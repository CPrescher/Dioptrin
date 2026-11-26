//! End-to-end image auto-calibration: raw 2D image → refined PONI with zero user clicks.
//!
//! Given a 2D powder diffraction image of a known calibrant (e.g. LaB6, CeO2),
//! the beam wavelength, and the detector pixel pitch (plus an optional rough
//! starting distance), recover the full detector geometry `(distance, poni1,
//! poni2, rot1, rot2)` automatically.
//!
//! ## Pipeline
//!
//! 1. **Automatic valid-pixel masking**: excludes NaN/Inf, negative sentinel
//!    pixels (Pilatus/Eiger module gaps `-1`/`-2`), uint32 dead sentinels, and
//!    contiguous zero border regions (e.g. MarCCD circular active-area corners).
//! 2. **Robust peak extraction**: fills invalid pixels with the image median,
//!    applies light pre-smoothing and local background subtraction, thresholds
//!    with MAD-based noise estimation, rejects peaks near mask edges, and keeps
//!    the strongest peaks across the full detector.
//! 3. **Ring-center estimation + radial-sharpness refinement**: 3-point circle
//!    RANSAC followed by multi-scale pattern search maximizing radial histogram
//!    sharpness `Σ count[b]²`. Works whether the beam center is inside or
//!    outside the detector.
//! 4. **1D radial-profile distance matching**: computes the azimuthal-average
//!    1D profile `I(r)` directly from valid image pixels around the refined
//!    center, subtracts a rolling lower-quantile baseline, and scans candidate
//!    distances with a symmetric F1-style matched-filter score (rewarding both
//!    predicted rings hitting 1D peaks and observed 1D peaks being explained).
//!    This eliminates cubic harmonic aliases (e.g. `D/2`, `2D`) that fool
//!    ratio-only ellipse matching on dense calibrants like LaB6.
//! 5. **Iterative 2D ring-guided sub-pixel refinement**: for the top distance
//!    candidates, projects predicted calibrant rings onto the 2D image using
//!    exact ray-plane intersection, extracts sub-pixel radial maxima along
//!    azimuthal sectors, and refines `(distance, poni1, poni2, rot1, rot2)`
//!    with Huber-weighted Levenberg-Marquardt. Iterates 3 passes with
//!    tightening search windows so moderate tilts are tracked cleanly.
//! 6. **Tilt / strip-detector fallback**: if the circular/radial-profile path
//!    fails verification (e.g. a heavily tilted strip detector), falls back to
//!    coarse `(distance, rot1, rot2, poni1, poni2)` grid search followed by
//!    2D ring-guided sub-pixel polish.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::poni::Poni;

use super::auto::{AutoCalibrateDiagnostics, AutoCalibrateError};
use super::calibrant::Calibrant;
use super::center::{RingCenterOptions, estimate_ring_center};
use super::grid_search::{GridSearchOptions, grid_search_top_k};
use super::massif::gaussian_blur;
use super::refinement::{
    CalibrationOptions, CalibrationPoints, CalibrationResult, Loss, ParamMask, refine,
};

/// Configuration for [`auto_calibrate_image`].
#[derive(Debug, Clone)]
pub struct AutoCalibrateImageOptions {
    /// Optional rough detector distance in metres (e.g. `Some(0.20)`).
    /// When `None`, a blind search over `[distance_min, distance_max]` is used.
    pub starting_distance: Option<f64>,
    /// Multiplicative search margin around `starting_distance`:
    /// searches `[d0 / factor, d0 * factor]`. Default: `2.0`.
    pub distance_tolerance_factor: f64,
    /// Lower distance bound (metres) when `starting_distance` is `None`.
    pub distance_min: f64,
    /// Upper distance bound (metres) when `starting_distance` is `None`.
    pub distance_max: f64,
    /// Parameter mask for the Levenberg–Marquardt refinement.
    /// Defaults to wavelength and `rot3` fixed, `(distance, poni1, poni2, rot1, rot2)` free.
    pub refine_mask: ParamMask,
    /// Fall back to coarse 5D grid search if the fast radial-profile path does
    /// not achieve a verified low-RMS fit (handles heavily tilted / strip detectors).
    pub enable_tilt_fallback: bool,
    /// Maximum calibrant ring index considered when scanning 1D profiles and
    /// extracting 2D ring peaks (default `65`; raise if only very high-order
    /// rings are visible on the detector).
    pub max_ring_index: usize,
    /// Maximum L-M iterations per refinement pass.
    pub max_iter: usize,
    /// L-M convergence tolerance.
    pub tol: f64,
}

impl Default for AutoCalibrateImageOptions {
    fn default() -> Self {
        let mut refine_mask = ParamMask::fix_wavelength();
        refine_mask.rot3 = false;
        Self {
            starting_distance: None,
            distance_tolerance_factor: 2.0,
            distance_min: 0.04,
            distance_max: 1.5,
            refine_mask,
            enable_tilt_fallback: true,
            max_ring_index: 65,
            max_iter: 200,
            tol: 1e-9,
        }
    }
}

/// Automatically calibrate a 2D powder diffraction image without any manual clicks.
///
/// # Arguments
///
/// * `image` — row-major 2D detector frame of length `dim1 * dim2` (same
///   orientation convention as [`crate::frame::Frame`]).
/// * `dim1`, `dim2` — image height (rows, slow axis) and width (cols, fast axis).
/// * `calibrant` — reference powder calibrant (e.g. LaB6, CeO2).
/// * `wavelength_m` — X-ray wavelength in metres.
/// * `pixel1`, `pixel2` — pixel pitch along `dim1` and `dim2` in metres.
/// * `options` — optional starting distance and solver knobs.
pub fn auto_calibrate_image(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &AutoCalibrateImageOptions,
) -> Result<(CalibrationResult, AutoCalibrateDiagnostics), AutoCalibrateError> {
    assert_eq!(image.len(), dim1 * dim2, "image size mismatch");
    if dim1 < 16 || dim2 < 16 {
        return Err(AutoCalibrateError::NoEllipses);
    }

    let (d_min, d_max) = distance_bounds(options);

    // 1. Automatic valid-pixel mask (excludes gaps, bad pixels, dead borders).
    let (valid, valid_median) = build_valid_mask(image, dim1, dim2);

    // 2. Robust image-wide peak extraction (also returns lightly-smoothed residual & noise floor).
    let (peaks, residual, sigma_r) =
        extract_robust_peaks(image, &valid, valid_median, dim1, dim2, 5000);
    if peaks.len() < 15 {
        return Err(AutoCalibrateError::NoEllipses);
    }

    let is_strip_detector = (dim1.max(dim2) as f64) / (dim1.min(dim2) as f64).max(1.0) > 2.5;

    // 3. Fast radial-profile path (handles centered, off-center, and moderately tilted detectors).
    // Skip for narrow strip detectors when tilt fallback is enabled, since circular ring-center
    // assumptions are under-constrained on short arcs.
    let radial_attempt = if is_strip_detector && options.enable_tilt_fallback {
        None
    } else {
        try_radial_profile_calibration(
            image,
            &residual,
            sigma_r,
            &valid,
            dim1,
            dim2,
            &peaks,
            calibrant,
            wavelength_m,
            pixel1,
            pixel2,
            d_min,
            d_max,
            options,
        )
    };

    let radial_inlier_frac = radial_attempt
        .as_ref()
        .map(|(res, _, _)| peak_inlier_fraction(&peaks, &res.poni, calibrant, 2.5e-3))
        .unwrap_or(0.0);

    if !is_strip_detector
        && let Some((res, diag, n_rings)) = &radial_attempt
        && is_verified_calibration(res, *n_rings)
        && radial_inlier_frac >= 0.55
    {
        return Ok((res.clone(), diag.clone()));
    }

    // 4. Tilt / strip-detector fallback via coarse 5D grid search + 2D image polish.
    if options.enable_tilt_fallback
        && let Some((grid_res, grid_diag, grid_rings)) = try_grid_fallback(
            &residual,
            sigma_r,
            &valid,
            dim1,
            dim2,
            &peaks,
            calibrant,
            wavelength_m,
            pixel1,
            pixel2,
            d_min,
            d_max,
            is_strip_detector,
            options,
        )
    {
        let grid_inlier_frac = peak_inlier_fraction(&peaks, &grid_res.poni, calibrant, 1.5e-3);
        let use_grid = match &radial_attempt {
            None => true,
            Some((r_res, _, r_rings)) => {
                let g_ok = is_verified_calibration(&grid_res, grid_rings);
                let r_ok = is_verified_calibration(r_res, *r_rings);
                if g_ok != r_ok {
                    g_ok
                } else {
                    let g_score = grid_inlier_frac.max(0.05) / (grid_res.rms_rad + 5.0e-5);
                    let r_score = radial_inlier_frac.max(0.05) / (r_res.rms_rad + 5.0e-5);
                    g_score > r_score
                }
            }
        };
        if use_grid {
            return Ok((grid_res, grid_diag));
        }
    }

    // If fallback didn't improve or was disabled, return the radial attempt if any.
    if let Some((res, diag, _)) = radial_attempt {
        return Ok((res, diag));
    }

    Err(AutoCalibrateError::NoConfidentIdentity)
}

fn distance_bounds(options: &AutoCalibrateImageOptions) -> (f64, f64) {
    if let Some(d0) = options.starting_distance
        && d0.is_finite()
        && d0 > 0.0
    {
        let factor = options.distance_tolerance_factor.max(1.1);
        return ((d0 / factor).max(0.01), d0 * factor);
    }
    let lo = options.distance_min.max(0.01);
    let hi = options.distance_max.max(lo * 1.1);
    (lo, hi)
}

fn is_verified_calibration(res: &CalibrationResult, n_rings: usize) -> bool {
    res.poni.distance.is_finite()
        && res.poni.distance > 0.005
        && res.rms_rad.is_finite()
        && res.rms_rad < 1.5e-3
        && n_rings >= 3
        && res.assignments.iter().filter(|a| a.is_some()).count() >= 40
}

fn peak_inlier_fraction(
    peaks: &[(f64, f64)],
    poni: &Poni,
    calibrant: &Calibrant,
    tol_rad: f64,
) -> f64 {
    if peaks.is_empty() {
        return 0.0;
    }
    let targets: Vec<f64> = calibrant
        .two_theta_rad(poni.wavelength)
        .into_iter()
        .flatten()
        .take(35)
        .collect();
    if targets.is_empty() {
        return 0.0;
    }
    let (sr1, cr1) = poni.rot1.sin_cos();
    let (sr2, cr2) = poni.rot2.sin_cos();
    let (sr3, cr3) = poni.rot3.sin_cos();
    let r00 = cr2 * cr3;
    let r01 = cr3 * sr1 * sr2 - cr1 * sr3;
    let r02 = -(cr1 * cr3 * sr2 + sr1 * sr3);
    let r10 = cr2 * sr3;
    let r11 = cr1 * cr3 + sr1 * sr2 * sr3;
    let r12 = cr3 * sr1 - cr1 * sr2 * sr3;
    let r20 = sr2;
    let r21 = -cr2 * sr1;
    let r22 = cr1 * cr2;

    let mut hits = 0usize;
    for &(d1, d2) in peaks {
        let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
        let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
        let p3 = poni.distance;
        let x = p1 * r00 + p2 * r01 + p3 * r02;
        let y = p1 * r10 + p2 * r11 + p3 * r12;
        let z = p1 * r20 + p2 * r21 + p3 * r22;
        let tth = (x * x + y * y).sqrt().atan2(z);
        if targets.iter().any(|&t| (tth - t).abs() <= tol_rad) {
            hits += 1;
        }
    }
    hits as f64 / peaks.len() as f64
}

// ---------- Step 1: Automatic valid-pixel mask ----------

fn subsample_median(values: &[f64], predicate: impl Fn(f64) -> bool) -> f64 {
    let step = (values.len() / 16_384).max(1);
    let mut sample: Vec<f64> = values
        .iter()
        .step_by(step)
        .copied()
        .filter(|&v| predicate(v))
        .collect();
    if sample.is_empty() {
        return 0.0;
    }
    let mid = sample.len() / 2;
    sample.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    sample[mid]
}

fn build_valid_mask(image: &[f64], dim1: usize, dim2: usize) -> (Vec<bool>, f64) {
    let n = dim1 * dim2;
    let p50_pos = subsample_median(image, |v| v.is_finite() && v > 0.0 && v < 4.0e9);
    let mask_zero_regions = p50_pos > 20.0;

    let mut valid = vec![true; n];
    for i in 0..dim1 {
        let row_off = i * dim2;
        for j in 0..dim2 {
            let idx = row_off + j;
            let v = image[idx];
            if !v.is_finite() || v < 0.0 || v >= 4.0e9 {
                valid[idx] = false;
                continue;
            }
            if mask_zero_regions && v == 0.0 {
                // Contiguous zero border/dead region (e.g. MarCCD circular aperture corners).
                let up = i == 0 || image[idx - dim2] <= 0.0;
                let dn = i + 1 == dim1 || image[idx + dim2] <= 0.0;
                let lt = j == 0 || image[idx - 1] <= 0.0;
                let rt = j + 1 == dim2 || image[idx + 1] <= 0.0;
                if up && dn && lt && rt {
                    valid[idx] = false;
                }
            }
        }
    }

    let valid_median = {
        let step = (n / 16_384).max(1);
        let mut sample = Vec::with_capacity(16_384);
        let mut k = 0;
        while k < n {
            if valid[k] {
                sample.push(image[k]);
            }
            k += step;
        }
        if sample.is_empty() {
            0.0
        } else {
            let mid = sample.len() / 2;
            sample.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            sample[mid]
        }
    };

    (valid, valid_median)
}

/// Check whether all pixels within `±margin` of `(i, j)` are valid.
#[inline]
fn neighborhood_valid(
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    i: usize,
    j: usize,
    margin: usize,
) -> bool {
    if i < margin || j < margin || i + margin >= dim1 || j + margin >= dim2 {
        return false;
    }
    for di in (i - margin)..=(i + margin) {
        let row = di * dim2;
        for dj in (j - margin)..=(j + margin) {
            if !valid[row + dj] {
                return false;
            }
        }
    }
    true
}

// ---------- Step 2: Robust image-wide peak extraction ----------

fn extract_robust_peaks(
    image: &[f64],
    valid: &[bool],
    valid_median: f64,
    dim1: usize,
    dim2: usize,
    max_peaks: usize,
) -> (Vec<(f64, f64)>, Vec<f64>, f64) {
    let n = dim1 * dim2;
    let filled: Vec<f64> = (0..n)
        .map(|k| if valid[k] { image[k] } else { valid_median })
        .collect();

    let smooth = gaussian_blur(&filled, dim1, dim2, 0.8);
    let sigma_bg = (dim1.max(dim2) as f64 / 80.0).clamp(6.0, 20.0);
    let bg = gaussian_blur(&smooth, dim1, dim2, sigma_bg);

    let residual: Vec<f64> = smooth.iter().zip(&bg).map(|(s, b)| s - b).collect();

    // Robust median and MAD of residual over valid pixels (subsampled).
    let step = (n / 32_768).max(1);
    let mut r_sample = Vec::with_capacity(32_768);
    let mut k = 0;
    while k < n {
        if valid[k] {
            r_sample.push(residual[k]);
        }
        k += step;
    }
    if r_sample.is_empty() {
        return (Vec::new(), residual, 0.0);
    }
    let mid = r_sample.len() / 2;
    r_sample.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let med_r = r_sample[mid];

    let mut abs_dev: Vec<f64> = r_sample.iter().map(|&v| (v - med_r).abs()).collect();
    abs_dev.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let mut sigma_r = 1.4826 * abs_dev[mid];
    if sigma_r < 1e-12 {
        let mean_r = r_sample.iter().sum::<f64>() / r_sample.len() as f64;
        let var_r = r_sample
            .iter()
            .map(|&v| (v - mean_r) * (v - mean_r))
            .sum::<f64>()
            / r_sample.len() as f64;
        sigma_r = var_r.sqrt();
    }
    if sigma_r < 1e-12 {
        return (Vec::new(), residual, 0.0);
    }

    let threshold = med_r + 3.5 * sigma_r;

    // Collect local maxima with their prominence so we can keep the strongest
    // peaks across the ENTIRE detector instead of truncating in row order.
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new();
    for i in 2..(dim1.saturating_sub(2)) {
        let row = i * dim2;
        for j in 2..(dim2.saturating_sub(2)) {
            let idx = row + j;
            let res_v = residual[idx];
            if res_v <= threshold {
                continue;
            }
            if !neighborhood_valid(valid, dim1, dim2, i, j, 2) {
                continue;
            }
            let v = smooth[idx];
            let mut is_max = true;
            'nbr: for di in -1isize..=1 {
                let nr = (i as isize + di) as usize * dim2;
                for dj in -1isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let nc = (j as isize + dj) as usize;
                    if smooth[nr + nc] > v {
                        is_max = false;
                        break 'nbr;
                    }
                }
            }
            if !is_max {
                continue;
            }
            // Parabolic sub-pixel refinement along row and col.
            let vm_r = smooth[(i - 1) * dim2 + j];
            let vp_r = smooth[(i + 1) * dim2 + j];
            let denom_r = vm_r - 2.0 * v + vp_r;
            let off_r = if denom_r.abs() > 1e-12 {
                (0.5 * (vm_r - vp_r) / denom_r).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let vm_c = smooth[idx - 1];
            let vp_c = smooth[idx + 1];
            let denom_c = vm_c - 2.0 * v + vp_c;
            let off_c = if denom_c.abs() > 1e-12 {
                (0.5 * (vm_c - vp_c) / denom_c).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            candidates.push((i as f64 + off_r, j as f64 + off_c, res_v));
        }
    }

    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    if candidates.len() > max_peaks {
        candidates.truncate(max_peaks);
    }
    let peaks = candidates.into_iter().map(|(d1, d2, _)| (d1, d2)).collect();
    (peaks, residual, sigma_r)
}

// ---------- Step 3: Ring-center estimation + local sharpness refinement ----------

fn radial_sharpness_score(
    peaks: &[(f64, f64)],
    center: (f64, f64),
    bin_width: f64,
    max_radius: f64,
) -> f64 {
    let n_bins = (max_radius / bin_width).ceil() as usize;
    if n_bins == 0 {
        return 0.0;
    }
    let mut hist = vec![0u32; n_bins];
    for &(d1, d2) in peaks {
        let r = (d1 - center.0).hypot(d2 - center.1);
        if !r.is_finite() || r >= max_radius {
            continue;
        }
        let bin = (r / bin_width) as usize;
        if bin < n_bins {
            hist[bin] += 1;
        }
    }
    hist.iter().map(|&c| (c as f64) * (c as f64)).sum()
}

fn refine_center_sharpness(peaks: &[(f64, f64)], init: (f64, f64), max_radius: f64) -> (f64, f64) {
    let working: Vec<(f64, f64)> = if peaks.len() > 1500 {
        let step = peaks.len().div_ceil(1500);
        peaks.iter().step_by(step).copied().collect()
    } else {
        peaks.to_vec()
    };

    let mut best = init;
    for &(step_px, bin_w) in &[
        (16.0, 4.0),
        (8.0, 3.0),
        (4.0, 2.0),
        (2.0, 1.5),
        (1.0, 1.0),
        (0.5, 1.0),
    ] {
        let mut best_score = radial_sharpness_score(&working, best, bin_w, max_radius);
        for _ in 0..6 {
            let mut improved = false;
            let origin = best;
            for &(dx, dy) in &[
                (-1.0, 0.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (0.0, 1.0),
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            ] {
                let cand = (origin.0 + dx * step_px, origin.1 + dy * step_px);
                let s = radial_sharpness_score(&working, cand, bin_w, max_radius);
                if s > best_score {
                    best_score = s;
                    best = cand;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    best
}

// ---------- Step 4: 1D radial profile + F1-scored distance scan ----------

struct RadialProfile1D {
    snr: Vec<f64>,
    r_min_obs: usize,
    r_max_obs: usize,
    peaks_r: Vec<(f64, f64)>, // (radius_px, snr)
}

fn build_radial_profile_1d(
    image: &[f64],
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    center: (f64, f64),
) -> Option<RadialProfile1D> {
    // Maximum distance from `center` to any corner of the detector.
    let corners = [
        (0.0, 0.0),
        (0.0, dim2 as f64),
        (dim1 as f64, 0.0),
        (dim1 as f64, dim2 as f64),
    ];
    let r_max = corners
        .iter()
        .map(|&(y, x)| (y - center.0).hypot(x - center.1))
        .fold(0.0_f64, f64::max)
        .ceil() as usize
        + 2;
    if !(20..=25_000).contains(&r_max) {
        return None;
    }

    // Parallel accumulation of 1-pixel radial bins.
    let chunk_rows = (dim1 / rayon::current_num_threads().max(1)).max(32);
    let (sum, count) = (0..dim1)
        .into_par_iter()
        .chunks(chunk_rows)
        .map(|rows| {
            let mut local_sum = vec![0.0_f64; r_max];
            let mut local_cnt = vec![0u32; r_max];
            for i in rows {
                let dy = i as f64 - center.0;
                let dy2 = dy * dy;
                let row_off = i * dim2;
                for j in 0..dim2 {
                    let idx = row_off + j;
                    if !valid[idx] {
                        continue;
                    }
                    let dx = j as f64 - center.1;
                    let bin = (dy2 + dx * dx).sqrt() as usize;
                    if bin < r_max {
                        local_sum[bin] += image[idx];
                        local_cnt[bin] += 1;
                    }
                }
            }
            (local_sum, local_cnt)
        })
        .reduce(
            || (vec![0.0_f64; r_max], vec![0u32; r_max]),
            |mut a, b| {
                for k in 0..r_max {
                    a.0[k] += b.0[k];
                    a.1[k] += b.1[k];
                }
                a
            },
        );

    let min_cnt = 15u32;
    let r_min_obs = count.iter().position(|&c| c >= min_cnt)?;
    let r_max_obs = count.iter().rposition(|&c| c >= min_cnt)?;
    if r_max_obs <= r_min_obs + 25 {
        return None;
    }

    let mut mean = vec![0.0_f64; r_max];
    for r in 0..r_max {
        if count[r] > 0 {
            mean[r] = sum[r] / count[r] as f64;
        }
    }

    // Rolling 20th-percentile baseline over ±20 px window within valid bins.
    let half_w = 20usize;
    let mut bg = vec![0.0_f64; r_max];
    let mut win = Vec::with_capacity(2 * half_w + 1);
    for r in r_min_obs..=r_max_obs {
        win.clear();
        let lo = r.saturating_sub(half_w).max(r_min_obs);
        let hi = (r + half_w).min(r_max_obs);
        for k in lo..=hi {
            if count[k] >= min_cnt {
                win.push(mean[k]);
            }
        }
        if !win.is_empty() {
            let q = win.len() / 5; // 20th percentile
            win.select_nth_unstable_by(q, |a, b| a.total_cmp(b));
            bg[r] = win[q];
        }
    }

    // Light 5-point boxcar smooth of the baseline.
    let mut bg_smooth = bg.clone();
    for r in r_min_obs..=r_max_obs {
        let lo = r.saturating_sub(2).max(r_min_obs);
        let hi = (r + 2).min(r_max_obs);
        let mut s = 0.0;
        let mut c = 0;
        for k in lo..=hi {
            s += bg[k];
            c += 1;
        }
        bg_smooth[r] = s / c as f64;
    }

    // Residual and robust 1D noise floor (MAD).
    let mut diff_vals = Vec::with_capacity(r_max_obs - r_min_obs + 1);
    let mut signal = vec![0.0_f64; r_max];
    for r in r_min_obs..=r_max_obs {
        if count[r] >= min_cnt {
            let d = mean[r] - bg_smooth[r];
            diff_vals.push(d);
            signal[r] = d.max(0.0);
        }
    }
    if diff_vals.len() < 20 {
        return None;
    }
    let mid = diff_vals.len() / 2;
    diff_vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let med = diff_vals[mid];
    let mut abs_dev: Vec<f64> = diff_vals.iter().map(|&v| (v - med).abs()).collect();
    abs_dev.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let mut noise = 1.4826 * abs_dev[mid];
    if noise < 1e-12 {
        let max_s = signal.iter().copied().fold(0.0_f64, f64::max);
        noise = (max_s * 1e-3).max(1e-9);
    }

    let snr: Vec<f64> = signal.iter().map(|&s| s / noise).collect();

    // 3-point smoothed SNR for clean 1D peak detection.
    let mut snr_s = vec![0.0_f64; r_max];
    for r in (r_min_obs + 1)..r_max_obs {
        snr_s[r] = 0.25 * snr[r - 1] + 0.5 * snr[r] + 0.25 * snr[r + 1];
    }

    // Extract strong 1D observed peaks with non-maximum suppression (sep >= 4 px).
    let mut raw_peaks: Vec<(usize, f64)> = Vec::new();
    for r in (r_min_obs + 2)..r_max_obs.saturating_sub(2) {
        if snr_s[r] >= 4.0 && snr_s[r] > snr_s[r - 1] && snr_s[r] >= snr_s[r + 1] {
            raw_peaks.push((r, snr_s[r]));
        }
    }
    raw_peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut peaks_r: Vec<(f64, f64)> = Vec::new();
    for (r, val) in raw_peaks {
        if peaks_r.iter().all(|&(pr, _)| (pr - r as f64).abs() >= 4.0) {
            // Sub-pixel 1D peak position.
            let vm = snr_s[r - 1];
            let v0 = snr_s[r];
            let vp = snr_s[r + 1];
            let denom = vm - 2.0 * v0 + vp;
            let off = if denom.abs() > 1e-12 {
                (0.5 * (vm - vp) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            peaks_r.push((r as f64 + off + 0.5, val));
            if peaks_r.len() >= 35 {
                break;
            }
        }
    }

    Some(RadialProfile1D {
        snr,
        r_min_obs,
        r_max_obs,
        peaks_r,
    })
}

/// Build a 1D radial profile from the *count* of detected 2D peaks per radial bin.
///
/// For spotty / coarse-grain powder patterns with bright single-crystal outliers,
/// pixel-mean azimuthal averaging dilutes discrete spots while letting a single
/// hot outlier create a false radial spike. Counting detected peaks per 1-pixel
/// radial bin is naturally outlier-resistant: 12 ring spots stack into one bin,
/// whereas 40 random outliers scatter into 40 bins of count 1.
fn build_peak_radial_profile_1d(
    peaks: &[(f64, f64)],
    dim1: usize,
    dim2: usize,
    center: (f64, f64),
) -> Option<RadialProfile1D> {
    if peaks.len() < 15 {
        return None;
    }
    let corners = [
        (0.0, 0.0),
        (0.0, dim2 as f64),
        (dim1 as f64, 0.0),
        (dim1 as f64, dim2 as f64),
    ];
    let r_max = corners
        .iter()
        .map(|&(y, x)| (y - center.0).hypot(x - center.1))
        .fold(0.0_f64, f64::max)
        .ceil() as usize
        + 2;
    if !(20..=25_000).contains(&r_max) {
        return None;
    }

    let mut counts = vec![0.0_f64; r_max];
    let mut r_min_obs = r_max;
    let mut r_max_obs = 0usize;
    for &(d1, d2) in peaks {
        let r = (d1 - center.0).hypot(d2 - center.1);
        if !r.is_finite() {
            continue;
        }
        let bin = r.round() as usize;
        if bin < r_max {
            counts[bin] += 1.0;
            r_min_obs = r_min_obs.min(bin);
            r_max_obs = r_max_obs.max(bin);
        }
    }
    if r_max_obs <= r_min_obs + 20 {
        return None;
    }

    // Light 5-point Gaussian-like smooth ([1, 4, 6, 4, 1] / 16) so 1-2 px ring
    // scatter merges cleanly into sharp 1D peaks.
    let mut smooth = vec![0.0_f64; r_max];
    for r in 2..(r_max - 2) {
        smooth[r] = (counts[r - 2]
            + 4.0 * counts[r - 1]
            + 6.0 * counts[r]
            + 4.0 * counts[r + 1]
            + counts[r + 2])
            / 16.0;
    }

    // Scale to SNR-like units (a bin with 3+ peaks is a clear ring).
    let snr: Vec<f64> = smooth.iter().map(|&c| c * 4.0).collect();

    let mut raw_peaks: Vec<(usize, f64)> = Vec::new();
    for r in (r_min_obs.max(2))..r_max_obs.min(r_max - 2) {
        if snr[r] >= 3.5 && snr[r] > snr[r - 1] && snr[r] >= snr[r + 1] {
            raw_peaks.push((r, snr[r]));
        }
    }
    raw_peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut peaks_r: Vec<(f64, f64)> = Vec::new();
    for (r, val) in raw_peaks {
        if peaks_r.iter().all(|&(pr, _)| (pr - r as f64).abs() >= 4.0) {
            peaks_r.push((r as f64, val));
            if peaks_r.len() >= 35 {
                break;
            }
        }
    }
    if peaks_r.len() < 3 {
        return None;
    }

    Some(RadialProfile1D {
        snr,
        r_min_obs,
        r_max_obs,
        peaks_r,
    })
}

fn scan_distances_1d(
    profile: &RadialProfile1D,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel_mean: f64,
    d_min: f64,
    d_max: f64,
    starting_distance: Option<f64>,
    max_ring_index: usize,
    top_k: usize,
) -> Vec<(f64, f64)> {
    let tan_rings: Vec<f64> = calibrant
        .two_theta_rad(wavelength_m)
        .into_iter()
        .flatten()
        .map(|t| t.tan())
        .filter(|&t| t.is_finite() && t > 0.0)
        .take(max_ring_index.max(10))
        .collect();
    if tan_rings.len() < 3 || profile.peaks_r.is_empty() {
        return Vec::new();
    }

    let r_lo = (profile.r_min_obs as f64).max(20.0);
    let r_hi = (profile.r_max_obs as f64) * 0.98;
    let total_obs_w: f64 = profile
        .peaks_r
        .iter()
        .map(|&(_, snr)| snr / (snr + 6.0))
        .sum();
    if total_obs_w <= 0.0 {
        return Vec::new();
    }

    let n_steps = 800usize;
    let ratio = (d_max / d_min).powf(1.0 / (n_steps - 1) as f64);
    let mut scores: Vec<(f64, f64)> = Vec::with_capacity(n_steps);

    for s in 0..n_steps {
        let d = d_min * ratio.powi(s as i32);
        let scale = d / pixel_mean;

        let mut pred_r: Vec<f64> = Vec::with_capacity(32);
        let mut forward_sum = 0.0;
        let mut n_hit = 0usize;

        for &tan_t in &tan_rings {
            let rk = scale * tan_t;
            if rk < r_lo {
                continue;
            }
            if rk > r_hi {
                break;
            }
            pred_r.push(rk);
            let tol = (0.012 * rk).clamp(3.0, 12.0);
            let b_lo = ((rk - tol).floor().max(0.0) as usize).min(profile.snr.len() - 1);
            let b_hi = ((rk + tol).ceil().max(0.0) as usize).min(profile.snr.len() - 1);
            let mut max_snr = 0.0_f64;
            for b in b_lo..=b_hi {
                if profile.snr[b] > max_snr {
                    max_snr = profile.snr[b];
                }
            }
            let q = max_snr / (max_snr + 6.0);
            forward_sum += q;
            if max_snr >= 4.0 {
                n_hit += 1;
            }
        }

        if pred_r.len() < 3 || n_hit < 3 {
            scores.push((d, 0.0));
            continue;
        }

        let forward = forward_sum / pred_r.len() as f64;

        let mut matched_obs_w = 0.0;
        for &(ro, obs_snr) in &profile.peaks_r {
            let tol = (0.012 * ro).clamp(3.0, 12.0);
            let matched = pred_r.iter().any(|&rk| (ro - rk).abs() <= tol);
            if matched {
                matched_obs_w += obs_snr / (obs_snr + 6.0);
            }
        }
        let backward = matched_obs_w / total_obs_w;

        let f1 = if forward + backward > 0.0 {
            2.0 * forward * backward / (forward + backward)
        } else {
            0.0
        };
        let count_weight = (n_hit as f64) / (n_hit as f64 + 1.5);
        let prior = match starting_distance {
            Some(d0) if d0 > 0.0 => {
                let log_err = (d / d0).ln();
                (1.0 - 0.05 * log_err * log_err).max(0.8)
            }
            _ => 1.0,
        };
        scores.push((d, f1 * count_weight * prior));
    }

    // Extract distinct local maxima (separated by at least 5% in distance).
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 1..(n_steps - 1) {
        let sc = scores[i].1;
        if sc > 0.05 && sc >= scores[i - 1].1 && sc >= scores[i + 1].1 {
            peaks.push(scores[i]);
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut chosen: Vec<(f64, f64)> = Vec::new();
    for (d, sc) in peaks {
        if chosen.iter().all(|&(cd, _)| (d / cd - 1.0).abs() > 0.05) {
            chosen.push((d, sc));
            if chosen.len() >= top_k {
                break;
            }
        }
    }
    chosen
}

// ---------- Step 5: Exact ray-plane ring projection + 2D sub-pixel refinement ----------

/// Exact detector pixel `(d1, d2)` where the Debye-Scherrer cone `(tth, phi)`
/// intersects the detector plane for the given `Poni`.
///
/// Exact inverse of `refinement::tth_at`.
#[inline]
fn project_ring_pixel(poni: &Poni, tth: f64, phi: f64) -> Option<(f64, f64)> {
    let (st, ct) = tth.sin_cos();
    let (sp, cp) = phi.sin_cos();
    let ux = st * cp;
    let uy = st * sp;
    let uz = ct;

    let (sr1, cr1) = poni.rot1.sin_cos();
    let (sr2, cr2) = poni.rot2.sin_cos();
    let (sr3, cr3) = poni.rot3.sin_cos();

    // Rotation matrix R maps detector frame (p1, p2, p3) -> lab frame (x, y, z):
    //   x = p1 * r00 + p2 * r01 + p3 * r02
    //   y = p1 * r10 + p2 * r11 + p3 * r12
    //   z = p1 * r20 + p2 * r21 + p3 * r22
    let r00 = cr2 * cr3;
    let r01 = cr3 * sr1 * sr2 - cr1 * sr3;
    let r02 = -(cr1 * cr3 * sr2 + sr1 * sr3);

    let r10 = cr2 * sr3;
    let r11 = cr1 * cr3 + sr1 * sr2 * sr3;
    let r12 = cr3 * sr1 - cr1 * sr2 * sr3;

    let r20 = sr2;
    let r21 = -cr2 * sr1;
    let r22 = cr1 * cr2;

    // Detector-frame ray direction q = R^T * u.
    let q1 = r00 * ux + r10 * uy + r20 * uz;
    let q2 = r01 * ux + r11 * uy + r21 * uz;
    let q3 = r02 * ux + r12 * uy + r22 * uz;

    if q3 <= 1e-6 {
        return None;
    }
    let t = poni.distance / q3;
    let p1 = t * q1;
    let p2 = t * q2;
    let d1 = (p1 + poni.poni1) / poni.pixel1 - 0.5;
    let d2 = (p2 + poni.poni2) / poni.pixel2 - 0.5;
    if d1.is_finite() && d2.is_finite() {
        Some((d1, d2))
    } else {
        None
    }
}

#[inline]
fn sample_bilinear_valid(
    image: &[f64],
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    d1: f64,
    d2: f64,
) -> Option<f64> {
    if d1 < 0.0 || d2 < 0.0 {
        return None;
    }
    let i0 = d1.floor() as usize;
    let j0 = d2.floor() as usize;
    if i0 + 1 >= dim1 || j0 + 1 >= dim2 {
        return None;
    }
    let idx00 = i0 * dim2 + j0;
    let idx01 = idx00 + 1;
    let idx10 = idx00 + dim2;
    let idx11 = idx10 + 1;
    if !(valid[idx00] && valid[idx01] && valid[idx10] && valid[idx11]) {
        return None;
    }
    let fy = d1 - i0 as f64;
    let fx = d2 - j0 as f64;
    let top = image[idx00] * (1.0 - fx) + image[idx01] * fx;
    let bot = image[idx10] * (1.0 - fx) + image[idx11] * fx;
    Some(top * (1.0 - fy) + bot * fy)
}

/// Sample sub-pixel ring maxima along azimuthal sectors for each visible
/// calibrant ring predicted by `poni`, using the background-subtracted residual.
fn extract_ring_peaks_2d(
    residual: &[f64],
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    calibrant: &Calibrant,
    half_width_px: usize,
    n_az: usize,
    max_rings: usize,
    max_ring_index: usize,
) -> (CalibrationPoints, usize) {
    let tth_list = calibrant.two_theta_rad(poni.wavelength);
    let mut pts = CalibrationPoints::new();
    let mut n_rings_with_pts = 0usize;
    let w = half_width_px as isize;
    let margin = (half_width_px + 2) as f64;
    let mut line = Vec::with_capacity(2 * half_width_px + 1);
    let min_peak_snr = 2.5 * sigma_r;

    for (ring_k, opt_tth) in tth_list
        .iter()
        .enumerate()
        .take(max_ring_index.max(max_rings))
    {
        if n_rings_with_pts >= max_rings {
            break;
        }
        let tth = match opt_tth {
            Some(t) if t.is_finite() && *t > 1e-4 && *t < PI * 0.49 => *t,
            _ => continue,
        };
        let mut ring_pts = 0usize;

        for a in 0..n_az {
            let phi = 2.0 * PI * (a as f64) / (n_az as f64);
            let (d1, d2) = match project_ring_pixel(poni, tth, phi) {
                Some(p) => p,
                None => continue,
            };
            if d1 < margin
                || d2 < margin
                || d1 > (dim1 as f64 - 1.0 - margin)
                || d2 > (dim2 as f64 - 1.0 - margin)
            {
                continue;
            }
            // Local outward radial normal via symmetric finite difference in 2θ.
            let delta = 1e-4;
            let (p_hi, p_lo) = match (
                project_ring_pixel(poni, tth + delta, phi),
                project_ring_pixel(poni, (tth - delta).max(1e-5), phi),
            ) {
                (Some(hi), Some(lo)) => (hi, lo),
                _ => continue,
            };
            let du1 = p_hi.0 - p_lo.0;
            let du2 = p_hi.1 - p_lo.1;
            let norm = du1.hypot(du2);
            if norm < 1e-9 {
                continue;
            }
            let u1 = du1 / norm;
            let u2 = du2 / norm;

            // Sample residual intensity along the normal.
            line.clear();
            let mut all_ok = true;
            for s in -w..=w {
                let sd = s as f64;
                match sample_bilinear_valid(residual, valid, dim1, dim2, d1 + sd * u1, d2 + sd * u2)
                {
                    Some(v) => line.push(v),
                    None => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if !all_ok || line.len() < 5 {
                continue;
            }

            // Find strict interior maximum.
            let mut best_idx = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            let mut min_val = f64::INFINITY;
            for (idx, &v) in line.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
                if v < min_val {
                    min_val = v;
                }
            }
            if best_idx == 0 || best_idx + 1 == line.len() {
                continue;
            }

            // Require clear prominence above both global background noise and window floor.
            if best_val < min_peak_snr || (best_val - min_val) < min_peak_snr {
                continue;
            }

            // Parabolic 3-point sub-pixel refinement.
            let vm = line[best_idx - 1];
            let v0 = line[best_idx];
            let vp = line[best_idx + 1];
            let denom = vm - 2.0 * v0 + vp;
            let sub = if denom.abs() > 1e-12 {
                (0.5 * (vm - vp) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let s_refined = (best_idx as isize - w) as f64 + sub;
            pts.push(d1 + s_refined * u1, d2 + s_refined * u2, Some(ring_k));
            ring_pts += 1;
        }

        if ring_pts >= 5 {
            n_rings_with_pts += 1;
        }
    }

    (pts, n_rings_with_pts)
}

fn iterative_2d_refine(
    residual: &[f64],
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    init_poni: Poni,
    calibrant: &Calibrant,
    options: &AutoCalibrateImageOptions,
    max_ring_index: usize,
    passes: &[(usize, f64, usize, usize)], // (half_width_px, huber_c, n_az, max_rings)
) -> Option<(CalibrationResult, usize)> {
    let mut current_poni = init_poni;
    let mut last_result: Option<(CalibrationResult, usize)> = None;

    for &(half_w, huber_c, n_az, max_rings) in passes {
        let (pts, n_rings) = extract_ring_peaks_2d(
            residual,
            sigma_r,
            valid,
            dim1,
            dim2,
            &current_poni,
            calibrant,
            half_w,
            n_az,
            max_rings,
            max_ring_index,
        );
        if pts.len() < 15 || n_rings < 2 {
            break;
        }
        let geom = Geometry::new(dim1, dim2, current_poni.clone());
        let refine_opts = CalibrationOptions {
            mask: options.refine_mask,
            loss: Loss::Huber { c: huber_c },
            auto_assign_rings: false,
            assign_tolerance_rad: 0.05,
            max_iter: options.max_iter,
            tol: options.tol,
        };
        match refine(&geom, &pts, calibrant, &refine_opts) {
            Ok(res) if res.poni.distance.is_finite() && res.poni.distance > 0.005 => {
                current_poni = res.poni.clone();
                last_result = Some((res, n_rings));
            }
            _ => break,
        }
    }

    last_result
}

fn refine_on_detected_peaks(
    peaks: &[(f64, f64)],
    dim1: usize,
    dim2: usize,
    init_poni: Poni,
    calibrant: &Calibrant,
    options: &AutoCalibrateImageOptions,
) -> Option<(CalibrationResult, usize)> {
    let mut pts = CalibrationPoints::new();
    for &(d1, d2) in peaks {
        pts.push(d1, d2, None);
    }
    let mut current = init_poni;
    let mut last: Option<(CalibrationResult, usize)> = None;
    for &(assign_tol, huber_c) in &[(1.0e-2, 3.0e-3), (4.0e-3, 1.5e-3), (2.5e-3, 8.0e-4)] {
        let geom = Geometry::new(dim1, dim2, current.clone());
        let opts = CalibrationOptions {
            mask: options.refine_mask,
            loss: Loss::Huber { c: huber_c },
            auto_assign_rings: true,
            assign_tolerance_rad: assign_tol,
            max_iter: options.max_iter,
            tol: options.tol,
        };
        match refine(&geom, &pts, calibrant, &opts) {
            Ok(res) if res.poni.distance.is_finite() && res.poni.distance > 0.005 => {
                let mut rings: Vec<usize> = res.assignments.iter().flatten().copied().collect();
                rings.sort_unstable();
                rings.dedup();
                if rings.len() < 3 {
                    break;
                }
                current = res.poni.clone();
                last = Some((res, rings.len()));
            }
            _ => break,
        }
    }
    last
}

fn try_radial_profile_calibration(
    image: &[f64],
    residual: &[f64],
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    peaks: &[(f64, f64)],
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    d_min: f64,
    d_max: f64,
    options: &AutoCalibrateImageOptions,
) -> Option<(CalibrationResult, AutoCalibrateDiagnostics, usize)> {
    let diag_px = (dim1 as f64).hypot(dim2 as f64);
    let max_radius = (2.5 * diag_px).max(2000.0);

    // Subsample peaks for fast 3-point circle RANSAC.
    let ransac_peaks: Vec<(f64, f64)> = if peaks.len() > 600 {
        let step = peaks.len().div_ceil(600);
        peaks.iter().step_by(step).copied().collect()
    } else {
        peaks.to_vec()
    };
    let center_opts = RingCenterOptions {
        max_trials: 2500,
        radial_bin_width_px: 4.0,
        top_k_bins: 12,
        max_radius_px: max_radius,
        ..RingCenterOptions::default()
    };
    let coarse_center = estimate_ring_center(&ransac_peaks, &center_opts)?;
    let center = refine_center_sharpness(peaks, (coarse_center.d1, coarse_center.d2), max_radius);

    // Build BOTH the pixel-mean 1D profile (ideal for smooth continuous rings)
    // and the peak-count 1D profile (ideal for spotty patterns with outliers),
    // and merge their top distance candidates.
    let pixel_mean = (pixel1 * pixel2).sqrt();
    let mut dist_candidates: Vec<(f64, f64)> = Vec::new();
    let mut n_detected_1d = 0usize;

    if let Some(profile) = build_radial_profile_1d(image, valid, dim1, dim2, center) {
        n_detected_1d = n_detected_1d.max(profile.peaks_r.len());
        dist_candidates.extend(scan_distances_1d(
            &profile,
            calibrant,
            wavelength_m,
            pixel_mean,
            d_min,
            d_max,
            options.starting_distance,
            options.max_ring_index,
            4,
        ));
    }
    if let Some(peak_profile) = build_peak_radial_profile_1d(peaks, dim1, dim2, center) {
        n_detected_1d = n_detected_1d.max(peak_profile.peaks_r.len());
        for (d, sc) in scan_distances_1d(
            &peak_profile,
            calibrant,
            wavelength_m,
            pixel_mean,
            d_min,
            d_max,
            options.starting_distance,
            options.max_ring_index,
            4,
        ) {
            if dist_candidates
                .iter()
                .all(|&(cd, _)| (d / cd - 1.0).abs() > 0.05)
            {
                dist_candidates.push((d, sc));
            }
        }
    }
    if dist_candidates.is_empty() {
        return None;
    }

    let bootstrap_p1 = (center.0 + 0.5) * pixel1;
    let bootstrap_p2 = (center.1 + 0.5) * pixel2;
    let passes = [
        (14usize, 3.0e-3, 96usize, 16usize),
        (7, 1.5e-3, 128, 22),
        (4, 8.0e-4, 128, 22),
    ];

    let mut best: Option<(CalibrationResult, AutoCalibrateDiagnostics, usize, f64)> = None;
    for &(d_cand, d_score) in &dist_candidates {
        let mut init = Poni::new();
        init.pixel1 = pixel1;
        init.pixel2 = pixel2;
        init.distance = d_cand;
        init.poni1 = bootstrap_p1;
        init.poni2 = bootstrap_p2;
        init.rot1 = 0.0;
        init.rot2 = 0.0;
        init.rot3 = 0.0;
        init.wavelength = wavelength_m;

        // Try both ring-guided 2D profile refinement (best for continuous rings)
        // and detected-peak refinement (best for spotty / coarse-grain rings).
        let smooth_res = iterative_2d_refine(
            residual,
            sigma_r,
            valid,
            dim1,
            dim2,
            init.clone(),
            calibrant,
            options,
            options.max_ring_index,
            &passes,
        );
        let peak_res = refine_on_detected_peaks(peaks, dim1, dim2, init, calibrant, options);

        for (res, n_rings) in [smooth_res, peak_res].into_iter().flatten() {
            let inlier_frac = peak_inlier_fraction(peaks, &res.poni, calibrant, 2.0e-3);
            let quality = (inlier_frac.max(0.05) * (n_rings as f64).sqrt() * d_score.max(0.1))
                / (res.rms_rad + 3.0e-5);
            let is_better = match &best {
                None => true,
                Some((b_res, _, b_rings, b_qual)) => {
                    let cur_ok = is_verified_calibration(&res, n_rings);
                    let best_ok = is_verified_calibration(b_res, *b_rings);
                    if cur_ok != best_ok {
                        cur_ok
                    } else {
                        quality > *b_qual
                    }
                }
            };
            if is_better {
                let diag = AutoCalibrateDiagnostics {
                    n_detected_ellipses: n_detected_1d,
                    n_identified_ellipses: n_rings,
                    bootstrap_distance: d_cand,
                    bootstrap_ring_center_d1: bootstrap_p1,
                    bootstrap_ring_center_d2: bootstrap_p2,
                };
                best = Some((res, diag, n_rings, quality));
            }
        }
    }

    best.map(|(res, diag, n_rings, _)| (res, diag, n_rings))
}

// ---------- Step 6: Tilt / strip-detector fallback ----------

fn try_grid_fallback(
    residual: &[f64],
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    peaks: &[(f64, f64)],
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    d_min: f64,
    d_max: f64,
    is_strip_detector: bool,
    options: &AutoCalibrateImageOptions,
) -> Option<(CalibrationResult, AutoCalibrateDiagnostics, usize)> {
    // Use the highest-prominence peaks (peaks are sorted strongest-first).
    let strong_peaks: &[(f64, f64)] = &peaks[..peaks.len().min(800)];
    let cx = strong_peaks.iter().map(|p| p.0).sum::<f64>() / strong_peaks.len() as f64;
    let cy = strong_peaks.iter().map(|p| p.1).sum::<f64>() / strong_peaks.len() as f64;

    let (grid_d_min, grid_d_max, grid_d_steps) = match options.starting_distance {
        Some(d0) if d0.is_finite() && d0 > 0.0 => ((d0 / 1.6).max(0.02), (d0 * 1.6).min(0.8), 16),
        _ => (d_min, d_max.min(0.6), 18),
    };
    let grid_opts = GridSearchOptions {
        distance_min: grid_d_min,
        distance_max: grid_d_max,
        distance_steps: grid_d_steps,
        rot_max: 45.0_f64.to_radians(),
        rot_steps: 15,
        poni_search_radius_px: (0.30 * dim1.max(dim2) as f64).clamp(150.0, 450.0),
        poni_grid_steps: 13,
        tolerance_rad: 8.0e-3,
        max_peaks_for_grid: 400,
        max_ring_index: 12,
        top_k: 40,
    };

    let candidates = grid_search_top_k(
        strong_peaks,
        cx,
        cy,
        calibrant,
        wavelength_m,
        pixel1,
        pixel2,
        &grid_opts,
    );
    if candidates.is_empty() {
        return None;
    }

    let mut pts = CalibrationPoints::new();
    for &(d1, d2) in strong_peaks {
        pts.push(d1, d2, None);
    }
    let refine_opts = CalibrationOptions {
        mask: options.refine_mask,
        loss: Loss::Huber { c: 2.0e-3 },
        auto_assign_rings: true,
        assign_tolerance_rad: grid_opts.tolerance_rad.max(5.0e-3),
        max_iter: options.max_iter,
        tol: options.tol,
    };

    let mut best_grid: Option<(
        CalibrationResult,
        &super::grid_search::GridSearchCandidate,
        f64,
    )> = None;
    for cand in &candidates {
        let geom = Geometry::new(dim1, dim2, cand.poni.clone());
        if let Ok(res) = refine(&geom, &pts, calibrant, &refine_opts) {
            if !res.poni.distance.is_finite() || res.poni.distance <= 0.005 {
                continue;
            }
            let inlier_frac = peak_inlier_fraction(strong_peaks, &res.poni, calibrant, 3.5e-3);
            let score = inlier_frac / (res.rms_rad + 2.0e-4);
            if best_grid.as_ref().is_none_or(|(_, _, s)| score > *s) {
                best_grid = Some((res, cand, score));
            }
        }
    }
    let (grid_res, winning, grid_score) = best_grid?;

    // Polish on the actual 2D residual via ring-guided sub-pixel extraction.
    let n_az = if is_strip_detector { 360 } else { 128 };
    let polish_passes = [(8usize, 1.5e-3, n_az, 18usize), (4, 8.0e-4, n_az, 18)];
    if let Some((polished, n_rings)) = iterative_2d_refine(
        residual,
        sigma_r,
        valid,
        dim1,
        dim2,
        grid_res.poni.clone(),
        calibrant,
        options,
        18,
        &polish_passes,
    ) {
        let pol_inlier_frac = peak_inlier_fraction(strong_peaks, &polished.poni, calibrant, 3.5e-3);
        let pol_score = pol_inlier_frac / (polished.rms_rad + 2.0e-4);
        if n_rings >= 3 && pol_score >= grid_score * 0.9 {
            let diag = AutoCalibrateDiagnostics {
                n_detected_ellipses: candidates.len(),
                n_identified_ellipses: n_rings,
                bootstrap_distance: winning.poni.distance,
                bootstrap_ring_center_d1: winning.poni.poni1,
                bootstrap_ring_center_d2: winning.poni.poni2,
            };
            return Some((polished, diag, n_rings));
        }
    }

    let n_rings = {
        let mut rings: Vec<usize> = grid_res.assignments.iter().flatten().copied().collect();
        rings.sort_unstable();
        rings.dedup();
        rings.len()
    };
    let diag = AutoCalibrateDiagnostics {
        n_detected_ellipses: candidates.len(),
        n_identified_ellipses: winning.n_inliers,
        bootstrap_distance: winning.poni.distance,
        bootstrap_ring_center_d1: winning.poni.poni1,
        bootstrap_ring_center_d2: winning.poni.poni2,
    };
    Some((grid_res, diag, n_rings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_ring_pixel_inverts_tth_for_tilted_poni() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.72e-4;
        poni.pixel2 = 1.72e-4;
        poni.distance = 0.246;
        poni.poni1 = 0.0177;
        poni.poni2 = 0.1007;
        poni.rot1 = -0.552;
        poni.rot2 = 0.08;
        poni.rot3 = 0.12;
        poni.wavelength = 2.91e-11;

        for tth in [0.05, 0.12, 0.25, 0.4] {
            for a in 0..16 {
                let phi = 2.0 * PI * (a as f64) / 16.0;
                if let Some((d1, d2)) = project_ring_pixel(&poni, tth, phi) {
                    // Forward-check using the same tth formula as refinement.
                    let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
                    let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
                    let p3 = poni.distance;
                    let (sr1, cr1) = poni.rot1.sin_cos();
                    let (sr2, cr2) = poni.rot2.sin_cos();
                    let (sr3, cr3) = poni.rot3.sin_cos();
                    let x = p1 * cr2 * cr3 + p2 * (cr3 * sr1 * sr2 - cr1 * sr3)
                        - p3 * (cr1 * cr3 * sr2 + sr1 * sr3);
                    let y = p1 * cr2 * sr3 + p2 * (cr1 * cr3 + sr1 * sr2 * sr3)
                        - p3 * (-(cr3 * sr1) + cr1 * sr2 * sr3);
                    let z = p1 * sr2 - p2 * cr2 * sr1 + p3 * cr1 * cr2;
                    let tth_check = (x * x + y * y).sqrt().atan2(z);
                    assert!(
                        (tth_check - tth).abs() < 1e-12,
                        "tth mismatch: {tth_check} vs {tth}"
                    );
                }
            }
        }
    }
}
