//! End-to-end image auto-calibration: raw 2D image → refined PONI with zero user clicks.
//!
//! Given a 2D powder diffraction image of a known calibrant (e.g. LaB6, CeO2),
//! the beam wavelength, and the detector pixel pitch (plus an optional rough
//! starting distance and optional pixel mask), recover the full detector
//! geometry `(distance, poni1, poni2, rot1, rot2)` automatically.
//!
//! ## Pipeline
//!
//! 1. **Automatic valid-pixel masking**: combines any caller-provided mask
//!    (`true` = excluded) with automatic exclusion of NaN/Inf, uint32 dead
//!    sentinels, negative sentinel gaps (`-1`/`-2`), and border-connected zero
//!    regions (e.g. MarCCD circular active-area corners, dilated by 1 px).
//!    Dark-subtracted images whose background noise fluctuates around 0 are
//!    preserved cleanly.
//! 2. **Robust peak extraction**: fills invalid pixels with the valid-pixel
//!    median, applies light pre-smoothing and local background subtraction,
//!    thresholds with MAD-based noise estimation, rejects peaks near mask
//!    edges, and keeps the strongest peaks across the full detector.
//! 3. **Isotropic ring-center estimation + radial-sharpness refinement**: works
//!    in an isotropic metric coordinate frame `(d1·pixel1, d2·pixel2) / p_min`
//!    so non-square pixels (`pixel1 != pixel2`) stay circular, using 3-point
//!    circle RANSAC followed by multi-scale radial-sharpness refinement.
//! 4. **1D radial-profile distance matching**: computes both the pixel-mean 1D
//!    profile (ideal for smooth continuous rings) and the peak-count 1D profile
//!    (ideal for spotty patterns with single-crystal outliers), and scans
//!    candidate distances with a symmetric F1 matched-filter score.
//! 5. **Iterative 2D sub-pixel refinement**: for each top distance candidate,
//!    runs both exact ray-plane ring-guided 2D profile refinement and
//!    detected-peak refinement with Huber-weighted Levenberg–Marquardt.
//! 6. **Verification & tilt/strip-detector fallback**: verifies every candidate
//!    with consistent ring support, RMS, and 2D peak inlier coverage. If the
//!    fast radial-profile path fails verification (or on narrow strip detectors
//!    with aspect ratio > 2.5), falls back to coarse 5D grid search + 2D polish.
//!    Returns `Err(NoConfidentIdentity)` if no candidate passes verification.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::collections::VecDeque;
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

/// Configuration for [`auto_calibrate_image`] and [`auto_calibrate_image_with_mask`].
#[derive(Debug, Clone)]
pub struct AutoCalibrateImageOptions {
    /// Optional rough detector distance in metres (e.g. `Some(0.20)`).
    /// When `None`, a blind search over `[distance_min, distance_max]` is used.
    /// When `refine_mask.distance` is `false` and `starting_distance` is `Some(d0)`,
    /// `d0` is used directly as the fixed distance.
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
/// Convenience wrapper around [`auto_calibrate_image_with_mask`] with `mask = None`.
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
    auto_calibrate_image_with_mask(
        image,
        dim1,
        dim2,
        None,
        calibrant,
        wavelength_m,
        pixel1,
        pixel2,
        options,
    )
}

/// Automatically calibrate a 2D powder diffraction image with an optional pixel mask.
///
/// # Arguments
///
/// * `image` — row-major 2D detector frame of length `dim1 * dim2` (same
///   orientation convention as [`crate::frame::Frame`], i.e. row 0 at bottom).
/// * `dim1`, `dim2` — image height (rows, slow axis) and width (cols, fast axis).
/// * `mask` — optional row-major boolean mask of length `dim1 * dim2` where
///   `true` marks an excluded/masked pixel (pyFAI / Dioptas convention).
/// * `calibrant` — reference powder calibrant (e.g. LaB6, CeO2).
/// * `wavelength_m` — X-ray wavelength in metres (must be finite and `> 0`).
/// * `pixel1`, `pixel2` — pixel pitch along `dim1` and `dim2` in metres (finite and `> 0`).
/// * `options` — optional starting distance and solver knobs.
pub fn auto_calibrate_image_with_mask(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &AutoCalibrateImageOptions,
) -> Result<(CalibrationResult, AutoCalibrateDiagnostics), AutoCalibrateError> {
    let expected_len = dim1
        .checked_mul(dim2)
        .ok_or_else(|| AutoCalibrateError::InvalidInput("image dimensions overflow".into()))?;
    if image.len() != expected_len {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "image length {} does not match dim1*dim2 ({}x{} = {})",
            image.len(),
            dim1,
            dim2,
            expected_len
        )));
    }
    if dim1 < 16 || dim2 < 16 {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "image too small for auto-calibration ({dim1}x{dim2}; need at least 16x16)"
        )));
    }
    if let Some(m) = mask
        && m.len() != expected_len
    {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "mask length {} does not match image length {}",
            m.len(),
            expected_len
        )));
    }
    if !wavelength_m.is_finite() || wavelength_m <= 0.0 {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "wavelength_m must be finite and positive, got {wavelength_m}"
        )));
    }
    if !pixel1.is_finite() || pixel1 <= 0.0 || !pixel2.is_finite() || pixel2 <= 0.0 {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "pixel sizes must be finite and positive, got ({pixel1}, {pixel2})"
        )));
    }
    if let Some(d0) = options.starting_distance
        && (!d0.is_finite() || d0 <= 0.0)
    {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "starting_distance must be finite and positive, got {d0}"
        )));
    }

    let (d_min, d_max) = distance_bounds(options);

    // 1. Automatic valid-pixel mask (combined with optional user mask).
    let (valid, valid_median) = build_valid_mask(image, mask, dim1, dim2);

    // 2. Robust image-wide peak extraction (also returns lightly-smoothed residual & noise floor).
    let (peaks, residual, sigma_r) =
        extract_robust_peaks(image, &valid, valid_median, dim1, dim2, 5000);
    if peaks.len() < 15 {
        return Err(AutoCalibrateError::NoEllipses);
    }

    let is_strip_detector = (dim1.max(dim2) as f64) / (dim1.min(dim2) as f64).max(1.0) > 2.5;
    const VERIFY_TOL_RAD: f64 = 2.0e-3;

    // 3. Fast radial-profile path (handles centered, off-center, non-square, and moderately tilted detectors).
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
        .map(|(res, _, _)| peak_inlier_fraction(&peaks, &res.poni, calibrant, VERIFY_TOL_RAD))
        .unwrap_or(0.0);

    let radial_verified = radial_attempt.as_ref().is_some_and(|(res, _, n_rings)| {
        is_verified_calibration(
            res,
            *n_rings,
            radial_inlier_frac,
            d_min,
            d_max,
            options.starting_distance.is_some(),
        )
    });

    if !is_strip_detector
        && radial_verified
        && radial_inlier_frac >= 0.55
        && let Some((res, diag, _)) = radial_attempt
    {
        return Ok((res, diag));
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
        let grid_inlier_frac =
            peak_inlier_fraction(&peaks, &grid_res.poni, calibrant, VERIFY_TOL_RAD);
        let grid_verified = is_verified_calibration(
            &grid_res,
            grid_rings,
            grid_inlier_frac,
            d_min,
            d_max,
            options.starting_distance.is_some(),
        );
        if grid_verified {
            let use_grid = match &radial_attempt {
                None => true,
                Some((r_res, _, _)) => {
                    if !radial_verified {
                        true
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
    }

    // Return the radial result only if it actually passed verification.
    if radial_verified && let Some((res, diag, _)) = radial_attempt {
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

/// Count distinct calibrant rings that have at least `min_pts_per_ring` assigned points.
fn count_supported_rings(assignments: &[Option<usize>], min_pts_per_ring: usize) -> usize {
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for r in assignments.iter().flatten() {
        *counts.entry(*r).or_insert(0) += 1;
    }
    counts.values().filter(|&&c| c >= min_pts_per_ring).count()
}

fn is_verified_calibration(
    res: &CalibrationResult,
    n_rings: usize,
    inlier_frac: f64,
    d_min: f64,
    d_max: f64,
    has_distance_hint: bool,
) -> bool {
    if !res.poni.distance.is_finite()
        || res.poni.distance <= 0.005
        || !res.rms_rad.is_finite()
        || res.rms_rad >= 1.5e-3
        || n_rings < 3
        || inlier_frac < 0.45
    {
        return false;
    }
    if res.assignments.iter().filter(|a| a.is_some()).count() < 40 {
        return false;
    }
    if has_distance_hint && (res.poni.distance < d_min * 0.5 || res.poni.distance > d_max * 1.5) {
        return false;
    }
    true
}

/// 3x3 rotation matrix mapping detector frame `(p1, p2, p3)` to lab frame `(x, y, z)`,
/// matching `refinement::tth_at`.
#[inline]
fn rotation_matrix(poni: &Poni) -> [[f64; 3]; 3] {
    let (sr1, cr1) = poni.rot1.sin_cos();
    let (sr2, cr2) = poni.rot2.sin_cos();
    let (sr3, cr3) = poni.rot3.sin_cos();
    [
        [
            cr2 * cr3,
            cr3 * sr1 * sr2 - cr1 * sr3,
            -(cr1 * cr3 * sr2 + sr1 * sr3),
        ],
        [
            cr2 * sr3,
            cr1 * cr3 + sr1 * sr2 * sr3,
            cr3 * sr1 - cr1 * sr2 * sr3,
        ],
        [sr2, -cr2 * sr1, cr1 * cr2],
    ]
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
    let r = rotation_matrix(poni);

    let mut hits = 0usize;
    for &(d1, d2) in peaks {
        let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
        let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
        let p3 = poni.distance;
        let x = p1 * r[0][0] + p2 * r[0][1] + p3 * r[0][2];
        let y = p1 * r[1][0] + p2 * r[1][1] + p3 * r[1][2];
        let z = p1 * r[2][0] + p2 * r[2][1] + p3 * r[2][2];
        let tth = (x * x + y * y).sqrt().atan2(z);
        if targets.iter().any(|&t| (tth - t).abs() <= tol_rad) {
            hits += 1;
        }
    }
    hits as f64 / peaks.len() as f64
}

// ---------- Step 1: Automatic valid-pixel mask ----------

fn subsample_median_and_mad(values: &[f64], predicate: impl Fn(f64) -> bool) -> (f64, f64) {
    let step = (values.len() / 16_384).max(1);
    let mut sample: Vec<f64> = values
        .iter()
        .step_by(step)
        .copied()
        .filter(|&v| predicate(v))
        .collect();
    if sample.is_empty() {
        return (0.0, 0.0);
    }
    let mid = sample.len() / 2;
    sample.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let med = sample[mid];
    let mut dev: Vec<f64> = sample.iter().map(|&v| (v - med).abs()).collect();
    dev.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    (med, 1.4826 * dev[mid])
}

fn build_valid_mask(
    image: &[f64],
    user_mask: Option<&[bool]>,
    dim1: usize,
    dim2: usize,
) -> (Vec<bool>, f64) {
    let n = dim1 * dim2;
    let (p50_finite, sigma_finite) =
        subsample_median_and_mad(image, |v| v.is_finite() && v < 4.0e9);
    let (p50_pos, _) = subsample_median_and_mad(image, |v| v.is_finite() && v > 0.0 && v < 4.0e9);

    // Positive-baseline images (e.g. photon-counting Pilatus/Eiger or CCD with dark offset)
    // use negative values as sentinel gap/bad-pixel markers (-1, -2). Dark-subtracted float
    // images have background noise fluctuating around 0, so we only mask extreme negative
    // outliers or contiguous exact -1/-2 gap lines there.
    let positive_baseline = p50_finite > 5.0 && (p50_finite - 4.0 * sigma_finite) > 0.0;
    let neg_outlier_floor = p50_finite - 10.0 * sigma_finite.max(1.0);
    let mask_zero_border = p50_pos > 20.0;

    let mut valid = vec![true; n];
    for idx in 0..n {
        if let Some(m) = user_mask
            && m[idx]
        {
            valid[idx] = false;
            continue;
        }
        let v = image[idx];
        if !v.is_finite() || v >= 4.0e9 {
            valid[idx] = false;
            continue;
        }
        if positive_baseline {
            if v < 0.0 {
                valid[idx] = false;
            }
        } else if v < neg_outlier_floor
            || ((v == -1.0 || v == -2.0) && has_same_cardinal_neighbor(image, dim1, dim2, idx, v))
        {
            valid[idx] = false;
        }
    }

    // Flood-fill exact-zero regions connected to the image border (e.g. MarCCD circular
    // active-area corners) and dilate by 1 px so no partial-zero rim stays valid.
    if mask_zero_border {
        let mut border_zero = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for j in 0..dim2 {
            for &idx in &[j, (dim1 - 1) * dim2 + j] {
                if image[idx] == 0.0 && !border_zero[idx] {
                    border_zero[idx] = true;
                    queue.push_back(idx);
                }
            }
        }
        for i in 1..(dim1 - 1) {
            for &idx in &[i * dim2, i * dim2 + dim2 - 1] {
                if image[idx] == 0.0 && !border_zero[idx] {
                    border_zero[idx] = true;
                    queue.push_back(idx);
                }
            }
        }
        while let Some(p) = queue.pop_front() {
            let i = p / dim2;
            let j = p % dim2;
            let nbrs = [
                (i > 0).then(|| p - dim2),
                (i + 1 < dim1).then(|| p + dim2),
                (j > 0).then(|| p - 1),
                (j + 1 < dim2).then(|| p + 1),
            ];
            for q in nbrs.into_iter().flatten() {
                if image[q] == 0.0 && !border_zero[q] {
                    border_zero[q] = true;
                    queue.push_back(q);
                }
            }
        }
        for i in 0..dim1 {
            let row = i * dim2;
            for j in 0..dim2 {
                let idx = row + j;
                if border_zero[idx] {
                    valid[idx] = false;
                    // 1-px dilation around border-zero regions.
                    if i > 0 {
                        valid[idx - dim2] = false;
                    }
                    if i + 1 < dim1 {
                        valid[idx + dim2] = false;
                    }
                    if j > 0 {
                        valid[idx - 1] = false;
                    }
                    if j + 1 < dim2 {
                        valid[idx + 1] = false;
                    }
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

#[inline]
fn has_same_cardinal_neighbor(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    idx: usize,
    val: f64,
) -> bool {
    let i = idx / dim2;
    let j = idx % dim2;
    (i > 0 && image[idx - dim2] == val)
        || (i + 1 < dim1 && image[idx + dim2] == val)
        || (j > 0 && image[idx - 1] == val)
        || (j + 1 < dim2 && image[idx + 1] == val)
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

// ---------- Step 3: Isotropic ring-center estimation + sharpness refinement ----------

fn radial_sharpness_score(
    iso_peaks: &[(f64, f64)],
    center_iso: (f64, f64),
    bin_width: f64,
    max_radius: f64,
) -> f64 {
    let n_bins = (max_radius / bin_width).ceil() as usize;
    if n_bins == 0 {
        return 0.0;
    }
    let mut hist = vec![0u32; n_bins];
    for &(y, x) in iso_peaks {
        let r = (y - center_iso.0).hypot(x - center_iso.1);
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

fn refine_center_sharpness(
    iso_peaks: &[(f64, f64)],
    init_iso: (f64, f64),
    max_radius: f64,
) -> (f64, f64) {
    let working: Vec<(f64, f64)> = if iso_peaks.len() > 1500 {
        let step = iso_peaks.len().div_ceil(1500);
        iso_peaks.iter().step_by(step).copied().collect()
    } else {
        iso_peaks.to_vec()
    };

    let mut best = init_iso;
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

// ---------- Step 4: 1D radial profiles (isotropic) + F1-scored distance scan ----------

struct RadialProfile1D {
    snr: Vec<f64>,
    r_min_obs: usize,
    r_max_obs: usize,
    peaks_r: Vec<(f64, f64)>, // (radius_iso, snr)
}

fn build_radial_profile_1d(
    image: &[f64],
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    center_iso: (f64, f64),
    scale1: f64,
    scale2: f64,
) -> Option<RadialProfile1D> {
    let corners = [
        (0.0, 0.0),
        (0.0, dim2 as f64 * scale2),
        (dim1 as f64 * scale1, 0.0),
        (dim1 as f64 * scale1, dim2 as f64 * scale2),
    ];
    let r_max = corners
        .iter()
        .map(|&(y, x)| (y - center_iso.0).hypot(x - center_iso.1))
        .fold(0.0_f64, f64::max)
        .ceil() as usize
        + 2;
    if !(20..=25_000).contains(&r_max) {
        return None;
    }

    let chunk_rows = (dim1 / rayon::current_num_threads().max(1)).max(32);
    let (sum, count) = (0..dim1)
        .into_par_iter()
        .chunks(chunk_rows)
        .map(|rows| {
            let mut local_sum = vec![0.0_f64; r_max];
            let mut local_cnt = vec![0u32; r_max];
            for i in rows {
                let dy = i as f64 * scale1 - center_iso.0;
                let dy2 = dy * dy;
                let row_off = i * dim2;
                for j in 0..dim2 {
                    let idx = row_off + j;
                    if !valid[idx] {
                        continue;
                    }
                    let dx = j as f64 * scale2 - center_iso.1;
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

    // Rolling 20th-percentile baseline over ±20 bins within valid range.
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
            let q = win.len() / 5;
            win.select_nth_unstable_by(q, |a, b| a.total_cmp(b));
            bg[r] = win[q];
        }
    }

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

    let mut snr_s = vec![0.0_f64; r_max];
    for r in (r_min_obs + 1)..r_max_obs {
        snr_s[r] = 0.25 * snr[r - 1] + 0.5 * snr[r] + 0.25 * snr[r + 1];
    }

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

/// Build a 1D radial profile from the *count* of detected 2D peaks per isotropic radial bin.
/// Naturally outlier-resistant for spotty / coarse-grain patterns with single-crystal spots.
fn build_peak_radial_profile_1d(
    iso_peaks: &[(f64, f64)],
    dim1: usize,
    dim2: usize,
    center_iso: (f64, f64),
    scale1: f64,
    scale2: f64,
) -> Option<RadialProfile1D> {
    if iso_peaks.len() < 15 {
        return None;
    }
    let corners = [
        (0.0, 0.0),
        (0.0, dim2 as f64 * scale2),
        (dim1 as f64 * scale1, 0.0),
        (dim1 as f64 * scale1, dim2 as f64 * scale2),
    ];
    let r_max = corners
        .iter()
        .map(|&(y, x)| (y - center_iso.0).hypot(x - center_iso.1))
        .fold(0.0_f64, f64::max)
        .ceil() as usize
        + 2;
    if !(20..=25_000).contains(&r_max) {
        return None;
    }

    let mut counts = vec![0.0_f64; r_max];
    let mut r_min_obs = r_max;
    let mut r_max_obs = 0usize;
    for &(y, x) in iso_peaks {
        let r = (y - center_iso.0).hypot(x - center_iso.1);
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

    let mut smooth = vec![0.0_f64; r_max];
    for r in 2..(r_max - 2) {
        smooth[r] = (counts[r - 2]
            + 4.0 * counts[r - 1]
            + 6.0 * counts[r]
            + 4.0 * counts[r + 1]
            + counts[r + 2])
            / 16.0;
    }

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
    iso_pixel_m: f64,
    d_min: f64,
    d_max: f64,
    starting_distance: Option<f64>,
    max_ring_index: usize,
    top_k: usize,
) -> Vec<(f64, f64)> {
    let mut tan_rings: Vec<f64> = calibrant
        .two_theta_rad(wavelength_m)
        .into_iter()
        .take(max_ring_index.max(10))
        .flatten()
        .map(|t| t.tan())
        .filter(|&t| t.is_finite() && t > 0.0)
        .collect();
    tan_rings.sort_by(|a, b| a.total_cmp(b));
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
        let scale = d / iso_pixel_m;

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
#[inline]
fn project_ring_pixel(poni: &Poni, tth: f64, phi: f64) -> Option<(f64, f64)> {
    let (st, ct) = tth.sin_cos();
    let (sp, cp) = phi.sin_cos();
    let ux = st * cp;
    let uy = st * sp;
    let uz = ct;

    let r = rotation_matrix(poni);

    // Detector-frame ray direction q = R^T * u.
    let q1 = r[0][0] * ux + r[1][0] * uy + r[2][0] * uz;
    let q2 = r[0][1] * ux + r[1][1] * uy + r[2][1] * uz;
    let q3 = r[0][2] * ux + r[1][2] * uy + r[2][2] * uz;

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

    for (ring_k, opt_tth) in tth_list.iter().enumerate().take(max_ring_index) {
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

            if best_val < min_peak_snr || (best_val - min_val) < min_peak_snr {
                continue;
            }

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
    passes: &[(usize, f64, usize, usize)],
) -> Option<(CalibrationResult, usize)> {
    let mut current_poni = init_poni;
    let mut best_result: Option<(CalibrationResult, usize)> = None;

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
                let supported = count_supported_rings(&res.assignments, 5);
                let replace = match &best_result {
                    None => true,
                    Some((prev, prev_rings)) => {
                        supported >= *prev_rings.min(&3) && res.rms_rad <= prev.rms_rad * 1.25
                    }
                };
                if replace {
                    best_result = Some((res, supported));
                }
            }
            _ => break,
        }
    }

    best_result
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
                let supported = count_supported_rings(&res.assignments, 5);
                if supported < 3 {
                    break;
                }
                current = res.poni.clone();
                last = Some((res, supported));
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
    // Work in an isotropic metric coordinate frame scaled by p_min so non-square
    // pixels (pixel1 != pixel2) still form circles for center estimation and 1D profiles.
    let p_min = pixel1.min(pixel2);
    let scale1 = pixel1 / p_min;
    let scale2 = pixel2 / p_min;
    let iso_peaks: Vec<(f64, f64)> = peaks
        .iter()
        .map(|&(d1, d2)| (d1 * scale1, d2 * scale2))
        .collect();

    let diag_iso = (dim1 as f64 * scale1).hypot(dim2 as f64 * scale2);
    let max_radius = (2.5 * diag_iso).max(2000.0);

    let ransac_peaks: Vec<(f64, f64)> = if iso_peaks.len() > 600 {
        let step = iso_peaks.len().div_ceil(600);
        iso_peaks.iter().step_by(step).copied().collect()
    } else {
        iso_peaks.clone()
    };
    let center_opts = RingCenterOptions {
        max_trials: 2500,
        radial_bin_width_px: 4.0,
        top_k_bins: 12,
        max_radius_px: max_radius,
        ..RingCenterOptions::default()
    };
    let coarse_center = estimate_ring_center(&ransac_peaks, &center_opts)?;
    let center_iso =
        refine_center_sharpness(&iso_peaks, (coarse_center.d1, coarse_center.d2), max_radius);

    // If distance is fixed and a starting_distance was supplied, use it directly.
    let mut dist_candidates: Vec<(f64, f64)> = Vec::new();
    let mut n_detected_1d = 0usize;
    if !options.refine_mask.distance
        && let Some(d0) = options.starting_distance
    {
        dist_candidates.push((d0, 1.0));
    } else {
        if let Some(profile) =
            build_radial_profile_1d(image, valid, dim1, dim2, center_iso, scale1, scale2)
        {
            n_detected_1d = n_detected_1d.max(profile.peaks_r.len());
            dist_candidates.extend(scan_distances_1d(
                &profile,
                calibrant,
                wavelength_m,
                p_min,
                d_min,
                d_max,
                options.starting_distance,
                options.max_ring_index,
                4,
            ));
        }
        if let Some(peak_profile) =
            build_peak_radial_profile_1d(&iso_peaks, dim1, dim2, center_iso, scale1, scale2)
        {
            n_detected_1d = n_detected_1d.max(peak_profile.peaks_r.len());
            for (d, sc) in scan_distances_1d(
                &peak_profile,
                calibrant,
                wavelength_m,
                p_min,
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
    }
    if dist_candidates.is_empty() {
        return None;
    }

    // Convert isotropic center back to detector metres (pyFAI pixel-center +0.5 convention).
    let bootstrap_p1 = center_iso.0 * p_min + 0.5 * pixel1;
    let bootstrap_p2 = center_iso.1 * p_min + 0.5 * pixel2;
    let passes = [
        (14usize, 3.0e-3, 96usize, 16usize),
        (7, 1.5e-3, 128, 22),
        (4, 8.0e-4, 128, 22),
    ];

    let has_hint = options.starting_distance.is_some();
    let mut best: Option<(
        CalibrationResult,
        AutoCalibrateDiagnostics,
        usize,
        f64,
        bool,
    )> = None;
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
            let cur_ok =
                is_verified_calibration(&res, n_rings, inlier_frac, d_min, d_max, has_hint);
            let quality = (inlier_frac.max(0.05) * (n_rings as f64).sqrt() * d_score.max(0.1))
                / (res.rms_rad + 3.0e-5);
            let is_better = match &best {
                None => true,
                Some((_, _, _, b_qual, best_ok)) => {
                    if cur_ok != *best_ok {
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
                best = Some((res, diag, n_rings, quality, cur_ok));
            }
        }
    }

    best.map(|(res, diag, n_rings, _, _)| (res, diag, n_rings))
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
    let strong_peaks: &[(f64, f64)] = &peaks[..peaks.len().min(800)];
    let cx = strong_peaks.iter().map(|p| p.0).sum::<f64>() / strong_peaks.len() as f64;
    let cy = strong_peaks.iter().map(|p| p.1).sum::<f64>() / strong_peaks.len() as f64;

    let (grid_d_min, grid_d_max, grid_d_steps) = match options.starting_distance {
        Some(d0) if d0.is_finite() && d0 > 0.0 => {
            if !options.refine_mask.distance {
                (d0, d0, 1)
            } else {
                let factor = options.distance_tolerance_factor.clamp(1.1, 1.6);
                ((d0 / factor).max(0.01), d0 * factor, 16)
            }
        }
        _ => {
            let hi = d_max.min(0.6).max(d_min * 1.5);
            (d_min, hi, 18)
        }
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
        max_ring_index: options.max_ring_index.min(12),
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
            let inlier_frac = peak_inlier_fraction(strong_peaks, &res.poni, calibrant, 2.0e-3);
            let score = inlier_frac / (res.rms_rad + 2.0e-4);
            if best_grid.as_ref().is_none_or(|(_, _, s)| score > *s) {
                best_grid = Some((res, cand, score));
            }
        }
    }
    let (grid_res, winning, grid_score) = best_grid?;

    let n_az = if is_strip_detector { 360 } else { 128 };
    let polish_ring_cap = options.max_ring_index.min(18);
    let polish_passes = [
        (8usize, 1.5e-3, n_az, polish_ring_cap),
        (4, 8.0e-4, n_az, polish_ring_cap),
    ];
    if let Some((polished, n_rings)) = iterative_2d_refine(
        residual,
        sigma_r,
        valid,
        dim1,
        dim2,
        grid_res.poni.clone(),
        calibrant,
        options,
        polish_ring_cap,
        &polish_passes,
    ) {
        let pol_inlier_frac = peak_inlier_fraction(strong_peaks, &polished.poni, calibrant, 2.0e-3);
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

    let n_rings = count_supported_rings(&grid_res.assignments, 5);
    let diag = AutoCalibrateDiagnostics {
        n_detected_ellipses: candidates.len(),
        n_identified_ellipses: n_rings,
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
                    let r = rotation_matrix(&poni);
                    let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
                    let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
                    let p3 = poni.distance;
                    let x = p1 * r[0][0] + p2 * r[0][1] + p3 * r[0][2];
                    let y = p1 * r[1][0] + p2 * r[1][1] + p3 * r[1][2];
                    let z = p1 * r[2][0] + p2 * r[2][1] + p3 * r[2][2];
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
