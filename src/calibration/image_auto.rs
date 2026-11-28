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
use super::grid_search::{GridSearchOptions, grid_search_top_k, grid_search_top_k_beam_centered};
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
    /// extracting 2D ring peaks (default `100`; raise if even higher-order
    /// rings are visible on the detector).
    pub max_ring_index: usize,
    /// Optional upper intensity limit (same units as `image`). Pixels with
    /// values `>= limit` are excluded during masking (useful for rejecting
    /// saturated pixels or bright single-crystal spots, e.g. `Some(55_000.0)`).
    /// When `None` (default), non-finite values, uint32 dead-pixel sentinels
    /// (`>= 4e9`), and auto-detected flat-topped saturation plateaus
    /// (`>= 65_000`) are excluded automatically.
    pub intensity_limit: Option<f64>,
    /// Optional starting MAD-based peak detection threshold multiplier
    /// (`med + factor * sigma`). When `None` (default), starts at `3.5σ`.
    /// Either way, `extract_robust_peaks` uses locally normalized SNR on large
    /// detectors and adapts the threshold automatically if coverage is sparse
    /// or flooded into background noise.
    pub peak_threshold_sigma: Option<f64>,
    /// Optional base half-width (pixels) for the 2D ring-profile peak search.
    /// When `None` (default), the search width adapts automatically to the
    /// measured 1D ring width and is clamped per-ring to `< 0.45` of the local
    /// inter-ring spacing so adjacent dense/overlapping rings cannot steal peaks.
    pub search_half_width_px: Option<usize>,
    /// Optional linear X-ray polarization factor in `[-1, 1]` (e.g. `Some(0.99)`
    /// for horizontal synchrotron polarization). When set with `|P| >= 0.05`,
    /// `rot3` is calibrated post-refinement from the azimuthal ring intensity
    /// distribution via [`super::calibrate_rot3_polarization`].
    pub polarization_factor: Option<f64>,
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
            max_ring_index: 100,
            intensity_limit: None,
            peak_threshold_sigma: None,
            search_half_width_px: None,
            polarization_factor: None,
            max_iter: 200,
            tol: 1e-9,
        }
    }
}

/// Stage of the end-to-end image auto-calibration pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoCalibrateStage {
    /// Building valid-pixel mask and detecting image-wide peaks.
    DetectingPeaks,
    /// Estimating isotropic ring center via 3-point circle RANSAC + sharpness refinement.
    EstimatingCenter,
    /// Scanning 1D radial profiles for candidate sample-to-detector distances.
    ScanningDistances,
    /// Refining candidate geometries against 2D ring profiles and detected peaks.
    RefiningCandidate,
    /// Running coarse 5D grid search fallback (tilted / strip detector).
    GridSearch,
    /// Polishing the best grid-search geometry against 2D ring profiles.
    PolishingGrid,
    /// Calibration verified and complete.
    Done,
}

/// Live progress and intermediate plottable diagnostics emitted during
/// [`auto_calibrate_image_with_progress`].
#[derive(Debug, Clone)]
pub struct AutoCalibrateProgress {
    pub stage: AutoCalibrateStage,
    /// Human-readable status message describing the current step.
    pub message: String,
    /// Fractional progress in `[0.0, 1.0]`.
    pub progress: f64,
    /// Subsampled detected peaks `(d1, d2)` in pixel coordinates so a UI can
    /// plot the points the algorithm is using.
    pub peaks: Option<Vec<(f64, f64)>>,
    /// Estimated ring center `(d1, d2)` in pixel coordinates once found.
    pub ring_center_px: Option<(f64, f64)>,
    /// Best candidate `Poni` found so far (for live ring overlay previews).
    pub candidate_poni: Option<Poni>,
    /// RMS 2θ residual (radians) of `candidate_poni`, if available.
    pub candidate_rms_rad: Option<f64>,
    /// Number of supported calibrant rings for `candidate_poni`, if available.
    pub n_rings: Option<usize>,
}

fn subsample_preview_peaks(peaks: &[(f64, f64)], max_count: usize) -> Vec<(f64, f64)> {
    if peaks.len() <= max_count {
        peaks.to_vec()
    } else {
        let step = peaks.len().div_ceil(max_count);
        peaks.iter().step_by(step).copied().collect()
    }
}

/// Configuration for [`extract_ring_peaks`] and [`refine_image`].
#[derive(Debug, Clone)]
pub struct RefineImageOptions {
    /// Parameter mask for Levenberg–Marquardt refinement.
    /// Defaults to wavelength and `rot3` fixed, `(distance, poni1, poni2, rot1, rot2)` free.
    /// Note: `rot3` is always kept fixed during `2θ` refinement (since `2θ` is
    /// invariant to `rot3`); pass `polarization_factor` to calibrate `rot3`.
    pub refine_mask: ParamMask,
    /// Maximum calibrant ring index searched (default `100`).
    pub max_ring_index: usize,
    /// Optional cap on the number of visible calibrant rings included.
    /// When `None` (default), refines progressively and then automatically
    /// extends to every geometrically visible resolved ring on the detector
    /// (up to `max_ring_index`). When `Some(k)`, strictly respects `k`.
    pub max_rings: Option<usize>,
    /// Number of azimuthal sectors sampled around each ring (default `72`).
    pub n_azimuth: usize,
    /// Optional upper intensity limit (`None` auto-detects flat-topped
    /// saturation plateaus `>= 65_000`).
    pub intensity_limit: Option<f64>,
    /// Optional base radial search half-width in pixels (`>= 3`). When `None`
    /// (default), auto-selects from detector size and shrinks across refinement
    /// passes, skipping unresolved `< 5 px` doublets and clamping each ring to
    /// `<= 0.45 × local_ring_spacing`.
    pub search_half_width_px: Option<usize>,
    /// Number of iterative re-extraction + L-M polish passes in [`refine_image`]
    /// (default `3`, clamped to `1..=6`).
    pub n_passes: usize,
    /// Optional linear X-ray polarization factor in `[-1, 1]` (e.g. `Some(0.99)`).
    /// Values with `|P| >= 0.05` run [`super::calibrate_rot3_polarization`]
    /// after refinement and update `poni.rot3` if `reliable` is true. Values
    /// with `|P| < 0.05` skip `rot3` calibration; non-finite or `|P| > 1`
    /// return `InvalidInput`.
    pub polarization_factor: Option<f64>,
    /// Maximum L-M iterations per pass (default `200`).
    pub max_iter: usize,
    /// L-M convergence tolerance (default `1e-9`).
    pub tol: f64,
}

impl Default for RefineImageOptions {
    fn default() -> Self {
        let mut refine_mask = ParamMask::fix_wavelength();
        refine_mask.rot3 = false;
        Self {
            refine_mask,
            max_ring_index: 100,
            max_rings: None,
            n_azimuth: 72,
            intensity_limit: None,
            search_half_width_px: None,
            n_passes: 3,
            polarization_factor: None,
            max_iter: 200,
            tol: 1e-9,
        }
    }
}

/// Output of [`refine_image`].
#[derive(Debug, Clone)]
pub struct RefineImageResult {
    /// Final Levenberg–Marquardt calibration result (with `poni.rot3` updated
    /// when `rot3_calibration` is present and `reliable`).
    pub calibration: CalibrationResult,
    /// Sub-pixel ring control points `(d1, d2, Some(ring_k))` extracted in the
    /// final pass (suitable for plotting or further inspection).
    pub points: CalibrationPoints,
    /// Number of calibrant rings with `>= 5` matched control points.
    pub n_rings: usize,
    /// RMS `2θ` residual (radians) of `start_poni` evaluated on `points`, for
    /// verifying that refinement improved upon the starting geometry.
    pub start_rms_rad: Option<f64>,
    /// Result of post-refinement polarization `rot3` calibration, when
    /// `options.polarization_factor` was set with `|P| >= 0.05` and sufficient
    /// signal existed.
    pub rot3_calibration: Option<super::polarization::PolarizationRot3Result>,
}

/// Extract sub-pixel control points uniformly around all visible calibrant
/// rings predicted by `poni`, using the mask-aware DoG residual and `8×8`
/// local-MAD noise model (`2.5 × sigma_loc` prominence).
pub fn extract_ring_peaks(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    calibrant: &Calibrant,
    options: &RefineImageOptions,
) -> Result<CalibrationPoints, AutoCalibrateError> {
    validate_image_and_poni(image, dim1, dim2, mask, poni, options)?;
    let explicit_upper = options
        .intensity_limit
        .filter(|v| v.is_finite() && *v > 0.0);
    let (valid, _, _) = build_valid_mask(image, mask, dim1, dim2, explicit_upper);
    let (_smooth, residual, noise, sigma_r) = compute_residual_and_noise(image, &valid, dim1, dim2);
    if sigma_r < 1e-12 {
        return Ok(CalibrationPoints::new());
    }
    let hw = options
        .search_half_width_px
        .unwrap_or_else(|| ((dim1.max(dim2) as f64 / 70.0).round() as usize).clamp(10, 24));
    let n_az = options.n_azimuth.clamp(12, 360);
    let max_ring_index = options.max_ring_index.max(1);
    let max_rings = match options.max_rings {
        Some(k) => k.max(1),
        None => detector_visible_ring_limit(poni, dim1, dim2, calibrant)
            .min(max_ring_index)
            .max(1),
    };
    let (pts, _) = extract_ring_peaks_2d_with_tangent(
        &residual,
        &noise,
        sigma_r,
        &valid,
        dim1,
        dim2,
        poni,
        calibrant,
        hw,
        n_az,
        max_rings,
        max_ring_index,
        0,
        None,
    );
    Ok(pts)
}

/// Precomputed mask, DoG residual, and local noise grid for repeated ring-peak
/// extraction across iterative distortion-correction passes without recomputing
/// image filters on each pass.
pub(crate) struct PreparedRingImage {
    dim1: usize,
    dim2: usize,
    valid: Vec<bool>,
    residual: Vec<f64>,
    noise: LocalNoiseGrid,
    sigma_r: f64,
}

impl PreparedRingImage {
    pub(crate) fn prepare(
        image: &[f64],
        mask: Option<&[bool]>,
        dim1: usize,
        dim2: usize,
        explicit_upper: Option<f64>,
    ) -> Option<Self> {
        let (valid, _, _) = build_valid_mask(image, mask, dim1, dim2, explicit_upper);
        let (_smooth, residual, noise, sigma_r) =
            compute_residual_and_noise(image, &valid, dim1, dim2);
        if sigma_r < 1e-12 {
            return None;
        }
        Some(Self {
            dim1,
            dim2,
            valid,
            residual,
            noise,
            sigma_r,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn extract_with_distortion(
        &self,
        poni: &Poni,
        calibrant: &Calibrant,
        half_width_px: usize,
        n_az: usize,
        max_rings: Option<usize>,
        max_ring_index: usize,
        tangent_half_px: usize,
        distortion: Option<&super::distortion::DistortionField>,
    ) -> (CalibrationPoints, usize) {
        let max_idx = max_ring_index.max(1);
        let eff_max_rings = match max_rings {
            Some(k) => k.max(1),
            None => detector_visible_ring_limit(poni, self.dim1, self.dim2, calibrant)
                .min(max_idx)
                .max(1),
        };
        extract_ring_peaks_2d_with_tangent(
            &self.residual,
            &self.noise,
            self.sigma_r,
            &self.valid,
            self.dim1,
            self.dim2,
            poni,
            calibrant,
            half_width_px,
            n_az,
            eff_max_rings,
            max_idx,
            tangent_half_px,
            distortion,
        )
    }
}

/// Evaluate RMS `2θ` residual (radians) of `poni` against already-assigned
/// `points`, returning `None` if no valid ring points exist.
fn eval_points_rms_rad(
    poni: &Poni,
    dim1: usize,
    dim2: usize,
    points: &CalibrationPoints,
    calibrant: &Calibrant,
) -> Option<f64> {
    let tths = calibrant.two_theta_rad(poni.wavelength);
    let geom = Geometry::new(dim1, dim2, poni.clone());
    let tc = crate::geometry::math::TrigonometricConstants::new(poni);
    let p3 = poni.distance;
    let mut sum2 = 0.0;
    let mut cnt = 0usize;
    let _ = geom;
    for i in 0..points.len() {
        let Some(rk) = points.ring[i] else {
            continue;
        };
        let Some(Some(target)) = tths.get(rk) else {
            continue;
        };
        let p1 = (points.d1[i] + 0.5) * poni.pixel1 - poni.poni1;
        let p2 = (points.d2[i] + 0.5) * poni.pixel2 - poni.poni2;
        let x1 = crate::geometry::math::f_x1(&p1, &p2, &p3, &tc);
        let x2 = crate::geometry::math::f_x2(&p1, &p2, &p3, &tc);
        let x3 = crate::geometry::math::f_x3(&p1, &p2, &p3, &tc);
        let pred = crate::geometry::math::f_tth(&x1, &x2, &x3);
        let r = pred - target;
        sum2 += r * r;
        cnt += 1;
    }
    (cnt > 0).then(|| (sum2 / cnt as f64).sqrt())
}

/// Refine an existing `start_poni` directly against a 2D diffraction image
/// using iterative local-MAD 2D ring-profile peak extraction + Huber-weighted
/// Levenberg–Marquardt, with optional post-refinement polarization `rot3`
/// calibration.
///
/// Designed for interactive "Refine" workflows (after initial manual or
/// automatic calibration, or when refining a loaded `.poni` against a new
/// calibrant image) with zero `intensity_min_factor` tuning.
pub fn refine_image(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    start_poni: &Poni,
    calibrant: &Calibrant,
    options: &RefineImageOptions,
) -> Result<RefineImageResult, AutoCalibrateError> {
    validate_image_and_poni(image, dim1, dim2, mask, start_poni, options)?;
    let explicit_upper = options
        .intensity_limit
        .filter(|v| v.is_finite() && *v > 0.0);
    let (valid, _, _) = build_valid_mask(image, mask, dim1, dim2, explicit_upper);
    let (smooth, residual, noise, sigma_r) = compute_residual_and_noise(image, &valid, dim1, dim2);
    if sigma_r < 1e-12 {
        return Err(AutoCalibrateError::InvalidInput(
            "image has no valid non-constant pixels".into(),
        ));
    }

    let mut refine_mask = options.refine_mask;
    // 2θ is invariant to rot3; keep rot3 locked during L-M so finite-difference
    // noise cannot drift it (rot3 is calibrated via `polarization_factor`).
    refine_mask.rot3 = false;

    let n_passes = options.n_passes.clamp(1, 6);
    let n_az = options.n_azimuth.clamp(12, 360);
    let max_ring_index = options.max_ring_index.max(1);
    let base_max_rings = options.max_rings.map(|k| k.max(1)).unwrap_or(16);
    let base_hw = options
        .search_half_width_px
        .unwrap_or_else(|| ((dim1.max(dim2) as f64 / 70.0).round() as usize).clamp(10, 24));

    let run_2d_passes = |init: Poni,
                         hw0: usize,
                         passes_cnt: usize|
     -> Option<(CalibrationResult, CalibrationPoints, usize)> {
        let mut p = init;
        let mut best: Option<(CalibrationResult, CalibrationPoints, usize)> = None;
        let hw_floor = hw0.min(6);
        for pass in 0..passes_cnt {
            let hw = if passes_cnt == 1 {
                hw0
            } else {
                let frac = 1.0 - 0.55 * (pass as f64 / (passes_cnt - 1) as f64);
                ((hw0 as f64 * frac).round() as usize).clamp(hw_floor, hw0)
            };
            let pass_rings = if passes_cnt >= 3 && pass == 0 {
                base_max_rings.min(8)
            } else if passes_cnt >= 3 && pass == 1 {
                base_max_rings.min(12)
            } else {
                base_max_rings
            };
            let (pts, _) = extract_ring_peaks_2d(
                &residual,
                &noise,
                sigma_r,
                &valid,
                dim1,
                dim2,
                &p,
                calibrant,
                hw,
                n_az,
                pass_rings,
                max_ring_index,
            );
            if pts.len() < 15 {
                break;
            }
            let huber_c = if pass == 0 { 2.5e-3 } else { 1.0e-3 };
            let cal_opts = CalibrationOptions {
                mask: refine_mask,
                loss: Loss::Huber { c: huber_c },
                auto_assign_rings: false,
                assign_tolerance_rad: 0.05,
                max_iter: options.max_iter,
                tol: options.tol,
            };
            let geom = Geometry::new(dim1, dim2, p.clone());
            match refine(&geom, &pts, calibrant, &cal_opts) {
                Ok(res) if res.poni.distance.is_finite() && res.poni.distance > 0.005 => {
                    let nr = count_supported_rings(&res.assignments, 5);
                    if nr < 2 {
                        break;
                    }
                    let replace = match &best {
                        None => true,
                        Some((prev, _, prev_rings)) => {
                            if nr > *prev_rings {
                                res.rms_rad <= (prev.rms_rad * 1.6).max(5.0e-4)
                            } else {
                                nr >= (*prev_rings).min(3) && res.rms_rad <= prev.rms_rad * 1.25
                            }
                        }
                    };
                    if replace {
                        p = res.poni.clone();
                        best = Some((res, pts, nr));
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        best
    };

    let max_px_shift = (0.12 * dim1.max(dim2) as f64).max(40.0);
    let verify_fit = |cal: &CalibrationResult,
                      pts: &CalibrationPoints,
                      nr: usize,
                      v_peaks: &[(f64, f64)]|
     -> bool {
        if nr < 2 || cal.rms_rad > 3.5e-3 {
            return false;
        }
        let rel_d_shift = (cal.poni.distance / start_poni.distance - 1.0).abs();
        let dp1_shift = (cal.poni.poni1 - start_poni.poni1).abs() / start_poni.pixel1;
        let dp2_shift = (cal.poni.poni2 - start_poni.poni2).abs() / start_poni.pixel2;
        if rel_d_shift > 0.15 || dp1_shift.max(dp2_shift) > max_px_shift {
            return false;
        }
        if let Some(s_rms) = eval_points_rms_rad(start_poni, dim1, dim2, pts, calibrant)
            && s_rms < 1.5e-3
            && cal.rms_rad > s_rms * 1.5 + 2.0e-4
        {
            return false;
        }
        if v_peaks.len() >= 15 {
            let m = evaluate_peak_match(
                v_peaks,
                &valid,
                dim1,
                dim2,
                &cal.poni,
                calibrant,
                2.0e-3,
                max_ring_index,
            );
            if m.inlier_frac < 0.40 || m.ring_recall < 0.55 || m.inlier_rms_rad > 1.2e-3 {
                return false;
            }
        }
        true
    };

    let mut best_fit = run_2d_passes(start_poni.clone(), base_hw, n_passes);
    let (verify_peaks, _) = scan_robust_peaks(
        image, &smooth, &residual, &noise, sigma_r, &valid, dim1, dim2, 5000, None,
    );

    // If 2D azimuthal ring profiling alone didn't lock tightly or failed
    // verification (e.g. a spotty pattern, dense calibrant, or larger initial
    // offset), pull in using the already-computed peaks + auto ring assignment,
    // then polish with 2D ring profiles.
    let need_peak_pull = match &best_fit {
        Some((res, pts, nr)) => {
            *nr < 6 || res.rms_rad >= 1.5e-3 || !verify_fit(res, pts, *nr, &verify_peaks)
        }
        None => true,
    };
    if need_peak_pull && verify_peaks.len() >= 15 {
        let mut pk_pts = CalibrationPoints::new();
        for &(d1, d2) in &verify_peaks {
            pk_pts.push(d1, d2, None);
        }
        let mut pk_poni = start_poni.clone();
        let mut pk_ok = false;
        for &(assign_tol, huber_c) in &[(1.2e-2, 3.0e-3), (4.5e-3, 1.5e-3), (2.5e-3, 8.0e-4)] {
            let geom = Geometry::new(dim1, dim2, pk_poni.clone());
            let opts_pk = CalibrationOptions {
                mask: refine_mask,
                loss: Loss::Huber { c: huber_c },
                auto_assign_rings: true,
                assign_tolerance_rad: assign_tol,
                max_iter: options.max_iter,
                tol: options.tol,
            };
            if let Ok(res) = refine(&geom, &pk_pts, calibrant, &opts_pk)
                && res.poni.distance.is_finite()
                && res.poni.distance > 0.005
            {
                pk_poni = res.poni;
                pk_ok = true;
            }
        }
        if pk_ok && let Some((res2, pts2, nr2)) = run_2d_passes(pk_poni, base_hw.min(10), 2) {
            let score = |nr: usize, rms: f64| (nr.min(16) as f64).sqrt() / (rms + 1.0e-4);
            let better = match &best_fit {
                None => true,
                Some((prev, prev_pts, prev_nr)) => {
                    let prev_ok = verify_fit(prev, prev_pts, *prev_nr, &verify_peaks);
                    let new_ok = verify_fit(&res2, &pts2, nr2, &verify_peaks);
                    (new_ok && !prev_ok)
                        || (new_ok == prev_ok
                            && score(nr2, res2.rms_rad) > score(*prev_nr, prev.rms_rad))
                }
            };
            if better {
                best_fit = Some((res2, pts2, nr2));
            }
        }
    }

    let (mut calibration, mut points, mut n_rings) =
        best_fit.ok_or(AutoCalibrateError::NoConfidentIdentity)?;

    if !verify_fit(&calibration, &points, n_rings, &verify_peaks) {
        return Err(AutoCalibrateError::NoConfidentIdentity);
    }

    // Automatically extend to every geometrically visible resolved ring on the
    // detector (up to `max_ring_index`) when the caller did not set `max_rings`.
    if options.max_rings.is_none()
        && let Some((ext_cal, ext_pts, ext_rings)) = extend_to_all_detector_rings(
            &calibration,
            Some(&points),
            n_rings,
            max_ring_index,
            &residual,
            &noise,
            sigma_r,
            &valid,
            dim1,
            dim2,
            calibrant,
            refine_mask,
            n_az,
            options.max_iter,
            options.tol,
        )
    {
        calibration = ext_cal;
        points = ext_pts;
        n_rings = ext_rings;
    }

    // Compute `start_rms_rad` on the final returned `points` so it is directly
    // comparable to `calibration.rms_rad`.
    let start_rms_rad = eval_points_rms_rad(start_poni, dim1, dim2, &points, calibrant);

    let mut rot3_calibration = None;
    if let Some(pf) = options.polarization_factor
        && pf.abs() >= 0.05
    {
        let inv_mask: Vec<bool> = valid.iter().map(|&v| !v).collect();
        if let Ok(pol) = super::polarization::calibrate_rot3_polarization(
            image,
            dim1,
            dim2,
            Some(&inv_mask),
            &calibration.poni,
            calibrant,
            pf,
            &super::polarization::PolarizationRot3Options::default(),
        ) {
            if pol.reliable {
                calibration.poni.rot3 = pol.rot3_rad;
            }
            rot3_calibration = Some(pol);
        }
    }

    Ok(RefineImageResult {
        calibration,
        points,
        n_rings,
        start_rms_rad,
        rot3_calibration,
    })
}

fn validate_image_and_poni(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    options: &RefineImageOptions,
) -> Result<(), AutoCalibrateError> {
    let expected_len = dim1
        .checked_mul(dim2)
        .ok_or_else(|| AutoCalibrateError::InvalidInput("image dimensions overflow".into()))?;
    if image.len() != expected_len {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "image length {} does not match {}x{}",
            image.len(),
            dim1,
            dim2
        )));
    }
    if dim1 < 16 || dim2 < 16 {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "image too small ({dim1}x{dim2}; need at least 16x16)"
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
    if let Some(hw) = options.search_half_width_px
        && hw < 3
    {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "search_half_width_px must be >= 3, got {hw}"
        )));
    }
    if let Some(pf) = options.polarization_factor
        && (!pf.is_finite() || pf.abs() > 1.0)
    {
        return Err(AutoCalibrateError::InvalidInput(format!(
            "polarization_factor must be finite and in [-1, 1], got {pf}"
        )));
    }
    if !poni.distance.is_finite()
        || poni.distance <= 0.0
        || !poni.wavelength.is_finite()
        || poni.wavelength <= 0.0
        || !poni.pixel1.is_finite()
        || poni.pixel1 <= 0.0
        || !poni.pixel2.is_finite()
        || poni.pixel2 <= 0.0
    {
        return Err(AutoCalibrateError::InvalidInput(
            "poni has non-finite or non-positive distance, wavelength, or pixel size".into(),
        ));
    }
    Ok(())
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
/// Convenience wrapper around [`auto_calibrate_image_with_progress`] without a
/// progress callback.
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
    auto_calibrate_image_with_progress(
        image,
        dim1,
        dim2,
        mask,
        calibrant,
        wavelength_m,
        pixel1,
        pixel2,
        options,
        |_| true,
    )
}

/// Automatically calibrate a 2D powder diffraction image with an optional pixel
/// mask and a live progress / cancellation callback.
///
/// `on_progress` is invoked at major pipeline stages with an
/// [`AutoCalibrateProgress`] snapshot containing human-readable status text,
/// detected peaks, estimated ring center, and intermediate candidate `Poni`
/// geometries suitable for live plotting. Returning `false` from `on_progress`
/// cancels calibration early with [`AutoCalibrateError::Cancelled`].
pub fn auto_calibrate_image_with_progress<F>(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &AutoCalibrateImageOptions,
    mut on_progress: F,
) -> Result<(CalibrationResult, AutoCalibrateDiagnostics), AutoCalibrateError>
where
    F: FnMut(AutoCalibrateProgress) -> bool,
{
    // 2θ residuals are invariant under rot3 (rotation around the incident
    // beam), so leaving rot3 free in L-M introduces a zero Jacobian column
    // that drifts rot3 randomly from finite-difference noise. Always keep
    // rot3 locked during 2θ refinement; post-refinement polarization
    // calibration (`polarization_factor`) updates rot3 when reliable.
    let mut locked_options = options.clone();
    locked_options.refine_mask.rot3 = false;
    auto_calibrate_image_impl(
        image,
        dim1,
        dim2,
        mask,
        calibrant,
        wavelength_m,
        pixel1,
        pixel2,
        &locked_options,
        &mut on_progress,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn auto_calibrate_image_impl(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &AutoCalibrateImageOptions,
    mut on_progress: &mut dyn FnMut(AutoCalibrateProgress) -> bool,
    allow_retry: bool,
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

    // 1. Automatic valid-pixel mask (combined with optional user mask and
    //    explicit or auto-detected upper saturation limit).
    let explicit_upper = options
        .intensity_limit
        .filter(|v| v.is_finite() && *v > 0.0);
    let (valid, valid_median, effective_upper) =
        build_valid_mask(image, mask, dim1, dim2, explicit_upper);

    // 2. Robust image-wide peak extraction with local noise normalization &
    //    uniformity-guided adaptive threshold.
    let (peaks, residual, noise, sigma_r, used_sigma) = extract_robust_peaks(
        image,
        &valid,
        valid_median,
        dim1,
        dim2,
        5000,
        options.peak_threshold_sigma,
    );
    if peaks.len() < 15 {
        return Err(AutoCalibrateError::NoEllipses);
    }

    let sat_note = if effective_upper < 3.9e9 {
        format!(", upper limit {:.0}", effective_upper)
    } else {
        String::new()
    };
    let sigma_note = match options.peak_threshold_sigma.filter(|s| s.is_finite()) {
        Some(s0) if (used_sigma - s0).abs() > 0.05 => {
            format!("adapted {:.1}σ → {:.1}σ", s0, used_sigma)
        }
        _ => format!("{:.1}σ", used_sigma),
    };
    if !on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::DetectingPeaks,
        message: format!(
            "Detected {} peaks ({}{})",
            peaks.len(),
            sigma_note,
            sat_note
        ),
        progress: 0.15,
        peaks: Some(subsample_preview_peaks(&peaks, 600)),
        ring_center_px: None,
        candidate_poni: None,
        candidate_rms_rad: None,
        n_rings: None,
    }) {
        return Err(AutoCalibrateError::Cancelled);
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
            &noise,
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
            &mut on_progress,
        )?
    };

    let radial_metrics = radial_attempt
        .as_ref()
        .map(|(res, _, _)| {
            evaluate_peak_match(
                &peaks,
                &valid,
                dim1,
                dim2,
                &res.poni,
                calibrant,
                VERIFY_TOL_RAD,
                options.max_ring_index,
            )
        })
        .unwrap_or_default();

    let radial_verified = radial_attempt.as_ref().is_some_and(|(res, _, n_rings)| {
        is_verified_calibration(
            res,
            *n_rings,
            &radial_metrics,
            peaks.len(),
            d_min,
            d_max,
            options.starting_distance.is_some(),
        )
    });

    let radial_tight = radial_attempt
        .as_ref()
        .is_some_and(|(res, _, n_rings)| res.rms_rad < 4.5e-4 && *n_rings >= 4);

    if !is_strip_detector
        && radial_verified
        && radial_tight
        && radial_metrics.inlier_frac >= 0.55
        && radial_metrics.ring_recall >= 0.65
        && radial_metrics.inlier_rms_rad < 6.5e-4
        && let Some((res, diag, n_rings)) = radial_attempt
    {
        return Ok(finalize_auto_calibration(
            res,
            diag,
            n_rings,
            "Verified",
            image,
            &peaks,
            &radial_metrics,
            &residual,
            &noise,
            sigma_r,
            &valid,
            dim1,
            dim2,
            calibrant,
            options,
            on_progress,
        ));
    }

    // 4. Tilt / strip-detector fallback via coarse 5D grid search + 2D image polish.
    if options.enable_tilt_fallback
        && let Some((grid_res, grid_diag, grid_rings)) = try_grid_fallback(
            &residual,
            &noise,
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
            &mut on_progress,
        )?
    {
        let grid_metrics = evaluate_peak_match(
            &peaks,
            &valid,
            dim1,
            dim2,
            &grid_res.poni,
            calibrant,
            VERIFY_TOL_RAD,
            options.max_ring_index,
        );
        let grid_verified = is_verified_calibration(
            &grid_res,
            grid_rings,
            &grid_metrics,
            peaks.len(),
            d_min,
            d_max,
            options.starting_distance.is_some(),
        );
        if grid_verified {
            let use_grid = match &radial_attempt {
                None => true,
                Some((r_res, _, r_rings)) => {
                    if !radial_verified {
                        true
                    } else {
                        let g_rms = grid_res.rms_rad + 0.35 * grid_metrics.inlier_rms_rad;
                        let r_rms = r_res.rms_rad + 0.35 * radial_metrics.inlier_rms_rad;
                        let g_score = (grid_metrics.inlier_frac.max(0.05)
                            * grid_metrics.ring_recall.max(0.05)
                            * (grid_rings as f64).min(24.0).sqrt())
                            / (g_rms + 5.0e-5);
                        let r_score = (radial_metrics.inlier_frac.max(0.05)
                            * radial_metrics.ring_recall.max(0.05)
                            * (*r_rings as f64).min(24.0).sqrt())
                            / (r_rms + 5.0e-5);
                        g_score > r_score
                    }
                }
            };
            if use_grid {
                return Ok(finalize_auto_calibration(
                    grid_res,
                    grid_diag,
                    grid_rings,
                    "Verified tilted fit",
                    image,
                    &peaks,
                    &grid_metrics,
                    &residual,
                    &noise,
                    sigma_r,
                    &valid,
                    dim1,
                    dim2,
                    calibrant,
                    options,
                    on_progress,
                ));
            }
        }
    }

    // Return the radial result only if it actually passed verification.
    if radial_verified && let Some((res, diag, n_rings)) = radial_attempt {
        return Ok(finalize_auto_calibration(
            res,
            diag,
            n_rings,
            "Verified",
            image,
            &peaks,
            &radial_metrics,
            &residual,
            &noise,
            sigma_r,
            &valid,
            dim1,
            dim2,
            calibrant,
            options,
            on_progress,
        ));
    }

    // 5. Automatic parameter restart when primary calibration fails
    //    verification (e.g. faint rings missed at 3.5σ, bright rings clipped
    //    by auto-saturation, or too many outer rings confusing initial search).
    let can_retry =
        allow_retry && (used_sigma > 2.1 || (explicit_upper.is_none() && effective_upper < 3.9e9));
    if can_retry {
        let mut retry_opts = options.clone();
        retry_opts.peak_threshold_sigma = Some(2.0);
        retry_opts.max_ring_index = options.max_ring_index.min(12);
        if explicit_upper.is_none() && effective_upper < 3.9e9 {
            retry_opts.intensity_limit = Some(4.0e9);
        }
        if !on_progress(AutoCalibrateProgress {
            stage: AutoCalibrateStage::DetectingPeaks,
            message:
                "Primary fit failed verification — retrying with relaxed threshold and inner rings…"
                    .into(),
            progress: 0.15,
            peaks: None,
            ring_center_px: None,
            candidate_poni: None,
            candidate_rms_rad: None,
            n_rings: None,
        }) {
            return Err(AutoCalibrateError::Cancelled);
        }
        match auto_calibrate_image_impl(
            image,
            dim1,
            dim2,
            mask,
            calibrant,
            wavelength_m,
            pixel1,
            pixel2,
            &retry_opts,
            on_progress,
            false,
        ) {
            Ok((res, diag)) => return Ok((res, diag)),
            Err(AutoCalibrateError::Cancelled) => return Err(AutoCalibrateError::Cancelled),
            Err(_) => {}
        }
    }

    Err(AutoCalibrateError::NoConfidentIdentity)
}

/// Compute how many calibrant rings geometrically intersect the detector
/// (all rings with `2θ <= tth_max_det`, where `tth_max_det` is the maximum `2θ`
/// across the detector's corners and edge midpoints).
fn detector_visible_ring_limit(
    poni: &Poni,
    dim1: usize,
    dim2: usize,
    calibrant: &Calibrant,
) -> usize {
    let tc = crate::geometry::math::TrigonometricConstants::new(poni);
    let p3 = poni.distance;
    let r_hi = dim1.saturating_sub(1) as f64;
    let c_hi = dim2.saturating_sub(1) as f64;
    let r_mid = 0.5 * r_hi;
    let c_mid = 0.5 * c_hi;
    let probes = [
        (0.0, 0.0),
        (0.0, c_mid),
        (0.0, c_hi),
        (r_mid, 0.0),
        (r_mid, c_hi),
        (r_hi, 0.0),
        (r_hi, c_mid),
        (r_hi, c_hi),
    ];
    let mut tth_max = 0.0_f64;
    for &(r, c) in &probes {
        let p1 = (r + 0.5) * poni.pixel1 - poni.poni1;
        let p2 = (c + 0.5) * poni.pixel2 - poni.poni2;
        let x1 = crate::geometry::math::f_x1(&p1, &p2, &p3, &tc);
        let x2 = crate::geometry::math::f_x2(&p1, &p2, &p3, &tc);
        let x3 = crate::geometry::math::f_x3(&p1, &p2, &p3, &tc);
        let t = crate::geometry::math::f_tth(&x1, &x2, &x3);
        if t.is_finite() && t > tth_max {
            tth_max = t;
        }
    }
    let tths = calibrant.two_theta_rad(poni.wavelength);
    let mut last_idx = 0usize;
    for (k, opt_t) in tths.iter().enumerate() {
        if let Some(t) = opt_t
            && t.is_finite()
            && *t > 1e-4
            && *t <= tth_max
        {
            last_idx = k + 1;
        }
    }
    last_idx
}

/// After calibration converges, automatically determine how many calibrant
/// rings geometrically fall on the detector (up to `max_ring_index`) and
/// extend the 2D ring-profile polish across all resolved visible rings using
/// 2 shrinking passes with azimuthal tangent averaging (`√7×` SNR boost on
/// faint outer arcs), never degrading the verified geometry.
#[allow(clippy::too_many_arguments)]
fn extend_to_all_detector_rings(
    prev: &CalibrationResult,
    prev_points: Option<&CalibrationPoints>,
    prev_n_rings: usize,
    max_ring_index: usize,
    residual: &[f64],
    noise: &LocalNoiseGrid,
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    calibrant: &Calibrant,
    mut refine_mask: ParamMask,
    n_az: usize,
    max_iter: usize,
    tol: f64,
) -> Option<(CalibrationResult, CalibrationPoints, usize)> {
    // 2θ is invariant to rot3; never let rot3 drift during ring extension.
    refine_mask.rot3 = false;

    let n_det = detector_visible_ring_limit(&prev.poni, dim1, dim2, calibrant).min(max_ring_index);
    if n_det <= prev_n_rings {
        return None;
    }

    let mut cur_poni = prev.poni.clone();
    let mut final_res: Option<(CalibrationResult, CalibrationPoints, usize)> = None;
    for &(hw, huber_c) in &[(6usize, 1.0e-3), (4usize, 7.0e-4)] {
        let (pts, _) = extract_ring_peaks_2d_with_tangent(
            residual, noise, sigma_r, valid, dim1, dim2, &cur_poni, calibrant, hw, n_az, n_det,
            n_det, 3, None,
        );
        if pts.len() < 15 {
            break;
        }
        let geom = Geometry::new(dim1, dim2, cur_poni.clone());
        let cal_opts = CalibrationOptions {
            mask: refine_mask,
            loss: Loss::Huber { c: huber_c },
            auto_assign_rings: false,
            assign_tolerance_rad: 0.05,
            max_iter,
            tol,
        };
        let Ok(ext) = refine(&geom, &pts, calibrant, &cal_opts) else {
            break;
        };
        if !ext.poni.distance.is_finite() || ext.poni.distance <= 0.005 {
            break;
        }
        let ext_rings = count_supported_rings(&ext.assignments, 5);
        cur_poni = ext.poni.clone();
        final_res = Some((ext, pts, ext_rings));
    }

    let (ext, pts, ext_rings) = final_res?;
    let rel_d = (ext.poni.distance / prev.poni.distance - 1.0).abs();
    let dp1 = (ext.poni.poni1 - prev.poni.poni1).abs() / prev.poni.pixel1;
    let dp2 = (ext.poni.poni2 - prev.poni.poni2).abs() / prev.poni.pixel2;
    let inner_ok = match prev_points {
        Some(pp) => {
            match (
                eval_points_rms_rad(&prev.poni, dim1, dim2, pp, calibrant),
                eval_points_rms_rad(&ext.poni, dim1, dim2, pp, calibrant),
            ) {
                (Some(r_prev), Some(r_ext)) => r_ext <= (r_prev * 1.25).max(r_prev + 1.0e-4),
                _ => true,
            }
        }
        None => true,
    };
    let rms_ok = ext.rms_rad <= (prev.rms_rad * 2.2).max(1.2e-3);
    if ext_rings > prev_n_rings && inner_ok && rms_ok && rel_d < 0.015 && dp1.max(dp2) < 15.0 {
        Some((ext, pts, ext_rings))
    } else {
        None
    }
}

fn finalize_auto_calibration(
    mut res: CalibrationResult,
    mut diag: AutoCalibrateDiagnostics,
    mut n_rings: usize,
    label: &str,
    image: &[f64],
    peaks: &[(f64, f64)],
    prev_metrics: &PeakMatchMetrics,
    residual: &[f64],
    noise: &LocalNoiseGrid,
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    calibrant: &Calibrant,
    options: &AutoCalibrateImageOptions,
    on_progress: &mut dyn FnMut(AutoCalibrateProgress) -> bool,
) -> (CalibrationResult, AutoCalibrateDiagnostics) {
    if let Some((ext_res, _ext_pts, ext_rings)) = extend_to_all_detector_rings(
        &res,
        None,
        n_rings,
        options.max_ring_index,
        residual,
        noise,
        sigma_r,
        valid,
        dim1,
        dim2,
        calibrant,
        options.refine_mask,
        72,
        options.max_iter,
        options.tol,
    ) {
        // Re-verify against the detected peaks so outer-ring extension can
        // never degrade the already-verified inner calibration.
        let ext_metrics = evaluate_peak_match(
            peaks,
            valid,
            dim1,
            dim2,
            &ext_res.poni,
            calibrant,
            2.0e-3,
            options.max_ring_index,
        );
        let non_regression = ext_metrics.inlier_rms_rad
            <= (prev_metrics.inlier_rms_rad * 1.15).max(prev_metrics.inlier_rms_rad + 5.0e-5)
            && ext_metrics.ring_recall >= prev_metrics.ring_recall - 0.02
            && ext_metrics.inlier_frac >= prev_metrics.inlier_frac - 0.02;
        if non_regression {
            res = ext_res;
            n_rings = ext_rings;
            diag.n_identified_ellipses = ext_rings;
        }
    }

    let mut rot3_note = String::new();
    if let Some(pf) = options.polarization_factor
        && pf.is_finite()
        && pf.abs() >= 0.05
        && pf.abs() <= 1.0
    {
        let inv_mask: Vec<bool> = valid.iter().map(|&v| !v).collect();
        if let Ok(pol) = super::polarization::calibrate_rot3_polarization(
            image,
            dim1,
            dim2,
            Some(&inv_mask),
            &res.poni,
            calibrant,
            pf,
            &super::polarization::PolarizationRot3Options::default(),
        ) && pol.reliable
        {
            res.poni.rot3 = pol.rot3_rad;
            rot3_note = format!(", rot3 {:.2}°", pol.rot3_rad.to_degrees());
        }
    }
    let _ = on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::Done,
        message: format!(
            "{label} ({n_rings} rings, RMS {:.4}°{rot3_note})",
            res.rms_rad.to_degrees()
        ),
        progress: 1.0,
        peaks: None,
        ring_center_px: None,
        candidate_poni: Some(res.poni.clone()),
        candidate_rms_rad: Some(res.rms_rad),
        n_rings: Some(n_rings),
    });
    (res, diag)
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

#[derive(Debug, Clone, Copy, Default)]
struct PeakMatchMetrics {
    inlier_frac: f64,
    inlier_rms_rad: f64,
    ring_recall: f64,
}

fn is_verified_calibration(
    res: &CalibrationResult,
    n_rings: usize,
    metrics: &PeakMatchMetrics,
    n_total_peaks: usize,
    d_min: f64,
    d_max: f64,
    has_distance_hint: bool,
) -> bool {
    if !res.poni.distance.is_finite()
        || res.poni.distance <= 0.005
        || !res.rms_rad.is_finite()
        || res.rms_rad >= 1.5e-3
        || n_rings < 3
        || metrics.inlier_frac < 0.45
        || metrics.ring_recall < 0.55
        || !metrics.inlier_rms_rad.is_finite()
        || metrics.inlier_rms_rad >= 1.5e-3
    {
        return false;
    }
    let min_assigned = ((n_total_peaks * 2) / 3).clamp(15, 40);
    if res.assignments.iter().filter(|a| a.is_some()).count() < min_assigned {
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

/// Evaluate unbiased image-wide peak agreement with a candidate geometry:
///
/// - Uses a spacing-aware tolerance `min(tol_rad, 0.35 * local_ring_spacing)` so
///   artificially small distances with dense high-order rings cannot score 100%
///   inliers by chance.
/// - Computes `ring_recall` over the predicted calibrant rings inside the
///   observed 2θ span `[p02, p98]` (weighted by calibrant intensity when known),
///   catching sub-multiple aliases (e.g. `D / √2` on LaB6 where half of the
///   predicted rings are completely absent).
fn evaluate_peak_match(
    peaks: &[(f64, f64)],
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    calibrant: &Calibrant,
    tol_rad: f64,
    max_ring_index: usize,
) -> PeakMatchMetrics {
    if peaks.is_empty() {
        return PeakMatchMetrics::default();
    }
    let max_cal_i = calibrant
        .intensities
        .iter()
        .filter_map(|&v| v)
        .fold(0.0_f64, f64::max);

    let mut raw_targets: Vec<(f64, f64)> = calibrant
        .two_theta_rad(poni.wavelength)
        .into_iter()
        .enumerate()
        .take(max_ring_index.max(35))
        .filter_map(|(idx, opt_t)| {
            let t = opt_t?;
            if !t.is_finite() || t <= 1e-4 {
                return None;
            }
            let w = if max_cal_i > 0.0 {
                calibrant
                    .intensities
                    .get(idx)
                    .copied()
                    .flatten()
                    .map(|v| (v / max_cal_i).sqrt().max(0.15))
                    .unwrap_or(0.25)
            } else {
                1.0
            };
            Some((t, w))
        })
        .collect();
    if raw_targets.is_empty() {
        return PeakMatchMetrics::default();
    }
    raw_targets.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n_t = raw_targets.len();
    // (tth, local_tol, weight)
    let mut targets: Vec<(f64, f64, f64)> = Vec::with_capacity(n_t);
    for k in 0..n_t {
        let gap_lo = if k > 0 {
            raw_targets[k].0 - raw_targets[k - 1].0
        } else {
            f64::INFINITY
        };
        let gap_hi = if k + 1 < n_t {
            raw_targets[k + 1].0 - raw_targets[k].0
        } else {
            f64::INFINITY
        };
        let spacing = gap_lo.min(gap_hi);
        let local_tol = if spacing.is_finite() {
            tol_rad.min(0.35 * spacing).max(2.0e-4)
        } else {
            tol_rad
        };
        targets.push((raw_targets[k].0, local_tol, raw_targets[k].1));
    }

    let r = rotation_matrix(poni);
    let mut hits = 0usize;
    let mut sq_err_sum = 0.0_f64;
    let mut ring_hits = vec![0usize; n_t];
    let mut obs_tths = Vec::with_capacity(peaks.len());

    for &(d1, d2) in peaks {
        let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
        let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
        let p3 = poni.distance;
        let x = p1 * r[0][0] + p2 * r[0][1] + p3 * r[0][2];
        let y = p1 * r[1][0] + p2 * r[1][1] + p3 * r[1][2];
        let z = p1 * r[2][0] + p2 * r[2][1] + p3 * r[2][2];
        let tth = (x * x + y * y).sqrt().atan2(z);
        if !tth.is_finite() {
            continue;
        }
        obs_tths.push(tth);

        let pos = targets.partition_point(|&(t, _, _)| t < tth);
        let mut best_k = usize::MAX;
        let mut best_err = f64::INFINITY;
        for cand_k in [pos.saturating_sub(1), pos] {
            if cand_k < n_t {
                let err = (tth - targets[cand_k].0).abs();
                if err < best_err {
                    best_err = err;
                    best_k = cand_k;
                }
            }
        }
        if best_k != usize::MAX && best_err <= targets[best_k].1 {
            hits += 1;
            sq_err_sum += best_err * best_err;
            ring_hits[best_k] += 1;
        }
    }

    if hits == 0 || obs_tths.is_empty() {
        return PeakMatchMetrics::default();
    }

    let inlier_frac = hits as f64 / peaks.len() as f64;
    let inlier_rms_rad = (sq_err_sum / hits as f64).sqrt();

    // Compute observed 2θ span [p02, p98] (robust to a few stray outlier spots
    // and beamstop-masked inner regions).
    obs_tths.sort_by(|a, b| a.total_cmp(b));
    let n_obs = obs_tths.len();
    let tth_lo = obs_tths[(n_obs / 50).min(n_obs - 1)];
    let tth_hi = obs_tths[((n_obs * 49) / 50).min(n_obs - 1)];

    let min_hits_for_ring = 2usize;
    // Direct beam position in pixel coordinates (for checking whether an inner
    // missing ring lies well outside any central beamstop).
    let beam_px = project_ring_pixel(poni, 0.0, 0.0);
    let mut visible_w = 0.0_f64;
    let mut hit_w = 0.0_f64;
    let mut n_checked = 0usize;
    for (k, &(tth_k, local_tol, w)) in targets.iter().enumerate() {
        if tth_k - local_tol > tth_hi {
            break;
        }
        if tth_k + local_tol < tth_lo {
            // Check whether this inner ring projects cleanly onto valid
            // detector pixels outside a central beamstop (~40 px radius).
            // Catches sub-multiple aliases (e.g. D/√2 on LaB6) where the
            // innermost predicted ring is completely absent.
            let mut n_valid_az = 0usize;
            for a in 0..16 {
                let phi = 2.0 * PI * (a as f64) / 16.0;
                if let Some((d1, d2)) = project_ring_pixel(poni, tth_k, phi)
                    && d1 >= 4.0
                    && d2 >= 4.0
                    && d1 < (dim1 as f64 - 5.0)
                    && d2 < (dim2 as f64 - 5.0)
                {
                    let far_from_beam = beam_px
                        .map(|(b1, b2)| (d1 - b1).hypot(d2 - b2) >= 40.0)
                        .unwrap_or(true);
                    let idx = (d1.round() as usize) * dim2 + (d2.round() as usize);
                    if far_from_beam && valid[idx] {
                        n_valid_az += 1;
                    }
                }
            }
            if n_valid_az < 8 {
                continue;
            }
        }
        visible_w += w;
        if ring_hits[k] >= min_hits_for_ring {
            hit_w += w;
        }
        n_checked += 1;
        if n_checked >= 16 {
            break;
        }
    }
    let ring_recall = if visible_w > 0.0 {
        hit_w / visible_w
    } else {
        0.0
    };

    PeakMatchMetrics {
        inlier_frac,
        inlier_rms_rad,
        ring_recall,
    }
}

/// Select strong calibrant rings (sorted by 2θ) for the 1D radial distance
/// scan. When relative intensities are available, drops very weak reflections
/// (`< 3%` of max) so dense 100–600 ring calibrants don't create false aliases,
/// while still honoring `max_ring_index` for high-order-only geometries.
fn select_initial_tth_rings(
    calibrant: &Calibrant,
    wavelength_m: f64,
    max_ring_index: usize,
) -> Vec<f64> {
    let raw = calibrant.two_theta_rad(wavelength_m);
    let intensities = &calibrant.intensities;
    let has_intensities = intensities.iter().any(|v| v.is_some());
    let pairs: Vec<(f64, f64)> = raw
        .into_iter()
        .take(max_ring_index)
        .enumerate()
        .filter_map(|(k, opt_t)| {
            let t = opt_t.filter(|v| v.is_finite() && *v > 1e-4 && *v < PI * 0.49)?;
            let w = intensities.get(k).copied().flatten().unwrap_or(100.0);
            Some((t, w))
        })
        .collect();
    if pairs.is_empty() {
        return Vec::new();
    }
    let max_i = pairs.iter().map(|p| p.1).fold(0.0_f64, f64::max);
    if has_intensities && max_i > 0.0 {
        let strong: Vec<f64> = pairs
            .iter()
            .filter(|&&(_, w)| w >= 0.03 * max_i)
            .map(|&(t, _)| t)
            .collect();
        if strong.len() >= 6 {
            return strong;
        }
    }
    pairs.into_iter().map(|(t, _)| t).collect()
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
    explicit_upper_limit: Option<f64>,
) -> (Vec<bool>, f64, f64) {
    let n = dim1 * dim2;
    // Auto-detect flat-topped hardware saturation plateaus at any bit depth
    // (12-bit 4095, 14-bit 16383, 16-bit 65535, 20-bit, or dark-subtracted
    // ceilings) when the caller did not pass an explicit `intensity_limit`.
    // Skips user-masked pixels and requires >= 8 cardinally-adjacent pixel
    // pairs near `max_v` (`>= 0.995 * max_v`, `|v - nb| <= 0.005 * max_v`) so
    // unsaturated bright 32-bit/summed powder rings are not falsely masked.
    let upper_limit = match explicit_upper_limit {
        Some(lim) => lim,
        None => {
            let max_v = (0..n)
                .into_par_iter()
                .filter_map(|k| {
                    if user_mask.is_some_and(|m| m[k]) {
                        return None;
                    }
                    let v = image[k];
                    (v.is_finite() && v < 4.0e9).then_some(v)
                })
                .reduce(|| 0.0_f64, f64::max);
            if max_v >= 4_000.0 && dim1 >= 2 && dim2 >= 2 {
                let sat_thresh = 0.995 * max_v;
                let flat_eps = 0.005 * max_v;
                let flat_pairs: usize = (0..dim1)
                    .into_par_iter()
                    .map(|i| {
                        let row = i * dim2;
                        let mut pairs = 0usize;
                        for j in 0..dim2 {
                            let idx = row + j;
                            if user_mask.is_some_and(|m| m[idx]) {
                                continue;
                            }
                            let v = image[idx];
                            if v.is_finite() && v < 4.0e9 && v >= sat_thresh {
                                if j + 1 < dim2
                                    && !user_mask.is_some_and(|m| m[idx + 1])
                                    && (v - image[idx + 1]).abs() <= flat_eps
                                {
                                    pairs += 1;
                                }
                                if i + 1 < dim1
                                    && !user_mask.is_some_and(|m| m[idx + dim2])
                                    && (v - image[idx + dim2]).abs() <= flat_eps
                                {
                                    pairs += 1;
                                }
                            }
                        }
                        pairs
                    })
                    .sum();
                if flat_pairs >= 8 {
                    0.995 * max_v
                } else {
                    4.0e9
                }
            } else {
                4.0e9
            }
        }
    };
    let (p50_finite, sigma_finite) =
        subsample_median_and_mad(image, |v| v.is_finite() && v < upper_limit);
    let (p50_pos, _) =
        subsample_median_and_mad(image, |v| v.is_finite() && v > 0.0 && v < upper_limit);

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
        if !v.is_finite() || v >= upper_limit {
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

    // Automatically detect internal exact-zero module gap rows/columns (e.g.
    // Pilatus/Eiger/Jungfrau gaps encoded as 0.0 rather than -1.0) on images
    // with a positive background baseline. Single row-major pass for cache locality.
    if p50_pos > 10.0 && dim1 >= 32 && dim2 >= 32 {
        let row_zero_thresh = (dim2 * 3) / 4;
        let col_zero_thresh = (dim1 * 3) / 4;
        let mut gap_rows = vec![false; dim1];
        let mut col_zeros = vec![0usize; dim2];
        for i in 0..dim1 {
            let row = i * dim2;
            let mut zc = 0usize;
            for j in 0..dim2 {
                if image[row + j] == 0.0 {
                    zc += 1;
                    col_zeros[j] += 1;
                }
            }
            if zc >= row_zero_thresh {
                gap_rows[i] = true;
            }
        }
        let gap_cols: Vec<bool> = col_zeros
            .into_iter()
            .map(|c| c >= col_zero_thresh)
            .collect();
        for i in 0..dim1 {
            if gap_rows[i] {
                let lo = i.saturating_sub(1);
                let hi = (i + 1).min(dim1 - 1);
                for ii in lo..=hi {
                    let row = ii * dim2;
                    for j in 0..dim2 {
                        valid[row + j] = false;
                    }
                }
            }
        }
        for j in 0..dim2 {
            if gap_cols[j] {
                let lo = j.saturating_sub(1);
                let hi = (j + 1).min(dim2 - 1);
                for i in 0..dim1 {
                    let row = i * dim2;
                    for jj in lo..=hi {
                        valid[row + jj] = false;
                    }
                }
            }
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

    (valid, valid_median, upper_limit)
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

/// 1D horizontal sliding-window box blur (`f32` storage, `f64` accumulator)
/// with clamp-to-edge boundaries. Strictly O(1) per pixel regardless of
/// `radius`, parallelized across rows.
fn box_blur_horizontal_f32(src: &[f32], dst: &mut [f32], dim1: usize, dim2: usize, radius: usize) {
    if dim1 == 0 || dim2 == 0 {
        return;
    }
    let w = 2 * radius + 1;
    let inv_w = 1.0 / w as f64;
    dst.par_chunks_mut(dim2)
        .enumerate()
        .for_each(|(i, out_row)| {
            let in_row = &src[i * dim2..(i + 1) * dim2];
            let mut acc = in_row[0] as f64 * (radius + 1) as f64;
            for k in 1..=radius {
                acc += in_row[k.min(dim2 - 1)] as f64;
            }
            let last = dim2 - 1;
            for x in 0..dim2 {
                out_row[x] = (acc * inv_w) as f32;
                let add_idx = (x + radius + 1).min(last);
                let sub_idx = x.saturating_sub(radius);
                acc += in_row[add_idx] as f64 - in_row[sub_idx] as f64;
            }
        });
}

/// Cache-friendly tiled parallel transpose `(dim1 × dim2) -> (dim2 × dim1)`.
fn transpose_f32(src: &[f32], dst: &mut [f32], dim1: usize, dim2: usize) {
    const TILE: usize = 32;
    dst.par_chunks_mut(TILE * dim1)
        .enumerate()
        .for_each(|(tile_j, dst_strip)| {
            let j0 = tile_j * TILE;
            let j_end = (j0 + TILE).min(dim2);
            for i0 in (0..dim1).step_by(TILE) {
                let i_end = (i0 + TILE).min(dim1);
                for i in i0..i_end {
                    let src_row = i * dim2;
                    for j in j0..j_end {
                        dst_strip[(j - j0) * dim1 + i] = src[src_row + j];
                    }
                }
            }
        });
}

/// Fast, low-memory 3-pass separable box blur (`f32`, reusing 2 scratch buffers)
/// approximating a Gaussian of standard deviation `sigma`.
fn fast_gaussian_blur_bg_f32(src: &[f32], dim1: usize, dim2: usize, sigma: f64) -> Vec<f32> {
    let r = ((sigma * sigma + 0.25).sqrt() - 0.5).round().max(1.0) as usize;
    let n = dim1 * dim2;
    let mut a = vec![0.0_f32; n];
    let mut b = vec![0.0_f32; n];

    // 3 horizontal passes: src -> a -> b -> a
    box_blur_horizontal_f32(src, &mut a, dim1, dim2, r);
    box_blur_horizontal_f32(&a, &mut b, dim1, dim2, r);
    box_blur_horizontal_f32(&b, &mut a, dim1, dim2, r);

    // Transpose a -> b, run 3 horizontal passes on columns (b -> a -> b -> a), transpose a -> b.
    transpose_f32(&a, &mut b, dim1, dim2);
    box_blur_horizontal_f32(&b, &mut a, dim2, dim1, r);
    box_blur_horizontal_f32(&a, &mut b, dim2, dim1, r);
    box_blur_horizontal_f32(&b, &mut a, dim2, dim1, r);
    transpose_f32(&a, &mut b, dim2, dim1);
    b
}

/// Select up to `max_peaks` strongest peaks while guaranteeing spatial
/// representation across an `8×8` detector tile grid for genuinely strong
/// peaks (`>= tile_min_score`), so bright inner rings or single-crystal spots
/// cannot starve fainter outer rings without filling empty tiles with noise.
fn select_spatially_balanced_peaks(
    mut candidates: Vec<(f64, f64, f64)>,
    dim1: usize,
    dim2: usize,
    max_peaks: usize,
    tile_min_score: f64,
) -> Vec<(f64, f64)> {
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    if candidates.len() <= max_peaks {
        return candidates.into_iter().map(|(d1, d2, _)| (d1, d2)).collect();
    }

    const TILES: usize = 8;
    let per_tile_quota = (max_peaks / (TILES * TILES)).max(16);
    let mut tile_counts = [0usize; TILES * TILES];
    let mut selected = Vec::with_capacity(max_peaks);
    let mut overflow = Vec::new();

    for &(d1, d2, score) in &candidates {
        let ti = ((d1 / dim1 as f64) * TILES as f64).clamp(0.0, (TILES - 1) as f64) as usize;
        let tj = ((d2 / dim2 as f64) * TILES as f64).clamp(0.0, (TILES - 1) as f64) as usize;
        let t = ti * TILES + tj;
        if score >= tile_min_score && tile_counts[t] < per_tile_quota && selected.len() < max_peaks
        {
            tile_counts[t] += 1;
            selected.push((d1, d2, score));
        } else {
            overflow.push((d1, d2, score));
        }
    }

    let remaining = max_peaks.saturating_sub(selected.len());
    selected.extend(overflow.into_iter().take(remaining));
    selected.sort_by(|a, b| b.2.total_cmp(&a.2));
    selected.into_iter().map(|(d1, d2, _)| (d1, d2)).collect()
}

/// Coarse `8×8` spatially-varying residual noise model (`med`, `sigma`) with
/// bilinear interpolation across tile centers. Prevents bright/noisy regions
/// near an off-center beam from inflating the threshold and blinding fainter
/// sections of the same Debye–Scherrer ring.
struct LocalNoiseGrid {
    dim1: usize,
    dim2: usize,
    med: [f64; 64],
    sigma: [f64; 64],
    sigma_unclamped: [f64; 64],
    valid_tiles: usize,
    tile_valid: [bool; 64],
}

impl LocalNoiseGrid {
    const TILES: usize = 8;

    fn build(
        residual: &[f64],
        valid: &[bool],
        dim1: usize,
        dim2: usize,
        global_med: f64,
        global_sigma: f64,
    ) -> Self {
        let mut buckets: Vec<Vec<f64>> = (0..64).map(|_| Vec::with_capacity(512)).collect();
        // 2D-strided sampling (~180×180 grid) so every tile is sampled evenly
        // across both rows and columns (avoids 1D-stride column aliasing on
        // power-of-two detectors like 2048×2048 or 4096×4096).
        let row_step = (dim1 / 180).max(1);
        let col_step = (dim2 / 180).max(1);
        let mut i = 0;
        while i < dim1 {
            let ti = ((i * Self::TILES) / dim1.max(1)).min(Self::TILES - 1);
            let row_off = i * dim2;
            let mut j = 0;
            while j < dim2 {
                let k = row_off + j;
                if valid[k] {
                    let tj = ((j * Self::TILES) / dim2.max(1)).min(Self::TILES - 1);
                    buckets[ti * Self::TILES + tj].push(residual[k]);
                }
                j += col_step;
            }
            i += row_step;
        }

        let use_local = dim1.min(dim2) >= 768;
        let sigma_lo = 0.65 * global_sigma;
        let sigma_lo_raw = 0.30 * global_sigma;
        let sigma_hi = 2.5 * global_sigma;
        let mut med = [global_med; 64];
        let mut sigma = [global_sigma; 64];
        let mut sigma_unclamped = [global_sigma; 64];
        let mut tile_valid = [false; 64];
        let mut valid_tiles = 0usize;

        for t in 0..64 {
            let b = &mut buckets[t];
            if b.len() >= 24 {
                tile_valid[t] = true;
                valid_tiles += 1;
                if use_local {
                    let mid = b.len() / 2;
                    b.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                    let m = b[mid];
                    let mut dev: Vec<f64> = b.iter().map(|&v| (v - m).abs()).collect();
                    dev.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                    let raw_s = 1.4826 * dev[mid];
                    med[t] = m;
                    sigma[t] = raw_s.clamp(sigma_lo, sigma_hi);
                    sigma_unclamped[t] = raw_s.clamp(sigma_lo_raw, sigma_hi);
                }
            }
        }

        Self {
            dim1,
            dim2,
            med,
            sigma,
            sigma_unclamped,
            valid_tiles: valid_tiles.max(1),
            tile_valid,
        }
    }

    /// Bilinearly interpolated `(med, sigma)` at pixel `(i, j)`.
    #[inline]
    fn at(&self, i: f64, j: f64) -> (f64, f64) {
        self.interp(i, j, &self.sigma)
    }

    /// Bilinearly interpolated `(med, sigma_unclamped)` at pixel `(i, j)` for
    /// known-geometry outer-ring extension into quiet detector regions.
    #[inline]
    fn at_unclamped(&self, i: f64, j: f64) -> (f64, f64) {
        self.interp(i, j, &self.sigma_unclamped)
    }

    #[inline]
    fn interp(&self, i: f64, j: f64, sig_grid: &[f64; 64]) -> (f64, f64) {
        let fy = (i / self.dim1.max(1) as f64) * Self::TILES as f64 - 0.5;
        let fx = (j / self.dim2.max(1) as f64) * Self::TILES as f64 - 0.5;
        let fy = fy.clamp(0.0, (Self::TILES - 1) as f64);
        let fx = fx.clamp(0.0, (Self::TILES - 1) as f64);
        let y0 = fy.floor() as usize;
        let x0 = fx.floor() as usize;
        let y1 = (y0 + 1).min(Self::TILES - 1);
        let x1 = (x0 + 1).min(Self::TILES - 1);
        let wy = fy - y0 as f64;
        let wx = fx - x0 as f64;

        let i00 = y0 * Self::TILES + x0;
        let i01 = y0 * Self::TILES + x1;
        let i10 = y1 * Self::TILES + x0;
        let i11 = y1 * Self::TILES + x1;

        let m_top = self.med[i00] * (1.0 - wx) + self.med[i01] * wx;
        let m_bot = self.med[i10] * (1.0 - wx) + self.med[i11] * wx;
        let s_top = sig_grid[i00] * (1.0 - wx) + sig_grid[i01] * wx;
        let s_bot = sig_grid[i10] * (1.0 - wx) + sig_grid[i11] * wx;
        (
            m_top * (1.0 - wy) + m_bot * wy,
            s_top * (1.0 - wy) + s_bot * wy,
        )
    }

    #[inline]
    fn tile_index(&self, d1: f64, d2: f64) -> usize {
        let ti = ((d1 / self.dim1.max(1) as f64) * Self::TILES as f64)
            .clamp(0.0, (Self::TILES - 1) as f64) as usize;
        let tj = ((d2 / self.dim2.max(1) as f64) * Self::TILES as f64)
            .clamp(0.0, (Self::TILES - 1) as f64) as usize;
        ti * Self::TILES + tj
    }
}

fn compute_residual_and_noise(
    image: &[f64],
    valid: &[bool],
    dim1: usize,
    dim2: usize,
) -> (Vec<f64>, Vec<f64>, LocalNoiseGrid, f64) {
    let n = dim1 * dim2;
    // Mask-aware normalized convolution: multiply by validity mask, blur both
    // signal and mask, then divide. Buffers are dropped promptly and the large
    // background blur runs sequentially in f32 (each blur already saturates
    // all Rayon threads) to keep peak RSS low on 4096×4096 detectors.
    let (smooth, bg_mask_f32): (Vec<f64>, Vec<f32>) = {
        let masked_img: Vec<f64> = image
            .par_iter()
            .zip(valid.par_iter())
            .map(|(&v, &ok)| if ok { v } else { 0.0 })
            .collect();
        let smooth_num = gaussian_blur(&masked_img, dim1, dim2, 0.8);
        drop(masked_img);

        let mask_f64: Vec<f64> = valid
            .par_iter()
            .map(|&ok| if ok { 1.0 } else { 0.0 })
            .collect();
        let smooth_den = gaussian_blur(&mask_f64, dim1, dim2, 0.8);
        drop(mask_f64);

        let mut smooth_out = vec![0.0_f64; n];
        let mut bg_mask = vec![0.0_f32; n];
        smooth_out
            .par_iter_mut()
            .zip(bg_mask.par_iter_mut())
            .enumerate()
            .for_each(|(k, (s, m))| {
                if valid[k] && smooth_den[k] > 0.25 {
                    *s = smooth_num[k] / smooth_den[k];
                    *m = 1.0_f32;
                }
            });
        (smooth_out, bg_mask)
    };

    let sigma_bg = (dim1.max(dim2) as f64 / 80.0).clamp(6.0, 20.0);
    let residual: Vec<f64> = {
        let smooth_f32: Vec<f32> = smooth.par_iter().map(|&v| v as f32).collect();
        let bg_num = fast_gaussian_blur_bg_f32(&smooth_f32, dim1, dim2, sigma_bg);
        drop(smooth_f32);
        let bg_den = fast_gaussian_blur_bg_f32(&bg_mask_f32, dim1, dim2, sigma_bg);
        drop(bg_mask_f32);

        smooth
            .par_iter()
            .zip(bg_num.par_iter())
            .zip(bg_den.par_iter())
            .zip(valid.par_iter())
            .map(|(((&s, &bn), &bd), &ok)| {
                if ok && bd > 0.15 {
                    s - (bn / bd) as f64
                } else {
                    0.0
                }
            })
            .collect()
    };

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
        let noise = LocalNoiseGrid::build(&residual, valid, dim1, dim2, 0.0, 0.0);
        return (smooth, residual, noise, 0.0);
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
    let noise = LocalNoiseGrid::build(&residual, valid, dim1, dim2, med_r, sigma_r);
    (smooth, residual, noise, sigma_r)
}

fn extract_robust_peaks(
    image: &[f64],
    valid: &[bool],
    _valid_median: f64,
    dim1: usize,
    dim2: usize,
    max_peaks: usize,
    peak_threshold_sigma: Option<f64>,
) -> (Vec<(f64, f64)>, Vec<f64>, LocalNoiseGrid, f64, f64) {
    let (smooth, residual, noise, sigma_r) = compute_residual_and_noise(image, valid, dim1, dim2);
    if sigma_r < 1e-12 {
        return (Vec::new(), residual, noise, 0.0, 3.5);
    }
    let (peaks, used_sigma) = scan_robust_peaks(
        image,
        &smooth,
        &residual,
        &noise,
        sigma_r,
        valid,
        dim1,
        dim2,
        max_peaks,
        peak_threshold_sigma,
    );
    (peaks, residual, noise, sigma_r, used_sigma)
}

#[allow(clippy::too_many_arguments)]
fn scan_robust_peaks(
    image: &[f64],
    smooth: &[f64],
    residual: &[f64],
    noise: &LocalNoiseGrid,
    sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    max_peaks: usize,
    peak_threshold_sigma: Option<f64>,
) -> (Vec<(f64, f64)>, f64) {
    let n = dim1 * dim2;

    // Estimate raw (unsmoothed) pixel noise σ from |image - smooth| MAD so
    // isolated single-pixel hot-pixel rejection scales with raw noise rather
    // than the ~3× smaller smoothed residual σ.
    let step = (n / 32_768).max(1);
    let mut raw_dev = Vec::with_capacity(32_768);
    let mut k = 0;
    while k < n {
        if valid[k] {
            raw_dev.push((image[k] - smooth[k]).abs());
        }
        k += step;
    }
    let sigma_raw = if !raw_dev.is_empty() {
        let rmid = raw_dev.len() / 2;
        raw_dev.select_nth_unstable_by(rmid, |a, b| a.total_cmp(b));
        (1.4826 * raw_dev[rmid]).max(sigma_r)
    } else {
        sigma_r
    };

    // Treat `peak_threshold_sigma` (e.g. 3.0 forwarded from a GUI) as the
    // starting point for uniformity-guided threshold adaptation: scan once at
    // `scan_floor` using locally normalized SNR `(res - med_loc) / sigma_loc`,
    // then select the cleanest threshold that gives uniform tile coverage
    // without flooding into the noise floor.
    let s0 = match peak_threshold_sigma {
        Some(s) if s.is_finite() => s.clamp(1.5, 20.0),
        _ => 3.5,
    };
    let scan_floor = (s0 * 0.55).clamp(1.8, s0);

    let valid_margin: usize = if dim1.min(dim2) >= 128 { 3 } else { 2 };
    let nms_r: isize = if dim1.max(dim2) >= 900 { 2 } else { 1 };
    let border = valid_margin.max(nms_r as usize);
    let row_end = dim1.saturating_sub(border);
    let col_end = dim2.saturating_sub(border);

    let all_candidates: Vec<(f64, f64, f64)> = (border..row_end)
        .into_par_iter()
        .flat_map_iter(|i| {
            let row = i * dim2;
            let i_f = i as f64;
            let mut row_cands = Vec::new();
            for j in border..col_end {
                let idx = row + j;
                let res_v = residual[idx];
                let (m_loc, s_loc) = noise.at(i_f, j as f64);
                let local_snr = (res_v - m_loc) / s_loc;
                if local_snr <= scan_floor {
                    continue;
                }
                if !neighborhood_valid(valid, dim1, dim2, i, j, valid_margin) {
                    continue;
                }
                // Reject isolated single-pixel hot pixels / cosmic rays:
                // require both a huge raw excess (> 12σ_raw) AND that even
                // the single brightest 8-neighbor sits near background
                // (< 12% of the center excess). Preserves real Pilatus
                // 1-px Bragg spots (which extend along the ring azimuth).
                let local_bg = smooth[idx] - res_v;
                let raw_ex = image[idx] - local_bg;
                if raw_ex > 12.0 * sigma_raw {
                    let mut n_max1 = f64::NEG_INFINITY;
                    for di in -1isize..=1 {
                        let nr = (i as isize + di) as usize * dim2;
                        for dj in -1isize..=1 {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            let nv = image[nr + (j as isize + dj) as usize];
                            if nv > n_max1 {
                                n_max1 = nv;
                            }
                        }
                    }
                    if (n_max1 - local_bg).max(0.0) < 0.12 * raw_ex {
                        continue;
                    }
                }

                let v = smooth[idx];
                let mut is_max = true;
                'nbr: for di in -nms_r..=nms_r {
                    let nr = (i as isize + di) as usize * dim2;
                    for dj in -nms_r..=nms_r {
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

                row_cands.push((i_f + off_r, j as f64 + off_c, local_snr));
            }
            row_cands
        })
        .collect();

    // Evaluate spatial tile coverage & count at a threshold in O(|all_candidates|).
    let eval_sigma = |thr: f64| -> (usize, f64) {
        let mut counts = [0usize; 64];
        let mut total = 0usize;
        for &(d1, d2, snr) in &all_candidates {
            if snr > thr {
                counts[noise.tile_index(d1, d2)] += 1;
                total += 1;
            }
        }
        let occupied = (0..64)
            .filter(|&t| noise.tile_valid[t] && counts[t] >= 2)
            .count();
        (total, occupied as f64 / noise.valid_tiles as f64)
    };

    let noise_cap = (n / 25).max(30_000);
    let (n0, cov0) = eval_sigma(s0);
    let mut used_sigma = s0;

    if n0 > noise_cap {
        // Starting threshold flooded into background noise — step upward until clean.
        for mult in [1.35, 1.75, 2.25] {
            let cand_s = (s0 * mult).min(20.0);
            let (nc, cov) = eval_sigma(cand_s);
            if nc >= 40 && cov >= 0.35 {
                used_sigma = cand_s;
                if nc <= noise_cap {
                    break;
                }
            }
        }
    } else if n0 < 40 || (cov0 < 0.50 && n0 < (n / 400).max(1000)) {
        // Sparse or uneven coverage at `s0` — step downward to pick up fainter
        // ring arcs uniformly without entering the background noise floor.
        let mut best_cov = cov0;
        let mut best_n = n0;
        for mult in [0.80, 0.65, 0.55] {
            let cand_s = (s0 * mult).max(scan_floor);
            if cand_s >= used_sigma - 0.05 {
                continue;
            }
            let (nc, cov) = eval_sigma(cand_s);
            if nc > noise_cap {
                break;
            }
            // Guard against flooding into the background noise floor: a sudden
            // >2.2× jump in peak count relative to `n0` signals random noise
            // peaks rather than missing ring arcs; stop stepping down.
            if n0 >= 40 && nc > (n0 as f64 * 2.2) as usize {
                break;
            }
            if (best_n < 40 && nc > best_n) || (cov >= best_cov + 0.08) {
                used_sigma = cand_s;
                best_cov = cov;
                best_n = nc;
                if best_cov >= 0.60 && best_n >= 200 {
                    break;
                }
            }
        }
    }

    let candidates: Vec<(f64, f64, f64)> = all_candidates
        .into_iter()
        .filter(|&(_, _, snr)| snr > used_sigma)
        .collect();

    let tile_min_score = (used_sigma + 1.5).max(4.5);
    let peaks = select_spatially_balanced_peaks(candidates, dim1, dim2, max_peaks, tile_min_score);
    (peaks, used_sigma)
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

#[allow(clippy::too_many_arguments)]
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
    r_min_valid_hint: Option<usize>,
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

    // Scale peak-count SNR up gently for sparse spotty patterns (< 200 total
    // peaks) where even a true ring has only ~10–15 discrete spots spread over
    // a few radial bins.
    let sparse_boost = (200.0 / (iso_peaks.len().max(30) as f64))
        .sqrt()
        .clamp(1.0, 2.5);
    let snr: Vec<f64> = smooth.iter().map(|&c| c * 4.0 * sparse_boost).collect();

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

    // Use the detector's valid-pixel inner radius (with a small floor to remain
    // robust to central beamstops) so `scan_distances_1d` still penalizes
    // sub-multiple distance aliases (e.g. D/√2) that predict missing inner rings
    // inside the innermost detected peak.
    let r_min_effective = match r_min_valid_hint {
        Some(rv) => rv.max(35).min(r_min_obs),
        None => r_min_obs,
    };

    Some(RadialProfile1D {
        snr,
        r_min_obs: r_min_effective,
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
    let mut tan_rings: Vec<f64> =
        select_initial_tth_rings(calibrant, wavelength_m, max_ring_index.max(10))
            .into_iter()
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

/// Estimate median ring FWHM (in isotropic pixels) from the strongest peaks of
/// a 1D radial SNR profile, used to auto-adapt the 2D ring search half-width.
fn estimate_profile_fwhm_px(profile: &RadialProfile1D) -> Option<f64> {
    let n = profile.snr.len();
    let mut widths: Vec<f64> = Vec::new();
    for &(r_pk, peak_snr) in profile.peaks_r.iter().take(8) {
        let center = r_pk.round() as usize;
        if center == 0 || center + 1 >= n || peak_snr <= 0.0 {
            continue;
        }
        let half = 0.5 * profile.snr[center].max(peak_snr);
        let mut lo = center;
        while lo > 0 && profile.snr[lo] > half && (center - lo) < 30 {
            lo -= 1;
        }
        let mut hi = center;
        while hi + 1 < n && profile.snr[hi] > half && (hi - center) < 30 {
            hi += 1;
        }
        let w = (hi - lo) as f64;
        if (1.0..50.0).contains(&w) {
            widths.push(w);
        }
    }
    if widths.is_empty() {
        return None;
    }
    widths.sort_by(|a, b| a.total_cmp(b));
    Some(widths[widths.len() / 2])
}

/// Sample sub-pixel ring maxima along azimuthal sectors for each visible
/// calibrant ring predicted by `poni`, using the background-subtracted residual.
fn extract_ring_peaks_2d(
    residual: &[f64],
    noise: &LocalNoiseGrid,
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
    extract_ring_peaks_2d_with_tangent(
        residual,
        noise,
        sigma_r,
        valid,
        dim1,
        dim2,
        poni,
        calibrant,
        half_width_px,
        n_az,
        max_rings,
        max_ring_index,
        0,
        None,
    )
}

/// Scan one azimuthal sector along the ring arc around `(d1, d2)` for a
/// standout unsaturated Bragg spot and, if found, return its 2D sub-pixel
/// center `(d1, d2)`. Uses a fast circular-arc tangent expansion with zero
/// trig calls and stack-only buffers. Returns `None` on smooth continuous
/// rings (so the caller keeps the uniform sector-center ray), on single-pixel
/// zingers, or when a candidate touches masked / saturated pixels.
#[allow(clippy::too_many_arguments)]
fn try_lock_sector_spot(
    residual: &[f64],
    noise: &LocalNoiseGrid,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    d1: f64,
    d2: f64,
    u1: f64,
    u2: f64,
    dphi: f64,
    r_ring_px: f64,
    w_eff: usize,
    margin: f64,
) -> Option<(f64, f64)> {
    let arc_px = r_ring_px * dphi;
    if arc_px < 6.0 || w_eff < 2 || r_ring_px < 20.0 {
        return None;
    }
    let n_sub = (((0.86 * arc_px) / 2.0).round().clamp(3.0, 23.0) as usize) | 1;
    let half_arc = 0.43 * arc_px;
    let step_arc = (2.0 * half_arc) / (n_sub - 1) as f64;
    let w_scan = w_eff.min(3) as isize;
    let inv_2r = 0.5 / r_ring_px;
    let t1 = -u2;
    let t2 = u1;

    let mut sub_vals = [0.0_f64; 23];
    let mut n_valid = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    let mut best_cand: Option<(f64, f64, f64, f64, f64)> = None; // (p1, p2, lu1, lu2, s_peak)

    for j in 0..n_sub {
        let td = -half_arc + (j as f64) * step_arc;
        let sag = td * td * inv_2r;
        let p1 = d1 + td * t1 - sag * u1;
        let p2 = d2 + td * t2 - sag * u2;
        if p1 < margin
            || p2 < margin
            || p1 > (dim1 as f64 - 1.0 - margin)
            || p2 > (dim2 as f64 - 1.0 - margin)
        {
            continue;
        }
        // Local unit radial rotated by small angle td / r_ring_px (no trig).
        let ang = td / r_ring_px;
        let lu1_raw = u1 + ang * t1;
        let lu2_raw = u2 + ang * t2;
        let inv_n = 1.0 / lu1_raw.hypot(lu2_raw);
        let lu1 = lu1_raw * inv_n;
        let lu2 = lu2_raw * inv_n;

        // Require all radial taps in [-w_scan, +w_scan] to be valid so
        // saturated spot cores and detector gaps are skipped.
        let mut ray_best_v = f64::NEG_INFINITY;
        let mut ray_best_s = 0.0_f64;
        let mut ray_ok = true;
        for s in -w_scan..=w_scan {
            let sd = s as f64;
            match sample_bilinear_valid(residual, valid, dim1, dim2, p1 + sd * lu1, p2 + sd * lu2) {
                Some(v) => {
                    if v > ray_best_v {
                        ray_best_v = v;
                        ray_best_s = sd;
                    }
                }
                None => {
                    ray_ok = false;
                    break;
                }
            }
        }
        if !ray_ok {
            continue;
        }
        sub_vals[n_valid] = ray_best_v;
        n_valid += 1;
        if ray_best_v > best_v {
            best_v = ray_best_v;
            best_cand = Some((p1, p2, lu1, lu2, ray_best_s));
        }
    }

    if n_valid < 3 {
        return None;
    }
    let (bp1, bp2, lu1, lu2, s_init) = best_cand?;
    let valid_slice = &mut sub_vals[..n_valid];
    let mid = n_valid / 2;
    valid_slice.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let sec_med = valid_slice[mid];
    let (m_loc, s_loc) = noise.at(bp1, bp2);

    // Only lock onto the sector peak when it is a standout Bragg spot
    // (smooth continuous rings stay on the uniform sector-center ray).
    if (best_v - m_loc) < 5.0 * s_loc || (best_v - m_loc) <= 3.0 * (sec_med - m_loc).max(s_loc) {
        return None;
    }

    // Refine tangent position along a 7-tap 1 px strip through the candidate
    // spot; require all 7 taps valid (rejecting halos next to saturated cores)
    // and at least one adjacent tap >= 25% of peak excess (rejecting 1-px zingers).
    let lt1 = -lu2;
    let lt2 = lu1;
    let c1 = bp1 + s_init * lu1;
    let c2 = bp2 + s_init * lu2;
    let mut t_vals = [0.0_f64; 7];
    let mut best_ti = 3usize;
    let mut best_tv = f64::NEG_INFINITY;
    for (ti, slot) in t_vals.iter_mut().enumerate() {
        let td = (ti as isize - 3) as f64;
        let v = sample_bilinear_valid(residual, valid, dim1, dim2, c1 + td * lt1, c2 + td * lt2)?;
        *slot = v;
        if v > best_tv {
            best_tv = v;
            best_ti = ti;
        }
    }
    if best_ti == 0 || best_ti == 6 {
        return None;
    }
    let tm = t_vals[best_ti - 1];
    let t0 = t_vals[best_ti];
    let tp = t_vals[best_ti + 1];
    let peak_excess = t0 - m_loc;
    if peak_excess < 5.0 * s_loc
        || ((tm - m_loc) < 0.25 * peak_excess && (tp - m_loc) < 0.25 * peak_excess)
    {
        return None;
    }
    let tden = tm - 2.0 * t0 + tp;
    let tsub = if tden.abs() > 1e-12 {
        (0.5 * (tm - tp) / tden).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let t_off = (best_ti as isize - 3) as f64 + tsub;

    // Full single-ray radial refinement through the snapped spot center.
    let sc1 = bp1 + t_off * lt1;
    let sc2 = bp2 + t_off * lt2;
    let w = w_eff.min(16) as isize;
    let mut r_line = [0.0_f64; 33];
    let mut r_len = 0usize;
    let mut r_best_idx = 0usize;
    let mut r_best_v = f64::NEG_INFINITY;
    let mut r_min_v = f64::INFINITY;
    for s in -w..=w {
        let sd = s as f64;
        let v = sample_bilinear_valid(residual, valid, dim1, dim2, sc1 + sd * lu1, sc2 + sd * lu2)?;
        if v > r_best_v {
            r_best_v = v;
            r_best_idx = r_len;
        }
        if v < r_min_v {
            r_min_v = v;
        }
        r_line[r_len] = v;
        r_len += 1;
    }
    if r_len < 5
        || r_best_idx == 0
        || r_best_idx + 1 >= r_len
        || (r_best_v - m_loc) < 5.0 * s_loc
        || (r_best_v - r_min_v) < 3.0 * s_loc
    {
        return None;
    }
    let rm = r_line[r_best_idx - 1];
    let r0 = r_line[r_best_idx];
    let rp = r_line[r_best_idx + 1];
    let rden = rm - 2.0 * r0 + rp;
    let rsub = if rden.abs() > 1e-12 {
        (0.5 * (rm - rp) / rden).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let s_final = (r_best_idx as isize - w) as f64 + rsub;
    Some((sc1 + s_final * lu1, sc2 + s_final * lu2))
}

/// Like [`extract_ring_peaks_2d`], with optional azimuthal tangent averaging
/// (`2 * tangent_half_px + 1` parallel rays spaced 1.5 px along the ring
/// tangent) to boost faint outer-ring SNR by `√(n_taps)` without radial blur,
/// and per-sector Bragg-spot locking on spotty rings.
#[allow(clippy::too_many_arguments)]
fn extract_ring_peaks_2d_with_tangent(
    residual: &[f64],
    noise: &LocalNoiseGrid,
    _sigma_r: f64,
    valid: &[bool],
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    calibrant: &Calibrant,
    half_width_px: usize,
    n_az: usize,
    max_rings: usize,
    max_ring_index: usize,
    tangent_half_px: usize,
    distortion: Option<&super::distortion::DistortionField>,
) -> (CalibrationPoints, usize) {
    let tc_dist = distortion.map(|_| crate::geometry::math::TrigonometricConstants::new(poni));
    let tth_list = calibrant.two_theta_rad(poni.wavelength);
    // Precompute local angular spacing to nearest neighboring calibrant ring so
    // the radial search window never jumps onto an adjacent overlapping ring.
    // Include one ring past `max_ring_index` so the last ring sees its outer neighbor.
    let valid_tths: Vec<f64> = tth_list
        .iter()
        .take(max_ring_index.saturating_add(1))
        .flatten()
        .copied()
        .filter(|t| t.is_finite() && *t > 1e-4 && *t < PI * 0.49)
        .collect();
    let mut pts = CalibrationPoints::new();
    let mut n_rings_with_pts = 0usize;
    let margin = (half_width_px + (tangent_half_px * 2) + 2) as f64;
    let mut line = Vec::with_capacity(2 * half_width_px + 1);

    for (ring_k, opt_tth) in tth_list.iter().enumerate().take(max_ring_index) {
        if n_rings_with_pts >= max_rings {
            break;
        }
        let tth = match opt_tth {
            Some(t) if t.is_finite() && *t > 1e-4 && *t < PI * 0.49 => *t,
            _ => continue,
        };
        let local_spacing_rad = valid_tths
            .iter()
            .filter_map(|&other| {
                let d = (other - tth).abs();
                (d > 1e-6).then_some(d)
            })
            .fold(f64::INFINITY, f64::min);
        let mut ring_pts = 0usize;
        let ring_first_idx = pts.len();
        // Adaptive arc-length sampling: keep roughly constant pixel spacing
        // along outer rings so clipped high-2θ corner arcs receive plenty of
        // points instead of being starved by equal-azimuth sampling.
        let r_ref = (0.25 * dim1.max(dim2) as f64).max(64.0);
        let r_ring_px = (poni.distance / poni.pixel1.min(poni.pixel2)) * tth.tan();
        let n_az_min = n_az.min(360) as f64;
        let n_az_ring = ((n_az as f64) * (r_ring_px / r_ref).max(1.0))
            .round()
            .clamp(n_az_min, 360.0) as usize;

        let dphi = 2.0 * PI / (n_az_ring as f64);
        for a in 0..n_az_ring {
            let phi = dphi * (a as f64);
            let (d1_ideal, d2_ideal) = match project_ring_pixel(poni, tth, phi) {
                Some(p) => p,
                None => continue,
            };
            let max_shift_px = (0.04 * dim1.max(dim2) as f64).clamp(16.0, 48.0);
            let (d1, d2) = match (distortion, tc_dist.as_ref()) {
                (Some(field), Some(tc)) => match super::distortion::invert_distortion_pixel(
                    field,
                    poni,
                    tc,
                    d1_ideal,
                    d2_ideal,
                    max_shift_px,
                ) {
                    Some(p) => p,
                    None => continue,
                },
                _ => (d1_ideal, d2_ideal),
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
            let t1 = -u2;
            let t2 = u1;

            // Skip unresolved doublets (two calibrant rings closer than 5 px)
            // so a radial search window cannot jump onto an adjacent ring;
            // otherwise clamp half-width to <= 0.45 of local inter-ring spacing.
            let w_eff = if local_spacing_rad.is_finite() {
                let px_per_rad = norm / (2.0 * delta);
                let spacing_px = local_spacing_rad * px_per_rad;
                if spacing_px < 5.0 {
                    continue;
                }
                let max_w = (0.45 * spacing_px).floor() as usize;
                half_width_px.min(max_w)
            } else {
                half_width_px
            };

            // On spotty rings, lock this azimuthal sector onto its strongest
            // unsaturated Bragg spot (preserving ~1 pick per sector); on smooth
            // continuous rings, `try_lock_sector_spot` returns `None` and we
            // fall through to the uniform sector-center ray.
            if let Some((out_d1, out_d2)) = try_lock_sector_spot(
                residual, noise, valid, dim1, dim2, d1, d2, u1, u2, dphi, r_ring_px, w_eff, margin,
            ) {
                let dup_prev = ring_pts > 0
                    && pts
                        .d1
                        .last()
                        .zip(pts.d2.last())
                        .is_some_and(|(&pd1, &pd2)| (out_d1 - pd1).hypot(out_d2 - pd2) < 2.0);
                let dup_first = ring_pts > 1
                    && (out_d1 - pts.d1[ring_first_idx]).hypot(out_d2 - pts.d2[ring_first_idx])
                        < 2.0;
                if dup_prev || dup_first {
                    continue;
                }
                pts.push(out_d1, out_d2, Some(ring_k));
                ring_pts += 1;
                continue;
            }
            let w = w_eff as isize;
            // Only average along the straight tangent on rings with large enough
            // radius that sagittal curvature (td² / (2·r)) stays < 0.04 px.
            let eff_tangent_half = if r_ring_px < 120.0 {
                0
            } else if r_ring_px < 250.0 {
                tangent_half_px.min(1)
            } else {
                tangent_half_px.min(4)
            };
            let th = eff_tangent_half as isize;
            let mut min_cnt = 2 * eff_tangent_half + 1;

            line.clear();
            let mut all_ok = true;
            for s in -w..=w {
                let sd = s as f64;
                let base1 = d1 + sd * u1;
                let base2 = d2 + sd * u2;
                if th == 0 {
                    match sample_bilinear_valid(residual, valid, dim1, dim2, base1, base2) {
                        Some(v) => line.push(v),
                        None => {
                            all_ok = false;
                            break;
                        }
                    }
                } else {
                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    for kt in -th..=th {
                        let td = kt as f64 * 1.5;
                        if let Some(v) = sample_bilinear_valid(
                            residual,
                            valid,
                            dim1,
                            dim2,
                            base1 + td * t1,
                            base2 + td * t2,
                        ) {
                            sum += v;
                            cnt += 1;
                        }
                    }
                    if cnt * 2 < (2 * eff_tangent_half + 1) {
                        all_ok = false;
                        break;
                    }
                    min_cnt = min_cnt.min(cnt);
                    line.push(sum / cnt as f64);
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

            let (m_loc, s_loc) = if eff_tangent_half > 0 {
                noise.at_unclamped(d1, d2)
            } else {
                noise.at(d1, d2)
            };
            // Account for residual σ=0.8 px blur correlation across 1.5 px
            // tangent taps (effective independent taps ≈ 1 + 0.5*(cnt - 1)).
            let n_eff = if eff_tangent_half > 0 {
                1.0 + 0.5 * (min_cnt.saturating_sub(1) as f64)
            } else {
                1.0
            };
            let snr_mult = if eff_tangent_half > 0 { 3.2 } else { 2.5 };
            let min_peak_snr = snr_mult * s_loc / n_eff.sqrt();
            if (best_val - m_loc) < min_peak_snr || (best_val - min_val) < min_peak_snr {
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

            // When tangent averaging is active, distinguish a continuous ring
            // from an isolated Bragg spot grazing the edge of the tangent strip:
            // - If a single spot dominates the strip, snap onto its interior
            //   tangent peak (skipping edge-grazed spots so dots never land in
            //   dark space 4 px beside a spot).
            // - On a continuous ring, require the center ray itself to carry
            //   genuine ring signal above background.
            let mut s_final = s_refined;
            let mut t_refined = 0.0_f64;
            if th > 0 {
                let b1 = d1 + s_refined * u1;
                let b2 = d2 + s_refined * u2;
                let mut tap_vals = [0.0_f64; 9];
                let mut tap_ok = [false; 9];
                let n_taps_u = (2 * eff_tangent_half + 1).min(9);
                let mut valid_vals = Vec::with_capacity(n_taps_u);
                let mut max_ti = eff_tangent_half;
                let mut max_tv = f64::NEG_INFINITY;
                for ti in 0..n_taps_u {
                    let kt = ti as isize - th;
                    let td = kt as f64 * 1.5;
                    if let Some(v) = sample_bilinear_valid(
                        residual,
                        valid,
                        dim1,
                        dim2,
                        b1 + td * t1,
                        b2 + td * t2,
                    ) {
                        tap_vals[ti] = v;
                        tap_ok[ti] = true;
                        valid_vals.push(v);
                        if v > max_tv {
                            max_tv = v;
                            max_ti = ti;
                        }
                    }
                }
                if valid_vals.is_empty() {
                    continue;
                }
                let mid = valid_vals.len() / 2;
                valid_vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                let tap_med = valid_vals[mid];
                let center_v = if tap_ok[eff_tangent_half] {
                    tap_vals[eff_tangent_half]
                } else {
                    tap_med
                };

                let is_spotty_strip = (max_tv - m_loc) >= 5.0 * s_loc
                    && (max_tv - m_loc) > 2.5 * (tap_med - m_loc).max(s_loc);
                if is_spotty_strip {
                    // Isolated Bragg spot: reject if it's on the strip boundary
                    // (would be an off-center graze), otherwise snap onto the
                    // spot's actual 2D center (tangent + radial re-refinement).
                    if max_ti == 0 || max_ti + 1 >= n_taps_u {
                        continue;
                    }
                    if !tap_ok[max_ti - 1] || !tap_ok[max_ti + 1] {
                        continue;
                    }
                    let tm = tap_vals[max_ti - 1];
                    let t0 = tap_vals[max_ti];
                    let tp = tap_vals[max_ti + 1];
                    let tden = tm - 2.0 * t0 + tp;
                    let tsub = if tden.abs() > 1e-12 {
                        (0.5 * (tm - tp) / tden).clamp(-0.5, 0.5)
                    } else {
                        0.0
                    };
                    t_refined = ((max_ti as isize - th) as f64 + tsub) * 1.5;

                    // Re-refine radial position along the single ray through the
                    // snapped spot center so both radial and tangent coords land
                    // dead-center on the Bragg spot; skip if radial check fails.
                    let sc1 = d1 + t_refined * t1;
                    let sc2 = d2 + t_refined * t2;
                    let mut r_best_s = s_refined;
                    let mut r_best_v = f64::NEG_INFINITY;
                    let mut r_line = Vec::with_capacity(2 * w_eff + 1);
                    let mut r_ok = true;
                    for s in -w..=w {
                        let sd = s as f64;
                        match sample_bilinear_valid(
                            residual,
                            valid,
                            dim1,
                            dim2,
                            sc1 + sd * u1,
                            sc2 + sd * u2,
                        ) {
                            Some(v) => {
                                if v > r_best_v {
                                    r_best_v = v;
                                    r_best_s = sd;
                                }
                                r_line.push(v);
                            }
                            None => {
                                r_ok = false;
                                break;
                            }
                        }
                    }
                    if !r_ok || r_line.len() < 5 {
                        continue;
                    }
                    let r_idx = (r_best_s.round() as isize + w) as usize;
                    if r_idx == 0 || r_idx + 1 >= r_line.len() || (r_best_v - m_loc) < 2.5 * s_loc {
                        continue;
                    }
                    let rm = r_line[r_idx - 1];
                    let r0 = r_line[r_idx];
                    let rp = r_line[r_idx + 1];
                    let rden = rm - 2.0 * r0 + rp;
                    let rsub = if rden.abs() > 1e-12 {
                        (0.5 * (rm - rp) / rden).clamp(-0.5, 0.5)
                    } else {
                        0.0
                    };
                    s_final = (r_idx as isize - w) as f64 + rsub;
                } else if (center_v - m_loc) < 0.45 * (best_val - m_loc)
                    || (center_v - min_val) < 1.5 * s_loc / n_eff.sqrt()
                {
                    continue;
                }
            }

            let out_d1 = d1 + s_final * u1 + t_refined * t1;
            let out_d2 = d2 + s_final * u2 + t_refined * t2;
            // Dedupe adjacent overlapping strips that snapped to the same spot.
            if ring_pts > 0
                && let (Some(&prev_d1), Some(&prev_d2)) = (pts.d1.last(), pts.d2.last())
                && (out_d1 - prev_d1).hypot(out_d2 - prev_d2) < 1.5
            {
                continue;
            }
            pts.push(out_d1, out_d2, Some(ring_k));
            ring_pts += 1;
        }

        if ring_pts >= 5 {
            n_rings_with_pts += 1;
        }
    }

    (pts, n_rings_with_pts)
}

#[allow(clippy::too_many_arguments)]
fn iterative_2d_refine(
    residual: &[f64],
    noise: &LocalNoiseGrid,
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
            noise,
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
        let mut mask = options.refine_mask;
        // 2θ is invariant to rot3; never let rot3 drift from finite-difference
        // noise during L-M refinement (rot3 is calibrated via polarization).
        mask.rot3 = false;
        let refine_opts = CalibrationOptions {
            mask,
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
                        if supported > *prev_rings {
                            res.rms_rad <= (prev.rms_rad * 1.6).max(5.0e-4)
                        } else {
                            supported >= *prev_rings.min(&3) && res.rms_rad <= prev.rms_rad * 1.25
                        }
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn try_radial_profile_calibration<F>(
    image: &[f64],
    residual: &[f64],
    noise: &LocalNoiseGrid,
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
    on_progress: &mut F,
) -> Result<Option<(CalibrationResult, AutoCalibrateDiagnostics, usize)>, AutoCalibrateError>
where
    F: FnMut(AutoCalibrateProgress) -> bool,
{
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
    let Some(coarse_center) = estimate_ring_center(&ransac_peaks, &center_opts) else {
        return Ok(None);
    };
    let center_iso =
        refine_center_sharpness(&iso_peaks, (coarse_center.d1, coarse_center.d2), max_radius);
    let center_px = (center_iso.0 / scale1, center_iso.1 / scale2);
    if !on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::EstimatingCenter,
        message: format!(
            "Estimated ring center ({:.1}, {:.1}) px",
            center_px.0, center_px.1
        ),
        progress: 0.30,
        peaks: None,
        ring_center_px: Some(center_px),
        candidate_poni: None,
        candidate_rms_rad: None,
        n_rings: None,
    }) {
        return Err(AutoCalibrateError::Cancelled);
    }

    // If distance is fixed and a starting_distance was supplied, use it directly.
    let mut dist_candidates: Vec<(f64, f64)> = Vec::new();
    let mut n_detected_1d = 0usize;
    let mut est_fwhm_px: Option<f64> = None;
    if !options.refine_mask.distance
        && let Some(d0) = options.starting_distance
    {
        dist_candidates.push((d0, 1.0));
    } else {
        let mut r_min_valid_hint: Option<usize> = None;
        if let Some(profile) =
            build_radial_profile_1d(image, valid, dim1, dim2, center_iso, scale1, scale2)
        {
            r_min_valid_hint = Some(profile.r_min_obs);
            est_fwhm_px = estimate_profile_fwhm_px(&profile);
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
                6,
            ));
        }
        if let Some(peak_profile) = build_peak_radial_profile_1d(
            &iso_peaks,
            dim1,
            dim2,
            center_iso,
            scale1,
            scale2,
            r_min_valid_hint,
        ) {
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
                6,
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
        return Ok(None);
    }

    if !on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::ScanningDistances,
        message: format!(
            "Found {} distance candidates (top {:.1} mm)",
            dist_candidates.len(),
            dist_candidates[0].0 * 1e3
        ),
        progress: 0.42,
        peaks: None,
        ring_center_px: Some(center_px),
        candidate_poni: None,
        candidate_rms_rad: None,
        n_rings: None,
    }) {
        return Err(AutoCalibrateError::Cancelled);
    }

    // Convert isotropic center back to detector metres (pyFAI pixel-center +0.5 convention).
    let bootstrap_p1 = center_iso.0 * p_min + 0.5 * pixel1;
    let bootstrap_p2 = center_iso.1 * p_min + 0.5 * pixel2;
    let fine_w = options
        .search_half_width_px
        .map(|w| w.clamp(3, 24))
        .unwrap_or_else(|| {
            est_fwhm_px
                .map(|w| (w * 1.1).round().clamp(4.0, 12.0) as usize)
                .unwrap_or(4)
        });
    let mid_w = (fine_w * 2).clamp(7, 20);
    let coarse_w = (fine_w * 3).clamp(14, 28);
    // Progressive coarse-to-fine ring expansion: lock the coarse geometry on
    // the innermost 6 rings first (preventing outer ring-jumping when initial
    // tilt/distance is approximate), then expand to 12 and finally up to
    // `max_ring_index.min(28)` rings for final polish.
    let final_max_rings = options.max_ring_index.clamp(8, 28);
    let mid_max_rings = final_max_rings.min(12);
    let coarse_max_rings = final_max_rings.min(6);
    let passes = [
        (coarse_w, 3.0e-3, 96usize, coarse_max_rings),
        (mid_w, 1.5e-3, 128, mid_max_rings),
        (fine_w, 8.0e-4, 128, final_max_rings),
    ];

    let has_hint = options.starting_distance.is_some();
    let mut best: Option<(
        CalibrationResult,
        AutoCalibrateDiagnostics,
        usize,
        f64,
        bool,
    )> = None;
    let n_cands = dist_candidates.len().max(1);
    for (cand_idx, &(d_cand, d_score)) in dist_candidates.iter().enumerate() {
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
            noise,
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
        // Only fall back to peak-only refinement when 2D ring extraction did
        // not already achieve a tight multi-ring fit (e.g. spotty rings where
        // continuous azimuthal profiles are sparse).
        let need_peak_fallback = match &smooth_res {
            Some((r, nr)) => *nr < 6 || r.rms_rad >= 1.5e-3,
            None => true,
        };
        let peak_res = if need_peak_fallback {
            refine_on_detected_peaks(peaks, dim1, dim2, init, calibrant, options)
        } else {
            None
        };

        let mut slam_dunk = false;
        for (res, n_rings) in [smooth_res, peak_res].into_iter().flatten() {
            let m = evaluate_peak_match(
                peaks,
                valid,
                dim1,
                dim2,
                &res.poni,
                calibrant,
                2.0e-3,
                options.max_ring_index,
            );
            let cur_ok =
                is_verified_calibration(&res, n_rings, &m, peaks.len(), d_min, d_max, has_hint);
            let clear_top1 = n_cands == 1 || dist_candidates[0].1 > 1.35 * dist_candidates[1].1;
            if (cand_idx >= 1 || clear_top1)
                && cur_ok
                && res.rms_rad < 3.5e-4
                && n_rings >= 6
                && m.inlier_frac >= 0.70
                && m.ring_recall >= 0.75
                && m.inlier_rms_rad < 5.0e-4
            {
                slam_dunk = true;
            }
            let eff_rms = res.rms_rad + 0.35 * m.inlier_rms_rad;
            let quality = (m.inlier_frac.max(0.05)
                * m.ring_recall.max(0.05).powi(2)
                * (n_rings as f64).sqrt()
                * d_score.max(0.1))
                / (eff_rms + 5.0e-5);
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

        let frac = 0.45 + 0.30 * ((cand_idx + 1) as f64 / n_cands as f64);
        let (cand_poni, cand_rms, cand_rings, detail) = match &best {
            Some((r, _, nr, _, _)) => (
                Some(r.poni.clone()),
                Some(r.rms_rad),
                Some(*nr),
                format!(
                    "Refining candidate {}/{} (best {:.1} mm, {} rings, RMS {:.4}°)",
                    cand_idx + 1,
                    n_cands,
                    r.poni.distance * 1e3,
                    nr,
                    r.rms_rad.to_degrees()
                ),
            ),
            None => (
                None,
                None,
                None,
                format!(
                    "Refining candidate {}/{} ({:.1} mm)",
                    cand_idx + 1,
                    n_cands,
                    d_cand * 1e3
                ),
            ),
        };
        if !on_progress(AutoCalibrateProgress {
            stage: AutoCalibrateStage::RefiningCandidate,
            message: detail,
            progress: frac,
            peaks: None,
            ring_center_px: Some(center_px),
            candidate_poni: cand_poni,
            candidate_rms_rad: cand_rms,
            n_rings: cand_rings,
        }) {
            return Err(AutoCalibrateError::Cancelled);
        }
        if slam_dunk {
            break;
        }
    }

    Ok(best.map(|(res, diag, n_rings, _, _)| (res, diag, n_rings)))
}

// ---------- Step 6: Tilt / strip-detector fallback ----------

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn try_grid_fallback<F>(
    residual: &[f64],
    noise: &LocalNoiseGrid,
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
    on_progress: &mut F,
) -> Result<Option<(CalibrationResult, AutoCalibrateDiagnostics, usize)>, AutoCalibrateError>
where
    F: FnMut(AutoCalibrateProgress) -> bool,
{
    if !on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::GridSearch,
        message: "Running 5D tilted/strip grid search…".into(),
        progress: 0.78,
        peaks: None,
        ring_center_px: None,
        candidate_poni: None,
        candidate_rms_rad: None,
        n_rings: None,
    }) {
        return Err(AutoCalibrateError::Cancelled);
    }

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
            let hi = d_max.min(0.8).max(d_min * 1.5);
            (d_min, hi, 24)
        }
    };

    let grid_opts = GridSearchOptions {
        distance_min: grid_d_min,
        distance_max: grid_d_max,
        distance_steps: grid_d_steps,
        rot_max: 45.0_f64.to_radians(),
        rot_steps: 15,
        poni_search_radius_px: (0.30 * dim1.max(dim2) as f64).clamp(150.0, 450.0),
        poni_grid_steps: if is_strip_detector { 17 } else { 13 },
        tolerance_rad: 8.0e-3,
        max_peaks_for_grid: 400,
        max_ring_index: options.max_ring_index.min(36),
        top_k: 40,
    };

    // Search both regimes:
    // 1. PONI near peak centroid (side/off-axis tilted detectors where the detector
    //    faces the sample at high 2θ and the direct beam is far off-detector).
    // 2. Direct beam near peak centroid (tilted detectors where the beam stays
    //    near/on the detector and PONI shifts off-detector by D·tan(tilt)).
    let (mut candidates, beam_candidates) = rayon::join(
        || {
            grid_search_top_k(
                strong_peaks,
                cx,
                cy,
                calibrant,
                wavelength_m,
                pixel1,
                pixel2,
                &grid_opts,
            )
        },
        || {
            if is_strip_detector {
                Vec::new()
            } else {
                let mut beam_opts = grid_opts.clone();
                beam_opts.poni_grid_steps = 9;
                grid_search_top_k_beam_centered(
                    strong_peaks,
                    cx,
                    cy,
                    calibrant,
                    wavelength_m,
                    pixel1,
                    pixel2,
                    &beam_opts,
                )
            }
        },
    );
    candidates.extend(beam_candidates);
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut pts = CalibrationPoints::new();
    for &(d1, d2) in strong_peaks {
        pts.push(d1, d2, None);
    }
    let refine_opts_coarse = CalibrationOptions {
        mask: options.refine_mask,
        loss: Loss::Huber { c: 2.0e-3 },
        auto_assign_rings: true,
        assign_tolerance_rad: grid_opts.tolerance_rad.max(5.0e-3),
        max_iter: options.max_iter,
        tol: options.tol,
    };
    let refine_opts_fine = CalibrationOptions {
        mask: options.refine_mask,
        loss: Loss::Huber { c: 1.0e-3 },
        auto_assign_rings: true,
        assign_tolerance_rad: 3.0e-3,
        max_iter: options.max_iter,
        tol: options.tol,
    };

    let max_rot_rad = 60.0_f64.to_radians();
    let refine_and_score = |start_poni: &Poni| -> Option<(CalibrationResult, f64)> {
        let geom = Geometry::new(dim1, dim2, start_poni.clone());
        let res_coarse = refine(&geom, &pts, calibrant, &refine_opts_coarse).ok()?;
        if !res_coarse.poni.distance.is_finite()
            || res_coarse.poni.distance <= 0.005
            || res_coarse.poni.rot1.abs() > max_rot_rad
            || res_coarse.poni.rot2.abs() > max_rot_rad
        {
            return None;
        }
        let geom_fine = Geometry::new(dim1, dim2, res_coarse.poni.clone());
        let res = refine(&geom_fine, &pts, calibrant, &refine_opts_fine)
            .ok()
            .filter(|r| {
                r.poni.distance.is_finite()
                    && r.poni.distance > 0.005
                    && r.poni.rot1.abs() <= max_rot_rad
                    && r.poni.rot2.abs() <= max_rot_rad
                    && count_supported_rings(&r.assignments, 5) >= 3
            })
            .unwrap_or(res_coarse);
        let m = evaluate_peak_match(
            strong_peaks,
            valid,
            dim1,
            dim2,
            &res.poni,
            calibrant,
            2.0e-3,
            options.max_ring_index,
        );
        let eff_rms = res.rms_rad + 0.35 * m.inlier_rms_rad;
        let score = (m.inlier_frac * m.ring_recall.max(0.05).powi(2)) / (eff_rms + 2.0e-4);
        Some((res, score))
    };

    let best_grid: Option<(
        CalibrationResult,
        &super::grid_search::GridSearchCandidate,
        f64,
        usize,
    )> = candidates
        .par_iter()
        .enumerate()
        .filter_map(|(idx, cand)| {
            refine_and_score(&cand.poni).map(|(res, score)| (res, cand, score, idx))
        })
        .max_by(|a, b| a.2.total_cmp(&b.2).then_with(|| b.3.cmp(&a.3)));
    let Some((mut grid_res, winning, mut grid_score, _)) = best_grid else {
        return Ok(None);
    };

    // Check the mirror-tilt twin around the direct-beam intersection: for a
    // tilted detector where the beam is near the detector, `(rot1, rot2)` and
    // `(-rot1, -rot2)` produce nearly identical ellipse axis ratios and differ
    // only in the small sign of the center shift. Testing the twin takes ~1 ms
    // and breaks any mirror-tilt local minimum.
    let p = &grid_res.poni;
    if p.rot1.abs().max(p.rot2.abs()) > 5.0_f64.to_radians() && p.rot1.cos().abs() > 0.2 {
        let b1 = p.poni1 + p.distance * p.rot2.tan() / p.rot1.cos();
        let b2 = p.poni2 - p.distance * p.rot1.tan();
        let mut twin = p.clone();
        twin.rot1 = -p.rot1;
        twin.rot2 = -p.rot2;
        twin.poni1 = b1 - twin.distance * twin.rot2.tan() / twin.rot1.cos();
        twin.poni2 = b2 + twin.distance * twin.rot1.tan();
        if let Some((twin_res, twin_score)) = refine_and_score(&twin)
            && twin_score > grid_score
        {
            grid_res = twin_res;
            grid_score = twin_score;
        }
    }

    let grid_n_rings = count_supported_rings(&grid_res.assignments, 5);
    if !on_progress(AutoCalibrateProgress {
        stage: AutoCalibrateStage::PolishingGrid,
        message: format!(
            "Polishing tilted geometry ({:.1} mm, {} rings, RMS {:.4}°)",
            grid_res.poni.distance * 1e3,
            grid_n_rings,
            grid_res.rms_rad.to_degrees()
        ),
        progress: 0.90,
        peaks: None,
        ring_center_px: None,
        candidate_poni: Some(grid_res.poni.clone()),
        candidate_rms_rad: Some(grid_res.rms_rad),
        n_rings: Some(grid_n_rings),
    }) {
        return Err(AutoCalibrateError::Cancelled);
    }

    let n_az = if is_strip_detector { 360 } else { 128 };
    let polish_ring_cap = options.max_ring_index;
    let polish_fine = options
        .search_half_width_px
        .map(|w| w.clamp(3, 16))
        .unwrap_or(4);
    let polish_coarse = (polish_fine * 2).clamp(8, 20);
    let final_polish_rings = options.max_ring_index.clamp(8, 28);
    let polish_passes = [
        (polish_coarse, 2.0e-3, n_az, final_polish_rings.min(8)),
        (polish_fine, 1.0e-3, n_az, final_polish_rings.min(16)),
        (polish_fine, 8.0e-4, n_az, final_polish_rings),
    ];
    if let Some((polished, n_rings)) = iterative_2d_refine(
        residual,
        noise,
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
        let m = evaluate_peak_match(
            strong_peaks,
            valid,
            dim1,
            dim2,
            &polished.poni,
            calibrant,
            2.0e-3,
            options.max_ring_index,
        );
        let eff_rms = polished.rms_rad + 0.35 * m.inlier_rms_rad;
        let pol_score = (m.inlier_frac * m.ring_recall.max(0.05).powi(2)) / (eff_rms + 2.0e-4);
        if n_rings >= 3 && pol_score >= grid_score * 0.9 {
            let diag = AutoCalibrateDiagnostics {
                n_detected_ellipses: candidates.len(),
                n_identified_ellipses: n_rings,
                bootstrap_distance: winning.poni.distance,
                bootstrap_ring_center_d1: winning.poni.poni1,
                bootstrap_ring_center_d2: winning.poni.poni2,
            };
            return Ok(Some((polished, diag, n_rings)));
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
    Ok(Some((grid_res, diag, n_rings)))
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

    #[test]
    fn fast_box_blur_preserves_constant_and_symmetry() {
        let dim1 = 64;
        let dim2 = 80;
        let constant = vec![42.5_f32; dim1 * dim2];
        let blurred = fast_gaussian_blur_bg_f32(&constant, dim1, dim2, 6.0);
        for &v in &blurred {
            assert!((v - 42.5).abs() < 1e-4, "constant drift: {v}");
        }

        let mut impulse = vec![0.0_f32; dim1 * dim2];
        impulse[32 * dim2 + 40] = 1000.0;
        let b = fast_gaussian_blur_bg_f32(&impulse, dim1, dim2, 6.0);
        assert!(b[32 * dim2 + 40] > b[32 * dim2 + 45]);
        assert!((b[32 * dim2 + 35] - b[32 * dim2 + 45]).abs() < 1e-4);
        assert!((b[27 * dim2 + 40] - b[37 * dim2 + 40]).abs() < 1e-4);
    }

    #[test]
    fn cancel_during_retry_returns_cancelled() {
        // Pure noise with a few bright fake spots so primary fails verification
        // and triggers the automatic retry; cancel as soon as retry begins.
        let dim1 = 128;
        let dim2 = 128;
        let mut img = vec![100.0; dim1 * dim2];
        for k in 0..40 {
            let r = 10 + (k * 23) % 108;
            let c = 10 + (k * 37) % 108;
            for dr in 0..3 {
                for dc in 0..3 {
                    img[(r + dr) * dim2 + (c + dc)] = 500.0;
                }
            }
        }
        let cal = Calibrant::builtin("CeO2").expect("builtin CeO2");
        let opts = AutoCalibrateImageOptions {
            enable_tilt_fallback: false,
            ..Default::default()
        };
        let mut saw_retry = false;
        let res = auto_calibrate_image_with_progress(
            &img,
            dim1,
            dim2,
            None,
            &cal,
            0.31e-10,
            100e-6,
            100e-6,
            &opts,
            |ev| {
                if ev.message.contains("retrying") {
                    saw_retry = true;
                    return false;
                }
                true
            },
        );
        assert!(saw_retry, "expected retry message to fire");
        assert!(
            matches!(res, Err(AutoCalibrateError::Cancelled)),
            "expected Cancelled during retry, got {res:?}"
        );
    }

    #[test]
    fn build_valid_mask_auto_detects_saturation_plateaus() {
        let (dim1, dim2) = (64, 64);

        // 1. 14-bit clipped 4x4 plateau at 16_383 (plus a masked 1e7 hot pixel).
        let mut img = vec![100.0; dim1 * dim2];
        for i in 20..24 {
            for j in 20..24 {
                img[i * dim2 + j] = 16_383.0;
            }
        }
        let mut mask = vec![false; dim1 * dim2];
        img[5 * dim2 + 5] = 1.0e7;
        mask[5 * dim2 + 5] = true;
        let (valid, _, upper) = build_valid_mask(&img, Some(&mask), dim1, dim2, None);
        assert!(
            upper < 16_380.0 && upper > 16_000.0,
            "expected 14-bit ceiling ~16300, got {upper}"
        );
        assert!(!valid[21 * dim2 + 21], "saturated core must be masked");

        // 2. Dark-subtracted 16-bit plateau (65_420..65_535, within 0.2%).
        let mut img_dark = vec![50.0; dim1 * dim2];
        for i in 30..34 {
            for j in 30..34 {
                img_dark[i * dim2 + j] = 65_535.0 - ((i + j) % 7) as f64 * 15.0;
            }
        }
        let (valid_d, _, upper_d) = build_valid_mask(&img_dark, None, dim1, dim2, None);
        assert!(
            upper_d < 65_500.0 && upper_d > 64_000.0,
            "expected dark-subtracted 16-bit ceiling, got {upper_d}"
        );
        assert!(!valid_d[31 * dim2 + 31]);

        // 3. Unsaturated smooth Gaussian peak at 30_000 must stay unmasked.
        let mut img_unsat = vec![50.0; dim1 * dim2];
        for i in 0..dim1 {
            for j in 0..dim2 {
                let d2 = (i as f64 - 32.0).powi(2) + (j as f64 - 32.0).powi(2);
                img_unsat[i * dim2 + j] += 30_000.0 * (-d2 / 8.0).exp();
            }
        }
        let (valid_u, _, upper_u) = build_valid_mask(&img_unsat, None, dim1, dim2, None);
        assert!(
            upper_u > 3.9e9,
            "unsaturated peak should not trigger ceiling, got {upper_u}"
        );
        assert!(valid_u[32 * dim2 + 32]);
    }
}
