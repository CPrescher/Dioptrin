//! End-to-end auto calibration: peaks → ellipses → ring identity → refine.
//!
//! Given a set of 2-D peak coordinates (e.g. from `find_peaks_image`) and a
//! known calibrant + wavelength + pixel size, recover a refined PONI with
//! **no manual seed clicks and no detector-geometry hint** (no distance, no
//! beam center, no rotations).
//!
//! Pipeline:
//!
//! 1. `ransac_ellipses` — group peaks into ring ellipses (with optional
//!    near-perpendicular axis-ratio constraint and dedup).
//! 2. `identify_rings` — assign each ellipse to a calibrant ring index by
//!    matching tan(2θ) ratios. Returns `None` for spurious ellipses.
//! 3. Build a starting PONI. Two regimes:
//!    - **near-perpendicular** (median ellipse axis-ratio < 1.05):
//!      `(D = mean(s_i·px / tan(2θ_k)), PONI = ring-center, rot = 0)`.
//!      The L-M refinement picks up the small rotations.
//!    - **visibly tilted** (median axis-ratio ≥ 1.05): per-ring closed-form
//!      geometry inversion (see `inversion`) → median over rings yields
//!      `(D, PONI, rot1, rot2)` exactly for a flat tilted detector.
//! 4. Convert each matched ellipse's inlier points into
//!    `CalibrationPoints` with the ring index from step 2.
//! 5. Call `refine` with the bootstrapped PONI and a sensible default
//!    parameter mask. Returns the refined `CalibrationResult`.

use crate::geometry::Geometry;
use crate::poni::Poni;

use super::calibrant::Calibrant;
use super::center::{RingCenterOptions, estimate_ring_center};
use super::ellipse::{RansacOptions, ransac_ellipses};
use super::radial::{RadialExtractOptions, extract_rings_radial};
use super::refinement::{
    CalibrationError, CalibrationOptions, CalibrationPoints, CalibrationResult, Loss, ParamMask,
    refine,
};
use super::ring_identity::{RingIdOptions, identify_rings};

/// Strategy for clustering peaks into ring ellipses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutoMethod {
    /// 3-point circle RANSAC to estimate the beam center, then radial
    /// histogram binning to assign each peak to a ring, then a per-ring
    /// ellipse least-squares fit. Robust to off-detector beams, partial
    /// arcs, dense rings, and noisy peak sets. This is the default.
    #[default]
    Radial,
    /// 5-point ellipse RANSAC (the original primitive, kept for cases
    /// where the user wants per-ring outlier rejection on extremely
    /// noisy data, or for diagnostics).
    Ransac,
}

#[derive(Debug, Clone)]
pub struct AutoCalibrateOptions {
    pub method: AutoMethod,
    pub center: RingCenterOptions,
    pub radial: RadialExtractOptions,
    pub ransac: RansacOptions,
    pub ring_id: RingIdOptions,
    /// Parameter mask for the final L-M refinement.
    pub refine_mask: ParamMask,
    /// Loss function for the final refinement. `Huber` is the right default
    /// for a bootstrapped fit because the geometric estimate from step 3 is
    /// imperfect; outliers from RANSAC mis-assignments are downweighted.
    pub refine_loss: Loss,
    pub max_iter: usize,
    pub tol: f64,
    /// Used only for the legacy `AutoMethod::Ransac` path. Maximum distance
    /// (pixels) of an ellipse center from the consensus center for it to be
    /// kept. Set to `0.0` to disable.
    pub center_consistency_px: f64,
}

impl Default for AutoCalibrateOptions {
    fn default() -> Self {
        let ransac = RansacOptions {
            max_axis_ratio: Some(1.2), // assume roughly perpendicular detector
            ..RansacOptions::default()
        };
        let ring_id = RingIdOptions {
            ratio_tolerance: 0.03,
            ..RingIdOptions::default()
        };
        Self {
            method: AutoMethod::default(),
            center: RingCenterOptions::default(),
            radial: RadialExtractOptions::default(),
            ransac,
            ring_id,
            refine_mask: ParamMask::default(), // wavelength fixed
            refine_loss: Loss::Huber { c: 2.0e-3 },
            max_iter: 200,
            tol: 1e-9,
            center_consistency_px: 30.0,
        }
    }
}

/// Return the indices of ellipses whose centers lie within `tol_px` of the
/// median ellipse center. The median is robust to outlier centers and is
/// what cleans up RANSAC "ghost" ellipses that span peaks from multiple
/// physical rings.
fn consensus_center_filter(ellipses: &[crate::calibration::Ellipse], tol_px: f64) -> Vec<usize> {
    if ellipses.is_empty() {
        return Vec::new();
    }
    let mut xs: Vec<f64> = ellipses.iter().map(|e| e.center.0).collect();
    let mut ys: Vec<f64> = ellipses.iter().map(|e| e.center.1).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cx = xs[xs.len() / 2];
    let cy = ys[ys.len() / 2];
    let tol2 = tol_px * tol_px;
    ellipses
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let dx = e.center.0 - cx;
            let dy = e.center.1 - cy;
            if dx * dx + dy * dy <= tol2 {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct AutoCalibrateDiagnostics {
    pub n_detected_ellipses: usize,
    pub n_identified_ellipses: usize,
    /// Mean of `semi_minor_k · pixel / tan(2θ_k)` across matched (ellipse,
    /// ring) pairs. For tilted detectors this is unbiased (semi_minor is
    /// the perpendicular-projection radius); for the perpendicular case
    /// it equals the geometric distance.
    pub bootstrap_distance: f64,
    /// Mean ring-center fed into the bootstrap PONI fields. **Caveat: this
    /// is the geometric center of the projected ellipses (≈ PONI only for
    /// near-perpendicular detectors), not the actual PONI in the tilted
    /// case.** See `center.rs` for the distinction.
    pub bootstrap_ring_center_d1: f64,
    pub bootstrap_ring_center_d2: f64,
}

#[derive(Debug)]
pub enum AutoCalibrateError {
    /// Fewer ellipses detected than `min_inliers` requires — image too sparse
    /// or RANSAC parameters too strict.
    NoEllipses,
    /// Ring identity matched fewer ellipses than `RingIdOptions::min_matches`.
    NoConfidentIdentity,
    /// The final refinement failed.
    Refine(CalibrationError),
}

impl std::fmt::Display for AutoCalibrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoCalibrateError::NoEllipses => write!(f, "no ellipses detected by RANSAC"),
            AutoCalibrateError::NoConfidentIdentity => {
                write!(
                    f,
                    "could not confidently assign ellipses to calibrant rings"
                )
            }
            AutoCalibrateError::Refine(err) => write!(f, "refinement failed: {}", err),
        }
    }
}

impl std::error::Error for AutoCalibrateError {}

/// End-to-end auto calibration from picked peaks.
///
/// # Arguments
///
/// * `peaks` — list of `(d1, d2)` peak coordinates, e.g. from
///   `dioptrin::calibration::find_peaks_image`.
/// * `calibrant` — reference powder (LaB6, CeO2, ...).
/// * `wavelength_m` — beam wavelength in metres.
/// * `pixel1`, `pixel2` — detector pixel pitch in metres.
/// * `options` — knobs for the three internal stages.
///
/// Returns the refined `CalibrationResult` plus diagnostics about the
/// intermediate stages.
pub fn auto_calibrate(
    peaks: &[(f64, f64)],
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &AutoCalibrateOptions,
) -> Result<(CalibrationResult, AutoCalibrateDiagnostics), AutoCalibrateError> {
    // 1. Cluster peaks into ring ellipses.
    let (ellipses, point_assignments) = match options.method {
        AutoMethod::Radial => {
            // Estimate beam center robustly (handles partial arcs &
            // off-detector beams), then bin peaks radially and fit one
            // ellipse per ring with least-squares (no Monte Carlo over
            // 5-point ellipse samples).
            let center = estimate_ring_center(peaks, &options.center)
                .ok_or(AutoCalibrateError::NoEllipses)?;
            let radial = extract_rings_radial(peaks, (center.d1, center.d2), &options.radial);
            if radial.ellipses.is_empty() {
                return Err(AutoCalibrateError::NoEllipses);
            }
            (radial.ellipses, radial.assignments)
        }
        AutoMethod::Ransac => {
            let ransac = ransac_ellipses(peaks, &options.ransac);
            if ransac.ellipses.is_empty() {
                return Err(AutoCalibrateError::NoEllipses);
            }
            // Shared-center filter: drop spurious ellipses whose centers
            // are far from the consensus.
            let kept_indices = if ransac.ellipses.len() >= 3 && options.center_consistency_px > 0.0
            {
                consensus_center_filter(&ransac.ellipses, options.center_consistency_px)
            } else {
                (0..ransac.ellipses.len()).collect()
            };
            let kept_ellipses: Vec<crate::calibration::Ellipse> =
                kept_indices.iter().map(|&i| ransac.ellipses[i]).collect();
            if kept_ellipses.is_empty() {
                return Err(AutoCalibrateError::NoEllipses);
            }
            let old_to_new: std::collections::HashMap<usize, usize> = kept_indices
                .iter()
                .enumerate()
                .map(|(new_idx, &old_idx)| (old_idx, new_idx))
                .collect();
            let kept_assignments: Vec<Option<usize>> = ransac
                .assignments
                .iter()
                .map(|a| a.and_then(|k| old_to_new.get(&k).copied()))
                .collect();
            (kept_ellipses, kept_assignments)
        }
    };

    // Wrap into a result-like view for the rest of the pipeline.
    let ransac = super::ellipse::RansacResult {
        ellipses,
        assignments: point_assignments,
    };

    // 2. Identify which calibrant ring each ellipse is.
    let ident = identify_rings(&ransac.ellipses, calibrant, wavelength_m, &options.ring_id);
    if !ident.confident {
        return Err(AutoCalibrateError::NoConfidentIdentity);
    }

    // 3. Bootstrap PONI from the matched ellipses.
    //
    // Two regimes:
    //
    // - **Near-perpendicular** (median axis_ratio < ~1.05): the
    //   tilt-aware inversion is numerically degenerate because it
    //   solves `sin²(τ) = (1-C)/(1+tan²(2θ))` with `C = (b/a)²`, which
    //   amplifies ellipse-fit noise in `b/a` into spurious tilt
    //   estimates. The simple "PONI = ring center, rot = 0,
    //   D = mean(b·px/tan(2θ))" bootstrap is more robust here and the
    //   L-M refiner picks up the small rotations.
    //
    // - **Visibly tilted** (median axis_ratio ≥ 1.05): the closed-form
    //   inversion in `super::inversion` recovers (D, poni1, poni2,
    //   rot1, rot2) exactly for a flat tilted detector and we take the
    //   median across rings.
    let ring_tth = calibrant.two_theta_rad(wavelength_m);
    let mut centers_d1: Vec<f64> = Vec::new();
    let mut centers_d2: Vec<f64> = Vec::new();
    let mut axis_ratios: Vec<f64> = Vec::new();
    let mut perp_distances: Vec<f64> = Vec::new();
    let mut inversions: Vec<super::inversion::InvertedGeometry> = Vec::new();
    let pixel_mean = (pixel1 * pixel2).sqrt();
    for (e_idx, ring) in ident.ring_index.iter().enumerate() {
        if let Some(k) = ring
            && let Some(Some(t)) = ring_tth.get(*k)
        {
            let e = &ransac.ellipses[e_idx];
            centers_d1.push(e.center.0);
            centers_d2.push(e.center.1);
            axis_ratios.push(e.semi_major / e.semi_minor);
            let tan_2th = t.tan();
            if tan_2th > 0.0 {
                perp_distances.push((e.semi_minor * pixel_mean) / tan_2th);
            }
            if let Some(inv) = super::inversion::invert(e, *t, pixel1, pixel2) {
                inversions.push(inv);
            }
        }
    }
    if centers_d1.is_empty() || perp_distances.is_empty() {
        return Err(AutoCalibrateError::NoConfidentIdentity);
    }

    // Median axis-ratio decides which bootstrap regime to use.
    let median_axis_ratio = {
        let mut ratios = axis_ratios.clone();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ratios[ratios.len() / 2]
    };
    const TILT_THRESHOLD: f64 = 1.05;

    let cx = centers_d1.iter().sum::<f64>() / centers_d1.len() as f64;
    let cy = centers_d2.iter().sum::<f64>() / centers_d2.len() as f64;
    let bootstrap_ring_center_d1 = (cx + 0.5) * pixel1;
    let bootstrap_ring_center_d2 = (cy + 0.5) * pixel2;

    let (dist, poni1, poni2, rot1, rot2) = if median_axis_ratio >= TILT_THRESHOLD
        && !inversions.is_empty()
        && let Some(agg) = super::inversion::aggregate(&inversions)
    {
        (agg.distance, agg.poni1, agg.poni2, agg.rot1, agg.rot2)
    } else {
        let d = perp_distances.iter().sum::<f64>() / perp_distances.len() as f64;
        (
            d,
            bootstrap_ring_center_d1,
            bootstrap_ring_center_d2,
            0.0,
            0.0,
        )
    };

    let mut poni = Poni::new();
    poni.pixel1 = pixel1;
    poni.pixel2 = pixel2;
    poni.distance = dist;
    poni.poni1 = poni1;
    poni.poni2 = poni2;
    poni.rot1 = rot1;
    poni.rot2 = rot2;
    poni.rot3 = 0.0;
    poni.wavelength = wavelength_m;

    // 4. Convert matched ellipses' inlier peaks → CalibrationPoints.
    let mut pts = CalibrationPoints::new();
    for (i, slot) in ransac.assignments.iter().enumerate() {
        if let Some(e_idx) = slot
            && let Some(ring) = ident.ring_index.get(*e_idx).copied().flatten()
        {
            let (d1, d2) = peaks[i];
            pts.push(d1, d2, Some(ring));
        }
    }
    if pts.is_empty() {
        return Err(AutoCalibrateError::NoConfidentIdentity);
    }

    // 5. Run refine with the bootstrapped PONI.
    let geometry = Geometry::new(0, 0, poni);
    let refine_opts = CalibrationOptions {
        mask: options.refine_mask,
        loss: options.refine_loss,
        auto_assign_rings: false,
        assign_tolerance_rad: 0.05,
        max_iter: options.max_iter,
        tol: options.tol,
    };
    let result =
        refine(&geometry, &pts, calibrant, &refine_opts).map_err(AutoCalibrateError::Refine)?;

    let diagnostics = AutoCalibrateDiagnostics {
        n_detected_ellipses: ransac.ellipses.len(),
        n_identified_ellipses: ident.n_matched,
        bootstrap_distance: dist,
        bootstrap_ring_center_d1,
        bootstrap_ring_center_d2,
    };
    Ok((result, diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn lab6() -> Calibrant {
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").unwrap()
    }

    /// Geometrically project a Debye-Scherrer ring of 2θ onto a flat
    /// detector at distance `dist` with the beam center at pixel
    /// `(cx, cy)`, returning N points on the ring (no tilt, no rotation).
    fn ring_points(
        dist_m: f64,
        cx_px: f64,
        cy_px: f64,
        pixel: f64,
        tth: f64,
        n: usize,
    ) -> Vec<(f64, f64)> {
        let r_px = (dist_m * tth.tan()) / pixel;
        (0..n)
            .map(|i| {
                let a = 2.0 * PI * (i as f64) / (n as f64);
                (cx_px + r_px * a.cos(), cy_px + r_px * a.sin())
            })
            .collect()
    }

    #[test]
    fn auto_calibrate_recovers_synthetic_geometry() {
        // Ground-truth synthetic LaB6 image-equivalent: rings at known
        // distance, no tilt, beam at (1000, 1000) pixels, 100 µm pitch.
        let cal = lab6();
        let wl = 1.5406e-10;
        let dist = 0.200;
        let cx_px = 1000.0;
        let cy_px = 1000.0;
        let pix = 1.0e-4;
        let tth = cal.two_theta_rad(wl);
        // Use rings 0..5, with plenty of points per ring so 5-point RANSAC
        // samples can land on a single ring with reasonable probability.
        let mut peaks: Vec<(f64, f64)> = Vec::new();
        for t in tth.iter().take(6).flatten() {
            peaks.extend(ring_points(dist, cx_px, cy_px, pix, *t, 60));
        }
        // Tune RANSAC for this synthetic, point-dense case.
        let opts = AutoCalibrateOptions {
            ransac: RansacOptions {
                max_iterations: 8000,
                min_inliers: 20,
                ..AutoCalibrateOptions::default().ransac
            },
            ..AutoCalibrateOptions::default()
        };
        let (result, diag) =
            auto_calibrate(&peaks, &cal, wl, pix, pix, &opts).expect("auto_calibrate ok");

        // Refined PONI should match the truth to numerical precision on
        // synthetic data. Allow ~1 µm of tolerance (the bootstrap is a
        // few hundred µm off; refine should converge to truth).
        assert!(
            (result.poni.distance - dist).abs() < 1e-6,
            "distance: {} vs {}",
            result.poni.distance,
            dist
        );
        // Beam center in meters: (cx + 0.5) * pix
        let truth_p1 = (cx_px + 0.5) * pix;
        let truth_p2 = (cy_px + 0.5) * pix;
        assert!((result.poni.poni1 - truth_p1).abs() < 1e-6);
        assert!((result.poni.poni2 - truth_p2).abs() < 1e-6);
        assert!(result.poni.rot1.abs() < 1e-6);
        assert!(result.poni.rot2.abs() < 1e-6);
        // The bootstrap should already be close.
        assert!((diag.bootstrap_distance - dist).abs() < dist * 0.05);
        assert!(diag.n_identified_ellipses >= 5);
    }
}
