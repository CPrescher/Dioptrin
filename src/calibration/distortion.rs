//! Spatial detector distortion estimation from powder calibration rings.
//!
//! Given a converged [`Poni`] and ring control points [`CalibrationPoints`]
//! (e.g. from [`super::extract_ring_peaks`], [`super::refine_image`], or manual
//! picking), estimates systematic detector spatial distortion and converts the
//! result directly into a [`crate::detector::spline::Spline`] (Fit2D / pyFAI
//! `.spline` format) or a [`crate::detector::Detector`] (`PixelMap::Corners`
//! for 3D bow / per-module models).
//!
//! ## Observability & Gauge Freedom
//!
//! Powder Debye–Scherrer rings measure the **radial** component of detector
//! distortion (`∂2θ/∂d₁ · δd₁ + ∂2θ/∂d₂ · δd₂ + ∂2θ/∂z · δz ≈ 2θ_target − 2θ_0`),
//! and are invariant to purely tangential twist along the rings. The lowest-order
//! radial modes are degenerate with rigid `(distance, poni1, poni2, rot1, rot2)`,
//! so each model excludes or regularizes rigid modes and optionally alternates
//! PONI re-refinement with distortion fitting.
//!
//! ## Supported Models
//!
//! - [`DistortionModel::RadialOptical`]: radial pincushion/barrel (`r³, r⁵, ...`)
//!   plus low-order anisotropic/astigmatic/coma terms for CCDs and fiber-optic tapers.
//! - [`DistortionModel::SmoothSpline`]: regularized 2D cubic tensor-product
//!   B-spline displacement `(δd₁, δd₂)` matching [`crate::detector::spline::Spline`].
//! - [`DistortionModel::BowZ`]: smooth out-of-plane detector bow `δz(d₁, d₂)`
//!   (degree `2..=max_degree`, excluding constant/tilt) for large flat panels.
//! - [`DistortionModel::PerModule`]: rigid `(Δd₁, Δd₂)` (and optional `Δz`)
//!   offsets per tile for multi-module detectors (Pilatus, Eiger, Jungfrau).

use std::fmt;
use std::sync::Arc;

use rayon::prelude::*;

use crate::detector::spline::{Spline, Surface, bspline_basis};
use crate::detector::{Detector, PixelMap};
use crate::geometry::Geometry;
use crate::geometry::math::{TrigonometricConstants, f_tth, f_x1, f_x2, f_x3};
use crate::poni::Poni;

use super::calibrant::Calibrant;
use super::refinement::{CalibrationOptions, CalibrationPoints, Loss, ParamMask, refine};

/// Spatial distortion parameterization to fit from ring residuals.
#[derive(Debug, Clone, PartialEq)]
pub enum DistortionModel {
    /// Optical / fiber-optic taper distortion: radial pincushion/barrel
    /// (`r³, r⁵, …` up to `max_order`, clamped to `3..=7`) plus low-order
    /// anisotropic harmonic terms (`r² cos/sin 2φ`, `r³ cos/sin 3φ`, …).
    RadialOptical {
        /// Maximum radial polynomial order (default `5`, valid `3..=7`).
        max_order: usize,
    },
    /// Regularized 2D cubic tensor-product B-spline displacement field
    /// `(δd₁, δd₂)` in pixels — directly compatible with Fit2D / pyFAI `.spline`.
    SmoothSpline {
        /// Number of spline control points along fast/col (`d2`) axis (default `8`, `>= 4`).
        grid_nx: usize,
        /// Number of spline control points along slow/row (`d1`) axis (default `8`, `>= 4`).
        grid_ny: usize,
        /// Second-difference bending-energy smoothness weight (default `1.0`, `> 0`).
        smoothness: f64,
    },
    /// Smooth out-of-plane detector bow `δz(d₁, d₂)` in metres (polynomial
    /// terms of total degree `2..=max_degree`, excluding constant/tilt which
    /// are degenerate with `distance, rot1, rot2`).
    BowZ {
        /// Maximum polynomial degree (default `3`, valid `2..=5`).
        max_degree: usize,
        /// Mild ridge regularization weight (default `1e-3`).
        smoothness: f64,
    },
    /// Per-module rigid offsets `(Δd₁, Δd₂)` (and optional `Δz`) for tiled
    /// multi-module detectors (Pilatus, Eiger, Jungfrau).
    PerModule {
        /// Height of each active module in pixels (slow/row axis).
        module_height: usize,
        /// Width of each active module in pixels (fast/col axis).
        module_width: usize,
        /// Inter-module gap height in pixels (slow/row axis).
        gap_height: usize,
        /// Inter-module gap width in pixels (fast/col axis).
        gap_width: usize,
        /// Also fit per-module out-of-plane offset `Δz` in metres.
        fit_z: bool,
        /// Ridge regularization weight keeping poorly-constrained modules near 0
        /// (default `1e-2`).
        regularization: f64,
    },
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self::SmoothSpline {
            grid_nx: 8,
            grid_ny: 8,
            smoothness: 1.0,
        }
    }
}

/// Configuration for [`estimate_distortion`].
#[derive(Debug, Clone)]
pub struct DistortionOptions {
    /// Distortion model family and hyperparameters.
    pub model: DistortionModel,
    /// Huber IRLS threshold in pixels (default `1.5` px).
    pub huber_c_px: f64,
    /// Alternate PONI re-refinement with distortion fitting so rigid
    /// `(distance, poni1, poni2, rot1, rot2)` absorb all rigid modes (default `true`).
    pub refine_poni: bool,
    /// Parameter mask used when `refine_poni` is `true` (`rot3` is always kept fixed).
    pub refine_mask: ParamMask,
    /// Number of cross-validation folds for out-of-sample `cv_rms_px` and
    /// `significant` gating (default `5`; `0` disables CV).
    pub cv_folds: usize,
}

impl Default for DistortionOptions {
    fn default() -> Self {
        let mut refine_mask = ParamMask::fix_wavelength();
        refine_mask.rot3 = false;
        Self {
            model: DistortionModel::default(),
            huber_c_px: 1.5,
            refine_poni: true,
            refine_mask,
            cv_folds: 5,
        }
    }
}

/// Rigid alignment offset fitted for one detector tile in [`DistortionModel::PerModule`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleOffset {
    /// Module grid index along slow/row axis (`0..n_mod_rows`).
    pub row_index: usize,
    /// Module grid index along fast/col axis (`0..n_mod_cols`).
    pub col_index: usize,
    /// Fitted slow/row shift `Δd₁` in pixels.
    pub dd1_px: f64,
    /// Fitted fast/col shift `Δd₂` in pixels.
    pub dd2_px: f64,
    /// Fitted out-of-plane shift `Δz` in metres (`0.0` when `fit_z == false`).
    pub dz_m: f64,
    /// Number of calibration control points that fell on this module.
    pub n_points: usize,
}

/// Fitted spatial distortion field, evaluable at arbitrary detector pixel
/// coordinates and convertible into a [`Spline`] or [`Detector`].
#[derive(Debug, Clone)]
pub enum DistortionField {
    RadialOptical {
        center_d1: f64,
        center_d2: f64,
        r_scale: f64,
        max_order: usize,
        coeffs: Vec<f64>,
    },
    SmoothSpline {
        /// Slow/row displacement `δd₁(col, row)` in pixels.
        surface_d1: Surface,
        /// Fast/col displacement `δd₂(col, row)` in pixels.
        surface_d2: Surface,
    },
    BowZ {
        center_d1: f64,
        center_d2: f64,
        scale_d1: f64,
        scale_d2: f64,
        max_degree: usize,
        /// Polynomial coefficients in metres for `(x^i y^j)` with `2 <= i+j <= max_degree`.
        coeffs: Vec<f64>,
    },
    PerModule {
        module_height: usize,
        module_width: usize,
        gap_height: usize,
        gap_width: usize,
        n_mod_rows: usize,
        n_mod_cols: usize,
        offsets: Vec<ModuleOffset>,
    },
}

impl DistortionField {
    /// Evaluate the in-plane pixel displacement `(δd₁, δd₂)` (slow/row, fast/col)
    /// at detector pixel coordinate `(d1, d2)`.
    pub fn displacement_px(&self, d1: f64, d2: f64) -> (f64, f64) {
        match self {
            Self::RadialOptical {
                center_d1,
                center_d2,
                r_scale,
                max_order,
                coeffs,
            } => eval_radial_optical(*center_d1, *center_d2, *r_scale, *max_order, coeffs, d1, d2),
            Self::SmoothSpline {
                surface_d1,
                surface_d2,
            } => (surface_d1.evaluate(d2, d1), surface_d2.evaluate(d2, d1)),
            Self::BowZ { .. } => (0.0, 0.0),
            Self::PerModule {
                module_height,
                module_width,
                gap_height,
                gap_width,
                n_mod_rows,
                n_mod_cols,
                offsets,
            } => {
                if let Some(m) = module_index(
                    d1,
                    d2,
                    *module_height,
                    *module_width,
                    *gap_height,
                    *gap_width,
                    *n_mod_rows,
                    *n_mod_cols,
                ) {
                    let o = &offsets[m];
                    (o.dd1_px, o.dd2_px)
                } else {
                    (0.0, 0.0)
                }
            }
        }
    }

    /// Evaluate the out-of-plane displacement `δz` in metres at `(d1, d2)`.
    pub fn dz_m(&self, d1: f64, d2: f64) -> f64 {
        match self {
            Self::BowZ {
                center_d1,
                center_d2,
                scale_d1,
                scale_d2,
                max_degree,
                coeffs,
            } => {
                let x = (d2 - center_d2) / scale_d2;
                let y = (d1 - center_d1) / scale_d1;
                let mut sum = 0.0;
                let mut idx = 0usize;
                for deg in 2..=*max_degree {
                    for ix in 0..=deg {
                        let iy = deg - ix;
                        sum += coeffs[idx] * x.powi(ix as i32) * y.powi(iy as i32);
                        idx += 1;
                    }
                }
                sum
            }
            Self::PerModule {
                module_height,
                module_width,
                gap_height,
                gap_width,
                n_mod_rows,
                n_mod_cols,
                offsets,
            } => {
                if let Some(m) = module_index(
                    d1,
                    d2,
                    *module_height,
                    *module_width,
                    *gap_height,
                    *gap_width,
                    *n_mod_rows,
                    *n_mod_cols,
                ) {
                    offsets[m].dz_m
                } else {
                    0.0
                }
            }
            Self::RadialOptical { .. } | Self::SmoothSpline { .. } => 0.0,
        }
    }

    /// Convert a smooth in-plane distortion field (`SmoothSpline` or
    /// `RadialOptical`) into a validated Fit2D / pyFAI [`Spline`].
    ///
    /// Returns `Err` for 3D / discontinuous models (`BowZ`, `PerModule`), which
    /// should use [`Self::to_detector`] (`PixelMap::Corners`) instead.
    pub fn to_spline(
        &self,
        dim1: usize,
        dim2: usize,
        pixel1: f64,
        pixel2: f64,
    ) -> Result<Spline, String> {
        let region = [0.0, 0.0, dim2 as f64, dim1 as f64];
        let pixels_um = [pixel2 * 1e6, pixel1 * 1e6];
        match self {
            Self::SmoothSpline {
                surface_d1,
                surface_d2,
            } => {
                // Shift knots by +0.5 so `Detector::position` (which evaluates
                // the spline at pixel centers `d + 0.5`) matches the pixel-index
                // coordinates `d` used during fitting.
                let shift_surface = |s: &Surface| Surface {
                    x: s.x.iter().map(|&k| k + 0.5).collect(),
                    y: s.y.iter().map(|&k| k + 0.5).collect(),
                    coefficients: s.coefficients.clone(),
                };
                let sx = shift_surface(surface_d2);
                let sy = shift_surface(surface_d1);
                let grid = (dim1.max(dim2) as f64) / ((sx.x.len() - 4).max(1) as f64);
                Spline::from_surfaces(region, pixels_um, grid, sx, sy)
            }
            Self::RadialOptical { .. } => {
                // Project the smooth radial-optical field onto a 12×12 cubic
                // B-spline grid via exact least-squares on a 32×32 sample grid,
                // shifted by +0.5 to match `Detector::position` pixel centers.
                let nx = 12usize;
                let ny = 12usize;
                let tx = clamped_uniform_knots(0.5, dim2 as f64 + 0.5, nx);
                let ty = clamped_uniform_knots(0.5, dim1 as f64 + 0.5, ny);
                let mut xs = Vec::with_capacity(32 * 32);
                let mut ys = Vec::with_capacity(32 * 32);
                let mut d1_vals = Vec::with_capacity(32 * 32);
                let mut d2_vals = Vec::with_capacity(32 * 32);
                for ir in 0..32 {
                    let y = 0.5 + (ir as f64 / 31.0) * dim1 as f64;
                    for ic in 0..32 {
                        let x = 0.5 + (ic as f64 / 31.0) * dim2 as f64;
                        let (dd1, dd2) = self.displacement_px(y - 0.5, x - 0.5);
                        xs.push(x);
                        ys.push(y);
                        d1_vals.push(dd1);
                        d2_vals.push(dd2);
                    }
                }
                let coeff_d1 = fit_bspline_surface_ls(&tx, &ty, &xs, &ys, &d1_vals, 1e-8)?;
                let coeff_d2 = fit_bspline_surface_ls(&tx, &ty, &xs, &ys, &d2_vals, 1e-8)?;
                let surf_x = Surface {
                    x: tx,
                    y: ty.clone(),
                    coefficients: coeff_d2,
                };
                let surf_y = Surface {
                    x: surf_x.x.clone(),
                    y: ty,
                    coefficients: coeff_d1,
                };
                let grid = (dim1.max(dim2) as f64) / (nx as f64);
                Spline::from_surfaces(region, pixels_um, grid, surf_x, surf_y)
            }
            Self::BowZ { .. } => {
                Err("BowZ is a 3D out-of-plane distortion; use to_detector() instead".into())
            }
            Self::PerModule { .. } => Err(
                "PerModule is a discontinuous tile distortion; use to_detector() instead".into(),
            ),
        }
    }

    /// Build a [`Detector`] carrying this distortion field:
    /// - `SmoothSpline` / `RadialOptical`: attaches a [`Spline`] to `base` (or
    ///   a new custom detector).
    /// - `BowZ` / `PerModule`: populates `PixelMap::Corners` with exact 3D
    ///   `[z, slow, fast]` pixel corners in metres.
    pub fn to_detector(
        &self,
        dim1: usize,
        dim2: usize,
        pixel1: f64,
        pixel2: f64,
        base: Option<&Detector>,
    ) -> Result<Detector, String> {
        let mut det = match base {
            Some(b) => b.clone(),
            None => Detector::custom(pixel1, pixel2),
        };
        det.pixel1 = pixel1;
        det.pixel2 = pixel2;
        det.max_shape = Some([dim1, dim2]);

        match self {
            Self::SmoothSpline { .. } | Self::RadialOptical { .. } => {
                let sp = self.to_spline(dim1, dim2, pixel1, pixel2)?;
                det.spline = Some(Arc::new(sp));
                det.validate()?;
                Ok(det)
            }
            Self::BowZ { .. } => {
                // Precompute the (dim1 + 1) × (dim2 + 1) corner vertex grid in
                // parallel so every interior corner is evaluated once instead of 4×.
                // Shift by -0.5 to match pixel-index coordinates used in fitting.
                let stride = dim2 + 1;
                let grid: Vec<[f64; 3]> = (0..=dim1)
                    .into_par_iter()
                    .flat_map_iter(|r| {
                        let rc = r as f64;
                        (0..=dim2).map(move |c| {
                            let cc = c as f64;
                            let dz = self.dz_m(rc - 0.5, cc - 0.5);
                            if let Some(b) = base {
                                let p = b.position([dim1, dim2], rc, cc);
                                [p[0] + dz, p[1], p[2]]
                            } else {
                                [dz, rc * pixel1, cc * pixel2]
                            }
                        })
                    })
                    .collect();
                let corners: Vec<[[f64; 3]; 4]> = (0..dim1)
                    .into_par_iter()
                    .flat_map_iter(|r| {
                        let g = &grid;
                        (0..dim2).map(move |c| {
                            [
                                g[r * stride + c],
                                g[(r + 1) * stride + c],
                                g[(r + 1) * stride + c + 1],
                                g[r * stride + c + 1],
                            ]
                        })
                    })
                    .collect();
                det.spline = None;
                // `base.position` already applied `base.orientation`; reset to
                // identity orientation (3) so corners are not flipped twice.
                det.orientation = 3;
                det.map = PixelMap::Corners {
                    shape: [dim1, dim2],
                    values: Arc::new(corners),
                };
                det.validate()?;
                Ok(det)
            }
            Self::PerModule { .. } => {
                let corners: Vec<[[f64; 3]; 4]> = (0..dim1)
                    .into_par_iter()
                    .flat_map_iter(|r| {
                        let r0 = r as f64;
                        let r1 = r0 + 1.0;
                        (0..dim2).map(move |c| {
                            let c0 = c as f64;
                            let c1 = c0 + 1.0;
                            let (dd1, dd2) = self.displacement_px(r0 + 0.5, c0 + 0.5);
                            let dz = self.dz_m(r0 + 0.5, c0 + 0.5);
                            let make_corner = |rc: f64, cc: f64| -> [f64; 3] {
                                if let Some(b) = base {
                                    let p = b.position([dim1, dim2], rc, cc);
                                    [p[0] + dz, p[1] + dd1 * pixel1, p[2] + dd2 * pixel2]
                                } else {
                                    [dz, (rc + dd1) * pixel1, (cc + dd2) * pixel2]
                                }
                            };
                            [
                                make_corner(r0, c0),
                                make_corner(r1, c0),
                                make_corner(r1, c1),
                                make_corner(r0, c1),
                            ]
                        })
                    })
                    .collect();
                det.spline = None;
                det.orientation = 3;
                det.map = PixelMap::Corners {
                    shape: [dim1, dim2],
                    values: Arc::new(corners),
                };
                det.validate()?;
                Ok(det)
            }
        }
    }
}

/// One calibration exposure contributing to [`estimate_distortion_multi`].
///
/// Combining multiple exposures measured at different detector positions
/// (`poni1`, `poni2`, `distance`) or wavelengths breaks single-position
/// tangential and rigid-PONI degeneracies because rings cross the same pixels
/// at different angles and `2θ` values.
#[derive(Debug, Clone, Copy)]
pub struct DistortionObservation<'a> {
    pub points: &'a CalibrationPoints,
    pub poni: &'a Poni,
    pub calibrant: &'a Calibrant,
}

/// Result of [`estimate_distortion`] or [`estimate_distortion_multi`].
#[derive(Debug, Clone)]
pub struct DistortionResult {
    /// Re-refined PONI of the first observation (unchanged if `options.refine_poni == false`).
    pub poni: Poni,
    /// Re-refined PONI for every input observation (length 1 for [`estimate_distortion`]).
    pub ponis: Vec<Poni>,
    /// Model configuration that was fitted.
    pub model: DistortionModel,
    /// Fitted distortion field.
    pub field: DistortionField,
    /// Radial 2θ RMS before distortion correction (radians).
    pub rms_before_rad: f64,
    /// Radial 2θ RMS after distortion correction (radians).
    pub rms_after_rad: f64,
    /// Radial residual RMS before distortion correction (pixels).
    pub rms_before_px: f64,
    /// Radial residual RMS after distortion correction (pixels).
    pub rms_after_px: f64,
    /// Out-of-sample K-fold cross-validated radial RMS (pixels); equals
    /// `rms_after_px` if `cv_folds < 2`.
    pub cv_rms_px: f64,
    /// Whether the distortion correction produces a statistically meaningful
    /// out-of-sample improvement (`cv_rms_px < 0.95 * rms_before_px`).
    pub significant: bool,
    /// Maximum in-plane displacement magnitude across the control points (pixels).
    pub max_displacement_px: f64,
    /// Number of valid ring-assigned control points used.
    pub n_points_used: usize,
    /// Signed radial residual per input point before correction (pixels; `0.0` for unassigned).
    pub point_residuals_before_px: Vec<f64>,
    /// Signed radial residual per input point after correction (pixels; `0.0` for unassigned).
    pub point_residuals_after_px: Vec<f64>,
    /// Per-module fitted offsets (populated for [`DistortionModel::PerModule`]).
    pub module_offsets: Vec<ModuleOffset>,
}

/// Failure modes for [`estimate_distortion`].
#[derive(Debug, Clone)]
pub enum DistortionError {
    InvalidInput(String),
    InsufficientPoints(String),
    Singular(String),
}

impl fmt::Display for DistortionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid distortion input: {msg}"),
            Self::InsufficientPoints(msg) => write!(f, "insufficient control points: {msg}"),
            Self::Singular(msg) => write!(f, "singular distortion system: {msg}"),
        }
    }
}

impl std::error::Error for DistortionError {}

/// Invert a [`DistortionField`] at an ideal undistorted pixel coordinate
/// `(d1_ideal, d2_ideal)` via damped fixed-point iterations, returning the
/// observed detector pixel `(d1, d2)` where the distorted ring physically lands.
///
/// Returns `None` if the displacement is non-finite, fails to converge within
/// `0.25 px` (e.g. bouncing across a `PerModule` gap), or exceeds `max_shift_px`
/// (guarding against wild polynomial extrapolation in unobserved corners).
pub(crate) fn invert_distortion_pixel(
    field: &DistortionField,
    poni: &Poni,
    tc: &TrigonometricConstants,
    d1_ideal: f64,
    d2_ideal: f64,
    max_shift_px: f64,
) -> Option<(f64, f64)> {
    const DAMP: f64 = 0.7;
    let mut d1 = d1_ideal;
    let mut d2 = d2_ideal;
    for _ in 0..8 {
        let (dd1, dd2) = field.displacement_px(d1, d2);
        let dz = field.dz_m(d1, d2);
        let (dz_d1, dz_d2) = if dz.abs() > 0.0 {
            let (g1, g2, gz) = tth_sensitivities(poni, tc, d1, d2);
            let g2_norm = g1 * g1 + g2 * g2;
            if g2_norm.is_finite() && g2_norm > 1e-24 {
                let s = (gz * dz) / g2_norm;
                (s * g1, s * g2)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };
        if !dd1.is_finite() || !dd2.is_finite() || !dz_d1.is_finite() || !dz_d2.is_finite() {
            return None;
        }
        let target_d1 = d1_ideal - dd1 - dz_d1;
        let target_d2 = d2_ideal - dd2 - dz_d2;
        let err = (target_d1 - d1).hypot(target_d2 - d2);
        d1 += DAMP * (target_d1 - d1);
        d2 += DAMP * (target_d2 - d2);
        if err < 1e-3 {
            break;
        }
    }
    // Verify convergence and bound total shift against runaway extrapolation.
    let (dd1, dd2) = field.displacement_px(d1, d2);
    let dz = field.dz_m(d1, d2);
    let (dz_d1, dz_d2) = if dz.abs() > 0.0 {
        let (g1, g2, gz) = tth_sensitivities(poni, tc, d1, d2);
        let g2_norm = g1 * g1 + g2 * g2;
        if g2_norm.is_finite() && g2_norm > 1e-24 {
            let s = (gz * dz) / g2_norm;
            (s * g1, s * g2)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };
    if !dd1.is_finite() || !dd2.is_finite() || !dz_d1.is_finite() || !dz_d2.is_finite() {
        return None;
    }
    let residual = (d1 + dd1 + dz_d1 - d1_ideal).hypot(d2 + dd2 + dz_d2 - d2_ideal);
    let shift = (d1 - d1_ideal).hypot(d2 - d2_ideal);
    if residual > 0.25 || shift > max_shift_px {
        return None;
    }
    Some((d1, d2))
}

/// Invert a physical [`Detector`] mapping (`.spline`, `PixelMap::Corners`,
/// `PixelMap::Separable`, or non-identity orientation) at an ideal flat-detector
/// pixel coordinate `(d1_ideal, d2_ideal)` via damped fixed-point iterations.
///
/// Returns the observed detector pixel `(d1, d2)` whose physical position
/// `det.position([dim1, dim2], d1 + 0.5, d2 + 0.5)` produces the same `2θ` as
/// the ideal flat pixel `(d1_ideal, d2_ideal)`. Returns `None` if the mapping
/// is non-finite, fails to converge within `0.25 px`, or shifts beyond
/// `max_shift_px`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invert_detector_pixel(
    det: &Detector,
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    tc: &TrigonometricConstants,
    d1_ideal: f64,
    d2_ideal: f64,
    max_shift_px: f64,
) -> Option<(f64, f64)> {
    const DAMP: f64 = 0.7;
    let s1 = if matches!(det.orientation, 1 | 2) {
        -1.0
    } else {
        1.0
    };
    let s2 = if matches!(det.orientation, 1 | 4) {
        -1.0
    } else {
        1.0
    };
    let d1_init = if s1 < 0.0 {
        dim1 as f64 - 1.0 - d1_ideal
    } else {
        d1_ideal
    };
    let d2_init = if s2 < 0.0 {
        dim2 as f64 - 1.0 - d2_ideal
    } else {
        d2_ideal
    };
    let mut d1 = d1_init;
    let mut d2 = d2_init;
    let inv_p1 = 1.0 / poni.pixel1;
    let inv_p2 = 1.0 / poni.pixel2;
    let slow_ideal = (d1_ideal + 0.5) * poni.pixel1;
    let fast_ideal = (d2_ideal + 0.5) * poni.pixel2;

    let step_residual = |r: f64, c: f64| -> Option<(f64, f64)> {
        let pos = det.position([dim1, dim2], r + 0.5, c + 0.5);
        if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
            return None;
        }
        let dz = pos[0];
        let (dz_d1, dz_d2) = if dz.abs() > 0.0 {
            let r_eff = pos[1] * inv_p1 - 0.5;
            let c_eff = pos[2] * inv_p2 - 0.5;
            let (g1, g2, gz) = tth_sensitivities(poni, tc, r_eff, c_eff);
            let g2_norm = g1 * g1 + g2 * g2;
            if g2_norm.is_finite() && g2_norm > 1e-24 {
                let s = (gz * dz) / g2_norm;
                (s * g1, s * g2)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };
        if !dz_d1.is_finite() || !dz_d2.is_finite() {
            return None;
        }
        let err1 = ((slow_ideal - pos[1]) * inv_p1 - dz_d1) * s1;
        let err2 = ((fast_ideal - pos[2]) * inv_p2 - dz_d2) * s2;
        Some((err1, err2))
    };

    for _ in 0..8 {
        let (err1, err2) = step_residual(d1, d2)?;
        d1 += DAMP * err1;
        d2 += DAMP * err2;
        if err1.hypot(err2) < 1e-3 {
            break;
        }
    }
    let (err1, err2) = step_residual(d1, d2)?;
    let residual = err1.hypot(err2);
    let shift = (d1 - d1_init).hypot(d2 - d2_init);
    if residual > 0.25 || shift > max_shift_px {
        return None;
    }
    Some((d1, d2))
}

/// One 2D diffraction image exposure for [`estimate_distortion_image_multi`].
#[derive(Clone, Copy)]
pub struct DistortionImageObservation<'a> {
    pub image: &'a [f64],
    pub dim1: usize,
    pub dim2: usize,
    pub mask: Option<&'a [bool]>,
    pub poni: &'a Poni,
    pub calibrant: &'a Calibrant,
}

/// Configuration for [`estimate_distortion_image`] and
/// [`estimate_distortion_image_multi`].
#[derive(Debug, Clone)]
pub struct DistortionImageOptions {
    /// Underlying distortion model, IRLS, PONI refinement, and CV settings.
    pub distortion: DistortionOptions,
    /// Number of iterative `(re-pick along distorted rings → fit distortion)`
    /// passes (default `3`, clamped to `1..=6`). Pass 0 uses the undistorted
    /// PONI; passes `1..n_passes` project rings through the current
    /// [`DistortionField`] so outer rings with large (> 4 px) distortion stay
    /// centered inside the radial search window.
    pub n_passes: usize,
    /// Maximum calibrant ring index to consider (default `100`).
    pub max_ring_index: usize,
    /// Optional cap on number of rings; `None` (default) uses all visible rings
    /// up to `max_ring_index`.
    pub max_rings: Option<usize>,
    /// Base azimuthal sampling density around each ring (default `72`).
    pub n_azimuth: usize,
    /// Optional explicit intensity ceiling; `None` auto-detects saturation plateaus.
    pub intensity_limit: Option<f64>,
    /// Optional fixed radial search half-width in pixels; `None` shrinks
    /// progressively (`8 → 6 → 4 px`) across passes.
    pub search_half_width_px: Option<usize>,
}

impl Default for DistortionImageOptions {
    fn default() -> Self {
        Self {
            distortion: DistortionOptions::default(),
            n_passes: 3,
            max_ring_index: 100,
            max_rings: None,
            n_azimuth: 72,
            intensity_limit: None,
            search_half_width_px: None,
        }
    }
}

/// Output of [`estimate_distortion_image`] / [`estimate_distortion_image_multi`].
#[derive(Debug, Clone)]
pub struct DistortionImageResult {
    /// Final distortion fit from the last re-picking pass.
    pub distortion: DistortionResult,
    /// Final re-picked control points for each input image exposure.
    pub points: Vec<CalibrationPoints>,
    /// Number of rings with control points on each input image exposure.
    pub n_rings: Vec<usize>,
    /// Baseline undistorted radial RMS (pixels) from pass 0 before any
    /// distortion-guided re-picking.
    pub initial_rms_px: f64,
    /// Number of re-picking passes actually completed.
    pub n_passes_completed: usize,
}

/// Iteratively pick ring peaks from a 2D image and estimate spatial detector
/// distortion, re-projecting rings through the fitted [`DistortionField`] on
/// each pass so outer rings with large (> 4 px) distortion are pulled in cleanly.
#[allow(clippy::too_many_arguments)]
pub fn estimate_distortion_image(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    calibrant: &Calibrant,
    options: &DistortionImageOptions,
) -> Result<DistortionImageResult, DistortionError> {
    estimate_distortion_image_multi(
        &[DistortionImageObservation {
            image,
            dim1,
            dim2,
            mask,
            poni,
            calibrant,
        }],
        options,
    )
}

/// Multi-exposure version of [`estimate_distortion_image`]: precomputes the
/// DoG residual and local noise grid once per image, then alternates
/// distortion-guided ring re-picking across all exposures with
/// [`estimate_distortion_multi`].
pub fn estimate_distortion_image_multi(
    observations: &[DistortionImageObservation<'_>],
    options: &DistortionImageOptions,
) -> Result<DistortionImageResult, DistortionError> {
    if observations.is_empty() {
        return Err(DistortionError::InsufficientPoints(
            "no image observations provided".into(),
        ));
    }
    let dim1 = observations[0].dim1;
    let dim2 = observations[0].dim2;
    if dim1 == 0 || dim2 == 0 {
        return Err(DistortionError::InvalidInput(
            "detector dimensions must be positive".into(),
        ));
    }
    let explicit_upper = options
        .intensity_limit
        .filter(|v| v.is_finite() && *v > 0.0);

    let mut prepared = Vec::with_capacity(observations.len());
    let mut cur_ponis = Vec::with_capacity(observations.len());
    for (e, obs) in observations.iter().enumerate() {
        if obs.dim1 != dim1 || obs.dim2 != dim2 {
            return Err(DistortionError::InvalidInput(format!(
                "observation {e} shape ({}, {}) does not match ({dim1}, {dim2})",
                obs.dim1, obs.dim2
            )));
        }
        let expected = dim1
            .checked_mul(dim2)
            .ok_or_else(|| DistortionError::InvalidInput("image dimensions overflow".into()))?;
        if obs.image.len() != expected {
            return Err(DistortionError::InvalidInput(format!(
                "observation {e} image length {} does not match {dim1}x{dim2}",
                obs.image.len()
            )));
        }
        if let Some(m) = obs.mask
            && m.len() != expected
        {
            return Err(DistortionError::InvalidInput(format!(
                "observation {e} mask length {} does not match {expected}",
                m.len()
            )));
        }
        let prep = super::image_auto::PreparedRingImage::prepare(
            obs.image,
            obs.mask,
            dim1,
            dim2,
            explicit_upper,
        )
        .ok_or_else(|| {
            DistortionError::InsufficientPoints(format!(
                "observation {e} has insufficient signal variance for ring extraction"
            ))
        })?;
        prepared.push(prep);
        cur_ponis.push(obs.poni.clone());
    }

    let n_passes = options.n_passes.clamp(1, 6);
    let n_az = options.n_azimuth.clamp(12, 360);
    let max_ring_index = options.max_ring_index.max(1);

    let mut cur_field: Option<DistortionField> = None;
    let mut last_result: Option<DistortionResult> = None;
    let mut last_points: Vec<CalibrationPoints> = Vec::new();
    let mut last_n_rings: Vec<usize> = Vec::new();
    let mut last_input_ponis: Vec<Poni> = Vec::new();
    let mut initial_rms_px = f64::NAN;
    let mut passes_done = 0usize;
    let mut ran_final_cv = false;

    for pass in 0..n_passes {
        // Progressive shrinking window: wider on pass 0 to catch distorted
        // outer rings before any correction, tightening on later passes once
        // ring trajectories follow the fitted distortion field.
        let hw = options.search_half_width_px.unwrap_or(match pass {
            0 => 8,
            1 => 6,
            _ => 4,
        });
        let tangent_half = if pass == 0 { 2 } else { 3 };

        let mut pass_pts = Vec::with_capacity(observations.len());
        let mut pass_n_rings = Vec::with_capacity(observations.len());
        for (e, obs) in observations.iter().enumerate() {
            let (pts, nr) = prepared[e].extract_with_distortion(
                &cur_ponis[e],
                obs.calibrant,
                hw,
                n_az,
                options.max_rings,
                max_ring_index,
                tangent_half,
                cur_field.as_ref(),
            );
            pass_pts.push(pts);
            pass_n_rings.push(nr);
        }

        // Skip expensive CV on intermediate passes; run full CV on the final pass.
        let is_final_pass = pass + 1 == n_passes;
        let mut pass_opts = options.distortion.clone();
        if !is_final_pass {
            pass_opts.cv_folds = 0;
        }
        let obs_refs: Vec<DistortionObservation<'_>> = observations
            .iter()
            .enumerate()
            .map(|(e, o)| DistortionObservation {
                points: &pass_pts[e],
                poni: &cur_ponis[e],
                calibrant: o.calibrant,
            })
            .collect();

        match estimate_distortion_multi(&obs_refs, dim1, dim2, &pass_opts) {
            Ok(res) => {
                if pass == 0 {
                    initial_rms_px = res.rms_before_px;
                }
                last_input_ponis = cur_ponis;
                cur_ponis = res.ponis.clone();
                cur_field = Some(res.field.clone());
                last_result = Some(res);
                last_points = pass_pts;
                last_n_rings = pass_n_rings;
                passes_done = pass + 1;
                ran_final_cv = is_final_pass;
            }
            Err(e) => {
                if last_result.is_some() {
                    break;
                }
                return Err(e);
            }
        }
    }

    // If a later pass failed after an earlier pass succeeded without CV,
    // re-run the last good pass with full CV options so `cv_rms_px` and
    // `significant` are never silently based on in-sample residuals.
    if !ran_final_cv && options.distortion.cv_folds >= 2 && last_result.is_some() {
        let obs_refs: Vec<DistortionObservation<'_>> = observations
            .iter()
            .enumerate()
            .map(|(e, o)| DistortionObservation {
                points: &last_points[e],
                poni: &last_input_ponis[e],
                calibrant: o.calibrant,
            })
            .collect();
        if let Ok(res_cv) = estimate_distortion_multi(&obs_refs, dim1, dim2, &options.distortion) {
            last_result = Some(res_cv);
        }
    }

    let distortion = last_result.ok_or_else(|| {
        DistortionError::InsufficientPoints("no distortion pass succeeded".into())
    })?;
    Ok(DistortionImageResult {
        distortion,
        points: last_points,
        n_rings: last_n_rings,
        initial_rms_px,
        n_passes_completed: passes_done,
    })
}

/// Estimate spatial detector distortion from calibration control points and a
/// converged [`Poni`].
pub fn estimate_distortion(
    points: &CalibrationPoints,
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    calibrant: &Calibrant,
    options: &DistortionOptions,
) -> Result<DistortionResult, DistortionError> {
    estimate_distortion_multi(
        &[DistortionObservation {
            points,
            poni,
            calibrant,
        }],
        dim1,
        dim2,
        options,
    )
}

/// Jointly estimate a single shared detector distortion field from multiple
/// calibration exposures (e.g. different detector positions, distances, tilts,
/// or wavelengths).
///
/// Each exposure retains its own [`Poni`] (re-refined independently when
/// `options.refine_poni` is `true`) while sharing one physical detector
/// distortion field `F(d1, d2)`. Multiple detector positions let rings cross
/// the same pixels at different angles, breaking tangential and rigid-PONI
/// degeneracies that are unobservable from a single exposure.
pub fn estimate_distortion_multi(
    observations: &[DistortionObservation<'_>],
    dim1: usize,
    dim2: usize,
    options: &DistortionOptions,
) -> Result<DistortionResult, DistortionError> {
    if dim1 == 0 || dim2 == 0 {
        return Err(DistortionError::InvalidInput(
            "detector dimensions must be positive".into(),
        ));
    }
    if observations.is_empty() {
        return Err(DistortionError::InsufficientPoints(
            "no calibration observations provided".into(),
        ));
    }
    if !options.huber_c_px.is_finite() || options.huber_c_px <= 0.0 {
        return Err(DistortionError::InvalidInput(
            "huber_c_px must be positive and finite".into(),
        ));
    }

    let mut cur_ponis: Vec<Poni> = Vec::with_capacity(observations.len());
    let mut active_per_obs: Vec<Vec<usize>> = Vec::with_capacity(observations.len());
    let mut total_active = 0usize;

    for (e, obs_in) in observations.iter().enumerate() {
        let p = obs_in.poni;
        if !p.distance.is_finite()
            || p.distance <= 0.0
            || !p.wavelength.is_finite()
            || p.wavelength <= 0.0
            || !p.pixel1.is_finite()
            || p.pixel1 <= 0.0
            || !p.pixel2.is_finite()
            || p.pixel2 <= 0.0
        {
            return Err(DistortionError::InvalidInput(format!(
                "observation {e} poni has non-finite or non-positive distance, wavelength, or pixel size"
            )));
        }
        cur_ponis.push(p.clone());
        let tth_rings = obs_in.calibrant.two_theta_rad(p.wavelength);
        let mut active = Vec::with_capacity(obs_in.points.len());
        for i in 0..obs_in.points.len() {
            if let Some(rk) = obs_in.points.ring[i]
                && let Some(Some(tgt)) = tth_rings.get(rk)
                && tgt.is_finite()
                && obs_in.points.d1[i].is_finite()
                && obs_in.points.d2[i].is_finite()
            {
                active.push(i);
            }
        }
        total_active += active.len();
        active_per_obs.push(active);
    }

    if total_active < 12 {
        return Err(DistortionError::InsufficientPoints(format!(
            "need at least 12 ring-assigned control points across observations, got {total_active}"
        )));
    }

    let (rms_before_rad, rms_before_px, point_residuals_before_px) =
        evaluate_residuals_multi(observations, &active_per_obs, &cur_ponis, None);

    let n_outer = if options.refine_poni { 3 } else { 1 };
    let mut field = fit_model_multi(
        observations,
        &active_per_obs,
        dim1,
        dim2,
        &cur_ponis,
        &options.model,
        options.huber_c_px,
    )?;

    for it in 0..n_outer {
        if it > 0 {
            field = fit_model_multi(
                observations,
                &active_per_obs,
                dim1,
                dim2,
                &cur_ponis,
                &options.model,
                options.huber_c_px,
            )?;
        }
        if options.refine_poni && it + 1 < n_outer {
            let mut mask = options.refine_mask;
            mask.rot3 = false;
            let cal_opts = CalibrationOptions {
                mask,
                loss: Loss::Huber { c: 0.002 },
                auto_assign_rings: false,
                assign_tolerance_rad: 0.05,
                max_iter: 40,
                tol: 1e-10,
            };
            for (e, obs_in) in observations.iter().enumerate() {
                if active_per_obs[e].len() < 6 {
                    continue;
                }
                let tc = TrigonometricConstants::new(&cur_ponis[e]);
                let mut shifted = CalibrationPoints::new();
                for &i in &active_per_obs[e] {
                    let d1 = obs_in.points.d1[i];
                    let d2 = obs_in.points.d2[i];
                    let (mut dd1, mut dd2) = field.displacement_px(d1, d2);
                    let dz = field.dz_m(d1, d2);
                    if dz.abs() > 0.0 {
                        let (g1, g2, gz) = tth_sensitivities(&cur_ponis[e], &tc, d1, d2);
                        let g2_norm = g1 * g1 + g2 * g2;
                        if g2_norm.is_finite() && g2_norm > 1e-24 {
                            // Equivalent radial shift for out-of-plane dz so BowZ
                            // and 3D PerModule also couple cleanly with PONI refine.
                            let scale = (gz * dz) / g2_norm;
                            dd1 += scale * g1;
                            dd2 += scale * g2;
                        }
                    }
                    shifted.push(d1 + dd1, d2 + dd2, obs_in.points.ring[i]);
                }
                let geom = Geometry::new(dim1, dim2, cur_ponis[e].clone());
                if let Ok(res) = refine(&geom, &shifted, obs_in.calibrant, &cal_opts)
                    && res.poni.distance.is_finite()
                    && res.poni.distance > 1e-3
                {
                    cur_ponis[e] = res.poni;
                }
            }
        }
    }

    let (rms_after_rad, rms_after_px, point_residuals_after_px) =
        evaluate_residuals_multi(observations, &active_per_obs, &cur_ponis, Some(&field));

    let mut max_displacement_px = 0.0_f64;
    for (e, obs_in) in observations.iter().enumerate() {
        for &i in &active_per_obs[e] {
            let (dd1, dd2) = field.displacement_px(obs_in.points.d1[i], obs_in.points.d2[i]);
            max_displacement_px = max_displacement_px.max(dd1.hypot(dd2));
        }
    }

    // K-fold cross-validation stratified across all observations, reusing
    // precomputed sensitivities and running folds in parallel via Rayon.
    let (cv_rms_px, cv_ran) = if options.cv_folds >= 2 && total_active >= options.cv_folds * 6 {
        let k_folds = options.cv_folds;
        let all_obs = build_observations_multi(observations, &active_per_obs, &cur_ponis);
        let (sq_sum, n_cv): (f64, usize) = (0..k_folds)
            .into_par_iter()
            .map(|fold| {
                let mut train_obs = Vec::with_capacity(all_obs.len());
                let mut test_per_obs: Vec<Vec<usize>> = vec![Vec::new(); observations.len()];
                let mut test_total = 0usize;
                for (pos, &o) in all_obs.iter().enumerate() {
                    if pos % k_folds == fold {
                        test_per_obs[o.obs_idx].push(o.pt_idx);
                        test_total += 1;
                    } else {
                        train_obs.push(o);
                    }
                }
                if test_total == 0 {
                    return (0.0, 0);
                }
                match fit_from_obs(
                    &train_obs,
                    dim1,
                    dim2,
                    &cur_ponis[0],
                    &options.model,
                    options.huber_c_px,
                ) {
                    Ok(fold_field) => {
                        let (_, fold_rms_px, _) = evaluate_residuals_multi(
                            observations,
                            &test_per_obs,
                            &cur_ponis,
                            Some(&fold_field),
                        );
                        (fold_rms_px * fold_rms_px * (test_total as f64), test_total)
                    }
                    Err(_) => (0.0, 0),
                }
            })
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        if n_cv > 0 {
            ((sq_sum / n_cv as f64).sqrt(), true)
        } else {
            (rms_after_px, false)
        }
    } else {
        (rms_after_px, false)
    };

    // Only mark significant when out-of-sample CV actually ran and improved RMS.
    let significant = cv_ran && cv_rms_px < 0.95 * rms_before_px && rms_after_px < rms_before_px;
    let module_offsets = match &field {
        DistortionField::PerModule { offsets, .. } => offsets.clone(),
        _ => Vec::new(),
    };
    let primary_poni = cur_ponis[0].clone();

    Ok(DistortionResult {
        poni: primary_poni,
        ponis: cur_ponis,
        model: options.model.clone(),
        field,
        rms_before_rad,
        rms_after_rad,
        rms_before_px,
        rms_after_px,
        cv_rms_px,
        significant,
        max_displacement_px,
        n_points_used: total_active,
        point_residuals_before_px,
        point_residuals_after_px,
        module_offsets,
    })
}

// ---------- Geometry & residual helpers ----------

#[inline]
fn eval_tth_at(poni: &Poni, tc: &TrigonometricConstants, d1: f64, d2: f64, dz: f64) -> f64 {
    let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
    let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
    let p3 = poni.distance + dz;
    let x1 = f_x1(&p1, &p2, &p3, tc);
    let x2 = f_x2(&p1, &p2, &p3, tc);
    let x3 = f_x3(&p1, &p2, &p3, tc);
    f_tth(&x1, &x2, &x3)
}

#[inline]
fn tth_sensitivities(
    poni: &Poni,
    tc: &TrigonometricConstants,
    d1: f64,
    d2: f64,
) -> (f64, f64, f64) {
    let hd = 1e-4;
    let hz = 1e-6;
    let g1 = (eval_tth_at(poni, tc, d1 + hd, d2, 0.0) - eval_tth_at(poni, tc, d1 - hd, d2, 0.0))
        / (2.0 * hd);
    let g2 = (eval_tth_at(poni, tc, d1, d2 + hd, 0.0) - eval_tth_at(poni, tc, d1, d2 - hd, 0.0))
        / (2.0 * hd);
    let gz = (eval_tth_at(poni, tc, d1, d2, hz) - eval_tth_at(poni, tc, d1, d2, -hz)) / (2.0 * hz);
    (g1, g2, gz)
}

fn evaluate_residuals_multi(
    observations: &[DistortionObservation<'_>],
    active_per_obs: &[Vec<usize>],
    ponis: &[Poni],
    field: Option<&DistortionField>,
) -> (f64, f64, Vec<f64>) {
    let total_len: usize = observations.iter().map(|o| o.points.len()).sum();
    let mut per_point_px = Vec::with_capacity(total_len);
    let mut sum_rad2 = 0.0;
    let mut sum_px2 = 0.0;
    let mut n_tot = 0usize;

    for (e, obs_in) in observations.iter().enumerate() {
        let poni = &ponis[e];
        let tc = TrigonometricConstants::new(poni);
        let tth_rings = obs_in.calibrant.two_theta_rad(poni.wavelength);
        let base_off = per_point_px.len();
        per_point_px.resize(base_off + obs_in.points.len(), 0.0);
        for &i in &active_per_obs[e] {
            let d1 = obs_in.points.d1[i];
            let d2 = obs_in.points.d2[i];
            let Some(Some(tgt)) = obs_in.points.ring[i].and_then(|rk| tth_rings.get(rk)) else {
                continue;
            };
            let (dd1, dd2, dz) = match field {
                Some(f) => {
                    let (a, b) = f.displacement_px(d1, d2);
                    (a, b, f.dz_m(d1, d2))
                }
                None => (0.0, 0.0, 0.0),
            };
            let pred = eval_tth_at(poni, &tc, d1 + dd1, d2 + dd2, dz);
            let dr_rad = tgt - pred;
            let (g1, g2, _) = tth_sensitivities(poni, &tc, d1, d2);
            let g_norm = g1.hypot(g2).max(1e-12);
            let dr_px = dr_rad / g_norm;
            per_point_px[base_off + i] = dr_px;
            sum_rad2 += dr_rad * dr_rad;
            sum_px2 += dr_px * dr_px;
            n_tot += 1;
        }
    }
    let n = n_tot.max(1) as f64;
    ((sum_rad2 / n).sqrt(), (sum_px2 / n).sqrt(), per_point_px)
}

#[derive(Clone, Copy)]
struct ObsPoint {
    obs_idx: usize,
    pt_idx: usize,
    d1: f64,
    d2: f64,
    /// Unit radial sensitivity `(g1/|g|, g2/|g|)` in pixels⁻¹.
    u1: f64,
    u2: f64,
    /// Out-of-plane sensitivity `gz / |g|` (pixels per metre of `δz`).
    uz: f64,
    /// Target radial shift `(2θ_target − 2θ_0) / |g|` in pixels.
    rhs_px: f64,
}

fn build_observations_multi(
    observations: &[DistortionObservation<'_>],
    active_per_obs: &[Vec<usize>],
    ponis: &[Poni],
) -> Vec<ObsPoint> {
    let cap: usize = active_per_obs.iter().map(Vec::len).sum();
    let mut obs = Vec::with_capacity(cap);
    for (e, obs_in) in observations.iter().enumerate() {
        let poni = &ponis[e];
        let tc = TrigonometricConstants::new(poni);
        let tth_rings = obs_in.calibrant.two_theta_rad(poni.wavelength);
        for &i in &active_per_obs[e] {
            let d1 = obs_in.points.d1[i];
            let d2 = obs_in.points.d2[i];
            let Some(Some(tgt)) = obs_in.points.ring[i].and_then(|rk| tth_rings.get(rk)) else {
                continue;
            };
            let pred = eval_tth_at(poni, &tc, d1, d2, 0.0);
            let (g1, g2, gz) = tth_sensitivities(poni, &tc, d1, d2);
            let g_norm = g1.hypot(g2);
            if !g_norm.is_finite() || !gz.is_finite() || !pred.is_finite() || g_norm < 1e-12 {
                continue;
            }
            obs.push(ObsPoint {
                obs_idx: e,
                pt_idx: i,
                d1,
                d2,
                u1: g1 / g_norm,
                u2: g2 / g_norm,
                uz: gz / g_norm,
                rhs_px: (tgt - pred) / g_norm,
            });
        }
    }
    obs
}

// ---------- Model dispatch ----------

fn fit_model_multi(
    observations: &[DistortionObservation<'_>],
    active_per_obs: &[Vec<usize>],
    dim1: usize,
    dim2: usize,
    ponis: &[Poni],
    model: &DistortionModel,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    let obs = build_observations_multi(observations, active_per_obs, ponis);
    fit_from_obs(&obs, dim1, dim2, &ponis[0], model, huber_c_px)
}

fn fit_from_obs(
    obs: &[ObsPoint],
    dim1: usize,
    dim2: usize,
    ref_poni: &Poni,
    model: &DistortionModel,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    if obs.len() < 12 {
        return Err(DistortionError::InsufficientPoints(
            "too few valid observations".into(),
        ));
    }
    match model {
        DistortionModel::RadialOptical { max_order } => {
            fit_radial_optical(obs, dim1, dim2, ref_poni, *max_order, huber_c_px)
        }
        DistortionModel::SmoothSpline {
            grid_nx,
            grid_ny,
            smoothness,
        } => fit_smooth_spline(obs, dim1, dim2, *grid_nx, *grid_ny, *smoothness, huber_c_px),
        DistortionModel::BowZ {
            max_degree,
            smoothness,
        } => fit_bow_z(obs, dim1, dim2, *max_degree, *smoothness, huber_c_px),
        DistortionModel::PerModule {
            module_height,
            module_width,
            gap_height,
            gap_width,
            fit_z,
            regularization,
        } => fit_per_module(
            obs,
            dim1,
            dim2,
            *module_height,
            *module_width,
            *gap_height,
            *gap_width,
            *fit_z,
            *regularization,
            huber_c_px,
        ),
    }
}

// ---------- Model 1: RadialOptical ----------

fn radial_optical_basis_radial(rho: f64, phi: f64, max_order: usize) -> Vec<f64> {
    let order = max_order.clamp(3, 7);
    let mut b = Vec::with_capacity(12);
    // Purely radial pincushion/barrel terms: ρ³, ρ⁵, (ρ⁷)
    b.push(rho.powi(3));
    if order >= 5 {
        b.push(rho.powi(5));
    }
    if order >= 7 {
        b.push(rho.powi(7));
    }
    // Anisotropic / astigmatic / trefoil / quadrafoil terms orthogonal to
    // rigid PONI modes (1, cos φ, sin φ, ρ, ρ² cos φ, ρ² sin φ):
    let r2 = rho * rho;
    let r3 = r2 * rho;
    b.push(r2 * (2.0 * phi).cos());
    b.push(r2 * (2.0 * phi).sin());
    b.push(r3 * (2.0 * phi).cos());
    b.push(r3 * (2.0 * phi).sin());
    b.push(r3 * (3.0 * phi).cos());
    b.push(r3 * (3.0 * phi).sin());
    if order >= 5 {
        let r4 = r2 * r2;
        b.push(r4 * (2.0 * phi).cos());
        b.push(r4 * (2.0 * phi).sin());
        b.push(r4 * (4.0 * phi).cos());
        b.push(r4 * (4.0 * phi).sin());
    }
    b
}

fn eval_radial_optical(
    center_d1: f64,
    center_d2: f64,
    r_scale: f64,
    max_order: usize,
    coeffs: &[f64],
    d1: f64,
    d2: f64,
) -> (f64, f64) {
    let dy = d1 - center_d1;
    let dx = d2 - center_d2;
    let r = dy.hypot(dx);
    if r < 1e-9 {
        return (0.0, 0.0);
    }
    let rho = r / r_scale;
    let phi = dy.atan2(dx);
    let basis = radial_optical_basis_radial(rho, phi, max_order);
    let dr: f64 = basis.iter().zip(coeffs.iter()).map(|(b, c)| b * c).sum();
    (dr * (dy / r), dr * (dx / r))
}

fn fit_radial_optical(
    obs: &[ObsPoint],
    dim1: usize,
    dim2: usize,
    poni: &Poni,
    max_order: usize,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    let order = max_order.clamp(3, 7);
    let center_d1 = poni.poni1 / poni.pixel1 - 0.5;
    let center_d2 = poni.poni2 / poni.pixel2 - 0.5;
    let r_scale = (0.5 * (dim1 as f64).hypot(dim2 as f64)).max(1.0);
    let n_params = radial_optical_basis_radial(0.5, 0.0, order).len();

    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(obs.len());
    let mut rhs: Vec<f64> = Vec::with_capacity(obs.len());
    for o in obs {
        let dy = o.d1 - center_d1;
        let dx = o.d2 - center_d2;
        let r = dy.hypot(dx).max(1e-6);
        let rho = r / r_scale;
        let phi = dy.atan2(dx);
        let radial_dir_proj = o.u1 * (dy / r) + o.u2 * (dx / r);
        let mut row = radial_optical_basis_radial(rho, phi, order);
        for v in &mut row {
            *v *= radial_dir_proj;
        }
        rows.push(row);
        rhs.push(o.rhs_px);
    }

    let reg = vec![1e-4; n_params];
    let coeffs = solve_huber_irls_dense(&rows, &rhs, &reg, None, huber_c_px)?;
    Ok(DistortionField::RadialOptical {
        center_d1,
        center_d2,
        r_scale,
        max_order: order,
        coeffs,
    })
}

// ---------- Model 2: SmoothSpline ----------

fn clamped_uniform_knots(lo: f64, hi: f64, n_ctrl: usize) -> Vec<f64> {
    let n = n_ctrl.max(4);
    let mut t = Vec::with_capacity(n + 4);
    for _ in 0..4 {
        t.push(lo);
    }
    let n_int = n - 4;
    for k in 1..=n_int {
        t.push(lo + (hi - lo) * (k as f64) / ((n_int + 1) as f64));
    }
    for _ in 0..4 {
        t.push(hi);
    }
    t
}

/// Build the thin-plate bending energy penalty matrix `P` (second differences
/// `D_xx`, `D_xy`, `D_yy`) for one `nx × ny` B-spline surface.
fn add_bending_penalty(
    h: &mut [f64],
    total_n: usize,
    offset: usize,
    nx: usize,
    ny: usize,
    lambda: f64,
) {
    let mut add_stencil = |terms: &[(usize, f64)], weight: f64| {
        for &(i, ci) in terms {
            for &(j, cj) in terms {
                h[(offset + i) * total_n + (offset + j)] += weight * ci * cj;
            }
        }
    };
    // D_xx along fast/x axis
    for ix in 1..nx.saturating_sub(1) {
        for iy in 0..ny {
            let idx = |a: usize, b: usize| a * ny + b;
            add_stencil(
                &[
                    (idx(ix - 1, iy), 1.0),
                    (idx(ix, iy), -2.0),
                    (idx(ix + 1, iy), 1.0),
                ],
                lambda,
            );
        }
    }
    // D_yy along slow/y axis
    for ix in 0..nx {
        for iy in 1..ny.saturating_sub(1) {
            let idx = |a: usize, b: usize| a * ny + b;
            add_stencil(
                &[
                    (idx(ix, iy - 1), 1.0),
                    (idx(ix, iy), -2.0),
                    (idx(ix, iy + 1), 1.0),
                ],
                lambda,
            );
        }
    }
    // 2 * D_xy mixed twist
    for ix in 0..nx.saturating_sub(1) {
        for iy in 0..ny.saturating_sub(1) {
            let idx = |a: usize, b: usize| a * ny + b;
            add_stencil(
                &[
                    (idx(ix, iy), 1.0),
                    (idx(ix + 1, iy), -1.0),
                    (idx(ix, iy + 1), -1.0),
                    (idx(ix + 1, iy + 1), 1.0),
                ],
                2.0 * lambda,
            );
        }
    }
    // Mild zero-order ridge suppresses the unobservable tangential nullspace
    // and anchors unobserved corners smoothly near zero.
    let ridge = (1e-3 * lambda).max(1e-6);
    for k in 0..(nx * ny) {
        h[(offset + k) * total_n + (offset + k)] += ridge;
    }
}

/// Add a weak gauge-fixing penalty on the 3 global rigid-body modes of the
/// `SmoothSpline` control-point grid (mean `d1` translation, mean `d2`
/// translation, and global rigid rotation) plus a mild discrete-curl floor
/// `∂(δd₁)/∂d₂ − ∂(δd₂)/∂d₁`.
///
/// On a single centered exposure, observations only constrain the radial
/// component `δ · r̂`: a global rigid rotation `(δd₁, δd₂) = α(x − x̄, −(y − ȳ))`
/// has both zero radial component and zero second-difference bending energy, so
/// without gauge fixing it can drift freely into a multi-pixel swirl. Purely
/// radial optical fields have identically zero curl and zero projection onto
/// the rotation mode, while the weights are kept mild enough that multi-exposure
/// fits with shifted beam centers still recover real local twists and shears
/// accurately.
#[allow(clippy::too_many_arguments)]
fn add_rigid_gauge_and_curl_penalty(
    h: &mut [f64],
    total_n: usize,
    nx: usize,
    ny: usize,
    tx: &[f64],
    ty: &[f64],
    dim1: usize,
    dim2: usize,
    lambda: f64,
) {
    let n_surf = nx * ny;
    let idx = |ix: usize, iy: usize| ix * ny + iy;

    // Centered Greville abscissae for the clamped cubic B-spline control grid,
    // scaled isotropically so aspect ratio is preserved for rigid rotation.
    let gx: Vec<f64> = (0..nx)
        .map(|ix| (tx[ix + 1] + tx[ix + 2] + tx[ix + 3]) / 3.0)
        .collect();
    let gy: Vec<f64> = (0..ny)
        .map(|iy| (ty[iy + 1] + ty[iy + 2] + ty[iy + 3]) / 3.0)
        .collect();
    let x_mean = gx.iter().sum::<f64>() / (nx as f64);
    let y_mean = gy.iter().sum::<f64>() / (ny as f64);
    let scale = (0.5 * (dim1.max(dim2) as f64)).max(1.0);
    let u: Vec<f64> = gx.iter().map(|&x| (x - x_mean) / scale).collect();
    let v: Vec<f64> = gy.iter().map(|&y| (y - y_mean) / scale).collect();

    // Rank-1 helper: H += weight * (vec ⊗ vec) where `vec` is unit-normalized.
    let mut add_unit_mode = |mode: &[f64], weight: f64| {
        let norm2: f64 = mode.iter().map(|&x| x * x).sum();
        if norm2 <= 1e-24 || weight <= 0.0 {
            return;
        }
        let scale_w = weight / norm2;
        for i in 0..total_n {
            let mi = mode[i];
            if mi == 0.0 {
                continue;
            }
            let row = i * total_n;
            let wmi = scale_w * mi;
            for j in 0..total_n {
                let mj = mode[j];
                if mj != 0.0 {
                    h[row + j] += wmi * mj;
                }
            }
        }
    };

    // 1 & 2. Weak mean-translation gauge: ∑ c_d1 = 0, ∑ c_d2 = 0.
    let mut mode_ty = vec![0.0_f64; total_n];
    let mut mode_tx = vec![0.0_f64; total_n];
    // 3. Global rigid-rotation gauge: ∑ (u_i · c_d1 − v_j · c_d2) = 0.
    let mut mode_rot = vec![0.0_f64; total_n];
    for (ix, &ui) in u.iter().enumerate() {
        for (iy, &vj) in v.iter().enumerate() {
            let k = idx(ix, iy);
            mode_ty[k] = 1.0;
            mode_tx[n_surf + k] = 1.0;
            mode_rot[k] = ui;
            mode_rot[n_surf + k] = -vj;
        }
    }
    let w_trans = (0.5 * lambda).max(1e-3);
    let w_rot = (10.0 * lambda).max(1e-1);
    add_unit_mode(&mode_ty, w_trans);
    add_unit_mode(&mode_tx, w_trans);
    add_unit_mode(&mode_rot, w_rot);

    // 4. Mild discrete curl floor on each grid cell:
    //    ω = ∂(δd₁)/∂d₂ − ∂(δd₂)/∂d₁ ≈ 0.
    //    Identically zero for all purely radial distortion fields, suppresses
    //    higher-order tangential nullspace swirl on single-exposure grids.
    let dx = (dim2 as f64) / ((nx - 1).max(1) as f64);
    let dy = (dim1 as f64) / ((ny - 1).max(1) as f64);
    let h_avg = 0.5 * (dx + dy);
    let ax = 0.5 * h_avg / dx.max(1e-9);
    let ay = 0.5 * h_avg / dy.max(1e-9);
    let w_curl = (0.01 * lambda).max(1e-5);

    for ix in 0..nx.saturating_sub(1) {
        for iy in 0..ny.saturating_sub(1) {
            let k00 = idx(ix, iy);
            let k10 = idx(ix + 1, iy);
            let k01 = idx(ix, iy + 1);
            let k11 = idx(ix + 1, iy + 1);
            let terms = [
                // +∂(δd₁)/∂d₂
                (k00, -ax),
                (k01, -ax),
                (k10, ax),
                (k11, ax),
                // −∂(δd₂)/∂d₁
                (n_surf + k00, ay),
                (n_surf + k10, ay),
                (n_surf + k01, -ay),
                (n_surf + k11, -ay),
            ];
            for &(i, ci) in &terms {
                let row = i * total_n;
                let wci = w_curl * ci;
                for &(j, cj) in &terms {
                    h[row + j] += wci * cj;
                }
            }
        }
    }
}

fn fit_smooth_spline(
    obs: &[ObsPoint],
    dim1: usize,
    dim2: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: f64,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    if !smoothness.is_finite() || smoothness <= 0.0 {
        return Err(DistortionError::InvalidInput(
            "smoothness must be positive and finite".into(),
        ));
    }
    let nx = grid_nx.clamp(4, 20);
    let ny = grid_ny.clamp(4, 20);
    let tx = clamped_uniform_knots(0.0, dim2 as f64, nx);
    let ty = clamped_uniform_knots(0.0, dim1 as f64, ny);
    let n_surf = nx * ny;
    let total_n = 2 * n_surf;

    // Precompute sparse 16-tap B-spline basis per observation for fast IRLS.
    let mut sparse_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(obs.len());
    let mut rhs: Vec<f64> = Vec::with_capacity(obs.len());
    for o in obs {
        let (ix, bx) = bspline_basis(&tx, o.d2);
        let (iy, by) = bspline_basis(&ty, o.d1);
        let mut entries = Vec::with_capacity(32);
        for (a, &wx) in bx.iter().enumerate() {
            for (b, &wy) in by.iter().enumerate() {
                let w = wx * wy;
                if w.abs() > 1e-15 {
                    let k = (ix + a) * ny + (iy + b);
                    entries.push((k, o.u1 * w)); // δd₁ surface
                    entries.push((n_surf + k, o.u2 * w)); // δd₂ surface
                }
            }
        }
        sparse_rows.push(entries);
        rhs.push(o.rhs_px);
    }

    // Scale bending penalty proportionally to observation density per control cell
    // so `smoothness = 1.0` gives gentle regularization across both sparse and
    // dense ring sets without over-smoothing real optical warps.
    let lambda = 0.05 * smoothness * (obs.len() as f64 / n_surf as f64).max(1.0);
    let mut penalty = vec![0.0_f64; total_n * total_n];
    add_bending_penalty(&mut penalty, total_n, 0, nx, ny, lambda);
    add_bending_penalty(&mut penalty, total_n, n_surf, nx, ny, lambda);
    add_rigid_gauge_and_curl_penalty(&mut penalty, total_n, nx, ny, &tx, &ty, dim1, dim2, lambda);

    let coeffs = solve_huber_irls_sparse(&sparse_rows, &rhs, total_n, &penalty, huber_c_px)?;

    Ok(DistortionField::SmoothSpline {
        surface_d1: Surface {
            x: tx.clone(),
            y: ty.clone(),
            coefficients: coeffs[..n_surf].to_vec(),
        },
        surface_d2: Surface {
            x: tx,
            y: ty,
            coefficients: coeffs[n_surf..].to_vec(),
        },
    })
}

fn fit_bspline_surface_ls(
    tx: &[f64],
    ty: &[f64],
    xs: &[f64],
    ys: &[f64],
    vals: &[f64],
    ridge: f64,
) -> Result<Vec<f64>, String> {
    let nx = tx.len() - 4;
    let ny = ty.len() - 4;
    let n = nx * ny;
    let mut h = vec![0.0_f64; n * n];
    let mut g = vec![0.0_f64; n];
    for ((&x, &y), &val) in xs.iter().zip(ys.iter()).zip(vals.iter()) {
        let (ix, bx) = bspline_basis(tx, x);
        let (iy, by) = bspline_basis(ty, y);
        let mut idx_w = [(0usize, 0.0_f64); 16];
        let mut cnt = 0usize;
        for (a, &wx) in bx.iter().enumerate() {
            for (b, &wy) in by.iter().enumerate() {
                let w = wx * wy;
                if w.abs() > 1e-15 {
                    idx_w[cnt] = ((ix + a) * ny + (iy + b), w);
                    cnt += 1;
                }
            }
        }
        for &(i, wi) in &idx_w[..cnt] {
            g[i] += wi * val;
            for &(j, wj) in &idx_w[..cnt] {
                h[i * n + j] += wi * wj;
            }
        }
    }
    for i in 0..n {
        h[i * n + i] += ridge;
    }
    solve_spd(&mut h, &mut g, n).map_err(|e| e.to_string())?;
    Ok(g)
}

// ---------- Model 3: BowZ ----------

fn fit_bow_z(
    obs: &[ObsPoint],
    dim1: usize,
    dim2: usize,
    max_degree: usize,
    smoothness: f64,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    let deg_max = max_degree.clamp(2, 5);
    let center_d1 = 0.5 * dim1 as f64;
    let center_d2 = 0.5 * dim2 as f64;
    let scale_d1 = (0.5 * dim1 as f64).max(1.0);
    let scale_d2 = (0.5 * dim2 as f64).max(1.0);
    let mut n_terms = 0usize;
    for d in 2..=deg_max {
        n_terms += d + 1;
    }

    // Scale δz in millimetres inside the linear system for good conditioning.
    const Z_SCALE: f64 = 1e-3;
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(obs.len());
    let mut rhs: Vec<f64> = Vec::with_capacity(obs.len());
    for o in obs {
        let x = (o.d2 - center_d2) / scale_d2;
        let y = (o.d1 - center_d1) / scale_d1;
        let mut row = Vec::with_capacity(n_terms);
        for d in 2..=deg_max {
            for ix in 0..=d {
                let iy = d - ix;
                row.push(o.uz * Z_SCALE * x.powi(ix as i32) * y.powi(iy as i32));
            }
        }
        rows.push(row);
        rhs.push(o.rhs_px);
    }

    let reg = vec![smoothness.max(1e-8); n_terms];
    let mut coeffs = solve_huber_irls_dense(&rows, &rhs, &reg, None, huber_c_px)?;
    for c in &mut coeffs {
        *c *= Z_SCALE;
    }
    Ok(DistortionField::BowZ {
        center_d1,
        center_d2,
        scale_d1,
        scale_d2,
        max_degree: deg_max,
        coeffs,
    })
}

// ---------- Model 4: PerModule ----------

#[allow(clippy::too_many_arguments)]
fn module_index(
    d1: f64,
    d2: f64,
    module_height: usize,
    module_width: usize,
    gap_height: usize,
    gap_width: usize,
    n_mod_rows: usize,
    n_mod_cols: usize,
) -> Option<usize> {
    if !d1.is_finite()
        || !d2.is_finite()
        || d1 < 0.0
        || d2 < 0.0
        || module_height == 0
        || module_width == 0
    {
        return None;
    }
    let pitch_r = module_height + gap_height;
    let pitch_c = module_width + gap_width;
    let r = d1.floor() as usize;
    let c = d2.floor() as usize;
    let mr = r / pitch_r;
    let mc = c / pitch_c;
    if mr >= n_mod_rows || mc >= n_mod_cols {
        return None;
    }
    if (r % pitch_r) >= module_height || (c % pitch_c) >= module_width {
        return None;
    }
    Some(mr * n_mod_cols + mc)
}

#[allow(clippy::too_many_arguments)]
fn fit_per_module(
    obs: &[ObsPoint],
    dim1: usize,
    dim2: usize,
    module_height: usize,
    module_width: usize,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
    huber_c_px: f64,
) -> Result<DistortionField, DistortionError> {
    if module_height == 0 || module_width == 0 {
        return Err(DistortionError::InvalidInput(
            "module_height and module_width must be positive".into(),
        ));
    }
    let pitch_r = module_height + gap_height;
    let pitch_c = module_width + gap_width;
    let n_mod_rows = (dim1 + gap_height).div_ceil(pitch_r).max(1);
    let n_mod_cols = (dim2 + gap_width).div_ceil(pitch_c).max(1);
    let n_mods = n_mod_rows * n_mod_cols;
    let stride = if fit_z { 3 } else { 2 };
    let total_n = n_mods * stride;
    const Z_SCALE: f64 = 1e-3;

    let mut counts = vec![0usize; n_mods];
    let mut sparse_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(obs.len());
    let mut rhs: Vec<f64> = Vec::with_capacity(obs.len());
    for o in obs {
        if let Some(m) = module_index(
            o.d1,
            o.d2,
            module_height,
            module_width,
            gap_height,
            gap_width,
            n_mod_rows,
            n_mod_cols,
        ) {
            counts[m] += 1;
            let base = m * stride;
            let mut entries = Vec::with_capacity(stride);
            entries.push((base, o.u1));
            entries.push((base + 1, o.u2));
            if fit_z {
                entries.push((base + 2, o.uz * Z_SCALE));
            }
            sparse_rows.push(entries);
            rhs.push(o.rhs_px);
        }
    }
    if sparse_rows.len() < 12 {
        return Err(DistortionError::InsufficientPoints(
            "too few control points fell inside active modules".into(),
        ));
    }

    // Diagonal ridge + weak zero-mean gauge penalty restricted to populated
    // modules (so empty corner modules stay at exactly 0 instead of being
    // pushed to −Σ_others).
    let reg = regularization.max(1e-6);
    let mut penalty = vec![0.0_f64; total_n * total_n];
    for i in 0..total_n {
        penalty[i * total_n + i] += reg;
    }
    let populated: Vec<usize> = (0..n_mods).filter(|&m| counts[m] > 0).collect();
    if populated.len() > 1 {
        let gauge_w = 1.0 / (populated.len() as f64);
        for comp in 0..stride {
            for &m1 in &populated {
                for &m2 in &populated {
                    penalty[(m1 * stride + comp) * total_n + (m2 * stride + comp)] += gauge_w;
                }
            }
        }
    }

    let coeffs = solve_huber_irls_sparse(&sparse_rows, &rhs, total_n, &penalty, huber_c_px)?;
    let mut offsets = Vec::with_capacity(n_mods);
    for mr in 0..n_mod_rows {
        for mc in 0..n_mod_cols {
            let m = mr * n_mod_cols + mc;
            let base = m * stride;
            let has_pts = counts[m] > 0;
            offsets.push(ModuleOffset {
                row_index: mr,
                col_index: mc,
                dd1_px: if has_pts { coeffs[base] } else { 0.0 },
                dd2_px: if has_pts { coeffs[base + 1] } else { 0.0 },
                dz_m: if has_pts && fit_z {
                    coeffs[base + 2] * Z_SCALE
                } else {
                    0.0
                },
                n_points: counts[m],
            });
        }
    }

    Ok(DistortionField::PerModule {
        module_height,
        module_width,
        gap_height,
        gap_width,
        n_mod_rows,
        n_mod_cols,
        offsets,
    })
}

// ---------- Huber IRLS & Cholesky SPD linear solvers ----------

fn solve_huber_irls_dense(
    rows: &[Vec<f64>],
    rhs: &[f64],
    diag_reg: &[f64],
    extra_penalty: Option<&[f64]>,
    huber_c: f64,
) -> Result<Vec<f64>, DistortionError> {
    let n = diag_reg.len();
    let mut x = vec![0.0_f64; n];
    let mut h = vec![0.0_f64; n * n];
    let mut g = vec![0.0_f64; n];

    for iter in 0..10 {
        h.fill(0.0);
        g.fill(0.0);
        if let Some(p) = extra_penalty {
            h.copy_from_slice(p);
        }
        for i in 0..n {
            h[i * n + i] += diag_reg[i];
        }
        for (row, &y) in rows.iter().zip(rhs.iter()) {
            let w = if iter == 0 {
                1.0
            } else {
                let pred: f64 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                let r = (y - pred).abs();
                if r <= huber_c { 1.0 } else { huber_c / r }
            };
            for i in 0..n {
                let wi = w * row[i];
                g[i] += wi * y;
                let row_off = i * n;
                for j in 0..=i {
                    h[row_off + j] += wi * row[j];
                }
            }
        }
        solve_spd(&mut h, &mut g, n)?;
        let step = g
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        x.copy_from_slice(&g);
        if iter >= 2 && step < 1e-8 {
            break;
        }
    }
    Ok(x)
}

fn solve_huber_irls_sparse(
    rows: &[Vec<(usize, f64)>],
    rhs: &[f64],
    n: usize,
    penalty: &[f64],
    huber_c: f64,
) -> Result<Vec<f64>, DistortionError> {
    let mut x = vec![0.0_f64; n];
    let mut h = vec![0.0_f64; n * n];
    let mut g = vec![0.0_f64; n];

    for iter in 0..10 {
        h.copy_from_slice(penalty);
        g.fill(0.0);
        for (row, &y) in rows.iter().zip(rhs.iter()) {
            let w = if iter == 0 {
                1.0
            } else {
                let pred: f64 = row.iter().map(|&(idx, a)| a * x[idx]).sum();
                let r = (y - pred).abs();
                if r <= huber_c { 1.0 } else { huber_c / r }
            };
            for &(i, ai) in row {
                let wi = w * ai;
                g[i] += wi * y;
                let row_off = i * n;
                for &(j, aj) in row {
                    if j <= i {
                        h[row_off + j] += wi * aj;
                    }
                }
            }
        }
        solve_spd(&mut h, &mut g, n)?;
        let step = g
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        x.copy_from_slice(&g);
        if iter >= 2 && step < 1e-7 {
            break;
        }
    }
    Ok(x)
}

/// In-place Cholesky decomposition `A = L Lᵀ` and solve `A x = b` (overwriting `b` with `x`).
fn solve_spd(a: &mut [f64], b: &mut [f64], n: usize) -> Result<(), DistortionError> {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-18 {
                    return Err(DistortionError::Singular(format!(
                        "normal matrix not positive definite at pivot {i} ({sum:.3e})"
                    )));
                }
                a[i * n + j] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }
    // Forward substitution: L y = b
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    // Back substitution: Lᵀ x = y
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ring_points_with_distortion(
        poni: &Poni,
        cal: &Calibrant,
        dim1: usize,
        dim2: usize,
        distort: impl Fn(f64, f64) -> (f64, f64, f64), // (true_dd1, true_dd2, true_dz)
    ) -> CalibrationPoints {
        let tc = TrigonometricConstants::new(poni);
        let tths = cal.two_theta_rad(poni.wavelength);
        let mut pts = CalibrationPoints::new();
        let c1 = poni.poni1 / poni.pixel1 - 0.5;
        let c2 = poni.poni2 / poni.pixel2 - 0.5;
        for (rk, opt_t) in tths.iter().enumerate().take(16) {
            let Some(tth) = opt_t else { continue };
            if !tth.is_finite() || *tth < 0.03 || *tth > 0.7 {
                continue;
            }
            let r_approx = (poni.distance / poni.pixel1) * tth.tan();
            for a in 0..72 {
                let phi = 2.0 * std::f64::consts::PI * (a as f64) / 72.0;
                // Find pixel (d1, d2) whose distorted physical position lands on tth
                let mut r = r_approx;
                let mut ok = false;
                for _ in 0..12 {
                    let d1 = c1 + r * phi.sin();
                    let d2 = c2 + r * phi.cos();
                    let (dd1, dd2, dz) = distort(d1, d2);
                    let f = eval_tth_at(poni, &tc, d1 + dd1, d2 + dd2, dz) - *tth;
                    if f.abs() < 1e-10 {
                        ok = true;
                        break;
                    }
                    let eps = 1e-3;
                    let d1p = c1 + (r + eps) * phi.sin();
                    let d2p = c2 + (r + eps) * phi.cos();
                    let (dd1p, dd2p, dzp) = distort(d1p, d2p);
                    let fp = eval_tth_at(poni, &tc, d1p + dd1p, d2p + dd2p, dzp) - *tth;
                    let df = (fp - f) / eps;
                    if df.abs() < 1e-12 {
                        break;
                    }
                    r -= f / df;
                }
                if !ok {
                    continue;
                }
                let d1 = c1 + r * phi.sin();
                let d2 = c2 + r * phi.cos();
                if d1 > 8.0 && d2 > 8.0 && d1 < (dim1 - 8) as f64 && d2 < (dim2 - 8) as f64 {
                    pts.push(d1, d2, Some(rk));
                }
            }
        }
        pts
    }

    fn make_test_poni() -> Poni {
        Poni {
            pixel1: 172e-6,
            pixel2: 172e-6,
            distance: 0.20,
            poni1: 512.0 * 172e-6,
            poni2: 512.0 * 172e-6,
            rot1: 0.04,
            rot2: -0.03,
            rot3: 0.0,
            wavelength: 0.4066e-10,
            ..Poni::new()
        }
    }

    #[test]
    fn radial_optical_recovers_pincushion_and_exports_valid_spline() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 1024usize;
        let dim2 = 1024usize;
        let c1 = 511.5;
        let c2 = 511.5;
        let r_scale = 0.5 * (1024.0_f64).hypot(1024.0);

        // Synthetic pincushion + astigmatism distortion (~4 px at edge)
        let distort = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let dy = d1 - c1;
            let dx = d2 - c2;
            let r = dy.hypot(dx).max(1e-6);
            let rho = r / r_scale;
            let phi = dy.atan2(dx);
            let dr = 4.5 * rho.powi(3) - 1.2 * rho.powi(2) * (2.0 * phi).cos();
            (dr * dy / r, dr * dx / r, 0.0)
        };
        let pts = sample_ring_points_with_distortion(&poni, &cal, dim1, dim2, distort);

        let res = estimate_distortion(
            &pts,
            dim1,
            dim2,
            &poni,
            &cal,
            &DistortionOptions {
                model: DistortionModel::RadialOptical { max_order: 5 },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("RadialOptical fit");

        assert!(res.significant, "expected significant distortion: {res:?}");
        assert!(res.rms_before_px > 0.5, "before RMS={}", res.rms_before_px);
        assert!(
            res.rms_after_px < 0.03,
            "after RMS too large: {:.4} px",
            res.rms_after_px
        );
        assert!(
            res.cv_rms_px < 0.05,
            "CV RMS too large: {:.4} px",
            res.cv_rms_px
        );

        // Export to Fit2D Spline and round-trip parse it!
        let sp = res
            .field
            .to_spline(dim1, dim2, poni.pixel1, poni.pixel2)
            .expect("to_spline");
        let reparsed = Spline::parse(&sp.content).expect("Spline::parse round-trip");
        // Detector::position evaluates the spline at pixel center (d2 + 0.5, d1 + 0.5)
        let (dx_fit, dy_fit) = reparsed.displacement(850.5, 750.5);
        let (true_dd1, true_dd2, _) = distort(750.0, 850.0);
        assert!(
            (dy_fit - true_dd1).abs() < 0.05 && (dx_fit - true_dd2).abs() < 0.05,
            "spline mismatch at (750,850): got ({dy_fit:.3}, {dx_fit:.3}), expected ({true_dd1:.3}, {true_dd2:.3})"
        );
    }

    #[test]
    fn smooth_spline_recovers_general_warp_and_builds_detector() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 1024usize;
        let dim2 = 1024usize;
        let c1 = 511.5;
        let c2 = 511.5;
        let r_scale = 512.0;

        let distort = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let dy = d1 - c1;
            let dx = d2 - c2;
            let r = dy.hypot(dx).max(1e-6);
            let rho = r / r_scale;
            let phi = dy.atan2(dx);
            let dr = 3.0 * rho.powi(3) + 1.5 * rho.powi(2) * (3.0 * phi).sin();
            (dr * dy / r, dr * dx / r, 0.0)
        };
        let pts = sample_ring_points_with_distortion(&poni, &cal, dim1, dim2, distort);

        let res = estimate_distortion(
            &pts,
            dim1,
            dim2,
            &poni,
            &cal,
            &DistortionOptions {
                model: DistortionModel::SmoothSpline {
                    grid_nx: 8,
                    grid_ny: 8,
                    smoothness: 0.2,
                },
                refine_poni: true,
                ..Default::default()
            },
        )
        .expect("SmoothSpline fit");

        assert!(res.significant);
        assert!(
            res.rms_after_px < 0.05,
            "SmoothSpline after RMS={:.4} px",
            res.rms_after_px
        );
        assert!(
            res.cv_rms_px < 0.08,
            "SmoothSpline CV RMS={:.4} px",
            res.cv_rms_px
        );

        let det = res
            .field
            .to_detector(dim1, dim2, poni.pixel1, poni.pixel2, None)
            .expect("to_detector");
        assert!(det.spline.is_some());
    }

    #[test]
    fn bow_z_recovers_out_of_plane_detector_curvature() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 1024usize;
        let dim2 = 1024usize;

        // 2 mm quadratic saddle bow across the detector
        let distort = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let x = (d2 - 512.0) / 512.0;
            let y = (d1 - 512.0) / 512.0;
            let dz = 2.0e-3 * (x * x - 0.6 * y * y);
            (0.0, 0.0, dz)
        };
        let pts = sample_ring_points_with_distortion(&poni, &cal, dim1, dim2, distort);

        let res = estimate_distortion(
            &pts,
            dim1,
            dim2,
            &poni,
            &cal,
            &DistortionOptions {
                model: DistortionModel::BowZ {
                    max_degree: 3,
                    smoothness: 1e-6,
                },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("BowZ fit");

        assert!(res.significant);
        assert!(
            res.rms_after_px < 0.02,
            "BowZ after RMS={:.4} px",
            res.rms_after_px
        );
        let dz_est = res.field.dz_m(512.0, 900.0);
        let dz_true = distort(512.0, 900.0).2;
        assert!(
            (dz_est - dz_true).abs() < 1e-4,
            "BowZ dz mismatch: est={:.3}mm, true={:.3}mm",
            dz_est * 1e3,
            dz_true * 1e3
        );
    }

    #[test]
    fn per_module_recovers_tile_offsets() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 1024usize;
        let dim2 = 1024usize;
        // 2×2 tile layout (500×500 modules, 24 px gaps)
        let mod_h = 500usize;
        let mod_w = 500usize;
        let gap_h = 24usize;
        let gap_w = 24usize;
        let true_shifts = [
            (1.2, -0.8),  // (0, 0)
            (-1.2, 0.8),  // (0, 1)
            (0.7, 1.1),   // (1, 0)
            (-0.7, -1.1), // (1, 1)
        ];

        let distort = |d1: f64, d2: f64| -> (f64, f64, f64) {
            if let Some(m) = module_index(d1, d2, mod_h, mod_w, gap_h, gap_w, 2, 2) {
                (true_shifts[m].0, true_shifts[m].1, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        };
        let pts = sample_ring_points_with_distortion(&poni, &cal, dim1, dim2, distort);

        let res = estimate_distortion(
            &pts,
            dim1,
            dim2,
            &poni,
            &cal,
            &DistortionOptions {
                model: DistortionModel::PerModule {
                    module_height: mod_h,
                    module_width: mod_w,
                    gap_height: gap_h,
                    gap_width: gap_w,
                    fit_z: false,
                    regularization: 1e-4,
                },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("PerModule fit");

        assert!(res.significant);
        assert!(
            res.rms_after_px < 0.03,
            "PerModule after RMS={:.4} px",
            res.rms_after_px
        );
        assert_eq!(res.module_offsets.len(), 4);
        for (m, off) in res.module_offsets.iter().enumerate() {
            assert!(
                (off.dd1_px - true_shifts[m].0).abs() < 0.08
                    && (off.dd2_px - true_shifts[m].1).abs() < 0.08,
                "module {m} mismatch: got ({:.2}, {:.2}), expected {:?}",
                off.dd1_px,
                off.dd2_px,
                true_shifts[m]
            );
        }
    }

    #[test]
    fn undistorted_detector_is_flagged_not_significant() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 1024usize;
        let dim2 = 1024usize;
        let mut pts =
            sample_ring_points_with_distortion(&poni, &cal, dim1, dim2, |_, _| (0.0, 0.0, 0.0));
        // Add small i.i.d. random jitter (~0.15 px) with zero true distortion
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_u = || -> f64 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((rng >> 11) as f64) / ((1u64 << 53) as f64)) - 0.5
        };
        for i in 0..pts.len() {
            pts.d1[i] += 0.3 * next_u();
            pts.d2[i] += 0.3 * next_u();
        }
        let res = estimate_distortion(
            &pts,
            dim1,
            dim2,
            &poni,
            &cal,
            &DistortionOptions {
                model: DistortionModel::SmoothSpline {
                    grid_nx: 8,
                    grid_ny: 8,
                    smoothness: 1.0,
                },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("undistorted fit");
        assert!(
            !res.significant,
            "pure noise must not be flagged significant: before={:.4}, cv={:.4}",
            res.rms_before_px, res.cv_rms_px
        );
    }

    #[test]
    fn multi_position_recovers_full_2d_shear_and_twist_distortion() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let dim1 = 1024usize;
        let dim2 = 1024usize;

        // Non-radial 2D shear + twist warp (partially tangential from any single
        // center, so multiple detector positions are required to recover both
        // δd₁ and δd₂ across the whole detector).
        let distort = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let y = (d1 - 512.0) / 512.0;
            let x = (d2 - 512.0) / 512.0;
            let dd1 = 2.5 * x * y + 1.5 * x * x * y;
            let dd2 = -2.0 * y * y + 1.8 * x * y * y;
            (dd1, dd2, 0.0)
        };

        // 3 translated detector positions + slightly different distances
        let mut p1 = make_test_poni();
        p1.poni1 = 360.0 * p1.pixel1;
        p1.poni2 = 380.0 * p1.pixel2;
        p1.distance = 0.19;

        let mut p2 = make_test_poni();
        p2.poni1 = 680.0 * p2.pixel1;
        p2.poni2 = 400.0 * p2.pixel2;
        p2.distance = 0.21;

        let mut p3 = make_test_poni();
        p3.poni1 = 440.0 * p3.pixel1;
        p3.poni2 = 690.0 * p3.pixel2;
        p3.distance = 0.23;

        let pts1 = sample_ring_points_with_distortion(&p1, &cal, dim1, dim2, distort);
        let pts2 = sample_ring_points_with_distortion(&p2, &cal, dim1, dim2, distort);
        let pts3 = sample_ring_points_with_distortion(&p3, &cal, dim1, dim2, distort);

        let obs = [
            DistortionObservation {
                points: &pts1,
                poni: &p1,
                calibrant: &cal,
            },
            DistortionObservation {
                points: &pts2,
                poni: &p2,
                calibrant: &cal,
            },
            DistortionObservation {
                points: &pts3,
                poni: &p3,
                calibrant: &cal,
            },
        ];

        let res = estimate_distortion_multi(
            &obs,
            dim1,
            dim2,
            &DistortionOptions {
                model: DistortionModel::SmoothSpline {
                    grid_nx: 8,
                    grid_ny: 8,
                    smoothness: 0.2,
                },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("multi-position SmoothSpline fit");

        assert_eq!(res.ponis.len(), 3);
        assert!(res.significant);
        assert!(
            res.rms_after_px < 0.04,
            "multi-position after RMS={:.4} px",
            res.rms_after_px
        );
        assert!(
            res.cv_rms_px < 0.06,
            "multi-position CV RMS={:.4} px",
            res.cv_rms_px
        );

        // Verify BOTH (δd₁, δd₂) components recovered accurately in the interior!
        for &(r, c) in &[
            (300.0, 300.0),
            (700.0, 350.0),
            (350.0, 720.0),
            (680.0, 680.0),
        ] {
            let (est_d1, est_d2) = res.field.displacement_px(r, c);
            let (true_d1, true_d2, _) = distort(r, c);
            assert!(
                (est_d1 - true_d1).abs() < 0.15 && (est_d2 - true_d2).abs() < 0.15,
                "2D displacement mismatch at ({r},{c}): got ({est_d1:.3}, {est_d2:.3}), expected ({true_d1:.3}, {true_d2:.3})"
            );
        }
    }

    #[test]
    fn image_repicking_pulls_in_large_corner_distortion() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_test_poni();
        let dim1 = 512usize;
        let dim2 = 512usize;
        let mut p = poni.clone();
        p.poni1 = 256.0 * p.pixel1;
        p.poni2 = 256.0 * p.pixel2;
        p.distance = 0.11;
        let tc = TrigonometricConstants::new(&p);
        let tths: Vec<f64> = cal
            .two_theta_rad(p.wavelength)
            .into_iter()
            .take(12)
            .flatten()
            .filter(|t| t.is_finite() && *t > 0.03 && *t < 0.7)
            .collect();

        // Synthesize a 2D image with a strong 6 px corner pincushion distortion
        // (larger than a 4 px radial search half-width!)
        let c1 = 255.5;
        let c2 = 255.5;
        let r_scale = 256.0;
        let mut img = vec![50.0_f64; dim1 * dim2];
        for r in 0..dim1 {
            for c in 0..dim2 {
                let dy = r as f64 - c1;
                let dx = c as f64 - c2;
                let rad = dy.hypot(dx).max(1e-6);
                let rho = rad / r_scale;
                let dr = 6.0 * rho.powi(3);
                let dd1 = dr * dy / rad;
                let dd2 = dr * dx / rad;
                let tth = eval_tth_at(&p, &tc, r as f64 + dd1, c as f64 + dd2, 0.0);
                let mut val = 50.0;
                for &rt in &tths {
                    let dt = (tth - rt) / 0.002;
                    if dt.abs() < 5.0 {
                        val += 2000.0 * (-0.5 * dt * dt).exp();
                    }
                }
                img[r * dim2 + c] = val;
            }
        }

        let opts_1pass = DistortionImageOptions {
            distortion: DistortionOptions {
                model: DistortionModel::RadialOptical { max_order: 5 },
                refine_poni: false,
                ..Default::default()
            },
            n_passes: 1,
            search_half_width_px: Some(4),
            ..Default::default()
        };
        let res_1pass = estimate_distortion_image(&img, dim1, dim2, None, &p, &cal, &opts_1pass)
            .expect("1-pass estimate_distortion_image");

        let opts = DistortionImageOptions {
            distortion: DistortionOptions {
                model: DistortionModel::RadialOptical { max_order: 5 },
                refine_poni: false,
                ..Default::default()
            },
            n_passes: 3,
            ..Default::default()
        };
        let res = estimate_distortion_image(&img, dim1, dim2, None, &p, &cal, &opts)
            .expect("estimate_distortion_image");
        assert_eq!(res.n_passes_completed, 3);
        assert!(res.distortion.significant);
        // Iterative re-picking must pull in outer points missed by the 1-pass ±4 px window.
        assert!(
            res.points[0].len() > res_1pass.points[0].len(),
            "expected 3-pass re-picking to recover more points than 1-pass: {} vs {}",
            res.points[0].len(),
            res_1pass.points[0].len()
        );
        let outer_pts = (0..res.points[0].len())
            .filter(|&i| {
                let r = (res.points[0].d1[i] - c1).hypot(res.points[0].d2[i] - c2);
                r > 235.0
            })
            .count();
        assert!(
            outer_pts >= 20,
            "expected >= 20 outer corner points (r > 235 px), got {outer_pts}"
        );
        // Verify fitted displacement at corner matches true +6*rho^3 in both sign and magnitude.
        let (est_d1, est_d2) = res.distortion.field.displacement_px(440.0, 440.0);
        let dy = 440.0 - c1;
        let dx = 440.0 - c2;
        let rad = dy.hypot(dx);
        let dr_true = 6.0 * (rad / r_scale).powi(3);
        let true_d1 = dr_true * dy / rad;
        let true_d2 = dr_true * dx / rad;
        assert!(
            (est_d1 - true_d1).abs() < 0.35 && (est_d2 - true_d2).abs() < 0.35,
            "corner displacement mismatch: got ({est_d1:.2}, {est_d2:.2}), expected ({true_d1:.2}, {true_d2:.2})"
        );
        assert!(
            res.distortion.rms_after_px < 0.15,
            "expected < 0.15 px after iterative re-picking, got {:.3} px (init={:.3} px)",
            res.distortion.rms_after_px,
            res.initial_rms_px
        );
        assert!(
            res.distortion.cv_rms_px < 0.20,
            "CV RMS too large: {:.3} px",
            res.distortion.cv_rms_px
        );
    }

    #[test]
    fn smooth_spline_suppresses_rigid_rotation_nullspace_while_preserving_real_twist() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let dim1 = 1024usize;
        let dim2 = 1024usize;
        let mut p = make_test_poni();
        p.poni1 = 512.0 * p.pixel1;
        p.poni2 = 512.0 * p.pixel2;
        p.rot1 = 1.2_f64.to_radians();
        p.rot2 = -0.9_f64.to_radians();

        // 1. Single centered exposure with pure radial pincushion distortion + small noise:
        //    SmoothSpline at grid=10 and grid=16 must have near-zero net rigid rotation
        //    and near-zero mean curl (|mean_curl * R| << max |δ|).
        let r_scale = 512.0;
        let distort_radial = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let dy = d1 - 512.0;
            let dx = d2 - 512.0;
            let r = dy.hypot(dx).max(1e-6);
            let rho = r / r_scale;
            let dr = 3.5 * rho.powi(3) - 0.8 * rho.powi(5);
            (dr * dy / r, dr * dx / r, 0.0)
        };
        let mut pts = sample_ring_points_with_distortion(&p, &cal, dim1, dim2, distort_radial);
        // Add deterministic ~0.08 px measurement noise to excite any nullspace mode.
        let mut seed = 0x1234_5678_9abc_def0_u64;
        for i in 0..pts.len() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            pts.d1[i] += 0.15 * u1;
            pts.d2[i] += 0.15 * u2;
        }

        for grid in [10usize, 16usize] {
            let res = estimate_distortion(
                &pts,
                dim1,
                dim2,
                &p,
                &cal,
                &DistortionOptions {
                    model: DistortionModel::SmoothSpline {
                        grid_nx: grid,
                        grid_ny: grid,
                        smoothness: 1.0,
                    },
                    refine_poni: true,
                    ..Default::default()
                },
            )
            .expect("SmoothSpline single centered radial fit");

            assert!(res.significant);
            assert!(
                res.rms_after_px < 0.12,
                "grid={grid}: expected rms_after_px < 0.12, got {:.4}",
                res.rms_after_px
            );

            // Measure max |δ|, mean curl (∂δd₁/∂d₂ - ∂δd₂/∂d₁), and net rigid rotation
            // amplitude across the interior.
            let mut max_disp = 0.0_f64;
            let mut curl_sum = 0.0_f64;
            let mut rot_num = 0.0_f64;
            let mut rot_den = 0.0_f64;
            let mut cnt = 0usize;
            let h = 4.0;
            for gr in 0..16 {
                let d1 = 96.0 + (gr as f64) * (832.0 / 15.0);
                let dy = d1 - 512.0;
                for gc in 0..16 {
                    let d2 = 96.0 + (gc as f64) * (832.0 / 15.0);
                    let dx = d2 - 512.0;
                    let (dd1, dd2) = res.field.displacement_px(d1, d2);
                    max_disp = max_disp.max(dd1.hypot(dd2));
                    rot_num += dx * dd1 - dy * dd2;
                    rot_den += dx * dx + dy * dy;
                    let (p_x1, _) = res.field.displacement_px(d1, d2 + h);
                    let (m_x1, _) = res.field.displacement_px(d1, d2 - h);
                    let (_, p_y2) = res.field.displacement_px(d1 + h, d2);
                    let (_, m_y2) = res.field.displacement_px(d1 - h, d2);
                    let curl = (p_x1 - m_x1) / (2.0 * h) - (p_y2 - m_y2) / (2.0 * h);
                    curl_sum += curl;
                    cnt += 1;
                }
            }
            let mean_curl_px = (curl_sum / cnt as f64).abs() * r_scale;
            let net_rot_px = (rot_num / rot_den).abs() * r_scale;
            assert!(
                max_disp > 1.5,
                "grid={grid}: expected real radial field (max |δ| > 1.5 px), got {max_disp:.3}"
            );
            assert!(
                mean_curl_px < 0.05 * max_disp,
                "grid={grid}: spurious curl swirl too large: |mean curl|*R = {mean_curl_px:.4} px vs max |δ| = {max_disp:.3} px"
            );
            assert!(
                net_rot_px < 0.02 * max_disp,
                "grid={grid}: spurious net rigid rotation too large: {net_rot_px:.4} px vs max |δ| = {max_disp:.3} px"
            );
        }

        // 2. Multi-exposure fit with a real non-uniform twist + shear across shifted
        //    beam centers must still recover both (δd₁, δd₂) accurately.
        let distort_twist = |d1: f64, d2: f64| -> (f64, f64, f64) {
            let y = (d1 - 512.0) / 512.0;
            let x = (d2 - 512.0) / 512.0;
            let r2 = x * x + y * y;
            // Non-uniform radial-dependent twist (zero mean rigid rotation) + shear:
            let swirl = 2.0 * (r2 - 0.5);
            let dd1 = swirl * x + 1.5 * x * y;
            let dd2 = -swirl * y + 1.2 * (x * x - y * y);
            (dd1, dd2, 0.0)
        };
        let mut p1 = make_test_poni();
        p1.poni1 = 350.0 * p1.pixel1;
        p1.poni2 = 370.0 * p1.pixel2;
        let mut p2 = make_test_poni();
        p2.poni1 = 680.0 * p2.pixel1;
        p2.poni2 = 390.0 * p2.pixel2;
        let mut p3 = make_test_poni();
        p3.poni1 = 460.0 * p3.pixel1;
        p3.poni2 = 690.0 * p3.pixel2;

        let pts1 = sample_ring_points_with_distortion(&p1, &cal, dim1, dim2, distort_twist);
        let pts2 = sample_ring_points_with_distortion(&p2, &cal, dim1, dim2, distort_twist);
        let pts3 = sample_ring_points_with_distortion(&p3, &cal, dim1, dim2, distort_twist);
        let obs = [
            DistortionObservation {
                points: &pts1,
                poni: &p1,
                calibrant: &cal,
            },
            DistortionObservation {
                points: &pts2,
                poni: &p2,
                calibrant: &cal,
            },
            DistortionObservation {
                points: &pts3,
                poni: &p3,
                calibrant: &cal,
            },
        ];
        let res_multi = estimate_distortion_multi(
            &obs,
            dim1,
            dim2,
            &DistortionOptions {
                model: DistortionModel::SmoothSpline {
                    grid_nx: 10,
                    grid_ny: 10,
                    smoothness: 0.2,
                },
                refine_poni: false,
                ..Default::default()
            },
        )
        .expect("multi-exposure real twist SmoothSpline fit");

        assert!(res_multi.significant);
        assert!(
            res_multi.rms_after_px < 0.12,
            "multi-exposure twist rms_after_px={:.4}",
            res_multi.rms_after_px
        );
        for &(r, c) in &[
            (280.0, 280.0),
            (740.0, 300.0),
            (310.0, 730.0),
            (720.0, 720.0),
        ] {
            let (est1, est2) = res_multi.field.displacement_px(r, c);
            let (t1, t2, _) = distort_twist(r, c);
            assert!(
                (est1 - t1).abs() < 0.25 && (est2 - t2).abs() < 0.25,
                "multi-exposure twist mismatch at ({r}, {c}): got ({est1:.3}, {est2:.3}), expected ({t1:.3}, {t2:.3})"
            );
        }

        // 3. Real centered MarCCD dataset (LaB6_40keV_MarCCD, grid=10, smoothness=1.0):
        //    must have near-zero net rigid rotation and near-zero mean curl instead of
        //    drifting into a multi-pixel swirl.
        if std::path::Path::new("test_data/LaB6_40keV_MarCCD.tif").exists() {
            let frame =
                crate::frame::Frame::from_tiff("test_data/LaB6_40keV_MarCCD.tif".to_string())
                    .expect("load LaB6_40keV_MarCCD.tif");
            let poni_mar = Poni::load("test_data/LaB6_40keV_MarCCD.poni")
                .expect("load LaB6_40keV_MarCCD.poni");
            let cal_lab6 = Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D")
                .expect("load LaB6_SRM660b.D");
            let mar_res = estimate_distortion_image(
                &frame.data,
                frame.dim1,
                frame.dim2,
                None,
                &poni_mar,
                &cal_lab6,
                &DistortionImageOptions {
                    distortion: DistortionOptions {
                        model: DistortionModel::SmoothSpline {
                            grid_nx: 10,
                            grid_ny: 10,
                            smoothness: 1.0,
                        },
                        refine_poni: true,
                        ..Default::default()
                    },
                    n_passes: 2,
                    ..Default::default()
                },
            )
            .expect("LaB6_40keV_MarCCD SmoothSpline fit");

            let r_half = 0.5 * (frame.dim1.max(frame.dim2) as f64);
            let c1 = 0.5 * (frame.dim1 as f64);
            let c2 = 0.5 * (frame.dim2 as f64);
            let mut max_disp = 0.0_f64;
            let mut curl_sum = 0.0_f64;
            let mut rot_num = 0.0_f64;
            let mut rot_den = 0.0_f64;
            let mut cnt = 0usize;
            let h = 4.0;
            for gr in 0..16 {
                let d1 = 0.15 * frame.dim1 as f64 + (gr as f64) * (0.70 * frame.dim1 as f64 / 15.0);
                let dy = d1 - c1;
                for gc in 0..16 {
                    let d2 =
                        0.15 * frame.dim2 as f64 + (gc as f64) * (0.70 * frame.dim2 as f64 / 15.0);
                    let dx = d2 - c2;
                    let (dd1, dd2) = mar_res.distortion.field.displacement_px(d1, d2);
                    max_disp = max_disp.max(dd1.hypot(dd2));
                    rot_num += dx * dd1 - dy * dd2;
                    rot_den += dx * dx + dy * dy;
                    let (p_x1, _) = mar_res.distortion.field.displacement_px(d1, d2 + h);
                    let (m_x1, _) = mar_res.distortion.field.displacement_px(d1, d2 - h);
                    let (_, p_y2) = mar_res.distortion.field.displacement_px(d1 + h, d2);
                    let (_, m_y2) = mar_res.distortion.field.displacement_px(d1 - h, d2);
                    curl_sum += (p_x1 - m_x1) / (2.0 * h) - (p_y2 - m_y2) / (2.0 * h);
                    cnt += 1;
                }
            }
            let mean_curl_px = (curl_sum / cnt as f64).abs() * r_half;
            let net_rot_px = (rot_num / rot_den).abs() * r_half;
            assert!(
                net_rot_px < 0.05 && mean_curl_px < 0.15,
                "LaB6_40keV_MarCCD drifted into rigid rotation / curl swirl: net_rot={net_rot_px:.4} px, |mean_curl|*R={mean_curl_px:.4} px (max|δ|={max_disp:.3} px)"
            );
        }
    }
}
