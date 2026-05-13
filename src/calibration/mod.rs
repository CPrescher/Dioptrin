//! Geometry refinement (calibration) for X-ray powder diffraction.
//!
//! Given a starting-guess `Poni`, a set of pixel-position control points each
//! tagged with (or assignable to) a Debye–Scherrer ring of a reference
//! calibrant, refine the PONI parameters so that the predicted 2θ at every
//! point matches the calibrant's known 2θ for that ring.
//!
//! Design goals (compared to pyFAI's `GeometryRefinement` and Dioptas's
//! wrapper):
//!
//! - **Robust to spotty patterns**: optional Huber/Tukey IRLS loss so a few
//!   bad points (single-crystal spots, mis-clicks) don't pull the fit.
//! - **Works with off-detector beam centers**: the forward model uses
//!   physical coordinates; there is no "PONI must lie inside the detector"
//!   assertion anywhere.
//! - **Auto ring-assignment**: if a control point's ring is unknown, it can be
//!   assigned to the nearest predicted-2θ ring at each iteration.
//! - **Closed parameter set in one pass**: distance, poni1/poni2, rot1/2/3,
//!   and wavelength can be free or fixed independently.
//!
//! ## Coordinate convention
//!
//! Pixel positions are `(d1, d2)` where `d1` is the row index (slow / Y axis)
//! and `d2` is the column index (fast / X axis), matching the rest of
//! Dioptrin's `Geometry`. Both are `f64` so sub-pixel centroids are fine.

mod calibrant;
mod ellipse;
mod massif;
mod refinement;

#[cfg(test)]
mod tests;

pub use calibrant::{Calibrant, CalibrantError};
pub use ellipse::{Ellipse, RansacOptions, RansacResult, fit_least_squares, ransac_ellipses};
pub use massif::{MassifOptions, find_peaks};
pub use refinement::{
    CalibrationError, CalibrationOptions, CalibrationPoints, CalibrationResult, Loss, ParamMask,
    refine,
};
