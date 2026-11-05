//! Unified LUT API
//!
//! - 1D/2D CSR LUTs share unified structs: `Lut1d` and `Lut2d`.
//! - Builders use the `build_*` prefix:
//!   - `build_lut1d_csr`, `build_lut1d_ss_csr`, `build_lut1d_ps_csr`
//!   - `build_lut2d_csr`, `build_lut2d_ss_csr`, `build_lut2d_ps_csr`
//! - Integrators:
//!   - `integrate1d(intensity, &Lut1d, hist_min, hist_step)`
//!   - `integrate2d(intensity, &Lut2d)`
//!   - `integrate1d_with_errors(intensity, variance, &Lut1d, mask)`
//!   - `integrate2d_with_errors(intensity, variance, &Lut2d, mask)`
//!
//! The `values: Option<Vec<f64>>` field on LUT structs indicates whether the LUT
//! is weighted (supersampled/pixel-split) or unweighted (counting/mean of contributing pixels).
//! HashMap- and Vec-of-Vec-based LUTs have been removed from the public API.
//!
//! Module layout
//! - `counting_csr.rs`: unweighted CSR LUT builders (`build_lut1d_csr`, `build_lut2d_csr`), types `Lut1d`, `Lut2d`.
//! - `pixel_split.rs`: pixel-splitting CSR LUT builders (`build_lut1d_ps_csr`, `build_lut2d_ps_csr`).
//! - `supersampled.rs`: supersampled CSR LUT builders (`build_lut1d_ss_csr`, `build_lut2d_ss_csr`).
//! - `integrate.rs`: shared integrators (`integrate1d`, `integrate2d`) that work with any CSR LUT.

pub(crate) mod azimuthal;
pub use azimuthal::{AzimuthalIntegrationResult, integrate1d_with_azimuthal_std};

pub mod common;
mod counting_csr;
mod csc;
#[cfg(test)]
pub(crate) mod fast;
mod integrate;
mod pixel_split;
pub(crate) mod prepared;
mod supersampled;

// Re-export RadialUnit from integration module for use in LUT builders
pub use crate::integration::RadialUnit;

// Re-export unified public API
pub use common::hull_grid;
pub use counting_csr::{
    Lut1d, Lut2d, build_lut1d_csr, build_lut1d_csr_with_range, build_lut2d_csr,
    build_lut2d_csr_with_azimuthal_range,
};
pub use integrate::{
    IntermediateResult1d, IntermediateResult2d, integrate1d, integrate1d_raw,
    integrate1d_raw_with_variance, integrate1d_with_errors, integrate2d, integrate2d_raw,
    integrate2d_raw_with_variance, integrate2d_with_errors,
};
pub use pixel_split::{
    build_lut1d_ps_csr, build_lut1d_ps_csr_with_range, build_lut2d_ps_csr,
    build_lut2d_ps_csr_with_azimuthal_range,
};
pub use supersampled::build_lut2d_ss_csr_with_azimuthal_range;
pub use supersampled::{build_lut1d_ss_csr, build_lut1d_ss_csr_with_range, build_lut2d_ss_csr};
