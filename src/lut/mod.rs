//! Unified LUT API
//!
//! - 1D/2D CSR LUTs share unified structs: `Lut1d` and `Lut2d`.
//! - Builders use the `build_*` prefix:
//!   - `build_lut1d_csr`, `build_lut1d_ss_csr`, `build_lut1d_ps_csr`
//!   - `build_lut2d_csr`, `build_lut2d_ss_csr`, `build_lut2d_ps_csr`
//! - Integrators:
//!   - `integrate_lut1d(intensity, &Lut1d, hist_min, hist_step)`
//!   - `integrate_lut2d(intensity, &Lut2d)`
//!
//! The `values: Option<Vec<f64>>` field on LUT structs indicates whether the LUT
//! is weighted (supersampled/pixel-split) or unweighted (counting/mean of contributing pixels).
//! HashMap- and Vec-of-Vec-based LUTs have been removed from the public API.

pub mod common;
mod counting_csr;
mod pixel_split;
mod supersampled;
#[cfg(feature = "lut_metrics")]
pub mod metrics;

// Re-export unified public API
pub use common::hull_grid;
pub use counting_csr::{
    Lut1d, Lut2d,
    build_lut1d_csr,
    build_lut2d_csr,
    integrate_lut1d, integrate_lut2d,
};
pub use supersampled::{build_lut1d_ss_csr, build_lut2d_ss_csr};
pub use pixel_split::{build_lut1d_ps_csr, build_lut2d_ps_csr};
#[cfg(feature = "lut_metrics")]
pub use metrics::{analyze_csr_locality, LutLocalityStats};
