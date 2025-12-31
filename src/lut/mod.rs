pub mod common;
pub mod base;
pub mod cake2d;
pub mod supersampled;
pub mod csr;
#[cfg(feature = "lut_metrics")]
pub mod metrics;

// Re-export public API to keep crate::lut::* stable
pub use common::hull_grid;
pub use base::{integrate, lut};
pub use cake2d::{integrate2d, lut_2d};
pub use supersampled::{integrate2d_ss, integrate_ss, lut_2d_ss, lut_ss};
pub use csr::{
    CsrLut, CsrLut2d, CsrLut2dSs, CsrLutSs, integrate2d_csr, integrate2d_ss_csr, integrate_csr,
    integrate_ss_csr, lut2d_csr_from_tth_azi, lut2d_ss_csr, lut_csr_from_tth, lut_ss_csr,
};
#[cfg(feature = "lut_metrics")]
pub use metrics::{
    analyze_csr_locality, analyze_lut2d_locality, analyze_lut2d_ss_locality, analyze_lut_locality,
    analyze_lut_ss_locality, LutLocalityStats,
};
