pub mod common;
pub mod counting;
pub mod supersampled;
pub mod pixel_split;
#[cfg(feature = "lut_metrics")]
pub mod metrics;

// Re-export public API to keep crate::lut::* stable
pub use common::hull_grid;
pub use counting::one_d::{integrate, lut};
pub use counting::two_d::{integrate2d, lut_2d};
pub use counting::csr::{
    CsrLut, CsrLut2d, integrate2d_csr, integrate_csr, lut2d_csr_from_tth_azi, lut_csr_from_tth,
};
pub use supersampled::one_d::{integrate_ss, lut_ss};
pub use supersampled::two_d::{integrate2d_ss, lut_2d_ss};
pub use supersampled::csr::{
    CsrLut2dSs, CsrLutSs, integrate2d_ss_csr, integrate_ss_csr, lut2d_ss_csr, lut_ss_csr,
};
pub use pixel_split::one_d::lut_ps;
pub use pixel_split::two_d::lut_2d_ps;
pub use pixel_split::csr::{lut2d_ps_csr, lut_ps_csr};
#[cfg(feature = "lut_metrics")]
pub use metrics::{
    analyze_csr_locality, analyze_lut2d_locality, analyze_lut2d_ss_locality, analyze_lut_locality,
    analyze_lut_ss_locality, LutLocalityStats,
};
