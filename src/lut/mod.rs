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
