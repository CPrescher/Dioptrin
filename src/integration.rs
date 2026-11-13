#![allow(dead_code)]
mod arithmetic;
mod cache;
mod corrections;
mod dynamic;
pub(crate) mod grid;
pub use grid::{AzimuthalRange2dResult, IntegrationGrid2d, PixelSplitMode};
mod fixed;
mod fixed_csr;
mod layout;
pub use dynamic::{Dynamic1dPlan, Dynamic1dResult, Dynamic1dWorker};
pub use fixed::{Fixed1dPlan, Fixed1dResult, Fixed1dWorker, InputOrder};
pub use fixed_csr::{FixedCsr1dInputPlan, FixedCsr1dPlan, FixedGrid2dPlan};
pub use layout::{LayoutPrediction, predict_1d_layout};
mod methods;
#[cfg(test)]
mod tests;
mod tiff_input;

use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

use crate::geometry::Geometry;
use crate::mask::Mask;

use cache::{Lut1dCache, Lut2dCache};

/// Mutable geometry state that can be lazily initialized.
/// Wrapped in RwLock for thread-safe access.
struct GeometryState {
    /// Dimensions the state was computed for
    dim1: usize,
    dim2: usize,
    /// Two-theta values in radians
    tth: Vec<f64>,
    /// Radial values in the selected unit
    radial: Vec<f64>,
    azimuth: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    /// Combined normalization: polarization * solid_angle (pyFAI-style).
    /// Wrapped in Arc so per-frame integrations can share it without copying.
    normalization: Arc<Vec<f64>>,
}

impl GeometryState {
    fn new() -> Self {
        Self {
            dim1: 0,
            dim2: 0,
            tth: vec![],
            radial: vec![],
            azimuth: vec![],
            solid_angle: vec![],
            polarization: vec![],
            normalization: Arc::new(vec![]),
        }
    }
}

/// Radial unit for integration output.
///
/// Defines the unit used for the radial (scattering angle / momentum transfer) axis
/// in 1D and 2D integrations. The LUT is built directly on the grid in the selected unit,
/// ensuring correct binning for non-linear transforms (e.g., tth to q).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RadialUnit {
    /// Two-theta in radians (default)
    #[default]
    TthRad,
    /// Two-theta in degrees
    TthDeg,
    /// Scattering vector q in inverse Angstroms (Å⁻¹)
    QAngstrom,
    /// Scattering vector q in inverse nanometers (nm⁻¹)
    QNm,
}

impl RadialUnit {
    /// Convert two-theta values (in radians) to this unit.
    ///
    /// For q units, requires the wavelength in meters.
    /// Panics if wavelength is required but not provided or is zero/negative.
    pub fn convert_from_tth(&self, tth_rad: &[f64], wavelength: f64) -> Vec<f64> {
        match self {
            RadialUnit::TthRad => tth_rad.to_vec(),
            RadialUnit::TthDeg => tth_rad.iter().map(|&t| t.to_degrees()).collect(),
            RadialUnit::QAngstrom => {
                assert!(
                    wavelength > 0.0,
                    "Wavelength must be positive for q conversion"
                );
                // q = 4π sin(θ) / λ, where θ = tth/2
                // wavelength in meters, convert to Angstroms (1e10)
                let wl_angstrom = wavelength * 1e10;
                let factor = 4.0 * PI / wl_angstrom;
                tth_rad.iter().map(|&t| factor * (t / 2.0).sin()).collect()
            }
            RadialUnit::QNm => {
                assert!(
                    wavelength > 0.0,
                    "Wavelength must be positive for q conversion"
                );
                // q = 4π sin(θ) / λ, where θ = tth/2
                // wavelength in meters, convert to nm (1e9)
                let wl_nm = wavelength * 1e9;
                let factor = 4.0 * PI / wl_nm;
                tth_rad.iter().map(|&t| factor * (t / 2.0).sin()).collect()
            }
        }
    }

    /// Returns true if this unit requires a wavelength for conversion.
    pub fn requires_wavelength(&self) -> bool {
        matches!(self, RadialUnit::QAngstrom | RadialUnit::QNm)
    }
}

/// Integration method selection (applies to both 1D and 2D integrations).
///
/// Examples
/// ```
/// use dioptrin::integration::{Integrator, IntegrationMethod};
/// use dioptrin::{frame::Frame, geometry::Geometry, poni::Poni};
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let frame = Frame {
///     name: String::from("demo"),
///     data: vec![1.0, 2.0, 3.0, 4.0],
///     dim1: 2,
///     dim2: 2,
/// };
///
/// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
/// let mut integrator = Integrator::new(geometry)
///     .with_method(IntegrationMethod::Counting);
/// let (_x, y) = integrator.integrate1d(&frame, 4);
/// assert_eq!(y.len(), 4);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum IntegrationMethod {
    /// Histogram (unweighted) integration.
    Histogram,
    /// Average of contributing pixels per bin (unweighted).
    Counting,
    /// Pixel-splitting with fractional/bilinear weights.
    PixelSplit,
    /// Approximate pyFAI-compatible full splitting for 1D and 2D.
    PyfaiFull,
    /// Supersampling with `n_ss x n_ss` subpixels per pixel.
    Supersampled { n_ss: usize },
}

/// How masks are expected to change between frames. Both modes use float64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MaskMode {
    /// Compile the current mask into CSR once and cache normalization.
    /// Subsequent mask edits are supported and refresh only the prepared table.
    #[default]
    Fixed,
    /// Keep geometry tables independent of the mask and accumulate normalization
    /// on each frame. Fast arithmetic uses CSC; strict arithmetic uses fused CSR.
    Dynamic,
}

/// Sparse execution layout for fast 1D integration. Strict arithmetic and
/// azimuthal spread retain CSR; 2D policy is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SparseLayout {
    #[default]
    Csc,
    Csr,
}

pub struct Integrator {
    geometry: Geometry,
    /// Mutable geometry state (tth, radial, azimuth, corrections) in RwLock for thread-safe access
    state: RwLock<GeometryState>,
    cache_1d: RwLock<Lut1dCache>,
    cache_2d: RwLock<Lut2dCache>,
    mask: Option<Mask>,
    method: IntegrationMethod,
    polarization_factor: Option<f64>,
    radial_unit: RadialUnit,
    radial_range: Option<(f64, f64)>,
    fast_math: bool,
    mask_mode: MaskMode,
    sparse_layout: SparseLayout,
    /// Default 2D radial edges for `integrate2d_on_azimuthal_range`.
    radial_edges_2d: RwLock<Option<(grid::RadialEdgesKey, (f64, f64))>>,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            state: RwLock::new(GeometryState::new()),
            cache_1d: RwLock::new(Lut1dCache::new()),
            cache_2d: RwLock::new(Lut2dCache::new()),
            mask: None,
            method: IntegrationMethod::PixelSplit,
            polarization_factor: None,
            radial_unit: RadialUnit::default(),
            radial_range: None,
            fast_math: true,
            mask_mode: MaskMode::default(),
            sparse_layout: SparseLayout::default(),
            radial_edges_2d: RwLock::new(None),
        }
    }

    // builder pattern for method
    /// Select reassociated CSR accumulation (and hardware FMA where available).
    /// Geometry, LUT weights, histogram and azimuthal-spread algorithms are unchanged.
    /// Enabled by default. Use false for strict accumulation; rounding can differ.
    pub fn with_fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }

    pub fn fast_math(&self) -> bool {
        self.fast_math
    }

    /// Changes accumulation without rebuilding the calibration/LUT.
    pub fn set_fast_math(&mut self, enabled: bool) {
        self.fast_math = enabled;
    }

    pub fn sparse_layout(&self) -> SparseLayout {
        self.sparse_layout
    }
    pub fn with_sparse_layout(mut self, layout: SparseLayout) -> Self {
        self.set_sparse_layout(layout);
        self
    }
    /// Select execution without discarding geometry, weights or valid caches.
    pub fn set_sparse_layout(&mut self, layout: SparseLayout) {
        self.sparse_layout = layout;
        if let Some(lut) = self
            .cache_1d
            .write()
            .expect("1D LUT cache poisoned")
            .lut
            .as_mut()
        {
            lut.set_sparse_layout(layout);
        }
    }

    pub fn mask_mode(&self) -> MaskMode {
        self.mask_mode
    }

    pub fn with_mask_mode(mut self, mode: MaskMode) -> Self {
        self.set_mask_mode(mode);
        self
    }

    /// Switch execution policy without rebuilding geometry or pixel weights.
    pub fn set_mask_mode(&mut self, mode: MaskMode) {
        if self.mask_mode != mode {
            self.mask_mode = mode;
            if let Some(lut) = self
                .cache_1d
                .write()
                .expect("1D LUT cache lock poisoned")
                .lut
                .as_mut()
            {
                lut.set_mask_mode(mode);
            }
            if let Some(lut) = self
                .cache_2d
                .write()
                .expect("2D LUT cache lock poisoned")
                .lut
                .as_mut()
            {
                lut.set_mask_mode(mode);
            }
        }
    }

    // builder pattern for method
    pub fn with_method(mut self, method: IntegrationMethod) -> Self {
        self.method = method;
        self
    }

    // getter for method
    pub fn method(&self) -> IntegrationMethod {
        self.method
    }

    // setter for method
    pub fn set_method(&mut self, method: IntegrationMethod) {
        self.method = method;
    }

    // builder pattern for radial unit
    pub fn with_radial_unit(mut self, unit: RadialUnit) -> Self {
        self.radial_unit = unit;
        self.invalidate_caches();
        self
    }

    // getter for radial unit
    pub fn radial_unit(&self) -> RadialUnit {
        self.radial_unit
    }

    // setter for radial unit
    pub fn set_radial_unit(&mut self, unit: RadialUnit) {
        if self.radial_unit != unit {
            self.radial_unit = unit;
            // Recompute radial values and invalidate caches
            let mut state = self.state.write().expect("geometry state lock poisoned");
            if !state.tth.is_empty() {
                state.radial = self
                    .radial_unit
                    .convert_from_tth(&state.tth, self.geometry.poni.wavelength);
            }
            drop(state);
            self.invalidate_caches();
        }
    }

    // builder pattern for mask
    pub fn with_mask(mut self, mask: Mask) -> Self {
        self.mask = Some(mask);
        self.invalidate_mask();
        self
    }

    // clearing current mask
    pub fn clear_mask(&mut self) {
        self.mask = None;
        self.invalidate_mask();
    }

    // getter for mask
    pub fn mask(&self) -> Option<&Mask> {
        self.mask.as_ref()
    }

    // setter for mask
    pub fn set_mask(&mut self, mask: Mask) {
        self.mask = Some(mask);
        self.invalidate_mask();
    }

    // builder pattern for radial range
    pub fn with_radial_range(mut self, min: f64, max: f64) -> Self {
        self.radial_range = Some((min, max));
        self.invalidate_caches();
        self
    }

    // getter for radial range
    pub fn radial_range(&self) -> Option<(f64, f64)> {
        self.radial_range
    }

    // setter for radial range
    pub fn set_radial_range(&mut self, range: Option<(f64, f64)>) {
        if self.radial_range != range {
            self.radial_range = range;
            self.invalidate_caches();
        }
    }

    // builder pattern for polarization factor
    pub fn with_polarization_factor(mut self, polarization_factor: f64) -> Self {
        self.polarization_factor = Some(polarization_factor);
        self.update_polarization();
        self
    }

    // getter for polarization factor
    pub fn polarization_factor(&self) -> Option<f64> {
        self.polarization_factor
    }

    /// Update polarization corrections only when the factor changes.
    /// Reapplying the current factor preserves correction arrays and cached normalization.
    pub fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) {
        if self.polarization_factor == polarization_factor {
            return;
        }
        self.polarization_factor = polarization_factor;
        self.update_polarization();
    }

    // getter for geometry
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    // getter for solid angle correction for each pixel as a flattened vector
    pub fn solid_angle(&self) -> Vec<f64> {
        self.state
            .read()
            .expect("geometry state lock poisoned")
            .solid_angle
            .clone()
    }

    // getter for polarization correction for each pixel as a flattened vector
    pub fn polarization(&self) -> Vec<f64> {
        self.state
            .read()
            .expect("geometry state lock poisoned")
            .polarization
            .clone()
    }

    // getter for tth vector for each pixel as a flattened vector (always in radians)
    pub fn tth(&self) -> Vec<f64> {
        self.state
            .read()
            .expect("geometry state lock poisoned")
            .tth
            .clone()
    }

    // getter for radial vector for each pixel in the selected unit
    pub fn radial(&self) -> Vec<f64> {
        self.state
            .read()
            .expect("geometry state lock poisoned")
            .radial
            .clone()
    }

    // getter for azimuth vector for each pixel as a flattened vector
    pub fn azimuth(&self) -> Vec<f64> {
        self.state
            .read()
            .expect("geometry state lock poisoned")
            .azimuth
            .clone()
    }

    // helper to invalidate LUT caches
    fn invalidate_caches(&self) {
        let mut cache_1d = self.cache_1d.write().expect("1D LUT cache lock poisoned");
        cache_1d.key = None;
        cache_1d.lut = None;
        drop(cache_1d);
        let mut cache_2d = self.cache_2d.write().expect("2D LUT cache lock poisoned");
        cache_2d.key = None;
        cache_2d.lut = None;
    }
}
