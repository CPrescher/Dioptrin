#![allow(dead_code)]
mod cache;
mod corrections;
mod methods;
#[cfg(test)]
mod tests;

use std::cell::RefCell;

use crate::geometry::Geometry;
use crate::mask::Mask;

use cache::{Lut1dCache, Lut2dCache};

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
    /// Supersampling with `n_ss x n_ss` subpixels per pixel.
    Supersampled { n_ss: usize },
}

pub struct Integrator {
    geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    cache_1d: RefCell<Lut1dCache>,
    cache_2d: RefCell<Lut2dCache>,
    mask: Option<Mask>,
    method: IntegrationMethod,
    polarization_factor: Option<f64>,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            tth: vec![],
            chi: vec![],
            solid_angle: vec![],
            polarization: vec![],
            cache_1d: RefCell::new(Lut1dCache::new()),
            cache_2d: RefCell::new(Lut2dCache::new()),
            mask: None,
            method: IntegrationMethod::PixelSplit,
            polarization_factor: None,
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

    // builder pattern for mask
    pub fn with_mask(mut self, mask: Mask) -> Self {
        self.mask = Some(mask);
        self
    }

    // clearing current mask
    pub fn clear_mask(&mut self) {
        self.mask = None;
    }

    // getter for mask
    pub fn mask(&self) -> Option<&Mask> {
        self.mask.as_ref()
    }

    // setter for mask
    pub fn set_mask(&mut self, mask: Mask) {
        self.mask = Some(mask);
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

    // setter for polarization factor
    pub fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) {
        self.polarization_factor = polarization_factor;
        self.update_polarization();
    }

    // getter for geometry
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    // getter for solid angle correction for each pixel as a flattened vector
    pub fn solid_angle(&self) -> &Vec<f64> {
        &self.solid_angle
    }

    // getter for polarization correction for each pixel as a flattened vector
    pub fn polarization(&self) -> &Vec<f64> {
        &self.polarization
    }

    // getter for tth vector for each pixel as a flattened vector
    pub fn tth(&self) -> &Vec<f64> {
        &self.tth
    }

    // getter for chi vector for each pixel as a flattened vector
    pub fn chi(&self) -> &Vec<f64> {
        &self.chi
    }
}
