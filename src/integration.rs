#![allow(dead_code)]
use crate::lut;
use crate::math::histogram;
use std::cell::RefCell;

use super::frame::Frame;
use super::geometry::Geometry;
use crate::mask::Mask;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OneDKey {
    Counting { n_points: usize },
    PixelSplit { n_points: usize },
    Supersampled { n_points: usize, n_ss: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoDKey {
    Counting {
        n_tth: usize,
        n_chi: usize,
    },
    PixelSplit {
        n_tth: usize,
        n_chi: usize,
    },
    Supersampled {
        n_tth: usize,
        n_chi: usize,
        n_ss: usize,
    },
    CountingChi {
        n_tth: usize,
        n_chi: usize,
        chi: ChiRangeKey,
    },
    PixelSplitChi {
        n_tth: usize,
        n_chi: usize,
        chi: ChiRangeKey,
    },
    SupersampledChi {
        n_tth: usize,
        n_chi: usize,
        n_ss: usize,
        chi: ChiRangeKey,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChiRangeKey {
    start_bits: u64,
    step_bits: u64,
}

impl ChiRangeKey {
    fn new(start: f64, step: f64) -> Self {
        Self { start_bits: start.to_bits(), step_bits: step.to_bits() }
    }
}

struct LutCache<K, L> {
    key: Option<K>,
    lut: Option<L>,
}

impl<K, L> LutCache<K, L> {
    fn new() -> Self {
        Self {
            key: None,
            lut: None,
        }
    }
}

type Lut1dCache = LutCache<OneDKey, lut::Lut1d>;
type Lut2dCache = LutCache<TwoDKey, lut::Lut2d>;

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

    /// Integrate a frame to 1D using the configured integration method.
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
    /// let (_x, y_count) = integrator.integrate1d(&frame, 4);
    /// integrator.set_method(IntegrationMethod::PixelSplit);
    /// let (_x, y_ps) = integrator.integrate1d(&frame, 4);
    /// integrator.set_method(IntegrationMethod::Supersampled { n_ss: 2 });
    /// let (_x, y_ss) = integrator.integrate1d(&frame, 4);
    /// assert_eq!(y_count.len(), 4);
    /// assert_eq!(y_ps.len(), 4);
    /// assert_eq!(y_ss.len(), 4);
    /// ```
    pub fn integrate1d(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d(&intensity, &*lut1d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 1D and return per-bin errors (standard deviations).
    pub fn integrate1d_with_errors(
        &mut self,
        frame: &Frame,
        n_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => histogram::histogram_with_errors(
                &self.tth, &intensity, &variance, n_points, mask_data,
            ),
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d_with_errors(&intensity, &variance, &*lut1d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D using the configured integration method.
    ///
    /// Example
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
    ///     .with_method(IntegrationMethod::PixelSplit);
    /// let y = integrator.integrate2d(&frame, 2, 2);
    /// assert_eq!(y.len(), 4);
    /// ```
    pub fn integrate2d(&mut self, frame: &Frame, n_tth: usize, n_chi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_tth, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_chi, n_ss });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D using an explicit chi start and step.
    ///
    /// For a full-circle chi range, use `chi_step = 2*pi / n_chi` and a
    /// `chi_start` that defines the center of the first bin (e.g. `0.0` for 0..2*pi).
    /// Chi bins wrap periodically, so angles outside the total range fold into
    /// `0..n_chi`.
    /// Note: `IntegrationMethod::Histogram` ignores chi binning.
    pub fn integrate2d_with_chi_range(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
        chi_start: f64,
        chi_step: f64,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let chi = ChiRangeKey::new(chi_start, chi_step);
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_tth, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingChi { n_tth, n_chi, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitChi { n_tth, n_chi, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi });
                crate::lut::integrate2d(&intensity, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D and return per-bin errors (standard deviations).
    pub fn integrate2d_with_errors(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &self.tth, &intensity, &variance, n_tth, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_chi, n_ss });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
        };
        result
    }

    /// Integrate a frame to 2D with errors using an explicit chi binning definition.
    ///
    /// `chi_start` is the center of the first chi bin; the effective start edge is
    /// `chi_start - chi_step / 2`. Chi bins wrap periodically, so angles outside
    /// the total range fold into `0..n_chi`.
    /// Note: `IntegrationMethod::Histogram` ignores chi binning.
    pub fn integrate2d_with_chi_range_and_errors(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_chi: usize,
        chi_start: f64,
        chi_step: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let (intensity, variance) = self.correct_intensity_and_variance(&frame.data);
        let mask_data = self.mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let chi = ChiRangeKey::new(chi_start, chi_step);
        let result = match self.method {
            IntegrationMethod::Histogram => {
                let (_bins, y, sigma) = histogram::histogram_with_errors(
                    &self.tth, &intensity, &variance, n_tth, mask_data,
                );
                (y, sigma)
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::CountingChi { n_tth, n_chi, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplitChi { n_tth, n_chi, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi });
                crate::lut::integrate2d_with_errors(&intensity, &variance, &*lut2d, mask_data)
            }
        };
        result
    }

    /// checks whether current tth, chi, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated
    fn update_geometry_arrays(&mut self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;
        let stale = self.geometry.dim1 != dim1
            || self.geometry.dim2 != dim2
            || self.tth.len() != size
            || self.chi.len() != size
            || self.solid_angle.len() != size
            || self.polarization.len() != size;
        if stale {
            self.geometry.dim1 = dim1;
            self.geometry.dim2 = dim2;
            self.tth = self.geometry.calc_tth_all();
            self.chi = self.geometry.calc_chi_all();
            self.solid_angle = self.geometry.calc_solid_angle();
            self.update_polarization();
            let mut cache_1d = self.cache_1d.borrow_mut();
            cache_1d.key = None;
            cache_1d.lut = None;
            drop(cache_1d);
            let mut cache_2d = self.cache_2d.borrow_mut();
            cache_2d.key = None;
            cache_2d.lut = None;
        }
    }

    fn update_polarization(&mut self) {
        self.polarization = match self.polarization_factor {
            Some(factor) => self
                .geometry
                .calc_polarization(factor, &self.tth, &self.chi),
            None => vec![1.0; self.geometry.size()],
        };
    }

    /// checks whether current lut was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    fn ensure_lut1d(&self, key: OneDKey) -> std::cell::Ref<'_, lut::Lut1d> {
        let rebuild = {
            let cache = self.cache_1d.borrow();
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };
        if rebuild {
            let lut1d = match key {
                OneDKey::Counting { n_points } => lut::build_lut1d_csr(&self.tth, n_points),
                OneDKey::PixelSplit { n_points } => {
                    lut::build_lut1d_ps_csr(&self.geometry, &self.tth, n_points)
                }
                OneDKey::Supersampled { n_points, n_ss } => {
                    lut::build_lut1d_ss_csr(&self.geometry, &self.tth, n_points, n_ss)
                }
            };
            let mut cache = self.cache_1d.borrow_mut();
            cache.key = Some(key);
            cache.lut = Some(lut1d);
        }
        std::cell::Ref::map(self.cache_1d.borrow(), |cache| cache.lut.as_ref().unwrap())
    }

    /// checks whether current lut_2d was calculated for the same number of points in tth and chi, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_chi` - number of points in chi for the 2d integration
    fn ensure_lut2d(&self, key: TwoDKey) -> std::cell::Ref<'_, lut::Lut2d> {
        let rebuild = {
            let cache = self.cache_2d.borrow();
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };
        if rebuild {
            let lut2d = match key {
                TwoDKey::Counting { n_tth, n_chi } => {
                    lut::build_lut2d_csr(&self.tth, &self.chi, n_tth, n_chi)
                }
                TwoDKey::PixelSplit { n_tth, n_chi } => {
                    lut::build_lut2d_ps_csr(&self.geometry, &self.tth, &self.chi, n_tth, n_chi)
                }
                TwoDKey::Supersampled { n_tth, n_chi, n_ss } => lut::build_lut2d_ss_csr(
                    &self.geometry,
                    &self.tth,
                    &self.chi,
                    n_tth,
                    n_chi,
                    n_ss,
                ),
                TwoDKey::CountingChi { n_tth, n_chi, chi } => {
                    lut::build_lut2d_csr_with_chi_range(
                        &self.tth,
                        &self.chi,
                        n_tth,
                        n_chi,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
                TwoDKey::PixelSplitChi { n_tth, n_chi, chi } => {
                    lut::build_lut2d_ps_csr_with_chi_range(
                        &self.geometry,
                        &self.tth,
                        &self.chi,
                        n_tth,
                        n_chi,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
                TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi } => {
                    lut::build_lut2d_ss_csr_with_chi_range(
                        &self.geometry,
                        &self.tth,
                        &self.chi,
                        n_tth,
                        n_chi,
                        n_ss,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
            };
            let mut cache = self.cache_2d.borrow_mut();
            cache.key = Some(key);
            cache.lut = Some(lut2d);
        }
        std::cell::Ref::map(self.cache_2d.borrow(), |cache| cache.lut.as_ref().unwrap())
    }

    /// Apply polarization and solid angle corrections to the intensity.
    fn correct_intensity(&mut self, data: &Vec<f64>) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, y)| y / self.polarization[i] / self.solid_angle[i])
            .collect::<Vec<f64>>()
    }

    fn correct_intensity_and_variance(&mut self, data: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut intensity = Vec::with_capacity(data.len());
        let mut variance = Vec::with_capacity(data.len());
        for (i, &y) in data.iter().enumerate() {
            let corr = self.polarization[i] * self.solid_angle[i];
            let inv = 1.0 / corr;
            intensity.push(y * inv);
            variance.push(y * inv * inv);
        }
        (intensity, variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math;
    use crate::poni::Poni;
    use crate::util;
    use rayon::prelude::*;

    #[test]
    fn test_integrator() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Histogram)
            .with_polarization_factor(0.99);
        let (tth, int) = integrator.integrate1d(&frame, 850);

        util::save_vec2("pattern.txt", &tth, &int);

        let tth_pyfai = util::read_vec("pattern_tth_pyFAI.txt").unwrap();
        let int_pyfai = util::read_vec("pattern_int_pyFAI.txt").unwrap();

        util::compare_vecs(&tth, &tth_pyfai, "tth", 1e-6);

        let int_diff: f64 = int
            .par_iter()
            .zip(int_pyfai.par_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        let int_diff = int_diff / int.len() as f64;
        assert!(int_diff < math::max(&int) * 1e-4); // 0.001 difference in total
    }

    #[test]
    fn test_integrate_counting_1d() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Counting)
            .with_polarization_factor(0.99);
        let (tth, int) = integrator.integrate1d(&frame, 850);

        util::save_vec2("pattern_lut.txt", &tth, &int);

        let tth_pyfai = util::read_vec("pattern_tth_pyFAI.txt").unwrap();
        let int_pyfai = util::read_vec("pattern_int_pyFAI.txt").unwrap();

        util::compare_vecs(&tth, &tth_pyfai, "tth", 1e-6);

        let int_diff: f64 = int
            .par_iter()
            .zip(int_pyfai.par_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        let int_diff = int_diff / int.len() as f64;
        assert!(int_diff < math::max(&int) * 1e-4); // 0.001 difference in total
    }

    #[test]
    fn test_integrate_counting_1d_with_errors() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Counting)
            .with_polarization_factor(0.99);

        let (tth, int, err) = integrator.integrate1d_with_errors(&frame, 850);
        util::save_vec3("pattern_lut_err.txt", &tth, &int, &err);
    }

    #[test]
    fn test_integrate_pixel_split_1d() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::PixelSplit)
            .with_polarization_factor(0.99);
        let (tth, int) = integrator.integrate1d(&frame, 850);
        util::save_vec2("pattern_lut_ps.txt", &tth, &int);
    }

    #[test]
    fn test_integrate_pixel_split_1d_with_errors() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::PixelSplit)
            .with_polarization_factor(0.99);

        let (tth, int, err) = integrator.integrate1d_with_errors(&frame, 850);
        util::save_vec3("pattern_lut_ps_err.txt", &tth, &int, &err);
    }

    #[test]
    fn test_integrate_supersampled_1d() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator =
            Integrator::new(geometry).with_method(IntegrationMethod::Supersampled { n_ss: 5 });
        let (tth, int) = integrator.integrate1d(&frame, 850);

        util::save_vec2("pattern_lut_ss.txt", &tth, &int);
    }

    #[test]
    fn test_integrate_counting_2d() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Counting)
            .with_polarization_factor(0.99);
        let cake = integrator.integrate2d(&frame, 850, 360);
        util::save_vec("cake_lut.txt", &cake);
    }

    #[test]
    fn test_integrate_supersampled_2d() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Supersampled { n_ss: 5 })
            .with_polarization_factor(0.99);
        let cake = integrator.integrate2d(&frame, 850, 360);
        util::save_vec("cake_lut_ss.txt", &cake);
    }

    #[test]
    fn test_lut_cache_cleared_on_geometry_change() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 1.0;
        let geometry = Geometry::new(2, 2, poni);
        let mut integrator = Integrator::new(geometry).with_method(IntegrationMethod::Counting);

        integrator.update_geometry_arrays(2, 2);
        let _ = integrator.ensure_lut1d(OneDKey::Counting { n_points: 4 });
        let _ = integrator.ensure_lut2d(TwoDKey::Counting { n_tth: 4, n_chi: 4 });

        let cache_1d = integrator.cache_1d.borrow();
        assert!(cache_1d.key.is_some());
        assert!(cache_1d.lut.is_some());
        drop(cache_1d);
        let cache_2d = integrator.cache_2d.borrow();
        assert!(cache_2d.key.is_some());
        assert!(cache_2d.lut.is_some());
        drop(cache_2d);

        integrator.update_geometry_arrays(3, 3);

        let cache_1d = integrator.cache_1d.borrow();
        assert!(cache_1d.key.is_none());
        assert!(cache_1d.lut.is_none());
        drop(cache_1d);
        let cache_2d = integrator.cache_2d.borrow();
        assert!(cache_2d.key.is_none());
        assert!(cache_2d.lut.is_none());
    }

    #[test]
    fn test_integrate_with_different_dimension() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame1 = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let dim1 = frame1.dim1 / 2;
        let dim2 = frame1.dim2 / 2;
        let size = dim1 * dim2;
        let frame2 = Frame {
            name: String::from("test_data/CeO2_Pilatus1M.tif"),
            data: frame1.data[0..size].to_vec(),
            dim1,
            dim2,
        };
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Counting)
            .with_polarization_factor(0.99);

        integrator.integrate1d(&frame1, 850);
        integrator.integrate1d(&frame2, 850);
    }

    #[test]
    fn test_set_and_clear_mask() {
        let poni = Poni::new();
        let geometry = Geometry::new(2, 2, poni);
        let mut integrator = Integrator::new(geometry);
        assert!(integrator.mask().is_none());

        let mask = Mask::from_vec(vec![0, 1, 0, 1], 2, 2).unwrap();
        integrator.set_mask(mask);
        assert!(integrator.mask().is_some());

        integrator.clear_mask();
        assert!(integrator.mask().is_none());
    }

    #[test]
    fn test_with_mask_chain() {
        let poni = Poni::new();
        let geometry = Geometry::new(2, 2, poni);
        let mask_data = vec![1, 0, 0, 1];
        let mask = Mask::from_vec(mask_data.clone(), 2, 2).unwrap();

        let mut integrator = Integrator::new(geometry).with_mask(mask);
        let stored = integrator.mask().unwrap();
        assert_eq!(stored.data, mask_data);

        integrator.clear_mask();
        assert!(integrator.mask().is_none());
    }
}
