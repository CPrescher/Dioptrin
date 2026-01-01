#![allow(dead_code)]
use crate::lut;
use crate::math::histogram;

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
        n_azi: usize,
    },
    PixelSplit {
        n_tth: usize,
        n_azi: usize,
    },
    Supersampled {
        n_tth: usize,
        n_azi: usize,
        n_ss: usize,
    },
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
/// ```no_run
/// use dioptrin::integration::{Integrator, IntegrationMethod};
/// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
/// # let poni: Poni = unimplemented!();
/// # let frame: Frame = unimplemented!();
/// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
/// let mut integrator = Integrator::new(geometry);
/// integrator.set_method(IntegrationMethod::Counting);
/// let (_x, y) = integrator.integrate1d(&frame, 1250);
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
    pub geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    cache_1d: Lut1dCache,
    cache_2d: Lut2dCache,
    mask: Option<Mask>,
    method: IntegrationMethod,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            tth: vec![],
            chi: vec![],
            solid_angle: vec![],
            polarization: vec![],
            cache_1d: Lut1dCache::new(),
            cache_2d: Lut2dCache::new(),
            mask: None,
            method: IntegrationMethod::PixelSplit,
        }
    }

    pub fn set_method(&mut self, method: IntegrationMethod) {
        self.method = method;
    }

    pub fn set_mask(&mut self, mask: Mask) {
        self.mask = Some(mask);
    }

    pub fn clear_mask(&mut self) {
        self.mask = None;
    }

    /// Integrate a frame to 1D using the configured integration method.
    ///
    /// Examples
    /// ```no_run
    /// use dioptrin::integration::{Integrator, IntegrationMethod};
    /// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
    /// # let poni: Poni = unimplemented!();
    /// # let frame: Frame = unimplemented!();
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry);
    /// integrator.set_method(IntegrationMethod::Counting);
    /// let (_x, y_count) = integrator.integrate1d(&frame, 1250);
    /// integrator.set_method(IntegrationMethod::PixelSplit);
    /// let (_x, y_ps) = integrator.integrate1d(&frame, 1250);
    /// integrator.set_method(IntegrationMethod::Supersampled { n_ss: 4 });
    /// let (_x, y_ss) = integrator.integrate1d(&frame, 1250);
    /// ```
    pub fn integrate1d(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask = self.mask.take();
        let mask_data = mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_points, mask_data)
            }
            IntegrationMethod::Counting => {
                let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
                crate::lut::integrate1d(&intensity, lut1d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
                crate::lut::integrate1d(&intensity, lut1d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
                crate::lut::integrate1d(&intensity, lut1d, mask_data)
            }
        };
        self.mask = mask;
        result
    }

    /// Integrate a frame to 2D using the configured integration method.
    ///
    /// Example
    /// ```no_run
    /// use dioptrin::integration::{Integrator, IntegrationMethod};
    /// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
    /// # let poni: Poni = unimplemented!();
    /// # let frame: Frame = unimplemented!();
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry);
    /// integrator.set_method(IntegrationMethod::PixelSplit);
    /// let y = integrator.integrate2d(&frame, 1250, 360);
    /// assert_eq!(y.len(), 1250*360);
    /// ```
    pub fn integrate2d(&mut self, frame: &Frame, n_tth: usize, n_azi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let mask = self.mask.take();
        let mask_data = mask.as_ref().map(|mask| {
            mask.assert_compatible(frame.dim1, frame.dim2);
            &mask.data
        });
        let result = match self.method {
            IntegrationMethod::Histogram => {
                histogram::histogram(&self.tth, &intensity, n_tth, mask_data).1
            }
            IntegrationMethod::Counting => {
                let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_azi });
                crate::lut::integrate2d(&intensity, lut2d, mask_data)
            }
            IntegrationMethod::PixelSplit => {
                let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_azi });
                crate::lut::integrate2d(&intensity, lut2d, mask_data)
            }
            IntegrationMethod::Supersampled { n_ss } => {
                let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_azi, n_ss });
                crate::lut::integrate2d(&intensity, lut2d, mask_data)
            }
        };
        self.mask = mask;
        result
    }

    /// checks whether current tth, chi, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated
    fn update_geometry_arrays(&mut self, dim1: usize, dim2: usize) {
        if self.geometry.dim1 != dim1 || self.geometry.dim2 != dim2 {
            self.geometry.dim1 = dim1;
            self.geometry.dim2 = dim2;
            self.tth = self.geometry.calc_tth_all();
            self.chi = self.geometry.calc_chi_all();
            self.solid_angle = self.geometry.calc_solid_angle();
            self.polarization = self.geometry.calc_polarization(0.99, &self.tth, &self.chi);
        }
    }

    /// checks whether current lut was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    fn ensure_lut1d(&mut self, key: OneDKey) -> &lut::Lut1d {
        let rebuild = match &self.cache_1d.key {
            None => true,
            Some(k) => k != &key,
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
            self.cache_1d.key = Some(key);
            self.cache_1d.lut = Some(lut1d);
        }
        self.cache_1d.lut.as_ref().unwrap()
    }

    /// checks whether current supersampled lookup table (LUT) was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_points` - number of points for the 1d integration
    /// * `n_ss` - super samplling factor,  number of subpixels in each direction for a pixel
    fn update_lut_ss(&mut self, n_points: usize, n_ss: usize) {
        let _ = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
    }

    /// checks whether current pixel-split lut was calculated for the same number of points
    fn update_lut_ps(&mut self, n_points: usize) {
        let _ = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
    }

    /// checks whether current lut_2d was calculated for the same number of points in tth and azimuth, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_azimuth` - number of points in azimuth for the 2d integration
    fn ensure_lut2d(&mut self, key: TwoDKey) -> &lut::Lut2d {
        let rebuild = match &self.cache_2d.key {
            None => true,
            Some(k) => k != &key,
        };
        if rebuild {
            let lut2d = match key {
                TwoDKey::Counting { n_tth, n_azi } => {
                    lut::build_lut2d_csr(&self.tth, &self.chi, n_tth, n_azi)
                }
                TwoDKey::PixelSplit { n_tth, n_azi } => {
                    lut::build_lut2d_ps_csr(&self.geometry, &self.tth, &self.chi, n_tth, n_azi)
                }
                TwoDKey::Supersampled { n_tth, n_azi, n_ss } => lut::build_lut2d_ss_csr(
                    &self.geometry,
                    &self.tth,
                    &self.chi,
                    n_tth,
                    n_azi,
                    n_ss,
                ),
            };
            self.cache_2d.key = Some(key);
            self.cache_2d.lut = Some(lut2d);
        }
        self.cache_2d.lut.as_ref().unwrap()
    }

    /// checks whether current supersampled lut_2d was calculated for the same number of points in tth and azimuth, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_azimuth` - number of points in azimuth for the 2d integration
    /// * `n_ss` - super samplling factor,  number of subpixels in each direction for a pixel
    fn update_lut_2d(&mut self, n_tth: usize, n_azi: usize) {
        let _ = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_azi });
    }
    fn update_lut_2d_ss(&mut self, n_tth: usize, n_azi: usize, n_ss: usize) {
        let _ = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_azi, n_ss });
    }

    /// checks whether current pixel-split 2d lut was calculated for the same points
    fn update_lut_2d_ps(&mut self, n_tth: usize, n_azi: usize) {
        let _ = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_azi });
    }

    /// Integrate a frame using look up table method
    ///
    /// # Arguments
    /// * `frame` - The frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    pub fn integrate_lut(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut1d = self.ensure_lut1d(OneDKey::Counting { n_points });
        lut::integrate1d(&intensity, lut1d, None)
    }

    /// Integrate a frame using lookup table (LUT) method with supersampling
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    /// * `n_ss` - Number of subpixels per pixel (e.g. 2 means 2x2 subpixels per pixel)
    pub fn integrate_lut_ss(
        &mut self,
        frame: &Frame,
        n_points: usize,
        n_ss: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut1d = self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss });
        lut::integrate1d(&intensity, lut1d, None)
    }

    /// Integrate a frame using pixel-splitting method (1D)
    pub fn integrate_lut_ps(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
        lut::integrate1d(&intensity, lut1d, None)
    }

    /// Ingrating a frame into a 2D grid (cake) with tth and azimuthal angle as axes using
    /// lookup table (LUT) method
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_tth` - Number of points in tth direction
    /// * `n_azi` - Number of points in azimuthal angle direction
    pub fn integrate_2d_lut(&mut self, frame: &Frame, n_tth: usize, n_azi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut2d = self.ensure_lut2d(TwoDKey::Counting { n_tth, n_azi });
        lut::integrate2d(&intensity, lut2d, None)
    }

    // Ingrating a frame into a 2D grid (cake) with tth and azimuthal angle as axes using
    /// lookup table (LUT) method with supersampling
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_tth` - Number of points in tth direction
    /// * `n_azi` - Number of points in azimuthal angle direction
    /// * `n_ss` - Number of subpixels per pixel (e.g. 2 means 2x2 subpixels per pixel)
    pub fn integrate2d_lut_ss(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_azi: usize,
        n_ss: usize,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut2d = self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_azi, n_ss });
        lut::integrate2d(&intensity, lut2d, None)
    }

    /// Integrate a frame into a 2D grid using pixel-splitting method
    pub fn integrate2d_lut_ps(&mut self, frame: &Frame, n_tth: usize, n_azi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_azi });
        lut::integrate2d(&intensity, lut2d, None)
    }

    /// Apply polarization and solid angle corrections to the intensity.
    fn correct_intensity(&mut self, data: &Vec<f64>) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, y)| y / self.polarization[i] / self.solid_angle[i])
            .collect::<Vec<f64>>()
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
        let mut integrator = Integrator::new(geometry);
        integrator.set_method(IntegrationMethod::Histogram);
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
    fn test_integrate_lut() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_lut(&frame, 850);

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
    fn test_integrate_pixel_split_lut() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_lut_ps(&frame, 850);
        util::save_vec2("pattern_lut_ps.txt", &tth, &int);
    }

    #[test]
    fn test_integrate_supersampled_lut() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_lut_ss(&frame, 850, 5);

        util::save_vec2("pattern_lut_ss.txt", &tth, &int);
    }

    #[test]
    fn test_integrate_2d_lut() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let cake = integrator.integrate_2d_lut(&frame, 850, 360);
        util::save_vec("cake_lut.txt", &cake);
    }

    #[test]
    fn test_integrate_2d_lut_ss() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let cake = integrator.integrate2d_lut_ss(&frame, 850, 360, 5);
        util::save_vec("cake_lut_ss.txt", &cake);
    }
}
