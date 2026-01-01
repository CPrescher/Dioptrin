#![allow(dead_code)]
use crate::lut;
use crate::math::histogram;

use super::frame::Frame;
use super::geometry::Geometry;

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

/// LUT method selection (applies to both 1D and 2D integrations).
///
/// Examples
/// ```no_run
/// use dioptrin::integration::{Integrator, LutMethod};
/// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
/// # let poni: Poni = unimplemented!();
/// # let frame: Frame = unimplemented!();
/// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
/// let mut integrator = Integrator::new(geometry);
/// let (_x, y) = integrator.integrate_lut_with(&frame, 1250, LutMethod::Counting);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum LutMethod {
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
        }
    }

    /// Integrate a frame to 1D using the selected LUT method
    ///
    /// Examples
    /// ```no_run
    /// use dioptrin::integration::{Integrator, LutMethod};
    /// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
    /// # let poni: Poni = unimplemented!();
    /// # let frame: Frame = unimplemented!();
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry);
    /// let (_x, y_count) = integrator.integrate_lut_with(&frame, 1250, LutMethod::Counting);
    /// let (_x, y_ps) = integrator.integrate_lut_with(&frame, 1250, LutMethod::PixelSplit);
    /// let (_x, y_ss) = integrator.integrate_lut_with(&frame, 1250, LutMethod::Supersampled { n_ss: 4 });
    /// ```
    pub fn integrate_lut_with(
        &mut self,
        frame: &Frame,
        n_points: usize,
        method: LutMethod,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut1d = match method {
            LutMethod::Counting => self.ensure_lut1d(OneDKey::Counting { n_points }),
            LutMethod::PixelSplit => self.ensure_lut1d(OneDKey::PixelSplit { n_points }),
            LutMethod::Supersampled { n_ss } => {
                self.ensure_lut1d(OneDKey::Supersampled { n_points, n_ss })
            }
        };
        crate::lut::integrate_lut1d(&intensity, lut1d)
    }

    /// Integrate a frame to 2D using the selected LUT method
    ///
    /// Example
    /// ```no_run
    /// use dioptrin::integration::{Integrator, LutMethod};
    /// # use dioptrin::{geometry::Geometry, frame::Frame, poni::Poni};
    /// # let poni: Poni = unimplemented!();
    /// # let frame: Frame = unimplemented!();
    /// let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
    /// let mut integrator = Integrator::new(geometry);
    /// let y = integrator.integrate2d_lut_with(&frame, 1250, 360, LutMethod::PixelSplit);
    /// assert_eq!(y.len(), 1250*360);
    /// ```
    pub fn integrate2d_lut_with(
        &mut self,
        frame: &Frame,
        n_tth: usize,
        n_azi: usize,
        method: LutMethod,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut2d = match method {
            LutMethod::Counting => self.ensure_lut2d(TwoDKey::Counting { n_tth, n_azi }),
            LutMethod::PixelSplit => self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_azi }),
            LutMethod::Supersampled { n_ss } => {
                self.ensure_lut2d(TwoDKey::Supersampled { n_tth, n_azi, n_ss })
            }
        };
        crate::lut::integrate_lut2d(&intensity, lut2d)
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
        lut::integrate_lut1d(&intensity, lut1d)
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
        lut::integrate_lut1d(&intensity, lut1d)
    }

    /// Integrate a frame using pixel-splitting method (1D)
    pub fn integrate_lut_ps(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut1d = self.ensure_lut1d(OneDKey::PixelSplit { n_points });
        lut::integrate_lut1d(&intensity, lut1d)
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
        lut::integrate_lut2d(&intensity, lut2d)
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
        lut::integrate_lut2d(&intensity, lut2d)
    }

    /// Integrate a frame into a 2D grid using pixel-splitting method
    pub fn integrate2d_lut_ps(&mut self, frame: &Frame, n_tth: usize, n_azi: usize) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        let lut2d = self.ensure_lut2d(TwoDKey::PixelSplit { n_tth, n_azi });
        lut::integrate_lut2d(&intensity, lut2d)
    }

    /// Integration of the frame using a simple histogram method
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    pub fn integrate(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        histogram::histogram(&self.tth, &intensity, n_points)
    }

    /// Apply polarization and solid angle corrections to the intensity
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
        let (tth, int) = integrator.integrate(&frame, 850);

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
