#![allow(dead_code)]
use std::collections::HashMap;

use crate::lut;
use crate::math::histogram;
#[cfg(feature = "lut_metrics")]
use crate::lut::metrics::{
    analyze_lut2d_locality, analyze_lut2d_ss_locality, analyze_lut_locality, analyze_lut_ss_locality,
};

use super::frame::Frame;
use super::geometry::Geometry;

pub struct Integrator {
    pub geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    lut: Vec<Vec<usize>>,                     // lookup table for 1d integration
    lut_ss: Vec<HashMap<usize, f64>>,         // supersampled lookup table for 1d integration
    lut_ss_sub_pixel: usize, // number of subpixels for supersampling for 1d integration
    lut_2d: Vec<Vec<Vec<usize>>>, // lookup table for 2d integration
    lut_2d_ss: Vec<Vec<HashMap<usize, f64>>>, // supersampled lookup table for 2d integration
    lust_2d_ss_sub_pixel: usize, // number of subpixels for supersampling for 2d integraion
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            tth: vec![],
            chi: vec![],
            solid_angle: vec![],
            polarization: vec![],
            lut: vec![],
            lut_ss: vec![],
            lut_ss_sub_pixel: 0,
            lut_2d: vec![],
            lut_2d_ss: vec![],
            lust_2d_ss_sub_pixel: 0,
        }
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
    fn update_lut(&mut self, n_points: usize) {
        if self.lut.len() != n_points {
            self.lut = lut::lut(&self.tth, n_points);
            #[cfg(feature = "lut_metrics")]
            {
                let stats = analyze_lut_locality(&self.lut);
                println!(
                    "[lut_metrics] 1D LUT: bins={} empty={} avg_stride={:.2} avg_cache_lines={:.2}",
                    stats.total_bins, stats.empty_bins, stats.avg_stride, stats.avg_cache_lines
                );
            }
        }
    }

    /// checks whether current supersampled lookup table (LUT) was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_points` - number of points for the 1d integration
    /// * `n_ss` - super samplling factor,  number of subpixels in each direction for a pixel
    fn update_lut_ss(&mut self, n_points: usize, n_ss: usize) {
        if self.lut_ss.len() != n_points || self.lut_ss_sub_pixel != n_ss {
            self.lut_ss = lut::lut_ss(&self.geometry, &self.tth, n_points, n_ss);
            self.lut_ss_sub_pixel = n_ss;
            #[cfg(feature = "lut_metrics")]
            {
                let stats = analyze_lut_ss_locality(&self.lut_ss);
                println!(
                    "[lut_metrics] 1D LUT SS: bins={} empty={} avg_stride={:.2} avg_cache_lines={:.2}",
                    stats.total_bins, stats.empty_bins, stats.avg_stride, stats.avg_cache_lines
                );
            }
        }
    }

    /// checks whether current lut_2d was calculated for the same number of points in tth and azimuth, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_azimuth` - number of points in azimuth for the 2d integration
    fn update_lut_2d(&mut self, n_tth: usize, n_azi: usize) {
        if self.lut_2d.len() != n_tth || self.lut_2d[0].len() != n_azi {
            self.lut_2d = lut::lut_2d(&self.tth, &self.chi, n_tth, n_azi);
            #[cfg(feature = "lut_metrics")]
            {
                let stats = analyze_lut2d_locality(&self.lut_2d);
                println!(
                    "[lut_metrics] 2D LUT: bins={} empty={} avg_stride={:.2} avg_cache_lines={:.2}",
                    stats.total_bins, stats.empty_bins, stats.avg_stride, stats.avg_cache_lines
                );
            }
        }
    }

    /// checks whether current supersampled lut_2d was calculated for the same number of points in tth and azimuth, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_azimuth` - number of points in azimuth for the 2d integration
    /// * `n_ss` - super samplling factor,  number of subpixels in each direction for a pixel
    fn update_lut_2d_ss(&mut self, n_tth: usize, n_azi: usize, n_ss: usize) {
        if self.lut_2d_ss.len() != n_tth
            || self.lut_2d_ss[0].len() != n_azi
            || self.lust_2d_ss_sub_pixel != n_ss
        {
            self.lut_2d_ss =
                lut::lut_2d_ss(&self.geometry, &self.tth, &self.chi, n_tth, n_azi, n_ss);
            self.lust_2d_ss_sub_pixel = n_ss;
            #[cfg(feature = "lut_metrics")]
            {
                let stats = analyze_lut2d_ss_locality(&self.lut_2d_ss);
                println!(
                    "[lut_metrics] 2D LUT SS: bins={} empty={} avg_stride={:.2} avg_cache_lines={:.2}",
                    stats.total_bins, stats.empty_bins, stats.avg_stride, stats.avg_cache_lines
                );
            }
        }
    }

    /// Integrate a frame using look up table method
    ///
    /// # Arguments
    /// * `frame` - The frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    pub fn integrate_lut(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_lut(n_points);

        let intensity = self.correct_intensity(&frame.data);

        let (hist_min, _max, hist_step) = histogram::min_max_step(&self.tth, n_points);

        lut::integrate(&intensity, &self.lut, hist_min, hist_step)
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
        self.update_lut_ss(n_points, n_ss);
        let intensity = self.correct_intensity(&frame.data);

        let (hist_min, _hist_max, hist_step) =
            histogram::ss_tth_min_max_step(&self.geometry, &self.tth, n_points, n_ss);

        lut::integrate_ss(&intensity, &self.lut_ss, hist_min, hist_step)
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
        self.update_lut_2d(n_tth, n_azi);
        let intensity = self.correct_intensity(&frame.data);

        lut::integrate2d(&intensity, &self.lut_2d)
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
        self.update_lut_2d_ss(n_tth, n_azi, n_ss);
        let intensity = self.correct_intensity(&frame.data);

        lut::integrate2d_ss(&intensity, &self.lut_2d_ss)
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

        util::save_vec("pattern_tth.txt", &tth);
        util::save_vec("pattern_int.txt", &int);

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

        util::save_vec("pattern_tth_lut.txt", &tth);
        util::save_vec("pattern_int_lut.txt", &int);

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
    fn test_integrate_supersampled_lut() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_lut_ss(&frame, 850, 3);

        util::save_vec("pattern_tth_lut_ss.txt", &tth);
        util::save_vec("pattern_int_lut_ss.txt", &int);
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
