#![allow(dead_code)]
use std::collections::HashMap;

use crate::{csr, histogram};

use super::frame::Frame;
use super::geometry::Geometry;

pub struct Integrator {
    pub geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    csr: Vec<Vec<usize>>,
    csr_supersampled: Vec<HashMap<usize, f64>>,
    csr_supersampled_sub_pixel: usize,
    csr2d: Vec<Vec<Vec<usize>>>,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            tth: vec![],
            chi: vec![],
            solid_angle: vec![],
            polarization: vec![],
            csr: vec![],
            csr_supersampled: vec![],
            csr_supersampled_sub_pixel: 0,
            csr2d: vec![],
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

    /// checks whether current csr was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    fn update_csr(&mut self, n_points: usize) {
        if self.csr.len() != n_points {
            self.csr = csr::csr(&self.tth, n_points);
        }
    }

    /// checks whether current supersampled csr was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    fn update_csr_supersampled(&mut self, n_points: usize, n_subpixels: usize) {
        if self.csr_supersampled.len() != n_points || self.csr_supersampled_sub_pixel != n_subpixels
        {
            self.csr_supersampled =
                csr::csr_supersampled(&self.geometry, &self.tth, n_points, n_subpixels);
            self.csr_supersampled_sub_pixel = n_subpixels;
        }
    }

    /// checks whether current csr2d was calculated for the same number of points in tth and azimuth, if not,
    /// it is recalculated
    fn update_csr2d(&mut self, n_points_tth: usize, n_points_azi: usize) {
        if self.csr2d.len() != n_points_tth || self.csr2d[0].len() != n_points_azi {
            self.csr2d = csr::csr2d(&self.tth, &self.chi, n_points_tth, n_points_azi);
        }
    }

    /// Integrate a frame using CSR method
    ///
    /// # Arguments
    /// * `frame` - The frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    pub fn integrate_csr(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_csr(n_points);

        let intensity = self.correct_intensity(&frame.data);

        let (hist_min, _max, hist_step) = histogram::min_max_step(&self.tth, n_points);

        csr::integrate(&intensity, &self.csr, hist_min, hist_step)
    }

    /// Integrate a frame using CSR method with supersampling
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_points` - the number of points of the resulting pattern
    /// * `n_subpixels` - Number of subpixels per pixel (e.g. 2 means 2x2 subpixels per pixel)
    pub fn integrate_csr_supersampled(
        &mut self,
        frame: &Frame,
        n_points: usize,
        n_subpixels: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_csr_supersampled(n_points, n_subpixels);
        let intensity = self.correct_intensity(&frame.data);

        let (hist_min, _hist_max, hist_step) =
            histogram::supersampled_min_max_step(&self.geometry, &self.tth, n_points, n_subpixels);

        csr::integrate_supersampled(&intensity, &self.csr_supersampled, hist_min, hist_step)
    }

    /// Ingrating a frame into a 2D grid (cake) with tth and azimuthal angle as axes using
    /// CSR method
    ///
    /// # Arguments
    /// * `frame` - Frame to integrate
    /// * `n_points_tth` - Number of points in tth direction
    /// * `n_points_azi` - Number of points in azimuthal angle direction
    pub fn integrate2d_csr(
        &mut self,
        frame: &Frame,
        n_points_tth: usize,
        n_points_azi: usize,
    ) -> Vec<f64> {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_csr2d(n_points_tth, n_points_azi);
        let intensity = self.correct_intensity(&frame.data);

        csr::integrate2d(&intensity, &self.csr2d)
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
    use rayon::prelude::*;
    #[path = "../../util.rs"]
    mod util;

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
    fn test_integrate_csr() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_csr(&frame, 850);

        util::save_vec("pattern_tth_csr.txt", &tth);
        util::save_vec("pattern_int_csr.txt", &int);

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
    fn test_integrate_supersampled_csr() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_csr_supersampled(&frame, 850, 3);

        util::save_vec("pattern_tth_csr_ss.txt", &tth);
        util::save_vec("pattern_int_csr_ss.txt", &int);
    }

    #[test]
    fn test_integrate2d_csr() {
        util::download_pilatus_test_file();
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let cake = integrator.integrate2d_csr(&frame, 850, 360);
        util::save_vec("cake_csr.txt", &cake);
    }
}
