#![allow(dead_code)]

use crate::math::get_bin;

use super::frame::Frame;
use super::geometry::Geometry;
use rayon::prelude::*;

use super::math::{divide, get_hist_min_max_step, histogram, histogram_bins};

pub struct Integrator {
    pub geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
    csr: Vec<Vec<usize>>,
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
        }
    }

    /// checks whether current tth, chi, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated
    fn update_geometry_arrays(&mut self, dim1: usize, dim2: usize) {
        if self.geometry.dim1 != dim1 || self.geometry.dim2 != dim2 {
            self.geometry.dim1 = dim1;
            self.geometry.dim2 = dim2;
            self.tth = self.geometry.calc_tth();
            self.chi = self.geometry.calc_chi();
            self.solid_angle = self.geometry.calc_solid_angle();
            self.polarization = self.geometry.calc_polarization(0.99, &self.tth, &self.chi);
        }
    }

    /// checks whether current csr was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    fn update_csr(&mut self, n_points: usize) {
        if self.csr.len() != n_points {
            self.csr = self.calc_csr(n_points);
        }
    }

    /// Calculates a CSR implementation
    /// This is a vector with length n_points (same as the output pattern). Each element of this vector is
    /// a vector of indices of the pixels that contribute to the pattern point.
    fn calc_csr(&self, n_points: usize) -> Vec<Vec<usize>> {
        let mut csr: Vec<Vec<usize>> = vec![Vec::new(); n_points];

        let (min, _max, step) = get_hist_min_max_step(&self.tth, n_points);

        let index: Vec<usize> = self
            .tth
            .par_iter()
            .map(|tth| get_bin(*tth, min, step))
            .collect();

        for i in 0..self.tth.len() {
            csr[index[i]].push(i);
        }
        csr
    }

    /// Integrate a frame using CSR method
    pub fn integrate_csr(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_csr(n_points);

        let (min, _max, step) = get_hist_min_max_step(&self.tth, n_points);
        let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

        let intensity = self.correct_intensity(&frame.data);

        // summing all saved pixel indices in the csr for each tth
        for i in 0..n_points {
            pattern_intensity[i] += self.csr[i].iter().map(|j| intensity[*j]).sum::<f64>();
        }

        // obtain the length of each tth vector in the csr
        let csr_len = self
            .csr
            .par_iter()
            .map(|v| v.len() as f64)
            .collect::<Vec<f64>>();
        // divide the summed intensity by the length of each tth vector in the csr for normalization
        let pattern_intensity: Vec<f64> = divide(&pattern_intensity, &csr_len);

        let pattern_tth = histogram_bins(n_points, min, step);

        (pattern_tth, pattern_intensity)
    }

    /// Integration of the frame using a simple histogram method
    pub fn integrate(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        histogram(&self.tth, &intensity, n_points)
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
}
