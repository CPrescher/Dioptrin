//! Geometry calculations for each pixel in the detector
//!
#![allow(dead_code)]

use crate::poni::Poni;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Geometry {
    pub dim1: usize,
    pub dim2: usize,
    pub poni: Poni,
}

impl Geometry {
    pub fn new(dim1: usize, dim2: usize, poni: Poni) -> Geometry {
        Geometry { dim1, dim2, poni }
    }

    pub fn size(&self) -> usize {
        self.dim1 * self.dim2
    }

    /// Returns pixel corner coordinates (d1, d2) for the full detector grid.
    pub fn corner_grid(&self) -> (Vec<f64>, Vec<f64>) {
        let count = (self.dim1 + 1) * (self.dim2 + 1);
        let mut d1 = Vec::with_capacity(count);
        let mut d2 = Vec::with_capacity(count);

        for i in 0..=self.dim1 {
            for j in 0..=self.dim2 {
                d1.push(i as f64);
                d2.push(j as f64);
            }
        }

        (d1, d2)
    }

    /// Calculates the pixel positions in the detector reference coordinate system
    /// whereby the lower left corner of the detector is (0, 0). return coordinates
    /// are in cartesian coordinates. Be aware that y and x are swapped in this framework.
    /// This results in | being the d1 direction and _ being the d2 direction
    ///
    /// Arguments:
    ///   * `d1` - pixel positions in Y image (|) direction
    ///   * `d2` - pixel positions in X image (_) direction
    fn calc_pixel_position_det_reference(
        &self,
        d1: &[f64],
        d2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        assert!(d1.len() == d2.len());

        let pixel1 = self.poni.pixel1;
        let pixel2 = self.poni.pixel2;

        let mut d1_cart = Vec::with_capacity(d1.len());
        let mut d2_cart = Vec::with_capacity(d1.len());

        for i in 0..d1.len() {
            d1_cart.push(d1[i] * pixel1);
            d2_cart.push(d2[i] * pixel2);
        }
        (d1_cart, d2_cart)
    }

    /// Calculates the pixel center positions of all pixels in the detector reference
    /// coordinate system/ return coordinates are in cartesian coordinates. Be aware
    /// that y and x are swapped in this framework. This results in | being the d1
    /// direction and _ being the d2 direction
    fn calc_pixel_position_det_reference_all(&self) -> (Vec<f64>, Vec<f64>) {
        let mut d1_cart = Vec::with_capacity(self.size());
        let mut d2_cart = Vec::with_capacity(self.size());

        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                d1_cart.push((i as f64 + 0.5) * self.poni.pixel1);
                d2_cart.push((j as f64 + 0.5) * self.poni.pixel2);
            }
        }

        (d1_cart, d2_cart)
    }

    /// Calculates the pixel center positions of all pixels in the PONI reference
    /// coordinate system. return coordinates are in cartesian coordinates. Be aware
    /// that y and x are swapped in this framework. This results in | being the d1 (first)
    /// direction and _ being the d2 (second) direction.
    fn calc_pixel_positions_with_poni_reference_all(&self) -> (Vec<f64>, Vec<f64>) {
        let size = self.size();
        let (d1_cart, d2_cart) = self.calc_pixel_position_det_reference_all();

        let mut d1_cart_poni = Vec::with_capacity(size);
        let mut d2_cart_poni = Vec::with_capacity(size);

        let poni1 = self.poni.poni1;
        let poni2 = self.poni.poni2;

        for i in 0..size {
            d1_cart_poni.push(d1_cart[i] - poni1);
            d2_cart_poni.push(d2_cart[i] - poni2);
        }
        (d1_cart_poni, d2_cart_poni)
    }

    /// Converts detector referenced coordinates into the PONI reference.
    /// Both arguments need to be of the same length. The detector referenced coordinates could come
    /// from the calc_pixel_position_det_reference() function.
    ///
    /// Arguments:
    ///  * `d1` - d1 in Y image (|) direction with detector reference in meters
    ///  * `d2` - d2 in X image (_) direction with detector reference in meters
    fn calc_pixel_positions_with_poni_reference(
        &self,
        d1: &[f64],
        d2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        assert!(d1.len() == d2.len());

        let mut d1_cart_poni = Vec::with_capacity(d1.len());
        let mut d2_cart_poni = Vec::with_capacity(d2.len());

        let poni1 = self.poni.poni1;
        let poni2 = self.poni.poni2;

        for i in 0..d1.len() {
            d1_cart_poni.push(d1[i] - poni1);
            d2_cart_poni.push(d2[i] - poni2);
        }
        (d1_cart_poni, d2_cart_poni)
    }

    /// Calculates all pixel center positions in the sample reference coordinate system
    fn calc_pixel_positions_with_sample_reference_all(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (d1_cart_poni, d2_cart_poni) = self.calc_pixel_positions_with_poni_reference_all();
        let tc = math::TrigonometricConstants::new(&self.poni);

        let x_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(&d2_cart_poni)
            .map(|(d1, d2)| math::f_x1(d1, d2, &self.poni.distance, &tc))
            .collect();

        let y_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(&d2_cart_poni)
            .map(|(d1, d2)| math::f_x2(d1, d2, &self.poni.distance, &tc))
            .collect();

        let z_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(&d2_cart_poni)
            .map(|(d1, d2)| math::f_x3(d1, d2, &self.poni.distance, &tc))
            .collect();

        (x_sample, y_sample, z_sample)
    }

    /// Calculates the x, y, z positions in sample reference of given the coordinates in poni reference.
    ///
    /// Arguments:
    ///   * `d1_cart_poni` - d1 in Y image (|) direction with poni reference in meters
    ///   * `d2_cart_poni` - d2 in X image (_) direction with poni reference in meters
    fn calc_pixel_positions_with_sample_reference(
        &self,
        d1_cart_poni: &[f64],
        d2_cart_poni: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let tc = math::TrigonometricConstants::new(&self.poni);

        let x_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(d2_cart_poni)
            .map(|(d1, d2)| math::f_x1(d1, d2, &self.poni.distance, &tc))
            .collect();

        let y_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(d2_cart_poni)
            .map(|(d1, d2)| math::f_x2(d1, d2, &self.poni.distance, &tc))
            .collect();

        let z_sample: Vec<f64> = d1_cart_poni
            .par_iter()
            .zip(d2_cart_poni)
            .map(|(d1, d2)| math::f_x3(d1, d2, &self.poni.distance, &tc))
            .collect();

        (x_sample, y_sample, z_sample)
    }

    /// Calculates the two theta angle in radians for each pixel in the detector
    pub fn calc_tth_all(&self) -> Vec<f64> {
        let (x_sample, y_sample, z_sample) = self.calc_pixel_positions_with_sample_reference_all();
        let mut tth = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            tth.push(math::f_tth(&x_sample[i], &y_sample[i], &z_sample[i]));
        }
        tth
    }

    /// Calculates the two theta angle in radians for given pixel coordinates.
    ///
    /// Arguments:
    ///   * `d1` - pixel positions in Y image (|) direction
    ///   * `d2` - pixel positions in X image (_) direction
    pub fn calc_tth(&self, d1: &[f64], d2: &[f64]) -> Vec<f64> {
        let (d1_cart, d2_cart) = self.calc_pixel_position_det_reference(d1, d2);
        let (d1_cart_poni, d2_cart_poni) =
            self.calc_pixel_positions_with_poni_reference(&d1_cart, &d2_cart);
        let (x_sample, y_sample, z_sample) =
            self.calc_pixel_positions_with_sample_reference(&d1_cart_poni, &d2_cart_poni);

        x_sample
            .par_iter()
            .zip(&y_sample)
            .zip(&z_sample)
            .map(|((x, y), z)| math::f_tth(x, y, z))
            .collect()
    }

    /// Calculates the azimuthal angle in radians for each pixel in the detector
    pub fn calc_azimuth_all(&self) -> Vec<f64> {
        let (x_sample, y_sample, _z_sample) = self.calc_pixel_positions_with_sample_reference_all();
        let azimuth = x_sample
            .par_iter()
            .zip(&y_sample)
            .map(|(x, y)| math::f_azimuth(x, y))
            .collect();
        azimuth
    }

    /// Calculates the azimuthal angle in radians for given pixel coordinates.
    ///
    /// Arguments:
    ///   * `d1` - pixel positions in Y image (|) direction
    ///   * `d2` - pixel positions in X image (_) direction
    pub fn calc_azimuth(&self, d1: &[f64], d2: &[f64]) -> Vec<f64> {
        let (d1_cart, d2_cart) = self.calc_pixel_position_det_reference(d1, d2);
        let (d1_cart_poni, d2_cart_poni) =
            self.calc_pixel_positions_with_poni_reference(&d1_cart, &d2_cart);
        let (x_sample, y_sample, _z_sample) =
            self.calc_pixel_positions_with_sample_reference(&d1_cart_poni, &d2_cart_poni);

        x_sample
            .par_iter()
            .zip(&y_sample)
            .map(|(x, y)| math::f_azimuth(x, y))
            .collect()
    }

    /// Calculates two theta and azimuthal angle in radians for each pixel in the detector.
    /// Combined function to reduce overhead for calculating each time the transform into sample reference geometry.
    pub fn calc_tth_and_azimuth(&self, d1: &[f64], d2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (d1_cart, d2_cart) = self.calc_pixel_position_det_reference(d1, d2);
        let (d1_cart_poni, d2_cart_poni) =
            self.calc_pixel_positions_with_poni_reference(&d1_cart, &d2_cart);
        let (x_sample, y_sample, z_sample) =
            self.calc_pixel_positions_with_sample_reference(&d1_cart_poni, &d2_cart_poni);

        let tth = x_sample
            .par_iter()
            .zip(&y_sample)
            .zip(&z_sample)
            .map(|((x, y), z)| math::f_tth(x, y, z))
            .collect();

        let azimuth = x_sample
            .par_iter()
            .zip(&y_sample)
            .map(|(x, y)| math::f_azimuth(x, y))
            .collect();

        (tth, azimuth)
    }

    /// Calculates Solid Angle for each pixel in the detector
    pub fn calc_solid_angle(&self) -> Vec<f64> {
        let (d1_cart_poni, d2_cart_poni) = self.calc_pixel_positions_with_poni_reference_all();
        let solid_angle = d1_cart_poni
            .par_iter()
            .zip(&d2_cart_poni)
            .map(|(d1, d2)| math::f_solid_angle(d1, d2, &self.poni.distance))
            .collect::<Vec<f64>>();
        solid_angle
    }

    /// Calculates polarization for each pixel in the detector
    ///
    /// Arguments:
    ///  * `factor` - polarization factor (value between -1 and 1)
    ///  * `tth` - two theta array in radians
    ///  * `azimuth` - azimuth array in radians
    pub fn calc_polarization(&self, factor: f64, tth: &[f64], azimuth: &[f64]) -> Vec<f64> {
        let cos2_tth = tth
            .par_iter()
            .map(|x| x.cos().powi(2))
            .collect::<Vec<f64>>();

        let polarization = cos2_tth
            .par_iter()
            .zip(azimuth)
            .map(|(cos2_tth, azimuth)| math::f_polarization(&factor, cos2_tth, azimuth))
            .collect();
        polarization
    }
}

mod math {
    pub struct TrigonometricConstants {
        sin_rot1: f64,
        cos_rot1: f64,
        sin_rot2: f64,
        cos_rot2: f64,
        sin_rot3: f64,
        cos_rot3: f64,
    }

    impl TrigonometricConstants {
        pub fn new(poni: &super::Poni) -> TrigonometricConstants {
            TrigonometricConstants {
                sin_rot1: poni.rot1.sin(),
                cos_rot1: poni.rot1.cos(),
                sin_rot2: poni.rot2.sin(),
                cos_rot2: poni.rot2.cos(),
                sin_rot3: poni.rot3.sin(),
                cos_rot3: poni.rot3.cos(),
            }
        }
    }

    /// Calculates x1 for a pixel position in the sample reference coordinate system
    pub fn f_x1(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
        p1 * tc.cos_rot2 * tc.cos_rot3
            + p2 * (tc.cos_rot3 * tc.sin_rot1 * tc.sin_rot2 - tc.cos_rot1 * tc.sin_rot3)
            - p3 * (tc.cos_rot1 * tc.cos_rot3 * tc.sin_rot2 + tc.sin_rot1 * tc.sin_rot3)
    }

    /// Calculates x2 for a pixel position in the sample reference coordinate system
    pub fn f_x2(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
        p1 * tc.cos_rot2 * tc.sin_rot3
            + p2 * (tc.cos_rot1 * tc.cos_rot3 + tc.sin_rot1 * tc.sin_rot2 * tc.sin_rot3)
            - p3 * (-(tc.cos_rot3 * tc.sin_rot1) + tc.cos_rot1 * tc.sin_rot2 * tc.sin_rot3)
    }

    /// Calculates x3 for a pixel position in the sample reference coordinate system
    /// Note that the z axis is inverted in the sample reference coordinate system
    pub fn f_x3(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
        p1 * tc.sin_rot2 - p2 * tc.cos_rot2 * tc.sin_rot1 + p3 * tc.cos_rot1 * tc.cos_rot2
    }

    /// Calculates the 2theta angle for a pixel position in the sample reference coordinate system
    /// Parameters are the x, y and z values of the pixel position
    pub fn f_tth(x: &f64, y: &f64, z: &f64) -> f64 {
        (y * y + x * x).sqrt().atan2(*z)
    }

    /// Calculates the azimuthal angle for a pixel position in the sample reference coordinate system
    /// Parameters are the x and y values of the pixel position
    pub fn f_azimuth(x: &f64, y: &f64) -> f64 {
        x.atan2(*y)
    }

    /// Calculates the solid angle for a pixel position in the detector reference coordinate system
    /// p1 and p2 are coordinates in the detector reference coordinate system
    /// p3 is the distance from the sample to the detector
    pub fn f_solid_angle(p1: &f64, p2: &f64, p3: &f64) -> f64 {
        let cosa = f_cosa(p1, p2, p3);
        cosa.powi(3)
    }

    /// Calculates the cosine of the angle between the pixel position and the sample
    /// p1 and p2 are coordinates in the detector reference coordinate system
    /// p3 is the distance from the sample to the detector
    pub fn f_cosa(p1: &f64, p2: &f64, p3: &f64) -> f64 {
        p3 / (p1 * p1 + p2 * p2 + p3 * p3).sqrt()
    }

    /// Calculates polarization
    ///
    /// Arguments:
    ///   * `factor` - polarization factor (value between -1 and 1)
    ///   * `cos2_tth` - cosine squared of the 2theta angle in radians
    ///   * `azimuth` - azimuthal angle in radians
    pub fn f_polarization(factor: &f64, cos2_tth: &f64, azimuth: &f64) -> f64 {
        0.5 * (1.0 + cos2_tth - factor * (2.0 * azimuth).cos() * (1.0 - cos2_tth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;

    static DIM1: usize = 10;
    static DIM2: usize = 20;
    static SIZE: usize = DIM1 * DIM2;
    static TEST_INDEX: usize = 65;

    #[test]
    fn create_geometry() {
        let poni = Poni::new();
        let _corners = Geometry::new(DIM1, DIM2, poni);
    }

    #[test]
    fn calc_pixel_positions_det_reference() {
        let pixel1 = 0.0002;
        let pixel2 = 0.0003;
        let mut poni = Poni::new();
        poni.pixel1 = pixel1;
        poni.pixel2 = pixel2;
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let (d1_cart, d2_cart) = geometry.calc_pixel_position_det_reference_all();

        util::save_vec("d1_cart.txt", &d1_cart);
        util::save_vec("d2_cart.txt", &d2_cart);

        let d1_cart_reference = util::read_vec("d1_cart.txt").unwrap();
        let d2_cart_reference = util::read_vec("d2_cart.txt").unwrap();

        util::compare_vecs(&d1_cart_reference, &d1_cart, "d1_cart", 1e-10);
        util::compare_vecs(&d2_cart_reference, &d2_cart, "d1_cart", 1e-10);
    }

    #[test]
    fn calc_pixel_positions_poni_reference() {
        let mut poni: Poni = Poni::new();
        poni.pixel1 = 0.0002;
        poni.pixel2 = 0.0003;
        poni.poni1 = 0.032;
        poni.poni2 = 0.034;
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let (d1_cart_poni, d2_cart_poni) = geometry.calc_pixel_positions_with_poni_reference_all();
        assert_eq!(d1_cart_poni.len(), SIZE);
        assert_eq!(d2_cart_poni.len(), SIZE);
        assert_eq!(d1_cart_poni[0], -0.0319);
        assert!((d2_cart_poni[0] + 0.03385).abs() < 0.0000001);

        util::save_vec("d1_cart_poni.txt", &d1_cart_poni);
        util::save_vec("d2_cart_poni.txt", &d2_cart_poni);

        let d1_cart_poni_reference = util::read_vec("d1_cart_poni.txt").unwrap();
        let d2_cart_poni_reference = util::read_vec("d2_cart_poni.txt").unwrap();

        util::compare_vecs(
            &d1_cart_poni_reference,
            &d1_cart_poni,
            "d1_cart_poni",
            1e-10,
        );
        util::compare_vecs(
            &d2_cart_poni_reference,
            &d2_cart_poni,
            "d2_cart_poni",
            1e-10,
        );
    }

    #[test]
    fn calc_pixel_positions_with_sample_reference() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let (x, y, z) = geometry.calc_pixel_positions_with_sample_reference_all();

        util::save_vec("x.txt", &x);
        util::save_vec("y.txt", &y);
        util::save_vec("z.txt", &z);

        let x_ref = util::read_vec("x_pyFAI.txt").unwrap();
        let y_ref = util::read_vec("y_pyFAI.txt").unwrap();
        let z_ref = util::read_vec("z_pyFAI.txt").unwrap();

        util::compare_vecs(&x, &x_ref, "x", 1e-10);
        util::compare_vecs(&y, &y_ref, "x", 1e-10);
        util::compare_vecs(&z, &z_ref, "x", 1e-10);
    }

    #[test]
    fn calc_tth_all() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let tth = geometry.calc_tth_all();

        util::save_vec("tth.txt", &tth);

        let tth_ref = util::read_vec("tth_pyFAI.txt").unwrap();
        util::compare_vecs(&tth, &tth_ref, "tth", 1e-6);
    }

    #[test]
    fn calc_tth() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);

        let mut d1 = Vec::with_capacity(geometry.size());
        let mut d2 = Vec::with_capacity(geometry.size());

        for i in 0..DIM1 {
            for j in 0..DIM2 {
                d1.push(i as f64 + 0.5);
                d2.push(j as f64 + 0.5);
            }
        }

        let tth = geometry.calc_tth(&d1, &d2);
        let tth_all = geometry.calc_tth_all();
        util::compare_vecs(&tth, &tth_all, "tth", 1e-10);
    }

    #[test]
    fn calc_azimuth_all() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let azimuth = geometry.calc_azimuth_all();

        util::save_vec("azimuth.txt", &azimuth);

        let azimuth_ref = util::read_vec("azimuth_pyFAI.txt").unwrap();
        util::compare_vecs(&azimuth, &azimuth_ref, "azimuth", 1e-6);
    }

    #[test]
    fn calc_azimuth() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);

        let mut d1 = Vec::with_capacity(geometry.size());
        let mut d2 = Vec::with_capacity(geometry.size());

        for i in 0..DIM1 {
            for j in 0..DIM2 {
                d1.push(i as f64 + 0.5);
                d2.push(j as f64 + 0.5);
            }
        }

        let azimuth = geometry.calc_azimuth(&d1, &d2);
        let azimuth_all = geometry.calc_azimuth_all();
        util::compare_vecs(&azimuth, &azimuth_all, "tth", 1e-10);
    }

    #[test]
    fn calc_tth_and_azimuth() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);

        let mut d1 = Vec::with_capacity(geometry.size());
        let mut d2 = Vec::with_capacity(geometry.size());

        for i in 0..DIM1 {
            for j in 0..DIM2 {
                d1.push(i as f64 + 0.5);
                d2.push(j as f64 + 0.5);
            }
        }

        let (tth, azimuth) = geometry.calc_tth_and_azimuth(&d1, &d2);

        let tth_all = geometry.calc_tth_all();
        let azimuth_all = geometry.calc_azimuth_all();

        util::compare_vecs(&tth, &tth_all, "tth", 1e-10);
        util::compare_vecs(&azimuth, &azimuth_all, "azimuth", 1e-10);
    }

    #[test]
    fn calc_solid_angle() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let solid_angle = geometry.calc_solid_angle();

        util::save_vec("solid_angle.txt", &solid_angle);

        let solid_angle_ref = util::read_vec("solid_angle_pyFAI.txt").unwrap();
        util::compare_vecs(&solid_angle, &solid_angle_ref, "solid_angle", 1e-6);
    }

    #[test]
    fn calc_polarization() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(DIM1, DIM2, poni);
        let tth = geometry.calc_tth_all();
        let azimuth = geometry.calc_azimuth_all();
        let polarization = geometry.calc_polarization(0.99, &tth, &azimuth);

        util::save_vec("polarization.txt", &polarization);

        let polarization_ref = util::read_vec("polarization_pyFAI.txt").unwrap();
        util::compare_vecs(&polarization, &polarization_ref, "polarization", 1e-6);
    }
}
