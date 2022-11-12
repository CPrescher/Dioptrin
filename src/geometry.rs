//! Geometry calculations for each pixel in the detector
//!
#![allow(dead_code)]

use crate::poni::Poni;

pub struct Geometry {
    pub dim1: usize,
    pub dim2: usize,
    size: usize,
    pub poni: Poni,
    pub tth: Vec<f64>,
    pub chi: Vec<f64>,
}

impl Geometry {
    pub fn new(dim1: usize, dim2: usize, poni: Poni) -> Geometry {
        let size = dim1 * dim2;

        Geometry {
            dim1,
            dim2,
            size,
            poni,
            tth: Vec::with_capacity(size),
            chi: Vec::with_capacity(size),
        }
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
        &mut self,
        d1: Vec<f64>,
        d2: Vec<f64>,
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
    fn calc_pixel_position_det_reference_all(&mut self) -> (Vec<f64>, Vec<f64>) {
        let mut d1_cart = Vec::with_capacity(self.size);
        let mut d2_cart = Vec::with_capacity(self.size);

        for i in 0..self.dim1 {
            for j in 0..self.dim2 {
                d1_cart.push((i as f64 + 0.5) * self.poni.pixel1);
                d2_cart.push((j as f64 + 0.5) * self.poni.pixel2);
            }
        }
        return (d1_cart, d2_cart);
    }

    /// Calculates the pixel center positions of all pixels in the PONI reference
    /// coordinate system. return coordinates are in cartesian coordinates. Be aware
    /// that y and x are swapped in this framework. This results in | being the d1 (first)
    /// direction and _ being the d2 (second) direction.
    fn calc_pixel_positions_with_poni_reference(&mut self) -> (Vec<f64>, Vec<f64>) {
        let (x, y) = self.calc_pixel_position_det_reference_all();

        let mut d1_cart_poni = Vec::with_capacity(self.size);
        let mut d2_cart_poni = Vec::with_capacity(self.size);

        let poni1 = self.poni.poni1;
        let poni2 = self.poni.poni2;

        for i in 0..self.size {
            d1_cart_poni.push(x[i] - poni1);
            d2_cart_poni.push(y[i] - poni2);
        }
        return (d1_cart_poni, d2_cart_poni);
    }

    /// Calculates the pixel center positions in the sample reference coordinate system
    fn calc_pixel_positions_with_sample_reference(&mut self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (d1_cart_poni, d2_cart_poni) = self.calc_pixel_positions_with_poni_reference();

        let mut x_sample = Vec::with_capacity(self.size);
        let mut y_sample = Vec::with_capacity(self.size);
        let mut z_sample = Vec::with_capacity(self.size);

        let tc = TrigonometricConstants::new(&self.poni);

        for i in 0..self.size {
            x_sample.push(f_x1(
                &d1_cart_poni[i],
                &d2_cart_poni[i],
                &self.poni.distance,
                &tc,
            ));
            y_sample.push(f_x2(
                &d1_cart_poni[i],
                &d2_cart_poni[i],
                &self.poni.distance,
                &tc,
            ));
            z_sample.push(f_x3(
                &d1_cart_poni[i],
                &d2_cart_poni[i],
                &self.poni.distance,
                &tc,
            ));
        }
        return (x_sample, y_sample, z_sample);
    }

    /// Calculates the two theta angle for each pixel in the detector
    /// The result is saved in radians in the tth field of the Geometry struct
    fn calc_tth(&mut self) {
        let (x_sample, y_sample, z_sample) = self.calc_pixel_positions_with_sample_reference();
        for i in 0..self.size {
            self.tth
                .push(f_tth(&x_sample[i], &y_sample[i], &z_sample[i]));
        }
    }

    /// Calculates the chi angle for each pixel in the detector
    /// The output is saved in radians in the chi field of the Geometry struct
    fn calc_chi(&mut self) {
        let (x_sample, y_sample, _z_sample) = self.calc_pixel_positions_with_sample_reference();
        for i in 0..self.size {
            self.chi.push(f_chi(&x_sample[i], &y_sample[i]));
        }
    }

    /// Calculates Solid Angle for each pixel in the detector
    fn calc_solid_angle(&mut self) -> Vec<f64> {
        let (d1_cart_poni, d2_cart_poni) = self.calc_pixel_positions_with_poni_reference();
        let mut solid_angle = Vec::with_capacity(self.size);
        for i in 0..self.size {
            solid_angle.push(f_solid_angle(
                &d1_cart_poni[i],
                &d2_cart_poni[i],
                &self.poni.distance,
            ));
        }
        solid_angle
    }

    /// Calculates polarization for each pixel in the detector
    ///
    /// Arguments:
    ///   * `factor` - polarization factor (value between -1 and 1)
    fn calc_polarization(&mut self, factor: f64) -> Vec<f64> {
        if self.tth.len() == 0 {
            self.calc_tth();
        }
        if self.chi.len() == 0 {
            self.calc_chi();
        }

        let mut polarization = Vec::with_capacity(self.size);

        let mut cos2_tth = Vec::with_capacity(self.size);
        for i in 0..self.size {
            cos2_tth.push(self.tth[i].cos().powi(2));
        }

        for i in 0..self.size {
            polarization.push(f_polarization(&factor, &cos2_tth[i], &self.chi[i]));
        }
        polarization
    }
}

struct TrigonometricConstants {
    sin_rot1: f64,
    cos_rot1: f64,
    sin_rot2: f64,
    cos_rot2: f64,
    sin_rot3: f64,
    cos_rot3: f64,
}

impl TrigonometricConstants {
    fn new(poni: &Poni) -> TrigonometricConstants {
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
fn f_x1(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.cos_rot2 * tc.cos_rot3
        + p2 * (tc.cos_rot3 * tc.sin_rot1 * tc.sin_rot2 - tc.cos_rot1 * tc.sin_rot3)
        - p3 * (tc.cos_rot1 * tc.cos_rot3 * tc.sin_rot2 + tc.sin_rot1 * tc.sin_rot3)
}

/// Calculates x2 for a pixel position in the sample reference coordinate system
fn f_x2(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.cos_rot2 * tc.sin_rot3
        + p2 * (tc.cos_rot1 * tc.cos_rot3 + tc.sin_rot1 * tc.sin_rot2 * tc.sin_rot3)
        - p3 * (-(tc.cos_rot3 * tc.sin_rot1) + tc.cos_rot1 * tc.sin_rot2 * tc.sin_rot3)
}

/// Calculates x3 for a pixel position in the sample reference coordinate system
/// Note that the z axis is inverted in the sample reference coordinate system
fn f_x3(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.sin_rot2 - p2 * tc.cos_rot2 * tc.sin_rot1 + p3 * tc.cos_rot1 * tc.cos_rot2
}

/// Calculates the 2theta angle for a pixel position in the sample reference coordinate system
/// Parameters are the x, y and z values of the pixel position
fn f_tth(x: &f64, y: &f64, z: &f64) -> f64 {
    (y * y + x * x).sqrt().atan2(*z)
}

/// Calculates the azimuthal angle for a pixel position in the sample reference coordinate system
/// Parameters are the x and y values of the pixel position
fn f_chi(x: &f64, y: &f64) -> f64 {
    x.atan2(*y)
}

/// Calculates the solid angle for a pixel position in the detector reference coordinate system
/// p1 and p2 are coordinates in the detector reference coordinate system
/// p3 is the distance from the sample to the detector
fn f_solid_angle(p1: &f64, p2: &f64, p3: &f64) -> f64 {
    let cosa = f_cosa(p1, p2, p3);
    return cosa.powi(3);
}

/// Calculates the cosine of the angle between the pixel position and the sample
/// p1 and p2 are coordinates in the detector reference coordinate system
/// p3 is the distance from the sample to the detector
fn f_cosa(p1: &f64, p2: &f64, p3: &f64) -> f64 {
    p3 / (p1 * p1 + p2 * p2 + p3 * p3).sqrt()
}

/// Calculates polarization
///
/// Arguments:
///   * `factor` - polarization factor (value between -1 and 1)
///   * `cos2_tth` - cosine squared of the 2theta angle
///   * `chi` - azimuthal angle in radians
fn f_polarization(factor: &f64, cos2_tth: &f64, chi: &f64) -> f64 {
    0.5 * (1.0 + cos2_tth - factor * (2.0 * chi).cos() * (1.0 - cos2_tth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    static DIM1: usize = 10;
    static DIM2: usize = 20;
    static SIZE: usize = DIM1 * DIM2;
    static TEST_INDEX: usize = 65;

    #[test]
    fn create_geometry() {
        let poni = Poni::new();
        let corners = Geometry::new(DIM1, DIM2, poni);
        assert_eq!(corners.tth.capacity(), SIZE);
    }

    #[test]
    fn calc_pixel_positions_det_reference() {
        let pixel1 = 0.0002;
        let pixel2 = 0.0003;
        let mut poni = Poni::new();
        poni.pixel1 = pixel1;
        poni.pixel2 = pixel2;
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        let (d1_cart, d2_cart) = geometry.calc_pixel_position_det_reference_all();

        save_vec("d1_cart.txt", &d1_cart);
        save_vec("d2_cart.txt", &d2_cart);

        assert_eq!(d1_cart.len(), SIZE);
        assert_eq!(d2_cart.len(), SIZE);
        assert_eq!(d1_cart[0], 0.0001);
        assert_eq!(d2_cart[0], 0.00015);

        let d1_last_pixel = (DIM1 as f64 - 0.5) * pixel1;
        let d2_last_pixel = (DIM2 as f64 - 0.5) * pixel2;
        assert!((d1_cart[d1_cart.len() - 1] - d1_last_pixel).abs() < 0.0000001);
        assert!((d2_cart[d2_cart.len() - 1] - d2_last_pixel).abs() < 0.0000001);
    }

    #[test]
    fn calc_pixel_positions_poni_reference() {
        let mut poni: Poni = Poni::new();
        poni.pixel1 = 0.0002;
        poni.pixel2 = 0.0003;
        poni.poni1 = 0.032;
        poni.poni2 = 0.034;
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        let (d1_cart_poni, d2_cart_poni) = geometry.calc_pixel_positions_with_poni_reference();
        assert_eq!(d1_cart_poni.len(), SIZE);
        assert_eq!(d2_cart_poni.len(), SIZE);
        assert_eq!(d1_cart_poni[0], -0.0319);
        assert!((d2_cart_poni[0] + 0.03385).abs() < 0.0000001);

        save_vec("d1_cart_poni.txt", &d1_cart_poni);
        save_vec("d2_cart_poni.txt", &d2_cart_poni);
    }

    #[test]
    fn calc_pixel_positions_with_sample_reference() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        let (x, y, z) = geometry.calc_pixel_positions_with_sample_reference();

        save_vec("x.txt", &x);
        save_vec("y.txt", &y);
        save_vec("z.txt", &z);

        assert_eq!(x.len(), SIZE);
        assert_eq!(y.len(), SIZE);
        assert_eq!(z.len(), SIZE);

        assert!((x[0] + 0.1880643204202641).abs() < 0.0000001);
        assert!((y[0] + 0.0837083064488318).abs() < 0.0000001);
        assert!((z[0] - 0.2063665578247668).abs() < 0.0000001);

        assert!((x[TEST_INDEX] + 0.18754839043309052).abs() < 0.0000001);
        assert!((y[TEST_INDEX] + 0.08284845270907473).abs() < 0.0000001);
        assert!((z[TEST_INDEX] - 0.20638455211721887).abs() < 0.0000001);
    }

    #[test]
    fn calc_tth() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        geometry.calc_tth();

        save_vec("tth.txt", &geometry.tth);

        assert_eq!(geometry.tth.len(), SIZE);
        assert!((geometry.tth[0] - 0.7841512459252314).abs() < 0.0000001);
        assert!((geometry.tth[TEST_INDEX] - 0.7821114852040393).abs() < 0.0000001);
    }

    #[test]
    fn calc_chi() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        geometry.calc_chi();

        save_vec("chi.txt", &geometry.chi);

        assert_eq!(geometry.chi.len(), SIZE);
        assert!((geometry.chi[0] + 1.9895718171402677).abs() < 0.0000001);
        assert!((geometry.chi[TEST_INDEX] + 1.986763738717781).abs() < 0.0000001);
    }

    #[test]
    fn calc_solid_angle() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        let solid_angle = geometry.calc_solid_angle();

        save_vec("solid_angle.txt", &solid_angle);

        assert!((solid_angle[0] - 0.3667949927780228).abs() < 0.0000001);
        assert!((solid_angle[TEST_INDEX] - 0.36893966446276066).abs() < 0.0000001);
    }

    #[test]
    fn calc_polarization() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(DIM1, DIM2, poni);
        let polarization = geometry.calc_polarization(0.99);

        save_vec("polarization.txt", &polarization);

        assert!((polarization[0] - 0.915858268737793).abs() < 0.0000001);
        assert!((polarization[TEST_INDEX] - 0.9172257781028748).abs() < 0.0000001);
    }

    fn save_vec(path: &str, vec: &Vec<f64>) {
        let dir_str: String = "notebooks/test_output/".to_owned();
        fs::create_dir(&dir_str).unwrap_or_default();
        let path_str = dir_str + path;
        let path: &Path = Path::new(path_str.as_str());
        let mut file = fs::File::create(path).unwrap();
        for val in vec.iter() {
            file.write_all(format!("{}\n", val).as_bytes()).unwrap();
        }
    }
}
