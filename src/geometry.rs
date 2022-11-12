//! Geometry calculations for each pixel in the detector
//!
#![allow(dead_code)]

use crate::poni::Poni;

pub struct Geometry {
    pub x_dim: usize,
    pub y_dim: usize,
    size: usize,
    pub poni: Poni,
    pub tth: Vec<f64>,
    pub chi: Vec<f64>,
}

impl Geometry {
    pub fn new(x_dim: usize, y_dim: usize, poni: Poni) -> Geometry {
        let size = x_dim * y_dim;

        Geometry {
            x_dim,
            y_dim,
            size,
            poni,
            tth: Vec::with_capacity(size),
            chi: Vec::with_capacity(size),
        }
    }

    /// Calculates the pixel positions in the detector reference coordinate system
    /// whereby the lower left corner of the detector is (0, 0)
    ///
    /// Arguments:
    ///   * `d1` - pixel positions in Y direction
    ///   * `d2` - pixel positions in X direction
    fn calc_pixel_position_det_reference(
        &mut self,
        d1: Vec<f64>,
        d2: Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        assert!(d1.len() == d2.len());

        let pixel1 = self.poni.pixel1;
        let pixel2 = self.poni.pixel2;

        let mut x = Vec::with_capacity(d1.len());
        let mut y = Vec::with_capacity(d1.len());

        for i in 0..d1.len() {
            x.push(d1[i] * pixel1);
            y.push(d2[i] * pixel2);
        }
        (x, y)
    }

    /// Calculates the pixel center positions of all pixels in the detector reference coordinate system
    fn calc_pixel_position_det_reference_all(&mut self) -> (Vec<f64>, Vec<f64>) {
        let pixel1 = self.poni.pixel1;
        let pixel2 = self.poni.pixel2;

        let mut x = Vec::with_capacity(self.size);
        let mut y = Vec::with_capacity(self.size);

        for i in 0..self.x_dim {
            for j in 0..self.y_dim {
                x.push((i as f64 + 0.5) * pixel1);
                y.push((j as f64 + 0.5) * pixel2);
            }
        }
        return (x, y);
    }

    /// Calculates the pixel center positions in the PONI reference coordinate system
    fn calc_pixel_positions_with_poni_reference(&mut self) -> (Vec<f64>, Vec<f64>) {
        let (x, y) = self.calc_pixel_position_det_reference_all();

        let mut x_poni = Vec::with_capacity(self.size);
        let mut y_poni = Vec::with_capacity(self.size);

        let poni1 = self.poni.poni1;
        let poni2 = self.poni.poni2;

        for i in 0..self.size {
            x_poni.push(x[i] - poni1);
            y_poni.push(y[i] - poni2);
        }
        return (x_poni, y_poni);
    }

    /// Calculates the pixel center positions in the sample reference coordinate system
    fn calc_pixel_positions_with_sample_reference(&mut self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (x_poni, y_poni) = self.calc_pixel_positions_with_poni_reference();

        let mut x_sample = Vec::with_capacity(self.size);
        let mut y_sample = Vec::with_capacity(self.size);
        let mut z_sample = Vec::with_capacity(self.size);

        let tc = TrigonometricConstants::new(&self.poni);

        for i in 0..self.size {
            y_sample.push(f_t1(&x_poni[i], &y_poni[i], &self.poni.distance, &tc));
            x_sample.push(f_t2(&x_poni[i], &y_poni[i], &self.poni.distance, &tc));
            z_sample.push(f_t3(&x_poni[i], &y_poni[i], &self.poni.distance, &tc));
        }
        return (x_sample, y_sample, z_sample);
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

/// Calculates t1 (y value) for a pixel position in the sample reference coordinate system
fn f_t1(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.cos_rot2 * tc.cos_rot3
        + p2 * (tc.cos_rot3 * tc.sin_rot1 * tc.sin_rot2 - tc.cos_rot1 * tc.sin_rot3)
        - p3 * (tc.cos_rot1 * tc.cos_rot3 * tc.sin_rot2 + tc.sin_rot1 * tc.sin_rot3)
}

/// Calculates t2 (x value) for a pixel position in the sample reference coordinate system
fn f_t2(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.cos_rot2 * tc.sin_rot3
        + p2 * (tc.cos_rot1 * tc.cos_rot3 + tc.sin_rot1 * tc.sin_rot2 * tc.sin_rot3)
        - p3 * (-(tc.cos_rot3 * tc.sin_rot1) + tc.cos_rot1 * tc.sin_rot2 * tc.sin_rot3)
}

/// Calculates t3 (-z value) for a pixel position in the sample reference coordinate system
/// Note that the z axis is inverted in the sample reference coordinate system
fn f_t3(p1: &f64, p2: &f64, p3: &f64, tc: &TrigonometricConstants) -> f64 {
    p1 * tc.sin_rot2 - p2 * tc.cos_rot2 * tc.sin_rot1 + p3 * tc.cos_rot1 * tc.cos_rot2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_geometry() {
        let poni = Poni::new();
        let corners = Geometry::new(101, 103, poni);
        assert_eq!(corners.tth.capacity(), 10403);
    }

    #[test]
    fn calc_pixel_positions_det_reference() {
        let mut poni = Poni::new();
        poni.pixel1 = 0.0002;
        poni.pixel2 = 0.0003;
        let mut geometry = Geometry::new(101, 103, poni);
        let (x, y) = geometry.calc_pixel_position_det_reference_all();
        assert_eq!(x.len(), 10403);
        assert_eq!(y.len(), 10403);
        assert_eq!(x[0], 0.0001);
        assert_eq!(y[0], 0.00015);
        assert_eq!(x[x.len() - 1], 0.0201);
        assert!((y[y.len() - 1] - 0.03075).abs() < 0.0000001);
    }

    #[test]
    fn calc_pixel_positions_poni_reference() {
        let mut poni: Poni = Poni::new();
        poni.pixel1 = 0.0002;
        poni.pixel2 = 0.0003;
        poni.poni1 = 0.032;
        poni.poni2 = 0.034;
        let mut geometry = Geometry::new(101, 103, poni);
        let (x, y) = geometry.calc_pixel_positions_with_poni_reference();
        assert_eq!(x.len(), 10403);
        assert_eq!(y.len(), 10403);
        assert_eq!(x[0], -0.0319);
        assert!(y[0] + 0.03385 < 0.0000001);
    }

    #[test]
    fn calc_pixel_positions_with_sample_reference() {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let mut geometry = Geometry::new(101, 103, poni);
        let (x, y, z) = geometry.calc_pixel_positions_with_sample_reference();

        assert_eq!(x.len(), 10403);
        assert_eq!(y.len(), 10403);
        assert_eq!(z.len(), 10403);

        assert!((x[0] + 0.0837083064488318).abs() < 0.0000001);
        assert!((y[0] + 0.1880643204202641).abs() < 0.0000001);
        assert!((z[0] - 0.2063665578247668).abs() < 0.0000001);

        assert!((x[3465] + 0.07235823705252044).abs() < 0.0000001);
        assert!((y[3465] + 0.18238923487242173).abs() < 0.0000001);
        assert!((z[3465] - 0.20659938549405718).abs() < 0.0000001);
    }
}
