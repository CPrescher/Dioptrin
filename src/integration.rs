#![allow(dead_code)]

use super::frame::Frame;
use super::geometry::Geometry;

struct Integrator {
    pub geometry: Geometry,
    tth: Vec<f64>,
    chi: Vec<f64>,
    solid_angle: Vec<f64>,
    polarization: Vec<f64>,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator {
            geometry,
            tth: vec![],
            chi: vec![],
            solid_angle: vec![],
            polarization: vec![],
        }
    }

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

    pub fn integrate(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        let intensity = self.correct_intensity(&frame.data);
        histogram(&self.tth, &intensity, n_points)
    }

    fn correct_intensity(&mut self, data: &Vec<f64>) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, y)| y / self.polarization[i] / self.solid_angle[i])
            .collect::<Vec<f64>>()
    }
}

fn histogram(data: &Vec<f64>, weights: &Vec<f64>, n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let min = min(data);
    let max = max(data);
    let step = (max - min) / (n_points as f64 - 1.0);

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    // Calulate the histogram
    for (i, tth) in data.iter().enumerate() {
        let index = ((tth - min) / step).floor() as usize;
        sum[index] += weights[i];
        count[index] += 1.0;
    }

    // normalize to amount of points
    let normed_sum = sum
        .iter()
        .zip(count.iter())
        .map(|(sum, count)| sum / count)
        .collect::<Vec<f64>>();

    // create tth values for bin centers
    let bin = (0..n_points)
        .map(|i| min + ((i as f64) + 0.5) * step)
        .collect::<Vec<f64>>();
    (bin, normed_sum)
}

fn min(v: &Vec<f64>) -> f64 {
    let mut min = v[0];
    for i in 1..v.len() {
        if v[i] < min {
            min = v[i];
        }
    }
    min
}

fn max(v: &Vec<f64>) -> f64 {
    let mut max = v[0];
    for i in 1..v.len() {
        if v[i] > max {
            max = v[i];
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poni::Poni;
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_integrator() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate(&frame, 850);
        save_vec("pattern_tth.txt", &tth);
        save_vec("pattern_int.txt", &int);
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
