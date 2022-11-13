#![allow(dead_code)]

use super::frame::Frame;
use super::geometry::Geometry;

struct Integrator {
    pub geometry: Geometry,
}

impl Integrator {
    pub fn new(geometry: Geometry) -> Integrator {
        Integrator { geometry }
    }

    pub fn integrate(&self, image: &Vec<f64>, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let tth = self.geometry.calc_tth();
        let chi: Vec<f64> = self.geometry.calc_chi();
        let solid_angle = self.geometry.calc_solid_angle();
        let polarization = self.geometry.calc_polarization(0.99, &tth, &chi);

        let intensity = image
            .iter()
            .enumerate()
            .map(|(i, y)| y / polarization[i] / solid_angle[i])
            .collect::<Vec<f64>>();

        let min_tth = min(&tth);
        let max_tth = max(&tth);
        let step_tth = (max_tth - min_tth) / (n_points as f64 - 1.0);

        let mut pattern_sum = vec![0.0; n_points];
        let mut pattern_count = vec![0.0; n_points];

        // Calulate the histogram
        for (i, tth) in tth.iter().enumerate() {
            let index = ((tth - min_tth) / step_tth).floor() as usize;
            pattern_sum[index] += intensity[i];
            pattern_count[index] += 1.0;
        }

        // normalize to amount of points
        let pattern_intensity = pattern_sum
            .iter()
            .zip(pattern_count.iter())
            .map(|(sum, count)| sum / count)
            .collect::<Vec<f64>>();

        // create tth values for bin centers
        let pattern_tth = (0..n_points)
            .map(|i| min_tth + ((i as f64) + 0.5) * step_tth)
            .collect::<Vec<f64>>();
        (pattern_tth, pattern_intensity)
    }
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
        let geometry = Geometry::new(frame.dim2, frame.dim1, poni);
        let integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate(&frame.data, 850);
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
