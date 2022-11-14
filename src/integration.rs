#![allow(dead_code)]

use super::frame::Frame;
use super::geometry::Geometry;
use rayon::prelude::*;

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

    fn update_csr(&mut self, n_points: usize) {
        if self.csr.len() != n_points {
            self.csr = self.calc_csr(n_points);
        }
    }

    fn calc_csr(&self, n_points: usize) -> Vec<Vec<usize>> {
        let mut csr: Vec<Vec<usize>> = vec![Vec::new(); n_points];

        let min = min(&self.tth);
        let max = max(&self.tth) + f64::EPSILON;
        let step = (max - min) / (n_points as f64);

        let index: Vec<usize> = self
            .tth
            .par_iter()
            .map(|tth| ((tth - min) / step) as usize)
            .collect();

        for i in 0..self.tth.len() {
            csr[index[i]].push(i);
        }
        csr
    }

    pub fn integrate_csr(&mut self, frame: &Frame, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        self.update_geometry_arrays(frame.dim1, frame.dim2);
        self.update_csr(n_points);
        let intensity = self.correct_intensity(&frame.data);

        let min = min(&self.tth);
        let max = max(&self.tth);
        let step = (max - min) / (n_points as f64);

        let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

        for i in 0..n_points {
            pattern_intensity[i] += self.csr[i].iter().map(|j| intensity[*j]).sum::<f64>();
        }

        let csr_len = self
            .csr
            .par_iter()
            .map(|v| v.len() as f64)
            .collect::<Vec<f64>>();
        let pattern_intensity: Vec<f64> = divide(&pattern_intensity, &csr_len);

        // create tth values for bin centers
        let pattern_tth = histogram_bins(n_points, min, step);

        (pattern_tth, pattern_intensity)
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
    let max = max(data) * (1.0 + f64::EPSILON);
    let step = (max - min) / n_points as f64;

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    // calculate bin index for each data point
    let index: Vec<usize> = data
        .par_iter()
        .map(|x| ((x - min) / step) as usize)
        .collect();

    // Calulate the histogram
    for (i, y) in weights.iter().enumerate() {
        sum[index[i]] += y;
        count[index[i]] += 1.0;
    }

    // normalize to amount of points
    let normed_sum = sum
        .iter()
        .zip(count.iter())
        .map(|(sum, count)| sum / count)
        .collect::<Vec<f64>>();

    let bins = histogram_bins(n_points, min, step);
    (bins, normed_sum)
}

use std::sync::mpsc;
use std::thread;

fn histogram_parallel(
    data: &Vec<f64>,
    weights: &Vec<f64>,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let min = min(data);
    let max = max(data);
    let step = (max - min) / n_points as f64;

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    let (tx, rx) = mpsc::channel();

    let n_threads = 16;
    let chunk_size = data.len() / n_threads;

    for i in 0..n_threads {
        let tx_thread = mpsc::Sender::clone(&tx);
        let data_thread = data[i * chunk_size..(i + 1) * chunk_size].to_vec();
        let weights_thread = weights[i * chunk_size..(i + 1) * chunk_size].to_vec();

        thread::spawn(move || {
            let mut sum_thread = vec![0.0; n_points];
            let mut count_thread = vec![0.0; n_points];

            for (i, y) in data_thread.iter().enumerate() {
                let index = ((y - min) / step).floor() as usize;
                sum_thread[index] += weights_thread[i];
                count_thread[index] += 1.0;
            }

            tx_thread.send((sum_thread, count_thread)).unwrap();
        });
    }
    drop(tx);

    for (sum_thread, count_thread) in rx {
        for i in 0..n_points {
            sum[i] += sum_thread[i];
            count[i] += count_thread[i];
        }
    }

    // normalize to amount of points
    let normed_sum = sum
        .iter()
        .zip(count.iter())
        .map(|(sum, count)| sum / count)
        .collect::<Vec<f64>>();

    // create tth values for bin centers
    let bin = histogram_bins(n_points, min, step);
    (bin, normed_sum)
}

fn histogram_bins(n_points: usize, min: f64, step: f64) -> Vec<f64> {
    (0..n_points)
        .map(|i| min + ((i as f64) + 0.5) * step)
        .collect::<Vec<f64>>()
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

fn divide(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<f64>>()
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

    #[test]
    fn test_integrate_csr() {
        let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
        let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry);
        let (tth, int) = integrator.integrate_csr(&frame, 850);
        save_vec("pattern_tth_csr.txt", &tth);
        save_vec("pattern_int_csr.txt", &int);
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
