#![allow(dead_code)]

use std::sync::mpsc;
use std::thread;

use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::{csr, math};

/// Calculates a normalized histogram of weights at data values
pub fn histogram(data: &Vec<f64>, weights: &Vec<f64>, n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    // calculate bin index for each data point
    let index: Vec<usize> = data.par_iter().map(|x| get_bin(*x, min, step)).collect();

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

    let bins = bins(n_points, min, step);
    (bins, normed_sum)
}

/// Calculates a normalized histogram of weights at data values using a parallel implementation
pub fn histogram_parallel(
    data: &Vec<f64>,
    weights: &Vec<f64>,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);

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
                let index = get_bin(*y, min, step);
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
    let bin = bins(n_points, min, step);
    (bin, normed_sum)
}

/// Calculates the bin index for a value in a histogram
/// min is the min value of the data, step is the bin width
pub fn get_bin(value: f64, min: f64, step: f64) -> usize {
    ((value - min) / step) as usize
}

pub fn supersampled_min_max_step(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_subpixels: usize,
) -> (f64, f64, f64) {
    let (hull_d1, hull_d2) = csr::hull_grid(geometry.dim1, geometry.dim2, n_subpixels);
    let hull_tth = geometry.calc_tth(&hull_d1, &hull_d2);

    let min = math::min(&tth).min(math::min(&hull_tth));
    let max = math::max(&tth).max(math::max(&hull_tth)) + f64::EPSILON;
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

/// Estimates min, max and step size for a histogram
pub fn min_max_step(data: &Vec<f64>, n_points: usize) -> (f64, f64, f64) {
    let min = math::min(data);
    let max = math::max(data) + f64::EPSILON;
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

/// Calculates the bin centers for a histogram
pub fn bins(n_points: usize, min: f64, step: f64) -> Vec<f64> {
    (0..n_points)
        .map(|i| min + ((i as f64) + 0.5) * step)
        .collect::<Vec<f64>>()
}
