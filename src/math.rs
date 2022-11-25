#![allow(dead_code)]

use rayon::prelude::*;

/// Estimates the minimum value of a vector
pub fn min(v: &Vec<f64>) -> f64 {
    let mut min = v[0];
    for i in 1..v.len() {
        if v[i] < min {
            min = v[i];
        }
    }
    min
}

/// Estimates the maximum value of a vector
pub fn max(v: &Vec<f64>) -> f64 {
    let mut max = v[0];
    for i in 1..v.len() {
        if v[i] > max {
            max = v[i];
        }
    }
    max
}

/// Divides vector a by vector b element-wise
pub fn divide(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<f64>>()
}
