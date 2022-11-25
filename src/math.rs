#![allow(dead_code)]

use rayon::prelude::*;
use std::cmp::PartialOrd;
use std::marker::Copy;

/// Estimates the minimum value of a vector
pub fn min<T: PartialOrd + Copy>(v: &Vec<T>) -> T {
    let mut min = v[0];
    for i in 1..v.len() {
        if v[i] < min {
            min = v[i];
        }
    }
    min
}

/// Estimates the maximum value of a vector
pub fn max<T: PartialOrd + Copy>(v: &Vec<T>) -> T {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min() {
        let v = vec![3.0, 1.0, 3.0, 4.0, 5.0];
        assert_eq!(min(&v), 1.0);
        let v = vec![3, 2, 4, 1, 5, 6, 7, 8, 9, 10];
        assert_eq!(min(&v), 1);
        let v = vec!['a', 'b', 'z', 'm'];
        assert_eq!(min(&v), 'a');
    }

    fn test_max() {
        let v = vec![3.0, 1.0, 3.0, 4.0, 5.0];
        assert_eq!(min(&v), 5.0);
        let v = vec![3, 2, 4, 1, 5, 6, 7, 8, 9, 10];
        assert_eq!(min(&v), 10);
        let v = vec!['a', 'b', 'z', 'm'];
        assert_eq!(min(&v), 'z');
    }
}
