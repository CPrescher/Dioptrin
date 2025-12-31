use rayon::prelude::*;

use crate::math::histogram;

/// Calculates a look up table (LUT) implementation for a given geometry
/// A LUT is a vector with length n_points (same as the output pattern). Each element of this vector is
/// a vector of indices of the pixels that contribute to the pattern point.
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `n_points` - number of pattern points in ouput pattern
pub fn lut(tth: &Vec<f64>, n_points: usize) -> Vec<Vec<usize>> {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    let index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min, step))
        .collect();

    // Count elements per bin first for better memory allocation
    let mut counts = vec![0usize; n_points];
    for &idx in &index {
        counts[idx] += 1;
    }

    // Pre-allocate with exact capacity
    let mut lut: Vec<Vec<usize>> = counts
        .iter()
        .map(|&count| Vec::with_capacity(count))
        .collect();

    for (i, &idx) in index.iter().enumerate() {
        lut[idx].push(i);
    }
    lut
}

/// Integrates an image using a lookup table implementation
///
/// Arguments:
/// * `intensity` - image intensity
/// * `lut` - lookup table created before by the lut(..) function
/// * `hist_min` - minimum value of the histogram
/// * `hist_step` - step size of the histogram
pub fn integrate(
    intensity: &Vec<f64>,
    lut: &Vec<Vec<usize>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.len();

    // summing all saved pixel indices in the lookup table for each tth
    let pattern_intensity: Vec<f64> = lut
        .par_iter()
        .map(|indices| {
            let sum: f64 = indices.iter().map(|j| intensity[*j]).sum();
            let count = indices.len() as f64;
            if count > 0.0 {
                sum / count
            } else {
                0.0
            }
        })
        .collect();

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

