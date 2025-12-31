use rayon::prelude::*;
use std::collections::HashMap;

use crate::geometry::Geometry;
use crate::math::histogram;
use crate::lut::common::subgrid;

/// Calculates a lookup table (LUT) implementation with supersampling
/// This is a vector with length n_points (same as the output pattern). Each element of this vector is
/// a vector of indices of the pixels that contribute to the pattern point.
///
/// Arguments:
/// * `geometry` - geometry of the detector
/// * `tth` - 2theta values of the pixels
/// * `n_points` - number of points in the output pattern
/// * `n_ss` - super sampling factor, number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn lut_ss(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
) -> Vec<HashMap<usize, f64>> {
    let mut lut: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_points];

    // Estimate the 2theta values of the hull of the detector for detemining the min and max of the integrated pattern
    let (min, _max, step) = histogram::ss_tth_min_max_step(&geometry, &tth, n_points, n_ss);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = subgrid(row, geometry.dim2, n_ss);

        let tth = geometry.calc_tth(&d1, &d2);
        let tth_bin_indices: Vec<usize> = tth
            .par_iter()
            .map(|tth| histogram::get_bin(*tth, min, step))
            .collect();

        for (i, bin_index) in tth_bin_indices.iter().enumerate() {
            let count = lut[*bin_index].entry(actual_index[i]).or_insert(0.0);
            *count += 1.0;
        }
    }

    // obtain the length of each tth vector in the lookup table
    let lut_len = lut
        .par_iter()
        .map(|hash| hash.values().sum::<f64>())
        .collect::<Vec<f64>>();

    // divide all values in the lookup table by the respective length of the tth vector
    lut.par_iter()
        .enumerate()
        .map(|(i, hash)| {
            hash.par_iter()
                .map(|(k, v)| (*k, v / lut_len[i]))
                .collect()
        })
        .collect()
}

pub fn lut_2d_ss(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
    n_ss: usize,
) -> Vec<Vec<HashMap<usize, f64>>> {
    // matrix initialization - optimized allocation
    let mut lut_2d: Vec<Vec<HashMap<usize, f64>>> = vec![vec![HashMap::new(); n_azi]; n_tth];

    let (tth_min, _max, tth_step) = histogram::ss_tth_min_max_step(&geometry, &tth, n_tth, n_ss);

    let (azi_min, _max, azi_step) =
        histogram::supersampled_chi_min_max_step(&geometry, &azi, n_azi, n_ss);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = subgrid(row, geometry.dim2, n_ss);

        let (tth, azi) = geometry.calc_tth_and_chi(&d1, &d2);
        let tth_indices: Vec<usize> = tth
            .par_iter()
            .map(|tth| histogram::get_bin(*tth, tth_min, tth_step))
            .collect();

        let azi_indices: Vec<usize> = azi
            .par_iter()
            .map(|azi| histogram::get_bin(*azi, azi_min, azi_step))
            .collect();

        for (i, (tth_index, azi_index)) in tth_indices.iter().zip(&azi_indices).enumerate() {
            let azi_capped = azi_index % n_azi;
            let count = lut_2d[*tth_index][azi_capped]
                .entry(actual_index[i])
                .or_insert(0.0);
            *count += 1.0;
        }
    }

    // obtain the length of each intensity vector in the lut
    let lut_len = lut_2d
        .par_iter()
        .map(|v| {
            v.par_iter()
                .map(|hash| hash.values().sum::<f64>())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    // divide all values in the lut by the respective length of the intensity vector
    lut_2d
        .par_iter()
        .enumerate()
        .map(|(i, v)| {
            v.par_iter()
                .enumerate()
                .map(|(j, hash)| {
                    hash.par_iter()
                        .map(|(k, v)| (*k, v / lut_len[i][j]))
                        .collect()
                })
                .collect::<Vec<HashMap<usize, f64>>>()
        })
        .collect()
}

pub fn integrate2d_ss(intensity: &Vec<f64>, lut: &Vec<Vec<HashMap<usize, f64>>>) -> Vec<f64> {
    let n_points_tth = lut.len();
    let n_points_azi = lut[0].len();

    // Pre-compute all (tth_idx, azi_idx) pairs to avoid repeated calculations
    let indices: Vec<(usize, usize)> = (0..n_points_tth * n_points_azi)
        .map(|i| (i % n_points_tth, i / n_points_tth))
        .collect();

    // add all pixel intensities for each tth and azi to the corresponding pixel in the cake
    let cake_intensity: Vec<f64> = indices
        .par_iter()
        .map(|(tth_idx, azi_idx)| {
            lut[*tth_idx][*azi_idx]
                .iter()
                .map(|(j, v)| intensity[*j] * v)
                .sum::<f64>()
        })
        .collect();

    cake_intensity
}

/// Integrates an image using a LUT implementation with supersampling
///
/// Arguments:
/// * `intensity` - image intensity
/// * `lut` - lookup table previously generated with the lut_ss(..) function
/// * `hist_min` - minimum value of the histogram
/// * `hist_step` - step size of the histogram
pub fn integrate_ss(
    intensity: &Vec<f64>,
    lut: &Vec<HashMap<usize, f64>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.len();
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    pattern_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += lut[i].iter().map(|(j, v)| intensity[*j] * *v).sum::<f64>();
        });

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

