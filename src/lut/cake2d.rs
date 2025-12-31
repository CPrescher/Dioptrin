use rayon::prelude::*;

use crate::math::histogram;

/// Calculates a lookup table (LUT) for a 2D (cake) integration
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `azi` - azimuthal values for each pixel
/// * `n_tth` - number of pattern points in ouput pattern in tth direction
/// * `n_azi` - number of pattern points in ouput pattern in azi direction
pub fn lut_2d(tth: &Vec<f64>, azi: &Vec<f64>, n_tth: usize, n_azi: usize) -> Vec<Vec<Vec<usize>>> {
    // matrix initialization - optimized allocation
    let mut lut_2d: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n_azi]; n_tth];

    // find the binning for tth and azi
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    // find the indices for tth and azimuth for each pixel
    let indices: Vec<(usize, usize)> = tth
        .par_iter()
        .zip(azi.par_iter())
        .map(|(t, a)| {
            (
                histogram::get_bin(*t, min_tth, step_tth),
                histogram::get_bin(*a, min_azi, step_azi),
            )
        })
        .collect();

    // build LUT
    for (i, (tth_index, azi_index)) in indices.iter().enumerate() {
        lut_2d[*tth_index][*azi_index].push(i);
    }
    lut_2d
}

/// Integrates a image into a 2d pattern using a lookup table implementation
/// The output cake is a matrix with dimensions n_points_tth x n_points_azi flattened into a vector
///
/// Arguments:
/// * `intensity` - image intensity
/// * `lut_2d` - lookup table created before by the lut_2d(..) function
pub fn integrate2d(intensity: &Vec<f64>, lut_2d: &Vec<Vec<Vec<usize>>>) -> Vec<f64> {
    let n_points_tth = lut_2d.len();
    let n_points_azi = lut_2d[0].len();

    // Pre-compute all (tth_idx, azi_idx) pairs to avoid repeated calculations
    let indices: Vec<(usize, usize)> = (0..n_points_tth * n_points_azi)
        .map(|i| (i % n_points_tth, i / n_points_tth))
        .collect();

    // add all pixel intensities and normalize in one pass
    let cake_intensity: Vec<f64> = indices
        .par_iter()
        .map(|(tth_idx, azi_idx)| {
            let pixel_indices = &lut_2d[*tth_idx][*azi_idx];
            let sum: f64 = pixel_indices.iter().map(|j| intensity[*j]).sum();
            let num_points = pixel_indices.len() as f64;
            if num_points > 0.0 {
                sum / num_points
            } else {
                0.0
            }
        })
        .collect();

    cake_intensity
}

