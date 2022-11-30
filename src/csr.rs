use crate::{geometry::Geometry, histogram, math};
use rayon::prelude::*;
use std::collections::HashMap;

/// Creates a d1, d2 subgrid for a column in a dim1 x dim2 grid
///
/// Arguments:
///   * `row` - column index for the sub grid
///   * `dim2` - number of points in horizontal direction (-)
///   * `n_subpixels` - number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn subgrid(row: usize, dim2: usize, n_subpixels: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut d1 = Vec::with_capacity(dim2 * n_subpixels);
    let mut d2 = Vec::with_capacity(dim2 * n_subpixels);
    let mut actual_index = Vec::with_capacity(dim2 * n_subpixels);

    let dim_sub = dim2 * n_subpixels;

    for i in 0..n_subpixels {
        for j in 0..dim_sub {
            d1.push(row as f64 + (i as f64 + 0.5) / n_subpixels as f64);
            d2.push((j as f64 + 0.5) / n_subpixels as f64);
            actual_index.push(row * dim2 + j / n_subpixels);
        }
    }

    return (d1, d2, actual_index);
}

/// Creates a grid of d1, d2 coordinates for the hull of dim1 x dim2 grid (i.e. the outermost points)
///
/// Arguments:
/// * `dim1` - number of points in vertical direction (-)
/// * `dim2` - number of points in horizontal direction (-)
/// * `n_subpixels` - number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn hull_grid(dim1: usize, dim2: usize, n_subpixels: usize) -> (Vec<f64>, Vec<f64>) {
    let size = 2 * (dim1 + dim2 - 2) * n_subpixels;
    let mut d1 = Vec::with_capacity(size);
    let mut d2 = Vec::with_capacity(size);

    let border_offset = 0.5 / n_subpixels as f64;
    let step = 1.0 / n_subpixels as f64;

    // left
    for i in 0..dim1 * n_subpixels {
        d1.push(border_offset + i as f64 * step);
        d2.push(border_offset);
    }

    // right
    for i in 0..dim1 * n_subpixels {
        d1.push(border_offset + i as f64 * step);
        d2.push(dim2 as f64 - border_offset);
    }

    // bottom
    for i in 1..(dim2 * n_subpixels) {
        d1.push(border_offset);
        d2.push(border_offset + i as f64 * step);
    }

    // top
    for i in 1..(dim2 * n_subpixels) {
        d1.push(dim1 as f64 - border_offset);
        d2.push(border_offset + i as f64 * step);
    }
    return (d1, d2);
}

/// Calculates a CSR implementation
/// A CSR is a vector with length n_points (same as the output pattern). Each element of this vector is
/// a vector of indices of the pixels that contribute to the pattern point.
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `n_points` - number of pattern points in ouput pattern
pub fn csr(tth: &Vec<f64>, n_points: usize) -> Vec<Vec<usize>> {
    let mut csr: Vec<Vec<usize>> = vec![Vec::new(); n_points];

    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    let index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min, step))
        .collect();

    for i in 0..tth.len() {
        csr[index[i]].push(i);
    }
    csr
}

/// Calculates a CSR table for a 2D (cake) integration
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `azi` - azimuthal values for each pixel
/// * `n_points_tth` - number of pattern points in ouput pattern in tth direction
/// * `n_points_azi` - number of pattern points in ouput pattern in azi direction
pub fn csr2d(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_points_tth: usize,
    n_points_azi: usize,
) -> Vec<Vec<Vec<usize>>> {
    // matrix initialization
    let mut csr_2d: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_points_tth);
    for v in 0..n_points_tth {
        csr_2d.push(Vec::with_capacity(n_points_azi));
        for _u in 0..n_points_azi {
            csr_2d[v].push(Vec::new());
        }
    }

    // find the binning for tth and azi
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_points_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_points_azi);

    // find the indices for tth and azimuth for each pixel
    let tth_index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min_tth, step_tth))
        .collect();

    let azi_index: Vec<usize> = azi
        .par_iter()
        .map(|azi| histogram::get_bin(*azi, min_azi, step_azi))
        .collect();

    // build CSR
    for (i, (tth_index, azi_index)) in tth_index.iter().zip(&azi_index).enumerate() {
        csr_2d[*tth_index][*azi_index].push(i);
    }

    for (i, (tth_index, azi_index)) in tth_index.iter().zip(&azi_index).enumerate() {
        csr_2d[*tth_index][*azi_index].push(i);
    }
    csr_2d
}

/// Integrates an image using a CSR implementation
///
/// Arguments:
/// * `intensity` - image intensity
/// * `csr` - CSR implementation
/// * `hist_min` - minimum value of the histogram
/// * `hist_step` - step size of the histogram
pub fn integrate(
    intensity: &Vec<f64>,
    csr: &Vec<Vec<usize>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = csr.len();
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    // summing all saved pixel indices in the csr for each tth
    pattern_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += csr[i].iter().map(|j| intensity[*j]).sum::<f64>();
        });

    // obtain the length of each tth vector in the csr
    let csr_len = csr.par_iter().map(|v| v.len() as f64).collect::<Vec<f64>>();
    // divide the summed intensity by the length of each tth vector in the csr for normalization
    let pattern_intensity: Vec<f64> = math::divide(&pattern_intensity, &csr_len);

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

/// Integrates a image into a 2d pattern using a CSR implementation
/// The output cake is a matrix with dimensions n_points_tth x n_points_azi flattened into a vector
///
/// Arguments:
/// * `intensity` - image intensity
/// * `csr` - csr2D values
pub fn integrate2d(intensity: &Vec<f64>, csr: &Vec<Vec<Vec<usize>>>) -> Vec<f64> {
    let n_points_tth = csr.len();
    let n_points_azi = csr[0].len();

    let mut cake_intensity: Vec<f64> = vec![0.0; n_points_tth * n_points_azi];

    // add all pixel intensities for each tth and azi to the corresponding pixel in the cake
    cake_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += csr[i % n_points_tth][i / n_points_tth]
                .iter()
                .map(|j| intensity[*j])
                .sum::<f64>();
        });

    // normalize the cake based on the number of pixels in each tth and azi bin
    cake_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            let num_points = csr[i % n_points_tth][i / n_points_tth].len() as f64;
            if num_points > 0 as f64 {
                *int /= num_points;
            }
        });
    cake_intensity
}

/// Calculates a CSR implementation with supersampling
/// This is a vector with length n_points (same as the output pattern). Each element of this vector is
/// a vector of indices of the pixels that contribute to the pattern point.
///
/// Arguments:
/// * `geometry` - geometry of the detector
/// * `tth` - 2theta values of the pixels
/// * `n_points` - number of points in the output pattern
/// * `n_subpixels` - number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn csr_supersampled(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_subpixels: usize,
) -> Vec<HashMap<usize, f64>> {
    let mut csr: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_points];

    // Estimate the 2theta values of the hull of the detector for detemining the min and max of the integrated pattern
    let (min, _max, step) =
        histogram::supersampled_min_max_step(&geometry, &tth, n_points, n_subpixels);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = subgrid(row, geometry.dim2, n_subpixels);

        let tth = geometry.calc_tth(&d1, &d2);
        let bin_indices: Vec<usize> = tth
            .par_iter()
            .map(|tth| histogram::get_bin(*tth, min, step))
            .collect();

        for (i, bin_index) in bin_indices.iter().enumerate() {
            let count = csr[*bin_index].entry(actual_index[i]).or_insert(1.0);
            *count += 1.0;
        }
    }

    // obtain the length of each tth vector in the csr
    let csr_len = csr
        .par_iter()
        .map(|hash| hash.values().map(|v| *v as f64).sum())
        .collect::<Vec<f64>>();

    // divide all values in the csr by the respective length of the tth vector
    csr.par_iter()
        .enumerate()
        .map(|(i, hash)| {
            hash.par_iter()
                .map(|(k, v)| (*k, *v as f64 / csr_len[i]))
                .collect()
        })
        .collect()
}

/// Integrates an image using a CSR implementation with supersampling
///
/// Arguments:
/// * `intensity` - image intensity
/// * `csr` - CSR implementation
/// * `hist_min` - minimum value of the histogram
/// * `hist_step` - step size of the histogram
pub fn integrate_supersampled(
    intensity: &Vec<f64>,
    csr: &Vec<HashMap<usize, f64>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = csr.len();
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    pattern_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += csr[i].iter().map(|(j, v)| intensity[*j] * *v).sum::<f64>();
        });

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

#[cfg(test)]
pub mod tests {}
