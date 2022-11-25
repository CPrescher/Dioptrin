use crate::{geometry::Geometry, histogram, math};
use rayon::prelude::*;

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
    for i in 1..(dim1 * n_subpixels) {
        d1.push(border_offset);
        d2.push(border_offset + i as f64 * step);
    }

    // top
    for i in 1..(dim1 * n_subpixels) {
        d1.push(dim1 as f64 - border_offset);
        d2.push(border_offset + i as f64 * step);
    }
    return (d1, d2);
}

/// Calculates a CSR implementation
/// This is a vector with length n_points (same as the output pattern). Each element of this vector is
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
) -> Vec<Vec<usize>> {
    let mut csr: Vec<Vec<usize>> = vec![Vec::new(); n_points];

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
            csr[*bin_index].push(actual_index[i]);
        }
    }
    csr
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
    for i in 0..n_points {
        pattern_intensity[i] += csr[i].iter().map(|j| intensity[*j]).sum::<f64>();
    }

    // obtain the length of each tth vector in the csr
    let csr_len = csr.par_iter().map(|v| v.len() as f64).collect::<Vec<f64>>();
    // divide the summed intensity by the length of each tth vector in the csr for normalization
    let pattern_intensity: Vec<f64> = math::divide(&pattern_intensity, &csr_len);

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
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
    csr: &Vec<Vec<usize>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = csr.len();
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    // summing all saved pixel indices in the csr for each tth
    for i in 0..n_points {
        pattern_intensity[i] += csr[i].iter().map(|j| intensity[*j]).sum::<f64>();
    }

    // obtain the length of each tth vector in the csr
    let csr_len = csr.par_iter().map(|v| v.len() as f64).collect::<Vec<f64>>();
    // divide the summed intensity by the length of each tth vector in the csr for normalization
    let pattern_intensity: Vec<f64> = math::divide(&pattern_intensity, &csr_len);

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

#[cfg(test)]
pub mod tests {}
