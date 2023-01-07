use crate::{geometry::Geometry, math, math::histogram};
use rayon::prelude::*;
use std::collections::HashMap;

/// Creates a d1, d2 subgrid for a column in a dim1 x dim2 grid
///
/// Arguments:
///   * `row` - column index for the sub grid
///   * `dim2` - number of points in horizontal direction (-)
///   * `n_ss` - supersampling factor -> number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn subgrid(row: usize, dim2: usize, n_ss: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut d1 = Vec::with_capacity(dim2 * n_ss);
    let mut d2 = Vec::with_capacity(dim2 * n_ss);
    let mut actual_index = Vec::with_capacity(dim2 * n_ss);

    let dim_sub = dim2 * n_ss;

    for i in 0..n_ss {
        for j in 0..dim_sub {
            d1.push(row as f64 + (i as f64 + 0.5) / n_ss as f64);
            d2.push((j as f64 + 0.5) / n_ss as f64);
            actual_index.push(row * dim2 + j / n_ss);
        }
    }

    return (d1, d2, actual_index);
}

/// Creates a grid of d1, d2 coordinates for the hull of dim1 x dim2 grid (i.e. the outermost points)
///
/// Arguments:
/// * `dim1` - number of points in vertical direction (-)
/// * `dim2` - number of points in horizontal direction (-)
/// * `n_ss` - supersampling factor -> number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn hull_grid(dim1: usize, dim2: usize, n_ss: usize) -> (Vec<f64>, Vec<f64>) {
    let size = 2 * (dim1 + dim2 - 2) * n_ss;
    let mut d1 = Vec::with_capacity(size);
    let mut d2 = Vec::with_capacity(size);

    let border_offset = 0.5 / n_ss as f64;
    let step = 1.0 / n_ss as f64;

    // left
    for i in 0..dim1 * n_ss {
        d1.push(border_offset + i as f64 * step);
        d2.push(border_offset);
    }

    // right
    for i in 0..dim1 * n_ss {
        d1.push(border_offset + i as f64 * step);
        d2.push(dim2 as f64 - border_offset);
    }

    // bottom
    for i in 1..(dim2 * n_ss) {
        d1.push(border_offset);
        d2.push(border_offset + i as f64 * step);
    }

    // top
    for i in 1..(dim2 * n_ss) {
        d1.push(dim1 as f64 - border_offset);
        d2.push(border_offset + i as f64 * step);
    }
    return (d1, d2);
}

/// Calculates a look up table (LUT) implementation for a given geometry
/// A LUT is a vector with length n_points (same as the output pattern). Each element of this vector is
/// a vector of indices of the pixels that contribute to the pattern point.
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `n_points` - number of pattern points in ouput pattern
pub fn lut(tth: &Vec<f64>, n_points: usize) -> Vec<Vec<usize>> {
    let mut lut: Vec<Vec<usize>> = vec![Vec::new(); n_points];

    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    let index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min, step))
        .collect();

    for i in 0..tth.len() {
        lut[index[i]].push(i);
    }
    lut
}

/// Calculates a lookup table (LUT) for a 2D (cake) integration
///
/// Arguments:
/// * `tth` - two theta values for each pixel
/// * `azi` - azimuthal values for each pixel
/// * `n_tth` - number of pattern points in ouput pattern in tth direction
/// * `n_azi` - number of pattern points in ouput pattern in azi direction
pub fn lut_2d(tth: &Vec<f64>, azi: &Vec<f64>, n_tth: usize, n_azi: usize) -> Vec<Vec<Vec<usize>>> {
    // matrix initialization
    let mut lut_2d: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_tth);
    for v in 0..n_tth {
        lut_2d.push(Vec::with_capacity(n_azi));
        for _u in 0..n_azi {
            lut_2d[v].push(Vec::new());
        }
    }

    // find the binning for tth and azi
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    // find the indices for tth and azimuth for each pixel
    let tth_index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min_tth, step_tth))
        .collect();

    let azi_index: Vec<usize> = azi
        .par_iter()
        .map(|azi| histogram::get_bin(*azi, min_azi, step_azi))
        .collect();

    // build LUT
    for (i, (tth_index, azi_index)) in tth_index.iter().zip(&azi_index).enumerate() {
        lut_2d[*tth_index][*azi_index].push(i);
    }
    lut_2d
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
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    // summing all saved pixel indices in the lookup table for each tth
    pattern_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += lut[i].iter().map(|j| intensity[*j]).sum::<f64>();
        });

    // obtain the length of each tth vector in the lut
    let lut_len = lut.par_iter().map(|v| v.len() as f64).collect::<Vec<f64>>();
    // divide the summed intensity by the length of each tth vector in the lut for normalization
    let pattern_intensity: Vec<f64> = math::divide(&pattern_intensity, &lut_len);

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
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

    let mut cake_intensity: Vec<f64> = vec![0.0; n_points_tth * n_points_azi];

    // add all pixel intensities for each tth and azi to the corresponding pixel in the cake
    cake_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += lut_2d[i % n_points_tth][i / n_points_tth]
                .iter()
                .map(|j| intensity[*j])
                .sum::<f64>();
        });

    // normalize the cake based on the number of pixels in each tth and azi bin
    cake_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            let num_points = lut_2d[i % n_points_tth][i / n_points_tth].len() as f64;
            if num_points > 0 as f64 {
                *int /= num_points;
            }
        });
    cake_intensity
}

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
            let count = lut[*bin_index].entry(actual_index[i]).or_insert(1.0);
            *count += 1.0;
        }
    }

    // obtain the length of each tth vector in the lookup table
    let lut_len = lut
        .par_iter()
        .map(|hash| hash.values().map(|v| *v as f64).sum())
        .collect::<Vec<f64>>();

    // divide all values in the lookup table by the respective length of the tth vector
    lut.par_iter()
        .enumerate()
        .map(|(i, hash)| {
            hash.par_iter()
                .map(|(k, v)| (*k, *v as f64 / lut_len[i]))
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
    // matrix initialization
    let mut lut_2d: Vec<Vec<HashMap<usize, f64>>> = Vec::with_capacity(n_tth);
    for v in 0..n_tth {
        lut_2d.push(Vec::with_capacity(n_azi));
        for _u in 0..n_azi {
            lut_2d[v].push(HashMap::new());
        }
    }

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
                .or_insert(1.0);
            *count += 1.0;
        }
    }

    // obtain the length of each intensity vector in the lut
    let lut_len = lut_2d
        .par_iter()
        .map(|v| {
            v.par_iter()
                .map(|hash| hash.values().map(|v| *v as f64).sum())
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
                        .map(|(k, v)| (*k, *v as f64 / lut_len[i][j]))
                        .collect()
                })
                .collect::<Vec<HashMap<usize, f64>>>()
        })
        .collect()
}

pub fn integrate2d_ss(intensity: &Vec<f64>, lut: &Vec<Vec<HashMap<usize, f64>>>) -> Vec<f64> {
    let n_points_tth = lut.len();
    let n_points_azi = lut[0].len();

    let mut cake_intensity: Vec<f64> = vec![0.0; n_points_tth * n_points_azi];

    // add all pixel intensities for each tth and azi to the corresponding pixel in the cake
    cake_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += lut[i % n_points_tth][i / n_points_tth]
                .iter()
                .map(|(j, v)| intensity[*j] * *v)
                .sum::<f64>();
        });

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

#[cfg(test)]
pub mod tests {}
