use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::math::histogram;
use crate::lut::counting_csr::{Lut1d, Lut2d};

/// Build a supersampled 1D CSR LUT directly (no HashMap in the public type).
///
/// Example (pseudo-code)
/// ```no_run
/// # use dioptrin::geometry::Geometry;
/// # let geometry: Geometry = unimplemented!();
/// # let tth: Vec<f64> = unimplemented!();
/// let lut = dioptrin::lut::build_lut1d_ss_csr(&geometry, &tth, 1024, 4);
/// # let intensity: Vec<f64> = unimplemented!();
/// let (_bins, y) = dioptrin::lut::integrate_lut1d(&intensity, &lut);
/// ```
pub fn build_lut1d_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
) -> Lut1d {
    let (hist_min, _hist_max, hist_step) =
        histogram::ss_tth_min_max_step(geometry, tth, n_points, n_ss);

    // Collect all (bin, pixel) occurrences from supersampling
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    pairs.reserve(geometry.dim1 * geometry.dim2 * n_ss * n_ss);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = crate::lut::common::subgrid(row, geometry.dim2, n_ss);
        let tth_ss = geometry.calc_tth(&d1, &d2);
        let bins: Vec<usize> = tth_ss
            .par_iter()
            .map(|&v| histogram::get_bin(v, hist_min, hist_step))
            .collect();
        for (i, b) in bins.into_iter().enumerate() {
            pairs.push((b, actual_index[i]));
        }
    }

    // Sort by (bin, pixel) and compress to unique with counts
    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(n_points + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_bin = pairs[i].0;
        while row_ptr.len() - 1 < current_bin {
            row_ptr.push(row_ptr.last().copied().unwrap());
        }
        let mut bin_total: usize = 0;
        let bin_start = col_idx.len();
        while i < pairs.len() && pairs[i].0 == current_bin {
            let pixel = pairs[i].1;
            let mut cnt = 1usize;
            i += 1;
            while i < pairs.len() && pairs[i].0 == current_bin && pairs[i].1 == pixel {
                cnt += 1;
                i += 1;
            }
            col_idx.push(pixel);
            values.push(cnt as f64);
            bin_total += cnt;
        }
        if bin_total > 0 { let denom = bin_total as f64; for w in &mut values[bin_start..] { *w /= denom; } }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < n_points + 1 { row_ptr.push(col_idx.len()); }

    Lut1d { row_ptr, col_idx, values: Some(values), hist_min, hist_step }
}

/// Build a supersampled 2D CSR LUT directly (no HashMap in the public type).
///
/// Example (pseudo-code)
/// ```no_run
/// # use dioptrin::geometry::Geometry;
/// # let geometry: Geometry = unimplemented!();
/// # let tth: Vec<f64> = unimplemented!();
/// # let chi: Vec<f64> = unimplemented!();
/// let lut2d = dioptrin::lut::build_lut2d_ss_csr(&geometry, &tth, &chi, 1024, 360, 4);
/// # let intensity: Vec<f64> = unimplemented!();
/// let y = dioptrin::lut::integrate_lut2d(&intensity, &lut2d);
/// assert_eq!(y.len(), 1024*360);
/// ```
pub fn build_lut2d_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
    n_ss: usize,
) -> Lut2d {
    let (tth_min, _tth_max, tth_step) = histogram::ss_tth_min_max_step(geometry, tth, n_tth, n_ss);
    let (azi_min, _azi_max, azi_step) =
        histogram::supersampled_chi_min_max_step(geometry, azi, n_azi, n_ss);

    let rows = n_tth * n_azi;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    pairs.reserve(geometry.dim1 * geometry.dim2 * n_ss * n_ss);

    for r in 0..geometry.dim1 {
        let (d1, d2, actual_index) = crate::lut::common::subgrid(r, geometry.dim2, n_ss);
        let (tth_ss, azi_ss) = geometry.calc_tth_and_chi(&d1, &d2);
        let t_bins: Vec<usize> = tth_ss
            .par_iter()
            .map(|&v| histogram::get_bin(v, tth_min, tth_step))
            .collect();
        let a_bins: Vec<usize> = azi_ss
            .par_iter()
            .map(|&v| histogram::get_bin(v, azi_min, azi_step))
            .collect();
        for i in 0..t_bins.len() {
            let tb = t_bins[i];
            let ab = a_bins[i] % n_azi;
            let row = ab * n_tth + tb;
            pairs.push((row, actual_index[i]));
        }
    }

    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_row = pairs[i].0;
        while row_ptr.len() - 1 < current_row { row_ptr.push(col_idx.len()); }
        let bin_start = col_idx.len();
        let mut bin_total = 0usize;
        while i < pairs.len() && pairs[i].0 == current_row {
            let pixel = pairs[i].1;
            let mut cnt = 1usize;
            i += 1;
            while i < pairs.len() && pairs[i].0 == current_row && pairs[i].1 == pixel {
                cnt += 1;
                i += 1;
            }
            col_idx.push(pixel);
            values.push(cnt as f64);
            bin_total += cnt;
        }
        if bin_total > 0 { let denom = bin_total as f64; for w in &mut values[bin_start..] { *w /= denom; } }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < rows + 1 { row_ptr.push(col_idx.len()); }

    Lut2d { row_ptr, col_idx, values: Some(values), n_tth, n_azi }
}
