use rayon::prelude::*;

use crate::math::histogram;

pub struct CsrLut {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

pub struct CsrLut2d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub n_tth: usize,
    pub n_azi: usize,
}

pub fn lut_csr_from_tth(tth: &Vec<f64>, n_points: usize) -> (CsrLut, f64, f64) {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    let index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min, step))
        .collect();

    let mut counts = vec![0usize; n_points];
    for &idx in &index {
        counts[idx] += 1;
    }

    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    let mut next = row_ptr.clone();
    let mut col_idx = vec![0usize; tth.len()];
    for (i, &bin) in index.iter().enumerate() {
        let pos = next[bin];
        col_idx[pos] = i;
        next[bin] += 1;
    }

    (CsrLut { row_ptr, col_idx }, min, step)
}

pub fn integrate_csr(
    intensity: &Vec<f64>,
    lut: &CsrLut,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let pattern_intensity: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                0.0
            } else {
                let sum: f64 = lut.col_idx[start..end]
                    .iter()
                    .map(|&j| intensity[j])
                    .sum();
                sum / (end - start) as f64
            }
        })
        .collect();
    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);
    (pattern_tth, pattern_intensity)
}

pub fn lut2d_csr_from_tth_azi(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> CsrLut2d {
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    let rows = n_tth * n_azi;
    let mut counts = vec![0usize; rows];

    let row_index: Vec<usize> = tth
        .par_iter()
        .zip(azi.par_iter())
        .map(|(t, a)| {
            let tb = histogram::get_bin(*t, min_tth, step_tth);
            let ab = histogram::get_bin(*a, min_azi, step_azi) % n_azi;
            ab * n_tth + tb
        })
        .collect();
    for &row in &row_index {
        counts[row] += 1;
    }

    let mut row_ptr = vec![0usize; rows + 1];
    for i in 0..rows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    let mut next = row_ptr.clone();
    let mut col_idx = vec![0usize; tth.len()];
    for (i, &row) in row_index.iter().enumerate() {
        let pos = next[row];
        col_idx[pos] = i;
        next[row] += 1;
    }

    CsrLut2d {
        row_ptr,
        col_idx,
        n_tth,
        n_azi,
    }
}

pub fn integrate2d_csr(intensity: &Vec<f64>, lut: &CsrLut2d) -> Vec<f64> {
    let rows = lut.row_ptr.len() - 1;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            if start == end {
                0.0
            } else {
                let sum: f64 = lut.col_idx[start..end]
                    .iter()
                    .map(|&j| intensity[j])
                    .sum();
                sum / (end - start) as f64
            }
        })
        .collect()
}

