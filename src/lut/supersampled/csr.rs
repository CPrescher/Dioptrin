use rayon::prelude::*;

use crate::math::histogram;
use crate::geometry::Geometry;
use crate::lut::{lut_ss, lut_2d_ss};

pub struct CsrLutSs {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

pub struct CsrLut2dSs {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
    pub n_tth: usize,
    pub n_azi: usize,
}

pub fn lut_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
) -> (CsrLutSs, f64, f64) {
    let lut_hm = lut_ss(geometry, tth, n_points, n_ss);
    let (hist_min, _hist_max, hist_step) =
        histogram::ss_tth_min_max_step(geometry, tth, n_points, n_ss);

    let mut row_ptr = Vec::with_capacity(n_points + 1);
    row_ptr.push(0);
    for bin in &lut_hm {
        row_ptr.push(row_ptr.last().unwrap() + bin.len());
    }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];

    for (i, bin) in lut_hm.into_iter().enumerate() {
        let start = row_ptr[i];
        if start == row_ptr[i + 1] {
            continue;
        }
        let mut entries: Vec<(usize, f64)> = bin.into_iter().collect();
        entries.sort_unstable_by_key(|e| e.0);
        for (j, (col, val)) in entries.into_iter().enumerate() {
            col_idx[start + j] = col;
            values[start + j] = val;
        }
    }

    (
        CsrLutSs {
            row_ptr,
            col_idx,
            values,
        },
        hist_min,
        hist_step,
    )
}

pub fn integrate_ss_csr(
    intensity: &Vec<f64>,
    lut: &CsrLutSs,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let pattern_intensity: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            lut.values[start..end]
                .iter()
                .zip(lut.col_idx[start..end].iter())
                .map(|(w, &j)| w * intensity[j])
                .sum::<f64>()
        })
        .collect();
    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);
    (pattern_tth, pattern_intensity)
}

pub fn lut2d_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
    n_ss: usize,
) -> CsrLut2dSs {
    let lut_hm = lut_2d_ss(geometry, tth, azi, n_tth, n_azi, n_ss);
    let rows = n_tth * n_azi;
    let mut row_ptr = Vec::with_capacity(rows + 1);
    row_ptr.push(0);
    for ab in 0..n_azi {
        for tb in 0..n_tth {
            let len = lut_hm[tb][ab].len();
            row_ptr.push(row_ptr.last().unwrap() + len);
        }
    }

    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];

    for ab in 0..n_azi {
        for tb in 0..n_tth {
            let start = row_ptr[ab * n_tth + tb];
            let end = row_ptr[ab * n_tth + tb + 1];
            if start == end {
                continue;
            }
            let mut entries: Vec<(usize, f64)> = lut_hm[tb][ab].iter().map(|(k, v)| (*k, *v)).collect();
            entries.sort_unstable_by_key(|e| e.0);
            for (j, (col, val)) in entries.into_iter().enumerate() {
                col_idx[start + j] = col;
                values[start + j] = val;
            }
        }
    }

    CsrLut2dSs {
        row_ptr,
        col_idx,
        values,
        n_tth,
        n_azi,
    }
}

pub fn integrate2d_ss_csr(intensity: &Vec<f64>, lut: &CsrLut2dSs) -> Vec<f64> {
    let rows = lut.row_ptr.len() - 1;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            lut.values[start..end]
                .iter()
                .zip(lut.col_idx[start..end].iter())
                .map(|(w, &j)| w * intensity[j])
                .sum::<f64>()
        })
        .collect()
}
