use rayon::prelude::*;

use crate::math::histogram;

// Unified CSR LUT types used for both counting and supersampled variants
// values=None -> unweighted (mean of contributing pixels)
// values=Some(..) -> weighted (supersampled / pixel-split)
/// CSR-based 1D LUT. If `values` is `None`, integration averages contributions.
/// If `Some(weights)`, contributions are weighted (e.g., supersampled/pixel-split).
pub struct Lut1d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Option<Vec<f64>>, // optional weights per entry
    pub hist_min: f64,
    pub hist_step: f64,
}

impl Lut1d {
    /// Number of bins in the 1D LUT.
    pub fn n_points(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// Compute the histogram bin centers for this LUT.
    pub fn bins(&self) -> Vec<f64> {
        histogram::bins(self.n_points(), self.hist_min, self.hist_step)
    }
}

/// CSR-based 2D LUT (rows flattened as `azi * n_tth + tth`). Same weighting rules as `Lut1d`.
pub struct Lut2d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Option<Vec<f64>>, // optional weights per entry
    pub n_tth: usize,
    pub n_azi: usize,
}

/// Build an unweighted 1D CSR LUT from pixel `tth` to `n_points` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut1d_csr, integrate_lut1d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let intensity = vec![10.0, 20.0, 30.0, 40.0];
/// let lut = build_lut1d_csr(&tth, 3);
/// let (_bins, y) = integrate_lut1d(&intensity, &lut);
/// assert_eq!(y.len(), 3);
/// ```
pub fn build_lut1d_csr(tth: &Vec<f64>, n_points: usize) -> Lut1d {
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

    Lut1d { row_ptr, col_idx, values: None, hist_min: min, hist_step: step }
}

/// Integrate using either unweighted (mean) or weighted 1D CSR LUT.
///
/// See `build_lut1d_csr` for an example.
pub fn integrate_lut1d(intensity: &Vec<f64>, lut: &Lut1d) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let pattern_intensity: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            match &lut.values {
                Some(values) => values[start..end]
                    .iter()
                    .zip(lut.col_idx[start..end].iter())
                    .map(|(w, &j)| w * intensity[j])
                    .sum::<f64>(),
                None => {
                    if start == end { 0.0 } else {
                        let sum: f64 = lut.col_idx[start..end]
                            .iter()
                            .map(|&j| intensity[j])
                            .sum();
                        sum / (end - start) as f64
                    }
                }
            }
        })
        .collect();
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity)
}

/// Build an unweighted 2D CSR LUT from pixel `tth`/`azi` to `(n_tth, n_azi)` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut2d_csr, integrate_lut2d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let chi = vec![0.0, 0.1, 0.2, 0.3];
/// let intensity = vec![1.0, 2.0, 3.0, 4.0];
/// let lut2d = build_lut2d_csr(&tth, &chi, 2, 3);
/// let y = integrate_lut2d(&intensity, &lut2d);
/// assert_eq!(y.len(), 2*3);
/// ```
pub fn build_lut2d_csr(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> Lut2d {
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

    Lut2d {
        row_ptr,
        col_idx,
        values: None,
        n_tth,
        n_azi,
    }
}

/// Integrate using either unweighted (mean) or weighted 2D CSR LUT.
///
/// See `build_lut2d_csr` for an example.
pub fn integrate_lut2d(intensity: &Vec<f64>, lut: &Lut2d) -> Vec<f64> {
    let rows = lut.row_ptr.len() - 1;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            match &lut.values {
                Some(values) => values[start..end]
                    .iter()
                    .zip(lut.col_idx[start..end].iter())
                    .map(|(w, &j)| w * intensity[j])
                    .sum::<f64>(),
                None => {
                    if start == end { 0.0 } else {
                        let sum: f64 = lut.col_idx[start..end]
                            .iter()
                            .map(|&j| intensity[j])
                            .sum();
                        sum / (end - start) as f64
                    }
                }
            }
        })
        .collect()
}
