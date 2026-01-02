use rayon::prelude::*;

// Counting (unweighted) CSR LUT builders and unified CSR types.
/// Use `crate::lut::integrate1d` / `integrate2d` to integrate any built LUT.
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

/// CSR-based 2D LUT (rows flattened as `chi * n_tth + tth`). Same weighting rules as `Lut1d`.
pub struct Lut2d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Option<Vec<f64>>, // optional weights per entry
    pub n_tth: usize,
    pub n_chi: usize,
}

/// Build an unweighted 1D CSR LUT from pixel `tth` to `n_points` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut1d_csr, integrate1d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let intensity = vec![10.0, 20.0, 30.0, 40.0];
/// let lut = build_lut1d_csr(&tth, 3);
/// let (_bins, y) = integrate1d(&intensity, &lut, None);
/// assert_eq!(y.len(), 3);
/// ```
pub fn build_lut1d_csr(radial: &Vec<f64>, n_points: usize) -> Lut1d {
    let (min, _max, step) = histogram::min_max_step(radial, n_points);

    let index: Vec<usize> = radial
        .par_iter()
        .map(|r| histogram::get_bin_clamped(*r, min, step, n_points))
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
    let mut col_idx = vec![0usize; radial.len()];
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
// integrate1d moved to src/lut/integrate.rs

/// Build an unweighted 2D CSR LUT from pixel `tth`/`chi` to `(n_tth, n_chi)` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut2d_csr, integrate2d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let chi = vec![0.0, 0.1, 0.2, 0.3];
/// let intensity = vec![1.0, 2.0, 3.0, 4.0];
/// let lut2d = build_lut2d_csr(&tth, &chi, 2, 3);
/// let y = integrate2d(&intensity, &lut2d, None);
/// assert_eq!(y.len(), 2*3);
/// ```
pub fn build_lut2d_csr(
    radial: &Vec<f64>,
    chi: &Vec<f64>,
    n_radial: usize,
    n_chi: usize,
) -> Lut2d {
    let (min_radial, _max_radial, step_radial) = histogram::min_max_step(radial, n_radial);
    let (min_chi, _max_chi, step_chi) = histogram::min_max_step(chi, n_chi);

    let rows = n_radial * n_chi;
    let mut counts = vec![0usize; rows];

    let row_index: Vec<usize> = radial
        .par_iter()
        .zip(chi.par_iter())
        .map(|(r, a)| {
            let rb = histogram::get_bin_clamped(*r, min_radial, step_radial, n_radial);
            let ab = histogram::get_bin(*a, min_chi, step_chi) % n_chi;
            ab * n_radial + rb
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
    let mut col_idx = vec![0usize; radial.len()];
    for (i, &row) in row_index.iter().enumerate() {
        let pos = next[row];
        col_idx[pos] = i;
        next[row] += 1;
    }

    Lut2d {
        row_ptr,
        col_idx,
        values: None,
        n_tth: n_radial,
        n_chi,
    }
}

/// Build an unweighted 2D CSR LUT from pixel `radial`/`chi` to `(n_radial, n_chi)` bins
/// using an explicit chi binning definition.
///
/// `chi_start` is the center of the first chi bin; the effective start edge is
/// `chi_start - chi_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_chi`.
pub fn build_lut2d_csr_with_chi_range(
    radial: &Vec<f64>,
    chi: &Vec<f64>,
    n_radial: usize,
    n_chi: usize,
    chi_start: f64,
    chi_step: f64,
) -> Lut2d {
    assert!(chi_step > 0.0, "chi_step must be positive");
    let (min_radial, _max_radial, step_radial) = histogram::min_max_step(radial, n_radial);
    let min_chi = chi_start - 0.5 * chi_step;

    let rows = n_radial * n_chi;
    let mut counts = vec![0usize; rows];

    let row_index: Vec<usize> = radial
        .par_iter()
        .zip(chi.par_iter())
        .map(|(r, a)| {
            let rb = histogram::get_bin_clamped(*r, min_radial, step_radial, n_radial);
            let ab = histogram::get_bin_wrapped(*a, min_chi, chi_step, n_chi);
            ab * n_radial + rb
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
    let mut col_idx = vec![0usize; radial.len()];
    for (i, &row) in row_index.iter().enumerate() {
        let pos = next[row];
        col_idx[pos] = i;
        next[row] += 1;
    }

    Lut2d {
        row_ptr,
        col_idx,
        values: None,
        n_tth: n_radial,
        n_chi,
    }
}
