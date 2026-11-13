use rayon::prelude::*;

// Counting (unweighted) CSR LUT builders and unified CSR types.
/// Use `crate::lut::integrate1d` / `integrate2d` to integrate any built LUT.
use crate::math::histogram;

// Unified CSR LUT types used for both counting and supersampled variants
// values=None -> unweighted (mean of contributing pixels)
// values=Some(..) -> weighted (supersampled / pixel-split)
/// CSR-based 1D LUT. If `values` is `None`, integration averages contributions.
/// If `Some(weights)`, contributions are weighted (e.g., supersampled/pixel-split).
#[derive(Clone)]
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

/// CSR-based 2D LUT (rows flattened as `azimuthal * n_tth + tth`). Same weighting rules as `Lut1d`.
pub struct Lut2d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Option<Vec<f64>>, // optional weights per entry
    pub n_tth: usize,
    pub n_azimuthal: usize,
}

/// Build an unweighted 1D CSR LUT from pixel `tth` to `n_points` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut1d_csr, integrate1d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let intensity = vec![10.0, 20.0, 30.0, 40.0];
/// let lut = build_lut1d_csr(&tth, 3);
/// let norm = vec![1.0; 4];
/// let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
/// assert_eq!(y.len(), 3);
/// ```
pub fn build_lut1d_csr(radial: &Vec<f64>, n_points: usize) -> Lut1d {
    let (min, _max, step) = histogram::min_max_step(radial, n_points);

    // Every value lies in [min, max]; the clamp only catches the maximum
    // rounding to index n_points.
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

    Lut1d {
        row_ptr,
        col_idx,
        values: None,
        hist_min: min,
        hist_step: step,
    }
}

/// Build an unweighted 1D CSR LUT with explicit radial range.
///
/// Pixels whose radial value lies outside `[radial_min, radial_max]` are left
/// out of the table; they are not assigned to the first or last bin.
pub fn build_lut1d_csr_with_range(
    radial: &Vec<f64>,
    n_points: usize,
    radial_min: f64,
    radial_max: f64,
) -> Lut1d {
    let (min, max, step) = histogram::min_max_step_from_range(radial_min, radial_max, n_points);

    let index: Vec<Option<usize>> = radial
        .par_iter()
        .map(|r| histogram::get_bin_in_range(*r, min, max, step, n_points))
        .collect();

    let mut counts = vec![0usize; n_points];
    for &idx in index.iter().flatten() {
        counts[idx] += 1;
    }

    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    let mut next = row_ptr.clone();
    let mut col_idx = vec![0usize; row_ptr[n_points]];
    for (i, bin) in index.iter().enumerate() {
        if let Some(bin) = *bin {
            col_idx[next[bin]] = i;
            next[bin] += 1;
        }
    }

    Lut1d {
        row_ptr,
        col_idx,
        values: None,
        hist_min: min,
        hist_step: step,
    }
}

/// Build an unweighted 2D CSR LUT from pixel `tth`/`azimuthal` to `(n_tth, n_azimuthal)` bins.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut2d_csr, integrate2d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let azimuthal = vec![0.0, 0.1, 0.2, 0.3];
/// let intensity = vec![1.0, 2.0, 3.0, 4.0];
/// let lut2d = build_lut2d_csr(&tth, &azimuthal, 2, 3);
/// let norm = vec![1.0; 4];
/// let y = integrate2d(&intensity, &lut2d, None, &norm);
/// assert_eq!(y.len(), 2*3);
/// ```
pub fn build_lut2d_csr(
    radial: &Vec<f64>,
    azimuthal: &Vec<f64>,
    n_radial: usize,
    n_azimuthal: usize,
) -> Lut2d {
    let (min_radial, _max_radial, step_radial) = histogram::min_max_step(radial, n_radial);
    let (min_azimuthal, _max_azimuthal, step_azimuthal) =
        histogram::min_max_step(azimuthal, n_azimuthal);

    let rows = n_radial * n_azimuthal;
    let mut counts = vec![0usize; rows];

    let row_index: Vec<usize> = radial
        .par_iter()
        .zip(azimuthal.par_iter())
        .map(|(r, a)| {
            let rb = histogram::get_bin_clamped(*r, min_radial, step_radial, n_radial);
            // The grid ends at the data's own maximum (plus a few epsilon of
            // padding), so no value lies past it; only the maximum itself can
            // round to index n_azimuthal. It belongs in the last bin, not (by
            // wrapping) in bin 0.
            let ab = histogram::get_bin_clamped(*a, min_azimuthal, step_azimuthal, n_azimuthal);
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
        n_azimuthal,
    }
}

/// Build an unweighted 2D CSR LUT from pixel `radial`/`azimuthal` to `(n_radial, n_azimuthal)` bins
/// using an explicit azimuthal binning definition.
///
/// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
/// `azimuthal_start - azimuthal_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_azimuthal`.
pub fn build_lut2d_csr_with_azimuthal_range(
    radial: &Vec<f64>,
    azimuthal: &Vec<f64>,
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_start: f64,
    azimuthal_step: f64,
) -> Lut2d {
    assert!(azimuthal_step > 0.0, "azimuthal_step must be positive");
    let (min_radial, _max_radial, step_radial) = histogram::min_max_step(radial, n_radial);
    let min_azimuthal = azimuthal_start - 0.5 * azimuthal_step;

    let rows = n_radial * n_azimuthal;
    let mut counts = vec![0usize; rows];

    let row_index: Vec<usize> = radial
        .par_iter()
        .zip(azimuthal.par_iter())
        .map(|(r, a)| {
            let rb = histogram::get_bin_clamped(*r, min_radial, step_radial, n_radial);
            let ab = histogram::get_bin_wrapped(*a, min_azimuthal, azimuthal_step, n_azimuthal);
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
        n_azimuthal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_range_keeps_every_pixel() {
        // The pixel at the maximum can round to bin index n; it belongs to the
        // last bin and must not be lost.
        for hi in [3.0, 30.8, 60.0, 1234.5] {
            for n in [7, 100, 1000, 1500, 3000] {
                let radial: Vec<f64> = (0..5000)
                    .map(|i| 1.0 + (hi - 1.0) * i as f64 / 4999.0)
                    .collect();
                let lut = build_lut1d_csr(&radial, n);
                assert_eq!(lut.col_idx.len(), radial.len(), "hi {hi}, n {n}");
                let last = &lut.col_idx[lut.row_ptr[n - 1]..];
                assert!(last.contains(&4999), "hi {hi}, n {n}");
            }
        }
    }

    #[test]
    fn automatic_azimuthal_range_keeps_the_maximum_in_the_last_bin() {
        // The largest azimuth can round to bin index n_azimuthal. The grid ends
        // there, so it belongs to the last azimuthal bin, not (wrapped) to bin 0.
        let azimuthal: Vec<f64> = (0..5000).map(|i| 1.0 + 59.0 * i as f64 / 4999.0).collect();
        let radial = vec![0.5; azimuthal.len()];
        for n in [7, 10, 36, 100] {
            let lut = build_lut2d_csr(&radial, &azimuthal, 1, n);
            assert_eq!(lut.col_idx.len(), azimuthal.len(), "n {n}");
            let first = &lut.col_idx[lut.row_ptr[0]..lut.row_ptr[1]];
            let last = &lut.col_idx[lut.row_ptr[n - 1]..lut.row_ptr[n]];
            assert!(!first.contains(&4999), "n {n}: maximum wrapped to bin 0");
            assert!(last.contains(&4999), "n {n}");
            // Only the minimum end of the range is in the first bin.
            assert!(
                first.iter().all(|&p| azimuthal[p] < 1.0 + 59.0 / n as f64),
                "n {n}"
            );
        }
    }

    #[test]
    fn explicit_range_includes_both_limits_at_any_magnitude() {
        for hi in [3.0, 30.8, 60.0, 1234.5] {
            for n in [7, 100, 1000, 1500, 3000] {
                let radial = vec![1.0, hi, hi + hi * 1e-12, 1.0 - 1e-12];
                let lut = build_lut1d_csr_with_range(&radial, n, 1.0, hi);
                assert_eq!(lut.col_idx, vec![0, 1], "hi {hi}, n {n}");
                assert_eq!(lut.row_ptr[1], 1, "hi {hi}, n {n}");
                assert_eq!(lut.row_ptr[n - 1], 1, "hi {hi}, n {n}");
            }
        }
    }

    #[test]
    fn explicit_range_drops_pixels_outside_it() {
        // Bins [1, 2) and [2, 3]. Values below 1 or above 3 belong to no bin.
        let radial = vec![0.5, 1.5, 2.5, 3.5, 1.0, 3.0, -4.0, f64::INFINITY];
        let lut = build_lut1d_csr_with_range(&radial, 2, 1.0, 3.0);
        assert_eq!(lut.row_ptr, vec![0, 2, 4]);
        assert_eq!(lut.col_idx, vec![1, 4, 2, 5]);
    }
}
