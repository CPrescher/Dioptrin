use rayon::prelude::*;

use crate::math::histogram;
use crate::{geometry::Geometry, lut};

/// CSR-like LUT for 1D integration (unweighted)
/// row_ptr len = n_bins + 1, col_idx len = n_nonzeros (pixels assigned)
#[allow(dead_code)]
pub struct CsrLut {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

/// CSR-like LUT for 1D integration with supersampling (weighted)
pub struct CsrLutSs {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

/// CSR-like LUT for 2D integration (unweighted), rows are flattened as: row = azi * n_tth + tth
pub struct CsrLut2d {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub n_tth: usize,
    pub n_azi: usize,
}

/// CSR-like LUT for 2D integration with supersampling (weighted)
pub struct CsrLut2dSs {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
    pub n_tth: usize,
    pub n_azi: usize,
}

/// Build CSR LUT from tth values and number of bins.
/// The per-bin indices are in ascending order for improved locality.
pub fn lut_csr_from_tth(tth: &Vec<f64>, n_points: usize) -> (CsrLut, f64, f64) {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    // Bin index per pixel
    let index: Vec<usize> = tth
        .par_iter()
        .map(|tth| histogram::get_bin(*tth, min, step))
        .collect();

    // Count per bin
    let mut counts = vec![0usize; n_points];
    for &idx in &index {
        counts[idx] += 1;
    }

    // Prefix sum to row_ptr
    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    // Fill col_idx in pixel order; this yields ascending indices within each bin
    let mut next = row_ptr.clone();
    let mut col_idx = vec![0usize; tth.len()];
    for (i, &bin) in index.iter().enumerate() {
        let pos = next[bin];
        col_idx[pos] = i;
        next[bin] += 1;
    }

    (CsrLut { row_ptr, col_idx }, min, step)
}

/// Integrate intensity using CSR LUT (unweighted mean per bin)
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

/// Build CSR 2D LUT from tth/azi values and bin sizes
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

    // First pass: compute row index per pixel (in parallel), then count sequentially
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

    // Prefix sum
    let mut row_ptr = vec![0usize; rows + 1];
    for i in 0..rows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    // Reset counts to use as write cursors
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

/// Integrate intensity using CSR 2D LUT (unweighted mean per bin), returns flattened cake
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

/// Build CSR LUT for supersampled 1D integration by converting the existing HashMap-based LUT.
/// Leaves building logic unchanged; focuses on cache-friendly storage for integration.
pub fn lut_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
) -> (CsrLutSs, f64, f64) {
    // Reuse existing builder and convert to CSR
    let lut_ss = lut::lut_ss(geometry, tth, n_points, n_ss);
    let (hist_min, _hist_max, hist_step) =
        crate::math::histogram::ss_tth_min_max_step(geometry, tth, n_points, n_ss);

    // Determine total nnz and prefix
    let mut row_ptr = Vec::with_capacity(n_points + 1);
    row_ptr.push(0);
    for bin in &lut_ss {
        row_ptr.push(row_ptr.last().unwrap() + bin.len());
    }

    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];

    // Fill per bin in ascending key order for locality
    for (i, bin) in lut_ss.into_iter().enumerate() {
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

/// Integrate intensity using CSR supersampled LUT (weighted sum per bin)
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

/// Build CSR 2D supersampled LUT by converting the HashMap-based LUT
pub fn lut2d_ss_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
    n_ss: usize,
) -> CsrLut2dSs {
    let lut_hm = lut::lut_2d_ss(geometry, tth, azi, n_tth, n_azi, n_ss);
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

    // Fill rows in the same flattening order: ab outer, tb inner
    let mut _cursor = 0usize;
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
            _cursor = end;
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

/// Integrate intensity using CSR 2D supersampled LUT (weighted sum)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut;
    use crate::poni::Poni;
    use crate::geometry::Geometry;

    #[test]
    fn test_integrate_csr_matches_vecvec() {
        let tth = vec![0.1, 0.2, 0.21, 0.4, 0.41, 0.9];
        let intensity = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let n_points = 3;

        // original LUT path
        let lut_vv = lut::lut(&tth, n_points);
        let (min, _max, step) = histogram::min_max_step(&tth, n_points);
        let (_bins_a, out_a) = lut::integrate(&intensity, &lut_vv, min, step);

        // CSR path
        let (lut_csr, min2, step2) = lut_csr_from_tth(&tth, n_points);
        let (_bins_b, out_b) = integrate_csr(&intensity, &lut_csr, min2, step2);

        assert_eq!(out_a.len(), out_b.len());
        for i in 0..out_a.len() {
            assert!((out_a[i] - out_b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_integrate_ss_csr_matches_hashmap() {
        // Small synthetic geometry
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.2;
        poni.poni1 = 0.0;
        poni.poni2 = 0.0;
        let mut geom = Geometry::new(0, 0, poni);
        geom.dim1 = 8;
        geom.dim2 = 7;

        let tth = geom.calc_tth_all();
        let n_points = 12;
        let n_ss = 3;

        // Build both LUTs
        let lut_hash = lut::lut_ss(&geom, &tth, n_points, n_ss);
        let (lut_csr_ss, _min, _step) = lut_ss_csr(&geom, &tth, n_points, n_ss);

        // Synthetic intensity
        let intensity: Vec<f64> = (0..geom.size()).map(|i| (i as f64) * 0.1 + 1.0).collect();

        // Integrate both ways
        let (_bins_a, out_a) = lut::integrate_ss(&intensity, &lut_hash, 0.0, 1.0);
        let (_bins_b, out_b) = integrate_ss_csr(&intensity, &lut_csr_ss, 0.0, 1.0);

        assert_eq!(out_a.len(), out_b.len());
        for i in 0..out_a.len() {
            assert!((out_a[i] - out_b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_integrate2d_csr_matches_vecvec() {
        // Small synthetic geometry
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.2;
        poni.poni1 = 0.0;
        poni.poni2 = 0.0;
        let mut geom = Geometry::new(0, 0, poni);
        geom.dim1 = 16;
        geom.dim2 = 12;

        let tth = geom.calc_tth_all();
        let chi = geom.calc_chi_all();

        let n_tth = 8;
        let n_azi = 18;

        // Original LUT 2D path
        let lut2d = lut::lut_2d(&tth, &chi, n_tth, n_azi);
        let int_classic = lut::integrate2d(&make_intensity(geom.size()), &lut2d);

        // CSR path
        let csr2d = lut2d_csr_from_tth_azi(&tth, &chi, n_tth, n_azi);
        let int_csr = integrate2d_csr(&make_intensity(geom.size()), &csr2d);

        assert_eq!(int_classic.len(), int_csr.len());
        for i in 0..int_classic.len() {
            assert!((int_classic[i] - int_csr[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_integrate2d_ss_csr_matches_hashmap() {
        // Small synthetic geometry
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.2;
        poni.poni1 = 0.0;
        poni.poni2 = 0.0;
        let mut geom = Geometry::new(0, 0, poni);
        geom.dim1 = 16;
        geom.dim2 = 12;

        let tth = geom.calc_tth_all();
        let chi = geom.calc_chi_all();

        let n_tth = 8;
        let n_azi = 18;
        let n_ss = 3;

        // Original supersampled LUT 2D path
        let lut2d_ss = lut::lut_2d_ss(&geom, &tth, &chi, n_tth, n_azi, n_ss);
        let int_classic = lut::integrate2d_ss(&make_intensity(geom.size()), &lut2d_ss);

        // CSR SS path
        let csr2d_ss = lut2d_ss_csr(&geom, &tth, &chi, n_tth, n_azi, n_ss);
        let int_csr = integrate2d_ss_csr(&make_intensity(geom.size()), &csr2d_ss);

        assert_eq!(int_classic.len(), int_csr.len());
        for i in 0..int_classic.len() {
            assert!((int_classic[i] - int_csr[i]).abs() < 1e-10);
        }
    }

    fn make_intensity(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.5 + (i as f64) * 0.01).collect()
    }
}
