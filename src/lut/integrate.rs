//! Shared LUT integrators used by all methods (counting, pixel-split, supersampled).
//! These functions operate on the unified CSR types (`Lut1d`, `Lut2d`).
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::math::histogram;

const N_AZIMUTHAL_SUB_BINS: usize = 3600; // 0.1 degree resolution

/// Compute per-bin azimuthal coverage fraction for a 1D LUT.
///
/// For each radial bin, determines what fraction of the full 2pi azimuthal range
/// is covered by detector pixels. Returns a Vec of coverage fractions in [0, 1].
/// Empty bins return 1.0 to avoid division by zero in normalization.
pub fn compute_azimuthal_coverage(lut: &Lut1d, azimuth: &[f64]) -> Vec<f64> {
    let n_points = lut.row_ptr.len() - 1;
    (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                return 1.0;
            }
            let mut occupied = vec![false; N_AZIMUTHAL_SUB_BINS];
            for k in start..end {
                let pixel = lut.col_idx[k];
                let az = azimuth[pixel];
                // Map [-pi, pi] to [0, N_SUB_BINS)
                let sub_bin = ((az + PI) / (2.0 * PI) * N_AZIMUTHAL_SUB_BINS as f64).floor()
                    as usize;
                let sub_bin = sub_bin.min(N_AZIMUTHAL_SUB_BINS - 1);
                occupied[sub_bin] = true;
            }
            let count = occupied.iter().filter(|&&b| b).count();
            count as f64 / N_AZIMUTHAL_SUB_BINS as f64
        })
        .collect()
}

/// Integrate using either unweighted (mean) or weighted 1D CSR LUT.
///
/// Examples
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate1d};
/// // Simple pixel-split (weighted) 1D example
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut = build_lut1d_ps_csr(&geometry, &tth, 4, RadialUnit::TthRad);
/// let (_x, y) = integrate1d(&intensity, &lut, None);
/// assert_eq!(y.len(), 4);
/// ```
pub fn integrate1d(intensity: &[f64], lut: &Lut1d, mask: Option<&[u8]>) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let pattern_intensity: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            match &lut.values {
                Some(values) => {
                    let mut sum = 0.0;
                    let mut k = start;
                    // Safety: row_ptr bounds ensure k < values/col_idx length; col_idx values index intensity.
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum += w * *intensity_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum += w * *intensity_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    sum
                }
                None => {
                    if start == end {
                        0.0
                    } else {
                        let mut sum = 0.0;
                        let mut count = 0usize;
                        let mut k = start;
                        unsafe {
                            let col_ptr = lut.col_idx.as_ptr();
                            let intensity_ptr = intensity.as_ptr();
                            match mask {
                                Some(mask) => {
                                    let mask_ptr = mask.as_ptr();
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        sum += *intensity_ptr.add(j);
                                        count += 1;
                                        k += 1;
                                    }
                                }
                            }
                        }
                        if count == 0 { 0.0 } else { sum / count as f64 }
                    }
                }
            }
        })
        .collect();
    let mut pattern_intensity = pattern_intensity;
    if let Some(coverage) = &lut.azimuthal_coverage {
        for (y, &cov) in pattern_intensity.iter_mut().zip(coverage.iter()) {
            if cov > 0.0 {
                *y /= cov;
            }
        }
    }
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity)
}

/// Integrate using either unweighted (mean) or weighted 1D CSR LUT, returning errors.
///
/// The returned error is the per-bin standard deviation computed from per-pixel variances.
pub fn integrate1d_with_errors(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            match &lut.values {
                Some(values) => {
                    let mut sum = 0.0;
                    let mut sum_var = 0.0;
                    let mut k = start;
                    // Safety: row_ptr bounds ensure k < values/col_idx length; col_idx values index intensity.
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum, sum_var)
                }
                None => {
                    if start == end {
                        (0.0, 0.0)
                    } else {
                        let mut sum = 0.0;
                        let mut sum_var = 0.0;
                        let mut count = 0usize;
                        let mut k = start;
                        unsafe {
                            let col_ptr = lut.col_idx.as_ptr();
                            let intensity_ptr = intensity.as_ptr();
                            let variance_ptr = variance.as_ptr();
                            match mask {
                                Some(mask) => {
                                    let mask_ptr = mask.as_ptr();
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            sum_var += *variance_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        sum += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        count += 1;
                                        k += 1;
                                    }
                                }
                            }
                        }
                        if count == 0 {
                            (0.0, 0.0)
                        } else {
                            let count_f = count as f64;
                            (sum / count_f, sum_var / (count_f * count_f))
                        }
                    }
                }
            }
        })
        .collect();
    let mut pattern_intensity = Vec::with_capacity(n_points);
    let mut pattern_sigma = Vec::with_capacity(n_points);
    for (y, var) in results {
        pattern_intensity.push(y);
        pattern_sigma.push(var.max(0.0).sqrt());
    }
    if let Some(coverage) = &lut.azimuthal_coverage {
        for i in 0..n_points {
            if coverage[i] > 0.0 {
                pattern_intensity[i] /= coverage[i];
                pattern_sigma[i] /= coverage[i];
            }
        }
    }
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity, pattern_sigma)
}

/// Integrate using either unweighted (mean) or weighted 2D CSR LUT.
///
/// Example
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let azimuthal = geometry.calc_azimuth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut2d = build_lut2d_ps_csr(&geometry, &tth, &azimuthal, 4, 4, RadialUnit::TthRad);
/// let y = integrate2d(&intensity, &lut2d, None);
/// assert_eq!(y.len(), 4 * 4);
/// ```
pub fn integrate2d(intensity: &[f64], lut: &Lut2d, mask: Option<&[u8]>) -> Vec<f64> {
    let rows = lut.row_ptr.len() - 1;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            match &lut.values {
                Some(values) => {
                    let mut sum = 0.0;
                    let mut k = start;
                    // Safety: row_ptr bounds ensure k < values/col_idx length; col_idx values index intensity.
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum += w * *intensity_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum += w * *intensity_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    sum
                }
                None => {
                    if start == end {
                        0.0
                    } else {
                        let mut sum = 0.0;
                        let mut count = 0usize;
                        let mut k = start;
                        unsafe {
                            let col_ptr = lut.col_idx.as_ptr();
                            let intensity_ptr = intensity.as_ptr();
                            match mask {
                                Some(mask) => {
                                    let mask_ptr = mask.as_ptr();
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        sum += *intensity_ptr.add(j);
                                        count += 1;
                                        k += 1;
                                    }
                                }
                            }
                        }
                        if count == 0 { 0.0 } else { sum / count as f64 }
                    }
                }
            }
        })
        .collect()
}

/// Integrate using either unweighted (mean) or weighted 2D CSR LUT, returning errors.
///
/// The returned error is the per-bin standard deviation computed from per-pixel variances.
pub fn integrate2d_with_errors(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut2d,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
    let rows = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            match &lut.values {
                Some(values) => {
                    let mut sum = 0.0;
                    let mut sum_var = 0.0;
                    let mut k = start;
                    // Safety: row_ptr bounds ensure k < values/col_idx length; col_idx values index intensity.
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum, sum_var)
                }
                None => {
                    if start == end {
                        (0.0, 0.0)
                    } else {
                        let mut sum = 0.0;
                        let mut sum_var = 0.0;
                        let mut count = 0usize;
                        let mut k = start;
                        unsafe {
                            let col_ptr = lut.col_idx.as_ptr();
                            let intensity_ptr = intensity.as_ptr();
                            let variance_ptr = variance.as_ptr();
                            match mask {
                                Some(mask) => {
                                    let mask_ptr = mask.as_ptr();
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            sum_var += *variance_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k);
                                        sum += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        count += 1;
                                        k += 1;
                                    }
                                }
                            }
                        }
                        if count == 0 {
                            (0.0, 0.0)
                        } else {
                            let count_f = count as f64;
                            (sum / count_f, sum_var / (count_f * count_f))
                        }
                    }
                }
            }
        })
        .collect();
    let mut intensity_out = Vec::with_capacity(rows);
    let mut sigma_out = Vec::with_capacity(rows);
    for (y, var) in results {
        intensity_out.push(y);
        sigma_out.push(var.max(0.0).sqrt());
    }
    (intensity_out, sigma_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_integrate1d_with_errors_matches_unweighted() {
        let tth = vec![0.1, 0.2, 0.3, 0.4];
        let intensity = vec![1.0, 2.0, 3.0, 4.0];
        let variance = intensity.clone();
        let lut = crate::lut::build_lut1d_csr(&tth, 2);
        let (_bins, y) = integrate1d(&intensity, &lut, None);
        let (_bins_err, y_err, _sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate1d_with_errors_matches_weighted() {
        let intensity = vec![2.0, 4.0, 6.0, 8.0];
        let variance = intensity.clone();
        let lut = Lut1d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 2, 3],
            values: Some(vec![0.5, 0.5, 0.25, 0.75]),
            hist_min: 0.0,
            hist_step: 1.0,
            azimuthal_coverage: None,
        };
        let (_bins, y) = integrate1d(&intensity, &lut, None);
        let (_bins_err, y_err, _sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate2d_with_errors_matches_unweighted() {
        let tth = vec![0.1, 0.2, 0.3, 0.4];
        let azimuthal = vec![0.0, 0.1, 0.2, 0.3];
        let intensity = vec![1.0, 2.0, 3.0, 4.0];
        let variance = intensity.clone();
        let lut = crate::lut::build_lut2d_csr(&tth, &azimuthal, 2, 2);
        let y = integrate2d(&intensity, &lut, None);
        let (y_err, _sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate2d_with_errors_matches_weighted() {
        let intensity = vec![2.0, 4.0, 6.0, 8.0];
        let variance = intensity.clone();
        let lut = Lut2d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 2, 3],
            values: Some(vec![0.5, 0.5, 0.25, 0.75]),
            n_tth: 2,
            n_azimuthal: 1,
        };
        let y = integrate2d(&intensity, &lut, None);
        let (y_err, _sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate1d_with_errors_sigma_unweighted() {
        let tth = vec![0.1, 0.2];
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let lut = crate::lut::build_lut1d_csr(&tth, 1);
        let (_bins, _y, sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None);
        assert_eq!(sigma.len(), 1);
        assert!((sigma[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_with_errors_sigma_weighted() {
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.25, 0.75]),
            hist_min: 0.0,
            hist_step: 1.0,
            azimuthal_coverage: None,
        };
        let (_bins, _y, sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None);
        assert_eq!(sigma.len(), 1);
        let expected_var: f64 = 0.25 * 0.25 * 1.0 + 0.75 * 0.75 * 3.0;
        let expected_sigma = expected_var.sqrt();
        assert!((sigma[0] - expected_sigma).abs() < 1e-12);
    }

    #[test]
    fn test_integrate2d_with_errors_sigma_unweighted() {
        let tth = vec![0.1, 0.2];
        let azimuthal = vec![0.0, 0.1];
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let lut = crate::lut::build_lut2d_csr(&tth, &azimuthal, 1, 1);
        let (_y, sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None);
        assert_eq!(sigma.len(), 1);
        assert!((sigma[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate2d_with_errors_sigma_weighted() {
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let lut = Lut2d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.25, 0.75]),
            n_tth: 1,
            n_azimuthal: 1,
        };
        let (_y, sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None);
        assert_eq!(sigma.len(), 1);
        let expected_var: f64 = 0.25 * 0.25 * 1.0 + 0.75 * 0.75 * 3.0;
        let expected_sigma = expected_var.sqrt();
        assert!((sigma[0] - expected_sigma).abs() < 1e-12);
    }
}
