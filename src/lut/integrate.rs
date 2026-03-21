//! Shared LUT integrators used by all methods (counting, pixel-split, supersampled).
//! These functions operate on the unified CSR types (`Lut1d`, `Lut2d`).
use rayon::prelude::*;

use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::math::histogram;

/// Integrate using either unweighted or weighted 1D CSR LUT with normalization.
///
/// Uses pyFAI-style weighted-average: `I = Sum(c*signal) / Sum(c*norm)`.
///
///   - Weighted (values=Some): sum_sig = Sum(w*signal), sum_norm = Sum(w*norm), result = sum_sig/sum_norm
///   - Unweighted (values=None): sum_sig = Sum(signal), sum_norm = Sum(norm), result = sum_sig/sum_norm
///
/// # Examples
///
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
/// let norm = vec![1.0; geometry.size()];
/// let lut = build_lut1d_ps_csr(&geometry, &tth, 4, RadialUnit::TthRad);
/// let (_x, y) = integrate1d(&intensity, &lut, None, &norm);
/// assert_eq!(y.len(), 4);
/// ```
pub fn integrate1d(
    intensity: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let pattern_intensity: Vec<f64> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                return 0.0;
            }
            match &lut.values {
                Some(values) => {
                    // Weighted: I = Sum(w*signal) / Sum(w*norm)
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        0.0
                    } else {
                        sum_sig / sum_norm
                    }
                }
                None => {
                    // Unweighted: I = Sum(signal) / Sum(norm)
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        0.0
                    } else {
                        sum_sig / sum_norm
                    }
                }
            }
        })
        .collect();
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity)
}

/// Integrate using either unweighted or weighted 1D CSR LUT, returning errors.
///
/// sigma = sqrt(Sum(w^2 * variance)) / sum_norm.
/// The returned error is the per-bin standard deviation computed from per-pixel variances.
pub fn integrate1d_with_errors(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_points = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                return (0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (sum_sig / sum_norm, sum_var / (sum_norm * sum_norm))
                    }
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_var += *variance_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (sum_sig / sum_norm, sum_var / (sum_norm * sum_norm))
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
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity, pattern_sigma)
}

/// Integrate using either unweighted or weighted 2D CSR LUT with normalization.
///
/// Uses pyFAI-style weighted-average: `I = Sum(c*signal) / Sum(c*norm)`.
///
/// # Example
///
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
/// let norm = vec![1.0; geometry.size()];
/// let lut2d = build_lut2d_ps_csr(&geometry, &tth, &azimuthal, 4, 4, RadialUnit::TthRad);
/// let y = integrate2d(&intensity, &lut2d, None, &norm);
/// assert_eq!(y.len(), 4 * 4);
/// ```
pub fn integrate2d(
    intensity: &[f64],
    lut: &Lut2d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> Vec<f64> {
    let rows = lut.row_ptr.len() - 1;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            if start == end {
                return 0.0;
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        0.0
                    } else {
                        sum_sig / sum_norm
                    }
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        0.0
                    } else {
                        sum_sig / sum_norm
                    }
                }
            }
        })
        .collect()
}

/// Integrate using either unweighted or weighted 2D CSR LUT, returning errors.
///
/// sigma = sqrt(Sum(w^2 * variance)) / sum_norm.
/// The returned error is the per-bin standard deviation computed from per-pixel variances.
pub fn integrate2d_with_errors(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut2d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let rows = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            if start == end {
                return (0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (sum_sig / sum_norm, sum_var / (sum_norm * sum_norm))
                    }
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut k = start;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_var += *variance_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    if sum_norm == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (sum_sig / sum_norm, sum_var / (sum_norm * sum_norm))
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

/// Intermediate (pre-division) result for multi-detector merging.
pub struct IntermediateResult1d {
    pub bins: Vec<f64>,
    pub sum_signal: Vec<f64>,
    pub sum_norm: Vec<f64>,
    pub sum_variance: Option<Vec<f64>>,
    pub count: Vec<f64>,
}

/// Intermediate (pre-division) result for 2D multi-detector merging.
pub struct IntermediateResult2d {
    pub sum_signal: Vec<f64>,
    pub sum_norm: Vec<f64>,
    pub sum_variance: Option<Vec<f64>>,
    pub count: Vec<f64>,
    pub n_radial: usize,
    pub n_azimuthal: usize,
}

/// Raw 1D integration returning intermediate sums without dividing.
pub fn integrate1d_raw(
    intensity: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> IntermediateResult1d {
    let n_points = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64, f64)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                return (0.0, 0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                        count += w;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    count += w;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, count)
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                        count += 1.0;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    count += 1.0;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, count)
                }
            }
        })
        .collect();
    let mut sum_signal = Vec::with_capacity(n_points);
    let mut sum_norm = Vec::with_capacity(n_points);
    let mut count = Vec::with_capacity(n_points);
    for (s, n, c) in results {
        sum_signal.push(s);
        sum_norm.push(n);
        count.push(c);
    }
    let bins = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    IntermediateResult1d {
        bins,
        sum_signal,
        sum_norm,
        sum_variance: None,
        count,
    }
}

/// Raw 1D integration with variance, returning intermediate sums without dividing.
pub fn integrate1d_raw_with_variance(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> IntermediateResult1d {
    let n_points = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64, f64, f64)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let start = lut.row_ptr[i];
            let end = lut.row_ptr[i + 1];
            if start == end {
                return (0.0, 0.0, 0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                        count += w;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    count += w;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, sum_var, count)
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                        count += 1.0;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_var += *variance_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    count += 1.0;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, sum_var, count)
                }
            }
        })
        .collect();
    let mut sum_signal = Vec::with_capacity(n_points);
    let mut sum_norm_v = Vec::with_capacity(n_points);
    let mut sum_variance = Vec::with_capacity(n_points);
    let mut count = Vec::with_capacity(n_points);
    for (s, n, v, c) in results {
        sum_signal.push(s);
        sum_norm_v.push(n);
        sum_variance.push(v);
        count.push(c);
    }
    let bins = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    IntermediateResult1d {
        bins,
        sum_signal,
        sum_norm: sum_norm_v,
        sum_variance: Some(sum_variance),
        count,
    }
}

/// Raw 2D integration returning intermediate sums without dividing.
pub fn integrate2d_raw(
    intensity: &[f64],
    lut: &Lut2d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> IntermediateResult2d {
    let rows = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64, f64)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            if start == end {
                return (0.0, 0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                        count += w;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    count += w;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, count)
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                        count += 1.0;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    count += 1.0;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, count)
                }
            }
        })
        .collect();
    let mut sum_signal = Vec::with_capacity(rows);
    let mut sum_norm = Vec::with_capacity(rows);
    let mut count = Vec::with_capacity(rows);
    for (s, n, c) in results {
        sum_signal.push(s);
        sum_norm.push(n);
        count.push(c);
    }
    IntermediateResult2d {
        sum_signal,
        sum_norm,
        sum_variance: None,
        count,
        n_radial: lut.n_tth,
        n_azimuthal: lut.n_azimuthal,
    }
}

/// Raw 2D integration with variance, returning intermediate sums without dividing.
pub fn integrate2d_raw_with_variance(
    intensity: &[f64],
    variance: &[f64],
    lut: &Lut2d,
    mask: Option<&[u8]>,
    normalization: &[f64],
) -> IntermediateResult2d {
    let rows = lut.row_ptr.len() - 1;
    let results: Vec<(f64, f64, f64, f64)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = lut.row_ptr[row];
            let end = lut.row_ptr[row + 1];
            if start == end {
                return (0.0, 0.0, 0.0, 0.0);
            }
            match &lut.values {
                Some(values) => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let values_ptr = values.as_ptr();
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        let w = *values_ptr.add(k);
                                        sum_sig += w * *intensity_ptr.add(j);
                                        sum_var += w * w * *variance_ptr.add(j);
                                        sum_norm += w * *norm_ptr.add(j);
                                        count += w;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let w = *values_ptr.add(k);
                                    let j = *col_ptr.add(k);
                                    sum_sig += w * *intensity_ptr.add(j);
                                    sum_var += w * w * *variance_ptr.add(j);
                                    sum_norm += w * *norm_ptr.add(j);
                                    count += w;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, sum_var, count)
                }
                None => {
                    let mut sum_sig = 0.0;
                    let mut sum_var = 0.0;
                    let mut sum_norm = 0.0;
                    let mut count = 0.0;
                    unsafe {
                        let col_ptr = lut.col_idx.as_ptr();
                        let intensity_ptr = intensity.as_ptr();
                        let variance_ptr = variance.as_ptr();
                        let norm_ptr = normalization.as_ptr();
                        match mask {
                            Some(mask) => {
                                let mask_ptr = mask.as_ptr();
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    if *mask_ptr.add(j) == 0 {
                                        sum_sig += *intensity_ptr.add(j);
                                        sum_var += *variance_ptr.add(j);
                                        sum_norm += *norm_ptr.add(j);
                                        count += 1.0;
                                    }
                                    k += 1;
                                }
                            }
                            None => {
                                let mut k = start;
                                while k < end {
                                    let j = *col_ptr.add(k);
                                    sum_sig += *intensity_ptr.add(j);
                                    sum_var += *variance_ptr.add(j);
                                    sum_norm += *norm_ptr.add(j);
                                    count += 1.0;
                                    k += 1;
                                }
                            }
                        }
                    }
                    (sum_sig, sum_norm, sum_var, count)
                }
            }
        })
        .collect();
    let mut sum_signal = Vec::with_capacity(rows);
    let mut sum_norm_v = Vec::with_capacity(rows);
    let mut sum_variance = Vec::with_capacity(rows);
    let mut count = Vec::with_capacity(rows);
    for (s, n, v, c) in results {
        sum_signal.push(s);
        sum_norm_v.push(n);
        sum_variance.push(v);
        count.push(c);
    }
    IntermediateResult2d {
        sum_signal,
        sum_norm: sum_norm_v,
        sum_variance: Some(sum_variance),
        count,
        n_radial: lut.n_tth,
        n_azimuthal: lut.n_azimuthal,
    }
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

    // Use normalization of all 1.0 to replicate simple mean/weighted-sum behavior
    fn ones(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    #[test]
    fn test_integrate1d_with_errors_matches_unweighted() {
        let tth = vec![0.1, 0.2, 0.3, 0.4];
        let intensity = vec![1.0, 2.0, 3.0, 4.0];
        let variance = intensity.clone();
        let norm = ones(4);
        let lut = crate::lut::build_lut1d_csr(&tth, 2);
        let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
        let (_bins_err, y_err, _sigma) =
            integrate1d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate1d_with_errors_matches_weighted() {
        let intensity = vec![2.0, 4.0, 6.0, 8.0];
        let variance = intensity.clone();
        let norm = ones(4);
        let lut = Lut1d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 2, 3],
            values: Some(vec![0.5, 0.5, 0.25, 0.75]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
        let (_bins_err, y_err, _sigma) =
            integrate1d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate2d_with_errors_matches_unweighted() {
        let tth = vec![0.1, 0.2, 0.3, 0.4];
        let azimuthal = vec![0.0, 0.1, 0.2, 0.3];
        let intensity = vec![1.0, 2.0, 3.0, 4.0];
        let variance = intensity.clone();
        let norm = ones(4);
        let lut = crate::lut::build_lut2d_csr(&tth, &azimuthal, 2, 2);
        let y = integrate2d(&intensity, &lut, None, &norm);
        let (y_err, _sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate2d_with_errors_matches_weighted() {
        let intensity = vec![2.0, 4.0, 6.0, 8.0];
        let variance = intensity.clone();
        let norm = ones(4);
        let lut = Lut2d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 2, 3],
            values: Some(vec![0.5, 0.5, 0.25, 0.75]),
            n_tth: 2,
            n_azimuthal: 1,
        };
        let y = integrate2d(&intensity, &lut, None, &norm);
        let (y_err, _sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_vec_close(&y, &y_err);
    }

    #[test]
    fn test_integrate1d_with_errors_sigma_unweighted() {
        let tth = vec![0.1, 0.2];
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let norm = ones(2);
        let lut = crate::lut::build_lut1d_csr(&tth, 1);
        let (_bins, _y, sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_eq!(sigma.len(), 1);
        assert!((sigma[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_with_errors_sigma_weighted() {
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let norm = ones(2);
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.25, 0.75]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let (_bins, _y, sigma) = integrate1d_with_errors(&intensity, &variance, &lut, None, &norm);
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
        let norm = ones(2);
        let lut = crate::lut::build_lut2d_csr(&tth, &azimuthal, 1, 1);
        let (_y, sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_eq!(sigma.len(), 1);
        assert!((sigma[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate2d_with_errors_sigma_weighted() {
        let intensity = vec![1.0, 3.0];
        let variance = intensity.clone();
        let norm = ones(2);
        let lut = Lut2d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.25, 0.75]),
            n_tth: 1,
            n_azimuthal: 1,
        };
        let (_y, sigma) = integrate2d_with_errors(&intensity, &variance, &lut, None, &norm);
        assert_eq!(sigma.len(), 1);
        let expected_var: f64 = 0.25 * 0.25 * 1.0 + 0.75 * 0.75 * 3.0;
        let expected_sigma = expected_var.sqrt();
        assert!((sigma[0] - expected_sigma).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_with_normalization() {
        // Two pixels, both in same bin, signal=[10, 20], norm=[2, 4]
        // Expected: (10+20)/(2+4) = 5.0
        let intensity = vec![10.0, 20.0];
        let norm = vec![2.0, 4.0];
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: None,
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
        assert_eq!(y.len(), 1);
        assert!((y[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_weighted_with_normalization() {
        // Two pixels, weights=[0.5, 0.5], signal=[10, 20], norm=[2, 4]
        // sum_sig = 0.5*10 + 0.5*20 = 15, sum_norm = 0.5*2 + 0.5*4 = 3
        // Expected: 15/3 = 5.0
        let intensity = vec![10.0, 20.0];
        let norm = vec![2.0, 4.0];
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.5, 0.5]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
        assert_eq!(y.len(), 1);
        assert!((y[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_raw_returns_sums() {
        let intensity = vec![10.0, 20.0];
        let norm = vec![2.0, 4.0];
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: None,
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let raw = integrate1d_raw(&intensity, &lut, None, &norm);
        assert_eq!(raw.sum_signal.len(), 1);
        assert!((raw.sum_signal[0] - 30.0).abs() < 1e-12); // 10 + 20
        assert!((raw.sum_norm[0] - 6.0).abs() < 1e-12); // 2 + 4
        assert!((raw.count[0] - 2.0).abs() < 1e-12);
        assert!(raw.sum_variance.is_none());
        // Manual division should match integrate1d
        let divided = raw.sum_signal[0] / raw.sum_norm[0];
        assert!((divided - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_raw_weighted() {
        let intensity = vec![10.0, 20.0];
        let norm = vec![2.0, 4.0];
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.5, 0.5]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let raw = integrate1d_raw(&intensity, &lut, None, &norm);
        // sum_sig = 0.5*10 + 0.5*20 = 15
        assert!((raw.sum_signal[0] - 15.0).abs() < 1e-12);
        // sum_norm = 0.5*2 + 0.5*4 = 3
        assert!((raw.sum_norm[0] - 3.0).abs() < 1e-12);
        // count = 0.5 + 0.5 = 1.0
        assert!((raw.count[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_raw_with_variance_returns_variance() {
        let intensity = vec![1.0, 3.0];
        let variance = vec![1.0, 3.0];
        let norm = ones(2);
        let lut = Lut1d {
            row_ptr: vec![0, 2],
            col_idx: vec![0, 1],
            values: Some(vec![0.25, 0.75]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let raw = integrate1d_raw_with_variance(&intensity, &variance, &lut, None, &norm);
        let sum_var = raw.sum_variance.unwrap();
        // sum_var = 0.25^2 * 1.0 + 0.75^2 * 3.0
        let expected_var = 0.25 * 0.25 * 1.0 + 0.75 * 0.75 * 3.0;
        assert!((sum_var[0] - expected_var).abs() < 1e-12);
    }

    #[test]
    fn test_integrate1d_raw_with_mask() {
        let intensity = vec![10.0, 20.0, 30.0];
        let norm = vec![1.0, 1.0, 1.0];
        let mask = vec![0u8, 1, 0]; // mask out pixel 1
        let lut = Lut1d {
            row_ptr: vec![0, 3],
            col_idx: vec![0, 1, 2],
            values: None,
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let raw = integrate1d_raw(&intensity, &lut, Some(&mask), &norm);
        // Only pixels 0 and 2 contribute
        assert!((raw.sum_signal[0] - 40.0).abs() < 1e-12); // 10 + 30
        assert!((raw.sum_norm[0] - 2.0).abs() < 1e-12);
        assert!((raw.count[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_integrate2d_raw_returns_sums() {
        let intensity = vec![10.0, 20.0, 30.0, 40.0];
        let norm = ones(4);
        let lut = Lut2d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 2, 3],
            values: None,
            n_tth: 2,
            n_azimuthal: 1,
        };
        let raw = integrate2d_raw(&intensity, &lut, None, &norm);
        assert_eq!(raw.sum_signal.len(), 2);
        assert!((raw.sum_signal[0] - 30.0).abs() < 1e-12); // 10 + 20
        assert!((raw.sum_signal[1] - 70.0).abs() < 1e-12); // 30 + 40
        assert!(raw.sum_variance.is_none());
        assert_eq!(raw.n_radial, 2);
        assert_eq!(raw.n_azimuthal, 1);
    }

    #[test]
    fn test_integrate1d_raw_empty_bins() {
        let intensity = vec![10.0];
        let norm = vec![1.0];
        let lut = Lut1d {
            row_ptr: vec![0, 0, 1], // first bin empty, second has one pixel
            col_idx: vec![0],
            values: None,
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let raw = integrate1d_raw(&intensity, &lut, None, &norm);
        assert_eq!(raw.sum_signal.len(), 2);
        assert!((raw.sum_signal[0]).abs() < 1e-12); // empty bin
        assert!((raw.sum_norm[0]).abs() < 1e-12);
        assert!((raw.count[0]).abs() < 1e-12);
        assert!((raw.sum_signal[1] - 10.0).abs() < 1e-12);
    }
}
