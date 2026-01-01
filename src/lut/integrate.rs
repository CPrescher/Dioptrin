//! Shared LUT integrators used by all methods (counting, pixel-split, supersampled).
//! These functions operate on the unified CSR types (`Lut1d`, `Lut2d`).
use rayon::prelude::*;

use crate::math::histogram;
use crate::lut::counting_csr::{Lut1d, Lut2d};

/// Integrate using either unweighted (mean) or weighted 1D CSR LUT.
///
/// Examples
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate1d};
/// // Simple pixel-split (weighted) 1D example
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut = build_lut1d_ps_csr(&geometry, &tth, 4);
/// let (_x, y) = integrate1d(&intensity, &lut, None);
/// assert_eq!(y.len(), 4);
/// ```
pub fn integrate1d(
    intensity: &Vec<f64>,
    lut: &Lut1d,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
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
                                    let j = *col_ptr.add(k) as usize;
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
                                    let j = *col_ptr.add(k) as usize;
                                    sum += w * *intensity_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    sum
                }
                None => {
                    if start == end { 0.0 } else {
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
                                        let j = *col_ptr.add(k) as usize;
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k) as usize;
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
    let pattern_tth = histogram::bins(n_points, lut.hist_min, lut.hist_step);
    (pattern_tth, pattern_intensity)
}

/// Integrate using either unweighted (mean) or weighted 2D CSR LUT.
///
/// Example
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let chi = geometry.calc_chi_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut2d = build_lut2d_ps_csr(&geometry, &tth, &chi, 4, 4);
/// let y = integrate2d(&intensity, &lut2d, None);
/// assert_eq!(y.len(), 4 * 4);
/// ```
pub fn integrate2d(
    intensity: &Vec<f64>,
    lut: &Lut2d,
    mask: Option<&[u8]>,
) -> Vec<f64> {
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
                                    let j = *col_ptr.add(k) as usize;
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
                                    let j = *col_ptr.add(k) as usize;
                                    sum += w * *intensity_ptr.add(j);
                                    k += 1;
                                }
                            }
                        }
                    }
                    sum
                }
                None => {
                    if start == end { 0.0 } else {
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
                                        let j = *col_ptr.add(k) as usize;
                                        if *mask_ptr.add(j) == 0 {
                                            sum += *intensity_ptr.add(j);
                                            count += 1;
                                        }
                                        k += 1;
                                    }
                                }
                                None => {
                                    while k < end {
                                        let j = *col_ptr.add(k) as usize;
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
