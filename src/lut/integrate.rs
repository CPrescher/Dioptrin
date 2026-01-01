//! Shared LUT integrators used by all methods (counting, pixel-split, supersampled).
//! These functions operate on the unified CSR types (`Lut1d`, `Lut2d`).
use rayon::prelude::*;

use crate::math::histogram;
use crate::lut::counting_csr::{Lut1d, Lut2d};

/// Integrate using either unweighted (mean) or weighted 1D CSR LUT.
///
/// Examples
/// ```
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate_lut1d};
/// // Simple pixel-split (weighted) 1D example
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let intensity = vec![10.0, 20.0, 30.0, 40.0];
/// let lut = build_lut1d_ps_csr(&tth, 3);
/// let (_x, y) = integrate_lut1d(&intensity, &lut);
/// assert_eq!(y.len(), 3);
/// ```
/// ```no_run
/// // Supersampled example (requires Geometry)
/// use dioptrin::lut::{build_lut1d_ss_csr, integrate_lut1d};
/// # use dioptrin::geometry::Geometry;
/// # let geometry: Geometry = unimplemented!();
/// # let tth: Vec<f64> = unimplemented!();
/// # let intensity: Vec<f64> = unimplemented!();
/// let lut = build_lut1d_ss_csr(&geometry, &tth, 1024, 4);
/// let (_x, y) = integrate_lut1d(&intensity, &lut);
/// ```
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

/// Integrate using either unweighted (mean) or weighted 2D CSR LUT.
///
/// Example
/// ```
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate_lut2d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let chi = vec![0.0, 0.1, 0.2, 0.3];
/// let intensity = vec![1.0, 2.0, 3.0, 4.0];
/// let lut2d = build_lut2d_ps_csr(&tth, &chi, 2, 3);
/// let y = integrate_lut2d(&intensity, &lut2d);
/// assert_eq!(y.len(), 2*3);
/// ```
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
