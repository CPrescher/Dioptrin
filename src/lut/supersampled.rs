//! Supersampled CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate1d` / `integrate2d`.
use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::integration::RadialUnit;
use crate::math::histogram;
use crate::lut::counting_csr::{Lut1d, Lut2d};

/// Build a supersampled 1D CSR LUT directly (no HashMap in the public type).
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// Example (pseudo-code)
/// ```no_run
/// # use dioptrin::geometry::Geometry;
/// # use dioptrin::integration::RadialUnit;
/// # let geometry: Geometry = unimplemented!();
/// # let radial: Vec<f64> = unimplemented!();
/// let lut = dioptrin::lut::build_lut1d_ss_csr(&geometry, &radial, 1024, 4, RadialUnit::TthRad);
/// # let intensity: Vec<f64> = unimplemented!();
/// let (_bins, y) = dioptrin::lut::integrate1d(&intensity, &lut, None);
/// ```
pub fn build_lut1d_ss_csr(
    geometry: &Geometry,
    radial: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
    unit: RadialUnit,
) -> Lut1d {
    let (hist_min, _hist_max, hist_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_points, n_ss, unit);

    let wavelength = geometry.poni.wavelength;

    // Collect all (bin, pixel) occurrences from supersampling
    let per_row = geometry.dim2 * n_ss * n_ss;
    let mut pairs: Vec<(usize, usize)> = (0..geometry.dim1)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(per_row),
            |mut acc, row| {
                let (d1, d2, actual_index) = crate::lut::common::subgrid(row, geometry.dim2, n_ss);
                let tth_ss = geometry.calc_tth(&d1, &d2);
                let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                for (i, &v) in radial_ss.iter().enumerate() {
                    let b = histogram::get_bin(v, hist_min, hist_step);
                    acc.push((b, actual_index[i]));
                }
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        );

    // Sort by (bin, pixel) and compress to unique with counts
    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(n_points + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_bin = pairs[i].0;
        while row_ptr.len() - 1 < current_bin {
            row_ptr.push(row_ptr.last().copied().unwrap());
        }
        let mut bin_total: usize = 0;
        let bin_start = col_idx.len();
        while i < pairs.len() && pairs[i].0 == current_bin {
            let pixel = pairs[i].1;
            let mut cnt = 1usize;
            i += 1;
            while i < pairs.len() && pairs[i].0 == current_bin && pairs[i].1 == pixel {
                cnt += 1;
                i += 1;
            }
            col_idx.push(pixel);
            values.push(cnt as f64);
            bin_total += cnt;
        }
        if bin_total > 0 { let denom = bin_total as f64; for w in &mut values[bin_start..] { *w /= denom; } }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < n_points + 1 { row_ptr.push(col_idx.len()); }

    Lut1d { row_ptr, col_idx, values: Some(values), hist_min, hist_step }
}

/// Build a supersampled 2D CSR LUT directly (no HashMap in the public type).
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// Example (pseudo-code)
/// ```no_run
/// # use dioptrin::geometry::Geometry;
/// # use dioptrin::integration::RadialUnit;
/// # let geometry: Geometry = unimplemented!();
/// # let radial: Vec<f64> = unimplemented!();
/// # let chi: Vec<f64> = unimplemented!();
/// let lut2d = dioptrin::lut::build_lut2d_ss_csr(&geometry, &radial, &chi, 1024, 360, 4, RadialUnit::TthRad);
/// # let intensity: Vec<f64> = unimplemented!();
/// let y = dioptrin::lut::integrate2d(&intensity, &lut2d, None);
/// assert_eq!(y.len(), 1024*360);
/// ```
pub fn build_lut2d_ss_csr(
    geometry: &Geometry,
    radial: &Vec<f64>,
    chi: &Vec<f64>,
    n_radial: usize,
    n_chi: usize,
    n_ss: usize,
    unit: RadialUnit,
) -> Lut2d {
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let (chi_min, _chi_max, chi_step) =
        histogram::supersampled_chi_min_max_step(geometry, chi, n_chi, n_ss);

    let wavelength = geometry.poni.wavelength;

    let rows = n_radial * n_chi;
    let per_row = geometry.dim2 * n_ss * n_ss;
    let mut pairs: Vec<(usize, usize)> = (0..geometry.dim1)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(per_row),
            |mut acc, r| {
                let (d1, d2, actual_index) = crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                let (tth_ss, chi_ss) = geometry.calc_tth_and_chi(&d1, &d2);
                let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                for i in 0..radial_ss.len() {
                    let rb = histogram::get_bin(radial_ss[i], radial_min, radial_step);
                    let ab = histogram::get_bin(chi_ss[i], chi_min, chi_step) % n_chi;
                    let row = ab * n_radial + rb;
                    acc.push((row, actual_index[i]));
                }
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        );

    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_row = pairs[i].0;
        while row_ptr.len() - 1 < current_row { row_ptr.push(col_idx.len()); }
        let bin_start = col_idx.len();
        let mut bin_total = 0usize;
        while i < pairs.len() && pairs[i].0 == current_row {
            let pixel = pairs[i].1;
            let mut cnt = 1usize;
            i += 1;
            while i < pairs.len() && pairs[i].0 == current_row && pairs[i].1 == pixel {
                cnt += 1;
                i += 1;
            }
            col_idx.push(pixel);
            values.push(cnt as f64);
            bin_total += cnt;
        }
        if bin_total > 0 { let denom = bin_total as f64; for w in &mut values[bin_start..] { *w /= denom; } }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < rows + 1 { row_ptr.push(col_idx.len()); }

    Lut2d { row_ptr, col_idx, values: Some(values), n_tth: n_radial, n_chi }
}

/// Build a supersampled 2D CSR LUT using an explicit chi binning definition.
///
/// `chi_start` is the center of the first chi bin; the effective start edge is
/// `chi_start - chi_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_chi`.
pub fn build_lut2d_ss_csr_with_chi_range(
    geometry: &Geometry,
    radial: &Vec<f64>,
    n_radial: usize,
    n_chi: usize,
    n_ss: usize,
    chi_start: f64,
    chi_step: f64,
    unit: RadialUnit,
) -> Lut2d {
    assert!(chi_step > 0.0, "chi_step must be positive");
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let min_chi = chi_start - 0.5 * chi_step;

    let wavelength = geometry.poni.wavelength;

    let rows = n_radial * n_chi;
    let per_row = geometry.dim2 * n_ss * n_ss;
    let mut pairs: Vec<(usize, usize)> = (0..geometry.dim1)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(per_row),
            |mut acc, r| {
                let (d1, d2, actual_index) = crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                let (tth_ss, chi_ss) = geometry.calc_tth_and_chi(&d1, &d2);
                let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                for i in 0..radial_ss.len() {
                    let rb = histogram::get_bin(radial_ss[i], radial_min, radial_step);
                    let ab = histogram::get_bin_wrapped(chi_ss[i], min_chi, chi_step, n_chi);
                    let row = ab * n_radial + rb;
                    acc.push((row, actual_index[i]));
                }
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, mut b| {
                a.append(&mut b);
                a
            },
        );

    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_row = pairs[i].0;
        while row_ptr.len() - 1 < current_row { row_ptr.push(col_idx.len()); }
        let bin_start = col_idx.len();
        let mut bin_total = 0usize;
        while i < pairs.len() && pairs[i].0 == current_row {
            let pixel = pairs[i].1;
            let mut cnt = 1usize;
            i += 1;
            while i < pairs.len() && pairs[i].0 == current_row && pairs[i].1 == pixel {
                cnt += 1;
                i += 1;
            }
            col_idx.push(pixel);
            values.push(cnt as f64);
            bin_total += cnt;
        }
        if bin_total > 0 { let denom = bin_total as f64; for w in &mut values[bin_start..] { *w /= denom; } }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < rows + 1 { row_ptr.push(col_idx.len()); }

    Lut2d { row_ptr, col_idx, values: Some(values), n_tth: n_radial, n_chi }
}
