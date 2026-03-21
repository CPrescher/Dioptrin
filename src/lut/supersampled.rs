//! Supersampled CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate1d` / `integrate2d`.
#![allow(clippy::items_after_test_module)]
use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::integration::RadialUnit;
use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::math::histogram;

/// Build a supersampled 1D CSR LUT directly (no HashMap in the public type).
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// # Example
///
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::poni::Poni;
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(10, 10, poni);
/// let radial = geometry.calc_tth_all();
/// let lut = dioptrin::lut::build_lut1d_ss_csr(&geometry, &radial, 100, 2, RadialUnit::TthRad);
/// let intensity = vec![1.0; 100];
/// let norm = vec![1.0; 100];
/// let (_bins, y) = dioptrin::lut::integrate1d(&intensity, &lut, None, &norm);
/// assert_eq!(y.len(), 100);
/// ```
pub fn build_lut1d_ss_csr(
    geometry: &Geometry,
    radial: &[f64],
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
                    let b = histogram::get_bin_clamped(v, hist_min, hist_step, n_points);
                    acc.push((b, actual_index[i]));
                }
                acc
            },
        )
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

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
        if bin_total > 0 {
            let denom = bin_total as f64;
            for w in &mut values[bin_start..] {
                *w /= denom;
            }
        }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < n_points + 1 {
        row_ptr.push(col_idx.len());
    }

    Lut1d {
        row_ptr,
        col_idx,
        values: Some(values),
        hist_min,
        hist_step,
    }
}

/// Build a supersampled 1D CSR LUT with explicit radial range.
pub fn build_lut1d_ss_csr_with_range(
    geometry: &Geometry,
    _radial: &[f64],
    n_points: usize,
    n_ss: usize,
    unit: RadialUnit,
    radial_min: f64,
    radial_max: f64,
) -> Lut1d {
    let (hist_min, _hist_max, hist_step) =
        histogram::min_max_step_from_range(radial_min, radial_max, n_points);

    let wavelength = geometry.poni.wavelength;

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
                    let b = histogram::get_bin_clamped(v, hist_min, hist_step, n_points);
                    acc.push((b, actual_index[i]));
                }
                acc
            },
        )
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

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
        if bin_total > 0 {
            let denom = bin_total as f64;
            for w in &mut values[bin_start..] {
                *w /= denom;
            }
        }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < n_points + 1 {
        row_ptr.push(col_idx.len());
    }

    Lut1d {
        row_ptr,
        col_idx,
        values: Some(values),
        hist_min,
        hist_step,
    }
}

/// Build a supersampled 2D CSR LUT directly (no HashMap in the public type).
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// # Example
///
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::poni::Poni;
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(10, 10, poni);
/// let radial = geometry.calc_tth_all();
/// let azimuthal = geometry.calc_azimuth_all();
/// let lut2d = dioptrin::lut::build_lut2d_ss_csr(
///     &geometry, &radial, &azimuthal, 32, 36, 2, RadialUnit::TthRad
/// );
/// let intensity = vec![1.0; 100];
/// let norm = vec![1.0; 100];
/// let y = dioptrin::lut::integrate2d(&intensity, &lut2d, None, &norm);
/// assert_eq!(y.len(), 32 * 36);
/// ```
pub fn build_lut2d_ss_csr(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    n_ss: usize,
    unit: RadialUnit,
) -> Lut2d {
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let (azimuthal_min, _azimuthal_max, azimuthal_step) =
        histogram::supersampled_azimuthal_min_max_step(geometry, azimuthal, n_azimuthal, n_ss);

    let wavelength = geometry.poni.wavelength;

    let rows = n_radial * n_azimuthal;
    let per_row = geometry.dim2 * n_ss * n_ss;
    let mut pairs: Vec<(usize, usize)> = (0..geometry.dim1)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(per_row),
            |mut acc, r| {
                let (d1, d2, actual_index) = crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                let (tth_ss, azimuthal_ss) = geometry.calc_tth_and_azimuth(&d1, &d2);
                let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                for i in 0..radial_ss.len() {
                    let rb =
                        histogram::get_bin_clamped(radial_ss[i], radial_min, radial_step, n_radial);
                    let ab = histogram::get_bin(azimuthal_ss[i], azimuthal_min, azimuthal_step)
                        % n_azimuthal;
                    let row = ab * n_radial + rb;
                    acc.push((row, actual_index[i]));
                }
                acc
            },
        )
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_row = pairs[i].0;
        while row_ptr.len() - 1 < current_row {
            row_ptr.push(col_idx.len());
        }
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
        if bin_total > 0 {
            let denom = bin_total as f64;
            for w in &mut values[bin_start..] {
                *w /= denom;
            }
        }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < rows + 1 {
        row_ptr.push(col_idx.len());
    }

    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: n_radial,
        n_azimuthal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::poni::Poni;

    fn create_test_geometry() -> Geometry {
        let mut poni = Poni::new();
        poni.pixel1 = 0.0001; // 100 µm
        poni.pixel2 = 0.0001;
        poni.distance = 0.1; // 10 cm
        poni.wavelength = 1e-10; // 1 Angstrom
        Geometry::new(10, 10, poni)
    }

    #[test]
    fn test_build_lut1d_ss_csr_tth_rad() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0);
        assert!(lut.hist_step > 0.0);
        assert!(lut.values.is_some());
    }

    #[test]
    fn test_build_lut1d_ss_csr_tth_deg() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0);
    }

    #[test]
    fn test_build_lut1d_ss_csr_q_angstrom() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_q, 5, 2, RadialUnit::QAngstrom);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut1d_ss_csr_q_nm() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_q, 5, 2, RadialUnit::QNm);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut2d_ss_csr_with_q_unit() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let azimuthal = geometry.calc_azimuth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut2d_ss_csr(
            &geometry,
            &radial_q,
            &azimuthal,
            5,
            4,
            2,
            RadialUnit::QAngstrom,
        );

        assert_eq!(lut.n_tth, 5);
        assert_eq!(lut.n_azimuthal, 4);
        assert_eq!(lut.row_ptr.len(), 5 * 4 + 1);
        assert!(lut.values.is_some());
    }

    #[test]
    fn test_build_lut2d_ss_csr_with_azimuthal_range_q_unit() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut2d_ss_csr_with_azimuthal_range(
            &geometry,
            &radial_q,
            5,
            4,
            2,
            0.0,
            std::f64::consts::PI / 2.0,
            RadialUnit::QNm,
        );

        assert_eq!(lut.n_tth, 5);
        assert_eq!(lut.n_azimuthal, 4);
    }

    #[test]
    fn test_supersampled_different_units_different_bins() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();

        // Build LUT in radians
        let lut_rad = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);

        // Build LUT in degrees
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_deg = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);

        // The bin centers should differ by factor of 180/π
        let ratio = lut_deg.hist_min / lut_rad.hist_min;
        let expected_ratio = 180.0 / std::f64::consts::PI;
        assert!(
            (ratio - expected_ratio).abs() / expected_ratio < 0.01,
            "Degree bins should be 180/π times radian bins, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_supersampled_q_angstrom_vs_q_nm_ratio() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();

        // Build LUT in q (Å⁻¹)
        let radial_qa = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_qa = build_lut1d_ss_csr(&geometry, &radial_qa, 5, 2, RadialUnit::QAngstrom);

        // Build LUT in q (nm⁻¹)
        let radial_qn = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_qn = build_lut1d_ss_csr(&geometry, &radial_qn, 5, 2, RadialUnit::QNm);

        // q in nm⁻¹ should be 10x q in Å⁻¹
        let ratio = lut_qn.hist_min / lut_qa.hist_min;
        assert!(
            (ratio - 10.0).abs() < 0.01,
            "q_nm bins should be 10x q_angstrom bins, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_supersampled_integration_with_different_units() {
        use crate::lut::integrate1d;

        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let intensity: Vec<f64> = (0..geometry.size()).map(|i| (i + 1) as f64).collect();

        // Build and integrate in radians
        let lut_rad = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);
        let norm = vec![1.0; geometry.size()];
        let (bins_rad, int_rad) = integrate1d(&intensity, &lut_rad, None, &norm);

        // Build and integrate in degrees
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_deg = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);
        let (bins_deg, int_deg) = integrate1d(&intensity, &lut_deg, None, &norm);

        // Bin centers should differ by 180/π
        for (r, d) in bins_rad.iter().zip(bins_deg.iter()) {
            let ratio = d / r;
            let expected = 180.0 / std::f64::consts::PI;
            assert!(
                (ratio - expected).abs() / expected < 0.01,
                "Bin centers should differ by 180/π"
            );
        }

        // Intensities should be the same since it's the same data
        for (ir, id) in int_rad.iter().zip(int_deg.iter()) {
            assert!(
                (ir - id).abs() < 1e-10,
                "Intensities should match: {} vs {}",
                ir,
                id
            );
        }
    }
}

/// Build a supersampled 2D CSR LUT using an explicit azimuthal binning definition.
///
/// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
/// `azimuthal_start - azimuthal_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_azimuthal`.
#[allow(clippy::too_many_arguments)]
pub fn build_lut2d_ss_csr_with_azimuthal_range(
    geometry: &Geometry,
    radial: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    n_ss: usize,
    azimuthal_start: f64,
    azimuthal_step: f64,
    unit: RadialUnit,
) -> Lut2d {
    assert!(azimuthal_step > 0.0, "azimuthal_step must be positive");
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let min_azimuthal = azimuthal_start - 0.5 * azimuthal_step;

    let wavelength = geometry.poni.wavelength;

    let rows = n_radial * n_azimuthal;
    let per_row = geometry.dim2 * n_ss * n_ss;
    let mut pairs: Vec<(usize, usize)> = (0..geometry.dim1)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(per_row),
            |mut acc, r| {
                let (d1, d2, actual_index) = crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                let (tth_ss, azimuthal_ss) = geometry.calc_tth_and_azimuth(&d1, &d2);
                let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                for i in 0..radial_ss.len() {
                    let rb =
                        histogram::get_bin_clamped(radial_ss[i], radial_min, radial_step, n_radial);
                    let ab = histogram::get_bin_wrapped(
                        azimuthal_ss[i],
                        min_azimuthal,
                        azimuthal_step,
                        n_azimuthal,
                    );
                    let row = ab * n_radial + rb;
                    acc.push((row, actual_index[i]));
                }
                acc
            },
        )
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    pairs.par_sort_unstable();

    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);

    let mut i = 0usize;
    while i < pairs.len() {
        let current_row = pairs[i].0;
        while row_ptr.len() - 1 < current_row {
            row_ptr.push(col_idx.len());
        }
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
        if bin_total > 0 {
            let denom = bin_total as f64;
            for w in &mut values[bin_start..] {
                *w /= denom;
            }
        }
        row_ptr.push(col_idx.len());
    }

    while row_ptr.len() < rows + 1 {
        row_ptr.push(col_idx.len());
    }

    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: n_radial,
        n_azimuthal,
    }
}
