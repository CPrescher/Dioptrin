//! Exact regression oracle: the pre-optimization collect/sort implementation.
//! Kept test-only, on small tilted geometries, to catch boundary arithmetic,
//! sample counting, CSR ordering and floating-point normalization changes.
#![allow(clippy::too_many_arguments)] // Preserve the previous kernel signatures as test oracles.
use super::*;
use crate::poni::Poni;
mod legacy {
    use super::*;
    pub(super) fn build_lut1d_ss_csr(
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
                    let (d1, d2, actual_index) =
                        crate::lut::common::subgrid(row, geometry.dim2, n_ss);
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

    pub(super) fn build_lut1d_ss_csr_with_range(
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
                    let (d1, d2, actual_index) =
                        crate::lut::common::subgrid(row, geometry.dim2, n_ss);
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

    pub(super) fn build_lut2d_ss_csr(
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
                    let (d1, d2, actual_index) =
                        crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                    let (tth_ss, azimuthal_ss) = geometry.calc_tth_and_azimuth(&d1, &d2);
                    let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                    for i in 0..radial_ss.len() {
                        let rb = histogram::get_bin_clamped(
                            radial_ss[i],
                            radial_min,
                            radial_step,
                            n_radial,
                        );
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

    pub(super) fn build_lut2d_ss_csr_with_azimuthal_range(
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
                    let (d1, d2, actual_index) =
                        crate::lut::common::subgrid(r, geometry.dim2, n_ss);
                    let (tth_ss, azimuthal_ss) = geometry.calc_tth_and_azimuth(&d1, &d2);
                    let radial_ss = unit.convert_from_tth(&tth_ss, wavelength);
                    for i in 0..radial_ss.len() {
                        let rb = histogram::get_bin_clamped(
                            radial_ss[i],
                            radial_min,
                            radial_step,
                            n_radial,
                        );
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
}

fn assert_csr(actual: (&[usize], &[usize], &[f64]), expected: (&[usize], &[usize], &[f64])) {
    assert_eq!(actual.0, expected.0, "CSR row pointers changed");
    assert_eq!(actual.1, expected.1, "CSR pixel ordering changed");
    assert_eq!(
        actual.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "CSR weights must be bitwise identical"
    );
}

fn compare_geometry(geometry: &Geometry) {
    for unit in [
        RadialUnit::TthRad,
        RadialUnit::TthDeg,
        RadialUnit::QAngstrom,
        RadialUnit::QNm,
    ] {
        let radial = unit.convert_from_tth(&geometry.calc_tth_all(), geometry.poni.wavelength);
        let azimuth = geometry.calc_azimuth_all();
        let max = radial.iter().copied().fold(0.0_f64, f64::max);
        for n in [1, 2, 3, 5, 15] {
            for bins in [1, 7, 49] {
                let a = build_lut1d_ss_csr(geometry, &radial, bins, n, unit);
                let e = legacy::build_lut1d_ss_csr(geometry, &radial, bins, n, unit);
                assert_eq!(a.hist_min.to_bits(), e.hist_min.to_bits());
                assert_eq!(a.hist_step.to_bits(), e.hist_step.to_bits());
                assert_csr(
                    (&a.row_ptr, &a.col_idx, a.values.as_ref().unwrap()),
                    (&e.row_ptr, &e.col_idx, e.values.as_ref().unwrap()),
                );
                for (lo, hi) in [(0.0, max * 2.0), (max * 0.2, max * 0.8)] {
                    let a = build_lut1d_ss_csr_with_range(geometry, &radial, bins, n, unit, lo, hi);
                    let e = legacy::build_lut1d_ss_csr_with_range(
                        geometry, &radial, bins, n, unit, lo, hi,
                    );
                    assert_csr(
                        (&a.row_ptr, &a.col_idx, a.values.as_ref().unwrap()),
                        (&e.row_ptr, &e.col_idx, e.values.as_ref().unwrap()),
                    );
                    assert_eq!(a.col_idx.len(), a.col_idx.capacity());
                    assert_eq!(
                        a.values.as_ref().unwrap().len(),
                        a.values.as_ref().unwrap().capacity()
                    );
                }
                let a = build_lut2d_ss_csr(geometry, &radial, &azimuth, bins, 11, n, unit);
                let e = legacy::build_lut2d_ss_csr(geometry, &radial, &azimuth, bins, 11, n, unit);
                assert_csr(
                    (&a.row_ptr, &a.col_idx, a.values.as_ref().unwrap()),
                    (&e.row_ptr, &e.col_idx, e.values.as_ref().unwrap()),
                );
                let a = build_lut2d_ss_csr_with_azimuthal_range(
                    geometry, &radial, bins, 11, n, -2.9, 0.6, unit,
                );
                let e = legacy::build_lut2d_ss_csr_with_azimuthal_range(
                    geometry, &radial, bins, 11, n, -2.9, 0.6, unit,
                );
                assert_csr(
                    (&a.row_ptr, &a.col_idx, a.values.as_ref().unwrap()),
                    (&e.row_ptr, &e.col_idx, e.values.as_ref().unwrap()),
                );
            }
        }
    }
}

#[test]
fn bitwise_legacy_supersampling_all_units_and_ranges() {
    let mut poni = Poni::new();
    poni.pixel1 = 0.000172;
    poni.pixel2 = 0.000075;
    poni.distance = 0.1839103128035422;
    poni.poni1 = 0.00031;
    poni.poni2 = 0.00041;
    poni.wavelength = 3.738e-11;
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            compare_geometry(&Geometry::new(7, 9, poni.clone()));
            let mut tilted = poni.clone();
            tilted.rot1 = 0.137;
            tilted.rot2 = -0.073;
            tilted.rot3 = 0.419;
            compare_geometry(&Geometry::new(3, 4, tilted));
        });
    }
}
