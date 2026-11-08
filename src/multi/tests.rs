use super::*;
use crate::{
    geometry::Geometry,
    integration::{MaskMode, RadialUnit},
    mask::Mask,
    poni::Poni,
};
use std::f64::consts::{PI, TAU};

fn geometry(rows: usize, cols: usize) -> Geometry {
    let mut p = Poni::new();
    p.pixel1 = 0.0001;
    p.pixel2 = 0.00012;
    p.distance = 0.1;
    p.poni1 = 0.0007;
    p.poni2 = 0.0011;
    p.wavelength = 1e-10;
    Geometry::new(rows, cols, p)
}
fn ai(g: Geometry, method: IntegrationMethod) -> Integrator {
    Integrator::new(g)
        .with_method(method)
        .with_radial_range(0.0, 0.1)
}
fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= 2e-11 * a.abs().max(b.abs()).max(1.0),
            "bin {i}: {a} != {b}"
        );
    }
}
fn methods() -> [IntegrationMethod; 3] {
    [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 4 },
    ]
}

#[test]
fn single_matches_ordinary_2d_on_matched_edges() {
    let mut g = geometry(13, 19);
    g.poni.poni1 = -0.001;
    g.poni.poni2 = -0.001;
    let data: Vec<_> = (0..g.size()).map(|i| 10.0 + (i % 23) as f64).collect();
    // Legacy split-polygon weighting is area-based rather than fractional;
    // the new polygon path is checked against fractional-pixel identities below.
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::Supersampled { n_ss: 4 },
    ] {
        let radial = g.calc_tth_all();
        let (lo, hi, _) = match method {
            IntegrationMethod::Counting => crate::math::histogram::min_max_step(&radial, 11),
            _ => {
                crate::math::histogram::ss_tth_min_max_step(&g, &radial, 11, 4, RadialUnit::TthRad)
            }
        };
        let ordinary = ai(g.clone(), method);
        let (y, e) = ordinary.integrate2d_with_azimuthal_range_and_errors_from_slice(
            &data,
            g.dim1,
            g.dim2,
            11,
            17,
            -PI + PI / 17.0,
            TAU / 17.0,
        );
        let single =
            MultiIntegrator::try_new(vec![ai(g.clone(), method).with_radial_range(lo, hi)])
                .unwrap();
        let out = single
            .integrate2d_from_slices(
                &[(&data, g.dim1, g.dim2)],
                11,
                17,
                &Multi2dOptions {
                    errors: true,
                    ..Default::default()
                },
            )
            .unwrap();
        close(&out.intensity, &y);
        close(out.errors.as_ref().unwrap(), &e);
        assert_eq!(out.radial.len(), 11);
        assert_eq!(out.azimuthal.len(), 17);
    }
}

#[test]
fn identical_detectors_preserve_intensity_and_reduce_std() {
    for method in methods() {
        for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
            for fast in [false, true] {
                let g = geometry(9, 12);
                let data = vec![81.0; g.size()];
                let make = || {
                    ai(g.clone(), method)
                        .with_mask_mode(mode)
                        .with_fast_math(fast)
                };
                let one = MultiIntegrator::try_new(vec![make()]).unwrap();
                let two = MultiIntegrator::try_new(vec![make(), make()]).unwrap();
                let options = Multi2dOptions {
                    errors: true,
                    ..Default::default()
                };
                let a = one
                    .integrate2d_from_slices(&[(&data, 9, 12)], 9, 16, &options)
                    .unwrap();
                let b = two
                    .integrate2d_from_slices(&[(&data, 9, 12), (&data, 9, 12)], 9, 16, &options)
                    .unwrap();
                close(&a.intensity, &b.intensity);
                close(
                    &a.errors
                        .unwrap()
                        .iter()
                        .map(|v| v / 2.0_f64.sqrt())
                        .collect::<Vec<_>>(),
                    &b.errors.unwrap(),
                );
                close(
                    &a.raw.count.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                    &b.raw.count,
                );
            }
        }
    }
}

#[test]
fn unequal_geometries_match_independent_counting_oracle() {
    let g1 = geometry(7, 11);
    let mut g2 = geometry(5, 9);
    g2.poni.pixel1 *= 2.0;
    g2.poni.pixel2 *= 1.7;
    g2.poni.distance *= 1.4;
    g2.poni.rot1 = 0.012;
    g2.poni.rot2 = -0.008;
    g2.poni.rot3 = 0.13;
    let geoms = [g1, g2];
    let data = [vec![13.0; 77], vec![47.0; 45]];
    let masks: Vec<Vec<u8>> = geoms
        .iter()
        .map(|g| (0..g.size()).map(|i| u8::from(i % 7 == 0)).collect())
        .collect();
    let grid = IntegrationGrid2d::new(8, 13, (0.0, 0.1), (-PI, PI)).unwrap();
    for norm in [MultiNormalization::Relative, MultiNormalization::Absolute] {
        let integrators = geoms
            .iter()
            .enumerate()
            .map(|(i, g)| {
                ai(g.clone(), IntegrationMethod::Counting)
                    .with_polarization_factor(0.3 + i as f64 * 0.2)
                    .with_mask(Mask::from_vec(masks[i].clone(), g.dim1, g.dim2).unwrap())
            })
            .collect();
        let multi = MultiIntegrator::try_new(integrators).unwrap();
        let options = Multi2dOptions {
            errors: true,
            normalization: norm,
            normalization_factors: Some(vec![2.0, 3.0]),
            ..Default::default()
        };
        let slices: Vec<_> = geoms
            .iter()
            .zip(&data)
            .map(|(g, d)| (d.as_slice(), g.dim1, g.dim2))
            .collect();
        let out = multi
            .integrate2d_from_slices(&slices, 8, 13, &options)
            .unwrap();
        let mut signal = vec![0.0; 104];
        let mut denom = vec![0.0; 104];
        let mut variance = vec![0.0; 104];
        for (i, g) in geoms.iter().enumerate() {
            let r = g.calc_tth_all();
            let a = g.calc_azimuth_all();
            let sa = g.calc_solid_angle();
            let pol = g.calc_polarization(0.3 + i as f64 * 0.2, &r, &a);
            let scale = (i as f64 + 2.0)
                * if norm == MultiNormalization::Absolute {
                    g.poni.pixel1 * g.poni.pixel2 / g.poni.distance.powi(2)
                } else {
                    1.0
                };
            for p in 0..g.size() {
                if masks[i][p] != 0 {
                    continue;
                }
                // Independent half-open bin calculation, not the builder helper.
                let rb = (r[p] / 0.1 * 8.0).floor() as usize;
                let ab = ((a[p] + PI).rem_euclid(TAU) / TAU * 13.0).floor() as usize;
                if rb >= 8 || ab >= 13 {
                    continue;
                }
                let b = ab * 8 + rb;
                signal[b] += data[i][p];
                variance[b] += data[i][p];
                denom[b] += sa[p] * pol[p] * scale;
            }
        }
        close(&out.raw.sum_signal, &signal);
        close(&out.raw.sum_norm, &denom);
        close(out.raw.sum_variance.as_ref().unwrap(), &variance);
        close(
            &out.intensity,
            &signal
                .iter()
                .zip(&denom)
                .map(|(s, n)| if *n == 0.0 { 0.0 } else { s / n })
                .collect::<Vec<_>>(),
        );
        close(
            &out.errors.unwrap(),
            &variance
                .iter()
                .zip(&denom)
                .map(|(v, n)| if *n == 0.0 { 0.0 } else { v.sqrt() / n })
                .collect::<Vec<_>>(),
        );
        assert!(out.raw.count.contains(&0.0));
        assert_eq!(grid.shape(), (13, 8));
    }
}

#[test]
fn fractional_split_weights_do_not_give_each_detector_equal_weight() {
    for method in methods() {
        let mut g = geometry(2, 3);
        g.poni.poni1 = -0.001;
        g.poni.poni2 = -0.001;
        let mut h = g.clone();
        h.dim1 = 1;
        h.dim2 = 1;
        let multi =
            MultiIntegrator::try_new(vec![ai(g.clone(), method), ai(h.clone(), method)]).unwrap();
        let data = vec![10.0; 6];
        let data2 = vec![90.0];
        let options = Multi2dOptions {
            errors: true,
            ..Default::default()
        };
        let out = multi
            .integrate2d_from_slices(&[(&data, 2, 3), (&data2, 1, 1)], 1, 1, &options)
            .unwrap();
        close(&out.raw.sum_signal, &[150.0]);
        close(&out.raw.count, &[7.0]);
        close(out.raw.sum_variance.as_ref().unwrap(), &[150.0]);
        let norm = g.calc_solid_angle().iter().sum::<f64>() + h.calc_solid_angle()[0];
        close(&out.intensity, &[150.0 / norm]);
        close(&out.errors.unwrap(), &[150.0_f64.sqrt() / norm]);
    }
}

#[test]
fn disjoint_coverage_masks_and_clipped_ranges() {
    for method in methods() {
        let mut g = geometry(3, 4);
        g.poni.poni1 = -0.002;
        g.poni.poni2 = -0.002;
        let mut h = g.clone();
        h.poni.rot2 = 0.06;
        let single = MultiIntegrator::try_new(vec![ai(g.clone(), method)]).unwrap();
        let second = MultiIntegrator::try_new(vec![ai(h.clone(), method)]).unwrap();
        let both =
            MultiIntegrator::try_new(vec![ai(g.clone(), method), ai(h.clone(), method)]).unwrap();
        let d = vec![5.0; 12];
        let e = vec![17.0; 12];
        let opts = Multi2dOptions::default();
        let a = single
            .integrate2d_from_slices(&[(&d, 3, 4)], 20, 12, &opts)
            .unwrap();
        let b = second
            .integrate2d_from_slices(&[(&e, 3, 4)], 20, 12, &opts)
            .unwrap();
        assert!(
            a.raw
                .count
                .iter()
                .zip(&b.raw.count)
                .all(|(x, y)| *x == 0.0 || *y == 0.0)
        );
        let c = both
            .integrate2d_from_slices(&[(&d, 3, 4), (&e, 3, 4)], 20, 12, &opts)
            .unwrap();
        close(
            &c.intensity,
            &a.intensity
                .iter()
                .zip(&b.intensity)
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        );
        let masked = MultiIntegrator::try_new(vec![
            ai(g.clone(), method).with_mask(Mask::from_vec(vec![1; 12], 3, 4).unwrap()),
        ])
        .unwrap();
        let m = masked
            .integrate2d_from_slices(
                &[(&d, 3, 4)],
                20,
                12,
                &Multi2dOptions {
                    errors: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            m.intensity
                .iter()
                .chain(m.errors.as_ref().unwrap())
                .all(|x| *x == 0.0)
        );
        // Neither radial nor azimuthal outliers may be clamped/folded into a sector.
        let excluded =
            MultiIntegrator::try_new(vec![ai(g.clone(), method).with_radial_range(0.08, 0.1)])
                .unwrap();
        assert!(
            excluded
                .integrate2d_from_slices(&[(&d, 3, 4)], 5, 5, &opts)
                .unwrap()
                .raw
                .count
                .iter()
                .all(|x| *x == 0.0)
        );
        let sector = Multi2dOptions {
            azimuthal_range: (-PI, -PI / 2.0),
            ..Default::default()
        };
        assert!(
            single
                .integrate2d_from_slices(&[(&d, 3, 4)], 5, 5, &sector)
                .unwrap()
                .raw
                .count
                .iter()
                .all(|x| *x == 0.0)
        );
    }
}

#[test]
fn units_and_periodic_seam_conserve_fractional_contributions() {
    for method in methods() {
        let g = geometry(9, 12);
        let d = vec![8.0; 108];
        let mut counts = Vec::new();
        for (unit, range) in [
            (RadialUnit::TthRad, 0.1),
            (RadialUnit::TthDeg, 0.1_f64.to_degrees()),
            (RadialUnit::QAngstrom, 1.0),
            (RadialUnit::QNm, 10.0),
        ] {
            let multi = MultiIntegrator::try_new(vec![
                ai(g.clone(), method)
                    .with_radial_unit(unit)
                    .with_radial_range(0.0, range),
            ])
            .unwrap();
            let result = multi
                .integrate2d_from_slices(&[(&d, 9, 12)], 8, 9, &Multi2dOptions::default())
                .unwrap();
            close(&[result.raw.count.iter().sum()], &[108.0]);
            close(&[result.raw.sum_signal.iter().sum()], &[864.0]);
            counts.push(result.raw.count);
        }
        close(&counts[0], &counts[1]);
        close(&counts[2], &counts[3]);
        let multi = MultiIntegrator::try_new(vec![ai(g.clone(), method)]).unwrap();
        let a = multi
            .integrate2d_from_slices(
                &[(&d, 9, 12)],
                8,
                9,
                &Multi2dOptions {
                    azimuthal_range: (0.0, TAU),
                    ..Default::default()
                },
            )
            .unwrap();
        close(&[a.raw.count.iter().sum()], &[108.0]);
        let a = multi
            .integrate2d_from_slices(
                &[(&d, 9, 12)],
                8,
                9,
                &Multi2dOptions {
                    azimuthal_range: (TAU, 2.0 * TAU),
                    ..Default::default()
                },
            )
            .unwrap();
        close(&[a.raw.count.iter().sum()], &[108.0]);
    }
}

#[test]
fn invalid_inputs_return_errors() {
    let g = geometry(2, 3);
    let data = vec![1.0; 6];
    let multi = MultiIntegrator::try_new(vec![ai(g.clone(), IntegrationMethod::Counting)]).unwrap();
    let good = [(data.as_slice(), 2, 3)];
    assert!(
        multi
            .integrate2d_from_slices(&[], 5, 5, &Default::default())
            .is_err()
    );
    assert!(
        multi
            .integrate2d_from_slices(&good, 0, 5, &Default::default())
            .is_err()
    );
    assert!(
        multi
            .integrate2d_from_slices(&good, usize::MAX, 5, &Default::default())
            .is_err()
    );
    assert!(
        multi
            .integrate2d_from_slices(&[(&data, 3, 3)], 5, 5, &Default::default())
            .is_err()
    );
    for range in [(0.0, 0.0), (1.0, -1.0), (f64::NAN, 1.0), (-PI, PI + 0.1)] {
        assert!(
            multi
                .integrate2d_from_slices(
                    &good,
                    5,
                    5,
                    &Multi2dOptions {
                        azimuthal_range: range,
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }
    for factors in [vec![], vec![0.0], vec![-1.0], vec![f64::NAN]] {
        assert!(
            multi
                .integrate2d_from_slices(
                    &good,
                    5,
                    5,
                    &Multi2dOptions {
                        normalization_factors: Some(factors),
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }
    let masked = MultiIntegrator::try_new(vec![
        ai(g.clone(), IntegrationMethod::Counting)
            .with_mask(Mask::from_vec(vec![0; 4], 2, 2).unwrap()),
    ])
    .unwrap();
    assert!(
        masked
            .integrate2d_from_slices(&good, 5, 5, &Default::default())
            .is_err()
    );
    for method in [
        IntegrationMethod::Supersampled { n_ss: 0 },
        IntegrationMethod::Supersampled { n_ss: usize::MAX },
    ] {
        let m = MultiIntegrator::try_new(vec![ai(g.clone(), method)]).unwrap();
        assert!(
            m.integrate2d_from_slices(&good, 5, 5, &Default::default())
                .is_err()
        );
    }
    let m = MultiIntegrator::try_new(vec![
        ai(g.clone(), IntegrationMethod::Counting).with_radial_unit(RadialUnit::QAngstrom),
    ])
    .unwrap();
    let mut invalid = g.clone();
    invalid.poni.wavelength = 0.0;
    let q = MultiIntegrator::try_new(vec![
        ai(invalid, IntegrationMethod::Counting).with_radial_unit(RadialUnit::QAngstrom),
    ])
    .unwrap();
    assert!(
        q.integrate2d_from_slices(&good, 5, 5, &Default::default())
            .is_err()
    );
    assert!(
        m.integrate2d_from_slices(&[(&[f64::NAN; 6], 2, 3)], 5, 5, &Default::default())
            .is_err()
    );
    assert!(MultiIntegrator::try_new(vec![]).is_err());
}

#[test]
fn single_polygon_matches_ordinary_for_corrected_uniform_signal() {
    let mut g = geometry(8, 11);
    g.poni.poni1 = -0.001;
    g.poni.poni2 = -0.001;
    let (d1, d2) = g.corner_grid();
    let corners = g.calc_tth(&d1, &d2);
    let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 10.0 * f64::EPSILON;
    let data: Vec<_> = g.calc_solid_angle().iter().map(|s| s * 50.0).collect();
    let ordinary = ai(g.clone(), IntegrationMethod::PixelSplit);
    let y = ordinary.integrate2d_with_azimuthal_range_from_slice(
        &data,
        8,
        11,
        10,
        13,
        -PI + PI / 13.0,
        TAU / 13.0,
    );
    let multi = MultiIntegrator::try_new(vec![
        ai(g, IntegrationMethod::PixelSplit).with_radial_range(lo, hi),
    ])
    .unwrap();
    let result = multi
        .integrate2d_from_slices(&[(&data, 8, 11)], 10, 13, &Default::default())
        .unwrap();
    close(&result.intensity, &y);
}

#[test]
fn partial_pixels_are_clipped_and_variance_uses_merged_fraction_squared() {
    let mut g = geometry(1, 1);
    g.poni.poni1 = -0.001;
    g.poni.poni2 = -0.001;
    let (d1, d2) = g.corner_grid();
    let corners = g.calc_tth(&d1, &d2);
    let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mid = (lo + hi) * 0.5;
    for method in [
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 12 },
    ] {
        let mut fractions = Vec::new();
        for (min, max) in [(0.0, mid), (mid, hi + 0.001)] {
            let m =
                MultiIntegrator::try_new(vec![ai(g.clone(), method).with_radial_range(min, max)])
                    .unwrap();
            let out = m
                .integrate2d_from_slices(
                    &[(&[100.0], 1, 1)],
                    1,
                    1,
                    &Multi2dOptions {
                        errors: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let weight = out.raw.count[0];
            assert!(weight > 0.0 && weight < 1.0);
            close(&out.raw.sum_signal, &[100.0 * weight]);
            close(
                out.raw.sum_variance.as_ref().unwrap(),
                &[100.0 * weight * weight],
            );
            fractions.push(weight);
        }
        close(&[fractions.iter().sum()], &[1.0]);
    }
}

#[test]
fn complementary_masks_partition_raw_contributions_for_all_methods() {
    let g = geometry(7, 9);
    let data: Vec<_> = (0..g.size()).map(|i| 5.0 + (i % 17) as f64).collect();
    for method in methods() {
        for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
            let make = || ai(g.clone(), method).with_mask_mode(mode);
            let mask_a: Vec<_> = (0..g.size()).map(|i| u8::from(i % 3 == 0)).collect();
            let mask_b: Vec<_> = mask_a.iter().map(|v| 1 - v).collect();
            let unmasked = MultiIntegrator::try_new(vec![make()]).unwrap();
            let partition = MultiIntegrator::try_new(vec![
                make().with_mask(Mask::from_vec(mask_a, 7, 9).unwrap()),
                make().with_mask(Mask::from_vec(mask_b, 7, 9).unwrap()),
            ])
            .unwrap();
            let opts = Multi2dOptions {
                errors: true,
                ..Default::default()
            };
            let a = unmasked
                .integrate2d_from_slices(&[(&data, 7, 9)], 9, 13, &opts)
                .unwrap();
            let b = partition
                .integrate2d_from_slices(&[(&data, 7, 9), (&data, 7, 9)], 9, 13, &opts)
                .unwrap();
            close(&a.intensity, &b.intensity);
            close(a.errors.as_ref().unwrap(), b.errors.as_ref().unwrap());
            close(&a.raw.sum_signal, &b.raw.sum_signal);
            close(&a.raw.sum_norm, &b.raw.sum_norm);
            close(&a.raw.count, &b.raw.count);
            close(
                a.raw.sum_variance.as_ref().unwrap(),
                b.raw.sum_variance.as_ref().unwrap(),
            );
        }
    }
}
