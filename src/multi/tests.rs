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

#[test]
fn singular_beam_pixel_has_eightfold_coverage_and_positive_variance() {
    let mut g = geometry(1, 1);
    g.poni.pixel1 = 1e-4;
    g.poni.pixel2 = 1e-4;
    g.poni.poni1 = 5e-5;
    g.poni.poni2 = 5e-5;
    for mode in [PixelSplitMode::Fractional, PixelSplitMode::PyfaiHistogram] {
        let multi = MultiIntegrator::try_new(vec![ai(g.clone(), IntegrationMethod::PixelSplit)])
            .unwrap()
            .with_pixel_split_mode(mode);
        let out = multi
            .integrate2d_from_slices(
                &[(&[25.0], 1, 1)],
                1,
                8,
                &Multi2dOptions {
                    errors: true,
                    ..Default::default()
                },
            )
            .unwrap();
        close(&out.raw.count, &[0.125; 8]);
        close(&out.raw.sum_signal, &[25.0 / 8.0; 8]);
        close(out.raw.sum_variance.as_ref().unwrap(), &[25.0 / 64.0; 8]);
        assert!(out.intensity.iter().all(|x| x.is_finite() && *x > 0.0));
        let sector = multi
            .integrate2d_from_slices(
                &[(&[25.0], 1, 1)],
                1,
                2,
                &Multi2dOptions {
                    azimuthal_range: (0.0, PI / 2.0),
                    ..Default::default()
                },
            )
            .unwrap();
        close(&sector.raw.count, &[0.125; 2]);
    }
    // A tilted detector's beam pixel need not contain the PONI point.
    g.poni.pixel1 = 1.7e-4;
    g.poni.pixel2 = 2.1e-4;
    g.poni.distance = 0.14;
    g.poni.poni1 = 0.00131;
    g.poni.poni2 = 0.00113;
    g.poni.rot1 = 0.007;
    g.poni.rot2 = -0.009;
    g.poni.rot3 = 0.05;
    let m = MultiIntegrator::try_new(vec![ai(g, IntegrationMethod::PixelSplit)]).unwrap();
    let out = m
        .integrate2d_from_slices(&[(&[1.0], 1, 1)], 1, 8, &Default::default())
        .unwrap();
    close(&[out.raw.count.iter().sum()], &[1.0]);
    assert!(out.raw.count.iter().all(|w| *w > 0.0));
}

#[test]
fn histogram_compatibility_is_opt_in_and_is_part_of_the_cache_key() {
    let mut g = geometry(1, 1);
    g.poni.poni1 = -0.001;
    g.poni.poni2 = -0.001;
    let corners = g.calc_tth(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 0.0, 1.0]);
    let hi = (corners.iter().copied().fold(f64::INFINITY, f64::min)
        + corners.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        * 0.5;
    let grid = IntegrationGrid2d::new(1, 1, (0.0, hi), (-PI, PI)).unwrap();
    let ai = ai(g, IntegrationMethod::PixelSplit);
    let a = ai
        .integrate2d_raw_on_grid_from_slice(&[1.0], 1, 1, &grid, true)
        .unwrap();
    let compat = grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram);
    let b = ai
        .integrate2d_raw_on_grid_from_slice(&[1.0], 1, 1, &compat, true)
        .unwrap();
    assert!(a.count[0] > 0.0 && a.count[0] < 1.0);
    close(&b.count, &[1.0]);
    close(b.sum_variance.as_ref().unwrap(), &[1.0]);
    let again = ai
        .integrate2d_raw_on_grid_from_slice(&[1.0], 1, 1, &grid, true)
        .unwrap();
    assert_eq!(a.sum_signal, again.sum_signal);
    assert_eq!(a.sum_norm, again.sum_norm);
}

#[test]
fn histogram_raster_matches_pyfai_boundary_pixel_golden_weight() {
    // pyFAI 2025.12.1 fullSplit2D_engine, no masks/corrections. A pixel
    // straddling an azimuthal edge: polygon area gives 0.1010677423, while
    // the reference's line-area raster gives 0.0106259221 in this bin.
    let mut g = geometry(31, 37);
    g.poni.poni1 = -0.003;
    g.poni.poni2 = -0.002;
    let ai = ai(g, IntegrationMethod::PixelSplit);
    let mut image = vec![0.0; 31 * 37];
    image[8 * 37 + 34] = 1.0;
    let grid = IntegrationGrid2d::new(12, 17, (0.0, 0.2 * (1.0 + f32::EPSILON as f64)), (-PI, PI))
        .unwrap();
    let fraction = ai
        .integrate2d_raw_on_grid_from_slice(&image, 31, 37, &grid, false)
        .unwrap();
    let raster = ai
        .integrate2d_raw_on_grid_from_slice(
            &image,
            31,
            37,
            &grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram),
            false,
        )
        .unwrap();
    assert!((fraction.sum_signal[9 * 12 + 4] - 0.1010677423).abs() < 1e-8);
    assert!((raster.sum_signal[9 * 12 + 4] - 0.0106259221).abs() < 3e-5);
    close(&[raster.sum_signal.iter().sum()], &[1.0]);
    assert!(raster.sum_signal.iter().all(|w| *w >= 0.0 && *w <= 1.0));
}

#[test]
fn histogram_mode_does_not_change_1d_or_other_2d_methods() {
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::Supersampled { n_ss: 3 },
    ] {
        let g = geometry(7, 9);
        let data = vec![20.0; g.size()];
        let a = MultiIntegrator::try_new(vec![ai(g.clone(), method)]).unwrap();
        let b = MultiIntegrator::try_new(vec![ai(g, method)])
            .unwrap()
            .with_pixel_split_mode(PixelSplitMode::PyfaiHistogram);
        let ra = a
            .integrate2d_from_slices(&[(&data, 7, 9)], 10, 13, &Default::default())
            .unwrap();
        let rb = b
            .integrate2d_from_slices(&[(&data, 7, 9)], 10, 13, &Default::default())
            .unwrap();
        assert_eq!(ra.intensity, rb.intensity);
        assert_eq!(
            a.integrate1d_from_slices(&[(&data, 7, 9)], 10),
            b.integrate1d_from_slices(&[(&data, 7, 9)], 10)
        );
    }
}

#[test]
fn azint_style_twenty_one_subpixels_match_vector_geometry_oracle() {
    let mut g1 = geometry(3, 4);
    g1.poni.poni1 = 0.00013;
    g1.poni.poni2 = 0.00017;
    let mut g2 = geometry(4, 3);
    g2.poni.pixel1 = 0.00017;
    g2.poni.distance = 0.14;
    g2.poni.rot1 = 0.006;
    g2.poni.rot2 = -0.009;
    g2.poni.rot3 = 0.05;
    for (unit, hi) in [
        (RadialUnit::TthRad, 0.1),
        (RadialUnit::TthDeg, 0.1_f64.to_degrees()),
        (RadialUnit::QAngstrom, 0.8),
        (RadialUnit::QNm, 8.0),
    ] {
        let geoms = [g1.clone(), g2.clone()];
        let data = [vec![13.0; 12], vec![31.0; 12]];
        let masks: Vec<Vec<u8>> = (0..2)
            .map(|d| (0..12).map(|i| u8::from((i + d) % 5 == 0)).collect())
            .collect();
        let integrators = geoms
            .iter()
            .enumerate()
            .map(|(i, g)| {
                ai(g.clone(), IntegrationMethod::Supersampled { n_ss: 21 })
                    .with_radial_unit(unit)
                    .with_radial_range(0.0, hi)
                    .with_polarization_factor(0.6)
                    .with_mask(Mask::from_vec(masks[i].clone(), g.dim1, g.dim2).unwrap())
            })
            .collect();
        let multi = MultiIntegrator::try_new(integrators).unwrap();
        let slices: Vec<_> = geoms
            .iter()
            .zip(&data)
            .map(|(g, d)| (d.as_slice(), g.dim1, g.dim2))
            .collect();
        let out = multi
            .integrate2d_from_slices(
                &slices,
                8,
                17,
                &Multi2dOptions {
                    errors: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut signal = vec![0.0; 136];
        let mut variance = signal.clone();
        let mut norm = signal.clone();
        let mut count = signal.clone();
        for (d, g) in geoms.iter().enumerate() {
            let mut y = Vec::new();
            let mut x = Vec::new();
            // Independent vector geometry calculation, rather than the scalar
            // transform used to build the production supersampled LUT.
            for row in 0..g.dim1 {
                for col in 0..g.dim2 {
                    for i in 0..21 {
                        for j in 0..21 {
                            y.push(row as f64 + (i as f64 + 0.5) / 21.0);
                            x.push(col as f64 + (j as f64 + 0.5) / 21.0);
                        }
                    }
                }
            }
            let (tth, az) = g.calc_tth_and_azimuth(&y, &x);
            let radial = unit.convert_from_tth(&tth, g.poni.wavelength);
            let centers = g.calc_tth_all();
            let center_az = g.calc_azimuth_all();
            let sa = g.calc_solid_angle();
            let pol = g.calc_polarization(0.6, &centers, &center_az);
            for pixel in 0..12 {
                if masks[d][pixel] != 0 {
                    continue;
                }
                let mut n = vec![0; 136];
                for sample in pixel * 441..(pixel + 1) * 441 {
                    if radial[sample] < 0.0 || radial[sample] >= hi {
                        continue;
                    }
                    let r = (radial[sample] / hi * 8.0).floor() as usize;
                    let a = ((az[sample] + PI).rem_euclid(TAU) / TAU * 17.0).floor() as usize;
                    n[a * 8 + r] += 1;
                }
                for b in 0..136 {
                    let w = n[b] as f64 / 441.0;
                    signal[b] += w * data[d][pixel];
                    variance[b] += w * w * data[d][pixel];
                    norm[b] += w * sa[pixel] * pol[pixel];
                    count[b] += w;
                }
            }
        }
        close(&out.raw.sum_signal, &signal);
        close(out.raw.sum_variance.as_ref().unwrap(), &variance);
        close(&out.raw.sum_norm, &norm);
        close(&out.raw.count, &count);
    }
}

#[test]
fn beam_on_pixel_edge_or_corner_retains_small_radial_disk() {
    let mut g = geometry(1, 2);
    g.poni.pixel1 = 1e-4;
    g.poni.pixel2 = 1e-4;
    let radial_hi = 1e-4_f64;
    for mode in [PixelSplitMode::Fractional, PixelSplitMode::PyfaiHistogram] {
        for corner in [false, true] {
            g.poni.poni1 = if corner { 0.0 } else { 5e-5 };
            g.poni.poni2 = if corner { 0.0 } else { 1e-4 };
            let m = MultiIntegrator::try_new(vec![
                ai(g.clone(), IntegrationMethod::PixelSplit).with_radial_range(0.0, radial_hi),
            ])
            .unwrap()
            .with_pixel_split_mode(mode);
            let out = m
                .integrate2d_from_slices(&[(&[1.0, 1.0], 1, 2)], 1, 8, &Default::default())
                .unwrap();
            let disk = PI * (radial_hi.tan() * g.poni.distance / g.poni.pixel1).powi(2);
            let expected = if corner { disk / 4.0 } else { disk };
            let actual: f64 = out.raw.count.iter().sum();
            // Independent analytic detector-plane area; the polar corner quad
            // misses this entire disk for a beam lying inside a pixel edge.
            assert!(
                (actual - expected).abs() < 3e-4,
                "mode {mode:?}, corner {corner}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn explicit_variances_follow_squared_weights_and_detector_normalization() {
    for method in methods() {
        let g = geometry(9, 12);
        let data = vec![25.0; g.size()];
        let variance = vec![25.0; g.size()];
        let multi =
            MultiIntegrator::try_new(vec![ai(g.clone(), method), ai(g.clone(), method)]).unwrap();
        let slices = [(&data[..], 9, 12), (&data[..], 9, 12)];
        let options = Multi2dOptions {
            errors: true,
            normalization: MultiNormalization::Absolute,
            ..Default::default()
        };
        let poisson = multi
            .integrate2d_from_slices(&slices, 17, 11, &options)
            .unwrap();
        let explicit = multi
            .integrate2d_from_slices_with_variances(
                &slices,
                &[&variance, &variance],
                17,
                11,
                &options,
            )
            .unwrap();
        close(&explicit.intensity, &poisson.intensity);
        close(
            explicit.errors.as_ref().unwrap(),
            poisson.errors.as_ref().unwrap(),
        );
        // An independent gain of -3 multiplies intensity by -3, std by 3.
        // This cannot be recovered by interpreting corrected intensity as counts.
        let corrected = vec![-75.0; g.size()];
        let propagated = vec![225.0; g.size()];
        let changed = multi
            .integrate2d_from_slices_with_variances(
                &[(&corrected, 9, 12), (&corrected, 9, 12)],
                &[&propagated, &propagated],
                17,
                11,
                &options,
            )
            .unwrap();
        close(
            &changed.intensity,
            &poisson
                .intensity
                .iter()
                .map(|v| -3.0 * v)
                .collect::<Vec<_>>(),
        );
        close(
            changed.errors.as_ref().unwrap(),
            &poisson
                .errors
                .as_ref()
                .unwrap()
                .iter()
                .map(|v| 3.0 * v)
                .collect::<Vec<_>>(),
        );
        let mut invalid = variance.clone();
        invalid[2] = -1.0;
        assert!(
            multi
                .integrate2d_from_slices_with_variances(
                    &slices,
                    &[&variance, &invalid],
                    17,
                    11,
                    &options
                )
                .is_err()
        );
        invalid[2] = f64::NAN;
        assert!(
            multi
                .integrate2d_from_slices_with_variances(
                    &slices,
                    &[&variance, &invalid],
                    17,
                    11,
                    &options
                )
                .is_err()
        );
        assert!(
            multi
                .integrate2d_from_slices_with_variances(
                    &slices,
                    &[&variance[..3], &variance],
                    17,
                    11,
                    &options
                )
                .is_err()
        );
        let recovered = multi
            .integrate2d_from_slices_with_variances(
                &slices,
                &[&variance, &variance],
                17,
                11,
                &options,
            )
            .unwrap();
        close(
            recovered.errors.as_ref().unwrap(),
            explicit.errors.as_ref().unwrap(),
        );
    }
}

/// Cut one detector into an 3-row and a 17-row strip. The merged 1D pattern of
/// the strips must equal the pattern of the whole detector: every pixel keeps
/// its own weight, however many pixels its strip puts into a bin.
#[test]
fn split_detector_merges_to_the_whole_detector_pattern_in_1d() {
    let (rows, cols, cut) = (20, 20, 3);
    let mut whole = geometry(rows, cols);
    whole.poni.poni1 = -0.001;
    whole.poni.poni2 = -0.001;
    let mut top = whole.clone();
    top.dim1 = cut;
    let mut bottom = whole.clone();
    bottom.dim1 = rows - cut;
    bottom.poni.poni1 -= cut as f64 * whole.poni.pixel1;
    let top = Geometry::new(top.dim1, cols, top.poni);
    let bottom = Geometry::new(bottom.dim1, cols, bottom.poni);

    let data: Vec<_> = (0..rows * cols)
        .map(|i| 10.0 + (i * 7 % 31) as f64 + 40.0 * (i / cols < cut) as u8 as f64)
        .collect();
    let (data_top, data_bottom) = data.split_at(cut * cols);
    let n = 40;
    let ranged = |g: Geometry, method| {
        Integrator::new(g)
            .with_method(method)
            .with_radial_range(0.012, 0.046)
    };
    for method in methods() {
        let (_, expected, expected_sigma) =
            ranged(whole.clone(), method).integrate1d_with_errors_from_slice(&data, rows, cols, n);
        let multi = MultiIntegrator::new(vec![
            ranged(top.clone(), method),
            ranged(bottom.clone(), method),
        ]);
        let slices = [(data_top, cut, cols), (data_bottom, rows - cut, cols)];
        let (_, merged) = multi.integrate1d_from_slices(&slices, n);
        let (_, merged_e, sigma) = multi.integrate1d_with_errors_from_slices(&slices, n);
        assert!(expected.iter().filter(|&&y| y > 0.0).count() > 20);
        for (name, got, want) in [
            ("intensity", &merged, &expected),
            ("intensity (errors path)", &merged_e, &expected),
            ("sigma", &sigma, &expected_sigma),
        ] {
            for (bin, (&got, &want)) in got.iter().zip(want).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "{method:?} {name} bin {bin}: {got} != {want}"
                );
            }
        }
    }
}

/// Pixels entirely outside the radial range must not reach any bin, in
/// particular not the first or last one, for one detector or several.
#[test]
fn pixels_outside_the_radial_range_do_not_leak_into_edge_bins() {
    let (rows, cols, n) = (20, 20, 12);
    let mut g = geometry(rows, cols);
    g.poni.poni1 = -0.001;
    g.poni.poni2 = -0.001;
    // A second detector further out, entirely above the range.
    let mut outside = g.clone();
    outside.poni.poni1 = -0.006;
    let outside = Geometry::new(rows, cols, outside.poni);
    let (lo, hi) = (0.018, 0.034);
    let (tth, _) = g.angular_corners();
    let out_of_range: Vec<bool> = (0..g.size())
        .map(|p| {
            let r = g.corner_indices(p).map(|i| tth[i]);
            r.iter().all(|&r| r < lo) || r.iter().all(|&r| r > hi)
        })
        .collect();
    assert!(out_of_range.iter().filter(|&&o| o).count() > 50);
    assert!(out_of_range.iter().filter(|&&o| !o).count() > 50);
    let (tth_outside, _) = outside.angular_corners();
    assert!(tth_outside.iter().all(|&r| r > hi));

    let clean: Vec<_> = (0..g.size()).map(|i| 10.0 + (i % 13) as f64).collect();
    let dirty: Vec<_> = clean
        .iter()
        .zip(&out_of_range)
        .map(|(&v, &out)| if out { 1e6 } else { v })
        .collect();
    let bright = vec![1e6; rows * cols];
    for method in methods().into_iter().chain([IntegrationMethod::PyfaiFull]) {
        let ranged = |g: &Geometry| {
            Integrator::new(g.clone())
                .with_method(method)
                .with_radial_range(lo, hi)
        };
        let (_, expected, expected_sigma) =
            ranged(&g).integrate1d_with_errors_from_slice(&clean, rows, cols, n);
        assert!(expected.iter().all(|&y| y > 0.0), "{method:?}");
        let (_, single, single_sigma) =
            ranged(&g).integrate1d_with_errors_from_slice(&dirty, rows, cols, n);
        let multi = MultiIntegrator::new(vec![ranged(&g), ranged(&outside)]);
        let (_, merged, merged_sigma) = multi
            .integrate1d_with_errors_from_slices(&[(&clean, rows, cols), (&bright, rows, cols)], n);
        for (name, got, want) in [
            ("single intensity", &single, &expected),
            ("single sigma", &single_sigma, &expected_sigma),
            ("merged intensity", &merged, &expected),
            ("merged sigma", &merged_sigma, &expected_sigma),
        ] {
            for (bin, (&got, &want)) in got.iter().zip(want).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "{method:?} {name} bin {bin}: {got} != {want}"
                );
            }
        }
    }
}
