//! Tests for `Integrator::integrate2d_on_azimuthal_range*`.
use super::*;
use crate::geometry::Geometry;
use crate::mask::Mask;
use crate::poni::Poni;
use std::f64::consts::PI;

const ROWS: usize = 60;
const COLS: usize = 80;
const NR: usize = 12;
const NA: usize = 18;

/// Beam inside the detector (row 30, col 40) with a slight tilt, so every
/// direction around the beam is covered.
fn geometry() -> Geometry {
    let mut p = Poni::new();
    p.pixel1 = 1e-4;
    p.pixel2 = 1e-4;
    p.distance = 0.05;
    p.poni1 = 0.003;
    p.poni2 = 0.004;
    p.rot1 = 0.01;
    p.rot2 = -0.02;
    p.wavelength = 1e-10;
    Geometry::new(ROWS, COLS, p)
}

fn integrator(method: IntegrationMethod) -> Integrator {
    Integrator::new(geometry())
        .with_method(method)
        .with_radial_unit(RadialUnit::TthDeg)
}

fn methods() -> [IntegrationMethod; 3] {
    [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
    ]
}

fn image() -> Vec<f64> {
    (0..ROWS * COLS)
        .map(|i| 10.0 + ((i * 37) % 101) as f64)
        .collect()
}

fn run(
    ai: &Integrator,
    data: &[f64],
    range: (f64, f64),
    errors: bool,
) -> Result<AzimuthalRange2dResult, String> {
    ai.integrate2d_on_azimuthal_range_from_slice(data, ROWS, COLS, NR, NA, range, errors)
}

/// Pixel azimuth unwrapped around the range centre, so a sector has no seam.
fn unwrapped_azimuth(range: (f64, f64)) -> Vec<f64> {
    let c = 0.5 * (range.0 + range.1);
    geometry()
        .calc_azimuth_all()
        .iter()
        .map(|a| c - PI + (a - c + PI).rem_euclid(TAU))
        .collect()
}

#[test]
fn counting_puts_each_pixel_in_the_bin_holding_its_azimuth() {
    for range in [
        (0.0, PI / 2.0),
        (2.5, 2.5 + PI / 2.0),
        (-PI, PI),
        (0.3, 0.3 + TAU),
    ] {
        let ai = integrator(IntegrationMethod::Counting);
        let ones = vec![1.0; ROWS * COLS];
        let azimuth = unwrapped_azimuth(range);
        let norm = run(&ai, &ones, range, false).unwrap();
        let signal = run(&ai, &azimuth, range, false).unwrap();
        let step = (range.1 - range.0) / NA as f64;
        let full = range.1 - range.0 > 6.0;
        let mut filled_rows = 0;
        for ia in 0..NA {
            let lo = range.0 + ia as f64 * step;
            assert!((norm.azimuthal[ia] - (lo + 0.5 * step)).abs() < 1e-12);
            let mut any = false;
            for ir in 0..NR {
                let k = ia * NR + ir;
                if norm.intensity[k] == 0.0 || (full && (ia == 0 || ia == NA - 1)) {
                    continue; // full circle: the unwrap seam sits in the end bins
                }
                any = true;
                let mean = signal.intensity[k] / norm.intensity[k];
                assert!(
                    mean >= lo - 1e-9 && mean <= lo + step + 1e-9,
                    "{range:?} bin {ia}: mean azimuth {mean} outside [{lo}, {}]",
                    lo + step
                );
            }
            filled_rows += any as usize;
        }
        assert!(
            filled_rows >= NA - 2,
            "{range:?}: only {filled_rows} rows filled"
        );
    }
}

#[test]
fn sectors_exclude_pixels_outside_their_edges() {
    let g = geometry();
    // Distance from the beam in pixels, from 2θ (the tilt moves the beam).
    let radius_px: Vec<f64> = g
        .calc_tth_all()
        .iter()
        .map(|t| g.poni.distance * t.tan() / g.poni.pixel1)
        .collect();
    assert_eq!(radius_px.len(), g.size());
    for method in methods() {
        for range in [(0.0, PI / 2.0), (2.5, 2.5 + PI / 2.0), (-2.0, -1.0)] {
            let ai = integrator(method);
            let azimuth = unwrapped_azimuth(range);
            // Pixels at least 15 px from the beam span < 0.07 rad of azimuth,
            // so one whose centre is 0.1 rad outside cannot touch the sector.
            let far: Vec<f64> = azimuth
                .iter()
                .zip(&radius_px)
                .map(|(a, r)| {
                    (*r > 15.0 && (*a < range.0 - 0.1 || *a > range.1 + 0.1)) as u8 as f64
                })
                .collect();
            assert!(far.iter().sum::<f64>() > 100.0);
            let out = run(&ai, &far, range, false).unwrap();
            assert!(
                out.intensity.iter().all(|v| *v == 0.0),
                "{method:?} {range:?}: pixels outside the sector contributed"
            );
            let inside = run(&ai, &image(), range, false).unwrap();
            assert!(inside.intensity.iter().filter(|v| **v > 0.0).count() > 40);
        }
    }
}

/// With the default path's own azimuth edges the cake equals the default one:
/// identical for counting and supersampling, close for pixel splitting (exact
/// fractional polygon areas vs the legacy pixel-split table). Pixels within
/// 3 px of the beam are masked: the shared grid samples the pixel holding the
/// beam densely, which changes the innermost radial bin. Supersampled row 0
/// may differ slightly too: the default lower azimuth edge is estimated from
/// hull points and subpixels just below it are clamped into bin 0 there,
/// whereas the shared grid leaves them out.
#[test]
fn default_edges_reproduce_the_default_cake() {
    let data = image();
    let g = geometry();
    let near_beam: Vec<u8> = g
        .calc_tth_all()
        .iter()
        .map(|t| (g.poni.distance * t.tan() / g.poni.pixel1 < 3.0) as u8)
        .collect();
    for method in methods() {
        let mut ai = integrator(method);
        ai.set_mask(Mask {
            data: near_beam.clone(),
            dim1: ROWS,
            dim2: COLS,
        });
        ai.update_geometry_arrays(ROWS, COLS);
        let edges = {
            let state = ai.state.read().unwrap();
            match method {
                IntegrationMethod::Counting => {
                    let (lo, hi, _) = histogram::min_max_step(&state.azimuth, NA);
                    (lo, hi)
                }
                IntegrationMethod::Supersampled { n_ss } => {
                    let (lo, hi, _) = histogram::supersampled_azimuthal_min_max_step(
                        &ai.geometry,
                        &state.azimuth,
                        NA,
                        n_ss,
                    );
                    (lo, hi)
                }
                _ => {
                    let (_, corners) = ai.geometry.angular_corners();
                    (
                        math::min(&corners),
                        math::max(&corners) + 10.0 * f64::EPSILON,
                    )
                }
            }
        };
        let (want, want_err) = ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, NR, NA);
        let got = run(&ai, &data, edges, true).unwrap();
        let got_err = got.errors.unwrap();
        if matches!(method, IntegrationMethod::PixelSplit) {
            let mut rel: Vec<f64> = got
                .intensity
                .iter()
                .zip(&want)
                .filter(|(a, b)| **a != 0.0 && **b != 0.0)
                .map(|(a, b)| ((a - b) / b).abs())
                .collect();
            rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = rel[rel.len() / 2];
            assert!(
                median < 1e-2,
                "pixel split median relative difference {median}"
            );
        } else {
            let pairs = got
                .intensity
                .iter()
                .zip(&want)
                .chain(got_err.iter().zip(&want_err));
            for (k, (a, b)) in pairs.enumerate() {
                let row0 = (k % (NR * NA)) < NR;
                let tol = match method {
                    IntegrationMethod::Supersampled { .. } if row0 => 1e-2,
                    _ => 1e-12,
                };
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{method:?} value {k}: {a} != {b}"
                );
            }
        }
    }
}

/// An explicit full circle keeps the default radial grid and applies the same
/// corrections: an image equal to the normalization integrates to 1 in every
/// filled cell, as it does on the default path.
#[test]
fn explicit_full_circle_keeps_radial_grid_and_corrections() {
    for method in methods() {
        let mut ai = integrator(method);
        ai.set_polarization_factor(Some(0.95));
        ai.update_geometry_arrays(ROWS, COLS);
        let normalization = ai.get_normalization().to_vec();
        let default = ai.integrate2d_from_slice(&normalization, ROWS, COLS, NR, NA);
        let full = run(&ai, &normalization, (-PI, PI), false).unwrap();
        for cake in [&default, &full.intensity] {
            let filled: Vec<_> = cake.iter().filter(|v| **v != 0.0).collect();
            assert!(filled.len() > NR * NA / 2);
            assert!(
                filled.iter().all(|v| (**v - 1.0).abs() < 1e-12),
                "{method:?}"
            );
        }
        // Radial centres are those of the default grid.
        let step = (full.radial[NR - 1] - full.radial[0]) / (NR - 1) as f64;
        let lo = full.radial[0] - 0.5 * step;
        let expected = ai.default_radial_edges_2d(ROWS, COLS, NR);
        assert!(
            (lo - expected.0).abs() < 1e-12 && (lo + NR as f64 * step - expected.1).abs() < 1e-9
        );
    }
}

#[test]
fn mask_and_errors_follow_the_default_path() {
    let data = image();
    let mask: Vec<u8> = (0..ROWS * COLS)
        .map(|i| (i % 7 == 0 || i / COLS < 5) as u8)
        .collect();
    let mut garbage = data.clone();
    for (v, m) in garbage.iter_mut().zip(&mask) {
        if *m != 0 {
            *v = 1e12;
        }
    }
    for method in methods() {
        let mut ai = integrator(method);
        ai.set_mask(Mask {
            data: mask.clone(),
            dim1: ROWS,
            dim2: COLS,
        });
        let range = (0.5, 0.5 + PI);
        let clean = run(&ai, &data, range, true).unwrap();
        let dirty = run(&ai, &garbage, range, true).unwrap();
        assert_eq!(
            clean.intensity, dirty.intensity,
            "{method:?}: masked pixels leaked"
        );
        assert_eq!(clean.errors, dirty.errors);
        let errors = clean.errors.unwrap();
        for (i, e) in clean.intensity.iter().zip(&errors) {
            assert!(e.is_finite() && *e >= 0.0 && (*i == 0.0) == (*e == 0.0));
        }
        let plain = run(&ai, &data, range, false).unwrap();
        assert_eq!(plain.intensity, clean.intensity);
        assert_eq!(plain.errors, None);
    }
}

#[test]
fn invalid_ranges_and_histogram_are_rejected() {
    let ai = integrator(IntegrationMethod::Counting);
    let data = image();
    for range in [
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (1.0, 1.0),
        (1.0, -1.0),
        (0.0, TAU + 0.01),
        (-7.0, -6.0),
    ] {
        assert!(run(&ai, &data, range, false).is_err(), "{range:?} accepted");
    }
    let hist = integrator(IntegrationMethod::Histogram);
    let err = run(&hist, &data, (-PI, PI), false).unwrap_err();
    assert!(err.contains("Histogram"), "{err}");
}

fn nonfinite(v: &[f64]) -> Vec<usize> {
    (0..v.len()).filter(|&i| !v[i].is_finite()).collect()
}

fn bit_identical(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Non-finite pixels are not rejected: as on the default path they affect only
/// the bins they contribute to, for intensity and errors, strict and fast math.
#[test]
fn non_finite_pixels_follow_the_default_path() {
    // A pixel well away from the beam, at an azimuth inside (0, pi/2).
    let g = geometry();
    let azimuth = g.calc_azimuth_all();
    let tth = g.calc_tth_all();
    let pixel = (0..g.size())
        .find(|&i| (0.6..0.9).contains(&azimuth[i]) && tth[i] > 0.03)
        .unwrap();
    let mut one_hot = vec![0.0; g.size()];
    one_hot[pixel] = 1.0;
    let all = [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::PyfaiFull,
    ];
    for bad in [f64::NAN, f64::INFINITY] {
        let mut data = image();
        data[pixel] = bad;
        for method in all {
            for fast in [true, false] {
                let ai = integrator(method).with_fast_math(fast);
                let (want, want_err) =
                    ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, NR, NA);
                let got = run(&ai, &data, (-PI, PI), true).unwrap();
                let got_err = got.errors.clone().unwrap();
                let touched = run(&ai, &one_hot, (-PI, PI), false).unwrap();
                let touched: Vec<usize> = (0..NR * NA)
                    .filter(|&k| touched.intensity[k] != 0.0)
                    .collect();
                assert!(!touched.is_empty());
                assert_eq!(
                    nonfinite(&got.intensity),
                    touched,
                    "{bad} {method:?} fast={fast}"
                );
                // NaN signal gives NaN intensity but a finite Poisson error,
                // because max(NaN, 0) is 0; inf gives inf for both.
                let err_bad = if bad.is_nan() {
                    vec![]
                } else {
                    touched.clone()
                };
                assert_eq!(nonfinite(&got_err), err_bad, "{bad} {method:?} fast={fast}");
                assert_eq!(
                    nonfinite(&want).len(),
                    touched.len(),
                    "{bad} {method:?} default"
                );
                if matches!(method, IntegrationMethod::PyfaiFull) {
                    // The default pyfai_full grid is the same full circle.
                    assert!(bit_identical(&got.intensity, &want), "{bad} fast={fast}");
                    assert!(bit_identical(&got_err, &want_err), "{bad} fast={fast}");
                }
                // A sector that does not contain the pixel stays finite.
                let other = run(&ai, &data, (-2.0, -1.0), true).unwrap();
                assert!(
                    nonfinite(&other.intensity).is_empty()
                        && nonfinite(&other.errors.unwrap()).is_empty()
                );
            }
        }
    }
}

/// Close detector: the largest radial value exceeds 32 in 2θ degrees and in
/// q nm⁻¹, where `min_max_step`'s 10 ulp margin no longer lifts the upper
/// edge above it. Counting must still put that pixel in the last bin, as the
/// default path does, and reproduce the default cake on its own edges.
#[test]
fn counting_keeps_the_outermost_pixel_in_every_unit() {
    let mut g = geometry();
    g.poni.distance = 0.005;
    for unit in [
        RadialUnit::TthRad,
        RadialUnit::TthDeg,
        RadialUnit::QAngstrom,
        RadialUnit::QNm,
    ] {
        let ai = Integrator::new(g.clone())
            .with_method(IntegrationMethod::Counting)
            .with_radial_unit(unit);
        let radial = unit.convert_from_tth(&g.calc_tth_all(), g.poni.wavelength);
        let outer = (0..radial.len())
            .max_by(|a, b| radial[*a].total_cmp(&radial[*b]))
            .unwrap();
        if matches!(unit, RadialUnit::TthDeg | RadialUnit::QNm) {
            assert!(radial[outer] > 32.0, "{unit:?}: {}", radial[outer]);
        }
        let (a0, a1, _) = histogram::min_max_step(&g.calc_azimuth_all(), NA);
        let mut one_hot = vec![0.0; g.size()];
        one_hot[outer] = 1000.0;
        let want = ai.integrate2d_from_slice(&one_hot, ROWS, COLS, NR, NA);
        let got = run(&ai, &one_hot, (a0, a1), false).unwrap();
        assert!(want.iter().sum::<f64>() > 0.0);
        assert!(
            bit_identical(&got.intensity, &want),
            "{unit:?}: outermost pixel"
        );
        let full = run(&ai, &one_hot, (-PI, PI), false).unwrap();
        assert!(
            full.intensity.iter().sum::<f64>() > 0.0,
            "{unit:?}: dropped on (-pi, pi)"
        );
        let data = image();
        let (want, want_err) = ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, NR, NA);
        let got = run(&ai, &data, (a0, a1), true).unwrap();
        assert!(bit_identical(&got.intensity, &want), "{unit:?}");
        assert!(bit_identical(&got.errors.unwrap(), &want_err), "{unit:?}");
    }
}

/// pyfai_full: the explicit full circle is the default pyfai_full cake, and
/// a sector keeps only pixel fractions inside it.
#[test]
fn pyfai_full_on_the_range_path() {
    let data = image();
    let ai = integrator(IntegrationMethod::PyfaiFull);
    let (want, want_err) = ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, NR, NA);
    let got = run(&ai, &data, (-PI, PI), true).unwrap();
    assert!(bit_identical(&got.intensity, &want));
    assert!(bit_identical(&got.errors.unwrap(), &want_err));
    let range = (0.0, PI / 2.0);
    let azimuth = unwrapped_azimuth(range);
    let radius_px: Vec<f64> = geometry()
        .calc_tth_all()
        .iter()
        .map(|t| 0.05 * t.tan() / 1e-4)
        .collect();
    let far: Vec<f64> = azimuth
        .iter()
        .zip(&radius_px)
        .map(|(a, r)| (*r > 15.0 && (*a < range.0 - 0.1 || *a > range.1 + 0.1)) as u8 as f64)
        .collect();
    assert!(
        run(&ai, &far, range, false)
            .unwrap()
            .intensity
            .iter()
            .all(|v| *v == 0.0)
    );
    let sector = run(&ai, &data, range, false).unwrap();
    assert!(sector.intensity.iter().filter(|v| **v > 0.0).count() > 40);
}
