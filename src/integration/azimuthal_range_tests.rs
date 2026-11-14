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

/// Radial values of every subpixel, exactly as the shared-grid and default
/// supersampled tables compute them.
fn subpixel_radial(g: &Geometry, unit: RadialUnit, n_ss: usize) -> Vec<f64> {
    let transform = g.subpixel_transform();
    let mut out = Vec::with_capacity(g.size() * n_ss * n_ss);
    for row in 0..g.dim1 {
        for col in 0..g.dim2 {
            for i in 0..n_ss {
                for j in 0..n_ss {
                    let d1 = row as f64 + (i as f64 + 0.5) / n_ss as f64;
                    let d2 = ((col * n_ss + j) as f64 + 0.5) / n_ss as f64;
                    let (tth, _) = transform.tth_and_azimuth(d1, d2);
                    let q = |scale: f64| 4.0 * PI / (g.poni.wavelength * scale) * (tth * 0.5).sin();
                    out.push(match unit {
                        RadialUnit::TthRad => tth,
                        RadialUnit::TthDeg => tth.to_degrees(),
                        RadialUnit::QAngstrom => q(1e10),
                        RadialUnit::QNm => q(1e9),
                    });
                }
            }
        }
    }
    out
}

/// Default-path azimuthal edges for `ai`'s method.
fn default_azimuthal_edges(ai: &Integrator, n_azimuthal: usize) -> (f64, f64) {
    let state = ai.state.read().unwrap();
    match ai.method {
        IntegrationMethod::Counting => {
            let (lo, hi, _) = histogram::min_max_step(&state.azimuth, n_azimuthal);
            (lo, hi)
        }
        IntegrationMethod::Supersampled { n_ss } => {
            let (lo, hi, _) = histogram::supersampled_azimuthal_min_max_step(
                &ai.geometry,
                &state.azimuth,
                n_azimuthal,
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
}

/// `a` equals `b` to rounding (`1e-12` relative), or both are NaN.
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs() || (a.is_nan() && b.is_nan())
}

/// With the default path's own azimuth edges the cake equals the default one:
/// identical for counting and supersampling, including the bins next to the
/// beam centre, and close for pixel splitting (exact fractional polygon areas
/// vs the legacy pixel-split table). For pixel splitting pixels within 3 px of
/// the beam are masked: the shared grid samples the pixel holding the beam
/// densely, which changes the innermost radial bin.
///
/// Supersampling needs the two-sided radial clamp here: subpixels of the
/// pixels next to the beam lie closer to it than any pixel centre or hull
/// point, below the lower radial edge, and the default table puts them in
/// radial bin 0.
///
/// A subpixel inside the edges whose azimuthal index rounds to `n_azimuthal`
/// lands in the last row on both paths.
#[test]
fn default_edges_reproduce_the_default_cake() {
    let data = image();
    let g = geometry();
    let near_beam: Vec<u8> = g
        .calc_tth_all()
        .iter()
        .map(|t| (g.poni.distance * t.tan() / g.poni.pixel1 < 3.0) as u8)
        .collect();
    let mut below_lower_edge = 0;
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 2 },
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::Supersampled { n_ss: 5 },
    ] {
        let mut ai = integrator(method);
        if matches!(method, IntegrationMethod::PixelSplit) {
            ai.set_mask(Mask {
                data: near_beam.clone(),
                dim1: ROWS,
                dim2: COLS,
            });
        }
        ai.update_geometry_arrays(ROWS, COLS);
        if let IntegrationMethod::Supersampled { n_ss } = method {
            let (r0, _) = ai.default_radial_edges_2d(ROWS, COLS, NR);
            below_lower_edge += subpixel_radial(&g, RadialUnit::TthDeg, n_ss)
                .iter()
                .filter(|r| **r < r0)
                .count();
        }
        let edges = default_azimuthal_edges(&ai, NA);
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
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                    "{method:?} value {k}: {a} != {b}"
                );
            }
        }
    }
    assert!(below_lower_edge > 0, "no subpixel below the lower edge");
}

/// The largest float below `x` (finite, nonzero `x`).
fn next_down(x: f64) -> f64 {
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// Values inside the edges whose bin index rounds to `n` stay in the last bin
/// on the range path (the default tables clamp the same way); without the
/// clamp, as on `MultiIntegrator` grids, they are dropped. In a full circle a
/// value just below the start edge wraps to the last bin.
#[test]
fn index_rounding_stays_in_the_last_bin() {
    // Search for an azimuth just below `hi` whose index rounds to `n`.
    let n = NA;
    let mut found = None;
    'search: for k in 1..2000 {
        let lo = -PI + k as f64 * 1e-4;
        let hi = lo + 2.0;
        let step = (hi - lo) / n as f64;
        let mut v = hi;
        for _ in 0..4 {
            v = next_down(v);
            if ((v - lo) / step) as usize >= n {
                found = Some((lo, hi, v));
                break 'search;
            }
        }
    }
    let (lo, hi, v) = found.expect("no rounding case found");
    let plain = IntegrationGrid2d::new(NR, n, (0.0, 1.0), (lo, hi)).unwrap();
    let clamped = plain.with_clamped_index_rounding();
    assert_eq!(plain.bin(0.5, v), None);
    assert_eq!(clamped.bin(0.5, v), Some((n - 1) * NR + NR / 2));
    assert_eq!(clamped.bin(0.5, hi), None);
    assert_eq!(clamped.bin(0.5, lo - 1e-12), None);

    // Radial: a value just below the upper edge that rounds to `n_radial`.
    let mut found = None;
    'radial: for k in 1..2000 {
        let r1 = 30.0 + k as f64 * 1e-3;
        let step = r1 / NR as f64;
        let mut v = r1;
        for _ in 0..4 {
            v = next_down(v);
            if (v / step) as usize >= NR {
                found = Some((r1, v));
                break 'radial;
            }
        }
    }
    let (r1, v) = found.expect("no radial rounding case found");
    let plain = IntegrationGrid2d::new(NR, n, (0.0, r1), (0.0, 1.0)).unwrap();
    assert_eq!(plain.bin(v, 0.5), None);
    assert_eq!(
        plain.with_clamped_index_rounding().bin(v, 0.5),
        Some(9 * NR + NR - 1)
    );

    // Full circle: the smallest step below -π wraps into the last bin.
    let circle = IntegrationGrid2d::new(NR, n, (0.0, 1.0), (-PI, PI))
        .unwrap()
        .with_clamped_index_rounding();
    let below = next_down(-PI);
    assert_eq!(circle.bin(0.5, below), Some((n - 1) * NR + NR / 2));
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

/// Supersampled twin of the test above: the default upper radial edge is the
/// largest subpixel value plus three epsilons, which no longer lifts it for
/// values of 8 or more (2θ in degrees, q in nm⁻¹). The outermost subpixel must
/// still land in the last radial bin, as on the default path.
#[test]
fn supersampled_keeps_the_outermost_subpixel_in_every_unit() {
    let mut g = geometry();
    g.poni.distance = 0.005;
    let near_beam: Vec<u8> = g
        .calc_tth_all()
        .iter()
        .map(|t| (g.poni.distance * t.tan() / g.poni.pixel1 < 3.0) as u8)
        .collect();
    let data = image();
    for n_ss in [1, 2, 3, 4] {
        for unit in [
            RadialUnit::TthRad,
            RadialUnit::TthDeg,
            RadialUnit::QAngstrom,
            RadialUnit::QNm,
        ] {
            let mut ai = Integrator::new(g.clone())
                .with_method(IntegrationMethod::Supersampled { n_ss })
                .with_radial_unit(unit);
            ai.set_mask(Mask {
                data: near_beam.clone(),
                dim1: ROWS,
                dim2: COLS,
            });
            ai.update_geometry_arrays(ROWS, COLS);
            let (a0, a1, _) = {
                let state = ai.state.read().unwrap();
                histogram::supersampled_azimuthal_min_max_step(
                    &ai.geometry,
                    &state.azimuth,
                    NA,
                    n_ss,
                )
            };
            let want = ai.integrate2d_from_slice(&data, ROWS, COLS, NR, NA);
            let got = run(&ai, &data, (a0, a1), false).unwrap();
            for (k, (a, b)) in got.intensity.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs(),
                    "n_ss={n_ss} {unit:?} row {} radial bin {}: {a} vs {b}",
                    k / NR,
                    k % NR
                );
            }
        }
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

/// Full circles written as `(0, 2π)` or `(-2π, 0)` give the `(-π, π)` cake
/// rotated by half a turn (rows shifted by `n_azimuthal / 2`), with the bin
/// centres in the requested range: identical for counting and supersampling,
/// equal up to rounding for pixel splitting.
#[test]
fn full_circle_can_start_anywhere() {
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::PyfaiFull,
    ] {
        let ai = integrator(method);
        let base = run(&ai, &image(), (-PI, PI), true).unwrap();
        for range in [(0.0, TAU), (-TAU, 0.0)] {
            let got = run(&ai, &image(), range, true).unwrap();
            assert!(got.azimuthal[0] > range.0 && got.azimuthal[NA - 1] < range.1);
            assert_eq!(got.radial, base.radial);
            let (gi, ge) = (&got.intensity, got.errors.as_ref().unwrap());
            let (bi, be) = (&base.intensity, base.errors.as_ref().unwrap());
            let mut worst = 0.0f64;
            for row in 0..NA {
                let src = (row + NA / 2) % NA;
                for r in 0..NR {
                    let (k, j) = (row * NR + r, src * NR + r);
                    for (a, b) in [(gi[k], bi[j]), (ge[k], be[j])] {
                        worst = worst.max((a - b).abs() / b.abs().max(1.0));
                    }
                }
            }
            // pyFAI-style splitting places pixel polygons relative to the
            // range start, so individual pyfai_full bins depend on how the
            // circle is written (documented caveat); only the grid is checked.
            if !matches!(method, IntegrationMethod::PyfaiFull) {
                assert!(worst <= 1e-9, "{method:?} {range:?}: {worst}");
            }
        }
    }
}

/// The default supersampled upper radial edge is a maximum over pixel centres
/// and a hull grid whose floating-point coordinates differ slightly from the
/// real subpixels', so the outermost subpixel can lie an ulp or two above it.
/// The default table keeps it in the last radial bin; so must the range path.
/// Before the fix row 10's last bin was 2.8278168 here instead of 2.8292121.
#[test]
fn supersampled_keeps_a_subpixel_just_above_the_upper_edge() {
    let (rows, cols) = (35, 34);
    let mut p = Poni::new();
    p.pixel1 = 1e-4;
    p.pixel2 = 1.3e-4;
    p.poni1 = -0.0002616938494362545;
    p.poni2 = 0.00048819508704195624;
    p.distance = 0.004909075938396821;
    p.rot1 = 0.02547498522443803;
    p.rot2 = 0.09350941219077002;
    p.rot3 = 0.008132999189795431;
    p.wavelength = 1.1542966923311957e-10;
    let g = Geometry::new(rows, cols, p);
    let n_ss = 3;
    let ai = Integrator::new(g.clone())
        .with_method(IntegrationMethod::Supersampled { n_ss })
        .with_radial_unit(RadialUnit::TthDeg);
    ai.update_geometry_arrays(rows, cols);
    let (_, r1) = ai.default_radial_edges_2d(rows, cols, NR);
    assert!(
        subpixel_radial(&g, RadialUnit::TthDeg, n_ss)
            .iter()
            .any(|r| *r > r1),
        "geometry no longer puts a subpixel above the upper edge"
    );
    let ones = vec![1.0; rows * cols];
    let want = ai.integrate2d_from_slice(&ones, rows, cols, NR, NA);
    let edges = default_azimuthal_edges(&ai, NA);
    let got = ai
        .integrate2d_on_azimuthal_range_from_slice(&ones, rows, cols, NR, NA, edges, false)
        .unwrap();
    let k = 10 * NR + NR - 1;
    assert!((want[k] - 2.8292121).abs() < 1e-6, "default {}", want[k]);
    assert!(
        same(got.intensity[k], want[k]),
        "{} vs {}",
        got.intensity[k],
        want[k]
    );
    for (k, (a, b)) in got.intensity.iter().zip(&want).enumerate() {
        assert!(
            same(*a, *b),
            "row {} radial bin {}: {a} vs {b}",
            k / NR,
            k % NR
        );
    }
}

/// Deterministic sweep over random geometries, units and subpixel counts: on
/// the default path's own edges the range path's radial axis matches the
/// default cake, in particular the last radial bin (subpixels a few ulp above
/// the upper edge) and bin 0 (subpixels below the lower edge). The first and
/// last azimuthal rows are skipped: the default table also clamps a subpixel
/// just outside its data-derived azimuthal edges into them, while this path
/// treats azimuthal edges as true edges.
#[test]
fn supersampled_radial_axis_matches_the_default_on_random_geometries() {
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let (mut above, mut below) = (0, 0);
    for _ in 0..24 {
        let (rows, cols) = (20 + (rnd() * 30.0) as usize, 20 + (rnd() * 30.0) as usize);
        let mut p = Poni::new();
        p.pixel1 = 1e-4;
        p.pixel2 = 1.3e-4;
        p.distance = 0.002 + rnd() * 0.1;
        p.poni1 = (rnd() * 1.4 - 0.2) * rows as f64 * 1e-4;
        p.poni2 = (rnd() * 1.4 - 0.2) * cols as f64 * 1.3e-4;
        p.rot1 = rnd() * 0.6 - 0.3;
        p.rot2 = rnd() * 0.6 - 0.3;
        p.rot3 = rnd() * 0.6 - 0.3;
        p.wavelength = 1e-10 * (0.3 + rnd());
        let g = Geometry::new(rows, cols, p);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| 10.0 + ((i * 37) % 101) as f64)
            .collect();
        for unit in [
            RadialUnit::TthRad,
            RadialUnit::TthDeg,
            RadialUnit::QAngstrom,
            RadialUnit::QNm,
        ] {
            for n_ss in [1, 2, 3, 5] {
                let ai = Integrator::new(g.clone())
                    .with_method(IntegrationMethod::Supersampled { n_ss })
                    .with_radial_unit(unit);
                ai.update_geometry_arrays(rows, cols);
                let (r0, r1) = ai.default_radial_edges_2d(rows, cols, NR);
                let radial = subpixel_radial(&g, unit, n_ss);
                above += radial.iter().any(|r| *r > r1) as usize;
                below += radial.iter().any(|r| *r < r0) as usize;
                let edges = default_azimuthal_edges(&ai, NA);
                let want = ai.integrate2d_from_slice(&data, rows, cols, NR, NA);
                let got = ai
                    .integrate2d_on_azimuthal_range_from_slice(
                        &data, rows, cols, NR, NA, edges, false,
                    )
                    .unwrap();
                let pairs = got.intensity.iter().zip(&want).enumerate();
                for (k, (&a, &b)) in pairs.take((NA - 1) * NR).skip(NR) {
                    assert!(
                        same(a, b),
                        "{rows}x{cols} {:?} {unit:?} n_ss={n_ss} row {} radial bin {}: {a} vs {b}",
                        g.poni,
                        k / NR,
                        k % NR
                    );
                }
            }
        }
    }
    assert!(
        above > 0 && below > 0,
        "sweep exercises: {above} above, {below} below"
    );
}

/// The two-sided radial clamp is only for the range path's data-derived
/// edges. On a user-chosen radial range (`integrate2d_raw_on_grid*`,
/// `MultiIntegrator`) values below the lower edge and at or above the upper
/// edge are still dropped, for every method.
#[test]
fn user_radial_ranges_still_drop_values_outside_them() {
    let g = geometry();
    let ones = vec![1.0; ROWS * COLS];
    let (r0, r1) = (1.0, 3.0);
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::Supersampled { n_ss: 5 },
    ] {
        let n_ss = match method {
            IntegrationMethod::Supersampled { n_ss } => n_ss,
            _ => 1,
        };
        // With one sample at the centre the subpixel is the pixel centre.
        let radial = subpixel_radial(&g, RadialUnit::TthDeg, n_ss);
        let inside = radial.iter().filter(|r| (r0..r1).contains(*r)).count();
        assert!(inside > 100 && inside + 100 < radial.len());
        let expected = inside as f64 / (n_ss * n_ss) as f64;
        let grid = IntegrationGrid2d::new(NR, NA, (r0, r1), (-PI, PI)).unwrap();
        let ai = integrator(method);
        let raw = ai
            .integrate2d_raw_on_grid_from_slice(&ones, ROWS, COLS, &grid, false)
            .unwrap();
        let total: f64 = raw.sum_signal.iter().sum();
        assert!(
            (total - expected).abs() < 1e-9,
            "{method:?} raw: {total} vs {expected}"
        );
        let multi = crate::multi::MultiIntegrator::try_new(vec![
            integrator(method).with_radial_range(r0, r1),
        ])
        .unwrap();
        let out = multi
            .integrate2d_from_slices(
                &[(&ones, ROWS, COLS)],
                NR,
                NA,
                &crate::multi::Multi2dOptions::default(),
            )
            .unwrap();
        let total: f64 = out.raw.sum_signal.iter().sum();
        assert!(
            (total - expected).abs() < 1e-9,
            "{method:?} multi: {total} vs {expected}"
        );
    }
}

/// `MultiIntegrator` and `integrate2d_raw_on_grid*` keep a value inside the
/// edges whose bin index rounds to `n` (in the last bin) but still drop a
/// value exactly at the upper edge and values outside a sector: bins stay
/// half-open.
#[test]
fn shared_grids_keep_values_whose_index_rounds_to_n() {
    // Arithmetic: the largest value below the upper edge rounds to `n`.
    for ((lo, hi), n) in [((0.5, 25.0), 1500), ((0.0, 8.0), 7), ((1.0, 30.8), 100)] {
        let v = next_down(hi);
        let step = (hi - lo) / n as f64;
        assert!(((v - lo) / step) as usize >= n, "({lo}, {hi}) n={n}");
        let grid = IntegrationGrid2d::new(n, 4, (lo, hi), (-PI, PI)).unwrap();
        assert_eq!(grid.bin(v, 0.1), None);
        let grid = grid.with_clamped_index_rounding();
        assert_eq!(grid.bin(v, 0.1), Some(2 * n + n - 1));
        assert_eq!(grid.bin(hi, 0.1), None);
    }
    // A full circle written as (0, 2π): a tiny negative azimuth wraps to 2π.
    let circle = IntegrationGrid2d::new(NR, NA, (0.0, 1.0), (0.0, TAU)).unwrap();
    for az in [-1e-16, -4e-16] {
        assert_eq!(circle.bin(0.5, az), None);
        let clamped = circle.with_clamped_index_rounding();
        assert_eq!(clamped.bin(0.5, az), Some((NA - 1) * NR + NR / 2));
    }
    let sector = IntegrationGrid2d::new(NR, NA, (0.0, 1.0), (0.0, 1.0))
        .unwrap()
        .with_clamped_index_rounding();
    for az in [-1e-16, 1.0, 2.0, -PI] {
        assert_eq!(sector.bin(0.5, az), None, "azimuth {az}");
    }

    // Through the real paths: a one-hot pixel just below the upper edge.
    let ai = integrator(IntegrationMethod::Counting);
    ai.update_geometry_arrays(ROWS, COLS);
    let (radial, azimuth) = {
        let state = ai.state.read().unwrap();
        (state.radial.clone(), state.azimuth.clone())
    };
    let (pixel, lo, n) = (0..radial.len())
        .filter(|&i| radial[i] > 1.0 && (0.2..1.0).contains(&azimuth[i]))
        .flat_map(|i| (0..20).flat_map(move |k| (1..400).map(move |n| (i, k as f64 * 0.05, n))))
        .find(|&(i, lo, n)| {
            let (v, hi) = (radial[i], f64::from_bits(radial[i].to_bits() + 1));
            ((v - lo) / ((hi - lo) / n as f64)) as usize >= n
        })
        .expect("no rounding case found");
    let v = radial[pixel];
    let hi = f64::from_bits(v.to_bits() + 1);
    let mut one_hot = vec![0.0; ROWS * COLS];
    one_hot[pixel] = 1.0;
    let kept = |lo: f64, hi: f64, azimuthal_range: (f64, f64)| -> [f64; 2] {
        let grid = IntegrationGrid2d::new(n, NA, (lo, hi), azimuthal_range).unwrap();
        let raw = ai
            .integrate2d_raw_on_grid_from_slice(&one_hot, ROWS, COLS, &grid, false)
            .unwrap();
        let multi = crate::multi::MultiIntegrator::try_new(vec![
            integrator(IntegrationMethod::Counting).with_radial_range(lo, hi),
        ])
        .unwrap();
        let options = crate::multi::Multi2dOptions {
            azimuthal_range,
            ..Default::default()
        };
        let out = multi
            .integrate2d_from_slices(&[(&one_hot, ROWS, COLS)], n, NA, &options)
            .unwrap();
        let last = |s: &[f64]| (0..NA).map(|a| s[a * n + n - 1]).sum::<f64>();
        [last(&raw.sum_signal), last(&out.raw.sum_signal)]
    };
    assert_eq!(
        kept(lo, hi, (-PI, PI)),
        [1.0, 1.0],
        "rounding to n: lo {lo}, n {n}"
    );
    // Exactly at the upper edge: dropped.
    assert_eq!(kept(lo, v, (-PI, PI)), [0.0, 0.0]);
    // Outside a sector: dropped.
    assert_eq!(kept(lo, hi, (-2.0, -1.0)), [0.0, 0.0]);
}
