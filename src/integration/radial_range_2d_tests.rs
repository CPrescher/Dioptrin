//! 2D integration on an explicit radial range (`Integrator::with_radial_range`).
use super::cache::TwoDKey;
use super::*;
use crate::frame::Frame;
use crate::mask::Mask;
use crate::poni::Poni;
use std::f64::consts::{PI, TAU};

const ROWS: usize = 60;
const COLS: usize = 80;
const NA: usize = 12;
/// Full range in degrees: covers every pixel corner of [`geometry`] (2θ < 6°).
/// With `N_FULL` bins each bin is 0.25° wide, so every edge below is exact.
const FULL: (f64, f64) = (0.0, 8.0);
const N_FULL: usize = 32;
/// Sub-range aligned on the full-range edges: bins 8..16 of the full grid.
const SUB: (f64, f64) = (2.0, 4.0);
const N_SUB: usize = 8;
const FIRST: usize = 8;

/// Beam inside the detector with a slight tilt, as in `azimuthal_range_tests`.
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
        .with_polarization_factor(0.7)
}

fn methods() -> [IntegrationMethod; 4] {
    [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::PyfaiFull,
    ]
}

fn image() -> Vec<f64> {
    (0..ROWS * COLS)
        .map(|i| 10.0 + ((i * 37) % 101) as f64)
        .collect()
}

fn mask() -> Mask {
    Mask {
        data: (0..ROWS * COLS).map(|i| u8::from(i % 13 == 0)).collect(),
        dim1: ROWS,
        dim2: COLS,
    }
}

fn centers((lo, hi): (f64, f64), n: usize) -> Vec<f64> {
    let step = (hi - lo) / n as f64;
    (0..n).map(|i| lo + (i as f64 + 0.5) * step).collect()
}

/// Every 2D entry point that honours the radial range, as
/// `(name, intensity, errors)`, each `NA` rows of `n_radial` bins.
fn all_entry_points(ai: &Integrator, n_radial: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let data = image();
    let frame = Frame {
        name: "frame".into(),
        data: data.clone(),
        dim1: ROWS,
        dim2: COLS,
    };
    let (start, step) = (-PI + PI / NA as f64, TAU / NA as f64);
    let plain = ai.integrate2d_from_slice(&data, ROWS, COLS, n_radial, NA);
    let (with_errors, errors) =
        ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, n_radial, NA);
    let legacy = ai.integrate2d_with_azimuthal_range(&frame, n_radial, NA, start, step);
    let (legacy_e, legacy_err) =
        ai.integrate2d_with_azimuthal_range_and_errors(&frame, n_radial, NA, start, step);
    let range = |r: (f64, f64)| {
        let out = ai
            .integrate2d_on_azimuthal_range(&frame, n_radial, NA, r, true)
            .unwrap();
        assert_eq!(
            out.radial,
            ai.radial_centers_2d(ROWS, COLS, n_radial).unwrap()
        );
        (out.intensity, out.errors.unwrap())
    };
    let (full_circle, full_circle_err) = range((-PI, PI));
    let (sector, sector_err) = range((0.3, 0.3 + PI / 2.0));
    vec![
        ("integrate2d", ai.integrate2d(&frame, n_radial, NA), vec![]),
        ("integrate2d_from_slice", plain, vec![]),
        ("integrate2d_with_errors", with_errors, errors),
        ("integrate2d_with_azimuthal_range", legacy, vec![]),
        ("..._and_errors", legacy_e, legacy_err),
        (
            "on_azimuthal_range full circle",
            full_circle,
            full_circle_err,
        ),
        ("on_azimuthal_range sector", sector, sector_err),
    ]
}

/// (a) The radial axis is `n` equal bins between the requested edges, and
/// (b) bins inside a sub-range agree with the same bins of a full-range
/// integration on matching edges, for every method, entry point and mask
/// mode. The end bins of the sub-range match too, which shows that
/// contributions outside it are dropped rather than clamped into them.
///
/// Counting, pixel splitting and supersampling put every (sub)pixel or
/// polygon piece into the same bin with the same weight on both grids (the
/// 0.25° edges are exact in binary), so they agree bit for bit. `PyfaiFull`
/// splits with pyFAI's histogram approximation, which renormalizes a pixel
/// clipped by the radial range to its part inside: pixels straddling a
/// sub-range edge weigh more in the end bins of the sub-range, so only bins at
/// least two away from its edges are compared, to rounding.
#[test]
fn sub_ranges_match_the_full_range_on_matching_edges() {
    for method in methods() {
        for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
            let make = |range: (f64, f64)| {
                integrator(method)
                    .with_mask(mask())
                    .with_mask_mode(mode)
                    .with_radial_range(range.0, range.1)
            };
            let full = make(FULL);
            let sub = make(SUB);
            assert_eq!(
                full.radial_centers_2d(ROWS, COLS, N_FULL).unwrap(),
                centers(FULL, N_FULL)
            );
            assert_eq!(
                sub.radial_centers_2d(ROWS, COLS, N_SUB).unwrap(),
                centers(SUB, N_SUB)
            );
            let pyfai = matches!(method, IntegrationMethod::PyfaiFull);
            let compared = if pyfai { 2..N_SUB - 2 } else { 0..N_SUB };
            let a = all_entry_points(&full, N_FULL);
            let b = all_entry_points(&sub, N_SUB);
            for ((name, full_i, full_e), (_, sub_i, sub_e)) in a.iter().zip(&b) {
                let mut filled = 0;
                for row in 0..NA {
                    for bin in compared.clone() {
                        let pairs = [(full_i, sub_i), (full_e, sub_e)];
                        for (f, s) in pairs.iter().filter(|(f, _)| !f.is_empty()) {
                            let (x, y) = (f[row * N_FULL + FIRST + bin], s[row * N_SUB + bin]);
                            let ok = if pyfai {
                                (x - y).abs() <= 1e-9 * x.abs().max(1.0)
                                    || (x.is_nan() && y.is_nan())
                            } else {
                                x.to_bits() == y.to_bits()
                            };
                            assert!(
                                ok,
                                "{method:?} {mode:?} {name} row {row} bin {bin}: {x} != {y}"
                            );
                        }
                        filled += usize::from(full_i[row * N_FULL + FIRST + bin] > 0.0);
                    }
                }
                assert!(filled > 10, "{method:?} {name}: only {filled} bins filled");
            }
        }
    }
}

/// Bins outside the requested range get nothing: a sub-range starting above
/// the innermost pixels receives none of the signal of pixels below it.
#[test]
fn contributions_outside_the_range_are_dropped() {
    let g = geometry();
    let tth_deg: Vec<f64> = g.calc_tth_all().iter().map(|t| t.to_degrees()).collect();
    for method in methods() {
        // Only pixels well inside 1.5° are lit; with 2θ edges at 2° and 4°
        // nothing of them (not even a subpixel or polygon corner) is inside.
        let data: Vec<f64> = tth_deg.iter().map(|t| f64::from(*t < 1.5)).collect();
        let ai = integrator(method).with_radial_range(SUB.0, SUB.1);
        let out = ai.integrate2d_from_slice(&data, ROWS, COLS, N_SUB, NA);
        assert!(
            out.iter().all(|v| *v == 0.0 || v.is_nan()),
            "{method:?}: pixels below the range contributed"
        );
        let lit = ai
            .integrate2d_on_azimuthal_range_from_slice(
                &data,
                ROWS,
                COLS,
                N_SUB,
                NA,
                (-PI, PI),
                false,
            )
            .unwrap();
        assert!(lit.intensity.iter().all(|v| *v == 0.0 || v.is_nan()));
        // Pixels inside the range do contribute.
        let inside: Vec<f64> = tth_deg
            .iter()
            .map(|t| f64::from((2.5..3.5).contains(t)))
            .collect();
        let out = ai.integrate2d_from_slice(&inside, ROWS, COLS, N_SUB, NA);
        assert!(
            out.iter().any(|v| *v > 0.0),
            "{method:?}: nothing inside the range"
        );
    }
}

/// (c) Without a radial range every 2D entry point keeps its default table:
/// the same cache key as before, so the same builder and bit-identical bins.
#[test]
fn default_calls_keep_the_default_tables() {
    let data = image();
    for method in methods() {
        let ai = integrator(method);
        let key = |ai: &Integrator| ai.cache_2d.read().unwrap().key.unwrap();
        let plain = ai.integrate2d_from_slice(&data, ROWS, COLS, N_FULL, NA);
        let expected = match method {
            IntegrationMethod::Counting => TwoDKey::Counting {
                n_radial: N_FULL,
                n_azimuthal: NA,
                unit: RadialUnit::TthDeg,
            },
            IntegrationMethod::PixelSplit => TwoDKey::PixelSplit {
                n_radial: N_FULL,
                n_azimuthal: NA,
                unit: RadialUnit::TthDeg,
            },
            IntegrationMethod::Supersampled { n_ss } => TwoDKey::Supersampled {
                n_radial: N_FULL,
                n_azimuthal: NA,
                n_ss,
                unit: RadialUnit::TthDeg,
            },
            _ => TwoDKey::PyfaiFull {
                n_radial: N_FULL,
                n_azimuthal: NA,
                unit: RadialUnit::TthDeg,
                radial_range: None,
                azimuthal: None,
            },
        };
        assert_eq!(key(&ai), expected, "{method:?}");
        let _ = ai.integrate2d_with_azimuthal_range_from_slice(
            &data,
            ROWS,
            COLS,
            N_FULL,
            NA,
            0.1,
            TAU / NA as f64,
        );
        assert!(
            !matches!(key(&ai), TwoDKey::SharedGrid { .. })
                || matches!(method, IntegrationMethod::PyfaiFull),
            "{method:?}: legacy azimuthal range left its table"
        );
        // Setting and clearing a range returns to the very same table.
        let mut toggled = integrator(method);
        toggled.set_radial_range(Some(SUB));
        let _ = toggled.integrate2d_from_slice(&data, ROWS, COLS, N_FULL, NA);
        toggled.set_radial_range(None);
        let again = toggled.integrate2d_from_slice(&data, ROWS, COLS, N_FULL, NA);
        assert_eq!(key(&toggled), expected);
        assert!(
            plain
                .iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

/// (d) Changing the range on the same integrator never serves a table, radial
/// axis or edge cache of the previous range.
#[test]
fn changing_the_range_is_never_stale() {
    let data = image();
    for method in methods() {
        let mut ai = integrator(method);
        let mut seen = Vec::new();
        for range in [Some(FULL), Some(SUB), Some((1.0, 5.0)), None, Some(SUB)] {
            ai.set_radial_range(range);
            let mut fresh = integrator(method);
            fresh.set_radial_range(range);
            for (n_radial, azimuthal) in [(N_SUB, None), (N_SUB, Some((-PI, PI))), (16, None)] {
                let run = |ai: &Integrator| match azimuthal {
                    None => ai.integrate2d_with_errors_from_slice(&data, ROWS, COLS, n_radial, NA),
                    Some(r) => {
                        let out = ai
                            .integrate2d_on_azimuthal_range_from_slice(
                                &data, ROWS, COLS, n_radial, NA, r, true,
                            )
                            .unwrap();
                        assert_eq!(
                            out.radial,
                            ai.radial_centers_2d(ROWS, COLS, n_radial).unwrap()
                        );
                        (out.intensity, out.errors.unwrap())
                    }
                };
                let got = run(&ai);
                let want = run(&fresh);
                assert_eq!(
                    ai.radial_centers_2d(ROWS, COLS, n_radial).unwrap(),
                    fresh.radial_centers_2d(ROWS, COLS, n_radial).unwrap()
                );
                if let Some(range) = range {
                    assert_eq!(
                        ai.radial_centers_2d(ROWS, COLS, n_radial).unwrap(),
                        centers(range, n_radial)
                    );
                }
                let same = |a: &[f64], b: &[f64]| {
                    a.iter()
                        .zip(b)
                        .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
                };
                assert!(
                    same(&got.0, &want.0) && same(&got.1, &want.1),
                    "{method:?} {range:?} {n_radial} {azimuthal:?}: stale result"
                );
                seen.push((range, n_radial, azimuthal.is_some(), got.0));
            }
        }
        // Different ranges give different cakes (the cache was not reused).
        let at = |r: Option<(f64, f64)>| {
            seen.iter()
                .find(|(range, n, az, _)| *range == r && *n == N_SUB && !az)
                .map(|s| s.3.clone())
                .unwrap()
        };
        assert_ne!(at(Some(SUB)), at(Some((1.0, 5.0))), "{method:?}");
        assert_ne!(at(Some(SUB)), at(None), "{method:?}");
    }
}

/// A range the 2D grid cannot use (negative lower edge, as for
/// `MultiIntegrator` and `PyfaiFull`) is an error, never silently ignored.
#[test]
fn unusable_ranges_are_errors() {
    let data = image();
    for method in methods() {
        let ai = integrator(method).with_radial_range(-1.0, 4.0);
        let err = ai
            .integrate2d_on_azimuthal_range_from_slice(&data, ROWS, COLS, 8, NA, (-PI, PI), false)
            .unwrap_err();
        assert!(err.contains("nonnegative"), "{method:?}: {err}");
        assert!(ai.radial_centers_2d(ROWS, COLS, 8).is_err());
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ai.integrate2d_from_slice(&data, ROWS, COLS, 8, NA)
        }));
        assert!(panicked.is_err(), "{method:?}: invalid range ignored");
    }
}

/// Default radial axis: the edges the default tables use.
#[test]
fn default_radial_axis_matches_the_default_edges() {
    for method in methods() {
        let ai = integrator(method);
        let edges = ai.default_radial_edges_2d(&ai.for_shape(ROWS, COLS), 20);
        assert_eq!(
            ai.radial_centers_2d(ROWS, COLS, 20).unwrap(),
            centers(edges, 20)
        );
    }
    let ai = integrator(IntegrationMethod::Histogram);
    assert!(ai.radial_centers_2d(ROWS, COLS, 20).is_err());
}

/// `prepare2d` (used by Python `batch2d` over arrays) builds exactly the table
/// the following integration uses, with and without radial and azimuthal
/// ranges, so parallel workers share it instead of each building one.
#[test]
fn prepare2d_builds_the_table_the_call_uses() {
    let data = image();
    for method in methods() {
        for range in [None, Some(SUB)] {
            for azimuthal in [None, Some((0.3, 0.3 + PI))] {
                let mut ai = integrator(method);
                ai.set_radial_range(range);
                ai.prepare2d(ROWS, COLS, N_SUB, NA, azimuthal);
                let table = |ai: &Integrator| {
                    let cache = ai.cache_2d.read().unwrap();
                    std::sync::Arc::as_ptr(cache.lut.as_ref().unwrap())
                };
                let prepared = table(&ai);
                match azimuthal {
                    None => drop(ai.integrate2d_from_slice(&data, ROWS, COLS, N_SUB, NA)),
                    Some(r) => drop(
                        ai.integrate2d_on_azimuthal_range_from_slice(
                            &data, ROWS, COLS, N_SUB, NA, r, false,
                        )
                        .unwrap(),
                    ),
                }
                assert_eq!(prepared, table(&ai), "{method:?} {range:?} {azimuthal:?}");
            }
        }
    }
}
