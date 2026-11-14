//! Transformed pixel maps (`Detector::transformed`, Dioptas flips and
//! rotations) on binned images. A transformed detector on the transformed
//! image must give what its base detector gives on the original image, with
//! the same binning, for every method and path, since every binned pixel sits
//! at the same physical position. The bases have module gaps inside binned
//! pixels, non-uniform pitch, a curved surface and a mirrored orientation, so
//! a binned pixel spans several unequal base pixels and a gap.
//!
//! The full matrix runs with the beam beside the detector: a beam pixel is
//! sampled at 65,536 points per table build, which dominates debug builds.
//! One beam-on case per method and path keeps the singular-pixel paths.
use super::*;
use crate::detector::{Detector, DetectorTransform, PixelMap};
use crate::multi::{Multi2dOptions, MultiIntegrator};
use crate::poni::Poni;
use DetectorTransform::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

/// Unbinned base shape; binnings [2, 2] and [2, 3] give 18 × 36 and 18 × 24.
const FULL: [usize; 2] = [36, 72];
const PIXEL: f64 = 5e-5;
const OPERATIONS: [DetectorTransform; 7] = [
    Half, Transpose, Transverse, Left, Right, Horizontal, Vertical,
];

/// Separable base with gaps before fine rows 7 and 20 and columns 11 and 40
/// (all inside binned pixels), a slightly non-uniform pitch and, if `curved`,
/// a curved surface.
fn base(orientation: u8, curved: bool) -> Detector {
    let edges = |n: usize, gaps: [usize; 2], axis: usize| {
        let mut lo = 0.;
        (0..n)
            .map(|i| {
                if gaps.contains(&i) {
                    lo += 0.4 * PIXEL;
                }
                let hi = lo + PIXEL * (1. + 0.05 * (i as f64 * 0.7).sin());
                let z = |x: f64| {
                    if curved {
                        2e-4 * (x / 3e-3).powi(2)
                    } else {
                        0.
                    }
                };
                let (mut a, mut b) = ([z(lo), 0., 0.], [z(hi), 0., 0.]);
                a[axis] = lo;
                b[axis] = hi;
                lo = hi;
                [a, b]
            })
            .collect()
    };
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.max_shape = Some(FULL);
    d.orientation = orientation;
    d.map = PixelMap::Separable {
        slow: edges(FULL[0], [7, 20], 1),
        fast: edges(FULL[1], [11, 40], 2),
    };
    d
}

fn bases() -> [(&'static str, Detector); 2] {
    [
        ("gapped", base(3, false)),
        ("gapped curved orientation 2", base(2, true)),
    ]
}

/// Tilted; the beam either on the detector (singular pixel) or beside it, past
/// the last column. The ±π seam crosses the image either way.
fn poni(beam_on: bool) -> Poni {
    let mut p = Poni::new();
    p.pixel1 = PIXEL;
    p.pixel2 = PIXEL;
    p.distance = 0.03;
    p.poni1 = 15.3 * PIXEL;
    p.poni2 = if beam_on { 41.7 * PIXEL } else { 85.0 * PIXEL };
    p.rot1 = 0.02;
    p.rot2 = -0.015;
    p.wavelength = 1e-10;
    p
}

/// Base pixel of pixel (r, c) of a transformed image of shape `[h, w]`.
fn inverse(op: DetectorTransform, [h, w]: [usize; 2], r: usize, c: usize) -> (usize, usize) {
    match op {
        Identity => (r, c),
        Half => (h - 1 - r, w - 1 - c),
        Transpose => (c, r),
        Transverse => (w - 1 - c, h - 1 - r),
        Left => (c, h - 1 - r),
        Right => (w - 1 - c, r),
        Horizontal => (r, w - 1 - c),
        Vertical => (h - 1 - r, c),
    }
}

/// One binned case: the transformed detector's geometry, image and shape, and
/// the base detector's.
struct Case {
    label: String,
    bin: [usize; 2],
    op: DetectorTransform,
    beam_on: bool,
    method: IntegrationMethod,
    test: Geometry,
    image: Vec<f64>,
    shape: [usize; 2],
    reference: Geometry,
    reference_image: Vec<f64>,
    reference_shape: [usize; 2],
}

impl Case {
    fn integrator(&self, geometry: &Geometry, range: Option<(f64, f64)>) -> Integrator {
        let ai = Integrator::new(geometry.clone()).with_method(self.method);
        match range {
            Some((lo, hi)) => ai.with_radial_range(lo, hi),
            None => ai,
        }
    }
    fn test(&self, range: Option<(f64, f64)>) -> Integrator {
        self.integrator(&self.test, range)
    }
    fn reference(&self, range: Option<(f64, f64)>) -> Integrator {
        self.integrator(&self.reference, range)
    }
}

/// Every base, binning and operation with the beam off the detector, plus one
/// beam-on case: the curved, mirrored base at [2, 3] under Transverse (axes
/// swapped and mirrored).
fn cases(method: IntegrationMethod) -> Vec<Case> {
    let mut matrix = vec![];
    for (name, base) in bases() {
        for bin in [[2, 2], [2, 3]] {
            for op in OPERATIONS {
                matrix.push((name, base.clone(), bin, op, false));
            }
        }
    }
    let (name, base) = bases().into_iter().last().unwrap();
    matrix.push((name, base, [2, 3], Transverse, true));
    matrix
        .into_iter()
        .map(|(name, base, bin, op, beam_on)| case(method, name, base, bin, op, beam_on))
        .collect()
}

fn case(
    method: IntegrationMethod,
    name: &str,
    base: Detector,
    bin: [usize; 2],
    op: DetectorTransform,
    beam_on: bool,
) -> Case {
    let bshape = [FULL[0] / bin[0], FULL[1] / bin[1]];
    let reference_image: Vec<f64> = (0..bshape[0] * bshape[1])
        .map(|i| {
            let (r, c) = ((i / bshape[1]) as f64, (i % bshape[1]) as f64);
            100.0 + 3.0 * r + 7.0 * c + 0.05 * r * c
        })
        .collect();
    let geometry = |d: Detector, shape: [usize; 2]| {
        Geometry::new(shape[0], shape[1], poni(beam_on))
            .with_detector(Arc::new(d))
            .unwrap()
    };
    let d = base.transformed(op).unwrap();
    // The same binning of the base pixels, in the transformed axes.
    let shape = if d.max_shape == Some(FULL) {
        bshape
    } else {
        [bshape[1], bshape[0]]
    };
    let image = (0..shape[0] * shape[1])
        .map(|i| {
            let (r, c) = inverse(op, shape, i / shape[1], i % shape[1]);
            reference_image[r * bshape[1] + c]
        })
        .collect();
    Case {
        label: format!("{name}, bin {bin:?}, {op:?}, beam on {beam_on}, {method:?}"),
        bin,
        op,
        beam_on,
        method,
        test: geometry(d, shape),
        image,
        shape,
        reference: geometry(base, bshape),
        reference_image,
        reference_shape: bshape,
    }
}

/// Agreement to 1e-6 of the largest reference value (raw sums, or patterns
/// without empty bins).
#[track_caller]
fn assert_scaled(label: &str, test: &[f64], reference: &[f64]) {
    assert_eq!(test.len(), reference.len(), "{label}: length");
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{label}: empty reference");
    for (i, (t, r)) in test.iter().zip(reference).enumerate() {
        assert!(
            (t - r).abs() <= 1e-6 * scale,
            "{label}: bin {i}: {t} vs {r}"
        );
    }
}

/// Normalized intensities agree to 1e-6 relative in bins the reference covers
/// by more than a rounding sliver (pyFAI kernels accumulate in f32).
#[track_caller]
fn assert_covered(label: &str, test: &[f64], reference: &[f64], norm: &[f64]) {
    assert_eq!(test.len(), reference.len(), "{label}: length");
    let max = norm.iter().fold(0.0f64, |m, v| m.max(*v));
    let mut checked = 0;
    for (i, (t, r)) in test.iter().zip(reference).enumerate() {
        if norm[i] > 1e-6 * max {
            checked += 1;
            assert!(
                (t - r).abs() <= 1e-6 * r.abs(),
                "{label}: bin {i}: {t} vs {r}"
            );
        }
    }
    assert!(checked > 0, "{label}: nothing compared");
}

const METHODS: [IntegrationMethod; 4] = [
    IntegrationMethod::PyfaiFull,
    IntegrationMethod::PixelSplit,
    IntegrationMethod::Counting,
    IntegrationMethod::Supersampled { n_ss: 3 },
];

/// Every case of every method, in parallel.
fn each_case(methods: &[IntegrationMethod], check: impl Fn(&Case) + Sync + Send) {
    let cases: Vec<Case> = methods.iter().flat_map(|&m| cases(m)).collect();
    cases.par_iter().for_each(check);
}

/// 1D patterns with automatic radial ranges (supersampled ranges come from
/// edge subpixels) and with an explicit range.
#[test]
fn binned_transformed_maps_1d_match_the_base_detector() {
    each_case(&METHODS, |c| {
        for range in [None, Some((0.004, 0.09))] {
            let run = |ai: Integrator, image: &[f64], [d1, d2]: [usize; 2]| {
                ai.integrate1d_from_slice(image, d1, d2, 40)
            };
            let (t, r) = (
                run(c.test(range), &c.image, c.shape),
                run(c.reference(range), &c.reference_image, c.reference_shape),
            );
            let label = format!("{} 1D {range:?}", c.label);
            assert_scaled(&format!("{label} radial"), &t.0, &r.0);
            assert_scaled(&label, &t.1, &r.1);
        }
    });
}

/// Legacy cakes (`integrate2d`): `pyfai_full` (shared corner grid),
/// `pixel_split` (per-pixel corners) and supersampled (automatic radial and
/// azimuthal ranges from edge subpixels).
#[test]
fn binned_transformed_maps_cakes_match_the_base_detector() {
    each_case(&METHODS, |c| {
        let label = format!("{} cake", c.label);
        let [d1, d2] = c.shape;
        let [r1, r2] = c.reference_shape;
        let t = c
            .test(None)
            .integrate2d_from_slice(&c.image, d1, d2, 20, 36);
        let r = c
            .reference(None)
            .integrate2d_from_slice(&c.reference_image, r1, r2, 20, 36);
        if matches!(c.method, IntegrationMethod::PyfaiFull) {
            let [lo, hi] = crate::lut::pyfai_radial_range(&c.reference, RadialUnit::TthRad);
            let grid = IntegrationGrid2d::new(20, 36, (lo, hi), (-PI, PI)).unwrap();
            let ones = vec![1.0; r1 * r2];
            let norm = c
                .reference(None)
                .integrate2d_raw_on_grid_from_slice(&ones, r1, r2, &grid, false)
                .unwrap()
                .sum_norm;
            assert_covered(&label, &t, &r, &norm);
        } else {
            assert_scaled(&label, &t, &r);
        }
    });
}

/// Shared-grid raw sums (`integrate2d_raw_on_grid*`), both pixel-split modes
/// for the methods that split by corners (counting and supersampled ignore
/// the mode).
#[test]
fn binned_transformed_maps_shared_grid_raw_sums_match_the_base_detector() {
    each_case(&METHODS, |c| {
        let modes: &[PixelSplitMode] = match c.method {
            IntegrationMethod::PyfaiFull | IntegrationMethod::PixelSplit => {
                &[PixelSplitMode::Fractional, PixelSplitMode::PyfaiHistogram]
            }
            _ => &[PixelSplitMode::Fractional],
        };
        for &mode in modes {
            for azimuth in [(-PI, PI), (2.0, 2.0 + PI)] {
                let grid = IntegrationGrid2d::new(16, 18, (0.0, 0.09), azimuth)
                    .unwrap()
                    .with_pixel_split_mode(mode);
                let [d1, d2] = c.shape;
                let [r1, r2] = c.reference_shape;
                let t = c
                    .test(None)
                    .integrate2d_raw_on_grid_from_slice(&c.image, d1, d2, &grid, false)
                    .unwrap();
                let r = c
                    .reference(None)
                    .integrate2d_raw_on_grid_from_slice(&c.reference_image, r1, r2, &grid, false)
                    .unwrap();
                let label = format!("{} grid {mode:?} {azimuth:?}", c.label);
                assert_scaled(&format!("{label} signal"), &t.sum_signal, &r.sum_signal);
                assert_scaled(&format!("{label} norm"), &t.sum_norm, &r.sum_norm);
            }
        }
    });
}

/// `MultiIntegrator` 2D with a binned transformed detector next to an
/// untransformed one. It builds the shared-grid tables tested above, so only
/// the wiring is checked here: unequal binning, an axis swap with a mirror
/// and a pure mirror, plus the beam-on case.
#[test]
fn binned_transformed_maps_in_a_multi_integrator_match_the_base_detector() {
    each_case(
        &[IntegrationMethod::PyfaiFull, IntegrationMethod::PixelSplit],
        |c| {
            if c.bin != [2, 3] || !(c.beam_on || matches!(c.op, Transverse | Horizontal)) {
                return;
            }
            let range = Some((0.0, 0.09));
            let other_image: Vec<f64> = c.reference_image.iter().map(|v| 2.0 * v - 50.0).collect();
            let test = MultiIntegrator::try_new(vec![c.test(range), c.reference(range)]).unwrap();
            let reference =
                MultiIntegrator::try_new(vec![c.reference(range), c.reference(range)]).unwrap();
            let [d1, d2] = c.shape;
            let [r1, r2] = c.reference_shape;
            let test_slices: [(&[f64], usize, usize); 2] =
                [(&c.image, d1, d2), (&other_image, r1, r2)];
            let reference_slices: [(&[f64], usize, usize); 2] =
                [(&c.reference_image, r1, r2), (&other_image, r1, r2)];
            for azimuthal_range in [(-PI, PI), (2.5, 2.5 + PI / 2.0)] {
                let options = Multi2dOptions {
                    azimuthal_range,
                    ..Default::default()
                };
                let t = test
                    .integrate2d_from_slices(&test_slices, 16, 12, &options)
                    .unwrap();
                let r = reference
                    .integrate2d_from_slices(&reference_slices, 16, 12, &options)
                    .unwrap();
                let label = format!("{} multi 2D {azimuthal_range:?}", c.label);
                assert_scaled(&format!("{label} norm"), &t.raw.sum_norm, &r.raw.sum_norm);
                assert_covered(&label, &t.intensity, &r.intensity, &r.raw.sum_norm);
            }
        },
    );
}

/// Solid angles of curved detectors take the pixel normal in image-index
/// order (pyFAI's `cos_incidence`), ignoring a mirrored base orientation; a
/// transformed map must take the normal its base takes for that pixel.
#[test]
fn transformed_curved_maps_solid_angle_matches_the_base_detector() {
    for orientation in [1, 2, 3, 4] {
        let base = base(orientation, true);
        for bin in [[1, 1], [2, 2], [2, 3]] {
            let bshape = [FULL[0] / bin[0], FULL[1] / bin[1]];
            let solid_angle = |d: Detector, [d1, d2]: [usize; 2]| {
                Geometry::new(d1, d2, poni(true))
                    .with_detector(Arc::new(d))
                    .unwrap()
                    .calc_solid_angle()
            };
            let reference = solid_angle(base.clone(), bshape);
            for op in OPERATIONS {
                let d = base.transformed(op).unwrap();
                let shape = if d.max_shape == Some(FULL) {
                    bshape
                } else {
                    [bshape[1], bshape[0]]
                };
                let test = solid_angle(d, shape);
                for (i, t) in test.iter().enumerate() {
                    let (r, c) = inverse(op, shape, i / shape[1], i % shape[1]);
                    let expected = reference[r * bshape[1] + c];
                    assert!(
                        (t - expected).abs() <= 1e-12 * expected,
                        "orientation {orientation}, bin {bin:?}, {op:?}: pixel {i}: {t} vs {expected}"
                    );
                }
            }
        }
    }
}
