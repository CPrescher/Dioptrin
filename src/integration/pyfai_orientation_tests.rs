//! `PyfaiFull` with pixel corners that trace the detector mirrored: detector
//! orientations 2 and 4 (one axis flipped) and transformed pixel maps. A
//! mirrored detector on an image must give what the default orientation gives
//! on the correspondingly flipped image, since every pixel sits at the same
//! physical position. pyFAI's full splitting is winding-sensitive (seam test
//! and signed raster areas), so this fails when the winding is not accounted
//! for.
//!
//! The reference kernels accumulate in f32, and the mirrored contour starts at
//! another corner, so agreement is to f32 rounding: about 1e-6 per pixel
//! weight (orientation 1, which only rotates the corner order, shows the same).
//! Rounding can also leave or remove a sliver of weight (~1e-8 of a pixel) in
//! an otherwise empty bin; normalized intensities are compared only in bins
//! with real coverage.
use super::*;
use crate::detector::{Detector, DetectorTransform, PixelMap};
use crate::lut::Lut1d;
use crate::multi::{Multi2dOptions, MultiIntegrator};
use crate::poni::Poni;
use std::f64::consts::PI;
use std::sync::Arc;

const N1: usize = 10;
const N2: usize = 14;
const PIXEL: f64 = 1e-4;

/// Beam on the detector inside pixel (4, 7), so the ±π seam crosses the
/// image and the beam pixel is singular; or beam off the detector, where no
/// pixel crosses the seam.
fn poni(beam_on: bool) -> Poni {
    let mut p = Poni::new();
    p.pixel1 = PIXEL;
    p.pixel2 = PIXEL;
    p.distance = 0.02;
    (p.poni1, p.poni2) = if beam_on {
        (4.3 * PIXEL, 7.7 * PIXEL)
    } else {
        (-5.0 * PIXEL, -7.0 * PIXEL)
    };
    p.wavelength = 1e-10;
    p
}

/// Smooth image without mirror symmetry, on a shape of `dims`.
fn image(dims: (usize, usize)) -> Vec<f64> {
    (0..dims.0 * dims.1)
        .map(|i| {
            let (r, c) = ((i / dims.1) as f64, (i % dims.1) as f64);
            100.0 + 3.0 * r + 7.0 * c + 0.05 * r * c
        })
        .collect()
}

/// Index of pixel `i` of an N1 x N2 image after flipping rows and/or columns.
fn flip_index(i: usize, rows: bool, cols: bool) -> usize {
    let (r, c) = (i / N2, i % N2);
    let r = if rows { N1 - 1 - r } else { r };
    let c = if cols { N2 - 1 - c } else { c };
    r * N2 + c
}

fn flipped(image: &[f64], rows: bool, cols: bool) -> Vec<f64> {
    (0..N1 * N2)
        .map(|i| image[flip_index(i, rows, cols)])
        .collect()
}

/// (rows, cols) flipped by a pyFAI orientation relative to orientation 3.
fn flips(orientation: u8) -> (bool, bool) {
    (matches!(orientation, 1 | 2), matches!(orientation, 1 | 4))
}

fn uniform(orientation: u8) -> Detector {
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.orientation = orientation;
    d
}

/// Separable map; with `gap` the lower half of the rows is shifted, so the
/// detector is not contiguous and every pixel stores its own corners.
fn separable(orientation: u8, gap: f64) -> Detector {
    let slow = (0..N1)
        .map(|i| {
            let g = if i >= N1 / 2 { gap } else { 0.0 };
            let lo = i as f64 * PIXEL + g;
            [[0., lo, 0.], [0., lo + PIXEL, 0.]]
        })
        .collect();
    let fast = (0..N2)
        .map(|j| {
            let lo = j as f64 * PIXEL;
            [[0., 0., lo], [0., 0., lo + PIXEL]]
        })
        .collect();
    let mut d = uniform(orientation);
    d.max_shape = Some([N1, N2]);
    d.map = PixelMap::Separable { slow, fast };
    d
}

fn geometry(detector: Detector, beam_on: bool, dims: (usize, usize)) -> Geometry {
    Geometry::new(dims.0, dims.1, poni(beam_on))
        .with_detector(Arc::new(detector))
        .unwrap()
}

fn pyfai(detector: Detector, beam_on: bool, dims: (usize, usize)) -> Integrator {
    Integrator::new(geometry(detector, beam_on, dims)).with_method(IntegrationMethod::PyfaiFull)
}

/// Per-bin agreement to 1e-6 relative (bins far below the largest one are
/// compared on its scale), in the bins `covered` selects (all if None).
#[track_caller]
fn assert_close(label: &str, test: &[f64], reference: &[f64], covered: Option<&[bool]>) {
    assert_eq!(test.len(), reference.len(), "{label}: length");
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{label}: empty reference");
    let mut checked = 0;
    for (i, (t, r)) in test.iter().zip(reference).enumerate() {
        if covered.is_some_and(|c| !c[i]) {
            continue;
        }
        checked += usize::from(*r != 0.0);
        let tol = 1e-6 * r.abs() + 1e-9 * scale;
        assert!((t - r).abs() <= tol, "{label}: bin {i}: {t} vs {r}");
    }
    assert!(checked > 0, "{label}: no filled bins compared");
}

/// Bins of the given edges that `reference` covers by more than a rounding
/// sliver, from its own shared-grid normalization sums.
fn coverage(
    reference: &Integrator,
    dims: (usize, usize),
    (n_radial, n_azimuthal): (usize, usize),
    radial: (f64, f64),
    azimuthal: (f64, f64),
) -> Vec<bool> {
    let grid = IntegrationGrid2d::new(n_radial, n_azimuthal, radial, azimuthal).unwrap();
    let ones = vec![1.0; dims.0 * dims.1];
    let norm = reference
        .integrate2d_raw_on_grid_from_slice(&ones, dims.0, dims.1, &grid, false)
        .unwrap()
        .sum_norm;
    let max = norm.iter().fold(0.0f64, |m, v| m.max(*v));
    norm.iter().map(|n| *n > 1e-6 * max).collect()
}

/// Edges spanned by bin centres (as returned by the range APIs).
fn edges(centres: &[f64]) -> (f64, f64) {
    let half = 0.5 * (centres[1] - centres[0]);
    (centres[0] - half, centres[centres.len() - 1] + half)
}

/// Ordinary 1D pattern and 2D cake: `test` on `image` must equal `reference`
/// on `reference_image`.
fn assert_paths_agree(
    label: &str,
    test: &Integrator,
    reference: &Integrator,
    (image, dims): (&[f64], (usize, usize)),
    (reference_image, reference_dims): (&[f64], (usize, usize)),
) {
    let one_d = |ai: &Integrator, data: &[f64], (d1, d2): (usize, usize)| {
        ai.integrate1d_from_slice(data, d1, d2, 20).1
    };
    assert_close(
        &format!("{label} 1D"),
        &one_d(test, image, dims),
        &one_d(reference, reference_image, reference_dims),
        None,
    );
    let cake = |ai: &Integrator, data: &[f64], (d1, d2): (usize, usize)| {
        ai.integrate2d_from_slice(data, d1, d2, 12, 18)
    };
    let radial = reference.radial_range().unwrap_or_else(|| {
        let [lo, hi] = crate::lut::pyfai_radial_range(reference.geometry(), RadialUnit::TthRad);
        (lo, hi)
    });
    assert_close(
        &format!("{label} 2D cake"),
        &cake(test, image, dims),
        &cake(reference, reference_image, reference_dims),
        Some(&coverage(
            reference,
            reference_dims,
            (12, 18),
            radial,
            (-PI, PI),
        )),
    );
}

/// Dense (pixel, bin) weights of a 1D table, pixels renumbered by `pixel`.
fn dense(lut: &Lut1d, pixel: impl Fn(usize) -> usize) -> Vec<f64> {
    let bins = lut.row_ptr.len() - 1;
    let values = lut.values.as_ref().unwrap();
    let mut weights = vec![0.0; bins * N1 * N2];
    for bin in 0..bins {
        for k in lut.row_ptr[bin]..lut.row_ptr[bin + 1] {
            weights[pixel(lut.col_idx[k]) * bins + bin] += values[k];
        }
    }
    weights
}

/// The 1D table the Integrator builds (`build_lut1d_pyfai_csr` without a
/// sector, automatic or explicit radial range), and the sector variant only
/// reachable by calling it directly: every pixel's weights must match those of
/// the same physical pixel of the unmirrored detector. Comparing weights
/// rather than patterns isolates the pixels on the seam; summed into a
/// pattern their error is diluted to ~1e-5.
#[test]
fn pyfai_full_1d_weights_of_mirrored_orientations_match_the_flipped_detector() {
    for beam_on in [true, false] {
        for orientation in [1, 2, 4] {
            let (rows, cols) = flips(orientation);
            for sector in [
                None,
                Some([2.5, 2.5 + PI / 2.0]),
                Some([-PI, -PI / 2.0]),
                Some([0.0, 2.0 * PI]),
            ] {
                for radial in [None, Some([0.002, 0.03])] {
                    let build = |o| {
                        crate::lut::build_lut1d_pyfai_csr(
                            &geometry(uniform(o), beam_on, (N1, N2)),
                            RadialUnit::TthRad,
                            30,
                            sector,
                            radial,
                        )
                        .unwrap()
                    };
                    let test = dense(&build(orientation), |p| flip_index(p, rows, cols));
                    let reference = dense(&build(3), |p| p);
                    for (i, (t, r)) in test.iter().zip(&reference).enumerate() {
                        assert!(
                            (t - r).abs() < 1e-5,
                            "beam on {beam_on}, orientation {orientation}, sector {sector:?}, \
                             radial {radial:?}: pixel {} bin {}: {t} vs {r}",
                            i / 30,
                            i % 30
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pyfai_full_mirrored_orientations_match_the_flipped_image() {
    let img = image((N1, N2));
    for beam_on in [true, false] {
        for orientation in [1, 2, 4] {
            let (rows, cols) = flips(orientation);
            let reference_image = flipped(&img, rows, cols);
            let label = format!("orientation {orientation}, beam on detector: {beam_on}");
            // An explicit radial range cutting through the image.
            let cut = if beam_on { (0.004, 0.03) } else { (0.06, 0.12) };
            for radial_range in [None, Some(cut)] {
                let make = |o| {
                    let ai = pyfai(uniform(o), beam_on, (N1, N2));
                    match radial_range {
                        Some((lo, hi)) => ai.with_radial_range(lo, hi),
                        None => ai,
                    }
                };
                assert_paths_agree(
                    &format!("{label}, radial range {radial_range:?}"),
                    &make(orientation),
                    &make(3),
                    (&img, (N1, N2)),
                    (&reference_image, (N1, N2)),
                );
            }
        }
    }
}

/// Shared-grid 2D tables, true azimuthal ranges, brackets and the legacy
/// centre/step range all go through the same pyFAI 2D kernel.
#[test]
fn pyfai_full_range_and_shared_grid_paths_match_the_flipped_image() {
    let img = image((N1, N2));
    for orientation in [1, 2, 4] {
        let (rows, cols) = flips(orientation);
        let reference_image = flipped(&img, rows, cols);
        let test = pyfai(uniform(orientation), true, (N1, N2));
        let reference = pyfai(uniform(3), true, (N1, N2));
        let label = format!("orientation {orientation}");

        for azimuth in [(-PI, PI), (2.0, 2.0 + PI)] {
            let grid = IntegrationGrid2d::new(16, 18, (0.0, 0.04), azimuth).unwrap();
            let raw = |ai: &Integrator, data: &[f64]| {
                ai.integrate2d_raw_on_grid_from_slice(data, N1, N2, &grid, false)
                    .unwrap()
            };
            let (t, r) = (raw(&test, &img), raw(&reference, &reference_image));
            // Raw sums: a rounding sliver is negligible on the largest bin's scale.
            let scaled = |label: &str, t: &[f64], r: &[f64]| {
                let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for (i, (t, r)) in t.iter().zip(r).enumerate() {
                    assert!(
                        (t - r).abs() <= 1e-6 * scale,
                        "{label}: bin {i}: {t} vs {r}"
                    );
                }
            };
            scaled(
                &format!("{label} shared grid {azimuth:?} signal"),
                &t.sum_signal,
                &r.sum_signal,
            );
            scaled(
                &format!("{label} shared grid {azimuth:?} norm"),
                &t.sum_norm,
                &r.sum_norm,
            );
        }

        let ranges = [(-PI, PI), (2.5, 2.5 + PI / 2.0), (-0.3, 1.2)];
        for range in ranges {
            let run = |ai: &Integrator, data: &[f64]| {
                ai.integrate2d_on_azimuthal_range_from_slice(data, N1, N2, 16, 12, range, true)
                    .unwrap()
            };
            let (t, r) = (run(&test, &img), run(&reference, &reference_image));
            let covered = coverage(&reference, (N1, N2), (16, 12), edges(&r.radial), range);
            let label = format!("{label} azimuthal range {range:?}");
            assert_close(&label, &t.intensity, &r.intensity, Some(&covered));
            assert_close(
                &label,
                t.errors.as_ref().unwrap(),
                r.errors.as_ref().unwrap(),
                Some(&covered),
            );
        }

        let (center_range, half) = ((0.0, 2.0 * PI), PI / 18.0);
        let brackets = |ai: &Integrator, data: &[f64]| {
            ai.integrate2d_on_azimuthal_brackets_from_slice(
                data,
                N1,
                N2,
                16,
                center_range,
                half,
                false,
            )
            .unwrap()
        };
        let (t, r) = (
            brackets(&test, &img),
            brackets(&reference, &reference_image),
        );
        let (n_azimuthal, azimuth) = azimuthal_bracket_bins(center_range, half).unwrap();
        let covered = coverage(
            &reference,
            (N1, N2),
            (16, n_azimuthal),
            edges(&r.radial),
            azimuth,
        );
        assert_close(
            &format!("{label} brackets"),
            &t.intensity,
            &r.intensity,
            Some(&covered),
        );
    }
}

#[test]
fn pyfai_full_multi_integrator_with_mirrored_detectors_matches_the_flipped_images() {
    let img = image((N1, N2));
    let other: Vec<f64> = img.iter().map(|v| 2.0 * v - 50.0).collect();
    let ai = |o| pyfai(uniform(o), true, (N1, N2)).with_radial_range(0.0, 0.03);
    let test = MultiIntegrator::try_new(vec![ai(2), ai(4)]).unwrap();
    let reference = MultiIntegrator::try_new(vec![ai(3), ai(3)]).unwrap();
    let rows = flipped(&img, true, false);
    let cols = flipped(&other, false, true);
    let test_slices: [(&[f64], usize, usize); 2] = [(&img, N1, N2), (&other, N1, N2)];
    let reference_slices: [(&[f64], usize, usize); 2] = [(&rows, N1, N2), (&cols, N1, N2)];
    assert_close(
        "multi 1D",
        &test.integrate1d_from_slices(&test_slices, 30).1,
        &reference.integrate1d_from_slices(&reference_slices, 30).1,
        None,
    );
    for azimuthal_range in [(-PI, PI), (2.5, 2.5 + PI / 2.0)] {
        let options = Multi2dOptions {
            azimuthal_range,
            ..Default::default()
        };
        let run = |m: &MultiIntegrator, slices: &[(&[f64], usize, usize)]| {
            m.integrate2d_from_slices(slices, 16, 12, &options).unwrap()
        };
        let (t, r) = (run(&test, &test_slices), run(&reference, &reference_slices));
        let max = r.raw.sum_norm.iter().fold(0.0f64, |m, v| m.max(*v));
        let covered: Vec<bool> = r.raw.sum_norm.iter().map(|n| *n > 1e-6 * max).collect();
        assert_close(
            &format!("multi 2D {azimuthal_range:?}"),
            &t.intensity,
            &r.intensity,
            Some(&covered),
        );
    }
}

/// Non-uniform maps: separable detectors (contiguous, and with a gap so each
/// pixel stores its own reordered corners) under orientations 1, 2 and 4, and
/// transformed maps. A transformed map keeps its base detector's corner order
/// and reports orientation 3 even when its base is mirrored, so the winding
/// cannot be read from `orientation`.
#[test]
fn pyfai_full_non_uniform_and_transformed_maps_match_the_unmirrored_detector() {
    let img = image((N1, N2));
    for gap in [0.0, 0.3 * PIXEL] {
        for orientation in [1, 2, 4] {
            let (rows, cols) = flips(orientation);
            assert_paths_agree(
                &format!("separable gap {gap}, orientation {orientation}"),
                &pyfai(separable(orientation, gap), true, (N1, N2)),
                &pyfai(separable(3, gap), true, (N1, N2)),
                (&img, (N1, N2)),
                (&flipped(&img, rows, cols), (N1, N2)),
            );
        }
        // Transformed detector on the transformed image == base on the image.
        for op in [
            DetectorTransform::Horizontal,
            DetectorTransform::Vertical,
            DetectorTransform::Half,
            DetectorTransform::Left,
            DetectorTransform::Transpose,
        ] {
            let base = separable(3, gap);
            let transformed = base.transformed(op).unwrap();
            let dims = match op {
                DetectorTransform::Left | DetectorTransform::Transpose => (N2, N1),
                _ => (N1, N2),
            };
            let transformed_image: Vec<f64> = (0..dims.0 * dims.1)
                .map(|i| {
                    let (r, c) = (i / dims.1, i % dims.1);
                    let (br, bc) = match op {
                        DetectorTransform::Horizontal => (r, N2 - 1 - c),
                        DetectorTransform::Vertical => (N1 - 1 - r, c),
                        DetectorTransform::Half => (N1 - 1 - r, N2 - 1 - c),
                        DetectorTransform::Left => (c, dims.0 - 1 - r),
                        _ => (c, r),
                    };
                    img[br * N2 + bc]
                })
                .collect();
            assert_paths_agree(
                &format!("separable gap {gap}, transformed {op:?}"),
                &pyfai(transformed, true, dims),
                &pyfai(base, true, (N1, N2)),
                (&transformed_image, dims),
                (&img, (N1, N2)),
            );
        }
        // A mirrored base under a transformation: orientation says 3, the
        // corners are still mirrored.
        let transformed = separable(2, gap)
            .transformed(DetectorTransform::Horizontal)
            .unwrap();
        assert_eq!(transformed.orientation, 3);
        assert_paths_agree(
            &format!("separable gap {gap}, orientation 2 transformed Horizontal"),
            &pyfai(transformed, true, (N1, N2)),
            &pyfai(separable(3, gap), true, (N1, N2)),
            (&img, (N1, N2)),
            (&flipped(&img, true, true), (N1, N2)),
        );
    }
}
