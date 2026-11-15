//! An attached detector with a fixed shape (`Detector::max_shape`) accepts
//! only that shape or an integer binning of it, on every integration entry
//! point and not only in `Geometry::with_detector`. Another image shape must
//! fail with a message naming both shapes instead of integrating with
//! truncated binning, overflowing in `Detector::corners` or returning no
//! finite bins. A detector without a fixed shape keeps accepting any image.
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, mpsc};

use super::{IntegrationGrid2d, IntegrationMethod, Integrator, MaskMode, SparseLayout};
use crate::batch::{
    Integration1D, Integration2D, WorkItem, process_files, process_files_sequential,
    process_files_streaming, process_queue_streaming,
};
use crate::detector::{Detector, DetectorTransform, PixelMap, spline::Spline};
use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::multi::{Multi2dOptions, MultiIntegrator};
use crate::poni::Poni;

type Pattern = (Vec<f64>, Vec<f64>);

const FULL: [usize; 2] = [12, 36];
const PIXEL: f64 = 1e-4;

/// Slightly non-uniform pitch with a gap before fine row 5 and column 17.
fn separable() -> Detector {
    let edges = |n: usize, gap: usize, axis: usize| {
        let mut lo = 0.;
        (0..n)
            .map(|i| {
                if i == gap {
                    lo += 0.3 * PIXEL;
                }
                let hi = lo + PIXEL * (1. + 0.04 * (i as f64 * 0.9).sin());
                let (mut a, mut b) = ([0.; 3], [0.; 3]);
                a[axis] = lo;
                b[axis] = hi;
                lo = hi;
                [a, b]
            })
            .collect()
    };
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.max_shape = Some(FULL);
    d.map = PixelMap::Separable {
        slow: edges(FULL[0], 5, 1),
        fast: edges(FULL[1], 17, 2),
    };
    d
}

fn corner_map() -> Detector {
    let s = separable();
    let values = (0..FULL[0])
        .flat_map(|r| (0..FULL[1]).map(move |c| (r, c)))
        .map(|(r, c)| s.corners(FULL, r, c))
        .collect();
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.max_shape = Some(FULL);
    d.map = PixelMap::Corners {
        shape: FULL,
        values: Arc::new(values),
    };
    d
}

fn spline() -> Detector {
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.max_shape = Some(FULL);
    d.spline = Some(Arc::new(
        Spline::parse(include_str!("../../test_data/detector_reference.spline")).unwrap(),
    ));
    d
}

/// Uniform with a fixed shape: on its own shape the geometry does not even
/// consult the detector, but the shape contract still holds.
fn uniform_fixed() -> Detector {
    let mut d = Detector::custom(PIXEL, PIXEL);
    d.max_shape = Some(FULL);
    d
}

/// Detectors with a fixed shape, with that shape.
fn fixed_detectors() -> Vec<(&'static str, Detector)> {
    vec![
        ("separable", separable()),
        ("corners", corner_map()),
        ("spline", spline()),
        (
            "transposed separable",
            separable()
                .transformed(DetectorTransform::Transpose)
                .unwrap(),
        ),
        ("uniform with max_shape", uniform_fixed()),
    ]
}

/// One more pixel, twice the size, and not a divisor (7 divides neither 12
/// nor 36).
fn invalid_shapes(full: [usize; 2]) -> [[usize; 2]; 3] {
    [
        [full[0] + 1, full[1] + 1],
        [2 * full[0], 2 * full[1]],
        [7, full[1]],
    ]
}

fn poni() -> Poni {
    let mut p = Poni::new();
    p.pixel1 = PIXEL;
    p.pixel2 = PIXEL;
    p.distance = 0.02;
    // Beam beside the detector: no singular pixel to sample.
    p.poni1 = 5.3 * PIXEL;
    p.poni2 = -4.0 * PIXEL;
    p.rot1 = 0.01;
    p.wavelength = 1e-10;
    p
}

fn integrator(detector: &Detector, method: IntegrationMethod) -> Integrator {
    let full = detector.max_shape.unwrap_or(FULL);
    let geometry = Geometry::new(full[0], full[1], poni())
        .with_detector(Arc::new(detector.clone()))
        .unwrap();
    Integrator::new(geometry).with_method(method)
}

fn frame([dim1, dim2]: [usize; 2]) -> Frame {
    Frame {
        name: format!("{dim1}x{dim2}"),
        data: (0..dim1 * dim2)
            .map(|p| ((p * 37 + dim2) % 101) as f64 + 1.0)
            .collect(),
        dim1,
        dim2,
    }
}

fn panic_message(f: impl FnOnce()) -> Option<String> {
    let payload = catch_unwind(AssertUnwindSafe(f)).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    )
}

/// The message names the image shape and the detector's shape, and, for an
/// image that is no integer binning (a crop, or a larger image), the remedy.
fn assert_shape_message(message: &str, shape: [usize; 2], full: [usize; 2], what: &str) {
    let image = format!("[{}, {}]", shape[0], shape[1]);
    let detector = format!("[{}, {}]", full[0], full[1]);
    let not_a_binning = !shape.contains(&0)
        && !(full[0].is_multiple_of(shape[0]) && full[1].is_multiple_of(shape[1]));
    assert!(
        message.contains("does not fit detector")
            && message.contains(&image)
            && message.contains(&detector)
            && (!not_a_binning
                || message.contains(
                    "for a cropped image, attach a custom detector without a fixed shape, or none"
                )),
        "{what}: unexpected message {message:?}"
    );
}

fn assert_panics(shape: [usize; 2], full: [usize; 2], what: &str, f: impl FnOnce()) {
    match panic_message(f) {
        Some(message) => assert_shape_message(&message, shape, full, what),
        None => panic!("{what}: image {shape:?} on a {full:?} detector was integrated"),
    }
}

fn assert_err<T>(
    shape: [usize; 2],
    full: [usize; 2],
    what: &str,
    f: impl FnOnce() -> Result<T, String>,
) {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Err(message)) => assert_shape_message(&message, shape, full, what),
        Ok(Ok(_)) => panic!("{what}: image {shape:?} on a {full:?} detector was integrated"),
        Err(_) => panic!("{what}: image {shape:?} panicked instead of returning an error"),
    }
}

/// Every panicking `Integrator` entry point that takes an image of arbitrary
/// shape: `Frame`, slice, raw, variance, error and azimuthal-range variants.
fn panicking_entry_points(ai: &Integrator, f: &Frame, what: &str, full: [usize; 2]) {
    let shape = [f.dim1, f.dim2];
    let (d, d1, d2) = (f.data.as_slice(), f.dim1, f.dim2);
    let check = |name: &str, g: &dyn Fn()| assert_panics(shape, full, &format!("{what} {name}"), g);
    check("integrate1d", &|| drop(ai.integrate1d(f, 8)));
    check("integrate1d_with_errors", &|| {
        drop(ai.integrate1d_with_errors(f, 8))
    });
    check("integrate1d_from_slice", &|| {
        drop(ai.integrate1d_from_slice(d, d1, d2, 8))
    });
    check("integrate1d_with_errors_from_slice", &|| {
        drop(ai.integrate1d_with_errors_from_slice(d, d1, d2, 8))
    });
    check("integrate1d_with_azimuthal_std", &|| {
        drop(ai.integrate1d_with_azimuthal_std(f, 8, true))
    });
    check("integrate1d_raw", &|| drop(ai.integrate1d_raw(f, 8)));
    check("integrate1d_raw_with_variance", &|| {
        drop(ai.integrate1d_raw_with_variance(f, 8))
    });
    check("integrate1d_raw_from_slice", &|| {
        drop(ai.integrate1d_raw_from_slice(d, d1, d2, 8))
    });
    check("integrate1d_raw_with_variance_from_slice", &|| {
        drop(ai.integrate1d_raw_with_variance_from_slice(d, d1, d2, 8))
    });
    check("integrate2d", &|| drop(ai.integrate2d(f, 8, 4)));
    check("integrate2d_from_slice", &|| {
        drop(ai.integrate2d_from_slice(d, d1, d2, 8, 4))
    });
    check("integrate2d_with_errors", &|| {
        drop(ai.integrate2d_with_errors(f, 8, 4))
    });
    check("integrate2d_with_azimuthal_range", &|| {
        drop(ai.integrate2d_with_azimuthal_range(f, 8, 4, -3.0, 0.5))
    });
}

/// Every `Result` entry point: the shape error is returned, not panicked.
fn result_entry_points(ai: &Integrator, f: &Frame, what: &str, full: [usize; 2]) {
    let shape = [f.dim1, f.dim2];
    let (d, d1, d2) = (f.data.as_slice(), f.dim1, f.dim2);
    let grid = IntegrationGrid2d::new(8, 4, (0.0, 0.2), (-3.1, 3.1)).unwrap();
    let check = |name: &str, g: &dyn Fn() -> Result<(), String>| {
        assert_err(shape, full, &format!("{what} {name}"), g)
    };
    check("integrate2d_on_azimuthal_range", &|| {
        ai.integrate2d_on_azimuthal_range(f, 8, 4, (-3.0, 1.5), false)
            .map(drop)
    });
    check("integrate2d_on_azimuthal_brackets_from_slice", &|| {
        ai.integrate2d_on_azimuthal_brackets_from_slice(d, d1, d2, 8, (0.0, 6.0), 0.5, false)
            .map(drop)
    });
    check("integrate2d_raw_on_grid_from_slice", &|| {
        ai.integrate2d_raw_on_grid_from_slice(d, d1, d2, &grid, true)
            .map(drop)
    });
    check("prepare_dynamic1d", &|| {
        ai.prepare_dynamic1d((d1, d2), 8).map(drop)
    });
    check("prepare_fixed2d_on_grid", &|| {
        ai.prepare_fixed2d_on_grid((d1, d2), &grid).map(drop)
    });
}

#[test]
fn invalid_shapes_fail_on_every_entry_point() {
    let methods = [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::PyfaiFull,
        IntegrationMethod::Supersampled { n_ss: 2 },
    ];
    for (name, detector) in fixed_detectors() {
        let full = detector.max_shape.unwrap();
        for shape in invalid_shapes(full) {
            let f = frame(shape);
            for method in methods {
                let what = format!("{name} {method:?} {shape:?}");
                let ai = integrator(&detector, method);
                panicking_entry_points(&ai, &f, &what, full);
                result_entry_points(&ai, &f, &what, full);
                // The plan builders return the error before any table work.
                let fixed = integrator(&detector, method)
                    .with_mask_mode(MaskMode::Fixed)
                    .with_fast_math(true)
                    .with_sparse_layout(SparseLayout::Csc);
                assert_err(shape, full, &format!("{what} prepare_fixed1d"), || {
                    fixed.prepare_fixed1d((shape[0], shape[1]), 8).map(drop)
                });
            }
            let ai = integrator(&detector, IntegrationMethod::Histogram);
            assert_panics(shape, full, &format!("{name} Histogram {shape:?}"), || {
                drop(ai.integrate1d(&f, 8))
            });
        }
    }
}

/// An integer binning without a manufacturer pitch is rejected as in
/// `Geometry::with_detector`; an empty image is rejected too.
#[test]
fn binnings_without_a_pitch_and_empty_images_are_rejected() {
    let mut detector = uniform_fixed();
    detector.binning_pitches = vec![(1, PIXEL), (2, 2.0 * PIXEL)];
    let ai = integrator(&detector, IntegrationMethod::Counting);
    let (_, y) = ai.integrate1d(&frame([6, 18]), 8);
    assert!(y.iter().any(|v| v.is_finite()));
    for shape in [[4, 12], [0, 36]] {
        let message = panic_message(|| drop(ai.integrate1d(&frame(shape), 8))).unwrap();
        assert_shape_message(&message, shape, FULL, &format!("{shape:?}"));
    }
}

/// A hand-built transformed map must keep its base's fixed shape: without it,
/// `with_detector` and every later shape check would accept any image. Every
/// map `Detector::transformed` produces still validates.
#[test]
fn transformed_maps_keep_their_base_shape() {
    use DetectorTransform::*;
    let ops = [
        Identity, Half, Transpose, Transverse, Left, Right, Horizontal, Vertical,
    ];
    let mut free_spline = spline();
    free_spline.max_shape = None;
    for base in [
        separable(),
        corner_map(),
        spline(),
        uniform_fixed(),
        free_spline,
    ] {
        for first in ops {
            let once = base.transformed(first).unwrap();
            for second in ops {
                let twice = once.transformed(second).unwrap();
                assert_eq!(twice.validate(), Ok(()), "{first:?} then {second:?}");
            }
        }
    }
    let base = Arc::new(separable());
    for (operation, max_shape) in [
        (Transpose, None),
        (Transpose, Some(FULL)),
        (Half, Some([FULL[1], FULL[0]])),
    ] {
        let mut d = separable();
        d.map = PixelMap::Transformed {
            base: Arc::clone(&base),
            operation,
        };
        d.max_shape = max_shape;
        let attach = Geometry::new(7, 5, poni()).with_detector(Arc::new(d.clone()));
        assert!(
            d.validate().is_err() && attach.is_err(),
            "{operation:?} {max_shape:?}"
        );
    }
}

#[test]
fn multi_integrator_rejects_an_invalid_shape() {
    let detector = separable();
    let full = FULL;
    let shape = [7, 36];
    let make = || integrator(&detector, IntegrationMethod::PixelSplit).with_radial_range(0.0, 0.2);
    let multi = MultiIntegrator::try_new(vec![make(), make()]).unwrap();
    let (good, bad) = (frame(FULL), frame(shape));
    assert_panics(shape, full, "MultiIntegrator::integrate1d", || {
        drop(multi.integrate1d(&[&good, &bad], 8))
    });
    let options = Multi2dOptions::default();
    assert_err(shape, full, "MultiIntegrator::integrate2d", || {
        multi.integrate2d(&[&good, &bad], 8, 4, &options).map(drop)
    });
}

/// Shapes the detector defines keep integrating: its own shape and integer
/// binnings, on detectors with and without a fixed shape.
#[test]
fn valid_shapes_still_integrate() {
    for (name, detector) in fixed_detectors() {
        let full = detector.max_shape.unwrap();
        for shape in [full, [full[0] / 2, full[1] / 2], [full[0] / 3, full[1]]] {
            let f = frame(shape);
            let ai = integrator(&detector, IntegrationMethod::PixelSplit);
            let (_, y) = ai.integrate1d(&f, 8);
            assert!(
                y.iter().any(|v| v.is_finite() && *v > 0.0),
                "{name} {shape:?}: no finite bins"
            );
            ai.integrate2d_on_azimuthal_range(&f, 8, 4, (-3.0, 1.5), false)
                .unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
            ai.prepare_dynamic1d((shape[0], shape[1]), 8)
                .unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
        }
    }
    // Without a fixed shape any image is accepted, as before.
    let ai = integrator(&Detector::custom(PIXEL, PIXEL), IntegrationMethod::Counting);
    for shape in invalid_shapes(FULL) {
        let (_, y) = ai.integrate1d(&frame(shape), 8);
        assert!(y.iter().any(|v| v.is_finite()), "free {shape:?}");
    }
}

/// `Geometry::set_shape` after `with_detector`: a shape the detector cannot
/// represent returns the message the `Integrator` gives and leaves the
/// geometry unchanged, instead of producing arrays for a truncated binning (a
/// crop), NaN arrays (a larger uniform image) or a panic on an index overflow
/// or a division by zero at first use. `with_detector` rejects the same shapes
/// with the same message.
#[test]
fn set_shape_rejects_shapes_the_detector_cannot_represent() {
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    for (name, detector) in fixed_detectors() {
        let full = detector.max_shape.unwrap();
        let binned = [full[0] / 2, full[1] / 2];
        let attach = |s: [usize; 2]| {
            Geometry::new(s[0], s[1], poni()).with_detector(Arc::new(detector.clone()))
        };
        for start in [full, binned] {
            let mut shapes = invalid_shapes(full).to_vec();
            shapes.extend([[0, full[1]], [0, 0]]);
            for shape in shapes {
                let what = format!("{name} {start:?} -> {shape:?}");
                let mut g = attach(start).unwrap();
                let before = bits(g.calc_tth_all());
                assert_err(shape, full, &what, || g.set_shape(shape[0], shape[1]));
                assert_eq!(g.shape(), (start[0], start[1]), "{what}");
                assert!(bits(g.calc_tth_all()) == before, "{what}");
                assert_err(shape, full, &format!("{what} with_detector"), || {
                    attach(shape)
                });
            }
        }
    }
}

/// Shapes the detector does represent (another integer binning, the full
/// shape from a binned start), set through `set_shape`, give the arrays
/// `with_detector` gives for that shape, bit for bit; without a fixed shape
/// every shape stays valid.
#[test]
fn valid_set_shape_shapes_match_with_detector() {
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let arrays = |g: &Geometry| {
        let (corner_r, corner_a) = g.angular_corners();
        let position = (0..g.size())
            .flat_map(|i| g.detector_position((i / g.dim2()) as f64 + 0.5, (i % g.dim2()) as f64))
            .collect();
        [
            g.calc_tth_all(),
            g.calc_azimuth_all(),
            g.calc_solid_angle(),
            corner_r,
            corner_a,
            position,
        ]
        .map(bits)
    };
    let mut cases = fixed_detectors();
    cases.push(("custom without max_shape", Detector::custom(PIXEL, PIXEL)));
    for (name, detector) in cases {
        let full = detector.max_shape.unwrap_or(FULL);
        let binned = [full[0] / 2, full[1] / 2];
        let mut pairs = vec![
            (full, binned),
            (binned, full),
            (full, [full[0] / 3, full[1]]),
        ];
        if detector.max_shape.is_none() {
            pairs.extend([(full, [7, full[1]]), (full, [full[0] + 1, full[1] + 1])]);
        }
        for (start, shape) in pairs {
            let attach = |s: [usize; 2]| {
                Geometry::new(s[0], s[1], poni())
                    .with_detector(Arc::new(detector.clone()))
                    .unwrap()
            };
            let mut g = attach(start);
            g.set_shape(shape[0], shape[1]).unwrap();
            assert!(
                arrays(&g) == arrays(&attach(shape)),
                "{name} {start:?} -> {shape:?}"
            );
        }
    }
}

/// A batch with one invalid shape among valid ones reports that item and
/// integrates the others like a direct call, whatever the worker layout.
#[test]
fn mixed_shape_batch_reports_the_invalid_item() {
    let detector = separable();
    let shapes = [FULL, [7, 36], [6, 18], FULL, [13, 37], [4, 12]];
    let invalid = |s: [usize; 2]| s == [7, 36] || s == [13, 37];
    let reference = integrator(&detector, IntegrationMethod::Counting);
    let task = Integration2D {
        n_radial: 8,
        n_azimuthal: 4,
    };
    let (tx, rx) = mpsc::channel();
    for (index, &shape) in shapes.iter().enumerate() {
        let f = frame(shape);
        tx.send(WorkItem::Frame {
            index,
            name: f.name.clone(),
            frame: f,
        })
        .unwrap();
    }
    drop(tx);
    let ai = Arc::new(integrator(&detector, IntegrationMethod::Counting));
    let results: Vec<_> = process_queue_streaming(rx, ai, task, 2, 1)
        .into_iter()
        .collect();
    assert_eq!(results.len(), shapes.len(), "a worker died");
    for (index, name, result) in results {
        let shape = shapes[index];
        if invalid(shape) {
            assert_shape_message(&result.unwrap_err(), shape, FULL, &name);
        } else {
            assert_eq!(
                result.unwrap(),
                reference.integrate2d(&frame(shape), 8, 4),
                "{name}"
            );
        }
    }

    // File batches: the same through every helper that reads TIFFs.
    let dir = std::env::temp_dir().join(format!("dioptrin-detector-shape-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let files: Vec<String> = shapes
        .iter()
        .enumerate()
        .map(|(i, &[h, w])| {
            use tiff::encoder::{TiffEncoder, colortype};
            let path = dir.join(format!("{i}.tif"));
            let pixels: Vec<u16> = (0..h * w).map(|p| (p % 97 + 1) as u16).collect();
            TiffEncoder::new(std::fs::File::create(&path).unwrap())
                .unwrap()
                .write_image::<colortype::Gray16>(w as u32, h as u32, &pixels)
                .unwrap();
            path.to_str().unwrap().to_owned()
        })
        .collect();
    let task = Integration1D { n_bins: 8 };
    let check = |what: &str, results: Vec<(String, Result<Pattern, String>)>| {
        assert_eq!(results.len(), files.len(), "{what}: a worker died");
        for (name, result) in results {
            let index = files.iter().position(|f| *f == name).unwrap();
            let shape = shapes[index];
            if invalid(shape) {
                let message = result.expect_err(&format!("{what} {name}"));
                assert_shape_message(&message, shape, FULL, what);
            } else {
                let expected = reference.integrate1d(&Frame::from_tiff(name.clone()).unwrap(), 8);
                assert_eq!(result.unwrap(), expected, "{what} {name}");
            }
        }
    };
    let ai = || Arc::new(integrator(&detector, IntegrationMethod::Counting));
    check("process_files", process_files(&files, ai(), task, 2, 1));
    check(
        "process_files_sequential",
        process_files_sequential(&files, &ai(), task),
    );
    check(
        "process_files_streaming",
        process_files_streaming(&files, ai(), task, 2, 1)
            .into_iter()
            .map(|(_, name, result)| (name, result))
            .collect(),
    );
    // Queues of the same files, as paths or as frames loaded beforehand.
    let queue = |as_frames: bool| {
        let (tx, rx) = mpsc::channel();
        for (index, path) in files.iter().enumerate() {
            let item = if as_frames {
                let frame = Frame::from_tiff(path.clone()).unwrap();
                let name = path.clone();
                WorkItem::Frame { index, name, frame }
            } else {
                let path = path.clone();
                WorkItem::Path { index, path }
            };
            tx.send(item).unwrap();
        }
        rx
    };
    let named = |results: Vec<(usize, String, Result<Pattern, String>)>| {
        results.into_iter().map(|(_, n, r)| (n, r)).collect()
    };
    check(
        "process_queue_streaming paths",
        named(
            process_queue_streaming(queue(false), ai(), task, 2, 1)
                .into_iter()
                .collect(),
        ),
    );
    #[cfg(feature = "python")]
    {
        use crate::batch::{
            FileWorkers, StreamControl, process_files_streaming_with_pool, process_files_with_pool,
            process_queue_with_pool,
        };
        let pool = || FileWorkers::new(2, 1).unwrap();
        check(
            "process_files_with_pool",
            process_files_with_pool(&files, ai(), task, pool()),
        );
        check(
            "process_files_streaming_with_pool",
            named(
                process_files_streaming_with_pool(&files, ai(), task, pool())
                    .into_iter()
                    .collect(),
            ),
        );
        for (as_frames, what) in [(false, "paths"), (true, "frames")] {
            let control = StreamControl::new();
            let batch = process_queue_with_pool(queue(as_frames), ai(), task, pool(), control);
            check(
                &format!("process_queue_with_pool {what}"),
                named(batch.into_iter().collect()),
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
