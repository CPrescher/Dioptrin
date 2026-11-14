use dioptrin::{
    detector::{self, spline::Spline},
    fit2d::Fit2d,
    geometry::Geometry,
    poni::Poni,
};
use serde_json::Value;
use std::sync::Arc;
fn reference() -> Value {
    serde_json::from_str(include_str!("../test_data/detector_reference.json")).unwrap()
}
#[test]
fn catalogue_coordinates_and_builtin_masks_match_pyfai() {
    for row in reference()["detectors"].as_array().unwrap() {
        let d = detector::named(row["name"].as_str().unwrap()).unwrap();
        d.validate().unwrap();
        let shape = serde_json::from_value(row["shape"].clone()).unwrap();
        for i in 0..row["rows"].as_array().unwrap().len() {
            let p = d.position(
                shape,
                row["rows"][i].as_f64().unwrap(),
                row["cols"][i].as_f64().unwrap(),
            );
            for (k, v) in p.iter().enumerate() {
                assert!(
                    (v - row["positions"][i][k].as_f64().unwrap()).abs() < 1e-7,
                    "{} sample {i} axis {k}: {p:?} != {}",
                    d.name,
                    row["positions"][i]
                );
            }
        }
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(d.builtin_mask(shape).unwrap())),
            row["mask_sha256"].as_str().unwrap(),
            "{} mask pixels",
            d.name
        );
        assert_eq!(
            d.builtin_mask(shape)
                .unwrap()
                .iter()
                .filter(|v| **v != 0)
                .count() as u64,
            row["masked"].as_u64().unwrap(),
            "{}",
            d.name
        );
    }
}
#[test]
fn fit2d_cubic_spline_matches_fitpack_including_clamped_edges() {
    let s = Spline::parse(include_str!("../test_data/detector_reference.spline")).unwrap();
    for row in reference()["spline"].as_array().unwrap() {
        let (x, y) = s.displacement(row[0].as_f64().unwrap(), row[1].as_f64().unwrap());
        assert!((x - row[2].as_f64().unwrap()).abs() < 1e-5, "{row}: {x}");
        assert!((y - row[3].as_f64().unwrap()).abs() < 1e-5, "{row}: {y}");
    }
    assert!(Spline::parse("broken spline").is_err());
}
#[test]
fn fit2d_conversion_matches_pyfai_and_reproduces_direct_distance_and_beam_center() {
    for row in reference()["fit2d"].as_array().unwrap() {
        let mut p = Poni::new();
        p.distance = 0.234;
        p.poni1 = 0.043;
        p.poni2 = 0.055;
        p.pixel1 = 79e-6;
        p.pixel2 = 123e-6;
        p.wavelength = 0.4e-10;
        p.rot1 = row["rotations"][0].as_f64().unwrap();
        p.rot2 = row["rotations"][1].as_f64().unwrap();
        p.rot3 = row["rotations"][2].as_f64().unwrap();
        let f = Fit2d::from_poni(&p);
        let v = &row["values"];
        for (key, a) in [
            ("directDist", f.direct_dist),
            ("centerX", f.center_x),
            ("centerY", f.center_y),
            ("tilt", f.tilt),
            ("tiltPlanRotation", f.tilt_plane_rotation),
            ("pixelX", f.pixel_x),
            ("pixelY", f.pixel_y),
        ] {
            assert!(
                (a - v[key].as_f64().unwrap()).abs() < 1e-9,
                "{key}: {a} != {}",
                v[key]
            );
        }
        let q = f.to_poni(p.wavelength).unwrap();
        let round = Fit2d::from_poni(&q);
        assert!((f.center_x - round.center_x).abs() < 1e-8);
        assert!((f.center_y - round.center_y).abs() < 1e-8);
        assert!((p.distance - q.distance).abs() < 1e-12);
    }
}
#[test]
fn orientations_and_transforms_keep_physical_positions_and_mask_pixels() {
    let mut d = detector::Detector::custom(0.001, 0.002);
    d.max_shape = Some([3, 5]);
    d.mask = vec![[1, 2]];
    for orientation in 0..=4 {
        d.orientation = orientation;
        let p = d.position([3, 5], 0.5, 1.5);
        let r = if matches!(orientation, 1 | 2) {
            2.5
        } else {
            0.5
        };
        let c = if matches!(orientation, 1 | 4) {
            3.5
        } else {
            1.5
        };
        assert_eq!(p, [0., r * 0.001, c * 0.002]);
    }
    d.orientation = 3;
    d.spline = Some(Arc::new(
        Spline::parse(include_str!("../test_data/detector_reference.spline")).unwrap(),
    ));
    let g = Geometry::new(3, 5, Poni::new())
        .with_detector(Arc::new(d.clone()))
        .unwrap();
    assert_eq!(g.angular_corners().0.len(), 24);
    let t = d.transformed(detector::DetectorTransform::Left).unwrap();
    assert_eq!(t.max_shape, Some([5, 3]));
    assert_eq!(t.position([5, 3], 3.5, 0.5), d.position([3, 5], 0.5, 1.5));
    assert_eq!(t.builtin_mask([5, 3]).unwrap()[9], 1);
    let mut repeated = d.clone();
    for _ in 0..1024 {
        repeated = repeated
            .transformed(detector::DetectorTransform::Left)
            .unwrap();
    }
    assert_eq!(
        repeated.position([3, 5], 0.5, 1.5),
        d.position([3, 5], 0.5, 1.5)
    );
    assert_eq!(
        repeated.builtin_mask([3, 5]).unwrap(),
        d.builtin_mask([3, 5]).unwrap()
    );
    assert!(!matches!(
        repeated.map,
        detector::PixelMap::Transformed { .. }
    ));
}
/// A transformed map moves whole pixels, so the offset inside a pixel turns
/// with it: positions on the logical corner grid (the shared corners of 2D
/// tables) must be the corners of the base pixel each logical pixel maps to.
#[test]
fn transformed_positions_on_the_corner_grid_are_the_pixel_corners() {
    use detector::DetectorTransform::*;
    let (n1, n2, p1, p2) = (3, 5, 1e-3, 2e-3);
    let slow = (0..n1)
        .map(|i| [[0., i as f64 * p1, 0.], [0., (i + 1) as f64 * p1, 0.]])
        .collect();
    let fast = (0..n2)
        .map(|j| [[0., 0., j as f64 * p2], [0., 0., (j + 1) as f64 * p2]])
        .collect();
    let mut base = detector::Detector::custom(p1, p2);
    base.max_shape = Some([n1, n2]);
    base.map = detector::PixelMap::Separable { slow, fast };
    for op in [
        Identity, Half, Transpose, Transverse, Left, Right, Horizontal, Vertical,
    ] {
        let d = base.transformed(op).unwrap();
        let shape = d.max_shape.unwrap();
        for r in 0..shape[0] {
            for c in 0..shape[1] {
                let mut corners = d.corners(shape, r, c).to_vec();
                for (dr, dc) in [(0., 0.), (1., 0.), (1., 1.), (0., 1.)] {
                    let p = d.position(shape, r as f64 + dr, c as f64 + dc);
                    let k = corners
                        .iter()
                        .position(|q| (0..3).all(|k| (p[k] - q[k]).abs() < 1e-12))
                        .unwrap_or_else(|| {
                            panic!("{op:?} pixel ({r}, {c}) corner +({dr}, {dc}) at {p:?}")
                        });
                    corners.swap_remove(k);
                }
            }
        }
    }
}
/// Base detectors for the transform invariant: non-uniform contiguous,
/// gapped (gaps inside and between binned pixels), curved, NeXus corners and
/// a uniform spline detector, each 6 × 12 with masked pixels.
fn transform_bases() -> Vec<(&'static str, detector::Detector)> {
    let (n1, n2, p1, p2) = (6usize, 12usize, 1e-4, 1.3e-4);
    let edges = |n: usize, p: f64, gaps: &[usize], axis: usize, curve: f64| {
        let mut lo = 0.;
        (0..n)
            .map(|i| {
                if gaps.contains(&i) {
                    lo += 0.37 * p;
                }
                let hi = lo + p * (1. + 0.1 * (i as f64).sin());
                let z = |x: f64| curve * (x / (n as f64 * p) - 0.4).powi(2);
                let mut a = [z(lo), 0., 0.];
                let mut b = [z(hi), 0., 0.];
                a[axis] = lo;
                b[axis] = hi;
                lo = hi;
                [a, b]
            })
            .collect::<Vec<_>>()
    };
    let separable = |gaps1: &[usize], gaps2: &[usize], curve: f64| {
        let mut d = detector::Detector::custom(p1, p2);
        d.max_shape = Some([n1, n2]);
        d.map = detector::PixelMap::Separable {
            slow: edges(n1, p1, gaps1, 1, curve),
            fast: edges(n2, p2, gaps2, 2, 0.5 * curve),
        };
        d.mask = vec![[3, 5], [n2 + 7, n2 + 8], [3 * n2 + 10, 4 * n2 + 2]];
        d
    };
    let contiguous = separable(&[], &[], 0.);
    // Row gaps before fine rows 1 and 4 (inside 2- and 3-binned pixels);
    // column gaps before 5 (inside both) and 9 (inside a 2-binned pixel, on a
    // 3-binned edge).
    let gapped = separable(&[1, 4], &[5, 9], 0.);
    let curved = separable(&[], &[], 2e-3);
    let mut corners = gapped.clone();
    let detector::PixelMap::Separable { slow, fast } = &gapped.map else {
        unreachable!()
    };
    corners.map = detector::PixelMap::Corners {
        shape: [n1, n2],
        values: Arc::new(
            (0..n1 * n2)
                .map(|i| {
                    let (s, f) = (slow[i / n2], fast[i % n2]);
                    let sum = |a: [f64; 3], b: [f64; 3]| std::array::from_fn(|k| a[k] + b[k]);
                    [
                        sum(s[0], f[0]),
                        sum(s[1], f[0]),
                        sum(s[1], f[1]),
                        sum(s[0], f[1]),
                    ]
                })
                .collect(),
        ),
    };
    let mut spline = detector::Detector::custom(p1, p2);
    spline.max_shape = Some([n1, n2]);
    spline.mask = contiguous.mask.clone();
    spline.spline = Some(Arc::new(
        Spline::parse(include_str!("../test_data/detector_reference.spline")).unwrap(),
    ));
    vec![
        ("contiguous", contiguous),
        ("gapped", gapped),
        ("curved", curved),
        ("corners", corners),
        ("spline", spline),
    ]
}
/// Logical point of a transformed image of shape `[h, w]` → base image point.
fn untransform(op: detector::DetectorTransform, [h, w]: [usize; 2], r: f64, c: f64) -> [f64; 2] {
    use detector::DetectorTransform::*;
    let (h, w) = (h as f64, w as f64);
    match op {
        Identity => [r, c],
        Half => [h - r, w - c],
        Transpose => [c, r],
        Transverse => [w - c, h - r],
        Left => [c, h - r],
        Right => [w - c, r],
        Horizontal => [r, w - c],
        Vertical => [h - r, c],
    }
}
/// A transformed map is the base detector seen through a flipped or rotated
/// image, at every binning: positions at corner-grid nodes and inside pixels,
/// pixel corners, masks and spline areas must all be the base detector's at
/// the inverse-mapped point. Binned pixels spanning module gaps, several
/// non-uniform or curved base pixels keep the base's binned quadrilateral.
#[test]
fn transformed_maps_match_the_base_detector_at_every_binning() {
    use detector::DetectorTransform::*;
    let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() <= 1e-15);
    for (name, mut base) in transform_bases() {
        for orientation in 0..=4 {
            base.orientation = orientation;
            for op in [
                Identity, Half, Transpose, Transverse, Left, Right, Horizontal, Vertical,
            ] {
                let t = base.transformed(op).unwrap();
                let full = base.max_shape.unwrap();
                for bin in [[1, 1], [2, 2], [2, 3], [3, 2], [1, 3]] {
                    let bshape = [full[0] / bin[0], full[1] / bin[1]];
                    let shape = if matches!(op, Transpose | Transverse | Left | Right) {
                        [bshape[1], bshape[0]]
                    } else {
                        bshape
                    };
                    assert_eq!(base.binning(bshape).unwrap(), bin);
                    let case = format!("{name} orientation {orientation} {op:?} bin {bin:?}");
                    let (tmask, bmask) = (
                        t.builtin_mask(shape).unwrap(),
                        base.builtin_mask(bshape).unwrap(),
                    );
                    assert!(bmask.contains(&1), "{case}: mask");
                    for r in 0..=shape[0] {
                        for c in 0..=shape[1] {
                            let [br, bc] = untransform(op, shape, r as f64, c as f64);
                            let (p, q) = (
                                t.position(shape, r as f64, c as f64),
                                base.position(bshape, br, bc),
                            );
                            assert!(close(p, q), "{case}: node ({r}, {c}) {p:?} != {q:?}");
                            if r == shape[0] || c == shape[1] {
                                continue;
                            }
                            for (fr, fc) in [(0.5, 0.5), (0.25, 0.7), (0.9, 0.05)] {
                                let (y, x) = (r as f64 + fr, c as f64 + fc);
                                let [br, bc] = untransform(op, shape, y, x);
                                let (p, q) =
                                    (t.position(shape, y, x), base.position(bshape, br, bc));
                                assert!(close(p, q), "{case}: ({y}, {x}) {p:?} != {q:?}");
                            }
                            let [br, bc] = untransform(op, shape, r as f64 + 0.5, c as f64 + 0.5);
                            let (br, bc) = (br.floor() as usize, bc.floor() as usize);
                            let mut expected = base.corners(bshape, br, bc).to_vec();
                            for p in t.corners(shape, r, c) {
                                let k = expected.iter().position(|q| close(p, *q));
                                let k = k.unwrap_or_else(|| {
                                    panic!("{case}: pixel ({r}, {c}) corner {p:?}")
                                });
                                expected.swap_remove(k);
                            }
                            assert_eq!(
                                tmask[r * shape[1] + c],
                                bmask[br * bshape[1] + bc],
                                "{case}: mask ({r}, {c})"
                            );
                            let (y, x) = (r as f64 + 0.5, c as f64 + 0.5);
                            assert_eq!(
                                t.spline_area(shape, y, x),
                                base.spline_area(bshape, br as f64 + 0.5, bc as f64 + 0.5),
                                "{case}: spline area ({r}, {c})"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn identification_requires_unique_consistent_evidence() {
    assert_eq!(detector::named("Aarhus").unwrap().config["radius"], 0.29989);
    assert_eq!(detector::named("Perkin").unwrap().config["pixel1"], 0.0002);
    assert!(detector::identify([1043, 981], None, "").selected.is_none());
    let d = detector::identify(
        [1043, 981],
        Some([172e-6; 2]),
        "DECTRIS Pilatus 1M, S/N 123",
    )
    .selected
    .unwrap();
    assert!(d.name.contains("Pilatus"));
    assert!(
        detector::identify([1043, 981], Some([79e-6; 2]), "Pilatus 1M")
            .selected
            .is_none()
    );
    assert!(detector::identify([3, 5], None, "").selected.is_none());
}

#[test]
fn refinement_uses_nonuniform_physical_positions_in_every_residual_and_jacobian() {
    use dioptrin::calibration::{Calibrant, CalibrationOptions, CalibrationPoints, ParamMask};
    let mut d = detector::Detector::custom(1e-4, 1.2e-4);
    d.max_shape = Some([64, 96]);
    let slow = (0..64)
        .map(|r| [r, r + 1].map(|r| [0., r as f64 * 1e-4 + 2e-5 * (r as f64 / 64.).powi(2), 0.]))
        .collect();
    let fast = (0..96)
        .map(|c| [c, c + 1].map(|x| [0., 0., x as f64 * 1.2e-4 + if c >= 48 { 1e-4 } else { 0. }]))
        .collect();
    d.map = detector::PixelMap::Separable { slow, fast };
    let mut p = Poni::new();
    p.distance = 0.045;
    p.poni1 = 0.0032;
    p.poni2 = 0.004;
    p.pixel1 = d.pixel1;
    p.pixel2 = d.pixel2;
    p.wavelength = 0.4e-10;
    let truth = Geometry::new(64, 96, p.clone())
        .with_detector(Arc::new(d))
        .unwrap();
    let targets: [f64; 3] = [0.04, 0.07, 0.10];
    let reference = Calibrant::from_d_spacings(
        "synthetic",
        targets
            .map(|t| p.wavelength * 1e10 / (2. * (t / 2.).sin()))
            .to_vec(),
    );
    let mut points = CalibrationPoints::new();
    for (ring, target) in targets.iter().enumerate() {
        for r in (0..64).step_by(4) {
            for c in 0..95 {
                let f = |col: f64| truth.calc_tth(&[r as f64 + 0.5], &[col + 0.5])[0] - target;
                if f(c as f64) * f(c as f64 + 1.) < 0. {
                    let (mut a, mut b) = (c as f64, c as f64 + 1.);
                    let sign = f(a);
                    for _ in 0..45 {
                        let mid = (a + b) / 2.;
                        if f(mid) * sign > 0. {
                            a = mid;
                        } else {
                            b = mid;
                        }
                    }
                    if f((a + b) / 2.).abs() < 1e-10 {
                        points.push(r as f64, (a + b) / 2., Some(ring));
                    }
                }
            }
        }
    }
    assert!(points.len() > 30);
    let mut guess = truth.clone();
    guess.poni.distance += 0.003;
    guess.poni.poni1 += 0.0002;
    guess.poni.poni2 -= 0.0003;
    let options = CalibrationOptions {
        mask: ParamMask {
            distance: true,
            poni1: true,
            poni2: true,
            rot1: false,
            rot2: false,
            rot3: false,
            wavelength: false,
        },
        tol: 1e-14,
        max_iter: 300,
        ..Default::default()
    };
    let fitted = dioptrin::calibration::refine(&guess, &points, &reference, &options).unwrap();
    assert!(
        (fitted.poni.distance - p.distance).abs() < 1e-8,
        "distance {} vs {}, rms {}",
        fitted.poni.distance,
        p.distance,
        fitted.rms_rad
    );
    assert!((fitted.poni.poni1 - p.poni1).abs() < 1e-8);
    assert!((fitted.poni.poni2 - p.poni2).abs() < 1e-8);
    assert!(fitted.rms_rad < 1e-9);
}
