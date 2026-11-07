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
#[test]
fn identification_requires_unique_consistent_evidence() {
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
