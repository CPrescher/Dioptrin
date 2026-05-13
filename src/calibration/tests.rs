//! Unit tests for the calibration module.
//!
//! Strategy: build a synthetic ground-truth `Poni`, sample N control points
//! along chosen calibrant rings using the *same* forward model the refiner
//! uses, then perturb each PONI parameter and check that refinement
//! recovers the ground truth to numerical tolerance.

use super::*;
use crate::geometry::Geometry;
use crate::poni::Poni;

fn truth_poni() -> Poni {
    let mut p = Poni::new();
    p.pixel1 = 7.9e-5;
    p.pixel2 = 7.9e-5;
    p.distance = 0.2; // 200 mm
    p.poni1 = 1024.0 * 7.9e-5; // ~81 mm — near the center of a ~2048x2048 detector
    p.poni2 = 1024.0 * 7.9e-5;
    p.rot1 = 0.01;
    p.rot2 = -0.005;
    p.rot3 = 0.0;
    p.wavelength = 3.1e-11; // ~40 keV
    p
}

/// Same closed-form 2θ used by the refiner — duplicated here so the test
/// does not lean on the function it is testing.
fn tth_truth(poni: &Poni, d1: f64, d2: f64) -> f64 {
    let p1 = (d1 + 0.5) * poni.pixel1 - poni.poni1;
    let p2 = (d2 + 0.5) * poni.pixel2 - poni.poni2;
    let p3 = poni.distance;
    let (sr1, cr1) = poni.rot1.sin_cos();
    let (sr2, cr2) = poni.rot2.sin_cos();
    let (sr3, cr3) = poni.rot3.sin_cos();
    let x =
        p1 * cr2 * cr3 + p2 * (cr3 * sr1 * sr2 - cr1 * sr3) - p3 * (cr1 * cr3 * sr2 + sr1 * sr3);
    let y =
        p1 * cr2 * sr3 + p2 * (cr1 * cr3 + sr1 * sr2 * sr3) - p3 * (-(cr3 * sr1) + cr1 * sr2 * sr3);
    let z = p1 * sr2 - p2 * cr2 * sr1 + p3 * cr1 * cr2;
    (x * x + y * y).sqrt().atan2(z)
}

/// Walk a circular arc in azimuth space until the predicted 2θ matches the
/// target. Returns `(d1, d2)` of the matched pixel via bisection along a
/// radial ray from the beam center for the given azimuth.
fn pixel_on_ring(poni: &Poni, ring_tth: f64, azimuth_rad: f64) -> Option<(f64, f64)> {
    // Beam center in pixel space (poni / pixel size). May lie off the detector.
    let cx = poni.poni1 / poni.pixel1;
    let cy = poni.poni2 / poni.pixel2;
    // Try radii from ~5 pixels up to a generous outer bound; bisect on tth.
    // Direction (du, dv) at this azimuth in pixel coordinates.
    let du = azimuth_rad.cos();
    let dv = azimuth_rad.sin();

    // Build a function r -> tth(cx + r*du, cy + r*dv).
    let f = |r: f64| -> f64 { tth_truth(poni, cx + r * du, cy + r * dv) };

    // Find a bracket where the sign of (f(r) - ring_tth) changes.
    let mut lo = 0.5;
    let mut hi = 0.5;
    let max_r = 5000.0;
    let mut g_lo = f(lo) - ring_tth;
    let mut found = false;
    while hi < max_r {
        hi *= 1.5;
        let g_hi = f(hi) - ring_tth;
        if g_lo.signum() != g_hi.signum() {
            found = true;
            break;
        }
        lo = hi;
        g_lo = g_hi;
    }
    if !found {
        return None;
    }
    // Bisect.
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let g_mid = f(mid) - ring_tth;
        if g_mid.signum() == g_lo.signum() {
            lo = mid;
            g_lo = g_mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-9 {
            break;
        }
    }
    let r = 0.5 * (lo + hi);
    Some((cx + r * du, cy + r * dv))
}

fn sample_rings(poni: &Poni, ring_tth: &[f64], points_per_ring: usize) -> CalibrationPoints {
    let mut pts = CalibrationPoints::new();
    for (k, &t) in ring_tth.iter().enumerate() {
        for j in 0..points_per_ring {
            let az = 2.0 * std::f64::consts::PI * (j as f64) / (points_per_ring as f64);
            if let Some((d1, d2)) = pixel_on_ring(poni, t, az) {
                pts.push(d1, d2, Some(k));
            }
        }
    }
    pts
}

fn lab6_calibrant() -> Calibrant {
    Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").unwrap()
}

#[test]
fn calibrant_parses_lab6() {
    let cal = lab6_calibrant();
    assert_eq!(cal.name, "LaB6_SRM660b");
    // First d-spacing should be the (1 0 0) at 4.15689 Å.
    assert!((cal.d_spacings[0] - 4.15689).abs() < 1e-5);
    // Sorted descending.
    for i in 1..cal.d_spacings.len() {
        assert!(cal.d_spacings[i] <= cal.d_spacings[i - 1]);
    }
}

#[test]
fn calibrant_parses_ceo2() {
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
    assert_eq!(cal.name, "CeO2");
    assert!((cal.d_spacings[0] - 3.12441816).abs() < 1e-7);
}

#[test]
fn calibrant_two_theta_at_known_wavelength() {
    // Cu Kα ≈ 1.5406 Å; for LaB6 (1 0 0) at d=4.15689 Å, 2θ ≈ 21.36°.
    let cal = lab6_calibrant();
    let wl = 1.5406e-10;
    let tth = cal.two_theta_rad(wl);
    let first = tth[0].unwrap();
    let expected = 2.0 * (1.5406_f64 / (2.0 * 4.15689)).asin();
    assert!((first - expected).abs() < 1e-12);
}

#[test]
fn refine_recovers_truth_with_small_perturbation() {
    let truth = truth_poni();
    let cal = lab6_calibrant();
    let ring_tth: Vec<f64> = cal
        .two_theta_rad(truth.wavelength)
        .into_iter()
        .flatten()
        .take(6)
        .collect();
    let pts = sample_rings(&truth, &ring_tth, 24);
    assert!(pts.len() >= 100, "need plenty of control points");

    // Perturb the truth → starting guess.
    let mut guess = truth.clone();
    guess.distance *= 1.02; // +2%
    guess.poni1 += 5.0 * truth.pixel1; // +5 pixels
    guess.poni2 -= 3.0 * truth.pixel2; // −3 pixels
    guess.rot1 += 5e-3;
    guess.rot2 -= 3e-3;

    let geom = Geometry::new(2048, 2048, guess);
    let opts = CalibrationOptions {
        mask: ParamMask::fix_wavelength(),
        ..CalibrationOptions::default()
    };
    let result = refine(&geom, &pts, &cal, &opts).expect("refine ok");

    // Distance and PONI should recover to sub-µm level on synthetic data.
    assert!(
        (result.poni.distance - truth.distance).abs() < 1e-7,
        "distance: got {} truth {}",
        result.poni.distance,
        truth.distance
    );
    assert!((result.poni.poni1 - truth.poni1).abs() < 1e-7);
    assert!((result.poni.poni2 - truth.poni2).abs() < 1e-7);
    assert!((result.poni.rot1 - truth.rot1).abs() < 1e-6);
    assert!((result.poni.rot2 - truth.rot2).abs() < 1e-6);
    // RMS residual essentially numerical noise.
    assert!(
        result.rms_rad < 1e-9,
        "RMS too high: {} rad",
        result.rms_rad
    );
}

#[test]
fn refine_handles_off_detector_beam_center() {
    // PONI outside the [0, 2048*pixel] detector window (negative offset).
    // 50 pixels off the corner is a realistic "tilted detector with off-axis
    // mount" case — the beam center physically exists in space but does not
    // fall on any detector pixel.
    let mut truth = truth_poni();
    truth.poni1 = -50.0 * truth.pixel1;
    truth.poni2 = -50.0 * truth.pixel2;

    let cal = lab6_calibrant();
    let ring_tth: Vec<f64> = cal
        .two_theta_rad(truth.wavelength)
        .into_iter()
        .flatten()
        .take(4)
        .collect();

    // Sweep full azimuth, keep whatever survives the on-detector filter.
    let mut pts = CalibrationPoints::new();
    for (k, &t) in ring_tth.iter().enumerate() {
        for j in 0..120 {
            let az = 2.0 * std::f64::consts::PI * (j as f64) / 120.0;
            if let Some((d1, d2)) = pixel_on_ring(&truth, t, az)
                && (0.0..2048.0).contains(&d1)
                && (0.0..2048.0).contains(&d2)
            {
                pts.push(d1, d2, Some(k));
            }
        }
    }
    assert!(
        pts.len() >= 30,
        "expected enough on-detector points; got {}",
        pts.len()
    );

    // Perturb.
    let mut guess = truth.clone();
    guess.distance *= 1.01;
    guess.poni1 += 10.0 * truth.pixel1;
    guess.poni2 -= 10.0 * truth.pixel2;

    let geom = Geometry::new(2048, 2048, guess);
    let opts = CalibrationOptions {
        mask: ParamMask::fix_wavelength(),
        ..CalibrationOptions::default()
    };
    let result = refine(&geom, &pts, &cal, &opts).expect("refine ok off-detector");
    assert!((result.poni.distance - truth.distance).abs() < 1e-6);
    assert!((result.poni.poni1 - truth.poni1).abs() < 1e-6);
    assert!((result.poni.poni2 - truth.poni2).abs() < 1e-6);
    // Sanity: the refined beam center IS off the detector.
    assert!(result.poni.poni1 < 0.0);
    assert!(result.poni.poni2 < 0.0);
}

#[test]
fn huber_loss_resists_outliers() {
    let truth = truth_poni();
    let cal = lab6_calibrant();
    let ring_tth: Vec<f64> = cal
        .two_theta_rad(truth.wavelength)
        .into_iter()
        .flatten()
        .take(5)
        .collect();
    let mut pts = sample_rings(&truth, &ring_tth, 24);
    // Plant 10% outliers: shift a few points off their ring by a large radial step.
    let n = pts.len();
    let n_bad = n / 10;
    for i in 0..n_bad {
        let idx = (i * 17 + 3) % n;
        // Shift the point ~50 pixels along the radial direction, simulating a
        // mis-assigned single-crystal spot.
        let cx = truth.poni1 / truth.pixel1;
        let cy = truth.poni2 / truth.pixel2;
        let dr1 = pts.d1[idx] - cx;
        let dr2 = pts.d2[idx] - cy;
        let norm = (dr1 * dr1 + dr2 * dr2).sqrt();
        if norm > 0.0 {
            pts.d1[idx] += 50.0 * dr1 / norm;
            pts.d2[idx] += 50.0 * dr2 / norm;
        }
    }

    let mut guess = truth.clone();
    guess.distance *= 1.005;
    guess.poni1 += 2.0 * truth.pixel1;

    let geom = Geometry::new(2048, 2048, guess);
    // Huber threshold: ~1 mrad in 2θ.
    let opts = CalibrationOptions {
        mask: ParamMask::fix_wavelength(),
        loss: Loss::Huber { c: 1.0e-3 },
        ..CalibrationOptions::default()
    };
    let robust = refine(&geom, &pts, &cal, &opts).expect("refine robust");

    // For comparison, plain L2 should be worse.
    let opts_l2 = CalibrationOptions {
        mask: ParamMask::fix_wavelength(),
        loss: Loss::L2,
        ..CalibrationOptions::default()
    };
    let l2 = refine(&geom, &pts, &cal, &opts_l2).expect("refine l2");

    let robust_err = (robust.poni.distance - truth.distance).abs();
    let l2_err = (l2.poni.distance - truth.distance).abs();
    assert!(
        robust_err < l2_err,
        "Huber should beat L2 on outlier-contaminated data: robust={:e} l2={:e}",
        robust_err,
        l2_err
    );
    // And it should recover distance reasonably well — within ~0.5 mm.
    assert!(
        robust_err < 5.0e-4,
        "robust distance error too large: {:e}",
        robust_err
    );
}

#[test]
fn auto_assign_finds_rings_when_left_blank() {
    let truth = truth_poni();
    let cal = lab6_calibrant();
    let ring_tth: Vec<f64> = cal
        .two_theta_rad(truth.wavelength)
        .into_iter()
        .flatten()
        .take(4)
        .collect();
    let mut pts = sample_rings(&truth, &ring_tth, 16);
    // Wipe ring assignments.
    for r in pts.ring.iter_mut() {
        *r = None;
    }

    let mut guess = truth.clone();
    guess.distance *= 1.005;
    guess.poni1 += 1.5 * truth.pixel1;

    let geom = Geometry::new(2048, 2048, guess);
    let opts = CalibrationOptions {
        mask: ParamMask::fix_wavelength(),
        auto_assign_rings: true,
        assign_tolerance_rad: 0.01,
        ..CalibrationOptions::default()
    };
    let result = refine(&geom, &pts, &cal, &opts).expect("refine auto-assign");
    assert!((result.poni.distance - truth.distance).abs() < 1e-6);
    // Every point should have been assigned a ring.
    assert!(result.assignments.iter().all(|r| r.is_some()));
}

#[test]
fn calibrant_rejects_empty_file() {
    let err = Calibrant::from_d_str("empty", "# only comments\n# no data\n").unwrap_err();
    matches!(err, CalibrantError::Empty);
}
