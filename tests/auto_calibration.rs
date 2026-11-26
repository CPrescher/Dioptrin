use std::path::Path;

use dioptrin::calibration::{AutoCalibrateImageOptions, Calibrant, auto_calibrate_image};
use dioptrin::frame::Frame;
use dioptrin::poni::Poni;

fn assert_poni_close(
    label: &str,
    got: &Poni,
    truth: &Poni,
    max_rel_dist: f64,
    max_poni_px: f64,
    max_rot_rad: f64,
) {
    let rel_dist = (got.distance - truth.distance).abs() / truth.distance;
    let dp1_px = (got.poni1 - truth.poni1).abs() / truth.pixel1;
    let dp2_px = (got.poni2 - truth.poni2).abs() / truth.pixel2;
    let drot1 = (got.rot1 - truth.rot1).abs();
    let drot2 = (got.rot2 - truth.rot2).abs();
    assert!(
        rel_dist < max_rel_dist,
        "{label}: distance rel err {rel_dist:.3e} >= {max_rel_dist:.3e} (got {}, truth {})",
        got.distance,
        truth.distance
    );
    assert!(
        dp1_px < max_poni_px && dp2_px < max_poni_px,
        "{label}: poni err ({dp1_px:.2} px, {dp2_px:.2} px) >= {max_poni_px} px"
    );
    assert!(
        drot1 < max_rot_rad && drot2 < max_rot_rad,
        "{label}: rot err ({drot1:.3e}, {drot2:.3e}) >= {max_rot_rad:.3e} rad"
    );
}

#[test]
fn auto_calibrate_image_ceo2_pilatus1m_blind_and_hinted() {
    let frame = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".to_string()).expect("load tiff");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load calibrant");
    let truth = Poni::load("test_data/v1.poni").expect("load truth poni");

    for starting_distance in [None, Some(0.20)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance,
            ..AutoCalibrateImageOptions::default()
        };
        let (res, diag) = auto_calibrate_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            &cal,
            truth.wavelength,
            truth.pixel1,
            truth.pixel2,
            &opts,
        )
        .expect("CeO2_Pilatus1M auto-calibration should succeed");

        assert!(res.rms_rad < 5e-4, "RMS too high: {:e}", res.rms_rad);
        assert!(
            diag.n_identified_ellipses >= 10,
            "expected >= 10 rings, got {}",
            diag.n_identified_ellipses
        );
        assert_poni_close(
            &format!("CeO2_Pilatus1M (d0={starting_distance:?})"),
            &res.poni,
            &truth,
            2e-3,
            3.0,
            2e-3,
        );
    }
}

#[test]
fn auto_calibrate_image_lab6_marccd_blind_and_hinted() {
    if !Path::new("test_data/LaB6_40keV_MarCCD.tif").exists() {
        return;
    }
    let frame = Frame::from_tiff("test_data/LaB6_40keV_MarCCD.tif".to_string()).expect("load tiff");
    let cal =
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load calibrant");
    let truth = Poni::load("test_data/LaB6_40keV_MarCCD.poni").expect("load truth poni");

    for starting_distance in [None, Some(0.20)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance,
            ..AutoCalibrateImageOptions::default()
        };
        let (res, diag) = auto_calibrate_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            &cal,
            truth.wavelength,
            truth.pixel1,
            truth.pixel2,
            &opts,
        )
        .expect("LaB6_40keV_MarCCD auto-calibration should succeed");

        assert!(res.rms_rad < 5e-4, "RMS too high: {:e}", res.rms_rad);
        assert!(
            diag.n_identified_ellipses >= 12,
            "expected >= 12 rings, got {}",
            diag.n_identified_ellipses
        );
        assert_poni_close(
            &format!("LaB6_40keV_MarCCD (d0={starting_distance:?})"),
            &res.poni,
            &truth,
            2e-3,
            8.0,
            3e-3,
        );
    }
}

#[test]
fn auto_calibrate_image_lab6_offcenter_pe_blind_and_hinted() {
    if !Path::new("test_data/LaB6_OffCenter_PE.tif").exists() {
        return;
    }
    let frame = Frame::from_tiff("test_data/LaB6_OffCenter_PE.tif".to_string()).expect("load tiff");
    let cal =
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load calibrant");
    let wavelength = 3.344e-11;
    let pixel = 200e-6;

    for starting_distance in [None, Some(0.50)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance,
            ..AutoCalibrateImageOptions::default()
        };
        let (res, diag) = auto_calibrate_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            &cal,
            wavelength,
            pixel,
            pixel,
            &opts,
        )
        .expect("LaB6_OffCenter_PE auto-calibration should succeed");

        assert!(
            (res.poni.distance - 0.500).abs() < 0.01,
            "expected distance ~0.500 m, got {}",
            res.poni.distance
        );
        assert!(res.rms_rad < 1e-3, "RMS too high: {:e}", res.rms_rad);
        assert!(
            diag.n_identified_ellipses >= 12,
            "expected >= 12 rings, got {}",
            diag.n_identified_ellipses
        );
        // Beam center lies off the 2048-wide detector along axis 2 (> 2048 * 200 µm = 0.4096 m).
        assert!(
            res.poni.poni2 > 2048.0 * pixel,
            "expected off-detector poni2, got {}",
            res.poni.poni2
        );
    }
}

#[test]
fn auto_calibrate_image_ceo2_side_strip_blind_and_hinted() {
    if !Path::new("test_data/ceo2_side.tif").exists() {
        return;
    }
    let frame = Frame::from_tiff("test_data/ceo2_side.tif".to_string()).expect("load tiff");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load calibrant");
    let truth = Poni::load("test_data/ceo2_side.poni").expect("load truth poni");

    for starting_distance in [None, Some(0.25)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance,
            ..AutoCalibrateImageOptions::default()
        };
        let (res, diag) = auto_calibrate_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            &cal,
            truth.wavelength,
            truth.pixel1,
            truth.pixel2,
            &opts,
        )
        .expect("ceo2_side auto-calibration should succeed");

        assert!(res.rms_rad < 1e-3, "RMS too high: {:e}", res.rms_rad);
        assert!(
            diag.n_identified_ellipses >= 8,
            "expected >= 8 rings, got {}",
            diag.n_identified_ellipses
        );
        assert_poni_close(
            &format!("ceo2_side (d0={starting_distance:?})"),
            &res.poni,
            &truth,
            5e-3,
            6.0,
            5e-3,
        );
    }
}
