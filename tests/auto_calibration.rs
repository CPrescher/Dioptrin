use std::f64::consts::PI;

use dioptrin::calibration::{
    AutoCalibrateError, AutoCalibrateImageOptions, Calibrant, auto_calibrate_image,
    auto_calibrate_image_with_mask,
};
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

// ---------- Synthetic diffraction image helper ----------

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_f64(state: &mut u64) -> f64 {
    (next_u64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn tth_forward(poni: &Poni, d1: f64, d2: f64) -> f64 {
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

struct SynthSpec {
    dim1: usize,
    dim2: usize,
    ring_sigma_rad: f64,
    baseline: f64,
    noise_amp: f64,
    spots_per_ring: Option<usize>,
    n_outliers: usize,
    seed: u64,
}

fn render_synthetic(poni: &Poni, cal: &Calibrant, spec: &SynthSpec) -> Vec<f64> {
    let mut rng = spec.seed;
    let mut img = vec![spec.baseline; spec.dim1 * spec.dim2];
    let rings: Vec<f64> = cal
        .two_theta_rad(poni.wavelength)
        .into_iter()
        .flatten()
        .take(20)
        .collect();

    if let Some(n_spots) = spec.spots_per_ring {
        let (sr1, cr1) = poni.rot1.sin_cos();
        let (sr2, cr2) = poni.rot2.sin_cos();
        let (sr3, cr3) = poni.rot3.sin_cos();
        let r00 = cr2 * cr3;
        let r01 = cr3 * sr1 * sr2 - cr1 * sr3;
        let r02 = -(cr1 * cr3 * sr2 + sr1 * sr3);
        let r10 = cr2 * sr3;
        let r11 = cr1 * cr3 + sr1 * sr2 * sr3;
        let r12 = cr3 * sr1 - cr1 * sr2 * sr3;
        let r20 = sr2;
        let r21 = -cr2 * sr1;
        let r22 = cr1 * cr2;

        for (k, &tth) in rings.iter().enumerate() {
            let amp_base = 800.0 / (1.0 + 0.15 * k as f64);
            for _ in 0..n_spots {
                let phi = 2.0 * PI * rand_f64(&mut rng);
                let (st, ct) = tth.sin_cos();
                let (sp, cp) = phi.sin_cos();
                let (ux, uy, uz) = (st * cp, st * sp, ct);
                let q1 = r00 * ux + r10 * uy + r20 * uz;
                let q2 = r01 * ux + r11 * uy + r21 * uz;
                let q3 = r02 * ux + r12 * uy + r22 * uz;
                if q3 <= 1e-6 {
                    continue;
                }
                let t = poni.distance / q3;
                let d1 = (t * q1 + poni.poni1) / poni.pixel1 - 0.5;
                let d2 = (t * q2 + poni.poni2) / poni.pixel2 - 0.5;
                if d1 < 3.0
                    || d2 < 3.0
                    || d1 >= spec.dim1 as f64 - 3.0
                    || d2 >= spec.dim2 as f64 - 3.0
                {
                    continue;
                }
                let amp = amp_base * (0.5 + rand_f64(&mut rng));
                let sig = 1.5;
                let i0 = (d1 as isize - 5).max(0) as usize;
                let i1 = (d1 as isize + 5).min(spec.dim1 as isize - 1) as usize;
                let j0 = (d2 as isize - 5).max(0) as usize;
                let j1 = (d2 as isize + 5).min(spec.dim2 as isize - 1) as usize;
                for i in i0..=i1 {
                    let dy = i as f64 - d1;
                    for j in j0..=j1 {
                        let dx = j as f64 - d2;
                        img[i * spec.dim2 + j] +=
                            amp * (-(dy * dy + dx * dx) / (2.0 * sig * sig)).exp();
                    }
                }
            }
        }
    } else {
        let two_s2 = 2.0 * spec.ring_sigma_rad * spec.ring_sigma_rad;
        for i in 0..spec.dim1 {
            for j in 0..spec.dim2 {
                let tth = tth_forward(poni, i as f64, j as f64);
                let mut val = spec.baseline;
                for (k, &rt) in rings.iter().enumerate() {
                    let dt = tth - rt;
                    if dt.abs() < 4.5 * spec.ring_sigma_rad {
                        let amp = 600.0 / (1.0 + 0.12 * k as f64);
                        val += amp * (-(dt * dt) / two_s2).exp();
                    }
                }
                img[i * spec.dim2 + j] = val;
            }
        }
    }

    for _ in 0..spec.n_outliers {
        let d1 = 8.0 + rand_f64(&mut rng) * (spec.dim1 - 16) as f64;
        let d2 = 8.0 + rand_f64(&mut rng) * (spec.dim2 - 16) as f64;
        let amp = 3000.0 + 4000.0 * rand_f64(&mut rng);
        let sig = 1.8;
        let i0 = (d1 as isize - 6).max(0) as usize;
        let i1 = (d1 as isize + 6).min(spec.dim1 as isize - 1) as usize;
        let j0 = (d2 as isize - 6).max(0) as usize;
        let j1 = (d2 as isize + 6).min(spec.dim2 as isize - 1) as usize;
        for i in i0..=i1 {
            let dy = i as f64 - d1;
            for j in j0..=j1 {
                let dx = j as f64 - d2;
                img[i * spec.dim2 + j] += amp * (-(dy * dy + dx * dx) / (2.0 * sig * sig)).exp();
            }
        }
    }

    if spec.noise_amp > 0.0 {
        for v in &mut img {
            let u = rand_f64(&mut rng) + rand_f64(&mut rng) - 1.0;
            *v += u * spec.noise_amp;
        }
    }
    img
}

// ---------- Self-contained synthetic tests (always run on CI) ----------

#[test]
fn auto_calibrate_image_synthetic_variants() {
    let ceo2 = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");

    // (a) Standard 512x512 tilted CeO2, blind and hinted.
    let mut truth = Poni::new();
    truth.pixel1 = 100e-6;
    truth.pixel2 = 100e-6;
    truth.distance = 0.15;
    truth.poni1 = (270.3 + 0.5) * 100e-6;
    truth.poni2 = (241.7 + 0.5) * 100e-6;
    truth.rot1 = 0.03;
    truth.rot2 = -0.02;
    truth.wavelength = 3.0e-11;

    let img = render_synthetic(
        &truth,
        &ceo2,
        &SynthSpec {
            dim1: 512,
            dim2: 512,
            ring_sigma_rad: 8.0e-4,
            baseline: 100.0,
            noise_amp: 5.0,
            spots_per_ring: None,
            n_outliers: 0,
            seed: 42,
        },
    );
    for d0 in [None, Some(0.18)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance: d0,
            ..Default::default()
        };
        let (res, diag) = auto_calibrate_image(
            &img,
            512,
            512,
            &ceo2,
            truth.wavelength,
            truth.pixel1,
            truth.pixel2,
            &opts,
        )
        .expect("synthetic CeO2 should calibrate");
        assert!(diag.n_identified_ellipses >= 5);
        assert_poni_close(
            &format!("synth standard (d0={d0:?})"),
            &res.poni,
            &truth,
            5e-4,
            0.5,
            5e-4,
        );
    }

    // (b) Gap row/col (-1), zero border, and explicit user mask.
    let mut img_masked = img.clone();
    for j in 0..512 {
        img_masked[256 * 512 + j] = -1.0;
        for b in 0..10 {
            img_masked[b * 512 + j] = 0.0;
            img_masked[(511 - b) * 512 + j] = 0.0;
        }
    }
    for i in 0..512 {
        img_masked[i * 512 + 256] = -1.0;
    }
    let mut user_mask = vec![false; 512 * 512];
    for i in 0..40 {
        for j in 0..40 {
            user_mask[i * 512 + j] = true;
            img_masked[i * 512 + j] = 1e6; // bright artifact hidden behind user mask
        }
    }
    let (res_m, _) = auto_calibrate_image_with_mask(
        &img_masked,
        512,
        512,
        Some(&user_mask),
        &ceo2,
        truth.wavelength,
        truth.pixel1,
        truth.pixel2,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("masked synthetic CeO2 should calibrate");
    assert_poni_close("synth masked", &res_m.poni, &truth, 1e-3, 1.0, 1e-3);

    // (c) Non-square pixels (100 x 150 um).
    let mut truth_ns = truth.clone();
    truth_ns.pixel1 = 100e-6;
    truth_ns.pixel2 = 150e-6;
    truth_ns.poni1 = (270.3 + 0.5) * 100e-6;
    truth_ns.poni2 = (180.4 + 0.5) * 150e-6;
    let img_ns = render_synthetic(
        &truth_ns,
        &ceo2,
        &SynthSpec {
            dim1: 512,
            dim2: 384,
            ring_sigma_rad: 8.0e-4,
            baseline: 100.0,
            noise_amp: 5.0,
            spots_per_ring: None,
            n_outliers: 0,
            seed: 43,
        },
    );
    let (res_ns, _) = auto_calibrate_image(
        &img_ns,
        512,
        384,
        &ceo2,
        truth_ns.wavelength,
        truth_ns.pixel1,
        truth_ns.pixel2,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("non-square pixel synthetic CeO2 should calibrate");
    assert_poni_close("synth non-square", &res_ns.poni, &truth_ns, 1e-3, 1.0, 1e-3);

    // (d) Dark-subtracted image with zero-mean background noise (~50% negative pixels).
    let img_dark = render_synthetic(
        &truth,
        &ceo2,
        &SynthSpec {
            dim1: 512,
            dim2: 512,
            ring_sigma_rad: 8.0e-4,
            baseline: 0.0,
            noise_amp: 12.0,
            spots_per_ring: None,
            n_outliers: 0,
            seed: 44,
        },
    );
    let (res_dark, _) = auto_calibrate_image(
        &img_dark,
        512,
        512,
        &ceo2,
        truth.wavelength,
        truth.pixel1,
        truth.pixel2,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("dark-subtracted synthetic CeO2 should calibrate");
    assert_poni_close(
        "synth dark-subtracted",
        &res_dark.poni,
        &truth,
        1e-3,
        1.0,
        1e-3,
    );

    // (e) Spotty powder (24 discrete spots/ring + 20 bright outliers), blind.
    let mut truth_sp = truth.clone();
    truth_sp.distance = 0.10;
    let img_spotty = render_synthetic(
        &truth_sp,
        &ceo2,
        &SynthSpec {
            dim1: 512,
            dim2: 512,
            ring_sigma_rad: 8.0e-4,
            baseline: 100.0,
            noise_amp: 6.0,
            spots_per_ring: Some(24),
            n_outliers: 20,
            seed: 45,
        },
    );
    let (res_sp, _) = auto_calibrate_image(
        &img_spotty,
        512,
        512,
        &ceo2,
        truth_sp.wavelength,
        truth_sp.pixel1,
        truth_sp.pixel2,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("spotty synthetic CeO2 should calibrate blind");
    assert_poni_close("synth spotty", &res_sp.poni, &truth_sp, 2e-3, 3.0, 3e-3);
}

#[test]
fn auto_calibrate_image_rejects_invalid_and_uncalibratable_inputs() {
    let ceo2 = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
    let opts = AutoCalibrateImageOptions::default();

    // Shape mismatch → InvalidInput (must not panic).
    let err =
        auto_calibrate_image(&[1.0; 100], 16, 16, &ceo2, 3e-11, 1e-4, 1e-4, &opts).unwrap_err();
    assert!(matches!(err, AutoCalibrateError::InvalidInput(_)));

    // Too small → InvalidInput.
    let err = auto_calibrate_image(&[1.0; 64], 8, 8, &ceo2, 3e-11, 1e-4, 1e-4, &opts).unwrap_err();
    assert!(matches!(err, AutoCalibrateError::InvalidInput(_)));

    // Non-positive wavelength / pixel size → InvalidInput.
    let img = vec![10.0; 64 * 64];
    assert!(matches!(
        auto_calibrate_image(&img, 64, 64, &ceo2, 0.0, 1e-4, 1e-4, &opts),
        Err(AutoCalibrateError::InvalidInput(_))
    ));
    assert!(matches!(
        auto_calibrate_image(&img, 64, 64, &ceo2, 3e-11, -1e-4, 1e-4, &opts),
        Err(AutoCalibrateError::InvalidInput(_))
    ));

    // Flat image → Err.
    assert!(auto_calibrate_image(&img, 64, 64, &ceo2, 3e-11, 1e-4, 1e-4, &opts).is_err());

    // All-NaN image → Err.
    let nan_img = vec![f64::NAN; 64 * 64];
    assert!(auto_calibrate_image(&nan_img, 64, 64, &ceo2, 3e-11, 1e-4, 1e-4, &opts).is_err());

    // Pure random noise → Err.
    let mut rng = 99u64;
    let noise: Vec<f64> = (0..128 * 128)
        .map(|_| 100.0 + 10.0 * rand_f64(&mut rng))
        .collect();
    assert!(auto_calibrate_image(&noise, 128, 128, &ceo2, 3e-11, 1e-4, 1e-4, &opts).is_err());

    // Long-distance strip (D=1.0 m) with hint 1.0 m must NOT return an unverified wrong geometry.
    let mut strip_poni = Poni::new();
    strip_poni.pixel1 = 172e-6;
    strip_poni.pixel2 = 172e-6;
    strip_poni.distance = 1.0;
    strip_poni.poni1 = 150.0 * 172e-6;
    strip_poni.poni2 = 200.0 * 172e-6;
    strip_poni.rot1 = 0.05;
    strip_poni.wavelength = 2.0e-11;
    let strip_img = render_synthetic(
        &strip_poni,
        &ceo2,
        &SynthSpec {
            dim1: 200,
            dim2: 800,
            ring_sigma_rad: 5.0e-4,
            baseline: 100.0,
            noise_amp: 5.0,
            spots_per_ring: None,
            n_outliers: 0,
            seed: 77,
        },
    );
    let strip_opts = AutoCalibrateImageOptions {
        starting_distance: Some(1.0),
        ..Default::default()
    };
    if let Ok((res, _)) = auto_calibrate_image(
        &strip_img,
        200,
        800,
        &ceo2,
        strip_poni.wavelength,
        strip_poni.pixel1,
        strip_poni.pixel2,
        &strip_opts,
    ) {
        assert!(
            (res.poni.distance - 1.0).abs() / 1.0 < 0.02,
            "returned Ok with wrong distance: {}",
            res.poni.distance
        );
    }
}

// ---------- Real detector image tests ----------

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
fn auto_calibrate_legacy_ceo2_side_grid() {
    use dioptrin::calibration::{
        AutoCalibrateOptions, AutoMethod, ImagePeakOptions, auto_calibrate, find_peaks_image,
    };
    let frame = Frame::from_tiff("test_data/ceo2_side.tif".to_string()).expect("load tiff");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load calibrant");
    let truth = Poni::load("test_data/ceo2_side.poni").expect("load truth poni");
    let peaks = find_peaks_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        &ImagePeakOptions {
            threshold_factor: 3.0,
            max_peaks: 5000,
            ..ImagePeakOptions::default()
        },
    );
    let mut opts = AutoCalibrateOptions {
        method: AutoMethod::Auto,
        ..AutoCalibrateOptions::default()
    };
    opts.grid.distance_min = 0.05;
    opts.grid.distance_max = 0.6;
    opts.grid.distance_steps = 18;
    opts.grid.rot_max = 45.0_f64.to_radians();
    opts.grid.rot_steps = 15;
    opts.grid.poni_search_radius_px = 400.0;
    opts.grid.poni_grid_steps = 13;
    opts.grid.tolerance_rad = 8.0e-3;
    opts.grid.max_peaks_for_grid = 400;
    opts.grid.max_ring_index = 12;
    opts.grid.top_k = 50;
    let (res, _diag) = auto_calibrate(
        &peaks,
        &cal,
        truth.wavelength,
        truth.pixel1,
        truth.pixel2,
        &opts,
    )
    .expect("legacy auto_calibrate ceo2_side");
    assert_poni_close("legacy ceo2_side", &res.poni, &truth, 5e-3, 6.0, 1e-2);
}

#[test]
#[ignore = "requires untracked local fixture test_data/LaB6_OffCenter_PE.tif"]
fn auto_calibrate_image_lab6_offcenter_pe_blind_and_hinted() {
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
        assert!(
            res.rms_rad < 1e-3,
            "RMS too high: {:e} (poni={:?}, rings={})",
            res.rms_rad,
            res.poni,
            diag.n_identified_ellipses
        );
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
    let frame = Frame::from_tiff("test_data/ceo2_side.tif".to_string()).expect("load tiff");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load calibrant");
    let truth = Poni::load("test_data/ceo2_side.poni").expect("load truth poni");

    for starting_distance in [None, Some(0.25)] {
        let opts = AutoCalibrateImageOptions {
            starting_distance,
            ..AutoCalibrateImageOptions::default()
        };
        let mut trace = Vec::new();
        let (res, diag) = dioptrin::calibration::auto_calibrate_image_with_progress(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &cal,
            truth.wavelength,
            truth.pixel1,
            truth.pixel2,
            &opts,
            |ev| {
                trace.push(format!("{:?}: {}", ev.stage, ev.message));
                true
            },
        )
        .unwrap_or_else(|e| panic!("ceo2_side failed: {e}\ntrace:\n{}", trace.join("\n")));

        assert!(
            res.rms_rad < 1e-3,
            "RMS too high: {:e} (poni={:?}, rings={})",
            res.rms_rad,
            res.poni,
            diag.n_identified_ellipses
        );
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

#[test]
#[ignore = "requires untracked local fixture test_data/LaB6_OffCenter_PE.tif"]
fn auto_calibrate_lab6_offcenter_pe_quarter_sections() {
    use std::time::Instant;

    let frame = Frame::from_tiff("test_data/LaB6_OffCenter_PE.tif".to_string()).expect("load tiff");
    let cal =
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load calibrant");
    let wavelength = 3.344e-11;
    let pixel = 200e-6;

    // Reference geometry from the full 2048×2048 image.
    let (full_res, _) = auto_calibrate_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        &cal,
        wavelength,
        pixel,
        pixel,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("full LaB6_OffCenter_PE calibration");
    let truth = full_res.poni;
    println!(
        "\nFull-frame truth: D={:.4} m, poni=({:.1}, {:.1}) px, rot=({:.2}°, {:.2}°)",
        truth.distance,
        truth.poni1 / pixel,
        truth.poni2 / pixel,
        truth.rot1.to_degrees(),
        truth.rot2.to_degrees(),
    );

    // 1/4 area sections (1024×1024) tiled across a 3×3 grid covering every region.
    let crop_h = frame.dim1 / 2;
    let crop_w = frame.dim2 / 2;
    let offsets = [0usize, frame.dim1 / 4, frame.dim1 / 2];

    println!(
        "{:<18} | {:<40} | {:<40}",
        "SECTION (r0,c0)", "BLIND (d0=None)", "HINTED (d0=0.50 m)"
    );
    println!("{}", "-".repeat(104));

    let mut blind_ok = 0usize;
    let mut hinted_ok = 0usize;
    let mut total = 0usize;

    for &r0 in &offsets {
        for &c0 in &offsets {
            total += 1;
            let mut crop = Vec::with_capacity(crop_h * crop_w);
            for i in r0..(r0 + crop_h) {
                let row = i * frame.dim2;
                crop.extend_from_slice(&frame.data[row + c0..row + c0 + crop_w]);
            }
            let expected_p1 = truth.poni1 - (r0 as f64) * pixel;
            let expected_p2 = truth.poni2 - (c0 as f64) * pixel;

            let mut cols = Vec::new();
            for (idx, d0) in [None, Some(0.50)].into_iter().enumerate() {
                let opts = AutoCalibrateImageOptions {
                    starting_distance: d0,
                    ..AutoCalibrateImageOptions::default()
                };
                let t0 = Instant::now();
                let out = auto_calibrate_image(
                    &crop, crop_h, crop_w, &cal, wavelength, pixel, pixel, &opts,
                );
                let ms = t0.elapsed().as_millis();
                match out {
                    Ok((res, diag)) => {
                        let rel_d = (res.poni.distance - truth.distance).abs() / truth.distance;
                        let dp1 = (res.poni.poni1 - expected_p1).abs() / pixel;
                        let dp2 = (res.poni.poni2 - expected_p2).abs() / pixel;
                        let drot = (res.poni.rot1 - truth.rot1)
                            .abs()
                            .max((res.poni.rot2 - truth.rot2).abs())
                            .to_degrees();
                        let ok = rel_d < 0.015 && dp1.max(dp2) < 15.0 && drot < 0.75;
                        if ok {
                            if idx == 0 {
                                blind_ok += 1;
                            } else {
                                hinted_ok += 1;
                            }
                        }
                        let tag = if ok { "OK  " } else { "WRNG" };
                        cols.push(format!(
                            "{tag} d={:.2}% p={:.1}px r={:.2}° {}r {ms}ms",
                            rel_d * 100.0,
                            dp1.max(dp2),
                            drot,
                            diag.n_identified_ellipses
                        ));
                    }
                    Err(e) => {
                        cols.push(format!("ERR  ({e:?}) {ms}ms"));
                    }
                }
            }
            println!(
                "({:>4},{:>4}) {:>4}x{:<4} | {:<40} | {:<40}",
                r0, c0, crop_h, crop_w, cols[0], cols[1]
            );
        }
    }
    println!("{}", "-".repeat(104));
    println!("SUMMARY: blind {blind_ok}/{total} passed, hinted {hinted_ok}/{total} passed\n");
}

#[test]
fn extract_ring_peaks_and_refine_image_from_perturbed_poni() {
    use dioptrin::calibration::{RefineImageOptions, extract_ring_peaks, refine_image};
    use dioptrin::geometry::Geometry;
    use dioptrin::simulation::{RingIntensity, SimulationOptions, simulate_diffraction};

    // 1. Real CeO2_Pilatus1M from a perturbed starting PONI (offset by ~8 px and +2% distance).
    let frame = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".into()).expect("load CeO2");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2.D");
    let truth = Poni::load("test_data/v1.poni").expect("load v1.poni");

    let mut perturbed = truth.clone();
    perturbed.distance *= 1.02;
    perturbed.poni1 += 6.0 * truth.pixel1;
    perturbed.poni2 -= 7.0 * truth.pixel2;
    perturbed.rot1 += 0.5_f64.to_radians();

    let pts = extract_ring_peaks(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &truth,
        &cal,
        &RefineImageOptions::default(),
    )
    .expect("extract_ring_peaks should succeed");
    assert!(
        pts.len() >= 200,
        "expected >= 200 ring points, got {}",
        pts.len()
    );

    let refined = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &perturbed,
        &cal,
        &RefineImageOptions::default(),
    )
    .expect("refine_image on CeO2_Pilatus1M should succeed");
    assert!(refined.n_rings >= 12);
    assert!(
        refined.points.len() >= 150,
        "expected >= 150 refined ring points, got {}",
        refined.points.len()
    );
    assert_poni_close(
        "refine_image CeO2_Pilatus1M",
        &refined.calibration.poni,
        &truth,
        1.5e-3,
        1.5,
        2e-3,
    );

    // 2. Simulated polarized image with known rot3 = +24°: refine_image recovers
    //    geometry AND rot3 in one call.
    let mut truth_pol = Poni {
        pixel1: 172e-6,
        pixel2: 172e-6,
        distance: 0.12,
        poni1: 256.0 * 172e-6,
        poni2: 240.0 * 172e-6,
        rot1: 5.0_f64.to_radians(),
        rot2: -4.0_f64.to_radians(),
        rot3: 24.0_f64.to_radians(),
        wavelength: 0.4066e-10,
        ..Poni::new()
    };
    let geom = Geometry::new(512, 512, truth_pol.clone());
    let sim = simulate_diffraction(
        &geom,
        &cal,
        &SimulationOptions {
            fwhm_rad: 2.5e-3,
            ring_intensity: RingIntensity::FromCalibrant { fallback: 20_000.0 },
            background: 50.0,
            apply_solid_angle: true,
            polarization_factor: Some(0.99),
            poisson_noise: true,
            rng_seed: 2026,
            ..Default::default()
        },
    )
    .expect("simulate polarized");

    let mut start_pol = truth_pol.clone();
    start_pol.distance *= 0.985;
    start_pol.poni1 -= 5.0 * truth_pol.pixel1;
    start_pol.rot3 = 0.0;

    let opts_pol = RefineImageOptions {
        polarization_factor: Some(0.99),
        ..Default::default()
    };
    let ref_pol = refine_image(
        &sim.data, sim.dim1, sim.dim2, None, &start_pol, &cal, &opts_pol,
    )
    .expect("refine_image polarized should succeed");
    let pol = ref_pol
        .rot3_calibration
        .as_ref()
        .expect("rot3_calibration should be populated");
    assert!(pol.reliable, "expected reliable rot3 fit, got {pol:?}");
    let rot3_err_deg = (ref_pol.calibration.poni.rot3 - truth_pol.rot3)
        .abs()
        .to_degrees();
    assert!(
        rot3_err_deg < 2.0,
        "expected rot3 ~24°, got {:.2}° (err={rot3_err_deg:.2}°)",
        ref_pol.calibration.poni.rot3.to_degrees()
    );
    truth_pol.rot3 = ref_pol.calibration.poni.rot3;
    assert_poni_close(
        "refine_image polarized",
        &ref_pol.calibration.poni,
        &truth_pol,
        1e-3,
        1.0,
        2e-3,
    );

    // 3. Fast 2D multi-pass path on small (2 px) perturbation without fallback.
    let mut small_pert = truth.clone();
    small_pert.poni1 += 2.0 * truth.pixel1;
    small_pert.poni2 -= 2.0 * truth.pixel2;
    let ref_small = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &small_pert,
        &cal,
        &RefineImageOptions {
            search_half_width_px: Some(3), // must not panic when hw < 6
            ..Default::default()
        },
    )
    .expect("small perturbation with hw=3 should succeed without panicking");
    assert_poni_close(
        "refine_image small pert hw=3",
        &ref_small.calibration.poni,
        &truth,
        1.5e-3,
        1.5,
        2e-3,
    );

    // 4. Input validation & wrong-calibrant rejection.
    assert!(matches!(
        refine_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &truth,
            &cal,
            &RefineImageOptions {
                search_half_width_px: Some(0),
                ..Default::default()
            },
        ),
        Err(dioptrin::calibration::AutoCalibrateError::InvalidInput(_))
    ));
    assert!(matches!(
        refine_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &truth,
            &cal,
            &RefineImageOptions {
                polarization_factor: Some(1.5),
                ..Default::default()
            },
        ),
        Err(dioptrin::calibration::AutoCalibrateError::InvalidInput(_))
    ));
    let lab6 = Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load LaB6.D");
    assert!(
        refine_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &truth,
            &lab6,
            &RefineImageOptions::default(),
        )
        .is_err(),
        "wrong calibrant (LaB6 on CeO2) must be rejected"
    );
}

#[test]
#[ignore = "requires untracked local fixture test_data/LaB6_OffCenter_PE.tif"]
fn auto_calibrate_and_refine_lab6_offcenter_extends_to_full_detector_rings() {
    use dioptrin::calibration::{RefineImageOptions, refine_image};

    let frame = Frame::from_tiff("test_data/LaB6_OffCenter_PE.tif".into()).expect("load tiff");
    let cal =
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load calibrant");
    let wavelength = 3.344e-11;
    let pixel = 200e-6;

    let (auto_res, auto_diag) = auto_calibrate_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        &cal,
        wavelength,
        pixel,
        pixel,
        &AutoCalibrateImageOptions::default(),
    )
    .expect("auto_calibrate_image LaB6_OffCenter_PE");

    println!(
        "auto_calibrate_image LaB6_OffCenter_PE: rings={}, D={:.5}m, poni=({:.2}, {:.2})px, rot=({:.3}°, {:.3}°), rms={:.5}°",
        auto_diag.n_identified_ellipses,
        auto_res.poni.distance,
        auto_res.poni.poni1 / pixel,
        auto_res.poni.poni2 / pixel,
        auto_res.poni.rot1.to_degrees(),
        auto_res.poni.rot2.to_degrees(),
        auto_res.rms_rad.to_degrees(),
    );

    let ref_res = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &auto_res.poni,
        &cal,
        &RefineImageOptions {
            polarization_factor: Some(0.99),
            ..Default::default()
        },
    )
    .expect("refine_image LaB6_OffCenter_PE");

    println!(
        "refine_image         LaB6_OffCenter_PE: rings={}, pts={}, D={:.5}m, poni=({:.2}, {:.2})px, rot=({:.3}°, {:.3}°, {:.2}°), rms={:.5}°",
        ref_res.n_rings,
        ref_res.points.len(),
        ref_res.calibration.poni.distance,
        ref_res.calibration.poni.poni1 / pixel,
        ref_res.calibration.poni.poni2 / pixel,
        ref_res.calibration.poni.rot1.to_degrees(),
        ref_res.calibration.poni.rot2.to_degrees(),
        ref_res.calibration.poni.rot3.to_degrees(),
        ref_res.calibration.rms_rad.to_degrees(),
    );
    let mut per_ring = std::collections::BTreeMap::<usize, usize>::new();
    for r in ref_res.points.ring.iter().flatten() {
        *per_ring.entry(*r).or_default() += 1;
    }
    let max_ring_idx = per_ring.keys().copied().max().unwrap_or(0);
    println!("max ring index reached: {max_ring_idx}");

    // Must extend all the way into outer rings (past index 60) with zero drift.
    assert!(
        auto_diag.n_identified_ellipses >= 35,
        "expected >= 35 rings, got {}",
        auto_diag.n_identified_ellipses
    );
    assert!(
        ref_res.n_rings >= 35 && max_ring_idx >= 60,
        "expected refine_image >= 35 rings reaching index >= 60, got {} rings (max idx {max_ring_idx})",
        ref_res.n_rings
    );
    let dp1 = (ref_res.calibration.poni.poni1 - auto_res.poni.poni1).abs() / pixel;
    let dp2 = (ref_res.calibration.poni.poni2 - auto_res.poni.poni2).abs() / pixel;
    assert!(
        dp1.max(dp2) < 10.0,
        "refine_image drifted from auto_calibrate_image by ({dp1:.1}, {dp2:.1}) px"
    );
}

#[test]
fn refine_image_respects_explicit_ring_caps() {
    use dioptrin::calibration::{RefineImageOptions, refine_image};

    let frame = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".into()).expect("load CeO2");
    let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2.D");
    let truth = Poni::load("test_data/v1.poni").expect("load v1.poni");

    let capped_rings = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &truth,
        &cal,
        &RefineImageOptions {
            max_rings: Some(4),
            ..Default::default()
        },
    )
    .expect("capped max_rings=4 should succeed");
    assert!(
        capped_rings.n_rings <= 4,
        "expected <= 4 rings when max_rings=Some(4), got {}",
        capped_rings.n_rings
    );

    let capped_idx = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &truth,
        &cal,
        &RefineImageOptions {
            max_ring_index: 6,
            ..Default::default()
        },
    )
    .expect("capped max_ring_index=6 should succeed");
    let max_idx = capped_idx
        .points
        .ring
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0);
    assert!(
        max_idx < 6,
        "expected all ring indices < 6 when max_ring_index=6, got {max_idx}"
    );
}

#[test]
fn ring_extension_does_not_hallucinate_absent_outer_rings() {
    use dioptrin::calibration::{RefineImageOptions, refine_image};
    use dioptrin::geometry::Geometry;
    use dioptrin::simulation::{RingIntensity, SimulationOptions, simulate_diffraction};

    let full_cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2.D");
    // Render only the first 5 rings into a noisy 1024×1024 image, then refine
    // with the full CeO2 calibrant (15+ rings on detector): extension must not
    // hallucinate phantom outer rings from background noise.
    let five_ring_cal = Calibrant::from_d_spacings("CeO2-5", full_cal.d_spacings[..5].to_vec());
    let truth = Poni {
        pixel1: 172e-6,
        pixel2: 172e-6,
        distance: 0.18,
        poni1: 512.0 * 172e-6,
        poni2: 512.0 * 172e-6,
        rot1: 2.0_f64.to_radians(),
        rot2: -1.5_f64.to_radians(),
        rot3: 0.0,
        wavelength: 0.4066e-10,
        ..Poni::new()
    };
    let geom = Geometry::new(1024, 1024, truth.clone());
    let frame = simulate_diffraction(
        &geom,
        &five_ring_cal,
        &SimulationOptions {
            fwhm_rad: 2.0e-3,
            ring_intensity: RingIntensity::FromCalibrant { fallback: 1_500.0 },
            background: 50.0,
            poisson_noise: true,
            readout_noise: 5.0,
            rng_seed: 909,
            ..Default::default()
        },
    )
    .expect("simulate 5-ring CeO2");

    let res = refine_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        None,
        &truth,
        &full_cal,
        &RefineImageOptions::default(),
    )
    .expect("refine_image on 5-ring CeO2 should succeed");

    assert!(
        res.n_rings <= 6,
        "extension hallucinated phantom outer rings: expected <= 6, got {}",
        res.n_rings
    );
    assert_poni_close(
        "5-ring CeO2 no phantom extension",
        &res.calibration.poni,
        &truth,
        2e-3,
        5.0,
        5e-3,
    );

    // 14-bit saturation (clipped at 16_383, well below 65_000): must auto-detect
    // the flat-topped plateau and still calibrate accurately without explicit intensity_limit.
    let mut sat14 = simulate_diffraction(
        &geom,
        &full_cal,
        &SimulationOptions {
            fwhm_rad: 2.0e-3,
            ring_intensity: RingIntensity::FromCalibrant { fallback: 40_000.0 },
            background: 50.0,
            poisson_noise: true,
            readout_noise: 5.0,
            rng_seed: 910,
            ..Default::default()
        },
    )
    .expect("simulate 14-bit saturated");
    for v in &mut sat14.data {
        if *v > 16_383.0 {
            *v = 16_383.0;
        }
    }
    let res_sat14 = refine_image(
        &sat14.data,
        sat14.dim1,
        sat14.dim2,
        None,
        &truth,
        &full_cal,
        &RefineImageOptions::default(),
    )
    .expect("refine_image on 14-bit saturated image should succeed");
    assert_poni_close(
        "14-bit saturated CeO2",
        &res_sat14.calibration.poni,
        &truth,
        1e-3,
        1.0,
        2e-3,
    );
}
