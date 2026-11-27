use std::time::Instant;

use dioptrin::calibration::{
    AutoCalibrateError, AutoCalibrateImageOptions, AutoCalibrateStage, Calibrant,
    auto_calibrate_image, auto_calibrate_image_with_progress,
};
use dioptrin::geometry::Geometry;
use dioptrin::poni::Poni;
use dioptrin::simulation::{
    PeakProfile, RingIntensity, SimulationOptions, SpottyOptions, simulate_diffraction,
};

struct Case {
    name: &'static str,
    calibrant_path: &'static str,
    dim1: usize,
    dim2: usize,
    poni: Poni,
    sim: SimulationOptions,
    /// Optional synthetic beamstop radius in pixels around the beam center.
    beamstop_px: Option<f64>,
    /// Optional number of bright random single-crystal outlier spots.
    n_outliers: usize,
}

#[allow(clippy::too_many_arguments)]
fn make_poni(
    pixel1: f64,
    pixel2: f64,
    distance: f64,
    poni1_px: f64,
    poni2_px: f64,
    rot1_deg: f64,
    rot2_deg: f64,
    wavelength_a: f64,
) -> Poni {
    Poni {
        pixel1,
        pixel2,
        distance,
        poni1: (poni1_px + 0.5) * pixel1,
        poni2: (poni2_px + 0.5) * pixel2,
        rot1: rot1_deg.to_radians(),
        rot2: rot2_deg.to_radians(),
        rot3: 0.0,
        spline_file: String::new(),
        wavelength: wavelength_a * 1e-10,
    }
}

fn sim_realistic() -> SimulationOptions {
    SimulationOptions {
        profile: PeakProfile::PseudoVoigt { eta: 0.3 },
        fwhm_rad: 2.0e-3,
        i_max: 1000.0,
        background: 50.0,
        ring_intensity: RingIntensity::FromCalibrant { fallback: 0.1 },
        spotty: None,
        oversampling: 1,
        apply_solid_angle: true,
        polarization_factor: Some(0.95),
        mask_value: Some(0.0),
        poisson_noise: true,
        readout_noise: 3.0,
        rng_seed: 12345,
    }
}

fn case(
    name: &'static str,
    calibrant_path: &'static str,
    dim1: usize,
    dim2: usize,
    poni: Poni,
    sim: SimulationOptions,
) -> Case {
    Case {
        name,
        calibrant_path,
        dim1,
        dim2,
        poni,
        sim,
        beamstop_px: None,
        n_outliers: 0,
    }
}

fn add_outliers(data: &mut [f64], dim1: usize, dim2: usize, n_outliers: usize, seed: u64) {
    let mut s = seed;
    let mut next_f = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
    };
    for _ in 0..n_outliers {
        let r0 = 8.0 + next_f() * (dim1 - 16) as f64;
        let c0 = 8.0 + next_f() * (dim2 - 16) as f64;
        let amp = 3000.0 + 5000.0 * next_f();
        let sig = 1.6;
        let i0 = (r0 as isize - 6).max(0) as usize;
        let i1 = (r0 as isize + 6).min(dim1 as isize - 1) as usize;
        let j0 = (c0 as isize - 6).max(0) as usize;
        let j1 = (c0 as isize + 6).min(dim2 as isize - 1) as usize;
        for i in i0..=i1 {
            let dy = i as f64 - r0;
            for j in j0..=j1 {
                let dx = j as f64 - c0;
                data[i * dim2 + j] += amp * (-(dy * dy + dx * dx) / (2.0 * sig * sig)).exp();
            }
        }
    }
}

fn add_beamstop(data: &mut [f64], dim1: usize, dim2: usize, poni: &Poni, radius_px: f64) {
    let bc1 = poni.poni1 / poni.pixel1 - 0.5;
    let bc2 = poni.poni2 / poni.pixel2 - 0.5;
    let r2 = radius_px * radius_px;
    for i in 0..dim1 {
        let dy = i as f64 - bc1;
        for j in 0..dim2 {
            let dx = j as f64 - bc2;
            if dy * dy + dx * dx <= r2 || (dx.abs() < 6.0 && dy > 0.0) {
                data[i * dim2 + j] = 0.0;
            }
        }
    }
}

fn assert_case_passes(c: &Case, starting_distance: Option<f64>) {
    let cal = Calibrant::from_d_file(c.calibrant_path).expect("load cal");
    let geom = Geometry::new(c.dim1, c.dim2, c.poni.clone());
    let mut frame = simulate_diffraction(&geom, &cal, &c.sim).expect("simulate");
    if c.n_outliers > 0 {
        add_outliers(&mut frame.data, frame.dim1, frame.dim2, c.n_outliers, 777);
    }
    if let Some(r_bs) = c.beamstop_px {
        add_beamstop(&mut frame.data, frame.dim1, frame.dim2, &c.poni, r_bs);
    }
    let opts = AutoCalibrateImageOptions {
        starting_distance,
        ..Default::default()
    };
    let (res, _diag) = auto_calibrate_image(
        &frame.data,
        frame.dim1,
        frame.dim2,
        &cal,
        c.poni.wavelength,
        c.poni.pixel1,
        c.poni.pixel2,
        &opts,
    )
    .unwrap_or_else(|e| panic!("{} (d0={starting_distance:?}) failed: {e}", c.name));

    let rel_d = (res.poni.distance - c.poni.distance).abs() / c.poni.distance;
    let dp1 = (res.poni.poni1 - c.poni.poni1).abs() / c.poni.pixel1;
    let dp2 = (res.poni.poni2 - c.poni.poni2).abs() / c.poni.pixel2;
    let drot = (res.poni.rot1 - c.poni.rot1)
        .abs()
        .max((res.poni.rot2 - c.poni.rot2).abs())
        .to_degrees();
    assert!(
        rel_d < 0.01 && dp1.max(dp2) < 10.0 && drot < 0.5,
        "{} (d0={starting_distance:?}) inaccurate: rel_d={:.3}%, dp=({dp1:.2}, {dp2:.2})px, drot={drot:.3}°",
        c.name,
        rel_d * 100.0
    );
}

#[test]
fn auto_calibrate_image_simulated_edge_cases() {
    let ceo2 = "test_data/calibrants/CeO2.D";
    let lab6 = "test_data/calibrants/LaB6_SRM660b.D";

    // Wavelength extremes (Cu Kα and 80 keV)
    assert_case_passes(
        &case(
            "CuKa_1.54A_short_dist",
            lab6,
            512,
            512,
            make_poni(172e-6, 172e-6, 0.06, 256.0, 256.0, 2.0, -1.0, 1.5406),
            SimulationOptions {
                fwhm_rad: 4.0e-3,
                ..sim_realistic()
            },
        ),
        None,
    );
    assert_case_passes(
        &case(
            "80keV_0.155A_lab6_dense",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.35, 256.0, 256.0, 1.5, -1.0, 0.155),
            SimulationOptions {
                fwhm_rad: 8.0e-4,
                ..sim_realistic()
            },
        ),
        None,
    );

    // Sharp and very broad peak widths + Lorentzian profile
    assert_case_passes(
        &case(
            "sharp_0.3mrad_oversamp2",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 260.0, 250.0, 2.0, -1.5, 0.31),
            SimulationOptions {
                fwhm_rad: 3.0e-4,
                oversampling: 2,
                ..sim_realistic()
            },
        ),
        None,
    );
    assert_case_passes(
        &case(
            "broad_12mrad",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.15, 256.0, 256.0, 2.0, -1.5, 0.37),
            SimulationOptions {
                fwhm_rad: 1.2e-2,
                ..sim_realistic()
            },
        ),
        None,
    );
    assert_case_passes(
        &case(
            "lorentzian_5mrad",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 2.0, -1.5, 0.37),
            SimulationOptions {
                profile: PeakProfile::Lorentzian,
                fwhm_rad: 5.0e-3,
                ..sim_realistic()
            },
        ),
        None,
    );

    // Off-detector beam position
    assert_case_passes(
        &case(
            "off_detector_side_100px",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.30, 256.0, -100.0, 2.0, -1.0, 0.31),
            sim_realistic(),
        ),
        None,
    );

    // Heavy tilts: both PONI-on-detector (side mount) and beam-centered (PONI shifted +839 px)
    assert_case_passes(
        &case(
            "tilt_25deg_poni_on_det",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 256.0, 256.0, 25.0, 0.0, 0.31),
            sim_realistic(),
        ),
        None,
    );
    assert_case_passes(
        &case(
            "tilt_25deg_beam_centered",
            ceo2,
            512,
            512,
            make_poni(
                100e-6,
                100e-6,
                0.18,
                256.0,
                256.0 + (0.18 * 25.0_f64.to_radians().tan()) / 100e-6,
                25.0,
                0.0,
                0.31,
            ),
            sim_realistic(),
        ),
        None,
    );

    // Tilted strip detector (with rough distance hint)
    assert_case_passes(
        &case(
            "strip_200x800_tilted_20deg",
            ceo2,
            200,
            800,
            make_poni(172e-6, 172e-6, 0.25, 100.0, 400.0, -20.0, 3.0, 0.31),
            sim_realistic(),
        ),
        Some(0.2875),
    );

    // Spotty coarse-grain powder + beamstop + single-crystal outliers
    let mut c_spotty = case(
        "spotty_24_plus_beamstop_and_outliers",
        ceo2,
        512,
        512,
        make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 2.5, -2.0, 0.31),
        SimulationOptions {
            spotty: Some(SpottyOptions {
                spots_per_ring: 24,
                azimuthal_fwhm_rad: 0.025,
                continuous_fraction: 0.0,
            }),
            ..sim_realistic()
        },
    );
    c_spotty.beamstop_px = Some(35.0);
    c_spotty.n_outliers = 25;
    assert_case_passes(&c_spotty, None);

    // Sparse ambiguous 12-spot LaB6 must not return a false D/√2 alias
    let cal_lab6 = Calibrant::from_d_file(lab6).expect("load lab6");
    let poni_sparse = make_poni(100e-6, 100e-6, 0.18, 250.0, 260.0, 3.0, -2.0, 0.31);
    let geom_sparse = Geometry::new(512, 512, poni_sparse.clone());
    let frame_sparse = simulate_diffraction(
        &geom_sparse,
        &cal_lab6,
        &SimulationOptions {
            spotty: Some(SpottyOptions {
                spots_per_ring: 12,
                azimuthal_fwhm_rad: 0.02,
                continuous_fraction: 0.0,
            }),
            ..sim_realistic()
        },
    )
    .expect("simulate sparse");
    if let Ok((res, _)) = auto_calibrate_image(
        &frame_sparse.data,
        512,
        512,
        &cal_lab6,
        poni_sparse.wavelength,
        poni_sparse.pixel1,
        poni_sparse.pixel2,
        &AutoCalibrateImageOptions::default(),
    ) {
        let rel_d = (res.poni.distance - poni_sparse.distance).abs() / poni_sparse.distance;
        assert!(
            rel_d < 0.02,
            "sparse LaB6 returned aliased distance {} vs {}",
            res.poni.distance,
            poni_sparse.distance
        );
    }
}

#[test]
fn auto_calibrate_progress_cancellation_and_auto_saturation() {
    let ceo2 = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
    let poni = make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 2.0, -1.5, 0.31);
    let geom = Geometry::new(512, 512, poni.clone());
    let mut frame = simulate_diffraction(&geom, &ceo2, &sim_realistic()).expect("simulate");

    // Inject flat-topped 16-bit CCD saturation blocks (65535) with NO explicit intensity_limit.
    for i in 120..145 {
        for j in 120..145 {
            frame.data[i * 512 + j] = 65535.0;
        }
    }

    let mut stages = Vec::new();
    let mut saw_peaks = false;
    let mut saw_center = false;
    let mut saw_candidate_poni = false;
    let (res, _diag) = auto_calibrate_image_with_progress(
        &frame.data,
        512,
        512,
        None,
        &ceo2,
        poni.wavelength,
        poni.pixel1,
        poni.pixel2,
        &AutoCalibrateImageOptions::default(),
        |p| {
            stages.push(p.stage);
            if p.peaks.as_ref().is_some_and(|v| !v.is_empty()) {
                saw_peaks = true;
            }
            if p.ring_center_px.is_some() {
                saw_center = true;
            }
            if p.candidate_poni.is_some() {
                saw_candidate_poni = true;
            }
            true
        },
    )
    .expect("auto-calibration should succeed with auto-detected saturation");

    assert!(saw_peaks, "expected preview peaks in progress callback");
    assert!(saw_center, "expected ring center in progress callback");
    assert!(
        saw_candidate_poni,
        "expected intermediate candidate PONI in progress callback"
    );
    assert_eq!(stages.first(), Some(&AutoCalibrateStage::DetectingPeaks));
    assert_eq!(stages.last(), Some(&AutoCalibrateStage::Done));
    let rel_d = (res.poni.distance - poni.distance).abs() / poni.distance;
    assert!(rel_d < 0.005, "distance err {rel_d}");

    // Returning `false` cancels cleanly with AutoCalibrateError::Cancelled.
    let err = auto_calibrate_image_with_progress(
        &frame.data,
        512,
        512,
        None,
        &ceo2,
        poni.wavelength,
        poni.pixel1,
        poni.pixel2,
        &AutoCalibrateImageOptions::default(),
        |_| false,
    )
    .unwrap_err();
    assert!(matches!(err, AutoCalibrateError::Cancelled));
}

fn build_full_matrix() -> Vec<Case> {
    let ceo2 = "test_data/calibrants/CeO2.D";
    let lab6 = "test_data/calibrants/LaB6_SRM660b.D";

    let mut cases = vec![
        case(
            "01_default_sim_zero_bg_ceo2",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.15, 256.0, 256.0, 2.0, -1.5, 0.31),
            SimulationOptions::default(),
        ),
        case(
            "02_realistic_centered_lab6_40keV",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 260.0, 245.0, 3.0, -2.0, 0.31),
            sim_realistic(),
        ),
        case(
            "03_wl_CuKa_1.54A_short_dist",
            lab6,
            512,
            512,
            make_poni(172e-6, 172e-6, 0.06, 256.0, 256.0, 2.0, -1.0, 1.5406),
            SimulationOptions {
                fwhm_rad: 4.0e-3,
                ..sim_realistic()
            },
        ),
        case(
            "04_wl_MoKa_0.71A_med_dist",
            ceo2,
            512,
            512,
            make_poni(172e-6, 172e-6, 0.12, 250.0, 265.0, 3.0, 2.0, 0.7107),
            sim_realistic(),
        ),
        case(
            "05_wl_60keV_0.207A_ceo2",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.25, 256.0, 256.0, 2.0, -2.0, 0.2066),
            SimulationOptions {
                fwhm_rad: 1.2e-3,
                ..sim_realistic()
            },
        ),
        case(
            "06_wl_80keV_0.155A_lab6_dense",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.35, 256.0, 256.0, 1.5, -1.0, 0.155),
            SimulationOptions {
                fwhm_rad: 8.0e-4,
                ..sim_realistic()
            },
        ),
        case(
            "07_sharp_peaks_0.3mrad_oversamp2",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 260.0, 250.0, 2.0, -1.5, 0.31),
            SimulationOptions {
                fwhm_rad: 3.0e-4,
                oversampling: 2,
                ..sim_realistic()
            },
        ),
        case(
            "08_broad_peaks_6mrad",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 260.0, 250.0, 2.0, -1.5, 0.31),
            SimulationOptions {
                fwhm_rad: 6.0e-3,
                ..sim_realistic()
            },
        ),
        case(
            "09_very_broad_peaks_12mrad",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.15, 256.0, 256.0, 2.0, -1.5, 0.37),
            SimulationOptions {
                fwhm_rad: 1.2e-2,
                ..sim_realistic()
            },
        ),
        case(
            "10_ultra_broad_18mrad_long_px",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.12, 256.0, 256.0, 2.0, -1.5, 0.50),
            SimulationOptions {
                fwhm_rad: 1.8e-2,
                ..sim_realistic()
            },
        ),
        case(
            "11_lorentzian_broad_5mrad",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 2.0, -1.5, 0.37),
            SimulationOptions {
                profile: PeakProfile::Lorentzian,
                fwhm_rad: 5.0e-3,
                ..sim_realistic()
            },
        ),
        case(
            "12_corner_beam_on_detector",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.25, 40.0, 45.0, 2.0, -1.5, 0.31),
            sim_realistic(),
        ),
        case(
            "13_off_detector_side_100px",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.30, 256.0, -100.0, 2.0, -1.0, 0.31),
            sim_realistic(),
        ),
        case(
            "14_off_detector_corner_150px",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.35, -150.0, -150.0, 1.5, -1.5, 0.31),
            sim_realistic(),
        ),
        case(
            "15_tilt_10deg_centered",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 10.0, 0.0, 0.31),
            sim_realistic(),
        ),
        case(
            "16_tilt_15deg_combined",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 12.0, -10.0, 0.31),
            sim_realistic(),
        ),
        case(
            "17_tilt_25deg_single_axis",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 256.0, 256.0, 25.0, 0.0, 0.31),
            sim_realistic(),
        ),
        case(
            "18_tilt_35deg_poni_on_det",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 256.0, 256.0, 35.0, -5.0, 0.31),
            sim_realistic(),
        ),
        case(
            "19_tilt_25deg_beam_centered",
            ceo2,
            512,
            512,
            make_poni(
                100e-6,
                100e-6,
                0.18,
                256.0,
                256.0 + (0.18 * 25.0_f64.to_radians().tan()) / 100e-6,
                25.0,
                0.0,
                0.31,
            ),
            sim_realistic(),
        ),
        case(
            "20_short_dist_0.06m_wide_angle",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.06, 256.0, 256.0, 2.0, -1.5, 0.4133),
            SimulationOptions {
                fwhm_rad: 3.0e-3,
                ..sim_realistic()
            },
        ),
        case(
            "21_long_dist_0.80m_1024px_60keV",
            lab6,
            1024,
            1024,
            make_poni(172e-6, 172e-6, 0.80, 512.0, 512.0, 1.0, -1.0, 0.2066),
            SimulationOptions {
                fwhm_rad: 8.0e-4,
                ..sim_realistic()
            },
        ),
        case(
            "22_low_snr_noisy",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 260.0, 245.0, 2.5, -1.5, 0.31),
            SimulationOptions {
                i_max: 60.0,
                background: 200.0,
                poisson_noise: true,
                readout_noise: 15.0,
                ..sim_realistic()
            },
        ),
        case(
            "23_low_count_photon_counting",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.16, 260.0, 245.0, 2.5, -1.5, 0.31),
            SimulationOptions {
                i_max: 25.0,
                background: 1.0,
                poisson_noise: true,
                readout_noise: 0.0,
                ..sim_realistic()
            },
        ),
        case(
            "24_strip_200x800_tilted_20deg",
            ceo2,
            200,
            800,
            make_poni(172e-6, 172e-6, 0.25, 100.0, 400.0, -20.0, 3.0, 0.31),
            sim_realistic(),
        ),
        case(
            "25_rect_nonsquare_px_tilted_12deg",
            ceo2,
            400,
            700,
            make_poni(75e-6, 150e-6, 0.20, 210.0, 340.0, 12.0, -4.0, 0.31),
            sim_realistic(),
        ),
        case(
            "26_spotty_pure_24_spots_ceo2",
            ceo2,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.15, 260.0, 245.0, 2.5, -1.5, 0.31),
            SimulationOptions {
                spotty: Some(SpottyOptions {
                    spots_per_ring: 24,
                    azimuthal_fwhm_rad: 0.025,
                    continuous_fraction: 0.0,
                }),
                ..sim_realistic()
            },
        ),
        case(
            "27_spotty_sparse_12_spots_lab6",
            lab6,
            512,
            512,
            make_poni(100e-6, 100e-6, 0.18, 250.0, 260.0, 3.0, -2.0, 0.31),
            SimulationOptions {
                spotty: Some(SpottyOptions {
                    spots_per_ring: 12,
                    azimuthal_fwhm_rad: 0.02,
                    continuous_fraction: 0.0,
                }),
                ..sim_realistic()
            },
        ),
        case(
            "28_grainy_tilted_20deg_beam_ctr",
            ceo2,
            512,
            512,
            make_poni(
                100e-6,
                100e-6,
                0.16,
                256.0,
                256.0 + (0.16 * 20.0_f64.to_radians().tan()) / 100e-6,
                20.0,
                0.0,
                0.31,
            ),
            SimulationOptions {
                spotty: Some(SpottyOptions {
                    spots_per_ring: 30,
                    azimuthal_fwhm_rad: 0.03,
                    continuous_fraction: 0.15,
                }),
                ..sim_realistic()
            },
        ),
    ];

    let mut c_bs = case(
        "29_beamstop_plus_30_dac_outliers",
        ceo2,
        512,
        512,
        make_poni(100e-6, 100e-6, 0.16, 256.0, 256.0, 2.5, -2.0, 0.31),
        SimulationOptions {
            spotty: Some(SpottyOptions {
                spots_per_ring: 36,
                azimuthal_fwhm_rad: 0.03,
                continuous_fraction: 0.25,
            }),
            ..sim_realistic()
        },
    );
    c_bs.beamstop_px = Some(35.0);
    c_bs.n_outliers = 30;
    cases.push(c_bs);

    cases
}

/// Full 29-case diagnostic matrix (blind + hinted). Run manually with:
/// `cargo test --test auto_calibration_stress stress_matrix_report -- --ignored --nocapture`
#[test]
#[ignore = "slow diagnostic stress matrix (~35s); run with --ignored --nocapture"]
fn stress_matrix_report() {
    let cases = build_full_matrix();
    let mut blind_pass = 0usize;
    let mut hinted_pass = 0usize;

    println!(
        "\n{:<40} | {:<36} | {:<36}",
        "CASE", "BLIND (d0=None)", "HINTED (d0=1.15*d)"
    );
    println!("{}", "-".repeat(118));

    for c in &cases {
        let cal = Calibrant::from_d_file(c.calibrant_path).expect("load cal");
        let geom = Geometry::new(c.dim1, c.dim2, c.poni.clone());
        let mut frame = simulate_diffraction(&geom, &cal, &c.sim).expect("simulate");
        if c.n_outliers > 0 {
            add_outliers(&mut frame.data, frame.dim1, frame.dim2, c.n_outliers, 777);
        }
        if let Some(r_bs) = c.beamstop_px {
            add_beamstop(&mut frame.data, frame.dim1, frame.dim2, &c.poni, r_bs);
        }

        let mut results = Vec::new();
        for (idx, d0) in [None, Some(c.poni.distance * 1.15)].into_iter().enumerate() {
            let opts = AutoCalibrateImageOptions {
                starting_distance: d0,
                ..Default::default()
            };
            let t0 = Instant::now();
            let res = auto_calibrate_image(
                &frame.data,
                frame.dim1,
                frame.dim2,
                &cal,
                c.poni.wavelength,
                c.poni.pixel1,
                c.poni.pixel2,
                &opts,
            );
            let ms = t0.elapsed().as_millis();
            match res {
                Ok((cal_res, _diag)) => {
                    let rel_d = (cal_res.poni.distance - c.poni.distance).abs() / c.poni.distance;
                    let dp1 = (cal_res.poni.poni1 - c.poni.poni1).abs() / c.poni.pixel1;
                    let dp2 = (cal_res.poni.poni2 - c.poni.poni2).abs() / c.poni.pixel2;
                    let drot = (cal_res.poni.rot1 - c.poni.rot1)
                        .abs()
                        .max((cal_res.poni.rot2 - c.poni.rot2).abs())
                        .to_degrees();
                    let ok = rel_d < 0.01 && dp1.max(dp2) < 12.0 && drot < 0.5;
                    if ok {
                        if idx == 0 {
                            blind_pass += 1;
                        } else {
                            hinted_pass += 1;
                        }
                    }
                    let tag = if ok { "OK  " } else { "WRNG" };
                    results.push(format!(
                        "{tag} d={:.2}% p={:.1}px r={:.2}° {ms}ms",
                        rel_d * 100.0,
                        dp1.max(dp2),
                        drot
                    ));
                }
                Err(e) => {
                    results.push(format!("ERR  ({e:?}) {ms}ms"));
                }
            }
        }
        println!("{:<40} | {:<36} | {:<36}", c.name, results[0], results[1]);
    }
    println!("{}", "-".repeat(118));
    println!(
        "SUMMARY: blind {blind_pass}/{} passed, hinted {hinted_pass}/{} passed\n",
        cases.len(),
        cases.len()
    );
}
