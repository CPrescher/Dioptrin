use super::*;

use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::math;
use crate::poni::Poni;
use crate::util;
use rayon::prelude::*;
use std::f64::consts::PI;

// ============================================================================
// RadialUnit conversion tests
// ============================================================================

#[test]
fn test_radial_unit_tth_rad_identity() {
    let tth_rad = vec![0.1, 0.2, 0.3, 0.5, 1.0];
    let wavelength = 1e-10; // 1 Angstrom
    let result = RadialUnit::TthRad.convert_from_tth(&tth_rad, wavelength);
    assert_eq!(result, tth_rad);
}

#[test]
fn test_radial_unit_tth_deg_conversion() {
    let tth_rad = vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];
    let wavelength = 1e-10;
    let result = RadialUnit::TthDeg.convert_from_tth(&tth_rad, wavelength);
    let expected = [0.0, 30.0, 45.0, 60.0, 90.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-10, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_radial_unit_q_angstrom_conversion() {
    // q = 4π sin(θ) / λ, where θ = tth/2
    // For tth = 0, q = 0
    // For tth = π (180°), θ = π/2, sin(θ) = 1, q = 4π/λ
    let wavelength = 1e-10; // 1 Angstrom = 1e-10 m
    let tth_rad = vec![0.0, PI];
    let result = RadialUnit::QAngstrom.convert_from_tth(&tth_rad, wavelength);

    // At tth=0: q = 0
    assert!(result[0].abs() < 1e-10, "q at tth=0 should be 0");

    // At tth=π: q = 4π / 1 Å = 4π Å⁻¹
    let expected_q_max = 4.0 * PI; // ≈ 12.566 Å⁻¹
    assert!(
        (result[1] - expected_q_max).abs() < 1e-10,
        "Expected q={}, got {}",
        expected_q_max,
        result[1]
    );
}

#[test]
fn test_radial_unit_q_nm_conversion() {
    // q = 4π sin(θ) / λ, where θ = tth/2
    let wavelength = 1e-10; // 1 Angstrom = 0.1 nm
    let tth_rad = vec![0.0, PI];
    let result = RadialUnit::QNm.convert_from_tth(&tth_rad, wavelength);

    // At tth=0: q = 0
    assert!(result[0].abs() < 1e-10, "q at tth=0 should be 0");

    // At tth=π: q = 4π / 0.1 nm = 40π nm⁻¹
    let expected_q_max = 4.0 * PI / 0.1; // ≈ 125.66 nm⁻¹
    assert!(
        (result[1] - expected_q_max).abs() < 1e-8,
        "Expected q={}, got {}",
        expected_q_max,
        result[1]
    );
}

#[test]
fn test_radial_unit_q_angstrom_vs_q_nm_ratio() {
    // q in nm⁻¹ should be 10x q in Å⁻¹ (since 1 nm = 10 Å)
    let wavelength = 1.54e-10; // Cu Kα ≈ 1.54 Å
    let tth_rad = vec![0.1, 0.2, 0.5, 1.0];

    let q_angstrom = RadialUnit::QAngstrom.convert_from_tth(&tth_rad, wavelength);
    let q_nm = RadialUnit::QNm.convert_from_tth(&tth_rad, wavelength);

    for (qa, qn) in q_angstrom.iter().zip(q_nm.iter()) {
        let ratio = qn / qa;
        assert!(
            (ratio - 10.0).abs() < 1e-10,
            "q_nm / q_angstrom should be 10, got {}",
            ratio
        );
    }
}

#[test]
fn test_radial_unit_requires_wavelength() {
    assert!(!RadialUnit::TthRad.requires_wavelength());
    assert!(!RadialUnit::TthDeg.requires_wavelength());
    assert!(RadialUnit::QAngstrom.requires_wavelength());
    assert!(RadialUnit::QNm.requires_wavelength());
}

#[test]
#[should_panic(expected = "Wavelength must be positive")]
fn test_radial_unit_q_angstrom_panics_on_zero_wavelength() {
    let tth_rad = vec![0.1];
    RadialUnit::QAngstrom.convert_from_tth(&tth_rad, 0.0);
}

#[test]
#[should_panic(expected = "Wavelength must be positive")]
fn test_radial_unit_q_nm_panics_on_negative_wavelength() {
    let tth_rad = vec![0.1];
    RadialUnit::QNm.convert_from_tth(&tth_rad, -1e-10);
}

#[test]
fn test_integrator() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::Histogram)
        .with_polarization_factor(0.99);
    let (tth, int) = integrator.integrate1d(&frame, 850);

    util::save_vec2("pattern.txt", &tth, &int);

    let tth_pyfai = util::read_vec("pattern_tth_pyFAI.txt").unwrap();
    let int_pyfai = util::read_vec("pattern_int_pyFAI.txt").unwrap();

    util::compare_vecs(&tth, &tth_pyfai, "tth", 1e-6);

    let int_diff: f64 = int
        .par_iter()
        .zip(int_pyfai.par_iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    let int_diff = int_diff / int.len() as f64;
    assert!(int_diff < math::max(&int) * 1e-4); // 0.001 difference in total
}

#[test]
fn test_integrate_counting_1d() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_polarization_factor(0.99);
    let (tth, int) = integrator.integrate1d(&frame, 850);

    util::save_vec2("pattern_lut.txt", &tth, &int);

    let tth_pyfai = util::read_vec("pattern_tth_pyFAI.txt").unwrap();
    let int_pyfai = util::read_vec("pattern_int_pyFAI.txt").unwrap();

    util::compare_vecs(&tth, &tth_pyfai, "tth", 1e-6);

    let int_diff: f64 = int
        .par_iter()
        .zip(int_pyfai.par_iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    let int_diff = int_diff / int.len() as f64;
    assert!(int_diff < math::max(&int) * 1e-4);
}

#[test]
fn test_integrate_counting_1d_with_errors() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_polarization_factor(0.99);

    let (tth, int, err) = integrator.integrate1d_with_errors(&frame, 850);
    util::save_vec3("pattern_lut_err.txt", &tth, &int, &err);
}

#[test]
fn test_integrate_pixel_split_1d() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.99);
    let (tth, int) = integrator.integrate1d(&frame, 850);
    util::save_vec2("pattern_lut_ps.txt", &tth, &int);
}

#[test]
fn test_integrate_pixel_split_1d_with_errors() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.99);

    let (tth, int, err) = integrator.integrate1d_with_errors(&frame, 850);
    util::save_vec3("pattern_lut_ps_err.txt", &tth, &int, &err);
}

#[test]
fn test_integrate_supersampled_1d() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let integrator =
        Integrator::new(geometry).with_method(IntegrationMethod::Supersampled { n_ss: 5 });
    let (tth, int) = integrator.integrate1d(&frame, 850);

    util::save_vec2("pattern_lut_ss.txt", &tth, &int);
}

// ============================================================================
// Integrator radial unit tests
// ============================================================================

#[test]
fn test_integrator_default_unit_is_tth_rad() {
    let poni = Poni::new();
    let geometry = Geometry::new(10, 10, poni);
    let integrator = Integrator::new(geometry);
    assert_eq!(integrator.radial_unit(), RadialUnit::TthRad);
}

#[test]
fn test_integrator_with_radial_unit_builder() {
    let poni = Poni::new();
    let geometry = Geometry::new(10, 10, poni);
    let integrator = Integrator::new(geometry).with_radial_unit(RadialUnit::TthDeg);
    assert_eq!(integrator.radial_unit(), RadialUnit::TthDeg);
}

#[test]
fn test_integrator_set_radial_unit() {
    let mut poni = Poni::new();
    poni.wavelength = 1e-10;
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 1.0;
    let geometry = Geometry::new(10, 10, poni);
    let mut integrator = Integrator::new(geometry);

    assert_eq!(integrator.radial_unit(), RadialUnit::TthRad);

    integrator.set_radial_unit(RadialUnit::QAngstrom);
    assert_eq!(integrator.radial_unit(), RadialUnit::QAngstrom);
}

#[test]
fn test_integrator_tth_deg_output_range() {
    util::download_pilatus_test_file();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();

    // Integrate in radians
    let poni1 = Poni::load("test_data/v1.poni").unwrap();
    let geometry1 = Geometry::new(0, 0, poni1);
    let integrator_rad = Integrator::new(geometry1)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthRad);
    let (x_rad, _) = integrator_rad.integrate1d(&frame, 100);

    // Integrate in degrees
    let poni2 = Poni::load("test_data/v1.poni").unwrap();
    let geometry2 = Geometry::new(0, 0, poni2);
    let integrator_deg = Integrator::new(geometry2)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthDeg);
    let (x_deg, _) = integrator_deg.integrate1d(&frame, 100);

    // Check that degree values are radians * 180/π
    for (r, d) in x_rad.iter().zip(x_deg.iter()) {
        let expected_deg = r.to_degrees();
        assert!(
            (d - expected_deg).abs() < 1e-6,
            "Expected {} deg, got {} deg",
            expected_deg,
            d
        );
    }
}

#[test]
fn test_integrator_q_angstrom_output_range() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let wavelength = poni.wavelength;
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();

    // Integrate in q (Å⁻¹)
    let geometry = Geometry::new(0, 0, poni);
    let integrator_q = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::QAngstrom);
    let (x_q, _) = integrator_q.integrate1d(&frame, 100);

    // Check q values are positive and in expected range
    let wl_angstrom = wavelength * 1e10;
    let q_max_theoretical = 4.0 * PI / wl_angstrom;

    for q in &x_q {
        assert!(*q >= 0.0, "q should be non-negative, got {}", q);
        assert!(
            *q <= q_max_theoretical,
            "q should be <= {}, got {}",
            q_max_theoretical,
            q
        );
    }

    // Verify monotonicity: q should increase with tth for tth < π
    for i in 1..x_q.len() {
        assert!(x_q[i] >= x_q[i - 1], "q should be monotonically increasing");
    }
}

#[test]
fn test_integrator_q_nm_vs_q_angstrom_ratio() {
    util::download_pilatus_test_file();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();

    // Integrate in q (Å⁻¹)
    let poni1 = Poni::load("test_data/v1.poni").unwrap();
    let geometry1 = Geometry::new(0, 0, poni1);
    let integrator_qa = Integrator::new(geometry1)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::QAngstrom);
    let (x_qa, _) = integrator_qa.integrate1d(&frame, 100);

    // Integrate in q (nm⁻¹)
    let poni2 = Poni::load("test_data/v1.poni").unwrap();
    let geometry2 = Geometry::new(0, 0, poni2);
    let integrator_qn = Integrator::new(geometry2)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::QNm);
    let (x_qn, _) = integrator_qn.integrate1d(&frame, 100);

    // q in nm⁻¹ should be 10x q in Å⁻¹
    for (qa, qn) in x_qa.iter().zip(x_qn.iter()) {
        if *qa > 1e-10 {
            // Avoid division by zero for very small values
            let ratio = qn / qa;
            assert!(
                (ratio - 10.0).abs() < 0.01,
                "q_nm / q_angstrom should be ~10, got {}",
                ratio
            );
        }
    }
}

#[test]
fn test_integrator_intensity_invariant_across_units() {
    // The integrated intensity should be approximately the same regardless of radial unit
    // (since we're integrating the same data, just with different x-axis)
    util::download_pilatus_test_file();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();

    let units = [
        RadialUnit::TthRad,
        RadialUnit::TthDeg,
        RadialUnit::QAngstrom,
        RadialUnit::QNm,
    ];

    let mut total_intensities = Vec::new();

    for unit in &units {
        let poni = Poni::load("test_data/v1.poni").unwrap();
        let geometry = Geometry::new(0, 0, poni);
        let integrator = Integrator::new(geometry)
            .with_method(IntegrationMethod::Counting)
            .with_radial_unit(*unit);
        let (_, int) = integrator.integrate1d(&frame, 100);

        // Sum all intensities (excluding NaN)
        let total: f64 = int.iter().filter(|x| !x.is_nan()).sum();
        total_intensities.push(total);
    }

    // All total intensities should be approximately equal
    let reference = total_intensities[0];
    for (i, total) in total_intensities.iter().enumerate() {
        let rel_diff = (total - reference).abs() / reference;
        assert!(
            rel_diff < 0.01,
            "Unit {:?}: total intensity differs by {:.2}%",
            units[i],
            rel_diff * 100.0
        );
    }
}

#[test]
fn test_integrator_pixel_split_with_q_unit() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);

    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::QAngstrom);

    let (x_q, int) = integrator.integrate1d(&frame, 100);

    // Basic sanity checks
    assert_eq!(x_q.len(), 100);
    assert_eq!(int.len(), 100);
    assert!(x_q[0] >= 0.0);
    assert!(x_q[99] > x_q[0]); // q should increase
}

#[test]
fn test_integrator_supersampled_with_q_unit() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);

    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::Supersampled { n_ss: 3 })
        .with_radial_unit(RadialUnit::QNm);

    let (x_q, int) = integrator.integrate1d(&frame, 100);

    // Basic sanity checks
    assert_eq!(x_q.len(), 100);
    assert_eq!(int.len(), 100);
    assert!(x_q[0] >= 0.0);
    assert!(x_q[99] > x_q[0]); // q should increase
}

#[test]
fn test_integrator_unit_change_invalidates_cache() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthRad);

    // First integration in radians
    let (x_rad, _) = integrator.integrate1d(&frame, 50);

    // Change unit to degrees
    integrator.set_radial_unit(RadialUnit::TthDeg);

    // Second integration should be in degrees
    let (x_deg, _) = integrator.integrate1d(&frame, 50);

    // Verify they're different (degrees should be larger values)
    assert!(
        x_deg[25] > x_rad[25] * 50.0,
        "After unit change, x values should be in degrees (larger)"
    );
}

// ============================================================================
// Radial range tests
// ============================================================================

#[test]
fn test_radial_range_builder() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 1.0;
    let geometry = Geometry::new(10, 10, poni);
    let integrator = Integrator::new(geometry).with_radial_range(0.1, 0.5);
    assert_eq!(integrator.radial_range(), Some((0.1, 0.5)));
}

#[test]
fn test_radial_range_setter() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 1.0;
    let geometry = Geometry::new(10, 10, poni);
    let mut integrator = Integrator::new(geometry);
    assert_eq!(integrator.radial_range(), None);
    integrator.set_radial_range(Some((0.1, 0.5)));
    assert_eq!(integrator.radial_range(), Some((0.1, 0.5)));
    integrator.set_radial_range(None);
    assert_eq!(integrator.radial_range(), None);
}

#[test]
fn test_radial_range_controls_bin_grid() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let geometry = Geometry::new(10, 10, poni);

    let frame = Frame {
        name: String::from("test"),
        data: vec![1.0; 100],
        dim1: 10,
        dim2: 10,
    };

    // Without radial range
    let integrator_auto =
        Integrator::new(geometry.clone()).with_method(IntegrationMethod::Counting);
    let (bins_auto, _) = integrator_auto.integrate1d(&frame, 50);

    // With explicit radial range wider than data
    let integrator_range = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 1.0);
    let (bins_range, _) = integrator_range.integrate1d(&frame, 50);

    // The ranged bins should start near 0.0 and end near 1.0
    assert!(
        bins_range[0] < 0.02,
        "First bin should be near 0.0, got {}",
        bins_range[0]
    );
    assert!(
        bins_range[49] > 0.95,
        "Last bin should be near 1.0, got {}",
        bins_range[49]
    );

    // Auto bins should be much narrower (detector only covers small angle)
    assert!(
        bins_auto[49] < bins_range[49],
        "Auto range should be narrower"
    );
}

// ============================================================================
// integrate1d_raw tests
// ============================================================================

#[test]
fn test_integrate1d_raw_matches_integrate1d() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let geometry = Geometry::new(10, 10, poni);

    let frame = Frame {
        name: String::from("test"),
        data: vec![5.0; 100],
        dim1: 10,
        dim2: 10,
    };

    let integrator = Integrator::new(geometry).with_method(IntegrationMethod::Counting);

    let (bins, intensity) = integrator.integrate1d(&frame, 20);
    let raw = integrator.integrate1d_raw(&frame, 20);

    // Bins should match
    assert_eq!(bins.len(), raw.bins.len());
    for (a, b) in bins.iter().zip(raw.bins.iter()) {
        assert!((a - b).abs() < 1e-12, "Bins differ: {} vs {}", a, b);
    }

    // Manual division of raw sums should match integrate1d result
    for (i, &int_val) in intensity.iter().enumerate() {
        let expected = if raw.sum_norm[i] == 0.0 {
            0.0
        } else {
            raw.sum_signal[i] / raw.sum_norm[i]
        };
        assert!(
            (int_val - expected).abs() < 1e-12,
            "Bin {}: integrate1d={}, raw divided={}",
            i,
            int_val,
            expected
        );
    }
}

#[test]
fn test_integrate1d_raw_with_variance_matches_integrate1d_with_errors() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let geometry = Geometry::new(10, 10, poni);

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    let integrator = Integrator::new(geometry).with_method(IntegrationMethod::PixelSplit);

    let (bins, intensity, sigma) = integrator.integrate1d_with_errors(&frame, 20);
    let raw = integrator.integrate1d_raw_with_variance(&frame, 20);

    // Bins should match
    for (a, b) in bins.iter().zip(raw.bins.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // Manual computation from raw should match
    let sum_var = raw.sum_variance.unwrap();
    for (i, ((&int_val, &sig_val), &norm_val)) in intensity
        .iter()
        .zip(sigma.iter())
        .zip(raw.sum_norm.iter())
        .enumerate()
    {
        let expected_int = if norm_val == 0.0 {
            0.0
        } else {
            raw.sum_signal[i] / norm_val
        };
        let expected_sig = if norm_val == 0.0 {
            0.0
        } else {
            (sum_var[i] / (norm_val * norm_val)).max(0.0).sqrt()
        };
        assert!(
            (int_val - expected_int).abs() < 1e-10,
            "Bin {}: intensity mismatch",
            i
        );
        assert!(
            (sig_val - expected_sig).abs() < 1e-10,
            "Bin {}: sigma mismatch",
            i
        );
    }
}

#[test]
#[should_panic(expected = "not supported for Histogram")]
fn test_integrate1d_raw_panics_for_histogram() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let geometry = Geometry::new(10, 10, poni);
    let frame = Frame {
        name: String::from("test"),
        data: vec![1.0; 100],
        dim1: 10,
        dim2: 10,
    };
    let integrator = Integrator::new(geometry).with_method(IntegrationMethod::Histogram);
    let _ = integrator.integrate1d_raw(&frame, 20);
}

// ============================================================================
// MultiIntegrator tests
// ============================================================================

#[test]
fn test_multi_integrator_single_detector_matches_plain() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let geometry = Geometry::new(10, 10, poni.clone());

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    // Plain integrator with radial range
    let plain = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let (bins_plain, int_plain) = plain.integrate1d(&frame, 50);

    // MultiIntegrator with one detector
    let multi_int = Integrator::new(geometry)
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let multi = crate::multi::MultiIntegrator::new(vec![multi_int]);
    let (bins_multi, int_multi) = multi.integrate1d(&[&frame], 50);

    // Results should match exactly
    assert_eq!(bins_plain.len(), bins_multi.len());
    for i in 0..bins_plain.len() {
        assert!(
            (bins_plain[i] - bins_multi[i]).abs() < 1e-12,
            "Bin {} differs: {} vs {}",
            i,
            bins_plain[i],
            bins_multi[i]
        );
        assert!(
            (int_plain[i] - int_multi[i]).abs() < 1e-12,
            "Intensity {} differs: {} vs {}",
            i,
            int_plain[i],
            int_multi[i]
        );
    }
}

#[test]
fn test_multi_integrator_two_identical_detectors() {
    // Two identical detectors with same data should produce same intensity as one
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    // Single detector result
    let (_, int_single) = int1.integrate1d(&frame, 50);

    let int2 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let int3 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    let multi = crate::multi::MultiIntegrator::new(vec![int2, int3]);
    let (_, int_double) = multi.integrate1d(&[&frame, &frame], 50);

    // With identical detectors and data, signal doubles and norm doubles,
    // so intensity = 2*signal / 2*norm = signal/norm = same as single
    for i in 0..50 {
        assert!(
            (int_single[i] - int_double[i]).abs() < 1e-10,
            "Bin {}: single={}, double={}",
            i,
            int_single[i],
            int_double[i]
        );
    }
}

#[test]
fn test_multi_integrator_with_errors() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    let multi = crate::multi::MultiIntegrator::new(vec![int1]);
    let (bins, intensity, sigma) = multi.integrate1d_with_errors(&[&frame], 50);

    assert_eq!(bins.len(), 50);
    assert_eq!(intensity.len(), 50);
    assert_eq!(sigma.len(), 50);

    // Sigma should be non-negative
    for &s in &sigma {
        assert!(s >= 0.0, "Sigma should be non-negative, got {}", s);
    }
}

#[test]
fn test_multi_integrator_from_slices() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;

    let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let frame = Frame {
        name: String::from("test"),
        data: data.clone(),
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let int2 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    let multi = crate::multi::MultiIntegrator::new(vec![int1, int2]);

    // Frame-based
    let (bins_frame, int_frame) = multi.integrate1d(&[&frame, &frame], 50);
    // Slice-based
    let (bins_slice, int_slice) =
        multi.integrate1d_from_slices(&[(&data, 10, 10), (&data, 10, 10)], 50);

    for i in 0..50 {
        assert!(
            (bins_frame[i] - bins_slice[i]).abs() < 1e-12,
            "Bin {} differs",
            i
        );
        assert!(
            (int_frame[i] - int_slice[i]).abs() < 1e-12,
            "Intensity {} differs",
            i
        );
    }
}

#[test]
#[should_panic(expected = "does not support Histogram")]
fn test_multi_integrator_rejects_histogram() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let int1 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Histogram)
        .with_radial_range(0.0, 0.2);
    let _ = crate::multi::MultiIntegrator::new(vec![int1]);
}

#[test]
#[should_panic(expected = "radial_range set")]
fn test_multi_integrator_rejects_missing_radial_range() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let int1 =
        Integrator::new(Geometry::new(10, 10, poni)).with_method(IntegrationMethod::Counting);
    let _ = crate::multi::MultiIntegrator::new(vec![int1]);
}

#[test]
#[should_panic(expected = "same radial_range")]
fn test_multi_integrator_rejects_mismatched_radial_range() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let int2 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.5);
    let _ = crate::multi::MultiIntegrator::new(vec![int1, int2]);
}

#[test]
#[should_panic(expected = "same radial_unit")]
fn test_multi_integrator_rejects_mismatched_units() {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;
    poni.wavelength = 1e-10;
    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthRad)
        .with_radial_range(0.0, 0.2);
    let int2 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthDeg)
        .with_radial_range(0.0, 0.2);
    let _ = crate::multi::MultiIntegrator::new(vec![int1, int2]);
}

#[test]
fn test_multi_integrator_flat_field_uniform_intensity() {
    // Uniform illumination across two detectors at different distances
    // should produce uniform merged intensity
    let mut poni1 = Poni::new();
    poni1.pixel1 = 1.0;
    poni1.pixel2 = 1.0;
    poni1.distance = 100.0;

    let mut poni2 = Poni::new();
    poni2.pixel1 = 1.0;
    poni2.pixel2 = 1.0;
    poni2.distance = 200.0; // different distance

    let frame1 = Frame {
        name: String::from("flat1"),
        data: vec![1.0; 100],
        dim1: 10,
        dim2: 10,
    };
    let frame2 = Frame {
        name: String::from("flat2"),
        data: vec![1.0; 100],
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni1))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.15);
    let int2 = Integrator::new(Geometry::new(10, 10, poni2))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.15);

    let multi = crate::multi::MultiIntegrator::new(vec![int1, int2]);
    let (_, intensity) = multi.integrate1d(&[&frame1, &frame2], 20);

    // All non-zero bins should have similar intensity (uniform flat field)
    let nonzero: Vec<f64> = intensity.iter().copied().filter(|&v| v > 0.0).collect();
    if nonzero.len() > 1 {
        let mean = nonzero.iter().sum::<f64>() / nonzero.len() as f64;
        for &v in &nonzero {
            // With uniform signal and normalization, merged intensity should be consistent
            assert!(
                (v - mean).abs() / mean < 0.5,
                "Flat field not uniform: {} vs mean {}",
                v,
                mean
            );
        }
    }
}

#[test]
fn test_multi_integrator_pixel_split_method() {
    // Verify MultiIntegrator works with pixel-split method
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_range(0.0, 0.2);

    // Single detector with PixelSplit
    let (_, int_single) = int1.integrate1d(&frame, 30);

    let int2 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_range(0.0, 0.2);
    let int3 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_range(0.0, 0.2);

    let multi = crate::multi::MultiIntegrator::new(vec![int2, int3]);
    let (_, int_double) = multi.integrate1d(&[&frame, &frame], 30);

    // Same data, same geometry → same result
    for i in 0..30 {
        assert!(
            (int_single[i] - int_double[i]).abs() < 1e-10,
            "Bin {}: single={}, double={}",
            i,
            int_single[i],
            int_double[i]
        );
    }
}

#[test]
fn test_multi_integrator_error_propagation() {
    // Verify error propagation: two identical detectors should produce
    // smaller errors than one (sqrt(2*var) / (2*norm) < sqrt(var) / norm)
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 100.0;

    let frame = Frame {
        name: String::from("test"),
        data: (1..=100).map(|x| x as f64).collect(),
        dim1: 10,
        dim2: 10,
    };

    let int1 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    let single = crate::multi::MultiIntegrator::new(vec![int1]);
    let (_, _, sigma_single) = single.integrate1d_with_errors(&[&frame], 30);

    let int2 = Integrator::new(Geometry::new(10, 10, poni.clone()))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);
    let int3 = Integrator::new(Geometry::new(10, 10, poni))
        .with_method(IntegrationMethod::Counting)
        .with_radial_range(0.0, 0.2);

    let double = crate::multi::MultiIntegrator::new(vec![int2, int3]);
    let (_, _, sigma_double) = double.integrate1d_with_errors(&[&frame, &frame], 30);

    // With two identical measurements, sigma should decrease by ~1/sqrt(2)
    for i in 0..30 {
        if sigma_single[i] > 0.0 {
            let ratio = sigma_double[i] / sigma_single[i];
            let expected_ratio = 1.0 / 2.0_f64.sqrt();
            assert!(
                (ratio - expected_ratio).abs() < 0.01,
                "Bin {}: error reduction ratio = {}, expected ~{}",
                i,
                ratio,
                expected_ratio
            );
        }
    }
}

#[test]
fn azimuthal_std_preserves_patterns_errors_and_units() {
    let mut poni = Poni::new();
    poni.pixel1 = 1e-4;
    poni.pixel2 = 1e-4;
    poni.distance = 0.03;
    poni.poni1 = 0.001;
    poni.poni2 = 0.001;
    poni.wavelength = 1e-10;
    let frame = Frame {
        name: "synthetic".into(),
        dim1: 20,
        dim2: 20,
        data: (0..400).map(|i| 20.0 + (i % 17) as f64).collect(),
    };
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::Histogram,
    ] {
        for unit in [
            RadialUnit::TthRad,
            RadialUnit::TthDeg,
            RadialUnit::QAngstrom,
            RadialUnit::QNm,
        ] {
            let integrator = Integrator::new(Geometry::new(20, 20, poni.clone()))
                .with_method(method)
                .with_radial_unit(unit)
                .with_polarization_factor(0.8);
            let (radial, intensity, errors) = integrator.integrate1d_with_errors(&frame, 12);
            let result = integrator.integrate1d_with_azimuthal_std(&frame, 12, true);
            assert_eq!(radial, result.radial);
            for (a, b) in intensity.iter().zip(&result.intensity) {
                assert!((a - b).abs() < 1e-10);
            }
            for (a, b) in errors.iter().zip(result.errors.as_ref().unwrap()) {
                assert!((a - b).abs() < 1e-10);
            }
            assert!(
                result
                    .azimuthal_std
                    .iter()
                    .all(|s| s.is_finite() && *s > 0.0)
            );
            let slice = integrator.integrate1d_with_azimuthal_std_from_slice(
                &frame.data,
                20,
                20,
                12,
                false,
            );
            assert_eq!(slice.azimuthal_std, result.azimuthal_std);
            assert!(slice.errors.is_none());
        }
    }
}
#[test]
fn fast_math_toggle_reuses_calibration_and_restores_exact_strict_results() {
    let mut poni = Poni::new();
    poni.pixel1 = 0.001;
    poni.pixel2 = 0.001;
    poni.distance = 0.2;
    let geometry = Geometry::new(8, 9, poni);
    let frame = Frame {
        name: "fast-math".into(),
        dim1: 8,
        dim2: 9,
        data: (0..72)
            .map(|i| 0.1 + (i as f64 * 0.31).sin().abs() * 100.0)
            .collect(),
    };
    let mut ai = Integrator::new(geometry).with_method(IntegrationMethod::Supersampled { n_ss: 3 });
    assert!(ai.fast_math());
    ai.set_fast_math(false);
    let reference = ai.integrate1d_with_errors(&frame, 13);
    let snapshot = {
        let cache = ai.cache_1d.read().unwrap();
        let lut = cache.lut.as_ref().unwrap();
        (
            lut.row_ptr.clone(),
            lut.col_idx.clone(),
            lut.values
                .as_ref()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            lut.col_idx.as_ptr(),
        )
    };
    ai.set_fast_math(true);
    assert!(ai.fast_math());
    let fast = ai.integrate1d_with_errors(&frame, 13);
    assert_eq!(fast.0, reference.0);
    for (a, b) in fast
        .1
        .iter()
        .chain(&fast.2)
        .zip(reference.1.iter().chain(&reference.2))
    {
        assert!((a - b).abs() < 1e-11 * b.abs().max(1.0));
    }
    {
        let cache = ai.cache_1d.read().unwrap();
        let lut = cache.lut.as_ref().unwrap();
        assert_eq!(lut.row_ptr, snapshot.0);
        assert_eq!(lut.col_idx, snapshot.1);
        assert_eq!(
            lut.values
                .as_ref()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            snapshot.2
        );
        assert_eq!(lut.col_idx.as_ptr(), snapshot.3);
    }
    ai.set_fast_math(false);
    let restored = ai.integrate1d_with_errors(&frame, 13);
    for (a, b) in restored
        .0
        .iter()
        .chain(&restored.1)
        .chain(&restored.2)
        .zip(reference.0.iter().chain(&reference.1).chain(&reference.2))
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn prepared_denominators_follow_mask_correction_range_and_unit_changes() {
    let mut poni = Poni::new();
    poni.pixel1 = 0.001;
    poni.pixel2 = 0.001;
    poni.distance = 0.2;
    poni.wavelength = 1e-10;
    let g = Geometry::new(7, 9, poni);
    let data: Vec<_> = (0..63)
        .map(|j| 0.1 + (j as f64 * 0.7).sin() * 100.0)
        .collect();
    let mask: Vec<_> = (0..63).map(|j| u8::from(j % 7 == 0)).collect();
    // PixelSplit's existing atomic scatter can change entry order between
    // independent parallel builds. Fix construction order for this cache test;
    // prepared-kernel tests compare a single table at arbitrary row lengths.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for method in [
                IntegrationMethod::Counting,
                IntegrationMethod::PixelSplit,
                IntegrationMethod::Supersampled { n_ss: 3 },
            ] {
                for (fast, mode) in [false, true]
                    .into_iter()
                    .flat_map(|fast| [MaskMode::Fixed, MaskMode::Dynamic].map(|mode| (fast, mode)))
                {
                    let mut ai = Integrator::new(g.clone())
                        .with_method(method)
                        .with_fast_math(fast)
                        .with_mask_mode(mode);
                    ai.integrate1d_with_errors_from_slice(&data, 7, 9, 11);
                    ai.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4);
                    for step in 0..6 {
                        match step {
                            0 => ai
                                .set_mask(crate::mask::Mask::from_vec(mask.clone(), 7, 9).unwrap()),
                            1 => ai.set_polarization_factor(Some(0.965)),
                            2 => ai.clear_mask(),
                            3 => ai.set_radial_range(Some((0.0, 0.1))),
                            4 => ai.set_radial_unit(RadialUnit::TthDeg),
                            _ => ai.set_polarization_factor(None),
                        }
                        let mut fresh = Integrator::new(g.clone())
                            .with_method(method)
                            .with_fast_math(fast)
                            .with_mask_mode(mode)
                            .with_radial_unit(ai.radial_unit());
                        fresh.set_radial_range(ai.radial_range());
                        fresh.set_polarization_factor(ai.polarization_factor());
                        if ai.mask().is_some() {
                            fresh
                                .set_mask(crate::mask::Mask::from_vec(mask.clone(), 7, 9).unwrap());
                        }
                        let a = ai.integrate1d_with_errors_from_slice(&data, 7, 9, 11);
                        let b = fresh.integrate1d_with_errors_from_slice(&data, 7, 9, 11);
                        for (a, b) in
                            a.0.iter()
                                .chain(&a.1)
                                .chain(&a.2)
                                .zip(b.0.iter().chain(&b.1).chain(&b.2))
                        {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "method={method:?}, fast={fast}, step={step}, a={a}, b={b}"
                            );
                        }
                        let a = ai.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4);
                        let b = fresh.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4);
                        for (a, b) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "method={method:?}, fast={fast}, step={step}, a={a}, b={b}"
                            );
                        }
                    }
                }
            }
        });
}

#[test]
fn concurrent_first_use_prepares_one_shared_denominator_without_blocking_the_pool() {
    use rayon::prelude::*;
    let mut poni = Poni::new();
    poni.pixel1 = 0.001;
    poni.pixel2 = 0.001;
    poni.distance = 0.2;
    let ai = Integrator::new(Geometry::new(7, 9, poni))
        .with_method(IntegrationMethod::Supersampled { n_ss: 3 });
    let data = vec![3.0; 63];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let out: Vec<_> = pool.install(|| {
        (0..8)
            .into_par_iter()
            .map(|_| ai.integrate1d_with_errors_from_slice(&data, 7, 9, 11))
            .collect()
    });
    for x in &out[1..] {
        assert_eq!(x, &out[0]);
    }
}

#[test]
fn polarization_setter_reuses_unchanged_corrections_and_updates_changed_factors() {
    use std::sync::Arc;

    let mut poni = Poni::new();
    poni.pixel1 = 0.01;
    poni.pixel2 = 0.01;
    poni.distance = 0.1;
    poni.wavelength = 1e-10;
    let geometry = Geometry::new(7, 9, poni);
    let data: Vec<_> = (0..63).map(|p| 10.0 + p as f64).collect();
    let mask: Vec<_> = (0..63).map(|p| u8::from(p % 7 == 0)).collect();

    for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
        for fast in [false, true] {
            // Counting has deterministic LUT construction, so results can be
            // compared bit-for-bit against independently built integrators.
            let make = || {
                let mut ai = Integrator::new(geometry.clone())
                    .with_method(IntegrationMethod::Counting)
                    .with_mask_mode(mode)
                    .with_fast_math(fast);
                ai.set_mask(crate::mask::Mask::from_vec(mask.clone(), 7, 9).unwrap());
                ai
            };
            let mut ai = make();
            let mut previous_factor = None;
            let mut previous_normalization = None;
            let mut previous_intensity = None;
            for factor in [None, Some(0.965), Some(0.4), None] {
                ai.set_polarization_factor(factor);
                assert_eq!(ai.polarization_factor(), factor);
                let actual = ai.integrate1d_with_errors_from_slice(&data, 7, 9, 11);
                let cake = ai.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4);
                let normalization = ai.get_normalization();
                let polarization = ai.polarization();

                if factor != previous_factor {
                    if let Some(previous) = &previous_normalization {
                        assert!(!Arc::ptr_eq(&normalization, previous));
                        assert_ne!(normalization.as_ref(), previous.as_ref());
                    }
                    if let Some(previous) = &previous_intensity {
                        assert_ne!(&actual.1, previous);
                    }
                }
                if factor.is_none() {
                    assert!(polarization.iter().all(|&value| value == 1.0));
                }

                // Repeating the setter must keep the same allocated correction
                // storage, not merely recalculate numerically equal values.
                ai.set_polarization_factor(factor);
                assert!(Arc::ptr_eq(&normalization, &ai.get_normalization()));
                assert_eq!(polarization, ai.polarization());
                assert_eq!(
                    actual,
                    ai.integrate1d_with_errors_from_slice(&data, 7, 9, 11)
                );
                assert_eq!(
                    cake,
                    ai.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4)
                );

                let mut fresh = make();
                fresh.set_polarization_factor(factor);
                assert_eq!(
                    actual,
                    fresh.integrate1d_with_errors_from_slice(&data, 7, 9, 11)
                );
                assert_eq!(
                    cake,
                    fresh.integrate2d_with_errors_from_slice(&data, 7, 9, 11, 4)
                );
                previous_factor = factor;
                previous_normalization = Some(normalization);
                previous_intensity = Some(actual.1);
            }
        }
    }
}

/// The outermost pixel can round to bin index `n_points`; it belongs to the last
/// bin. The Histogram method used to index out of bounds here.
#[test]
fn histogram_method_keeps_the_outermost_pixel_in_range() {
    let (rows, cols) = (1043, 981);
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let data = vec![1.0; rows * cols];
    for n in [1000, 1500, 2000, 3000] {
        for unit in [RadialUnit::TthDeg, RadialUnit::QNm] {
            let integrator = Integrator::new(Geometry::new(rows, cols, poni.clone()))
                .with_method(IntegrationMethod::Histogram)
                .with_radial_unit(unit);
            let (x, y) = integrator.integrate1d_from_slice(&data, rows, cols, n);
            assert_eq!((x.len(), y.len()), (n, n));
            let (_, y, sigma) = integrator.integrate1d_with_errors_from_slice(&data, rows, cols, n);
            assert_eq!((y.len(), sigma.len()), (n, n));
        }
    }
}
