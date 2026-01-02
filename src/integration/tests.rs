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
    let expected = vec![0.0, 30.0, 45.0, 60.0, 90.0];
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
    let mut integrator = Integrator::new(geometry)
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
    let mut integrator = Integrator::new(geometry)
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
    assert!(int_diff < math::max(&int) * 1e-4); // 0.001 difference in total
}

#[test]
fn test_integrate_counting_1d_with_errors() {
    util::download_pilatus_test_file();
    let poni: Poni = Poni::load("test_data/v1.poni").unwrap();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();
    let geometry = Geometry::new(0, 0, poni);
    let mut integrator = Integrator::new(geometry)
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
    let mut integrator = Integrator::new(geometry)
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
    let mut integrator = Integrator::new(geometry)
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
    let mut integrator =
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
    let mut integrator_rad = Integrator::new(geometry1)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::TthRad);
    let (x_rad, _) = integrator_rad.integrate1d(&frame, 100);

    // Integrate in degrees
    let poni2 = Poni::load("test_data/v1.poni").unwrap();
    let geometry2 = Geometry::new(0, 0, poni2);
    let mut integrator_deg = Integrator::new(geometry2)
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
    let mut integrator_q = Integrator::new(geometry)
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
        assert!(
            x_q[i] >= x_q[i - 1],
            "q should be monotonically increasing"
        );
    }
}

#[test]
fn test_integrator_q_nm_vs_q_angstrom_ratio() {
    util::download_pilatus_test_file();
    let frame = Frame::from_tiff(String::from("test_data/CeO2_Pilatus1M.tif")).unwrap();

    // Integrate in q (Å⁻¹)
    let poni1 = Poni::load("test_data/v1.poni").unwrap();
    let geometry1 = Geometry::new(0, 0, poni1);
    let mut integrator_qa = Integrator::new(geometry1)
        .with_method(IntegrationMethod::Counting)
        .with_radial_unit(RadialUnit::QAngstrom);
    let (x_qa, _) = integrator_qa.integrate1d(&frame, 100);

    // Integrate in q (nm⁻¹)
    let poni2 = Poni::load("test_data/v1.poni").unwrap();
    let geometry2 = Geometry::new(0, 0, poni2);
    let mut integrator_qn = Integrator::new(geometry2)
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
        let mut integrator = Integrator::new(geometry)
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

    let mut integrator = Integrator::new(geometry)
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

    let mut integrator = Integrator::new(geometry)
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
