use super::*;

use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::math;
use crate::poni::Poni;
use crate::util;
use rayon::prelude::*;

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
