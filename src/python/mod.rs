use pyo3::prelude::*;

pub mod calibration;
pub mod dynamic;
pub mod integrator;
pub mod multi_integrator;
pub mod results;
pub mod utils;

use calibration::{
    PyAutoCalibrateResult, PyCalibrant, PyCalibrationPoints, PyCalibrationResult, PyEllipseFit,
    PyRansacEllipseResult, PyRingIdResult, auto_calibrate_py, estimate_ring_center_py,
    find_peaks_image_py, find_peaks_massif_py, identify_rings_py, ransac_ellipses_py, refine_py,
};
use integrator::PyIntegrator;
use multi_integrator::PyMultiIntegrator;
use results::{PyBatchIterator, PyBatchResult, PyIntegrationResult};
use utils::{rayon_num_threads, set_rayon_threads};

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(refine_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks_massif_py, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_ellipses_py, m)?)?;
    m.add_function(wrap_pyfunction!(identify_rings_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(auto_calibrate_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_ring_center_py, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<dynamic::PyDynamic1dPlan>()?;
    m.add_class::<dynamic::PyDynamic1dWorker>()?;
    m.add_class::<PyMultiIntegrator>()?;
    m.add_class::<PyIntegrationResult>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchIterator>()?;
    m.add_class::<PyCalibrant>()?;
    m.add_class::<PyCalibrationPoints>()?;
    m.add_class::<PyCalibrationResult>()?;
    m.add_class::<PyEllipseFit>()?;
    m.add_class::<PyRansacEllipseResult>()?;
    m.add_class::<PyRingIdResult>()?;
    m.add_class::<PyAutoCalibrateResult>()?;
    Ok(())
}
