use pyo3::prelude::*;

pub mod calibration;
pub mod integrator;
pub mod multi_integrator;
pub mod results;
pub mod utils;

use calibration::{
    PyCalibrant, PyCalibrationPoints, PyCalibrationResult, PyEllipseFit, PyRansacEllipseResult,
    find_peaks_massif_py, ransac_ellipses_py, refine_py,
};
use integrator::PyIntegrator;
use multi_integrator::PyMultiIntegrator;
use results::{PyBatchIterator, PyBatchResult, PyIntegrationResult};
use utils::{rayon_num_threads, set_rayon_threads};

pyo3::create_exception!(dioptrin, LicenseError, pyo3::exceptions::PyException);
pyo3::create_exception!(dioptrin, LicenseNotFoundError, LicenseError);
pyo3::create_exception!(dioptrin, LicenseExpiredError, LicenseError);

#[cfg(feature = "licensed")]
pub fn license_error_to_py(e: crate::license::LicenseError) -> PyErr {
    match &e {
        crate::license::LicenseError::NotFound => LicenseNotFoundError::new_err(e.to_string()),
        crate::license::LicenseError::Expired { .. } => LicenseExpiredError::new_err(e.to_string()),
        _ => LicenseError::new_err(e.to_string()),
    }
}

/// Validate the current license. Returns None if valid, raises LicenseError otherwise.
/// In unlicensed builds, always returns None (no check performed).
#[pyfunction]
fn validate_license() -> PyResult<()> {
    #[cfg(feature = "licensed")]
    {
        crate::license::verify_license().map_err(license_error_to_py)?;
    }
    Ok(())
}

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(validate_license, m)?)?;
    m.add_function(wrap_pyfunction!(refine_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks_massif_py, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_ellipses_py, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PyMultiIntegrator>()?;
    m.add_class::<PyIntegrationResult>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchIterator>()?;
    m.add_class::<PyCalibrant>()?;
    m.add_class::<PyCalibrationPoints>()?;
    m.add_class::<PyCalibrationResult>()?;
    m.add_class::<PyEllipseFit>()?;
    m.add_class::<PyRansacEllipseResult>()?;
    m.add("LicenseError", m.py().get_type::<LicenseError>())?;
    m.add(
        "LicenseNotFoundError",
        m.py().get_type::<LicenseNotFoundError>(),
    )?;
    m.add(
        "LicenseExpiredError",
        m.py().get_type::<LicenseExpiredError>(),
    )?;
    Ok(())
}
