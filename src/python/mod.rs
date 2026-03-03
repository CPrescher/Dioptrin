use pyo3::prelude::*;

pub mod integrator;
pub mod results;
pub mod utils;

use integrator::PyIntegrator;
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

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PyIntegrationResult>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchIterator>()?;
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
