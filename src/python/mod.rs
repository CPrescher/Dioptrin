use pyo3::prelude::*;

pub mod integrator;
pub mod results;
pub mod utils;

use integrator::PyIntegrator;
use results::{PyBatchIterator, PyBatchResult, PyIntegrationResult};
use utils::{rayon_num_threads, set_rayon_threads};

#[cfg(feature = "licensed")]
pyo3::create_exception!(dioptrin, LicenseError, pyo3::exceptions::PyException);

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PyIntegrationResult>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchIterator>()?;
    #[cfg(feature = "licensed")]
    m.add("LicenseError", m.py().get_type::<LicenseError>())?;
    Ok(())
}
