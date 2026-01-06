use pyo3::prelude::*;

pub mod integrator;
pub mod results;
pub mod utils;

use integrator::PyIntegrator;
use results::{PyBatchIterator, PyBatchResult, PyIntegrationResult};
use utils::{rayon_num_threads, set_rayon_threads};

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PyIntegrationResult>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchIterator>()?;
    Ok(())
}
