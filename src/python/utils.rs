use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::ThreadPoolBuilder;
use std::thread;

use crate::integration::{IntegrationMethod, RadialUnit};
use crate::poni::Poni;

#[derive(FromPyObject)]
pub enum ImageInput<'p> {
    Path(String),
    Array(PyReadonlyArray2<'p, f64>),
}

#[derive(FromPyObject)]
pub enum BatchImageInput<'p> {
    Paths(Vec<String>),
    Arrays(Vec<PyReadonlyArray2<'p, f64>>),
}

pub fn parse_method(method: Option<&str>, n_ss: Option<usize>) -> Result<IntegrationMethod, PyErr> {
    match method.unwrap_or("pixel_split").to_lowercase().as_str() {
        "pixel_split" | "pixelsplit" | "ps" | "pixelsplitting" => Ok(IntegrationMethod::PixelSplit),
        "counting" | "count" => Ok(IntegrationMethod::Counting),
        "histogram" | "hist" => Ok(IntegrationMethod::Histogram),
        "supersampled" | "supersample" | "ss" => {
            let n_ss = n_ss.unwrap_or(5);
            Ok(IntegrationMethod::Supersampled { n_ss })
        }
        other => Err(PyValueError::new_err(format!(
            "Unknown integration method: {}",
            other
        ))),
    }
}

pub fn parse_unit(unit: Option<&str>) -> Result<RadialUnit, PyErr> {
    match unit.unwrap_or("2th_rad").to_lowercase().as_str() {
        "2th_rad" | "tth_rad" | "2theta_rad" => Ok(RadialUnit::TthRad),
        "2th_deg" | "tth_deg" | "2theta_deg" => Ok(RadialUnit::TthDeg),
        "q_a^-1" | "q_angstrom" | "q_a" => Ok(RadialUnit::QAngstrom),
        "q_nm^-1" | "q_nm" => Ok(RadialUnit::QNm),
        other => Err(PyValueError::new_err(format!(
            "Unknown radial unit: {}. Valid options: 2th_rad, 2th_deg, q_A^-1, q_nm^-1",
            other
        ))),
    }
}

pub fn resolve_workers_threads(
    num_workers: Option<usize>,
    num_threads: Option<usize>,
) -> PyResult<(usize, usize)> {
    let num_threads = num_threads.unwrap_or(2);
    if num_threads == 0 {
        return Err(PyValueError::new_err("num_threads must be > 0"));
    }

    let num_workers = match num_workers {
        Some(workers) => {
            if workers == 0 {
                return Err(PyValueError::new_err("num_workers must be > 0"));
            }
            workers
        }
        None => {
            let cores = thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let max_workers = cores / num_threads;
            let computed = max_workers.saturating_sub(1);
            if computed < 1 {
                1
            } else {
                computed
            }
        }
    };

    Ok((num_workers, num_threads))
}

pub fn extract_array_dims<T>(
    array: &PyReadonlyArray2<T>,
    type_name: &str,
) -> PyResult<(usize, usize)>
where
    T: numpy::Element,
{
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(format!(
            "Array must be C-contiguous (row-major) and {}",
            type_name
        )));
    }
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];
    Ok((dim1, dim2))
}

pub fn poni_from_dict(poni_dict: &Bound<'_, PyDict>) -> PyResult<Poni> {
    let mut poni = Poni::new();

    if let Some(val) = poni_dict.get_item("pixel1")? {
        poni.pixel1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("pixel2")? {
        poni.pixel2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("distance")? {
        poni.distance = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("poni1")? {
        poni.poni1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("poni2")? {
        poni.poni2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot1")? {
        poni.rot1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot2")? {
        poni.rot2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot3")? {
        poni.rot3 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("wavelength")? {
        poni.wavelength = val.extract()?;
    }

    Ok(poni)
}

/// Get the current number of threads used by Rayon for parallel processing.
///
/// # Returns
///
/// Number of threads currently configured for parallel operations
///
/// # Example
///
/// ```python
/// import dioptrin
/// threads = dioptrin.rayon_num_threads()
/// print(f"Using {threads} threads")
/// ```
#[pyfunction]
pub fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Set the number of threads for Rayon parallel processing.
///
/// This affects all parallel operations in the library, including geometry
/// calculations and integration.
///
/// # Arguments
///
/// * `threads` - Number of threads to use (must be > 0)
///
/// # Raises
///
/// ValueError if thread count is invalid or cannot be set
///
/// # Example
///
/// ```python
/// import dioptrin
/// dioptrin.set_rayon_threads(8)  # Use 8 threads
/// ```
#[pyfunction]
#[pyo3(text_signature = "(threads)")]
pub fn set_rayon_threads(threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| PyValueError::new_err(format!("Failed to set Rayon threads: {}", err)))
}
