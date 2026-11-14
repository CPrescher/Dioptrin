use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::ThreadPoolBuilder;
use std::thread;

use crate::frame::Frame;
use crate::integration::{IntegrationMethod, Integrator, RadialUnit};
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
        "pyfai_full" | "pyfai" => Ok(IntegrationMethod::PyfaiFull),
        "counting" | "count" => Ok(IntegrationMethod::Counting),
        "histogram" | "hist" => Ok(IntegrationMethod::Histogram),
        "supersampled" | "supersample" | "ss" => {
            let n_ss = n_ss.unwrap_or(5);
            if n_ss == 0 {
                return Err(PyValueError::new_err("n_ss must be positive"));
            }
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
            if computed < 1 { 1 } else { computed }
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
    if dim1 == 0 || dim2 == 0 {
        return Err(PyValueError::new_err(format!(
            "Array must be non-empty, got shape ({dim1}, {dim2})"
        )));
    }
    Ok((dim1, dim2))
}

/// Raise ValueError unless a bin count is positive (e.g. "n_points must be positive").
pub fn check_positive(name: &str, value: usize) -> PyResult<()> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be positive")));
    }
    Ok(())
}

/// Shape of the integrator's mask, if one is set.
pub fn mask_shape(integrator: &Integrator) -> Option<(usize, usize)> {
    integrator.mask().map(|mask| (mask.dim1, mask.dim2))
}

/// Raise ValueError when an image does not match the mask shape; core
/// integration asserts on it. `label` names the image, e.g. "images[2]".
pub fn check_mask_shape(
    mask: Option<(usize, usize)>,
    image: (usize, usize),
    label: &str,
) -> PyResult<()> {
    match mask {
        Some(mask) if mask != image => Err(PyValueError::new_err(format!(
            "{label} shape {image:?} does not match mask shape {mask:?}"
        ))),
        _ => Ok(()),
    }
}

/// Validate Integrator options that core accepts unchecked (and would later
/// panic on or silently misuse).
pub fn validate_integrator_options(
    poni: &Poni,
    unit: RadialUnit,
    polarization_factor: Option<f64>,
    radial_range: Option<(f64, f64)>,
) -> PyResult<()> {
    check_unit_wavelength(unit, poni.wavelength)?;
    check_polarization_factor(polarization_factor)?;
    if let Some((min, max)) = radial_range
        && !(min.is_finite() && max.is_finite() && min < max)
    {
        return Err(PyValueError::new_err(format!(
            "radial_range must be finite with min < max, got ({min}, {max})"
        )));
    }
    Ok(())
}

/// Histogram always bins over the automatic radial range of the image, so a
/// `radial_range` would be silently ignored; reject the combination instead.
pub fn check_method_radial_range(
    method: &IntegrationMethod,
    radial_range: Option<(f64, f64)>,
) -> PyResult<()> {
    if matches!(method, IntegrationMethod::Histogram) && radial_range.is_some() {
        return Err(PyValueError::new_err(
            "radial_range is not supported by the 'histogram' method, which always \
             uses the automatic radial range; use 'counting', 'pixel_split', \
             'pyfai_full' or 'supersampled' for an explicit radial_range",
        ));
    }
    Ok(())
}

/// q units convert with the PONI wavelength, which must be positive.
pub fn check_unit_wavelength(unit: RadialUnit, wavelength: f64) -> PyResult<()> {
    if unit.requires_wavelength() && !(wavelength.is_finite() && wavelength > 0.0) {
        return Err(PyValueError::new_err(format!(
            "q units require a positive wavelength in the PONI calibration, got {wavelength}"
        )));
    }
    Ok(())
}

/// The polarization factor must be finite and within [-1, 1].
pub fn check_polarization_factor(factor: Option<f64>) -> PyResult<()> {
    if let Some(factor) = factor
        && !(factor.is_finite() && (-1.0..=1.0).contains(&factor))
    {
        return Err(PyValueError::new_err(format!(
            "polarization_factor must be finite and in [-1, 1], got {factor}"
        )));
    }
    Ok(())
}

/// [`extract_array_dims`] for float64 batch item `index`, naming it in errors.
pub fn extract_item_dims(array: &PyReadonlyArray2<f64>, index: usize) -> PyResult<(usize, usize)> {
    extract_array_dims(array, "float64")
        .map_err(|err| PyValueError::new_err(format!("images[{index}]: {}", err.value(array.py()))))
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

/// Classified input for batch operations.
///
/// Tries extraction in order: Vec<String> (paths), Vec<Array> (numpy),
/// then falls back to Python iterator.
pub enum ClassifiedInput<'py> {
    Paths(Vec<String>),
    Arrays(Vec<PyReadonlyArray2<'py, f64>>),
    Iterator(Bound<'py, PyAny>),
}

/// Classify a Python object as paths, arrays, or an iterator.
///
/// Preserves backward compatibility: lists of strings or arrays match first.
/// Other lists/tuples (e.g. mixing paths and arrays) are checked item by item
/// here, so an invalid item raises before any work starts instead of ending
/// the stream; generators/iterators fall through to the Iterator variant and
/// are checked as they are consumed.
pub fn classify_batch_input<'py>(obj: &Bound<'py, PyAny>) -> PyResult<ClassifiedInput<'py>> {
    // Try Vec<String> first
    if let Ok(paths) = obj.extract::<Vec<String>>() {
        return Ok(ClassifiedInput::Paths(paths));
    }
    // Try Vec<PyReadonlyArray2<f64>>
    if let Ok(arrays) = obj.extract::<Vec<PyReadonlyArray2<'py, f64>>>() {
        return Ok(ClassifiedInput::Arrays(arrays));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        for (index, item) in obj.try_iter()?.enumerate() {
            check_batch_item(&item?, index)?;
        }
    }
    // Fall back to iterator protocol
    let iter = obj.call_method0("__iter__").map_err(|_| {
        PyValueError::new_err(
            "images must be a list of file paths, a list of numpy arrays, or an iterable/generator",
        )
    })?;
    Ok(ClassifiedInput::Iterator(iter))
}

/// Describe a rejected batch item, e.g. "numpy array with dtype int32 and shape (4, 4)".
fn describe_item(obj: &Bound<'_, PyAny>) -> String {
    let type_name = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "object".to_owned());
    match (obj.getattr("dtype"), obj.getattr("shape")) {
        (Ok(dtype), Ok(shape)) => format!("{type_name} with dtype {dtype} and shape {shape}"),
        _ => type_name,
    }
}

fn item_type_error(obj: &Bound<'_, PyAny>, index: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "images[{}] must be a file path (str) or a 2D float64 numpy array, got {}",
        index,
        describe_item(obj)
    ))
}

/// Check that a batch item is a path or a 2D C-contiguous float64 array.
fn check_batch_item(obj: &Bound<'_, PyAny>, index: usize) -> PyResult<()> {
    if obj.extract::<String>().is_ok() {
        return Ok(());
    }
    let array: PyReadonlyArray2<'_, f64> =
        obj.extract().map_err(|_| item_type_error(obj, index))?;
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(format!(
            "images[{}] must be C-contiguous (row-major)",
            index
        )));
    }
    if view.is_empty() {
        return Err(PyValueError::new_err(format!(
            "images[{}] must be non-empty, got shape {:?}",
            index,
            view.shape()
        )));
    }
    Ok(())
}

/// Extract a Frame from a Python object (str path or numpy array).
pub fn extract_frame_from_pyany(obj: &Bound<'_, PyAny>, index: usize) -> PyResult<FrameOrPath> {
    // Try string first (file path)
    if let Ok(path) = obj.extract::<String>() {
        return Ok(FrameOrPath::Path(path));
    }
    // Try numpy array
    let array: PyReadonlyArray2<'_, f64> =
        obj.extract().map_err(|_| item_type_error(obj, index))?;
    let (dim1, dim2) = extract_item_dims(&array, index)?;
    let slice = array.as_slice().map_err(|err| {
        PyValueError::new_err(format!("images[{}] must be contiguous: {}", index, err))
    })?;
    Ok(FrameOrPath::Frame(Frame {
        name: format!("iter[{}]", index),
        data: slice.to_vec(),
        dim1,
        dim2,
    }))
}

/// Result of extracting a frame from a Python object.
pub enum FrameOrPath {
    Path(String),
    Frame(Frame),
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
