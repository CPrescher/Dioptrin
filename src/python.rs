use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rayon::ThreadPoolBuilder;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::batch::{
    process_files, process_files_streaming, Integration1D, Integration1DWithErrors, Integration2D,
    Integration2DWithAzimuthalRange, Integration2DWithAzimuthalRangeAndErrors,
    Integration2DWithErrors,
};
use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::integration::{IntegrationMethod, Integrator, RadialUnit};
use crate::poni::Poni;

/// Result of an integration operation.
///
/// Contains the integrated intensity data along with optional radial/azimuthal
/// coordinates and error estimates, depending on the integration type performed.
///
/// # Attributes
///
/// * `radial` - Radial bin centers (present for 1D integration, None for 2D)
/// * `azimuthal` - Azimuthal bin centers (present for 2D integration, None for 1D)
/// * `intensity` - Integrated intensity values
/// * `errors` - Error estimates (present if errors=True was specified)
/// * `n_radial` - Number of radial bins (for reshaping 2D data)
/// * `n_azimuthal` - Number of azimuthal bins (for reshaping 2D data)
#[pyclass(name = "IntegrationResult")]
#[derive(Clone)]
pub struct PyIntegrationResult {
    #[pyo3(get)]
    radial: Option<Vec<f64>>,
    #[pyo3(get)]
    azimuthal: Option<Vec<f64>>,
    #[pyo3(get)]
    intensity: Vec<f64>,
    #[pyo3(get)]
    errors: Option<Vec<f64>>,
    #[pyo3(get)]
    n_radial: Option<usize>,
    #[pyo3(get)]
    n_azimuthal: Option<usize>,
}

#[pymethods]
impl PyIntegrationResult {
    fn __repr__(&self) -> String {
        let kind = if self.n_azimuthal.is_some() {
            "2D"
        } else {
            "1D"
        };
        let errors = if self.errors.is_some() {
            ", with errors"
        } else {
            ""
        };
        format!(
            "IntegrationResult({}, {} points{})",
            kind,
            self.intensity.len(),
            errors
        )
    }
}

impl PyIntegrationResult {
    fn from_1d(radial: Vec<f64>, intensity: Vec<f64>) -> Self {
        Self {
            radial: Some(radial),
            azimuthal: None,
            intensity,
            errors: None,
            n_radial: None,
            n_azimuthal: None,
        }
    }

    fn from_1d_with_errors(radial: Vec<f64>, intensity: Vec<f64>, errors: Vec<f64>) -> Self {
        Self {
            radial: Some(radial),
            azimuthal: None,
            intensity,
            errors: Some(errors),
            n_radial: None,
            n_azimuthal: None,
        }
    }

    fn from_2d(intensity: Vec<f64>, n_radial: usize, n_azimuthal: usize) -> Self {
        Self {
            radial: None,
            azimuthal: None,
            intensity,
            errors: None,
            n_radial: Some(n_radial),
            n_azimuthal: Some(n_azimuthal),
        }
    }

    fn from_2d_with_errors(
        intensity: Vec<f64>,
        errors: Vec<f64>,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Self {
        Self {
            radial: None,
            azimuthal: None,
            intensity,
            errors: Some(errors),
            n_radial: Some(n_radial),
            n_azimuthal: Some(n_azimuthal),
        }
    }
}

/// Result of a single file in a batch integration.
///
/// # Attributes
///
/// * `filename` - Path to the processed file
/// * `result` - IntegrationResult if successful, None if failed
/// * `error` - Error message if failed, None if successful
#[pyclass(name = "BatchResult")]
#[derive(Clone)]
pub struct PyBatchResult {
    #[pyo3(get)]
    filename: String,
    #[pyo3(get)]
    result: Option<PyIntegrationResult>,
    #[pyo3(get)]
    error: Option<String>,
}

#[pymethods]
impl PyBatchResult {
    fn __repr__(&self) -> String {
        if self.error.is_some() {
            format!(
                "BatchResult({}, error={})",
                self.filename,
                self.error.as_ref().unwrap()
            )
        } else {
            format!("BatchResult({}, ok)", self.filename)
        }
    }

    /// Check if the integration was successful.
    fn is_ok(&self) -> bool {
        self.result.is_some()
    }

    /// Check if the integration failed.
    fn is_err(&self) -> bool {
        self.error.is_some()
    }
}

/// Internal enum to hold different receiver types for the iterator
enum BatchReceiver {
    Integration1D(Mutex<Receiver<(String, Result<(Vec<f64>, Vec<f64>), String>)>>),
    Integration1DWithErrors(
        Mutex<Receiver<(String, Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>)>>,
    ),
    Integration2D {
        receiver: Mutex<Receiver<(String, Result<Vec<f64>, String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
    },
    Integration2DWithErrors {
        receiver: Mutex<Receiver<(String, Result<(Vec<f64>, Vec<f64>), String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
    },
}

/// Iterator for streaming batch integration results.
///
/// Yields BatchResult objects as each file completes processing.
/// Use this for memory-efficient processing of large batches.
///
/// # Example
///
/// ```python
/// for result in integrator.batch_integrate1d_iter(files, n_points=1000):
///     if result.is_ok():
///         process(result.result.intensity)
///     else:
///         print(f"Failed: {result.filename}: {result.error}")
/// ```
#[pyclass(name = "BatchIterator")]
pub struct PyBatchIterator {
    receiver: BatchReceiver,
    total: usize,
    completed: usize,
}

#[pymethods]
impl PyBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<PyBatchResult> {
        let batch_result = match &self.receiver {
            BatchReceiver::Integration1D(rx) => {
                let guard = rx
                    .lock()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                match guard.recv() {
                    Ok((filename, result)) => match result {
                        Ok((radial, intensity)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_1d(radial, intensity)),
                            error: None,
                        },
                        Err(e) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(e),
                        },
                    },
                    Err(_) => return Err(PyStopIteration::new_err(())),
                }
            }
            BatchReceiver::Integration1DWithErrors(rx) => {
                let guard = rx
                    .lock()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                match guard.recv() {
                    Ok((filename, result)) => match result {
                        Ok((radial, intensity, errors)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_1d_with_errors(
                                radial, intensity, errors,
                            )),
                            error: None,
                        },
                        Err(e) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(e),
                        },
                    },
                    Err(_) => return Err(PyStopIteration::new_err(())),
                }
            }
            BatchReceiver::Integration2D {
                receiver,
                n_radial,
                n_azimuthal,
            } => {
                let guard = receiver
                    .lock()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                match guard.recv() {
                    Ok((filename, result)) => match result {
                        Ok(intensity) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d(
                                intensity,
                                *n_radial,
                                *n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(e) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(e),
                        },
                    },
                    Err(_) => return Err(PyStopIteration::new_err(())),
                }
            }
            BatchReceiver::Integration2DWithErrors {
                receiver,
                n_radial,
                n_azimuthal,
            } => {
                let guard = receiver
                    .lock()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                match guard.recv() {
                    Ok((filename, result)) => match result {
                        Ok((intensity, errors)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                errors,
                                *n_radial,
                                *n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(e) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(e),
                        },
                    },
                    Err(_) => return Err(PyStopIteration::new_err(())),
                }
            }
        };

        self.completed += 1;
        Ok(batch_result)
    }

    /// Get the total number of files being processed.
    #[getter]
    fn total(&self) -> usize {
        self.total
    }

    /// Get the number of files processed so far.
    #[getter]
    fn completed(&self) -> usize {
        self.completed
    }

    /// Get the progress as a fraction (0.0 to 1.0).
    #[getter]
    fn progress(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.completed as f64 / self.total as f64
        }
    }
}

fn parse_method(method: Option<&str>, n_ss: Option<usize>) -> Result<IntegrationMethod, PyErr> {
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

fn parse_unit(unit: Option<&str>) -> Result<RadialUnit, PyErr> {
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

fn resolve_workers_threads(
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

fn extract_array_dims(array: &PyReadonlyArray2<f64>) -> PyResult<(usize, usize)> {
    let view = array.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "Array must be C-contiguous (row-major) and float64",
        ));
    }
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];
    Ok((dim1, dim2))
}

fn poni_from_dict(poni_dict: &Bound<'_, PyDict>) -> PyResult<Poni> {
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
fn rayon_num_threads() -> usize {
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
fn set_rayon_threads(threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| PyValueError::new_err(format!("Failed to set Rayon threads: {}", err)))
}

#[pyclass(name = "Integrator", unsendable)]
struct PyIntegrator {
    integrator: Arc<Integrator>,
}

#[pymethods]
impl PyIntegrator {
    /// Create a new Integrator instance.
    ///
    /// The integrator is configured with detector geometry from a PONI file.
    /// The dimensions will be determined automatically when the first image
    /// is integrated.
    ///
    /// # Arguments
    ///
    /// * `poni_path` - Path to the PONI calibration file (v1 or v2 format)
    /// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
    /// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Returns
    ///
    /// New Integrator instance
    ///
    /// # Example
    ///
    /// ```python
    /// import dioptrin
    /// integrator = dioptrin.Integrator(
    ///     "calibration.poni",
    ///     method="pixel_split",
    ///     unit="2th_deg"
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (poni_path, method=None, polarization_factor=None, n_ss=None, unit=None))]
    fn new(
        poni_path: &str,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
    ) -> PyResult<Self> {
        let poni = Poni::load(poni_path)
            .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(parse_method(method, n_ss)?)
            .with_radial_unit(parse_unit(unit)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator: Arc::new(integrator),
        })
    }

    /// Create an Integrator from a PONI dictionary.
    ///
    /// The dictionary should contain PONI calibration parameters as keys:
    /// "pixel1", "pixel2", "distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength".
    ///
    /// # Arguments
    ///
    /// * `poni_dict` - Dictionary containing PONI calibration parameters
    /// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
    /// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Returns
    ///
    /// New Integrator instance
    ///
    /// # Example
    ///
    /// ```python
    /// import dioptrin
    /// poni_dict = {
    ///     "pixel1": 0.000172,
    ///     "pixel2": 0.000172,
    ///     "distance": 0.2,
    ///     "poni1": 0.1,
    ///     "poni2": 0.1,
    ///     "wavelength": 1e-10
    /// }
    /// integrator = dioptrin.Integrator.from_poni_dict(poni_dict)
    /// ```
    #[classmethod]
    #[pyo3(signature = (poni_dict, method=None, polarization_factor=None, n_ss=None, unit=None))]
    fn from_poni_dict(
        _cls: &Bound<'_, PyType>,
        poni_dict: &Bound<'_, PyDict>,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
    ) -> PyResult<Self> {
        let poni = poni_from_dict(poni_dict)?;
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(parse_method(method, n_ss)?)
            .with_radial_unit(parse_unit(unit)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator: Arc::new(integrator),
        })
    }

    /// Perform 1D radial integration from a TIFF file.
    ///
    /// # Arguments
    ///
    /// * `tiff_path` - Path to the TIFF image file
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    ///
    /// # Returns
    ///
    /// IntegrationResult with radial values, intensities, and optional errors
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// result = integrator.integrate1d("pattern.tif", n_points=1000)
    /// result = integrator.integrate1d("pattern.tif", n_points=1000, errors=True)
    /// ```
    #[pyo3(signature = (tiff_path, n_points, errors=false))]
    fn integrate1d(
        &self,
        tiff_path: &str,
        n_points: usize,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        if errors {
            let (radial, intensity, errs) =
                self.integrator.integrate1d_with_errors(&frame, n_points);
            Ok(PyIntegrationResult::from_1d_with_errors(
                radial, intensity, errs,
            ))
        } else {
            let (radial, intensity) = self.integrator.integrate1d(&frame, n_points);
            Ok(PyIntegrationResult::from_1d(radial, intensity))
        }
    }

    /// Perform 1D radial integration from a NumPy array.
    ///
    /// Integrates a 2D NumPy array (detector image) along the radial axis.
    /// The array must be C-contiguous (row-major) and of type float64.
    ///
    /// # Arguments
    ///
    /// * `array` - 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    ///
    /// # Returns
    ///
    /// IntegrationResult with radial values, intensities, and optional errors
    ///
    /// # Raises
    ///
    /// ValueError if the array is not C-contiguous or not float64
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// data = np.array(image_data, dtype=np.float64)
    /// result = integrator.integrate1d_from_array(data, n_points=1000)
    /// result = integrator.integrate1d_from_array(data, n_points=1000, errors=True)
    /// ```
    #[pyo3(signature = (array, n_points, errors=false))]
    fn integrate1d_from_array(
        &self,
        array: PyReadonlyArray2<f64>,
        n_points: usize,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let (dim1, dim2) = extract_array_dims(&array)?;
        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        if errors {
            let (radial, intensity, errs) = self
                .integrator
                .integrate1d_with_errors_from_slice(slice, dim1, dim2, n_points);
            Ok(PyIntegrationResult::from_1d_with_errors(
                radial, intensity, errs,
            ))
        } else {
            let (radial, intensity) =
                self.integrator
                    .integrate1d_from_slice(slice, dim1, dim2, n_points);
            Ok(PyIntegrationResult::from_1d(radial, intensity))
        }
    }

    /// Perform 2D integration from a TIFF file.
    ///
    /// Set `errors=True` to include per-bin error estimates. Use `azimuthal_range`
    /// to restrict the azimuthal range.
    ///
    /// # Arguments
    ///
    /// * `tiff_path` - Path to the TIFF image file
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    ///
    /// # Returns
    ///
    /// IntegrationResult with flat intensity (and optional errors) sized
    /// `n_radial * n_azimuthal`
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// result = integrator.integrate2d("pattern.tif", n_radial=100, n_azimuthal=360)
    /// result = integrator.integrate2d(
    ///     "pattern.tif", n_radial=100, n_azimuthal=90, azimuthal_range=(0, 1.57), errors=True
    /// )
    /// ```
    #[pyo3(signature = (tiff_path, n_radial, n_azimuthal, errors=false, azimuthal_range=None))]
    fn integrate2d(
        &self,
        tiff_path: &str,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
    ) -> PyResult<PyIntegrationResult> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;

        let result = match (errors, azimuthal_range) {
            (false, None) => {
                let intensity = self.integrator.integrate2d(&frame, n_radial, n_azimuthal);
                PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
            }
            (true, None) => {
                let (intensity, errs) =
                    self.integrator
                        .integrate2d_with_errors(&frame, n_radial, n_azimuthal);
                PyIntegrationResult::from_2d_with_errors(intensity, errs, n_radial, n_azimuthal)
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
                let intensity = self.integrator.integrate2d_with_azimuthal_range(
                    &frame,
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                );
                PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
                let (intensity, errs) =
                    self.integrator.integrate2d_with_azimuthal_range_and_errors(
                        &frame,
                        n_radial,
                        n_azimuthal,
                        azimuthal_min,
                        azimuthal_max,
                    );
                PyIntegrationResult::from_2d_with_errors(intensity, errs, n_radial, n_azimuthal)
            }
        };

        Ok(result)
    }

    /// Perform 2D integration from a NumPy array.
    ///
    /// Accepts a 2D NumPy array (float64, C-contiguous). Set `errors=True` to include
    /// per-bin error estimates. Use `azimuthal_range` to restrict the azimuthal range.
    ///
    /// # Arguments
    ///
    /// * `array` - 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    ///
    /// # Returns
    ///
    /// IntegrationResult with flat intensity (and optional errors) sized
    /// `n_radial * n_azimuthal`
    ///
    /// # Raises
    ///
    /// ValueError if the array is not C-contiguous or not float64
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// data = np.array(image_data, dtype=np.float64)
    /// result = integrator.integrate2d_from_array(data, n_radial=100, n_azimuthal=360)
    /// ```
    #[pyo3(signature = (array, n_radial, n_azimuthal, errors=false, azimuthal_range=None))]
    fn integrate2d_from_array(
        &self,
        array: PyReadonlyArray2<f64>,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
    ) -> PyResult<PyIntegrationResult> {
        let (dim1, dim2) = extract_array_dims(&array)?;
        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        let result = match (errors, azimuthal_range) {
            (false, None) => {
                let intensity =
                    self.integrator
                        .integrate2d_from_slice(slice, dim1, dim2, n_radial, n_azimuthal);
                PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
            }
            (true, None) => {
                let (intensity, errs) = self.integrator.integrate2d_with_errors_from_slice(
                    slice,
                    dim1,
                    dim2,
                    n_radial,
                    n_azimuthal,
                );
                PyIntegrationResult::from_2d_with_errors(intensity, errs, n_radial, n_azimuthal)
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
                let intensity = self.integrator.integrate2d_with_azimuthal_range_from_slice(
                    slice,
                    dim1,
                    dim2,
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                );
                PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
                let (intensity, errs) =
                    self.integrator
                        .integrate2d_with_azimuthal_range_and_errors_from_slice(
                            slice,
                            dim1,
                            dim2,
                            n_radial,
                            n_azimuthal,
                            azimuthal_min,
                            azimuthal_max,
                        );
                PyIntegrationResult::from_2d_with_errors(intensity, errs, n_radial, n_azimuthal)
            }
        };

        Ok(result)
    }

    /// Set the integration method.
    ///
    /// Changes the integration method used for subsequent integrations.
    /// Available methods: "pixel_split", "counting", "histogram", or "supersampled".
    ///
    /// # Arguments
    ///
    /// * `method` - Integration method name
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (required if method is "supersampled")
    ///
    /// # Raises
    ///
    /// ValueError if the method name is invalid
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_method("supersampled", n_ss=7)
    /// ```
    #[pyo3(signature = (method, n_ss=None))]
    fn set_method(&mut self, method: &str, n_ss: Option<usize>) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_method(parse_method(Some(method), n_ss)?);
        Ok(())
    }

    /// Set the polarization correction factor.
    ///
    /// Applies polarization correction to the integration. The factor should
    /// be between 0.0 and 1.0. Set to None to disable polarization correction.
    ///
    /// # Arguments
    ///
    /// * `polarization_factor` - Polarization factor (0.0 to 1.0) or None to disable
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_polarization_factor(0.99)  # Enable correction
    /// integrator.set_polarization_factor(None)  # Disable correction
    /// ```
    fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_polarization_factor(polarization_factor);
        Ok(())
    }

    /// Set the radial unit for integration output.
    ///
    /// Changes the unit used for the radial (scattering angle) axis in integration results.
    ///
    /// # Arguments
    ///
    /// * `unit` - Radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Raises
    ///
    /// ValueError if the unit name is invalid
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_unit("q_A^-1")  # Switch to q-space in inverse Angstroms
    /// ```
    fn set_unit(&mut self, unit: &str) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_radial_unit(parse_unit(Some(unit))?);
        Ok(())
    }

    /// Get the current radial unit as a string.
    ///
    /// Returns the unit currently configured for the radial axis.
    ///
    /// # Returns
    ///
    /// String representation of the current radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Example
    ///
    /// ```python
    /// unit = integrator.get_unit()
    /// print(f"Current unit: {unit}")
    /// ```
    fn get_unit(&self) -> &'static str {
        match self.integrator.radial_unit() {
            RadialUnit::TthRad => "2th_rad",
            RadialUnit::TthDeg => "2th_deg",
            RadialUnit::QAngstrom => "q_A^-1",
            RadialUnit::QNm => "q_nm^-1",
        }
    }

    /// Perform batch 1D integration.
    ///
    /// This is the recommended batch API. Returns a list of BatchResult objects
    /// with IntegrationResult containing optional error estimates.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename and IntegrationResult.
    ///
    /// # Example
    ///
    /// ```python
    /// results = integrator.batch1d(files, n_points=1000, errors=True)
    /// for r in results:
    ///     if r.is_ok():
    ///         print(r.result.intensity)
    ///         if r.result.errors:
    ///             print(r.result.errors)
    /// ```
    #[pyo3(signature = (tiff_paths, n_points, errors=false, num_workers=None, num_threads=None))]
    fn batch1d(
        &self,
        tiff_paths: Vec<String>,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<PyBatchResult>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        if errors {
            let task = Integration1DWithErrors { n_bins: n_points };
            let results = process_files(
                &tiff_paths,
                Arc::clone(&self.integrator),
                task,
                num_workers,
                num_threads,
            );
            Ok(results
                .into_iter()
                .map(|(filename, result)| match result {
                    Ok((radial, intensity, errs)) => PyBatchResult {
                        filename,
                        result: Some(PyIntegrationResult::from_1d_with_errors(
                            radial, intensity, errs,
                        )),
                        error: None,
                    },
                    Err(err) => PyBatchResult {
                        filename,
                        result: None,
                        error: Some(err),
                    },
                })
                .collect())
        } else {
            let task = Integration1D { n_bins: n_points };
            let results = process_files(
                &tiff_paths,
                Arc::clone(&self.integrator),
                task,
                num_workers,
                num_threads,
            );
            Ok(results
                .into_iter()
                .map(|(filename, result)| match result {
                    Ok((radial, intensity)) => PyBatchResult {
                        filename,
                        result: Some(PyIntegrationResult::from_1d(radial, intensity)),
                        error: None,
                    },
                    Err(err) => PyBatchResult {
                        filename,
                        result: None,
                        error: Some(err),
                    },
                })
                .collect())
        }
    }

    /// Perform streaming batch 1D integration.
    ///
    /// Returns an iterator that yields results as each file completes.
    /// More memory-efficient for large batches.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// BatchIterator that yields BatchResult objects as they complete.
    /// The iterator has `total`, `completed`, and `progress` properties.
    ///
    /// # Example
    ///
    /// ```python
    /// it = integrator.batch1d_iter(files, n_points=1000)
    /// for result in it:
    ///     print(f"Progress: {it.completed}/{it.total}")
    ///     if result.is_ok():
    ///         process(result.result)
    /// ```
    #[pyo3(signature = (tiff_paths, n_points, errors=false, num_workers=None, num_threads=None))]
    fn batch1d_iter(
        &self,
        tiff_paths: Vec<String>,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<PyBatchIterator> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let total = tiff_paths.len();

        if errors {
            let task = Integration1DWithErrors { n_bins: n_points };
            let batch = process_files_streaming(
                &tiff_paths,
                Arc::clone(&self.integrator),
                task,
                num_workers,
                num_threads,
            );
            Ok(PyBatchIterator {
                receiver: BatchReceiver::Integration1DWithErrors(Mutex::new(batch.into_receiver())),
                total,
                completed: 0,
            })
        } else {
            let task = Integration1D { n_bins: n_points };
            let batch = process_files_streaming(
                &tiff_paths,
                Arc::clone(&self.integrator),
                task,
                num_workers,
                num_threads,
            );
            Ok(PyBatchIterator {
                receiver: BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                total,
                completed: 0,
            })
        }
    }

    /// Perform batch 2D integration with unified result type.
    ///
    /// This is the recommended batch API for 2D integration. Returns a list of
    /// BatchResult objects with IntegrationResult containing optional error estimates.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename and IntegrationResult.
    ///
    /// # Example
    ///
    /// ```python
    /// # Full 2D integration
    /// results = integrator.batch2d(files, n_radial=100, n_azimuthal=360)
    ///
    /// # With azimuthal range and errors
    /// results = integrator.batch2d(
    ///     files, n_radial=100, n_azimuthal=90,
    ///     azimuthal_range=(0, 1.57),  # 0 to pi/2
    ///     errors=True
    /// )
    /// ```
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, errors=false, azimuthal_range=None, num_workers=None, num_threads=None))]
    fn batch2d(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<PyBatchResult>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match (errors, azimuthal_range) {
            (false, None) => {
                let task = Integration2D {
                    n_radial,
                    n_azimuthal,
                };
                let results = process_files(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok(intensity) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d(
                                intensity,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (true, None) => {
                let task = Integration2DWithErrors {
                    n_radial,
                    n_azimuthal,
                };
                let results = process_files(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok((intensity, errs)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                errs,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let results = process_files(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok(intensity) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d(
                                intensity,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRangeAndErrors {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let results = process_files(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok((intensity, errs)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                errs,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
        }
    }

    /// Perform streaming batch 2D integration.
    ///
    /// Returns an iterator that yields results as each file completes.
    /// More memory-efficient for large batches.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// BatchIterator that yields BatchResult objects as they complete.
    /// The iterator has `total`, `completed`, and `progress` properties.
    ///
    /// # Example
    ///
    /// ```python
    /// for result in integrator.batch2d_iter(files, n_radial=100, n_azimuthal=360):
    ///     if result.is_ok():
    ///         cake = result.result.intensity
    /// ```
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, errors=false, azimuthal_range=None, num_workers=None, num_threads=None))]
    fn batch2d_iter(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<PyBatchIterator> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let total = tiff_paths.len();

        match (errors, azimuthal_range) {
            (false, None) => {
                let task = Integration2D {
                    n_radial,
                    n_azimuthal,
                };
                let batch = process_files_streaming(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                })
            }
            (true, None) => {
                let task = Integration2DWithErrors {
                    n_radial,
                    n_azimuthal,
                };
                let batch = process_files_streaming(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                })
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let batch = process_files_streaming(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                })
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRangeAndErrors {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let batch = process_files_streaming(
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                })
            }
        }
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
    Ok(())
}
