use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use std::sync::mpsc::Receiver;
use std::sync::Mutex;

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
    pub radial: Option<Vec<f64>>,
    #[pyo3(get)]
    pub azimuthal: Option<Vec<f64>>,
    #[pyo3(get)]
    pub intensity: Vec<f64>,
    #[pyo3(get)]
    pub errors: Option<Vec<f64>>,
    #[pyo3(get)]
    pub n_radial: Option<usize>,
    #[pyo3(get)]
    pub n_azimuthal: Option<usize>,
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
    pub(crate) fn from_1d(radial: Vec<f64>, intensity: Vec<f64>) -> Self {
        Self {
            radial: Some(radial),
            azimuthal: None,
            intensity,
            errors: None,
            n_radial: None,
            n_azimuthal: None,
        }
    }

    pub(crate) fn from_1d_with_errors(
        radial: Vec<f64>,
        intensity: Vec<f64>,
        errors: Vec<f64>,
    ) -> Self {
        Self {
            radial: Some(radial),
            azimuthal: None,
            intensity,
            errors: Some(errors),
            n_radial: None,
            n_azimuthal: None,
        }
    }

    pub(crate) fn from_2d(intensity: Vec<f64>, n_radial: usize, n_azimuthal: usize) -> Self {
        Self {
            radial: None,
            azimuthal: None,
            intensity,
            errors: None,
            n_radial: Some(n_radial),
            n_azimuthal: Some(n_azimuthal),
        }
    }

    pub(crate) fn from_2d_with_errors(
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
    pub filename: String,
    #[pyo3(get)]
    pub result: Option<PyIntegrationResult>,
    #[pyo3(get)]
    pub error: Option<String>,
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
#[allow(clippy::type_complexity)]
pub(crate) enum BatchReceiver {
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
    pub(crate) receiver: BatchReceiver,
    pub(crate) total: usize,
    pub(crate) completed: usize,
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
