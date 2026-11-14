use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

/// Result of an integration operation.
///
/// Contains the integrated intensity data along with optional radial/azimuthal
/// coordinates and error estimates, depending on the integration type performed.
///
/// # Attributes
///
/// * `radial` - Radial bin centers (1D and MultiIntegrator 2D; otherwise None)
/// * `azimuthal` - Azimuthal bin centers in radians (MultiIntegrator 2D and Integrator 2D
///   with `azimuthal_range`; otherwise None)
/// * `intensity` - Integrated intensity values
/// * `errors` - Poisson error estimates (present if errors=True was specified)
/// * `azimuthal_std` - Weighted population spread (1D, when requested)
/// * `n_radial` - Number of radial bins (for reshaping 2D data)
/// * `n_azimuthal` - Number of azimuthal bins (for reshaping 2D data)
#[pyclass(name = "IntegrationResult", from_py_object)]
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
    pub azimuthal_std: Option<Vec<f64>>,
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
    pub(crate) fn from_azimuthal(result: crate::lut::AzimuthalIntegrationResult) -> Self {
        Self {
            radial: Some(result.radial),
            azimuthal: None,
            intensity: result.intensity,
            errors: result.errors,
            azimuthal_std: Some(result.azimuthal_std),
            n_radial: None,
            n_azimuthal: None,
        }
    }

    pub(crate) fn from_1d(radial: Vec<f64>, intensity: Vec<f64>) -> Self {
        Self {
            radial: Some(radial),
            azimuthal: None,
            intensity,
            errors: None,
            azimuthal_std: None,
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
            azimuthal_std: None,
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
            azimuthal_std: None,
            n_radial: Some(n_radial),
            n_azimuthal: Some(n_azimuthal),
        }
    }

    /// 2D result on an explicit azimuthal range: fills the azimuthal bin
    /// centres; `radial` stays None as for the default 2D path.
    pub(crate) fn from_azimuthal_range_2d(
        result: crate::integration::AzimuthalRange2dResult,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Self {
        Self {
            radial: None,
            azimuthal: Some(result.azimuthal),
            intensity: result.intensity,
            errors: result.errors,
            azimuthal_std: None,
            n_radial: Some(n_radial),
            n_azimuthal: Some(n_azimuthal),
        }
    }

    /// Batch entry for an explicit-range result: integration errors (invalid
    /// input) become the entry's error.
    pub(crate) fn batch_azimuthal_range_2d(
        filename: String,
        result: Result<AzimuthalRange2dOutput, String>,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> PyBatchResult {
        match result.and_then(|r| r) {
            Ok(r) => PyBatchResult {
                filename,
                result: Some(Self::from_azimuthal_range_2d(r, n_radial, n_azimuthal)),
                error: None,
            },
            Err(e) => PyBatchResult {
                filename,
                result: None,
                error: Some(e),
            },
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
            azimuthal_std: None,
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
#[pyclass(name = "BatchResult", from_py_object)]
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
        if let Some(err) = &self.error {
            format!("BatchResult({}, error={})", self.filename, err)
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
    Integration1DWithAzimuthalStd(
        Mutex<
            Receiver<(
                usize,
                String,
                Result<crate::lut::AzimuthalIntegrationResult, String>,
            )>,
        >,
    ),
    Integration1D(Mutex<Receiver<(usize, String, Result<(Vec<f64>, Vec<f64>), String>)>>),
    Integration1DWithErrors(
        Mutex<
            Receiver<(
                usize,
                String,
                Result<(Vec<f64>, Vec<f64>, Vec<f64>), String>,
            )>,
        >,
    ),
    Integration2D {
        receiver: Mutex<Receiver<(usize, String, Result<Vec<f64>, String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
    },
    Integration2DWithErrors {
        receiver: Mutex<Receiver<(usize, String, Result<(Vec<f64>, Vec<f64>), String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
    },
    /// 2D on an explicit `(min, max)` azimuthal range; carries bin centres.
    Integration2DOnAzimuthalRange {
        receiver: Mutex<Receiver<(usize, String, Result<AzimuthalRange2dOutput, String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
    },
}

/// Output of [`crate::batch::Integration2DOnAzimuthalRange`].
pub(crate) type AzimuthalRange2dOutput = Result<crate::integration::AzimuthalRange2dResult, String>;

/// Iterator for streaming batch integration results.
///
/// Yields BatchResult objects as each file completes processing.
/// Use this for memory-efficient processing of large batches.
///
/// # Example
///
/// ```python
/// for result in integrator.batch1d_iter(files, n_points=1000):
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
    buffer: BTreeMap<usize, PyBatchResult>,
    next_index: usize,
    stream_error: StreamError,
    finished: bool,
}

/// Failure that ends a stream at a given input position: an invalid input
/// item, an exception raised by the caller's iterator, or a worker panic.
///
/// Producers record it instead of sending a result for that position;
/// `__next__` raises it once every earlier result has been delivered.
#[derive(Clone, Default)]
pub(crate) struct StreamError(Arc<Mutex<Option<(usize, PyErr)>>>);

impl StreamError {
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<(usize, PyErr)>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record `err` for input `index`; the earliest position wins.
    pub(crate) fn set(&self, index: usize, err: PyErr) {
        let mut slot = self.slot();
        if slot.as_ref().is_none_or(|(stored, _)| index < *stored) {
            *slot = Some((index, err));
        }
    }

    fn take_at(&self, index: usize) -> Option<PyErr> {
        let mut slot = self.slot();
        if slot.as_ref().is_some_and(|(stored, _)| *stored == index) {
            slot.take().map(|(_, err)| err)
        } else {
            None
        }
    }

    fn take(&self) -> Option<PyErr> {
        self.slot().take().map(|(_, err)| err)
    }
}

/// Text of a caught panic payload.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

fn recv<T: Send>(py: Python<'_>, rx: &Mutex<Receiver<T>>) -> Option<T> {
    py.detach(|| rx.lock().ok().and_then(|guard| guard.recv().ok()))
}

impl PyBatchResult {
    fn from_result(filename: String, result: Result<PyIntegrationResult, String>) -> Self {
        match result {
            Ok(result) => Self {
                filename,
                result: Some(result),
                error: None,
            },
            Err(error) => Self {
                filename,
                result: None,
                error: Some(error),
            },
        }
    }
}

impl PyBatchIterator {
    /// Iterator over `total` results (0 when unknown, e.g. generator input).
    pub(crate) fn new(receiver: BatchReceiver, total: usize) -> Self {
        Self::with_stream_error(receiver, total, StreamError::default())
    }

    /// Like [`Self::new`], raising the failure producers record in `stream_error`.
    pub(crate) fn with_stream_error(
        receiver: BatchReceiver,
        total: usize,
        stream_error: StreamError,
    ) -> Self {
        Self {
            receiver,
            total,
            completed: 0,
            buffer: BTreeMap::new(),
            next_index: 0,
            stream_error,
            finished: false,
        }
    }

    /// The result channel closed: raise a recorded failure, or report results
    /// that never arrived instead of silently ending the iteration.
    fn end_of_stream(&mut self) -> PyErr {
        self.finished = true;
        if let Some(err) = self.stream_error.take() {
            return err;
        }
        if !self.buffer.is_empty() || self.next_index < self.total {
            return PyRuntimeError::new_err(format!(
                "batch ended without a result for item {}: a worker stopped unexpectedly",
                self.next_index
            ));
        }
        PyStopIteration::new_err(())
    }
}

#[pymethods]
impl PyBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<PyBatchResult> {
        if self.finished {
            return Err(PyStopIteration::new_err(()));
        }
        loop {
            // Check if the next expected result is already buffered
            if let Some(result) = self.buffer.remove(&self.next_index) {
                self.next_index += 1;
                self.completed += 1;
                return Ok(result);
            }
            // All earlier results were delivered: raise a failure recorded for
            // this position (bad input item, generator error, worker panic).
            if let Some(err) = self.stream_error.take_at(self.next_index) {
                self.finished = true;
                return Err(err);
            }

            // Receive the next result from the channel
            let received = match &self.receiver {
                BatchReceiver::Integration1DWithAzimuthalStd(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(PyIntegrationResult::from_azimuthal);
                        (index, PyBatchResult::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration1D(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(|(radial, intensity)| {
                            PyIntegrationResult::from_1d(radial, intensity)
                        });
                        (index, PyBatchResult::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration1DWithErrors(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(|(radial, intensity, errors)| {
                            PyIntegrationResult::from_1d_with_errors(radial, intensity, errors)
                        });
                        (index, PyBatchResult::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration2D {
                    receiver,
                    n_radial,
                    n_azimuthal,
                } => recv(py, receiver).map(|(index, filename, result)| {
                    let result = result.map(|intensity| {
                        PyIntegrationResult::from_2d(intensity, *n_radial, *n_azimuthal)
                    });
                    (index, PyBatchResult::from_result(filename, result))
                }),
                BatchReceiver::Integration2DWithErrors {
                    receiver,
                    n_radial,
                    n_azimuthal,
                } => recv(py, receiver).map(|(index, filename, result)| {
                    let result = result.map(|(intensity, errors)| {
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errors,
                            *n_radial,
                            *n_azimuthal,
                        )
                    });
                    (index, PyBatchResult::from_result(filename, result))
                }),
                BatchReceiver::Integration2DOnAzimuthalRange {
                    receiver,
                    n_radial,
                    n_azimuthal,
                } => recv(py, receiver).map(|(index, filename, result)| {
                    (
                        index,
                        PyIntegrationResult::batch_azimuthal_range_2d(
                            filename,
                            result,
                            *n_radial,
                            *n_azimuthal,
                        ),
                    )
                }),
            };
            let Some((index, batch_result)) = received else {
                return Err(self.end_of_stream());
            };

            // If this is the one we're waiting for, yield it directly
            if index == self.next_index {
                self.next_index += 1;
                self.completed += 1;
                return Ok(batch_result);
            }

            // Otherwise buffer it and keep receiving
            self.buffer.insert(index, batch_result);
        }
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
    ///
    /// Returns None when the total is unknown (e.g., generator input).
    #[getter]
    fn progress(&self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some(self.completed as f64 / self.total as f64)
        }
    }
}
