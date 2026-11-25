use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::npyffi::NPY_ARRAY_WRITEABLE;
use numpy::{PyArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use crate::batch::StreamControl;

/// Result of an integration operation.
///
/// Contains the integrated intensity data along with optional radial/azimuthal
/// coordinates and error estimates, depending on the integration type performed.
///
/// # Attributes
///
/// * `radial` - Radial bin centers (1D, MultiIntegrator 2D, and Integrator 2D with a
///   `radial_range` or an `azimuthal_range`; otherwise None)
/// * `azimuthal` - Azimuthal bin centers in radians (MultiIntegrator 2D and Integrator 2D
///   with `azimuthal_range`; otherwise None)
/// * `intensity` - Integrated intensity values
/// * `errors` - Poisson error estimates (present if errors=True was specified)
/// * `azimuthal_std` - Weighted population spread (1D, when requested)
/// * `n_radial` - Number of radial bins (for reshaping 2D data)
/// * `n_azimuthal` - Number of azimuthal bins (for reshaping 2D data)
///
/// Each vector also has a `*_array` property (`intensity_array`, ...) that
/// returns a read-only float64 NumPy view of the same data without copying;
/// 2D `intensity_array`/`errors_array` have shape `(n_azimuthal, n_radial)`.
//
// `frozen`: once the Python object exists, Rust code reaches the fields only
// through `&self`, so the vectors are never mutated or reallocated while it is
// alive. The `*_array` views rely on this; do not add setters or `&mut self`
// methods. (Extraction via `from_py_object` clones, leaving the object intact.)
#[pyclass(name = "IntegrationResult", frozen, from_py_object)]
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

/// Which vector of a [`PyIntegrationResult`] a NumPy view borrows.
#[derive(Clone, Copy)]
enum Field {
    Radial,
    Azimuthal,
    Intensity,
    Errors,
    AzimuthalStd,
}

impl PyIntegrationResult {
    fn field(&self, field: Field) -> Option<&[f64]> {
        match field {
            Field::Radial => self.radial.as_deref(),
            Field::Azimuthal => self.azimuthal.as_deref(),
            Field::Intensity => Some(&self.intensity),
            Field::Errors => self.errors.as_deref(),
            Field::AzimuthalStd => self.azimuthal_std.as_deref(),
        }
    }

    /// Read-only NumPy view of `field`, with `slf` as its base (so the view
    /// keeps the result alive).
    ///
    /// `as_2d` views 2D intensity/errors as `(n_azimuthal, n_radial)`, the
    /// layout the integrators write (index `azimuthal_bin * n_radial + radial_bin`).
    fn view<'py>(
        slf: &Bound<'py, Self>,
        field: Field,
        as_2d: bool,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let this = slf.get();
        let Some(data) = this.field(field) else {
            return Ok(None);
        };
        let owner = slf.clone().into_any();
        // SAFETY (borrow_from_array): the array points into the heap buffer of
        // a vector owned by `slf` and holds a reference to `slf` as its base,
        // so the buffer outlives the array. The class is frozen and has no
        // setters, so the vector is never mutated or reallocated once the
        // Python object exists.
        // SAFETY (flags): the array was just created and nothing else refers
        // to it yet; clearing WRITEABLE makes it read-only, and NumPy refuses
        // to set the flag again because the base exposes no writable buffer.
        let array = match (as_2d, this.n_radial, this.n_azimuthal) {
            (true, Some(n_radial), Some(n_azimuthal)) => {
                let view = ArrayView2::from_shape((n_azimuthal, n_radial), data).map_err(|e| {
                    PyValueError::new_err(format!(
                        "2D result of length {} does not match ({n_azimuthal}, {n_radial}): {e}",
                        data.len()
                    ))
                })?;
                let array = unsafe { PyArray::borrow_from_array(&view, owner) };
                unsafe { (*array.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE };
                array.into_any()
            }
            _ => {
                let view = ArrayView1::from(data);
                let array = unsafe { PyArray::borrow_from_array(&view, owner) };
                unsafe { (*array.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE };
                array.into_any()
            }
        };
        Ok(Some(array))
    }
}

#[pymethods]
impl PyIntegrationResult {
    /// `radial` as a read-only float64 NumPy view (no copy), or None.
    #[getter]
    fn radial_array<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::view(slf, Field::Radial, false)
    }

    /// `azimuthal` as a read-only float64 NumPy view (no copy), or None.
    #[getter]
    fn azimuthal_array<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::view(slf, Field::Azimuthal, false)
    }

    /// `intensity` as a read-only float64 NumPy view (no copy): shape `(n,)`
    /// for 1D, `(n_azimuthal, n_radial)` for 2D.
    #[getter]
    fn intensity_array<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::view(slf, Field::Intensity, true)
    }

    /// `errors` as a read-only float64 NumPy view (no copy), shaped like
    /// `intensity_array`, or None.
    #[getter]
    fn errors_array<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::view(slf, Field::Errors, true)
    }

    /// `azimuthal_std` as a read-only float64 NumPy view (no copy), or None.
    #[getter]
    fn azimuthal_std_array<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::view(slf, Field::AzimuthalStd, false)
    }

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

    /// Attach radial bin centres (2D results carry them only with an explicit
    /// radial or azimuthal range).
    pub(crate) fn with_radial(mut self, radial: Option<Vec<f64>>) -> Self {
        if radial.is_some() {
            self.radial = radial;
        }
        self
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

    /// 2D result on an explicit azimuthal range: fills the radial and
    /// azimuthal bin centres, which this path computes anyway.
    pub(crate) fn from_azimuthal_range_2d(
        result: crate::integration::AzimuthalRange2dResult,
        n_radial: usize,
        n_azimuthal: usize,
    ) -> Self {
        Self {
            radial: Some(result.radial),
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
    ) -> BatchEntry {
        match result.and_then(|r| r) {
            Ok(r) => BatchEntry {
                filename,
                result: Some(Self::from_azimuthal_range_2d(r, n_radial, n_azimuthal)),
                error: None,
            },
            Err(e) => BatchEntry {
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

/// Rust-side batch outcome, built off the interpreter by the batch paths and
/// turned into a Python [`PyBatchResult`] when it is handed to Python.
pub struct BatchEntry {
    pub filename: String,
    pub result: Option<PyIntegrationResult>,
    pub error: Option<String>,
}

impl<'py> IntoPyObject<'py> for BatchEntry {
    type Target = PyBatchResult;
    type Output = Bound<'py, PyBatchResult>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let result = self.result.map(|r| Py::new(py, r)).transpose()?;
        Bound::new(
            py,
            PyBatchResult {
                filename: self.filename,
                result,
                error: self.error,
            },
        )
    }
}

/// Result of a single file in a batch integration.
///
/// # Attributes
///
/// * `filename` - Path to the processed file
/// * `result` - IntegrationResult if successful, None if failed (the same
///   object on every read; no data is copied)
/// * `error` - Error message if failed, None if successful
#[pyclass(name = "BatchResult", frozen)]
pub struct PyBatchResult {
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub result: Option<Py<PyIntegrationResult>>,
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
        /// Radial bin centres for an explicit radial range.
        radial: Option<Vec<f64>>,
    },
    Integration2DWithErrors {
        receiver: Mutex<Receiver<(usize, String, Result<(Vec<f64>, Vec<f64>), String>)>>,
        n_radial: usize,
        n_azimuthal: usize,
        /// Radial bin centres for an explicit radial range.
        radial: Option<Vec<f64>>,
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
/// Fed by a list of arrays or a Python iterator, the stream reads at most
/// `4 * num_workers` items ahead of the results it has handed out; arrays from
/// a list are checked when the stream is opened and copied when the stream
/// reaches them. The integrator counts as in use while the stream runs. Once
/// the iterator has raised `StopIteration` or an error, it is free. Dropping
/// the iterator mid-way:
///
/// - fed by a list of arrays or a Python iterator: stops the workers and
///   waits for them, so the integrator is free when the drop returns. A
///   producer thread blocked in the caller's `__next__` stays there until
///   that call returns, holding neither the integrator nor the workers.
/// - over a list of TIFF paths: not waited for; each worker finishes the file
///   it is on, then stops.
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
    /// None once the stream is closed (finished or dropped).
    receiver: Option<BatchReceiver>,
    total: usize,
    completed: usize,
    buffer: BTreeMap<usize, BatchEntry>,
    next_index: usize,
    stream_error: StreamError,
    finished: bool,
    /// Present for queue-fed streams (array input to `batch2d_iter` and
    /// Python iterator input): lets the iterator stop its workers.
    control: Option<Arc<StreamControl>>,
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

impl BatchEntry {
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
            receiver: Some(receiver),
            total,
            completed: 0,
            buffer: BTreeMap::new(),
            next_index: 0,
            stream_error,
            finished: false,
            control: None,
        }
    }

    /// Iterator over a queue-fed stream run by `control`'s workers.
    pub(crate) fn with_control(
        receiver: BatchReceiver,
        total: usize,
        stream_error: StreamError,
        control: Arc<StreamControl>,
    ) -> Self {
        let mut iterator = Self::with_stream_error(receiver, total, stream_error);
        iterator.control = Some(control);
        iterator
    }

    /// Hand out the result for `next_index` and grant the producer credit
    /// for one more item.
    fn deliver(&mut self, result: BatchEntry) -> BatchEntry {
        self.next_index += 1;
        self.completed += 1;
        if let Some(control) = &self.control {
            control.deliver(self.next_index);
        }
        result
    }

    /// Stop the stream: no more items are pulled from the input, results
    /// still in flight are discarded, and for a queue-fed stream (a list of
    /// arrays or an iterator) wait (detached from the interpreter) until its workers have released the
    /// integrator, so its setters work as soon as the stream is closed.
    ///
    /// Workers never wait for Python or for this consumer, so the wait lasts
    /// at most one poll interval or the integration of one item. The
    /// producer thread is not waited for: it may be inside the caller's
    /// `__next__` (a generator waiting for a detector frame), which cannot be
    /// interrupted. It holds neither the integrator nor the workers and exits
    /// when that call returns.
    fn close(&mut self, py: Python<'_>) {
        self.finished = true;
        self.receiver = None;
        self.buffer.clear();
        let Some(control) = &self.control else {
            return;
        };
        control.cancel();
        py.detach(|| control.wait_workers_idle());
    }

    /// The result channel closed: raise a recorded failure, or report results
    /// that never arrived instead of silently ending the iteration.
    fn end_of_stream(&mut self, py: Python<'_>) -> PyErr {
        let missing = !self.buffer.is_empty() || self.next_index < self.total;
        self.close(py);
        if let Some(err) = self.stream_error.take() {
            return err;
        }
        if missing {
            return PyRuntimeError::new_err(format!(
                "batch ended without a result for item {}: a worker stopped unexpectedly",
                self.next_index
            ));
        }
        // Only a producer that saw the input end makes this a clean end of
        // the stream (the total of iterator input is unknown).
        if let Some(control) = &self.control
            && !control.input_exhausted()
        {
            return PyRuntimeError::new_err(format!(
                "batch ended without a result for item {}: \
                 the thread reading the input stopped unexpectedly",
                self.next_index
            ));
        }
        PyStopIteration::new_err(())
    }
}

impl Drop for PyBatchIterator {
    fn drop(&mut self) {
        // Only Python drops this value: from the object's deallocator, or
        // from pyo3's wrapper when a call fails to wrap the returned value.
        // Both run on an attached thread (also during interpreter shutdown),
        // so `attach` reuses that attachment and `close` can detach to wait.
        if self.finished && self.receiver.is_none() {
            return;
        }
        Python::attach(|py| self.close(py));
    }
}

#[pymethods]
impl PyBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<BatchEntry> {
        if self.finished {
            return Err(PyStopIteration::new_err(()));
        }
        loop {
            // Check if the next expected result is already buffered
            if let Some(result) = self.buffer.remove(&self.next_index) {
                return Ok(self.deliver(result));
            }
            // All earlier results were delivered: raise a failure recorded for
            // this position (bad input item, generator error, worker panic).
            if let Some(err) = self.stream_error.take_at(self.next_index) {
                self.close(py);
                return Err(err);
            }

            // Receive the next result from the channel
            let Some(receiver) = &self.receiver else {
                return Err(PyStopIteration::new_err(()));
            };
            let received = match receiver {
                BatchReceiver::Integration1DWithAzimuthalStd(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(PyIntegrationResult::from_azimuthal);
                        (index, BatchEntry::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration1D(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(|(radial, intensity)| {
                            PyIntegrationResult::from_1d(radial, intensity)
                        });
                        (index, BatchEntry::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration1DWithErrors(rx) => {
                    recv(py, rx).map(|(index, filename, result)| {
                        let result = result.map(|(radial, intensity, errors)| {
                            PyIntegrationResult::from_1d_with_errors(radial, intensity, errors)
                        });
                        (index, BatchEntry::from_result(filename, result))
                    })
                }
                BatchReceiver::Integration2D {
                    receiver,
                    n_radial,
                    n_azimuthal,
                    radial,
                } => recv(py, receiver).map(|(index, filename, result)| {
                    let result = result.map(|intensity| {
                        PyIntegrationResult::from_2d(intensity, *n_radial, *n_azimuthal)
                            .with_radial(radial.clone())
                    });
                    (index, BatchEntry::from_result(filename, result))
                }),
                BatchReceiver::Integration2DWithErrors {
                    receiver,
                    n_radial,
                    n_azimuthal,
                    radial,
                } => recv(py, receiver).map(|(index, filename, result)| {
                    let result = result.map(|(intensity, errors)| {
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errors,
                            *n_radial,
                            *n_azimuthal,
                        )
                        .with_radial(radial.clone())
                    });
                    (index, BatchEntry::from_result(filename, result))
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
                return Err(self.end_of_stream(py));
            };

            // If this is the one we're waiting for, yield it directly
            if index == self.next_index {
                return Ok(self.deliver(batch_result));
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
