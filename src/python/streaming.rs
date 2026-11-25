//! Queue-fed Python iterator/array streaming and reusable file-worker pools for `PyIntegrator`.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender};

use crate::batch::{
    FileWorkers, IntegrationTask, StreamControl, StreamingBatch, WorkItem,
    process_files_streaming_with_pool, process_files_with_pool,
};
use crate::frame::Frame;
use crate::integration::Integrator;

use super::integrator::PyIntegrator;
use super::results::{StreamError, panic_message};
use super::utils::{
    ClassifiedInput, FrameOrPath, check_mask_shape, extract_frame_from_pyany, extract_item_dims,
    mask_shape,
};

/// Items a Python iterator stream may have pulled but not yet handed to the
/// caller: queued, being integrated, or waiting in the result channel or the
/// reorder buffer. Bounds the memory of a stream nobody consumes (an endless
/// generator would otherwise be drained as fast as the workers run) while
/// keeping every worker busy: a full work queue (`2 * workers`), one item per
/// worker, and one finished result per worker.
fn iterator_window(num_workers: usize) -> usize {
    num_workers.max(1) * 4
}

/// Next work item from a Python iterator: `Ok(None)` at its end, `Err` for an
/// exception from the iterator, an invalid item or a mask-shape mismatch.
fn pull_work_item(
    iter: &Bound<'_, PyAny>,
    index: usize,
    mask: Option<(usize, usize)>,
) -> PyResult<Option<WorkItem>> {
    let item = match iter.call_method0("__next__") {
        Ok(item) => item,
        Err(err) if err.is_instance_of::<PyStopIteration>(iter.py()) => return Ok(None),
        Err(err) => return Err(err),
    };
    Ok(Some(match extract_frame_from_pyany(&item, index)? {
        FrameOrPath::Path(path) => WorkItem::Path { index, path },
        FrameOrPath::Frame(frame) => {
            check_mask_shape(mask, (frame.dim1, frame.dim2), &format!("images[{index}]"))?;
            WorkItem::Frame {
                index,
                name: frame.name.clone(),
                frame,
            }
        }
    }))
}

/// Next work item from a list of arrays checked when the stream was opened,
/// copied now: `Ok(None)` after the last one. Shape and mask are checked again,
/// as the caller may have changed an array in place since.
fn pull_array_item(
    array: Option<Py<PyArray2<f64>>>,
    py: Python<'_>,
    index: usize,
    mask: Option<(usize, usize)>,
) -> PyResult<Option<WorkItem>> {
    let Some(array) = array else {
        return Ok(None);
    };
    let array = array
        .bind(py)
        .try_readonly()
        .map_err(|err| PyRuntimeError::new_err(format!("images[{index}]: {err}")))?;
    let (dim1, dim2) = extract_item_dims(&array, index)?;
    let slice = array
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("Array {index} must be contiguous: {err}")))?;
    check_mask_shape(mask, (dim1, dim2), &format!("images[{index}]"))?;
    let name = format!("array[{index}]");
    Ok(Some(WorkItem::Frame {
        index,
        name: name.clone(),
        frame: Frame {
            name,
            data: slice.to_vec(),
            dim1,
            dim2,
        },
    }))
}

/// What the producer of a queue-fed stream reads.
pub(super) enum StreamInput {
    /// The caller's iterator or generator.
    Iterator(Py<PyAny>),
    /// A list of arrays, checked when the stream was opened and copied one at
    /// a time as the stream reaches them; each reference is dropped once its
    /// frame is copied.
    Arrays(std::vec::IntoIter<Py<PyArray2<f64>>>),
}

/// A started queue-fed stream, before its workers get their task.
pub(super) struct QueueStream {
    pub(super) work_rx: Receiver<WorkItem>,
    pub(super) pool: Arc<FileWorkers>,
    pub(super) stream_error: StreamError,
    pub(super) control: Arc<StreamControl>,
}

/// Pull items from a Python iterator or a list of arrays on a producer thread
/// and queue them for the workers. An invalid item, a mask-shape mismatch, an exception raised
/// by the iterator itself, or a Rust panic while pulling (pyo3 resumes a
/// `PanicException` from the caller's code as one) is recorded in
/// `stream_error` at its position, so the consumer raises it after all
/// earlier results, and ends production. Only reaching the end of the
/// input marks it exhausted; the consumer treats any other end of
/// the stream without a recorded error as a failure.
///
/// Before each pull the producer waits for credit (see [`iterator_window`])
/// and stops once the stream is closed. It cannot be interrupted while inside
/// the caller's `__next__`: if the iterator blocks there (a live feed waiting
/// for its next frame) after the stream was closed, this thread stays until
/// that call returns. It holds neither the integrator nor the workers, which
/// shut down independently.
fn spawn_producer(
    input: StreamInput,
    work_tx: SyncSender<WorkItem>,
    mask: Option<(usize, usize)>,
    stream_error: StreamError,
    control: Arc<StreamControl>,
    window: usize,
) {
    std::thread::spawn(move || {
        Python::attach(move |py| {
            // Moved in, so the remaining references are dropped while attached.
            let mut input = input;
            for index in 0.. {
                // Wait for credit (and below, for queue space) without
                // holding the GIL.
                if !py.detach(|| control.wait_for_credit(index, window)) {
                    break; // Stream closed
                }
                let pull = || match &mut input {
                    StreamInput::Iterator(iter) => pull_work_item(iter.bind(py), index, mask),
                    StreamInput::Arrays(arrays) => pull_array_item(arrays.next(), py, index, mask),
                };
                let work_item = match catch_unwind(AssertUnwindSafe(pull)) {
                    Ok(Ok(Some(item))) => item,
                    Ok(Ok(None)) => {
                        control.mark_input_exhausted();
                        break;
                    }
                    Ok(Err(err)) => {
                        stream_error.set(index, err);
                        break;
                    }
                    Err(payload) => {
                        stream_error.set(
                            index,
                            PyRuntimeError::new_err(format!(
                                "reading images[{index}] from the input panicked: {}",
                                panic_message(&*payload)
                            )),
                        );
                        break;
                    }
                };
                if py.detach(|| work_tx.send(work_item)).is_err() {
                    break; // Workers gone
                }
            }
        });
        // work_tx dropped here → channel closes → workers drain and stop
    });
}

impl PyIntegrator {
    pub(super) fn file_pool(&self, workers: usize, threads: usize) -> PyResult<Arc<FileWorkers>> {
        let mut cache = self
            .file_workers
            .lock()
            .expect("file worker cache lock poisoned");
        if let Some(pool) = cache.as_ref()
            && pool.workers() == workers
            && pool.num_threads == threads
        {
            return Ok(Arc::clone(pool));
        }
        let pool = FileWorkers::new(workers, threads).map_err(PyValueError::new_err)?;
        *cache = Some(Arc::clone(&pool));
        Ok(pool)
    }

    /// Workers for a stream fed by a Python iterator, run by `control`. Such a
    /// stream holds every worker of its pool until the iterator is exhausted
    /// (or the stream is dropped), so it gets a pool of its own: on the shared
    /// cached pool a later synchronous batch, or a second stream, would queue
    /// behind it for as long as the caller's generator keeps producing.
    ///
    /// Building a pool costs a few hundred microseconds (one OS thread and one
    /// Rayon pool per worker), noticeable for short streams. The pool of the
    /// previous iterator stream is therefore reused when its workers have all
    /// finished (the stream ended or was closed); a stream that is still open
    /// keeps its pool and the new stream gets a fresh one.
    pub(super) fn iterator_pool(
        &self,
        workers: usize,
        threads: usize,
        control: &Arc<StreamControl>,
    ) -> PyResult<Arc<FileWorkers>> {
        let mut cache = self
            .iterator_workers
            .lock()
            .expect("iterator worker cache lock poisoned");
        let pool = match cache.as_ref() {
            Some((pool, last))
                if pool.workers() == workers
                    && pool.num_threads == threads
                    && last.workers_idle() =>
            {
                Arc::clone(pool)
            }
            _ => FileWorkers::new(workers, threads).map_err(PyValueError::new_err)?,
        };
        *cache = Some((Arc::clone(&pool), Arc::clone(control)));
        Ok(pool)
    }

    /// Producer input for a stream over a list of arrays or an iterator, and
    /// its length (0 when unknown). A list is checked here, before the stream
    /// starts (shape, contiguity, mask), but not copied.
    pub(super) fn stream_input(
        &self,
        input: ClassifiedInput<'_>,
    ) -> PyResult<(StreamInput, usize)> {
        match input {
            ClassifiedInput::Arrays(arrays) => {
                let mut checked = Vec::with_capacity(arrays.len());
                for (i, array) in arrays.iter().enumerate() {
                    let (dim1, dim2) = extract_item_dims(array, i)?;
                    array.as_slice().map_err(|err| {
                        PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
                    })?;
                    self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
                    checked.push(array.as_unbound().clone_ref(array.py()));
                }
                let total = checked.len();
                Ok((StreamInput::Arrays(checked.into_iter()), total))
            }
            ClassifiedInput::Iterator(iter) => Ok((StreamInput::Iterator(iter.unbind()), 0)),
            ClassifiedInput::Paths(_) => unreachable!("path lists use the file streams"),
        }
    }

    /// Start a queue-fed stream: its own workers (see [`Self::iterator_pool`])
    /// and a producer thread that reads `input` within the credit window.
    pub(super) fn open_queue_stream(
        &self,
        input: StreamInput,
        num_workers: usize,
        num_threads: usize,
    ) -> PyResult<QueueStream> {
        let control = StreamControl::new();
        let pool = self.iterator_pool(num_workers, num_threads, &control)?;
        let window = iterator_window(num_workers);
        let (work_tx, work_rx) = std::sync::mpsc::sync_channel(num_workers * 2);
        let stream_error = StreamError::default();
        spawn_producer(
            input,
            work_tx,
            mask_shape(&self.integrator),
            stream_error.clone(),
            Arc::clone(&control),
            window,
        );
        Ok(QueueStream {
            work_rx,
            pool,
            stream_error,
            control,
        })
    }

    /// Integrate `paths` on the cached file workers and wait for all results.
    ///
    /// The wait releases the GIL: a stream fed by a Python iterator needs it to
    /// pull its next item, and its jobs may be ahead of this batch in a worker
    /// queue. Nothing inside the detached region touches Python objects.
    pub(super) fn process_files<T: IntegrationTask + Clone + 'static>(
        &self,
        py: Python<'_>,
        paths: &[String],
        integrator: Arc<Integrator>,
        task: T,
        workers: usize,
        threads: usize,
    ) -> Vec<(String, Result<T::Output, String>)>
    where
        T::Output: 'static,
    {
        // Worker construction errors are rare resource errors; preserve one
        // ordered error result per input file, including for empty batches.
        if paths.is_empty() {
            return Vec::new();
        }
        match self.file_pool(workers, threads) {
            Ok(pool) => py.detach(|| process_files_with_pool(paths, integrator, task, pool)),
            Err(e) => paths
                .iter()
                .map(|p| (p.clone(), Err(e.to_string())))
                .collect(),
        }
    }

    pub(super) fn process_files_streaming<T: IntegrationTask + Clone + 'static>(
        &self,
        paths: &[String],
        integrator: Arc<Integrator>,
        task: T,
        workers: usize,
        threads: usize,
    ) -> PyResult<StreamingBatch<T::Output>>
    where
        T::Output: 'static,
    {
        Ok(process_files_streaming_with_pool(
            paths,
            integrator,
            task,
            self.file_pool(workers, threads)?,
        ))
    }
}
