//! Batch processing of multiple files with parallel workers.
//!
//! This module provides utilities for processing multiple detector frames in parallel,
//! sharing a single `Integrator` instance across workers to minimize memory usage.
//!
//! The `IntegrationTask` trait allows you to specify which integration method to use
//! (1D, 2D, with/without errors, etc.) without duplicating the worker logic.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use dioptrin::batch::{process_files, Integration1D};
//! use dioptrin::geometry::Geometry;
//! use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
//! use dioptrin::poni::Poni;
//!
//! // Create a simple geometry for demonstration
//! let mut poni = Poni::new();
//! poni.pixel1 = 1.0;
//! poni.pixel2 = 1.0;
//! poni.distance = 1.0;
//!
//! let geometry = Geometry::new(10, 10, poni);
//! let integrator = Arc::new(
//!     Integrator::new(geometry)
//!         .with_method(IntegrationMethod::Counting)
//!         .with_radial_unit(RadialUnit::TthDeg)
//! );
//!
//! // In practice, you would use actual file paths
//! // let files = vec!["image1.tif".to_string(), "image2.tif".to_string()];
//! // let task = Integration1D { n_bins: 3000 };
//! // let results = process_files(&files, integrator, task, 4, 2);
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::frame::{Frame, TiffFrame};
use crate::integration::{IntegrationMethod, Integrator};

mod file_workers;
#[cfg(feature = "python")]
pub(crate) use file_workers::{
    FileWorkers, process_files_streaming_with_pool, process_files_with_pool,
    process_queue_with_pool,
};

/// Trait for defining integration tasks that can be executed on frames.
///
/// Implement this trait to specify which integration method to use
/// and what output type to produce.
pub trait IntegrationTask: Send + Sync {
    /// The output type produced by this integration task.
    type Output: Send;

    /// Execute the integration on a frame using the provided integrator.
    ///
    /// This method is called by workers in a thread pool context.
    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output;

    /// Compatibility adapter for custom tasks. Standard CSR tasks override this
    /// to consume the decoder's typed buffer without a full f64 image copy.
    #[doc(hidden)]
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        self.execute(integrator, &frame.to_frame())
    }

    /// The `Integrator::integrate2d*` method this task calls, if it is a 2D
    /// task. The batch helpers use it to reject the Histogram method, which
    /// has no 2D binning, with one error per input instead of a worker panic.
    fn integrate2d_method(&self) -> Option<&'static str> {
        None
    }
}

/// Error message for a task the integrator's method cannot run: a 2D task
/// with the Histogram method. Matches the `integrate2d*` panic message.
fn rejection<T: IntegrationTask>(integrator: &Integrator, task: &T) -> Option<String> {
    let method = task.integrate2d_method()?;
    matches!(integrator.method(), IntegrationMethod::Histogram)
        .then(|| format!("{method} is not supported for Histogram method"))
}

/// 1D integration task without error propagation.
///
/// Returns `(radial, intensity)` vectors.
#[derive(Clone, Copy, Debug)]
pub struct Integration1D {
    pub n_bins: usize,
}

impl IntegrationTask for Integration1D {
    type Output = (Vec<f64>, Vec<f64>);

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate1d(frame, self.n_bins)
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        let (x, y, _) = integrator.integrate_tiff1d::<false>(frame, self.n_bins);
        (x, y)
    }
}

/// 1D integration task with per-bin error estimation.
///
/// Returns `(radial, intensity, errors)` vectors.
#[derive(Clone, Copy, Debug)]
pub struct Integration1DWithErrors {
    pub n_bins: usize,
}

impl IntegrationTask for Integration1DWithErrors {
    type Output = (Vec<f64>, Vec<f64>, Vec<f64>);

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate1d_with_errors(frame, self.n_bins)
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        integrator.integrate_tiff1d::<true>(frame, self.n_bins)
    }
}

/// 1D integration with azimuthal population spread and optional Poisson errors.
#[derive(Clone, Copy, Debug)]
pub struct Integration1DWithAzimuthalStd {
    pub n_bins: usize,
    pub errors: bool,
}

impl IntegrationTask for Integration1DWithAzimuthalStd {
    type Output = crate::lut::AzimuthalIntegrationResult;

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate1d_with_azimuthal_std(frame, self.n_bins, self.errors)
    }
}

/// 2D integration task without error propagation.
///
/// Returns a flat vector of intensities `[n_radial * n_azimuthal]`.
#[derive(Clone, Copy, Debug)]
pub struct Integration2D {
    pub n_radial: usize,
    pub n_azimuthal: usize,
}

impl IntegrationTask for Integration2D {
    type Output = Vec<f64>;

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate2d(frame, self.n_radial, self.n_azimuthal)
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        integrator
            .integrate_tiff2d::<false>(frame, self.n_radial, self.n_azimuthal, None)
            .0
    }
    fn integrate2d_method(&self) -> Option<&'static str> {
        Some("integrate2d")
    }
}

/// 2D integration task with per-bin error estimation.
///
/// Returns `(intensity_flat, errors_flat)` vectors.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DWithErrors {
    pub n_radial: usize,
    pub n_azimuthal: usize,
}

impl IntegrationTask for Integration2DWithErrors {
    type Output = (Vec<f64>, Vec<f64>);

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate2d_with_errors(frame, self.n_radial, self.n_azimuthal)
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        integrator.integrate_tiff2d::<true>(frame, self.n_radial, self.n_azimuthal, None)
    }
    fn integrate2d_method(&self) -> Option<&'static str> {
        Some("integrate2d_with_errors")
    }
}

/// 2D integration with azimuthal range restriction.
///
/// Returns a flat vector of intensities `[n_radial * n_azimuthal]`.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DWithAzimuthalRange {
    pub n_radial: usize,
    pub n_azimuthal: usize,
    /// Centre of the first azimuthal bin in radians (the `azimuthal_start` of
    /// `Integrator::integrate2d_with_azimuthal_range`), despite the field name.
    pub azimuthal_min: f64,
    /// Azimuthal bin width in radians (`azimuthal_step`), despite the field name.
    pub azimuthal_max: f64,
}

impl IntegrationTask for Integration2DWithAzimuthalRange {
    type Output = Vec<f64>;

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate2d_with_azimuthal_range(
            frame,
            self.n_radial,
            self.n_azimuthal,
            self.azimuthal_min,
            self.azimuthal_max,
        )
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        integrator
            .integrate_tiff2d::<false>(
                frame,
                self.n_radial,
                self.n_azimuthal,
                Some((self.azimuthal_min, self.azimuthal_max)),
            )
            .0
    }
    fn integrate2d_method(&self) -> Option<&'static str> {
        Some("integrate2d_with_azimuthal_range")
    }
}

/// 2D integration with azimuthal range restriction and error estimation.
///
/// Returns `(intensity_flat, errors_flat)` vectors.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DWithAzimuthalRangeAndErrors {
    pub n_radial: usize,
    pub n_azimuthal: usize,
    /// Centre of the first azimuthal bin in radians (the `azimuthal_start` of
    /// `Integrator::integrate2d_with_azimuthal_range`), despite the field name.
    pub azimuthal_min: f64,
    /// Azimuthal bin width in radians (`azimuthal_step`), despite the field name.
    pub azimuthal_max: f64,
}

impl IntegrationTask for Integration2DWithAzimuthalRangeAndErrors {
    type Output = (Vec<f64>, Vec<f64>);

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate2d_with_azimuthal_range_and_errors(
            frame,
            self.n_radial,
            self.n_azimuthal,
            self.azimuthal_min,
            self.azimuthal_max,
        )
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        integrator.integrate_tiff2d::<true>(
            frame,
            self.n_radial,
            self.n_azimuthal,
            Some((self.azimuthal_min, self.azimuthal_max)),
        )
    }
    fn integrate2d_method(&self) -> Option<&'static str> {
        Some("integrate2d_with_azimuthal_range_and_errors")
    }
}

/// 2D integration on a true azimuthal range `(min, max)` in radians, via
/// [`Integrator::integrate2d_on_azimuthal_range`]; sectors are supported.
///
/// Returns the full result (intensity, optional errors, bin centres), or an
/// error message for invalid input.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DOnAzimuthalRange {
    pub n_radial: usize,
    pub n_azimuthal: usize,
    /// Lower azimuthal edge in radians.
    pub azimuthal_min: f64,
    /// Upper azimuthal edge in radians.
    pub azimuthal_max: f64,
    pub errors: bool,
}

impl IntegrationTask for Integration2DOnAzimuthalRange {
    type Output = Result<crate::integration::AzimuthalRange2dResult, String>;

    fn execute(&self, integrator: &Integrator, frame: &Frame) -> Self::Output {
        integrator.integrate2d_on_azimuthal_range(
            frame,
            self.n_radial,
            self.n_azimuthal,
            (self.azimuthal_min, self.azimuthal_max),
            self.errors,
        )
    }
    fn execute_tiff(&self, integrator: &Integrator, frame: &TiffFrame) -> Self::Output {
        self.execute(integrator, &frame.to_frame())
    }
    fn integrate2d_method(&self) -> Option<&'static str> {
        Some("integrate2d_on_azimuthal_range")
    }
}

/// Process multiple files in parallel using a shared integrator.
///
/// # Arguments
///
/// * `filenames` - Slice of file paths to process
/// * `integrator` - Shared integrator instance (geometry + LUT cache shared across workers)
/// * `task` - Integration task defining which method to use
/// * `num_workers` - Number of parallel workers to spawn
/// * `num_threads` - Number of threads per worker (for Rayon thread pool)
///
/// # Returns
///
/// Vector of `(filename, Result<Output>)` pairs in the same order as input filenames.
///
/// A 2D task (one whose [`IntegrationTask::integrate2d_method`] is `Some`) with an
/// integrator using `IntegrationMethod::Histogram` is not run: every input gets
/// `Err("<integrate2d method> is not supported for Histogram method")`.
///
/// # Notes
///
/// - Each worker gets its own Rayon thread pool with `num_threads` threads
/// - The integrator is wrapped in `Arc` and shared across all workers
/// - Geometry arrays and LUT cache are computed once and reused
/// - Total system threads used: `num_workers * num_threads`
///
/// # Examples
///
/// ```
/// # use std::sync::Arc;
/// # use dioptrin::batch::{process_files, Integration1DWithErrors};
/// # use dioptrin::geometry::Geometry;
/// # use dioptrin::integration::{IntegrationMethod, Integrator};
/// # use dioptrin::poni::Poni;
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(100, 100, poni);
/// let integrator = Arc::new(Integrator::new(geometry));
///
/// // In practice, use actual file paths
/// // let files = vec!["a.tif".to_string(), "b.tif".to_string()];
/// // let task = Integration1DWithErrors { n_bins: 1000 };
/// // let results = process_files(&files, integrator, task, 2, 4);
/// ```
pub fn process_files<T>(
    filenames: &[String],
    integrator: Arc<Integrator>,
    task: T,
    num_workers: usize,
    num_threads: usize,
) -> Vec<(String, Result<T::Output, String>)>
where
    T: IntegrationTask + Clone + 'static,
{
    if let Some(message) = rejection(&integrator, &task) {
        return filenames
            .iter()
            .map(|f| (f.clone(), Err(message.clone())))
            .collect();
    }
    let num_workers = num_workers.max(1);
    let num_files = filenames.len();

    // Distribute files across workers
    let chunk_size = num_files.div_ceil(num_workers).max(1);
    let mut handles = Vec::new();
    let mut file_chunks = Vec::new();

    for chunk_files in filenames.chunks(chunk_size) {
        file_chunks.push(chunk_files.to_vec());
    }

    for chunk in file_chunks {
        let integrator = Arc::clone(&integrator);
        let task = task.clone();

        let handle = thread::spawn(move || {
            // Create thread pool for this worker
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build thread pool");

            let mut results = Vec::new();
            // Reuse one Frame per worker so the 80 MB f64 buffer is allocated
            // once instead of per file.
            let mut frame = Frame {
                name: String::new(),
                data: Vec::new(),
                dim1: 0,
                dim2: 0,
            };

            for filename in chunk {
                let result = pool.install(|| match frame.reload_from_tiff(filename.clone()) {
                    Ok(()) => Ok(task.execute(&integrator, &frame)),
                    Err(e) => Err(e.to_string()),
                });

                results.push((filename, result));
            }

            results
        });

        handles.push(handle);
    }

    // Collect results from all workers
    let mut all_results = Vec::new();
    for handle in handles {
        let result = handle.join();
        match result {
            Ok(mut worker_results) => {
                all_results.append(&mut worker_results);
            }
            Err(_) => {
                eprintln!("Worker thread panicked");
            }
        }
    }

    all_results
}

/// Process multiple files sequentially (single-threaded).
///
/// This is a simpler alternative to `process_files` for cases where
/// parallel processing is not needed or desired.
///
/// # Arguments
///
/// * `filenames` - Slice of file paths to process
/// * `integrator` - Integrator instance to use
/// * `task` - Integration task defining which method to use
///
/// A 2D task (one whose [`IntegrationTask::integrate2d_method`] is `Some`) with an
/// integrator using `IntegrationMethod::Histogram` is not run: every input gets
/// `Err("<integrate2d method> is not supported for Histogram method")`.
///
/// # Examples
///
/// ```
/// # use dioptrin::batch::{process_files_sequential, Integration1D};
/// # use dioptrin::geometry::Geometry;
/// # use dioptrin::integration::Integrator;
/// # use dioptrin::poni::Poni;
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(100, 100, poni);
/// let integrator = Integrator::new(geometry);
///
/// // In practice, use actual file paths
/// // let files = vec!["a.tif".to_string(), "b.tif".to_string()];
/// // let task = Integration1D { n_bins: 1000 };
/// // let results = process_files_sequential(&files, &integrator, task);
/// ```
pub fn process_files_sequential<T>(
    filenames: &[String],
    integrator: &Integrator,
    task: T,
) -> Vec<(String, Result<T::Output, String>)>
where
    T: IntegrationTask,
{
    if let Some(message) = rejection(integrator, &task) {
        return filenames
            .iter()
            .map(|f| (f.clone(), Err(message.clone())))
            .collect();
    }
    let mut results = Vec::new();

    for filename in filenames {
        let result = Frame::from_tiff(filename.clone())
            .map_err(|e| e.to_string())
            .map(|frame| task.execute(integrator, &frame));

        results.push((filename.clone(), result));
    }

    results
}

/// A streaming batch processor that yields results as they complete.
///
/// Results are sent through a channel as each file is processed,
/// allowing the caller to handle results incrementally.
pub struct StreamingBatch<T> {
    receiver: Receiver<(usize, String, Result<T, String>)>,
    total_files: usize,
    completed: Arc<AtomicUsize>,
    #[allow(dead_code)]
    handles: Vec<thread::JoinHandle<()>>,
}

impl<T> StreamingBatch<T> {
    /// Get the total number of files being processed.
    pub fn total(&self) -> usize {
        self.total_files
    }

    /// Get the number of files completed so far.
    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    /// Consume the batch and return the receiver.
    pub fn into_receiver(self) -> Receiver<(usize, String, Result<T, String>)> {
        self.receiver
    }

    /// Consume the batch and return an iterator over results.
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl Iterator<Item = (usize, String, Result<T, String>)> {
        self.receiver.into_iter()
    }
}

/// Process multiple files in parallel, streaming results as they complete.
///
/// Unlike `process_files`, this function returns immediately with a `StreamingBatch`
/// that can be iterated to receive results as each file completes.
///
/// # Arguments
///
/// * `filenames` - Slice of file paths to process
/// * `integrator` - Shared integrator instance
/// * `task` - Integration task defining which method to use
/// * `num_workers` - Number of parallel workers to spawn
/// * `num_threads` - Number of threads per worker (for Rayon thread pool)
///
/// # Returns
///
/// A `StreamingBatch` that yields `(filename, Result<Output>)` pairs as they complete.
///
/// A 2D task (one whose [`IntegrationTask::integrate2d_method`] is `Some`) with an
/// integrator using `IntegrationMethod::Histogram` is not run: every input gets
/// `Err("<integrate2d method> is not supported for Histogram method")`.
pub fn process_files_streaming<T>(
    filenames: &[String],
    integrator: Arc<Integrator>,
    task: T,
    num_workers: usize,
    num_threads: usize,
) -> StreamingBatch<T::Output>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let num_workers = num_workers.max(1);
    let num_files = filenames.len();
    let completed = Arc::new(AtomicUsize::new(0));

    #[allow(clippy::type_complexity)]
    let (tx, rx): (Sender<(usize, String, Result<T::Output, String>)>, _) = mpsc::channel();

    if let Some(message) = rejection(&integrator, &task) {
        for (index, filename) in filenames.iter().enumerate() {
            let _ = tx.send((index, filename.clone(), Err(message.clone())));
        }
        completed.store(num_files, Ordering::Relaxed);
        return StreamingBatch {
            receiver: rx,
            total_files: num_files,
            completed,
            handles: Vec::new(),
        };
    }

    // Distribute files across workers
    let chunk_size = num_files.div_ceil(num_workers).max(1);
    let mut handles = Vec::new();
    let mut chunk_start = 0usize;

    for chunk_files in filenames.chunks(chunk_size) {
        let chunk: Vec<(usize, String)> = chunk_files
            .iter()
            .enumerate()
            .map(|(i, f)| (chunk_start + i, f.clone()))
            .collect();
        chunk_start += chunk_files.len();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let tx = tx.clone();
        let completed = Arc::clone(&completed);

        let handle = thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build thread pool");

            // One Frame per worker, reused across files (avoids per-file allocation).
            let mut frame = Frame {
                name: String::new(),
                data: Vec::new(),
                dim1: 0,
                dim2: 0,
            };

            for (index, filename) in chunk {
                let result = pool.install(|| match frame.reload_from_tiff(filename.clone()) {
                    Ok(()) => Ok(task.execute(&integrator, &frame)),
                    Err(e) => Err(e.to_string()),
                });

                completed.fetch_add(1, Ordering::Relaxed);
                // Ignore send errors (receiver may have been dropped)
                let _ = tx.send((index, filename, result));
            }
        });

        handles.push(handle);
    }

    // Drop the original sender so the channel closes when all workers finish
    drop(tx);

    StreamingBatch {
        receiver: rx,
        total_files: num_files,
        completed,
        handles,
    }
}

/// A work item for queue-based streaming processing.
///
/// Workers pull items from a shared queue, enabling dynamic load balancing
/// and lazy consumption of input (e.g., Python generators).
pub enum WorkItem {
    /// A file path — the worker loads the TIFF before integrating.
    Path { index: usize, path: String },
    /// An already-loaded frame (e.g., from a numpy array).
    Frame {
        index: usize,
        name: String,
        frame: Frame,
    },
}

/// Process work items from a shared queue, streaming results as they complete.
///
/// Workers pull items from the receiver using work-stealing (shared
/// `Arc<Mutex<Receiver<WorkItem>>>`), providing automatic load balancing.
///
/// # Arguments
///
/// * `work_rx` - Receiver end of the work queue
/// * `integrator` - Shared integrator instance
/// * `task` - Integration task defining which method to use
/// * `num_workers` - Number of parallel workers to spawn
/// * `num_threads` - Number of threads per worker (for Rayon thread pool)
///
/// # Returns
///
/// A `StreamingBatch` that yields `(name, Result<Output>)` pairs as they complete.
/// The `total_files` field is set to 0 since the total is unknown for queue-based input.
///
/// A 2D task (one whose [`IntegrationTask::integrate2d_method`] is `Some`) with an
/// integrator using `IntegrationMethod::Histogram` is not run: every input gets
/// `Err("<integrate2d method> is not supported for Histogram method")`.
pub fn process_queue_streaming<T>(
    work_rx: std::sync::mpsc::Receiver<WorkItem>,
    integrator: Arc<Integrator>,
    task: T,
    num_workers: usize,
    num_threads: usize,
) -> StreamingBatch<T::Output>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let num_workers = num_workers.max(1);
    let completed = Arc::new(AtomicUsize::new(0));

    #[allow(clippy::type_complexity)]
    let (tx, rx): (Sender<(usize, String, Result<T::Output, String>)>, _) = mpsc::channel();

    let work_rx = Arc::new(std::sync::Mutex::new(work_rx));
    let mut handles = Vec::new();
    // Queue input is consumed lazily, so a rejected task answers each item
    // with this error instead of integrating it.
    let rejected = rejection(&integrator, &task);

    for _ in 0..num_workers {
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let tx = tx.clone();
        let completed = Arc::clone(&completed);
        let work_rx = Arc::clone(&work_rx);
        let rejected = rejected.clone();

        let handle = thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build thread pool");

            // Reusable Frame for the path branch — kept across iterations of
            // the queue loop so we allocate the f64 buffer at most once.
            let mut path_frame = Frame {
                name: String::new(),
                data: Vec::new(),
                dim1: 0,
                dim2: 0,
            };

            loop {
                // Pull the next work item from the shared queue
                let item = {
                    let guard = match work_rx.lock() {
                        Ok(g) => g,
                        Err(_) => break,
                    };
                    match guard.recv() {
                        Ok(item) => item,
                        Err(_) => break, // Channel closed, no more work
                    }
                };

                let (index, name, result) = match (item, rejected.as_deref()) {
                    (
                        WorkItem::Path { index, path: name } | WorkItem::Frame { index, name, .. },
                        Some(message),
                    ) => (index, name, Err(message.to_owned())),
                    (WorkItem::Path { index, path }, None) => {
                        let name = path.clone();
                        let result = pool.install(|| match path_frame.reload_from_tiff(path) {
                            Ok(()) => Ok(task.execute(&integrator, &path_frame)),
                            Err(e) => Err(e.to_string()),
                        });
                        (index, name, result)
                    }
                    (WorkItem::Frame { index, name, frame }, None) => {
                        let result = pool.install(|| Ok(task.execute(&integrator, &frame)));
                        (index, name, result)
                    }
                };

                completed.fetch_add(1, Ordering::Relaxed);
                if tx.send((index, name, result)).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        handles.push(handle);
    }

    // Drop the original sender so the channel closes when all workers finish
    drop(tx);

    StreamingBatch {
        receiver: rx,
        total_files: 0, // Unknown for queue-based input
        completed,
        handles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::integration::IntegrationMethod;
    use crate::poni::Poni;

    fn test_poni() -> Poni {
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 1.0;
        poni
    }

    fn test_frame(name: &str, value: f64) -> Frame {
        Frame {
            name: String::from(name),
            data: vec![value; 4],
            dim1: 2,
            dim2: 2,
        }
    }

    #[test]
    fn test_queue_streaming_indices_match_input_order() {
        // Send 20 frames through process_queue_streaming with 4 workers.
        // Verify that the index in each result matches the name we gave it,
        // proving the index is threaded correctly through the pipeline.
        let poni = test_poni();
        let geometry = Geometry::new(2, 2, poni);
        let integrator =
            Arc::new(Integrator::new(geometry).with_method(IntegrationMethod::Counting));

        let (work_tx, work_rx) = mpsc::channel();
        let n_frames = 20;

        for i in 0..n_frames {
            let frame = test_frame(&format!("frame_{}", i), i as f64);
            work_tx
                .send(WorkItem::Frame {
                    index: i,
                    name: format!("frame_{}", i),
                    frame,
                })
                .unwrap();
        }
        drop(work_tx);

        let task = Integration1D { n_bins: 4 };
        let batch = process_queue_streaming(work_rx, integrator, task, 4, 1);

        // Collect all results (may arrive out of order)
        let results: Vec<_> = batch.into_iter().collect();
        assert_eq!(results.len(), n_frames);

        // Verify each result's index matches its name
        for (index, name, result) in &results {
            assert_eq!(name, &format!("frame_{}", index));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_queue_streaming_reorder_yields_input_order() {
        // Simulate the reorder buffer logic used by PyBatchIterator:
        // collect results from the channel, sort by index, verify sequential order.
        let poni = test_poni();
        let geometry = Geometry::new(2, 2, poni);
        let integrator =
            Arc::new(Integrator::new(geometry).with_method(IntegrationMethod::Counting));

        let (work_tx, work_rx) = mpsc::channel();
        let n_frames = 30;

        for i in 0..n_frames {
            let frame = test_frame(&format!("frame_{}", i), i as f64);
            work_tx
                .send(WorkItem::Frame {
                    index: i,
                    name: format!("frame_{}", i),
                    frame,
                })
                .unwrap();
        }
        drop(work_tx);

        let task = Integration1D { n_bins: 4 };
        let batch = process_queue_streaming(work_rx, integrator, task, 4, 1);

        // Collect and sort by index (this is what the reorder buffer does)
        let mut results: Vec<_> = batch.into_iter().collect();
        results.sort_by_key(|(index, _, _)| *index);

        // Verify we get all frames in sequential order
        assert_eq!(results.len(), n_frames);
        for (i, (index, name, _)) in results.iter().enumerate() {
            assert_eq!(*index, i, "index mismatch at position {}", i);
            assert_eq!(
                name,
                &format!("frame_{}", i),
                "name mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn empty_file_batches_finish_without_workers() {
        let integrator = Arc::new(Integrator::new(Geometry::new(2, 2, test_poni())));
        let task = Integration1DWithAzimuthalStd {
            n_bins: 4,
            errors: true,
        };
        assert!(process_files(&[], Arc::clone(&integrator), task, 2, 1).is_empty());
        assert_eq!(
            process_files_streaming(&[], integrator, task, 2, 1)
                .into_iter()
                .count(),
            0
        );
    }

    #[test]
    fn test_files_streaming_indices_are_sequential() {
        // process_files_streaming distributes files across worker chunks.
        // Verify indices are globally sequential (chunk_start + i), not per-chunk.
        // We can't use real TIFF files here, but we can test with the queue approach
        // which exercises the same indexing in process_queue_streaming.
        // Instead, test process_files_streaming with a few "files" that will fail
        // to load (non-existent), and verify the indices are still correct.
        let poni = test_poni();
        let geometry = Geometry::new(2, 2, poni);
        let integrator =
            Arc::new(Integrator::new(geometry).with_method(IntegrationMethod::Counting));

        let filenames: Vec<String> = (0..8).map(|i| format!("nonexistent_{}.tif", i)).collect();

        let task = Integration1D { n_bins: 4 };
        let batch = process_files_streaming(&filenames, integrator, task, 4, 1);

        let mut results: Vec<_> = batch.into_iter().collect();
        assert_eq!(results.len(), 8);

        // Sort by index and verify sequential
        results.sort_by_key(|(index, _, _)| *index);
        for (i, (index, name, result)) in results.iter().enumerate() {
            assert_eq!(*index, i, "index mismatch at position {}", i);
            assert_eq!(name, &format!("nonexistent_{}.tif", i));
            // All should be errors since files don't exist
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_integration_tasks() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 1.0;

        let frame = Frame {
            name: String::from("test"),
            data: vec![1.0, 2.0, 3.0, 4.0],
            dim1: 2,
            dim2: 2,
        };

        let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
        let integrator = Integrator::new(geometry).with_method(IntegrationMethod::Counting);

        // Test 1D integration
        let task_1d = Integration1D { n_bins: 4 };
        let (radial, intensity) = task_1d.execute(&integrator, &frame);
        assert_eq!(radial.len(), 4);
        assert_eq!(intensity.len(), 4);

        // Test 1D integration with errors
        let task_1d_errors = Integration1DWithErrors { n_bins: 4 };
        let (radial, intensity, errors) = task_1d_errors.execute(&integrator, &frame);
        assert_eq!(radial.len(), 4);
        assert_eq!(intensity.len(), 4);
        assert_eq!(errors.len(), 4);

        // Test 2D integration
        let task_2d = Integration2D {
            n_radial: 3,
            n_azimuthal: 2,
        };
        let intensity = task_2d.execute(&integrator, &frame);
        assert_eq!(intensity.len(), 6);

        // Test 2D integration with errors
        let task_2d_errors = Integration2DWithErrors {
            n_radial: 3,
            n_azimuthal: 2,
        };
        let (intensity, errors) = task_2d_errors.execute(&integrator, &frame);
        assert_eq!(intensity.len(), 6);
        assert_eq!(errors.len(), 6);
    }

    /// Writes `n` small TIFFs and returns their paths.
    fn write_tiffs(tag: &str, n: usize) -> Vec<String> {
        use tiff::encoder::{TiffEncoder, colortype};
        (0..n)
            .map(|i| {
                let path = std::env::temp_dir().join(format!(
                    "dioptrin-batch-{tag}-{}-{i}.tif",
                    std::process::id()
                ));
                let mut enc = TiffEncoder::new(std::fs::File::create(&path).unwrap()).unwrap();
                enc.write_image::<colortype::Gray16>(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
                    .unwrap();
                path.to_str().unwrap().to_owned()
            })
            .collect()
    }

    /// Every batch helper answers a 2D task with the Histogram method with one
    /// error per input. Before, `process_files` and both streaming helpers
    /// lost the inputs to worker panics and returned nothing.
    fn check_histogram_2d_rejected<T>(task: T, expected: &str)
    where
        T: IntegrationTask + Clone + 'static,
        T::Output: 'static,
    {
        let files = write_tiffs(expected, 4);
        let integrator = Arc::new(
            Integrator::new(Geometry::new(3, 3, test_poni()))
                .with_method(IntegrationMethod::Histogram),
        );
        let check = |helper: &str, mut results: Vec<(usize, String, Result<T::Output, String>)>| {
            assert_eq!(results.len(), files.len(), "{helper}");
            results.sort_by_key(|r| r.0);
            for (i, (index, name, result)) in results.into_iter().enumerate() {
                assert_eq!((index, &name), (i, &files[i]), "{helper}");
                assert_eq!(result.err().as_deref(), Some(expected), "{helper}");
            }
        };
        let indexed = |v: Vec<(String, Result<T::Output, String>)>| {
            v.into_iter()
                .enumerate()
                .map(|(i, (name, r))| (i, name, r))
                .collect()
        };

        let results = process_files(&files, Arc::clone(&integrator), task.clone(), 2, 1);
        check("process_files", indexed(results));
        let results = process_files_sequential(&files, &integrator, task.clone());
        check("process_files_sequential", indexed(results));
        let batch = process_files_streaming(&files, Arc::clone(&integrator), task.clone(), 2, 1);
        assert_eq!((batch.total(), batch.completed()), (4, 4));
        check("process_files_streaming", batch.into_iter().collect());
        let (work_tx, work_rx) = mpsc::channel();
        for (index, path) in files.iter().enumerate() {
            work_tx
                .send(WorkItem::Path {
                    index,
                    path: path.clone(),
                })
                .unwrap();
        }
        drop(work_tx);
        let batch = process_queue_streaming(work_rx, integrator, task, 2, 1);
        check("process_queue_streaming", batch.into_iter().collect());
        for file in files {
            std::fs::remove_file(file).unwrap();
        }
    }

    #[test]
    fn batch_helpers_reject_histogram_for_every_2d_task() {
        let (n_radial, n_azimuthal) = (5, 4);
        let (azimuthal_min, azimuthal_max) = (0.0, 0.5);
        check_histogram_2d_rejected(
            Integration2D {
                n_radial,
                n_azimuthal,
            },
            "integrate2d is not supported for Histogram method",
        );
        check_histogram_2d_rejected(
            Integration2DWithErrors {
                n_radial,
                n_azimuthal,
            },
            "integrate2d_with_errors is not supported for Histogram method",
        );
        check_histogram_2d_rejected(
            Integration2DWithAzimuthalRange {
                n_radial,
                n_azimuthal,
                azimuthal_min,
                azimuthal_max,
            },
            "integrate2d_with_azimuthal_range is not supported for Histogram method",
        );
        check_histogram_2d_rejected(
            Integration2DWithAzimuthalRangeAndErrors {
                n_radial,
                n_azimuthal,
                azimuthal_min,
                azimuthal_max,
            },
            "integrate2d_with_azimuthal_range_and_errors is not supported for Histogram method",
        );
        for errors in [false, true] {
            check_histogram_2d_rejected(
                Integration2DOnAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                    errors,
                },
                "integrate2d_on_azimuthal_range is not supported for Histogram method",
            );
        }
    }

    #[test]
    fn batch_helpers_still_run_1d_histogram_tasks() {
        let files = write_tiffs("hist1d", 3);
        let integrator = Arc::new(
            Integrator::new(Geometry::new(3, 3, test_poni()))
                .with_method(IntegrationMethod::Histogram),
        );
        let results = process_files(&files, integrator, Integration1D { n_bins: 4 }, 2, 1);
        assert_eq!(results.len(), 3);
        for (_, result) in results {
            let (x, y) = result.unwrap();
            assert_eq!((x.len(), y.len()), (4, 4));
        }
        for file in files {
            std::fs::remove_file(file).unwrap();
        }
    }
}
