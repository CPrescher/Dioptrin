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

use crate::frame::Frame;
use crate::integration::Integrator;

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
}

/// 2D integration with azimuthal range restriction.
///
/// Returns a flat vector of intensities `[n_radial * n_azimuthal]`.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DWithAzimuthalRange {
    pub n_radial: usize,
    pub n_azimuthal: usize,
    pub azimuthal_min: f64,
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
}

/// 2D integration with azimuthal range restriction and error estimation.
///
/// Returns `(intensity_flat, errors_flat)` vectors.
#[derive(Clone, Copy, Debug)]
pub struct Integration2DWithAzimuthalRangeAndErrors {
    pub n_radial: usize,
    pub n_azimuthal: usize,
    pub azimuthal_min: f64,
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
    let num_workers = num_workers.max(1);
    let num_files = filenames.len();

    // Distribute files across workers
    let chunk_size = num_files.div_ceil(num_workers);
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

            for filename in chunk {
                let result = pool.install(|| {
                    // Load frame
                    let frame = Frame::from_tiff(filename.clone()).map_err(|e| e.to_string())?;

                    // Execute integration task
                    Ok(task.execute(&integrator, &frame))
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
    receiver: Receiver<(String, Result<T, String>)>,
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
    pub fn into_receiver(self) -> Receiver<(String, Result<T, String>)> {
        self.receiver
    }

    /// Consume the batch and return an iterator over results.
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl Iterator<Item = (String, Result<T, String>)> {
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
    let (tx, rx): (Sender<(String, Result<T::Output, String>)>, _) = mpsc::channel();

    // Distribute files across workers
    let chunk_size = num_files.div_ceil(num_workers);
    let mut handles = Vec::new();

    for chunk_files in filenames.chunks(chunk_size) {
        let chunk = chunk_files.to_vec();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let tx = tx.clone();
        let completed = Arc::clone(&completed);

        let handle = thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build thread pool");

            for filename in chunk {
                let result = pool.install(|| {
                    let frame = Frame::from_tiff(filename.clone()).map_err(|e| e.to_string())?;
                    Ok(task.execute(&integrator, &frame))
                });

                completed.fetch_add(1, Ordering::Relaxed);
                // Ignore send errors (receiver may have been dropped)
                let _ = tx.send((filename, result));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::integration::IntegrationMethod;
    use crate::poni::Poni;

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
}
