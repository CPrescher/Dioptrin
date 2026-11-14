//! Persistent worker-owned TIFF buffers and Rayon contexts.
#![cfg_attr(not(feature = "python"), allow(dead_code))]
use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// How often a worker waiting for queue input checks whether its stream was
/// cancelled. Bounds how long a closed stream keeps its workers.
const CANCEL_POLL: Duration = Duration::from_millis(50);

/// Shared state of one queue-fed stream: its consumer (the Python
/// `BatchIterator`), its workers and, for Python iterator input, the producer
/// thread that pulls items from the caller's iterator.
///
/// - `cancel` stops the stream: idle workers leave their queue wait within
///   [`CANCEL_POLL`], busy ones after their current item, and the producer
///   before pulling another item.
/// - `deliver`/`wait_for_credit` give the producer back-pressure: it may only
///   pull item `i` once fewer than `window` items are pulled but not yet
///   delivered to the caller, which bounds the work queue, the result channel
///   and the consumer's reorder buffer together.
/// - `wait_workers_idle` waits until every worker of the stream has released
///   its share of the integrator.
pub(crate) struct StreamControl {
    cancelled: AtomicBool,
    input_exhausted: AtomicBool,
    /// Fed by a Python iterator, on a pool of its own.
    iterator_input: bool,
    state: Mutex<ControlState>,
    changed: Condvar,
}

#[derive(Default)]
struct ControlState {
    active_workers: usize,
    delivered: usize,
}

/// Held by one worker job of a stream; dropping it marks the job finished.
pub(crate) struct WorkerGuard(Arc<StreamControl>);

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let mut state = self.0.state();
        state.active_workers -= 1;
        drop(state);
        self.0.changed.notify_all();
    }
}

impl StreamControl {
    /// Control for a stream fed by a Python iterator on its own worker pool.
    pub(crate) fn for_iterator() -> Arc<Self> {
        Self::new(true)
    }

    /// Control for a finite stream on the shared cached worker pool.
    pub(crate) fn for_shared_pool() -> Arc<Self> {
        Self::new(false)
    }

    fn new(iterator_input: bool) -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            input_exhausted: AtomicBool::new(false),
            iterator_input,
            state: Mutex::new(ControlState::default()),
            changed: Condvar::new(),
        })
    }

    fn state(&self) -> MutexGuard<'_, ControlState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn iterator_input(&self) -> bool {
        self.iterator_input
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        // Take the lock so a waiter cannot miss the flag between its check
        // and its wait.
        drop(self.state());
        self.changed.notify_all();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// The producer reached the end of its input (as opposed to stopping on
    /// an error, a panic or cancellation).
    pub(crate) fn mark_input_exhausted(&self) {
        self.input_exhausted.store(true, Ordering::SeqCst);
    }

    pub(crate) fn input_exhausted(&self) -> bool {
        self.input_exhausted.load(Ordering::SeqCst)
    }

    /// The consumer has handed out the first `count` results.
    pub(crate) fn deliver(&self, count: usize) {
        let mut state = self.state();
        if count > state.delivered {
            state.delivered = count;
            drop(state);
            self.changed.notify_all();
        }
    }

    /// Block until item `index` may be pulled, i.e. fewer than `window` items
    /// are in flight. Returns false once the stream is cancelled.
    pub(crate) fn wait_for_credit(&self, index: usize, window: usize) -> bool {
        let mut state = self.state();
        loop {
            if self.is_cancelled() {
                return false;
            }
            if index < state.delivered.saturating_add(window.max(1)) {
                return true;
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Register a worker job; it counts as active until the guard is dropped.
    fn worker_started(self: &Arc<Self>) -> WorkerGuard {
        self.state().active_workers += 1;
        WorkerGuard(Arc::clone(self))
    }

    /// Block until every registered worker job has finished.
    pub(crate) fn wait_workers_idle(&self) {
        let mut state = self.state();
        while state.active_workers > 0 {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

/// Next item from a queue shared by several workers, or None once the queue
/// is closed or the stream cancelled. Waits in slices of [`CANCEL_POLL`] so a
/// cancelled stream frees its workers even while the producer is stuck (for
/// example in a Python generator that waits for a detector frame).
fn next_item(work_rx: &Mutex<Receiver<WorkItem>>, control: &StreamControl) -> Option<WorkItem> {
    let rx = work_rx.lock().expect("work queue lock poisoned");
    loop {
        // Checked after taking the lock too: workers queued on the lock
        // leave at once instead of each waiting out a poll interval.
        if control.is_cancelled() {
            return None;
        }
        match rx.recv_timeout(CANCEL_POLL) {
            Ok(item) => return Some(item),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

struct Job {
    run: Box<dyn FnOnce(&mut TiffFrame) + Send>,
    keep_alive: Arc<FileWorkers>,
}

pub(crate) struct FileWorkers {
    senders: Vec<Sender<Job>>,
    handles: Vec<thread::JoinHandle<()>>,
    pub(crate) num_threads: usize,
}

impl FileWorkers {
    pub(crate) fn new(workers: usize, threads: usize) -> Result<Arc<Self>, String> {
        let mut senders = Vec::new();
        let mut handles = Vec::new();
        for _ in 0..workers.max(1) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| e.to_string())?;
            let (tx, rx) = mpsc::channel::<Job>();
            // Building the Rayon context before spawning also makes failures
            // observable by the caller rather than silently losing file results.
            let handle = thread::spawn(move || {
                let mut frame = TiffFrame::default();
                while let Ok(job) = rx.recv() {
                    pool.install(|| (job.run)(&mut frame));
                    // Release on the owning OS worker, after Rayon returns.
                    // Releasing inside Rayon could join the OS thread that is
                    // waiting for that very Rayon closure to finish.
                    drop(job.keep_alive);
                }
            });
            senders.push(tx);
            handles.push(handle);
        }
        Ok(Arc::new(Self {
            senders,
            handles,
            num_threads: threads,
        }))
    }

    pub(crate) fn workers(&self) -> usize {
        self.senders.len()
    }

    fn submit(self: &Arc<Self>, worker: usize, job: impl FnOnce(&mut TiffFrame) + Send + 'static) {
        // A streaming receiver can outlive its Python Integrator. Keep this
        // pool alive until its admitted work finishes, without retaining images
        // or the integrator after the job completes.
        let keep_alive = Arc::clone(self);
        self.senders[worker]
            .send(Job {
                run: Box::new(job),
                keep_alive,
            })
            .expect("file worker stopped unexpectedly");
    }
}

impl Drop for FileWorkers {
    fn drop(&mut self) {
        self.senders.clear();
        for handle in self.handles.drain(..) {
            // The final reference may belong to a completed streaming job.
            if handle.thread().id() != thread::current().id() {
                let _ = handle.join();
            }
        }
    }
}

fn execute<T: IntegrationTask>(
    task: &T,
    integrator: &Integrator,
    frame: &mut TiffFrame,
    path: &str,
) -> Result<T::Output, String> {
    catch_unwind(AssertUnwindSafe(|| {
        frame.reload(path).map_err(|e| e.to_string())?;
        Ok(task.execute_tiff(integrator, frame))
    }))
    .unwrap_or_else(|_| Err("integration worker panicked".to_owned()))
}

pub(crate) fn process_files_with_pool<T>(
    filenames: &[String],
    integrator: Arc<Integrator>,
    task: T,
    pool: Arc<FileWorkers>,
) -> Vec<(String, Result<T::Output, String>)>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let total = filenames.len();
    let (tx, rx) = mpsc::channel();
    let chunk_size = total.div_ceil(pool.workers()).max(1);
    for (worker, chunk) in filenames.chunks(chunk_size).enumerate() {
        let paths = chunk.to_vec();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let tx = tx.clone();
        pool.submit(worker, move |frame| {
            let results: Vec<_> = paths
                .into_iter()
                .map(|path| {
                    let result = execute(&task, &integrator, frame, &path);
                    (path, result)
                })
                .collect();
            // Synchronous callers may change calibration immediately on return.
            drop(integrator);
            drop(task);
            let _ = tx.send((worker, results));
        });
    }
    drop(tx);
    let mut chunks: Vec<_> = rx.into_iter().collect();
    chunks.sort_unstable_by_key(|r| r.0);
    let results: Vec<_> = chunks.into_iter().flat_map(|(_, r)| r).collect();
    assert_eq!(results.len(), total, "a file worker lost an output");
    results
}

pub(crate) fn process_files_streaming_with_pool<T>(
    filenames: &[String],
    integrator: Arc<Integrator>,
    task: T,
    pool: Arc<FileWorkers>,
) -> StreamingBatch<T::Output>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let completed = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    let chunk_size = filenames.len().div_ceil(pool.workers()).max(1);
    for (worker, chunk) in filenames.chunks(chunk_size).enumerate() {
        let start = worker * chunk_size;
        let paths = chunk.to_vec();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let tx = tx.clone();
        let completed = Arc::clone(&completed);
        pool.submit(worker, move |frame| {
            for (offset, path) in paths.into_iter().enumerate() {
                let result = execute(&task, &integrator, frame, &path);
                completed.fetch_add(1, Ordering::Relaxed);
                if tx.send((start + offset, path, result)).is_err() {
                    break;
                }
            }
            // Release the integrator before this job's sender: once the last
            // sender is gone the consumer may change the integrator.
            drop(integrator);
            drop(task);
            drop(tx);
        });
    }
    drop(tx);
    StreamingBatch {
        receiver: rx,
        total_files: filenames.len(),
        completed,
        handles: Vec::new(),
    }
}

/// Integrate queued items on every worker of `pool`, streaming results.
///
/// The result channel is unbounded, so a worker never blocks on a consumer
/// that stopped reading: the shared cached pool must stay free for
/// synchronous batches. Memory is bounded upstream instead, by whatever feeds
/// `work_rx` (the Python iterator producer waits for credit from `control`).
/// Cancelling `control` makes every worker return within [`CANCEL_POLL`] (or
/// after the item it is integrating), releasing its share of `integrator`.
pub(crate) fn process_queue_with_pool<T>(
    work_rx: Receiver<WorkItem>,
    integrator: Arc<Integrator>,
    task: T,
    pool: Arc<FileWorkers>,
    control: Arc<StreamControl>,
) -> StreamingBatch<T::Output>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let work_rx = Arc::new(Mutex::new(work_rx));
    let completed = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    for worker in 0..pool.workers() {
        let work_rx = Arc::clone(&work_rx);
        let completed = Arc::clone(&completed);
        let tx = tx.clone();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        let control = Arc::clone(&control);
        // Registered before submission, so a consumer that closes the stream
        // at once still waits for this job.
        let active = control.worker_started();
        pool.submit(worker, move |buffer| {
            while let Some(item) = next_item(&work_rx, &control) {
                let (index, name, result) = match item {
                    WorkItem::Path { index, path } => {
                        let result = execute(&task, &integrator, buffer, &path);
                        (index, path, result)
                    }
                    WorkItem::Frame { index, name, frame } => {
                        let result =
                            catch_unwind(AssertUnwindSafe(|| task.execute(&integrator, &frame)))
                                .map_err(|_| "integration worker panicked".to_owned());
                        (index, name, result)
                    }
                };
                completed.fetch_add(1, Ordering::Relaxed);
                if control.is_cancelled() || tx.send((index, name, result)).is_err() {
                    break;
                }
            }
            // Release the integrator before reporting the job finished: a
            // closed stream waits for `active` and then lets the caller
            // change the integrator's calibration.
            drop(integrator);
            drop(task);
            drop(active);
        });
    }
    drop(tx);
    StreamingBatch {
        receiver: rx,
        total_files: 0,
        completed,
        handles: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn workers_reuse_threads_and_outlive_their_caller_without_leaking() {
        let pool = FileWorkers::new(2, 1).unwrap();
        let (tx, rx) = mpsc::channel();
        for pass in 0..2 {
            for worker in 0..2 {
                let tx = tx.clone();
                pool.submit(worker, move |_| {
                    tx.send((pass, worker, thread::current().id())).unwrap();
                });
            }
        }
        let mut ids = std::collections::HashMap::new();
        for _ in 0..4 {
            let (_, worker, id) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
            if let Some(old) = ids.insert(worker, id) {
                assert_eq!(old, id);
            }
        }
        let (release_tx, release_rx) = mpsc::channel();
        pool.submit(0, move |_| {
            release_rx.recv().unwrap();
        });
        // The last pool reference is released by a driver worker after Rayon
        // returns. Verify actual worker shutdown rather than just result delivery.
        let weak = Arc::downgrade(&pool);
        drop(pool);
        assert!(weak.strong_count() > 0);
        release_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while weak.strong_count() != 0 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(weak.strong_count(), 0);
        // New pool construction and dropping an unused pool also terminate.
        drop(FileWorkers::new(1, 2).unwrap());
    }

    /// Run `f` on a helper thread and fail instead of hanging the test suite.
    fn within<R: Send + 'static>(secs: u64, f: impl FnOnce() -> R + Send + 'static) -> R {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(secs))
            .expect("operation did not finish in time")
    }

    #[test]
    fn cancel_releases_workers_while_the_queue_stays_open() {
        use crate::geometry::Geometry;
        use crate::poni::Poni;
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 1.0;
        let integrator = Arc::new(
            Integrator::new(Geometry::new(2, 2, poni)).with_method(IntegrationMethod::Counting),
        );
        let pool = FileWorkers::new(2, 1).unwrap();
        let weak_pool = Arc::downgrade(&pool);
        let control = StreamControl::for_iterator();
        // The sender stays alive, like a producer stuck in a blocking
        // generator: without cancellation the workers would wait forever.
        let (work_tx, work_rx) = mpsc::sync_channel(4);
        let frame = Frame {
            name: "f".into(),
            data: vec![1.0; 4],
            dim1: 2,
            dim2: 2,
        };
        work_tx
            .send(WorkItem::Frame {
                index: 0,
                name: "f".into(),
                frame,
            })
            .unwrap();
        let batch = process_queue_with_pool(
            work_rx,
            Arc::clone(&integrator),
            Integration1D { n_bins: 4 },
            pool,
            Arc::clone(&control),
        );
        let rx = batch.into_receiver();
        let (index, _, result) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(index, 0);
        assert!(result.is_ok());
        assert!(Arc::strong_count(&integrator) > 1);

        control.cancel();
        let waiter = Arc::clone(&control);
        within(10, move || waiter.wait_workers_idle());
        assert_eq!(Arc::strong_count(&integrator), 1);
        // The jobs released the pool, which shuts its threads down, and the
        // work queue closed, so a blocked producer would see the error.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while weak_pool.strong_count() != 0 && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(weak_pool.strong_count(), 0);
        assert!(
            work_tx
                .send(WorkItem::Path {
                    index: 1,
                    path: String::new()
                })
                .is_err()
        );
        drop(rx);
    }

    #[test]
    fn credit_window_bounds_items_in_flight() {
        let control = StreamControl::for_iterator();
        assert!(control.wait_for_credit(0, 2));
        assert!(control.wait_for_credit(1, 2));
        // Item 2 needs the first result delivered.
        let waiter = Arc::clone(&control);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || tx.send(waiter.wait_for_credit(2, 2)).unwrap());
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
        control.deliver(1);
        assert!(rx.recv_timeout(Duration::from_secs(10)).unwrap());
        // Cancellation releases a waiting producer with `false`.
        let waiter = Arc::clone(&control);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || tx.send(waiter.wait_for_credit(10, 2)).unwrap());
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
        control.cancel();
        assert!(!rx.recv_timeout(Duration::from_secs(10)).unwrap());
        assert!(!control.wait_for_credit(0, 2));
    }
}
