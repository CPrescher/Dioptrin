//! Persistent worker-owned TIFF buffers and Rayon contexts.
#![cfg_attr(not(feature = "python"), allow(dead_code))]
use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

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

pub(crate) fn process_queue_with_pool<T>(
    work_rx: Receiver<WorkItem>,
    integrator: Arc<Integrator>,
    task: T,
    pool: Arc<FileWorkers>,
) -> StreamingBatch<T::Output>
where
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let work_rx = Arc::new(std::sync::Mutex::new(work_rx));
    let completed = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    for worker in 0..pool.workers() {
        let work_rx = Arc::clone(&work_rx);
        let completed = Arc::clone(&completed);
        let tx = tx.clone();
        let integrator = Arc::clone(&integrator);
        let task = task.clone();
        pool.submit(worker, move |buffer| {
            loop {
                let item = {
                    let guard = work_rx.lock().expect("work queue lock poisoned");
                    guard.recv()
                };
                let (index, name, result) = match item {
                    Ok(WorkItem::Path { index, path }) => {
                        let result = execute(&task, &integrator, buffer, &path);
                        (index, path, result)
                    }
                    Ok(WorkItem::Frame { index, name, frame }) => {
                        let result =
                            catch_unwind(AssertUnwindSafe(|| task.execute(&integrator, &frame)))
                                .map_err(|_| "integration worker panicked".to_owned());
                        (index, name, result)
                    }
                    Err(_) => break,
                };
                completed.fetch_add(1, Ordering::Relaxed);
                if tx.send((index, name, result)).is_err() {
                    break;
                }
            }
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
}
