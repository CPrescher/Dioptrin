//! Bounded, cancellable streaming of a batch: [`process_streaming`] and
//! [`BatchStream`].
use super::file_workers::{FileWorkers, QueueResult, StreamControl, submit_queue_jobs};
use super::*;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::SyncSender;

/// One input item of a [`process_streaming`]: a TIFF file to load, or a frame already in
/// memory.
///
/// Converts from `String` and `&str` (a path) and from [`Frame`], so the input
/// of [`process_streaming`] can be a `Vec<String>` of paths, a `Vec<Frame>`, or any
/// iterator over those.
#[non_exhaustive]
pub enum BatchInput {
    /// A TIFF file, loaded by the worker that integrates it. Its result is
    /// named after the path.
    Path(String),
    /// A frame in memory. Its result is named after [`Frame::name`].
    Frame(Frame),
}

impl From<String> for BatchInput {
    fn from(path: String) -> Self {
        Self::Path(path)
    }
}

impl From<&str> for BatchInput {
    fn from(path: &str) -> Self {
        Self::Path(path.to_owned())
    }
}

impl From<Frame> for BatchInput {
    fn from(frame: Frame) -> Self {
        Self::Frame(frame)
    }
}

/// Worker count, threads per worker and read-ahead of a [`process_streaming`].
///
/// ```
/// use dioptrin::batch::StreamOptions;
///
/// // 4 workers with 2 Rayon threads each; at most 16 items ahead of the consumer.
/// let options = StreamOptions::new(4, 2);
/// assert_eq!(options.read_ahead_limit(), 16);
///
/// // Keep fewer results waiting (each may be a large 2D cake).
/// let options = StreamOptions::new(4, 2).read_ahead(6);
/// assert_eq!(options.read_ahead_limit(), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamOptions {
    num_workers: usize,
    num_threads: usize,
    read_ahead: Option<usize>,
}

impl StreamOptions {
    /// Read-ahead per worker when none is set with [`Self::read_ahead`].
    pub const DEFAULT_READ_AHEAD_PER_WORKER: usize = 4;

    /// `num_workers` parallel workers (at least 1), each with its own Rayon
    /// pool of `num_threads` threads (0 lets Rayon choose), as for
    /// [`process_files`].
    pub fn new(num_workers: usize, num_threads: usize) -> Self {
        Self {
            num_workers: num_workers.max(1),
            num_threads,
            read_ahead: None,
        }
    }

    /// Let the stream run at most `items` items (at least 1) ahead of its
    /// consumer.
    ///
    /// An item counts from the moment the stream takes it from the input
    /// until the consumer receives its result, so this also bounds how many
    /// inputs are held (a queued [`Frame`] holds its image) and how many
    /// finished results wait to be received. The default,
    /// [`Self::DEFAULT_READ_AHEAD_PER_WORKER`] items per worker, keeps every
    /// worker busy while the consumer keeps up; a limit below the worker
    /// count leaves workers idle. `usize::MAX` removes the limit: the input
    /// is then read as fast as the workers integrate it, and results the
    /// consumer has not received accumulate without bound.
    pub fn read_ahead(mut self, items: usize) -> Self {
        self.read_ahead = Some(items.max(1));
        self
    }

    /// The read-ahead limit in effect.
    pub fn read_ahead_limit(&self) -> usize {
        self.read_ahead.unwrap_or(
            self.num_workers
                .saturating_mul(Self::DEFAULT_READ_AHEAD_PER_WORKER),
        )
    }
}

/// Integrate `input` in parallel and stream the results in input order, with
/// bounded read-ahead and early stop.
///
/// Items are taken from `input` lazily, on a thread of the stream, only while
/// fewer than [`StreamOptions::read_ahead_limit`] taken items have not yet been
/// received by the consumer. A consumer that reads slowly, or stops reading,
/// therefore holds at most that many inputs and results, and `input` may be
/// endless (a live feed of frames).
///
/// The returned [`BatchStream`] yields `(index, name, result)` for every input
/// item in order: `index` is the item's position in `input`, `name` its path or
/// [`Frame::name`]. As for the other batch helpers, an item that cannot be
/// loaded, whose shape the integrator's mask or detector does not accept, or
/// whose integration panics gets an `Err` and the stream continues. A 2D task
/// with the Histogram method (see [`IntegrationTask::integrate2d_method`]), or
/// with an explicit radial range the 2D grid cannot use (see
/// [`IntegrationTask::check_2d`]), is not run: every item gets that error. If `input` itself panics, the item at
/// that position gets an `Err` and the stream ends there.
///
/// Dropping the stream, or calling [`BatchStream::close`], stops it: no more
/// items are taken from `input`, queued items are not integrated, and the call
/// returns once every worker has released `integrator`. That takes at most
/// the item each worker is integrating, or, while the thread reading `input`
/// is inside `input.next()`, up to 50 ms for idle workers to notice. That
/// thread exits when its current call to `input.next()` returns; it does not
/// hold `integrator`.
///
/// Returns `Err` if a worker's thread or Rayon pool, or the thread reading
/// `input`, cannot be created.
///
/// To wait for all results of a list of files at once, use [`process_files`].
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use dioptrin::batch::{process_streaming, Integration1D, StreamOptions};
/// use dioptrin::frame::Frame;
/// use dioptrin::geometry::Geometry;
/// use dioptrin::integration::Integrator;
/// use dioptrin::poni::Poni;
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
/// let integrator = Arc::new(Integrator::new(Geometry::new(10, 10, poni)));
///
/// // An endless source of frames; with paths, pass a Vec<String> instead.
/// let frames = (0..).map(|i| Frame {
///     name: format!("frame {i}"),
///     data: vec![i as f64; 100],
///     dim1: 10,
///     dim2: 10,
/// });
///
/// let task = Integration1D { n_bins: 20 };
/// let options = StreamOptions::new(2, 1).read_ahead(4);
/// let mut results = process_streaming(frames, Arc::clone(&integrator), task, options).unwrap();
/// for expected in 0..10 {
///     let (index, name, result) = results.next().unwrap();
///     assert_eq!(index, expected);
///     assert_eq!(name, format!("frame {expected}"));
///     let (_radial, _intensity) = result.unwrap();
/// }
/// // Stop the endless stream; the workers have released the integrator.
/// drop(results);
/// assert_eq!(Arc::strong_count(&integrator), 1);
/// ```
pub fn process_streaming<I, T>(
    input: I,
    integrator: Arc<Integrator>,
    task: T,
    options: StreamOptions,
) -> Result<BatchStream<T::Output>, String>
where
    I: IntoIterator,
    I::IntoIter: Send + 'static,
    I::Item: Into<BatchInput>,
    T: IntegrationTask + Clone + 'static,
    T::Output: 'static,
{
    let control = StreamControl::new();
    let window = options.read_ahead_limit();
    let shared = Arc::new(Shared {
        clean_end: AtomicUsize::new(NOT_ENDED),
    });
    let (result_tx, result_rx) = mpsc::channel();
    let rejected = rejection(&integrator, &task);
    let (work_tx, pool) = if rejected.is_some() {
        // Every item gets the error; the reading thread sends it.
        (None, None)
    } else {
        let pool = FileWorkers::new(options.num_workers, options.num_threads)?;
        let (work_tx, work_rx) = mpsc::sync_channel(options.num_workers.saturating_mul(2));
        submit_queue_jobs(work_rx, &integrator, &task, &pool, &control, &result_tx);
        (Some(work_tx), Some(pool))
    };
    drop(integrator);
    let reader = Reader {
        input: input.into_iter(),
        work_tx,
        result_tx,
        rejected,
        control: Arc::clone(&control),
        shared: Arc::clone(&shared),
        window,
    };
    let mut stream = BatchStream {
        results: Some(result_rx),
        buffer: BTreeMap::new(),
        next_index: 0,
        control,
        shared,
        pool,
        finished: false,
    };
    if let Err(err) = thread::Builder::new()
        .name("dioptrin-stream-input".to_owned())
        .spawn(move || reader.run())
    {
        stream.close();
        return Err(format!("failed to start the stream's input thread: {err}"));
    }
    Ok(stream)
}

/// [`Shared::clean_end`] while the reading thread has not ended cleanly.
const NOT_ENDED: usize = usize::MAX;

/// State the reading thread hands to the consumer.
struct Shared {
    /// Number of items the reading thread sent on (to the workers, or as an
    /// error result), stored once it ended cleanly: at the end of the input or
    /// after reporting a panic of the input. [`NOT_ENDED`] otherwise.
    clean_end: AtomicUsize,
}

/// The stream's thread that takes items from the input within the credit
/// window and queues them for the workers.
struct Reader<I, O> {
    input: I,
    /// None when every item is answered with `rejected`.
    work_tx: Option<SyncSender<WorkItem>>,
    result_tx: Sender<QueueResult<O>>,
    rejected: Option<String>,
    control: Arc<StreamControl>,
    shared: Arc<Shared>,
    window: usize,
}

impl<I, O> Reader<I, O>
where
    I: Iterator,
    I::Item: Into<BatchInput>,
{
    fn run(self) {
        let Self {
            mut input,
            work_tx,
            result_tx,
            rejected,
            control,
            shared,
            window,
        } = self;
        let mut index = 0;
        let clean = loop {
            if !control.wait_for_credit(index, window) {
                break false; // Stream closed
            }
            let item = match catch_unwind(AssertUnwindSafe(|| input.next().map(Into::into))) {
                Ok(Some(item)) => item,
                Ok(None) => break true,
                Err(payload) => {
                    let message = format!(
                        "reading input item {index} panicked: {}",
                        panic_text(&*payload)
                    );
                    let _ = result_tx.send((index, String::new(), Err(message)));
                    index += 1;
                    break true;
                }
            };
            let item = match item {
                BatchInput::Path(path) => WorkItem::Path { index, path },
                BatchInput::Frame(frame) => WorkItem::Frame {
                    index,
                    name: frame.name.clone(),
                    frame,
                },
            };
            let sent = match (&work_tx, &rejected) {
                (Some(work_tx), _) => work_tx.send(item).is_ok(),
                (None, rejected) => {
                    let (WorkItem::Path { path: name, .. } | WorkItem::Frame { name, .. }) = item;
                    let message = rejected.clone().unwrap_or_default();
                    result_tx.send((index, name, Err(message))).is_ok()
                }
            };
            if !sent {
                break false; // Workers or consumer gone
            }
            index += 1;
        };
        if clean {
            shared.clean_end.store(index, Ordering::SeqCst);
        }
        // Closing the queue lets the workers finish; the result channel
        // closes once they and this thread have dropped their senders, after
        // `clean_end` is stored.
        drop(work_tx);
        drop(result_tx);
        // The input may hold the caller's data; a panic dropping it ends only
        // this thread.
        let _ = catch_unwind(AssertUnwindSafe(move || drop(input)));
    }
}

/// Text of a caught panic payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Results of a [`process_streaming`], in input order: an [`Iterator`] over
/// `(index, name, result)`.
///
/// Receiving a result lets the stream take one more item from its input.
/// Dropping the stream, or calling [`Self::close`], stops it (see [`process_streaming`]);
/// either may wait for the item each worker is integrating.
///
/// If the stream ends without a result for some item (a worker or the
/// thread reading the input stopped unexpectedly), that item gets an `Err`
/// and the iteration ends after it, rather than ending silently.
pub struct BatchStream<T> {
    /// None once the stream is finished or closed.
    results: Option<Receiver<QueueResult<T>>>,
    /// Results that arrived before an earlier item's.
    buffer: BTreeMap<usize, (String, Result<T, String>)>,
    next_index: usize,
    control: Arc<StreamControl>,
    shared: Arc<Shared>,
    /// Kept so the workers outlive a stream whose jobs already finished;
    /// None for a stream whose task is rejected.
    pool: Option<Arc<FileWorkers>>,
    finished: bool,
}

impl<T> BatchStream<T> {
    /// Stop the stream: take no more input, discard results not yet received,
    /// and return once every worker has released the integrator. Later calls
    /// to `next` return `None`. Called by `drop`.
    pub fn close(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.control.cancel();
        self.results = None;
        self.buffer.clear();
        self.control.wait_workers_idle();
        self.pool = None;
    }

    fn deliver(&mut self, name: String, result: Result<T, String>) -> QueueResult<T> {
        let index = self.next_index;
        self.next_index += 1;
        self.control.deliver(self.next_index);
        (index, name, result)
    }
}

impl<T> Iterator for BatchStream<T> {
    type Item = (usize, String, Result<T, String>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((name, result)) = self.buffer.remove(&self.next_index) {
                return Some(self.deliver(name, result));
            }
            let received = self.results.as_ref()?.recv();
            match received {
                Ok((index, name, result)) if index == self.next_index => {
                    return Some(self.deliver(name, result));
                }
                Ok((index, name, result)) => {
                    self.buffer.insert(index, (name, result));
                }
                Err(_) => {
                    // Every sender is gone, so the reading thread has ended.
                    let index = self.next_index;
                    let complete = self.shared.clean_end.load(Ordering::SeqCst) == index;
                    self.close();
                    return (!complete).then(|| {
                        let message = format!(
                            "batch stream ended without a result for item {index}: \
                             a worker or the thread reading the input stopped unexpectedly"
                        );
                        (index, String::new(), Err(message))
                    });
                }
            }
        }
    }
}

impl<T> std::iter::FusedIterator for BatchStream<T> {}

impl<T> Drop for BatchStream<T> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests;
