//! Tests of the bounded stream. Timeouts below only bound how long a broken
//! stream may hang a test; no assertion depends on a sleep.
use super::*;
use crate::geometry::Geometry;
use crate::integration::MaskMode;
use crate::mask::Mask;
use crate::poni::Poni;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(20);

fn small_poni() -> Poni {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 1.0;
    poni
}

fn small_integrator() -> Arc<Integrator> {
    Arc::new(
        Integrator::new(Geometry::new(2, 2, small_poni())).with_method(IntegrationMethod::Counting),
    )
}

fn small_frame(i: usize) -> Frame {
    Frame {
        name: format!("f{i}"),
        data: vec![i as f64; 4],
        dim1: 2,
        dim2: 2,
    }
}

/// Poll `done` until it holds; panics after [`TIMEOUT`].
fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < TIMEOUT, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

/// Counts its executions; panics for frames named "panic"; spends `work` per
/// item to stand in for a real integration.
#[derive(Clone)]
struct Probe {
    executed: Arc<AtomicUsize>,
    work: Duration,
    /// When set, every item waits until the gate is opened.
    gate: Option<Arc<Gate>>,
}

#[derive(Default)]
struct Gate {
    open: std::sync::Mutex<bool>,
    opened: std::sync::Condvar,
}

impl Gate {
    fn pass(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
    }
    fn open(&self) {
        *self.open.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.opened.notify_all();
    }
}

struct OpenOnDrop(Arc<Gate>);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

impl Probe {
    fn new(work: Duration) -> Self {
        Self {
            executed: Arc::new(AtomicUsize::new(0)),
            work,
            gate: None,
        }
    }
    fn gated(gate: &Arc<Gate>) -> Self {
        Self {
            gate: Some(Arc::clone(gate)),
            ..Self::new(Duration::ZERO)
        }
    }
}

impl IntegrationTask for Probe {
    type Output = String;
    fn execute(&self, _: &Integrator, frame: &Frame) -> String {
        self.executed.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.pass();
        }
        if !self.work.is_zero() {
            thread::sleep(self.work);
        }
        assert!(frame.name != "panic", "probe asked to panic");
        frame.name.clone()
    }
}

/// An endless (or `limit`-long) input of small frames that counts how many
/// items the stream took, checks every take against the read-ahead bound and
/// records when the stream drops it.
struct Source {
    limit: usize,
    pulled: Arc<AtomicUsize>,
    /// Incremented by the consumer *before* each `next`, so it is never below
    /// the number of results the stream has handed out.
    taken: Arc<AtomicUsize>,
    window: usize,
    violation: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
    panic_at: Option<usize>,
}

#[derive(Clone, Default)]
struct SourceState {
    pulled: Arc<AtomicUsize>,
    taken: Arc<AtomicUsize>,
    violation: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

impl SourceState {
    fn source(&self, limit: usize, window: usize) -> Source {
        Source {
            limit,
            pulled: Arc::clone(&self.pulled),
            taken: Arc::clone(&self.taken),
            window,
            violation: Arc::clone(&self.violation),
            dropped: Arc::clone(&self.dropped),
            panic_at: None,
        }
    }
    fn pulled(&self) -> usize {
        self.pulled.load(Ordering::SeqCst)
    }
}

impl Iterator for Source {
    type Item = Frame;
    fn next(&mut self) -> Option<Frame> {
        let index = self.pulled.load(Ordering::SeqCst);
        if index >= self.limit {
            return None;
        }
        if Some(index) == self.panic_at {
            panic!("source asked to panic");
        }
        // Taking item `index` means index + 1 items taken from the input.
        if index + 1
            > self
                .taken
                .load(Ordering::SeqCst)
                .saturating_add(self.window)
        {
            self.violation.store(true, Ordering::SeqCst);
        }
        self.pulled.fetch_add(1, Ordering::SeqCst);
        Some(small_frame(index))
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn read_ahead_limit_defaults_to_four_per_worker() {
    assert_eq!(StreamOptions::new(3, 1).read_ahead_limit(), 12);
    assert_eq!(StreamOptions::new(0, 1).read_ahead_limit(), 4);
    assert_eq!(StreamOptions::new(3, 1).read_ahead(5).read_ahead_limit(), 5);
    assert_eq!(StreamOptions::new(3, 1).read_ahead(0).read_ahead_limit(), 1);
}

/// A consumer that pauses after every result never has more than the
/// read-ahead limit taken from the input ahead of it, and the stream does
/// fill that window (so the bound, not slowness, is what stops it).
#[test]
fn read_ahead_bound_holds_for_a_slow_consumer() {
    for workers in [1, 2, 4, 8] {
        for read_ahead in [None, Some(1), Some(3), Some(5)] {
            let mut options = StreamOptions::new(workers, 1);
            if let Some(n) = read_ahead {
                options = options.read_ahead(n);
            }
            let window = options.read_ahead_limit();
            let state = SourceState::default();
            let probe = Probe::new(Duration::ZERO);
            let mut results = process_streaming(
                state.source(usize::MAX, window),
                small_integrator(),
                probe.clone(),
                options,
            )
            .unwrap();
            for expected in 0..25 {
                let taken = state.taken.fetch_add(1, Ordering::SeqCst) + 1;
                let (index, name, result) = results.next().unwrap();
                assert_eq!((index, name.as_str()), (expected, &*format!("f{expected}")));
                assert_eq!(result.unwrap(), name);
                // The pause: wait until the stream has filled its window.
                wait_until("the read-ahead window to fill", || {
                    state.pulled() >= taken + window
                });
                assert!(probe.executed.load(Ordering::SeqCst) <= state.pulled());
            }
            assert!(
                !state.violation.load(Ordering::SeqCst),
                "{workers} workers, window {window}: took more than the window ahead"
            );
            assert_eq!(
                state.pulled(),
                25 + window,
                "{workers} workers, window {window}"
            );
        }
    }
}

/// Dropping an endless stream releases the integrator, finishes only the
/// items the workers had started, leaves the queued items unintegrated and
/// drops the input. The workers are held on a gate until the stream is
/// cancelled, so the counts are exact.
#[test]
fn dropping_the_stream_stops_work_and_releases_the_integrator() {
    for workers in [1, 2, 4] {
        let integrator = small_integrator();
        let state = SourceState::default();
        let gate = Arc::new(Gate::default());
        let probe = Probe::gated(&gate);
        // A window the test never reaches, so the work queue fills.
        let options = StreamOptions::new(workers, 1).read_ahead(1 << 40);
        let results = process_streaming(
            state.source(usize::MAX, usize::MAX),
            Arc::clone(&integrator),
            probe.clone(),
            options,
        )
        .unwrap();
        // Opens the gate if an assertion below fails. Declared after
        // `results`, so it is dropped first while unwinding and the stream's
        // drop does not wait forever for the gated workers.
        let _open_on_unwind = OpenOnDrop(Arc::clone(&gate));
        // Every worker holds one item at the gate; the queue (2 per worker)
        // is full behind them.
        wait_until("the workers to start and the queue to fill", || {
            probe.executed.load(Ordering::SeqCst) == workers && state.pulled() >= 3 * workers
        });
        let control = Arc::clone(&results.control);
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(results);
            let _ = done_tx.send(());
        });
        wait_until("the stream to be cancelled", || control.is_cancelled());
        gate.open();
        done_rx
            .recv_timeout(TIMEOUT)
            .expect("dropping the stream did not return");
        assert_eq!(Arc::strong_count(&integrator), 1, "integrator still shared");
        assert_eq!(probe.executed.load(Ordering::SeqCst), workers);
        wait_until("the input to be dropped", || {
            state.dropped.load(Ordering::SeqCst)
        });
        assert!(state.pulled() >= 3 * workers);
        assert_eq!(probe.executed.load(Ordering::SeqCst), workers);
    }
}

/// Drop `results` on a helper thread; panics if that does not return.
fn drop_within_timeout<T: Send + 'static>(results: BatchStream<T>) {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        drop(results);
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(TIMEOUT)
        .expect("dropping the stream did not return");
}

/// Dropping a stream whose window is full: the input thread waits for
/// credit and the workers wait for input, so only cancellation frees them.
#[test]
fn dropping_a_stream_with_a_full_window() {
    for workers in [1, 3] {
        let integrator = small_integrator();
        let state = SourceState::default();
        let probe = Probe::new(Duration::ZERO);
        let mut results = process_streaming(
            state.source(usize::MAX, 4),
            Arc::clone(&integrator),
            probe.clone(),
            StreamOptions::new(workers, 1).read_ahead(4),
        )
        .unwrap();
        state.taken.fetch_add(1, Ordering::SeqCst);
        results.next().unwrap().2.unwrap();
        wait_until("the window to fill and the workers to finish it", || {
            state.pulled() == 5 && probe.executed.load(Ordering::SeqCst) == 5
        });
        drop_within_timeout(results);
        assert_eq!(Arc::strong_count(&integrator), 1);
        wait_until("the input to be dropped", || {
            state.dropped.load(Ordering::SeqCst)
        });
        assert_eq!(state.pulled(), 5);
        assert!(!state.violation.load(Ordering::SeqCst));
    }
}

#[test]
fn close_stops_the_stream_and_next_returns_none() {
    let integrator = small_integrator();
    let state = SourceState::default();
    let mut results = process_streaming(
        state.source(usize::MAX, usize::MAX),
        Arc::clone(&integrator),
        Integration1D { n_bins: 4 },
        StreamOptions::new(2, 1),
    )
    .unwrap();
    assert_eq!(results.next().unwrap().0, 0);
    results.close();
    assert_eq!(Arc::strong_count(&integrator), 1);
    assert!(results.next().is_none());
    results.close();
    assert!(results.next().is_none());
    wait_until("the input to be dropped", || {
        state.dropped.load(Ordering::SeqCst)
    });
}

fn realistic_integrator() -> Arc<Integrator> {
    let mut poni = Poni::new();
    poni.distance = 0.1;
    poni.poni1 = 0.003;
    poni.poni2 = 0.002;
    poni.pixel1 = 1e-4;
    poni.pixel2 = 1e-4;
    poni.wavelength = 3e-11;
    Arc::new(Integrator::new(Geometry::new(48, 64, poni)))
}

fn write_tiff(tag: &str, i: usize) -> String {
    use tiff::encoder::{TiffEncoder, colortype};
    let path = std::env::temp_dir().join(format!(
        "dioptrin-bounded-{tag}-{}-{i}.tif",
        std::process::id()
    ));
    let mut enc = TiffEncoder::new(std::fs::File::create(&path).unwrap()).unwrap();
    let data: Vec<u16> = (0..64 * 48)
        .map(|v| ((v * 7 + i * 131) % 4093) as u16)
        .collect();
    enc.write_image::<colortype::Gray16>(64, 48, &data).unwrap();
    path.to_str().unwrap().to_owned()
}

fn copy(frame: &Frame) -> Frame {
    Frame {
        name: frame.name.clone(),
        data: frame.data.clone(),
        dim1: frame.dim1,
        dim2: frame.dim2,
    }
}

fn copies(frames: &[Frame]) -> Vec<Frame> {
    frames.iter().map(copy).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Paths and frames come back in input order with results bit-identical to
/// `process_files` and to running the task on each frame in turn, for
/// several worker counts (more workers than items included).
#[test]
fn results_are_in_order_and_identical_to_process_files() {
    let integrator = realistic_integrator();
    let files: Vec<String> = (0..9).map(|i| write_tiff("order", i)).collect();
    let task = Integration1D { n_bins: 50 };
    let reference = process_files(&files, Arc::clone(&integrator), task, 3, 1);
    let frames: Vec<Frame> = files
        .iter()
        .map(|f| Frame::from_tiff(f.clone()).unwrap())
        .collect();
    for workers in [1, 2, 4, 16] {
        let options = StreamOptions::new(workers, 1);
        let streamed: Vec<_> =
            process_streaming(files.clone(), Arc::clone(&integrator), task, options)
                .unwrap()
                .collect();
        assert_eq!(streamed.len(), files.len());
        for (i, ((index, name, result), (ref_name, ref_result))) in
            streamed.iter().zip(&reference).enumerate()
        {
            assert_eq!((*index, name), (i, ref_name));
            let (radial, intensity) = result.as_ref().unwrap();
            let (ref_radial, ref_intensity) = ref_result.as_ref().unwrap();
            assert_eq!(bits(radial), bits(ref_radial));
            assert_eq!(bits(intensity), bits(ref_intensity));
        }
        let streamed: Vec<_> =
            process_streaming(copies(&frames), Arc::clone(&integrator), task, options)
                .unwrap()
                .collect();
        assert_eq!(streamed.len(), frames.len());
        for (i, (index, name, result)) in streamed.into_iter().enumerate() {
            assert_eq!((index, &name), (i, &frames[i].name));
            let (radial, intensity) = result.unwrap();
            let (ref_radial, ref_intensity) = task.execute(&integrator, &frames[i]);
            assert_eq!(bits(&radial), bits(&ref_radial));
            assert_eq!(bits(&intensity), bits(&ref_intensity));
        }
        let task2d = Integration2D {
            n_radial: 30,
            n_azimuthal: 12,
        };
        let streamed: Vec<_> =
            process_streaming(files.clone(), Arc::clone(&integrator), task2d, options)
                .unwrap()
                .collect();
        for (i, (index, _, result)) in streamed.into_iter().enumerate() {
            assert_eq!(index, i);
            let expected = task2d.execute(&integrator, &frames[i]);
            assert_eq!(bits(&result.unwrap()), bits(&expected));
        }
    }
    for f in files {
        let _ = std::fs::remove_file(f);
    }
}

/// The CeO2 Pilatus 1M image: streamed paths and frames match
/// `process_files` bit for bit.
#[test]
fn pilatus_results_match_process_files() {
    crate::util::download_pilatus_test_file();
    let poni = Poni::load("test_data/v1.poni").unwrap();
    let integrator = Arc::new(Integrator::new(Geometry::new(0, 0, poni)));
    let path = String::from("test_data/CeO2_Pilatus1M.tif");
    let files = vec![path.clone(), path.clone(), path];
    let task = Integration1D { n_bins: 1000 };
    let reference = process_files(&files, Arc::clone(&integrator), task, 2, 1);
    let frame = Frame::from_tiff(files[0].clone()).unwrap();
    let options = StreamOptions::new(2, 1);
    let from_paths: Vec<_> =
        process_streaming(files.clone(), Arc::clone(&integrator), task, options)
            .unwrap()
            .collect();
    let from_frames: Vec<_> = process_streaming(
        copies(&[copy(&frame), copy(&frame), frame]),
        Arc::clone(&integrator),
        task,
        options,
    )
    .unwrap()
    .collect();
    for ((streamed, framed), (_, reference)) in from_paths.iter().zip(&from_frames).zip(&reference)
    {
        let reference = reference.as_ref().unwrap();
        for (_, _, result) in [streamed, framed] {
            let result = result.as_ref().unwrap();
            assert_eq!(bits(&result.0), bits(&reference.0));
            assert_eq!(bits(&result.1), bits(&reference.1));
        }
    }
}

#[test]
fn per_item_errors_do_not_end_the_stream() {
    let integrator = small_integrator();
    let mut panicking = small_frame(2);
    panicking.name = "panic".to_owned();
    let input = vec![
        BatchInput::from(small_frame(0)),
        BatchInput::from("/nonexistent/dioptrin/missing.tif"),
        BatchInput::from(panicking),
        BatchInput::from(small_frame(3)),
    ];
    let results: Vec<_> = process_streaming(
        input,
        integrator,
        Probe::new(Duration::ZERO),
        StreamOptions::new(2, 1),
    )
    .unwrap()
    .collect();
    let indices: Vec<_> = results.iter().map(|r| r.0).collect();
    assert_eq!(indices, [0, 1, 2, 3]);
    assert_eq!(results[0].2.as_deref(), Ok("f0"));
    assert_eq!(results[1].1, "/nonexistent/dioptrin/missing.tif");
    assert!(results[1].2.is_err());
    assert_eq!(results[2].1, "panic");
    assert_eq!(
        results[2].2.as_ref().err().map(String::as_str),
        Some("integration worker panicked")
    );
    assert_eq!(results[3].2.as_deref(), Ok("f3"));
}

#[test]
fn a_mask_mismatched_item_fails_alone() {
    for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
        let integrator = Arc::new(
            Integrator::new(Geometry::new(2, 2, small_poni()))
                .with_method(IntegrationMethod::Counting)
                .with_mask_mode(mode)
                .with_mask(Mask {
                    data: vec![0; 4],
                    dim1: 2,
                    dim2: 2,
                }),
        );
        let wrong = Frame {
            name: "wrong".to_owned(),
            data: vec![1.0; 6],
            dim1: 3,
            dim2: 2,
        };
        let input = vec![small_frame(0), wrong, small_frame(2)];
        let results: Vec<_> = process_streaming(
            input,
            integrator,
            Integration1D { n_bins: 4 },
            StreamOptions::new(2, 1),
        )
        .unwrap()
        .collect();
        assert_eq!(results.len(), 3);
        assert!(results[0].2.is_ok());
        assert_eq!(
            results[1].2.as_ref().err().map(String::as_str),
            Some("image shape (3, 2) does not match mask shape (2, 2)")
        );
        assert!(results[2].2.is_ok());
    }
}

#[test]
fn a_panicking_input_ends_the_stream_with_an_error() {
    let state = SourceState::default();
    let mut source = state.source(usize::MAX, usize::MAX);
    source.panic_at = Some(3);
    let results: Vec<_> = process_streaming(
        source,
        small_integrator(),
        Integration1D { n_bins: 4 },
        StreamOptions::new(2, 1),
    )
    .unwrap()
    .collect();
    assert_eq!(results.len(), 4);
    assert!(results[..3].iter().all(|r| r.2.is_ok()));
    assert_eq!(results[3].0, 3);
    assert_eq!(
        results[3].2.as_ref().err().map(String::as_str),
        Some("reading input item 3 panicked: source asked to panic")
    );
}

#[test]
fn empty_input_ends_at_once() {
    let integrator = small_integrator();
    let mut results = process_streaming(
        Vec::<String>::new(),
        Arc::clone(&integrator),
        Integration1D { n_bins: 4 },
        StreamOptions::new(4, 1),
    )
    .unwrap();
    assert!(results.next().is_none());
    assert!(results.next().is_none());
    assert_eq!(Arc::strong_count(&integrator), 1);
}

#[test]
fn histogram_2d_tasks_get_one_error_per_item() {
    let integrator = Arc::new(
        Integrator::new(Geometry::new(2, 2, small_poni()))
            .with_method(IntegrationMethod::Histogram),
    );
    let task = Integration2D {
        n_radial: 4,
        n_azimuthal: 4,
    };
    let results: Vec<_> = process_streaming(
        (0..5).map(small_frame),
        Arc::clone(&integrator),
        task,
        StreamOptions::new(2, 1).read_ahead(2),
    )
    .unwrap()
    .collect();
    assert_eq!(results.len(), 5);
    for (i, (index, name, result)) in results.into_iter().enumerate() {
        assert_eq!((index, name), (i, format!("f{i}")));
        assert_eq!(
            result.err().as_deref(),
            Some("integrate2d is not supported for Histogram method")
        );
    }
    assert_eq!(Arc::strong_count(&integrator), 1);
}

/// Many short streams, consumed fully, partly or not at all, each finish
/// with the integrator released.
#[test]
fn repeated_open_and_close_cycles() {
    let integrator = small_integrator();
    for cycle in 0..200 {
        let state = SourceState::default();
        let limit = if cycle % 3 == 0 { 5 } else { usize::MAX };
        let mut results = process_streaming(
            state.source(limit, usize::MAX),
            Arc::clone(&integrator),
            Integration1D { n_bins: 4 },
            StreamOptions::new(1 + cycle % 4, 1),
        )
        .unwrap();
        for _ in 0..cycle % 7 {
            if results.next().is_none() {
                break;
            }
        }
        drop(results);
        assert_eq!(Arc::strong_count(&integrator), 1, "cycle {cycle}");
        wait_until("the input to be dropped", || {
            state.dropped.load(Ordering::SeqCst)
        });
    }
}
