//! `batch::process_streaming` leaves no threads behind after many streams are opened and
//! closed. A binary of its own, so no other test's threads are counted.
use std::sync::Arc;
use std::time::{Duration, Instant};

use dioptrin::batch::{Integration1D, StreamOptions, process_streaming};
use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator};
use dioptrin::poni::Poni;

/// Threads of this process, where the platform makes that easy to read.
fn thread_count() -> Option<usize> {
    if cfg!(target_os = "linux") {
        return std::fs::read_dir("/proc/self/task").ok().map(|d| d.count());
    }
    if cfg!(target_os = "macos") {
        // One header line, then one line per thread.
        let out = std::process::Command::new("ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        return Some(String::from_utf8_lossy(&out.stdout).lines().count() - 1);
    }
    None
}

fn frames() -> impl Iterator<Item = Frame> + Send + 'static {
    (0..).map(|i| Frame {
        name: format!("f{i}"),
        data: vec![i as f64; 4],
        dim1: 2,
        dim2: 2,
    })
}

fn run_streams(integrator: &Arc<Integrator>, cycles: usize) {
    for cycle in 0..cycles {
        let options = StreamOptions::new(1 + cycle % 4, 1);
        let task = Integration1D { n_bins: 4 };
        if cycle % 2 == 0 {
            // Endless input, closed early.
            let mut results =
                process_streaming(frames(), Arc::clone(integrator), task, options).unwrap();
            for _ in 0..cycle % 5 {
                results.next().unwrap().2.unwrap();
            }
        } else {
            // Finite input, read to the end.
            let n = process_streaming(frames().take(6), Arc::clone(integrator), task, options)
                .unwrap()
                .count();
            assert_eq!(n, 6);
        }
        assert_eq!(Arc::strong_count(integrator), 1);
    }
}

#[test]
fn many_streams_leave_no_threads_behind() {
    let Some(_) = thread_count() else {
        eprintln!("thread count not available on this platform; skipped");
        return;
    };
    let mut poni = Poni::new();
    poni.pixel1 = 1.0;
    poni.pixel2 = 1.0;
    poni.distance = 1.0;
    let integrator = Arc::new(
        Integrator::new(Geometry::new(2, 2, poni)).with_method(IntegrationMethod::Counting),
    );
    // Warm up (lazy statics, the integrator's caches), then count. Threads of
    // a closed stream exit just after it returns, so wait (bounded) for the
    // count to come down rather than expecting it at once.
    run_streams(&integrator, 4);
    let settle = |limit: usize| {
        let start = Instant::now();
        loop {
            let now = thread_count().unwrap();
            if now <= limit || start.elapsed() > Duration::from_secs(20) {
                return now;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    let baseline = settle(2);
    eprintln!("threads: baseline {baseline}");
    run_streams(&integrator, 300);
    let after = settle(baseline);
    assert!(
        after <= baseline,
        "{after} threads after 300 streams, {baseline} before"
    );
}
