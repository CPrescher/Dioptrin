"""A live or abandoned generator-fed batch*_iter stream never blocks a later
synchronous batch on the same Integrator.

Each scenario runs in a subprocess with a timeout, so a regression (a GIL
deadlock that cannot be interrupted) fails the test instead of hanging the
suite. Synthetic geometry; TIFF inputs are written to a temporary directory.
"""
import subprocess
import sys
import textwrap
import unittest

TIMEOUT = 60

PRELUDE = """
import itertools, os, tempfile, threading, time
import numpy as np, tifffile, dioptrin
poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0032, poni2=0.0048,
            rot1=0, rot2=0, rot3=0, wavelength=1e-10)
ai = dioptrin.Integrator.from_poni_dict(poni)
SHAPE = (64, 96)
d = tempfile.mkdtemp()
paths = []
for i in range(4):
    p = os.path.join(d, f"a{i}.tif")
    tifffile.imwrite(p, np.full(SHAPE, i + 1, dtype=np.float32))
    paths.append(p)
pulled = itertools.count()
def frames(n=None, delay=0.0):
    for _ in (range(n) if n is not None else itertools.count()):
        next(pulled)
        if delay:
            time.sleep(delay)
        yield np.ones(SHAPE)
def check_sync_batches():
    r1 = ai.batch1d(paths, 10, num_workers=2)
    assert len(r1) == 4 and all(r.is_ok() for r in r1), [r.error for r in r1]
    r2 = ai.batch2d(paths, 10, 8, num_workers=2)
    assert len(r2) == 4 and all(r.is_ok() for r in r2), [r.error for r in r2]
    r3 = ai.batch1d([np.ones(SHAPE)] * 3, 10, num_workers=2)
    assert len(r3) == 3 and all(r.is_ok() for r in r3)
def os_threads():
    # OS threads of this process where it can be observed without extra
    # dependencies (Linux, macOS); None elsewhere.
    import subprocess, sys
    if os.path.isdir("/proc/self/task"):
        return len(os.listdir("/proc/self/task"))
    if sys.platform == "darwin":
        out = subprocess.run(["ps", "-M", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout
        return len(out.strip().splitlines()) - 1
    return None
def until(condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True
def setter_works(unit="2th_deg"):
    try:
        ai.set_unit(unit)
        return True
    except ValueError as e:
        assert "in use" in str(e), e
        return False
"""


class SyncBatchWithOpenStreamTests(unittest.TestCase):
    def run_scenario(self, body):
        script = PRELUDE + textwrap.dedent(body) + "\nprint('SCENARIO OK')\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"scenario did not finish within {TIMEOUT}s (deadlock)")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SCENARIO OK", proc.stdout, proc.stderr)

    def test_open_1d_iterator_then_batch1d_paths(self):
        self.run_scenario("""
            it = ai.batch1d_iter(frames(200), 10, num_workers=2)
            assert next(it).is_ok()
            r = ai.batch1d(paths, 10, num_workers=2)
            assert len(r) == 4 and all(x.is_ok() for x in r)
            assert sum(1 for _ in it) == 199
        """)

    def test_open_2d_iterator_then_batch2d_paths(self):
        self.run_scenario("""
            it = ai.batch2d_iter(frames(200), 10, 8, num_workers=2)
            assert next(it).is_ok()
            r = ai.batch2d(paths, 10, 8, num_workers=2)
            assert len(r) == 4 and all(x.is_ok() for x in r)
            assert sum(1 for _ in it) == 199
        """)

    def test_open_endless_iterators_do_not_block_sync_batches(self):
        # An endless stream (e.g. a live detector feed) that the caller is not
        # consuming must not hold the workers that synchronous batches use.
        self.run_scenario("""
            it1 = ai.batch1d_iter(frames(), 10, num_workers=2)
            it2 = ai.batch2d_iter(frames(), 10, 8, num_workers=2)
            assert next(it1).is_ok() and next(it2).is_ok()
            check_sync_batches()
            assert next(it1).is_ok() and next(it2).is_ok()
        """)

    def test_second_stream_is_not_queued_behind_an_endless_first(self):
        self.run_scenario("""
            first = ai.batch1d_iter(frames(), 10, num_workers=2)
            second = ai.batch1d_iter(frames(5), 10, num_workers=2)
            assert sum(1 for r in second if r.is_ok()) == 5
            assert next(first).is_ok()
        """)

    def test_abandoned_iterator_then_sync_batches(self):
        # Slow endless generator, dropped mid-stream: the batch returns and
        # the producer stops pulling from the generator soon afterwards.
        self.run_scenario("""
            it = ai.batch1d_iter(frames(delay=0.05), 10, num_workers=2)
            assert next(it).is_ok()
            del it
            check_sync_batches()
            deadline = time.monotonic() + 20
            while True:
                before = next(pulled)
                time.sleep(1.0)
                if next(pulled) == before + 1:
                    break
                assert time.monotonic() < deadline, "producer kept pulling"
        """)


class StreamShutdownTests(unittest.TestCase):
    """Closing a generator-fed stream stops its workers; a failing producer
    ends the stream with an exception; an unconsumed stream stops pulling."""

    run_scenario = SyncBatchWithOpenStreamTests.run_scenario

    def test_abandoned_stream_over_blocking_generator_releases_integrator(self):
        # A live feed blocks in its generator waiting for the next frame. The
        # dropped stream must stop its workers anyway: the setters work again
        # and repeated rounds leave at most the blocked producer threads.
        self.run_scenario("""
            release = threading.Event()
            def live_feed():
                yield np.ones(SHAPE)
                release.wait()
            rounds = 4
            check_sync_batches()  # start the shared pools before counting
            baseline = os_threads()
            for i in range(rounds):
                if i % 2:
                    it = ai.batch2d_iter(live_feed(), 10, 8, num_workers=4, num_threads=2)
                else:
                    it = ai.batch1d_iter(live_feed(), 10, num_workers=4, num_threads=2)
                assert next(it).is_ok()
                assert not setter_works(), "stream should hold the integrator"
                del it
                assert until(setter_works), f"round {i}: integrator still in use"
            check_sync_batches()
            if baseline is not None:
                # One producer per round stays inside release.wait(); worker
                # and Rayon threads (13 per round before the fix) are gone.
                limit = baseline + rounds + 4
                assert until(lambda: os_threads() <= limit), (baseline, os_threads())
            release.set()
            if baseline is not None:
                assert until(lambda: os_threads() <= baseline + 4), (baseline, os_threads())
        """)

    def test_producer_panic_raises_after_earlier_results(self):
        # Calling the unsendable Integrator from the producer thread raises a
        # pyo3 PanicException inside the generator, which pyo3 resumes as a
        # Rust panic in the producer. The stream must not end silently.
        self.run_scenario("""
            def misbehaving():
                yield np.ones(SHAPE)
                yield np.ones(SHAPE)
                ai.set_unit("2th_deg")  # wrong thread: panics
                yield np.ones(SHAPE)
            for make in (lambda g: ai.batch1d_iter(g, 10, num_workers=2),
                         lambda g: ai.batch2d_iter(g, 10, 8, num_workers=2)):
                it = make(misbehaving())
                got = []
                try:
                    for r in it:
                        got.append(r.is_ok())
                except RuntimeError as e:
                    assert "panicked" in str(e), e
                else:
                    raise AssertionError(f"stream ended silently after {got}")
                assert got == [True, True], got
                del it
                assert until(setter_works)
        """)

    def test_unconsumed_endless_generator_stops_pulling(self):
        # Back-pressure: with nobody consuming, an endless generator is only
        # pulled a few items ahead of the consumer, bounded by the workers.
        self.run_scenario("""
            workers = 2
            bound = 8 * workers + 2  # generous; the stream allows 4 per worker
            for make in (lambda g: ai.batch1d_iter(g, 10, num_workers=workers),
                         lambda g: ai.batch2d_iter(g, 10, 8, num_workers=workers)):
                count = [0]
                def endless():
                    while True:
                        count[0] += 1
                        yield np.ones(SHAPE)
                it = make(endless())
                assert next(it).is_ok()
                time.sleep(2.0)
                assert count[0] <= bound, f"pulled {count[0]} items unconsumed"
                # Still a working stream: consuming more pulls more.
                assert all(next(it).is_ok() for _ in range(50))
                assert 51 <= count[0] <= 51 + bound, count[0]
                del it
                assert until(setter_works)
        """)


if __name__ == "__main__":
    unittest.main()
