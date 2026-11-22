"""Integrator.integrate2d releases the GIL while it loads a TIFF and while it
integrates, so other Python threads run meanwhile and two integrators
(one per thread; an Integrator cannot be shared between threads) integrate
concurrently.

Each scenario runs in a subprocess with a timeout and two Rayon threads, so
the two concurrent calls do not just compete for every core.
"""
import os
import subprocess
import sys
import textwrap
import unittest

TIMEOUT = 60

PRELUDE = """
import os, tempfile, threading, time
import numpy as np, tifffile, dioptrin
poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.1, poni2=0.1,
            rot1=0.01, rot2=-0.02, rot3=0, wavelength=1e-10)
SHAPE = (1500, 1500)
rng = np.random.default_rng(0)
IMAGE = rng.random(SHAPE) * 100
d = tempfile.mkdtemp()
TIFF = os.path.join(d, "frame.tif")
tifffile.imwrite(TIFF, IMAGE.astype(np.float32))
CALLS = {
    "plain": lambda ai, img: ai.integrate2d(img, 200, 72),
    "errors": lambda ai, img: ai.integrate2d(img, 200, 72, errors=True),
    "range": lambda ai, img: ai.integrate2d(img, 200, 36, azimuthal_range=(0.0, np.pi)),
    "tiff": lambda ai, img: ai.integrate2d(TIFF, 200, 72),
}
def make(method="counting"):
    return dioptrin.Integrator.from_poni_dict(poni, method=method)
def warm(ai):
    for call in CALLS.values():
        call(ai, IMAGE)
def snapshot(result):
    return tuple(None if v is None else np.asarray(v).tobytes()
                 for v in (result.intensity, result.errors, result.radial, result.azimuthal))
"""


def run_scenario(body):
    script = PRELUDE + textwrap.dedent(body) + "\nprint('SCENARIO OK')\n"
    env = dict(os.environ, RAYON_NUM_THREADS="2")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError("scenario timed out")
    return proc


class Integrate2dGilTests(unittest.TestCase):
    def check(self, body):
        proc = run_scenario(body)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SCENARIO OK", proc.stdout)

    def test_threaded_results_equal_single_threaded(self):
        self.check("""
        methods = ("counting", "pixel_split", "supersampled", "pyfai_full")
        expected = {}
        for method in methods:
            ai = make(method)
            for name, call in CALLS.items():
                expected[method, name] = snapshot(call(ai, IMAGE))
        got, errors = {}, []
        def worker(method):
            try:
                ai = make(method)  # created in this thread: Integrator is unsendable
                for _ in range(2):
                    for name, call in CALLS.items():
                        got[method, name] = snapshot(call(ai, IMAGE))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(repr(exc))
        threads = [threading.Thread(target=worker, args=(m,)) for m in methods]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, errors
        assert got == expected, [k for k in expected if got.get(k) != expected[k]]
        """)

    def test_two_integrators_overlap_in_time(self):
        """With the GIL held, the second thread could not even record its start
        time until the first call returned, so the calls could never overlap."""
        self.check("""
        def attempt():
            barrier = threading.Barrier(2)
            intervals = {0: [], 1: []}
            def worker(k):
                ai = make()
                warm(ai)
                barrier.wait()
                for _ in range(6):
                    start = time.perf_counter()
                    ai.integrate2d(IMAGE, 200, 72, errors=True)
                    intervals[k].append((start, time.perf_counter()))
            threads = [threading.Thread(target=worker, args=(k,)) for k in (0, 1)]
            for t in threads: t.start()
            for t in threads: t.join()
            return any(max(a[0], b[0]) < min(a[1], b[1])
                       for a in intervals[0] for b in intervals[1])
        assert any(attempt() for _ in range(3)), "integrate2d calls never overlapped"
        """)

    def test_python_threads_run_during_a_call(self):
        """A pure-Python thread keeps counting while integrate2d runs, for
        arrays and TIFF paths and on every branch."""
        self.check("""
        ai = make("pixel_split")
        warm(ai)
        def count_while(action):
            ticks, stop = [0], threading.Event()
            def counter():
                while not stop.is_set():
                    ticks[0] += 1
            t = threading.Thread(target=counter)
            t.start()
            time.sleep(0.02)
            before, start = ticks[0], time.perf_counter()
            action()
            during, elapsed = ticks[0] - before, time.perf_counter() - start
            stop.set()
            t.join()
            return during / elapsed
        for name, call in CALLS.items():
            def attempt():
                alone = count_while(lambda: time.sleep(0.1))
                during = count_while(lambda: [call(ai, IMAGE) for _ in range(5)])
                return during / alone
            # The counter keeps most of its solo rate while the GIL is free;
            # held, it only runs in the switch intervals (5 ms) between calls.
            ratios = [attempt() for _ in range(3)]
            print(name, ["%.2f" % r for r in ratios])
            assert max(ratios) > 0.6, (name, ratios)
        """)


if __name__ == "__main__":
    unittest.main()
