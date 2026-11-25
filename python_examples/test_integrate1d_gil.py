"""Integrator.integrate1d releases the GIL while it loads a TIFF and while it
integrates, so other Python threads run meanwhile and two integrators
(one per thread; an Integrator cannot be shared between threads) integrate
concurrently.

Each scenario runs in a subprocess with a timeout and two Rayon threads (one
for the counter scenario), so the calls do not just compete for every core.
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
SHAPE = (500, 500)
rng = np.random.default_rng(0)
IMAGE = rng.random(SHAPE) * 100
d = tempfile.mkdtemp()
TIFF = os.path.join(d, "frame.tif")
tifffile.imwrite(TIFF, IMAGE.astype(np.float32))
CALLS = {
    "plain": lambda ai, img: ai.integrate1d(img, 1000),
    "errors": lambda ai, img: ai.integrate1d(img, 1000, errors=True),
    "azimuthal_std": lambda ai, img: ai.integrate1d(img, 1000, errors=True, azimuthal_std=True),
    "tiff": lambda ai, img: ai.integrate1d(TIFF, 1000),
}
def make(method="counting"):
    return dioptrin.Integrator.from_poni_dict(poni, method=method)
def warm(ai):
    for call in CALLS.values():
        call(ai, IMAGE)
def snapshot(result):
    return tuple(None if v is None else np.asarray(v).tobytes()
                 for v in (result.intensity, result.errors, result.radial, result.azimuthal_std))
"""


def run_scenario(body, rayon_threads=2):
    script = PRELUDE + textwrap.dedent(body) + "\nprint('SCENARIO OK')\n"
    env = dict(os.environ, RAYON_NUM_THREADS=str(rayon_threads))
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


class Integrate1dGilTests(unittest.TestCase):
    def check(self, body, rayon_threads=2):
        proc = run_scenario(body, rayon_threads)
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
        del ai  # only the threads' integrators hold tables from here on
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
        """Single calls of two threads overlap in time. The switch interval is
        raised above the scenario's length, so CPython never forces a thread
        to drop the GIL: a thread that holds it through each call runs all
        its calls before the other thread's first (its intervals then never
        overlap), and only a call that releases the GIL lets the other thread
        start one meanwhile. With the default 5 ms interval a GIL-holding call
        would still overlap: a thread can be switched out after reading its
        start time, so its interval contains the other thread's call. With
        the switch interval raised, even a call of about a millisecond (one
        with azimuthal_std on IMAGE) separates the two cases."""
        self.check("""
        import sys
        sys.setswitchinterval(30.0)
        def attempt():
            barrier = threading.Barrier(2)
            intervals = {0: [], 1: []}
            def worker(k):
                ai = make()
                warm(ai)
                barrier.wait()
                for _ in range(6):
                    start = time.perf_counter()
                    ai.integrate1d(IMAGE, 1000, errors=True, azimuthal_std=True)
                    intervals[k].append((start, time.perf_counter()))
            threads = [threading.Thread(target=worker, args=(k,)) for k in (0, 1)]
            for t in threads: t.start()
            for t in threads: t.join()
            overlap = sum(max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
                          for a in intervals[0] for b in intervals[1])
            busy = min(sum(e - s for s, e in intervals[k]) for k in (0, 1))
            return overlap / busy
        # Released: the calls overlap for nearly all their time; held: never.
        fractions = [attempt() for _ in range(3)]
        print("overlap", ["%.2f" % f for f in fractions])
        assert max(fractions) > 0.5, ("integrate1d calls did not overlap", fractions)
        """)

    def test_python_threads_run_during_a_call(self):
        """A pure-Python thread keeps counting while integrate1d runs, for
        arrays and TIFF paths and on every branch. One Rayon thread, so the
        counter has a core of its own on a two-core machine."""
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
                during = count_while(lambda: [call(ai, IMAGE) for _ in range(40)])
                return during / alone
            # The counter keeps most of its solo rate while the GIL is free;
            # held, it only runs in the switch intervals (5 ms) between calls.
            ratios = [attempt() for _ in range(3)]
            print(name, ["%.2f" % r for r in ratios])
            assert max(ratios) > 0.6, (name, ratios)
        """, rayon_threads=1)


if __name__ == "__main__":
    unittest.main()
