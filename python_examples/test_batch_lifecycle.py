"""How every kind of batch*_iter stream ends, and when it gives the integrator
back.

Stream kinds: a list of TIFF paths, a list of arrays and a generator, for
batch1d_iter (plain, errors, azimuthal_std) and batch2d_iter (plain, errors,
azimuthal_range, azimuthal_bracket). A stream that ended (StopIteration or an
exception) has released the integrator, so its setters work at once; a stream
deleted mid-way releases it at once when fed by a generator and eventually
otherwise. Hangs are covered by test_batch_concurrency (subprocesses with a
timeout); everything here is bounded by a deadline.

Synthetic geometry and small images; TIFF inputs go to a temporary directory.
"""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import tifffile

import dioptrin

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0016, poni2=0.0024,
            wavelength=1e-10)
SHAPE = (32, 48)
N = 6
ARRAYS = [np.full(SHAPE, i + 1.0) for i in range(N)]

ONE_D = {
    "1d": lambda ai, src: ai.batch1d_iter(src, 10),
    "1d errors": lambda ai, src: ai.batch1d_iter(src, 10, errors=True),
    "1d azimuthal_std": lambda ai, src: ai.batch1d_iter(src, 10, azimuthal_std=True),
}
TWO_D = {
    "2d": lambda ai, src: ai.batch2d_iter(src, 10, 8),
    "2d errors": lambda ai, src: ai.batch2d_iter(src, 10, 8, errors=True),
    "2d azimuthal_range": lambda ai, src: ai.batch2d_iter(
        src, 10, 8, azimuthal_range=(0.0, np.pi)),
    "2d azimuthal_bracket": lambda ai, src: ai.batch2d_iter(
        src, 10, azimuthal_bracket=np.pi / 8),
}
STREAMS = {**ONE_D, **TWO_D}


def integrator():
    return dioptrin.Integrator.from_poni_dict(dict(PONI))


def setter_works(ai):
    try:
        ai.set_unit("2th_deg")
        return True
    except ValueError as e:
        assert "in use" in str(e), e
        return False


def until(condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


class LifecycleTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = []
        for i, image in enumerate(ARRAYS):
            path = Path(cls.tmp.name) / f"frame_{i}.tif"
            tifffile.imwrite(path, image.astype(np.float32))
            cls.paths.append(str(path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sources(self):
        """(label, make input, expected filenames, known total)."""
        return [
            ("arrays", lambda: list(ARRAYS), [f"array[{i}]" for i in range(N)], N),
            ("paths", lambda: list(self.paths), list(self.paths), N),
            ("generator", lambda: (a for a in ARRAYS), [f"iter[{i}]" for i in range(N)], 0),
        ]

    def assert_usable(self, ai):
        """Setters work now, and a synchronous batch on the integrator runs."""
        self.assertTrue(setter_works(ai), "integrator still in use")
        r1 = ai.batch1d(list(ARRAYS[:2]), 10, num_workers=2)
        r2 = ai.batch2d(self.paths[:2], 10, 8, num_workers=2)
        self.assertTrue(all(r.is_ok() for r in r1 + r2), [r.error for r in r1 + r2])


class FinishedStreamTests(LifecycleTestCase):
    def test_finished_stream_releases_integrator_at_once(self):
        # Regression: a finished batch1d_iter over arrays used to keep its
        # integrator for a moment after StopIteration (setter failed in ~97%
        # of rounds), so no retry here. Rounds per case are sized to catch
        # that race reliably while costing well under a second each.
        rounds = {"arrays": 200, "paths": 50, "generator": 30}
        ai = integrator()
        for label, make_input, names, total in self.sources():
            for kind, make in STREAMS.items():
                with self.subTest(source=label, stream=kind):
                    for round_ in range(rounds[label]):
                        it = make(ai, make_input())
                        if round_ == 0:
                            self.assertEqual((it.total, it.completed), (total, 0))
                            self.assertEqual(it.progress, 0.0 if total else None)
                        results = list(it)
                        self.assertTrue(setter_works(ai),
                                        f"in use after StopIteration, round {round_}")
                        self.assertEqual([r.filename for r in results], names)
                        self.assertTrue(all(r.is_ok() for r in results),
                                        [r.error for r in results])
                        self.assertEqual((it.total, it.completed), (total, N))
                        self.assertEqual(it.progress, 1.0 if total else None)
                        with self.assertRaises(StopIteration):
                            next(it)
        self.assert_usable(ai)

    def test_empty_input(self):
        ai = integrator()
        for label, source in (("list", lambda: []), ("generator", lambda: iter([]))):
            for kind, make in STREAMS.items():
                with self.subTest(source=label, stream=kind):
                    it = make(ai, source())
                    self.assertEqual(list(it), [])
                    self.assertEqual((it.total, it.completed, it.progress), (0, 0, None))
                    with self.assertRaises(StopIteration):
                        next(it)
                    self.assertTrue(setter_works(ai))
        self.assert_usable(ai)

    def test_failed_items_in_path_streams_do_not_end_the_stream(self):
        # A missing file, or a file whose shape does not match the mask (a
        # panic inside the worker), is an error result at its position.
        ai = integrator()
        ai.set_mask(np.zeros(SHAPE, np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.tif")
            wrong_shape = str(Path(directory) / "wrong_shape.tif")
            tifffile.imwrite(wrong_shape, np.ones((8, 8), np.float32))
            items = [self.paths[0], missing, self.paths[1], wrong_shape, self.paths[2]]
            for label, source in (("list", lambda: list(items)),
                                  ("generator", lambda: iter(items))):
                for kind, make in STREAMS.items():
                    with self.subTest(source=label, stream=kind):
                        results = list(make(ai, source()))
                        self.assertTrue(setter_works(ai))
                        self.assertEqual([r.filename for r in results], items)
                        self.assertEqual([r.is_ok() for r in results],
                                         [True, False, True, False, True])
        ai.set_mask(None)
        self.assert_usable(ai)


class FailedStreamTests(LifecycleTestCase):
    """A generator-fed stream that fails raises once, after every earlier
    result, then stops; its workers are gone by the time it raises."""

    def check_failure(self, ai, make, frames, error, pattern, rounds=10):
        for _ in range(rounds):
            it = make(ai, frames())
            first = [next(it), next(it)]
            self.assertEqual([r.filename for r in first], ["iter[0]", "iter[1]"])
            self.assertTrue(all(r.is_ok() for r in first), [r.error for r in first])
            with self.assertRaisesRegex(error, pattern):
                next(it)
            self.assertTrue(setter_works(ai), "in use after the stream raised")
            self.assertEqual((it.total, it.completed, it.progress), (0, 2, None))
            with self.assertRaises(StopIteration):
                next(it)

    def test_generator_raises(self):
        def frames():
            yield ARRAYS[0]
            yield ARRAYS[1]
            raise KeyError("boom")

        ai = integrator()
        for kind, make in STREAMS.items():
            with self.subTest(stream=kind):
                self.check_failure(ai, make, frames, KeyError, "boom")
        self.assert_usable(ai)

    def test_invalid_item(self):
        def frames():
            yield ARRAYS[0]
            yield ARRAYS[1]
            yield ARRAYS[2].astype(np.int32)
            yield ARRAYS[3]

        ai = integrator()
        for kind, make in STREAMS.items():
            with self.subTest(stream=kind):
                self.check_failure(ai, make, frames, TypeError, r"images\[2\].*int32")
        self.assert_usable(ai)

    def test_mask_shape_mismatch(self):
        def frames():
            yield ARRAYS[0]
            yield ARRAYS[1]
            yield np.ones((8, 8))
            yield ARRAYS[3]

        ai = integrator()
        ai.set_mask(np.zeros(SHAPE, np.uint8))
        for kind, make in STREAMS.items():
            with self.subTest(stream=kind):
                self.check_failure(ai, make, frames, ValueError,
                                   r"images\[2\] shape \(8, 8\) does not match mask")
        ai.set_mask(None)
        self.assert_usable(ai)


class AbandonedStreamTests(LifecycleTestCase):
    def test_deleted_generator_stream_releases_integrator_at_once(self):
        ai = integrator()
        for kind, make in STREAMS.items():
            with self.subTest(stream=kind):
                for _ in range(10):
                    it = make(ai, (a for a in ARRAYS * 4))
                    self.assertTrue(next(it).is_ok())
                    self.assertFalse(setter_works(ai), "stream should hold the integrator")
                    del it
                    self.assertTrue(setter_works(ai), "in use after del")
        self.assert_usable(ai)

    def test_deleted_list_stream_releases_integrator_eventually(self):
        # List streams are not waited for on del: 2D over arrays is cancelled,
        # 1D over arrays and path streams finish their remaining items.
        ai = integrator()
        for label, make_input in (("arrays", lambda: ARRAYS * 4),
                                  ("paths", lambda: self.paths * 4)):
            for kind, make in STREAMS.items():
                with self.subTest(source=label, stream=kind):
                    it = make(ai, make_input())
                    self.assertTrue(next(it).is_ok())
                    del it
                    self.assertTrue(until(lambda: setter_works(ai)),
                                    "integrator not released after del")
        self.assert_usable(ai)


if __name__ == "__main__":
    unittest.main()
