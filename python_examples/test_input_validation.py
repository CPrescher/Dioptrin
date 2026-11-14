"""Invalid Python input raises ValueError/TypeError, never a Rust panic, and is
never silently swallowed by the streaming batch APIs.

Synthetic geometry and arrays only; TIFF inputs are written to a temporary
directory with tifffile. No reference data or network needed.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

import dioptrin

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0032, poni2=0.0048, wavelength=1e-10)
SHAPE = (64, 96)
IMAGE = np.random.default_rng(0).random(SHAPE) * 100
GOOD = IMAGE
BAD = IMAGE.astype(np.int32)


def integrator(poni=PONI, **kw):
    return dioptrin.Integrator.from_poni_dict(dict(poni), **kw)


def write_tiff(directory, image):
    # TIFF input follows Dioptrin's vertical-flip convention.
    path = Path(directory) / f"frame_{len(list(Path(directory).iterdir()))}.tif"
    tifffile.imwrite(path, np.ascontiguousarray(image[::-1]))
    return str(path)


class StreamingBatchErrorTests(unittest.TestCase):
    """Item 1: bad items and iterator exceptions reach the caller."""

    def test_list_of_wrong_dtype_raises_up_front(self):
        ai = integrator()
        with self.assertRaisesRegex(TypeError, r"images\[0\].*float64.*int32"):
            ai.batch1d_iter([BAD, BAD], 10)
        with self.assertRaisesRegex(TypeError, r"images\[1\].*float64"):
            ai.batch2d_iter([GOOD, BAD], 10, 8)
        with self.assertRaisesRegex(TypeError, r"images\[2\].*got int"):
            ai.batch1d_iter([GOOD, GOOD, 5], 10)
        with self.assertRaisesRegex(ValueError, r"images\[1\].*contiguous"):
            ai.batch1d_iter([GOOD, GOOD[:, ::2]], 10)

    def check_bad_generator_item(self, make_iter):
        def frames():
            yield GOOD
            yield GOOD
            yield BAD
            yield GOOD

        it = make_iter(frames())
        first = [next(it), next(it)]
        self.assertTrue(all(r.is_ok() for r in first), [r.error for r in first])
        with self.assertRaisesRegex(TypeError, r"images\[2\].*float64.*int32"):
            next(it)
        with self.assertRaises(StopIteration):
            next(it)

    def test_bad_generator_item_raises_after_earlier_results(self):
        ai = integrator()
        self.check_bad_generator_item(lambda g: ai.batch1d_iter(g, 10))
        self.check_bad_generator_item(lambda g: ai.batch1d_iter(g, 10, errors=True))
        self.check_bad_generator_item(lambda g: ai.batch1d_iter(g, 10, azimuthal_std=True))
        self.check_bad_generator_item(lambda g: ai.batch2d_iter(g, 10, 8))
        self.check_bad_generator_item(
            lambda g: ai.batch2d_iter(g, 10, 8, azimuthal_range=(0.0, 1.0)))

    def test_generator_exception_propagates(self):
        def frames():
            yield GOOD
            raise KeyError("boom")

        ai = integrator()
        for make in (lambda: ai.batch1d_iter(frames(), 10),
                     lambda: ai.batch2d_iter(frames(), 10, 8, errors=True)):
            it = make()
            self.assertTrue(next(it).is_ok())
            with self.assertRaisesRegex(KeyError, "boom"):
                next(it)

    def test_valid_mixed_list_streams_like_arrays(self):
        ai = integrator()
        with tempfile.TemporaryDirectory() as directory:
            path = write_tiff(directory, IMAGE)
            mixed = list(ai.batch1d_iter([path, GOOD], 10))
        arrays = list(ai.batch1d_iter([GOOD, GOOD], 10))
        self.assertEqual(len(mixed), 2)
        for got, want in zip(mixed, arrays):
            self.assertTrue(got.is_ok(), got.error)
            np.testing.assert_allclose(got.result.intensity, want.result.intensity)


class MaskShapeTests(unittest.TestCase):
    """Item 2: an image that does not match the mask raises ValueError."""

    def test_every_image_entry_point(self):
        ai = integrator()
        ai.set_mask(np.zeros((32, 32), np.uint8))
        message = r"\(64, 96\) does not match mask shape \(32, 32\)"
        calls = {
            "integrate1d": lambda: ai.integrate1d(GOOD, 10),
            "integrate1d std": lambda: ai.integrate1d(GOOD, 10, azimuthal_std=True),
            "integrate2d": lambda: ai.integrate2d(GOOD, 10, 8),
            "integrate2d errors": lambda: ai.integrate2d(GOOD, 10, 8, errors=True),
            "integrate2d range": lambda: ai.integrate2d(GOOD, 10, 8, azimuthal_range=(0, 1)),
            "batch1d": lambda: ai.batch1d([GOOD], 10),
            "batch1d errors": lambda: ai.batch1d([GOOD], 10, errors=True),
            "batch1d_iter": lambda: ai.batch1d_iter([GOOD], 10),
            "batch2d_iter": lambda: ai.batch2d_iter([GOOD], 10, 8),
        }
        for name, call in calls.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                call()

        with tempfile.TemporaryDirectory() as directory:
            path = write_tiff(directory, IMAGE)
            with self.assertRaisesRegex(ValueError, message):
                ai.integrate1d(path, 10)
            with self.assertRaisesRegex(ValueError, message):
                ai.integrate2d(path, 10, 8)

        it = ai.batch1d_iter(iter([GOOD]), 10)
        with self.assertRaisesRegex(ValueError, r"images\[0\] shape"):
            next(it)

        # The integrator stays usable with a matching image.
        self.assertTrue(ai.batch1d([np.ones((32, 32))], 10)[0].is_ok())

    def test_multi_integrator_per_detector(self):
        masked = integrator(radial_range=(0.0, 0.1))
        masked.set_mask(np.zeros((32, 32), np.uint8))
        multi = dioptrin.MultiIntegrator([integrator(radial_range=(0.0, 0.1)), masked])
        message = r"detector 1 image shape \(64, 96\) does not match mask shape \(32, 32\)"
        with self.assertRaisesRegex(ValueError, message):
            multi.integrate1d([GOOD, GOOD], 10)
        with self.assertRaisesRegex(ValueError, "frame set 0, " + message):
            multi.batch1d([[GOOD, GOOD]], 10)
        with self.assertRaisesRegex(ValueError, message):
            multi.integrate2d([GOOD, GOOD], 10, 8)


class MultiIntegratorMixedInputTests(unittest.TestCase):
    """Item 3: lists mixing TIFF paths and arrays work for 1D."""

    def test_mixed_lists_match_arrays(self):
        other = np.random.default_rng(1).random((40, 50)) * 50
        multi = dioptrin.MultiIntegrator([integrator(radial_range=(0.0, 0.1)),
                                          integrator(radial_range=(0.0, 0.1))])
        with tempfile.TemporaryDirectory() as directory:
            path = write_tiff(directory, IMAGE)
            for errors in (False, True):
                arrays = multi.integrate1d([IMAGE, other], 20, errors=errors)
                mixed = multi.integrate1d([path, other], 20, errors=errors)
                np.testing.assert_allclose(mixed.intensity, arrays.intensity)
                np.testing.assert_allclose(mixed.radial, arrays.radial)
                if errors:
                    np.testing.assert_allclose(mixed.errors, arrays.errors)
                batch = multi.batch1d([[IMAGE, other], [path, other]], 20, errors=errors)
                for result in batch:
                    np.testing.assert_allclose(result.intensity, arrays.intensity)
            paths = multi.integrate1d([path, path], 20)
            np.testing.assert_allclose(
                paths.intensity, multi.integrate1d([IMAGE, IMAGE], 20).intensity)


class ArgumentValidationTests(unittest.TestCase):
    """Item 4: arguments that core would panic on (or misuse) are rejected."""

    def test_bin_counts(self):
        ai = integrator()
        multi = dioptrin.MultiIntegrator([integrator(radial_range=(0.0, 0.1))])
        calls = {
            "n_points": [
                lambda: ai.integrate1d(GOOD, 0),
                lambda: ai.batch1d([GOOD], 0),
                lambda: ai.batch1d_iter([GOOD], 0),
                lambda: multi.integrate1d([GOOD], 0),
                lambda: multi.batch1d([[GOOD]], 0),
                lambda: ai.prepare_dynamic1d(SHAPE, 0),
            ],
            "n_radial": [
                lambda: ai.integrate2d(GOOD, 0, 8),
                lambda: ai.batch2d([], 0, 8),
                lambda: ai.batch2d_iter([GOOD], 0, 8),
                lambda: ai.batch2d_iter([GOOD], 0, 8, azimuthal_range=(0, 1)),
                lambda: multi.integrate2d([GOOD], 0, 8),
            ],
            "n_azimuthal": [
                lambda: ai.integrate2d(GOOD, 10, 0),
                lambda: ai.batch2d([], 10, 0),
                lambda: ai.batch2d_iter([GOOD], 10, 0),
                lambda: multi.integrate2d([GOOD], 10, 0),
            ],
        }
        for name, group in calls.items():
            for call in group:
                with self.subTest(name), self.assertRaisesRegex(ValueError, "must be positive"):
                    call()

    def test_q_units_need_a_wavelength(self):
        no_wavelength = {k: v for k, v in PONI.items() if k != "wavelength"}
        for unit in ("q_A^-1", "q_nm^-1"):
            with self.assertRaisesRegex(ValueError, "wavelength"):
                integrator(no_wavelength, unit=unit)
        ai = integrator(no_wavelength)
        self.assertTrue(np.all(np.isfinite(ai.integrate1d(GOOD, 10).intensity)))
        with self.assertRaisesRegex(ValueError, "wavelength"):
            ai.set_unit("q_A^-1")
        self.assertEqual(ai.get_unit(), "2th_rad")

    def test_radial_range(self):
        for bad in [(0.1, 0.0), (0.1, 0.1), (float("nan"), 0.1), (0.0, float("inf"))]:
            with self.subTest(bad), self.assertRaisesRegex(ValueError, "radial_range"):
                integrator(radial_range=bad)

    def test_histogram_rejects_radial_range(self):
        # Histogram always bins over the automatic range; it would silently
        # ignore radial_range, so the combination is rejected everywhere.
        message = r"radial_range is not supported by the 'histogram' method.*automatic"
        for method in ("histogram", "hist", "HISTOGRAM"):
            with self.subTest(method), self.assertRaisesRegex(ValueError, message):
                integrator(method=method, radial_range=(0.01, 0.1))
        ai = integrator(method="counting", radial_range=(0.01, 0.1))
        with self.assertRaisesRegex(ValueError, message):
            ai.set_method("histogram")
        # The failed switch left the method and range unchanged.
        radial = np.asarray(ai.integrate1d(GOOD, 10).radial)
        self.assertGreaterEqual(radial.min(), 0.01)
        self.assertLessEqual(radial.max(), 0.1)
        for method in ("counting", "pixel_split", "pyfai_full", "supersampled"):
            ai.set_method(method)
        # Histogram without a range is unaffected and can be switched to/from.
        hist = integrator(method="histogram")
        self.assertTrue(np.any(np.asarray(hist.integrate1d(GOOD, 10).intensity) > 0))
        hist.set_method("counting")
        hist.set_method("histogram")

    def test_supersampling_factor(self):
        with self.assertRaisesRegex(ValueError, "n_ss must be positive"):
            integrator(method="supersampled", n_ss=0)
        ai = integrator()
        with self.assertRaisesRegex(ValueError, "n_ss must be positive"):
            ai.set_method("supersampled", n_ss=0)

    def test_polarization_factor(self):
        for bad in (1.5, -1.01, float("nan")):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, "polarization_factor"):
                integrator(polarization_factor=bad)
        ai = integrator(polarization_factor=-1.0)
        with self.assertRaisesRegex(ValueError, "polarization_factor"):
            ai.set_polarization_factor(2.0)
        ai.set_polarization_factor(0.99)
        ai.set_polarization_factor(None)

    def test_empty_images(self):
        ai = integrator(method="counting")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ai.integrate1d(np.ones((0, 5)), 10)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ai.batch1d_iter([np.ones((0, 5))], 10)
        with self.assertRaisesRegex(ValueError, r"images\[1\] must be non-empty"):
            ai.batch1d_iter([GOOD, np.ones((0, 5)), "x.tif"], 10)


if __name__ == "__main__":
    unittest.main()
