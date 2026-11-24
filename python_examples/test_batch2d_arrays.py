"""Integrator.batch2d accepts a list of 2D float64 NumPy arrays, borrowed
without copying and integrated in parallel with the GIL released, like
batch1d. Synthetic geometry; TIFF inputs are written to a temporary directory.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import dioptrin

try:
    import tifffile
except ImportError:  # pragma: no cover - optional test dependency
    tifffile = None

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0032, poni2=0.0048,
            rot1=0.01, rot2=-0.02, wavelength=1e-10)
SHAPE = (64, 96)
METHODS = ("counting", "pixel_split", "supersampled", "pyfai_full")
FIELDS = ("radial", "azimuthal", "intensity", "errors", "n_radial", "n_azimuthal")
CALLS = {
    "plain": dict(n_azimuthal=16),
    "errors": dict(n_azimuthal=16, errors=True),
    "azimuthal_range": dict(n_azimuthal=8, azimuthal_range=(0.2, 0.2 + np.pi / 2), errors=True),
    "azimuthal_bracket": dict(azimuthal_bracket=np.radians(10)),
}


def integrator(method="pixel_split", **kw):
    return dioptrin.Integrator.from_poni_dict(dict(PONI), method=method, n_ss=3, **kw)


def frames(n=4):
    rng = np.random.default_rng(1)
    return [rng.random(SHAPE) * 100 + 10 * i for i in range(n)]


def write_tiff(directory, image, index):
    # TIFF input follows Dioptrin's vertical-flip convention.
    path = Path(directory) / f"frame_{index}.tif"
    tifffile.imwrite(path, np.ascontiguousarray(image[::-1]))
    return str(path)


class Batch2dArrayTests(unittest.TestCase):
    def same(self, actual, expected, label=""):
        for field in FIELDS:
            value, reference = getattr(actual, field), getattr(expected, field)
            if reference is None:
                self.assertIsNone(value, f"{label} {field}")
            else:
                assert_array_equal(value, reference, err_msg=f"{label} {field}")

    def test_arrays_match_integrate2d_and_tiff_paths(self):
        images = frames()
        with tempfile.TemporaryDirectory() as directory:
            paths = None
            if tifffile is not None:
                paths = [write_tiff(directory, image, i) for i, image in enumerate(images)]
            for method in METHODS:
                for radial_range in (None, (0.02, 0.08)):
                    for name, kw in CALLS.items():
                        label = f"{method} {radial_range} {name}"
                        with self.subTest(label):
                            ai = integrator(method, radial_range=radial_range)
                            batch = ai.batch2d(images, 20, **kw)
                            self.assertEqual(
                                [r.filename for r in batch],
                                [f"array[{i}]" for i in range(len(images))],
                            )
                            for item, image in zip(batch, images):
                                self.assertTrue(item.is_ok(), item.error)
                                self.assertGreater(np.nansum(item.result.intensity), 0)
                                self.same(item.result, ai.integrate2d(image, 20, **kw), label)
                            if paths is not None:
                                for item, other in zip(batch, ai.batch2d(paths, 20, **kw)):
                                    self.assertTrue(other.is_ok(), other.error)
                                    self.same(item.result, other.result, label + " tiff")
                            # And the copying stream gives the same.
                            for item, other in zip(batch, ai.batch2d_iter(images, 20, **kw)):
                                self.same(item.result, other.result, label + " iter")

    def test_workers_shapes_and_borrowed_inputs(self):
        ai = integrator()
        mixed = [np.ones(SHAPE), np.full((48, 80), 2.0), np.full(SHAPE, 3.0)]
        expected = [ai.integrate2d(x, 20, 16, errors=True) for x in mixed]
        for workers, threads in ((None, None), (1, 1), (3, 2)):
            batch = integrator().batch2d(mixed, 20, 16, errors=True,
                                         num_workers=workers, num_threads=threads)
            for item, wanted in zip(batch, expected):
                self.same(item.result, wanted)
        # Read-only, memory-mapped frames are borrowed, not copied or written.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npy"
            np.save(path, np.stack(frames(3)))
            images = np.load(path, mmap_mode="r")
            batch = ai.batch2d(list(images), 20, 16)
            for item, image in zip(batch, images):
                self.same(item.result, ai.integrate2d(np.array(image), 20, 16))
            self.assertFalse(images.flags.writeable)
        self.assertEqual(ai.batch2d([], 20, 16), [])

    def test_worker_splits_match_integrate2d(self):
        """num_workers frames at a time, each on a pool of num_threads threads
        (the split itself is checked in Rust by
        batch::tests::map_in_pools_applies_the_requested_split); every split,
        including the default global pool, gives per-frame integrate2d bits."""
        images = frames(5)
        splits = [(None, None), (1, 1), (1, 4), (4, 1), (2, 2), (16, 1), (3, None), (None, 3)]
        for method in ("counting", "pixel_split", "supersampled"):
            for radial_range in (None, (0.02, 0.08)):
                ai = integrator(method, radial_range=radial_range)
                for name in ("plain", "errors", "azimuthal_range"):
                    kw = CALLS[name]
                    expected = [ai.integrate2d(image, 20, **kw) for image in images]
                    for workers, threads in splits:
                        label = f"{method} {radial_range} {name} {workers}x{threads}"
                        with self.subTest(label):
                            batch = ai.batch2d(images, 20, num_workers=workers,
                                               num_threads=threads, **kw)
                            self.assertEqual(len(batch), len(images))
                            for i, (item, wanted) in enumerate(zip(batch, expected)):
                                self.assertEqual(item.filename, f"array[{i}]")
                                self.assertTrue(item.is_ok(), item.error)
                                self.same(item.result, wanted, label)
        with self.assertRaises(ValueError):
            integrator().batch2d(images, 20, 16, num_workers=0)
        with self.assertRaises(ValueError):
            integrator().batch2d(images, 20, 16, num_threads=0)

    def test_invalid_inputs_raise_like_batch1d(self):
        image = np.ones(SHAPE)
        ai = integrator()
        bad = {
            "non-contiguous": ([image, image[:, ::2]], ValueError, "contiguous"),
            "wrong dtype": ([image.astype(np.int64)], TypeError, None),
            "wrong ndim": ([np.ones((2,) + SHAPE)], TypeError, None),
        }
        for name, (images, error, message) in bad.items():
            with self.subTest(name):
                with self.assertRaises(error) as one:
                    ai.batch1d(images, 20)
                with self.assertRaises(error) as two:
                    ai.batch2d(images, 20, 16)
                if message:
                    self.assertIn(message, str(two.exception))
                    self.assertEqual(
                        str(one.exception), str(two.exception)
                    )
        # Mask mismatch: same ValueError and message as batch1d.
        masked = integrator()
        masked.set_mask(np.zeros((32, 32), np.uint8))
        with self.assertRaises(ValueError) as one:
            masked.batch1d([image], 20)
        with self.assertRaises(ValueError) as two:
            masked.batch2d([image], 20, 16)
        self.assertEqual(str(one.exception), str(two.exception))
        self.assertIn("does not match mask shape (32, 32)", str(two.exception))
        # Mixed lists are rejected as in batch1d (batch2d_iter accepts them).
        with tempfile.TemporaryDirectory() as directory:
            if tifffile is not None:
                path = write_tiff(directory, image, 0)
                with self.assertRaises(TypeError):
                    ai.batch1d([path, image], 20)
                with self.assertRaises(TypeError) as caught:
                    ai.batch2d([path, image], 20, 16)
                self.assertEqual(
                    getattr(caught.exception, "__notes__", None),
                    ["while processing 'tiff_paths'"],
                )
                self.assertEqual(len(list(ai.batch2d_iter([path, image], 20, 16))), 2)
        # An invalid azimuthal range is rejected up front, as for paths.
        with self.assertRaises(ValueError):
            ai.batch2d([image], 20, 16, azimuthal_range=(1.0, 0.0))
        # The integrator stays usable and is free for setters afterwards.
        self.assertTrue(ai.batch2d([image], 20, 16)[0].is_ok())
        ai.set_unit("2th_deg")

    def test_inputs_are_not_modified(self):
        images = frames(3)
        copies = [x.copy() for x in images]
        integrator("supersampled").batch2d(images, 20, 16, errors=True)
        for image, copy in zip(images, copies):
            assert_array_equal(image, copy)

    def test_native_pixel_dtypes_match_float64(self):
        # Test enough frames to trigger the 4-frame SIMD-tiled batch path as well
        # as a small batch (single-frame fallback) across all native dtypes.
        rng = np.random.default_rng(7)
        base_ints = [rng.integers(0, 5000, size=SHAPE, dtype=np.int32) for _ in range(70)]
        ai = integrator("pixel_split")
        for dtype in (np.float32, np.uint16, np.uint32, np.int32):
            typed_all = [x.astype(dtype) for x in base_ints]
            f64_all = [x.astype(np.float64) for x in typed_all]
            for count in (3, len(typed_all)):
                typed = typed_all[:count]
                f64 = f64_all[:count]
                for errors in (False, True):
                    with self.subTest(dtype=dtype.__name__, count=count, errors=errors):
                        r1_ref = ai.batch1d(f64, 24, errors=errors)
                        r1_got = ai.batch1d(typed, 24, errors=errors)
                        for got, ref in zip(r1_got, r1_ref):
                            self.assertTrue(got.is_ok(), got.error)
                            assert_array_equal(got.result.radial, ref.result.radial)
                            assert_array_equal(got.result.intensity, ref.result.intensity)
                            if errors:
                                assert_array_equal(got.result.errors, ref.result.errors)

                        r2_ref = ai.batch2d(f64, 20, 16, errors=errors)
                        r2_got = ai.batch2d(typed, 20, 16, errors=errors)
                        for got, ref in zip(r2_got, r2_ref):
                            self.assertTrue(got.is_ok(), got.error)
                            self.same(got.result, ref.result, f"{dtype.__name__} {count} {errors}")


if __name__ == "__main__":
    unittest.main()
