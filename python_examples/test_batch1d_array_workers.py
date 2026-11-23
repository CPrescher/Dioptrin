"""Integrator.batch1d over NumPy arrays honours num_workers and num_threads:
with neither it runs on the global Rayon pool as before; with either, at most
num_workers frames run at a time, each on a pool of num_threads threads (the
split itself is checked in Rust by
batch::tests::map_in_pools_applies_the_requested_split). Every split gives
the bits of per-frame integrate1d. Synthetic geometry only."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import dioptrin

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0032, poni2=0.0048,
            rot1=0.01, rot2=-0.02, wavelength=1e-10)
SHAPE = (64, 96)
FIELDS = ("radial", "intensity", "errors", "azimuthal_std")
CALLS = {
    "plain": dict(),
    "errors": dict(errors=True),
    "azimuthal_std": dict(errors=True, azimuthal_std=True),
}
SPLITS = [(None, None), (1, 1), (1, 4), (4, 1), (2, 2), (16, 1), (3, None), (None, 3)]


def frames(n=5):
    rng = np.random.default_rng(2)
    return [rng.random(SHAPE) * 100 + 10 * i for i in range(n)]


class Batch1dArrayWorkerTests(unittest.TestCase):
    def same(self, actual, expected, label):
        for field in FIELDS:
            value, reference = getattr(actual, field), getattr(expected, field)
            if reference is None:
                self.assertIsNone(value, f"{label} {field}")
            else:
                assert_array_equal(value, reference, err_msg=f"{label} {field}")

    def test_worker_splits_match_integrate1d(self):
        images = frames()
        cases = [(m, None) for m in ("counting", "pixel_split", "supersampled", "histogram")]
        cases += [(m, (0.01, 0.08)) for m in ("counting", "pixel_split", "pyfai_full")]
        for method, radial_range in cases:
            ai = dioptrin.Integrator.from_poni_dict(
                dict(PONI), method=method, n_ss=3, polarization_factor=0.965,
                radial_range=radial_range,
            )
            ai.set_mask((np.arange(np.prod(SHAPE)).reshape(SHAPE) % 17 == 0).astype(np.uint8))
            for name, kw in CALLS.items():
                expected = [ai.integrate1d(image, 40, **kw) for image in images]
                for workers, threads in SPLITS:
                    label = f"{method} {radial_range} {name} {workers}x{threads}"
                    with self.subTest(label):
                        batch = ai.batch1d(images, 40, num_workers=workers,
                                           num_threads=threads, **kw)
                        self.assertEqual([r.filename for r in batch],
                                         [f"array[{i}]" for i in range(len(images))])
                        for item, wanted in zip(batch, expected):
                            self.assertTrue(item.is_ok(), item.error)
                            self.same(item.result, wanted, label)

    def test_mixed_shapes_empty_and_invalid_counts(self):
        ai = dioptrin.Integrator.from_poni_dict(dict(PONI))
        mixed = [np.ones(SHAPE), np.full((48, 80), 2.0), np.full(SHAPE, 3.0)]
        expected = [ai.integrate1d(x, 40, errors=True) for x in mixed]
        for workers, threads in SPLITS:
            batch = ai.batch1d(mixed, 40, errors=True, num_workers=workers, num_threads=threads)
            for item, wanted in zip(batch, expected):
                self.same(item.result, wanted, f"{workers}x{threads}")
        self.assertEqual(ai.batch1d([], 40, num_workers=2, num_threads=2), [])
        for bad in (dict(num_workers=0), dict(num_threads=0)):
            with self.assertRaises(ValueError):
                ai.batch1d(mixed, 40, **bad)
        # The integrator is free for setters afterwards.
        ai.set_unit("2th_deg")


if __name__ == "__main__":
    unittest.main()
