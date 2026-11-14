"""Parallel array/list/generator batches must use each image's own geometry."""
import unittest

import dioptrin
import numpy as np


PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.05,
            poni1=0.0024, poni2=0.0018, wavelength=1e-10)
METHODS = ("counting", "pixel_split", "pyfai_full", "supersampled")
SHAPES = ((48, 36), (20, 30), (36, 48))  # first/last have equal pixel counts


def images():
    return [((np.arange(h * w, dtype=np.float64) * 37 + w) % 101 + 1).reshape(h, w)
            for h, w in SHAPES]


class MixedShapeTests(unittest.TestCase):
    def assert_result_matches(self, actual, expected):
        for field in ("radial", "azimuthal", "intensity", "errors"):
            value = getattr(expected, field)
            if value is None:
                self.assertIsNone(getattr(actual, field))
            else:
                np.testing.assert_allclose(getattr(actual, field), value, rtol=1e-12, atol=1e-12)

    def test_parallel_1d_arrays_and_generators_match_independent_shapes(self):
        frames = images()
        order = [i % len(frames) for i in range(18)]
        for method in (*METHODS, "histogram"):
            for fast in (False, True):
                new = lambda: dioptrin.Integrator.from_poni_dict(
                    PONI, method=method, n_ss=2, fast_math=fast)
                for errors in (False, True):
                    expected = [new().integrate1d(image, 25, errors=errors) for image in frames]
                    shared = new()
                    eager = shared.batch1d([frames[i] for i in order], 25, errors=errors,
                                           num_workers=4, num_threads=2)
                    self.assertEqual(len(eager), len(order))
                    for i, item in zip(order, eager):
                        self.assertTrue(item.is_ok(), item.error)
                        self.assert_result_matches(item.result, expected[i])
                    for source in ([frames[i] for i in order], (frames[i] for i in order)):
                        batch = list(shared.batch1d_iter(source, 25, errors=errors,
                                                        num_workers=4, num_threads=2))
                        self.assertEqual(len(batch), len(order))
                        for i, item in zip(order, batch):
                            self.assertTrue(item.is_ok(), item.error)
                            self.assert_result_matches(item.result, expected[i])

    def test_parallel_2d_ranges_and_generators_match_independent_shapes(self):
        frames = images()
        order = [i % len(frames) for i in range(12)]
        for method in METHODS:
            for fast in (False, True):
                new = lambda: dioptrin.Integrator.from_poni_dict(
                    PONI, method=method, n_ss=2, fast_math=fast)
                shared = new()
                for angle_range in (None, (-3 * np.pi, -np.pi), (2.5, 2.5 + np.pi / 2)):
                    for errors in (False, True):
                        settings = dict(errors=errors, azimuthal_range=angle_range)
                        expected = [new().integrate2d(image, 12, 6, **settings) for image in frames]
                        for source in ([frames[i] for i in order], (frames[i] for i in order)):
                            batch = list(shared.batch2d_iter(source, 12, 6, num_workers=4,
                                                            num_threads=2, **settings))
                            self.assertEqual(len(batch), len(order))
                            for i, item in zip(order, batch):
                                self.assertTrue(item.is_ok(), item.error)
                                self.assert_result_matches(item.result, expected[i])


if __name__ == "__main__":
    unittest.main()
