"""Regression tests for the native 1D spottiness API (run with unittest)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import dioptrin
import numpy as np
from numpy.testing import assert_allclose


PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1,
            poni1=0.0032, poni2=0.0032, wavelength=1e-10)


def integrator(method="pixel_split", **kwargs):
    return dioptrin.Integrator.from_poni_dict(PONI, method=method, **kwargs)


class AzimuthalStdTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).uniform(10, 200, (64, 64))

    def compare_results(self, actual, expected):
        assert_allclose(actual.radial, expected.radial)
        assert_allclose(actual.intensity, expected.intensity)
        assert_allclose(actual.azimuthal_std, expected.azimuthal_std, equal_nan=True)
        if expected.errors is None:
            self.assertIsNone(actual.errors)
        else:
            assert_allclose(actual.errors, expected.errors)

    def test_optional_and_independent_of_poisson_errors(self):
        for method in ["counting", "pixel_split", "supersampled", "histogram"]:
            with self.subTest(method=method):
                ai = integrator(method, n_ss=3)
                ai.set_mask((np.indices(self.image.shape)[0] < 7).astype(np.uint8))
                plain = ai.integrate1d(self.image, 40)
                poisson = ai.integrate1d(self.image, 40, True)
                spread = ai.integrate1d(self.image, 40, azimuthal_std=True)
                both = ai.integrate1d(self.image, 40, errors=True, azimuthal_std=True)
                self.assertIsNone(plain.azimuthal_std)
                self.assertIsNone(poisson.azimuthal_std)
                self.assertIsNone(spread.errors)
                assert_allclose(both.intensity, plain.intensity, equal_nan=True)
                assert_allclose(both.errors, poisson.errors)
                assert_allclose(both.azimuthal_std, spread.azimuthal_std, equal_nan=True)
                valid = np.isfinite(both.azimuthal_std)
                self.assertTrue(np.any(valid))
                self.assertTrue(np.all(np.asarray(both.azimuthal_std)[valid] >= 0))
                self.assertIsNone(ai.integrate2d(self.image, 10, 10).azimuthal_std)

    def test_all_masked_and_single_contribution(self):
        ai = integrator("counting")
        mask = np.ones(self.image.shape, dtype=np.uint8)
        ai.set_mask(mask)
        empty = ai.integrate1d(self.image, 10, azimuthal_std=True, errors=True)
        self.assertTrue(np.all(np.isnan(empty.azimuthal_std)))
        assert_allclose(empty.intensity, 0)
        assert_allclose(empty.errors, 0)
        mask[4, 4] = 0
        ai.set_mask(mask)
        single = ai.integrate1d(self.image, 10, azimuthal_std=True)
        valid = np.isfinite(single.azimuthal_std)
        self.assertEqual(np.count_nonzero(valid), 1)
        assert_allclose(np.asarray(single.azimuthal_std)[valid], 0)

    def test_zero_bins_rejected(self):
        ai = integrator()
        for call in [ai.integrate1d, ai.batch1d, ai.batch1d_iter]:
            with self.assertRaisesRegex(ValueError, "n_points"):
                call(self.image if call == ai.integrate1d else [self.image],
                     0, azimuthal_std=True)

    def test_array_and_generator_batches(self):
        ai = integrator()
        frames = [self.image, self.image * 2, self.image * 3]
        for errors in [False, True]:
            expected = [ai.integrate1d(x, 20, errors=errors, azimuthal_std=True)
                        for x in frames]
            for batch in [ai.batch1d(frames, 20, errors, 2, 1, azimuthal_std=True),
                          ai.batch1d_iter(frames, 20, errors, 2, 1, azimuthal_std=True),
                          ai.batch1d_iter(iter(frames), 20, errors, 2, 1, azimuthal_std=True)]:
                results = list(batch)
                self.assertEqual(len(results), len(frames))
                for actual, reference in zip(results, expected):
                    self.assertTrue(actual.is_ok(), actual.error)
                    self.compare_results(actual.result, reference)
            assert_allclose(expected[1].azimuthal_std,
                            np.asarray(expected[0].azimuthal_std) * 2)
            if errors:
                assert_allclose(expected[1].errors, np.asarray(expected[0].errors) * np.sqrt(2))
        self.assertEqual(ai.batch1d([], 20, azimuthal_std=True), [])
        self.assertEqual(list(ai.batch1d_iter([], 20, azimuthal_std=True)), [])

    @unittest.skipUnless(importlib.util.find_spec("fabio"), "requires fabio")
    def test_tiff_and_file_batches(self):
        from fabio.tifimage import TifImage
        ai = integrator()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "frame.tif")
            # Use uint16, also exercising the native TIFF loader.
            data = self.image.astype(np.uint16)
            TifImage(data=data).write(path)
            for errors in [False, True]:
                expected = ai.integrate1d(data.astype(float), 20, errors, azimuthal_std=True)
                self.compare_results(ai.integrate1d(path, 20, errors, azimuthal_std=True), expected)
                paths = [path, str(Path(directory) / "missing.tif"), path]
                for batch in [ai.batch1d(paths, 20, errors, 2, 1, azimuthal_std=True),
                              ai.batch1d_iter(paths, 20, errors, 2, 1, azimuthal_std=True),
                              ai.batch1d_iter(iter(paths), 20, errors, 2, 1, azimuthal_std=True)]:
                    results = list(batch)
                    self.assertEqual(len(results), 3)
                    self.assertTrue(results[1].is_err())
                    self.compare_results(results[0].result, expected)
                    self.compare_results(results[2].result, expected)

    @unittest.skipUnless(importlib.util.find_spec("pyFAI"), "requires pyFAI")
    def test_equal_weight_reference_and_pyfai(self):
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        # Four equidistant pixels share exactly the same solid-angle correction.
        mask = np.ones((64, 64), dtype=np.uint8)
        pixels = [(10, 10), (10, 53), (53, 10), (53, 53)]
        image = np.zeros((64, 64), dtype=float)
        values = np.array([20., 40., 60., 200.])
        for pixel, value in zip(pixels, values):
            image[pixel] = value
            mask[pixel] = 0
        ai = integrator("counting")
        ai.set_mask(mask)
        result = ai.integrate1d(image, 1, azimuthal_std=True)
        reference = AzimuthalIntegrator(
            dist=PONI["distance"], poni1=PONI["poni1"], poni2=PONI["poni2"],
            pixel1=PONI["pixel1"], pixel2=PONI["pixel2"], wavelength=PONI["wavelength"])
        pyfai = reference.integrate1d(image, 1, mask=mask, method=("no", "csr", "cython"),
                                     error_model="azimuthal", unit="2th_rad")
        # Analytic solid angle for the perpendicular detector (pixel centers).
        radius_sq = 2 * ((10.5 * PONI["pixel1"]) - PONI["poni1"]) ** 2
        norm = (1 + radius_sq / PONI["distance"] ** 2) ** -1.5
        assert_allclose(result.intensity, [np.mean(values) / norm], rtol=1e-10)
        assert_allclose(result.azimuthal_std, [np.std(values) / norm], rtol=1e-10)
        assert_allclose(result.azimuthal_std, pyfai.std, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
