"""Regression: calibration caching must never cache image-dependent sums."""
import unittest

import dioptrin
import numpy as np


class ImageSwitchingTests(unittest.TestCase):
    def test_alternating_images_and_poisson_errors(self):
        poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1,
                    poni1=0.0016, poni2=0.0032, wavelength=1e-10)
        images = [np.full((32, 64), value, dtype=np.float64)
                  for value in (9.0, 25.0)]
        mask = (np.indices(images[0].shape)[0] < 3).astype(np.uint8)
        order = [0, 1, 0, 1, 1, 0, 0, 1]
        for method in ("supersampled", "pixel_split"):
            for fast in (False, True):
                with self.subTest(method=method, fast_math=fast):
                    ai = dioptrin.Integrator.from_poni_dict(
                        poni, method=method, n_ss=3,
                        polarization_factor=0.965, fast_math=fast)
                    ai.set_mask(mask)
                    reference = [ai.integrate1d(im, 40, errors=True)
                                 for im in images]
                    a, b = reference
                    self.assertTrue(np.any(np.asarray(a.intensity) > 0))
                    # Independent expectations for constant positive counts:
                    # intensity scales with counts, Poisson sigma with sqrt.
                    np.testing.assert_allclose(b.intensity,
                        np.asarray(a.intensity) * (25.0 / 9.0),
                        rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(b.errors,
                        np.asarray(a.errors) * (5.0 / 3.0),
                        rtol=2e-12, atol=2e-12)
                    sequential = [ai.integrate1d(images[j], 40, errors=True)
                                  for j in order]
                    batch = ai.batch1d([images[j] for j in order], 40,
                                      errors=True, num_workers=8,
                                      num_threads=1)
                    self.assertTrue(all(item.is_ok() for item in batch))
                    for results in (sequential, [item.result for item in batch]):
                        for j, result in zip(order, results):
                            for field in ("radial", "intensity", "errors"):
                                actual = np.asarray(getattr(result, field))
                                expected = np.asarray(getattr(reference[j], field))
                                self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
