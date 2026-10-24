"""Regression coverage for synchronous batches borrowing NumPy image data."""
from pathlib import Path
import tempfile
import unittest

import dioptrin
import numpy as np
from numpy.testing import assert_array_equal


PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1,
            poni1=0.0016, poni2=0.0032, wavelength=1e-10)


def integrator(method="pixel_split"):
    return dioptrin.Integrator.from_poni_dict(
        PONI, method=method, n_ss=3, polarization_factor=0.965)


class BatchArrayBorrowingTests(unittest.TestCase):
    def compare_results(self, actual, expected):
        for field in ("radial", "intensity", "errors", "azimuthal_std"):
            value, reference = getattr(actual, field), getattr(expected, field)
            if reference is None:
                self.assertIsNone(value)
            else:
                assert_array_equal(value, reference, err_msg=field)

    def test_views_aliases_and_optional_outputs_match_individual_calls(self):
        base = np.random.default_rng(42).uniform(-2, 200, (36, 64))
        view = base[2:34]
        readonly = np.array(view * 2)
        readonly.flags.writeable = False
        frames = [view, readonly, view]
        before = base.copy()
        mask = np.zeros(view.shape, dtype=np.uint8)
        mask[:3] = 1
        for method in ("counting", "pixel_split", "supersampled", "histogram"):
            for errors in (False, True):
                for spread in (False, True):
                    with self.subTest(method=method, errors=errors, spread=spread):
                        reference = integrator(method)
                        reference.set_mask(mask)
                        expected = [reference.integrate1d(
                            x, 40, errors=errors, azimuthal_std=spread) for x in frames]
                        # Start cold so parallel workers also exercise lazy setup.
                        batch = integrator(method)
                        batch.set_mask(mask)
                        actual = batch.batch1d(frames, 40, errors, 4, 1,
                                               azimuthal_std=spread)
                        self.assertEqual(len(actual), len(frames))
                        for i, (item, wanted) in enumerate(zip(actual, expected)):
                            self.assertTrue(item.is_ok(), item.error)
                            self.assertEqual(item.filename, f"array[{i}]")
                            self.compare_results(item.result, wanted)
        assert_array_equal(base, before)
        self.assertTrue(view.flags.writeable)
        self.assertFalse(readonly.flags.writeable)

    def test_readonly_memory_mapped_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npy"
            np.save(path, np.arange(3 * 32 * 64, dtype=np.float64).reshape(3, 32, 64))
            images = np.load(path, mmap_mode="r")
            ai = integrator()
            expected = [ai.integrate1d(x, 40, errors=True) for x in images]
            actual = ai.batch1d(list(images), 40, errors=True)
            for item, wanted in zip(actual, expected):
                self.compare_results(item.result, wanted)
            self.assertFalse(images.flags.writeable)

    def test_invalid_inputs_rejected_and_integrator_remains_usable(self):
        image = np.ones((32, 64), dtype=np.float64)
        ai = integrator()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            ai.batch1d([image, image[:, ::2]], 40)
        with self.assertRaises(TypeError):
            ai.batch1d([image.astype(np.float32)], 40)
        actual = ai.batch1d([image], 40)[0]
        self.assertTrue(actual.is_ok(), actual.error)
        self.compare_results(actual.result, ai.integrate1d(image, 40))
        self.assertEqual(ai.batch1d([], 40), [])

    def test_different_shapes_preserve_order_with_shared_geometry(self):
        frames = [np.ones((32, 64)), np.full((24, 48), 2.0), np.full((32, 64), 3.0)]
        reference = integrator()
        expected = [reference.integrate1d(x, 40, errors=True) for x in frames]
        actual = integrator().batch1d(frames, 40, errors=True)
        for item, wanted in zip(actual, expected):
            self.compare_results(item.result, wanted)


if __name__ == "__main__":
    unittest.main()
