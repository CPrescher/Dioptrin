"""Numerical/API regression coverage for default fast and explicit strict CSR accumulation."""
import unittest
import numpy as np
import dioptrin

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1,
            poni1=0.0016, poni2=0.0032, wavelength=1e-10)

class FastMathTests(unittest.TestCase):
    def test_fast_default_and_explicit_strict_opt_out(self):
        import tempfile
        ai = dioptrin.Integrator.from_poni_dict(PONI, method='supersampled', n_ss=3)
        self.assertTrue(ai.fast_math)
        image = np.random.default_rng(41).uniform(1, 200, (32, 64))
        reference = ai.integrate1d(image, 40, errors=True)
        ai.set_fast_math(False)
        self.assertFalse(ai.fast_math)
        ai.integrate1d(image, 40, errors=True)
        ai.set_fast_math(True)
        restored = ai.integrate1d(image, 40, errors=True)
        for field in ('radial', 'intensity', 'errors'):
            self.assertEqual(np.asarray(getattr(restored, field)).tobytes(),
                             np.asarray(getattr(reference, field)).tobytes())
        strict = dioptrin.Integrator.from_poni_dict(PONI, fast_math=False)
        self.assertFalse(strict.fast_math)
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            path = Path(directory) / "fixture.poni"
            path.write_text('poni_version: 1\nPixelSize1: 0.0001\nPixelSize2: 0.0001\nDistance: 0.1\nPoni1: 0.0016\nPoni2: 0.0032\nWavelength: 1e-10\n')
            self.assertTrue(dioptrin.Integrator(str(path)).fast_math)
            self.assertFalse(dioptrin.Integrator(str(path), fast_math=False).fast_math)

    def test_strict_toggle_constructor_and_batch_poisson(self):
        image = np.random.default_rng(19).uniform(-2, 200, (32, 64))
        mask = (np.indices(image.shape)[0] < 3).astype(np.uint8)
        for method in ('counting', 'pixel_split', 'supersampled'):
            with self.subTest(method=method):
                ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3, fast_math=False)
                self.assertFalse(ai.fast_math)
                ai.set_mask(mask)
                reference = ai.integrate1d(image, 40, errors=True)
                ai.set_fast_math(True)
                self.assertTrue(ai.fast_math)
                fast = ai.integrate1d(image, 40, errors=True)
                np.testing.assert_array_equal(fast.radial, reference.radial)
                for field in ('intensity','errors'):
                    np.testing.assert_allclose(getattr(fast,field),getattr(reference,field),rtol=2e-12,atol=2e-12)
                batch = ai.batch1d([image, image], 40, errors=True, num_workers=4, num_threads=1)
                for item in batch:
                    self.assertTrue(item.is_ok(), item.error)
                    np.testing.assert_array_equal(item.result.intensity, fast.intensity)
                    np.testing.assert_array_equal(item.result.errors, fast.errors)
                ai.set_fast_math(False)
                restored = ai.integrate1d(image,40,errors=True)
                for field in ('radial','intensity','errors'):
                    np.testing.assert_array_equal(np.asarray(getattr(restored,field)).view(np.uint64),np.asarray(getattr(reference,field)).view(np.uint64))
                constructor = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3, fast_math=True)
                constructor.set_mask(mask)
                np.testing.assert_array_equal(constructor.integrate1d(image,40,errors=True).intensity,fast.intensity)

    def test_2d_and_wrapped_range_match_normal_mode(self):
        image = np.random.default_rng(3).uniform(1, 30, (32, 64))
        for method in ('counting','pixel_split','supersampled'):
            for azimuth in (None, (2.5,3.8)):
                ai = dioptrin.Integrator.from_poni_dict(PONI,method=method,n_ss=3,fast_math=False)
                reference = ai.integrate2d(image,24,12,errors=True,azimuthal_range=azimuth)
                ai.set_fast_math(True)
                fast = ai.integrate2d(image,24,12,errors=True,azimuthal_range=azimuth)
                for field in ('intensity','errors'):
                    np.testing.assert_allclose(getattr(fast,field),getattr(reference,field),rtol=2e-12,atol=2e-12)

    def test_spread_and_histogram_retain_their_existing_algorithms(self):
        image = np.random.default_rng(9).uniform(1,100,(32,64))
        for method in ('counting','pixel_split','supersampled','histogram'):
            ai = dioptrin.Integrator.from_poni_dict(PONI,method=method,n_ss=3,fast_math=False)
            reference = ai.integrate1d(image,30,errors=True,azimuthal_std=True)
            ai.set_fast_math(True)
            fast = ai.integrate1d(image,30,errors=True,azimuthal_std=True)
            for field in ('intensity','errors','azimuthal_std'):
                np.testing.assert_array_equal(getattr(fast,field),getattr(reference,field))

if __name__ == '__main__':
    dioptrin.set_rayon_threads(4)
    unittest.main()
