"""Fixed/dynamic mask lifecycle, Python API and numerical regression tests."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import dioptrin

PONI=dict(pixel1=1e-4,pixel2=1e-4,distance=.1,poni1=.0016,poni2=.0032,wavelength=1e-10)

class MaskModeTests(unittest.TestCase):
    def test_constructor_setter_validation_and_shared_integrator_guard(self):
        ai=dioptrin.Integrator.from_poni_dict(PONI,radial_range=(0.,.1))
        self.assertEqual(ai.mask_mode,'fixed')
        ai.set_mask_mode('dynamic');self.assertEqual(ai.mask_mode,'dynamic')
        with self.assertRaisesRegex(ValueError,'fixed.*dynamic'):ai.set_mask_mode('auto')
        self.assertEqual(ai.mask_mode,'dynamic')
        with self.assertRaisesRegex(ValueError,'fixed.*dynamic'):
            dioptrin.Integrator.from_poni_dict(PONI,mask_mode='auto')
        shared=dioptrin.MultiIntegrator([ai])
        with self.assertRaisesRegex(ValueError,'in use'):ai.set_mask_mode('fixed')
        del shared
        ai.set_mask_mode('fixed')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'test.poni'
            p.write_text('poni_version: 1\nPixelSize1: 0.0001\nPixelSize2: 0.0001\nDistance: 0.1\nPoni1: 0.0016\nPoni2: 0.0032\nWavelength: 1e-10\n')
            self.assertEqual(dioptrin.Integrator(str(p),mask_mode='dynamic').mask_mode,'dynamic')

    def test_mask_edits_unmasking_corrections_and_2d(self):
        image=np.random.default_rng(72).uniform(-2,200,(32,64));image[0,0]=np.nan
        static=np.zeros(image.shape,np.uint8);static[0,:]=1
        masks=[static,np.logical_or(static,image>170),np.ones_like(static),None,static]
        # The unmasked frame is finite so nonfinite input never contaminates
        # a numerical comparison; masked NaN is explicitly exercised otherwise.
        for method in ['counting','pixel_split','supersampled']:
            ai=dioptrin.Integrator.from_poni_dict(PONI,method=method,n_ss=3)
            for mask in masks:
                values=np.nan_to_num(image,nan=0.) if mask is None else image
                ai.set_mask_mode('fixed');ai.set_fast_math(False);ai.set_mask(mask)
                strict=ai.integrate1d(values,40,errors=True)
                strict2=ai.integrate2d(values,24,12,errors=True,azimuthal_range=(2.5,3.8))
                ai.set_mask_mode('dynamic')
                dynamic=ai.integrate1d(values,40,errors=True)
                for f in ['radial','intensity','errors']:
                    np.testing.assert_array_equal(np.asarray(getattr(dynamic,f)).view(np.uint64),np.asarray(getattr(strict,f)).view(np.uint64))
                ai.set_fast_math(True)
                dynamic=ai.integrate1d(values,40,errors=True)
                dynamic2=ai.integrate2d(values,24,12,errors=True,azimuthal_range=(2.5,3.8))
                for a,b in [(dynamic,strict),(dynamic2,strict2)]:
                    for f in ['intensity','errors']:
                        np.testing.assert_allclose(getattr(a,f),getattr(b,f),rtol=2e-12,atol=2e-12)
                ai.set_polarization_factor(.965)
                changed=ai.integrate1d(values,40,errors=True)
                ai.set_mask_mode('fixed')
                fixed=ai.integrate1d(values,40,errors=True)
                for f in ['intensity','errors']:
                    np.testing.assert_allclose(getattr(changed,f),getattr(fixed,f),rtol=2e-12,atol=2e-12)
                ai.set_polarization_factor(None)

    def test_dynamic_batch_borrows_inputs_and_agrees_with_parallel_single_frame(self):
        images=np.random.default_rng(13).uniform(1,200,(3,600,600));images.flags.writeable=False
        mask=np.zeros((600,600),np.uint8);mask[:3,:]=1
        ai=dioptrin.Integrator.from_poni_dict(PONI,mask_mode='dynamic')
        ai.set_mask(mask)
        expected=[ai.integrate1d(image,100,errors=True) for image in images]
        got=ai.batch1d(list(images),100,errors=True,num_workers=4,num_threads=1)
        for actual,wanted in zip(got,expected):
            self.assertTrue(actual.is_ok(),actual.error)
            for f in ['intensity','errors']:
                np.testing.assert_allclose(getattr(actual.result,f),getattr(wanted,f),rtol=2e-12,atol=2e-12)
        self.assertFalse(images.flags.writeable)

if __name__=='__main__':
    dioptrin.set_rayon_threads(4)
    unittest.main()
