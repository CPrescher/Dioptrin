"""Shared-plan API, changing masks, snapshot isolation and thread ownership."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import unittest

import dioptrin
import numpy as np

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=.1,
            poni1=.0016, poni2=.0032, wavelength=1e-10)


class DynamicPlanTests(unittest.TestCase):
    def test_shared_workers_match_independent_integrators(self):
        rng = np.random.default_rng(9834)
        images = rng.uniform(-4, 400, (16, 32, 64))
        masks = (images > 370).astype(np.uint8)
        masks[-1] = 1
        for method in ('counting', 'pixel_split', 'supersampled'):
            for fast in (False, True):
                source = dioptrin.Integrator.from_poni_dict(PONI, method=method,
                    n_ss=2, unit='q_A^-1', polarization_factor=.965, fast_math=fast)
                plan = source.prepare_dynamic1d((32,64),40)
                workers = [plan.worker() for _ in range(8)]
                self.assertEqual({w.shared_plan_id for w in workers}, {plan.shared_plan_id})
                self.assertTrue(all(w.num_threads == 1 for w in workers))
                del plan, source
                gc.collect()
                for errors, spread in ((False,False),(True,False),(False,True),(True,True)):
                    references = []
                    for image, mask in zip(images, masks):
                        reference = dioptrin.Integrator.from_poni_dict(PONI, method=method,
                            n_ss=2,unit='q_A^-1',polarization_factor=.965,fast_math=fast,
                            mask_mode='dynamic')
                        reference.set_mask(mask)
                        references.append(reference.integrate1d(image,40,errors=errors,azimuthal_std=spread))
                    barrier = threading.Barrier(8)
                    def run(index):
                        barrier.wait(timeout=10)
                        return [(i, workers[index].integrate1d(images[i],masks[i],
                                    errors=errors,azimuthal_std=spread))
                                for i in range(index,len(images),8)]
                    with ThreadPoolExecutor(8) as pool:
                        output = [item for group in pool.map(run,range(8)) for item in group]
                    for i, result in output:
                        for field in ('radial','intensity','errors','azimuthal_std'):
                            actual = getattr(result,field)
                            expected = getattr(references[i],field)
                            if expected is None:
                                self.assertIsNone(actual)
                            else:
                                np.testing.assert_allclose(actual,expected,rtol=5e-12,atol=1e-9)

    def test_shape_validation_and_snapshot_isolation(self):
        image = np.ones((32,64),np.float64)
        source = dioptrin.Integrator.from_poni_dict(PONI,polarization_factor=.965)
        plan = source.prepare_dynamic1d(image.shape,40)
        worker = plan.worker()
        before = worker.integrate1d(image).intensity
        source.set_polarization_factor(.1)
        source.set_unit('q_A^-1')
        source.set_mask(np.ones(image.shape,np.uint8))
        np.testing.assert_array_equal(worker.integrate1d(image).intensity,before)
        with self.assertRaises(ValueError):worker.integrate1d(image[:1])
        with self.assertRaises(ValueError):worker.integrate1d(image,np.zeros((1,64),np.uint8))
        with self.assertRaises(ValueError):worker.integrate1d(np.asfortranarray(image))
        with self.assertRaises(ValueError):worker.integrate1d(image,np.asfortranarray(np.zeros(image.shape,np.uint8)))
        with self.assertRaises(ValueError):plan.worker(0)
        with self.assertRaises(ValueError):source.prepare_dynamic1d((0,64),40)
        with self.assertRaises(ValueError):source.prepare_dynamic1d(image.shape,0)

if __name__ == '__main__':unittest.main()
