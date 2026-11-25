"""IntegrationResult `*_array` properties: read-only float64 NumPy views of the
result's vectors (no copy) that match the list attributes, with 2D cakes shaped
(n_azimuthal, n_radial); and `BatchResult.result` returns one shared object."""
from pathlib import Path
import gc
import unittest

import numpy as np

import dioptrin

try:
    import tifffile
except ImportError:  # pragma: no cover - optional test dependency
    tifffile = None

ROOT = Path(__file__).resolve().parent.parent
TIFF = ROOT / "test_data" / "CeO2_Pilatus1M.tif"
PONI = ROOT / "test_data" / "v1.poni"
FIELDS = ("radial", "azimuthal", "intensity", "errors", "azimuthal_std")
# Asymmetric cake so a transposed view cannot pass.
N_RAD, N_AZ = 100, 36


def check_views(test, result):
    """Every *_array matches its list attribute (or is None with it)."""
    for name in FIELDS:
        values = getattr(result, name)
        arr = getattr(result, name + "_array")
        if values is None:
            test.assertIsNone(arr, name)
            continue
        test.assertIsInstance(arr, np.ndarray, name)
        test.assertEqual(arr.dtype, np.float64, name)
        test.assertFalse(arr.flags.writeable, name)
        test.assertIs(arr.base, result, name)
        expected = np.asarray(values, dtype=np.float64)
        np.testing.assert_array_equal(arr.ravel(), expected, err_msg=name)
        if result.n_azimuthal is not None and name in ("intensity", "errors"):
            shape = (result.n_azimuthal, result.n_radial)
            test.assertEqual(arr.shape, shape, name)
            np.testing.assert_array_equal(arr, expected.reshape(shape), err_msg=name)
        else:
            test.assertEqual(arr.shape, (len(values),), name)


@unittest.skipIf(tifffile is None, "tifffile not installed")
class ResultArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = tifffile.imread(TIFF).astype(np.float64)
        cls.ai = dioptrin.Integrator(str(PONI), method="counting", unit="2th_deg")

    def test_integrate1d(self):
        r = self.ai.integrate1d(self.image, 500)
        check_views(self, r)
        self.assertEqual(r.intensity_array.shape, (500,))
        self.assertIsNone(r.errors_array)
        self.assertIsNone(r.azimuthal_array)
        self.assertIsNone(r.azimuthal_std_array)
        r = self.ai.integrate1d(self.image, 500, errors=True)
        check_views(self, r)
        self.assertEqual(r.errors_array.shape, (500,))

    def test_integrate1d_azimuthal_std(self):
        r = self.ai.integrate1d(self.image, 300, azimuthal_std=True)
        self.assertIsNotNone(r.azimuthal_std)
        check_views(self, r)

    def test_integrate2d(self):
        for kwargs in ({}, {"errors": True}, {"azimuthal_range": (-np.pi, np.pi)}):
            r = self.ai.integrate2d(self.image, N_RAD, N_AZ, **kwargs)
            check_views(self, r)
            self.assertEqual(r.intensity_array.shape, (N_AZ, N_RAD))
        # Axis arrays stay 1D and match the cake's axes.
        r = self.ai.integrate2d(
            self.image, N_RAD, N_AZ, errors=True, azimuthal_range=(-np.pi, np.pi)
        )
        self.assertEqual(r.radial_array.shape, (N_RAD,))
        self.assertEqual(r.azimuthal_array.shape, (N_AZ,))
        self.assertEqual(r.errors_array.shape, (N_AZ, N_RAD))

    def test_2d_rows_are_azimuthal_sectors(self):
        """Physical layout check: an image lit only on one side of the beam
        centre gives cake rows (azimuthal sectors) that are either lit or dark,
        whereas the columns (radial bins) mix both."""
        h, w = self.image.shape
        ai = dioptrin.Integrator(str(PONI), method="counting", unit="2th_deg")
        cx = int(round(0.0799601126306 / 0.000172))  # Poni2 / pixel size
        half = np.zeros((h, w))
        half[:, :cx] = 1.0
        r = ai.integrate2d(half, N_RAD, N_AZ, azimuthal_range=(-np.pi, np.pi))
        cake = np.nan_to_num(r.intensity_array)
        # Rows are azimuthal sectors: the sectors on the dark side are 0 in
        # every radial bin, about half of them.
        dark_rows = int((cake.max(axis=1) < 0.01).sum())
        self.assertGreaterEqual(dark_rows, N_AZ // 4)
        self.assertLessEqual(dark_rows, 3 * N_AZ // 4)
        # Columns are radial bins and see both sides, so (almost) none is
        # dark. A transposed layout would swap these two counts.
        dark_cols = int((cake.max(axis=0) < 0.01).sum())
        self.assertLessEqual(dark_cols, N_RAD // 10)

    def test_read_only(self):
        r = self.ai.integrate1d(self.image, 200)
        arr = r.intensity_array
        with self.assertRaises(ValueError):
            arr[0] = 1.0
        with self.assertRaises(ValueError):
            arr.flags.writeable = True
        self.assertFalse(arr.flags.owndata)

    def test_zero_copy(self):
        r = self.ai.integrate1d(self.image, 200)
        a, b = r.intensity_array, r.intensity_array
        self.assertEqual(
            a.__array_interface__["data"][0], b.__array_interface__["data"][0]
        )
        self.assertTrue(np.shares_memory(a, b))

    def test_array_outlives_result(self):
        r = self.ai.integrate2d(self.image, N_RAD, N_AZ, errors=True)
        expected = np.asarray(r.intensity).reshape(N_AZ, N_RAD).copy()
        arr = r.intensity_array
        del r
        gc.collect()
        # Allocate a little to make reuse of freed memory likely.
        _ = [np.ones(N_RAD * N_AZ) for _ in range(20)]
        np.testing.assert_array_equal(arr, expected)
        self.assertIsInstance(arr.base, dioptrin.IntegrationResult)

    def test_batch1d(self):
        out = self.ai.batch1d([self.image, self.image], 400, errors=True)
        for item in out:
            self.assertTrue(item.is_ok())
            self.assertIs(item.result, item.result)
            check_views(self, item.result)
            self.assertEqual(item.result.intensity_array.shape, (400,))
        # The view keeps the result alive after the BatchResult goes away.
        arr = out[0].result.intensity_array
        expected = np.array(out[0].result.intensity)
        del out
        gc.collect()
        np.testing.assert_array_equal(arr, expected)

    def test_batch1d_iter(self):
        for item in self.ai.batch1d_iter([self.image], 400):
            self.assertIs(item.result, item.result)
            check_views(self, item.result)

    def test_batch2d(self):
        out = self.ai.batch2d([self.image, self.image], N_RAD, N_AZ, errors=True)
        out += self.ai.batch2d([str(TIFF)], N_RAD, N_AZ, errors=True)
        for item in out:
            self.assertTrue(item.is_ok(), item.error)
            self.assertIs(item.result, item.result)
            check_views(self, item.result)
            self.assertEqual(item.result.intensity_array.shape, (N_AZ, N_RAD))
            self.assertEqual(item.result.errors_array.shape, (N_AZ, N_RAD))

    def test_batch2d_iter(self):
        for item in self.ai.batch2d_iter([self.image], N_RAD, N_AZ):
            self.assertIs(item.result, item.result)
            check_views(self, item.result)

    def test_batch_failure_result_is_none(self):
        out = self.ai.batch1d([str(ROOT / "test_data" / "does_not_exist.tif")], 100)
        self.assertFalse(out[0].is_ok())
        self.assertIsNone(out[0].result)
        self.assertIsNotNone(out[0].error)

    def test_multi_integrator(self):
        multi = dioptrin.MultiIntegrator(
            [
                dioptrin.Integrator(
                    str(PONI), method="counting", unit="2th_deg", radial_range=(5.0, 40.0)
                )
            ]
        )
        r1 = multi.integrate1d([self.image], 300)
        check_views(self, r1)
        r2 = multi.integrate2d([self.image], N_RAD, N_AZ)
        check_views(self, r2)
        self.assertEqual(r2.intensity_array.shape, (N_AZ, N_RAD))

    @unittest.skipUnless(
        hasattr(dioptrin, "metal_available") and dioptrin.metal_available(),
        "Metal not available",
    )
    def test_metal_plans(self):
        shape = self.image.shape
        plan = self.ai.prepare_metal1d(shape, 400)
        r = plan.integrate1d(self.image, errors=True)
        check_views(self, r)
        for r in plan.batch1d([self.image, self.image]):
            check_views(self, r)
        plan2 = self.ai.prepare_metal2d(shape, N_RAD, N_AZ)
        r = plan2.integrate2d(self.image, errors=True)
        check_views(self, r)
        self.assertEqual(r.intensity_array.shape, (N_AZ, N_RAD))
        cpu = self.ai.integrate2d(self.image, N_RAD, N_AZ).intensity_array
        gpu = r.intensity_array
        ok = np.isfinite(cpu) & np.isfinite(gpu)
        np.testing.assert_allclose(gpu[ok], cpu[ok], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
