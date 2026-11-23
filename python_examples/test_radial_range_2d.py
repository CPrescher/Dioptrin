"""Integrator 2D with an explicit `radial_range`: every method bins onto the
requested radial edges, drops pixels outside them and returns the radial axis."""
from pathlib import Path
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
METHODS = ("counting", "pixel_split", "supersampled", "pyfai_full")
N_AZ = 36
# In 2th_deg: FULL covers the whole detector (2theta < 40 deg) with 0.25 deg
# bins; SUB is bins 64..128 of it. Every edge is exact in binary.
FULL, N_FULL = (0.0, 64.0), 256
SUB, N_SUB, FIRST = (16.0, 32.0), 64, 64


def centers(lo, hi, n):
    return lo + (np.arange(n) + 0.5) * ((hi - lo) / n)


def cake(values, n_radial):
    return np.asarray(values).reshape(-1, n_radial)


def integrator(method, radial_range=None):
    return dioptrin.Integrator(
        str(PONI), method=method, n_ss=3, unit="2th_deg", radial_range=radial_range
    )


def results(ai, image, n_radial):
    """Every Python 2D path, as name -> IntegrationResult."""
    out = {
        "integrate2d": ai.integrate2d(image, n_radial, N_AZ),
        "errors": ai.integrate2d(image, n_radial, N_AZ, errors=True),
        "azimuthal_range": ai.integrate2d(
            image, n_radial, N_AZ, errors=True, azimuthal_range=(-np.pi, np.pi)
        ),
        "sector": ai.integrate2d(image, n_radial, 9, azimuthal_range=(0.2, 0.2 + np.pi / 2)),
        "tiff": ai.integrate2d(str(TIFF), n_radial, N_AZ),
    }
    for name, item in [
        ("batch2d", ai.batch2d([str(TIFF)], n_radial, N_AZ, errors=True)[0]),
        ("batch2d_iter", list(ai.batch2d_iter([image], n_radial, N_AZ))[0]),
        ("batch2d_iter range", list(
            ai.batch2d_iter([str(TIFF)], n_radial, N_AZ, azimuthal_range=(-np.pi, np.pi))
        )[0]),
    ]:
        assert item.is_ok(), item.error
        out[name] = item.result
    return out


@unittest.skipIf(tifffile is None, "tifffile not installed")
class RadialRange2dTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = tifffile.imread(TIFF).astype(np.float64)

    def test_sub_range_matches_full_range_on_matching_edges(self):
        """The radial axis is the centres of n equal bins between the requested
        edges, and the bins of a sub-range equal the matching bins of a range
        covering the detector. Counting, pixel splitting and supersampling
        agree bit for bit (also in the end bins, so nothing outside the
        sub-range is clamped into them); pyfai_full renormalizes pixels
        clipped by the range to their part inside, so it is compared away from
        the sub-range edges, to rounding."""
        for method in METHODS:
            full = results(integrator(method, FULL), self.image, N_FULL)
            sub = results(integrator(method, SUB), self.image, N_SUB)
            for name in full:
                with self.subTest(method=method, path=name):
                    a, b = full[name], sub[name]
                    np.testing.assert_allclose(a.radial, centers(*FULL, N_FULL), rtol=1e-15, atol=0)
                    np.testing.assert_allclose(b.radial, centers(*SUB, N_SUB), rtol=1e-15, atol=0)
                    self.assertEqual(b.n_radial, N_SUB)
                    pairs = [(a.intensity, b.intensity)]
                    if a.errors is not None:
                        pairs.append((a.errors, b.errors))
                    for x, y in pairs:
                        x = cake(x, N_FULL)[:, FIRST:FIRST + N_SUB]
                        y = cake(y, N_SUB)
                        if method == "pyfai_full":
                            np.testing.assert_allclose(
                                x[:, 2:-2], y[:, 2:-2], rtol=1e-9, atol=1e-12
                            )
                        else:
                            np.testing.assert_array_equal(x, y)
                    self.assertGreater(np.count_nonzero(cake(b.intensity, N_SUB)), 100)

    def test_pixels_outside_the_range_are_left_out(self):
        # Light only pixels below 12 deg 2theta (untilted estimate; the real
        # tilt of ~1 deg leaves them well below the sub-range starting at 16).
        ii, jj = np.indices(self.image.shape)
        p1 = (ii + 0.5) * 0.000172 - 0.0872948482846
        p2 = (jj + 0.5) * 0.000172 - 0.0799601126306
        low = (np.degrees(np.arctan(np.hypot(p1, p2) / 0.208651380603)) < 12.0).astype(float)
        for method in METHODS:
            ai = integrator(method, SUB)
            with self.subTest(method=method):
                for out in (
                    ai.integrate2d(low, N_SUB, N_AZ),
                    ai.integrate2d(low, N_SUB, N_AZ, azimuthal_range=(-np.pi, np.pi)),
                ):
                    values = np.asarray(out.intensity)
                    self.assertFalse(np.any(values[np.isfinite(values)]))
                # The same pixels do land in the full range.
                full = integrator(method, FULL).integrate2d(low, N_FULL, N_AZ)
                self.assertTrue(np.nansum(full.intensity) > 0)

    def test_default_calls_are_unchanged(self):
        """Without a radial_range the default paths keep `radial = None`
        except on an azimuthal range, which reports its default radial edges;
        an explicit range changes the bins."""
        for method in METHODS:
            ai = integrator(method)
            plain = ai.integrate2d(self.image, 100, N_AZ)
            self.assertIsNone(plain.radial)
            self.assertIsNone(ai.integrate2d(self.image, 100, N_AZ, errors=True).radial)
            self.assertIsNone(ai.batch2d([str(TIFF)], 100, N_AZ)[0].result.radial)
            ranged = ai.integrate2d(self.image, 100, N_AZ, azimuthal_range=(-np.pi, np.pi))
            self.assertEqual(len(ranged.radial), 100)
            again = integrator(method).integrate2d(self.image, 100, N_AZ)
            np.testing.assert_array_equal(plain.intensity, again.intensity)
            explicit = integrator(method, SUB).integrate2d(self.image, 100, N_AZ)
            self.assertFalse(np.array_equal(plain.intensity, explicit.intensity))

    def test_negative_minimum_is_rejected_in_2d(self):
        ai = integrator("pixel_split", (-1.0, 30.0))
        self.assertEqual(len(ai.integrate1d(self.image, 50).radial), 50)  # 1D unchanged
        for call in (
            lambda: ai.integrate2d(self.image, 50, N_AZ),
            lambda: ai.integrate2d(self.image, 50, N_AZ, azimuthal_range=(0, 1)),
            lambda: ai.batch2d([str(TIFF)], 50, N_AZ),
            lambda: ai.batch2d_iter([self.image], 50, N_AZ),
        ):
            with self.assertRaisesRegex(ValueError, "radial_range .*nonnegative"):
                call()


if __name__ == "__main__":
    unittest.main()
