"""Integrator 2D `azimuthal_range=(min, max)`: true ranges, sectors included."""
from pathlib import Path
import unittest

import numpy as np

import dioptrin

# Untilted detector with the beam on a pixel corner: pixel azimuth is
# arctan2(p1, p2) and every direction around the beam is covered.
PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.1, poni1=0.0032, poni2=0.0048, wavelength=1e-10)
SHAPE = (64, 96)
N_RADIAL, N_AZIMUTHAL = 8, 36
METHODS = ("counting", "pixel_split", "supersampled")
ROOT = Path(__file__).resolve().parent.parent
TIFF = ROOT / "test_data" / "CeO2_Pilatus1M.tif"
TIFF_PONI = ROOT / "test_data" / "v1.poni"
RANGES = [(0.0, np.pi / 2), (2.5, 2.5 + np.pi / 2), (-np.pi, np.pi), (0.3, 0.3 + 2 * np.pi)]


def pixel_azimuth():
    i, j = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    p1 = (i + 0.5) * PONI["pixel1"] - PONI["poni1"]
    p2 = (j + 0.5) * PONI["pixel2"] - PONI["poni2"]
    return np.arctan2(p1, p2), np.hypot(p1, p2) / PONI["pixel1"]


def cake(result):
    return np.asarray(result.intensity).reshape(result.n_azimuthal, result.n_radial)


class AzimuthalRangeTests(unittest.TestCase):
    def test_bins_cover_exactly_their_edges(self):
        chi, radius = pixel_azimuth()
        ones = np.ones(SHAPE)
        for lo, hi in RANGES:
            step = (hi - lo) / N_AZIMUTHAL
            centre = 0.5 * (lo + hi)
            # unwrapped around the range centre: no seam inside a sector
            unwrapped = centre - np.pi + np.mod(chi - centre + np.pi, 2 * np.pi)
            ai = dioptrin.Integrator.from_poni_dict(PONI, method="counting")
            norm = ai.integrate2d(ones, N_RADIAL, N_AZIMUTHAL, azimuthal_range=(lo, hi))
            signal = ai.integrate2d(unwrapped, N_RADIAL, N_AZIMUTHAL, azimuthal_range=(lo, hi))
            np.testing.assert_allclose(norm.azimuthal, lo + (np.arange(N_AZIMUTHAL) + 0.5) * step, atol=1e-12)
            filled = cake(norm) > 0
            mean = np.where(filled, cake(signal) / np.where(filled, cake(norm), 1), np.nan)
            if hi - lo > 6:  # full circle: the unwrap seam sits in the end bins
                mean[[0, -1]] = np.nan
            low = lo + np.arange(N_AZIMUTHAL)[:, None] * step
            inside = np.isnan(mean) | ((mean >= low - 1e-9) & (mean <= low + step + 1e-9))
            self.assertTrue(inside.all(), f"bin means outside their bins for {(lo, hi)}")
            self.assertGreaterEqual(filled.any(axis=1).sum(), N_AZIMUTHAL - 2)
            # Pixels whose centre is 0.1 rad outside (and 15 px from the beam)
            # contribute nothing, for every method.
            far = ((radius > 15) & ((unwrapped < lo - 0.1) | (unwrapped > hi + 0.1))).astype(float)
            for method in METHODS:
                ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3)
                out = ai.integrate2d(far, N_RADIAL, N_AZIMUTHAL, azimuthal_range=(lo, hi))
                self.assertFalse(np.any(cake(out)), f"{method} {(lo, hi)} picked up outside pixels")

    def test_explicit_full_circle_is_close_to_default(self):
        # The default grid spans the detector's own azimuth extent instead of
        # exactly [-pi, pi), so the cakes agree closely but not bin for bin.
        image = np.random.default_rng(2).uniform(50, 100, SHAPE)
        for method in METHODS:
            ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3)
            default = cake(ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL))
            explicit = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=True, azimuthal_range=(-np.pi, np.pi))
            self.assertEqual(len(explicit.errors), N_RADIAL * N_AZIMUTHAL)
            both = (default != 0) & (cake(explicit) != 0)
            self.assertGreater(both.mean(), 0.5)
            rel = np.abs(cake(explicit) - default)[both] / default[both]
            self.assertLess(np.median(rel), 0.05, method)

    def test_invalid_ranges_raise_value_error(self):
        image = np.ones(SHAPE)
        ai = dioptrin.Integrator.from_poni_dict(PONI, method="counting")
        for rng in [(0.0, 7.0), (np.pi, -np.pi), (1.0, 1.0), (float("nan"), 1.0), (-7.0, -6.0)]:
            for errors in (False, True):
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.batch2d_iter([image], N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.batch2d_iter(iter([image]), N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)

    def test_batch_iterators_match_integrate2d_for_arrays(self):
        chi, _ = pixel_azimuth()
        for errors in (False, True):
            ai = dioptrin.Integrator.from_poni_dict(PONI, method="pixel_split")
            ref = ai.integrate2d(chi, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=(0.0, np.pi / 2))
            for images in ([chi], iter([chi])):
                r = list(ai.batch2d_iter(images, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=(0.0, np.pi / 2)))[0]
                self.assertTrue(r.is_ok(), r.error)
                np.testing.assert_array_equal(r.result.intensity, ref.intensity)
                np.testing.assert_array_equal(r.result.azimuthal, ref.azimuthal)
                self.assertEqual(r.result.errors, ref.errors)

    @unittest.skipUnless(TIFF.exists(), "CeO2 TIFF not available")
    def test_path_and_batch_entry_points(self):
        files = [str(TIFF)]
        for method in ("counting", "pixel_split"):
            ai = dioptrin.Integrator(str(TIFF_PONI), method=method)
            for errors in (False, True):
                for rng in [(0.0, 1.57), (-np.pi, np.pi)]:
                    ref = ai.integrate2d(str(TIFF), 50, 90, errors=errors, azimuthal_range=rng)
                    self.assertEqual(len(ref.azimuthal), 90)
                    self.assertAlmostEqual(ref.azimuthal[0], rng[0] + (rng[1] - rng[0]) / 180)
                    # every azimuthal bin of the sector receives data
                    self.assertEqual((cake(ref) != 0).any(axis=1).sum(), 90)
                    batch = ai.batch2d(files, 50, 90, errors=errors, azimuthal_range=rng)[0]
                    streamed = list(ai.batch2d_iter(files, 50, 90, errors=errors, azimuthal_range=rng))[0]
                    for r in (batch, streamed):
                        self.assertTrue(r.is_ok(), r.error)
                        np.testing.assert_array_equal(r.result.intensity, ref.intensity)
                        np.testing.assert_array_equal(r.result.azimuthal, ref.azimuthal)
                        self.assertEqual(r.result.errors, ref.errors)
            with self.assertRaises(ValueError):
                ai.batch2d(files, 50, 90, azimuthal_range=(0.0, 7.0))
            with self.assertRaises(ValueError):
                ai.integrate2d(str(TIFF), 50, 90, azimuthal_range=(0.0, 7.0))


if __name__ == "__main__":
    unittest.main()
