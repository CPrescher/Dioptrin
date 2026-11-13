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
            for method in METHODS + ("pyfai_full",):
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
        for rng in [(0.0, 7.0), (-4 * np.pi, 4 * np.pi), (np.pi, -np.pi), (1.0, 1.0), (float("nan"), 1.0)]:
            for errors in (False, True):
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.batch2d_iter([image], N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)
                with self.assertRaises(ValueError, msg=str(rng)):
                    ai.batch2d_iter(iter([image]), N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)

    def test_distant_full_turn_is_corrected_in_results_and_axis(self):
        image = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        for method in (*METHODS, "pyfai_full"):
            ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3)
            for errors in (False, True):
                expected = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=(-np.pi, np.pi))
                rng = (-3 * np.pi, -np.pi)
                direct = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng)
                results = [direct]
                for source in ([image], iter([image])):
                    batch = list(ai.batch2d_iter(source, N_RADIAL, N_AZIMUTHAL, errors=errors, azimuthal_range=rng))[0]
                    self.assertTrue(batch.is_ok(), batch.error)
                    results.append(batch.result)
                for result in results:
                    np.testing.assert_array_equal(result.intensity, expected.intensity)
                    if errors:
                        np.testing.assert_array_equal(result.errors, expected.errors)
                    np.testing.assert_array_equal(result.azimuthal, expected.azimuthal)
        # TIFF paths and the eager file batch use the same range validation.
        ai = dioptrin.Integrator(str(TIFF_PONI), method="counting")
        expected = ai.integrate2d(str(TIFF), 20, 36, azimuthal_range=(-np.pi, np.pi))
        shifted = ai.batch2d([str(TIFF)], 20, 36, azimuthal_range=(-3 * np.pi, -np.pi))[0]
        self.assertTrue(shifted.is_ok(), shifted.error)
        np.testing.assert_array_equal(shifted.result.intensity, expected.intensity)
        np.testing.assert_array_equal(shifted.result.azimuthal, expected.azimuthal)
        with self.assertRaisesRegex(ValueError, "one full turn"):
            ai.batch2d([str(TIFF)], 20, 36, azimuthal_range=(-4 * np.pi, 4 * np.pi))

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

    def test_nan_pixel_affects_only_its_bins(self):
        # As without a range: no error, and only the bins the pixel feeds are NaN.
        chi, radius = pixel_azimuth()
        pixel = np.argwhere((chi > 0.6) & (chi < 0.9) & (radius > 20))[0]
        image = np.random.default_rng(3).uniform(50, 100, SHAPE)
        image[tuple(pixel)] = np.nan
        one_hot = np.zeros(SHAPE)
        one_hot[tuple(pixel)] = 1.0
        for method in METHODS + ("pyfai_full",):
            ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3)
            self.assertTrue(np.isnan(cake(ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL))).any())
            for rng in [(0.0, np.pi / 2), (-np.pi, np.pi)]:
                fed = cake(ai.integrate2d(one_hot, N_RADIAL, N_AZIMUTHAL, azimuthal_range=rng)) != 0
                self.assertTrue(fed.any())
                r = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=True, azimuthal_range=rng)
                np.testing.assert_array_equal(np.isnan(cake(r)), fed, err_msg=f"{method} {rng}")
                b = list(ai.batch2d_iter([image], N_RADIAL, N_AZIMUTHAL, azimuthal_range=rng))[0]
                self.assertTrue(b.is_ok(), b.error)
                np.testing.assert_array_equal(np.isnan(cake(b.result)), fed)
            other = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, errors=True, azimuthal_range=(-2.0, -1.0))
            self.assertTrue(np.isfinite(other.intensity).all() and np.isfinite(other.errors).all())

    def test_counting_keeps_the_outermost_pixel(self):
        # Close detector: the largest 2θ (deg) and q (nm^-1) exceed 32.
        close = dict(PONI, distance=0.005)
        _, radius = pixel_azimuth()
        outer = np.unravel_index(np.argmax(radius), SHAPE)
        one_hot = np.zeros(SHAPE)
        one_hot[outer] = 1000.0
        for unit in ("2th_deg", "q_nm^-1", "2th_rad", "q_A^-1"):
            ai = dioptrin.Integrator.from_poni_dict(close, method="counting", unit=unit)
            default = np.sum(ai.integrate2d(one_hot, N_RADIAL, N_AZIMUTHAL).intensity)
            ranged = np.sum(ai.integrate2d(one_hot, N_RADIAL, N_AZIMUTHAL, azimuthal_range=(-np.pi, np.pi)).intensity)
            self.assertGreater(default, 0)
            self.assertGreater(ranged, 0, unit)

    def test_multi_worker_batches_with_a_range(self):
        chi, _ = pixel_azimuth()
        images = [chi * (k + 1) for k in range(5)]
        ai = dioptrin.Integrator.from_poni_dict(PONI, method="supersampled", n_ss=3)
        rng = (2.5, 2.5 + np.pi / 2)
        refs = [ai.integrate2d(img, N_RADIAL, N_AZIMUTHAL, errors=True, azimuthal_range=rng) for img in images]
        for source in (images, iter(images)):
            got = list(ai.batch2d_iter(source, N_RADIAL, N_AZIMUTHAL, errors=True, azimuthal_range=rng, num_workers=3, num_threads=1))
            self.assertEqual(len(got), len(images))
            for r, ref in zip(got, refs):
                self.assertTrue(r.is_ok(), r.error)
                np.testing.assert_array_equal(r.result.intensity, ref.intensity)
                np.testing.assert_array_equal(r.result.errors, ref.errors)

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
            many = ai.batch2d(files * 4, 50, 90, errors=True, azimuthal_range=(0.0, 1.57), num_workers=2, num_threads=2)
            ref = ai.integrate2d(str(TIFF), 50, 90, errors=True, azimuthal_range=(0.0, 1.57))
            self.assertEqual(len(many), 4)
            for r in many:
                self.assertTrue(r.is_ok(), r.error)
                np.testing.assert_array_equal(r.result.intensity, ref.intensity)
                self.assertEqual(r.result.errors, ref.errors)
            with self.assertRaises(ValueError):
                ai.batch2d(files, 50, 90, azimuthal_range=(0.0, 7.0))
            with self.assertRaises(ValueError):
                ai.integrate2d(str(TIFF), 50, 90, azimuthal_range=(0.0, 7.0))

    def test_bracket_gives_rows_centred_on_the_requested_directions(self):
        bracket = np.radians(2.5)
        image = np.random.default_rng(1).uniform(10, 100, SHAPE)
        for method in METHODS:
            ai = dioptrin.Integrator.from_poni_dict(PONI, method=method, n_ss=3)
            got = ai.integrate2d(image, N_RADIAL, azimuthal_bracket=bracket, errors=True)
            self.assertEqual(got.n_azimuthal, 72)
            np.testing.assert_allclose(np.degrees(got.azimuthal), np.arange(72) * 5.0, atol=1e-9)
            ref = ai.integrate2d(image, N_RADIAL, 72, errors=True,
                                 azimuthal_range=(-bracket, 2 * np.pi - bracket))
            np.testing.assert_allclose(got.intensity, ref.intensity, rtol=1e-9)
            np.testing.assert_allclose(got.errors, ref.errors, rtol=1e-9)
            # a sector: centres 0, 10, ..., 80 degrees
            sector = ai.integrate2d(image, N_RADIAL, azimuthal_range=(0, np.pi / 2),
                                    azimuthal_bracket=np.radians(5))
            np.testing.assert_allclose(np.degrees(sector.azimuthal), np.arange(9) * 10.0, atol=1e-9)
            for r in list(ai.batch2d_iter([image, image], N_RADIAL, azimuthal_bracket=bracket, errors=True)):
                self.assertTrue(r.is_ok(), r.error)
                np.testing.assert_array_equal(r.result.intensity, got.intensity)
                np.testing.assert_array_equal(r.result.azimuthal, got.azimuthal)
        ai = dioptrin.Integrator.from_poni_dict(PONI)
        for kwargs in (dict(), dict(n_azimuthal=72, azimuthal_bracket=bracket),
                       dict(azimuthal_bracket=np.radians(3.5)), dict(azimuthal_bracket=0.0)):
            with self.assertRaises(ValueError):
                ai.integrate2d(image, N_RADIAL, **kwargs)
            with self.assertRaises(ValueError):
                ai.batch2d_iter([image], N_RADIAL, **kwargs)
            with self.assertRaises(ValueError):
                ai.batch2d([str(TIFF)], N_RADIAL, **kwargs)

    @unittest.skipUnless(TIFF.exists(), "CeO2 TIFF not available")
    def test_bracket_in_file_batches(self):
        ai = dioptrin.Integrator(str(TIFF_PONI), method="pixel_split")
        bracket = np.radians(5)
        ref = ai.integrate2d(str(TIFF), 50, azimuthal_bracket=bracket)
        self.assertEqual(ref.n_azimuthal, 36)
        batch = ai.batch2d([str(TIFF)], 50, azimuthal_bracket=bracket)[0]
        self.assertTrue(batch.is_ok(), batch.error)
        np.testing.assert_array_equal(batch.result.intensity, ref.intensity)
        np.testing.assert_array_equal(batch.result.azimuthal, ref.azimuthal)



if __name__ == "__main__":
    unittest.main()
