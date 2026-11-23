"""azint-cached (mask=None after construction) must equal stock masked azint."""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import same_bits

try:
    import numpy as np
    import azint
    from azint.detector import Detector
except ImportError:  # pragma: no cover - only in environments without azint
    azint = None


@unittest.skipIf(azint is None, "azint is not installed")
class AzintCachedTests(unittest.TestCase):
    def setUp(self):
        shape = (72, 96)
        rng = np.random.default_rng(7)
        y, x = np.ogrid[: shape[0], : shape[1]]
        self.mask = ((x % 31 < 2) | (np.hypot(y - 30, x - 50) < 5)).astype(np.uint8)
        frames = rng.poisson(100.0, size=(4, *shape)).astype(np.float64)
        frames[:, self.mask != 0] = 0  # as fixtures.py prepares every input
        self.frames = list(frames)
        poni = azint.Poni(
            det=Detector(1e-4, 1e-4, shape),
            dist=0.02,
            poni1=shape[0] * 1e-4 * 0.47,
            poni2=shape[1] * 1e-4 * 0.53,
            rot1=0.012,
            rot2=-0.008,
            rot3=0.02,
            wavelength=0.5e-10,
        )
        self.make = lambda: azint.AzimuthalIntegrator(
            poni,
            n_splitting=3,
            radial_bins=np.linspace(0.0, 25.0, 65),
            unit="2th",
            mask=self.mask,
            solid_angle=True,
            polarization_factor=0.965,
            error_model="poisson",
        )

    def assert_identical(self, stock, cached):
        for index in (0, 1):  # intensity and Poisson sigma
            self.assertTrue(same_bits(stock[index], cached[index]))

    def test_cached_norm_is_bit_identical_including_threaded_copies(self):
        stock_ai = self.make()
        ai = self.make()
        ai.mask = None
        stock = [stock_ai.integrate(frame) for frame in self.frames]
        for frame, expected in zip(self.frames, stock):
            self.assert_identical(expected, ai.integrate(frame))
        # The runner's per-thread engines are shallow copies sharing norm/sparse.
        engines = [ai] + [copy.copy(ai) for _ in range(3)]
        work = [(engines[i % 4], i % 4) for i in range(64)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(lambda t: (t[1], t[0].integrate(self.frames[t[1]])), work)
            )
        for index, result in results:
            self.assert_identical(stock[index], result)
        self.assertIsNone(ai.mask)
        self.assertTrue(all(engine.mask is None for engine in engines))


if __name__ == "__main__":
    unittest.main()
