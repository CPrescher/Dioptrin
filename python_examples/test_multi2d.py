"""Native multi-detector cakes: Python API and matched-grid pyFAI checks.

Run with `python python_examples/test_multi2d.py`; requires numpy, fabio,
pyFAI (validated with 2025.12.1). No reference images or network needed.
"""
import json
import tempfile
import unittest
from pathlib import Path

import dioptrin
import numpy as np


PONI = dict(pixel1=1e-4, pixel2=1.2e-4, distance=0.1,
            poni1=0.00131, poni2=0.00113, rot1=0.0, rot2=0.0,
            rot3=0.0, wavelength=1e-10)


def integrator(poni=PONI, method="counting", radial_range=(0.0, 0.2), **kw):
    return dioptrin.Integrator.from_poni_dict(
        poni, method=method, radial_range=radial_range, **kw)


class ApiTests(unittest.TestCase):
    def test_mixed_paths_arrays_and_coordinates(self):
        from fabio.tifimage import TifImage
        a = np.arange(35, dtype=np.float64).reshape(5, 7) + 2
        b = np.arange(12, dtype=np.float64).reshape(3, 4) + 10
        multi = dioptrin.MultiIntegrator([integrator(), integrator()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.tif"
            # TIFF input follows Dioptrin's established vertical-flip convention.
            TifImage(data=np.ascontiguousarray(a[::-1])).write(str(path))
            mixed = multi.integrate2d([str(path), b], 9, 13, errors=True)
            arrays = multi.integrate2d([a, b], 9, 13, errors=True)
        np.testing.assert_allclose(mixed.intensity, arrays.intensity)
        np.testing.assert_allclose(mixed.errors, arrays.errors)
        self.assertEqual(np.asarray(mixed.intensity).reshape(13, 9).shape, (13, 9))
        self.assertEqual(len(mixed.radial), 9)
        self.assertEqual(len(mixed.azimuthal), 13)
        self.assertEqual((mixed.n_radial, mixed.n_azimuthal), (9, 13))
        self.assertIsNone(multi.integrate2d([a, b], 9, 13).errors)

    def test_invalid_inputs(self):
        data = np.ones((3, 4))
        multi = dioptrin.MultiIntegrator([integrator()])
        cases = [([], 5, 5, {}), ([data], 0, 5, {}),
                 ([data], 5, 5, {"normalization": "unknown"}),
                 ([data], 5, 5, {"normalization_factors": []}),
                 ([data], 5, 5, {"normalization_factors": [0.0]}),
                 ([data], 5, 5, {"azimuthal_range": (1.0, -1.0)}),
                 ([data], 5, 5, {"azimuthal_range": (0.0, 7.0)}),
                 ([np.empty((0, 3))], 5, 5, {}),
                 ([data[:, ::2]], 5, 5, {}),
                 ([np.full((3, 4), np.nan)], 5, 5, {})]
        for images, nr, na, options in cases:
            with self.subTest(options=options, shape=[x.shape for x in images]):
                with self.assertRaises(ValueError):
                    multi.integrate2d(images, nr, na, **options)
        with self.assertRaises(ValueError):
            dioptrin.MultiIntegrator([integrator(radial_range=(0, 1)),
                                      integrator(radial_range=(0, 2))])

    def test_monitor_and_absolute_scale(self):
        d = np.full((8, 11), 64.0)
        multi = dioptrin.MultiIntegrator([integrator()])
        relative = multi.integrate2d([d], 4, 8, errors=True)
        absolute = multi.integrate2d([d], 4, 8, errors=True,
                                     normalization="absolute", normalization_factors=[3.0])
        sac = PONI["pixel1"]*PONI["pixel2"]/PONI["distance"]**2
        np.testing.assert_allclose(absolute.intensity, np.asarray(relative.intensity)/(3*sac))
        np.testing.assert_allclose(absolute.errors, np.asarray(relative.errors)/(3*sac))


class PyFaiTests(unittest.TestCase):
    def test_matched_counting_and_full_split(self):
        import pyFAI
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        from pyFAI.multi_geometry import MultiGeometry
        geometries = [dict(PONI, poni1=-0.003, poni2=-0.002), dict(PONI, poni1=-0.003, poni2=-0.002, pixel1=1.7e-4, pixel2=2.1e-4,
                                distance=0.14, rot1=0.007, rot2=-0.009, rot3=0.05)]
        shapes = [(31, 37), (23, 29)]
        data = [np.ascontiguousarray(20 + i*17 + np.arange(np.prod(s)).reshape(s) % 31,
                                     dtype=np.float64) for i, s in enumerate(shapes)]
        masks = [np.ascontiguousarray(np.indices(s).sum(axis=0) % 11 == 0, dtype=np.uint8)
                 for s in shapes]
        receipts = []
        cases = [("off_axis", unit, hi, geometries) for unit, hi in
                 [("2th_rad", 0.2), ("2th_deg", np.degrees(0.2)),
                  ("q_A^-1", 1.5), ("q_nm^-1", 15.0)]]
        central = [PONI, dict(geometries[1], poni1=PONI["poni1"], poni2=PONI["poni2"])]
        cases += [("beam_and_seam", "2th_rad", 0.08, central),
                  ("radial_crop", "2th_rad", 0.08, geometries)]
        for scenario, unit, hi, geometries in cases:
            for split in ["no", "full"]:
                ais = [AzimuthalIntegrator(dist=p["distance"], pixel1=p["pixel1"], pixel2=p["pixel2"],
                                           poni1=p["poni1"], poni2=p["poni2"], rot1=p["rot1"],
                                           rot2=p["rot2"], rot3=p["rot3"], wavelength=p["wavelength"])
                       for p in geometries]
                mg = MultiGeometry(ais, unit=unit, radial_range=(0, hi), azimuth_range=(-180, 180),
                                   threadpoolsize=0)
                ref = mg.integrate2d(data, 12, 17, method=(split, "histogram", "cython"),
                                     correctSolidAngle=True, lst_mask=masks, error_model="poisson",
                                     polarization_factor=0.6, normalization_factor=[2.0, 3.0])
                # pyFAI expands its radial upper edge by float32 epsilon.
                # Recover the actual grid from returned centers for both libraries.
                dr = ref.radial[1]-ref.radial[0]
                rr = (max(0.0, float(ref.radial[0]-dr/2)), float(ref.radial[-1]+dr/2))
                ds = [integrator(p, method="counting" if split=="no" else "pixel_split",
                                 radial_range=rr, unit=unit, polarization_factor=0.6)
                      for p in geometries]
                for ai, mask in zip(ds, masks):
                    ai.set_mask(mask.astype(bool))
                out = dioptrin.MultiIntegrator(ds).integrate2d(
                    data, 12, 17, errors=True, normalization="absolute", normalization_factors=[2.0,3.0])
                actual = np.array(out.intensity).reshape(17,12)
                error = np.array(out.errors).reshape(17,12)
                valid = np.asarray(ref.count)>1e-6
                # pyFAI uses float32 geometry/corrections/weights internally.
                relative = np.abs(actual[valid]-ref.intensity[valid])/np.maximum(1,abs(ref.intensity[valid]))
                rel_sigma = np.abs(error[valid]-ref.sigma[valid])/np.maximum(1,abs(ref.sigma[valid]))
                receipt = dict(scenario=scenario, unit=unit, split=split, bins=int(valid.sum()),
                               dioptrin_only_nonzero_bins=int(np.count_nonzero((actual != 0) & ~valid)),
                               pyfai_only_nonzero_bins=int(np.count_nonzero((actual == 0) & valid)),
                               max_relative_intensity=float(relative.max()),
                               max_relative_std=float(rel_sigma.max()))
                receipts.append(receipt)
                print(json.dumps(receipt), flush=True)
                np.testing.assert_allclose(out.radial, ref.radial, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(np.degrees(out.azimuthal), ref.azimuthal, rtol=2e-6, atol=3e-5)
                if split == "no":
                    np.testing.assert_allclose(actual, ref.intensity, rtol=2e-6, atol=1e-3)
                    np.testing.assert_allclose(error, ref.sigma, rtol=2e-6, atol=1e-3)
                else:
                    # pyFAI's full histogram engine distributes integrated line
                    # areas through rows; Dioptrin clips polygons exactly in f64.
                    # These are deliberately measured, bounded method differences,
                    # not a claim of float32-level full-split equivalence.
                    self.assertEqual(receipt["dioptrin_only_nonzero_bins"],
                                     1 if scenario == "beam_and_seam" else 0)
                    self.assertEqual(receipt["pyfai_only_nonzero_bins"], 0)
                    np.testing.assert_allclose(actual[valid], ref.intensity[valid],
                                               rtol=0.03 if scenario=="beam_and_seam" else 0.01, atol=1e-3)
                    np.testing.assert_allclose(error[valid], ref.sigma[valid],
                                               rtol=0.10 if scenario=="beam_and_seam" else 0.04, atol=1e-3)
        report = dict(pyfai=pyFAI.version, comparisons=receipts)
        print(json.dumps(report, indent=2))
        if REPORT_PATH is not None:
            REPORT_PATH.write_text(json.dumps(report, indent=2)+"\n")


REPORT_PATH = None
if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write pyFAI comparison metrics as JSON")
    args, remaining = parser.parse_known_args()
    REPORT_PATH = args.report
    unittest.main(argv=[sys.argv[0], *remaining])
