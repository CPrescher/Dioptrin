"""Characterize cold preparation and cached Python multi-detector 2D latency.

RAYON_NUM_THREADS=8 python python_examples/bench_multi2d.py --report result.json
Includes NumPy input copies, raw merging and Python result-object construction; no output list getters or I/O.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import dioptrin
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--n-ss", type=int, default=3)
    parser.add_argument("--pixel-split-mode", choices=["fractional", "pyfai_histogram"], default="fractional")
    args = parser.parse_args()
    if args.n_ss < 1:
        parser.error("--n-ss must be positive")
    geometries = [
        dict(pixel1=1e-4, pixel2=1.2e-4, distance=0.1, poni1=0.009, poni2=0.011,
             rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1e-10),
        dict(pixel1=1.7e-4, pixel2=2.1e-4, distance=0.14, poni1=0.013, poni2=0.01,
             rot1=0.007, rot2=-0.009, rot3=0.05, wavelength=1e-10),
    ]
    shapes = [(256, 320), (192, 240)]
    images = [np.full(shape, 100.0) for shape in shapes]
    results = []
    for method in ["counting", "pixel_split", "supersampled"]:
        integrators = [dioptrin.Integrator.from_poni_dict(
            p, method=method, n_ss=args.n_ss, unit="2th_rad", radial_range=(0, 0.3),
            polarization_factor=0.6) for p in geometries]
        multi = dioptrin.MultiIntegrator(integrators, pixel_split_mode=args.pixel_split_mode)
        def run():
            return multi.integrate2d(images, 256, 180, errors=True,
                                     normalization="absolute", normalization_factors=[2, 3])
        start = time.perf_counter()
        first = run()
        cold_ms = (time.perf_counter()-start)*1000
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            current = run()
            samples.append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(current.intensity, first.intensity)
        np.testing.assert_array_equal(current.errors, first.errors)
        results.append(dict(method=method, cold_ms=cold_ms,
                            cached_median_ms=statistics.median(samples), repetitions=len(samples)))
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  rayon_threads=dioptrin.rayon_num_threads(), shapes=shapes,
                  n_radial=256, n_azimuthal=180, n_ss=args.n_ss, pixel_split_mode=args.pixel_split_mode, errors=True, results=results)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
