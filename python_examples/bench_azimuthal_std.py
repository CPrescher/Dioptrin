"""Warm-path timings for optional spottiness; run after maturin develop --release."""
import argparse
import statistics
import time

import dioptrin
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    dioptrin.set_rayon_threads(args.threads)
    data = np.random.default_rng(42).uniform(10, 200, (args.size, args.size))
    poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.2,
                poni1=args.size*5e-5, poni2=args.size*5e-5, wavelength=1e-10)
    print(f"{args.size}x{args.size}, {args.bins} bins, {args.threads} threads; median milliseconds")
    for method in ["counting", "pixel_split", "supersampled"]:
        ai = dioptrin.Integrator.from_poni_dict(poni, method=method, n_ss=3)
        samples = {name: [] for name in ["mean", "mean+errors", "mean+spread", "all"]}
        options = [dict(), dict(errors=True), dict(azimuthal_std=True),
                   dict(errors=True, azimuthal_std=True)]
        for opts in options:
            ai.integrate1d(data, args.bins, **opts)
        for _ in range(args.repeats):
            for (name, durations), opts in zip(samples.items(), options):
                start = time.perf_counter()
                ai.integrate1d(data, args.bins, **opts)
                durations.append((time.perf_counter() - start) * 1000)
        print(method, ", ".join(f"{name}: {statistics.median(values):.3f}" for name, values in samples.items()))


if __name__ == "__main__":
    main()
