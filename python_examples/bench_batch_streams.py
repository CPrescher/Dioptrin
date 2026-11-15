"""Throughput of the Python batch and stream methods, in frames per second.

Times ``batch1d``, ``batch2d`` (TIFF paths), ``batch1d_iter`` and
``batch2d_iter`` for each combination of frame size, worker count and input
kind (a list or a generator, of NumPy arrays or of TIFF paths), and prints the
median rate over the repeats with its range. Not a test: ``test_examples.py``
skips ``bench_*`` scripts, and CI does not run it.

    maturin develop --release --features python
    python python_examples/bench_batch_streams.py
    python python_examples/bench_batch_streams.py --sizes 256 1024 --workers 2 8 \\
        --apis batch2d batch2d_iter --inputs list-paths gen-paths --repeat 7

Synthetic frames come from a fixed seed. A pool of ``--pool`` distinct frames
is cycled to make ``--frames`` items, and path inputs are uncompressed float32
TIFFs written once to a temporary directory, so they are read from a warm file
cache: the rates are for decoding and integrating, not for disk reads.
Generators yield the pooled arrays or paths without copying, so they measure
the stream machinery rather than a producer's cost. One untimed call per case
builds the lookup table first, so every repeat is warm.

Run it on an otherwise idle machine; the spread column shows how much the
repeats disagree. ``RAYON_NUM_THREADS`` and ``--threads`` (threads per worker)
change the per-worker thread count.
"""
import argparse
import importlib.metadata
import os
import platform
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import dioptrin

ROOT = Path(__file__).resolve().parent.parent
PONI = ROOT / "test_data" / "v1.poni"
APIS = ("batch1d", "batch2d", "batch1d_iter", "batch2d_iter")
INPUTS = ("list-arrays", "gen-arrays", "list-paths", "gen-paths")


def write_float32_tiff(path, image):
    """Write a single-strip, uncompressed, little-endian float32 TIFF."""
    image = np.ascontiguousarray(image, dtype="<f4")
    rows, cols = image.shape
    data = image.tobytes()
    entries = [
        (256, 4, 1, cols),  # ImageWidth
        (257, 4, 1, rows),  # ImageLength
        (258, 3, 1, 32),  # BitsPerSample
        (259, 3, 1, 1),  # Compression: none
        (262, 3, 1, 1),  # Photometric: black is zero
        (273, 4, 1, 0),  # StripOffsets, patched below
        (277, 3, 1, 1),  # SamplesPerPixel
        (278, 4, 1, rows),  # RowsPerStrip
        (279, 4, 1, len(data)),  # StripByteCounts
        (339, 3, 1, 3),  # SampleFormat: IEEE float
    ]
    ifd_offset = 8
    data_offset = ifd_offset + 2 + 12 * len(entries) + 4
    with open(path, "wb") as handle:
        handle.write(b"II" + struct.pack("<HI", 42, ifd_offset))
        handle.write(struct.pack("<H", len(entries)))
        for tag, kind, count, value in entries:
            if tag == 273:
                value = data_offset
            if kind == 3:
                handle.write(struct.pack("<HHIHH", tag, kind, count, value, 0))
            else:
                handle.write(struct.pack("<HHII", tag, kind, count, value))
        handle.write(struct.pack("<I", 0))
        handle.write(data)


def make_pool(size, pool, directory, seed):
    rng = np.random.default_rng(seed)
    arrays = [rng.poisson(100.0, size=(size, size)).astype(np.float64) for _ in range(pool)]
    paths = []
    for index, array in enumerate(arrays):
        path = directory / f"frame_{size}_{index:03d}.tif"
        write_float32_tiff(path, array)
        paths.append(str(path))
    return arrays, paths


def items(kind, arrays, paths, n_frames):
    source = arrays if kind.endswith("arrays") else paths
    if kind.startswith("list"):
        return [source[i % len(source)] for i in range(n_frames)]
    return (source[i % len(source)] for i in range(n_frames))


def supported(api, kind):
    if api == "batch2d":
        return kind == "list-paths"  # arrays go through batch2d_iter
    if api == "batch1d":
        return kind.startswith("list")
    return True


def run_once(integrator, api, images, args):
    common = dict(num_workers=args.workers_now, num_threads=args.threads)
    if api == "batch1d":
        results = integrator.batch1d(images, args.n_points, **common)
    elif api == "batch1d_iter":
        results = list(integrator.batch1d_iter(images, args.n_points, **common))
    elif api == "batch2d":
        results = integrator.batch2d(images, args.n_radial, args.n_azimuthal, **common)
    else:
        results = list(integrator.batch2d_iter(images, args.n_radial, args.n_azimuthal, **common))
    failed = [r for r in results if not r.is_ok()]
    if failed:
        raise RuntimeError(f"{api}: {len(failed)} of {len(results)} frames failed: {failed[0].error}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024],
                        help="square frame edge lengths in pixels (default: 256 1024)")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--threads", type=int, default=None, help="threads per worker (default: the library's)")
    parser.add_argument("--apis", nargs="+", choices=APIS, default=list(APIS))
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=list(INPUTS))
    parser.add_argument("--method", default="pixel_split")
    parser.add_argument("--frames", type=int, default=None,
                        help="frames per call (default: 400 for frames up to 512 px, else 64)")
    parser.add_argument("--pool", type=int, default=8, help="distinct frames cycled through (default: 8)")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-radial", type=int, default=500)
    parser.add_argument("--n-azimuthal", type=int, default=360)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.repeat < 1 or args.pool < 1 or min(args.sizes) < 2 or min(args.workers) < 1:
        parser.error("--repeat, --pool and --workers must be positive and sizes at least 2")

    try:
        version = importlib.metadata.version("dioptrin")
    except importlib.metadata.PackageNotFoundError:
        version = "?"
    print(f"dioptrin {version} ({Path(dioptrin.__file__).parent}), Python {platform.python_version()}, "
          f"{platform.machine()}, {os.cpu_count()} CPUs, load {os.getloadavg()[0]:.2f}, "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset')}, method {args.method}")
    header = f"{'api':<13} {'input':<12} {'size':>6} {'workers':>7} {'frames':>6} {'median fps':>11} {'min':>9} {'max':>9} {'spread':>7}"
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory(prefix="dioptrin-bench-") as tmp:
        for size in args.sizes:
            arrays, paths = make_pool(size, args.pool, Path(tmp), args.seed)
            n_frames = args.frames or (400 if size <= 512 else 64)
            for api in args.apis:
                for kind in args.inputs:
                    if not supported(api, kind):
                        continue
                    for workers in args.workers:
                        args.workers_now = workers
                        # A fresh integrator per case: its table is built by the
                        # untimed warm-up call, not by the first timed repeat.
                        integrator = dioptrin.Integrator(str(PONI), method=args.method)
                        run_once(integrator, api, items(kind, arrays, paths, min(n_frames, 2 * workers)), args)
                        rates = []
                        for _ in range(args.repeat):
                            images = items(kind, arrays, paths, n_frames)
                            start = time.perf_counter()
                            done = run_once(integrator, api, images, args)
                            rates.append(done / (time.perf_counter() - start))
                        median = statistics.median(rates)
                        spread = (max(rates) - min(rates)) / median
                        print(f"{api:<13} {kind:<12} {size:>6} {workers:>7} {n_frames:>6} "
                              f"{median:>11.1f} {min(rates):>9.1f} {max(rates):>9.1f} {spread:>6.1%}")
                        sys.stdout.flush()
                        if hasattr(integrator, "clear_file_workers"):
                            integrator.clear_file_workers()
    print(f"load at end {os.getloadavg()[0]:.2f}")


if __name__ == "__main__":
    main()
