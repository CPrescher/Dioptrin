"""
Benchmark comparison: Dioptrin vs pyFAI

Compares 1D and 2D integration throughput and peak memory usage across
different parallelism strategies. Accepts a TIFF file and PONI file as
input so you can test with different detector dimensions.

Usage:
    python benchmark_comparison.py
    python benchmark_comparison.py --tiff my_image.tif --poni my_calib.poni
    python benchmark_comparison.py --files 100 --num-workers 8
"""

import argparse
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil

# pyFAI process pool globals
_worker_ai = None


def _init_pyfai_worker(poni_path, tiff_path, method):
    global _worker_ai
    import fabio
    import pyFAI

    _worker_ai = pyFAI.load(poni_path)
    # Warmup LUT with real data so solid angle arrays match
    data = fabio.open(tiff_path).data
    _worker_ai.integrate1d(data, 10, method=method)


def _pyfai_1d_task(args):
    import fabio

    tiff_path, n_points, method = args
    data = fabio.open(tiff_path).data
    result = _worker_ai.integrate1d(data, n_points, method=method)
    return result.radial, result.intensity


def _pyfai_2d_task(args):
    import fabio

    tiff_path, n_radial, n_azimuthal, method = args
    data = fabio.open(tiff_path).data
    result = _worker_ai.integrate2d(data, n_radial, n_azimuthal, method=method)
    return result.intensity


class PeakMemoryTracker:
    """Polls RSS of current process + children in a background thread."""

    def __init__(self, interval=0.05):
        self._interval = interval
        self._proc = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def _poll(self):
        while not self._stop.is_set():
            mem = self._sample()
            if mem > self._peak:
                self._peak = mem
            self._stop.wait(self._interval)

    def start(self):
        self._peak = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self._peak / (1024 * 1024)


def benchmark(label, func, baseline_mb):
    """Run a benchmark function and return elapsed time and peak memory above baseline."""
    # Track memory across warmup + measured run so we capture all allocations
    tracker = PeakMemoryTracker()
    tracker.start()

    # Warmup (memory tracked but time not)
    func()

    start = time.perf_counter()
    count = func()
    elapsed = time.perf_counter() - start
    peak_mb = tracker.stop()

    throughput = count / elapsed if elapsed > 0 else float("inf")
    mem_used = peak_mb - baseline_mb
    print(
        f"  {label:40s}  {elapsed:6.2f}s  {throughput:8.1f} files/s  {mem_used:7.1f} MB"
    )
    return elapsed, throughput, mem_used


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison: Dioptrin vs pyFAI"
    )
    parser.add_argument("--tiff", default="test_data/CeO2_Pilatus1M.tif")
    parser.add_argument("--poni", default="test_data/v1.poni")
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-radial", type=int, default=200)
    parser.add_argument("--n-azimuthal", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 4
    num_workers = args.num_workers or max(1, cpu_count - 2)
    files = [args.tiff for _ in range(args.files)]

    # Detect image dimensions
    import fabio

    img = fabio.open(args.tiff)
    shape = img.data.shape
    print(f"Image:       {args.tiff} ({shape[0]} x {shape[1]})")
    print(f"PONI:        {args.poni}")
    print(f"Files:       {args.files}")
    print(f"CPU cores:   {cpu_count}")
    print(f"Workers:     {num_workers}")
    print(f"1D bins:     {args.n_points}")
    print(f"2D bins:     {args.n_radial} x {args.n_azimuthal}")
    print()

    # Baseline memory before any benchmarks (just Python + numpy + fabio loaded)
    baseline_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Baseline RSS: {baseline_mb:.1f} MB")
    print()

    # ── pyFAI benchmarks ──────────────────────────────────────────────
    import pyFAI

    ai = pyFAI.load(args.poni)
    data = img.data.astype(np.float64)
    pyfai_method = "csr"

    # Warmup pyFAI LUT
    ai.integrate1d(data, args.n_points, method=pyfai_method)
    ai.integrate2d(data, args.n_radial, args.n_azimuthal, method=pyfai_method)

    header = f"  {'Method':40s}  {'Time':>6s}  {'Throughput':>12s}  {'Peak RSS':>8s}"

    print("=" * 80)
    print("1D Integration")
    print("=" * 80)
    print(header)
    print("-" * 80)

    def pyfai_serial_1d():
        for f in files:
            d = fabio.open(f).data
            ai.integrate1d(d, args.n_points, method=pyfai_method)
        return len(files)

    benchmark("pyFAI serial", pyfai_serial_1d, baseline_mb)

    def pyfai_process_1d():
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_pyfai_worker,
            initargs=(args.poni, args.tiff, pyfai_method),
        ) as executor:
            tasks = [(f, args.n_points, pyfai_method) for f in files]
            list(executor.map(_pyfai_1d_task, tasks))
        return len(files)

    benchmark(f"pyFAI process pool ({num_workers} workers)", pyfai_process_1d, baseline_mb)

    # Pre-warmed pool: create once, reuse for warmup + measured run
    pyfai_pool_1d = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_pyfai_worker,
        initargs=(args.poni, args.tiff, pyfai_method),
    )
    # Force all workers to start and initialize
    list(pyfai_pool_1d.map(_pyfai_1d_task, [(args.tiff, args.n_points, pyfai_method)]))

    def pyfai_prewarmed_1d():
        tasks = [(f, args.n_points, pyfai_method) for f in files]
        list(pyfai_pool_1d.map(_pyfai_1d_task, tasks))
        return len(files)

    benchmark(
        f"pyFAI pre-warmed pool ({num_workers} workers)", pyfai_prewarmed_1d, baseline_mb
    )
    pyfai_pool_1d.shutdown()

    # ── Dioptrin benchmarks ───────────────────────────────────────────
    import dioptrin

    integrator = dioptrin.Integrator(args.poni, method="pixel_split")
    # Warmup
    integrator.integrate1d(files[0], args.n_points)

    def dioptrin_serial_1d():
        for f in files:
            integrator.integrate1d(f, args.n_points)
        return len(files)

    benchmark("Dioptrin serial", dioptrin_serial_1d, baseline_mb)

    def dioptrin_batch_1d():
        integrator.batch1d(files, args.n_points, num_workers=num_workers)
        return len(files)

    benchmark(f"Dioptrin batch ({num_workers} workers)", dioptrin_batch_1d, baseline_mb)

    print()
    print("=" * 80)
    print("2D Integration")
    print("=" * 80)
    print(header)
    print("-" * 80)

    def pyfai_serial_2d():
        for f in files:
            d = fabio.open(f).data
            ai.integrate2d(d, args.n_radial, args.n_azimuthal, method=pyfai_method)
        return len(files)

    benchmark("pyFAI serial", pyfai_serial_2d, baseline_mb)

    def pyfai_process_2d():
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_pyfai_worker,
            initargs=(args.poni, args.tiff, pyfai_method),
        ) as executor:
            tasks = [
                (f, args.n_radial, args.n_azimuthal, pyfai_method) for f in files
            ]
            list(executor.map(_pyfai_2d_task, tasks))
        return len(files)

    benchmark(f"pyFAI process pool ({num_workers} workers)", pyfai_process_2d, baseline_mb)

    # Pre-warmed pool for 2D
    pyfai_pool_2d = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_pyfai_worker,
        initargs=(args.poni, args.tiff, pyfai_method),
    )
    list(
        pyfai_pool_2d.map(
            _pyfai_2d_task, [(args.tiff, args.n_radial, args.n_azimuthal, pyfai_method)]
        )
    )

    def pyfai_prewarmed_2d():
        tasks = [
            (f, args.n_radial, args.n_azimuthal, pyfai_method) for f in files
        ]
        list(pyfai_pool_2d.map(_pyfai_2d_task, tasks))
        return len(files)

    benchmark(
        f"pyFAI pre-warmed pool ({num_workers} workers)", pyfai_prewarmed_2d, baseline_mb
    )
    pyfai_pool_2d.shutdown()

    def dioptrin_serial_2d():
        for f in files:
            integrator.integrate2d(f, args.n_radial, args.n_azimuthal)
        return len(files)

    benchmark("Dioptrin serial", dioptrin_serial_2d, baseline_mb)

    def dioptrin_batch_2d():
        integrator.batch2d(
            files, args.n_radial, args.n_azimuthal, num_workers=num_workers
        )
        return len(files)

    benchmark(f"Dioptrin batch ({num_workers} workers)", dioptrin_batch_2d, baseline_mb)


if __name__ == "__main__":
    main()
