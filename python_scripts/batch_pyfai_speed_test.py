import argparse
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import fabio
import pyFAI

# Global integrator for process pool workers
_worker_ai = None
_worker_method = None


def _init_worker(poni_path, method):
    """Initialize worker process with its own integrator."""
    global _worker_ai, _worker_method
    _worker_ai = pyFAI.load(poni_path)
    _worker_method = method


def integrate1d_process_task(args):
    """Worker function for 1D integration (process pool)."""
    tiff_path, n_points = args
    data = fabio.open(tiff_path).data
    result = _worker_ai.integrate1d(data, n_points, method=_worker_method)
    # Return only numpy arrays (picklable), not the full result object
    return (result.radial, result.intensity)


def integrate2d_process_task(args):
    """Worker function for 2D integration (process pool)."""
    tiff_path, n_radial, n_azimuthal = args
    data = fabio.open(tiff_path).data
    result = _worker_ai.integrate2d(data, n_radial, n_azimuthal, method=_worker_method)
    # Return only numpy arrays (picklable), not the full result object
    return (result.radial, result.azimuthal, result.intensity)


def main() -> None:
    parser = argparse.ArgumentParser(description="pyFAI Batch API speed test")
    parser.add_argument("--poni", default="test_data/v1.poni")
    parser.add_argument("--tiff", default="test_data/CeO2_Pilatus1M.tif")
    parser.add_argument("tiff_path", nargs="?", default=None)
    parser.add_argument("--files", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-radial", type=int, default=200)
    parser.add_argument("--n-azimuthal", type=int, default=180)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--executor",
        choices=["serial", "thread", "process"],
        default="serial",
        help="Executor type for parallel processing",
    )
    parser.add_argument(
        "--method",
        default="csr",
        help="Integration method (e.g., csr, lut, bbox)",
    )
    args = parser.parse_args()

    tiff_path = args.tiff_path or args.tiff
    files = [tiff_path for _ in range(args.files)]

    ai = pyFAI.load(args.poni)
    print(f"Loaded PONI: {args.poni}")
    print(f"Integration method: {args.method}")
    print(f"Files: {args.files}")
    print(f"Executor: {args.executor}")
    if args.num_workers:
        print(f"Workers: {args.num_workers}")
    print()

    # Warmup - build LUT
    data = fabio.open(tiff_path).data
    ai.integrate1d(data, args.n_points, method=args.method)

    # 1D batch integration
    start = time.perf_counter()
    if args.executor == "serial":
        results_1d = []
        for f in files:
            d = fabio.open(f).data
            results_1d.append(ai.integrate1d(d, args.n_points, method=args.method))
    elif args.executor == "thread":
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            def task_1d(f):
                d = fabio.open(f).data
                return ai.integrate1d(d, args.n_points, method=args.method)

            results_1d = list(executor.map(task_1d, files))
    else:  # process
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.poni, args.method),
        ) as executor:
            tasks = [(f, args.n_points) for f in files]
            results_1d = list(executor.map(integrate1d_process_task, tasks))
    elapsed = time.perf_counter() - start

    print(f"1D batch ok: {len(results_1d)}/{len(results_1d)}")
    print(f"1D batch time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"1D throughput: {len(results_1d) / elapsed:.2f} files/s")
    print()

    # 2D batch integration
    start = time.perf_counter()
    if args.executor == "serial":
        results_2d = []
        for f in files:
            d = fabio.open(f).data
            results_2d.append(
                ai.integrate2d(d, args.n_radial, args.n_azimuthal, method=args.method)
            )
    elif args.executor == "thread":
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            def task_2d(f):
                d = fabio.open(f).data
                return ai.integrate2d(
                    d, args.n_radial, args.n_azimuthal, method=args.method
                )

            results_2d = list(executor.map(task_2d, files))
    else:  # process
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.poni, args.method),
        ) as executor:
            tasks = [(f, args.n_radial, args.n_azimuthal) for f in files]
            results_2d = list(executor.map(integrate2d_process_task, tasks))
    elapsed = time.perf_counter() - start

    print(f"2D batch ok: {len(results_2d)}/{len(results_2d)}")
    print(f"2D batch time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"2D throughput: {len(results_2d) / elapsed:.2f} files/s")


if __name__ == "__main__":
    main()
