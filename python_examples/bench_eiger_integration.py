"""Benchmark disk-to-integration streaming throughput.

Measures end-to-end performance of reading frames from disk and
integrating them via batch1d_iter with generator-based streaming.
This isolates the realistic pipeline: disk I/O + integration.

Requires optional dependencies:
    pip install dioptrin[benchmark]

Usage:
    python python_examples/bench_eiger_integration.py
    python python_examples/bench_eiger_integration.py --frames 100 --points 2000
    python python_examples/bench_eiger_integration.py --workers 14 --threads 1
"""

import argparse
import gc
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import dioptrin

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "test_data")
HDF5_PATH = os.path.join(DATA_DIR, "eiger_0000.h5")
PONI_PATH = os.path.join(DATA_DIR, "eiger_0000.poni")
MASK_PATH = os.path.join(DATA_DIR, "eiger_0000.mask")
TIFF_PATH = os.path.join(DATA_DIR, "eiger_0000.tiff")
HDF5_DATASET = "entry_0000/instrument/eiger/data"


# ---------------------------------------------------------------------------
# Frame generators (streaming from disk)
# ---------------------------------------------------------------------------

def _decompress_bitshuffle_lz4(raw_bytes, shape, dtype):
    """Decompress a single bitshuffle-LZ4 compressed chunk."""
    import bitshuffle

    block_size = struct.unpack(">I", raw_bytes[8:12])[0]
    buf = np.frombuffer(raw_bytes, dtype=np.uint8, offset=12)
    return bitshuffle.decompress_lz4(buf, shape, dtype, block_size).astype(np.float64)


def gen_frames_h5py(path: str, dataset: str, n_frames: int):
    """Generator that yields frames one at a time from h5py (serial)."""
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[dataset]
        n = min(n_frames, ds.shape[0])
        for i in range(n):
            yield ds[i].astype(np.float64)


def gen_frames_fabio(path: str, n_frames: int):
    """Generator that yields frames one at a time from fabio (serial)."""
    import fabio

    img = fabio.open(path)
    yield img.data.astype(np.float64)
    for _ in range(min(n_frames, img.nframes) - 1):
        img = img.next()
        yield img.data.astype(np.float64)


def gen_frames_parallel(
    path: str, dataset: str, n_frames: int, decomp_workers: int = 8
):
    """Generator that reads raw chunks and decompresses in parallel threads.

    Uses a sliding window of at most `decomp_workers` in-flight futures to
    avoid reading all raw chunks into memory and to limit CPU contention
    with downstream integration threads.
    """
    import h5py
    from collections import deque

    with h5py.File(path, "r") as f:
        ds = f[dataset]
        dtype = ds.dtype
        shape = (ds.chunks[1], ds.chunks[2])
        n = min(n_frames, ds.shape[0])

        with ThreadPoolExecutor(max_workers=decomp_workers) as pool:
            pending = deque()
            idx = 0

            # Fill initial window
            while idx < n and len(pending) < decomp_workers:
                raw = bytes(ds.id.read_direct_chunk((idx, 0, 0))[1])
                pending.append(pool.submit(_decompress_bitshuffle_lz4, raw, shape, dtype))
                idx += 1

            # Yield results and refill window one-for-one
            while pending:
                yield pending.popleft().result()
                if idx < n:
                    raw = bytes(ds.id.read_direct_chunk((idx, 0, 0))[1])
                    pending.append(pool.submit(_decompress_bitshuffle_lz4, raw, shape, dtype))
                    idx += 1


def gen_frames_tiff(path: str, n_frames: int):
    """Generator that re-reads a single TIFF file to simulate per-frame disk reads."""
    import fabio

    for _ in range(n_frames):
        yield fabio.open(path).data.astype(np.float64)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench_stream(
    integrator: dioptrin.Integrator,
    name: str,
    gen_fn,
    n_points: int,
    num_workers: int | None,
    num_threads: int | None,
) -> None:
    """Benchmark disk-to-integration streaming pipeline."""
    gc.collect()
    t0 = time.perf_counter()
    count = 0
    for _ in integrator.batch1d_iter(
        gen_fn(), n_points, num_workers=num_workers, num_threads=num_threads,
    ):
        count += 1
    elapsed = time.perf_counter() - t0
    print(f"  {name:30s}: {elapsed:7.2f}s  ({count:4d} frames, {count / elapsed:6.1f} fps)")
    gc.collect()


def bench_stream_to_array(
    integrator: dioptrin.Integrator,
    name: str,
    gen_fn,
    n_points: int,
    num_workers: int | None,
    num_threads: int | None,
) -> None:
    """Benchmark streaming pipeline with collection into a 2D NumPy array."""
    gc.collect()
    t0 = time.perf_counter()
    rows = []
    for batch_result in integrator.batch1d_iter(
        gen_fn(), n_points, num_workers=num_workers, num_threads=num_threads,
    ):
        if batch_result.is_ok():
            rows.append(batch_result.result.intensity)
    patterns = np.array(rows)  # shape: (n_frames, n_points)
    elapsed = time.perf_counter() - t0
    count = patterns.shape[0]
    print(
        f"  {name:30s}: {elapsed:7.2f}s  ({count:4d} frames, {count / elapsed:6.1f} fps)"
        f"  -> {patterns.shape} array"
    )
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark disk-to-integration streaming")
    parser.add_argument("--hdf5", default=HDF5_PATH, help="Path to HDF5 file")
    parser.add_argument("--dataset", default=HDF5_DATASET, help="HDF5 dataset path")
    parser.add_argument("--poni", default=PONI_PATH, help="Path to PONI calibration file")
    parser.add_argument("--mask", default=MASK_PATH, help="Path to mask file")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to integrate")
    parser.add_argument("--points", type=int, default=1000, help="Number of radial bins (1D)")
    parser.add_argument("--workers", type=int, default=None, help="Batch worker count")
    parser.add_argument("--threads", type=int, default=None, help="Rayon threads per worker")
    parser.add_argument("--decomp-workers", type=int, default=8, help="Decompression thread count")
    parser.add_argument(
        "--method",
        default="pixel_split",
        choices=["counting", "pixel_split", "supersampled"],
    )
    parser.add_argument("--no-mask", action="store_true", help="Skip loading mask")
    parser.add_argument("--flip", action="store_true", help="Flip frames vertically (frame[::-1])")
    args = parser.parse_args()

    for path, label in [(args.hdf5, "HDF5"), (args.poni, "PONI")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            sys.exit(1)

    print(f"HDF5 file      : {args.hdf5}")
    print(f"Method         : {args.method}")
    print(f"PONI           : {args.poni}")

    integrator = dioptrin.Integrator(args.poni, method=args.method)

    if not args.no_mask and os.path.exists(args.mask):
        integrator.set_mask(args.mask)
        print(f"Mask           : {args.mask}")
    else:
        print("Mask           : none")

    if args.workers:
        print(f"Workers        : {args.workers}")
    if args.threads:
        print(f"Threads        : {args.threads} (per worker)")
    else:
        print(f"Threads        : {dioptrin.rayon_num_threads()} (default)")
    print(f"Decomp workers : {args.decomp_workers}")

    # Warmup (builds LUT) using a single frame
    import h5py

    print("\nWarming up (building LUT)...")
    with h5py.File(args.hdf5, "r") as f:
        warmup_frame = f[args.dataset][0].astype(np.float64)
    if args.flip:
        warmup_frame = np.ascontiguousarray(warmup_frame[::-1])
    t0 = time.perf_counter()
    integrator.integrate1d(warmup_frame, args.points)
    warmup = time.perf_counter() - t0
    print(f"LUT build      : {warmup:.2f}s")
    del warmup_frame
    gc.collect()

    dw = args.decomp_workers
    flip = args.flip

    def maybe_flip(gen_fn):
        """Wrap a generator factory to flip frames vertically if --flip is set."""
        if not flip:
            return gen_fn
        def wrapped():
            for frame in gen_fn():
                yield np.ascontiguousarray(frame[::-1])
        return wrapped

    flip_label = " [flipped]" if flip else ""
    print(f"\n=== Disk streaming integration ({args.points} bins, {args.frames} frames){flip_label} ===")
    # bench_stream(
    #     integrator, "h5py",
    #     lambda: gen_frames_h5py(args.hdf5, args.dataset, args.frames),
    #     args.points, args.workers, args.threads,
    # )
    # bench_stream(
    #     integrator, "fabio",
    #     lambda: gen_frames_fabio(args.hdf5, args.frames),
    #     args.points, args.workers, args.threads,
    # )
    bitshuffle_gen = maybe_flip(
        lambda: gen_frames_parallel(args.hdf5, args.dataset, args.frames, decomp_workers=dw),
    )
    bench_stream(
        integrator, f"parallel bitshuffle ({dw}w)",
        bitshuffle_gen,
        args.points, args.workers, args.threads,
    )
    bench_stream_to_array(
        integrator, f"  + collect to array",
        bitshuffle_gen,
        args.points, args.workers, args.threads,
    )

    tiff_path = os.path.join(DATA_DIR, "eiger_0000.tiff")
    if os.path.exists(tiff_path):
        tiff_gen = maybe_flip(lambda: gen_frames_tiff(tiff_path, args.frames))
        bench_stream(
            integrator, "tiff re-read",
            tiff_gen,
            args.points, args.workers, args.threads,
        )
        bench_stream_to_array(
            integrator, "  + collect to array",
            tiff_gen,
            args.points, args.workers, args.threads,
        )

    print()


if __name__ == "__main__":
    main()
