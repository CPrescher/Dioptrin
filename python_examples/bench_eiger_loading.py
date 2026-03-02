"""Benchmark frame loading speed from Eiger HDF5 files.

Compares h5py vs fabio vs parallel-bitshuffle readers for raw frame
loading throughput, independent of integration performance.

Requires optional dependencies:
    pip install dioptrin[benchmark]

Usage:
    python python_examples/bench_eiger_loading.py
    python python_examples/bench_eiger_loading.py --frames 100
    python python_examples/bench_eiger_loading.py --decomp-workers 14
"""

import argparse
import gc
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "test_data")
HDF5_PATH = os.path.join(DATA_DIR, "eiger_0000.h5")
HDF5_DATASET = "entry_0000/instrument/eiger/data"
TIFF_PATH = os.path.join(DATA_DIR, "eiger_0000.tiff")


# ---------------------------------------------------------------------------
# Frame loaders (bulk)
# ---------------------------------------------------------------------------

def load_frames_h5py(path: str, dataset: str, n_frames: int) -> list[np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[dataset]
        n = min(n_frames, ds.shape[0])
        return [ds[i].astype(np.float64) for i in range(n)]


def load_frames_fabio(path: str, n_frames: int) -> list[np.ndarray]:
    import fabio

    img = fabio.open(path)
    frames = [img.data.astype(np.float64)]
    for _ in range(min(n_frames, img.nframes) - 1):
        img = img.next()
        frames.append(img.data.astype(np.float64))
    return frames


def _decompress_bitshuffle_lz4(raw_bytes, shape, dtype):
    """Decompress a single bitshuffle-LZ4 compressed chunk."""
    import bitshuffle

    block_size = struct.unpack(">I", raw_bytes[8:12])[0]
    buf = np.frombuffer(raw_bytes, dtype=np.uint8, offset=12)
    return bitshuffle.decompress_lz4(buf, shape, dtype, block_size).astype(np.float64)


def load_frames_parallel(
    path: str, dataset: str, n_frames: int, decomp_workers: int = 8
) -> list[np.ndarray]:
    """Load frames by reading raw HDF5 chunks and decompressing in parallel.

    Uses a sliding window to avoid holding all raw chunks in memory at once.
    """
    import h5py
    from collections import deque

    with h5py.File(path, "r") as f:
        ds = f[dataset]
        dtype = ds.dtype
        shape = (ds.chunks[1], ds.chunks[2])
        n = min(n_frames, ds.shape[0])

        frames = []
        with ThreadPoolExecutor(max_workers=decomp_workers) as pool:
            pending = deque()
            idx = 0

            while idx < n and len(pending) < decomp_workers:
                raw = bytes(ds.id.read_direct_chunk((idx, 0, 0))[1])
                pending.append(pool.submit(_decompress_bitshuffle_lz4, raw, shape, dtype))
                idx += 1

            while pending:
                frames.append(pending.popleft().result())
                if idx < n:
                    raw = bytes(ds.id.read_direct_chunk((idx, 0, 0))[1])
                    pending.append(pool.submit(_decompress_bitshuffle_lz4, raw, shape, dtype))
                    idx += 1

    return frames


# ---------------------------------------------------------------------------
# Frame generators (streaming from disk)
# ---------------------------------------------------------------------------

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
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_bulk_loader(name: str, load_fn) -> None:
    """Time a bulk loader, report results, and free memory."""
    gc.collect()
    t0 = time.perf_counter()
    frames = load_fn()
    elapsed = time.perf_counter() - t0
    n = len(frames)
    h, w = frames[0].shape
    mb_per_frame = h * w * 8 / 1e6
    print(f"  {name:30s}: {elapsed:7.2f}s  ({n:4d} frames, {n / elapsed:6.1f} fps, {mb_per_frame:.1f} MB/frame)")
    del frames
    gc.collect()


def bench_generator_loader(name: str, gen_fn) -> None:
    """Time consuming a frame generator, report results, and free memory."""
    gc.collect()
    t0 = time.perf_counter()
    count = 0
    shape = None
    for frame in gen_fn():
        shape = frame.shape
        count += 1
        del frame
    elapsed = time.perf_counter() - t0
    h, w = shape
    mb_per_frame = h * w * 8 / 1e6
    print(f"  {name:30s}: {elapsed:7.2f}s  ({count:4d} frames, {count / elapsed:6.1f} fps, {mb_per_frame:.1f} MB/frame)")
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frame loading speed from Eiger HDF5")
    parser.add_argument("--hdf5", default=HDF5_PATH, help="Path to HDF5 file")
    parser.add_argument("--dataset", default=HDF5_DATASET, help="HDF5 dataset path")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to load")
    parser.add_argument("--decomp-workers", type=int, default=8, help="Decompression thread count")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        print(f"Skipping: HDF5 file not found: {args.hdf5}")
        sys.exit(77)

    print(f"HDF5 file      : {args.hdf5}")
    print(f"Frames         : {args.frames}")
    print(f"Decomp workers : {args.decomp_workers}")

    dw = args.decomp_workers

    print(f"\n=== Bulk loading (all frames into memory) ===")
    bench_bulk_loader("h5py", lambda: load_frames_h5py(args.hdf5, args.dataset, args.frames))
    bench_bulk_loader("fabio", lambda: load_frames_fabio(args.hdf5, args.frames))
    bench_bulk_loader(f"parallel bitshuffle ({dw}w)", lambda: load_frames_parallel(args.hdf5, args.dataset, args.frames, decomp_workers=dw))

    print(f"\n=== Generator loading (one frame at a time) ===")
    bench_generator_loader("h5py generator", lambda: gen_frames_h5py(args.hdf5, args.dataset, args.frames))
    bench_generator_loader("fabio generator", lambda: gen_frames_fabio(args.hdf5, args.frames))
    bench_generator_loader(f"parallel bitshuffle gen ({dw}w)", lambda: gen_frames_parallel(args.hdf5, args.dataset, args.frames, decomp_workers=dw))

    tiff_path = os.path.join(DATA_DIR, "eiger_0000.tiff")
    if os.path.exists(tiff_path):
        bench_generator_loader("tiff re-read", lambda: gen_frames_tiff(tiff_path, args.frames))

    print()


if __name__ == "__main__":
    main()
