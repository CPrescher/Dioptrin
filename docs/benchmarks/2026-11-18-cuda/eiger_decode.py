"""Measure file-to-uint32 throughput with FabIO and dioptrin-io."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
import importlib.metadata
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from time import perf_counter

from hdf5_batch import digest_file, scan_info
from pyfai_hdf5 import NativeReader


def initialize(barrier):
    global ready
    ready = barrier


def worker(case):
    import numpy as np

    with ExitStack() as stack:
        if case["reader"] == "fabio":
            import fabio
            import hdf5plugin  # noqa: F401  # Registers bitshuffle.

            reader = stack.enter_context(fabio.open(f'{case["file"]}::{case["dataset"]}'))
            if (reader.nframes, *reader.shape) != tuple(case["shape"]):
                raise ValueError("unexpected FabIO dataset dimensions")
        else:
            reader = NativeReader(case["bridge"], case["file"], case["dataset"])
            stack.callback(reader.close)
            if reader.shape != tuple(case["shape"]):
                raise ValueError("unexpected dioptrin-io dataset dimensions")
            output = np.zeros(reader.shape[1:], dtype=np.uint32)
        ready.wait()
        started = perf_counter()
        for index in range(case["first"], case["stop"]):
            if case["reader"] == "fabio":
                output = reader.getframe(index).data
            else:
                reader.read(index, output)
        elapsed = perf_counter()
        if output.dtype != np.uint32:
            raise ValueError("decoder changed Eiger dtype")
        return started, elapsed


def scan_dummy_pixels(filename, bridge, dataset, frames, readers):
    """Collect locations pyFAI's float32 dummy comparison rejects for uint32 Eiger."""
    import numpy as np

    # Values from this midpoint through u32::MAX round to float32(2**32),
    # the representation of the Eiger -2 dummy code after conversion to uint32.
    first_rejected = (1 << 32) - 128

    def shard(i):
        source = NativeReader(bridge, filename, dataset)
        try:
            pixels = np.zeros(source.shape[1:], dtype=np.uint32)
            mask = np.zeros_like(pixels, dtype=bool)
            flagged = np.empty_like(mask)
            for frame in range(i * frames // readers, (i + 1) * frames // readers):
                source.read(frame, pixels)
                np.greater_equal(pixels, first_rejected, out=flagged)
                mask |= flagged
            return mask
        finally:
            source.close()

    with ThreadPoolExecutor(readers) as pool:
        mask = np.logical_or.reduce(list(pool.map(shard, range(readers))))
    return np.flatnonzero(mask).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--io-bridge", type=Path, required=True)
    parser.add_argument("--dataset", default="/entry_0000/ESRF-ID11/eiger/data")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--readers", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", type=Path)
    output.add_argument("--extra-mask-out", type=Path, help="Write a fixed union of sentinel locations")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    import h5py
    import hdf5plugin
    import numpy as np
    from fabio import open as fabio_open

    frames, shape, _, compressed, _ = scan_info(args.hdf5, args.metadata, args.dataset, args.frames)
    if max(args.readers) > frames:
        parser.error("each reader must have at least one frame")
    filename = str(args.hdf5.resolve())
    bridge = str(args.io_bridge.resolve())
    if args.extra_mask_out:
        indices = scan_dummy_pixels(filename, bridge, args.dataset, frames, max(args.readers))
        args.extra_mask_out.parent.mkdir(parents=True, exist_ok=True)
        args.extra_mask_out.write_text(json.dumps(indices) + "\n", encoding="utf-8")
        print(f"Found {len(indices)} sentinel pixel locations across {frames} frames", flush=True)
        return
    record = dict(
        file_sha256=digest_file(args.hdf5), io_bridge_sha256=digest_file(args.io_bridge),
        dataset=args.dataset, frames=frames, shape=shape, compressed_bytes=compressed,
        fabio=importlib.metadata.version("fabio"), hdf5=h5py.version.hdf5_version,
        hdf5plugin=hdf5plugin.version, samples=[],
    )
    native = NativeReader(bridge, filename, args.dataset)
    try:
        output = np.zeros(shape[1:], dtype=np.uint32)
        with fabio_open(f"{filename}::{args.dataset}") as source:
            for index in sorted({0, frames // 2, frames - 1}):
                native.read(index, output)
                np.testing.assert_array_equal(output, source.getframe(index).data)
    finally:
        native.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    configs = [(reader, count) for count in args.readers for reader in ("fabio", "dioptrin-io")]
    ctx = get_context("spawn")
    for repeat in range(args.repeats):
        for reader, count in configs[::1 if repeat % 2 == 0 else -1]:
            cases = [dict(file=filename, dataset=args.dataset, shape=shape, bridge=bridge,
                          reader=reader, first=i * frames // count, stop=(i + 1) * frames // count)
                     for i in range(count)]
            barrier = ctx.Barrier(count, timeout=60)
            with ProcessPoolExecutor(count, mp_context=ctx, initializer=initialize,
                                     initargs=(barrier,)) as pool:
                intervals = list(pool.map(worker, cases))
            seconds = max(end for _, end in intervals) - min(start for start, _ in intervals)
            row = dict(reader=reader, processes=count, seconds=seconds, frames_per_s=frames / seconds)
            record["samples"].append(row)
            args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f'{reader} x{count}: {row["frames_per_s"]:.1f} frames/s', flush=True)
    for reader, count in configs:
        samples = [r["frames_per_s"] for r in record["samples"]
                   if (r["reader"], r["processes"]) == (reader, count)]
        print(f"{reader} x{count} median: {median(samples):.1f} frames/s", flush=True)


if __name__ == "__main__":
    main()
