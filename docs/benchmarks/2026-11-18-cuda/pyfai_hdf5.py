"""Compare pyFAI OpenCL input buffers on the real Eiger HDF5 scan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from hdf5_batch import digest_file, scan_info
from pyfai_compare import BINS, select_gpu
from run import ROOT, command, cpu_model


class NativeReader:
    """Wrap the benchmark's caller-buffer dioptrin-io bridge."""

    def __init__(self, bridge, filename, dataset):
        self.library = dll = ctypes.CDLL(bridge)
        dll.eiger_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
        dll.eiger_open.restype = ctypes.c_void_p
        dll.eiger_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        dll.eiger_read.restype = ctypes.c_int
        dll.eiger_close.argtypes = [ctypes.c_void_p]
        dll.eiger_close.restype = None
        shape = (ctypes.c_size_t * 3)()
        self.handle = dll.eiger_open(filename.encode(), dataset.encode(), shape)
        if not self.handle:
            raise OSError("dioptrin-io could not open the Eiger snapshot")
        self.shape = tuple(shape)

    def read(self, i, output):
        if output.dtype.name != "uint32" or not (output.dtype.isnative and output.flags.carray):
            raise ValueError("dioptrin-io requires a writable contiguous native uint32 buffer")
        if self.library.eiger_read(self.handle, i, output.ctypes.data, output.size):
            raise OSError(f"dioptrin-io could not decode frame {i}")

    def close(self):
        self.library.eiger_close(self.handle)


def worker(config):
    import numpy as np
    import pyFAI
    import pyopencl as cl
    from pyFAI.detectors import detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.method_registry import IntegrationMethod
    from pyopencl.array import Array

    target, device = select_gpu(config["gpu_name"])
    method, = IntegrationMethod.select_method(
        dim=1, split="full", algo="csr", impl="opencl", target=target, degradable=False,
    )
    g = config["geometry"]
    detector = detector_factory(g["detector"], {"orientation": 3})
    ai = AzimuthalIntegrator(
        dist=g["distance_m"], poni1=g["center_row"] * g["pixel1"],
        poni2=g["center_col"] * g["pixel2"], wavelength=g["wavelength_m"], detector=detector,
    )
    frames, rows, columns = config["shape"]
    count = config["frames"]
    first_frame = config["first_frame"]
    frame_shape = (rows, columns)
    dtype = np.dtype("uint32")
    if detector.shape != frame_shape or (detector.pixel1, detector.pixel2) != (g["pixel1"], g["pixel2"]):
        raise ValueError("pyFAI catalog geometry differs from the Eiger scan")
    if config["extra_mask"]:
        mask = detector.mask.copy()
        indices = json.loads(Path(config["extra_mask"]).read_text(encoding="utf-8"))
        if any(not 0 <= index < mask.size for index in indices):
            raise ValueError("HDF5 mask index outside detector")
        mask.flat[indices] = 1
        detector.mask = mask

    def integrate(frame, safe=False):
        return ai.integrate1d(
            frame, BINS, unit="2th_deg", method=method, correctSolidAngle=True,
            error_model="poisson", safe=safe,
        )

    first = integrate(np.zeros(frame_shape, dtype=dtype), safe=True)
    if first.method != method or "opencl" not in first.compute_engine.lower():
        raise RuntimeError("pyFAI fell back from the requested OpenCL engine")
    engine = ai.engines[method].engine
    mode = config["mode"]
    direct = mode.endswith("opencl-host")
    native = mode.startswith("io-")
    fabio = mode == "fabio"
    mapper = cl.CommandQueue(engine.ctx) if direct else None
    free, ready = Queue(), Queue()
    def mapped(slot, flags):
        array, _ = cl.enqueue_map_buffer(
            mapper, slot.data, flags, 0, (1, rows, columns), dtype, is_blocking=True,
        )
        return array

    for _ in range(2):
        if direct:
            flags = cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR
            buf = cl.Buffer(engine.ctx, flags, size=rows * columns * dtype.itemsize)
            slot = Array(engine.queue, frame_shape, dtype, data=buf)
            if native:
                # Rust borrows a slice of initialized u32s from these mappings.
                view = mapped(slot, cl.map_flags.WRITE_INVALIDATE_REGION)
                view.fill(0)
                view.base.release(mapper).wait()
        elif fabio:
            slot = True
        elif native:
            slot = np.zeros((1, rows, columns), dtype=dtype)
        else:
            slot = np.empty((1, rows, columns), dtype=dtype)
        free.put(slot)
    patterns = np.empty((count, BINS, 2), dtype="<f8")

    def produce():
        read_s = 0.0
        try:
            with ExitStack() as stack:
                if native:
                    source = NativeReader(config["io_bridge"], config["hdf5"], config["dataset"])
                    stack.callback(source.close)
                    source_shape = source.shape
                elif fabio:
                    from fabio import open as fabio_open
                    import hdf5plugin  # noqa: F401  # Registers bitshuffle.
                    source = stack.enter_context(fabio_open(f'{config["hdf5"]}::{config["dataset"]}'))
                    source_shape = (source.nframes, *source.shape)
                else:
                    import h5py
                    import hdf5plugin  # noqa: F401  # Registers bitshuffle.
                    handle = stack.enter_context(h5py.File(config["hdf5"], "r"))
                    source = handle[config["dataset"]]
                    source_shape = source.shape
                if source_shape != (frames, rows, columns):
                    raise ValueError("HDF5 scan shape changed after discovery")
                for i in range(first_frame, first_frame + count):
                    slot = free.get()
                    if slot is None:
                        break
                    selection = np.s_[i:i + 1, :, :]
                    started = perf_counter()
                    if direct:
                        flags = cl.map_flags.WRITE if native else cl.map_flags.WRITE_INVALIDATE_REGION
                        view = mapped(slot, flags)
                        try:
                            if native:
                                source.read(i, view)
                            else:
                                source.read_direct(view, source_sel=selection)
                        finally:
                            view.base.release(mapper).wait()
                    elif native:
                        source.read(i, slot)
                    elif fabio:
                        slot = source.getframe(i).data
                        if slot.dtype != dtype:
                            raise ValueError("FabIO changed the dtype of Eiger pixels")
                    else:
                        source.read_direct(slot, source_sel=selection)
                    read_s += perf_counter() - started
                    ready.put(slot)
        finally:
            ready.put(None)
        return read_s

    compute_s = 0.0
    pipeline_start = perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(produce)
        try:
            for i in range(count):
                slot = ready.get()
                if slot is None:
                    reader.result()
                    raise RuntimeError("HDF5 reader stopped before completing the scan")
                started = perf_counter()
                result = integrate(slot if direct or fabio else slot[0])
                patterns[i, :, 0] = result.intensity
                patterns[i, :, 1] = result.sigma
                compute_s += perf_counter() - started
                free.put(True if fabio else slot)
        finally:
            free.put(None)
        read_s = reader.result()
    pipeline_s = perf_counter() - pipeline_start
    if not np.isfinite(patterns).all() or not np.any(patterns[:, :, 0] > 0):
        raise ValueError("invalid integration output")
    patterns.tofile(config["output"])
    stats = dict(
        mode=config["mode"], pyfai=pyFAI.version, pyopencl=cl.VERSION_TEXT,
        device=device, read_and_handoff_s=read_s, integrate_and_collect_s=compute_s,
        prepared_pipeline_s=pipeline_s, masked_pixels=int(detector.mask.sum()),
        mask_sha256=hashlib.sha256(detector.mask.astype(np.uint8).tobytes()).hexdigest(),
        axis_deg=first.radial.tolist(),
    )
    Path(config["stats"]).write_text(json.dumps(stats), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dataset", default="/entry_0000/ESRF-ID11/eiger/data")
    parser.add_argument("--gpu-name", default="RTX 5070 Ti")
    parser.add_argument("--frames", type=int, default=0, help="Use 0 for all frames")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--readers", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1])
    parser.add_argument("--inputs", nargs="+",
                        choices=("ordinary", "opencl-host", "fabio", "io-ordinary", "io-opencl-host"),
                        default=["ordinary", "opencl-host"])
    parser.add_argument("--io-bridge", type=Path)
    parser.add_argument("--extra-mask", type=Path, help="JSON list of extra masked flat pixel indices")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patterns-out", type=Path, help="Optional local f64 intensity and sigma pairs")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(json.loads(args.case.read_text(encoding="utf-8")))
        return
    if not all((args.hdf5, args.metadata, args.out)) or args.repeats < 1:
        parser.error("specify --hdf5, --metadata, --out, and a positive repeat count")
    if any(mode.startswith("io-") for mode in args.inputs) and not args.io_bridge:
        parser.error("dioptrin-io inputs require --io-bridge")

    import h5py
    import hdf5plugin
    import numpy as np

    count, shape, geometry, compressed, filters = scan_info(
        args.hdf5, args.metadata, args.dataset, args.frames,
    )
    if max(args.readers) > count:
        parser.error("there must be at least one frame per reader")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = dict(metadata=dict(
        commit=command("git", "rev-parse", "HEAD"), cpu=cpu_model(),
        harness_sha256=digest_file(Path(__file__)),
        file_sha256=digest_file(args.hdf5), metadata_sha256=digest_file(args.metadata),
        dataset=args.dataset, shape=shape, frames=count, compressed_bytes=compressed,
        geometry=geometry, hdf5_version=h5py.version.hdf5_version,
        hdf5plugin=hdf5plugin.version, hdf5_filter_ids=filters, bins=BINS,
        io_bridge_sha256=digest_file(args.io_bridge) if args.io_bridge else None,
        extra_mask_sha256=digest_file(args.extra_mask) if args.extra_mask else None,
        cache="normal OS cache after reading the whole file",
    ), records=[])
    env = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    env.setdefault("PYOPENCL_CACHE_DIR", str(ROOT / "target/paper-gpu-cache/opencl"))
    modes = [(mode, readers) for readers in args.readers for mode in args.inputs]
    with TemporaryDirectory(dir=args.out.parent) as directory:
        work = Path(directory)
        reference = None
        for repeat in range(args.repeats):
            for mode, readers in modes[::1 if repeat % 2 == 0 else -1]:
                def run_shard(i):
                    case, output, stats = (work / f"{name}-{i}" for name in ("case", "patterns", "stats"))
                    first_frame, end = i * count // readers, (i + 1) * count // readers
                    case.write_text(json.dumps(dict(
                        geometry=geometry, shape=shape, first_frame=first_frame,
                        frames=end - first_frame, gpu_name=args.gpu_name, mode=mode,
                        hdf5=str(args.hdf5.resolve()), dataset=args.dataset, output=str(output), stats=str(stats),
                        io_bridge=str(args.io_bridge.resolve()) if args.io_bridge else None,
                        extra_mask=str(args.extra_mask.resolve()) if args.extra_mask else None,
                    )), encoding="utf-8")
                    subprocess.run([sys.executable, "-E", "-s", "-B", __file__, "--case", str(case)],
                                   cwd=ROOT, env=env, check=True)
                    return output, stats

                started = perf_counter()
                with ThreadPoolExecutor(max_workers=readers) as pool:
                    shards = list(pool.map(run_shard, range(readers)))
                pattern = np.concatenate([np.fromfile(output, dtype="<f8") for output, _ in shards])
                pattern = pattern.reshape(count, BINS, 2)
                elapsed = perf_counter() - started
                rows = [json.loads(stats.read_text(encoding="utf-8")) for _, stats in shards]
                if reference is None:
                    reference = pattern
                    record["axis_deg"] = rows[0]["axis_deg"]
                    if args.patterns_out:
                        pattern.tofile(args.patterns_out)
                np.testing.assert_array_equal(pattern, reference)
                for row in rows:
                    np.testing.assert_array_equal(row.pop("axis_deg"), record["axis_deg"])
                row = dict(mode=mode, readers=readers, shards=rows, process_to_patterns_s=elapsed,
                           repeat=repeat, matches_ordinary=True)
                record["records"].append(row)
                args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                print(f"{mode} x{readers}: {elapsed:.3f} s for {count} frames", flush=True)
    for mode, readers in modes:
        results = [r["process_to_patterns_s"] for r in record["records"]
                   if (r["mode"], r["readers"]) == (mode, readers)]
        print(f"{mode} x{readers} median: {median(results):.3f} s", flush=True)


if __name__ == "__main__":
    main()
