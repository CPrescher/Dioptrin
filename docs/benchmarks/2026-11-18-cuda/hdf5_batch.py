"""Benchmark the real Eiger HDF5 file from compressed chunks to f64 patterns."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, command, cpu_model, gpu_snapshot
from startup import build

TEST = "integration::cuda::bench::hdf5::eiger_hdf5"
MARKER = "DIOPTRIN_HDF5_BATCH "
METHODS = ("counting", "pixel_split", "pyfai_full")
BACKENDS = ("cpu_csc", "cpu_csr", "cuda_native_block", "cuda_native_warp", "cuda_f64")
PATTERN = 6000


def trial(executable, env):
    args = [executable, TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    start = time.perf_counter_ns()
    with subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        rows, log = [], []
        for line in process.stdout:
            _, marker, payload = line.partition(MARKER)
            if marker:
                elapsed = (time.perf_counter_ns() - start) / 1e6
                row = json.loads(payload)
                if row["iteration"] == 0:
                    row["process_to_patterns_ms"] = elapsed
                rows.append(row)
            else:
                log.append(line)
        if process.wait() or len(rows) != 1 + int(env["DIOPTRIN_HDF5_REPEATS"]):
            raise RuntimeError("HDF5 benchmark failed\n" + "".join(log))
        return rows


def scan_info(hdf5, metadata, dataset, requested):
    import h5py
    import numpy as np

    with h5py.File(hdf5, "r") as source:
        data = source[dataset]
        if data.ndim != 3 or data.dtype != np.dtype("uint32") or data.chunks != (1, *data.shape[1:]):
            raise ValueError("expected frame-chunked uint32 Eiger data")
        frames = requested or data.shape[0]
        if not 0 < frames <= data.shape[0]:
            raise ValueError("frame count outside the dataset")
        properties = data.id.get_create_plist()
        filters = [properties.get_filter(i)[0] for i in range(properties.get_nfilters())]
        compressed_bytes = sum(data.id.get_chunk_info_by_coord((i, 0, 0)).size for i in range(frames))
        shape = tuple(data.shape)
    with h5py.File(metadata, "r") as sidecar:
        settings = sidecar["1.1/instrument"]
        detector = settings["eiger"]
        calibration = settings["crysalis"]
        if tuple(detector["image"].shape) != shape:
            raise ValueError("sidecar Eiger dimensions differ from the scan")
        if detector["image_rotation"][()] != b"NONE" or any(detector["image_flip"][()]):
            raise ValueError("this benchmark expects the detector's untransformed storage order")
        geometry = dict(
            detector="Eiger2 CdTe 4M",
            pixel1=float(detector["y_pixel_size"][()]),
            pixel2=float(detector["x_pixel_size"][()]),
            distance_m=float(calibration["distance"][()]) * 1e-3,
            center_row=float(calibration["ycenter"][()]),
            center_col=float(calibration["xcenter"][()]),
            wavelength_m=float(calibration["wavelength"][()]) * 1e-10,
        )
    return frames, shape, geometry, compressed_bytes, filters


def compare_output(actual, reference, frames):
    import numpy as np

    size = frames * PATTERN
    results = np.fromfile(actual, dtype="<f8") if isinstance(actual, Path) else actual
    expected = np.fromfile(reference, dtype="<f8") if isinstance(reference, Path) else reference
    if results.size != size or expected.size != size:
        raise ValueError("HDF5 integration returned the wrong number of patterns")
    signal, sigma = 0., 0.
    for start in range(0, size, 16 * PATTERN):
        a, b = results[start:start + 16 * PATTERN], expected[start:start + 16 * PATTERN]
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
        diff = np.abs(a - b)
        signal = max(signal, float(diff[::2].max()))
        sigma = max(sigma, float(diff[1::2].max()))
    return dict(max_intensity_absolute_error=signal, max_std_absolute_error=sigma)


def digest_file(path):
    digest = hashlib.sha256()
    buffer = bytearray(8 << 20)
    with path.open("rb") as file:
        while size := file.readinto(buffer):
            digest.update(memoryview(buffer)[:size])
    return digest.hexdigest()


@contextmanager
def trial_input(path, workspace, cache, sha256):
    """Give normal buffered readers a new file whose copy bypassed the Windows cache."""
    if cache == "warm":
        yield path.resolve()
        return
    source = path.resolve()
    with tempfile.TemporaryDirectory(prefix="uncached-scan-", dir=workspace) as directory:
        destination = Path(directory) / source.name
        copied = subprocess.run(
            ["robocopy", str(source.parent), directory, source.name, "/J", "/NOOFFLOAD",
             "/R:0", "/W:0", "/NDL", "/NFL", "/NJH", "/NJS", "/NP"],
            capture_output=True, text=True, errors="replace",
        )
        if copied.returncode >= 8 or not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Unbuffered scan copy failed\n{copied.stdout}\n{copied.stderr}")
        yield destination
        # The integrity check reads the copy only after the timed workers exit.
        if digest_file(destination) != sha256:
            raise ValueError("Benchmark copy differs from the source Eiger scan")


def disk_reads():
    try:
        import psutil
    except ModuleNotFoundError:
        return {}

    return {device: row.read_bytes for device, row in (psutil.disk_io_counters(perdisk=True) or {}).items()}


def disk_delta(before):
    return {device: value - before[device] for device, value in disk_reads().items() if device in before}


def run_readers(executable, env, workspace, frames, readers):
    import numpy as np

    paths = [workspace / f"shard-{i}" for i in range(readers)]
    def run(i):
        first, last = i * frames // readers, (i + 1) * frames // readers
        workers = 14 // readers + int(i < 14 % readers)
        return trial(executable, dict(
            env, DIOPTRIN_HDF5_FIRST=str(first), DIOPTRIN_HDF5_FRAMES=str(last - first),
            DIOPTRIN_HDF5_CPU_WORKERS=str(workers), DIOPTRIN_HDF5_OUTPUT=str(paths[i]),
        ))[0]

    start = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=readers) as pool:
        rows = list(pool.map(run, range(readers)))
    # Include bringing each process's completed patterns into the caller.
    # The temporary output files are buffered and are removed after the run.
    patterns = np.concatenate([np.fromfile(p, dtype="<f8") for p in paths])
    total_ms = (time.perf_counter_ns() - start) / 1e6
    return [dict(iteration=0, process_to_patterns_ms=total_ms, readers=rows)], patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", default="/entry_0000/ESRF-ID11/eiger/data")
    parser.add_argument("--library", type=Path, help="Path to native HDF5 library")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--benchmark-exe", type=Path, help="Use a prebuilt HDF5 benchmark test executable")
    parser.add_argument("--extra-mask", type=Path, help="JSON list of extra masked flat pixel indices")
    parser.add_argument("--cache", choices=("warm", "cold"), default="warm")
    parser.add_argument("--frames", type=int, default=64, help="Use 0 for all frames")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument("--repeats", type=int, default=3, help="Prepared repeats after one fresh batch")
    parser.add_argument("--readers", nargs="+", type=int, choices=(1, 2, 4, 8),
                        help="Split the scan across independent processes and time output collection")
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    if args.repeats < 0 or any(not 1 <= tile <= 64 for tile in args.tile_sizes):
        parser.error("tile sizes must be 1 to 64 and repeats cannot be negative")
    if args.readers and args.repeats != 0:
        parser.error("--readers measures fresh runs and requires --repeats 0")
    if args.cache == "cold" and (os.name != "nt" or not args.readers):
        parser.error("cold runs use Windows and --readers with --repeats 0")
    if not args.library and os.name != "nt":
        parser.error("pass --library outside Windows")

    import h5py
    import hdf5plugin

    library = (args.library or Path(h5py.__file__).parent / "hdf5.dll").resolve(strict=True)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("CUDA_CACHE_DISABLE", None)
    env.pop("DIOPTRIN_HDF5_FIRST", None)
    env.pop("DIOPTRIN_HDF5_CPU_WORKERS", None)
    env.pop("DIOPTRIN_HDF5_EXTRA_MASK", None)
    if args.extra_mask:
        env["DIOPTRIN_HDF5_EXTRA_MASK"] = str(args.extra_mask.resolve(strict=True))
    search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    libs = [str(library.parent), env.get(search)]
    if args.nvrtc:
        libs.insert(0, str(args.nvrtc.resolve()))
    env[search] = os.pathsep.join(filter(None, libs))
    env["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
    frames, shape, geometry, compressed, filters = scan_info(
        args.hdf5, args.metadata, args.dataset, args.frames,
    )
    if args.readers and max(args.readers) > frames:
        parser.error("there must be at least one frame per reader")
    env.update(
        DIOPTRIN_HDF5_FILE=str(args.hdf5.resolve()),
        DIOPTRIN_HDF5_LIBRARY=str(library),
        DIOPTRIN_HDF5_DATASET=args.dataset,
        DIOPTRIN_HDF5_GEOMETRY=json.dumps(geometry),
        DIOPTRIN_HDF5_FRAMES=str(frames),
        DIOPTRIN_HDF5_REPEATS=str(args.repeats),
    )
    executable = str(args.benchmark_exe.resolve(strict=True)) if args.benchmark_exe else build(args.cargo, env)
    # The full-scan comparisons start with the same cache conditions. Hash
    # the file as it is read to record the input without another full pass.
    full_digest = digest_file(args.hdf5) if args.readers else None
    cache_label = "normal operating-system cache"
    if args.cache == "cold":
        cache_label = "first buffered read of an unbuffered copy"
    elif args.readers:
        cache_label = "normal OS cache after reading the whole file"
    sources = [Path(__file__), ROOT / "src/integration/cuda.rs",
               ROOT / "src/integration/cuda/kernels.cu",
               *ROOT.glob("src/integration/cuda/bench/hdf5*.rs"),
               *ROOT.glob("src/integration/cuda/bench/hdf5/*.rs")]
    data = dict(
        metadata=dict(
            commit=command("git", "rev-parse", "HEAD"),
            source_sha256={p.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
            hdf5_version=h5py.version.hdf5_version,
            hdf5plugin_version=hdf5plugin.version,
            executable_sha256=digest_file(Path(executable)),
            extra_mask_sha256=digest_file(args.extra_mask) if args.extra_mask else None,
            file_bytes=args.hdf5.stat().st_size,
            file_sha256=full_digest, metadata_sha256=digest_file(args.metadata),
            dataset=args.dataset, shape=shape, frames=frames, compressed_bytes=compressed,
            hdf5_filter_ids=filters, geometry=geometry, cpu_workers=14,
            cache=cache_label, platform=platform.platform(),
            cpu=cpu_model(), gpu_before=gpu_snapshot(),
        ),
        records=[],
    )
    rng = random.Random(20261118)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        workspace = Path(directory)
        for method in args.methods:
            baseline, measurement = (workspace / name for name in ("baseline", "measurement"))
            variants = [(backend, tile, readers) for backend in args.backends
                        for tile in args.tile_sizes for readers in args.readers or [None]]
            rng.shuffle(variants)
            variants.sort(key=lambda x: not x[0].startswith("cpu"))
            reference = None
            for index, (backend, tile, readers) in enumerate(variants):
                temporary = baseline if index == 0 else measurement
                print(f"{method} {backend} tile={tile} readers={readers or 1} ({frames} frames)", flush=True)
                trial_env = dict(
                    env, DIOPTRIN_HDF5_BACKEND=backend, DIOPTRIN_HDF5_METHOD=method,
                    DIOPTRIN_HDF5_TILE=str(tile), DIOPTRIN_HDF5_OUTPUT=str(temporary),
                )
                if readers:
                    with trial_input(args.hdf5, workspace, args.cache, full_digest) as path:
                        before = disk_reads()
                        rows, patterns = run_readers(
                            executable, dict(trial_env, DIOPTRIN_HDF5_FILE=str(path)), workspace, frames, readers,
                        )
                        rows[0]["disk_read_bytes"] = disk_delta(before)
                else:
                    rows, patterns = trial(executable, trial_env), temporary
                if index == 0:
                    reference = patterns
                errors = compare_output(patterns, reference, frames)
                data["records"].append(dict(
                    method=method, backend=backend, tile=tile, readers=readers or 1,
                    samples=rows, **errors,
                ))
                output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    data["metadata"]["gpu_after"] = gpu_snapshot()
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
