"""Measure installed pyFAI CPU and OpenCL GPU engines on the CUDA fixtures."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from run import ROOT, available_host_bytes, command, cpu_model, gpu_snapshot

MARKER = "PYFAI_BENCHMARK "
BINS = 3000


def emit(event, **data):
    print(MARKER + json.dumps({"event": event, **data}), flush=True)


def fixture(case, eiger_dir):
    import fabio
    import numpy as np
    import pyFAI
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    if case == "pilatus":
        tiff = ROOT / "test_data/CeO2_Pilatus1M.tif"
        poni = ROOT / "test_data/v1.poni"
        mask_path = None
    elif case == "eiger":
        tiff = eiger_dir / "eiger_0000.tiff"
        poni = ROOT / "docs/benchmarks/2026-11-05-bslz4/calibration.poni"
        mask_path = eiger_dir / "eiger_0000.mask"
    else:
        raise ValueError(f"unknown real fixture {case}")
    with fabio.open(str(tiff)) as handle:
        # Match the row orientation of Dioptrin's Rust TIFF loader.
        signal = np.ascontiguousarray(handle.data[::-1], dtype=np.float64)
    original = pyFAI.load(str(poni))
    mask = (~np.isfinite(signal)) | (signal < 0)
    if mask_path:
        with fabio.open(str(mask_path)) as handle:
            mask |= handle.data[::-1] != 0
    # Dioptrin uses the uniform pixels of the PONI and the supplied mask.
    detector = Detector(
        pixel1=original.detector.pixel1, pixel2=original.detector.pixel2,
        max_shape=signal.shape, orientation=3,
    )
    ai = AzimuthalIntegrator(
        dist=original.dist, poni1=original.poni1, poni2=original.poni2,
        rot1=original.rot1, rot2=original.rot2, rot3=original.rot3,
        wavelength=original.wavelength, detector=detector,
    )
    ai.detector.mask = np.ascontiguousarray(mask, dtype=np.int8)
    masked = int(mask.sum())
    return ai, signal, masked


def select_gpu(name):
    import pyopencl as cl

    for platform_index, platform in enumerate(cl.get_platforms()):
        for device_index, device in enumerate(platform.get_devices()):
            if device.type & cl.device_type.GPU and name.lower() in device.name.lower():
                return (platform_index, device_index), {
                    "platform": platform.name, "device": device.name,
                    "driver": device.driver_version, "opencl": device.version,
                }
    raise RuntimeError(f"No OpenCL GPU matching {name!r}")


def worker(config):
    start = time.perf_counter()
    import numpy as np
    import pyFAI.integrator.azimuthal  # Populate the native method registry.
    from pyFAI.method_registry import IntegrationMethod
    from threadpoolctl import threadpool_info

    backend = config["backend"]
    gpu = backend == "gpu"
    target, device = select_gpu(config["gpu_name"]) if gpu else (None, None)
    impl = "opencl" if gpu else "cython"
    requested = (config["split"], config["algorithm"], impl)
    methods = IntegrationMethod.select_method(
        dim=1, split=requested[0], algo=requested[1], impl=impl, target=target,
        degradable=False,
    )
    if not methods:
        raise RuntimeError(f"Exact pyFAI method unavailable: {requested}, target={target}")
    method = methods[0]
    imports_ms = (time.perf_counter() - start) * 1000
    before_fixture = time.perf_counter()
    ai, signal, masked = fixture(config["case"], Path(config["eiger_dir"]))
    fixture_ms = (time.perf_counter() - before_fixture) * 1000

    def integrate(integrator, frame, errors, safe):
        return integrator.integrate1d(
            frame, BINS, unit="2th_deg", method=method, correctSolidAngle=True,
            polarization_factor=0.965, error_model="poisson" if errors else None,
            safe=safe,
        )

    def validate(pattern, errors):
        actual = pattern.method.method
        if (actual.split, actual.algo, actual.impl) != requested or (gpu and tuple(actual.target) != target):
            raise RuntimeError(f"pyFAI changed {requested}/{target} to {actual}")
        engine = str(pattern.compute_engine)
        if gpu and "opencl" not in engine.lower():
            raise RuntimeError(f"pyFAI reported an OpenCL method but executed {engine}")
        if pattern.intensity.shape != (BINS,) or not np.isfinite(pattern.intensity).all():
            raise RuntimeError("invalid intensity result")
        if not np.any(pattern.intensity > 0):
            raise RuntimeError("the real fixture produced no positive bins")
        if errors and (pattern.sigma is None or not np.isfinite(pattern.sigma).all()):
            raise RuntimeError("invalid Poisson result")
        return engine

    before_first = time.perf_counter()
    first = integrate(ai, signal, True, True)
    first_ms = (time.perf_counter() - before_first) * 1000
    engine = validate(first, True)
    emit(
        "first", import_and_selection_ms=imports_ms, fixture_ms=fixture_ms,
        plan_and_first_integration_ms=first_ms, engine=engine, target=target,
        device=device, shape=signal.shape, masked_pixels=masked,
        native_thread_pools=[{k: v for k, v in row.items() if k != "filepath"}
                             for row in threadpool_info()],
    )
    if config["setup_only"]:
        return

    # Match the established CPU comparison: share geometry but give each
    # concurrent caller its own CSR engine and per-frame scratch buffers.
    integrators = [ai]
    now = time.perf_counter()
    for _ in range(1, config["frame_workers"]):
        replica = copy.copy(ai)
        replica.engines = {}
        validate(integrate(replica, signal, True, True), True)
        integrators.append(replica)
    pool = ThreadPoolExecutor(max_workers=len(integrators)) if len(integrators) > 1 else None
    emit("workers", additional_worker_setup_ms=(time.perf_counter() - now) * 1000)

    def run_batch(frames, errors):
        if pool is None or len(frames) == 1:
            return [integrate(ai, frame, errors, False) for frame in frames]
        width = (len(frames) + len(integrators) - 1) // len(integrators)
        tasks = zip(integrators, (frames[i:i + width] for i in range(0, len(frames), width)))

        def chunk(task):
            integrator, images = task
            return [integrate(integrator, frame, errors, False) for frame in images]

        return [pattern for group in pool.map(chunk, tasks) for pattern in group]

    for dtype in config["dtypes"]:
        source = signal.astype(dtype, copy=False)
        count = max(config["batches"])
        host_limit = min(config["host_limit_bytes"], available_host_bytes() - 1024**3)
        if source.nbytes * count > host_limit:
            raise RuntimeError(f"{dtype} batch of {count} exceeds host limit")
        frames = np.repeat(source[np.newaxis, :, :], count, axis=0)
        for errors in (False, True):
            baseline = integrate(ai, frames[0], errors, True)
            validate(baseline, errors)
            for n in config["batches"]:
                samples = []
                for trial in range(config["repeats"] + 2):
                    started = time.perf_counter()
                    batch = run_batch(frames[:n], errors)
                    seconds = time.perf_counter() - started
                    if trial >= 2:
                        samples.append(seconds * 1000)
                for pattern in batch:
                    validate(pattern, errors)
                    np.testing.assert_allclose(pattern.intensity, baseline.intensity, rtol=1e-6, atol=1e-7)
                    if errors:
                        np.testing.assert_allclose(pattern.sigma, baseline.sigma, rtol=1e-6, atol=1e-7)
                emit("timing", dtype=dtype, frames=n, errors=errors, samples_ms_per_batch=samples)
        del frames
    if pool is not None:
        pool.shutdown()


def run_child(config, site, cache):
    env = dict(
        os.environ, OMP_NUM_THREADS=str(config["threads"]), OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        MPLCONFIGDIR=str(cache / "matplotlib"), XDG_CACHE_HOME=str(cache),
    )
    args = [sys.executable, "-B", __file__, "--child", json.dumps(config)]
    if site:
        args += ["--site", str(site)]
    start = time.perf_counter()
    rows = []
    other = []
    with subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            if line.startswith(MARKER):
                row = json.loads(line[len(MARKER):])
                if row["event"] == "first":
                    row["process_to_first_pattern_ms"] = (time.perf_counter() - start) * 1000
                rows.append(row)
            else:
                other.append(line)
        status = process.wait()
    return {"status": status, "records": rows, "diagnostics": "".join(other)}


def input_hashes(eiger_dir):
    paths = {
        "pilatus_image": ROOT / "test_data/CeO2_Pilatus1M.tif",
        "pilatus_poni": ROOT / "test_data/v1.poni",
        "eiger_image": eiger_dir / "eiger_0000.tiff",
        "eiger_mask": eiger_dir / "eiger_0000.mask",
        "eiger_poni": ROOT / "docs/benchmarks/2026-11-05-bslz4/calibration.poni",
    }
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items() if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--eiger-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--site", type=Path, help="Optional workspace-local wheel directory")
    parser.add_argument("--cache", type=Path, default=ROOT / "target" / "pyfai-cache")
    parser.add_argument("--gpu-name", default="NVIDIA")
    parser.add_argument("--cases", default="pilatus,eiger")
    parser.add_argument("--splits", default="no,full")
    parser.add_argument("--cpu-threads", default="1,14,16")
    parser.add_argument("--cpu-algorithms", default="csr,csc")
    parser.add_argument("--cpu-frame-workers", type=int, default=4)
    parser.add_argument("--backends", default="cpu,gpu")
    parser.add_argument("--batches", default="1,8,64")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--setup-repeats", type=int, default=3)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.site:
        sys.path.insert(0, str(args.site.resolve()))
    if args.child:
        worker(json.loads(args.child))
        return
    if not args.out or args.repeats < 3 or args.setup_repeats < 1:
        parser.error("provide --out, at least three timing repeats and one setup repeat")

    eiger_dir = args.eiger_dir.resolve()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    results = {
        "metadata": {
            "source_commit": command("git", "rev-parse", "HEAD"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version, "cpu": cpu_model(), "logical_processors": os.cpu_count(),
            "bins": BINS, "gpu_before": gpu_snapshot(), "sha256": input_hashes(eiger_dir),
            "packages": {name: importlib.metadata.version(name) for name in ("pyFAI", "pyopencl", "numpy", "scipy", "fabio", "silx")},
        },
        "runs": [],
    }
    configurations = []
    if "cpu" in args.backends.split(","):
        configurations.extend(("cpu", algo, int(threads), 1) for algo in args.cpu_algorithms.split(",")
                              for threads in args.cpu_threads.split(","))
        if args.cpu_frame_workers > 1:
            configurations.append(("cpu", "csr", 1, args.cpu_frame_workers))
    if "gpu" in args.backends.split(","):
        configurations.append(("gpu", "csr", 14, 1))
    failures = []
    for case in args.cases.split(","):
        for split in args.splits.split(","):
            for backend, algorithm, threads, workers in configurations:
                print(f"{case} {split} {backend} {algorithm} omp={threads} frame_workers={workers}", flush=True)
                config = {
                    "case": case, "split": split, "backend": backend, "algorithm": algorithm,
                    "frame_workers": workers,
                    "threads": threads, "eiger_dir": str(eiger_dir), "gpu_name": args.gpu_name,
                    "batches": [int(n) for n in args.batches.split(",")],
                    "dtypes": args.dtypes.split(","), "repeats": args.repeats,
                    "host_limit_bytes": min(15_000_000_000, available_host_bytes() - 2 * 1024**3),
                }
                for i in range(args.setup_repeats):
                    trial = dict(config, setup_only=i < args.setup_repeats - 1)
                    run = run_child(trial, args.site, args.cache.resolve())
                    # Keep the file portable even when a custom fixture directory was used.
                    del trial["eiger_dir"]
                    results["runs"].append(dict(run, configuration=trial, trial=i + 1))
                    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                    if run["status"]:
                        print(run["diagnostics"], file=sys.stderr, flush=True)
                        failures.append((case, split, backend, algorithm, threads, workers))
                        break
    results["metadata"]["gpu_after"] = gpu_snapshot()
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Failed configurations: {failures}")


if __name__ == "__main__":
    main()
