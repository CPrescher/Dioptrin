#!/usr/bin/env python3
"""Fresh-process CPU comparison of Dioptrin, azint, and pyFAI (1D, in memory)."""

import argparse
import copy
import datetime
import csv
import importlib.metadata
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fixtures import MANIFEST, manifest, prepare, sha256
from provenance import azint_signature, verify_azint_build, verify_build


ROOT = Path(__file__).resolve().parent.parent.parent
VARIANTS = (
    "dioptrin-pixel_split",
    "dioptrin-supersampled",
    "dioptrin-pyfai_full",
    "azint",
    "azint-cached",
    "pyfai-full",
    "pyfai-bbox",
    "pyfai-full-omp",
    "pyfai-full-opencl",
)


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
)
# Exact pyFAI (split, algo, impl); fallbacks are rejected.
PYFAI_METHODS = {
    "pyfai-full": ("full", "csr", "cython"),
    "pyfai-bbox": ("bbox", "csr", "cython"),
    "pyfai-full-omp": ("full", "csr", "cython"),
    "pyfai-full-opencl": ("full", "csr", "opencl"),
}
# One engine integrating frames in sequence: pyFAI parallelises inside the
# CSR kernel (OpenMP threads, or the OpenCL device) instead of over frames.
SEQUENTIAL_VARIANTS = ("pyfai-full-omp", "pyfai-full-opencl")
GPU_VARIANTS = ("pyfai-full-opencl",)
CONTROL = "@@"
# Multi-process blocks start this long after the parent sends the command.
START_LEAD_SECONDS = 0.1
SPIN_SECONDS = 0.03


def case_threads(variant, count, processes=1):
    """Per-case thread budget; with processes > 1 it is per process.

    Dioptrin gets a Rayon pool of `count`. azint's sparse-matrix construction
    (generate_matrix) is its only OpenMP region; its SpMV is serial and runs on
    `count` Python frame threads, so OMP_NUM_THREADS=count parallelises only
    set-up. pyFAI's Python-thread rows stay at 1: its macOS wheel has no
    OpenMP, and more OpenMP threads would change what is measured there.
    `pyfai-full-omp` is pyFAI's own CPU mode: one CSR engine with
    OMP_NUM_THREADS=count. The OpenCL row uses one host thread.
    """
    if count % processes:
        raise ValueError("The CPU budget must be divisible by the process count")
    per = str(count // processes)
    omp = variant.startswith("azint") or variant == "pyfai-full-omp"
    return {
        "OMP_NUM_THREADS": per if omp else "1",
        "RAYON_NUM_THREADS": per if variant.startswith("dioptrin-") else "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "FALSE",
    }


def same_bits(a, b):
    import numpy as np

    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contiguous_chunks(frames, workers):
    indexed = list(enumerate(frames))
    size = (len(frames) + workers - 1) // workers
    return [indexed[i * size : (i + 1) * size] for i in range(workers)]


def process_share(total, processes, rank):
    """Contiguous logical frame indices of one process (all for rank None)."""
    if rank is None:
        return list(range(total))
    if total % processes:
        raise ValueError("Frame count must be divisible by the process count")
    share = total // processes
    return list(range(rank * share, (rank + 1) * share))


OPENMP_SYMBOLS = (b"GOMP_parallel", b"__kmpc_fork_call", b"omp_get_max_threads")


def pyfai_openmp():
    """Whether pyFAI's full-split CSR extension was compiled with OpenMP.

    pyFAI has no runtime flag for this. The extension's import table names the
    OpenMP runtime entry points (GOMP_* for GCC builds, __kmpc_* for LLVM),
    and on Linux the loaded runtime library also appears in /proc/self/maps.
    """
    import pyFAI.ext.splitPixelFullCSR as module

    data = Path(module.__file__).read_bytes()
    symbols = [s.decode() for s in OPENMP_SYMBOLS if s in data]
    loaded = []
    maps = Path("/proc/self/maps")
    if maps.is_file():
        loaded = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(k in line for k in ("libgomp", "libomp", "libiomp"))
            }
        )
    return dict(
        openmp=bool(symbols),
        module=module.__file__,
        symbols=symbols,
        loaded_runtime=loaded,
    )


def opencl_target(device):
    """pyFAI method selection for --opencl-device (gpu, cpu, acc, any, or P:D)."""
    if ":" in device:
        platform_id, device_id = (int(v) for v in device.split(":"))
        return dict(target=(platform_id, device_id))
    if device not in ("gpu", "cpu", "acc", "any"):
        raise ValueError(
            f"--opencl-device must be gpu, cpu, acc, any or P:D, not {device}"
        )
    return {} if device == "any" else dict(target_type=device)


def opencl_method(device):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: F401  (registry)
    from pyFAI.method_registry import IntegrationMethod

    found = IntegrationMethod.select_method(
        dim=1,
        split="full",
        algo="csr",
        impl="opencl",
        degradable=False,
        **opencl_target(device),
    )
    return found[0] if found else None


def opencl_device_info(device):
    """Platform, device, and precision facts for a pyopencl Device."""
    import pyopencl

    extensions = device.extensions.split()
    return dict(
        platform=device.platform.name.strip(),
        platform_version=device.platform.version.strip(),
        device=device.name.strip(),
        device_type=pyopencl.device_type.to_string(device.type),
        device_version=device.version.strip(),
        driver_version=device.driver_version.strip(),
        device_fp64=("cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions),
        # pyFAI 2026.x azim_csr: float32 buffers/coefficients and float32
        # double-word (float2, compensated) accumulators in ocl_azim_CSR.cl,
        # whatever the device's fp64 support.
        single_precision_accumulation=True,
        accumulation="float32 arithmetic with double-word (float2) compensated "
        "accumulators (pyFAI ocl_azim_CSR.cl); float32 input and coefficients",
    )


def probe(device):
    """Availability of the optional pyFAI variants, in a fresh process."""
    result = {}
    try:
        result["pyfai_openmp"] = pyfai_openmp()
    except Exception as error:  # report, don't crash the parent
        result["pyfai_openmp"] = dict(openmp=False, reason=repr(error))
    opencl = dict(available=False, requested_device=device)
    try:
        import pyopencl  # noqa: F401
    except ImportError as error:
        opencl["reason"] = f"pyopencl not importable: {error}"
    else:
        try:
            method = opencl_method(device)
            if method is None:
                opencl["reason"] = f"no OpenCL device matches {device!r}"
            else:
                from pyFAI.opencl import ocl

                platform_id, device_id = method.target
                info = opencl_device_info(
                    ocl.platforms[platform_id].devices[device_id]._device
                )
                opencl.update(available=True, target=list(method.target), **info)
        except Exception as error:
            opencl["reason"] = repr(error)
    result["opencl"] = opencl
    print(json.dumps(result))


def setup_case(case, rank=None):
    """Load inputs, build and warm the engine, and return the measurement hooks.

    With rank None the case runs in this one process. With a rank, this
    process is one of `processes`: it copies only its own frames into memory it
    allocates (first touch, so NUMA binding places them locally) and builds its
    own tables with `workers / processes` threads.
    """
    import numpy as np

    fixture = Path(case["fixture"])
    meta = json.loads((fixture / "fixture.json").read_text())
    stack = np.load(fixture / "frames.npy", mmap_mode="r")
    processes = case.get("processes", 1)
    distinct = 1 if case["workload"] == "repeated" else len(stack)
    logical = process_share(case["frames"], processes, rank)
    if rank is None:
        needed = list(range(distinct))
        unique = {u: stack[u] for u in needed}
    else:
        needed = sorted({i % distinct for i in logical})
        unique = {u: np.array(stack[u]) for u in needed}
    del stack
    image = unique[needed[0]]
    mask = np.load(fixture / "mask.npy")
    frames = [unique[i % distinct] for i in logical]
    # Fault every input page in before setup/timing. All cases see the same
    # immutable frame sequence; frame generation/decoding is never timed.
    for frame in unique.values():
        float(np.sum(frame))
    n = case["bins"]
    workers = case["workers"] // processes
    errors = case["errors"]
    variant = case["variant"]
    bounds = meta["radial_range"]
    pool = None
    stock_check = None
    info = {}
    started = time.perf_counter()

    if variant.startswith("dioptrin-"):
        import dioptrin

        ai = dioptrin.Integrator(
            str(fixture / "geometry.poni"),
            method=variant.removeprefix("dioptrin-"),
            unit="2th_deg",
            n_ss=meta["n_splitting"],
            polarization_factor=0.965,
            radial_range=tuple(bounds),
        )
        ai.set_mask(mask)

        # The current array batch path uses the global Rayon pool rather than
        # its file-worker arguments. Set the real total CPU budget at startup.
        if dioptrin.rayon_num_threads() != workers:
            raise RuntimeError(
                "Dioptrin Rayon pool does not match requested CPU budget"
            )

        def integrate_frames(images):
            return ai.batch1d(
                images, n, errors=errors, num_workers=workers, num_threads=1
            )

        def run():
            return integrate_frames(frames)

        def extract(result):
            if not result.is_ok():
                raise RuntimeError(result.error)
            value = result.result
            return (
                np.asarray(value.radial),
                np.asarray(value.intensity),
                np.asarray(value.errors) if errors else None,
            )

        extract(integrate_frames([image])[0])
        setup_first = time.perf_counter() - started
        baseline = [extract(r) for r in integrate_frames([unique[u] for u in needed])]
    else:
        if variant in ("azint", "azint-cached"):
            import azint
            from azint.detector import Detector

            geometry = meta["geometry"].copy()
            pixel1, pixel2 = geometry.pop("pixel1"), geometry.pop("pixel2")
            poni = azint.Poni(det=Detector(pixel1, pixel2, image.shape), **geometry)
            ai = azint.AzimuthalIntegrator(
                poni,
                n_splitting=meta["n_splitting"],
                radial_bins=np.linspace(*bounds, n + 1),
                unit="2th",
                mask=mask,
                solid_angle=True,
                polarization_factor=0.965,
                error_model="poisson" if errors else None,
            )
            if variant == "azint-cached":
                # Masked pixels are already excluded from the sparse matrix, and
                # the constructor cached norm = A @ 1. With mask=None, integrate()
                # reuses that norm instead of multiplying the frame by the
                # inverted mask and recomputing A @ (1 - mask) on every call.
                # Shallow per-thread copies below inherit mask=None and only
                # read the shared norm array.
                ai.mask = None

                def stock_check(images, cached):
                    stock = copy.copy(ai)
                    stock.mask = mask
                    for frame, value in zip(images, cached):
                        reference = stock.integrate(frame)
                        for index in (0, 1):
                            if not same_bits(reference[index], value[index]):
                                raise AssertionError(
                                    "azint-cached differs from stock azint output"
                                )
                    return True

            def integrate(engine, frame):
                return engine.integrate(frame)

            def extract(result):
                return (
                    np.asarray(ai.radial_axis),
                    np.asarray(result[0]),
                    result[1] if errors else None,
                )

        else:
            import pyFAI
            from pyFAI.method_registry import IntegrationMethod

            ai = pyFAI.load(str(fixture / "geometry.poni"))
            ai.detector.mask = mask
            method = PYFAI_METHODS[variant]
            if variant != "pyfai-full-opencl":
                info["pyfai_openmp"] = pyfai_openmp()
                if variant == "pyfai-full-omp" and not info["pyfai_openmp"]["openmp"]:
                    raise RuntimeError(
                        "pyFAI's CSR extension has no OpenMP; pyfai-full-omp is meaningless here"
                    )
            if variant == "pyfai-full-opencl":
                selected = opencl_method(case["opencl_device"])
                if selected is None:
                    raise RuntimeError(
                        f"No pyFAI OpenCL device for {case['opencl_device']!r}"
                    )
                target = tuple(selected.target)
            else:
                selected, target = method, None
                if not IntegrationMethod.select_method(
                    dim=1,
                    split=method[0],
                    algo="csr",
                    impl=method[2],
                    degradable=False,
                ):
                    raise RuntimeError(
                        f"Requested exact pyFAI method unavailable: {method}"
                    )

            def integrate(engine, frame):
                return engine.integrate1d(
                    frame,
                    n,
                    unit="2th_deg",
                    method=selected,
                    radial_range=tuple(bounds),
                    correctSolidAngle=True,
                    polarization_factor=0.965,
                    error_model="poisson" if errors else None,
                    safe=True,
                )

            def extract(result):
                used = result.method.method
                if (used.split, used.algo, used.impl) != method or (
                    target is not None and tuple(used.target) != target
                ):
                    raise RuntimeError(f"pyFAI fell back from {selected} to {used}")
                return (
                    np.asarray(result.radial),
                    np.asarray(result.intensity),
                    result.sigma if errors else None,
                )

        extract(integrate(ai, image))
        setup_first = time.perf_counter() - started
        if variant == "pyfai-full-opencl":
            engine = next(
                e.engine for e in ai.engines.values() if hasattr(e.engine, "ctx")
            )
            info["opencl"] = dict(
                target=list(target), **opencl_device_info(engine.ctx.devices[0])
            )
        baseline = [extract(integrate(ai, unique[u])) for u in needed]
        threads = 1 if variant in SEQUENTIAL_VARIANTS else workers
        engines = [ai]
        for _ in range(1, threads):
            engine = copy.copy(ai)
            if variant.startswith("pyfai-"):
                # Share immutable geometry arrays, but each frame worker owns
                # its CSR engine and scratch buffers. Never concurrently use ai.
                engine.engines = {}
            integrate(engine, image)
            engines.append(engine)
        pool = ThreadPoolExecutor(max_workers=threads)
        chunks = contiguous_chunks(frames, threads)

        def chunk(task):
            engine, images = task
            return [(index, integrate(engine, frame)) for index, frame in images]

        def run():
            results = [None] * len(frames)
            for group in pool.map(chunk, zip(engines, chunks)):
                for index, result in group:
                    results[index] = result
            return results

    sample = tuple(
        np.stack([entry[i] for entry in baseline]) if i < 2 or errors else np.array([])
        for i in range(3)
    )
    row = {u: position for position, u in enumerate(needed)}.__getitem__
    baseline = sample
    if case.get("reference"):
        with np.load(case["reference"]) as reference:
            baseline = (
                reference["radial"].copy(),
                reference["intensity"].copy(),
                reference["sigma"].copy() if errors else np.array([]),
            )
        row = int  # the reference holds every distinct frame in order
    elif rank is not None:
        raise RuntimeError("Multi-process cases need a one-worker reference")

    def validate(results):
        if len(results) != len(frames):
            raise AssertionError("Wrong output frame count")
        for index, result in enumerate(results):
            radial, intensity, sigma = extract(result)
            expected = row(logical[index] % distinct)
            if intensity.shape != (n,) or not np.isfinite(intensity).all():
                raise AssertionError("Invalid integration output")
            if not np.any(intensity > 0):
                raise AssertionError("Empty integration output")
            np.testing.assert_allclose(
                radial, baseline[0][expected], rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                intensity, baseline[1][expected], rtol=1e-6, atol=1e-6
            )
            if errors:
                if sigma is None or not np.isfinite(sigma).all():
                    raise AssertionError("Invalid Poisson errors")
                np.testing.assert_allclose(
                    sigma, baseline[2][expected], rtol=1e-6, atol=1e-6
                )

    warm_result = run()
    setup_ready = time.perf_counter() - started
    validate(warm_result)
    if stock_check is not None:
        # Outside setup and timing: the cached-normalisation results must be
        # bit-identical (intensity and sigma) to stock masked azint.
        checked = min(len(needed), len(frames))
        stock_check(frames[:checked], warm_result[:checked])
    del warm_result
    negative = np.argwhere(sample[2] < 0).tolist() if errors else []
    return dict(
        meta=meta,
        run=run,
        validate=validate,
        pool=pool,
        sample=sample,
        needed=needed,
        frames=len(frames),
        distinct=distinct,
        setup_first=setup_first,
        setup_ready=setup_ready,
        stock_checked=stock_check is not None,
        negative_sigma=[[needed[i], b] for i, b in negative],
        info=info,
    )


def peak_rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1 if sys.platform == "darwin" else 1024) / 2**20


def case_record(case, state):
    return dict(
        **case,
        setup_first_s=state["setup_first"],
        setup_ready_s=state["setup_ready"],
        thread_environment={k: os.environ.get(k) for k in THREAD_VARIABLES},
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS"),
        stock_azint_bit_identical=True if state["stock_checked"] else None,
        distinct_frames=state["distinct"],
        self_check="passed",
        **state["info"],
    )


def numerical_fields(negative):
    return dict(
        numerical_status="negative_sigma" if negative else "ok",
        negative_sigma_bins=len(negative),
        negative_sigma_bin_indices=negative,
    )


def summary_line(case, meta, rate):
    processes = case.get("processes", 1)
    return (
        f"{meta['dataset']:14} {case['workload']:8} {case['variant']:25} workers={case['workers']:3} "
        + (f"processes={processes} " if processes > 1 else "")
        + f"{case['error_mode']:9} {rate:10.1f} frames/s"
    )


def worker(case_path, rank=None):
    import numpy as np

    case = json.loads(Path(case_path).read_text())
    if rank is not None:
        serve_rank(case, rank)
        return
    state = setup_case(case)
    run, validate = state["run"], state["validate"]
    # Warmup and validation are outside warmed timing blocks. Native batch API
    # scheduling and result allocation are included in every measured call.
    durations = []
    counts = []
    for _ in range(case["blocks"]):
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < case["seconds"]:
            result = run()
            count += len(result)
            del result
        durations.append(time.perf_counter() - t0)
        counts.append(count)
    validate(run())
    if state["pool"]:
        state["pool"].shutdown()
    rates = [count / duration for count, duration in zip(counts, durations)]
    result_path = Path(case["result"])
    sample = state["sample"]
    np.savez_compressed(
        result_path.with_suffix(".npz"),
        radial=sample[0],
        intensity=sample[1],
        sigma=sample[2] if case["errors"] else np.array([]),
    )
    negative = state["negative_sigma"]
    dump(
        result_path,
        dict(
            **case_record(case, state),
            frames_per_s=float(np.median(rates)),
            block_frames_per_s=rates,
            block_seconds=durations,
            block_frames=counts,
            peak_rss_mib=peak_rss(),
            **numerical_fields(negative),
        ),
    )
    print(summary_line(case, state["meta"], np.median(rates)), flush=True)
    if negative:
        print(
            f"NUMERICAL WARNING: {len(negative)} negative Poisson sigma bins; raw values retained",
            flush=True,
        )


def send(stream, tag, value):
    stream.write(f"{CONTROL}{tag} {json.dumps(value)}\n")
    stream.flush()


def serve_rank(case, rank):
    """One process of a multi-process case, driven over stdin/stdout.

    The parent releases all ranks at a common start time; each runs the same
    number of passes over its own frames and reports its finish time.
    """
    # Keep the control channel private: anything else written to stdout,
    # including by native code, goes to the rank's log (stderr).
    control = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    state = setup_case(case, rank)
    run = state["run"]
    send(
        control,
        "ready",
        dict(
            rank=rank,
            pid=os.getpid(),
            frames=state["frames"],
            setup_first_s=state["setup_first"],
            setup_ready_s=state["setup_ready"],
        ),
    )
    for line in sys.stdin:
        command = line.split()
        if not command or command[0] == "finish":
            break
        if command[0] != "run":
            raise RuntimeError(f"Unknown control command: {line!r}")
        repeats, start_at = int(command[1]), float(command[2])
        # Sleep, then spin through the last 30 ms: OS timer slack (several ms
        # on macOS) would otherwise delay the start of some ranks.
        while (remaining := start_at - time.monotonic()) > SPIN_SECONDS:
            time.sleep(remaining - SPIN_SECONDS)
        while time.monotonic() < start_at:
            pass
        started = time.monotonic()
        count = 0
        for _ in range(repeats):
            result = run()
            count += len(result)
            del result
        send(control, "done", dict(start=started, end=time.monotonic(), frames=count))
    state["validate"](run())
    if state["pool"]:
        state["pool"].shutdown()
    path = Path(case["result"]).with_suffix(f".rank{rank}.json")
    dump(
        path,
        dict(
            **case_record(case, state),
            rank=rank,
            pid=os.getpid(),
            local_frames=state["frames"],
            local_distinct_frames=state["needed"],
            peak_rss_mib=peak_rss(),
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            **numerical_fields(state["negative_sigma"]),
        ),
    )
    send(control, "result", str(path))


def azint_build_lines(output):
    try:
        environment = json.loads((output / "environment.json").read_text())
    except (OSError, ValueError):
        return []
    build = environment.get("azint_build")
    if not build:
        return []
    flags = (
        (build.get("recorded") or {}).get("flags") or build.get("legacy_flags") or {}
    )
    return [
        f"azint build: {build.get('variant', 'unrecorded')}; "
        f"CXXFLAGS `{flags.get('CXXFLAGS', '').strip()}`"
        + (
            f", LDFLAGS `{flags['LDFLAGS'].strip()}`"
            if flags.get("LDFLAGS", "").strip()
            else ""
        )
        + ("" if not build.get("problems") else " (UNVERIFIED)")
    ]


def extra_lines(output):
    """Skipped variants, OpenCL devices and beam-pixel masks, from run records."""
    lines = []
    try:
        skipped = json.loads((output / "skipped.json").read_text())
    except (OSError, ValueError):
        skipped = []
    for entry in skipped:
        lines.append(f"SKIPPED {entry['variant']}: {entry['reason']}")
    try:
        environment = json.loads((output / "environment.json").read_text())
    except (OSError, ValueError):
        environment = {}
    opencl = (environment.get("variant_probes") or {}).get("opencl") or {}
    if opencl.get("available"):
        lines.append(
            f"pyFAI OpenCL device: {opencl['device']} ({opencl['device_type']}) on "
            f"{opencl['platform']}; fp64 {'yes' if opencl['device_fp64'] else 'no'}; "
            "single-precision double-word accumulation"
        )
    for fixture in sorted(output.glob("*/fixture.json")):
        meta = json.loads(fixture.read_text())
        beam = meta.get("beam_pixel_mask")
        if beam:
            added = [p for p in beam["pixels"] if p not in beam["already_masked"]]
            lines.append(
                f"{meta['dataset']}: --mask-beam-pixel {beam['mode']} masked pixels "
                f"{beam['pixels']} ({len(added)} newly); minimum-2theta pixel "
                f"{beam['min_2theta_pixel']}"
            )
    return lines


def summarize(output):
    import numpy as np

    rows = [
        json.loads(p.read_text())
        for p in sorted((output / "cases").glob("*.result.json"))
    ]
    fields = [
        "dataset",
        "workload",
        "variant",
        "workers",
        "error_mode",
        "trial",
        "frames_per_s",
        "setup_first_s",
        "setup_ready_s",
        "peak_rss_mib",
        "omp_num_threads",
        "rayon_num_threads",
        "self_check",
        "numerical_status",
        "negative_sigma_bins",
        # Added columns (appended so earlier positions are unchanged).
        "processes",
        "workers_per_process",
        "binding",
        "mask_beam_pixel",
        "beam_pixels_added",
    ]
    for row in rows:
        row.setdefault("processes", 1)
        row.setdefault("workers_per_process", row["workers"])
        row.setdefault("binding", "none")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    numerical = []
    for row in rows:
        if row["workers"] != 1 or row["trial"] != 0:
            continue
        reference = next(
            (
                r
                for r in rows
                if r["dataset"] == row["dataset"]
                and r["workload"] == row["workload"]
                and r["variant"] == "dioptrin-pixel_split"
                and r["workers"] == 1
                and r["trial"] == 0
                and r["errors"] == row["errors"]
            ),
            None,
        )
        if reference is None:
            continue
        with np.load(Path(row["result"]).with_suffix(".npz")) as saved:
            a = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        with np.load(Path(reference["result"]).with_suffix(".npz")) as saved:
            b = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        metrics = dict(
            dataset=row["dataset"],
            workload=row["workload"],
            variant=row["variant"],
            error_mode=row["error_mode"],
            reference="dioptrin-pixel_split",
            negative_sigma_bins=row.get("negative_sigma_bins", 0),
            max_axis_difference_deg=float(np.max(np.abs(a["radial"] - b["radial"]))),
        )
        for key in ("intensity", "sigma"):
            if a[key].size:
                denominator = float(np.linalg.norm(b[key]))
                metrics[key + "_relative_l2"] = float(
                    np.linalg.norm(a[key] - b[key])
                ) / max(denominator, 1e-300)
        numerical.append(metrics)
    # azint-cached must reproduce stock azint exactly; compare every matching case.
    identity = []
    for row in rows:
        if row["variant"] != "azint-cached" or row["processes"] != 1:
            # Multi-process ranks check stock bit-identity in process only.
            continue
        stock = next(
            (
                r
                for r in rows
                if r["variant"] == "azint"
                and all(
                    r[k] == row[k]
                    for k in (
                        "dataset",
                        "workload",
                        "workers",
                        "processes",
                        "errors",
                        "trial",
                    )
                )
            ),
            None,
        )
        if stock is None:
            continue
        with np.load(Path(row["result"]).with_suffix(".npz")) as saved:
            a = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        with np.load(Path(stock["result"]).with_suffix(".npz")) as saved:
            b = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        identity.append(
            dict(
                **{
                    k: row[k]
                    for k in ("dataset", "workload", "workers", "error_mode", "trial")
                },
                **{
                    key + "_bit_identical": same_bits(a[key], b[key])
                    for key in ("intensity", "sigma")
                    if row["errors"] or key == "intensity"
                },
                in_process_stock_check=row.get("stock_azint_bit_identical"),
            )
        )
    dump(output / "numerical.json", numerical)
    dump(output / "azint_cached_identity.json", identity)
    lines = [
        "# CPU integration benchmark",
        "",
        "1D, preloaded float64 inputs; generation, decoding, and output writing excluded.",
        "Repeated reuses one image; frames cycles through distinct generated images.",
        "Cross-library algorithms, precision, and normalization differ; see numerical.json.",
        "azint reuses its constructor normalisation in azint-cached; azint_cached_identity.json",
        "records its bit-identity with stock azint. azint uses OMP_NUM_THREADS = CPU budget",
        "(matrix construction only); pyFAI uses 1 except pyfai-full-omp (one CSR engine,",
        "OMP_NUM_THREADS = budget); Dioptrin uses a Rayon pool of the budget. With several",
        "processes each gets budget/processes; throughput is all frames over the wall time",
        "from a common start to the last process finishing.",
        *azint_build_lines(output),
        *extra_lines(output),
        "",
        "| Dataset | Workload | Method | CPU budget | Processes | Errors | Median frames/s | Trials | Numerical status |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    keys = ("dataset", "workload", "variant", "workers", "processes", "error_mode")
    groups = sorted({tuple(r[f] for f in keys) for r in rows})
    for key in groups:
        members = [r for r in rows if tuple(r[f] for f in keys) == key]
        values = [r["frames_per_s"] for r in members]
        status = ", ".join(
            sorted({r.get("numerical_status", "unchecked") for r in members})
        )
        lines.append(
            "| "
            + " | ".join(
                map(
                    str, (*key, round(float(np.median(values)), 2), len(values), status)
                )
            )
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def numa_nodes():
    """NUMA nodes with CPUs in this process's affinity: {node: [cpus]}."""
    allowed = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    )
    nodes = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        cpus = set()
        for part in (path / "cpulist").read_text().strip().split(","):
            if part:
                low, _, high = part.partition("-")
                cpus.update(range(int(low), int(high or low) + 1))
        if cpus & allowed:
            nodes[int(path.name.removeprefix("node"))] = sorted(cpus & allowed)
    return nodes


def binding_commands(binding, processes):
    """Command prefix per rank. `numa` binds rank r's CPUs and memory to one
    NUMA node, round-robin over the nodes in the allocation."""
    if binding == "none":
        return [[] for _ in range(processes)]
    numactl = shutil.which("numactl")
    nodes = sorted(numa_nodes())
    if numactl is None or not nodes:
        raise RuntimeError("--bind numa needs Linux NUMA topology and numactl in PATH")
    return [
        [
            numactl,
            f"--cpunodebind={nodes[r % len(nodes)]}",
            f"--membind={nodes[r % len(nodes)]}",
        ]
        for r in range(processes)
    ]


def expect(process, tag, log):
    """Next control message `tag` from a rank; other stdout lines go to its log."""
    for line in process.stdout:
        if line.startswith(CONTROL + tag + " "):
            return json.loads(line[len(CONTROL + tag) + 1 :])
        log.write(line)
        log.flush()
    raise RuntimeError(f"Rank process exited (code {process.wait()}) before {tag!r}")


def run_processes(case, case_path, env, prefixes):
    """Run one case as `processes` coordinated processes and aggregate it.

    Each block releases every rank at the same start time; each integrates the
    same number of passes over its own frames. Untimed calibration blocks set
    the passes per block so that a block lasts about --seconds.
    """
    import numpy as np

    processes, logs = [], []
    try:
        for rank, prefix in enumerate(prefixes):
            log = case_path.with_suffix(f".rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        *prefix,
                        sys.executable,
                        "-E",
                        "-s",
                        __file__,
                        "--case",
                        str(case_path),
                        "--rank",
                        str(rank),
                    ],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
            )
        ready = [expect(p, "ready", log) for p, log in zip(processes, logs)]

        def block(repeats):
            start = time.monotonic() + START_LEAD_SECONDS
            for process in processes:
                process.stdin.write(f"run {repeats} {start!r}\n")
                process.stdin.flush()
            done = [expect(p, "done", log) for p, log in zip(processes, logs)]
            return (
                sum(d["frames"] for d in done),
                max(d["end"] for d in done) - start,
                max(d["start"] for d in done) - start,
            )

        # Two untimed calibration blocks: one pass, then about a quarter block.
        _, calibration, _ = block(1)
        trial_passes = max(1, math.ceil(0.25 * case["seconds"] / calibration))
        _, calibration, _ = block(trial_passes)
        calibration /= trial_passes
        repeats = max(1, math.ceil(case["seconds"] / calibration))
        blocks = [block(repeats) for _ in range(case["blocks"])]
        for process in processes:
            process.stdin.write("finish\n")
            process.stdin.flush()
        paths = [expect(p, "result", log) for p, log in zip(processes, logs)]
        for process in processes:
            if process.wait(timeout=600):
                raise RuntimeError(f"Rank exited with {process.returncode}")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.kill()
        raise
    finally:
        for log in logs:
            log.close()
    ranks = [json.loads(Path(path).read_text()) for path in paths]
    counts = [b[0] for b in blocks]
    durations = [b[1] for b in blocks]
    rates = [c / d for c, d in zip(counts, durations)]
    negative = sorted(
        {tuple(i) for r in ranks for i in r["negative_sigma_bin_indices"]}
    )
    first = ranks[0]
    record = dict(
        {k: v for k, v in first.items() if k not in ("rank", "pid", "peak_rss_mib")},
        setup_first_s=max(r["setup_first_s"] for r in ready),
        setup_ready_s=max(r["setup_ready_s"] for r in ready),
        stock_azint_bit_identical=True
        if all(r["stock_azint_bit_identical"] for r in ranks)
        else first["stock_azint_bit_identical"],
        frames_per_s=float(np.median(rates)),
        block_frames_per_s=rates,
        block_seconds=durations,
        block_frames=counts,
        passes_per_block=repeats,
        calibration_seconds_per_pass=calibration,
        max_start_lag_s=max(b[2] for b in blocks),
        peak_rss_mib=sum(r["peak_rss_mib"] for r in ranks),
        peak_rss_mib_per_process=[r["peak_rss_mib"] for r in ranks],
        rank_commands=[" ".join(prefix) or "(unbound)" for prefix in prefixes],
        rank_cpu_affinity=[r["cpu_affinity"] for r in ranks],
        rank_results=[str(path) for path in paths],
        **numerical_fields([list(i) for i in negative]),
    )
    for key in ("local_frames", "local_distinct_frames", "cpu_affinity"):
        record.pop(key, None)
    dump(case["result"], record)
    return summary_line(
        case,
        json.loads((Path(case["fixture"]) / "fixture.json").read_text()),
        record["frames_per_s"],
    ) + (
        f"\nNUMERICAL WARNING: {len(negative)} negative Poisson sigma bins; raw values retained"
        if negative
        else ""
    )


def probe_variants(variants, opencl_device):
    """Run the availability probe in a fresh process (same flags as cases)."""
    if not any(v in ("pyfai-full-omp", "pyfai-full-opencl") for v in variants):
        return {}
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            __file__,
            "--probe",
            "--opencl-device",
            opencl_device,
        ],
        capture_output=True,
        text=True,
        env=os.environ | case_threads("pyfai-full", 1),
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode or not lines:
        raise RuntimeError("Variant probe failed: " + result.stderr[-2000:])
    return json.loads(lines[-1])


def skipped_variants(variants, probes):
    skipped = []
    if "pyfai-full-omp" in variants and not probes["pyfai_openmp"]["openmp"]:
        skipped.append(
            dict(
                variant="pyfai-full-omp",
                reason="this pyFAI build has no OpenMP in its CSR extension "
                f"({probes['pyfai_openmp'].get('module', 'unknown')}); "
                "OMP_NUM_THREADS would have no effect (macOS wheels lack OpenMP)",
            )
        )
    if "pyfai-full-opencl" in variants and not probes["opencl"]["available"]:
        skipped.append(
            dict(
                variant="pyfai-full-opencl",
                reason=probes["opencl"].get("reason", "OpenCL unavailable"),
            )
        )
    return skipped


def main():
    definition = manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=definition["suites"], default="standard")
    parser.add_argument(
        "--output",
        type=Path,
        help="New results directory; default: benchmark-results/<timestamp>",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument(
        "--subset-dir",
        type=Path,
        help="Directory of prepare_subset.py bundles for the *-subset datasets; "
        "default: --data-dir",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "dioptrin-benchmarks",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use bundled/cached data only"
    )
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="Retain generated frame stacks after success",
    )
    parser.add_argument(
        "--allow-unverified-build",
        action="store_true",
        help="Explicitly allow an unrecorded/stale Dioptrin build, with a warning in the report",
    )
    parser.add_argument(
        "--workers",
        help="Comma-separated CPU budgets; default: suite budgets within CPU affinity",
    )
    parser.add_argument(
        "--processes",
        default="1",
        help="Comma-separated process counts. P > 1 splits each CPU budget over P "
        "processes (budget/P workers each, own frames and tables); budgets not "
        "divisible by P are skipped. Default: 1",
    )
    parser.add_argument(
        "--bind",
        choices=("none", "numa"),
        default="none",
        help="Multi-process CPU/memory binding: numa binds each process to one "
        "NUMA node with numactl (round-robin). Default: none",
    )
    parser.add_argument(
        "--mask-beam-pixel",
        nargs="?",
        const="poni",
        choices=("poni", "direct"),
        help="Mask, for all libraries, the pixel(s) containing or touching the "
        "point of normal incidence (poni, the default) or the direct-beam "
        "position (direct, 2theta = 0)",
    )
    parser.add_argument(
        "--opencl-device",
        default="gpu",
        help="pyfai-full-opencl device: gpu, cpu, acc, any, or PLATFORM:DEVICE ids",
    )
    parser.add_argument("--datasets", nargs="+", choices=definition["datasets"])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("repeated", "frames"),
        default=["repeated", "frames"],
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--bins", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(args.case, args.rank)
        return
    if args.probe:
        probe(args.opencl_device)
        return
    if args.list_datasets:
        print(json.dumps(definition, indent=2))
        return
    try:
        opencl_target(args.opencl_device)
    except ValueError as error:
        parser.error(str(error))
    defaults = definition["suites"][args.suite]
    for key in ("bins", "frames", "trials", "blocks", "seconds", "datasets"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    for key in ("bins", "frames", "trials", "blocks", "seconds"):
        if getattr(args, key) <= 0 or not math.isfinite(getattr(args, key)):
            parser.error(f"--{key} must be positive and finite")
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        workers = (
            sorted({int(w) for w in args.workers.split(",")})
            if args.workers
            else [w for w in defaults["workers"] if w <= available]
        )
        process_counts = sorted({int(p) for p in args.processes.split(",")} | {1})
    except ValueError:
        parser.error(
            "--workers and --processes must contain comma-separated positive integers"
        )
    if not workers or workers[0] < 1 or workers[-1] > available:
        parser.error(
            f"--workers must be within the process CPU allocation (1..{available})"
        )
    if process_counts[0] < 1:
        parser.error("--processes must be positive")
    workers = sorted(set(workers) | {1})
    bindings = {}
    for count in process_counts:
        if count > 1:
            try:
                bindings[count] = binding_commands(args.bind, count)
            except RuntimeError as error:
                parser.error(str(error))
    required = ["numpy", "pyFAI"]
    if any(v.startswith("dioptrin-") for v in args.variants):
        required.append("dioptrin")
    if any(v.startswith("azint") for v in args.variants):
        required.append("azint")
    if any(definition["datasets"][d]["kind"] == "subset" for d in args.datasets):
        required += ["h5py", "hdf5plugin"]
    try:
        versions = {name: importlib.metadata.version(name) for name in required}
    except importlib.metadata.PackageNotFoundError as error:
        parser.error(
            f"Missing benchmark dependency {error.name}; run python3 benchmarks/comparison/setup.py first"
        )
    for optional in ("pyopencl",):
        try:
            versions[optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            pass
    build_path = Path(sys.prefix) / "benchmark-build.json"
    installed_build = (
        json.loads(build_path.read_text()) if build_path.is_file() else None
    )
    build_problems = (
        verify_build(ROOT, installed_build) if "dioptrin" in required else []
    )
    azint_build = None
    if "azint" in required:
        installed_azint = azint_signature()
        recorded = (installed_build or {}).get("azint_build")
        azint_problems = verify_azint_build(installed_build, installed_azint)
        build_problems += azint_problems
        azint_build = dict(
            variant=(recorded or {}).get("variant", "unrecorded"),
            recorded=recorded,
            # Records written before azint_build existed share one flag set.
            legacy_flags=None if recorded else (installed_build or {}).get("flags"),
            installed_extension=installed_azint,
            problems=azint_problems,
        )
    if build_problems:
        message = "; ".join(build_problems)
        if not args.allow_unverified_build:
            parser.error(
                message
                + ". Rerun setup.py, or explicitly use --allow-unverified-build."
            )
        print("UNVERIFIED BUILD: " + message, file=sys.stderr, flush=True)
    try:
        probes = probe_variants(args.variants, args.opencl_device)
    except RuntimeError as error:
        parser.error(str(error))
    skipped = skipped_variants(args.variants, probes)
    for entry in skipped:
        print(f"SKIPPED {entry['variant']}: {entry['reason']}", flush=True)
    variants_run = [
        v for v in args.variants if v not in {e["variant"] for e in skipped}
    ]
    module_paths = {
        name: importlib.import_module(name).__file__
        for name in required
        if name not in ("h5py", "hdf5plugin")
    }
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "benchmark-results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    dump(output / "skipped.json", skipped)
    provenance = {}
    if (ROOT / ".git").exists():
        for key, command in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("status", ["git", "status", "--short"]),
        ):
            provenance[key] = subprocess.check_output(
                command, cwd=ROOT, text=True
            ).strip()
    dump(
        output / "environment.json",
        dict(
            versions=versions,
            python=sys.version,
            platform=platform.platform(),
            node=platform.node(),
            available_logical_cpus=available,
            cpu_budgets=workers,
            process_counts=process_counts,
            binding=args.bind,
            rank_commands={
                count: [" ".join(prefix) for prefix in prefixes]
                for count, prefixes in bindings.items()
            },
            numa_nodes=numa_nodes()
            if Path("/sys/devices/system/node").is_dir()
            else None,
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            source=provenance,
            manifest_sha256=sha256(MANIFEST),
            runner_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "compare.py",
                    "fixtures.py",
                    "provenance.py",
                    "prepare_subset.py",
                )
            },
            installed_build=installed_build,
            azint_build=azint_build,
            variant_probes=probes,
            skipped_variants=skipped,
            case_threads={
                variant: {
                    f"{count}x{p}" if p > 1 else count: case_threads(variant, count, p)
                    for count in workers
                    for p in process_counts
                    if p <= count and count % p == 0
                }
                for variant in args.variants
            },
            build_verification_problems=build_problems,
            module_paths=module_paths,
        ),
    )
    (output / "datasets.json").write_bytes(MANIFEST.read_bytes())
    metadata = {}
    for dataset in args.datasets:
        print(f"Preparing {dataset} outside timing...", flush=True)
        metadata[dataset] = prepare(
            output / dataset,
            dataset,
            args.data_dir,
            args.cache_dir,
            args.offline,
            beam_pixel=args.mask_beam_pixel,
            subset_dir=args.subset_dir,
        )
        beam = metadata[dataset].get("beam_pixel_mask")
        if beam:
            print(
                f"{dataset}: masked {beam['mode']} pixel(s) {beam['pixels']} "
                f"(already masked: {beam['already_masked']}; minimum-2theta pixel "
                f"{beam['min_2theta_pixel']})",
                flush=True,
            )
    failures = []
    for trial in range(args.trials):
        variants = variants_run if trial % 2 == 0 else list(reversed(variants_run))
        for dataset in args.datasets:
            beam = metadata[dataset].get("beam_pixel_mask")
            for workload in args.workloads:
                if workload == "frames" and metadata[dataset]["distinct_frames"] < 2:
                    continue
                for count in workers:
                    for processes in process_counts:
                        if processes > count or count % processes:
                            continue
                        for errors in (False, True):
                            for variant in variants:
                                if variant in GPU_VARIANTS and (
                                    count > 1 or processes > 1
                                ):
                                    continue  # one host thread drives the device
                                suffix = f"{dataset}-{workload}-{variant}"
                                mode = "poisson" if errors else "intensity"
                                split = f"-p{processes}" if processes > 1 else ""
                                label = f"{suffix}-w{count}{split}-{mode}-t{trial}"
                                case_path = output / "cases" / f"{label}.json"
                                case = dict(
                                    dataset=dataset,
                                    workload=workload,
                                    variant=variant,
                                    workers=count,
                                    errors=errors,
                                    error_mode=mode,
                                    trial=trial,
                                    fixture=str(output / dataset),
                                    frames=(
                                        (max(args.frames, count) + count - 1) // count
                                    )
                                    * count,
                                    bins=args.bins,
                                    blocks=args.blocks,
                                    seconds=args.seconds,
                                    result=str(case_path.with_suffix(".result.json")),
                                    processes=processes,
                                    workers_per_process=count // processes,
                                    binding=args.bind if processes > 1 else "none",
                                    mask_beam_pixel=args.mask_beam_pixel,
                                    beam_pixels_added=len(beam["pixels"])
                                    - len(beam["already_masked"])
                                    if beam
                                    else 0,
                                    opencl_device=args.opencl_device,
                                )
                                if count > 1 or trial > 0:
                                    case["reference"] = str(
                                        output
                                        / "cases"
                                        / f"{suffix}-w1-{mode}-t0.result.npz"
                                    )
                                case["threads"] = case_threads(
                                    variant, count, processes
                                )
                                dump(case_path, case)
                                env = os.environ | case["threads"]
                                log_path = case_path.with_suffix(".log")
                                if processes > 1:
                                    try:
                                        line = run_processes(
                                            case, case_path, env, bindings[processes]
                                        )
                                        log_path.write_text(line + "\n")
                                        code = 0
                                    except Exception as error:
                                        log_path.write_text(f"{error!r}\n")
                                        code = 1
                                else:
                                    with log_path.open("w") as log:
                                        code = subprocess.run(
                                            [
                                                sys.executable,
                                                "-E",
                                                "-s",
                                                __file__,
                                                "--case",
                                                str(case_path),
                                            ],
                                            env=env,
                                            stdout=log,
                                            stderr=subprocess.STDOUT,
                                        ).returncode
                                if code:
                                    failures.append(dict(case=label, exit_code=code))
                                    print(
                                        f"FAILED: {label}; see {log_path}",
                                        flush=True,
                                    )
                                else:
                                    print(log_path.read_text().strip(), flush=True)
                                summarize(output)
                                dump(output / "failures.json", failures)
    if failures:
        raise SystemExit(
            f"{len(failures)} cases failed; partial results and inputs retained in {output}"
        )
    if not args.keep_inputs:
        for dataset in args.datasets:
            for filename in ("frames.npy", "mask.npy"):
                (output / dataset / filename).unlink()
    print(f"Results: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    # Keep the parent and case processes on the same interpreter paths even if
    # a shell/module exports PYTHONPATH or enables user-installed packages.
    if not sys.flags.ignore_environment or not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, "-E", "-s", *sys.argv])
    main()
