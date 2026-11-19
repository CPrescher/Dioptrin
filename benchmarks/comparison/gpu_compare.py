#!/usr/bin/env python3
"""Compare Dioptrin CUDA and pyFAI OpenCL on the CPU suite's detector fixtures."""

import argparse
import importlib.metadata
from itertools import product
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from fixtures import manifest, prepare, sha256
from hardware import capture
from provenance import source_hashes


ROOT = Path(__file__).resolve().parents[2]
CUDA_TEST = "integration::cuda::bench::paper::paper_native_cuda"
BINS_DEFAULT = 2000


def build_cuda(cargo):
    if not (ROOT / "Cargo.lock").is_file():
        subprocess.run([cargo, "generate-lockfile"], cwd=ROOT, check=True)
    command = [cargo, "test", "--locked", "--release", "--features", "cuda", "--lib",
               "--no-run", "--message-format=json"]
    output = subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8")
    for line in output.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if (row.get("reason") == "compiler-artifact" and row.get("executable")
                    and row["target"]["name"] == "dioptrin" and row["profile"]["test"]):
                return Path(row["executable"])
    raise RuntimeError("Cargo did not provide the CUDA test executable")


def profile_opencl(engine, integrate, frames, repeats):
    """Sum the device execution intervals of the stock pyFAI kernels."""
    phases = {name: [] for name in ("memset_ng", "corrections4a", "csr_integrate4")}
    profile = []
    event_names = set()
    engine.set_profiling(True)
    try:
        for i in range(repeats + 2):
            engine.reset_log()
            for frame in frames:
                integrate(frame)
            engine.queue.finish()
            totals = dict.fromkeys(phases, 0.0)
            count = dict.fromkeys(phases, 0)
            for row in engine.events:
                name = row[0].removesuffix("_single")
                event_names.add(name)
                if not name.startswith("copy "):
                    if len(row) == 3:
                        _, start, end = row
                    else:
                        start, end = row[1].profile.start, row[1].profile.end
                    totals[name] = totals.get(name, 0.0) + (end - start) / 1e9
                    count[name] = count.get(name, 0) + 1
            if any(n != len(frames) for n in count.values()) or any(t <= 0 for t in totals.values()):
                raise RuntimeError(f"Incomplete pyFAI device kernel profile: {count}")
            if i >= 2:
                profile.append(sum(totals.values()))
                for name, total in totals.items():
                    phases.setdefault(name, []).append(total)
    finally:
        engine.reset_log()
        engine.set_profiling(False)
    return profile, phases, sorted(event_names)


def measure_opencl(fixture, meta, frames, bins, errors, repeats, gpu_name):
    started = time.perf_counter()
    import numpy as np
    import pyFAI
    import pyopencl as cl
    from pyFAI.method_registry import IntegrationMethod

    matching = [
        (pi, di, device) for pi, platform in enumerate(cl.get_platforms())
        for di, device in enumerate(platform.get_devices())
        if device.type & cl.device_type.GPU and gpu_name.lower() in device.name.lower()
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one OpenCL GPU matching {gpu_name!r}, found {len(matching)}")
    (pi, di, device), = matching
    target = (pi, di)
    ai = pyFAI.load(str(fixture / "geometry.poni"))
    ai.detector.mask = np.load(fixture / "mask.npy")
    method = next(iter(IntegrationMethod.select_method(
        dim=1, split="full", algo="csr", impl="opencl", target=target, degradable=False,
    )), None)
    if method is None:
        raise RuntimeError(f"Full CSR OpenCL unavailable for {device.name}")

    def integrate(frame, safe=False):
        return ai.integrate1d(
            frame, bins, unit="2th_deg", method=method, radial_range=tuple(meta["radial_range"]),
            correctSolidAngle=True, polarization_factor=0.965,
            error_model="poisson" if errors else None, safe=safe,
        )

    pattern = integrate(frames[0], safe=True)
    setup = time.perf_counter() - started
    implementation = pattern.method.method
    if (implementation.split, implementation.algo, implementation.impl, tuple(implementation.target)) != (
        "full", "csr", "opencl", target
    ) or "opencl" not in str(pattern.compute_engine).lower():
        raise RuntimeError(f"pyFAI used {implementation}, {pattern.compute_engine}")

    batch = None
    timings = []
    for iteration in range(repeats + 2):
        started = time.perf_counter()
        batch = [integrate(frame) for frame in frames]
        seconds = time.perf_counter() - started
        if iteration >= 2:
            timings.append(seconds)
    result = dict(
        radial=np.asarray(pattern.radial),
        intensity=np.stack([p.intensity for p in batch]),
        sigma=np.stack([p.sigma for p in batch]) if errors else np.empty((len(frames), 0)),
    )
    if not all(np.isfinite(values).all() for values in result.values()):
        raise RuntimeError("nonfinite pyFAI OpenCL output")
    np.testing.assert_allclose(
        result["intensity"], np.broadcast_to(pattern.intensity, result["intensity"].shape),
        rtol=1e-6, atol=1e-6,
    )
    if errors:
        np.testing.assert_allclose(
            result["sigma"], np.broadcast_to(pattern.sigma, result["sigma"].shape),
            rtol=1e-6, atol=1e-6,
        )
    host = dict(
        backend="pyfai-opencl", host_memory="pageable", layout="csr", device=device.name,
        driver=device.driver_version,
        engine=str(pattern.compute_engine), setup_first_s=setup, batch_seconds=timings,
    )
    device_seconds, phases, events = profile_opencl(ai.engines[method].engine, integrate, frames, repeats)
    original = frames[0]
    offset = np.arange(original.size, dtype=np.uint32).reshape(original.shape) % 7
    distinct = np.minimum(original, np.iinfo(np.uint32).max - offset) + offset
    alternate = integrate(distinct)
    result["validation_intensity"] = np.stack([pattern.intensity, alternate.intensity])
    if errors:
        result["validation_sigma"] = np.stack([pattern.sigma, alternate.sigma])
    kernels = dict(backend="pyfai-opencl-kernels", layout="csr", device=device.name,
                   driver=device.driver_version, engine=str(pattern.compute_engine),
                   batch_seconds=device_seconds, kernel_seconds=phases, opencl_events=events)
    return [host, kernels], result


def run_opencl_case(path):
    import numpy as np

    case = json.loads(path.read_text(encoding="utf-8"))
    native = np.fromfile(case["pixels"], dtype="<u4").reshape(case["shape"])
    rows, patterns = measure_opencl(
        path.parent, case, [native] * case["frames"], case["bins"],
        case["errors"], case["repeats"], case["gpu_name"],
    )
    np.savez_compressed(case["patterns"], **patterns)
    Path(case["result"]).write_text(json.dumps(rows), encoding="utf-8")


def compare_patterns(cuda, opencl, errors):
    import numpy as np

    for gpu, cpu in zip(cuda["gpu_patterns"], cuda["cpu_patterns"], strict=True):
        np.testing.assert_array_equal(gpu["radial"], cpu["radial"])
    reference = cuda["cpu_patterns"]
    axis = float(np.max(np.abs(np.asarray([row["radial"] for row in reference]) - opencl["radial"])))
    metrics = {"max_axis_deg": axis}
    for quantity in ("intensity", "sigma") if errors else ("intensity",):
        cpu = np.asarray([row[quantity] for row in reference])
        cl = opencl[f"validation_{quantity}"]
        if cpu.shape != cl.shape or not np.isfinite(cl).all():
            raise RuntimeError(f"invalid or incomplete OpenCL validation for {quantity}")
        delta = np.abs(cpu - cl)
        rel = np.max(np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(cpu, axis=1), 1e-300))
        paired = (cpu > 0) & (cl > 0)
        fractions = delta[paired] / cpu[paired]
        metrics[quantity] = dict(relative_l2=float(rel), max_abs=float(delta.max()),
                                 max_positive_bin_relative=float(fractions.max(initial=0)),
                                 p99_positive_bin_relative=float(np.percentile(fractions, 99)) if fractions.size else 0,
                                 zero_status_mismatches=int(np.count_nonzero((cpu == 0) != (cl == 0))))
        # Cross-engine tolerances allow different Poisson floors and preprocessing.
        if rel > 0.01:
            raise RuntimeError(f"{quantity} differs from CPU by {rel:.3%} relative L2")
    if axis > 0.0001:
        raise RuntimeError(f"OpenCL radial axis differs from CPU by {axis} degrees")
    return metrics


def measure_cuda(executable, config_path, config, nvrtc):
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    environment = os.environ | {
        "DIOPTRIN_PAPER_CUDA_CASE": str(config_path), "RAYON_NUM_THREADS": "14",
    }
    if nvrtc:
        search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        environment[search] = str(nvrtc.resolve()) + os.pathsep + environment.get(search, "")
    result = subprocess.run(
        [str(executable), CUDA_TEST, "--exact", "--ignored", "--nocapture"],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    config_path.with_suffix(".log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"CUDA benchmark failed\n{result.stdout}")
    return json.loads(Path(config["result"]).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-exe", type=Path)
    parser.add_argument("--allow-unverified-build", action="store_true",
                        help="Allow a prebuilt CUDA executable and label its source unverified")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--gpu-name", default="NVIDIA")
    parser.add_argument("--datasets", nargs="+", choices=manifest()["datasets"],
                        default=["pilatus", "synthetic-4m", "synthetic-9m"])
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--bins", type=int, default=BINS_DEFAULT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3, help="Independent process launches per backend and layout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--keep-inputs", action="store_true")
    parser.add_argument("--opencl-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.opencl_case:
        run_opencl_case(args.opencl_case)
        return
    if not args.output:
        parser.error("--output is required")
    if args.cuda_exe and not args.allow_unverified_build:
        parser.error("omit --cuda-exe to build this checkout, or explicitly allow an unverified build")
    if min(args.frames, args.bins, args.trials) < 1 or args.repeats < 3:
        parser.error("Use positive frame/bin/trial counts and at least three repeats")
    import numpy as np

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "target" / "paper-gpu-cache"
    os.environ.setdefault("PYOPENCL_CACHE_DIR", str(cache / "opencl"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    executable = args.cuda_exe.resolve() if args.cuda_exe else build_cuda(args.cargo)
    record = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True,
        ).strip(),
        cuda_executable=str(executable), cuda_sha256=sha256(executable),
        cuda_built_from_checkout=args.cuda_exe is None,
        cuda_source_sha256=source_hashes(ROOT) if args.cuda_exe is None else None,
        cargo_lock_sha256=sha256(ROOT / "Cargo.lock"), rustflags=os.environ.get("RUSTFLAGS"),
        harness_sha256=sha256(Path(__file__)),
        hardware=capture(), python=sys.version,
        pyfai=importlib.metadata.version("pyFAI"), pyopencl=importlib.metadata.version("pyopencl"),
        input="Prepared uint32 frames with separate host-to-pattern and device kernel timings",
        bins=args.bins, frames=args.frames, repeats=args.repeats, trials=args.trials, records=[],
    )
    for dataset in args.datasets:
        fixture = output / dataset
        meta = prepare(fixture, dataset, args.data_dir, output / "fixture-cache", offline=True)
        image = np.load(fixture / "frames.npy", mmap_mode="r")[0]
        native = image.astype("<u4")
        if not np.array_equal(image, native):
            raise ValueError(f"{dataset} pixels cannot be represented exactly as uint32")
        del image
        pixels = fixture / "pixels.u32"
        mask = fixture / "mask.u8"
        native.tofile(pixels)
        np.load(fixture / "mask.npy").tofile(mask)
        for errors, trial in product((False, True), range(args.trials)):
            mode = f"{'poisson' if errors else 'intensity'}-t{trial}"
            print(f"{dataset} {mode}", flush=True)
            case = dict(
                shape=meta["shape"], radial_range=meta["radial_range"],
                pixels=str(pixels), mask=str(mask), poni=str(fixture / "geometry.poni"),
                frames=args.frames, bins=args.bins, errors=errors, repeats=args.repeats,
            )
            opencl_case = fixture / f"opencl-{mode}-case.json"
            opencl_patterns = fixture / f"opencl-{mode}.npz"
            opencl_result = fixture / f"opencl-{mode}.json"
            opencl_case.write_text(json.dumps(case | dict(gpu_name=args.gpu_name,
                                   result=str(opencl_result), patterns=str(opencl_patterns))), encoding="utf-8")
            log_path = opencl_case.with_suffix(".log")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, "-E", "-s", "-B", __file__, "--opencl-case", str(opencl_case)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                raise RuntimeError(f"OpenCL benchmark failed. See {log_path}")
            records = json.loads(opencl_result.read_text(encoding="utf-8"))
            with np.load(opencl_patterns) as patterns:
                for layout in (("block", "warp") if trial % 2 == 0 else ("warp", "block")):
                    result = fixture / f"cuda-{mode}-{layout}.json"
                    cuda = measure_cuda(executable, fixture / f"config-{mode}-{layout}.json",
                                        case | dict(layout=layout, result=str(result)), args.nvrtc)
                    metrics = compare_patterns(cuda, patterns, errors)
                    records.extend(cuda["records"])
                    record.setdefault("numerical", []).append(dict(dataset=dataset, errors=errors,
                                                                 layout=layout, trial=trial, **metrics))
            for row in records:
                if args.gpu_name.lower() not in row["device"].lower():
                    raise RuntimeError(f"Benchmark selected {row['device']!r}")
                row.update(dataset=dataset, shape=meta["shape"], errors=errors, trial=trial,
                           frames_per_s=args.frames / statistics.median(row["batch_seconds"]))
                print(f"  {row['backend']} {row['layout']}: {row['frames_per_s']:.2f} frames/s", flush=True)
                record["records"].append(row)
            (output / "gpu-results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if not args.keep_inputs:
            for path in (pixels, mask, fixture / "frames.npy", fixture / "mask.npy"):
                path.unlink()


if __name__ == "__main__":
    main()
