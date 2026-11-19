#!/usr/bin/env python3
"""Compare Dioptrin CUDA and pyFAI OpenCL on the CPU suite's detector fixtures."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from fixtures import manifest, prepare, sha256
from hardware import capture


ROOT = Path(__file__).resolve().parents[2]
CUDA_TEST = "integration::cuda::bench::paper::paper_native_cuda"
BINS_DEFAULT = 2000


def build_cuda(cargo):
    command = [cargo, "test", "--release", "--features", "cuda", "--lib",
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
                if name in phases:
                    if len(row) == 3:
                        _, start, end = row
                    else:
                        start, end = row[1].profile.start, row[1].profile.end
                    totals[name] += (end - start) / 1e9
                    count[name] += 1
            if any(n != len(frames) for n in count.values()) or any(t <= 0 for t in totals.values()):
                raise RuntimeError(f"Incomplete pyFAI device kernel profile: {count}")
            if i >= 2:
                profile.append(sum(totals.values()))
                for name in phases:
                    phases[name].append(totals[name])
    finally:
        engine.reset_log()
        engine.set_profiling(False)
    return profile, phases


def measure_opencl(fixture, meta, frames, bins, errors, repeats, gpu_name):
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
    started = time.perf_counter()
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
    device_seconds, phases = profile_opencl(ai.engines[method].engine, integrate, frames, repeats)
    kernels = dict(backend="pyfai-opencl-kernels", layout="csr", device=device.name,
                   driver=device.driver_version, engine=str(pattern.compute_engine),
                   batch_seconds=device_seconds, kernel_seconds=phases)
    return [host, kernels], result


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
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--gpu-name", default="NVIDIA")
    parser.add_argument("--datasets", nargs="+", choices=manifest()["datasets"],
                        default=["pilatus", "synthetic-4m", "synthetic-9m"])
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--bins", type=int, default=BINS_DEFAULT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--keep-inputs", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.bins < 1 or args.repeats < 3:
        parser.error("Use positive frame/bin counts and at least three repeats")
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
        harness_sha256=sha256(Path(__file__)),
        hardware=capture(), python=sys.version,
        pyfai=importlib.metadata.version("pyFAI"), pyopencl=importlib.metadata.version("pyopencl"),
        input="Prepared uint32 frames with separate host-to-pattern and device kernel timings",
        bins=args.bins, frames=args.frames, repeats=args.repeats, records=[],
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
        for errors in (False, True):
            mode = "poisson" if errors else "intensity"
            print(f"{dataset} {mode}", flush=True)
            config = dict(
                shape=meta["shape"], radial_range=meta["radial_range"],
                pixels=str(pixels), mask=str(mask), poni=str(fixture / "geometry.poni"),
                frames=args.frames, bins=args.bins, errors=errors, repeats=args.repeats,
                result=str(fixture / f"cuda-{mode}.json"),
            )
            cuda = measure_cuda(executable, fixture / f"config-{mode}.json", config, args.nvrtc)
            pyfai, patterns = measure_opencl(
                fixture, meta, [native] * args.frames, args.bins, errors, args.repeats, args.gpu_name,
            )
            for row in [*cuda["records"], *pyfai]:
                if args.gpu_name.lower() not in row["device"].lower():
                    raise RuntimeError(f"Benchmark selected {row['device']!r}")
                row.update(dataset=dataset, shape=meta["shape"], errors=errors,
                           frames_per_s=args.frames / statistics.median(row["batch_seconds"]))
                print(f"  {row['backend']} {row['layout']}: {row['frames_per_s']:.2f} frames/s", flush=True)
                record["records"].append(row)
            signal = np.asarray(cuda["intensity"])
            sigma = np.asarray(cuda["sigma"])
            np.savez_compressed(fixture / f"opencl-{mode}.npz", **patterns)
            record.setdefault("numerical", []).append(dict(
                dataset=dataset, errors=errors,
                max_axis_deg=float(np.max(np.abs(np.asarray(cuda["radial"]) - patterns["radial"]))),
                intensity_relative_l2=float(
                    np.linalg.norm(signal - patterns["intensity"][0]) / max(np.linalg.norm(signal), 1e-300)
                ),
                sigma_relative_l2=float(
                    np.linalg.norm(sigma - patterns["sigma"][0]) / max(np.linalg.norm(sigma), 1e-300)
                ) if errors else None,
            ))
            (output / "gpu-results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if not args.keep_inputs:
            for path in (pixels, mask, fixture / "frames.npy", fixture / "mask.npy"):
                path.unlink()


if __name__ == "__main__":
    main()
