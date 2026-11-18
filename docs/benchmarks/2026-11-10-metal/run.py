#!/usr/bin/env python3
"""Reproduce the isolated Metal experiment on an Apple silicon Mac.

Only the Python standard library is needed; Rust, Swift, and Metal are required.
Large temporary fixtures are deleted after the run. No detector binaries are
added to the repository, and no production library code is modified.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=REPO, check=True, **kwargs)


def capture(*args):
    return command(args, capture_output=True, text=True).stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def accuracy(actual, reference, std=None):
    assert len(actual) == len(reference)
    absolute, relative, residual_std, failures = [], [], [], []
    for i, (a, r) in enumerate(zip(actual, reference)):
        delta = abs(a - r)
        finite = math.isfinite(a) and math.isfinite(r)
        # Tight enough to catch implementation errors while allowing FP32 sums.
        allowed = 1e-5 + 1e-5 * abs(r)
        if not finite or delta > allowed:
            failures.append({"bin": i, "gpu": a, "cpu": r, "abs_error": delta})
        absolute.append(delta)
        if r != 0:
            relative.append(delta / abs(r))
        if std is not None and std[i] > 0:
            residual_std.append(delta / std[i])
    return {
        "bins": len(reference), "passed": not failures,
        "tolerance": "abs_error <= 1e-5 + 1e-5 * abs(reference)",
        "max_abs_error": max(absolute, default=0),
        "max_relative_error_nonzero": max(relative, default=0),
        "rms_error": math.sqrt(statistics.mean(d*d for d in absolute)),
        "max_intensity_error_in_poisson_std": max(residual_std, default=0) if std is not None else None,
        "failed_bins": failures,
    }


def best_cpu(case, errors, layout=None):
    candidates = [x for x in case["cpu"] if x["errors"] == errors and
                  (layout is None or x["layout"] == layout)]
    return min(candidates, key=lambda x: x["timing"]["median_ms"])


def best_gpu(case, errors, mode, compensated=False):
    return min((x for x in case["metal"]["results"] if x["errors"] == errors and x["input"] == mode
                and x["compensated"] == compensated),
               key=lambda x: x["wall"]["median_ms"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=51)
    p.add_argument("--cpu-samples", type=int, default=31)
    p.add_argument("--output", type=Path, default=HERE / "results.json")
    p.add_argument("--small-only", action="store_true", help="skip the local Eiger fixture")
    args = p.parse_args()
    if args.samples <= 0 or args.cpu_samples <= 0:
        p.error("sample counts must be positive")
    command(["cargo", "build", "--release", "--example", "benchmark_metal_fixture"])
    cases = [
        ("pilatus_counting", "test_data/CeO2_Pilatus1M.tif", "test_data/v1.poni", "counting", "-"),
        ("pilatus_pixel_split", "test_data/CeO2_Pilatus1M.tif", "test_data/v1.poni", "pixel_split", "-"),
    ]
    if not args.small_only:
        cases += [
            ("eiger_counting", "test_data/eiger_0000.tiff", "docs/benchmarks/2026-11-05-bslz4/calibration.poni", "counting", "test_data/eiger_0000.mask"),
            ("eiger_pixel_split", "test_data/eiger_0000.tiff", "docs/benchmarks/2026-11-05-bslz4/calibration.poni", "pixel_split", "test_data/eiger_0000.mask"),
        ]
    provenance = {"commit": capture("git", "rev-parse", "HEAD"),
                  "git_status": capture("git", "status", "--short"),
                  "macos": capture("sw_vers"), "architecture": platform.machine(),
                  "cpu": capture("sysctl", "-n", "machdep.cpu.brand_string"),
                  "rust": capture("rustc", "--version"), "swift": capture("swift", "--version"),
                  "source_sha256": {str(f.relative_to(REPO)): digest(f) for f in
                      [REPO / "examples/benchmark_metal_fixture.rs", HERE / "MetalBench.swift", HERE / "integrate.metal", Path(__file__)]},
                  "fixture_sha256": {path: digest(REPO / path) for path in
                      sorted({path for _, image, poni, _, mask in cases for path in [image, poni, mask] if path != "-"})}}
    output = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "provenance": provenance, "cases": []}
    with tempfile.TemporaryDirectory(prefix="dioptrin-metal-") as scratch:
        root = Path(scratch)
        executable = root / "MetalBench"
        command(["swiftc", "-O", HERE / "MetalBench.swift", "-o", executable])
        for name, image, poni, method, mask in cases + [("edge", None, None, None, None)]:
            print(f"Benchmarking {name}", flush=True)
            fixture = root / name
            cmd = [REPO / "target/release/examples/benchmark_metal_fixture", fixture]
            cmd += ["edge"] if name == "edge" else [image, poni, method, "3000", mask, args.cpu_samples]
            command(cmd)
            gpu_file = fixture / "metal.json"
            command([executable, fixture, HERE / "integrate.metal", gpu_file, args.samples])
            case = json.loads((fixture / "fixture.json").read_text())
            metal = json.loads(gpu_file.read_text())
            for run in metal["results"]:
                y = run.pop("intensity")
                run["intensity_accuracy"] = accuracy(y, case["reference_intensity"], case["reference_std"])
                if run["errors"]:
                    run["std_accuracy"] = accuracy(run.pop("std"), case["reference_std"])
                # Preserve the small independent boundary fixture's actual output.
                if name == "edge":
                    run["intensity"] = y
                if name != "edge" and (not run["intensity_accuracy"]["passed"] or
                                       (run["errors"] and not run["std_accuracy"]["passed"])):
                    raise RuntimeError(f"accuracy failed: {name}, {run}")
                if name != "edge" and run["compensated"]:
                    assert run["intensity_accuracy"]["max_intensity_error_in_poisson_std"] < 0.01
            case["name"] = name
            case["metal"] = metal
            if name == "edge":
                # Conversion must demonstrably lose the unit increment above 2^24.
                assert all(any(f["bin"] == 6 for f in r["intensity_accuracy"]["failed_bins"])
                           for r in metal["results"])
                # Empty, fully masked/NaN, fractional, negative, zero rows must pass.
                assert all(not any(f["bin"] in [0, 1, 2, 3, 7] for f in r["intensity_accuracy"]["failed_bins"])
                           for r in metal["results"])
                assert all(not any(f["bin"] == 5 for f in r["intensity_accuracy"]["failed_bins"])
                           for r in metal["results"] if r["compensated"])
            else:
                for errors in [False, True]:
                    cpu = best_cpu(case, errors)
                    gpu = best_gpu(case, errors, "f64_convert", compensated=True)
                    print(f"  errors={errors}: CPU {cpu['timing']['median_ms']:.3f} ms "
                          f"({cpu['layout']}/{cpu['threads']} threads), "
                          f"Metal compensated/f64 input {gpu['wall']['median_ms']:.3f} ms, "
                          f"speedup {cpu['timing']['median_ms']/gpu['wall']['median_ms']:.2f}x", flush=True)
            # Full reference arrays retained once, rather than once per timing variant.
            output["cases"].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            # Release the large binary fixture before constructing the next case.
            import shutil
            shutil.rmtree(fixture)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
