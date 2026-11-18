"""Measure fresh-process time to the first f64-input Poisson pattern."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from run import ROOT, command, cpu_model, gpu_snapshot

TEST = "integration::cuda::bench::startup::first_pattern"
MARKER = "DIOPTRIN_FIRST_PATTERN "


def build(cargo, env):
    args = [
        cargo, "test", "--release", "--features", "cuda", "--lib",
        "--no-run", "--message-format=json",
    ]
    messages = subprocess.check_output(args, cwd=ROOT, env=env, text=True, encoding="utf-8")
    for line in messages.splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if (
            row.get("reason") == "compiler-artifact"
            and row["target"]["name"] == "dioptrin"
            and row["profile"]["test"]
            and row["executable"]
        ):
            return row["executable"]
    raise RuntimeError("Cargo did not report the CUDA test executable")


def first_pattern(executable, env):
    args = [executable, TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    start = time.perf_counter_ns()
    with subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        lines = []
        phases = None
        for line in process.stdout:
            _, marker, payload = line.partition(MARKER)
            if marker:
                total_ms = (time.perf_counter_ns() - start) / 1e6
                phases = json.loads(payload)
                phases["process_to_first_pattern_ms"] = total_ms
            else:
                lines.append(line)
        if process.wait() or phases is None:
            raise RuntimeError("First-pattern trial failed\n" + "".join(lines))
    return phases


def select(rows, case, method):
    group = [
        row for row in rows
        if row.get("case") == case and row.get("method") == method
        and row.get("batch") == "1" and row.get("errors")
    ]
    cpu = min(
        (r for r in group if r["type"] == "cpu"),
        key=lambda r: statistics.median(r["samples_ms_per_batch"]),
    )
    gpu = min(
        (r for r in group if r["type"] == "gpu" and r["mode"] == "compensated"),
        key=lambda r: statistics.median(s["host_f64_ms"] for s in r["samples_ms_per_batch"]),
    )
    return cpu, gpu


def input_hashes(cases, eiger_dir):
    files = {}
    if "pilatus" in cases:
        for name in ("test_data/CeO2_Pilatus1M.tif", "test_data/v1.poni"):
            files[name] = ROOT / name
    if "eiger" in cases or "eiger-synthetic" in cases:
        for name in ("eiger_0000.mask", "eiger_0000.poni"):
            files[name] = eiger_dir / name
        if "eiger" in cases:
            files["eiger_0000.tiff"] = eiger_dir / "eiger_0000.tiff"
        calibration = "docs/benchmarks/2026-11-05-bslz4/calibration.poni"
        files[calibration] = ROOT / calibration
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steady-results", nargs="+", type=Path, required=True)
    parser.add_argument("--eiger-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--cases", default="pilatus,eiger")
    parser.add_argument("--methods", default="counting,pixel_split,pyfai_full")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least three fresh processes per configuration")

    env = os.environ.copy()
    env.pop("CUDA_CACHE_DISABLE", None)
    eiger_dir = args.eiger_dir.resolve()
    env["DIOPTRIN_CUDA_EIGER_DIR"] = str(eiger_dir)
    if args.nvrtc:
        search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[search] = os.pathsep.join(filter(None, [str(args.nvrtc.resolve()), env.get(search)]))

    exe = build(args.cargo, env)
    rows = [
        row for path in args.steady_results
        for row in json.loads(path.read_text(encoding="utf-8"))["records"]
    ]
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = args.cases.split(",")
    data = {
        "metadata": {
            "commit": command("git", "rev-parse", "HEAD"),
            "cpu": cpu_model(),
            "gpu_before": gpu_snapshot(),
            "repeats": args.repeats,
            "fixture_sha256": input_hashes(cases, eiger_dir),
        },
        "trials": [],
    }
    random_order = random.Random(20261118)
    for case in cases:
        for method in args.methods.split(","):
            print(f"{case} {method}", flush=True)
            cpu, gpu = select(rows, case, method)
            run_env = dict(
                env,
                DIOPTRIN_CUDA_STARTUP_CASE=case,
                DIOPTRIN_CUDA_STARTUP_METHOD=method,
            )
            cpu_env = dict(
                run_env,
                DIOPTRIN_CUDA_STARTUP_BACKEND=cpu["layout"],
                DIOPTRIN_CUDA_STARTUP_THREADS=str(cpu["threads"]),
            )
            cuda_env = dict(
                run_env,
                DIOPTRIN_CUDA_STARTUP_BACKEND="cuda",
                DIOPTRIN_CUDA_STARTUP_LAYOUT=gpu["layout"],
                DIOPTRIN_CUDA_STARTUP_MEMORY=gpu["memory"],
            )
            # Prime the driver JIT cache and filesystem cache before trials.
            first_pattern(exe, cuda_env)
            configurations = [
                ("cpu", cpu_env, {"layout": cpu["layout"], "threads": cpu["threads"]}),
                ("cuda_cached", cuda_env, {"layout": gpu["layout"], "memory": gpu["memory"]}),
                (
                    "cuda_uncached", dict(cuda_env, CUDA_CACHE_DISABLE="1"),
                    {"layout": gpu["layout"], "memory": gpu["memory"]},
                ),
            ]
            for number in range(1, args.repeats + 1):
                random_order.shuffle(configurations)
                for backend, setup, options in configurations:
                    phases = first_pattern(exe, setup)
                    data["trials"].append(dict(
                        phases, case=case, method=method, backend=backend, trial=number, **options,
                    ))
            output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    data["metadata"]["gpu_after"] = gpu_snapshot()
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
