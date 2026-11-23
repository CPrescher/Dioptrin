"""Time the whole 64-frame Eiger TIFF batch with the normal filesystem cache."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, command, cpu_model, gpu_snapshot
from startup import build, input_hashes

TEST = "integration::cuda::bench::file_batch::eiger_files"
BACKENDS = ("cpu", "cuda_pageable", "cuda_pinned", "cuda_native", "cuda_native_direct")
METHODS = ("counting", "pixel_split", "pyfai_full")
FRAMES = 64
BATCH_MARKER = "DIOPTRIN_FILE_BATCH "
OUTPUT_MARKER = "DIOPTRIN_FILE_OUTPUT "


def prepare_files(source, digest):
    directory = ROOT / "target" / "cuda-disk-fixtures" / digest
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(FRAMES):
        path = directory / f"frame-{index:02}.tiff"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            temporary = path.with_suffix(".part")
            shutil.copyfile(source, temporary)
            temporary.replace(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Fixture copy differs from {source}")
    return directory


def trial(executable, env):
    args = [executable, TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    start = time.perf_counter_ns()
    with subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        messages, batches, pattern = [], {}, None
        for line in process.stdout:
            _, marker, payload = line.partition(BATCH_MARKER)
            if marker:
                elapsed = (time.perf_counter_ns() - start) / 1e6
                measurement = json.loads(payload)
                stage = measurement.pop("stage")
                if stage == "fresh":
                    measurement["process_to_patterns_ms"] = elapsed
                batches[stage] = measurement
            elif OUTPUT_MARKER in line:
                pattern = json.loads(line.partition(OUTPUT_MARKER)[2])
            else:
                messages.append(line)
        if process.wait() or set(batches) != {"fresh", "prepared"} or pattern is None:
            raise RuntimeError("File-batch trial failed\n" + "".join(messages))
    return batches, pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eiger-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--direct-io", action="store_true", help="Bypass the Windows filesystem data cache")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument("--layout", choices=("block", "warp"), default="block")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least three fresh trials per backend")
    if args.direct_io and os.name != "nt":
        parser.error("--direct-io requires Windows")
    eiger_dir = args.eiger_dir.resolve()
    source = eiger_dir / "eiger_0000.tiff"
    env = os.environ.copy()
    env.pop("CUDA_CACHE_DISABLE", None)
    env.pop("DIOPTRIN_CUDA_DISK_DIRECT_IO", None)
    if args.direct_io:
        env["DIOPTRIN_CUDA_DISK_DIRECT_IO"] = "1"
    if args.nvrtc:
        search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[search] = os.pathsep.join(filter(None, [str(args.nvrtc.resolve()), env.get(search)]))
    executable = build(args.cargo, env)
    hashes = input_hashes(["eiger"], eiger_dir)
    copies = prepare_files(source, hashes["eiger_0000.tiff"])
    env.update(
        DIOPTRIN_CUDA_EIGER_DIR=str(eiger_dir),
        DIOPTRIN_CUDA_DISK_DIR=str(copies),
        DIOPTRIN_CUDA_DISK_LAYOUT=args.layout,
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "src/integration/cuda.rs",
               ROOT / "src/integration/cuda/kernels.cu",
               ROOT / "src/integration/cuda/bench/file_batch.rs",
               ROOT / "src/frame/tiff_input.rs"]
    data = {
        "metadata": {
            "commit": command("git", "rev-parse", "HEAD"),
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "cpu": cpu_model(),
            "platform": platform.platform(),
            "gpu_before": gpu_snapshot(),
            "frames": FRAMES,
            "tiff_bytes_per_batch": FRAMES * source.stat().st_size,
            "fixture_sha256": hashes,
            "filesystem_cache": ("Windows unbuffered data reads" if args.direct_io
                                 else "normal filesystem cache, fixture copies read before timing"),
            "repeats": args.repeats,
            "cuda_layout": args.layout,
        },
        "trials": [],
    }
    rng = random.Random(20261118)
    for method in args.methods:
        reference = None
        for number in range(1, args.repeats + 1):
            order = list(args.backends)
            rng.shuffle(order)
            for backend in order:
                print(f"{method} {backend} trial {number}", flush=True)
                phases, pattern = trial(executable, dict(
                    env,
                    DIOPTRIN_CUDA_DISK_BACKEND=backend,
                    DIOPTRIN_CUDA_DISK_METHOD=method,
                ))
                if reference is None:
                    reference = pattern
                if len(pattern) != len(reference) or not all(
                    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                    for a, b in zip(pattern, reference)
                ):
                    raise RuntimeError(f"{method} output differs on {backend}")
                data["trials"].append(dict(
                    phases, backend=backend, method=method, trial=number,
                ))
                output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    data["metadata"]["gpu_after"] = gpu_snapshot()
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
