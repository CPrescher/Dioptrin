"""Build and measure the native Dioptrin CUDA prototype. Requires Python 3."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[3]
TEST = "integration::cuda::bench::benchmark_cuda"
CPU_TEST = "integration::cuda::bench::headroom::cpu_headroom"
INPUT_TEST = "integration::cuda::bench::inputs::cpu_inputs"
TRANSFER_TEST = "integration::cuda::bench::transfers::gpu_transfers"
GPU_FIELDS = (
    "name,driver_version,memory.total,memory.free,temperature.gpu,"
    "clocks.sm,clocks.mem,utilization.gpu,pstate"
)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8").strip()


def cpu_model():
    if os.name == "nt":
        import winreg

        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def gpu_snapshot():
    return command("nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv")


def available_host_bytes():
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total", ctypes.c_uint64),
                ("available", ctypes.c_uint64),
                ("other", ctypes.c_uint64 * 5),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.available
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError("Could not query available host memory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path, help="Directory with the NVRTC shared libraries")
    parser.add_argument("--cases", default="pilatus")
    parser.add_argument("--eiger-dir", type=Path, default=ROOT / "test_data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true", help="Compare CPU worker counts")
    mode.add_argument("--cpu-inputs", action="store_true", help="Compare f32 and f64 CPU inputs and arithmetic")
    mode.add_argument("--gpu-transfers", action="store_true", help="Compare CUDA conversion and transfer paths")
    parser.add_argument("--batches")
    parser.add_argument("--methods", default="counting,pixel_split,pyfai_full")
    parser.add_argument("--layouts", choices=("block,warp", "block", "warp"), default="block,warp")
    parser.add_argument("--cpu-threads")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host-bytes", type=int, default=15_000_000_000)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--rustc", default="rustc")
    args = parser.parse_args()
    cpu_only = args.cpu_only or args.cpu_inputs
    test, default_batches, default_threads = TEST, "1,8,64,256,max", "1,4,8,16"
    if args.cpu_inputs:
        test, default_batches, default_threads = INPUT_TEST, "1,64", "14"
    elif args.cpu_only:
        test, default_batches, default_threads = CPU_TEST, "1,64", "14,16"
    elif args.gpu_transfers:
        test, default_batches = TRANSFER_TEST, "1,8,64"
    batches = args.batches or default_batches
    cpu_threads = args.cpu_threads or default_threads
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_suffix(".jsonl")
    raw.write_text("", encoding="utf-8")
    host_available = available_host_bytes()
    host_limit = min(args.host_bytes, 15_000_000_000, host_available - 2 * 1024**3)
    if host_limit <= 0:
        parser.error("need at least 2 GiB of spare host memory")
    env = os.environ.copy()
    env.update(
        DIOPTRIN_CUDA_OUT=str(raw),
        DIOPTRIN_CUDA_CASES=args.cases,
        DIOPTRIN_CUDA_BATCHES=batches,
        DIOPTRIN_CUDA_METHODS=args.methods,
        DIOPTRIN_CUDA_LAYOUTS=args.layouts,
        DIOPTRIN_CUDA_CPU_THREADS=cpu_threads,
        DIOPTRIN_CUDA_EIGER_DIR=str(args.eiger_dir.resolve()),
        DIOPTRIN_CUDA_REPEATS=str(args.repeats),
        DIOPTRIN_CUDA_HOST_BYTES=str(host_limit),
    )
    if args.nvrtc:
        search = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[search] = os.pathsep.join(
            filter(None, [str(args.nvrtc.resolve()), env.get(search)])
        )
    metadata = {
        "commit": command("git", "rev-parse", "HEAD"),
        "tracked_changes": command("git", "status", "--porcelain", "--untracked-files=no"),
        "rustc": command(args.rustc, "--version", "--verbose"),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cores": os.cpu_count(),
        "host_bytes_available_before": host_available,
        "cases": args.cases,
        "batches": batches,
        "methods": args.methods,
        "layouts": args.layouts,
        "cpu_threads": cpu_threads,
        "cpu_only": cpu_only,
        "compare_cpu_inputs": args.cpu_inputs,
        "compare_gpu_transfers": args.gpu_transfers,
        "gpu_before": gpu_snapshot(),
    }
    sources = [Path(__file__), ROOT / "src/integration/cuda.rs", *(ROOT / "src/integration/cuda").rglob("*")]
    if args.cpu_inputs:
        sources.append(ROOT / "src/lut/csc.rs")
    metadata["source_sha256"] = {
        str(path.resolve().relative_to(ROOT)).replace(os.sep, "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
        if path.is_file()
    }
    run = [args.cargo, "test", "--release", "--features", "cuda", "--lib", test,
           "--", "--exact", "--ignored", "--nocapture"]
    completed = subprocess.run(run, cwd=ROOT, env=env)
    metadata["exit_status"] = completed.returncode
    metadata["gpu_after"] = gpu_snapshot()
    with raw.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n",
                      encoding="utf-8")
    completed.check_returncode()


if __name__ == "__main__":
    main()
