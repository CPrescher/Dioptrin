"""Build and measure the native Dioptrin CUDA prototype. Requires Python 3."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
TEST = "integration::cuda::bench::benchmark_cuda"
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


def prep_eiger(path):
    # The benchmark TIFF and mask must use the same row orientation. The
    # existing Rust TIFF reader flips rows, so apply the same flip to the mask.
    import fabio
    import numpy as np

    image = fabio.open(str(ROOT / "test_data/eiger_0000.mask")).data
    path.write_bytes(np.ascontiguousarray(image[::-1] != 0, dtype=np.uint8).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path, help="Directory with the NVRTC shared libraries")
    parser.add_argument("--cases", default="256,pilatus,1024,2048")
    parser.add_argument("--batches", default="1,8,64,256,max")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host-bytes", type=int, default=15_000_000_000)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--rustc", default="rustc")
    args = parser.parse_args()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_suffix(".jsonl")
    env = os.environ.copy()
    env.update(
        DIOPTRIN_CUDA_OUT=str(raw),
        DIOPTRIN_CUDA_CASES=args.cases,
        DIOPTRIN_CUDA_BATCHES=args.batches,
        DIOPTRIN_CUDA_REPEATS=str(args.repeats),
        DIOPTRIN_CUDA_HOST_BYTES=str(args.host_bytes),
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
        "cases": args.cases,
        "batches": args.batches,
        "gpu_before": gpu_snapshot(),
    }
    with tempfile.TemporaryDirectory(prefix="dioptrin-cuda-") as temp:
        if "eiger" in args.cases.split(","):
            mask = Path(temp) / "eiger-mask.u8"
            prep_eiger(mask)
            env["DIOPTRIN_CUDA_EIGER_MASK_U8"] = str(mask)
        run = [args.cargo, "test", "--release", "--features", "cuda", "--lib", TEST,
               "--", "--exact", "--ignored", "--nocapture"]
        subprocess.run(run, cwd=ROOT, env=env, check=True)
    metadata["gpu_after"] = gpu_snapshot()
    with raw.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
