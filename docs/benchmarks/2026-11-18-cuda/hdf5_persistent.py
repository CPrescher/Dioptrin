"""Compare fresh and retained shared CUDA plans on two reads of an Eiger scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from hdf5_batch import MARKER, PATTERN, cache_observation, compare_output, digest_file, disk_delta, disk_reads, peak_process_rss, read_file, scan_info, storage_for, trial_input
from prepare_io_benchmarks import executable_provenance
from run import ROOT, command, cpu_model, gpu_snapshot


TEST = "integration::cuda::bench::hdf5::persistent::eiger_hdf5_persistent"


def run_requests(executable, env, path, prime, workspace, frames, plan_key, cache, compressed):
    import numpy as np

    scans = [(path, frames)] if prime is None else [(prime, 1), (path, frames)]
    outputs = [workspace / f"patterns-{i}.f64" for i in range(len(scans))]
    requests = [json.dumps(dict(source=str(scan), output=str(out), plan_key=plan_key, frames=count)) + "\n"
                for (scan, count), out in zip(scans, outputs)]
    storage = storage_for(path)
    before = disk_reads()
    started = perf_counter_ns()
    args = [str(executable), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    with subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
        rows, patterns, log = [], [], []
        try:
            for i, (request, output) in enumerate(zip(requests, outputs)):
                if i:
                    if cache == "warm":
                        read_file(path)
                    before = disk_reads()
                    started = perf_counter_ns()
                proc.stdin.write(request)
                proc.stdin.flush()
                for line in proc.stdout:
                    _, marker, payload = line.partition(MARKER)
                    if marker:
                        row = json.loads(payload)
                        break
                    log.append(line)
                else:
                    raise RuntimeError("Shared CUDA worker stopped before returning patterns\n" + "".join(log))
                collected = np.fromfile(output, dtype="<f8")
                elapsed_ms = (perf_counter_ns() - started) / 1e6
                if row["iteration"] != i or row["pid"] != proc.pid or collected.size != scans[i][1] * PATTERN:
                    raise RuntimeError("Shared CUDA worker returned the wrong scan or frame count")
                row.update(request_to_patterns_ms=elapsed_ms, plan="retained" if i else "fresh",
                           **cache_observation(storage, disk_delta(before), cache, compressed))
                row["peak_worker_rss_bytes"] = peak_process_rss(proc.pid)
                if i and row["prepare_ms"] != 0.0:
                    raise RuntimeError("Shared CUDA worker prepared an additional plan")
                rows.append(row)
                patterns.append(collected)
            proc.stdin.close()
            log.append(proc.stdout.read())
            if proc.wait():
                raise RuntimeError("Shared CUDA worker failed\n" + "".join(log))
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
    return rows[-1], patterns[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--benchmark-exe", required=True, type=Path, help="Isolated dioptrin-io benchmark executable")
    parser.add_argument("--allow-unverified-build", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=("cuda", "cpu_csr"), default=["cuda"])
    parser.add_argument("--extra-mask", type=Path)
    parser.add_argument("--reference", type=Path, help="Optional frame-major f64 intensity/sigma pairs from the CPU")
    parser.add_argument("--dataset", default="/entry_0000/ESRF-ID11/eiger/data")
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[4])
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--layout", choices=("block", "warp"), default="block")
    parser.add_argument("--cache", choices=("warm", "cold"), default="cold")
    parser.add_argument("--frames", type=int, default=0, help="Use 0 for the full scan")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if (args.trials < 1 or any(not 1 <= n <= 14 for n in args.threads)
            or any(not 1 <= n <= 64 for n in args.tile_sizes)):
        parser.error("use positive trial counts, 1 to 14 decoder threads and tile sizes from 1 to 64")
    if args.cache == "cold" and os.name != "nt":
        parser.error("the cold copy protocol requires Windows")

    import hdf5plugin  # noqa: F401  # Allow discovering the filtered scan with h5py.
    import numpy as np

    frames, shape, geometry, compressed, filters = scan_info(args.hdf5, args.metadata, args.dataset, args.frames)
    executable = args.benchmark_exe.resolve(strict=True)
    build = executable_provenance(executable, args.allow_unverified_build)
    env = os.environ | dict(
        DIOPTRIN_HDF5_DATASET=args.dataset, DIOPTRIN_HDF5_GEOMETRY=json.dumps(geometry),
        DIOPTRIN_HDF5_FRAMES=str(frames), DIOPTRIN_HDF5_LAYOUT=args.layout,
    )
    env.pop("DIOPTRIN_HDF5_EXTRA_MASK", None)
    if args.extra_mask:
        env["DIOPTRIN_HDF5_EXTRA_MASK"] = str(args.extra_mask.resolve(strict=True))
    if args.nvrtc:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(filter(None, (str(args.nvrtc.resolve(strict=True)), env.get(var))))
    scan_digest = digest_file(args.hdf5)
    extra_mask = digest_file(args.extra_mask) if args.extra_mask else None
    plan_key = hashlib.sha256(json.dumps(dict(
        geometry=geometry, shape=shape[1:], extra_mask_sha256=extra_mask,
        method="pyfai_full", unit="2th_deg", bins=PATTERN // 2,
        solid_angle=True, errors="poisson", precision="f64",
    ), sort_keys=True).encode()).hexdigest()
    root = Path(__file__).resolve().parent
    provenance = [Path(__file__).resolve(), root / "io_reader_adapter.rs", root / "io_reader_persistent.rs",
                  root / "hdf5_batch.py", root / "dioptrin_io_caller_buffer.patch",
                  ROOT / "src/integration/cuda/bench/hdf5.rs"]
    data = dict(metadata=dict(
        commit=command("git", "rev-parse", "HEAD"), source_sha256={p.relative_to(ROOT).as_posix(): digest_file(p)
                                                                  for p in provenance},
        executable_sha256=digest_file(executable), executable_build=build, plan_key=plan_key,
        frames=frames, shape=shape, geometry=geometry, compressed_bytes=compressed,
        hdf5_filter_ids=filters, dataset=args.dataset, file_sha256=scan_digest,
        metadata_sha256=digest_file(args.metadata), cache=args.cache, layout=args.layout,
        extra_mask_sha256=extra_mask,
        source_storage=storage_for(args.hdf5), scratch_storage=storage_for(args.out.resolve().parent),
        reference_sha256=digest_file(args.reference) if args.reference else None,
        cpu=cpu_model(), gpu_before=gpu_snapshot(),
    ), records=[])
    reference = np.fromfile(args.reference, dtype="<f8", count=frames * PATTERN) if args.reference else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configurations = [(backend, threads, tile) for backend in args.backends
                      for threads in (args.threads if backend == "cuda" else [1]) for tile in args.tile_sizes]
    rng = random.Random(42019)
    for trial in range(args.trials):
        rng.shuffle(configurations)
        for backend, threads, tile in configurations:
            print(f"{backend}: {frames} frames, {threads} decoder threads, {tile}-frame tiles, trial {trial + 1}", flush=True)
            plans = ["fresh", "retained"]
            rng.shuffle(plans)
            rows = []
            for plan in plans:
                with TemporaryDirectory(dir=args.out.resolve().parent) as directory:
                    workspace = Path(directory)
                    with trial_input(args.hdf5, workspace, args.cache, scan_digest) as path:
                        row, result = run_requests(executable, env | dict(
                                DIOPTRIN_HDF5_DECODERS=str(threads), DIOPTRIN_HDF5_TILE=str(tile),
                                DIOPTRIN_HDF5_BACKEND=backend), path,
                            args.hdf5.resolve() if plan == "retained" else None,
                            workspace, frames, plan_key, args.cache, compressed)
                if reference is None:
                    reference = result
                row.update(compare_output(result, reference, frames))
                rows.append(row)
                print(f"  {row['plan']}: {row['request_to_patterns_ms'] / 1000:.3f} s ({row['cache_observation']})", flush=True)
            pinned_bytes = min(2, (frames + tile - 1) // tile) * min(tile, frames) * (4 * shape[1] * shape[2] + 8 * PATTERN)
            data["records"].append(dict(trial=trial, backend=backend, decoder_threads=threads, tile=tile,
                                        pinned_bytes=pinned_bytes if backend == "cuda" else 0, samples=rows))
            args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    data["metadata"]["gpu_after"] = gpu_snapshot()
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
