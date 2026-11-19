"""Compare fresh and retained shared CUDA plans on two reads of an Eiger scan."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from hdf5_batch import MARKER, PATTERN, compare_output, digest_file, disk_delta, disk_reads, scan_info, trial_input
from prepare_io_benchmarks import IO_COMMIT
from run import ROOT, command, cpu_model, gpu_snapshot


TEST = "integration::cuda::bench::hdf5::persistent::eiger_hdf5_persistent"


def run_pair(executable, env, paths, workspace, frames):
    import numpy as np

    outputs = [workspace / f"patterns-{i}.f64" for i in range(len(paths))]
    requests = [json.dumps(dict(source=str(path), output=str(out))) + "\n"
                for path, out in zip(paths, outputs)]
    before = disk_reads()
    started = perf_counter_ns()
    args = [str(executable), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    with subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
        rows, patterns, log = [], [], []
        try:
            for i, (request, output) in enumerate(zip(requests, outputs)):
                if i:
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
                if row["iteration"] != i or row["pid"] != proc.pid or collected.size != frames * PATTERN:
                    raise RuntimeError("Shared CUDA worker returned the wrong scan or frame count")
                row.update(request_to_patterns_ms=elapsed_ms, disk_read_bytes=disk_delta(before),
                           plan="retained" if i else "fresh")
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
    return rows, patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--benchmark-exe", required=True, type=Path, help="Isolated dioptrin-io benchmark executable")
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
    root = Path(__file__).resolve().parent
    provenance = [Path(__file__).resolve(), root / "io_reader_adapter.rs", root / "io_reader_persistent.rs",
                  root / "hdf5_batch.py", root / "dioptrin_io_caller_buffer.patch",
                  ROOT / "src/integration/cuda/bench/hdf5.rs"]
    data = dict(metadata=dict(
        commit=command("git", "rev-parse", "HEAD"), source_sha256={p.relative_to(ROOT).as_posix(): digest_file(p)
                                                                  for p in provenance},
        executable_sha256=digest_file(executable), dioptrin_io_commit=IO_COMMIT,
        frames=frames, shape=shape, geometry=geometry, compressed_bytes=compressed,
        hdf5_filter_ids=filters, dataset=args.dataset, file_sha256=scan_digest,
        metadata_sha256=digest_file(args.metadata), cache=args.cache, layout=args.layout,
        extra_mask_sha256=digest_file(args.extra_mask) if args.extra_mask else None,
        reference_sha256=digest_file(args.reference) if args.reference else None,
        cpu=cpu_model(), gpu_before=gpu_snapshot(),
    ), records=[])
    reference = np.fromfile(args.reference, dtype="<f8", count=frames * PATTERN) if args.reference else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configurations = [(threads, tile) for threads in args.threads for tile in args.tile_sizes]
    for trial in range(args.trials):
        for threads, tile in configurations[::1 if trial % 2 == 0 else -1]:
            print(f"{frames} frames, {threads} decoder threads, {tile}-frame tiles, trial {trial + 1}", flush=True)
            with TemporaryDirectory(dir=args.out.resolve().parent) as directory, ExitStack() as clean:
                workspace = Path(directory)
                paths = [clean.enter_context(trial_input(args.hdf5, workspace, args.cache, scan_digest)) for _ in range(2)]
                rows, patterns = run_pair(executable, env | dict(DIOPTRIN_HDF5_DECODERS=str(threads),
                                                                 DIOPTRIN_HDF5_TILE=str(tile)),
                                          paths, workspace, frames)
            if reference is None:
                reference = patterns[0]
            for row, result in zip(rows, patterns):
                row.update(compare_output(result, reference, frames))
                print(f"  {row['plan']}: {row['request_to_patterns_ms'] / 1000:.3f} s", flush=True)
            data["records"].append(dict(trial=trial, decoder_threads=threads, tile=tile, samples=rows))
            args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    data["metadata"]["gpu_after"] = gpu_snapshot()
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
