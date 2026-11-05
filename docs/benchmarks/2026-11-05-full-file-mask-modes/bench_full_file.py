"""Matched real-Eiger CPU benchmark against a pinned bslz4_to_sparse build.

Build examples/export_benchmark_lut.rs first (same source as the installed
Dioptrin wheel). See --help. Large inputs and exported LUTs stay outside Git.
The original PONI orientation=2 is handled by flipping Dioptrin inputs and
mapping the exported matrix's columns back to raw detector order for bslz4.
No detector values are clipped, thresholded, or background-subtracted.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import mmap
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import bslz4_to_sparse as b
import dioptrin
import fabio
import h5py
import hdf5plugin  # noqa: F401 -- registers the detector's compression filter
import numpy as np
import psutil
from scipy import sparse


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def other_process_cpu():
    result = {}
    for proc in psutil.process_iter(["pid", "cpu_times"]):
        try:
            if proc.pid != os.getpid() and proc.info["cpu_times"] is not None:
                t = proc.info["cpu_times"]
                result[proc.pid] = t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def timed(name, fn, frames, target, repeat):
    gc.collect()
    other0, process0 = other_process_cpu(), time.process_time()
    start = time.perf_counter()
    loops = 0
    while True:
        result = fn()
        # Consume the returned result; allocation and release are timed too.
        assert result.shape[0] == frames and np.isfinite(result).all()
        checksum = float(result.sum())
        del result
        loops += 1
        elapsed = time.perf_counter() - start
        if elapsed >= target:
            break
    process = time.process_time() - process0
    other1 = other_process_cpu()
    other = (
        sum(max(0.0, v - other0[pid]) for pid, v in other1.items() if pid in other0)
        / elapsed
    )
    row = dict(
        name=name,
        repeat=repeat,
        frames=frames,
        loops=loops,
        seconds=elapsed,
        fps=frames * loops / elapsed,
        other_busy_cores=other,
        process_cores=process / elapsed,
        checksum=checksum,
    )
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--dataset", default="entry_0000/instrument/eiger/data")
    parser.add_argument(
        "--poni",
        type=Path,
        required=True,
        help="Derived PONI with explicit pixel sizes; original orientation must be 2",
    )
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-seconds", type=float, default=0.6)
    parser.add_argument("--full-file", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--file-only", action="store_true")
    parser.add_argument("--include-reusable", action="store_true")
    parser.add_argument("--include-decode-control", action="store_true")
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Measure bslz4 batches and CPU pipeline choices on 128 sampled file chunks",
    )
    parser.add_argument(
        "--reuse-control-only",
        action="store_true",
        help="Compare reusable-buffer decompression + Dioptrin with fused integration",
    )
    parser.add_argument(
        "--pipeline-json",
        type=json.loads,
        default=None,
        help='Explicit upstream pipeline overrides, e.g. {"dot":"csc"}',
    )
    parser.add_argument("--input-sha256", default="not calculated")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dioptrin.set_rayon_threads(args.workers)
    record = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        python=sys.version,
        platform=platform.platform(),
        cpu=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        versions={
            k: importlib.metadata.version(k)
            for k in [
                "dioptrin",
                "bslz4_to_sparse",
                "numpy",
                "scipy",
                "h5py",
                "hdf5plugin",
            ]
        },
        bslz4_build=b.build_info(),
        binaries={},
        measurements=[],
    )
    record["binaries"]["dioptrin"] = [
        dict(path=str(p), sha256=digest(p))
        for p in Path(dioptrin.__file__).parent.glob("*.so")
    ]
    record["binaries"]["bslz4"] = [
        dict(path=str(p), sha256=digest(p))
        for p in Path(b.__file__).parent.glob("*.so")
    ]
    with h5py.File(args.hdf5, "r") as f:
        ds = f[args.dataset]
        count, height, width = ds.shape
        assert ds.chunks == (1, height, width) and ds.dtype == np.dtype("uint32")
        indices = np.linspace(0, count - 1, min(args.samples, count), dtype=int)
        locations = [ds.id.get_chunk_info_by_coord((i, 0, 0)) for i in range(count)]
        assert all(c.filter_mask == 0 for c in locations)
        offsets = np.array([c.byte_offset for c in locations], dtype=np.int64)
        lengths = np.array([c.size for c in locations], dtype=np.int64)
        raw = np.stack([ds[int(i)] for i in indices])
        filters = [
            repr(ds.id.get_create_plist().get_filter(i))
            for i in range(ds.id.get_create_plist().get_nfilters())
        ]
    # The supplied TIFF mask is stored in Dioptas's bottom-up convention.
    # Reversing it aligns it with raw HDF5 rows; all first-frame detector
    # sentinel pixels must be covered before timing either engine.
    mask = np.ascontiguousarray(fabio.open(str(args.mask)).data[::-1] != 0, np.uint8)
    assert mask.shape == (height, width)
    valid = np.ascontiguousarray(1 - mask)
    processed = [np.ascontiguousarray(frame[::-1], dtype=np.float64) for frame in raw]
    record["input"] = dict(
        path=str(args.hdf5.resolve()),
        sha256=args.input_sha256,
        bytes=args.hdf5.stat().st_size,
        shape=[count, height, width],
        dtype="uint32",
        filters=filters,
        sample_indices=indices.tolist(),
        sample_nonzero_fraction=float(np.count_nonzero(raw) / raw.size),
        unmasked_min=min(int(a[mask == 0].min()) for a in raw),
        unmasked_max=max(int(a[mask == 0].max()) for a in raw),
        first_frame_unmasked_sentinels=int(
            np.count_nonzero((raw[0] == (1 << 32) - 1) & (mask == 0))
        ),
        masked_pixels=int(mask.sum()),
        sampled_compressed_bytes=int(lengths[indices].sum()),
        poni_sha256=digest(args.poni),
        mask_sha256=digest(args.mask),
    )
    print(json.dumps({"input": record["input"]}), flush=True)
    assert record["input"]["first_frame_unmasked_sentinels"] == 0

    ptr = np.fromfile(args.lut / "indptr.u64", dtype="<u8").astype(np.int64)
    col = np.fromfile(args.lut / "indices.u32", dtype="<u4")
    # Exported LUT coordinates have row zero at the bottom; the compressed file
    # has row zero at the top (PONI orientation=2).
    col = ((height - 1 - col // width) * width + col % width).astype(np.int32)
    weights_path = args.lut / "weights.f64"
    weights = (
        np.fromfile(weights_path, dtype="<f8")
        if weights_path.exists()
        else np.ones(col.size)
    )
    matrix = sparse.csr_matrix(
        (weights, col, ptr), shape=(len(ptr) - 1, height * width)
    )
    matrix = matrix.multiply(valid.ravel()[None, :]).tocsr()
    matrix.eliminate_zeros()
    pixel_norm = (
        np.fromfile(args.lut / "normalization.f64", dtype="<f8")
        .reshape(height, width)[::-1]
        .ravel()
    )
    norm = np.asarray(matrix @ pixel_norm).ravel()
    nbins = matrix.shape[0]
    radial = np.fromfile(args.lut / "radial.f64", dtype="<f8")
    if args.errors:
        # Poisson variance is sum(w^2 * raw_counts); the exact same photon
        # counts are fed to the squared-weight rows in the same decoder pass.
        variance_matrix = matrix.copy()
        variance_matrix.data **= 2
        bmatrix = sparse.vstack([matrix, variance_matrix], format="csc")
        del variance_matrix
    else:
        bmatrix = matrix.tocsc()
    record["matrix"] = dict(
        bins=nbins,
        nnz=int(matrix.nnz),
        bslz4_nnz=int(bmatrix.nnz),
        bslz4_weights="float32 (required by unmodified upstream)",
        dioptrin_weights="float64",
        normalization="identical exported float64 denominator",
        errors="squared-weight rows in same pass" if args.errors else "disabled",
    )
    setup_start = time.perf_counter()
    integ = dioptrin.Integrator(
        str(args.poni),
        method="pixel_split",
        unit="2th_deg",
        polarization_factor=0.965,
        fast_math=True,
    )
    integ.set_mask(np.ascontiguousarray(mask[::-1]))
    first = integ.integrate1d(processed[0], nbins, errors=args.errors)
    record["dioptrin_first_result_s"] = time.perf_counter() - setup_start
    assert np.array_equal(first.radial, radial)
    setup_start = time.perf_counter()
    plan = b.plan(valid, bmatrix, np.uint32, codec="lz4", pipeline=args.pipeline_json)
    record["bslz4_plan_s"] = time.perf_counter() - setup_start
    del bmatrix, matrix, weights, col, ptr, pixel_norm
    gc.collect()
    tls = threading.local()
    pipelines = []
    lock = threading.Lock()

    def init_worker():
        tls.decoder = plan.context()

    def normalized(powder):
        out = np.zeros((len(powder), nbins * (2 if args.errors else 1)))
        np.divide(powder[:, :nbins], norm, out=out[:, :nbins], where=norm > 0)
        if args.errors:
            np.divide(
                np.sqrt(np.maximum(powder[:, nbins:], 0)),
                norm,
                out=out[:, nbins:],
                where=norm > 0,
            )
        return out

    cut = 1 << 32  # Above every uint32: disable pixel-list extraction only.

    def job(item):
        kind, ids, pipeline = item
        decoder = tls.decoder
        if pipeline is not None:
            if not hasattr(tls, "variants"):
                tls.variants = {}
            key = json.dumps(pipeline, sort_keys=True)
            if key not in tls.variants:
                tls.variants[key] = b.chunk2sparseCSC(
                    valid, plan, np.uint32, codec="lz4", pipeline=pipeline
                )
            decoder = tls.variants[key]
        if kind == "array":
            powder = decoder.from_frames(raw[ids[0] : ids[-1] + 1], cut)[2]
        elif kind == "compressed":
            powder = decoder.multi([chunks[i] for i in ids], cut)[2]
        else:
            powder = decoder.from_buffer(mapped, offsets[ids], lengths[ids], cut)[2]
        with lock:
            if len(pipelines) < args.workers:
                pipelines.append(list(map(int, decoder.pipeline)))
        return normalized(powder)

    def run_b(kind, ids, batch_size=2, pipeline=None):
        groups = [
            (kind, np.asarray(ids[i : i + batch_size], dtype=int), pipeline)
            for i in range(0, len(ids), batch_size)
        ]
        return np.concatenate(list(pool.map(job, groups)))

    def unpack(results):
        rows = []
        results = sorted(results, key=lambda r: int(re.findall(r"\d+", r.filename)[-1]))
        for result in results:
            if not result.is_ok():
                raise RuntimeError(result.error)
            r = result.result
            rows.append(
                np.concatenate([r.intensity, r.errors]) if args.errors else r.intensity
            )
        return np.asarray(rows)

    def run_d_array():
        return unpack(
            integ.batch1d(
                processed,
                nbins,
                errors=args.errors,
                num_workers=args.workers,
                num_threads=1,
            )
        )

    def run_d_file():
        def frames():
            with h5py.File(args.hdf5, "r") as f:
                ds = f[args.dataset]
                for i in file_ids:
                    yield np.ascontiguousarray(ds[int(i)][::-1], dtype=np.float64)

        return unpack(
            integ.batch1d_iter(
                frames(),
                nbins,
                errors=args.errors,
                num_workers=args.workers,
                num_threads=1,
            )
        )

    def decode_job(i):
        frame = b.decompress(
            [mapped[offsets[i] : offsets[i] + lengths[i]]], np.uint32
        ).reshape(height, width)
        return np.ascontiguousarray(frame[::-1], dtype=np.float64)

    def run_d_parallel_decode():
        # Bounded batches keep RAM usage independent of file length. Decoder
        # and integration stages each use W cores, consecutively (no overlap).
        rows = []
        for i in range(0, len(file_ids), 32):
            arrays = list(pool.map(decode_job, file_ids[i : i + 32]))
            rows.append(
                unpack(
                    integ.batch1d(
                        arrays,
                        nbins,
                        errors=args.errors,
                        num_workers=args.workers,
                        num_threads=1,
                    )
                )
            )
        return np.concatenate(rows)

    reusable_float = None

    def decode_reusable(item):
        slot, i = item
        if not hasattr(tls, "decode_buffer"):
            tls.decode_buffer = np.empty((height, width), np.uint32)
        b.decompress_from_buffer(
            mapped,
            offsets[i : i + 1],
            lengths[i : i + 1],
            np.uint32,
            out=tls.decode_buffer,
        )
        np.copyto(reusable_float[slot], tls.decode_buffer[::-1])

    def run_d_reusable_decode():
        # One typed buffer per decoder thread, W float64 frame buffers shared
        # across consecutive stages; synchronous batch1d borrows them safely.
        rows = []
        for i in range(0, len(file_ids), args.workers):
            ids = file_ids[i : i + args.workers]
            list(pool.map(decode_reusable, enumerate(ids)))
            arrays = [reusable_float[j] for j in range(len(ids))]
            rows.append(
                unpack(
                    integ.batch1d(
                        arrays,
                        nbins,
                        errors=args.errors,
                        num_workers=args.workers,
                        num_threads=1,
                    )
                )
            )
        return np.concatenate(rows)

    file_handle = open(args.hdf5, "rb")
    mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
    # Sampled compressed chunks are copied once for compute-only timings.
    chunks = [mapped[offsets[i] : offsets[i] + lengths[i]] for i in indices]
    for i in [0, len(indices) // 2, len(indices) - 1]:
        np.testing.assert_array_equal(
            b.decompress([chunks[i]], np.uint32).reshape(height, width), raw[i]
        )
    pool = ThreadPoolExecutor(max_workers=args.workers, initializer=init_worker)
    sample_ids = np.arange(len(indices))
    expected = run_d_array()
    array_result = run_b("array", sample_ids)
    compressed_result = run_b("compressed", sample_ids)
    numerical = {}
    for label, result in [
        ("bslz4_array", array_result),
        ("bslz4_compressed", compressed_result),
    ]:
        numerical[label] = {}
        for field, sl in [("intensity", slice(0, nbins))] + (
            [("poisson_sigma", slice(nbins, None))] if args.errors else []
        ):
            x, y = result[:, sl], expected[:, sl]
            rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
            numerical[label][field] = dict(
                relative_l2=rel, max_absolute=float(np.abs(x - y).max())
            )
            np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-8)
    record["numerical"] = numerical
    print(json.dumps({"numerical": numerical}), flush=True)
    np.savez_compressed(
        args.output.with_suffix(".patterns.npz"),
        radial=radial,
        frame_indices=indices,
        dioptrin=expected,
        bslz4=array_result,
        bslz4_compressed=compressed_result,
    )
    jobs = [
        ("dioptrin_array", run_d_array, len(indices)),
        ("bslz4_array", lambda: run_b("array", sample_ids), len(indices)),
        ("bslz4_compressed", lambda: run_b("compressed", sample_ids), len(indices)),
    ]
    file_ids = np.arange(count) if args.full_file else indices
    jobs += [
        ("dioptrin_h5py_stream", run_d_file, len(file_ids)),
        ("dioptrin_parallel_decode", run_d_parallel_decode, len(file_ids)),
        ("bslz4_mmap_stream", lambda: run_b("file", file_ids), len(file_ids)),
    ]
    if args.compute_only:
        jobs = jobs[:3]
    if args.tune_only:
        tune_ids = np.tile(indices, 4)
        jobs = [
            (
                f"bslz4_mmap_batch{batch}",
                lambda batch=batch: run_b("file", tune_ids, batch),
                len(tune_ids),
            )
            for batch in [1, 2, 4, 8]
        ]
        record["tuning_pipelines"] = {
            "csc": {"dot": "csc"},
            "csc_run": {"dot": "csc-run"},
            "neon": {"untranspose": "neon"},
            "stock": {"decode": "lz4-stock"},
        }
        jobs += [
            (
                f"bslz4_mmap_{name}",
                lambda pipeline=pipeline: run_b("file", tune_ids, pipeline=pipeline),
                len(tune_ids),
            )
            for name, pipeline in record["tuning_pipelines"].items()
        ]
    if args.reuse_control_only:
        del raw, processed, chunks, array_result, compressed_result
        gc.collect()
        reusable_float = np.empty((args.workers, height, width), np.float64)
        jobs = [
            ("dioptrin_reusable_decode", run_d_reusable_decode, len(file_ids)),
            ("bslz4_mmap_stream", lambda: run_b("file", file_ids), len(file_ids)),
        ]
    if args.file_only:
        jobs = jobs[3:]
        del raw, processed, chunks, array_result, compressed_result
        gc.collect()
    if args.include_reusable:
        reusable_float = np.empty((args.workers, height, width), np.float64)
        jobs.append(("dioptrin_reusable_decode", run_d_reusable_decode, len(file_ids)))
    if args.include_decode_control:
        def run_decode_control():
            out = np.empty((len(file_ids), 1))
            for i in range(0, len(file_ids), args.workers):
                ids = file_ids[i:i+args.workers]
                list(pool.map(decode_reusable, enumerate(ids)))
                out[i:i+len(ids),0] = reusable_float[:len(ids),0,0]
            return out
        jobs.append(("read_decode_convert_only",run_decode_control,len(file_ids)))

    file_reference = None
    for name, fn, n in jobs:
        print("warm " + name, flush=True)
        result = fn()
        assert result.shape[0] == n
        if args.tune_only:
            np.testing.assert_allclose(
                result, np.tile(expected, (4, 1)), rtol=2e-6, atol=1e-8
            )
        if name in ("dioptrin_h5py_stream", "dioptrin_reusable_decode"):
            file_reference = result.copy()
        elif name in ("dioptrin_parallel_decode", "bslz4_mmap_stream"):
            np.testing.assert_allclose(result, file_reference, rtol=2e-6, atol=1e-8)
            record["numerical"][name] = dict(
                relative_l2=float(
                    np.linalg.norm(result - file_reference)
                    / np.linalg.norm(file_reference)
                ),
                frames_checked=n,
            )
        del result
    record["file_reading_protocol"] = {
        "mask_mode": getattr(integ, "mask_mode", "legacy"),
        "file_frames": len(file_ids),
        "h5py": "Open dataset and read/decode each frame, flip/convert to float64 inside timed producer; integrate via streaming batch.",
        "parallel_decode": "Read chunks through read-only mmap, decode in eight threads, allocate flipped float64 frames, integrate bounded groups32.",
        "reusable_decode": "Same chunk decoder, per-thread uint32 buffers and eight reusable float64 frames; all read/decode/convert/integration timed.",
        "jon": "Read chunks from the same mapping, fused bitshuffle-LZ4 decode plus CSC integration, float32 weights and float64 output/normalization.",
        "setup_excluded": "HDF5 chunk-index discovery, calibration/LUT/plan construction, opening the raw mmap, sampled validation and buffer allocation before the timed passes.",
        "cache": "Warm/repeated whole-file passes; OS page cache is not flushed. Physical cold-disk bandwidth is not measured.",
        "decode_control": "Reading mapped chunks, decompressing to uint32 and flipping/copying to reusable float64; no integration. Optional diagnostic, not an integration result."
    }
    record["bslz4_pipelines"] = pipelines
    for rep in range(args.repeats):
        order = jobs if rep % 2 == 0 else list(reversed(jobs))
        for name, fn, n in order:
            record["measurements"].append(timed(name, fn, n, args.target_seconds, rep))
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    pool.shutdown()
    mapped.close()
    file_handle.close()
    summary = {
        name: dict(
            median_fps=statistics.median(
                x["fps"] for x in record["measurements"] if x["name"] == name
            ),
            median_other_busy_cores=statistics.median(
                x["other_busy_cores"]
                for x in record["measurements"]
                if x["name"] == name
            ),
        )
        for name, _, _ in jobs
    }
    record["summary"] = summary
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
