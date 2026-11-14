#!/usr/bin/env python3
"""Fresh-process CPU comparison of Dioptrin, azint, and pyFAI (1D, in memory)."""

import argparse
import copy
import datetime
import csv
import importlib.metadata
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fixtures import MANIFEST, manifest, prepare, sha256
from provenance import verify_build


ROOT = Path(__file__).resolve().parent.parent.parent
VARIANTS = (
    "dioptrin-pixel_split",
    "dioptrin-supersampled",
    "dioptrin-pyfai_full",
    "azint",
    "pyfai-full",
    "pyfai-bbox",
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contiguous_chunks(frames, workers):
    indexed = list(enumerate(frames))
    size = (len(frames) + workers - 1) // workers
    return [indexed[i * size : (i + 1) * size] for i in range(workers)]


def worker(case_path):
    import numpy as np

    case = json.loads(Path(case_path).read_text())
    fixture = Path(case["fixture"])
    meta = json.loads((fixture / "fixture.json").read_text())
    stack = np.load(fixture / "frames.npy", mmap_mode="r")
    unique = [stack[0]] if case["workload"] == "repeated" else list(stack)
    image = unique[0]
    mask = np.load(fixture / "mask.npy")
    frames = [unique[i % len(unique)] for i in range(case["frames"])]
    # Fault every input page in before setup/timing. All cases see the same
    # immutable frame sequence; frame generation/decoding is never timed.
    for frame in unique:
        float(np.sum(frame))
    n = case["bins"]
    workers = case["workers"]
    errors = case["errors"]
    variant = case["variant"]
    bounds = meta["radial_range"]
    pool = None
    started = time.perf_counter()

    if variant.startswith("dioptrin-"):
        import dioptrin

        ai = dioptrin.Integrator(
            str(fixture / "geometry.poni"),
            method=variant.removeprefix("dioptrin-"),
            unit="2th_deg",
            n_ss=meta["n_splitting"],
            polarization_factor=0.965,
            radial_range=tuple(bounds),
        )
        ai.set_mask(mask)

        # The current array batch path uses the global Rayon pool rather than
        # its file-worker arguments. Set the real total CPU budget at startup.
        if dioptrin.rayon_num_threads() != workers:
            raise RuntimeError(
                "Dioptrin Rayon pool does not match requested CPU budget"
            )

        def integrate_frames(images):
            return ai.batch1d(
                images, n, errors=errors, num_workers=workers, num_threads=1
            )

        def run():
            return integrate_frames(frames)

        def extract(result):
            if not result.is_ok():
                raise RuntimeError(result.error)
            value = result.result
            return (
                np.asarray(value.radial),
                np.asarray(value.intensity),
                np.asarray(value.errors) if errors else None,
            )

        extract(integrate_frames([image])[0])
        setup_first = time.perf_counter() - started
        baseline = [extract(r) for r in integrate_frames(unique)]
    else:
        if variant == "azint":
            import azint
            from azint.detector import Detector

            geometry = meta["geometry"].copy()
            pixel1, pixel2 = geometry.pop("pixel1"), geometry.pop("pixel2")
            poni = azint.Poni(det=Detector(pixel1, pixel2, image.shape), **geometry)
            ai = azint.AzimuthalIntegrator(
                poni,
                n_splitting=meta["n_splitting"],
                radial_bins=np.linspace(*bounds, n + 1),
                unit="2th",
                mask=mask,
                solid_angle=True,
                polarization_factor=0.965,
                error_model="poisson" if errors else None,
            )

            def integrate(engine, frame):
                return engine.integrate(frame)

            def extract(result):
                return (
                    np.asarray(ai.radial_axis),
                    np.asarray(result[0]),
                    result[1] if errors else None,
                )

        else:
            import pyFAI
            from pyFAI.method_registry import IntegrationMethod

            ai = pyFAI.load(str(fixture / "geometry.poni"))
            ai.detector.mask = mask
            method = (variant.removeprefix("pyfai-"), "csr", "cython")
            if not IntegrationMethod.select_method(
                dim=1, split=method[0], algo="csr", impl="cython", degradable=False
            ):
                raise RuntimeError(
                    f"Requested exact pyFAI method unavailable: {method}"
                )

            def integrate(engine, frame):
                return engine.integrate1d(
                    frame,
                    n,
                    unit="2th_deg",
                    method=method,
                    radial_range=tuple(bounds),
                    correctSolidAngle=True,
                    polarization_factor=0.965,
                    error_model="poisson" if errors else None,
                    safe=True,
                )

            def extract(result):
                used = result.method.method
                if (used.split, used.algo, used.impl) != method:
                    raise RuntimeError(f"pyFAI fell back from {method} to {used}")
                return (
                    np.asarray(result.radial),
                    np.asarray(result.intensity),
                    result.sigma if errors else None,
                )

        extract(integrate(ai, image))
        setup_first = time.perf_counter() - started
        baseline = [extract(integrate(ai, frame)) for frame in unique]
        engines = [ai]
        for _ in range(1, workers):
            engine = copy.copy(ai)
            if variant.startswith("pyfai-"):
                # Share immutable geometry arrays, but each frame worker owns
                # its CSR engine and scratch buffers. Never concurrently use ai.
                engine.engines = {}
            integrate(engine, image)
            engines.append(engine)
        pool = ThreadPoolExecutor(max_workers=workers)
        chunks = contiguous_chunks(frames, workers)

        def chunk(task):
            engine, images = task
            return [(index, integrate(engine, frame)) for index, frame in images]

        def run():
            results = [None] * len(frames)
            for group in pool.map(chunk, zip(engines, chunks)):
                for index, result in group:
                    results[index] = result
            return results

    sample = tuple(
        np.stack([entry[i] for entry in baseline]) if i < 2 or errors else np.array([])
        for i in range(3)
    )
    baseline = sample
    if case.get("reference"):
        with np.load(case["reference"]) as reference:
            baseline = (
                reference["radial"].copy(),
                reference["intensity"].copy(),
                reference["sigma"].copy() if errors else np.array([]),
            )

    def validate(results):
        if len(results) != len(frames):
            raise AssertionError("Wrong output frame count")
        for index, result in enumerate(results):
            radial, intensity, sigma = extract(result)
            expected = index % len(unique)
            if intensity.shape != (n,) or not np.isfinite(intensity).all():
                raise AssertionError("Invalid integration output")
            if not np.any(intensity > 0):
                raise AssertionError("Empty integration output")
            np.testing.assert_allclose(
                radial, baseline[0][expected], rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                intensity, baseline[1][expected], rtol=1e-6, atol=1e-6
            )
            if errors:
                if sigma is None or not np.isfinite(sigma).all():
                    raise AssertionError("Invalid Poisson errors")
                np.testing.assert_allclose(
                    sigma, baseline[2][expected], rtol=1e-6, atol=1e-6
                )

    warm_result = run()
    setup_ready = time.perf_counter() - started
    validate(warm_result)
    del warm_result
    # Warmup and validation are outside warmed timing blocks. Native batch API
    # scheduling and result allocation are included in every measured call.
    durations = []
    counts = []
    for _ in range(case["blocks"]):
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < case["seconds"]:
            result = run()
            count += len(result)
            del result
        durations.append(time.perf_counter() - t0)
        counts.append(count)
    validate(run())
    if pool:
        pool.shutdown()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    rates = [count / duration for count, duration in zip(counts, durations)]
    result_path = Path(case["result"])
    negative_sigma = np.argwhere(sample[2] < 0).tolist() if errors else []
    np.savez_compressed(
        result_path.with_suffix(".npz"),
        radial=sample[0],
        intensity=sample[1],
        sigma=sample[2] if errors else np.array([]),
    )
    dump(
        result_path,
        dict(
            **case,
            setup_first_s=setup_first,
            setup_ready_s=setup_ready,
            thread_environment={
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            frames_per_s=float(np.median(rates)),
            block_frames_per_s=rates,
            block_seconds=durations,
            block_frames=counts,
            peak_rss_mib=peak / 2**20,
            distinct_frames=len(unique),
            self_check="passed",
            numerical_status="negative_sigma" if negative_sigma else "ok",
            negative_sigma_bins=len(negative_sigma),
            negative_sigma_bin_indices=negative_sigma,
        ),
    )
    print(
        f"{meta['dataset']:14} {case['workload']:8} {variant:25} workers={workers:3} "
        f"{case['error_mode']:9} {np.median(rates):10.1f} frames/s",
        flush=True,
    )
    if negative_sigma:
        print(
            f"NUMERICAL WARNING: {len(negative_sigma)} negative Poisson sigma bins; raw values retained",
            flush=True,
        )


def summarize(output):
    import numpy as np

    rows = [
        json.loads(p.read_text())
        for p in sorted((output / "cases").glob("*.result.json"))
    ]
    fields = [
        "dataset",
        "workload",
        "variant",
        "workers",
        "error_mode",
        "trial",
        "frames_per_s",
        "setup_first_s",
        "setup_ready_s",
        "peak_rss_mib",
        "self_check",
        "numerical_status",
        "negative_sigma_bins",
    ]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    numerical = []
    for row in rows:
        if row["workers"] != 1 or row["trial"] != 0:
            continue
        reference = next(
            (
                r
                for r in rows
                if r["dataset"] == row["dataset"]
                and r["workload"] == row["workload"]
                and r["variant"] == "dioptrin-pixel_split"
                and r["workers"] == 1
                and r["trial"] == 0
                and r["errors"] == row["errors"]
            ),
            None,
        )
        if reference is None:
            continue
        with np.load(Path(row["result"]).with_suffix(".npz")) as saved:
            a = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        with np.load(Path(reference["result"]).with_suffix(".npz")) as saved:
            b = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        metrics = dict(
            dataset=row["dataset"],
            workload=row["workload"],
            variant=row["variant"],
            error_mode=row["error_mode"],
            reference="dioptrin-pixel_split",
            negative_sigma_bins=row.get("negative_sigma_bins", 0),
            max_axis_difference_deg=float(np.max(np.abs(a["radial"] - b["radial"]))),
        )
        for key in ("intensity", "sigma"):
            if a[key].size:
                denominator = float(np.linalg.norm(b[key]))
                metrics[key + "_relative_l2"] = float(
                    np.linalg.norm(a[key] - b[key])
                ) / max(denominator, 1e-300)
        numerical.append(metrics)
    dump(output / "numerical.json", numerical)
    lines = [
        "# CPU integration benchmark",
        "",
        "1D, preloaded float64 inputs; generation, decoding, and output writing excluded.",
        "Repeated reuses one image; frames cycles through distinct generated images.",
        "Cross-library algorithms, precision, and normalization differ; see numerical.json.",
        "",
        "| Dataset | Workload | Method | CPU budget | Errors | Median frames/s | Trials | Numerical status |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    groups = sorted({tuple(r[f] for f in fields[:5]) for r in rows})
    for key in groups:
        values = [
            r["frames_per_s"] for r in rows if tuple(r[f] for f in fields[:5]) == key
        ]
        status = ", ".join(
            sorted(
                {
                    r.get("numerical_status", "unchecked")
                    for r in rows
                    if tuple(r[f] for f in fields[:5]) == key
                }
            )
        )
        lines.append(
            "| "
            + " | ".join(
                map(
                    str, (*key, round(float(np.median(values)), 2), len(values), status)
                )
            )
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    definition = manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=definition["suites"], default="standard")
    parser.add_argument(
        "--output",
        type=Path,
        help="New results directory; default: benchmark-results/<timestamp>",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "dioptrin-benchmarks",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use bundled/cached data only"
    )
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="Retain generated frame stacks after success",
    )
    parser.add_argument(
        "--allow-unverified-build",
        action="store_true",
        help="Explicitly allow an unrecorded/stale Dioptrin build, with a warning in the report",
    )
    parser.add_argument(
        "--workers",
        help="Comma-separated CPU budgets; default: suite budgets within CPU affinity",
    )
    parser.add_argument("--datasets", nargs="+", choices=definition["datasets"])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("repeated", "frames"),
        default=["repeated", "frames"],
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--bins", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(args.case)
        return
    if args.list_datasets:
        print(json.dumps(definition, indent=2))
        return
    defaults = definition["suites"][args.suite]
    for key in ("bins", "frames", "trials", "blocks", "seconds", "datasets"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    for key in ("bins", "frames", "trials", "blocks", "seconds"):
        if getattr(args, key) <= 0 or not math.isfinite(getattr(args, key)):
            parser.error(f"--{key} must be positive and finite")
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        workers = (
            sorted({int(w) for w in args.workers.split(",")})
            if args.workers
            else [w for w in defaults["workers"] if w <= available]
        )
    except ValueError:
        parser.error("--workers must contain comma-separated positive integers")
    if not workers or workers[0] < 1 or workers[-1] > available:
        parser.error(
            f"--workers must be within the process CPU allocation (1..{available})"
        )
    workers = sorted(set(workers) | {1})
    required = ["numpy", "pyFAI"]
    if any(v.startswith("dioptrin-") for v in args.variants):
        required.append("dioptrin")
    if "azint" in args.variants:
        required.append("azint")
    try:
        versions = {name: importlib.metadata.version(name) for name in required}
    except importlib.metadata.PackageNotFoundError as error:
        parser.error(
            f"Missing benchmark dependency {error.name}; run python3 benchmarks/comparison/setup.py first"
        )
    build_path = Path(sys.prefix) / "benchmark-build.json"
    installed_build = (
        json.loads(build_path.read_text()) if build_path.is_file() else None
    )
    build_problems = (
        verify_build(ROOT, installed_build) if "dioptrin" in required else []
    )
    if build_problems:
        message = "; ".join(build_problems)
        if not args.allow_unverified_build:
            parser.error(
                message
                + ". Rerun setup.py, or explicitly use --allow-unverified-build."
            )
        print("UNVERIFIED BUILD: " + message, file=sys.stderr, flush=True)
    module_paths = {name: importlib.import_module(name).__file__ for name in required}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "benchmark-results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    provenance = {}
    if (ROOT / ".git").exists():
        for key, command in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("status", ["git", "status", "--short"]),
        ):
            provenance[key] = subprocess.check_output(
                command, cwd=ROOT, text=True
            ).strip()
    dump(
        output / "environment.json",
        dict(
            versions=versions,
            python=sys.version,
            platform=platform.platform(),
            node=platform.node(),
            available_logical_cpus=available,
            cpu_budgets=workers,
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            source=provenance,
            manifest_sha256=sha256(MANIFEST),
            runner_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in ("compare.py", "fixtures.py", "provenance.py")
            },
            installed_build=installed_build,
            build_verification_problems=build_problems,
            module_paths=module_paths,
        ),
    )
    (output / "datasets.json").write_bytes(MANIFEST.read_bytes())
    metadata = {}
    for dataset in args.datasets:
        print(f"Preparing {dataset} outside timing...", flush=True)
        metadata[dataset] = prepare(
            output / dataset, dataset, args.data_dir, args.cache_dir, args.offline
        )
    failures = []
    for trial in range(args.trials):
        variants = args.variants if trial % 2 == 0 else list(reversed(args.variants))
        for dataset in args.datasets:
            for workload in args.workloads:
                if workload == "frames" and metadata[dataset]["distinct_frames"] < 2:
                    continue
                for count in workers:
                    for errors in (False, True):
                        for variant in variants:
                            suffix = f"{dataset}-{workload}-{variant}"
                            mode = "poisson" if errors else "intensity"
                            label = f"{suffix}-w{count}-{mode}-t{trial}"
                            case_path = output / "cases" / f"{label}.json"
                            case = dict(
                                dataset=dataset,
                                workload=workload,
                                variant=variant,
                                workers=count,
                                errors=errors,
                                error_mode=mode,
                                trial=trial,
                                fixture=str(output / dataset),
                                frames=((max(args.frames, count) + count - 1) // count)
                                * count,
                                bins=args.bins,
                                blocks=args.blocks,
                                seconds=args.seconds,
                                result=str(case_path.with_suffix(".result.json")),
                            )
                            if count > 1 or trial > 0:
                                case["reference"] = str(
                                    output
                                    / "cases"
                                    / f"{suffix}-w1-{mode}-t0.result.npz"
                                )
                            dump(case_path, case)
                            env = os.environ | {
                                "OMP_NUM_THREADS": "1",
                                "RAYON_NUM_THREADS": str(count)
                                if variant.startswith("dioptrin-")
                                else "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "NUMEXPR_NUM_THREADS": "1",
                                "OMP_DYNAMIC": "FALSE",
                                "OMP_PROC_BIND": "FALSE",
                            }
                            with case_path.with_suffix(".log").open("w") as log:
                                result = subprocess.run(
                                    [
                                        sys.executable,
                                        "-E",
                                        "-s",
                                        __file__,
                                        "--case",
                                        str(case_path),
                                    ],
                                    env=env,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                )
                            if result.returncode:
                                failures.append(
                                    dict(case=label, exit_code=result.returncode)
                                )
                                print(
                                    f"FAILED: {label}; see {case_path.with_suffix('.log')}",
                                    flush=True,
                                )
                            else:
                                print(
                                    case_path.with_suffix(".log").read_text().strip(),
                                    flush=True,
                                )
                            summarize(output)
                            dump(output / "failures.json", failures)
    if failures:
        raise SystemExit(
            f"{len(failures)} cases failed; partial results and inputs retained in {output}"
        )
    if not args.keep_inputs:
        for dataset in args.datasets:
            for filename in ("frames.npy", "mask.npy"):
                (output / dataset / filename).unlink()
    print(f"Results: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    # Keep the parent and case processes on the same interpreter paths even if
    # a shell/module exports PYTHONPATH or enables user-installed packages.
    if not sys.flags.ignore_environment or not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, "-E", "-s", *sys.argv])
    main()
