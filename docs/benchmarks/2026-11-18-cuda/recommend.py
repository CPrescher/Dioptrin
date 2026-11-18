"""Recommend a same-precision path from a local TIFF or HDF5 benchmark."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def measurements(result, method, first, cpu_only):
    timings = defaultdict(list)
    if "trials" in result:
        for row in result["trials"]:
            if row["method"] != method or (cpu_only and row["backend"] != "cpu"):
                continue
            value = (row["fresh"]["process_to_patterns_ms"] if first
                     else row["prepared"]["batch_ms"])
            timings[row["backend"], None, 1].append(value)
    else:
        for row in result["records"]:
            if row["method"] != method or (cpu_only and not row["backend"].startswith("cpu")):
                continue
            frames = (r for r in row["samples"] if (r["iteration"] == 0) == first)
            key = "process_to_patterns_ms" if first else "batch_ms"
            timings[row["backend"], row["tile"], row.get("readers", 1)].extend(r[key] for r in frames)
    return [
        dict(backend=backend, tile=tile, readers=readers, median_ms=median(values), min_ms=min(values),
             max_ms=max(values), samples=len(values))
        for (backend, tile, readers), values in timings.items() if values
    ]


def recommend(options):
    ranked = sorted(options, key=lambda row: row["median_ms"])
    best = ranked[0]
    best_cpu = next((x for x in ranked if x["backend"].startswith("cpu")), None)
    if best_cpu and not best["backend"].startswith("cpu") and best["max_ms"] >= best_cpu["min_ms"]:
        return best_cpu, "CPU and CUDA timing ranges overlap", ranked
    reason = ("Lowest measured median for this workload" if best["samples"] > 1
              else "Lowest observed time from one run per configuration")
    return best, reason, ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--method", choices=("counting", "pixel_split", "pyfai_full"),
                        default="pyfai_full")
    parser.add_argument("--phase", choices=("first", "prepared"), default="prepared")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if "gpu_after" not in result["metadata"]:
        parser.error("use a completed benchmark result")
    options = measurements(result, args.method, args.phase == "first", args.cpu_only)
    if not options:
        parser.error("the requested method and phase were not measured")
    chosen, reason, ranked = recommend(options)
    print(json.dumps(dict(
        frames=result["metadata"]["frames"], method=args.method, phase=args.phase,
        precision="f64", recommended=chosen, reason=reason, measured=ranked,
    ), indent=2))


if __name__ == "__main__":
    main()
