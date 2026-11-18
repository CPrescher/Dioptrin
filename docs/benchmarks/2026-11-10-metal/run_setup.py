#!/usr/bin/env python3
"""Measure complete process-to-first-pattern startup and amortization.

Uses the benchmark-only static Rust bridge and the existing Metal kernel. All
measurements include Poisson std. Fresh processes retain normal OS file/shader
caches. Compilation and numerical reference validation are outside timing.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=REPO, check=True, **kwargs)


def stats(values):
    ordered = sorted(values)
    return {"median": statistics.median(values), "min": ordered[0],
            "max": ordered[-1], "values": values}


def crossover(cpu_first, gpu_first, cpu_frame, gpu_frame):
    """T(N) = F + (N-1)t. Distinguish permanent wins from early-only wins."""
    extra = gpu_first - cpu_first
    saving = cpu_frame - gpu_frame
    result = {"gpu_startup_extra_ms": extra, "gpu_saving_per_frame_ms": saving}
    if saving > 0:
        result.update(kind="gpu_wins_from", images=1 + math.ceil(max(0, extra) / saving))
    elif saving == 0:
        result.update(kind="gpu_wins_from" if extra <= 0 else "gpu_never_wins",
                      images=1 if extra <= 0 else None)
    elif extra > 0:
        result.update(kind="gpu_never_wins", images=None)
    else:
        result.update(kind="gpu_wins_initially_only", images=None,
                      last_winning_image=1 + math.floor(-extra / -saving))
    return result


def trial(executable, case, backend, samples, batch=0, idle=0):
    config = case["config"][backend]
    cmd = [executable, config["backend"], case["image"], case["poni"], case["mask"],
           case["method"], 3000, config["threads"], case["gpu_threads"], samples,
           batch, idle, HERE / "integrate.metal"]
    start = time.perf_counter_ns()
    proc = subprocess.Popen([str(a) for a in cmd], cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    events = []
    for line in proc.stdout:
        wall_ms = (time.perf_counter_ns() - start) / 1e6
        event = json.loads(line)
        event["parent_wall_ms"] = wall_ms
        if event["kind"].startswith("idle"):
            print(f"  {event}", flush=True)
        events.append(event)
    error = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd}: {error}")
    first = next(e for e in events if e["kind"] == "first")
    result = next(e for e in events if e["kind"] == "result")
    assert result["accuracy_passed"]
    return {"first": first, "result": result, "batch": next((e for e in events if e["kind"] == "batch"), None)}


def summarize(case):
    summary = {}
    for backend in ["best_cpu", "default_cpu", "gpu"]:
        samples = case["trials"][backend]
        summary[backend] = {
            "process_to_first_pattern_ms": stats([r["first"]["parent_wall_ms"] for r in samples]),
            "internal_to_first_pattern_ms": stats([r["first"]["internal_ms"] for r in samples]),
            "steady_ms": {},
        }
        modes = [r["input"] for r in samples[0]["result"]["steady"]]
        for mode in modes:
            values = [next(v["timing"]["median_ms"] for v in r["result"]["steady"] if v["input"] == mode) for r in samples]
            summary[backend]["steady_ms"][mode] = stats(values)
        first = summary[backend]["process_to_first_pattern_ms"]["median"]
        frame = summary[backend]["steady_ms"]["f64_convert"]["median"]
        summary[backend]["derived_setup_overhead_ms"] = first - frame
    for baseline in ["best_cpu", "default_cpu"]:
        cpu, gpu = summary[baseline], summary["gpu"]
        cross = crossover(cpu["process_to_first_pattern_ms"]["median"],
                          gpu["process_to_first_pattern_ms"]["median"],
                          cpu["steady_ms"]["f64_convert"]["median"],
                          gpu["steady_ms"]["f64_convert"]["median"])
        cross["per_repeat"] = [crossover(c["first"]["parent_wall_ms"],g["first"]["parent_wall_ms"],
            c["result"]["steady"][0]["timing"]["median_ms"],g["result"]["steady"][0]["timing"]["median_ms"])
            for c,g in zip(case["trials"][baseline],case["trials"]["gpu"])]
        summary[f"gpu_vs_{baseline}"] = cross
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=HERE / "setup-results.json")
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    assert args.repeats > 0 and args.samples > 0 and args.idle_seconds >= 0
    run(["cargo", "build", "--release", "--manifest-path", HERE / "setup/Cargo.toml",
         "--target-dir", REPO / "target/metal-setup"])
    old = json.loads((HERE / "results.json").read_text())
    cases = []
    for c in old["cases"]:
        if c["name"] == "edge" or (args.small_only and c["name"].startswith("eiger")):
            continue
        def best(layout=None):
            return min((r for r in c["cpu"] if r["errors"] and (layout is None or r["layout"] == layout)), key=lambda r:r["timing"]["median_ms"])
        chosen, default = best(), best("default_csc")
        g = min((r for r in c["metal"]["results"] if r["errors"] and r["compensated"] and r["input"] == "f64_convert"), key=lambda r:r["wall"]["median_ms"])
        eiger = c["name"].startswith("eiger")
        case = {"name":c["name"],"image":c["image"],"method":c["method"],
                "poni":"docs/benchmarks/2026-11-05-bslz4/calibration.poni" if eiger else "test_data/v1.poni",
                "mask":"test_data/eiger_0000.mask" if eiger else "-", "gpu_threads":g["threads"],
                "config":{}, "trials":{"best_cpu":[],"default_cpu":[],"gpu":[]}}
        for key, selected in [("best_cpu",chosen),("default_cpu",default)]:
            case["config"][key] = {"backend":{"default_csc":0,"fast_csr":1,"strict_csr":2}[selected["layout"]],
                                   "layout":selected["layout"],"threads":selected["threads"]}
        case["config"]["gpu"] = {"backend":3,"threads":chosen["threads"]}
        cases.append(case)
    files = [HERE / "setup/Cargo.toml", HERE / "setup/Cargo.lock", HERE / "setup/src/lib.rs",
             HERE / "setup/SetupBench.swift", HERE / "integrate.metal", Path(__file__)]
    output = {"date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "commit":run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
              "source_sha256":{str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              "steady_baseline":"original results.json selections, remeasured in the new harness",
              "cases":cases}
    def save():
        args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    with tempfile.TemporaryDirectory(prefix="dioptrin-metal-setup-") as scratch:
        executable = Path(scratch) / "SetupBench"
        run(["swiftc","-O",HERE/"setup/SetupBench.swift",REPO/"target/metal-setup/release/libdioptrin_metal_setup.a","-o",executable])
        rng = random.Random(20261110)
        for case in cases:
            print(f"Startup: {case['name']}",flush=True)
            for repeat in range(args.repeats):
                backends = ["best_cpu","default_cpu","gpu"]
                rng.shuffle(backends)
                for backend in backends:
                    record = trial(executable,case,backend,args.samples)
                    case["trials"][backend].append(record)
                    print(f"  repeat {repeat+1} {backend}: {record['first']['parent_wall_ms']:.1f} ms to first pattern",flush=True)
                save()
            case["summary"] = summarize(case)
            print(f"  crossover: {case['summary']['gpu_vs_best_cpu']}",flush=True)
            save()
        # Validate the model at batch sizes bracketing the predicted crossover,
        # only where a positive startup premium is amortized by faster frames.
        for case in cases:
            cross = case["summary"]["gpu_vs_best_cpu"]
            if cross["kind"] == "gpu_wins_from" and 1 < cross["images"] <= 3000:
                case["batch_validation"] = []
                for n in sorted({max(1,cross["images"]//2),cross["images"]*2}):
                    for repeat in range(3):
                        pair = {b:trial(executable,case,b,args.samples,batch=n) for b in ["best_cpu","gpu"]}
                        case["batch_validation"].append({"n":n,"repeat":repeat,"trials":pair})
                        print(f"Batch {case['name']} N={n}: CPU {pair['best_cpu']['batch']['parent_wall_ms']:.1f} ms, GPU {pair['gpu']['batch']['parent_wall_ms']:.1f} ms",flush=True)
                save()
        if args.idle_seconds > 0:
            case = next((c for c in cases if c["name"] == "eiger_pixel_split"),cases[-1])
            print(f"Idle test: {case['name']}, same Metal plan, 3 x {args.idle_seconds}s",flush=True)
            output["idle_test"] = {"case":case["name"],"trial":trial(executable,case,"gpu",args.samples,idle=args.idle_seconds)}
            save()
    print(f"Saved {args.output}",flush=True)


if __name__ == "__main__":
    main()
