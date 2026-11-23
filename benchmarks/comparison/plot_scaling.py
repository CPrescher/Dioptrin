#!/usr/bin/env python3
"""Plot measured CPU scaling and a separate GPU detector-size comparison."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = {"pilatus": "Pilatus 1M", "synthetic-4m": "Synthetic 4M", "synthetic-9m": "Synthetic 9M"}
CPU = {
    "dioptrin-pyfai_full": ("Dioptrin full", "#0072b2", "o"),
    "pyfai-full": ("pyFAI full CSR", "#d55e00", "s"),
    "dioptrin-supersampled": ("Dioptrin supersampled", "#009e73", "^"),
    "azint": ("AZINT subpixel", "#9b5597", "D"),
    "azint-cached": ("AZINT cached normalisation", "#6e3b82", "s"),
}
# Earlier name of azint-cached (results recorded before the rename).
CPU_ALIASES = {"azint-prepared-mask": "azint-cached"}
GPU = {
    "dioptrin-cuda-single": ("Dioptrin CUDA, per frame", "#009e73"),
    "dioptrin-cuda-pageable": ("Dioptrin CUDA, ordinary", "#56b4e9"),
    "dioptrin-cuda": ("Dioptrin CUDA, pinned", "#0072b2"),
    "pyfai-opencl": ("pyFAI OpenCL, ordinary", "#d55e00"),
}
GPU_KERNELS = {
    "dioptrin-cuda-kernels": ("Dioptrin CUDA", "#0072b2"),
    "pyfai-opencl-kernels": ("pyFAI OpenCL", "#d55e00"),
}


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cpu(directory, output, mode, processes=1):
    """1D rows of `processes` processes (rows recorded before these fields
    existed are 1D single-process); 2D cakes and other process counts are
    left out rather than pooled into the same points."""
    rates = defaultdict(list)
    frame_counts = defaultdict(set)
    for path in sorted((directory / "cases").glob("*.result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("dimension", "1d") != "1d" or row.get("processes", 1) != processes:
            continue
        if row["error_mode"] == mode and row["workload"] == "repeated":
            variant = CPU_ALIASES.get(row["variant"], row["variant"])
            rates[row["dataset"], variant, row["workers"]].append(row["frames_per_s"])
            frame_counts[row["dataset"]].add(row["frames"])
    datasets = [d for d in DATASETS if any(key[0] == d for key in rates)]
    if not datasets:
        raise ValueError(
            f"No 1D {processes}-process CPU results in {directory} for {mode}"
        )
    fig, axes = plt.subplots(
        2, len(datasets), figsize=(3.7 * len(datasets), 5.8),
        sharey="row", squeeze=False, layout="constrained",
    )
    handles = {}
    for i, dataset in enumerate(datasets):
        through, speed = axes[:, i]
        through.set_title(DATASETS[dataset])
        budgets = sorted({k[2] for k in rates if k[0] == dataset})
        speed.plot(budgets, budgets, linestyle=":", color="0.5", linewidth=1, zorder=0)
        for variant, (label, color, marker) in CPU.items():
            workers = [w for w in budgets if (dataset, variant, w) in rates]
            if not workers:
                continue
            samples = [rates[dataset, variant, w] for w in workers]
            centers = np.array([median(values) for values in samples])
            ranges = np.array([centers - [min(x) for x in samples], [max(x) for x in samples] - centers])
            handle = through.errorbar(
                workers, centers, yerr=ranges, color=color, marker=marker,
                linewidth=1.5, markersize=4, capsize=2,
            )
            handles[label] = handle
            if workers[0] == 1 and len(frame_counts[dataset]) == 1:
                speed.plot(workers, centers / centers[0], color=color, marker=marker,
                           linewidth=1.5, markersize=4)
        through.set_yscale("log")
        speed.set_ylim(bottom=0)
        for ax in (through, speed):
            ax.set_xticks(budgets)
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["right", "top"]].set_visible(False)
        speed.set_xlabel("CPU worker budget")
    axes[0, 0].set_ylabel("Throughput (frames/s)")
    axes[1, 0].set_ylabel("Speedup over one worker")
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2, frameon=False)
    suffix = "" if processes == 1 else f"-p{processes}"
    save(fig, output, f"cpu-scaling-{mode}{suffix}")


def plot_gpu(records, output, mode, series, metric, filename):
    rows = json.loads(records.read_text(encoding="utf-8"))
    error_mode = mode == "poisson"
    records = [r for r in rows["records"] if r["backend"] in series and r["errors"] == error_mode
               and r["layout"] in ("block", "csr")]
    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in records)]
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    x = np.arange(len(datasets))
    backends = {key: value for key, value in series.items()
                if any(r["backend"] == key for r in records)}
    width = .75 / len(backends)
    for i, (backend, (label, color)) in enumerate(backends.items()):
        values = []
        errors = []
        for dataset in datasets:
            candidates = [r for r in records if r["dataset"] == dataset
                          and r["backend"] == backend]
            fps = [r["frames_per_s"] for r in candidates]
            center = median(fps)
            values.append(center)
            errors.append([center - min(fps), max(fps) - center])
        bars = ax.bar(x + (i - (len(backends) - 1) / 2) * width, values, width,
                      yerr=np.asarray(errors).T,
                      capsize=3, label=label, color=color)
        ax.bar_label(bars, labels=[f"{x:,.0f}" for x in values], padding=3, fontsize=8)
    ax.set_xticks(x, [DATASETS[d] for d in datasets])
    ax.set_ylabel(metric)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", alpha=.22)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False)
    save(fig, output, f"{filename}-{mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("intensity", "poisson"), default="poisson")
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="CPU plot: use the 1D rows of this process count (compare.py "
        "--processes); default 1. 2D rows are never plotted.",
    )
    args = parser.parse_args()
    if not (args.cpu or args.gpu):
        parser.error("pass at least one of --cpu and --gpu")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        plot_cpu(args.cpu, args.output, args.mode, args.processes)
    if args.gpu:
        plot_gpu(args.gpu, args.output, args.mode, GPU,
                 "Host-to-pattern throughput (frames/s)", "gpu-detector-scaling")
        plot_gpu(args.gpu, args.output, args.mode, GPU_KERNELS,
                 "GPU kernel throughput (frames/s)", "gpu-kernel-scaling")


if __name__ == "__main__":
    main()
