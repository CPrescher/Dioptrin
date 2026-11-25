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
try:
    import resource
except ImportError:
    resource = None
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fixtures import MANIFEST, manifest, prepare, sha256
from hardware import capture
from provenance import azint_signature, verify_azint_build, verify_build


ROOT = Path(__file__).resolve().parent.parent.parent
VARIANTS = (
    "dioptrin-pixel_split",
    "dioptrin-supersampled",
    "dioptrin-pyfai_full",
    "azint",
    "azint-cached",
    "pyfai-full",
    "pyfai-bbox",
    "pyfai-full-omp",
    "pyfai-full-opencl",
)
# Dioptrin's CUDA 1D/2D plans (Integrator.prepare_cuda1d / prepare_cuda2d):
# (method, call). "frame" submits one frame per public call, like pyFAI's
# OpenCL engine; "batch" uses the plan's multi-frame batch1d / batch2d.
CUDA_VARIANTS = {
    "dioptrin-pixel_split-cuda": ("pixel_split", "frame"),
    "dioptrin-pyfai_full-cuda": ("pyfai_full", "frame"),
    "dioptrin-pixel_split-cuda-batch": ("pixel_split", "batch"),
    "dioptrin-pyfai_full-cuda-batch": ("pyfai_full", "batch"),
}
# Dioptrin's Metal 1D/2D plans on macOS (Integrator.prepare_metal1d / prepare_metal2d).
METAL_VARIANTS = {
    "dioptrin-pixel_split-metal": ("pixel_split", "frame"),
    "dioptrin-pyfai_full-metal": ("pyfai_full", "frame"),
    "dioptrin-pixel_split-metal-batch": ("pixel_split", "batch"),
    "dioptrin-pyfai_full-metal-batch": ("pyfai_full", "batch"),
}
DIOPTRIN_GPU_VARIANTS = {**CUDA_VARIANTS, **METAL_VARIANTS}
# f32 pixels and weights with double-word compensated sums, f64 denominators,
# division and output: the counterpart of pyFAI's OpenCL CSR engine.
CUDA_PRECISION = "f32_compensated"
GPU_PRECISIONS = ("f32_compensated", "f32", "f64")
INPUT_DTYPES = ("float64", "float32")
# Untimed agreement tolerances of GPU output with CPU Integrator by precision.
CUDA_AGREEMENT = dict(radial_rtol=1e-12, rtol=1e-5, atol_of_max=1e-5)
GPU_AGREEMENT_TOLERANCES = {
    "f64": dict(radial_rtol=1e-12, rtol=1e-11, atol_of_max=1e-11),
    "f32_compensated": CUDA_AGREEMENT,
    "f32": dict(radial_rtol=1e-12, rtol=1e-4, atol_of_max=1e-4),
}
CUDA_NO_API = "installed dioptrin has no CUDA API"
METAL_NO_API = "installed dioptrin has no Metal API"
# The default variant list; the CUDA/Metal rows are opt-in (--variants), so a
# default run never initialises a GPU device.
DEFAULT_VARIANTS = VARIANTS
# Every selectable variant.
VARIANTS += tuple(CUDA_VARIANTS) + tuple(METAL_VARIANTS)
# The GPU probe (device initialisation) runs in its own child process; a crash,
# signal or a child running longer than this marks the backend unavailable.
CUDA_PROBE_TIMEOUT_S = 600


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
)
# Exact pyFAI (split, algo, impl); fallbacks are rejected.
PYFAI_METHODS = {
    "pyfai-full": ("full", "csr", "cython"),
    "pyfai-bbox": ("bbox", "csr", "cython"),
    "pyfai-full-omp": ("full", "csr", "cython"),
    "pyfai-full-opencl": ("full", "csr", "opencl"),
}
# One engine integrating frames in sequence: pyFAI parallelises inside the
# CSR kernel (OpenMP threads, or the OpenCL device) instead of over frames.
SEQUENTIAL_VARIANTS = ("pyfai-full-omp", "pyfai-full-opencl", *DIOPTRIN_GPU_VARIANTS)
GPU_VARIANTS = ("pyfai-full-opencl", *DIOPTRIN_GPU_VARIANTS)
CONTROL = "@@"
# Multi-process blocks start this long after the parent sends the command.
START_LEAD_SECONDS = 0.1
SPIN_SECONDS = 0.03

# 2D ("cake") cases. Every library integrates onto chi in [-180, 180) degrees
# (pyFAI's convention, seam at +-180) and the fixture's 2theta range.
DEFAULT_CAKES = ("2000x360", "1000x72")
AZIMUTH_RANGE_DEG = (-180.0, 180.0)
# Variants whose 2D subpixel count comes from --subpixels (per side).
SUBPIXEL_VARIANTS = ("dioptrin-supersampled", "azint", "azint-cached")
DEFAULT_SUBPIXELS_2D = (10,)
DIOPTRIN_2D_PATHS = ("sequential", "stream", "batch")
# How --dioptrin-2d-path batch splits the CPU budget B in batch2d over arrays:
# frames = num_workers=B, num_threads=1 (B frames side by side, one thread
# each, no nested parallelism; the arrangement of the azint/pyFAI frame
# threads); nested = neither argument (all frames on the global Rayon pool of
# RAYON_NUM_THREADS = B threads, parallel inside frames too).
DIOPTRIN_2D_BATCH_SPLITS = ("frames", "nested")
# The CPU Dioptrin 1D call over the in-memory frame list (--dioptrin-1d-path):
# global = Integrator.batch1d(frames, n, errors=...) with neither num_workers
# nor num_threads, on the global Rayon pool of RAYON_NUM_THREADS = budget
# threads (the library's default call; 4-frame tiled kernels when the call has
# at least 4 frames per pool thread); split = num_workers=budget,
# num_threads=1, which since Dioptrin 0.15.0 runs `budget` frames at a time on
# one-thread pools that the call builds itself (in 0.14 it ran as global).
DIOPTRIN_1D_PATHS = ("global", "split")
# How a timed Dioptrin call turns its IntegrationResult into NumPy arrays:
# array = the zero-copy read-only accessors (intensity_array, errors_array),
# list = np.asarray of the list-returning attributes (builds without them).
DIOPTRIN_RESULT_CONVERSIONS = ("array", "list")
# Cross-library pairs in numerical.json besides the dioptrin-pixel_split reference.
COUNTERPARTS_2D = (
    ("pyfai-full", "dioptrin-pyfai_full"),
    ("pyfai-full-omp", "pyfai-full"),
    ("pyfai-full-opencl", "pyfai-full"),
    ("azint", "dioptrin-supersampled"),
    ("azint-cached", "dioptrin-supersampled"),
    ("dioptrin-pixel_split-cuda", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-cuda", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-cuda-batch", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-cuda-batch", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-metal", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-metal", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-metal-batch", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-metal-batch", "dioptrin-pyfai_full"),
)
# 1D (variant, counterpart) pairs in numerical.json besides the
# dioptrin-pixel_split reference: GPU rows against their CPU method.
COUNTERPARTS_1D = (
    ("dioptrin-pixel_split-cuda", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-cuda", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-cuda-batch", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-cuda-batch", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-metal", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-metal", "dioptrin-pyfai_full"),
    ("dioptrin-pixel_split-metal-batch", "dioptrin-pixel_split"),
    ("dioptrin-pyfai_full-metal-batch", "dioptrin-pyfai_full"),
    ("pyfai-full-opencl", "pyfai-full"),
)
# What each library does when 2D Poisson uncertainties are requested.
UNCERTAINTY_2D = {
    "dioptrin": "native",
    "pyfai": "native (error_model='poisson')",
    # azint.py (pinned commit) divides errors by norm before the 1D/2D branch
    # and errors_2d by norm again: sigma_2d = sqrt(sum w^2 I) / norm**2.
    "azint": "native; azint.py divides the 2D Poisson sigma by the normalisation "
    "twice (sqrt(var)/norm/norm), so it is not comparable with the other libraries",
}
# Memory model for --max-memory-fraction (bytes unless noted). Per process:
# interpreter and libraries, frames, per-pixel geometry arrays (including
# table-construction temporaries), one sparse table per engine, one batch of
# results (per output layer: intensity, and sigma with errors) and the
# validation sample plus reference. Calibrated on the Pilatus 1M and Eiger 4M
# smoke runs (README): it overestimates the measured peak RSS by roughly
# 1.1-1.6x, deliberately.
MEMORY_MODEL = {
    "base_mib": 200,
    "geometry_bytes_per_pixel": {"dioptrin": 300, "pyfai": 150, "azint": 300},
    "table_bytes_per_nnz": {"dioptrin": 32, "pyfai": 40, "azint": 16},
    "output_bytes_per_bin": {"dioptrin": 8, "pyfai": 40, "azint": 16},
}


UNSUPPORTED_EXIT = 3


class Unsupported(Exception):
    """The library cannot produce what the case asks for (recorded, not faked)."""


def library(variant):
    """dioptrin, pyfai or azint."""
    return variant.split("-", 1)[0]


def dioptrin_method(variant):
    """Integrator method of a Dioptrin variant (CPU, CUDA, or Metal)."""
    if variant in DIOPTRIN_GPU_VARIANTS:
        return DIOPTRIN_GPU_VARIANTS[variant][0]
    if not variant.startswith("dioptrin-"):
        raise ValueError(f"{variant!r} is not a Dioptrin variant")
    return variant.removeprefix("dioptrin-")


def jsonable(value):
    """Tuples (e.g. from dioptrin.cuda_info) as lists, recursively."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def cuda_version(value):
    """13040 -> '13.4' (CUDA driver/runtime integer); (12, 8) -> '12.8'."""
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ".".join(map(str, value))
    return f"{value // 1000}.{value % 1000 // 10}"


def cuda_agreement(
    gpu,
    cpu,
    errors,
    *,
    precision=CUDA_PRECISION,
    backend="cuda",
    input_dtype="float64",
):
    """Compare GPU 1D outputs with CPU Integrator.integrate1d of the same frames.

    gpu and cpu are lists of (radial, intensity, sigma) per frame. Raises
    AssertionError beyond GPU_AGREEMENT_TOLERANCES[precision]; returns the
    observed differences.
    """
    import numpy as np

    label = backend.upper()
    tolerance = GPU_AGREEMENT_TOLERANCES.get(precision, CUDA_AGREEMENT)
    radial_exact = True
    found = {key: dict(max_abs=0.0, max_rel_nonzero=None) for key in ("intensity", "sigma")}
    for (g_radial, *g_values), (c_radial, *c_values) in zip(gpu, cpu, strict=True):
        np.testing.assert_allclose(
            g_radial, c_radial, rtol=tolerance["radial_rtol"], atol=0,
            err_msg=f"{label} radial axis differs from CPU integrate1d",
        )
        radial_exact &= bool(np.array_equal(g_radial, c_radial))
        for key, g, c in zip(("intensity", "sigma"), g_values, c_values):
            if key == "sigma" and not errors:
                continue
            g, c = np.asarray(g, dtype=float), np.asarray(c, dtype=float)
            scale = float(np.max(np.abs(c))) if c.size else 0.0
            np.testing.assert_allclose(
                g, c, rtol=tolerance["rtol"], atol=tolerance["atol_of_max"] * scale,
                err_msg=f"{label} {key} differs from CPU integrate1d",
            )
            difference = np.abs(g - c)
            entry = found[key]
            entry["max_abs"] = max(entry["max_abs"], float(difference.max(initial=0.0)))
            nonzero = c != 0
            if nonzero.any():
                relative = float(np.max(difference[nonzero] / np.abs(c[nonzero])))
                entry["max_rel_nonzero"] = max(entry["max_rel_nonzero"] or 0.0, relative)
    if not errors:
        found.pop("sigma")
    return dict(
        reference=f"CPU Integrator.integrate1d (same Integrator, {input_dtype} frames)",
        frames_checked=len(gpu),
        tolerances=dict(tolerance),
        radial_exactly_equal=radial_exact,
        **found,
    )


def gpu_agreement_2d(
    gpu,
    cpu,
    errors,
    *,
    precision=CUDA_PRECISION,
    backend="cuda",
    input_dtype="float64",
):
    """Compare GPU 2D outputs with CPU Integrator.integrate2d of the same frames.

    gpu and cpu are lists of (radial, azimuthal, intensity, sigma) per frame.
    """
    import numpy as np

    label = backend.upper()
    tolerance = GPU_AGREEMENT_TOLERANCES.get(precision, CUDA_AGREEMENT)
    radial_exact = True
    azimuthal_exact = True
    found = {key: dict(max_abs=0.0, max_rel_nonzero=None) for key in ("intensity", "sigma")}
    for (g_rad, g_azi, *g_values), (c_rad, c_azi, *c_values) in zip(gpu, cpu, strict=True):
        np.testing.assert_allclose(
            g_rad, c_rad, rtol=tolerance["radial_rtol"], atol=0,
            err_msg=f"{label} 2D radial axis differs from CPU integrate2d",
        )
        np.testing.assert_allclose(
            g_azi, c_azi, rtol=tolerance["radial_rtol"], atol=1e-12,
            err_msg=f"{label} 2D azimuthal axis differs from CPU integrate2d",
        )
        radial_exact &= bool(np.array_equal(g_rad, c_rad))
        azimuthal_exact &= bool(np.array_equal(g_azi, c_azi))
        for key, g, c in zip(("intensity", "sigma"), g_values, c_values):
            if key == "sigma" and not errors:
                continue
            g, c = np.asarray(g, dtype=float), np.asarray(c, dtype=float)
            scale = float(np.max(np.abs(c))) if c.size else 0.0
            np.testing.assert_allclose(
                g, c, rtol=tolerance["rtol"], atol=tolerance["atol_of_max"] * scale,
                err_msg=f"{label} 2D {key} differs from CPU integrate2d",
            )
            difference = np.abs(g - c)
            entry = found[key]
            entry["max_abs"] = max(entry["max_abs"], float(difference.max(initial=0.0)))
            nonzero = c != 0
            if nonzero.any():
                relative = float(np.max(difference[nonzero] / np.abs(c[nonzero])))
                entry["max_rel_nonzero"] = max(entry["max_rel_nonzero"] or 0.0, relative)
    if not errors:
        found.pop("sigma")
    return dict(
        reference=f"CPU Integrator.integrate2d (same Integrator, {input_dtype} frames)",
        frames_checked=len(gpu),
        tolerances=dict(tolerance),
        radial_exactly_equal=radial_exact,
        azimuthal_exactly_equal=azimuthal_exact,
        **found,
    )


def parse_cake(text):
    """'2000x360' -> (2000, 360)."""
    try:
        radial, azimuthal = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(
            f"Cake sizes are RADIALxAZIMUTHAL, e.g. 2000x360, not {text!r}"
        )
    if radial < 1 or azimuthal < 1:
        raise ValueError(f"Cake bin counts must be positive: {text!r}")
    return radial, azimuthal


def roll_azint(array, n_azimuthal):
    """azint bins azimuth on [0, 360); move its rows onto [-180, 180).

    With n_azimuthal even, azint row j covers [j*s, (j+1)*s) and the harness
    row i covers [-180 + i*s, ...); they are the same angles for
    j = (i + n/2) mod n, i.e. a roll by n/2. Subpixel sampling has no seam, so
    this is an exact relabelling, not a resampling.
    """
    import numpy as np

    if n_azimuthal % 2:
        raise Unsupported(
            "azint rows can be mapped onto [-180, 180) only for even n_azimuthal"
        )
    return np.roll(array, n_azimuthal // 2, axis=0)


def azint_azimuth_axis(axis, n_azimuthal):
    import numpy as np

    rolled = roll_azint(np.asarray(axis), n_azimuthal)
    return np.where(rolled >= 180.0, rolled - 360.0, rolled)


def total_memory_bytes():
    """Physical RAM, or a smaller cgroup (Slurm) limit when one is readable."""
    if sys.platform == "win32" or not hasattr(os, "sysconf"):
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit() and 0 < int(value) < total:
            total = int(value)
    return total


def estimate_memory_mib(case, meta, estimates):
    """Expected peak RSS (MiB, summed over processes) of a 2D case."""
    lib = library(case["variant"])
    pixels = meta["shape"][0] * meta["shape"][1]
    processes = case.get("processes", 1)
    per_process = case["workers"] // processes
    distinct = 1 if case["workload"] == "repeated" else meta["distinct_frames"]
    frames = case["frames"] // processes
    held = min(distinct, frames) if processes > 1 else distinct
    bins = case["n_radial"] * case["n_azimuthal"]
    layers = 2 if case["errors"] else 1
    cake = estimates[case["cake"]]
    if case["variant"] in SUBPIXEL_VARIANTS:
        nnz = cake["subpixel_nnz"][str(case["subpixels"])]
    else:
        nnz = cake["polygon_nnz"]
    if lib == "pyfai":
        engines = 1 if case["variant"] in SEQUENTIAL_VARIANTS else per_process
    else:
        engines = 1
    model = MEMORY_MODEL
    pixel_bytes = 4 if case.get("input_dtype") == "float32" else 8
    one = (
        model["base_mib"] * 2**20
        + held * pixels * pixel_bytes
        + pixels * model["geometry_bytes_per_pixel"][lib]
        + engines * nnz * model["table_bytes_per_nnz"][lib]
        + frames * bins * model["output_bytes_per_bin"][lib] * layers
        + 2 * distinct * bins * 8 * layers
    )
    return dict(
        estimated_memory_mib=round(processes * one / 2**20, 1),
        estimated_table_nnz=int(nnz),
        estimated_engines=engines * processes,
    )


def case_threads(variant, count, processes=1):
    """Per-case thread budget; with processes > 1 it is per process.

    Dioptrin gets a Rayon pool of `count`. azint's sparse-matrix construction
    (generate_matrix) is its only OpenMP region; its SpMV is serial and runs on
    `count` Python frame threads, so OMP_NUM_THREADS=count parallelises only
    set-up. pyFAI's Python-thread rows stay at 1: its macOS wheel has no
    OpenMP, and more OpenMP threads would change what is measured there.
    `pyfai-full-omp` is pyFAI's own CPU mode: one CSR engine with
    OMP_NUM_THREADS=count. The OpenCL row uses one host thread.
    """
    if count % processes:
        raise ValueError("The CPU budget must be divisible by the process count")
    per = str(count // processes)
    omp = variant.startswith("azint") or variant == "pyfai-full-omp"
    return {
        "OMP_NUM_THREADS": per if omp else "1",
        "RAYON_NUM_THREADS": per if variant.startswith("dioptrin-") else "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "FALSE",
    }


def same_bits(a, b):
    import numpy as np

    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class DioptrinOutput:
    """One frame as a timed Dioptrin call hands it back to the harness.

    `intensity` and, with Poisson errors, `sigma` are NumPy float64 arrays
    created inside the timed call (1D: (n,), 2D: (n_azimuthal, n_radial)),
    which is what pyFAI's and azint's timed calls return. `result` is the
    IntegrationResult: the radial and azimuthal axes are per-integrator
    constants and are read from it outside timing, as azint's `radial_axis`.
    `filename` is the BatchResult label ("array[i]") of batch calls, checked
    against the frame position during validation.
    """

    __slots__ = ("result", "intensity", "sigma", "filename")

    def __init__(self, result, intensity, sigma, filename=None):
        self.result = result
        self.intensity = intensity
        self.sigma = sigma
        self.filename = filename


def dioptrin_result_conversion(result):
    """'array' when this IntegrationResult has the zero-copy accessors."""
    return "array" if hasattr(result, "intensity_array") else "list"


LIST_CONVERSION_ERROR = (
    "the installed Dioptrin build has no IntegrationResult *_array accessors "
    "(intensity_array, errors_array; Dioptrin 0.15.2 and older wheels lack "
    "them), so every timed Dioptrin call would build NumPy arrays from Python "
    "lists, serially under the GIL (for a 2000x360 cake about 32 ms against "
    "1.6 ms of integration), and the rows would mostly time list building. "
    "Install Dioptrin from a revision that has the accessors (rerun setup.py "
    "from this checkout), or pass --allow-list-conversion to time the list "
    "conversion anyway"
)
LIST_CONVERSION_WARNING = (
    "LIST CONVERSION: this Dioptrin build has no *_array accessors; Dioptrin "
    "rows time np.asarray over Python lists (serial, under the GIL), which can "
    "dominate the timed call. Do not compare these rows with array rows or "
    "with pyFAI/azint as integration throughput."
)


def dioptrin_has_result_arrays():
    """Whether the installed dioptrin's IntegrationResult has the accessors."""
    import dioptrin

    cls = getattr(dioptrin, "IntegrationResult", None)
    return cls is not None and hasattr(cls, "intensity_array")


def dioptrin_case_conversion(result, case):
    """dioptrin_result_conversion, refusing 'list' unless the case allows it.

    The run refuses list conversion before any case (main); this repeats the
    check in every case process and rank in case the build changed since.
    """
    conversion = dioptrin_result_conversion(result)
    if conversion == "list" and not case.get("allow_list_conversion"):
        raise RuntimeError("Refusing to time Dioptrin: " + LIST_CONVERSION_ERROR)
    return conversion


def dioptrin_converter(conversion, errors, shape=None):
    """IntegrationResult -> DioptrinOutput, run inside the timed call.

    `shape` is the 2D cake shape (n_azimuthal, n_radial): the list attributes
    of 2D results are flat, the accessors are already shaped (a mismatch fails
    validation rather than being reshaped away).
    """
    import numpy as np

    if conversion == "array":

        def convert(result, filename=None):
            return DioptrinOutput(
                result,
                result.intensity_array,
                result.errors_array if errors else None,
                filename,
            )

    elif conversion == "list":

        def convert(result, filename=None):
            intensity = np.asarray(result.intensity, dtype=np.float64)
            sigma = result.errors if errors else None
            if sigma is not None:
                sigma = np.asarray(sigma, dtype=np.float64)
            if shape is not None:
                intensity = intensity.reshape(shape)
                if sigma is not None:
                    sigma = sigma.reshape(shape)
            return DioptrinOutput(result, intensity, sigma, filename)

    else:
        raise ValueError(f"Unknown Dioptrin result conversion {conversion!r}")
    return convert


def dioptrin_batch_converter(convert):
    """BatchResult -> DioptrinOutput (error check and unwrap, then convert)."""

    def unwrap(item):
        if not item.is_ok():
            raise RuntimeError(item.error)
        return convert(item.result, item.filename)

    return unwrap


def dioptrin_axis(result, name, conversion):
    """An axis of an IntegrationResult as float64 (outside timing), or None."""
    import numpy as np

    value = getattr(result, f"{name}_array" if conversion == "array" else name)
    return None if value is None else np.asarray(value, dtype=np.float64)


def check_dioptrin_output(output, index, shape, errors):
    """What the timed call produced: float64 ndarrays of the expected shape,
    in frame order for batch calls. Missing sigma is left to the callers."""
    import numpy as np

    if not isinstance(output, DioptrinOutput):
        raise AssertionError(f"Timed Dioptrin call returned {type(output).__name__}")
    if index is not None and output.filename not in (None, f"array[{index}]"):
        raise AssertionError(f"Dioptrin batch result {index} is {output.filename!r}")
    for name in ("intensity", "sigma") if errors else ("intensity",):
        value = getattr(output, name)
        if value is None:
            continue
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise AssertionError(
                f"Dioptrin {name} is {type(value).__name__} "
                f"{getattr(value, 'dtype', '')}, not a float64 ndarray"
            )
        if value.shape != shape:
            raise AssertionError(f"Dioptrin {name} shape {value.shape}, expected {shape}")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contiguous_chunks(frames, workers):
    indexed = list(enumerate(frames))
    size, extra = divmod(len(frames), workers)
    return [
        indexed[i * size + min(i, extra) : (i + 1) * size + min(i + 1, extra)]
        for i in range(workers)
    ]


def process_share(total, processes, rank):
    """Contiguous logical frame indices of one process (all for rank None)."""
    if rank is None:
        return list(range(total))
    if total % processes:
        raise ValueError("Frame count must be divisible by the process count")
    share = total // processes
    return list(range(rank * share, (rank + 1) * share))


OPENMP_SYMBOLS = (
    b"GOMP_parallel",
    b"__kmpc_fork_call",
    b"omp_get_max_threads",
    b"vcomp140",
    b"vcomp140d",
    b"_vcomp_fork",
)


def pyfai_openmp():
    """Whether pyFAI's full-split CSR extension was compiled with OpenMP.

    pyFAI has no runtime flag for this. The extension's import table names the
    OpenMP runtime entry points (GOMP_* for GCC builds, __kmpc_* for LLVM),
    and on Linux the loaded runtime library also appears in /proc/self/maps.
    """
    import pyFAI.ext.splitPixelFullCSR as module

    data = Path(module.__file__).read_bytes()
    symbols = [s.decode() for s in OPENMP_SYMBOLS if s in data]
    loaded = []
    maps = Path("/proc/self/maps")
    if maps.is_file():
        loaded = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(k in line for k in ("libgomp", "libomp", "libiomp"))
            }
        )
    return dict(
        openmp=bool(symbols),
        module=module.__file__,
        symbols=symbols,
        loaded_runtime=loaded,
    )


def opencl_target(device):
    """pyFAI method selection for --opencl-device (gpu, cpu, acc, any, or P:D)."""
    if ":" in device:
        platform_id, device_id = (int(v) for v in device.split(":"))
        return dict(target=(platform_id, device_id))
    if device not in ("gpu", "cpu", "acc", "any"):
        raise ValueError(
            f"--opencl-device must be gpu, cpu, acc, any or P:D, not {device}"
        )
    return {} if device == "any" else dict(target_type=device)


def opencl_method(device, dim=1):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: F401  (registry)
    from pyFAI.method_registry import IntegrationMethod

    found = IntegrationMethod.select_method(
        dim=dim,
        split="full",
        algo="csr",
        impl="opencl",
        degradable=False,
        **opencl_target(device),
    )
    return found[0] if found else None


def opencl_device_info(device):
    """Platform, device, and precision facts for a pyopencl Device."""
    import pyopencl

    extensions = device.extensions.split()
    return dict(
        platform=device.platform.name.strip(),
        platform_version=device.platform.version.strip(),
        device=device.name.strip(),
        device_type=pyopencl.device_type.to_string(device.type),
        device_version=device.version.strip(),
        driver_version=device.driver_version.strip(),
        device_fp64=("cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions),
        # pyFAI 2026.x azim_csr: float32 buffers/coefficients and float32
        # double-word (float2, compensated) accumulators in ocl_azim_CSR.cl,
        # whatever the device's fp64 support.
        single_precision_accumulation=True,
        accumulation="float32 arithmetic with double-word (float2) compensated "
        "accumulators (pyFAI ocl_azim_CSR.cl); float32 input and coefficients",
    )


def cuda_probe(device):
    """dioptrin.cuda_info(device) with tuples as lists; never raises.

    Run only in a fresh process: it initialises CUDA, which is not fork-safe.
    """
    try:
        import dioptrin

        info = dioptrin.cuda_info(device)
    except (ImportError, AttributeError):
        return dict(compiled=False, available=False, reason=CUDA_NO_API, device=device)
    except BaseException as error:  # a Rust panic surfaces as a BaseException
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return dict(
            compiled=True,
            available=False,
            reason=f"dioptrin.cuda_info failed: {error!r}",
            device=device,
        )
    info = jsonable(dict(info))
    if not info.get("available") and not info.get("reason"):
        info["reason"] = "no reason reported"
    return info


def metal_probe(device):
    """dioptrin.metal_info(device) with tuples as lists; never raises."""
    try:
        import dioptrin

        info = dioptrin.metal_info(device)
    except (ImportError, AttributeError):
        return dict(compiled=False, available=False, reason=METAL_NO_API, device=device)
    except BaseException as error:  # a Rust panic surfaces as a BaseException
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return dict(
            compiled=True,
            available=False,
            reason=f"dioptrin.metal_info failed: {error!r}",
            device=device,
        )
    info = jsonable(dict(info))
    if not info.get("available") and not info.get("reason"):
        info["reason"] = "no reason reported"
    return info


def probe(device):
    """Availability of the optional pyFAI variants, in a fresh process."""
    result = {}
    try:
        result["pyfai_openmp"] = pyfai_openmp()
    except Exception as error:  # report, don't crash the parent
        result["pyfai_openmp"] = dict(openmp=False, reason=repr(error))
    opencl = dict(available=False, requested_device=device)
    try:
        import pyopencl  # noqa: F401
    except ImportError as error:
        opencl["reason"] = f"pyopencl not importable: {error}"
    else:
        try:
            method = opencl_method(device)
            if method is None:
                opencl["reason"] = f"no OpenCL device matches {device!r}"
            else:
                from pyFAI.opencl import ocl

                platform_id, device_id = method.target
                info = opencl_device_info(
                    ocl.platforms[platform_id].devices[device_id]._device
                )
                opencl.update(available=True, target=list(method.target), **info)
                opencl["available_2d"] = opencl_method(device, dim=2) is not None
        except Exception as error:
            opencl["reason"] = repr(error)
    result["opencl"] = opencl
    print(json.dumps(result))


CONVENTION_CAKE = (100, 36)
# Asymmetric azimuthal test pattern (pyFAI chi, degrees): no rotation, mirror
# or flip of chi maps it onto itself.
CONVENTION_BANDS = ((20.0, 80.0, 100.0), (-150.0, -120.0, 30.0))


def convention_rows(n_azimuthal):
    """Rows expected bright, medium and dark for CONVENTION_BANDS."""
    step = 360.0 / n_azimuthal
    lower = [AZIMUTH_RANGE_DEG[0] + step * i for i in range(n_azimuthal)]
    inside = [
        [i for i, a in enumerate(lower) if lo <= a and a + step <= hi]
        for lo, hi, _ in CONVENTION_BANDS
    ]
    touching = {
        i
        for lo, hi, _ in CONVENTION_BANDS
        for i, a in enumerate(lower)
        if a < hi and a + step > lo
    }
    dark = [i for i in range(n_azimuthal) if i not in touching]
    return inside[0], inside[1], dark


def check_convention(cakes, flats, n_azimuthal):
    """Row medians of image/flat cakes per library against the pattern.

    Dividing by the cake of a flat (all-ones) image removes each library's
    corrections and estimator, leaving ~100, ~31 and ~1 in the band rows.
    """
    import numpy as np

    bright, medium, dark = convention_rows(n_azimuthal)
    report = {}
    for name, cake in cakes.items():
        flat = flats[name]
        ratio = np.where(flat > 0, cake / np.where(flat > 0, flat, 1.0), np.nan)
        rows = np.array(
            [np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in ratio]
        )
        ok = bool(
            np.all(rows[bright] > 60)
            and np.all((rows[medium] > 15) & (rows[medium] < 50))
            and np.nanmax(rows[dark]) < 5
        )
        report[name] = dict(
            passed=ok,
            row_medians=[None if not np.isfinite(v) else float(v) for v in rows],
        )
    return report


def preflight_2d(fixture, cakes, subpixels, variants, output):
    """Per-dataset 2D checks, run in a fresh process before any 2D case.

    1. Azimuth convention: an asymmetric chi pattern is integrated by every
       requested library onto the harness grid; all must put it in the same rows.
    2. Fallback radial axes for older Dioptrin builds whose 2D results omit
       the axis. For those builds, pixel_split/supersampled use the default
       edges of the 1D grid of the same method.
    3. Table-size estimates for the memory model, from pixel-corner spans.
    """
    import numpy as np
    import pyFAI

    fixture = Path(fixture)
    meta = json.loads((fixture / "fixture.json").read_text())
    mask = np.load(fixture / "mask.npy")
    image0 = np.load(fixture / "frames.npy", mmap_mode="r")[0]
    shape = tuple(meta["shape"])
    poni = str(fixture / "geometry.poni")
    ai = pyFAI.load(poni)
    ai.detector.mask = mask
    libraries = {library(v) for v in variants}
    # Common 2D radial edges. Dioptrin's Integrator ignores radial_range in 2D
    # for pixel_split and supersampled; pixel_split uses the extremes of the
    # pixel-corner 2theta (plus 10 ulp), read here from its 1D default grid,
    # which follows the same rule. Every other 2D case is given these edges.
    corners = ai.array_from_unit(shape, typ="corner", unit="2th_deg")
    if "dioptrin" in libraries:
        import dioptrin

        engine = dioptrin.Integrator(poni, method="pixel_split", unit="2th_deg")
        engine.set_mask(mask)
        axis = np.asarray(engine.integrate1d(np.ascontiguousarray(image0), 1000).radial)
        step = (axis[-1] - axis[0]) / (len(axis) - 1)
        lo, hi = float(axis[0] - step / 2), float(axis[-1] + step / 2)
        range_source = (
            "Dioptrin pixel_split default edges (pixel-corner 2theta extremes)"
        )
        del engine
    else:
        lo, hi = float(corners[..., 0].min()), float(corners[..., 0].max())
        range_source = "pyFAI pixel-corner 2theta extremes (float32)"
    # Per-pixel 2theta and chi spans (from the corners) for the table estimates.
    valid = mask == 0
    tth = corners[..., 0][valid].astype(np.float64)
    angle = corners[..., 1][valid].astype(np.float64)
    del corners
    d_tth = tth.max(axis=1) - tth.min(axis=1)
    d_chi = angle.max(axis=1) - angle.min(axis=1)
    wrapped = np.mod(angle, 2 * np.pi)
    d_chi = np.minimum(d_chi, wrapped.max(axis=1) - wrapped.min(axis=1))
    del tth, angle, wrapped
    chi = np.degrees(ai.center_array(shape, unit="chi_rad"))
    pattern = np.ones(shape)
    for low, high, value in CONVENTION_BANDS:
        pattern[(chi > low) & (chi < high)] = value
    pattern[mask != 0] = 0
    flat = np.where(mask != 0, 0.0, 1.0)
    del chi
    nr, na = CONVENTION_CAKE
    cakes_out, flats, axes = {}, {}, {}
    for name, image, store in (("pattern", pattern, cakes_out), ("flat", flat, flats)):
        result = ai.integrate2d(
            image,
            nr,
            na,
            unit="2th_deg",
            method=("full", "csr", "cython"),
            radial_range=(lo, hi),
            azimuth_range=AZIMUTH_RANGE_DEG,
            correctSolidAngle=True,
            polarization_factor=0.965,
        )
        store["pyfai"] = np.asarray(result.intensity)
        axes["pyfai"] = np.asarray(result.azimuthal)
    if "dioptrin" in libraries:
        import dioptrin

        for method in ("pixel_split", "pyfai_full", "supersampled"):
            engine = dioptrin.Integrator(
                poni,
                method=method,
                unit="2th_deg",
                n_ss=2,
                polarization_factor=0.965,
                radial_range=(lo, hi),
            )
            engine.set_mask(mask)
            for image, store in ((pattern, cakes_out), (flat, flats)):
                result = engine.integrate2d(
                    image, nr, na, azimuthal_range=(-math.pi, math.pi)
                )
                store[f"dioptrin-{method}"] = np.asarray(result.intensity).reshape(
                    na, nr
                )
            axes[f"dioptrin-{method}"] = np.degrees(np.asarray(result.azimuthal))
    if "azint" in libraries:
        import azint
        from azint.detector import Detector

        geometry = meta["geometry"].copy()
        pixel1, pixel2 = geometry.pop("pixel1"), geometry.pop("pixel2")
        engine = azint.AzimuthalIntegrator(
            azint.Poni(det=Detector(pixel1, pixel2, shape), **geometry),
            n_splitting=2,
            radial_bins=np.linspace(lo, hi, nr + 1),
            azimuth_bins=np.linspace(0.0, 360.0, na + 1),
            unit="2th",
            mask=mask,
            solid_angle=True,
            polarization_factor=0.965,
        )
        cakes_out["azint"] = roll_azint(engine.integrate(pattern)[2], na)
        flats["azint"] = roll_azint(engine.integrate(flat)[2], na)
        axes["azint"] = azint_azimuth_axis(engine.azimuth_axis, na)
    report = check_convention(cakes_out, flats, na)
    for name, axis in axes.items():
        report[name]["max_azimuth_axis_difference_deg"] = float(
            np.max(np.abs(axis - axes["pyfai"]))
        )
        if report[name]["max_azimuth_axis_difference_deg"] > 1e-4:
            report[name]["passed"] = False
    convention = dict(
        cake=list(CONVENTION_CAKE),
        azimuth_range_deg=list(AZIMUTH_RANGE_DEG),
        bands_deg=[list(b) for b in CONVENTION_BANDS],
        reference="pyFAI chi (center_array chi_rad), seam at +-180 degrees",
        azint_mapping="azint azimuth bins [0, 360), rows rolled by n_azimuthal/2",
        libraries=report,
        passed=all(r["passed"] for r in report.values()),
    )
    del pattern, flat, cakes_out, flats

    # dioptrin-supersampled's numerical companion cake uses MultiIntegrator
    # (explicit edges). Check that MultiIntegrator reproduces
    # Integrator.integrate2d where both use the same edges (pixel_split).
    multi_check = None
    if "dioptrin-supersampled" in variants:
        import dioptrin

        engine = dioptrin.Integrator(
            poni,
            method="pixel_split",
            unit="2th_deg",
            polarization_factor=0.965,
            radial_range=(lo, hi),
        )
        engine.set_mask(mask)
        frame = np.ascontiguousarray(image0)
        span = (-math.pi, math.pi)
        a = np.asarray(
            engine.integrate2d(frame, nr, na, azimuthal_range=span).intensity
        )
        b = np.asarray(
            dioptrin.MultiIntegrator([engine])
            .integrate2d([frame], nr, na, azimuthal_range=span)
            .intensity
        )
        scale = np.maximum(np.abs(b), 1e-300)
        multi_check = dict(
            cake=list(CONVENTION_CAKE),
            method="pixel_split",
            max_relative_difference=float(np.max(np.abs(a - b) / scale)),
            relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
        )
        del engine

    radial_axes = {}
    methods = {
        dioptrin_method(v) for v in variants if v.startswith("dioptrin-")
    }
    radial_counts = sorted({parse_cake(c)[0] for c in cakes})
    if methods:
        import dioptrin

        for method in sorted(methods):
            counts = subpixels if method == "supersampled" else [None]
            for n_ss in counts:
                engine = dioptrin.Integrator(
                    poni,
                    method=method,
                    unit="2th_deg",
                    n_ss=n_ss or meta["n_splitting"],
                    # Same rule as the 2D grid: only pyfai_full takes the range.
                    radial_range=(lo, hi) if method == "pyfai_full" else None,
                )
                engine.set_mask(mask)
                for n in radial_counts:
                    axis = list(
                        engine.integrate1d(np.ascontiguousarray(image0), n).radial
                    )
                    key = method if n_ss is None else f"{method}-ss{n_ss}"
                    radial_axes.setdefault(key, {})[str(n)] = axis

    estimates = {}
    for text in cakes:
        n_rad, n_azi = parse_cake(text)
        cells = (d_tth / ((hi - lo) / n_rad) + 1.0) * (
            d_chi / (2 * np.pi / n_azi) + 1.0
        )
        estimates[text] = dict(
            polygon_nnz=float(cells.sum()),
            subpixel_nnz={
                str(n): float(np.minimum(cells, n * n).sum()) for n in subpixels
            },
        )
    dump(
        output,
        dict(
            radial_range_2d=[lo, hi],
            radial_range_2d_source=range_source,
            radial_range_1d=meta["radial_range"],
            convention=convention,
            multiintegrator_check=multi_check,
            dioptrin_radial_axes=radial_axes,
            table_estimates=estimates,
            unmasked_pixels=int(valid.sum()),
        ),
    )


def setup_case(case, rank=None):
    """Load inputs, build and warm the engine, and return the measurement hooks.

    With rank None the case runs in this one process. With a rank, this
    process is one of `processes`: it copies only its own frames into memory it
    allocates (first touch, so NUMA binding places them locally) and builds its
    own tables with `workers / processes` threads.
    """
    import numpy as np

    if case.get("dimension") == "2d":
        return setup_case_2d(case, rank)
    fixture = Path(case["fixture"])
    meta = json.loads((fixture / "fixture.json").read_text())
    stack = np.load(fixture / "frames.npy", mmap_mode="r")
    processes = case.get("processes", 1)
    distinct = 1 if case["workload"] == "repeated" else len(stack)
    logical = process_share(case["frames"], processes, rank)
    if rank is None:
        needed = list(range(distinct))
        unique = {u: stack[u] for u in needed}
    else:
        needed = sorted({i % distinct for i in logical})
        unique = {u: np.array(stack[u]) for u in needed}
    del stack
    if case.get("input_dtype") == "float32":
        cast = {}
        for u, frame in unique.items():
            f32 = np.ascontiguousarray(frame, dtype=np.float32)
            if not np.array_equal(f32.astype(frame.dtype), frame):
                raise AssertionError(
                    f"Casting frame {u} from {frame.dtype} to float32 was not lossless"
                )
            cast[u] = f32
        unique = cast
    image = unique[needed[0]]
    mask = np.load(fixture / "mask.npy")
    frames = [unique[i % distinct] for i in logical]
    # Fault every input page in before setup/timing. All cases see the same
    # immutable frame sequence; frame generation/decoding is never timed.
    for frame in unique.values():
        float(np.sum(frame))
    n = case["bins"]
    workers = case["workers"] // processes
    errors = case["errors"]
    variant = case["variant"]
    bounds = meta["radial_range"]
    pool = None
    stock_check = None
    cuda_check = None
    info = {}
    started = time.perf_counter()

    if variant.startswith("dioptrin-"):
        import dioptrin

        ai = dioptrin.Integrator(
            str(fixture / "geometry.poni"),
            method=dioptrin_method(variant),
            unit="2th_deg",
            n_ss=meta["n_splitting"],
            polarization_factor=0.965,
            radial_range=tuple(bounds),
        )
        ai.set_mask(mask)

        # The default array batch call (--dioptrin-1d-path global) runs on
        # the global Rayon pool: it must be exactly the CPU budget
        # (RAYON_NUM_THREADS, set by case_threads).
        if dioptrin.rayon_num_threads() != workers:
            raise RuntimeError(
                "Dioptrin Rayon pool does not match requested CPU budget"
            )

    if variant in DIOPTRIN_GPU_VARIANTS:
        # One host thread drives one device (w1, p1). The plan snapshots the
        # Integrator's mask, corrections, method, unit and radial range and
        # holds the table on the device. The timed call ends with NumPy
        # float64 intensity (and sigma) per frame, as for the CPU rows.
        backend = "cuda" if variant in CUDA_VARIANTS else "metal"
        kind = DIOPTRIN_GPU_VARIANTS[variant][1]
        precision = case.get("gpu_precision", CUDA_PRECISION)
        if backend == "cuda":
            plan = ai.prepare_cuda1d(
                image.shape, n, precision=precision, device=case["cuda_device"]
            )
        else:
            plan = ai.prepare_metal1d(
                image.shape, n, precision=precision, device=case["metal_device"]
            )

        def integrate_one(frame):
            return plan.integrate1d(frame, errors=errors)

        first = integrate_one(image)
        conversion = dioptrin_case_conversion(first, case)
        convert = dioptrin_converter(conversion, errors)

        if kind == "frame":

            def integrate_frames(images):
                return [convert(integrate_one(frame)) for frame in images]

        else:

            def integrate_frames(images):
                return [convert(r) for r in plan.batch1d(images, errors=errors)]

        def run():
            return integrate_frames(frames)

        def extract(output):
            check_dioptrin_output(output, None, (n,), errors)
            return (
                dioptrin_axis(output.result, "radial", conversion),
                output.intensity,
                output.sigma if errors else None,
            )

        extract(convert(first))
        setup_first = time.perf_counter() - started
        del first
        info["dioptrin_result_conversion"] = conversion
        baseline = [extract(r) for r in integrate_frames([unique[u] for u in needed])]

        def cuda_check(sample):
            """Untimed, after setup_ready: device facts, agreement with the CPU
            Integrator and, for batch rows, batch1d == per-frame integrate1d."""
            device = (
                dioptrin.cuda_info(plan.device)
                if backend == "cuda"
                else dioptrin.metal_info(plan.device)
            )
            gpu = [
                (sample[0][i], sample[1][i], sample[2][i] if errors else None)
                for i in range(len(needed))
            ]
            cpu = [
                extract(
                    convert(
                        ai.integrate1d(np.ascontiguousarray(unique[u]), n, errors=errors)
                    )
                )
                for u in needed
            ]
            identical = None
            if kind == "batch":
                images = [unique[u] for u in needed]
                batched = [convert(r) for r in plan.batch1d(images, errors=errors)]
                single = [convert(integrate_one(frame)) for frame in images]
                identical = len(batched) == len(single) and all(
                    same_bits(a, b)
                    for x, y in zip(batched, single)
                    for a, b in zip(extract(x), extract(y))
                )
                if not identical:
                    raise AssertionError(
                        f"{backend.upper()} batch1d differs from per-frame integrate1d"
                    )
            backend_info = dict(
                call="plan.integrate1d per frame"
                if kind == "frame"
                else "plan.batch1d over all frames",
                precision=plan.precision,
                requested_precision=precision,
                device=plan.device,
                device_name=plan.device_name,
                max_batch_frames=plan.max_batch_frames,
                table_entries=plan.table_entries,
                input_dtype=str(np.asarray(image).dtype),
                cpu_agreement=cuda_agreement(
                    gpu,
                    cpu,
                    errors,
                    precision=precision,
                    backend=backend,
                    input_dtype=str(np.asarray(image).dtype),
                ),
                batch_bit_identical=identical,
            )
            if backend == "cuda":
                backend_info.update(
                    compute_capability=device.get("compute_capability"),
                    total_memory_bytes=device.get("total_memory_bytes"),
                    driver_version=device.get("driver_version"),
                    nvrtc_version=device.get("nvrtc_version"),
                    bindings_version=device.get("bindings_version"),
                )
            else:
                backend_info.update(
                    has_unified_memory=device.get("has_unified_memory"),
                    recommended_max_working_set_size=device.get(
                        "recommended_max_working_set_size"
                    ),
                )
            info[backend] = jsonable(backend_info)
            if info[backend]["precision"] != precision:
                raise AssertionError(
                    f"{backend.upper()} plan precision {plan.precision!r}, expected {precision!r}"
                )

    elif variant.startswith("dioptrin-"):
        # --dioptrin-1d-path (DIOPTRIN_1D_PATHS): "global" is the library's
        # default call on the global pool of `workers` threads (checked
        # above); "split" passes num_workers=workers, num_threads=1.
        path = case.get("dioptrin_1d_path") or "global"
        if path == "global":
            shape = {}
        elif path == "split":
            shape = dict(num_workers=workers, num_threads=1)
        else:
            raise ValueError(f"Unknown Dioptrin 1D path {path!r}")

        def batch(images):
            return ai.batch1d(images, n, errors=errors, **shape)

        first = batch([image])[0]
        if not first.is_ok():
            raise RuntimeError(first.error)
        conversion = dioptrin_case_conversion(first.result, case)
        unwrap = dioptrin_batch_converter(dioptrin_converter(conversion, errors))

        def integrate_frames(images):
            # Inside the timed call: unwrap every BatchResult and hold NumPy
            # float64 intensity (and sigma) for every frame.
            return [unwrap(item) for item in batch(images)]

        def run():
            return integrate_frames(frames)

        def extract(output):
            check_dioptrin_output(output, None, (n,), errors)
            return (
                dioptrin_axis(output.result, "radial", conversion),
                output.intensity,
                output.sigma if errors else None,
            )

        extract(unwrap(first))
        setup_first = time.perf_counter() - started
        del first
        info["dioptrin_result_conversion"] = conversion
        baseline = [extract(r) for r in integrate_frames([unique[u] for u in needed])]
    else:
        if variant in ("azint", "azint-cached"):
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
            if variant == "azint-cached":
                # Masked pixels are already excluded from the sparse matrix, and
                # the constructor cached norm = A @ 1. With mask=None, integrate()
                # reuses that norm instead of multiplying the frame by the
                # inverted mask and recomputing A @ (1 - mask) on every call.
                # Shallow per-thread copies below inherit mask=None and only
                # read the shared norm array.
                ai.mask = None

                def stock_check(images, cached):
                    stock = copy.copy(ai)
                    stock.mask = mask
                    for frame, value in zip(images, cached):
                        reference = stock.integrate(frame)
                        for index in (0, 1):
                            if not same_bits(reference[index], value[index]):
                                raise AssertionError(
                                    "azint-cached differs from stock azint output"
                                )
                    return True

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
            method = PYFAI_METHODS[variant]
            if variant != "pyfai-full-opencl":
                info["pyfai_openmp"] = pyfai_openmp()
                if variant == "pyfai-full-omp" and not info["pyfai_openmp"]["openmp"]:
                    raise RuntimeError(
                        "pyFAI's CSR extension has no OpenMP; pyfai-full-omp is meaningless here"
                    )
            if variant == "pyfai-full-opencl":
                selected = opencl_method(case["opencl_device"])
                if selected is None:
                    raise RuntimeError(
                        f"No pyFAI OpenCL device for {case['opencl_device']!r}"
                    )
                target = tuple(selected.target)
            else:
                selected, target = method, None
                if not IntegrationMethod.select_method(
                    dim=1,
                    split=method[0],
                    algo="csr",
                    impl=method[2],
                    degradable=False,
                ):
                    raise RuntimeError(
                        f"Requested exact pyFAI method unavailable: {method}"
                    )

            def integrate(engine, frame):
                return engine.integrate1d(
                    frame,
                    n,
                    unit="2th_deg",
                    method=selected,
                    radial_range=tuple(bounds),
                    correctSolidAngle=True,
                    polarization_factor=0.965,
                    error_model="poisson" if errors else None,
                    safe=True,
                )

            def extract(result):
                used = result.method.method
                if (used.split, used.algo, used.impl) != method or (
                    target is not None and tuple(used.target) != target
                ):
                    raise RuntimeError(f"pyFAI fell back from {selected} to {used}")
                return (
                    np.asarray(result.radial),
                    np.asarray(result.intensity),
                    result.sigma if errors else None,
                )

        extract(integrate(ai, image))
        setup_first = time.perf_counter() - started
        if variant == "pyfai-full-opencl":
            engine = next(
                e.engine for e in ai.engines.values() if hasattr(e.engine, "ctx")
            )
            info["opencl"] = dict(
                target=list(target), **opencl_device_info(engine.ctx.devices[0])
            )
        baseline = [extract(integrate(ai, unique[u])) for u in needed]
        threads = 1 if variant in SEQUENTIAL_VARIANTS else workers
        engines = [ai]
        for _ in range(1, threads):
            engine = copy.copy(ai)
            if variant.startswith("pyfai-"):
                # Share immutable geometry arrays, but each frame worker owns
                # its CSR engine and scratch buffers. Never concurrently use ai.
                engine.engines = {}
            integrate(engine, image)
            engines.append(engine)
        pool = ThreadPoolExecutor(max_workers=threads)
        chunks = contiguous_chunks(frames, threads)

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
    row = {u: position for position, u in enumerate(needed)}.__getitem__
    baseline = sample
    if case.get("reference"):
        with np.load(case["reference"]) as reference:
            baseline = (
                reference["radial"].copy(),
                reference["intensity"].copy(),
                reference["sigma"].copy() if errors else np.array([]),
            )
        row = int  # the reference holds every distinct frame in order
    elif rank is not None:
        raise RuntimeError("Multi-process cases need a one-worker reference")

    def validate(results):
        if len(results) != len(frames):
            raise AssertionError("Wrong output frame count")
        for index, result in enumerate(results):
            if isinstance(result, DioptrinOutput):
                # The arrays the timed call produced, in frame order.
                check_dioptrin_output(result, index, (n,), errors)
            radial, intensity, sigma = extract(result)
            expected = row(logical[index] % distinct)
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
    if stock_check is not None:
        # Outside setup and timing: the cached-normalisation results must be
        # bit-identical (intensity and sigma) to stock masked azint.
        checked = min(len(needed), len(frames))
        stock_check(frames[:checked], warm_result[:checked])
    del warm_result
    if cuda_check is not None:
        cuda_check(sample)  # outside setup and timing
        # The check ran CPU integrations while the device idled; one more
        # untimed pass so the first timed block starts warm, as for
        # pyfai-full-opencl (setup_ready is unchanged).
        run()
    negative = np.argwhere(sample[2] < 0).tolist() if errors else []
    return dict(
        meta=meta,
        run=run,
        validate=validate,
        pool=pool,
        sample=sample,
        needed=needed,
        frames=len(frames),
        distinct=distinct,
        setup_first=setup_first,
        setup_ready=setup_ready,
        stock_checked=stock_check is not None,
        negative_sigma=[[needed[i], b] for i, b in negative],
        info=info,
    )


def load_inputs(case, rank):
    """Fixture, frames and mask of a case, as in the 1D setup (pages faulted in)."""
    import numpy as np

    fixture = Path(case["fixture"])
    meta = json.loads((fixture / "fixture.json").read_text())
    stack = np.load(fixture / "frames.npy", mmap_mode="r")
    processes = case.get("processes", 1)
    distinct = 1 if case["workload"] == "repeated" else len(stack)
    logical = process_share(case["frames"], processes, rank)
    if rank is None:
        needed = list(range(distinct))
        unique = {u: stack[u] for u in needed}
    else:
        needed = sorted({i % distinct for i in logical})
        unique = {u: np.array(stack[u]) for u in needed}
    del stack
    if case.get("input_dtype") == "float32":
        cast = {}
        for u, frame in unique.items():
            f32 = np.ascontiguousarray(frame, dtype=np.float32)
            if not np.array_equal(f32.astype(frame.dtype), frame):
                raise AssertionError(
                    f"Casting frame {u} from {frame.dtype} to float32 was not lossless"
                )
            cast[u] = f32
        unique = cast
    mask = np.load(fixture / "mask.npy")
    frames = [unique[i % distinct] for i in logical]
    for frame in unique.values():
        float(np.sum(frame))
    return fixture, meta, mask, unique, needed, frames, logical, distinct


def pyfai_engines(engine):
    """The integration engines a pyFAI AzimuthalIntegrator has built."""
    return [e.engine for e in engine.engines.values()]


def threaded_runner(ai, integrate, image, frames, threads, variant):
    """Per-thread engines (shallow copies) and a pool over contiguous chunks.

    pyFAI copies get their own engine dictionary, so every frame thread builds
    and owns its CSR engine and scratch buffers; azint copies share the sparse
    matrix (read only), as in the 1D cases.
    """
    engines = [ai]
    for _ in range(1, threads):
        engine = copy.copy(ai)
        if variant.startswith("pyfai-"):
            engine.engines = {}
        integrate(engine, image)
        engines.append(engine)
    pool = ThreadPoolExecutor(max_workers=threads)
    chunks = contiguous_chunks(frames, threads)

    def chunk(task):
        engine, images = task
        return [(index, integrate(engine, frame)) for index, frame in images]

    def run():
        results = [None] * len(frames)
        for group in pool.map(chunk, zip(engines, chunks)):
            for index, result in group:
                results[index] = result
        return results

    return engines, pool, run


def setup_case_2d(case, rank=None):
    """Build and warm a 2D (cake) case; same contract as the 1D setup_case.

    Every library gets the fixture's 2theta range with n_radial bins, chi in
    [-180, 180) degrees with n_azimuthal bins, the same mask, solid angle and
    polarization 0.965. Peak RSS is also sampled after the inputs are loaded
    and when all workers are ready, before any validation reference is read.
    """
    import numpy as np

    fixture, meta, mask, unique, needed, frames, logical, distinct = load_inputs(
        case, rank
    )
    image = unique[needed[0]]
    nr, na = case["n_radial"], case["n_azimuthal"]
    processes = case.get("processes", 1)
    workers = case["workers"] // processes
    errors = case["errors"]
    variant = case["variant"]
    lib = library(variant)
    bounds = case["radial_range_2d"]
    subpixels = case.get("subpixels")
    pool = None
    stock_check = None
    gpu_check_2d = None
    conversion = None  # Dioptrin rows: DIOPTRIN_RESULT_CONVERSIONS

    def convert(result):  # Dioptrin rows replace it with dioptrin_converter
        return result

    info = dict(rss_inputs_mib=peak_rss(), uncertainty_support=UNCERTAINTY_2D[lib])
    started = time.perf_counter()

    if lib == "dioptrin":
        import dioptrin

        method = dioptrin_method(variant)
        ai = dioptrin.Integrator(
            str(fixture / "geometry.poni"),
            method=method,
            unit="2th_deg",
            n_ss=subpixels or meta["n_splitting"],
            polarization_factor=0.965,
            radial_range=tuple(bounds),
        )
        ai.set_mask(mask)
        if dioptrin.rayon_num_threads() != workers:
            raise RuntimeError(
                "Dioptrin Rayon pool does not match requested CPU budget"
            )
        span = tuple(math.radians(v) for v in AZIMUTH_RANGE_DEG)
        fallback_radial_axis = np.asarray(case["dioptrin_radial_axis"])

        if variant in DIOPTRIN_GPU_VARIANTS:
            backend = "cuda" if variant in CUDA_VARIANTS else "metal"
            kind = DIOPTRIN_GPU_VARIANTS[variant][1]
            precision = case.get("gpu_precision", CUDA_PRECISION)
            if backend == "cuda":
                plan = ai.prepare_cuda2d(
                    image.shape,
                    nr,
                    na,
                    azimuthal_range=span,
                    precision=precision,
                    device=case["cuda_device"],
                )
            else:
                plan = ai.prepare_metal2d(
                    image.shape,
                    nr,
                    na,
                    azimuthal_range=span,
                    precision=precision,
                    device=case["metal_device"],
                )

            def integrate(engine, frame):
                return engine.integrate2d(frame, errors=errors)

            first = integrate(plan, image)
            conversion = dioptrin_case_conversion(first, case)
            convert = dioptrin_converter(conversion, errors, (na, nr))

            # The timed call ends with NumPy float64 cakes (n_azimuthal,
            # n_radial), and sigma with Poisson errors, for every frame.
            if kind == "frame":

                def run():
                    return [
                        convert(plan.integrate2d(frame, errors=errors))
                        for frame in frames
                    ]

            else:

                def run():
                    return [convert(r) for r in plan.batch2d(frames, errors=errors)]

            def extract(output):
                check_dioptrin_output(output, None, (na, nr), errors)
                radial = dioptrin_axis(output.result, "radial", conversion)
                sigma = None
                if errors:
                    if output.sigma is None:
                        raise Unsupported("Dioptrin GPU returned no 2D errors")
                    sigma = output.sigma
                return (
                    fallback_radial_axis if radial is None else radial,
                    np.degrees(dioptrin_axis(output.result, "azimuthal", conversion)),
                    output.intensity,
                    sigma,
                )

            setup_first = time.perf_counter() - started
            engines = [plan]
            info["table_nnz"] = plan.table_entries
            info["table_nnz_source"] = f"dioptrin.{backend}_2d_plan.table_entries"

            def gpu_check_2d(sample):
                """Untimed, after setup_ready: device facts, agreement with CPU
                Integrator.integrate2d and, for batch rows, batch2d == integrate2d."""
                device = (
                    dioptrin.cuda_info(plan.device)
                    if backend == "cuda"
                    else dioptrin.metal_info(plan.device)
                )
                gpu_frames = [
                    (
                        sample["radial"],
                        sample["azimuthal"],
                        sample["intensity"][i],
                        sample["sigma"][i] if errors else None,
                    )
                    for i in range(len(needed))
                ]
                cpu_frames = [
                    extract(
                        convert(
                            ai.integrate2d(
                                np.ascontiguousarray(unique[u]),
                                nr,
                                na,
                                errors=errors,
                                azimuthal_range=span,
                            )
                        )
                    )
                    for u in needed
                ]
                identical = None
                if kind == "batch":
                    images = [unique[u] for u in needed]
                    batched = [convert(r) for r in plan.batch2d(images, errors=errors)]
                    single = [convert(integrate(plan, frame)) for frame in images]
                    identical = len(batched) == len(single) and all(
                        same_bits(a, b)
                        for x, y in zip(batched, single)
                        for a, b in zip(extract(x), extract(y))
                    )
                    if not identical:
                        raise AssertionError(
                            f"{backend.upper()} batch2d differs from per-frame integrate2d"
                        )
                backend_info = dict(
                    call="plan.integrate2d per frame"
                    if kind == "frame"
                    else "plan.batch2d over all frames",
                    precision=plan.precision,
                    requested_precision=precision,
                    device=plan.device,
                    device_name=plan.device_name,
                    max_batch_frames=plan.max_batch_frames,
                    table_entries=plan.table_entries,
                    input_dtype=str(np.asarray(image).dtype),
                    cpu_agreement=gpu_agreement_2d(
                        gpu_frames,
                        cpu_frames,
                        errors,
                        precision=precision,
                        backend=backend,
                        input_dtype=str(np.asarray(image).dtype),
                    ),
                    batch_bit_identical=identical,
                )
                if backend == "cuda":
                    backend_info.update(
                        compute_capability=device.get("compute_capability"),
                        total_memory_bytes=device.get("total_memory_bytes"),
                        driver_version=device.get("driver_version"),
                        nvrtc_version=device.get("nvrtc_version"),
                        bindings_version=device.get("bindings_version"),
                    )
                else:
                    backend_info.update(
                        has_unified_memory=device.get("has_unified_memory"),
                        recommended_max_working_set_size=device.get(
                            "recommended_max_working_set_size"
                        ),
                    )
                info[backend] = jsonable(backend_info)
                if info[backend]["precision"] != precision:
                    raise AssertionError(
                        f"{backend.upper()} 2D plan precision {plan.precision!r}, expected {precision!r}"
                    )
        else:
            def integrate(engine, frame):
                return engine.integrate2d(
                    frame, nr, na, errors=errors, azimuthal_range=span
                )

            first = integrate(ai, image)
            setup_first = time.perf_counter() - started
            conversion = dioptrin_case_conversion(first, case)
            convert = dioptrin_converter(conversion, errors, (na, nr))
            # Every path's timed call ends with NumPy float64 cakes
            # (n_azimuthal, n_radial), and sigma with Poisson errors, for every
            # frame (DioptrinOutput); the axes are read outside timing.

            if case["dioptrin_2d_path"] == "sequential":
                # One frame at a time; each integrate2d runs its row-parallel SpMV
                # on the Rayon pool (RAYON_NUM_THREADS = CPU budget), no copies.
                def run():
                    return [convert(integrate(ai, frame)) for frame in frames]

            elif case["dioptrin_2d_path"] == "batch":
                # One synchronous batch2d call over the array list (Dioptrin builds
                # whose batch2d accepts arrays); the arrays are borrowed, not
                # copied. "frames": `workers` frames at a time, each in a Rayon pool
                # of one thread built by the call (needs a build that honours
                # num_workers/num_threads for arrays; older array builds ignored
                # them). "nested": neither argument, so all frames run on the global
                # pool (RAYON_NUM_THREADS = CPU budget, checked above). Either way
                # the process computes on `workers` threads.
                split = case["dioptrin_2d_batch_split"]
                if split == "frames":
                    shape = dict(num_workers=workers, num_threads=1)
                elif split == "nested":
                    shape = {}
                else:
                    raise ValueError(f"Unknown Dioptrin 2D batch split {split!r}")

                def batch(images):
                    return ai.batch2d(
                        images, nr, na, errors=errors, azimuthal_range=span, **shape
                    )

                try:
                    batch([image])
                except TypeError as error:
                    # Unsupported builds are skipped, never substituted.
                    raise Unsupported(
                        "--dioptrin-2d-path batch needs Integrator.batch2d over "
                        "NumPy arrays; the installed Dioptrin "
                        f"{importlib.metadata.version('dioptrin')} accepts TIFF "
                        f"paths only ({error})"
                    ) from None
                unwrap = dioptrin_batch_converter(convert)

                def run():
                    # BatchResult.result is read once per frame, inside timing.
                    return [unwrap(item) for item in batch(frames)]

            elif case["dioptrin_2d_path"] == "stream":
                # Frame-parallel stream: num_workers frames at once, one thread
                # each. The stream copies every array once before integrating it.
                def run():
                    results = [None] * len(frames)
                    for item in ai.batch2d_iter(
                        frames,
                        nr,
                        na,
                        errors=errors,
                        azimuthal_range=span,
                        num_workers=workers,
                        num_threads=1,
                    ):
                        if not item.is_ok():
                            raise RuntimeError(item.error)
                        results[int(item.filename.removeprefix("array[")[:-1])] = (
                            convert(item.result)
                        )
                    return results

            else:
                raise ValueError(f"Unknown Dioptrin 2D path {case['dioptrin_2d_path']!r}")

            def extract(output):
                check_dioptrin_output(output, None, (na, nr), errors)
                radial = dioptrin_axis(output.result, "radial", conversion)
                sigma = None
                if errors:
                    if output.sigma is None:
                        raise Unsupported("Dioptrin returned no 2D errors")
                    sigma = output.sigma
                return (
                    fallback_radial_axis if radial is None else radial,
                    np.degrees(dioptrin_axis(output.result, "azimuthal", conversion)),
                    output.intensity,
                    sigma,
                )

            engines = [ai]
            info["table_nnz_source"] = "not exposed by the Dioptrin Python API"
    else:
        if lib == "azint":
            import azint
            from azint.detector import Detector

            geometry = meta["geometry"].copy()
            pixel1, pixel2 = geometry.pop("pixel1"), geometry.pop("pixel2")
            poni = azint.Poni(det=Detector(pixel1, pixel2, image.shape), **geometry)
            ai = azint.AzimuthalIntegrator(
                poni,
                n_splitting=subpixels,
                radial_bins=np.linspace(*bounds, nr + 1),
                # azint measures azimuth on [0, 360); rows are rolled below.
                azimuth_bins=np.linspace(0.0, 360.0, na + 1),
                unit="2th",
                mask=mask,
                solid_angle=True,
                polarization_factor=0.965,
                error_model="poisson" if errors else None,
            )
            azimuth_axis = azint_azimuth_axis(ai.azimuth_axis, na)
            if variant == "azint-cached":
                ai.mask = None  # reuse the constructor's norm; see the 1D notes

                def stock_check(images, cached):
                    stock = copy.copy(ai)
                    stock.mask = mask
                    for frame, value in zip(images, cached):
                        reference = stock.integrate(frame)
                        for index in (2, 3):
                            if not same_bits(reference[index], value[index]):
                                raise AssertionError(
                                    "azint-cached differs from stock azint output"
                                )
                    return True

            def integrate(engine, frame):
                return engine.integrate(frame)

            def extract(result):
                sigma = None
                if errors:
                    if result[3] is None:
                        raise Unsupported("azint returned no 2D errors")
                    sigma = roll_azint(np.asarray(result[3]), na)
                return (
                    np.asarray(ai.radial_axis),
                    azimuth_axis,
                    roll_azint(np.asarray(result[2]), na),
                    sigma,
                )

            info["table_nnz_source"] = "not exposed by azint (_azint.Sparse)"
        else:
            import pyFAI
            from pyFAI.method_registry import IntegrationMethod

            ai = pyFAI.load(str(fixture / "geometry.poni"))
            ai.detector.mask = mask
            method = PYFAI_METHODS[variant]
            if variant != "pyfai-full-opencl":
                info["pyfai_openmp"] = pyfai_openmp()
                if variant == "pyfai-full-omp" and not info["pyfai_openmp"]["openmp"]:
                    raise RuntimeError(
                        "pyFAI's CSR extension has no OpenMP; pyfai-full-omp is meaningless here"
                    )
                selected, target = method, None
                if not IntegrationMethod.select_method(
                    dim=2, split=method[0], algo="csr", impl=method[2], degradable=False
                ):
                    raise RuntimeError(
                        f"Requested exact pyFAI 2D method unavailable: {method}"
                    )
            else:
                selected = opencl_method(case["opencl_device"], dim=2)
                if selected is None:
                    raise RuntimeError(
                        f"No pyFAI 2D OpenCL device for {case['opencl_device']!r}"
                    )
                target = tuple(selected.target)

            def integrate(engine, frame):
                return engine.integrate2d(
                    frame,
                    nr,
                    na,
                    unit="2th_deg",
                    method=selected,
                    radial_range=tuple(bounds),
                    azimuth_range=AZIMUTH_RANGE_DEG,
                    correctSolidAngle=True,
                    polarization_factor=0.965,
                    error_model="poisson" if errors else None,
                    safe=True,
                )

            def extract(result):
                used = result.method.method
                if (used.split, used.algo, used.impl) != method or (
                    target is not None and tuple(used.target) != target
                ):
                    raise RuntimeError(f"pyFAI fell back from {selected} to {used}")
                sigma = None
                if errors:
                    if result.sigma is None:
                        raise Unsupported("pyFAI returned no 2D sigma")
                    sigma = np.asarray(result.sigma)
                return (
                    np.asarray(result.radial),
                    np.asarray(result.azimuthal),
                    np.asarray(result.intensity),
                    sigma,
                )

        first = integrate(ai, image)
        setup_first = time.perf_counter() - started
        if variant == "pyfai-full-opencl":
            engine = next(e for e in pyfai_engines(ai) if hasattr(e, "ctx"))
            info["opencl"] = dict(
                target=list(target), **opencl_device_info(engine.ctx.devices[0])
            )
        threads = 1 if variant in SEQUENTIAL_VARIANTS else workers
        engines, pool, run = threaded_runner(
            ai, integrate, image, frames, threads, variant
        )

    def extract_run_result(index, result):
        if isinstance(result, DioptrinOutput):
            # The arrays the timed call produced, in frame order.
            check_dioptrin_output(result, index, (na, nr), errors)
        return extract(result)

    if conversion is not None:
        info["dioptrin_result_conversion"] = conversion
    first = extract(convert(first))
    warm_result = run()
    setup_ready = time.perf_counter() - started
    info["peak_rss_ready_mib"] = peak_rss()
    info["setup_rss_growth_mib"] = info["peak_rss_ready_mib"] - info["rss_inputs_mib"]
    if lib == "pyfai":
        counts = [getattr(e, "nnz", None) for a in engines for e in pyfai_engines(a)]
        if counts and all(c is not None for c in counts):
            info.update(
                table_nnz=int(counts[0]),
                table_nnz_total=int(sum(counts)),
                table_nnz_source=f"pyFAI engine.nnz, {len(counts)} engine(s)",
            )
        else:
            info["table_nnz_source"] = "not exposed by this pyFAI engine"

    # One output per distinct frame: the first time it appears in the warm
    # run, and (single process) any distinct frame the run did not reach.
    outputs = {}
    for index, result in enumerate(warm_result):
        outputs.setdefault(logical[index] % distinct, extract_run_result(index, result))
    for u in needed:
        if u not in outputs:
            outputs[u] = extract(convert(integrate(ai, unique[u])))
    for got, expected in zip(outputs[needed[0]][1:], first[1:]):
        if expected is not None:
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    keys = ("radial", "azimuthal", "intensity", "sigma")
    sample = {
        "radial": outputs[needed[0]][0],
        "azimuthal": outputs[needed[0]][1],
        "intensity": np.stack([outputs[u][2] for u in needed]),
        "sigma": np.stack([outputs[u][3] for u in needed]) if errors else np.array([]),
    }
    del outputs
    row = {u: position for position, u in enumerate(needed)}.__getitem__
    baseline = sample
    if case.get("reference"):
        if not Path(case["reference"]).is_file():
            raise RuntimeError(f"Missing one-worker reference {case['reference']}")
        with np.load(case["reference"]) as reference:
            baseline = {key: reference[key].copy() for key in keys}
        row = int  # the reference holds every distinct frame in order
    elif rank is not None:
        raise RuntimeError("Multi-process cases need a one-worker reference")

    def validate(results):
        if len(results) != len(frames):
            raise AssertionError("Wrong output frame count")
        for index, result in enumerate(results):
            radial, azimuthal, intensity, sigma = extract_run_result(index, result)
            expected = row(logical[index] % distinct)
            if intensity.shape != (na, nr) or not np.isfinite(intensity).all():
                raise AssertionError("Invalid integration output")
            if not np.any(intensity > 0):
                raise AssertionError("Empty integration output")
            np.testing.assert_allclose(radial, baseline["radial"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                azimuthal, baseline["azimuthal"], rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                intensity, baseline["intensity"][expected], rtol=1e-6, atol=1e-6
            )
            if errors:
                if sigma is None or not np.isfinite(sigma).all():
                    raise AssertionError("Invalid Poisson errors")
                np.testing.assert_allclose(
                    sigma, baseline["sigma"][expected], rtol=1e-6, atol=1e-6
                )

    validate(warm_result)
    if stock_check is not None:
        checked = min(len(needed), len(frames))
        stock_check(frames[:checked], warm_result[:checked])
    del warm_result
    if gpu_check_2d is not None:
        gpu_check_2d(sample)
        # Re-warm the GPU plan after the untimed CPU comparison, matching 1D.
        run()
    negative = np.argwhere(sample["sigma"] < 0).tolist() if errors else []

    def save(result_path):
        """The one-worker reference case (w1, t0, one process) keeps every
        distinct cake for the other cases' self-checks (removed after a
        successful run unless --keep-inputs) and its first cake, compressed,
        for numerical.json."""
        if case.get("reference"):
            return
        result_path = Path(result_path)
        first_cake = {
            key: (value[0] if key in ("intensity", "sigma") and value.size else value)
            for key, value in sample.items()
        }
        np.savez_compressed(result_path.with_suffix(".cake0.npz"), **first_cake)
        np.savez(result_path.with_suffix(".npz"), **sample)
        if variant == "dioptrin-supersampled":
            # Untimed, for numerical.json only: Integrator.integrate2d ignores
            # radial_range for supersampled, so its cakes sit on their own
            # radial grid. MultiIntegrator with this one integrator bins the
            # same subpixels onto the common explicit edges (for pixel_split it
            # reproduces Integrator.integrate2d; see preflight2d.json).
            import dioptrin

            common = dioptrin.MultiIntegrator([ai]).integrate2d(
                [np.ascontiguousarray(image)],
                nr,
                na,
                errors=errors,
                azimuthal_range=tuple(math.radians(v) for v in AZIMUTH_RANGE_DEG),
            )
            np.savez_compressed(
                result_path.with_suffix(".cake0common.npz"),
                radial=np.asarray(common.radial),
                azimuthal=np.degrees(np.asarray(common.azimuthal)),
                intensity=np.asarray(common.intensity).reshape(na, nr),
                sigma=np.asarray(common.errors).reshape(na, nr)
                if errors
                else np.array([]),
            )

    return dict(
        meta=meta,
        run=run,
        validate=validate,
        pool=pool,
        sample=sample,
        needed=needed,
        frames=len(frames),
        distinct=distinct,
        setup_first=setup_first,
        setup_ready=setup_ready,
        stock_checked=stock_check is not None,
        negative_sigma=[[needed[i], a, r] for i, a, r in negative],
        info=info,
        save=save,
    )


def peak_rss():
    if resource is None:
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / 2**20
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1 if sys.platform == "darwin" else 1024) / 2**20


def cpu_time():
    """User + system CPU seconds of this process (all threads)."""
    if resource is None:
        return time.process_time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def case_record(case, state):
    return dict(
        **{k: v for k, v in case.items() if k != "dioptrin_radial_axis"},
        setup_first_s=state["setup_first"],
        setup_ready_s=state["setup_ready"],
        thread_environment={k: os.environ.get(k) for k in THREAD_VARIABLES},
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS"),
        stock_azint_bit_identical=True if state["stock_checked"] else None,
        distinct_frames=state["distinct"],
        self_check="passed",
        **state["info"],
    )


def numerical_fields(negative):
    return dict(
        numerical_status="negative_sigma" if negative else "ok",
        negative_sigma_bins=len(negative),
        negative_sigma_bin_indices=negative,
    )


def summary_line(case, meta, rate):
    processes = case.get("processes", 1)
    variant = case["variant"]
    if case.get("dimension") == "2d":
        variant = f"2d {case['cake']} {variant}" + (
            f" ss{case['subpixels']}" if case.get("subpixels") else ""
        )
        if case.get("dioptrin_2d_batch_split"):
            variant += f" batch/{case['dioptrin_2d_batch_split']}"
    elif case.get("dioptrin_1d_path") not in (None, "global"):
        variant += f" {case['dioptrin_1d_path']}"
    return (
        f"{meta['dataset']:14} {case['workload']:8} {variant:25} workers={case['workers']:3} "
        + (f"processes={processes} " if processes > 1 else "")
        + f"{case['error_mode']:9} {rate:10.1f} frames/s"
    )


def worker(case_path, rank=None):
    import numpy as np

    case = json.loads(Path(case_path).read_text())
    try:
        if rank is not None:
            serve_rank(case, rank)
            return
        state = setup_case(case)
    except Unsupported as error:
        dump(
            Path(case["result"]).with_suffix(".unsupported.json"),
            dict(variant=case["variant"], reason=str(error)),
        )
        print(f"UNSUPPORTED: {error}", flush=True)
        raise SystemExit(UNSUPPORTED_EXIT)
    run, validate = state["run"], state["validate"]
    # Warmup and validation are outside warmed timing blocks. Native batch API
    # scheduling and result allocation are included in every measured call.
    durations = []
    counts = []
    cpu_seconds = []  # process user + system time per block, all threads
    for _ in range(case["blocks"]):
        count = 0
        c0 = cpu_time()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < case["seconds"]:
            result = run()
            count += len(result)
            del result
        durations.append(time.perf_counter() - t0)
        cpu_seconds.append(cpu_time() - c0)
        counts.append(count)
    validate(run())
    if state["pool"]:
        state["pool"].shutdown()
    rates = [count / duration for count, duration in zip(counts, durations)]
    result_path = Path(case["result"])
    sample = state["sample"]
    if "save" in state:
        state["save"](result_path)
    else:
        np.savez_compressed(
            result_path.with_suffix(".npz"),
            radial=sample[0],
            intensity=sample[1],
            sigma=sample[2] if case["errors"] else np.array([]),
        )
    negative = state["negative_sigma"]
    dump(
        result_path,
        dict(
            **case_record(case, state),
            frames_per_s=float(np.median(rates)),
            block_frames_per_s=rates,
            block_seconds=durations,
            block_frames=counts,
            block_cpu_seconds=cpu_seconds,
            # Busy CPUs during the timed blocks (CPU time / wall time); about
            # the CPU budget when every compute thread stays busy.
            cpu_per_wall=sum(cpu_seconds) / sum(durations),
            peak_rss_mib=peak_rss(),
            **numerical_fields(negative),
        ),
    )
    print(summary_line(case, state["meta"], np.median(rates)), flush=True)
    if negative:
        print(
            f"NUMERICAL WARNING: {len(negative)} negative Poisson sigma bins; raw values retained",
            flush=True,
        )


def send(stream, tag, value):
    stream.write(f"{CONTROL}{tag} {json.dumps(value)}\n")
    stream.flush()


def serve_rank(case, rank):
    """One process of a multi-process case, driven over stdin/stdout.

    The parent releases all ranks at a common start time; each runs the same
    number of passes over its own frames and reports its finish time.
    """
    # Keep the control channel private: anything else written to stdout,
    # including by native code, goes to the rank's log (stderr).
    control = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    state = setup_case(case, rank)
    run = state["run"]
    send(
        control,
        "ready",
        dict(
            rank=rank,
            pid=os.getpid(),
            frames=state["frames"],
            setup_first_s=state["setup_first"],
            setup_ready_s=state["setup_ready"],
        ),
    )
    for line in sys.stdin:
        command = line.split()
        if not command or command[0] == "finish":
            break
        if command[0] != "run":
            raise RuntimeError(f"Unknown control command: {line!r}")
        repeats, start_at = int(command[1]), float(command[2])
        # Sleep, then spin through the last 30 ms: OS timer slack (several ms
        # on macOS) would otherwise delay the start of some ranks.
        while (remaining := start_at - time.monotonic()) > SPIN_SECONDS:
            time.sleep(remaining - SPIN_SECONDS)
        while time.monotonic() < start_at:
            pass
        started = time.monotonic()
        count = 0
        for _ in range(repeats):
            result = run()
            count += len(result)
            del result
        send(control, "done", dict(start=started, end=time.monotonic(), frames=count))
    state["validate"](run())
    if state["pool"]:
        state["pool"].shutdown()
    path = Path(case["result"]).with_suffix(f".rank{rank}.json")
    dump(
        path,
        dict(
            **case_record(case, state),
            rank=rank,
            pid=os.getpid(),
            local_frames=state["frames"],
            local_distinct_frames=state["needed"],
            peak_rss_mib=peak_rss(),
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            **numerical_fields(state["negative_sigma"]),
        ),
    )
    send(control, "result", str(path))


def dioptrin_call_lines(rows, rows_2d):
    """summary.md lines: the CPU Dioptrin 1D call and the result conversion."""
    lines = []
    paths = sorted({r["dioptrin_1d_path"] for r in rows if r.get("dioptrin_1d_path")})
    if paths:
        calls = {
            "global": "batch1d(frames, n, errors) on the global Rayon pool of the budget",
            "split": "batch1d(frames, n, errors, num_workers=budget, num_threads=1)",
        }
        lines.append(
            "Dioptrin CPU 1D rows time "
            + "; ".join(f"{p}: {calls[p]}" for p in paths)
            + "."
        )
    conversions = sorted(
        {
            r["dioptrin_result_conversion"]
            for r in rows + rows_2d
            if r.get("dioptrin_result_conversion")
        }
    )
    if conversions:
        lines.append(
            "Every timed Dioptrin call ends with NumPy float64 intensity (and Poisson "
            "sigma) for every frame, as pyFAI's and azint's do; conversion: "
            + ", ".join(
                {"array": "array (zero-copy accessors)", "list": "list (np.asarray of lists)"}[c]
                for c in conversions
            )
            + "."
        )
    return lines


def azint_build_lines(output):
    try:
        environment = json.loads((output / "environment.json").read_text())
    except (OSError, ValueError):
        return []
    build = environment.get("azint_build")
    if not build:
        return []
    flags = (
        (build.get("recorded") or {}).get("flags") or build.get("legacy_flags") or {}
    )
    return [
        f"azint build: {build.get('variant', 'unrecorded')}; "
        f"CXXFLAGS `{flags.get('CXXFLAGS', '').strip()}`"
        + (
            f", LDFLAGS `{flags['LDFLAGS'].strip()}`"
            if flags.get("LDFLAGS", "").strip()
            else ""
        )
        + ("" if not build.get("problems") else " (UNVERIFIED)")
    ]


def extra_lines(output):
    """Skipped variants, OpenCL devices and beam-pixel masks, from run records."""
    lines = []
    try:
        skipped = json.loads((output / "skipped.json").read_text())
    except (OSError, ValueError):
        skipped = []
    for entry in skipped:
        name = entry.get("case", entry["variant"])
        lines.append(f"SKIPPED {name}: {entry['reason']}")
    try:
        environment = json.loads((output / "environment.json").read_text())
    except (OSError, ValueError):
        environment = {}
    opencl = (environment.get("variant_probes") or {}).get("opencl") or {}
    if opencl.get("available"):
        lines.append(
            f"pyFAI OpenCL device: {opencl['device']} ({opencl['device_type']}) on "
            f"{opencl['platform']}; fp64 {'yes' if opencl['device_fp64'] else 'no'}; "
            "single-precision double-word accumulation"
        )
    cuda = (environment.get("variant_probes") or {}).get("cuda") or {}
    if cuda.get("available"):
        capability = cuda.get("compute_capability")
        gpu_precision = (environment.get("arguments") or {}).get(
            "gpu_precision", CUDA_PRECISION
        )
        nvrtc = (
            "none (embedded PTX)"
            if cuda.get("nvrtc_version") is None
            else cuda_version(cuda.get("nvrtc_version"))
        )
        lines.append(
            f"Dioptrin CUDA device: {cuda.get('device_name')} (compute "
            f"{cuda_version(capability)}), driver "
            f"{cuda_version(cuda.get('driver_version'))}, NVRTC "
            f"{nvrtc}, precision {gpu_precision}"
        )
    metal = (environment.get("variant_probes") or {}).get("metal") or {}
    if metal.get("available"):
        gpu_precision = (environment.get("arguments") or {}).get(
            "gpu_precision", CUDA_PRECISION
        )
        unified = metal.get("has_unified_memory")
        if unified is None:
            unified = metal.get("unified_memory")
        lines.append(
            f"Dioptrin Metal device: {metal.get('device_name')}, "
            f"unified memory {'yes' if unified else 'no'}, "
            f"precision {gpu_precision}"
        )
    for fixture in sorted(output.glob("*/fixture.json")):
        meta = json.loads(fixture.read_text())
        beam = meta.get("beam_pixel_mask")
        if beam:
            added = [p for p in beam["pixels"] if p not in beam["already_masked"]]
            lines.append(
                f"{meta['dataset']}: --mask-beam-pixel {beam['mode']} masked pixels "
                f"{beam['pixels']} ({len(added)} newly); minimum-2theta pixel "
                f"{beam['min_2theta_pixel']}"
            )
    return lines


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
        "omp_num_threads",
        "rayon_num_threads",
        "self_check",
        "numerical_status",
        "negative_sigma_bins",
        # Added columns (appended so earlier positions are unchanged).
        "processes",
        "workers_per_process",
        "binding",
        "mask_beam_pixel",
        "beam_pixels_added",
        # 2D (cake) columns; empty for 1D rows.
        *CSV_2D_FIELDS,
        # Input dtype and GPU accumulation precision (appended).
        "input_dtype",
        "gpu_precision",
        # CPU Dioptrin 1D call (--dioptrin-1d-path; empty for other libraries,
        # GPU rows and 2D) and how every timed Dioptrin call produced its
        # NumPy arrays (empty for other libraries). Appended.
        "dioptrin_1d_path",
        "dioptrin_result_conversion",
    ]
    for row in rows:
        row.setdefault("processes", 1)
        row.setdefault("workers_per_process", row["workers"])
        row.setdefault("binding", "none")
        row.setdefault("dimension", "1d")
        row.setdefault("input_dtype", "float64")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    rows_2d = [r for r in rows if r["dimension"] == "2d"]
    rows = [r for r in rows if r["dimension"] == "1d"]
    def find_1d(row, variant):
        candidates = [
            r
            for r in rows
            if r["dataset"] == row["dataset"]
            and r["workload"] == row["workload"]
            and r["variant"] == variant
            and r["workers"] == 1
            and r["trial"] == 0
            and r["errors"] == row["errors"]
        ]
        # With several --dioptrin-1d-path values in one run, compare within
        # the same path; rows without one (other libraries, GPU rows) use the
        # default path.
        path = row.get("dioptrin_1d_path") or "global"
        for preferred in (path, "global"):
            same = [
                r for r in candidates if (r.get("dioptrin_1d_path") or "global") == preferred
            ]
            if same:
                return same[0]
        return candidates[0] if candidates else None

    def metrics_1d(row, reference, **extra):
        with np.load(Path(row["result"]).with_suffix(".npz")) as saved:
            a = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        with np.load(Path(reference["result"]).with_suffix(".npz")) as saved:
            b = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        metrics = dict(
            dataset=row["dataset"],
            workload=row["workload"],
            variant=row["variant"],
            error_mode=row["error_mode"],
            reference=reference["variant"],
            dioptrin_1d_path=row.get("dioptrin_1d_path"),
            reference_dioptrin_1d_path=reference.get("dioptrin_1d_path"),
            negative_sigma_bins=row.get("negative_sigma_bins", 0),
            max_axis_difference_deg=float(np.max(np.abs(a["radial"] - b["radial"]))),
        )
        for key in ("intensity", "sigma"):
            if a[key].size:
                denominator = float(np.linalg.norm(b[key]))
                metrics[key + "_relative_l2"] = float(
                    np.linalg.norm(a[key] - b[key])
                ) / max(denominator, 1e-300)
        metrics.update(extra)
        return metrics

    # Entry kinds (README, "Outputs"): a reference entry (no "comparison" key)
    # per one-worker trial-0 row against dioptrin-pixel_split, then a
    # "comparison": "counterpart" entry per COUNTERPARTS_1D pair. A pair whose
    # counterpart is dioptrin-pixel_split itself would repeat the reference
    # entry; instead that reference entry gets "also_counterpart": true.
    counterparts = dict(COUNTERPARTS_1D)
    numerical = []
    for row in rows:
        if row["workers"] != 1 or row["trial"] != 0:
            continue
        reference = find_1d(row, "dioptrin-pixel_split")
        if reference is None:
            continue
        extra = {}
        if counterparts.get(row["variant"]) == "dioptrin-pixel_split":
            extra["also_counterpart"] = True
        numerical.append(metrics_1d(row, reference, **extra))
    # Counterparts (GPU rows against their CPU method), after the reference
    # entries; a pair whose counterpart row is missing is left out.
    for row in rows:
        if row["workers"] != 1 or row["trial"] != 0:
            continue
        counterpart = counterparts.get(row["variant"])
        if counterpart is None or counterpart == "dioptrin-pixel_split":
            continue
        reference = find_1d(row, counterpart)
        if reference is not None:
            numerical.append(metrics_1d(row, reference, comparison="counterpart"))
    # azint-cached must reproduce stock azint exactly; compare every matching case.
    identity = []
    for row in rows:
        if row["variant"] != "azint-cached" or row["processes"] != 1:
            # Multi-process ranks check stock bit-identity in process only.
            continue
        stock = next(
            (
                r
                for r in rows
                if r["variant"] == "azint"
                and all(
                    r[k] == row[k]
                    for k in (
                        "dataset",
                        "workload",
                        "workers",
                        "processes",
                        "errors",
                        "trial",
                    )
                )
            ),
            None,
        )
        if stock is None:
            continue
        with np.load(Path(row["result"]).with_suffix(".npz")) as saved:
            a = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        with np.load(Path(stock["result"]).with_suffix(".npz")) as saved:
            b = {key: saved[key] for key in ("radial", "intensity", "sigma")}
        identity.append(
            dict(
                **{
                    k: row[k]
                    for k in ("dataset", "workload", "workers", "error_mode", "trial")
                },
                **{
                    key + "_bit_identical": same_bits(a[key], b[key])
                    for key in ("intensity", "sigma")
                    if row["errors"] or key == "intensity"
                },
                in_process_stock_check=row.get("stock_azint_bit_identical"),
            )
        )
    numerical += numerical_2d(rows_2d)
    identity += azint_identity_2d(rows_2d)
    dump(output / "numerical.json", numerical)
    dump(output / "azint_cached_identity.json", identity)
    # Runs without Dioptrin GPU rows keep the original wording byte for byte
    # (pyfai-full-opencl predates the Dioptrin GPU rows and may use a CPU device).
    present = {r["variant"] for r in rows + rows_2d}
    gpu_rows = (
        sorted(present & set(GPU_VARIANTS))
        if present & set(DIOPTRIN_GPU_VARIANTS)
        else []
    )
    gpu_label = (
        "GPU rows"
        if (present & set(CUDA_VARIANTS)) and (present & set(METAL_VARIANTS))
        else (
            "Metal and OpenCL rows"
            if present & set(METAL_VARIANTS)
            else "CUDA and OpenCL rows"
        )
    )
    gpu_lines = (
        [
            f"{gpu_label} (" + ", ".join(gpu_rows) + ")",
            "use one host thread driving one device at CPU budget 1 in one process; their",
            "timed calls include the host-to-device frame transfers and the transfer of",
            "the results back.",
        ]
        if gpu_rows
        else []
    )
    input_dtypes = sorted({r.get("input_dtype", "float64") for r in rows + rows_2d} or {"float64"})
    dtype_label = "/".join(input_dtypes)
    if rows and rows_2d:
        dim_label = "1D and 2D"
    elif rows_2d:
        dim_label = "2D"
    else:
        dim_label = "1D"
    lines = [
        "# CPU and GPU integration benchmark" if gpu_rows else "# CPU integration benchmark",
        "",
        f"{dim_label}, preloaded {dtype_label} inputs; generation, decoding, and output writing excluded.",
        *gpu_lines,
        *(
            ["", f"**WARNING: {LIST_CONVERSION_WARNING}**", ""]
            if any(r.get("dioptrin_result_conversion") == "list" for r in rows + rows_2d)
            else []
        ),
        "Repeated reuses one image; frames cycles through distinct generated images.",
        "Cross-library algorithms, precision, and normalization differ; see numerical.json.",
        "azint reuses its constructor normalisation in azint-cached; azint_cached_identity.json",
        "records its bit-identity with stock azint. azint uses OMP_NUM_THREADS = CPU budget",
        "(matrix construction only); pyFAI uses 1 except pyfai-full-omp (one CSR engine,",
        "OMP_NUM_THREADS = budget); Dioptrin uses a Rayon pool of the budget. With several",
        "processes each gets budget/processes; throughput is all frames over the wall time",
        "from a common start to the last process finishing.",
        *dioptrin_call_lines(rows, rows_2d),
        *azint_build_lines(output),
        *extra_lines(output),
    ]
    if rows or not rows_2d:
        lines += [
            "",
            "| Dataset | Workload | Method | CPU budget | Processes | Errors | Median frames/s | Trials | Numerical status |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    keys = ("dataset", "workload", "variant", "workers", "processes", "error_mode")
    # The CPU Dioptrin 1D path is shown with the method, not as a column.
    groups = sorted(
        {tuple(r[f] for f in keys) + (r.get("dioptrin_1d_path") or "",) for r in rows}
    )
    for key in groups:
        members = [
            r
            for r in rows
            if tuple(r[f] for f in keys) + (r.get("dioptrin_1d_path") or "",) == key
        ]
        *key, path = key
        if path:
            key[2] = f"{key[2]} ({path})"
        values = [r["frames_per_s"] for r in members]
        status = ", ".join(
            sorted({r.get("numerical_status", "unchecked") for r in members})
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
    lines += summary_2d(rows_2d)
    (output / "summary.md").write_text("\n".join(lines) + "\n")


CSV_2D_FIELDS = (
    "dimension",
    "n_radial",
    "n_azimuthal",
    "subpixels",
    "dioptrin_2d_path",
    "uncertainty_support",
    "table_nnz",
    "table_nnz_total",
    "table_nnz_source",
    "estimated_table_nnz",
    "estimated_memory_mib",
    "rss_inputs_mib",
    "peak_rss_ready_mib",
    "setup_rss_growth_mib",
    # Appended: batch2d split of --dioptrin-2d-path batch; empty otherwise.
    "dioptrin_2d_batch_split",
)
CAKE_KEYS = ("dataset", "workload", "n_radial", "n_azimuthal", "errors")


def cake_metrics(a, b):
    """Compare two cakes (n_azimuthal, n_radial) over bins populated in both.

    Rows 0 and n-1 are the bins on either side of the +-180 degree seam; they
    are reported separately and excluded from the interior figures.
    """
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    populated_a = np.isfinite(a) & (a != 0)
    populated_b = np.isfinite(b) & (b != 0)
    both = populated_a & populated_b
    seam = np.zeros(a.shape, dtype=bool)
    seam[[0, -1], :] = True

    def figures(selection):
        x, y = a[selection], b[selection]
        if not x.size:
            return dict(bins=0)
        relative = np.abs(x - y) / np.abs(y)
        return dict(
            bins=int(x.size),
            relative_l2=float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300)),
            bins_over_1pct=int(np.count_nonzero(relative > 0.01)),
            fraction_over_1pct=float(np.count_nonzero(relative > 0.01) / x.size),
            max_relative_difference=float(relative.max()),
            median_ratio=float(np.median(x / y)),
        )

    return dict(
        populated_in_both=int(both.sum()),
        populated_only_in_variant=int((populated_a & ~populated_b).sum()),
        populated_only_in_reference=int((populated_b & ~populated_a).sum()),
        all=figures(both),
        interior=figures(both & ~seam),
        seam_rows=figures(both & seam),
    )


def numerical_2d(rows):
    """Cross-library comparison of the first cake of every w1/t0 case.

    Each case is compared with dioptrin-pixel_split (polygon clipping) of the
    same dataset, workload, cake and error mode, and with its counterpart in
    COUNTERPARTS_2D (same subpixel count where both have one). Every pair is
    compared as integrated ("grid": "native"). dioptrin-supersampled cakes sit
    on their own radial grid; where one side is dioptrin-supersampled, a second
    entry ("grid": "common") uses its MultiIntegrator cake on the common edges.
    """
    import numpy as np

    references = [
        r for r in rows if r["workers"] == 1 and r["trial"] == 0 and r["processes"] == 1
    ]

    def load(path):
        if not path.is_file():
            return None
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}

    def cake(row, grid):
        base = Path(row["result"])
        if grid == "common":
            common = load(base.with_suffix(".cake0common.npz"))
            if common is not None:
                return common
        return load(base.with_suffix(".cake0.npz"))

    def find(row, variant):
        candidates = [
            r
            for r in references
            if r["variant"] == variant and all(r[k] == row[k] for k in CAKE_KEYS)
        ]
        same = [r for r in candidates if r.get("subpixels") == row.get("subpixels")]
        candidates = same or candidates
        # With several batch splits in one run, compare within the same split.
        split = row.get("dioptrin_2d_batch_split")
        same = [r for r in candidates if r.get("dioptrin_2d_batch_split") == split]
        return (same or candidates or [None])[0]

    entries = []
    for row in references:
        pairs = [("dioptrin-pixel_split", "reference")]
        pairs += [
            (ref, "counterpart") for v, ref in COUNTERPARTS_2D if v == row["variant"]
        ]
        for ref_variant, role in pairs:
            other = find(row, ref_variant)
            if other is None or other is row:
                continue
            grids = ["native"]
            if "dioptrin-supersampled" in (row["variant"], other["variant"]):
                grids.append("common")
            for grid in grids:
                a, b = cake(row, grid), cake(other, grid)
                if a is None or b is None:
                    continue
                entry = dict(
                    dimension="2d",
                    dataset=row["dataset"],
                    workload=row["workload"],
                    cake=row["cake"],
                    variant=row["variant"],
                    subpixels=row.get("subpixels"),
                    error_mode=row["error_mode"],
                    reference=ref_variant,
                    reference_subpixels=other.get("subpixels"),
                    dioptrin_2d_batch_split=row.get("dioptrin_2d_batch_split"),
                    reference_dioptrin_2d_batch_split=other.get(
                        "dioptrin_2d_batch_split"
                    ),
                    comparison=role,
                    grid=grid,
                    frame="first distinct frame",
                    negative_sigma_bins=row.get("negative_sigma_bins", 0),
                    max_radial_axis_difference_deg=float(
                        np.max(np.abs(a["radial"] - b["radial"]))
                    ),
                    max_azimuth_axis_difference_deg=float(
                        np.max(np.abs(a["azimuthal"] - b["azimuthal"]))
                    ),
                    intensity=cake_metrics(a["intensity"], b["intensity"]),
                )
                if a["sigma"].size and b["sigma"].size:
                    entry["sigma"] = cake_metrics(a["sigma"], b["sigma"])
                entries.append(entry)
    return entries


def azint_identity_2d(rows):
    """azint-cached against stock azint for the 2D reference cases (all cakes)."""
    import numpy as np

    identity = []
    for row in rows:
        if row["variant"] != "azint-cached" or row["workers"] != 1 or row["trial"]:
            continue
        stock = next(
            (
                r
                for r in rows
                if r["variant"] == "azint"
                and r["workers"] == 1
                and r["trial"] == 0
                and r.get("subpixels") == row.get("subpixels")
                and all(r[k] == row[k] for k in CAKE_KEYS)
            ),
            None,
        )
        paths = [Path(r["result"]).with_suffix(".npz") for r in (row, stock or row)]
        if stock is None or not all(p.is_file() for p in paths):
            continue
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            bits = {
                key + "_bit_identical": same_bits(a[key], b[key])
                for key in ("intensity", "sigma")
                if row["errors"] or key == "intensity"
            }
        identity.append(
            dict(
                dimension="2d",
                cake=row["cake"],
                subpixels=row.get("subpixels"),
                **{
                    k: row[k]
                    for k in ("dataset", "workload", "workers", "error_mode", "trial")
                },
                **bits,
                in_process_stock_check=row.get("stock_azint_bit_identical"),
            )
        )
    return identity


def summary_2d(rows):
    import numpy as np

    if not rows:
        return []
    lines = [
        "",
        "## 2D cakes",
        "",
        "chi in [-180, 180) degrees, the fixture's 2theta range, same mask and",
        "corrections for every library. First = time to the first cake (setup,",
        "table construction); ready = all workers built and warmed. Peak RSS at ready",
        "is sampled before validation references are read. nnz: stored table weights",
        "per engine where the library exposes them (pyFAI), else the estimate (~).",
        "",
        "| Dataset | Workload | Cake | Method | Subpixels | CPU budget | Processes | Errors | "
        "Median frames/s | First s | Ready s | Peak RSS ready MiB | nnz | Trials | Numerical status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    keys = (
        "dataset",
        "workload",
        "cake",
        "variant",
        "subpixels",
        "workers",
        "processes",
        "error_mode",
        "dioptrin_2d_batch_split",  # shown with the method, not as a column
    )
    groups = sorted(
        {tuple(r.get(f) for f in keys) for r in rows},
        key=lambda k: tuple("" if v is None else str(v).zfill(8) for v in k),
    )
    for key in groups:
        members = [r for r in rows if tuple(r.get(f) for f in keys) == key]
        *key, split = key
        if split:
            key[3] = f"{key[3]} (batch, {split})"
        median = lambda field: float(np.median([r[field] for r in members]))  # noqa: E731
        nnz = members[0].get("table_nnz")
        nnz = f"{nnz:,}" if nnz else f"~{members[0].get('estimated_table_nnz', 0):,.0f}"
        status = ", ".join(
            sorted({r.get("numerical_status", "unchecked") for r in members})
        )
        values = (
            *("-" if v is None else v for v in key),
            round(median("frames_per_s"), 2),
            round(median("setup_first_s"), 2),
            round(median("setup_ready_s"), 2),
            round(median("peak_rss_ready_mib")),
            nnz,
            len(members),
            status,
        )
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return lines


def numa_nodes():
    """NUMA nodes with CPUs in this process's affinity: {node: [cpus]}."""
    allowed = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    )
    nodes = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        cpus = set()
        for part in (path / "cpulist").read_text().strip().split(","):
            if part:
                low, _, high = part.partition("-")
                cpus.update(range(int(low), int(high or low) + 1))
        if cpus & allowed:
            nodes[int(path.name.removeprefix("node"))] = sorted(cpus & allowed)
    return nodes


def compact_cpus(cpus):
    """[0, 1, 2, 5] -> '0-2,5'."""
    parts, cpus = [], sorted(cpus)
    start = previous = None
    for cpu in cpus + [None]:
        if previous is not None and cpu == previous + 1:
            previous = cpu
            continue
        if start is not None:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    return ",".join(parts)


def binding_commands(binding, processes, workers_per_process=1):
    """Command prefix per rank. `numa` binds rank r's CPUs and memory to one
    NUMA node, round-robin over the nodes in the allocation.

    Fails when a node's CPUs in this process's affinity are fewer than the
    workers of the ranks bound to it (ranks on one node share its CPUs).
    """
    if binding == "none":
        return [[] for _ in range(processes)]
    numactl = shutil.which("numactl")
    available = numa_nodes()
    nodes = sorted(available)
    if numactl is None or not nodes:
        raise RuntimeError("--bind numa needs Linux NUMA topology and numactl in PATH")
    for index, node in enumerate(nodes):
        ranks = len(range(index, processes, len(nodes)))
        needed = ranks * workers_per_process
        if ranks and len(available[node]) < needed:
            raise RuntimeError(
                f"--bind numa: NUMA node {node} has {len(available[node])} CPU(s) "
                f"in this process's affinity ({compact_cpus(available[node])}), "
                f"fewer than the {needed} worker(s) of the {ranks} process(es) "
                f"bound to it ({workers_per_process} per process, "
                f"{processes} processes over nodes {nodes}); use fewer workers or "
                "processes, --bind none, or an allocation with more CPUs per node"
            )
    return [
        [
            numactl,
            f"--cpunodebind={nodes[r % len(nodes)]}",
            f"--membind={nodes[r % len(nodes)]}",
        ]
        for r in range(processes)
    ]


def expect(process, tag, log):
    """Next control message `tag` from a rank; other stdout lines go to its log."""
    for line in process.stdout:
        if line.startswith(CONTROL + tag + " "):
            return json.loads(line[len(CONTROL + tag) + 1 :])
        log.write(line)
        log.flush()
    raise RuntimeError(f"Rank process exited (code {process.wait()}) before {tag!r}")


def run_processes(case, case_path, env, prefixes):
    """Run one case as `processes` coordinated processes and aggregate it.

    Each block releases every rank at the same start time; each integrates the
    same number of passes over its own frames. Untimed calibration blocks set
    the passes per block so that a block lasts about --seconds.
    """
    import numpy as np

    processes, logs = [], []
    try:
        for rank, prefix in enumerate(prefixes):
            log = case_path.with_suffix(f".rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        *prefix,
                        sys.executable,
                        "-E",
                        "-s",
                        __file__,
                        "--case",
                        str(case_path),
                        "--rank",
                        str(rank),
                    ],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
            )
        ready = [expect(p, "ready", log) for p, log in zip(processes, logs)]

        def block(repeats):
            start = time.monotonic() + START_LEAD_SECONDS
            for process in processes:
                process.stdin.write(f"run {repeats} {start!r}\n")
                process.stdin.flush()
            done = [expect(p, "done", log) for p, log in zip(processes, logs)]
            return (
                sum(d["frames"] for d in done),
                max(d["end"] for d in done) - start,
                max(d["start"] for d in done) - start,
            )

        # Two untimed calibration blocks: one pass, then about a quarter block.
        _, calibration, _ = block(1)
        trial_passes = max(1, math.ceil(0.25 * case["seconds"] / calibration))
        _, calibration, _ = block(trial_passes)
        calibration /= trial_passes
        repeats = max(1, math.ceil(case["seconds"] / calibration))
        blocks = [block(repeats) for _ in range(case["blocks"])]
        for process in processes:
            process.stdin.write("finish\n")
            process.stdin.flush()
        paths = [expect(p, "result", log) for p, log in zip(processes, logs)]
        for process in processes:
            if process.wait(timeout=600):
                raise RuntimeError(f"Rank exited with {process.returncode}")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.kill()
        raise
    finally:
        for log in logs:
            log.close()
    ranks = [json.loads(Path(path).read_text()) for path in paths]
    counts = [b[0] for b in blocks]
    durations = [b[1] for b in blocks]
    rates = [c / d for c, d in zip(counts, durations)]
    negative = sorted(
        {tuple(i) for r in ranks for i in r["negative_sigma_bin_indices"]}
    )
    first = ranks[0]
    record = dict(
        {k: v for k, v in first.items() if k not in ("rank", "pid", "peak_rss_mib")},
        setup_first_s=max(r["setup_first_s"] for r in ready),
        setup_ready_s=max(r["setup_ready_s"] for r in ready),
        stock_azint_bit_identical=True
        if all(r["stock_azint_bit_identical"] for r in ranks)
        else first["stock_azint_bit_identical"],
        frames_per_s=float(np.median(rates)),
        block_frames_per_s=rates,
        block_seconds=durations,
        block_frames=counts,
        passes_per_block=repeats,
        calibration_seconds_per_pass=calibration,
        max_start_lag_s=max(b[2] for b in blocks),
        peak_rss_mib=sum(r["peak_rss_mib"] for r in ranks),
        peak_rss_mib_per_process=[r["peak_rss_mib"] for r in ranks],
        rank_commands=[" ".join(prefix) or "(unbound)" for prefix in prefixes],
        rank_cpu_affinity=[r["cpu_affinity"] for r in ranks],
        rank_results=[str(path) for path in paths],
        **numerical_fields([list(i) for i in negative]),
    )
    for key in ("local_frames", "local_distinct_frames", "cpu_affinity"):
        record.pop(key, None)
    # 2D memory and table sizes: every process holds its own frames and tables.
    for key in (
        "rss_inputs_mib",
        "peak_rss_ready_mib",
        "setup_rss_growth_mib",
        "table_nnz_total",
    ):
        if all(r.get(key) is not None for r in ranks):
            record[key] = sum(r[key] for r in ranks)
            record[key + "_per_process"] = [r[key] for r in ranks]
    dump(case["result"], record)
    return summary_line(
        case,
        json.loads((Path(case["fixture"]) / "fixture.json").read_text()),
        record["frames_per_s"],
    ) + (
        f"\nNUMERICAL WARNING: {len(negative)} negative Poisson sigma bins; raw values retained"
        if negative
        else ""
    )


def probe_variants(variants, opencl_device, cuda_device=0, metal_device=0):
    """Run the availability probes in fresh processes (same flags as cases).

    The pyFAI probe never touches CUDA or Metal. CUDA/Metal are probed (and
    initialised) only when a matching variant is requested, in a separate child
    with a timeout, so a crash or hang skips only those variants; this parent
    never touches the device.
    """
    result = {}
    if any(v in ("pyfai-full-omp", "pyfai-full-opencl") for v in variants):
        child = subprocess.run(
            [
                sys.executable,
                "-E",
                "-s",
                __file__,
                "--probe",
                "--opencl-device",
                opencl_device,
            ],
            capture_output=True,
            text=True,
            env=os.environ | case_threads("pyfai-full", 1),
        )
        lines = [line for line in child.stdout.splitlines() if line.startswith("{")]
        if child.returncode or not lines:
            raise RuntimeError("Variant probe failed: " + child.stderr[-2000:])
        result.update(json.loads(lines[-1]))
    if any(v in CUDA_VARIANTS for v in variants):
        result["cuda"] = probe_cuda_child(cuda_device)
    if any(v in METAL_VARIANTS for v in variants):
        result["metal"] = probe_metal_child(metal_device)
    return result


def probe_metal_child(device, timeout=CUDA_PROBE_TIMEOUT_S):
    """metal_probe(device) in its own child process; never raises."""
    command = [
        sys.executable,
        "-E",
        "-s",
        __file__,
        "--probe-metal",
        "--metal-device",
        str(device),
    ]
    failed = dict(compiled=None, available=False, probe_failed=True, device=device)
    try:
        child = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ | case_threads("dioptrin-pixel_split-metal", 1),
        )
    except subprocess.TimeoutExpired:
        return dict(failed, reason=f"Metal probe timed out after {timeout:g} s (killed)")
    tail = (child.stderr or "").strip()[-500:]
    if child.returncode:
        if child.returncode < 0:
            try:
                name = signal.Signals(-child.returncode).name
            except ValueError:
                name = f"signal {-child.returncode}"
            status = f"killed by {name} (exit status {child.returncode})"
        else:
            status = f"exit status {child.returncode}"
        return dict(
            failed,
            reason=f"Metal probe child failed: {status}" + (f"; {tail}" if tail else ""),
        )
    lines = [line for line in child.stdout.splitlines() if line.startswith("{")]
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return dict(
            failed,
            reason="Metal probe child printed no result (exit status 0)"
            + (f"; {tail}" if tail else ""),
        )


def probe_cuda_child(device, timeout=CUDA_PROBE_TIMEOUT_S):
    """cuda_probe(device) in its own child process; never raises.

    A non-zero exit, a signal, a timeout or unparsable output is recorded as
    unavailable (probe_failed) with the exit status in the reason.
    """
    command = [
        sys.executable,
        "-E",
        "-s",
        __file__,
        "--probe-cuda",
        "--cuda-device",
        str(device),
    ]
    failed = dict(compiled=None, available=False, probe_failed=True, device=device)
    try:
        child = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ | case_threads("dioptrin-pixel_split-cuda", 1),
        )
    except subprocess.TimeoutExpired:
        return dict(failed, reason=f"CUDA probe timed out after {timeout:g} s (killed)")
    tail = (child.stderr or "").strip()[-500:]
    if child.returncode:
        if child.returncode < 0:
            try:
                name = signal.Signals(-child.returncode).name
            except ValueError:
                name = f"signal {-child.returncode}"
            status = f"killed by {name} (exit status {child.returncode})"
        else:
            status = f"exit status {child.returncode}"
        return dict(
            failed, reason=f"CUDA probe child failed: {status}" + (f"; {tail}" if tail else "")
        )
    lines = [line for line in child.stdout.splitlines() if line.startswith("{")]
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return dict(
            failed,
            reason="CUDA probe child printed no result (exit status 0)"
            + (f"; {tail}" if tail else ""),
        )


def skipped_variants(variants, probes):
    skipped = []
    if "pyfai-full-omp" in variants and not probes["pyfai_openmp"]["openmp"]:
        skipped.append(
            dict(
                variant="pyfai-full-omp",
                reason="this pyFAI build has no OpenMP in its CSR extension "
                f"({probes['pyfai_openmp'].get('module', 'unknown')}); "
                "OMP_NUM_THREADS would have no effect (macOS wheels lack OpenMP)",
            )
        )
    if "pyfai-full-opencl" in variants and not probes["opencl"]["available"]:
        skipped.append(
            dict(
                variant="pyfai-full-opencl",
                reason=probes["opencl"].get("reason", "OpenCL unavailable"),
            )
        )
    cuda = probes.get("cuda") or dict(
        compiled=False, available=False, reason="CUDA was not probed"
    )
    for variant in (v for v in variants if v in CUDA_VARIANTS):
        if cuda.get("available"):
            continue
        if cuda.get("probe_failed"):
            reason = cuda.get("reason") or "CUDA probe child failed"
        elif not cuda.get("compiled"):
            reason = (
                "Dioptrin was built without the 'cuda' feature; rerun setup.py "
                "--with-cuda (environment.py --with-cuda / WITH_CUDA=1)"
            )
            if cuda.get("reason") == CUDA_NO_API:
                reason += f" ({CUDA_NO_API})"
        else:
            reason = (
                f"CUDA unavailable on device {cuda.get('device', 0)}: "
                f"{cuda.get('reason') or 'no reason reported'}"
            )
        skipped.append(dict(variant=variant, reason=reason))
    metal = probes.get("metal") or dict(
        compiled=False, available=False, reason="Metal was not probed"
    )
    for variant in (v for v in variants if v in METAL_VARIANTS):
        if metal.get("available"):
            continue
        if metal.get("probe_failed"):
            reason = metal.get("reason") or "Metal probe child failed"
        elif not metal.get("compiled"):
            reason = (
                "Dioptrin was built without the 'metal' feature; rerun setup.py "
                "--with-metal on macOS"
            )
            if metal.get("reason") == METAL_NO_API:
                reason += f" ({METAL_NO_API})"
        else:
            reason = (
                f"Metal unavailable on device {metal.get('device', 0)}: "
                f"{metal.get('reason') or 'no reason reported'}"
            )
        skipped.append(dict(variant=variant, reason=reason))
    return skipped


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
        "--subset-dir",
        type=Path,
        help="Directory of prepare_subset.py bundles for the *-subset datasets; "
        "default: --data-dir",
    )
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
        "--allow-list-conversion",
        action="store_true",
        help="Run Dioptrin variants even if the installed build lacks the "
        "IntegrationResult *_array accessors (0.15.2 and older), timing NumPy "
        "conversion of Python lists; warned about at start and in summary.md. "
        "Without it such a run is refused",
    )
    parser.add_argument(
        "--workers",
        help="Comma-separated CPU budgets; default: suite budgets within CPU affinity",
    )
    parser.add_argument(
        "--processes",
        default="1",
        help="Comma-separated process counts. P > 1 splits each CPU budget over P "
        "processes (budget/P workers each, own frames and tables); budgets not "
        "divisible by P are skipped. Default: 1",
    )
    parser.add_argument(
        "--bind",
        choices=("none", "numa"),
        default="none",
        help="Multi-process CPU/memory binding: numa binds each process to one "
        "NUMA node with numactl (round-robin). Default: none",
    )
    parser.add_argument(
        "--mask-beam-pixel",
        nargs="?",
        const="poni",
        choices=("poni", "direct"),
        help="Mask, for all libraries, the pixel(s) containing or touching the "
        "point of normal incidence (poni, the default) or the direct-beam "
        "position (direct, 2theta = 0)",
    )
    parser.add_argument(
        "--opencl-device",
        default="gpu",
        help="pyfai-full-opencl device: gpu, cpu, acc, any, or PLATFORM:DEVICE ids",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        help="CUDA device ordinal for the dioptrin-*-cuda variants (after "
        "CUDA_VISIBLE_DEVICES). Default: 0",
    )
    parser.add_argument(
        "--metal-device",
        type=int,
        help="Metal device ordinal for the dioptrin-*-metal variants on macOS. "
        "Default: 0",
    )
    parser.add_argument(
        "--gpu-precision",
        choices=GPU_PRECISIONS,
        default=CUDA_PRECISION,
        help="Accumulation precision for Dioptrin CUDA/Metal variants: "
        "f32_compensated (default, matches pyFAI OpenCL), f32, or f64 (CUDA only)",
    )
    parser.add_argument(
        "--input-dtype",
        choices=INPUT_DTYPES,
        default="float64",
        help="Preloaded input frame dtype passed to all libraries (default: float64)",
    )
    parser.add_argument("--datasets", nargs="+", choices=definition["datasets"])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("repeated", "frames"),
        default=["repeated", "frames"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help="Default: every variant except the opt-in dioptrin-*-cuda / -metal rows",
    )
    parser.add_argument("--bins", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=("1d", "2d"),
        default=["1d"],
        help="1d (default): azimuthal integration to 1D patterns; 2d: cakes "
        "(radial x azimuthal); both may be given",
    )
    parser.add_argument(
        "--cakes",
        nargs="+",
        default=list(DEFAULT_CAKES),
        help="2D cake sizes RADIALxAZIMUTHAL (chi -180..180 deg). Default: "
        + " ".join(DEFAULT_CAKES),
    )
    parser.add_argument(
        "--subpixels",
        nargs="+",
        type=int,
        default=list(DEFAULT_SUBPIXELS_2D),
        help="2D subpixels per side for dioptrin-supersampled, azint and "
        "azint-cached; several values give one case each. Default: 10. "
        "1D cases keep the per-dataset n_splitting",
    )
    parser.add_argument(
        "--dioptrin-1d-path",
        nargs="+",
        choices=DIOPTRIN_1D_PATHS,
        default=["global"],
        help="CPU Dioptrin 1D call over the frame list: global (default) is "
        "Integrator.batch1d(frames, n, errors=...) without num_workers/"
        "num_threads, on the global Rayon pool of RAYON_NUM_THREADS = CPU "
        "budget threads (the library's default call); split passes "
        "num_workers=budget, num_threads=1 (budget frames at a time on "
        "one-thread pools built per call, since Dioptrin 0.15.0). Several "
        "values give one case each; non-default paths are part of the case "
        "name",
    )
    parser.add_argument(
        "--dioptrin-2d-path",
        choices=DIOPTRIN_2D_PATHS,
        default="sequential",
        help="Dioptrin 2D multi-core path: sequential (default) integrates one "
        "frame at a time with Integrator.integrate2d, whose sparse product is "
        "row-parallel over the Rayon pool; stream uses batch2d_iter over the "
        "array list (frame-parallel, copies each frame); batch is one "
        "Integrator.batch2d call over the array list (no copies; split of the "
        "CPU budget set by --dioptrin-2d-batch-split; needs a Dioptrin whose "
        "batch2d accepts arrays, otherwise the cases are skipped as unsupported)",
    )
    parser.add_argument(
        "--dioptrin-2d-batch-split",
        nargs="+",
        choices=DIOPTRIN_2D_BATCH_SPLITS,
        default=["frames"],
        help="With --dioptrin-2d-path batch: how batch2d splits the CPU budget B. "
        "frames (default): num_workers=B, num_threads=1, one frame per thread "
        "as the azint/pyFAI frame threads; nested: neither argument, all frames "
        "on the global Rayon pool of B threads, parallel inside frames too. "
        "Several values give one case each",
    )
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.7,
        help="Skip 2D cases whose estimated peak memory exceeds this fraction of "
        "RAM (or of the cgroup limit). Default: 0.7",
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-cuda", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-metal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight-2d", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(args.case, args.rank)
        return
    if args.probe:
        probe(args.opencl_device)
        return
    if args.cuda_device is None:
        args.cuda_device = 0
    if args.metal_device is None:
        args.metal_device = 0
    if args.probe_cuda:
        # Own child process (probe_cuda_child): initialises CUDA.
        print(json.dumps(cuda_probe(args.cuda_device)), flush=True)
        return
    if args.probe_metal:
        # Own child process (probe_metal_child): initialises Metal.
        print(json.dumps(metal_probe(args.metal_device)), flush=True)
        return
    if args.cuda_device < 0:
        parser.error("--cuda-device must be a non-negative device ordinal")
    if args.metal_device < 0:
        parser.error("--metal-device must be a non-negative device ordinal")
    if args.gpu_precision == "f64" and any(v in METAL_VARIANTS for v in args.variants):
        parser.error("--gpu-precision f64 is not supported by Metal (use f32_compensated or f32)")
    if args.preflight_2d:
        preflight_2d(
            args.preflight_2d,
            args.cakes,
            args.subpixels,
            args.variants,
            args.preflight_2d / "preflight2d.json",
        )
        return
    if args.list_datasets:
        print(json.dumps(definition, indent=2))
        return
    try:
        opencl_target(args.opencl_device)
    except ValueError as error:
        parser.error(str(error))
    defaults = definition["suites"][args.suite]
    for key in ("bins", "frames", "trials", "blocks", "seconds", "datasets"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    for key in ("bins", "frames", "trials", "blocks", "seconds"):
        if getattr(args, key) <= 0 or not math.isfinite(getattr(args, key)):
            parser.error(f"--{key} must be positive and finite")
    try:
        cakes = [parse_cake(c) for c in args.cakes]
    except ValueError as error:
        parser.error(str(error))
    args.cakes = [f"{r}x{a}" for r, a in cakes]
    args.dioptrin_2d_batch_split = list(dict.fromkeys(args.dioptrin_2d_batch_split))
    args.dioptrin_1d_path = list(dict.fromkeys(args.dioptrin_1d_path))
    if min(args.subpixels) < 1:
        parser.error("--subpixels must be positive")
    if not 0 < args.max_memory_fraction <= 1:
        parser.error("--max-memory-fraction must be in (0, 1]")
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
        process_counts = sorted({int(p) for p in args.processes.split(",")} | {1})
    except ValueError:
        parser.error(
            "--workers and --processes must contain comma-separated positive integers"
        )
    if not workers or workers[0] < 1 or workers[-1] > available:
        parser.error(
            f"--workers must be within the process CPU allocation (1..{available})"
        )
    if process_counts[0] < 1:
        parser.error("--processes must be positive")
    workers = sorted(set(workers) | {1})
    bindings = {}
    for count in process_counts:
        if count > 1:
            # The largest per-process share scheduled with this process count.
            per_process = max(
                (w // count for w in workers if w >= count and w % count == 0),
                default=1,
            )
            try:
                bindings[count] = binding_commands(args.bind, count, per_process)
            except RuntimeError as error:
                parser.error(str(error))
    required = ["numpy", "pyFAI"]
    if any(v.startswith("dioptrin-") for v in args.variants):
        required.append("dioptrin")
    if any(v.startswith("azint") for v in args.variants):
        required.append("azint")
    if any(definition["datasets"][d]["kind"] == "subset" for d in args.datasets):
        required += ["h5py", "hdf5plugin"]
    try:
        versions = {name: importlib.metadata.version(name) for name in required}
    except importlib.metadata.PackageNotFoundError as error:
        parser.error(
            f"Missing benchmark dependency {error.name}; run python3 benchmarks/comparison/setup.py first"
        )
    for optional in ("pyopencl",):
        try:
            versions[optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            pass
    build_path = Path(sys.prefix) / "benchmark-build.json"
    installed_build = (
        json.loads(build_path.read_text()) if build_path.is_file() else None
    )
    build_problems = (
        verify_build(ROOT, installed_build) if "dioptrin" in required else []
    )
    azint_build = None
    if "azint" in required:
        installed_azint = azint_signature()
        recorded = (installed_build or {}).get("azint_build")
        azint_problems = verify_azint_build(installed_build, installed_azint)
        build_problems += azint_problems
        azint_build = dict(
            variant=(recorded or {}).get("variant", "unrecorded"),
            recorded=recorded,
            # Records written before azint_build existed share one flag set.
            legacy_flags=None if recorded else (installed_build or {}).get("flags"),
            installed_extension=installed_azint,
            problems=azint_problems,
        )
    if build_problems:
        message = "; ".join(build_problems)
        if not args.allow_unverified_build:
            parser.error(
                message
                + ". Rerun setup.py, or explicitly use --allow-unverified-build."
            )
        print("UNVERIFIED BUILD: " + message, file=sys.stderr, flush=True)
    # Every timed Dioptrin call converts its results to NumPy arrays; without
    # the accessors that is list building, which can dominate the timing.
    # Checked here, before any case runs (cases and ranks check again).
    result_arrays = (
        dioptrin_has_result_arrays() if "dioptrin" in required else None
    )
    if result_arrays is False:
        if not args.allow_list_conversion:
            parser.error("Refusing to run Dioptrin variants: " + LIST_CONVERSION_ERROR)
        banner = "!" * 78
        print(
            f"{banner}\nWARNING {LIST_CONVERSION_WARNING}\n{banner}",
            file=sys.stderr,
            flush=True,
        )
    try:
        probes = probe_variants(
            args.variants, args.opencl_device, args.cuda_device, args.metal_device
        )
    except RuntimeError as error:
        parser.error(str(error))
    skipped = skipped_variants(args.variants, probes)
    for entry in skipped:
        print(f"SKIPPED {entry['variant']}: {entry['reason']}", flush=True)
    variants_run = [
        v for v in args.variants if v not in {e["variant"] for e in skipped}
    ]
    module_paths = {
        name: importlib.import_module(name).__file__
        for name in required
        if name not in ("h5py", "hdf5plugin")
    }
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "benchmark-results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    dump(output / "skipped.json", skipped)
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
            hardware=capture(),
            cpu_budgets=workers,
            process_counts=process_counts,
            binding=args.bind,
            rank_commands={
                count: [" ".join(prefix) for prefix in prefixes]
                for count, prefixes in bindings.items()
            },
            numa_nodes=numa_nodes()
            if Path("/sys/devices/system/node").is_dir()
            else None,
            arguments={
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
                # CUDA/Metal options only when a matching row was requested, so
                # other runs record the same arguments as before.
                if (
                    k not in ("cuda_device", "probe_cuda")
                    or any(v in CUDA_VARIANTS for v in args.variants)
                )
                and (
                    k not in ("metal_device", "probe_metal")
                    or any(v in METAL_VARIANTS for v in args.variants)
                )
            },
            source=provenance,
            manifest_sha256=sha256(MANIFEST),
            runner_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "compare.py",
                    "fixtures.py",
                    "hardware.py",
                    "provenance.py",
                    "prepare_subset.py",
                )
            },
            installed_build=installed_build,
            azint_build=azint_build,
            variant_probes=probes,
            skipped_variants=skipped,
            case_threads={
                variant: {
                    f"{count}x{p}" if p > 1 else count: case_threads(variant, count, p)
                    for count in workers
                    for p in process_counts
                    if p <= count and count % p == 0
                }
                for variant in args.variants
            },
            build_verification_problems=build_problems,
            dioptrin_result_arrays=result_arrays,
            module_paths=module_paths,
            total_memory_mib=round(total_memory_bytes() / 2**20, 1),
            memory_model_2d=MEMORY_MODEL,
            uncertainty_support_2d=UNCERTAINTY_2D,
        ),
    )
    (output / "datasets.json").write_bytes(MANIFEST.read_bytes())
    metadata = {}
    for dataset in args.datasets:
        print(f"Preparing {dataset} outside timing...", flush=True)
        metadata[dataset] = prepare(
            output / dataset,
            dataset,
            args.data_dir,
            args.cache_dir,
            args.offline,
            beam_pixel=args.mask_beam_pixel,
            subset_dir=args.subset_dir,
        )
        beam = metadata[dataset].get("beam_pixel_mask")
        if beam:
            print(
                f"{dataset}: masked {beam['mode']} pixel(s) {beam['pixels']} "
                f"(already masked: {beam['already_masked']}; minimum-2theta pixel "
                f"{beam['min_2theta_pixel']})",
                flush=True,
            )
    failures = []
    memory_limit = args.max_memory_fraction * total_memory_bytes() / 2**20
    preflights = {}
    variants_2d = list(variants_run)
    if "2d" in args.dimensions and variants_2d:
        for dataset in args.datasets:
            print(
                f"2D preflight for {dataset} (azimuth convention, axes, estimates)...",
                flush=True,
            )
            budget = str(max(workers))
            log_path = output / dataset / "preflight2d.log"
            with log_path.open("w") as log:
                code = subprocess.run(
                    [
                        sys.executable,
                        "-E",
                        "-s",
                        __file__,
                        "--preflight-2d",
                        str(output / dataset),
                        "--cakes",
                        *args.cakes,
                        "--subpixels",
                        *map(str, args.subpixels),
                        "--variants",
                        *variants_2d,
                    ],
                    env=os.environ
                    | case_threads("dioptrin-pixel_split", 1)
                    | {"RAYON_NUM_THREADS": budget, "OMP_NUM_THREADS": budget},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
            if code:
                failures.append(dict(case=f"{dataset}-2d-preflight", exit_code=code))
                print(f"FAILED: 2D preflight for {dataset}; see {log_path}", flush=True)
                continue
            preflight = json.loads((output / dataset / "preflight2d.json").read_text())
            convention = preflight["convention"]
            print(
                f"{dataset}: azimuth convention "
                + ("agrees" if convention["passed"] else "DISAGREES")
                + " ("
                + ", ".join(
                    f"{k} {'ok' if v['passed'] else 'FAILED'}"
                    for k, v in convention["libraries"].items()
                )
                + ")",
                flush=True,
            )
            if not convention["passed"]:
                failures.append(
                    dict(
                        case=f"{dataset}-2d-convention",
                        reason="azimuth convention check failed; see preflight2d.json",
                    )
                )
                continue
            preflights[dataset] = preflight
        dump(output / "failures.json", failures)

    def skip(label, variant, reason, **extra):
        skipped.append(dict(case=label, variant=variant, reason=reason, **extra))
        dump(output / "skipped.json", skipped)
        print(f"SKIPPED {label}: {reason}", flush=True)

    def execute(label, case, case_path):
        """Run one case (fresh process, or coordinated processes) and record it."""
        processes = case["processes"]
        dump(case_path, case)
        env = os.environ | case["threads"]
        log_path = case_path.with_suffix(".log")
        if processes > 1:
            try:
                line = run_processes(case, case_path, env, bindings[processes])
                log_path.write_text(line + "\n")
                code = 0
            except Exception as error:
                log_path.write_text(f"{error!r}\n")
                code = 1
        else:
            with log_path.open("w") as log:
                code = subprocess.run(
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
                ).returncode
        unsupported = Path(case["result"]).with_suffix(".unsupported.json")
        if code == UNSUPPORTED_EXIT and unsupported.is_file():
            reason = json.loads(unsupported.read_text())["reason"]
            skip(label, case["variant"], f"unsupported: {reason}")
        elif code:
            failures.append(dict(case=label, exit_code=code))
            print(
                f"FAILED: {label}; see {log_path}",
                flush=True,
            )
        else:
            print(log_path.read_text().strip(), flush=True)
        summarize(output)
        dump(output / "failures.json", failures)

    def base_case(dataset, workload, variant, count, processes, errors, trial):
        beam = metadata[dataset].get("beam_pixel_mask")
        return dict(
            dataset=dataset,
            workload=workload,
            variant=variant,
            workers=count,
            errors=errors,
            error_mode="poisson" if errors else "intensity",
            trial=trial,
            fixture=str(output / dataset),
            frames=((max(args.frames, count) + count - 1) // count) * count,
            bins=args.bins,
            blocks=args.blocks,
            seconds=args.seconds,
            processes=processes,
            workers_per_process=count // processes,
            binding=args.bind if processes > 1 else "none",
            mask_beam_pixel=args.mask_beam_pixel,
            beam_pixels_added=len(beam["pixels"]) - len(beam["already_masked"])
            if beam
            else 0,
            opencl_device=args.opencl_device,
            allow_list_conversion=args.allow_list_conversion,
        )

    opencl_2d = (probes.get("opencl") or {}).get("available_2d", False)
    for trial in range(args.trials):
        variants = variants_run if trial % 2 == 0 else list(reversed(variants_run))
        for dataset in args.datasets:
            for workload in args.workloads:
                if workload == "frames" and metadata[dataset]["distinct_frames"] < 2:
                    continue
                if "1d" in args.dimensions:
                    for count in workers:
                        for processes in process_counts:
                            if processes > count or count % processes:
                                continue
                            for errors in (False, True):
                                for variant in variants:
                                    if variant in GPU_VARIANTS and (
                                        count > 1 or processes > 1
                                    ):
                                        continue  # one host thread drives the device
                                    dioptrin_cpu = (
                                        variant.startswith("dioptrin-")
                                        and variant not in DIOPTRIN_GPU_VARIANTS
                                    )
                                    for path_1d in (
                                        args.dioptrin_1d_path if dioptrin_cpu else [None]
                                    ):
                                        # Non-default 1D paths are part of the
                                        # case name, so several paths in one run
                                        # never share files or references.
                                        how = (
                                            f"-{path_1d}"
                                            if path_1d not in (None, "global")
                                            else ""
                                        )
                                        suffix = f"{dataset}-{workload}-{variant}{how}"
                                        mode = "poisson" if errors else "intensity"
                                        split = f"-p{processes}" if processes > 1 else ""
                                        label = f"{suffix}-w{count}{split}-{mode}-t{trial}"
                                        case_path = output / "cases" / f"{label}.json"
                                        case = base_case(
                                            dataset,
                                            workload,
                                            variant,
                                            count,
                                            processes,
                                            errors,
                                            trial,
                                        )
                                        case["dioptrin_1d_path"] = path_1d
                                        case["result"] = str(
                                            case_path.with_suffix(".result.json")
                                        )
                                        if count > 1 or trial > 0:
                                            case["reference"] = str(
                                                output
                                                / "cases"
                                                / f"{suffix}-w1-{mode}-t0.result.npz"
                                            )
                                        if variant in CUDA_VARIANTS:
                                            case["cuda_device"] = args.cuda_device
                                            case["gpu_precision"] = args.gpu_precision
                                        elif variant in METAL_VARIANTS:
                                            case["metal_device"] = args.metal_device
                                            case["gpu_precision"] = args.gpu_precision
                                        case["input_dtype"] = args.input_dtype
                                        case["threads"] = case_threads(
                                            variant, count, processes
                                        )
                                        execute(label, case, case_path)
                if "2d" not in args.dimensions or dataset not in preflights:
                    continue
                preflight = preflights[dataset]
                for cake in args.cakes:
                    n_radial, n_azimuthal = parse_cake(cake)
                    for count in workers:
                        for processes in process_counts:
                            if processes > count or count % processes:
                                continue
                            for errors in (False, True):
                                for variant in variants:
                                    if variant in GPU_VARIANTS and (
                                        count > 1 or processes > 1
                                    ):
                                        continue
                                    counts = (
                                        args.subpixels
                                        if variant in SUBPIXEL_VARIANTS
                                        else [None]
                                    )
                                    dioptrin = (
                                        variant.startswith("dioptrin-")
                                        and variant not in DIOPTRIN_GPU_VARIANTS
                                    )
                                    path_2d = (
                                        args.dioptrin_2d_path if dioptrin else None
                                    )
                                    splits = (
                                        args.dioptrin_2d_batch_split
                                        if path_2d == "batch"
                                        else [None]
                                    )
                                    for subpixels, batch_split in (
                                        (s, b) for s in counts for b in splits
                                    ):
                                        ss = f"-ss{subpixels}" if subpixels else ""
                                        # Non-default Dioptrin paths and batch
                                        # splits are part of the case name, so
                                        # several splits in one run never share
                                        # files or one-worker references.
                                        how = (
                                            f"-{path_2d}"
                                            if path_2d not in (None, "sequential")
                                            else ""
                                        ) + (f"-{batch_split}" if batch_split else "")
                                        suffix = f"{dataset}-{workload}-2d-{cake}-{variant}{ss}{how}"
                                        mode = "poisson" if errors else "intensity"
                                        split = (
                                            f"-p{processes}" if processes > 1 else ""
                                        )
                                        label = (
                                            f"{suffix}-w{count}{split}-{mode}-t{trial}"
                                        )
                                        case_path = output / "cases" / f"{label}.json"
                                        case = base_case(
                                            dataset,
                                            workload,
                                            variant,
                                            count,
                                            processes,
                                            errors,
                                            trial,
                                        )
                                        method = (
                                            dioptrin_method(variant)
                                            if variant.startswith("dioptrin-")
                                            else variant
                                        )
                                        case.update(
                                            dimension="2d",
                                            bins=n_radial * n_azimuthal,
                                            cake=cake,
                                            n_radial=n_radial,
                                            n_azimuthal=n_azimuthal,
                                            subpixels=subpixels,
                                            azimuth_range_deg=list(AZIMUTH_RANGE_DEG),
                                            radial_range_2d=preflight[
                                                "radial_range_2d"
                                            ],
                                            dioptrin_2d_path=path_2d,
                                            dioptrin_2d_batch_split=batch_split,
                                            result=str(
                                                case_path.with_suffix(".result.json")
                                            ),
                                        )
                                        if variant in CUDA_VARIANTS:
                                            case["cuda_device"] = args.cuda_device
                                            case["gpu_precision"] = args.gpu_precision
                                        elif variant in METAL_VARIANTS:
                                            case["metal_device"] = args.metal_device
                                            case["gpu_precision"] = args.gpu_precision
                                        case["input_dtype"] = args.input_dtype
                                        if variant.startswith("dioptrin-"):
                                            key = method + (
                                                f"-ss{subpixels}" if subpixels else ""
                                            )
                                            case["dioptrin_radial_axis"] = preflight[
                                                "dioptrin_radial_axes"
                                            ][key][str(n_radial)]
                                        case.update(
                                            estimate_memory_mib(
                                                case,
                                                metadata[dataset],
                                                preflight["table_estimates"],
                                            ),
                                            memory_limit_mib=round(memory_limit, 1),
                                        )
                                        if count > 1 or trial > 0 or processes > 1:
                                            case["reference"] = str(
                                                output
                                                / "cases"
                                                / f"{suffix}-w1-{mode}-t0.result.npz"
                                            )
                                        if (
                                            variant.startswith("azint")
                                            and n_azimuthal % 2
                                        ):
                                            skip(
                                                label,
                                                variant,
                                                "azint bins azimuth on [0, 360); mapping its rows "
                                                "onto [-180, 180) needs an even n_azimuthal",
                                            )
                                            continue
                                        if (
                                            variant == "pyfai-full-opencl"
                                            and not opencl_2d
                                        ):
                                            skip(
                                                label,
                                                variant,
                                                "no pyFAI 2D OpenCL CSR method",
                                            )
                                            continue
                                        if case["estimated_memory_mib"] > memory_limit:
                                            skip(
                                                label,
                                                variant,
                                                f"estimated peak memory {case['estimated_memory_mib']:.0f} MiB "
                                                f"exceeds {args.max_memory_fraction:g} of RAM "
                                                f"({memory_limit:.0f} MiB)",
                                                estimated_memory_mib=case[
                                                    "estimated_memory_mib"
                                                ],
                                                memory_limit_mib=round(memory_limit, 1),
                                            )
                                            continue
                                        if (
                                            case.get("reference")
                                            and not Path(case["reference"]).is_file()
                                        ):
                                            skip(
                                                label,
                                                variant,
                                                "its one-worker reference case did not run "
                                                "(skipped, unsupported or failed)",
                                            )
                                            continue
                                        case["threads"] = case_threads(
                                            variant, count, processes
                                        )
                                        execute(label, case, case_path)
    if failures:
        raise SystemExit(
            f"{len(failures)} cases failed; partial results and inputs retained in {output}"
        )
    if not args.keep_inputs:
        for dataset in args.datasets:
            for filename in ("frames.npy", "mask.npy"):
                (output / dataset / filename).unlink()
        # Full 2D reference stacks (every distinct cake) only serve the
        # self-checks; numerical.json uses the *.cake0.npz files, which stay.
        for path in (output / "cases").glob("*-2d-*.result.npz"):
            path.unlink()
    if not (output / "summary.md").is_file():
        summarize(output)  # no case ran (e.g. every variant skipped)
    print(f"Results: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    # Keep the parent and case processes on the same interpreter paths even if
    # a shell/module exports PYTHONPATH or enables user-installed packages.
    if not sys.flags.ignore_environment or not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, "-E", "-s", *sys.argv])
    main()
