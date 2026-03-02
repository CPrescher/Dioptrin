import argparse
import time

import numpy as np
import fabio
import matplotlib.pyplot as plt
from pyFAI.integrator.azimuthal import AzimuthalIntegrator


def time_method(label: str, func, runs: int = 5) -> None:
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    avg = sum(durations) / runs
    print(
        f"{label} timing over {runs} runs: avg={avg:.4f}s min={min(durations):.4f}s max={max(durations):.4f}s"
    )


def main(plot: bool = False) -> None:
    pattern_rust_hist = np.loadtxt("test_data/test_output/pattern.txt")
    tth_rust_hist = pattern_rust_hist[:, 0]
    int_rust_hist = pattern_rust_hist[:, 1]
    pattern_rust_lut = np.loadtxt("test_data/test_output/pattern_lut.txt")
    tth_rust_lut = pattern_rust_lut[:, 0]
    int_rust_lut = pattern_rust_lut[:, 1]
    pattern_rust_lut_ss = np.loadtxt("test_data/test_output/pattern_lut_ss.txt")
    tth_rust_lut_ss = pattern_rust_lut_ss[:, 0]
    int_rust_lut_ss = pattern_rust_lut_ss[:, 1]

    pattern_rust_ps = np.loadtxt("test_data/test_output/pattern_lut_ps.txt")
    tth_rust_ps = pattern_rust_ps[:, 0]
    int_rust_ps = pattern_rust_ps[:, 1]

    pattern_rust_lut_err = np.loadtxt("test_data/test_output/pattern_lut_err.txt")
    tth_rust_lut_err = pattern_rust_lut_err[:, 0]
    int_rust_lut_err = pattern_rust_lut_err[:, 1]
    err_rust_lut_err = pattern_rust_lut_err[:, 2]

    pattern_rust_lut_ps_err = np.loadtxt("test_data/test_output/pattern_lut_ps_err.txt")
    tth_rust_lut_ps_err = pattern_rust_lut_ps_err[:, 0]
    int_rust_lut_ps_err = pattern_rust_lut_ps_err[:, 1]
    err_rust_lut_ps_err = pattern_rust_lut_ps_err[:, 2]

    azimuthal_integrator = AzimuthalIntegrator()
    azimuthal_integrator.load("test_data/v1.poni")
    fabio_data = fabio.open("test_data/CeO2_Pilatus1M.tif").data[::-1]

    tth_pyfai_hist, int_pyfai_hist = azimuthal_integrator.integrate1d(
        fabio_data,
        len(int_rust_hist),
        unit="2th_rad",
        method=("no", "histogram", "cython"),
        polarization_factor=0.99,
    )
    time_method(
        "pyFAI hist",
        lambda: azimuthal_integrator.integrate1d(
            fabio_data,
            len(int_rust_hist),
            unit="2th_rad",
            method=("no", "histogram", "cython"),
            polarization_factor=0.99,
        ),
    )
    tth_pyfai_lut, int_pyfai_lut = azimuthal_integrator.integrate1d(
        fabio_data,
        len(int_rust_lut),
        unit="2th_rad",
        method=("no", "lut", "cython"),
        polarization_factor=0.99,
    )

    time_method(
        "pyFAI lut",
        lambda: azimuthal_integrator.integrate1d(
            fabio_data,
            len(int_rust_lut),
            unit="2th_rad",
            method=("no", "lut", "cython"),
            polarization_factor=0.99,
        ),
    )

    tth_pyfai_ps_bbox, int_pyfai_ps_bbox = azimuthal_integrator.integrate1d(
        fabio_data,
        len(int_rust_lut_ss),
        unit="2th_rad",
        method=("bbox", "csr", "cython"),
        polarization_factor=0.99,
    )

    time_method(
        "pyFAI ps_bbox",
        lambda: azimuthal_integrator.integrate1d(
            fabio_data,
            len(int_rust_lut_ss),
            unit="2th_rad",
            method=("bbox", "csr", "cython"),
            polarization_factor=0.99,
        ),
    )

    tth_pyfai_ps_full, int_pyfai_ps_full = azimuthal_integrator.integrate1d(
        fabio_data,
        len(int_rust_lut_ss),
        unit="2th_rad",
        method=("full", "csr", "cython"),
        polarization_factor=0.99,
    )

    time_method(
        "pyFAI ps_full",
        lambda: azimuthal_integrator.integrate1d(
            fabio_data,
            len(int_rust_lut_ss),
            unit="2th_rad",
            method=("full", "csr", "cython"),
            polarization_factor=0.99,
        ),
    )

    def print_comparison_metrics(label: str, actual: np.ndarray[np.float64], expected: np.ndarray[np.float64]) -> None:
        diff = actual - expected
        abs_diff = np.abs(diff)
        rel = abs_diff / (np.abs(expected) + 1e-12)
        print(
            f"{label}: mean_abs={abs_diff.mean():.6g}, "
            f"max_abs={abs_diff.max():.6g}, "
            f"mean_rel={rel.mean():.6g}, "
            f"max_rel={rel.max():.6g}"
        )

    print_comparison_metrics("hist rust vs pyFAI", int_rust_hist, int_pyfai_hist)
    print_comparison_metrics("lut rust vs pyFAI", int_rust_lut, int_pyfai_lut)
    print_comparison_metrics("ps rust vs pyFAI full", int_rust_ps, int_pyfai_ps_full)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(tth_rust_hist, int_rust_hist, label="rust_hist")
        plt.plot(tth_rust_lut, int_rust_lut, label="rust_lut")
        plt.plot(tth_pyfai_hist, int_pyfai_hist, label="pyFAI_hist")
        plt.plot(tth_pyfai_lut, int_pyfai_lut, label="pyFAI lut")
        plt.errorbar(tth_rust_lut_err, int_rust_lut_err, yerr=err_rust_lut_err, label="rust_lut_err")
        plt.legend()
        plt.xlabel("2theta")
        plt.ylabel("Intensity")
        plt.title("Histogram and LUT methods")

        plt.subplot(122)
        plt.errorbar(tth_rust_lut_ps_err, int_rust_lut_ps_err, yerr=err_rust_lut_ps_err, label="rust_lut_ps_err")
        plt.plot(tth_pyfai_ps_bbox, int_pyfai_ps_bbox, label="pyFAI ps_bbox")
        plt.plot(tth_pyfai_ps_full, int_pyfai_ps_full, label="pyFAI ps_full")
        plt.plot(tth_rust_lut_ss, int_rust_lut_ss, label="rust_lut_ss")
        plt.plot(tth_rust_ps, int_rust_ps, label="rust_ps")
        plt.title("Pixel-Split methods")
        plt.legend()
        plt.xlabel("2theta")
        plt.ylabel("Intensity")
        plt.tight_layout()
        plt.show()
        plt.savefig("test_data/test_output/pattern_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D integration comparison between Dioptrin and pyFAI")
    parser.add_argument("--plot", action="store_true", help="Generate and display plots")
    args = parser.parse_args()
    main(plot=args.plot)
