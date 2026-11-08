"""Inspect individual splitting weights and compare with dense detector sampling.

Run from any directory: python python_examples/inspect_full_split.py --report result.json
Builds the Rust weight-export example; requires pyFAI and NumPy. No installed
package is changed. The reference is uniform intensity within a flat pixel,
not a claim about the actual detector's subpixel response.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyFAI
from pyFAI.ext import splitPixel
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from test_multi2d import PONI

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    subprocess.run(["cargo", "build", "--release", "--example", "inspect_split_weights"], cwd=ROOT, check=True)
    executable = ROOT / "target/release/examples/inspect_split_weights"
    epsilon = 1 + np.finfo(np.float32).eps
    shapes = [(31, 37), (23, 29)]
    receipts = []
    panels = {}
    for scenario in ["off_axis", "beam_and_seam", "radial_crop"]:
        geometries = [PONI, dict(PONI, pixel1=1.7e-4, pixel2=2.1e-4,
                                distance=0.14, rot1=0.007, rot2=-0.009, rot3=0.05)]
        if scenario != "beam_and_seam":
            geometries = [dict(p, poni1=-0.003, poni2=-0.002) for p in geometries]
        hi = 0.2 if scenario == "off_axis" else 0.08
        for detector, (p, shape) in enumerate(zip(geometries, shapes)):
            ai = AzimuthalIntegrator(dist=p["distance"], pixel1=p["pixel1"], pixel2=p["pixel2"],
                                    poni1=p["poni1"], poni2=p["poni2"], rot1=p["rot1"],
                                    rot2=p["rot2"], rot3=p["rot3"], wavelength=p["wavelength"])
            corners = ai.corner_array(shape, unit="2th_rad").reshape(-1, 4, 2)
            matrices = {}
            with tempfile.TemporaryDirectory() as directory:
                poni = Path(directory)/"geometry.poni"
                output = Path(directory)/"weights.bin"
                ai.save(str(poni))
                for label, mode, n_ss in [("fractional", "fractional", 0),
                                           ("pyfai_histogram", "pyfai_histogram", 0),
                                           ("subpixels_21", "supersampled", 21),
                                           ("subpixels_41", "supersampled", 41),
                                           ("subpixels_256", "supersampled", 256)]:
                    command = [str(executable), str(poni), *map(str, shape), "12", "17",
                               "0", str(hi*epsilon), str(-np.pi), str(np.pi), "2th_rad",
                               str(output), mode]
                    if n_ss:
                        command.append(str(n_ss))
                    subprocess.run(command, cwd=ROOT, check=True)
                    matrices[label] = np.fromfile(output, dtype="<f8").reshape(-1, 17, 12)
            ref = np.empty_like(matrices["fractional"])
            for i in range(len(corners)):
                # Arrange the reference to have the same actual upper edges as
                # Dioptrin; pyFAI applies its float32-epsilon upper expansion.
                result = splitPixel.fullSplit2D_engine(
                    corners[i:i+1], np.ones(1), (12, 17),
                    pos0_range=(0, hi), pos1_range=(-np.pi, np.pi/epsilon))
                ref[i] = result.signal
            dense = matrices.pop("subpixels_256")
            matrices["pyfai_reference"] = ref
            comparison = {}
            for name, matrix in matrices.items():
                l1 = np.abs(matrix-dense).sum(axis=(1, 2))
                comparison[name] = dict(mean_pixel_L1_vs_dense=float(l1.mean()),
                                        max_pixel_L1_vs_dense=float(l1.max()),
                                        negative_weights=int(np.count_nonzero(matrix < 0)),
                                        minimum_weight=float(matrix.min()))
            if detector == 0:
                for name, pixel in ({"off_axis": [("off_axis", 8*37+34)],
                                     "beam_and_seam": [("beam", 13*37+9), ("seam", 13*37+8)],
                                     "radial_crop": [("crop", 26*37+31)]}[scenario]):
                    panels[name] = {label: matrix[pixel].copy() for label, matrix in matrices.items()}
                    panels[name]["subpixels_256"] = dense[pixel].copy()
            origin_pixel = int(np.argmin(corners[:, :, 0].mean(axis=1)))
            receipt = dict(scenario=scenario, detector=detector, shape=shape,
                           reference_pixels_unmasked=True, dense_subpixels_per_axis=256,
                           origin_pixel_row_column=list(divmod(origin_pixel, shape[1])),
                           comparisons=comparison)
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    report = dict(pyfai=pyFAI.version, comparisons=receipts)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    if args.figure:
        plot_panels(panels, args.figure)


def plot_panels(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"fractional": "Fractional (beam fallback)", "pyfai_histogram": "Compatibility",
              "subpixels_21": "21 × 21 samples", "subpixels_256": "256 × 256 reference",
              "pyfai_reference": "pyFAI full histogram"}
    colors = {"fractional": "#1565c0", "pyfai_histogram": "#ef6c00", "subpixels_21": "#2e7d32",
              "subpixels_256": "#222222", "pyfai_reference": "#c62828"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, key, title in zip(axes.flat, ["off_axis", "beam", "seam", "crop"],
                              ["Ordinary bin-boundary pixel", "Pixel containing the direct beam",
                               "Pixel crossing the azimuthal seam", "Pixel mostly outside radial range"]):
        if key == "crop":
            names = ["fractional", "pyfai_histogram", "subpixels_21", "subpixels_256"]
            ax.bar(range(4), [panels[key][name].sum() for name in names],
                   color=[colors[name] for name in names])
            ax.set_xticks(range(4), ["Fractional", "Compatibility", "21 × 21", "256 × 256"])
            ax.set_ylabel("Total in-range weight")
            ax.set_ylim(0, 1.1)
        else:
            for name in ["fractional", "pyfai_histogram", "subpixels_21", "subpixels_256", "pyfai_reference"]:
                values = panels[key][name].sum(axis=1)
                ax.plot(np.arange(17), values, label=labels[name], color=colors[name],
                        linewidth=1.5, linestyle="--" if name=="pyfai_reference" else "-",
                        marker="." if name=="subpixels_256" else None)
            ax.axhline(0, color="#888888", linewidth=0.6)
            ax.set_xlabel("Azimuthal bin index")
            ax.set_ylabel("Fraction of this pixel")
            if key == "off_axis":
                ax.set_xlim(8.5, 10.5)
                ax.legend(fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.15)
    fig.suptitle("Splitting conventions: isolated-pixel weights, matched grids", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
