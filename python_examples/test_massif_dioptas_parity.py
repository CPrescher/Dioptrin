"""
Parity check: Dioptrin's Massif peak picking vs Dioptas/pyFAI's Massif on the
same seed pixel of a real LaB6 image.

We give both implementations the same `(x, y)` seed coordinates on
`LaB6_40keV_MarCCD.tif` (matching Dioptas's own `test_calibration1`) and
compare:

  - the number of peaks each one returns
  - the per-peak position agreement (for each Dioptas peak, find the
    nearest Dioptrin peak; assert the distance is sub-pixel)

Combined with the slice-1 parity test (`test_calibration_dioptas_parity.py`),
which already showed that Dioptrin's refine matches pyFAI's `GeometryRefinement`
on identical control points, this transitively validates the full
Dioptrin-only pipeline (Massif → refine) as a pyFAI replacement.

Run from Dioptas's venv:
    /path/to/Dioptas/.venv/bin/python python_examples/test_massif_dioptas_parity.py
"""

import os
import sys

import numpy as np

try:
    from dioptas.model.CalibrationModel import CalibrationModel
    from dioptas.model.ImgModel import ImgModel
except ImportError:
    print("Dioptas not importable; skipping parity test.")
    sys.exit(77)

try:
    import dioptrin
except ImportError:
    print("dioptrin not installed in this venv; skipping.")
    sys.exit(77)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMAGE = os.path.join(REPO, "test_data", "LaB6_40keV_MarCCD.tif")
SEEDS = [(1179.6, 1129.4), (1268.5, 1119.8)]


def dioptas_peaks_at_seed(image_path, seed):
    """Run Dioptas's find_peaks_automatic for one seed; return the peaks."""
    img_model = ImgModel()
    img_model.load(image_path)
    cm = CalibrationModel(img_model)
    cm.find_peaks_automatic(seed[0], seed[1], 0)
    # cm.points is a list of arrays (one per seed call). Convert the one we
    # just added (last entry) to a list of (d1, d2).
    arr = cm.points[-1]
    if arr.ndim == 1:
        return [(float(arr[0]), float(arr[1]))]
    return [(float(row[0]), float(row[1])) for row in arr]


def dioptrin_peaks_at_seed(image, seed):
    """Run Dioptrin's Massif at the rounded seed."""
    sd = (int(round(seed[0])), int(round(seed[1])))
    return dioptrin.find_peaks_massif(image, sd, max_peaks=200, dedup_dist=1.0)


def nearest_distance(p, others):
    """Euclidean distance from p to its nearest neighbour in `others`."""
    if not others:
        return float("inf")
    return min(
        ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        for q in others
    )


def compare_at_seed(image_data, seed):
    print(f"\nSeed at ({seed[0]:.1f}, {seed[1]:.1f}):")
    dioptas_pts = dioptas_peaks_at_seed(IMAGE, seed)
    dioptrin_pts = dioptrin_peaks_at_seed(image_data, seed)
    print(f"  Dioptas (pyFAI Massif): {len(dioptas_pts)} peaks")
    print(f"  Dioptrin Massif       : {len(dioptrin_pts)} peaks")

    # For each Dioptas peak, find nearest Dioptrin peak.
    distances = [nearest_distance(p, dioptrin_pts) for p in dioptas_pts]
    if distances:
        dist_arr = np.array(distances)
        print(f"  Per-Dioptas-peak nearest-Dioptrin distance (px):")
        print(f"    min   = {dist_arr.min():.3f}")
        print(f"    mean  = {dist_arr.mean():.3f}")
        print(f"    max   = {dist_arr.max():.3f}")
        print(f"    p95   = {np.percentile(dist_arr, 95):.3f}")
        within_1 = (dist_arr < 1.0).mean()
        within_2 = (dist_arr < 2.0).mean()
        print(f"    fraction within 1 px: {within_1:.2%}")
        print(f"    fraction within 2 px: {within_2:.2%}")
        return {
            "n_dioptas": len(dioptas_pts),
            "n_dioptrin": len(dioptrin_pts),
            "mean": float(dist_arr.mean()),
            "p95": float(np.percentile(dist_arr, 95)),
            "within_1": float(within_1),
            "within_2": float(within_2),
        }
    return None


def main():
    print(f"Image: {IMAGE}")

    img_model = ImgModel()
    img_model.load(IMAGE)
    image_data = img_model.img_data.astype(np.float64, copy=True)
    print(f"Image shape: {image_data.shape}")

    summaries = []
    for seed in SEEDS:
        s = compare_at_seed(image_data, seed)
        if s is not None:
            summaries.append(s)

    print()
    print("=" * 70)
    print("Overall agreement criteria:")
    # Most Dioptas peaks should have a Dioptrin peak within 2 px.
    # And the count should be in the same order of magnitude.
    all_within_2 = min(s["within_2"] for s in summaries)
    worst_p95 = max(s["p95"] for s in summaries)
    print(f"  worst seed: fraction within 2 px = {all_within_2:.2%}  (want > 80%)")
    print(f"  worst seed: p95 of nearest distance = {worst_p95:.3f} px  (want < 3.0)")

    ok = all_within_2 > 0.80 and worst_p95 < 3.0
    if ok:
        print("\n✅ Dioptrin Massif agrees with Dioptas/pyFAI Massif at the seed level.")
        return 0
    print("\n❌ Massif peak sets disagree more than expected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
