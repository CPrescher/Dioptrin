"""
End-to-end test of `dioptrin.identify_rings` on real LaB6 data.

  1. Drive Dioptas to pick a multi-ring set of peaks on
     `LaB6_40keV_MarCCD.tif` (Dioptas labels each point with its ring index).
  2. Run `dioptrin.ransac_ellipses` on the un-labelled coordinates.
  3. Run `dioptrin.identify_rings` on the detected ellipses + the LaB6
     calibrant.
  4. Compare Dioptrin's inferred ring index for each ellipse against the
     Dioptas labels of its inlier points: the algorithm should converge
     on the same calibrant ring index Dioptas's iterative fit assigned.

This is the end-to-end "given an image + calibrant, identify which detected
ellipses are which Bragg rings, with no manual hints" validation.
"""

import io
import os
import sys
import warnings

import numpy as np

try:
    from dioptas.model.CalibrationModel import CalibrationModel
    from dioptas.model.ImgModel import ImgModel
    from dioptas import calibrants_path as dioptas_calibrants_path
except ImportError:
    print("Dioptas not importable; skipping.")
    sys.exit(77)

try:
    import dioptrin
except ImportError:
    print("dioptrin not installed; skipping.")
    sys.exit(77)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMAGE = os.path.join(REPO, "test_data", "LaB6_40keV_MarCCD.tif")
SEEDS = [(1179.6, 1129.4, 0), (1268.5, 1119.8, 1)]
WAVELENGTH = 3.1e-11   # 40 keV — from the LaB6_40keV PONI fixture


def silent(fn):
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn()
    finally:
        sys.stdout = saved


def gather_labelled_peaks():
    img = ImgModel()
    img.load(IMAGE)
    cm = CalibrationModel(img)
    for x, y, ring in SEEDS:
        cm.find_peaks_automatic(x, y, ring)
    cm.set_calibrant(os.path.join(dioptas_calibrants_path, "LaB6.D"))
    cm.calibrate()
    cm.setup_peak_search_algorithm("Massif")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ring_index in range(2, 10):
            try:
                silent(lambda ri=ring_index: cm.search_peaks_on_ring(ri))
            except Exception:
                break
    cm.refine()
    return cm.get_point_array().copy()


def main():
    pts3 = gather_labelled_peaks()
    n_dioptas_rings = int(pts3[:, 2].max()) + 1
    print(f"Dioptas labels: {len(pts3)} peaks across {n_dioptas_rings} rings")

    pts = [(float(p[0]), float(p[1])) for p in pts3]
    ring_truth = np.array([int(p[2]) for p in pts3])

    print()
    print("Step 1: RANSAC cluster the unlabelled peaks into ellipses…")
    ransac = dioptrin.ransac_ellipses(
        pts,
        inlier_threshold=1.5,
        min_inliers=8,
        max_iterations=3000,
        max_ellipses=n_dioptas_rings + 2,
        max_axis_ratio=1.15,
        rng_seed=42,
    )
    print(f"  RANSAC detected {ransac.n_ellipses()} ellipses")

    # Dedup near-duplicate ellipses (multiple RANSAC fits of the same
    # physical ring) before ring-identity matching. We keep the largest
    # ellipse out of each cluster of "similar size and centre". A native
    # dedup step inside `ransac_ellipses` is a planned slice-3c improvement;
    # for now the dedup happens here at the Python boundary.
    raw = list(ransac.ellipses)
    raw.sort(key=lambda e: e.semi_major)
    deduped = []
    for e in raw:
        merged = False
        for d in deduped:
            same_size = abs(e.semi_major - d.semi_major) / d.semi_major < 0.02
            same_center = (
                abs(e.center[0] - d.center[0]) ** 2 +
                abs(e.center[1] - d.center[1]) ** 2
            ) ** 0.5 < 5.0
            if same_size and same_center:
                merged = True
                break
        if not merged:
            deduped.append(e)
    print(f"  after dedup: {len(deduped)} ellipses")

    cal = dioptrin.Calibrant.from_file(
        os.path.join(dioptas_calibrants_path, "LaB6.D")
    )

    print()
    print("Step 2: Identify which calibrant ring each ellipse is…")
    ident = dioptrin.identify_rings(
        deduped, cal, WAVELENGTH,
        ratio_tolerance=0.03, max_anchor_ring=6, min_matches=3,
    )
    print(f"  confident: {ident.confident}, n_matched={ident.n_matched}")
    print(f"  anchor_ring (smallest ellipse's ring): {ident.anchor_ring}")

    print()
    print("Per-ellipse comparison (dioptrin ↔ Dioptas):")
    # For each deduped ellipse, find its Dioptas majority ring by selecting
    # the points within Sampson distance of that ellipse and looking at
    # their Dioptas ring labels.
    n_correct = 0
    n_with_label = 0
    for k, e in enumerate(deduped):
        diop_ring = ident.ring_index[k]
        # Points within 2 px of this ellipse curve
        on_ellipse = np.array([e.distance(p) for p in pts]) < 2.0
        if not on_ellipse.any():
            print(f"  ell {k}: semi={e.semi_major:6.1f}  dioptrin→{diop_ring}  "
                  f"(no Dioptas labels within 2 px)")
            continue
        counts = np.bincount(ring_truth[on_ellipse], minlength=n_dioptas_rings)
        dioptas_ring = int(counts.argmax())
        dioptas_purity = counts[dioptas_ring] / counts.sum()
        match = "✓" if diop_ring == dioptas_ring else "✗"
        if diop_ring == dioptas_ring:
            n_correct += 1
        n_with_label += 1
        print(f"  ell {k}: semi={e.semi_major:6.1f}  dioptrin→{diop_ring}  "
              f"dioptas→{dioptas_ring} (purity {dioptas_purity:.0%})  {match}")

    print()
    print(f"Correct assignments: {n_correct} / {n_with_label}")
    # Acceptance: most identified ellipses (i.e. those Dioptrin matched to
    # a calibrant ring) should agree with Dioptas's labels, allowing for
    # some unmatched ellipses where Dioptrin opted not to commit.
    ok = ident.confident and n_correct >= max(3, int(0.6 * n_with_label))
    if ok:
        print("\n✅ dioptrin.identify_rings agrees with Dioptas's ring labels.")
        return 0
    print("\n❌ Ring identity assignment disagrees more than expected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
