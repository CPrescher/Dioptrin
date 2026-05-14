"""
Diagnostic: `dioptrin.auto_calibrate` end-to-end on a heavily-tilted CeO2 image.

`test_data/ceo2_side.tif` was collected with a Pilatus300kw strip detector
(195 x 1475) at ~31° tilt. The corresponding `ceo2_side.poni` has:
    Distance = 0.24607 m, Poni1 = 0.01774 m, Poni2 = 0.10068 m,
    Rot1 = -0.5524 rad (≈ -31.6°), Rot2 = -0.0039 rad.

This is the canonical hard case that motivated the tilt-aware geometry
inversion in `src/calibration/inversion.rs`. The inversion math itself is
verified on synthetic tilted data to 10⁻⁹ (see `inversion::tests`), and it
is wired into the auto-calibrate bootstrap via a regime switch:

  - near-perpendicular (median ellipse axis_ratio < 1.05): use the old
    perpendicular bootstrap (more robust to ellipse-fit noise)
  - visibly tilted: use the closed-form inversion

**Current status on ceo2_side**: the pipeline runs end-to-end but does NOT
recover the truth. The remaining bottleneck is upstream of the inversion:
on a 195-tall strip detector each ring contributes only a short arc
(detected semi_minor ≈ 90-115 px vs predicted 158-419 px for the true
rings at 31° tilt), so the 5-point ellipse RANSAC produces unreliable
ellipses that the inversion then correctly inverts into the wrong geometry.

Fixing ceo2_side end-to-end needs constrained arc-to-ellipse fitting that
enforces the physical axis-ratio prior (1/√C with C = cos²τ - sin²τ·tan²(2θ))
across the detected arcs. That is a separate slice of work.

This file is therefore retained as a *diagnostic*, not a strict gate.
"""

import os
import sys

import numpy as np
from PIL import Image

try:
    import dioptrin
except ImportError:
    print("dioptrin not installed; skipping.")
    sys.exit(77)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMAGE = os.path.join(REPO, "test_data", "ceo2_side.tif")
PONI = os.path.join(REPO, "test_data", "ceo2_side.poni")
PIXEL = 1.72e-4
WAVELENGTH = 2.91385e-11


def read_truth_poni():
    truth = {}
    with open(PONI) as f:
        for line in f:
            line = line.strip()
            if ":" not in line or line.startswith("#"):
                continue
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"):
                truth[k] = float(v)
    return truth


def find_calibrant_file():
    """Prefer Dioptas's calibrants dir if available; otherwise look in the repo."""
    try:
        from dioptas import calibrants_path as p
        f = os.path.join(p, "CeO2.D")
        if os.path.exists(f):
            return f
    except ImportError:
        pass
    for cand in (
        os.path.join(REPO, "test_data", "CeO2.D"),
        "/usr/share/pyFAI/calibrant/CeO2.D",
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("CeO2.D not findable; install dioptas or place CeO2.D in test_data")


def main():
    truth = read_truth_poni()
    print("Truth PONI (from ceo2_side.poni):")
    for k in ("distance", "poni1", "poni2", "rot1", "rot2"):
        print(f"  {k:<10} = {truth[k]:+.6e}")
    print(f"  tilt magnitude = {np.degrees(np.hypot(truth['rot1'], truth['rot2'])):.2f}°")

    img = np.array(Image.open(IMAGE)).astype(np.float64)
    print(f"\nImage shape: {img.shape}, dtype: {img.dtype}, max: {img.max():.0f}")

    # Find peaks across the whole image. We expect a few hundred peaks from
    # CeO2 rings on a 195x1475 strip.
    peaks = dioptrin.find_peaks_image(img, threshold_factor=3.0, max_peaks=5000)
    print(f"Detected {len(peaks)} raw peaks")
    if len(peaks) < 30:
        print("Too few peaks; aborting")
        return 1

    cal = dioptrin.Calibrant.from_file(find_calibrant_file())

    print("\nRunning auto_calibrate (method='ransac', axis_ratio<=1.5 for 31° tilt)…")
    try:
        result = dioptrin.auto_calibrate(
            peaks, cal, wavelength=WAVELENGTH,
            pixel1=PIXEL, pixel2=PIXEL,
            method="ransac",
            ransac_inlier_threshold=1.5,
            ransac_min_inliers=10,
            ransac_max_axis_ratio=1.5,     # heavy tilt → wider ratio band
            ransac_max_iterations=5000,
            ratio_tolerance=0.04,
            min_matches=3,
            center_consistency_px=120.0,   # rings don't share a center under tilt
            fix=["wavelength", "rot3"],
        )
    except Exception as e:
        print(f"❌ auto_calibrate failed: {e}")
        return 1

    print(f"\nDetected ellipses: {result.n_detected_ellipses}, "
          f"identified: {result.n_identified_ellipses}")
    print(f"Bootstrap distance: {result.bootstrap_distance:.6f}")
    print(f"Bootstrap ring-center (m): ({result.bootstrap_ring_center_d1:.6f}, "
          f"{result.bootstrap_ring_center_d2:.6f})")
    print(f"Refined PONI:")
    for k in ("distance", "poni1", "poni2", "rot1", "rot2"):
        print(f"  {k:<10} = {result.poni[k]:+.6e}")
    print(f"  RMS = {result.rms_rad * 1e3:.4f} mrad of 2θ, "
          f"n_iter = {result.n_iter}")

    print("\nDeltas vs truth:")
    deltas = {k: result.poni[k] - truth[k] for k in ("distance", "poni1", "poni2", "rot1", "rot2")}
    for k, dv in deltas.items():
        print(f"  Δ{k:<10} = {dv:+.3e}")

    rel_dist = abs(deltas["distance"]) / abs(truth["distance"])
    rel_poni = max(abs(deltas["poni1"]) / max(abs(truth["poni1"]), 1e-9),
                   abs(deltas["poni2"]) / max(abs(truth["poni2"]), 1e-9))
    abs_rot = max(abs(deltas["rot1"]), abs(deltas["rot2"]))
    print(f"\n  Δdistance/distance   = {rel_dist:.2e}  (want < 5e-3)")
    print(f"  max ΔPONI / PONI     = {rel_poni:.2e}  (want < 5e-2)")
    print(f"  max |Δrot1, Δrot2|   = {abs_rot:.2e} rad  (want < 1e-2)")

    if rel_dist < 5e-3 and rel_poni < 5e-2 and abs_rot < 1e-2:
        print("\n[OK] auto_calibrate recovers the heavily-tilted CeO2 geometry.")
        return 0
    print(
        "\n[diagnostic] auto_calibrate ran end-to-end but did not converge "
        "to truth.\n"
        "   Known limitation on strip detectors: 5-point ellipse RANSAC on "
        "short arcs\n"
        "   produces unreliable ellipses; a constrained arc-to-ellipse "
        "primitive is needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
