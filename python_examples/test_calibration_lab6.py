"""
Calibration integration test on the real LaB6_40keV_MarCCD test image.

This test does *not* require the image data itself for the refinement: it
uses the reference PONI (a known-good calibration) as ground truth, builds
synthetic ring points using pyFAI's forward model, perturbs the PONI to
make a starting guess, and verifies that Dioptrin's refiner recovers the
ground-truth geometry from those synthetic peaks.

We compare against pyFAI's forward model (not Dioptrin's) on purpose, so
the test exercises the whole refinement chain end-to-end without trusting
Dioptrin's own 2θ computation to validate itself.
"""

import math
import sys

import dioptrin
import numpy as np

try:
    import pyFAI
except ImportError:
    print("pyFAI not installed; skipping calibration integration test.")
    sys.exit(77)


PONI_PATH = "test_data/LaB6_40keV_MarCCD.poni"
CALIBRANT_PATH = "test_data/calibrants/LaB6_SRM660b.D"
# Detector size of the MarCCD frame in the Dioptas test corpus.
DIM = 2048


def pyfai_2theta_at_pixel(ai, d1, d2):
    """2θ (rad) at a single sub-pixel position using pyFAI's geometry."""
    d1a = np.array([d1], dtype=float)
    d2a = np.array([d2], dtype=float)
    return float(ai.tth(d1a, d2a)[0])


def pixel_on_ring(ai, ring_tth, azimuth, dim):
    """Find a sub-pixel (d1, d2) on the given ring at the given azimuth via
    bisection along a radial ray from the beam center.

    Returns ``None`` if the ring does not intersect the radial ray within the
    detector window."""
    cx = ai.poni1 / ai.pixel1
    cy = ai.poni2 / ai.pixel2
    du, dv = math.cos(azimuth), math.sin(azimuth)

    def tth_at(r):
        return pyfai_2theta_at_pixel(ai, cx + r * du, cy + r * dv)

    # Bracket the ring 2θ.
    lo = 0.5
    hi = 0.5
    g_lo = tth_at(lo) - ring_tth
    found = False
    for _ in range(80):
        hi *= 1.5
        if hi > 10_000:
            break
        g_hi = tth_at(hi) - ring_tth
        if (g_lo < 0) != (g_hi < 0):
            found = True
            break
        lo, g_lo = hi, g_hi
    if not found:
        return None
    # Bisect.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g_mid = tth_at(mid) - ring_tth
        if (g_mid < 0) == (g_lo < 0):
            lo, g_lo = mid, g_mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-9:
            break
    r = 0.5 * (lo + hi)
    return cx + r * du, cy + r * dv


def main():
    ai = pyFAI.load(PONI_PATH)
    truth_poni = dict(
        pixel1=ai.pixel1,
        pixel2=ai.pixel2,
        distance=ai.dist,
        poni1=ai.poni1,
        poni2=ai.poni2,
        rot1=ai.rot1,
        rot2=ai.rot2,
        rot3=ai.rot3,
        wavelength=ai.wavelength,
    )
    print(f"Ground-truth PONI (from {PONI_PATH}):")
    for k, v in truth_poni.items():
        print(f"  {k}: {v}")

    cal = dioptrin.Calibrant.from_file(CALIBRANT_PATH)
    # Pick a few low-angle rings whose 2θ stays below the detector limit.
    ring_tth_all = cal.two_theta_rad(truth_poni["wavelength"])
    chosen_rings = []
    for k, t in enumerate(ring_tth_all):
        if t is None:
            continue
        # Skip rings beyond ~12° 2θ (cap roughly by detector extent).
        if t > 0.20:
            continue
        chosen_rings.append((k, t))
    print(f"Using {len(chosen_rings)} rings for synthetic peaks.")

    # Sample ~16 points per ring around full azimuth.
    pts = dioptrin.CalibrationPoints()
    n_added = 0
    for k, t in chosen_rings:
        for j in range(16):
            az = 2 * math.pi * j / 16
            xy = pixel_on_ring(ai, t, az, DIM)
            if xy is None:
                continue
            d1, d2 = xy
            if 0 <= d1 < DIM and 0 <= d2 < DIM:
                pts.add(d1, d2, k)
                n_added += 1
    print(f"Sampled {n_added} on-detector control points.")
    assert n_added > 80, f"too few points: {n_added}"

    # Perturb the truth → starting guess.
    guess = dict(truth_poni)
    guess["distance"] *= 1.02            # +2 %
    guess["poni1"] += 8.0 * ai.pixel1    # +8 pixels
    guess["poni2"] -= 5.0 * ai.pixel2    # −5 pixels
    guess["rot1"] += 5e-3
    guess["rot2"] -= 3e-3

    # Fix rot3 in addition to wavelength: with a near-perpendicular detector,
    # in-plane rotation is a near-degenerate parameter that can absorb poni1/2
    # changes. pyFAI/Dioptas also fix it by default.
    result = dioptrin.refine(
        guess, pts, cal, fix=["wavelength", "rot3"], max_iter=200
    )
    refined = result.poni
    print(f"\nrefine completed in {result.n_iter} iterations")
    print(f"  RMS residual: {result.rms_rad*1e6:.3f} µrad of 2θ")

    # Compare refined vs truth.
    print("\nParameter recovery (refined vs truth):")
    deltas = {
        "distance": refined["distance"] - truth_poni["distance"],
        "poni1":    refined["poni1"]    - truth_poni["poni1"],
        "poni2":    refined["poni2"]    - truth_poni["poni2"],
        "rot1":     refined["rot1"]     - truth_poni["rot1"],
        "rot2":     refined["rot2"]     - truth_poni["rot2"],
        "rot3":     refined["rot3"]     - truth_poni["rot3"],
    }
    for k, d in deltas.items():
        print(f"  Δ{k:<9} = {d:+.3e}")

    # Tolerances chosen for synthetic-from-pyFAI data: numerical-noise level.
    assert abs(deltas["distance"]) < 1e-6, "distance not recovered"
    assert abs(deltas["poni1"])    < 1e-6
    assert abs(deltas["poni2"])    < 1e-6
    assert abs(deltas["rot1"])     < 1e-5
    assert abs(deltas["rot2"])     < 1e-5
    assert result.rms_rad < 1e-7, "RMS residual too high"
    print("\n✅ Calibration recovered LaB6 reference PONI to numerical precision.")


if __name__ == "__main__":
    main()
