"""Where the azimuth seam lies: does a cake depend on how the full circle is written?

Integrates the CeO2 Pilatus 1M test image (test_data/v1.poni) into 300 x 72
bins (2theta in degrees) once with the seam at +-pi, i.e. the range (-pi, pi),
and once with the seam at 0, i.e. (0, 2pi) and (-2pi, 0). The second cake,
with its rows shifted back by half a turn, should equal the first. The script
reports how far it does for every Dioptrin method and for pyFAI's
full-splitting engines (setChiDiscAtPi vs setChiDiscAtZero), plus a
quarter-circle sector written as (2.5, 2.5 + pi/2) and shifted by -2pi.

Run from the repository root with a wheel of Dioptrin, pyFAI and fabio
installed:

    python scripts/compare_azimuth_seam.py docs/validation/pyfai-full-seam.json

See docs/validation/pyfai-full-seam.md for the results and discussion.
"""
import json
import subprocess
from importlib.metadata import version
import platform
import sys
from pathlib import Path

import fabio
import numpy as np
import pyFAI

import dioptrin

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "test_data" / "CeO2_Pilatus1M.tif"
PONI = ROOT / "test_data" / "v1.poni"
N_RADIAL, N_AZIMUTHAL = 300, 72
THRESHOLDS = (1e-9, 1e-6, 1e-3, 1e-2)


def compare(got, ref, radial=None):
    """Relative per-bin difference |got - ref| / max(|ref|, 1)."""
    d = np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)
    big = d > 1e-2
    return {
        "bins": int(d.size),
        "max_relative": float(d.max()),
        "median_relative_of_differing": float(np.median(d[d > 1e-9])) if (d > 1e-9).any() else 0.0,
        **{f"bins_above_{t:g}": int((d > t).sum()) for t in THRESHOLDS},
        "rows_above_1e-9": int(len(set(np.nonzero(d > 1e-9)[0].tolist()))),
        "max_radial_bin_above_1e-2": int(np.nonzero(big)[1].max()) if big.any() else None,
        **(
            {"max_2th_deg_above_1e-2": float(radial[np.nonzero(big)[1]].max()) if big.any() else None}
            if radial is not None
            else {}
        ),
    }, big


def dioptrin_cakes(image):
    out, masks = {}, {}
    for method in ["counting", "pixel_split", "supersampled", "pyfai_full"]:
        ai = dioptrin.Integrator(str(PONI), method=method, unit="2th_deg")

        def cake(rng):
            r = ai.integrate2d(image, N_RADIAL, N_AZIMUTHAL, azimuthal_range=rng)
            return np.asarray(r.intensity).reshape(N_AZIMUTHAL, N_RADIAL)

        base = np.roll(cake((-np.pi, np.pi)), -N_AZIMUTHAL // 2, axis=0)
        out[method] = {}
        for name, rng in [("(0, 2pi)", (0.0, 2 * np.pi)), ("(-2pi, 0)", (-2 * np.pi, 0.0))]:
            out[method][f"seam at 0: {name} vs (-pi, pi)"], big = compare(cake(rng), base)
            if name == "(0, 2pi)":
                masks[method] = big
        sector = (2.5, 2.5 + np.pi / 2)
        shifted = (sector[0] - 2 * np.pi, sector[1] - 2 * np.pi)
        out[method]["sector (2.5, 2.5 + pi/2) vs shifted by -2pi"], _ = compare(
            cake(shifted), cake(sector)
        )
    return out, masks


def pyfai_cakes(image):
    ai = pyFAI.load(str(PONI))
    out, masks = {}, {}
    for method in [("full", "histogram", "cython"), ("full", "csr", "cython"), ("bbox", "histogram", "cython")]:

        def cake(disc_at_zero, rng):
            ai.setChiDiscAtZero() if disc_at_zero else ai.setChiDiscAtPi()
            r = ai.integrate2d_ng(
                image, N_RADIAL, N_AZIMUTHAL, unit="2th_deg", method=method,
                azimuth_range=rng, correctSolidAngle=False,
            )
            return np.asarray(r.intensity, dtype=np.float64), np.asarray(r.radial)

        base, radial = cake(False, (-180.0, 180.0))
        got, _ = cake(True, (0.0, 360.0))
        key = "/".join(method)
        out[key], masks[key] = compare(got, np.roll(base, -N_AZIMUTHAL // 2, axis=0), radial)
    return out, masks


def main(path):
    image = fabio.open(str(IMAGE)).data.astype(np.float64)
    d, d_masks = dioptrin_cakes(image)
    p, p_masks = pyfai_cakes(image)
    a, b = d_masks["pyfai_full"], p_masks["full/histogram/cython"]
    result = {
        "image": IMAGE.name,
        "poni": PONI.name,
        "bins": [N_AZIMUTHAL, N_RADIAL],
        "unit": "2th_deg",
        "metric": "|got - ref| / max(|ref|, 1) per bin; ref is the seam-at-pi cake",
        "versions": {
            "dioptrin": version("dioptrin"),
            "dioptrin_commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "pyFAI": pyFAI.version,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "dioptrin": d,
        "pyFAI (setChiDiscAtPi (-180, 180) vs setChiDiscAtZero (0, 360))": p,
        "overlap of bins above 1e-2, dioptrin pyfai_full vs pyFAI full/histogram": {
            "dioptrin": int(a.sum()),
            "pyFAI": int(b.sum()),
            "both": int((a & b).sum()),
        },
    }
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pyfai-full-seam.json")
