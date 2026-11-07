"""Refresh the native catalogue from pyFAI; development/reference tool only.

No Python is needed to compile or use the resulting catalogue. Physical model
parameters and masks are data, not imported pyFAI implementation code. See
docs/detectors.md for provenance, precision and unsupported-model boundaries.
"""
import gc
import gzip
import json
from pathlib import Path
import numpy as np
import pyFAI
from pyFAI.detectors import ALL_DETECTORS

output = []
for cls in sorted(set(ALL_DETECTORS.values()), key=lambda c: c.__name__):
    if cls.__name__ == "Detector":
        continue
    d = cls()
    d._orientation = 3
    d.binning = (1,1)
    name = (d.aliases or [cls.__name__])[0]
    record = dict(name=name, aliases=sorted(set([cls.__name__, *d.aliases])),
                  pixel1=d.pixel1, pixel2=d.pixel2, binning_pitches=list(getattr(d,"BINNED_PIXEL_SIZE",{}).items()), max_shape=list(d.max_shape),
                  orientation=3, map="Uniform", mask=[], circular_mask=cls.__name__ in ("Mar345","Rayonix133","RayonixSx165"), spline=None, unsupported=None)
    if d.CORNERS != 4:
        record["unsupported"] = f"{name}: hexagonal pixel corners are not implemented in the pyFAI detector reference; supply a qualified quadrilateral NeXus definition."
    elif hasattr(d, "_get_compact_pixel_corners"):
        y, x, z = d._get_compact_pixel_corners()
        y, x, z = [np.asarray(a, dtype=np.float32).astype(float) for a in (y, x, z)]
        slow = np.zeros((len(y)-1, 2, 3))
        fast = np.zeros((len(x)-1, 2, 3))
        slow[:, 0, 1], slow[:, 1, 1] = y[:-1], y[1:]
        fast[:, 0, 2], fast[:, 1, 2] = x[:-1], x[1:]
        fast[:, 0, 0], fast[:, 1, 0] = z[:-1], z[1:]
        record["map"] = {"Separable": dict(slow=slow.tolist(), fast=fast.tolist())}
    elif cls.__name__ in ("Xpad_flat", "Cirpad"):
        corners = d.get_pixel_corners().astype(float)
        origin = corners[0, 0, 0]
        slow = corners[:, 0, :2].copy()
        fast = corners[0, :, [0, 3]].transpose(1, 0, 2).copy() - origin
        rng = np.random.default_rng(1947)
        for _ in range(500):
            r, c = [rng.integers(n) for n in d.shape]
            predicted = np.array([slow[r,0]+fast[c,0], slow[r,1]+fast[c,0], slow[r,1]+fast[c,1], slow[r,0]+fast[c,1]])
            if not np.allclose(predicted, corners[r,c], rtol=0, atol=1e-7):
                raise ValueError(f"Nonseparable corners: {name}")
        record["map"] = {"Separable": dict(slow=slow.tolist(), fast=fast.tolist())}
    elif hasattr(d, "calc_pixels_edges"):
        y, x = d.calc_pixels_edges()
        slow = np.zeros((len(y)-1, 2, 3)); fast = np.zeros((len(x)-1, 2, 3))
        slow[:, 0, 1], slow[:, 1, 1] = y[:-1], y[1:]
        fast[:, 0, 2], fast[:, 1, 2] = x[:-1], x[1:]
        record["map"] = {"Separable": dict(slow=slow.tolist(), fast=fast.tolist())}
    m = d.calc_mask()
    if m is not None and not record["circular_mask"]:
        flat = np.asarray(m, dtype=bool).ravel()
        edges = np.flatnonzero(np.diff(np.r_[False, flat, False]))
        record["mask"] = edges.reshape(-1, 2).tolist()
    output.append(record)
    print(name, "intervals", len(record["mask"]), flush=True)
    del d, m
    gc.collect()
root = Path(__file__).resolve().parents[1]
(root / "src/detector/catalog.json.gz").write_bytes(gzip.compress((json.dumps(output, separators=(",", ":")) + "\n").encode(),mtime=0))
print("pyFAI", pyFAI.version, "models", len(output))
