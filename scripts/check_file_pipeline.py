"""TIFF/Python regression checks. Requires numpy and tifffile; imagecodecs adds LZW.

Run against the installed candidate: python scripts/check_file_pipeline.py.
No input images or output fixtures are retained unless --record is supplied.
"""
import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile
import dioptrin


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), "integration result changed"


def compare(actual, reference, errors):
    same(actual.radial, reference.radial)
    same(actual.intensity, reference.intensity)
    if errors:
        same(actual.errors, reference.errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    kwargs = dict(pixel1=0.0001, pixel2=0.00012, distance=0.2,
                  poni1=0.0013, poni2=0.0017, rot1=0.04, rot2=-0.03,
                  rot3=0.08, wavelength=1e-10)
    checked = 0
    compression = [None, "deflate"]
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        pass
    else:
        compression.append("lzw")
    with tempfile.TemporaryDirectory(prefix="dioptrin-tiff-check-") as folder:
        root = Path(folder)
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64,
                      np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
            pixels = (np.arange(33 * 35).reshape(33, 35) % 121).astype(dtype)
            if np.issubdtype(dtype, np.integer):
                limits = np.iinfo(dtype)
                pixels[0, :3] = [limits.min, limits.max, 0]
                if pixels.dtype.itemsize == 8:
                    pixels[1, 0] = 9007199254740993
            else:
                pixels[0, :5] = [np.nan, np.inf, -np.inf, -0., -5.]
            for byteorder in ["<", ">"]:
                for codec in compression:
                    for tiled in [False, True]:
                        path = root / "changing.tif"
                        options = {"tile": (16, 16)} if tiled else {"rowsperstrip": 7}
                        tifffile.imwrite(path, pixels, compression=codec, byteorder=byteorder,
                                         photometric="minisblack", **options)
                        data = np.ascontiguousarray(pixels[::-1], dtype=np.float64)
                        for method in ["counting", "histogram", "pixel_split", "supersampled"]:
                            for fast in [False, True]:
                                ai = dioptrin.Integrator.from_poni_dict(
                                    kwargs, method=method, n_ss=3, fast_math=fast,
                                    polarization_factor=.965)
                                mask = np.zeros(data.shape, dtype=np.uint8)
                                mask[::5, ::3] = 1
                                ai.set_mask(mask)
                                for errors in [False, True]:
                                    ref = ai.integrate1d(data, 19, errors=errors)
                                    compare(ai.integrate1d(str(path), 19, errors=errors), ref, errors)
                                    batch = ai.batch1d([str(path)] * 3, 19, errors=errors,
                                                       num_workers=2, num_threads=1)
                                    for result in batch:
                                        assert result.is_ok(), result.error
                                        compare(result.result, ref, errors)
                                    checked += 4
                                if hasattr(ai, "clear_file_workers"):
                                    ai.clear_file_workers()
        # Distinct images repeatedly enter the same persistent worker buffers.
        paths = []
        arrays = []
        for i in range(2):
            image = (np.arange(33 * 35).reshape(33, 35) * (i + 3) - i * 9).astype(np.int32)
            path = root / f"alternating-{i}.tif"
            tifffile.imwrite(path, image)
            paths.append(str(path)); arrays.append(np.ascontiguousarray(image[::-1], dtype=np.float64))
        for method in ["pixel_split", "supersampled"]:
            for fast in [False, True]:
                ai = dioptrin.Integrator.from_poni_dict(kwargs, method=method, n_ss=3, fast_math=fast)
                refs = [ai.integrate1d(x, 23, errors=True) for x in arrays]
                for workers in [1, 2, 4, 8, 2]:
                    for round_ in range(3):
                        indices = [(i + round_) % 2 for i in range(32)]
                        names = [paths[i] for i in indices]
                        for mode in ["batch", "list-stream", "generator"]:
                            if mode == "batch":
                                rr = ai.batch1d(names, 23, errors=True, num_workers=workers, num_threads=1)
                            else:
                                source = names if mode == "list-stream" else (p for p in names)
                                rr = list(ai.batch1d_iter(source, 23, errors=True,
                                                        num_workers=workers, num_threads=1))
                            assert len(rr) == 32
                            for r, i in zip(rr, indices):
                                assert r.is_ok(), r.error
                                compare(r.result, refs[i], True)
                                checked += 1
                    # Sync returns must release worker-owned calibration refs.
                    ai.set_fast_math(fast)
                # A failed read must never reuse the previous frame's pixels.
                rr = ai.batch1d([paths[0], str(root / "missing.tif"), paths[1]], 23,
                                errors=True, num_workers=1, num_threads=1)
                assert rr[0].is_ok() and not rr[1].is_ok() and rr[2].is_ok()
                compare(rr[0].result, refs[0], True); compare(rr[2].result, refs[1], True)
                assert ai.batch1d([], 23, num_workers=8, num_threads=1) == []
                # Reopen a path whose contents change between admitted batches.
                changing = root / "overwrite.tif"
                for i in [0, 1, 0, 1]:
                    changing.write_bytes(Path(paths[i]).read_bytes())
                    r = ai.batch1d([str(changing)], 23, errors=True, num_workers=1, num_threads=1)
                    compare(r[0].result, refs[i], True)
                # 2D file batches, explicit azimuthal grid and errors share the typed path.
                for errors in [False, True]:
                    for azimuthal in [None, (-.3, .7)]:
                        refs2 = [ai.integrate2d(a, 17, 8, errors=errors,
                                               azimuthal_range=azimuthal) for a in arrays]
                        for mode in ["batch", "generator", "array-stream"]:
                            if mode == "batch":
                                rr = ai.batch2d(paths * 3, 17, 8, errors=errors,
                                                azimuthal_range=azimuthal, num_workers=2, num_threads=1)
                            else:
                                src = (p for p in paths * 3) if mode == "generator" else arrays * 3
                                rr = list(ai.batch2d_iter(src, 17, 8, errors=errors,
                                                          azimuthal_range=azimuthal, num_workers=2, num_threads=1))
                            assert len(rr) == 6
                            for j, r in enumerate(rr):
                                assert r.is_ok(), r.error
                                same(r.result.intensity, refs2[j % 2].intensity)
                                if errors: same(r.result.errors, refs2[j % 2].errors)
                                checked += 1
                if hasattr(ai, "clear_file_workers"):
                    ai.clear_file_workers()
        # A streaming iterator owns its admitted operation even if its parent dies.
        ai = dioptrin.Integrator.from_poni_dict(kwargs, method="counting")
        ref = ai.integrate1d(arrays[0], 17, errors=True)
        iterator = ai.batch1d_iter((paths[0] for _ in range(16)), 17, errors=True,
                                  num_workers=2, num_threads=1)
        del ai; gc.collect()
        rr = list(iterator); assert len(rr) == 16
        for r in rr: compare(r.result, ref, True)
    record = dict(passed=True, checked_results=checked + 16,
                  codecs=compression, types=10, byteorders=2,
                  layouts=["strips", "padded tiles"], workers=[1, 2, 4, 8],
                  persistent_api=hasattr(dioptrin.Integrator, "clear_file_workers"))
    if args.record: args.record.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
