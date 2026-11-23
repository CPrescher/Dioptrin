"""Versioned benchmark inputs: bundled data and deterministic synthetic series."""

import hashlib
import json
import os
from pathlib import Path
import ssl
import tempfile
import urllib.request


MANIFEST = Path(__file__).with_name("datasets.json")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    return json.loads(MANIFEST.read_text())


def cached_image(spec, data_dir, cache_dir, offline=False):
    """Use a verified Git/LFS fixture, or fetch its pinned public fallback once."""
    bundled = data_dir / spec["image"]
    digest = spec.get("sha256")
    if bundled.is_file() and (digest is None or sha256(bundled) == digest):
        return bundled
    if not spec.get("download"):
        raise FileNotFoundError(f"Missing benchmark input: {bundled}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{digest}-{spec['image']}"
    if cached.is_file() and sha256(cached) == digest:
        return cached
    if offline:
        raise FileNotFoundError(
            f"Verified fixture unavailable offline: {bundled}. Run git lfs pull or fetch it once without --offline."
        )
    import certifi

    context = ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE") or certifi.where()
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(
                spec["download"], context=context, timeout=60
            ) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if sha256(temporary) != digest:
            raise ValueError(
                f"SHA-256 mismatch for {spec['download']}; download rejected"
            )
        os.replace(temporary, cached)
        return cached
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def synthetic(spec, directory):
    import numpy as np
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    shape = tuple(spec["shape"])
    pixel = spec["pixel_size_m"]
    ai = AzimuthalIntegrator(
        dist=spec["distance_m"],
        poni1=shape[0] * pixel * 0.47,
        poni2=shape[1] * pixel * 0.53,
        rot1=0.012,
        rot2=-0.008,
        rot3=0.02,
        wavelength=0.5e-10,
        detector=Detector(pixel1=pixel, pixel2=pixel, max_shape=shape, orientation=3),
    )
    tth = ai.center_array(shape, unit="2th_rad")
    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = shape[0] * 0.47, shape[1] * 0.53
    radius = np.hypot(y - cy, x - cx)
    mask = radius < min(shape) * 0.025
    mask |= (y % 512 < 6) | (x % 512 < 4)
    mask = np.asarray(mask, dtype=np.uint8)
    profile = 15 + 35 * np.exp(-3 * tth)
    # Analytic powder rings plus an amorphous halo. Their width scales with the
    # angular detector pitch rather than collapsing to subpixel peaks at 16M.
    width = max(0.0005, pixel / spec["distance_m"] * 1.4)
    for center, height in (
        (0.045, 350),
        (0.09, 650),
        (0.15, 450),
        (0.22, 800),
        (0.30, 400),
        (0.4, 250),
    ):
        profile += height * np.exp(-0.5 * ((tth - center) / width) ** 2)
    profile += 55 * np.exp(-0.5 * ((tth - 0.18) / 0.035) ** 2)
    profile *= ai.solidAngleArray(shape) * ai.polarization(shape, factor=0.965)
    stack = np.lib.format.open_memmap(
        directory / "frames.npy",
        mode="w+",
        dtype=np.float64,
        shape=(spec["distinct_frames"], *shape),
    )
    for index in range(len(stack)):
        rng = np.random.Generator(np.random.PCG64(spec["seed"] + index))
        stack[index] = rng.poisson(profile * (1 + 0.025 * index))
        stack[index][mask != 0] = 0
    stack.flush()
    return ai, mask, stack, 3, {}, {}


def normalized_geometry(original, shape):
    """Orientation-3 integrator equivalent to `original` on flipped images.

    Returns the integrator, the original orientation and the flip that maps an
    image (or mask) in the calibration's stored orientation onto it.
    """
    import numpy as np
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    orientation = int(original.detector.orientation)
    if orientation not in (0, 1, 2, 3, 4):
        raise ValueError(f"Unsupported orientation: {orientation}")

    def flip(array):
        if orientation in (1, 2):
            array = array[::-1]
        if orientation in (1, 4):
            array = array[:, ::-1]
        return np.ascontiguousarray(array)

    ai = AzimuthalIntegrator(
        dist=original.dist,
        poni1=original.poni1,
        poni2=original.poni2,
        rot1=original.rot1,
        rot2=original.rot2,
        rot3=original.rot3,
        wavelength=original.wavelength,
        detector=Detector(
            pixel1=original.detector.pixel1,
            pixel2=original.detector.pixel2,
            max_shape=shape,
            orientation=3,
        ),
    )
    np.testing.assert_allclose(
        ai.center_array(shape, unit="2th_rad"),
        flip(original.center_array(shape, unit="2th_rad")),
        rtol=2e-6,
        atol=2e-8,
    )
    return ai, orientation, flip


def invalid_pixels(image):
    import numpy as np

    return (image < 0) | ~np.isfinite(image) | (image >= 2**31 - 1)


def real(spec, data_dir, cache_dir, directory, offline):
    import fabio
    import numpy as np
    import pyFAI

    image_path = cached_image(spec, data_dir, cache_dir, offline)
    poni_path = data_dir / spec["poni"]
    sources = {image_path.name: sha256(image_path), poni_path.name: sha256(poni_path)}
    with fabio.open(str(image_path)) as handle:
        image = handle.data.astype(np.float64)
    original = pyFAI.load(str(poni_path))
    original.detector.shape = image.shape
    mask = invalid_pixels(image).astype(np.uint8)
    if spec.get("mask"):
        mask_path = data_dir / spec["mask"]
        with fabio.open(str(mask_path)) as handle:
            mask |= (handle.data != 0).astype(np.uint8)
        sources[mask_path.name] = sha256(mask_path)
    if original.detector.mask is not None:
        mask |= (original.detector.mask != 0).astype(np.uint8)
    ai, orientation, flip = normalized_geometry(original, image.shape)
    image, mask = flip(image), flip(mask)
    image[mask != 0] = 0
    np.save(directory / "frames.npy", image[None, ...])
    return ai, mask, image[None, ...], orientation, sources, {}


def subset(spec, data_dir, directory):
    """Frames of an archivable subset file (prepare_subset.py) with its PONI/mask.

    Every decoded frame is checked against the subset's recorded hashes. The
    mask is the union of the bundle's mask file, the calibration detector's
    mask and pixels that are invalid in any subset frame; masked pixels are
    zeroed in every frame, as for the other datasets.
    """
    import fabio
    import h5py
    import numpy as np
    import pyFAI

    from prepare_subset import array_sha256, read_record, register_filters

    path = data_dir / spec["subset"]
    poni_path = data_dir / spec["poni"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing subset {path}; create it with prepare_subset.py and pass "
            "--subset-dir (or --data-dir) pointing to its directory"
        )
    register_filters()
    record = read_record(path)
    if spec.get("source_sha256") and record["source_sha256"] != spec["source_sha256"]:
        raise ValueError(
            f"{path.name} was not cut from the scan named in datasets.json"
        )
    sources = {path.name: sha256(path), poni_path.name: sha256(poni_path)}
    original = pyFAI.load(str(poni_path))
    with h5py.File(path, "r") as handle:
        data = handle[record["dataset"]]
        count, shape = len(data), tuple(data.shape[1:])
        original.detector.shape = shape
        mask = np.zeros(shape, dtype=np.uint8)
        if spec.get("mask"):
            mask_path = data_dir / spec["mask"]
            with fabio.open(str(mask_path)) as mask_handle:
                mask |= (mask_handle.data != 0).astype(np.uint8)
            sources[mask_path.name] = sha256(mask_path)
        if original.detector.mask is not None:
            mask |= (original.detector.mask != 0).astype(np.uint8)
        ai, orientation, flip = normalized_geometry(original, shape)
        stack = np.lib.format.open_memmap(
            directory / "frames.npy", mode="w+", dtype=np.float64, shape=(count, *shape)
        )
        for index in range(count):
            raw = data[index]
            if array_sha256(raw) != record["decoded_frame_sha256"][index]:
                raise ValueError(
                    f"Decoded frame {index} of {path.name} differs from its record"
                )
            mask |= invalid_pixels(raw).astype(np.uint8)
            stack[index] = flip(raw.astype(np.float64))
    mask = flip(mask)
    for frame in stack:
        frame[mask != 0] = 0
    stack.flush()
    extra = dict(
        subset=dict(
            file=path.name,
            source_file=record["source_file"],
            source_sha256=record["source_sha256"],
            dataset=record["dataset"],
            source_frame_indices=record["frame_indices"],
            decoded_dtype=record["dtype"],
        )
    )
    return ai, mask, stack, orientation, sources, extra


def beam_pixels(ai, shape, mode):
    """Pixels (row, column) whose closed area contains the chosen point.

    `poni`: the point of normal incidence (PONI). `direct`: where the direct
    beam meets the detector (2theta = 0). Coordinates are in pixel units with
    pixel i spanning [i, i + 1), the convention of pyFAI's PONI and Fit2D
    centre in the orientation-3 geometry the benchmark integrates.
    """
    import itertools
    import math

    if mode == "poni":
        point = (ai.poni1 / ai.detector.pixel1, ai.poni2 / ai.detector.pixel2)
    elif mode == "direct":
        fit2d = ai.getFit2D()
        point = (fit2d["centerY"], fit2d["centerX"])
    else:
        raise ValueError(f"Unknown beam-pixel mode: {mode}")
    candidates = []
    for value in point:
        base = math.floor(value)
        near = [base]
        if value - base < 1e-6:
            near.append(base - 1)  # on the edge shared with the previous pixel
        if base + 1 - value < 1e-6:
            near.append(base + 1)  # on the edge shared with the next pixel
        candidates.append(near)
    pixels = sorted(
        (int(r), int(c))
        for r, c in itertools.product(*candidates)
        if 0 <= r < shape[0] and 0 <= c < shape[1]
    )
    return [float(v) for v in point], pixels


def mask_beam_pixels(directory, ai, mask, mode):
    """Add the beam pixels to the mask, zero them in frames.npy, record them."""
    import numpy as np

    shape = mask.shape
    point, pixels = beam_pixels(ai, shape, mode)
    tth = ai.center_array(shape, unit="2th_rad")
    minimum = np.unravel_index(int(np.argmin(tth)), shape)
    record = dict(
        mode=mode,
        point_pixels=point,
        pixels=[list(p) for p in pixels],
        already_masked=[list(p) for p in pixels if mask[p]],
        min_2theta_pixel=[int(v) for v in minimum],
        min_2theta_pixel_masked_before=bool(mask[minimum]),
    )
    if pixels:
        rows, cols = zip(*pixels)
        mask[rows, cols] = 1
        frames = np.load(directory / "frames.npy", mmap_mode="r+")
        frames[:, rows, cols] = 0
        frames.flush()
        del frames
    return record


def prepare(
    directory,
    name,
    data_dir,
    cache_dir,
    offline=False,
    beam_pixel=None,
    subset_dir=None,
):
    import numpy as np

    definition = manifest()
    spec = definition["datasets"][name]
    directory.mkdir(parents=True)
    if spec["kind"] == "synthetic":
        ai, mask, frames, orientation, sources, extra = synthetic(spec, directory)
    elif spec["kind"] == "subset":
        ai, mask, frames, orientation, sources, extra = subset(
            spec, subset_dir or data_dir, directory
        )
    else:
        ai, mask, frames, orientation, sources, extra = real(
            spec, data_dir, cache_dir, directory, offline
        )
    shape = frames.shape[1:]
    if beam_pixel:
        # Same pixels masked (and zeroed) for every library in this run.
        extra["beam_pixel_mask"] = mask_beam_pixels(directory, ai, mask, beam_pixel)
        del frames
        frames = np.load(directory / "frames.npy", mmap_mode="r")
    ai.save(str(directory / "geometry.poni"))
    np.save(directory / "mask.npy", mask)
    corners = ai.array_from_unit(shape, typ="corner", unit="2th_deg")
    metadata = dict(
        dataset=name,
        specification=spec,
        generator_version=definition["generator_version"],
        manifest_sha256=sha256(MANIFEST),
        shape=list(shape),
        dtype="float64",
        distinct_frames=len(frames),
        frame_sha256=[hashlib.sha256(memoryview(f)).hexdigest() for f in frames],
        n_splitting=spec["n_splitting"],
        radial_range=[max(0.0, float(corners.min())), float(corners.max())],
        original_orientation=orientation,
        normalized_orientation=3,
        masked_pixels=int(mask.sum()),
        sources=sources,
        geometry=dict(
            pixel1=ai.detector.pixel1,
            pixel2=ai.detector.pixel2,
            dist=ai.dist,
            poni1=ai.poni1,
            poni2=ai.poni2,
            rot1=ai.rot1,
            rot2=ai.rot2,
            rot3=ai.rot3,
            wavelength=ai.wavelength,
        ),
        beam_pixel_mask=extra.pop("beam_pixel_mask", None),
        **extra,
    )
    (directory / "fixture.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata
