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
    return ai, mask, stack, 3, {}


def real(spec, data_dir, cache_dir, directory, offline):
    import fabio
    import numpy as np
    import pyFAI
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    image_path = cached_image(spec, data_dir, cache_dir, offline)
    poni_path = data_dir / spec["poni"]
    sources = {image_path.name: sha256(image_path), poni_path.name: sha256(poni_path)}
    with fabio.open(str(image_path)) as handle:
        image = handle.data.astype(np.float64)
    original = pyFAI.load(str(poni_path))
    original.detector.shape = image.shape
    mask = ((image < 0) | ~np.isfinite(image) | (image >= 2**31 - 1)).astype(np.uint8)
    if spec.get("mask"):
        mask_path = data_dir / spec["mask"]
        with fabio.open(str(mask_path)) as handle:
            mask |= (handle.data != 0).astype(np.uint8)
        sources[mask_path.name] = sha256(mask_path)
    if original.detector.mask is not None:
        mask |= (original.detector.mask != 0).astype(np.uint8)
    orientation = int(original.detector.orientation)
    if orientation not in (0, 1, 2, 3, 4):
        raise ValueError(f"Unsupported orientation: {orientation}")
    if orientation in (1, 2):
        image, mask = image[::-1], mask[::-1]
    if orientation in (1, 4):
        image, mask = image[:, ::-1], mask[:, ::-1]
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    image[mask != 0] = 0
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
            max_shape=image.shape,
            orientation=3,
        ),
    )
    angles = original.center_array(image.shape, unit="2th_rad")
    if orientation in (1, 2):
        angles = angles[::-1]
    if orientation in (1, 4):
        angles = angles[:, ::-1]
    np.testing.assert_allclose(
        ai.center_array(image.shape, unit="2th_rad"), angles, rtol=2e-6, atol=2e-8
    )
    np.save(directory / "frames.npy", image[None, ...])
    return ai, mask, image[None, ...], orientation, sources


def prepare(directory, name, data_dir, cache_dir, offline=False):
    import numpy as np

    definition = manifest()
    spec = definition["datasets"][name]
    directory.mkdir(parents=True)
    if spec["kind"] == "synthetic":
        ai, mask, frames, orientation, sources = synthetic(spec, directory)
    else:
        ai, mask, frames, orientation, sources = real(
            spec, data_dir, cache_dir, directory, offline
        )
    shape = frames.shape[1:]
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
    )
    (directory / "fixture.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata
