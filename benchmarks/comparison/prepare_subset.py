#!/usr/bin/env python3
"""Archivable frame subsets of compressed Eiger HDF5 scans, without recompression.

``subset`` copies N evenly spaced frames of a scan as their original compressed
chunks (``read_direct_chunk``/``write_direct_chunk``) into a small HDF5 file
with the same dataset path, dtype, chunking and filter pipeline. It records the
source SHA-256, the source frame indices and the dataset attributes, decodes
every copied frame again and compares it with the decoded source frame. The
calibration (PONI) and mask are written beside it, so the directory is a
self-contained bundle that compare.py can use (``--subset-dir``).

``tile`` repeats a subset's compressed chunks into a file with a requested
frame count (e.g. 625 or 2500), again by direct chunk writes, for file-reading
pipelines that need a scan-sized input built only from archived frames.

Requires h5py and hdf5plugin (both in the benchmark environment).
"""

import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import shutil

SCHEMA = 1
META = "benchmark_subset"
# Same invalid-pixel threshold as fixtures.py; it includes the uint32 sentinel.
INVALID = 2**31 - 1
TRANSFORMS = {
    "none": lambda a: a,
    "flipud": lambda a: a[::-1],
    "fliplr": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
}


@contextlib.contextmanager
def partial_file(target):
    """Write to a sibling file and publish it only after a complete success."""
    temporary = target.with_name(target.name + ".partial")
    try:
        yield temporary
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha256(array):
    import numpy as np

    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def even_indices(total, count):
    """Evenly spaced, distinct frame indices including the first and last."""
    import numpy as np

    if not 1 <= count <= total:
        raise ValueError(f"Cannot select {count} of {total} frames")
    if count == 1:
        return [0]
    indices = np.linspace(0, total - 1, count).round().astype(int).tolist()
    if len(set(indices)) != count:
        raise ValueError("Frame indices are not distinct")
    return indices


def filters(dataset):
    plist = dataset.id.get_create_plist()
    return [
        dict(
            id=int(fid),
            flags=int(flags),
            values=[int(v) for v in values],
            name=name.decode(errors="replace"),
        )
        for fid, flags, values, name in (
            plist.get_filter(i) for i in range(plist.get_nfilters())
        )
    ]


def per_frame_chunks(dataset):
    if dataset.ndim != 3 or dataset.chunks != (1, *dataset.shape[1:]):
        raise ValueError(
            f"Expected one frame per chunk, got shape {dataset.shape} and chunks {dataset.chunks}"
        )


def register_filters():
    """Use hdf5plugin's Bitshuffle/LZ4 before any file is opened.

    A system HDF5 plugin directory (e.g. /usr/local/hdf5/lib/plugin) can shadow
    it with a filter linked against another HDF5 library, whose set_local
    callback then fails when a dataset is created.
    """
    import hdf5plugin

    hdf5plugin.register(("bshuf", "lz4"), force=True)


def same_pipeline(a, b):
    """Filter ids and user parameters. Bitshuffle's first two values are the
    plugin/library versions its set_local callback fills in; the optional flag
    does not matter for verbatim chunks, whose filter mask is copied."""

    def key(entries):
        return [
            (f["id"], f["values"][2:] if f["id"] == BITSHUFFLE else f["values"])
            for f in entries
        ]

    return key(a) == key(b)


BITSHUFFLE = 32008


def clone_dataset(source, destination, path, frames):
    """Create `path` in `destination` with the source's chunking and filters.

    Following docs/benchmarks/2026-11-21-maxwell-eiger9m/benchmark_watcher.py,
    Bitshuffle/LZ4 is configured through hdf5plugin with the source's block
    size; chunks are later copied verbatim, so nothing is recompressed.
    """
    import hdf5plugin

    pipeline = filters(source)
    if len(pipeline) != 1:
        raise ValueError(f"Expected a single compression filter, got {pipeline}")
    (entry,) = pipeline
    if entry["id"] == BITSHUFFLE:
        values = entry["values"]
        if len(values) < 5 or values[4] != 2:
            raise ValueError(f"Expected Bitshuffle with LZ4 (cd_values {values})")
        compression = dict(**hdf5plugin.Bitshuffle(nelems=values[3], cname="lz4"))
    else:
        compression = dict(
            compression=entry["id"], compression_opts=tuple(entry["values"])
        )
    parent, name = path.strip("/").rsplit("/", 1)
    created = destination.require_group(parent).create_dataset(
        name,
        shape=(frames, *source.shape[1:]),
        dtype=source.dtype,
        chunks=source.chunks,
        fillvalue=source.fillvalue,
        track_times=False,
        **compression,
    )
    for key, value in source.attrs.items():
        created.attrs[key] = value
    if not same_pipeline(filters(created), pipeline) or created.chunks != source.chunks:
        raise RuntimeError(
            f"Destination filters {filters(created)} differ from the source {pipeline}"
        )
    return created


def attributes(dataset):
    def plain(value):
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    return {key: plain(value) for key, value in dataset.attrs.items()}


def write_mask(path, mask):
    import fabio.tifimage
    import numpy as np

    fabio.tifimage.TifImage(data=np.ascontiguousarray(mask, dtype=np.uint8)).write(
        str(path)
    )


def read_mask(path):
    import fabio

    with fabio.open(str(path)) as handle:
        return handle.data != 0


def choose_transform(mask, invalid, requested):
    """Orientation of a mask file relative to the HDF5 frames.

    `auto` accepts the transform under which the mask covers every invalid
    pixel of the subset frames, preferring `none`; it refuses to guess when no
    or several non-identity transforms qualify.
    """
    coverage = {
        name: int((invalid & transform(mask)).sum())
        for name, transform in TRANSFORMS.items()
    }
    total = int(invalid.sum())
    if requested != "auto":
        return requested, coverage, total
    if total == 0:
        raise ValueError(
            "No invalid pixels in the subset frames; pass --mask-transform explicitly"
        )
    # A few pixels can be invalid in only some frames and absent from a static
    # mask; the benchmark loader masks those anyway. Require one orientation to
    # cover nearly all invalid pixels and every other to be clearly worse.
    ranked = sorted(coverage, key=lambda name: (-coverage[name], name != "none"))
    best, runner_up = ranked[0], ranked[1]
    if coverage[best] >= 0.999 * total and (
        coverage[runner_up] < 0.99 * total
        or (best == "none" and coverage[runner_up] == coverage[best])
    ):
        return best, coverage, total
    raise ValueError(
        f"Cannot determine mask orientation (covered invalid pixels {coverage} of {total}); "
        "pass --mask-transform"
    )


def bundle_mask(args, invalid, all_zero):
    """Mask in the HDF5 frame orientation, copied or generated, and its record."""
    if args.mask:
        original = read_mask(args.mask)
        if original.shape != invalid.shape:
            raise ValueError(f"Mask shape {original.shape} differs from frames")
        transform, coverage, total = choose_transform(
            original, invalid, args.mask_transform
        )
        mask = TRANSFORMS[transform](original)
        mask_record = dict(
            origin="copied",
            source_file=args.mask.name,
            source_sha256=sha256(args.mask),
            transform_to_frame_orientation=transform,
            invalid_pixels_in_subset=total,
            invalid_pixels_covered_by_transform=coverage,
        )
    else:
        mask = all_zero | invalid
        mask_record = dict(
            origin="generated",
            rule="pixel is 0 in every subset frame (module gaps, dead pixels) "
            f"or >= {INVALID} in any subset frame (invalid-pixel sentinel)",
            zero_in_every_frame=int(all_zero.sum()),
            invalid_in_any_frame=int(invalid.sum()),
        )
        if args.catalog_detector:
            import pyFAI.detectors

            catalog = pyFAI.detectors.detector_factory(args.catalog_detector).mask
            if catalog is None or catalog.shape != mask.shape:
                raise ValueError(f"{args.catalog_detector} has no matching mask")
            catalog = catalog != 0
            mask_record["catalog_check"] = dict(
                detector=args.catalog_detector,
                catalog_masked=int(catalog.sum()),
                catalog_masked_covered=int((catalog & mask).sum()),
            )
    return mask, mask_record


def make_subset(args):
    import h5py

    register_filters()
    import numpy as np

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}-subset{args.frames}"
    target = out_dir / f"{stem}.h5"
    if target.exists():
        raise FileExistsError(target)
    print(f"Hashing {args.scan.name} ...", flush=True)
    source_digest = sha256(args.scan)
    with h5py.File(args.scan, "r") as src_file:
        source = src_file[args.dataset]
        per_frame_chunks(source)
        indices = even_indices(len(source), args.frames)
        decoded_hashes = []
        invalid = np.zeros(source.shape[1:], dtype=bool)
        all_zero = np.ones(source.shape[1:], dtype=bool)
        with partial_file(target) as temporary, h5py.File(temporary, "w") as out:
            copy = clone_dataset(source, out, args.dataset, len(indices))
            chunk_bytes = 0
            for position, index in enumerate(indices):
                flags, data = source.id.read_direct_chunk((index, 0, 0))
                copy.id.write_direct_chunk((position, 0, 0), data, filter_mask=flags)
                chunk_bytes += len(data)
            meta = out.create_group(META)
            meta.create_dataset("source_frame_indices", data=np.asarray(indices))
            out.flush()
            # Verify against independently decoded source frames.
            for position, index in enumerate(indices):
                original = source[index]
                restored = copy[position]
                if not np.array_equal(original, restored):
                    raise AssertionError(f"Decoded frame {index} differs after copy")
                decoded_hashes.append(array_sha256(restored))
                invalid |= restored >= INVALID
                all_zero &= restored == 0
            meta.create_dataset(
                "decoded_frame_sha256", data=np.asarray(decoded_hashes, dtype="S64")
            )
            record = dict(
                schema=SCHEMA,
                source_file=args.scan.name,
                source_bytes=args.scan.stat().st_size,
                source_sha256=source_digest,
                dataset=args.dataset,
                source_frames=len(source),
                frame_indices=indices,
                shape=list(source.shape[1:]),
                dtype=str(source.dtype),
                chunks=list(source.chunks),
                filters=filters(source),
                subset_filters=filters(copy),
                dataset_attributes=attributes(source),
                compressed_chunk_bytes=chunk_bytes,
                decoded_frame_sha256=decoded_hashes,
                verification="every copied frame decoded equal to the decoded source frame",
            )
            meta.attrs["record"] = json.dumps(record)
            # Decide the mask before the subset is published.
            mask, mask_record = bundle_mask(args, invalid, all_zero)

    # The HDF5 file holds no timestamp, so the same scan and plugin versions
    # give the same bytes; the creation time is kept in the sidecar only.
    bundle = dict(
        subset=target.name,
        subset_sha256=sha256(target),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **record,
    )
    bundle["subset_bytes"] = target.stat().st_size
    poni = out_dir / f"{args.name}.poni"
    shutil.copyfile(args.poni, poni)
    bundle["poni"] = dict(
        file=poni.name, source_file=args.poni.name, sha256=sha256(poni)
    )
    mask_path = out_dir / f"{args.name}.mask.tif"
    write_mask(mask_path, mask)
    if not np.array_equal(read_mask(mask_path), mask):
        raise AssertionError("Mask file does not read back identically")
    mask_record.update(
        file=mask_path.name,
        sha256=sha256(mask_path),
        masked_pixels=int(mask.sum()),
        orientation="same as the HDF5 frames",
    )
    bundle["mask"] = mask_record
    manifest = out_dir / f"{stem}.json"
    manifest.write_text(json.dumps(bundle, indent=2) + "\n")
    print(
        json.dumps({k: bundle[k] for k in ("subset", "subset_bytes", "subset_sha256")})
    )
    return bundle


def read_record(subset):
    import h5py

    with h5py.File(subset, "r") as handle:
        return json.loads(handle[META].attrs["record"])


def tile(args):
    import h5py

    register_filters()
    import numpy as np

    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    record = read_record(args.subset)
    with h5py.File(args.subset, "r") as sub_file:
        source = sub_file[record["dataset"]]
        per_frame_chunks(source)
        count = len(source)
        chunks = [source.id.read_direct_chunk((i, 0, 0)) for i in range(count)]
        with partial_file(target) as temporary, h5py.File(temporary, "w") as out:
            copy = clone_dataset(source, out, record["dataset"], args.frames)
            for index in range(args.frames):
                flags, data = chunks[index % count]
                copy.id.write_direct_chunk((index, 0, 0), data, filter_mask=flags)
            mapping = np.arange(args.frames) % count
            meta = out.create_group(META)
            meta.create_dataset("subset_frame_of_frame", data=mapping)
            meta.create_dataset(
                "source_frame_indices",
                data=np.asarray(record["frame_indices"])[mapping],
            )
            tiled = dict(
                record,
                tiled_from=Path(args.subset).name,
                tiled_from_sha256=sha256(args.subset),
                frames=args.frames,
                rule="frame i is subset frame i mod N, written as its compressed chunk",
            )
            meta.attrs["record"] = json.dumps(tiled)
            out.flush()
            for index in range(args.frames):
                if copy.id.read_direct_chunk((index, 0, 0)) != chunks[index % count]:
                    raise AssertionError(f"Chunk {index} differs after tiling")
            for index in range(min(count, args.frames)):
                if not np.array_equal(copy[index], source[index]):
                    raise AssertionError(f"Decoded frame {index} differs after tiling")
    result = dict(
        output=target.name,
        frames=args.frames,
        bytes=target.stat().st_size,
        sha256=sha256(target),
        tiled_from_sha256=tiled["tiled_from_sha256"],
    )
    print(json.dumps(result))
    return result


def verify(args):
    """Re-check a subset against its manifest record and, optionally, the scan."""
    import h5py

    register_filters()
    import numpy as np

    record = read_record(args.subset)
    with h5py.File(args.subset, "r") as sub_file:
        data = sub_file[record["dataset"]]
        hashes = [array_sha256(data[i]) for i in range(len(data))]
        if hashes != record["decoded_frame_sha256"]:
            raise AssertionError(
                "Decoded subset frames differ from the recorded hashes"
            )
        if args.scan:
            if sha256(args.scan) != record["source_sha256"]:
                raise AssertionError("Scan SHA-256 differs from the recorded source")
            with h5py.File(args.scan, "r") as src_file:
                source = src_file[record["dataset"]]
                for position, index in enumerate(record["frame_indices"]):
                    if not np.array_equal(source[index], data[position]):
                        raise AssertionError(f"Frame {index} differs from the scan")
    print(f"Verified {len(hashes)} frames of {Path(args.subset).name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("subset", help="Copy N evenly spaced compressed frames")
    sub.add_argument("--scan", type=Path, required=True)
    sub.add_argument("--dataset", required=True, help="HDF5 path of the frame stack")
    sub.add_argument("--frames", type=int, default=32)
    sub.add_argument("--name", required=True, help="Bundle name, e.g. eiger9m")
    sub.add_argument("--out-dir", type=Path, required=True)
    sub.add_argument("--poni", type=Path, required=True)
    sub.add_argument(
        "--mask",
        type=Path,
        help="Mask file; without it a mask is generated from the subset frames",
    )
    sub.add_argument(
        "--mask-transform",
        choices=("auto", *TRANSFORMS),
        default="auto",
        help="Orientation of --mask relative to the HDF5 frames (auto: verified "
        "by coverage of invalid pixels)",
    )
    sub.add_argument(
        "--catalog-detector",
        help="pyFAI detector whose gap mask a generated mask is checked against",
    )
    til = commands.add_parser("tile", help="Repeat a subset's compressed chunks")
    til.add_argument("--subset", type=Path, required=True)
    til.add_argument("--frames", type=int, required=True)
    til.add_argument("--output", type=Path, required=True)
    ver = commands.add_parser("verify", help="Re-check a subset file")
    ver.add_argument("--subset", type=Path, required=True)
    ver.add_argument("--scan", type=Path)
    args = parser.parse_args(argv)
    if getattr(args, "frames", 1) < 1:
        parser.error("--frames must be positive")
    try:
        return {"subset": make_subset, "tile": tile, "verify": verify}[args.command](
            args
        )
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
        parser.exit(1, f"{args.command} failed: {error}\n")


if __name__ == "__main__":
    main()
