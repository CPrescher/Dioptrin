"""Reproduce the Maxwell client-cold prepared-watcher benchmark on GPFS."""
import argparse
import csv
import ctypes
import hashlib
import json
import mmap
import os
import random
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fabio
import h5py
import hdf5plugin
import numpy as np

import watch_eiger as watch

ROOT = Path(__file__).resolve().parents[3]
PONI = ROOT / "docs/benchmarks/2026-11-05-bslz4/calibration.poni"
MASK = ROOT / "test_data/eiger_0000.mask"
DATASET = "entry_0000/instrument/eiger/data"
SCAN_SHA256 = "2b749693d7a942b35d89806d99b9a1025ee119850be0ade27428f1dc11d0a1f3"
LAYOUTS = [(1, "interleave", 16), (2, "interleave", 16), (2, "local", 16),
           (4, "local", 16), (4, "local", 4), (4, "local", 8), (4, "local", 24)]


def references(scan, poni, mask_path, bins_list):
    import dioptrin

    integrator = dioptrin.Integrator(str(poni), method="pixel_split", unit="2th_deg",
                                     polarization_factor=.965, fast_math=True)
    mask = np.ascontiguousarray(fabio.open(str(mask_path)).data != 0, dtype=np.uint8)
    integrator.set_mask(mask)
    with h5py.File(scan) as source:
        frames = source[DATASET]
        refs = {bins: np.empty((len(frames), 2 * bins)) for bins in bins_list}
        for index, raw in enumerate(frames):
            image = np.ascontiguousarray(raw[::-1], dtype=np.float64)
            for bins, patterns in refs.items():
                result = integrator.integrate1d(image, bins, errors=True)
                patterns[index, :bins] = result.intensity
                patterns[index, bins:] = result.errors
    return refs


def fixtures(root, scan):
    original = root / "forward"
    reverse = root / "reverse"
    template = root / "template"
    shutil.copyfile(scan, original)
    with h5py.File(original) as f:
        source = f[DATASET]
        filter_id, _, options, _ = source.id.get_create_plist().get_filter(0)
        assert source.chunks == (1, *source.shape[1:]) and filter_id == 32008 and options[4] == 2
        for dest, n in [(reverse, len(source)), (template, 1)]:
            with h5py.File(dest, "w") as out:
                parent, name = DATASET.rsplit("/", 1)
                dataset = out.require_group(parent).create_dataset(
                    name, shape=(n, *source.shape[1:]), dtype=source.dtype, chunks=source.chunks,
                    **hdf5plugin.Bitshuffle(nelems=options[3], cname="lz4"))
                assert dataset.id.get_create_plist().get_filter(0)[2][2:] == options[2:]
                for i in range(n):
                    flags, data = source.id.read_direct_chunk((n-1-i, 0, 0))
                    dataset.id.write_direct_chunk((i, 0, 0), data, filter_mask=flags)
    return template, [original, reverse]


def evict_scan(path):
    with path.open("rb", buffering=0) as f:
        fd = f.fileno()
        gpfs = ctypes.CDLL("libgpfs.so", use_errno=True)
        # GPFS v1 fcntl header followed by the per-file CLEAR_FILE_CACHE hint.
        request = (ctypes.c_int * 6)(24, 1, 0, 0, 8, 1004)
        if gpfs.gpfs_fcntl(fd, ctypes.byref(request)) != 0:
            raise OSError(ctypes.get_errno(), "GPFS_CLEAR_FILE_CACHE")
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        size = os.fstat(fd).st_size
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vector = (ctypes.c_ubyte * pages)()
        with mmap.mmap(fd, 0, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
        resident = sum(value & 1 for value in vector)
        if resident:
            raise RuntimeError(f"{resident} input pages still resident after cache eviction")
        return dict(bytes=size, linux_resident_pages=resident, pages=pages, gpfs_clear=True)


def received_bytes(counters):
    # InfiniBand port_rcv_data counts four-byte units.
    return 4 * sum(int(path.read_text()) for path in counters)


def publish(source, dest, delay):
    time.sleep(delay)
    start = time.monotonic()
    source.rename(dest)
    return start


def arrival(service, inbox, results, producer, rng, serial, source, errors, bins, expected, counters):
    state = evict_scan(source)
    before = received_bytes(counters)
    path = inbox / f"eiger-{serial:04d}.h5"
    arrival_time = producer.submit(publish, source, path, rng.uniform(.005, .095))
    deadline = time.monotonic() + 10
    while True:
        found = next(watch.pending(inbox, results), None)
        if found:
            break
        if time.monotonic() > deadline:
            arrival_time.result()
            raise TimeoutError("completed scan was not discovered")
        time.sleep(.05)
    detected = time.monotonic()
    actual, destination = found
    assert actual == path
    start = arrival_time.result()
    ready, saved = service.process(path, destination, errors, "/dev/shm")
    network = received_bytes(counters) - before
    if network < .99 * state["bytes"]:
        raise RuntimeError("InfiniBand traffic did not confirm a cold payload")
    with np.load(destination) as archive:
        intensity = archive["intensity"]
        np.testing.assert_allclose(intensity, expected[:, :bins], rtol=2e-11, atol=1e-10)
        error = float(np.max(np.abs(intensity - expected[:, :bins])))
        axis = archive["radial"]
        assert axis.shape == (bins,) and np.all(np.diff(axis) > 0)
        if errors:
            sigma = archive["sigma"]
            np.testing.assert_allclose(sigma, expected[:, bins:], rtol=2e-11, atol=1e-10)
            error = max(error, float(np.max(np.abs(sigma - expected[:, bins:]))))
        else:
            assert "sigma" not in archive
    assert not list(watch.pending(inbox, results))
    path.rename(source)
    destination.unlink()
    return dict(detect_s=detected-start, ready_s=ready-start, saved_s=saved-start, save_s=saved-ready,
                network_bytes=network, cache=state, max_abs=error)


def summarize(records, destination):
    keys = ("bins", "errors", "processes", "policy", "readers")
    groups = {}
    for row in records:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    assert len(groups) == 28
    summary = []
    for key, trials in sorted(groups.items()):
        assert sorted((r["block"], r["reversed"]) for r in trials) == [
            (b, rev) for b in (1, 2, 3) for rev in (False, True)]
        item = dict(zip(keys, key))
        for metric in ("detect_s", "ready_s", "saved_s", "save_s", "startup_s"):
            values = [r[metric] for r in trials]
            item["median_" + metric] = statistics.median(values)
            if metric in ("ready_s", "saved_s"):
                item["min_" + metric], item["max_" + metric] = min(values), max(values)
        item["max_abs_error"] = max(r["max_abs"] for r in trials)
        item["median_frames_s"] = 625 / item["median_ready_s"]
        summary.append(item)
    with destination.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="Original 625-frame Eiger 9M scan")
    parser.add_argument("--out", type=Path, required=True, help="New run directory on Maxwell GPFS")
    args = parser.parse_args()
    with args.scan.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SCAN_SHA256:
            parser.error("Scan SHA-256 differs from the published input")
    counters = list(Path("/sys/class/infiniband").glob("*/ports/*/counters/port_rcv_data"))
    if not counters or len(os.sched_getaffinity(0)) != 96:
        parser.error("This protocol requires the 96-thread allocation and InfiniBand receive counters")
    args.out.mkdir(parents=True, exist_ok=False)
    refs = references(args.scan, PONI, MASK, (2000, 3000))
    assert not np.allclose(refs[3000][0], refs[3000][-1])
    rng = random.Random(24601)
    records = []
    with tempfile.TemporaryDirectory(dir=args.out) as tmp, ThreadPoolExecutor(1) as producer:
        root = Path(tmp)
        inbox, results = root / "incoming", root / "results"
        inbox.mkdir()
        results.mkdir()
        seed, sources = fixtures(root, args.scan)
        for block in range(1, 4):
            cases = [(layout, bins) for layout in LAYOUTS for bins in refs]
            rng.shuffle(cases)
            for (processes, policy, readers), bins in cases:
                config = dict(poni=str(PONI), mask=str(MASK), dataset=DATASET, template=str(seed),
                              polarization=.965, bins=bins, processes=processes, readers=readers, policy=policy)
                start = time.monotonic()
                with watch.PreparedReaders(config) as service:
                    startup = time.monotonic() - start
                    reverse_first = rng.randrange(2)
                    first_error = bool(rng.randrange(2))
                    for ordinal in range(4):
                        reverse = (ordinal + reverse_first) % 2
                        errors = first_error if ordinal < 2 else not first_error
                        expected = refs[bins][::-1] if reverse else refs[bins]
                        row = arrival(service, inbox, results, producer, rng, len(records)+1,
                                      sources[reverse], errors, bins, expected, counters)
                        row.update(block=block, ordinal=ordinal, processes=processes, policy=policy,
                                   readers=readers, bins=bins, errors=errors, reversed=bool(reverse), startup_s=startup)
                        records.append(row)
                        with (args.out / "trials.jsonl").open("a") as f:
                            print(json.dumps(row), file=f)
    summarize(records, args.out / "summary.csv")


if __name__ == "__main__":
    main()
