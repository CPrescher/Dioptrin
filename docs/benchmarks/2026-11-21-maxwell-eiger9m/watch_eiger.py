"""Integrate completed Eiger HDF5 files on a Linux NUMA node using persistent LUTs.

Publish each closed scan by renaming it to a unique .h5 filename in the inbox.
Use a separate output directory for each calibration, mask, bin and error setting.
The template is a completed file with matching detector geometry. The mask is
in calibrated bottom-up coordinates, as required by dioptrin_io_fused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def cpu_groups(count):
    """Divide physical cores evenly, keeping sibling threads in the same group."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(int((topology / name).read_text()) for name in ("physical_package_id", "core_id"))
        cores.setdefault(key, []).append(cpu)
    if count < 1 or len(cores) % count:
        raise ValueError("Process/reader count must evenly divide the available physical cores")
    ordered = sorted(cores)
    groups = []
    for i in range(count):
        keys = ordered[i*len(cores)//count:(i+1)*len(cores)//count]
        cpus = [cpu for key in keys for cpu in cores[key]]
        nodes = {int(entry.name[4:]) for cpu in cpus for entry in Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*")}
        if not nodes:
            raise ValueError(f"Cannot determine the NUMA nodes for CPUs {cpus}")
        groups.append((cpus, nodes))
    return groups


def read_geometry(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    geometry = {k: float(fields[k]) for k in ("distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")}
    detector = json.loads(fields.get("detector_config", "{}"))
    if detector.get("orientation") != 2:
        raise ValueError("The fused Eiger watcher requires PONI Detector_config.orientation = 2 "
                         "(vertical row flip).")
    pixels = [fields.get(f"pixelsize{i}", detector.get(f"pixel{i}")) for i in (1, 2)]
    if None in pixels:
        import pyFAI
        model = pyFAI.load(str(path)).detector
        pixels = [model.pixel1, model.pixel2]
    geometry.update(pixel1=float(pixels[0]), pixel2=float(pixels[1]))
    return geometry


def worker():
    import fabio
    import numpy as np
    from dioptrin_io_fused import FusedHdf5Integrator
    config = json.loads(sys.stdin.readline())
    geometry = read_geometry(config["poni"])
    mask = np.ascontiguousarray(fabio.open(config["mask"]).data != 0, dtype=np.uint8)
    threads = config["readers"] // config["processes"]
    groups = cpu_groups(threads)
    plans = [None] * threads
    bins = config["bins"]
    dataset = config["dataset"]

    def build(j):
        plans[j] = FusedHdf5Integrator(config["template"], dataset, geometry, bins, mask=mask,
                    polarization_factor=config["polarization"], unit="2th_deg", num_workers=len(groups[j][0]))
        os.sched_setaffinity(0, groups[j][0])

    def integrate(j, command, result):
        plans[j].set_source(command["path"], dataset)
        reader = config["index"] * threads + j
        frames = command["frames"]
        for i in range(reader*frames//config["readers"], (reader+1)*frames//config["readers"]):
            obj = plans[j].integrate1d(i, errors=command["errors"])
            if i == 0:
                result[0] = obj.radial
            result[1+i] = obj.intensity
            if command["errors"]:
                result[1+frames+i] = obj.errors

    with ExitStack() as stack:
        pools = [stack.enter_context(ThreadPoolExecutor(1)) for _ in plans]
        def parallel(fn, *args):
            futures = [p.submit(fn, j, *args) for j, p in enumerate(pools)]
            # Drain every worker before reusing its plan or output on the next file.
            failures = [f.exception() for f in futures]
            for failure in failures:
                if failure is not None:
                    raise failure
        try:
            parallel(build)
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
                try:
                    command = json.loads(line)
                    result = np.memmap(command["buffer"], mode="r+", dtype="f8",
                               shape=(1+command["frames"]*(1+command["errors"]), bins))
                    parallel(integrate, command, result)
                    del result
                    reply = {"ok": True}
                except Exception as exc:
                    reply = {"error": str(exc)}
                print(json.dumps(reply), flush=True)
        finally:
            for j, pool in enumerate(pools):
                pool.submit(plans.__setitem__, j, None).result()


class PreparedReaders:
    def __init__(self, config):
        self.config = config
        self.processes = []

    def __enter__(self):
        config = self.config
        if config["readers"] < 1 or config["readers"] % config["processes"]:
            raise ValueError("Readers must be a positive multiple of processes")
        groups = cpu_groups(config["processes"])
        all_nodes = set.union(*(nodes for _, nodes in groups))
        try:
            for i, (cpus, nodes) in enumerate(groups):
                if config["policy"] == "local" and len(nodes) != 1:
                    raise ValueError("Each local process must fit on one NUMA node")
                local = config["policy"] == "local"
                policy = ("--membind=" if local else "--interleave=") + ",".join(map(str, sorted(nodes if local else all_nodes)))
                cmd = ["numactl", "--physcpubind="+",".join(map(str, cpus)), policy,
                       sys.executable, "-u", __file__, "--worker"]
                env = dict(os.environ, RAYON_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
                self.processes.append(p)
                self.send(p, dict(config, index=i))
            self.receive("ready")
        except BaseException:
            self.close()
            raise
        return self

    @staticmethod
    def send(p, command):
        print(json.dumps(command), file=p.stdin, flush=True)

    def receive(self, key):
        replies = [p.stdout.readline() for p in self.processes]
        for reply in replies:
            if not reply:
                raise ChildProcessError("Integration worker exited unexpectedly")
            message = json.loads(reply)
            if not message.get(key):
                raise RuntimeError(message.get("error", "Unexpected worker response"))

    def close(self):
        for p in self.processes:
            p.stdin.close()
        for p in self.processes:
            p.wait()
            p.stdout.close()

    def __exit__(self, *_):
        self.close()

    def process(self, path, dest, errors, scratch):
        import h5py
        import numpy as np
        with h5py.File(path, "r") as src:
            frames, _, _ = src[self.config["dataset"]].shape
        if frames < 1:
            raise ValueError("Empty scan")
        bins = self.config["bins"]
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            buffer = Path(temp)/"patterns"
            shape = (1+frames*(1+errors), bins)
            with buffer.open("wb") as file:
                # Reserve backing space before workers start writing through their mmap.
                os.posix_fallocate(file.fileno(), 0, 8 * shape[0] * shape[1])
            for p in self.processes:
                self.send(p, dict(path=str(path), frames=frames, errors=errors, buffer=str(buffer)))
            self.receive("ok")
            ready = time.monotonic()
            values = np.memmap(buffer, mode="r", dtype="f8", shape=shape)
            arrays = dict(radial=values[0], intensity=values[1:1+frames])
            if errors:
                arrays["sigma"] = values[1+frames:]
            # Only a fully written result gets its final name on the output filesystem.
            with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".partial", delete=False) as f:
                partial = Path(f.name)
                try:
                    np.savez(f, **arrays)
                    f.close()
                    os.replace(partial, dest)
                finally:
                    partial.unlink(missing_ok=True)
            published = time.monotonic()
            del arrays, values
        return ready, published


def pending(inbox, output):
    for path in sorted(inbox.glob("*.h5")):
        dest = output/(path.name+".npz")
        if not dest.exists():
            yield path, dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("inbox", "output", "template", "poni", "mask"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--dataset", default="entry_0000/instrument/eiger/data")
    ap.add_argument("--bins", type=int, default=3000)
    ap.add_argument("--poisson", action="store_true")
    ap.add_argument("--polarization", type=float, default=None)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--readers", type=int, default=16)
    ap.add_argument("--policy", choices=("local", "interleave"), default="local")
    ap.add_argument("--poll-ms", type=float, default=50)
    ap.add_argument("--scratch", type=Path, default=Path("/dev/shm"))
    args = ap.parse_args()
    if min(args.processes, args.readers, args.bins, args.poll_ms) <= 0:
        ap.error("Counts and polling interval must be positive")
    if not args.inbox.is_dir():
        ap.error("Inbox directory does not exist")
    args.output.mkdir(exist_ok=True, parents=True)
    config = {k: (str(v.resolve()) if isinstance(v, Path) else v) for k,v in vars(args).items()}
    with PreparedReaders(config) as service:
        print("Ready. Watching", args.inbox, flush=True)
        retry = {}
        while True:
            for path, dest in pending(args.inbox, args.output):
                if time.monotonic() < retry.get(path, 0):
                    continue
                try:
                    service.process(path, dest, args.poisson, args.scratch)
                except (OSError, RuntimeError, KeyError, ValueError) as exc:
                    if isinstance(exc, (ChildProcessError, BrokenPipeError)):
                        raise
                    print(f"{path}: {exc}", file=sys.stderr, flush=True)
                    retry[path] = time.monotonic()+5
                else:
                    retry.pop(path, None)
                    print(dest, flush=True)
            time.sleep(args.poll_ms/1000)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
