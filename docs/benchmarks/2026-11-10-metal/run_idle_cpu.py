#!/usr/bin/env python3
"""Compare a retained CPU plan after 30-second pauses with the GPU idle test."""
import hashlib
import json
from pathlib import Path
import tempfile

from run_setup import HERE, REPO, run, trial


def main():
    data = json.loads((HERE / "setup-results.json").read_text())
    case = next(c for c in data["cases"] if c["name"] == data["idle_test"]["case"])
    run(["cargo", "build", "--release", "--manifest-path", HERE / "setup/Cargo.toml",
         "--target-dir", REPO / "target/metal-setup"])
    with tempfile.TemporaryDirectory(prefix="dioptrin-cpu-idle-") as directory:
        executable = Path(directory) / "SetupBench"
        run(["swiftc", "-O", HERE / "setup/SetupBench.swift",
             REPO / "target/metal-setup/release/libdioptrin_metal_setup.a", "-o", executable])
        result = {"case":case["name"],"source_sha256":{
            str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),HERE / "run_setup.py",HERE / "setup/SetupBench.swift",
                      HERE / "setup/src/lib.rs"]},
            "trial":trial(executable,case,"best_cpu",51,idle=30)}
    path = HERE / "idle-cpu-results.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
