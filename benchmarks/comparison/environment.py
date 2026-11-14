#!/usr/bin/env python3
"""Persist native benchmark environments without reusing incompatible binaries."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import uuid

from provenance import source_hashes
from setup import compiler_environment


ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_VERSION = 1


def cpu_identity():
    """Exclude node names/core counts, but distinguish native instruction sets."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        allowed = os.sched_getaffinity(0)
        models, features = [], []
        for block in cpuinfo.read_text().split("\n\n"):
            fields = dict(
                line.split(":", 1) for line in block.splitlines() if ":" in line
            )
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            if int(fields.get("processor", -1)) not in allowed:
                continue
            models.append(
                {
                    key: fields.get(key, "")
                    for key in (
                        "vendor_id",
                        "cpu family",
                        "model",
                        "model name",
                        "stepping",
                        "CPU implementer",
                        "CPU part",
                    )
                }
            )
            features.append(
                set((fields.get("flags") or fields.get("Features", "")).split())
            )
        if not models or not all(features):
            raise RuntimeError("Cannot identify the allocated CPU instruction set")
        native_models = {
            json.dumps(
                {key: value for key, value in model.items() if key != "stepping"},
                sort_keys=True,
            )
            for model in models
        }
        if len(native_models) != 1 or any(flags != features[0] for flags in features):
            raise RuntimeError(
                "Native builds require a homogeneous CPU model/instruction set within the allocation"
            )
        return dict(
            models=sorted({json.dumps(model, sort_keys=True) for model in models}),
            features=sorted(set.intersection(*features)),
        )
    if sys.platform == "darwin":
        values = subprocess.check_output(
            ["sysctl", "-a"], text=True, stderr=subprocess.DEVNULL
        )
        return {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in values.splitlines()
            if line.startswith(
                (
                    "machdep.cpu.brand_string:",
                    "machdep.cpu.features:",
                    "machdep.cpu.leaf7_features:",
                    "hw.optional.",
                )
            )
        }
    raise RuntimeError(
        "Native environment caching requires identifiable Linux/macOS CPU features"
    )


def fingerprint(root, rust_toolchain):
    env = compiler_environment()
    compilers = {}
    for name in ("CC", "CXX"):
        command = shlex.split(env[name])
        compilers[name] = dict(
            command=command,
            executable=str(Path(shutil.which(command[0]) or command[0]).resolve()),
            version=subprocess.check_output([*command, "--version"], text=True),
        )
    recipes = ("environment.py", "setup.py", "requirements.txt", "provenance.py")
    return dict(
        schema_version=SCHEMA_VERSION,
        cpu=cpu_identity(),
        system=platform.system(),
        machine=platform.machine(),
        libc=platform.libc_ver(),
        os_release=Path("/etc/os-release").read_text()
        if Path("/etc/os-release").is_file()
        else platform.mac_ver()[0],
        python=dict(
            executable=str(Path(sys.executable).resolve()),
            base_prefix=sys.base_prefix,
            version=sys.version,
            soabi=sysconfig.get_config_var("SOABI"),
        ),
        compilers=compilers,
        rust_toolchain=rust_toolchain,
        flags={
            name: env.get(name, "")
            for name in (
                "CFLAGS",
                "CXXFLAGS",
                "RUSTFLAGS",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
            )
        },
        source=source_hashes(root),
        extra_build_inputs={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "build.rs",
                "rust-toolchain",
                "rust-toolchain.toml",
                ".cargo/config",
                ".cargo/config.toml",
            )
            if (root / name).is_file()
        },
        recipes={
            name: hashlib.sha256(
                (root / "benchmarks/comparison" / name).read_bytes()
            ).hexdigest()
            for name in recipes
        },
    )


def cache_key(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_cached(cache_root, identity, build, validate, rebuild=False):
    """Publish complete immutable generations; lock construction, not benchmark use."""
    key = cache_key(identity)
    entry = cache_root.resolve() / key
    entry.mkdir(parents=True, exist_ok=True)
    with (entry / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = entry / "ready.json"
        try:
            index = json.loads(index_path.read_text())
        except (OSError, ValueError):
            index = {}
        if not isinstance(index, dict):
            index = {}
        name = index.get("generation", "")
        if not isinstance(name, str):
            name = ""
        generation = entry / name
        if (
            not rebuild
            and index.get("key") == key
            and isinstance(name, str)
            and name.startswith("env-")
            and Path(name).name == name
            and generation.resolve().parent == entry
            and validate(generation)
        ):
            reused = True
        else:
            # Never relocate a venv, and never replace a generation another job
            # might still be using. A failed/forced rebuild leaves old generations.
            generation = entry / ("env-" + uuid.uuid4().hex)
            generation.mkdir()
            build(generation)
            if not validate(generation):
                raise RuntimeError(
                    "New environment failed validation; cache was not published"
                )
            atomic_json(
                index_path,
                dict(key=key, generation=generation.name, fingerprint=identity),
            )
            reused = False
    return dict(
        key=key,
        generation=str(generation),
        python=str(generation / "venv/bin/python"),
        reused=reused,
        fingerprint=identity,
    )


def probe_environment(generation, root):
    python = generation / "venv/bin/python"
    record = generation / "venv/benchmark-build.json"
    frozen = generation / "build/python-packages.txt"
    if not python.is_file() or not record.is_file() or not frozen.is_file():
        return False
    check = (
        "import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from provenance import verify_build;import numpy,pyFAI,azint;"
        "r=json.loads(Path(sys.argv[2]).read_text());"
        "problems=verify_build(Path(sys.argv[3]),r);"
        "print('; '.join(problems));sys.exit(bool(problems))"
    )
    try:
        result = subprocess.run(
            [
                str(python),
                "-E",
                "-s",
                "-c",
                check,
                str(root / "benchmarks/comparison"),
                str(record),
                str(root),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            return False
        actual = subprocess.check_output(
            [str(python), "-E", "-s", "-m", "pip", "freeze", "--all"], text=True
        )
        if sorted(actual.splitlines()) != sorted(frozen.read_text().splitlines()):
            return False
        return (
            subprocess.run(
                [str(python), "-E", "-s", "-m", "pip", "check"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def build_environment(generation, build_root, rust_toolchain):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("RUSTUP_TOOLCHAIN", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["CARGO_HOME"] = str(build_root / "cargo-home")
    env["RUSTUP_HOME"] = str(build_root / "rustup-home")
    env["CARGO_TARGET_DIR"] = str(build_root / "target")
    env["CARGO_BUILD_JOBS"] = env.get("BUILD_JOBS", "8")
    script = build_root / "rustup-init.sh"
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "https://sh.rustup.rs",
            "-o",
            str(script),
        ],
        check=True,
    )
    subprocess.run(
        [
            "sh",
            str(script),
            "-y",
            "--profile",
            "minimal",
            "--default-toolchain",
            rust_toolchain,
            "--no-modify-path",
        ],
        env=env,
        check=True,
    )
    env["PATH"] = env["CARGO_HOME"] + "/bin" + os.pathsep + env["PATH"]
    subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            str(ROOT / "benchmarks/comparison/setup.py"),
            "--venv",
            str(generation / "venv"),
            "--artifacts",
            str(generation / "build"),
        ],
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / ".benchmark-environments"
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        required=True,
        help="Temporary directory for Rust/build tools",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rust-toolchain", default="stable")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.build_root.mkdir(parents=True, exist_ok=True)
    try:
        identity = fingerprint(ROOT, args.rust_toolchain)

        def build(generation):
            print(f"Building persistent Python environment: {generation}", flush=True)
            build_environment(
                generation, args.build_root.resolve(), args.rust_toolchain
            )
            if fingerprint(ROOT, args.rust_toolchain) != identity:
                raise RuntimeError(
                    "Build inputs changed during setup; cache was not published"
                )

        receipt = ensure_cached(
            args.cache_root,
            identity,
            build,
            lambda generation: probe_environment(generation, ROOT),
            args.rebuild,
        )
        receipt["build_records"] = str(Path(receipt["generation"]) / "build")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt)
        shutil.copytree(
            receipt["build_records"],
            args.artifacts,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("wheels"),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Environment setup failed: {error}\n")
    print(
        ("Reusing" if receipt["reused"] else "Created")
        + f" cached environment: {receipt['python']}",
        flush=True,
    )


if __name__ == "__main__":
    if not sys.flags.ignore_environment or not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, "-E", "-s", *sys.orig_argv[1:]])
    main()
