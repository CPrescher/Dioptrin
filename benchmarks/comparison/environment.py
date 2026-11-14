#!/usr/bin/env python3
"""Persist native benchmark environments without reusing incompatible binaries."""

import argparse
from dataclasses import dataclass
import errno
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
SCHEMA_VERSION = 2
PROBE_TIMEOUT = 60
BUILD_FLAGS = (
    "CFLAGS",
    "CXXFLAGS",
    "RUSTFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)


def sanitized_build_environment():
    # Do not let sbatch's inherited CARGO_*, PIP_*, MATURIN_* or PYO3_*
    # overrides change a build without changing its cache identity.
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "TMPDIR",
        "TMP",
        "TEMP",
        "CC",
        "CXX",
        *BUILD_FLAGS,
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed or key.startswith("LC_")
    }
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INPUT="1", PYTHONNOUSERSITE="1")
    return env


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str = ""

    def __bool__(self):
        return self.ok


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
    for name, value in {
        "CC": env["CC"],
        "CXX": env["CXX"],
        "default_rust_linker": "cc",
    }.items():
        command = shlex.split(value)
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
        flags={name: env.get(name, "") for name in BUILD_FLAGS},
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
        cargo_configs={
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (root, *root.parents)
            for path in (directory / ".cargo/config", directory / ".cargo/config.toml")
            if path.is_file()
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


def read_generation(entry, key):
    try:
        index = json.loads((entry / "ready.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or index.get("key") != key:
        return None
    name = index.get("generation")
    if (
        not isinstance(name, str)
        or not name.startswith("env-")
        or Path(name).name != name
    ):
        return None
    candidate = entry / name
    return candidate if candidate.resolve().parent == entry else None


def ensure_cached(cache_root, identity, build, validate, rebuild=False):
    """Validate immutable warm entries without a lock; serialize construction only."""
    key = cache_key(identity)
    entry = cache_root.resolve() / key
    entry.mkdir(parents=True, exist_ok=True)

    def usable(generation):
        if generation is None:
            return False
        result = validate(generation)
        if not result:
            reason = getattr(result, "reason", "validation returned false")
            print(f"Cached environment invalid ({generation}): {reason}", flush=True)
        return bool(result)

    def receipt(generation, reused):
        return dict(
            key=key,
            generation=str(generation),
            python=str(generation / "venv/bin/python"),
            reused=reused,
            fingerprint=identity,
        )

    observed = read_generation(entry, key)
    if not rebuild and usable(observed):
        return receipt(observed, True)
    with (entry / "build.lock").open("a+") as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            print(f"Waiting for environment build lock: {entry}", flush=True)
            fcntl.lockf(lock, fcntl.LOCK_EX)
        lock.seek(0)
        lock.truncate()
        lock.write(
            json.dumps(
                dict(
                    pid=os.getpid(),
                    host=platform.node(),
                    job=os.environ.get("SLURM_JOB_ID"),
                )
            )
        )
        lock.flush()
        latest = read_generation(entry, key)
        # Simultaneous force requests share the rebuild that completed while
        # they waited, instead of each doing another complete installation.
        if (not rebuild or latest != observed) and usable(latest):
            return receipt(latest, True)
        generation = entry / ("env-" + uuid.uuid4().hex)
        generation.mkdir()
        try:
            build(generation)
            result = validate(generation)
            if not result:
                raise RuntimeError(
                    "New environment failed validation: "
                    + getattr(result, "reason", "validation returned false")
                )
        except BaseException:
            # This generation has never been visible to a benchmark job.
            shutil.rmtree(generation, ignore_errors=True)
            raise
        # Mark before publication: a marked generation might be in use, even
        # if a kill arrives between marker creation and pointer replacement.
        atomic_json(generation / "published.json", dict(key=key))
        atomic_json(
            entry / "ready.json",
            dict(key=key, generation=generation.name, fingerprint=identity),
        )
        return receipt(generation, False)


def probe_environment(generation, root, timeout=PROBE_TIMEOUT):
    python = generation / "venv/bin/python"
    record = generation / "venv/benchmark-build.json"
    frozen = generation / "build/python-packages.txt"
    if not python.is_file() or not record.is_file() or not frozen.is_file():
        return ProbeResult(False, "missing interpreter or build records")
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
            timeout=timeout,
        )
        if result.returncode:
            return ProbeResult(
                False,
                "imports/provenance: "
                + (result.stdout + result.stderr).strip()[-2000:],
            )
        result = subprocess.run(
            [str(python), "-E", "-s", "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            return ProbeResult(
                False, "pip freeze failed: " + result.stderr.strip()[-2000:]
            )
        if sorted(result.stdout.splitlines()) != sorted(
            frozen.read_text().splitlines()
        ):
            return ProbeResult(
                False, "installed package versions differ from the build record"
            )
        result = subprocess.run(
            [str(python), "-E", "-s", "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            return ProbeResult(
                False,
                "dependency check: " + (result.stdout + result.stderr).strip()[-2000:],
            )
        return ProbeResult(True)
    except subprocess.TimeoutExpired:
        return ProbeResult(False, f"validation subprocess timed out after {timeout}s")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return ProbeResult(False, str(error))


def build_environment(generation, build_root, rust_toolchain):
    env = sanitized_build_environment()
    env["CARGO_HOME"] = str(build_root / "cargo-home")
    env["RUSTUP_HOME"] = str(build_root / "rustup-home")
    env["CARGO_TARGET_DIR"] = str(build_root / "target")
    env["CARGO_BUILD_JOBS"] = os.environ.get("BUILD_JOBS", "8")
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
        env=env,
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
