#!/usr/bin/env python3
"""Build an isolated, CPU-targeted comparison environment from a Dioptrin clone."""

import argparse
import glob
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv

from provenance import digest, source_hashes


ROOT = Path(__file__).resolve().parent.parent.parent
AZINT_COMMIT = "4f78cc3b1d8cab67128c9679f55de6340c68f191"


def run(command, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def compiler_environment(portable=False):
    env = os.environ.copy()
    if "CXX" not in env:
        if sys.platform == "darwin":
            choices = glob.glob("/opt/homebrew/bin/g++-*") + glob.glob(
                "/usr/local/bin/g++-*"
            )
            choices = [
                p for p in choices if Path(p).name.removeprefix("g++-").isdigit()
            ]
            if not choices:
                raise RuntimeError(
                    "An OpenMP compiler is required. On macOS install Homebrew gcc (brew install gcc), or set CC/CXX to an OpenMP-enabled compiler."
                )
            env["CXX"] = max(
                choices, key=lambda p: int(Path(p).name.removeprefix("g++-"))
            )
        else:
            env["CXX"] = "g++"
    if "CC" not in env:
        env["CC"] = env["CXX"].replace("g++", "gcc").replace("clang++", "clang")
    native = (
        "-mcpu=native"
        if platform.machine().lower() in ("arm64", "aarch64")
        else "-march=native"
    )
    env["CFLAGS"] = f"{env.get('CFLAGS', '')} -O3" + (
        f" {native}" if not portable else ""
    )
    env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -O3" + (
        f" {native}" if not portable else ""
    )
    env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + (
        " -C target-cpu=native" if not portable else ""
    )
    return env


def accepts_flag(env, flag):
    """True when the C++ compiler compiles and links a trivial program with flag."""
    with tempfile.TemporaryDirectory(prefix="dioptrin-flag-") as temporary:
        source = Path(temporary) / "check.cpp"
        source.write_text("int main() { return 0; }\n")
        result = subprocess.run(
            [
                *shlex.split(env["CXX"]),
                flag,
                source,
                "-o",
                Path(temporary) / "check",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0


def azint_environment(env, fast_math=False, portable=False, probe=accepts_flag):
    """Compiler environment and provenance for the azint extension only.

    The default keeps the historical -O3 (+ native CPU) build. fast_math follows
    the 2026-10-24 "CPU/LTO/fast-math" azint variant: -ffast-math lets the
    compiler reassociate the sequential double reduction in azint's _spmv, which
    is what Dioptrin's default fast_math=True does for its CSR accumulation.
    -flto=thin is added only when the compiler accepts it (clang); GCC rejects
    it, and azint is a single translation unit, so nothing else is substituted.
    """
    env = env.copy()
    added = []
    lto = None
    if fast_math:
        added.append("-ffast-math")
        if probe(env, "-flto=thin"):
            added.append("-flto=thin")
            lto = "-flto=thin"
            env["LDFLAGS"] = f"{env.get('LDFLAGS', '')} -flto=thin"
        else:
            lto = "omitted: compiler rejects -flto=thin (single translation unit)"
    for key in ("CFLAGS", "CXXFLAGS"):
        env[key] = " ".join([env.get(key, ""), *added]).rstrip()
    record = dict(
        commit=AZINT_COMMIT,
        variant="fast-math" if fast_math else "default",
        fast_math=fast_math,
        native_cpu=not portable,
        lto=lto,
        added_flags=added,
        flags={
            k: env.get(k, "") for k in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        },
    )
    return env, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-benchmark")
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Omit native CPU targeting (for shared CI caches)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Build metadata/wheels; default: <venv>/build-record",
    )
    parser.add_argument(
        "--azint-fast-math",
        action="store_true",
        help="Build azint with -ffast-math (and -flto=thin where supported), "
        "matching Dioptrin's default fast_math=True reassociated accumulation",
    )
    parser.add_argument(
        "--azint-only",
        action="store_true",
        help="Rebuild only azint in an existing environment; keep the installed "
        "Dioptrin, pyFAI, and their build record unchanged",
    )
    args = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        parser.error(
            "The comparison tool currently supports Linux/macOS; on Windows use WSL2."
        )
    if not (3, 11) <= sys.version_info[:2] < (3, 16):
        parser.error("Use Python 3.11–3.15")
    for tool in ("cargo", "rustc", "git"):
        if shutil.which(tool) is None:
            parser.error(
                f"Missing {tool}; install the stable Rust toolchain and Git first"
            )
    target = args.venv.resolve()
    if (
        target.exists()
        and not (target / "pyvenv.cfg").is_file()
        and any(target.iterdir())
    ):
        parser.error(f"Refusing to replace a non-venv directory: {target}")
    artifacts = (args.artifacts or target / "build-record").resolve()
    previous = None
    if args.azint_only:
        try:
            previous = json.loads((target / "benchmark-build.json").read_text())
        except (OSError, ValueError):
            parser.error(
                f"--azint-only needs an existing environment with a build record: {target}"
            )
    try:
        env = compiler_environment(args.portable)
        with tempfile.TemporaryDirectory(prefix="dioptrin-openmp-") as temporary:
            source = Path(temporary) / "check.cpp"
            executable = Path(temporary) / "check"
            source.write_text(
                "#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n"
            )
            run(
                [
                    *shlex.split(env["CXX"]),
                    "-std=c++17",
                    "-fopenmp",
                    source,
                    "-o",
                    executable,
                ],
                env=env,
            )
            run([executable], env=env)
        if not (target / "pyvenv.cfg").exists():
            venv.create(target, with_pip=True)
        python = target / "bin" / "python"
        # Isolate the environment from module/user Python paths on clusters.
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["VIRTUAL_ENV"] = str(target)
        env["PATH"] = str(target / "bin") + os.pathsep + env["PATH"]
        artifacts.mkdir(parents=True, exist_ok=True)
        wheels = artifacts / "wheels"
        wheels.mkdir(exist_ok=True)
        if not args.azint_only:
            run([python, "-m", "pip", "install", "--upgrade", "pip"], env=env)
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    Path(__file__).with_name("requirements.txt"),
                ],
                env=env,
            )
            locked = ["--locked"] if (ROOT / "Cargo.lock").is_file() else []
            run(
                [
                    python,
                    "-m",
                    "maturin",
                    "build",
                    "--release",
                    "--features",
                    "python",
                    *locked,
                    "--interpreter",
                    python,
                    "--out",
                    wheels,
                ],
                cwd=ROOT,
                env=env,
            )
            wheel = max(
                wheels.glob("dioptrin-*.whl"), key=lambda p: p.stat().st_mtime_ns
            )
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    wheel,
                ],
                env=env,
            )
        azint_env, azint_build = azint_environment(
            env, args.azint_fast_math, args.portable
        )
        # Keep each azint flavour's wheel: both carry the same version/file name.
        azint_wheels = wheels / f"azint-{azint_build['variant']}"
        azint_wheels.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dioptrin-azint-") as temporary:
            source = Path(temporary) / "azint"
            run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "https://github.com/maxiv-science/azint.git",
                    source,
                ],
                env=env,
            )
            run(
                ["git", "-C", source, "checkout", "--quiet", "--detach", AZINT_COMMIT],
                env=env,
            )
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-cache-dir",
                    source,
                    "--wheel-dir",
                    azint_wheels,
                ],
                env=azint_env,
            )
        wheel = max(
            azint_wheels.glob("azint-*.whl"), key=lambda p: p.stat().st_mtime_ns
        )
        run(
            [python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel],
            env=env,
        )
        azint_build["wheel"] = dict(name=wheel.name, sha256=digest(wheel))
        # The installed extension identifies the binary that compare.py loads, and
        # -ffast-math must not have switched the process to flush-to-zero.
        azint_build.update(
            json.loads(
                subprocess.check_output(
                    [
                        str(python),
                        "-E",
                        "-s",
                        "-c",
                        "import sys,json;sys.path.insert(0,sys.argv[1]);"
                        "from provenance import azint_signature;import numpy as np;"
                        "import _azint;"
                        "tiny=[np.finfo(t).tiny for t in (np.float32,np.float64)];"
                        "print(json.dumps(dict(installed_extension=azint_signature(),"
                        "subnormals_preserved=all(np.array([x])[0]/2!=0 for x in tiny))))",
                        str(Path(__file__).parent),
                    ],
                    env=env,
                    text=True,
                )
            )
        )
        if not azint_build["subnormals_preserved"]:
            print(
                "WARNING: importing azint enabled flush-to-zero in this process",
                flush=True,
            )
        run([python, "-m", "pip", "check"], env=env)
        for filename, command in (
            ("python-packages.txt", [python, "-m", "pip", "freeze", "--all"]),
            ("rustc.txt", ["rustc", "-Vv"]),
            ("cxx.txt", [*shlex.split(env["CXX"]), "--version"]),
        ):
            (artifacts / filename).write_text(
                subprocess.check_output(list(map(str, command)), env=env, text=True)
            )
        if args.azint_only:
            # Dioptrin, pyFAI, and their source/binary record are left as built.
            record = dict(previous)
            record.update(azint_commit=AZINT_COMMIT, azint_build=azint_build)
            record.setdefault("history", []).append(
                dict(action="azint-only rebuild", azint_variant=azint_build["variant"])
            )
            record = json.dumps(record, indent=2) + "\n"
            (artifacts / "build.json").write_text(record)
            (target / "benchmark-build.json").write_text(record)
            print(f"\nRebuilt azint ({azint_build['variant']}): {azint_build['flags']}")
            return
        shutil.copy2(ROOT / "Cargo.lock", artifacts / "Cargo.lock")
        source = {}
        if (ROOT / ".git").exists():
            source = {
                key: subprocess.check_output(command, cwd=ROOT, text=True).strip()
                for key, command in (
                    ("commit", ["git", "rev-parse", "HEAD"]),
                    ("status", ["git", "status", "--short"]),
                )
            }
        # A version string alone cannot distinguish builds from two revisions
        # of the same release. Record actual source bytes with the installed env.
        source["sha256"] = source_hashes(ROOT)
        installed = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-E",
                    "-s",
                    "-c",
                    "import sys,json;sys.path.insert(0,sys.argv[1]);from provenance import installed_signature;print(json.dumps(installed_signature()))",
                    str(Path(__file__).parent),
                ],
                env=env,
                text=True,
            )
        )
        record = (
            json.dumps(
                dict(
                    azint_commit=AZINT_COMMIT,
                    azint_build=azint_build,
                    python=str(python),
                    platform=platform.platform(),
                    source=source,
                    installed_dioptrin=installed,
                    portable=args.portable,
                    flags={
                        k: env[k]
                        for k in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "RUSTFLAGS")
                    },
                ),
                indent=2,
            )
            + "\n"
        )
        (artifacts / "build.json").write_text(record)
        (target / "benchmark-build.json").write_text(record)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        parser.exit(1, f"Setup failed: {error}\n")
    print(
        "\nRun the standard suite:\n"
        + shlex.join([str(python), str(Path(__file__).with_name("compare.py"))])
    )


if __name__ == "__main__":
    main()
