#!/usr/bin/env python3
"""Build an isolated, CPU-targeted comparison environment from a Dioptrin clone."""

import argparse
import glob
import hashlib
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


ROOT = Path(__file__).resolve().parent.parent.parent
AZINT_COMMIT = "4f78cc3b1d8cab67128c9679f55de6340c68f191"


def run(command, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def compiler_environment():
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
    env["CFLAGS"] = f"{env.get('CFLAGS', '')} -O3 {native}"
    env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -O3 {native}"
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -C target-cpu=native"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-benchmark")
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Build metadata/wheels; default: <venv>/build-record",
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
    try:
        env = compiler_environment()
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
        wheel = max(wheels.glob("dioptrin-*.whl"), key=lambda p: p.stat().st_mtime_ns)
        run(
            [python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel],
            env=env,
        )
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
                    source,
                    "--wheel-dir",
                    wheels,
                ],
                env=env,
            )
        wheel = max(wheels.glob("azint-*.whl"), key=lambda p: p.stat().st_mtime_ns)
        run(
            [python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel],
            env=env,
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
        source["sha256"] = {}
        for path in sorted((ROOT / "src").rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    source["sha256"][str(path.relative_to(ROOT))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        record = (
            json.dumps(
                dict(
                    azint_commit=AZINT_COMMIT,
                    python=str(python),
                    platform=platform.platform(),
                    source=source,
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
