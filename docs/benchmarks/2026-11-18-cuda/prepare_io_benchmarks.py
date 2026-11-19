"""Stage isolated Rust sources for the dioptrin-io Eiger benchmark variants."""

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

from run import ROOT


HERE = Path(__file__).resolve().parent
IO_COMMIT = "97f5ba73ca1f8b169ce3037ff79b0a51b40f6098"
IO_DEPENDENCY = 'dioptrin-io = { path = "../dioptrin-io", features = ["hdf5"] }\n'


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digests():
    files = [ROOT / "Cargo.toml", Path(__file__),
             HERE / "dioptrin_io_caller_buffer.patch", *HERE.glob("io_reader_*.rs")]
    if (ROOT / "Cargo.lock").is_file():
        files.append(ROOT / "Cargo.lock")
    files.extend(p for p in (ROOT / "src").rglob("*") if p.is_file())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def build_record_path(executable):
    return executable.with_name(executable.name + ".benchmark-build.json")


def executable_provenance(executable, allow_unverified=False):
    path = build_record_path(executable)
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["executable_sha256"] == digest(executable) and record["source_sha256"] == source_digests()
                and record["dioptrin_io_commit"] == IO_COMMIT):
            return record | {"verified": True}
    if allow_unverified:
        return {"verified": False, "executable_sha256": digest(executable)}
    raise ValueError("Prebuilt executable has no matching build record. Stage it with prepare_io_benchmarks.py --build.")


def build_variants(cargo, source, bridge, record):
    for directory, command, target, kind in (
        (source, ["test", "--lib", "--features", "cuda", "--no-run"], "dioptrin", "bench"),
        (bridge, ["build"], "eiger_io_bridge", "bridge"),
    ):
        report = subprocess.check_output([cargo, *command, "--release", "--message-format=json"], cwd=directory, text=True, encoding="utf-8")
        for line in report.splitlines():
            row = json.loads(line)
            if row.get("reason") != "compiler-artifact" or row["target"]["name"] != target:
                continue
            if kind == "bench":
                if not row["profile"]["test"] or not row["executable"]:
                    continue
                artifact = Path(row["executable"])
            else:
                artifact = next(Path(name) for name in row["filenames"]
                                if Path(name).suffix in (".dll", ".so", ".dylib"))
            build_record_path(artifact).write_text(json.dumps(record | dict(
                executable_sha256=digest(artifact), staged_cargo_lock_sha256=digest(directory / "Cargo.lock"),
                cargo_arguments=[*command, "--release"], kind=kind,
            ), indent=2), encoding="utf-8")
            print(artifact, flush=True)
            break
        else:
            raise RuntimeError(f"Cargo did not produce {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-source", type=Path, required=True, help="Existing local dioptrin-io Git clone")
    parser.add_argument("--work", type=Path, required=True, help="New benchmark source directory")
    parser.add_argument("--build", action="store_true", help="Build both staged artifacts and record their provenance")
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    io = work / "dioptrin-io"
    archive = subprocess.check_output([
        "git", "-C", str(args.io_source.resolve()), "archive", "--format=zip", IO_COMMIT,
    ])
    with ZipFile(BytesIO(archive)) as source_archive:
        source_archive.extractall(io)
    subprocess.run(["git", "apply", str(HERE / "dioptrin_io_caller_buffer.patch")],
                   cwd=io, env=os.environ | {"GIT_CEILING_DIRECTORIES": str(work)}, check=True)

    source = work / "dioptrin"
    source.mkdir()
    shutil.copytree(ROOT / "src", source / "src")
    shutil.copytree(ROOT / "benches", source / "benches")
    (source / "test_data").mkdir()
    shutil.copy2(ROOT / "test_data" / "detector_reference.spline", source / "test_data")
    shutil.copy2(ROOT / "README.md", source)
    if (ROOT / "Cargo.lock").is_file():
        shutil.copy2(ROOT / "Cargo.lock", source)
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    (source / "Cargo.toml").write_text(
        manifest.replace("[dev-dependencies]\n", "[dev-dependencies]\n" + IO_DEPENDENCY, 1),
        encoding="utf-8",
    )
    bench = source / "src" / "integration" / "cuda" / "bench"
    hdf5 = bench / "hdf5.rs"
    harness = hdf5.read_text(encoding="utf-8")
    harness = harness.replace(
        "mod reader;", '#[path = "hdf5/io_reader.rs"]\nmod reader;\nmod persistent;', 1,
    )
    hdf5.write_text(harness.replace("Dataset<'_>", "Dataset"), encoding="utf-8")
    shutil.copy2(HERE / "io_reader_adapter.rs", bench / "hdf5" / "io_reader.rs")
    shutil.copy2(HERE / "io_reader_persistent.rs", bench / "hdf5" / "persistent.rs")

    bridge = work / "bridge"
    (bridge / "src").mkdir(parents=True)
    shutil.copy2(HERE / "io_reader_bridge.rs", bridge / "src" / "lib.rs")
    (bridge / "Cargo.toml").write_text(
        '[package]\nname = "eiger-io-bridge"\nversion = "0.1.0"\nedition = "2024"\n\n'
        '[lib]\ncrate-type = ["cdylib"]\n\n[dependencies]\n' + IO_DEPENDENCY +
        '\n[profile.release]\nlto = true\ncodegen-units = 1\nstrip = "symbols"\n',
        encoding="utf-8",
    )
    print(f"Prepared {source} and {bridge}")
    if args.build:
        build_variants(args.cargo, source, bridge, dict(
            dioptrin_io_commit=IO_COMMIT, source_sha256=source_digests(),
            rustflags=os.environ.get("RUSTFLAGS"),
            rustc=subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        ))


if __name__ == "__main__":
    main()
