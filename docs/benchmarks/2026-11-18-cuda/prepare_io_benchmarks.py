"""Stage isolated Rust sources for the dioptrin-io Eiger benchmark variants."""

import argparse
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

from run import ROOT


HERE = Path(__file__).resolve().parent
IO_COMMIT = "97f5ba73ca1f8b169ce3037ff79b0a51b40f6098"
IO_DEPENDENCY = 'dioptrin-io = { path = "../dioptrin-io", features = ["hdf5"] }\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-source", type=Path, required=True, help="Existing local dioptrin-io Git clone")
    parser.add_argument("--work", type=Path, required=True, help="New benchmark source directory")
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
    harness = harness.replace("mod reader;", '#[path = "hdf5/io_reader.rs"]\nmod reader;', 1)
    hdf5.write_text(harness.replace("Dataset<'_>", "Dataset"), encoding="utf-8")
    shutil.copy2(HERE / "io_reader_adapter.rs", bench / "hdf5" / "io_reader.rs")

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


if __name__ == "__main__":
    main()
