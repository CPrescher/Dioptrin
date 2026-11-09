"""Validate open-source release archives and reject accidentally packaged keys."""

from email.parser import BytesParser
import sys
import tarfile
import zipfile
from pathlib import Path


def check(directory):
    files = list(Path(directory).iterdir())
    if not files:
        raise RuntimeError("The publication directory is empty")
    for artifact in files:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise RuntimeError(f"Missing package metadata in {artifact.name}")
                license_expression = BytesParser().parsebytes(
                    archive.read(metadata[0])
                )["License-Expression"]
                if license_expression != "MIT":
                    raise RuntimeError(f"Missing MIT metadata in {artifact.name}")
            if not any(n.endswith((".so", ".pyd")) for n in names):
                raise RuntimeError(f"Missing native extension in {artifact.name}")
        elif artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact, "r:gz") as archive:
                names = archive.getnames()
            if not any(n.endswith("/Cargo.toml") for n in names):
                raise RuntimeError(f"Missing Rust source manifest in {artifact.name}")
        else:
            raise RuntimeError(f"Unexpected release artifact: {artifact.name}")
        for name in names:
            path = Path(name)
            if path.suffix in {".key", ".pem", ".p12"} or "keys" in path.parts:
                raise RuntimeError(f"Unexpected key file in {artifact.name}: {name}")
        if not any(Path(n).name == "LICENSE" for n in names):
            raise RuntimeError(f"Missing MIT license file in {artifact.name}")
        for notice in ("pyFAI-detectors.txt", "pyFAI-pixel-splitting.txt"):
            if not any(Path(n).name == notice for n in names):
                raise RuntimeError(f"Missing upstream notice {notice} in {artifact.name}")
        print(f"Checked {artifact.name}")


if __name__ == "__main__":
    check(sys.argv[1])
