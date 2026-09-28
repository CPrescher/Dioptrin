"""Reject source archives or private/source files before publishing wheels."""
import sys
import zipfile
from pathlib import Path


def check(directory):
    files = list(Path(directory).iterdir())
    if not files or any(p.suffix != ".whl" for p in files):
        raise RuntimeError("The publication directory must contain only wheels")
    for wheel in files:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            for name in names:
                path = Path(name)
                if (path.suffix in {".rs", ".key", ".pem"}
                        or path.name in {"Cargo.toml", "Cargo.lock"}
                        or "keys" in path.parts):
                    raise RuntimeError(f"Unexpected source or key file in {wheel.name}: {name}")
            if not any(n.endswith((".so", ".pyd")) for n in names):
                raise RuntimeError(f"Missing native extension in {wheel.name}")
        print(f"Checked {wheel.name}")


if __name__ == "__main__":
    check(sys.argv[1])
