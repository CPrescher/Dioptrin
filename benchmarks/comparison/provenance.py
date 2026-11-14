"""Identify the source and installed native code actually used by a comparison."""

import hashlib
import importlib
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root):
    files = [root / "Cargo.toml", root / "pyproject.toml"]
    files.extend(p for p in (root / "src").rglob("*") if p.is_file())
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def installed_signature():
    path = Path(importlib.import_module("dioptrin").__file__).resolve()
    files = {path}
    if path.suffix == ".py":
        files.update(path.parent.glob("dioptrin*.so"))
        files.update(path.parent.glob("dioptrin*.pyd"))
    return {"module": str(path), "sha256": {p.name: digest(p) for p in sorted(files)}}


def verify_build(root, record):
    """Return clear mismatch reasons; do not mistake a current Git SHA for a build."""
    if record is None:
        return ["No Dioptrin build record is installed in this environment"]
    problems = []
    if record.get("source", {}).get("sha256") != source_hashes(root):
        problems.append(
            "Dioptrin source/build configuration differs from the installed build"
        )
    if record.get("installed_dioptrin") != installed_signature():
        problems.append(
            "The installed Dioptrin module differs from the recorded binary"
        )
    return problems
