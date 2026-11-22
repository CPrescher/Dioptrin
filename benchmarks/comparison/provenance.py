"""Identify the source and installed native code actually used by a comparison."""

import hashlib
import importlib
import importlib.util
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


def azint_signature():
    """Path and hash of azint's native extension, located without importing it."""
    spec = importlib.util.find_spec("_azint")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve()
    return {"module": str(path), "sha256": digest(path)}


def verify_azint_build(record, installed):
    """Compare the loaded azint extension with the one setup.py recorded."""
    recorded = ((record or {}).get("azint_build") or {}).get("installed_extension")
    if recorded is None:
        return []
    if installed != recorded:
        return ["The installed azint extension differs from the recorded build"]
    return []


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
