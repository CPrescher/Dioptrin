"""Sphinx configuration for the Markdown user manual.

No native extension is imported: documentation builds do not need Rust,
detector data or the optional scientific Python dependencies.
"""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "pyproject.toml").open("rb") as metadata_file:
    metadata = tomllib.load(metadata_file)["project"]

project = "Dioptrin"
author = "Clemens Prescher"
copyright = "2026, Clemens Prescher"
release = metadata["version"]
version = release

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
nitpicky = True
myst_heading_anchors = 4

# Benchmark notebooks and historical reports are records, not manual chapters.
# Selected JSON files are copied only when referenced as downloads by a chapter.
exclude_patterns = ["_build", "README.md", "benchmarks/**"]

html_theme = "furo"
html_title = f"Dioptrin {release} manual"
html_theme_options = {
    "source_repository": "https://github.com/CPrescher/Dioptrin/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Check external destinations with `sphinx-build -b linkcheck`; HTML builds
# already validate internal document links and heading references.
linkcheck_retries = 2
linkcheck_timeout = 20
linkcheck_rate_limit_timeout = 30
linkcheck_workers = 2
