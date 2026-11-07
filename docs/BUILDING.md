# Building the manual

The manual uses [Sphinx](https://www.sphinx-doc.org/) with
[MyST](https://myst-parser.readthedocs.io/) for Markdown and the
[Furo](https://pradyunsg.me/furo/) theme. Installation, API examples and release
notes are included from their existing repository files, so edits have a single
source. No Rust build, detector data or Dioptrin installation is needed.

## Local build

From the repository root, using Python 3.13:

```bash
python3.13 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

On Windows, create the environment with `py -3.13 -m venv .venv-docs` and
activate it with `.venv-docs\Scripts\Activate.ps1` in PowerShell.

Open `docs/_build/html/index.html` in a browser. Alternatively, run
`python -m http.server --directory docs/_build/html 8000` and visit
`http://localhost:8000`. Warnings fail the build, including unresolved internal
document or heading references. To check external links as well:

```bash
python -m sphinx -n -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

The Documentation GitHub Actions workflow runs the strict HTML build for pushes
and pull requests. Generated HTML and documentation virtual environments are
ignored by Git. The Python type stub is available as a download. Native reader
measurements and their downloads are maintained in the
[dioptrin-io manual](https://dioptrin-io.readthedocs.io/en/latest/PERFORMANCE.html);
the historical benchmark directory here is excluded from document discovery.

## Read the Docs

The root `.readthedocs.yaml` selects Ubuntu 24.04, Python 3.13, the same pinned
documentation dependencies and `docs/conf.py`, with warnings treated as errors.
The displayed Dioptrin version comes from `pyproject.toml` in the checked-out
commit.

The [published manual](https://dioptrin.readthedocs.io/en/latest/) is managed in
the [Dioptrin Read the Docs project](https://app.readthedocs.org/projects/dioptrin/).
Its default branch is `main`, its configuration file is `.readthedocs.yaml`,
and a signed GitHub incoming webhook delivers repository updates for automatic
builds. The `latest` version tracks `main`; enable release versions when their
tags contain this documentation configuration. Older tags without it need
separate configuration and do not gain this manual automatically.

See the official [Sphinx deployment guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html)
and [versioning guide](https://docs.readthedocs.com/platform/stable/versions.html).
