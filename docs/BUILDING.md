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
ignored by Git. Only the benchmark JSON files linked from the performance guide
are included as downloads; the historical benchmark directory is excluded from
document discovery.

## Read the Docs

The root `.readthedocs.yaml` selects Ubuntu 24.04, Python 3.13, the same pinned
documentation dependencies and `docs/conf.py`, with warnings treated as errors.
The displayed Dioptrin version comes from `pyproject.toml` in the checked-out
commit.

To activate hosting, import `https://github.com/CPrescher/Dioptrin` into a
[Read the Docs Community](https://app.readthedocs.org/) account. Select `main`
as the default branch and `.readthedocs.yaml` as the configuration file. Connect
the repository integration for automatic builds and pull-request previews.
The `latest` version should track `main`; enable release versions when their
tags contain this documentation configuration. Older tags without it need
separate configuration and do not gain this manual automatically.

See the official [Sphinx deployment guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html)
and [versioning guide](https://docs.readthedocs.com/platform/stable/versions.html).
