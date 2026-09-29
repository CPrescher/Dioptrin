# Dioptrin

Dioptrin provides fast 1D and 2D diffraction integration using a compiled Rust
engine. It supports pixel splitting, supersampling, detector corrections,
Poisson uncertainties, azimuthal intensity spread, and batch integration.

## Installation

```bash
python -m pip install dioptrin
```

Dioptrin is distributed as binary wheels for standard CPython 3.11–3.15 on
Windows x86-64, macOS Intel and Apple silicon, and Linux x86-64 and ARM64.
Python 3.15 is currently a prerelease. Free-threaded Python is not supported.
No source distribution is provided; installation requires a compatible wheel.

## License activation

To request a license, contact [Clemens Prescher](mailto:clemens.prescher@gmail.com).

Downloading the package does not activate it. A valid Dioptrin license is required
to use the integration engine. Place your issued license file at
`~/.dioptrin/license.key`, where `~` is your user home directory. On Windows this
is normally `%USERPROFILE%\.dioptrin\license.key`.

For a beamline or server, the `DIOPTRIN_LICENSE` environment variable can instead
contain the complete license file contents (not the filename).

Verify activation with:

```bash
python -c "import dioptrin; dioptrin.validate_license()"
```

Activation and integration remain fully offline. An integrator retains the license
loaded when it was constructed, so create a new integrator after renewing a
license. New operations check expiration; a batch that has already started may
finish after expiry.

## Use with Dioptas

Install Dioptrin in the same Python environment as Dioptas, activate the license,
and restart Dioptas. The **Use Dioptrin** checkbox becomes available in its
integration options. Dioptas continues to work with pyFAI without Dioptrin.
