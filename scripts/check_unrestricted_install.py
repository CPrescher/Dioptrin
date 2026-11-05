"""Check an installed release wheel without activation or valid license state."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROBE = r"""
from pathlib import Path
import importlib.metadata
import numpy as np
import dioptrin

assert importlib.metadata.metadata("dioptrin")["License-Expression"] == "MIT"
assert dioptrin.validate_license() is None
assert issubclass(dioptrin.LicenseNotFoundError, dioptrin.LicenseError)
assert issubclass(dioptrin.LicenseExpiredError, dioptrin.LicenseError)
poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.2, poni1=0.0032,
            poni2=0.0032, rot1=0., rot2=0., rot3=0., wavelength=1e-10)
path = Path("release-check.poni")
path.write_text("poni_version: 1\nPixelSize1: 0.0001\nPixelSize2: 0.0001\n"
    "Distance: 0.2\nPoni1: 0.0032\nPoni2: 0.0032\nWavelength: 1e-10\n")
image = np.arange(64*64, dtype=np.float64).reshape(64, 64)
kwargs = dict(unit="2th_deg", radial_range=(0., 10.))
integrators = [dioptrin.Integrator(str(path), **kwargs),
    dioptrin.Integrator.from_poni_dict(poni, **kwargs)]
for ai in integrators:
    one = ai.integrate1d(image, 40, errors=True)
    assert len(one.intensity) == 40 and np.isfinite(one.intensity).all()
    two = ai.integrate2d(image, 40, 12, errors=True)
    assert len(two.intensity) == 480 and np.isfinite(two.intensity).all()
    for results in (ai.batch1d([image, image], 40, errors=True),
                    ai.batch1d_iter(iter([image, image]), 40, errors=True)):
        results = list(results)
        assert len(results) == 2 and all(r.is_ok() for r in results)
        for result in results:
            np.testing.assert_array_equal(result.result.intensity, one.intensity)
multi = dioptrin.MultiIntegrator(integrators)
assert np.isfinite(multi.integrate1d([image, image], 40, errors=True).intensity).all()
assert len(multi.batch1d([[image, image]], 40, errors=True)) == 1
print("PASS: constructors, 1D/2D, batches, streams and multi-detector integration")
"""


def main():
    with tempfile.TemporaryDirectory(prefix="dioptrin-release-check-") as directory:
        env = os.environ.copy()
        env.pop("DIOPTRIN_LICENSE", None)
        for content in (None, "invalid-release-verification-license"):
            if content is not None:
                env["DIOPTRIN_LICENSE"] = content
            subprocess.run(
                [sys.executable, "-I", "-c", PROBE],
                env=env,
                cwd=Path(directory),
                check=True,
            )
    print("Installed wheel works with missing and invalid activation state")


if __name__ == "__main__":
    main()
