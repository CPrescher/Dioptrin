"""Check an installed release wheel with no license and no workstation state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROBE = r'''
import dioptrin

poni = dict(pixel1=1e-4, pixel2=1e-4, distance=0.2, poni1=0.0032,
            poni2=0.0032, rot1=0., rot2=0., rot3=0., wavelength=1e-10)

def rejected(operation):
    try:
        operation()
    except dioptrin.LicenseError:
        return
    raise AssertionError("Release wheel did not enforce licensing")

rejected(dioptrin.validate_license)
for monkeypatch in (False, True):
    if monkeypatch:
        dioptrin.validate_license = lambda: None
        dioptrin.dioptrin.validate_license = lambda: None
    rejected(lambda: dioptrin.Integrator("nonexistent-release-check.poni"))
    rejected(lambda: dioptrin.Integrator.from_poni_dict(poni))
print("PASS: validation and both constructors reject; Python monkeypatch does not unlock")
'''


def main():
    with tempfile.TemporaryDirectory(prefix="dioptrin-license-check-") as directory:
        home = Path(directory)
        env = os.environ.copy()
        env.pop("DIOPTRIN_LICENSE", None)
        for variable in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                         "APPDATA", "LOCALAPPDATA"):
            env[variable] = str(home)
        for content in (None, "invalid-release-verification-license"):
            if content is not None:
                env["DIOPTRIN_LICENSE"] = content
            subprocess.run([sys.executable, "-I", "-c", PROBE], env=env, cwd=home, check=True)
    print("Installed wheel rejects missing and invalid licenses")


if __name__ == "__main__":
    main()
