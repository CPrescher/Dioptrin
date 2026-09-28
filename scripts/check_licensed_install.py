"""Verify the installed release wheel enforces licensing, without a secret."""
import os

# A nonempty invalid value overrides any workstation license file.
os.environ["DIOPTRIN_LICENSE"] = "invalid-release-verification-license"

import dioptrin

for operation in (
    dioptrin.validate_license,
    lambda: dioptrin.Integrator("nonexistent-release-check.poni"),
    lambda: dioptrin.Integrator.from_poni_dict({}),
):
    try:
        operation()
    except dioptrin.LicenseError:
        pass
    else:
        raise RuntimeError("Release wheel did not enforce licensing")

print("Installed wheel rejects an invalid license")
