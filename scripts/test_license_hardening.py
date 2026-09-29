"""Exercise licensed wheels with ephemeral test keys and native fault injection.

Run with a venv containing maturin, numpy, and cryptography. Requires Cargo.
Only temporary source copies and installations are changed. Never uses a real
license/signing key, adds a release feature, or publishes a test-key wheel.
"""
from base64 import b64encode
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import struct
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=0.2, poni1=0.0032,
            poni2=0.0032, rot1=0., rot2=0., rot3=0., wavelength=1e-10)

# Imported only in child processes after selecting the temporary wheel target.
def probe(mode):
    import numpy as np
    import dioptrin

    def rejected(operation):
        try:
            operation()
        except dioptrin.LicenseError:
            return
        raise AssertionError("Expected LicenseError")

    if mode == "diagnostic":
        assert dioptrin.validate_license() is None  # native diagnostic was stubbed
        rejected(lambda: dioptrin.Integrator("fixture.poni"))
        rejected(lambda: dioptrin.Integrator.from_poni_dict(PONI))
        print("PASS: native diagnostic/operation success does not unlock either constructor")
        return

    dioptrin.validate_license = lambda: None
    dioptrin.dioptrin.validate_license = lambda: None
    engine = dioptrin.Integrator.from_poni_dict(PONI, method="counting", unit="2th_deg")
    if mode == "benchmark":
        dioptrin.set_rayon_threads(1)
        for size in (64, 512):
            frame = np.ones((size, size), dtype=np.float64)
            engine.integrate1d(frame, 64)
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(100):
                    engine.integrate1d(frame, 64)
                samples.append((time.perf_counter() - started) / 100)
            print(f"TIMING {size}x{size} median_us={np.median(samples) * 1e6:.2f}", flush=True)
        return
    y, x = np.indices((64, 64))
    image = (10 + .1 * ((x - 31.5)**2 + (y - 31.5)**2)).astype(np.float64)
    # Tiny uncompressed float64 TIFF, generated without an external dataset.
    image_path = Path("fixture.tiff")
    tags = [(256, 4, 64), (257, 4, 64), (258, 3, 64), (259, 3, 1),
            (262, 3, 1), (273, 4, 134), (277, 3, 1), (278, 4, 64),
            (279, 4, image.nbytes), (339, 3, 3)]
    image_path.write_bytes(struct.pack("<2sHIH", b"II", 42, 8, len(tags)) +
                          b"".join(struct.pack("<HHII", tag, kind, 1, value)
                                   for tag, kind, value in tags) +
                          struct.pack("<I", 0) + image.astype("<f8").tobytes())
    common = dict(PONI)
    detector = dioptrin.Integrator.from_poni_dict(
        common, method="counting", unit="2th_deg", radial_range=(0, 1.3))
    multi = dioptrin.MultiIntegrator([detector, detector])
    if mode == "fault":
        # Both acquisition and MultiIntegrator construction were deliberately
        # bypassed in this fixture. The operation checks must still reject.
        operations = [
            lambda: engine.integrate1d(image, 16),
            lambda: engine.integrate2d(image, 16, 8),
            lambda: engine.batch1d([image], 16),
            lambda: engine.batch1d_iter(iter([image]), 16),
            lambda: engine.batch2d([], 16, 8),
            lambda: engine.batch2d_iter(iter([image]), 16, 8),
            lambda: multi.integrate1d([image, image], 16),
            lambda: multi.batch1d([[image, image]], 16),
        ]
        for operation in operations:
            rejected(operation)
        print("PASS: fabricated/expired constructor state rejected by all 8 operation boundaries")
        return

    poni_path = Path("fixture.poni")
    poni_path.write_text("poni_version: 1\nPixelSize1: 0.0001\nPixelSize2: 0.0001\n"
                         "Distance: 0.2\nPoni1: 0.0032\nPoni2: 0.0032\n"
                         "Rot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n")
    file_engine = dioptrin.Integrator(str(poni_path), method="counting", unit="2th_deg")
    np.testing.assert_allclose(file_engine.integrate1d(str(image_path), 16).intensity,
                               engine.integrate1d(image, 16).intensity)
    # Once admitted, a native object keeps its signed evidence; subsequent
    # environment changes must neither revoke it nor grant rights to new objects.
    os.environ["DIOPTRIN_LICENSE"] = "invalid-after-construction"
    rejected(lambda: dioptrin.Integrator.from_poni_dict(PONI))
    a = engine.integrate1d(image, 16, errors=True)
    np.testing.assert_allclose(engine.integrate1d(image * 2, 16).intensity,
                               np.asarray(a.intensity) * 2, rtol=1e-12)
    rho2 = ((x + .5) * 1e-4 - .0032)**2 + ((y + .5) * 1e-4 - .0032)**2
    theta = np.degrees(np.arctan2(np.sqrt(rho2), .2))
    edges = np.linspace(theta.min(), theta.max(), 17)
    reference = (np.histogram(theta, edges, weights=image)[0] /
                 np.histogram(theta, edges, weights=(1 + rho2 / .2**2)**-1.5)[0])
    np.testing.assert_allclose(a.intensity, reference, rtol=1e-12)
    np.testing.assert_allclose(a.radial, (edges[:-1] + edges[1:]) / 2, atol=1e-14)
    for method in ("counting", "pixel_split", "histogram", "supersampled"):
        engine.set_method(method, n_ss=2)
        one = engine.integrate1d(image, 16, errors=True, azimuthal_std=True)
        two = engine.integrate2d(image, 16, 8, errors=True)
        # Histogram intentionally ignores azimuthal binning in the existing API.
        expected_2d = 16 if method == "histogram" else 128
        assert len(one.intensity) == 16 and len(two.intensity) == expected_2d, method
        assert np.isfinite(one.intensity).all()
        for batch in (engine.batch1d([image], 16, errors=True),
                      engine.batch1d_iter(iter([image]), 16, errors=True)):
            result = list(batch)
            assert len(result) == 1 and result[0].is_ok()
            np.testing.assert_allclose(result[0].result.intensity, one.intensity)
        streamed = list(engine.batch2d_iter(iter([image]), 16, 8, errors=True))
        assert len(streamed) == 1 and streamed[0].is_ok()
        np.testing.assert_allclose(streamed[0].result.intensity, two.intensity)
        files = engine.batch2d([str(image_path)], 16, 8, errors=True)
        assert len(files) == 1 and files[0].is_ok()
        np.testing.assert_allclose(files[0].result.intensity, two.intensity)
    single = detector.integrate1d(image, 16)
    np.testing.assert_allclose(multi.integrate1d([image, image], 16).intensity,
                               single.intensity, equal_nan=True)
    assert len(multi.batch1d([[image, image]], 16)) == 1
    print("PASS: offline numerical reference, scaling, methods, batches and multi-detector")


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="dioptrin-hardening-") as temporary:
        temporary = Path(temporary)
        source = temporary / "source"
        source.mkdir()
        shutil.copytree(root / "src", source / "src")
        shutil.copytree(root / "benches", source / "benches")
        for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "README-pypi.md", "dioptrin.pyi"):
            if (root / name).exists():
                shutil.copy2(root / name, source / name)
        (source / "keys").mkdir()
        signing = Ed25519PrivateKey.generate()
        public = signing.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        (source / "keys/public.key").write_bytes(public)
        home = temporary / "home"
        home.mkdir()
        env = os.environ.copy()
        env.pop("DIOPTRIN_LICENSE", None)
        for name in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                     "APPDATA", "LOCALAPPDATA"):
            env[name] = str(home)
        # Keep Cargo's normal toolchain/cache for compilation; it does not run the
        # extension. Package test processes use only the isolated environment.
        build_env = os.environ.copy()
        build_env.pop("DIOPTRIN_LICENSE", None)
        build_env.pop("PYO3_PYTHON", None)
        for name in ("CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
            build_env.pop(name, None)

        def license_content(expires, valid=True):
            payload = f"email=fixture@example.invalid\nexpires={expires}"
            signature = signing.sign(payload.encode()) if valid else bytes(64)
            return payload + "\n---\n" + b64encode(signature).decode()

        def build(label, features="python,licensed"):
            wheels = temporary / (label + "-wheels")
            subprocess.run([sys.executable, "-m", "maturin", "build", "--release",
                            "--features", features, "--out", str(wheels),
                            "-i", sys.executable], cwd=source, env=build_env, check=True)
            target = temporary / (label + "-install")
            subprocess.run([sys.executable, "-m", "pip", "--isolated", "install",
                            "--no-deps", "--target", str(target),
                            str(next(wheels.glob("*.whl")))], env=env, check=True)
            return target

        def run(target, mode, content):
            child_env = env.copy()
            if content is not None:
                child_env["DIOPTRIN_LICENSE"] = content
            code = ("import runpy,sys; sys.path.insert(0, " + repr(str(target)) + "); "
                    "runpy.run_path(" + repr(str(Path(__file__).resolve())) + ")['probe']("
                    + repr(mode) + ")")
            subprocess.run([sys.executable, "-I", "-c", code], cwd=home,
                           env=child_env, check=True)

        target = build("unlicensed", features="python")
        print("BENCHMARK unlicensed", flush=True)
        run(target, "benchmark", None)
        target = build("licensed")
        run(target, "positive", license_content("2099-12-31"))
        license_file = home / ".dioptrin" / "license.key"
        license_file.parent.mkdir()
        license_file.write_text(license_content("2099-12-31"))
        try:
            run(target, "positive", None)
        finally:
            license_file.unlink()
        print("BENCHMARK licensed", flush=True)
        run(target, "benchmark", license_content("2099-12-31"))

        # Reproduce the *effect* of the old native zero-success patch without
        # architecture-specific offsets or changing the real repository binary.
        path = source / "src/license.rs"
        original = path.read_text()
        start = original.index("pub fn verify_license()")
        end = original.index("\n}\n", start) + 2
        stubbed = original[:start] + '''pub fn verify_license() -> Result<License, LicenseError> {
    Ok(License { email: String::new(), expires: String::new() })
}''' + original[end:]
        # Also disable the operation verifier in this fixture. Constructor
        # verification must remain independent of both successful return paths.
        begin = stubbed.index("    fn check_with_key_at(")
        end = stubbed.index("\n    }\n", begin) + len("\n    }")
        stubbed = stubbed[:begin] + """    fn check_with_key_at(
        &self, _key: &[u8; 32], _now: SystemTime,
    ) -> Result<(), LicenseError> { Ok(()) }""" + stubbed[end:]
        path.write_text(stubbed)
        target = build("diagnostic-and-operation-bypassed")
        for content in (None, "invalid", license_content("2099-12-31", valid=False)):
            run(target, "diagnostic", content)

        # Simulate skipping constructor checks while preserving well-formed
        # native objects, to test every downstream check deterministically.
        begin = original.index("        // Acquisition and operation admission")
        end = original.index("        Ok(grant)", begin)
        path.write_text(original[:begin] + original[end:])
        multi_path = source / "src/python/multi_integrator.rs"
        multi_source = multi_path.read_text()
        check = '''                integrator
                    .entitlement
                    .check()
                    .map_err(crate::python::license_error_to_py)?;
'''
        assert multi_source.count(check) == 1
        multi_path.write_text(multi_source.replace(check, ""))
        target = build("constructors-bypassed")
        run(target, "fault", license_content("2099-12-31", valid=False))
        run(target, "fault", license_content("2020-01-01"))
        print("All licensed hardening tests passed; temporary test-key wheels removed.")


if __name__ == "__main__":
    main()
