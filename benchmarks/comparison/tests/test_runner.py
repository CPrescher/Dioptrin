"""Regression checks for frame assignment, stale builds, and portable CI flags."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import VARIANTS, case_threads, contiguous_chunks, same_bits
from provenance import source_hashes, verify_azint_build, verify_build
from setup import azint_environment, compiler_environment


class RunnerTests(unittest.TestCase):
    def test_high_cpu_budgets_do_not_pin_each_worker_to_one_frame(self):
        frames = list(range(4)) * 8
        for workers in (4, 8):
            chunks = contiguous_chunks(frames, workers)
            self.assertEqual([i for chunk in chunks for i, _ in chunk], list(range(32)))
            self.assertEqual(len({len(chunk) for chunk in chunks}), 1)
            for chunk in chunks:
                self.assertEqual({frame for _, frame in chunk}, {0, 1, 2, 3})

    def test_stale_source_configuration_and_binary_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            files = {
                "src/lib.rs": "original source",
                "Cargo.toml": "cargo",
                "pyproject.toml": "python",
            }
            for name, value in files.items():
                (root / name).write_text(value)
            signature = {"module": "dioptrin.so", "sha256": {"dioptrin.so": "original"}}
            record = {
                "source": {"sha256": source_hashes(root)},
                "installed_dioptrin": signature,
            }
            with patch("provenance.installed_signature", return_value=signature):
                self.assertEqual(verify_build(root, record), [])
                for name, value in files.items():
                    (root / name).write_text("changed")
                    self.assertIn(
                        "source/build configuration", verify_build(root, record)[0]
                    )
                    (root / name).write_text(value)
            with patch("provenance.installed_signature", return_value={}):
                self.assertIn("recorded binary", verify_build(root, record)[0])
            self.assertIn("No Dioptrin build record", verify_build(root, None)[0])

    def test_ci_portable_mode_does_not_add_native_cpu_flags(self):
        with patch.dict(os.environ, {"CC": "gcc", "CXX": "g++"}, clear=True):
            portable = compiler_environment(portable=True)
            native = compiler_environment()
        for key in ("CFLAGS", "CXXFLAGS", "RUSTFLAGS"):
            self.assertNotIn("native", portable[key])
            self.assertIn("native", native[key])

    def test_default_variants_measure_both_azint_rows(self):
        self.assertIn("azint", VARIANTS)
        self.assertEqual(VARIANTS[VARIANTS.index("azint") + 1], "azint-cached")

    def test_thread_budget_per_library(self):
        for count in (1, 4):
            for variant in ("azint", "azint-cached"):
                threads = case_threads(variant, count)
                self.assertEqual(threads["OMP_NUM_THREADS"], str(count))
                self.assertEqual(threads["RAYON_NUM_THREADS"], "1")
            for variant in ("pyfai-full", "pyfai-bbox"):
                threads = case_threads(variant, count)
                self.assertEqual(threads["OMP_NUM_THREADS"], "1")
                self.assertEqual(threads["RAYON_NUM_THREADS"], "1")
            threads = case_threads("dioptrin-pixel_split", count)
            self.assertEqual(threads["OMP_NUM_THREADS"], "1")
            self.assertEqual(threads["RAYON_NUM_THREADS"], str(count))
            for variant in VARIANTS:
                threads = case_threads(variant, count)
                for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    self.assertEqual(threads[key], "1")

    def test_azint_fast_math_flags(self):
        base = {"CC": "gcc", "CXX": "g++", "CFLAGS": " -O3", "CXXFLAGS": " -O3"}
        env, record = azint_environment(base, probe=lambda env, flag: True)
        self.assertEqual(env["CXXFLAGS"], " -O3")
        self.assertEqual(record["variant"], "default")
        self.assertFalse(record["fast_math"])
        self.assertIsNone(record["lto"])

        env, record = azint_environment(base, True, probe=lambda env, flag: False)
        self.assertEqual(env["CXXFLAGS"].split(), ["-O3", "-ffast-math"])
        self.assertEqual(env["CFLAGS"].split(), ["-O3", "-ffast-math"])
        self.assertNotIn("lto", env.get("LDFLAGS", ""))
        self.assertTrue(record["lto"].startswith("omitted"))
        self.assertEqual(record["flags"]["CXXFLAGS"], env["CXXFLAGS"])

        env, record = azint_environment(base, True, probe=lambda env, flag: True)
        self.assertEqual(env["CXXFLAGS"].split(), ["-O3", "-ffast-math", "-flto=thin"])
        self.assertIn("-flto=thin", env["LDFLAGS"])
        self.assertEqual(record["variant"], "fast-math")
        self.assertEqual(base["CXXFLAGS"], " -O3")  # caller's env is untouched

    def test_azint_extension_must_match_record(self):
        signature = {"module": "_azint.so", "sha256": "a"}
        record = {"azint_build": {"installed_extension": signature}}
        self.assertEqual(verify_azint_build(record, signature), [])
        self.assertIn(
            "azint extension",
            verify_azint_build(record, {"module": "_azint.so", "sha256": "b"})[0],
        )
        # Records written before azint provenance existed are not rejected.
        self.assertEqual(verify_azint_build({"flags": {}}, signature), [])
        self.assertEqual(verify_azint_build(None, signature), [])

    def test_bit_identity_helper(self):
        self.assertTrue(same_bits([1.0, 2.0], [1.0, 2.0]))
        self.assertFalse(same_bits([1.0, 2.0], [1.0, 2.0 + 1e-15]))
        self.assertTrue(same_bits(None, None))
        self.assertFalse(same_bits(None, [1.0]))


if __name__ == "__main__":
    unittest.main()
