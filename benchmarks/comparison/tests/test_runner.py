"""Regression checks for frame assignment, stale builds, and portable CI flags."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import contiguous_chunks
from provenance import source_hashes, verify_build
from setup import compiler_environment


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


if __name__ == "__main__":
    unittest.main()
