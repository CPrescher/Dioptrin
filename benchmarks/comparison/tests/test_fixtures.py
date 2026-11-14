"""Input reproducibility, download integrity, and offline operation."""

import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixtures


class FixtureTests(unittest.TestCase):
    def test_series_reproducible_distinct_and_masked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = fixtures.prepare(
                root / "a", "synthetic-small", root, root, offline=True
            )
            b = fixtures.prepare(
                root / "b", "synthetic-small", root, root, offline=True
            )
            self.assertEqual(a["frame_sha256"], b["frame_sha256"])
            self.assertEqual(len(set(a["frame_sha256"])), a["distinct_frames"])
            frames = np.load(root / "a" / "frames.npy")
            mask = np.load(root / "a" / "mask.npy")
            self.assertTrue(np.all(frames[:, mask != 0] == 0))
            self.assertTrue(np.all(np.isfinite(frames)))
            self.assertTrue(np.all(frames >= 0))
            self.assertGreater(np.std(frames[0][mask == 0]), 0)
            self.assertGreater(a["masked_pixels"], 0)

    def test_lfs_pointer_downloads_once_and_then_works_offline(self):
        payload = b"verified public image bytes"
        spec = dict(
            image="image.tif",
            sha256=hashlib.sha256(payload).hexdigest(),
            download="https://example.invalid/image",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.tif").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
            )
            with patch(
                "urllib.request.urlopen", return_value=io.BytesIO(payload)
            ) as fetch:
                path = fixtures.cached_image(spec, root, root / "cache")
                self.assertEqual(path.read_bytes(), payload)
                fetch.assert_called_once()
            with patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("unexpected download"),
            ):
                self.assertEqual(
                    fixtures.cached_image(spec, root, root / "cache", offline=True),
                    path,
                )

    def test_corrupt_download_is_not_cached(self):
        spec = dict(
            image="image.tif", sha256="0" * 64, download="https://example.invalid/image"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    fixtures.cached_image(spec, root, root / "cache")
            self.assertEqual(list((root / "cache").iterdir()), [])
            with self.assertRaisesRegex(FileNotFoundError, "offline"):
                fixtures.cached_image(spec, root, root / "cache", offline=True)


if __name__ == "__main__":
    unittest.main()
