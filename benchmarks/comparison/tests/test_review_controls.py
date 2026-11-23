"""Regression checks for cache classification and ownership at the reader boundary."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docs/benchmarks/2026-11-18-cuda"))
from hdf5_batch import cache_observation
from pyfai_hdf5 import NativeReader


class ReviewControls(unittest.TestCase):
    def test_cache_labels_use_only_the_input_drive(self):
        storage = {"device": "PhysicalDrive0"}
        size = 10_000_000_000
        reads = {"PhysicalDrive0": size - 1, "PhysicalDrive1": size}
        self.assertEqual(cache_observation(storage, reads, "cold", size)["cache_observation"], "short-of-cold")
        self.assertEqual(cache_observation(storage, reads, "warm", size)["cache_observation"], "storage-traffic")
        self.assertEqual(cache_observation(storage, {"PhysicalDrive0": size}, "cold", size)["cache_observation"], "cold-observed")
        self.assertEqual(cache_observation(storage, {"PhysicalDrive1": size}, "cold", size)["cache_observation"], "unverified")

    def test_reader_close_is_idempotent_and_rejects_late_reads(self):
        library = MagicMock()
        library.eiger_open.return_value = 123
        with patch("pyfai_hdf5.ctypes.CDLL", return_value=library):
            reader = NativeReader("bridge", "scan.h5", "dataset")
        reader.close()
        reader.close()
        library.eiger_close.assert_called_once_with(123)
        with self.assertRaisesRegex(ValueError, "closed"):
            reader.read(0, np.zeros((3, 4), dtype=np.uint32))
        library.eiger_read.assert_not_called()
