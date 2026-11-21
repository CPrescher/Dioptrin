import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py

import watch_eiger as watch


class WatcherChecks(unittest.TestCase):
    def test_only_calibrated_row_orientation_is_accepted(self):
        calibration = Path(__file__).resolve().parents[1] / "2026-11-05-bslz4/calibration.poni"
        self.assertEqual(watch.read_geometry(calibration)["distance"], 0.1839103128035422)
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "other.poni"
            for config in ('{"orientation": 3}', '{}'):
                with self.subTest(config=config):
                    other.write_text(calibration.read_text().replace('{"orientation": 2}', config))
                    with self.assertRaisesRegex(ValueError, "orientation = 2"):
                        watch.read_geometry(other)

    def test_insufficient_scratch_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest, scratch = root/"scan.h5", root/"scan.npz", root/"scratch"
            scratch.mkdir()
            with h5py.File(source, "w") as f:
                f.create_dataset("data", shape=(2, 8, 8), dtype="u4")
            service = watch.PreparedReaders({"dataset": "data", "bins": 10})
            service.processes = [object()]
            with patch.object(watch.os, "posix_fallocate", side_effect=OSError(errno.ENOSPC, "full")) as reserve:
                with patch.object(service, "send") as send:
                    with self.assertRaises(OSError) as caught:
                        service.process(source, dest, True, scratch)
                send.assert_not_called()
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            self.assertEqual(reserve.call_args.args[1:], (0, 400))
            self.assertFalse(dest.exists())
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
