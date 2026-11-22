"""Archivable subsets, their dataset kind, beam-pixel masks, and process mode."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixtures
import prepare_subset
from compare import (
    case_threads,
    opencl_target,
    process_share,
    skipped_variants,
    binding_commands,
)

try:
    import h5py
    import hdf5plugin
except ImportError:  # pragma: no cover - only without the benchmark environment
    h5py = None

HERE = Path(__file__).resolve().parents[1]
DATASET = "entry_0000/instrument/eiger/data"
SHAPE = (24, 32)


def make_scan(path, frames=10):
    """Small Bitshuffle/LZ4 scan, one frame per chunk, with gaps and sentinels."""
    prepare_subset.register_filters()
    rng = np.random.default_rng(3)
    data = rng.poisson(50, size=(frames, *SHAPE)).astype(np.uint32)
    data[:, 10:12, :] = 0  # module gap
    data[:, 0, 0] = 2**32 - 1  # invalid in every frame (top-left)
    data[5, 5, 7] = 2**32 - 1  # invalid in one frame, never selected below
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            DATASET,
            data=data,
            chunks=(1, *SHAPE),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        dataset.attrs["interpretation"] = "image"
    return data


def write_poni(path, orientation=2):
    path.write_text(
        "poni_version: 2.1\n"
        "Detector: Detector\n"
        f'Detector_config: {{"pixel1": 0.0001, "pixel2": 0.0001, "orientation": {orientation}, '
        f'"max_shape": [{SHAPE[0]}, {SHAPE[1]}]}}\n'
        "Distance: 0.05\nPoni1: 0.0012\nPoni2: 0.0016\n"
        "Rot1: 0.01\nRot2: -0.02\nRot3: 0.0\nWavelength: 1e-10\n"
    )


@unittest.skipIf(h5py is None, "h5py/hdf5plugin are not installed")
class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scan = self.root / "scan.h5"
        self.data = make_scan(self.scan)
        self.poni = self.root / "scan.poni"
        write_poni(self.poni)
        # A static mask stored upside down relative to the HDF5 frames.
        mask = np.zeros(SHAPE, dtype=bool)
        mask[10:12, :] = True
        mask[0, 0] = True
        mask[2, 30] = True
        self.mask_in_frames = mask
        self.mask_file = self.root / "scan.mask.tif"
        prepare_subset.write_mask(self.mask_file, mask[::-1])

    def tearDown(self):
        self.temporary.cleanup()

    def subset(self, out, *extra, frames=4):
        return prepare_subset.main(
            [
                "subset",
                "--scan",
                str(self.scan),
                "--dataset",
                DATASET,
                "--frames",
                str(frames),
                "--name",
                "tiny",
                "--out-dir",
                str(out),
                "--poni",
                str(self.poni),
                *extra,
            ]
        )

    def test_subset_copies_compressed_chunks_and_records_provenance(self):
        bundle = self.subset(self.root / "a", "--mask", str(self.mask_file))
        self.assertEqual(bundle["frame_indices"], [0, 3, 6, 9])
        self.assertEqual(bundle["source_sha256"], prepare_subset.sha256(self.scan))
        self.assertEqual(bundle["dataset_attributes"], {"interpretation": "image"})
        self.assertEqual(bundle["mask"]["transform_to_frame_orientation"], "flipud")
        subset = self.root / "a" / bundle["subset"]
        with h5py.File(self.scan) as source, h5py.File(subset) as copy:
            for position, index in enumerate(bundle["frame_indices"]):
                self.assertEqual(
                    copy[DATASET].id.read_direct_chunk((position, 0, 0)),
                    source[DATASET].id.read_direct_chunk((index, 0, 0)),
                )
                np.testing.assert_array_equal(copy[DATASET][position], self.data[index])
            self.assertEqual(copy[DATASET].attrs["interpretation"], "image")
        mask = prepare_subset.read_mask(self.root / "a" / bundle["mask"]["file"])
        np.testing.assert_array_equal(mask, self.mask_in_frames)
        # No timestamps in the HDF5 file: the same inputs give the same bytes.
        again = self.subset(self.root / "b", "--mask", str(self.mask_file))
        self.assertEqual(again["subset_sha256"], bundle["subset_sha256"])
        prepare_subset.main(
            ["verify", "--subset", str(subset), "--scan", str(self.scan)]
        )

    def test_generated_mask_and_tiling(self):
        bundle = self.subset(self.root / "a", frames=5)
        mask = prepare_subset.read_mask(self.root / "a" / bundle["mask"]["file"])
        self.assertEqual(bundle["mask"]["origin"], "generated")
        self.assertTrue(mask[10:12].all() and mask[0, 0])
        self.assertFalse(mask[5, 7])  # frame 5 is not among indices [0, 2, 4, 7, 9]
        subset = self.root / "a" / bundle["subset"]
        tiled = self.root / "tiled.h5"
        result = prepare_subset.main(
            ["tile", "--subset", str(subset), "--frames", "12", "--output", str(tiled)]
        )
        self.assertEqual(result["frames"], 12)
        with h5py.File(tiled) as out, h5py.File(subset) as source:
            self.assertEqual(out[DATASET].shape, (12, *SHAPE))
            for index in range(12):
                self.assertEqual(
                    out[DATASET].id.read_direct_chunk((index, 0, 0)),
                    source[DATASET].id.read_direct_chunk((index % 5, 0, 0)),
                )
            np.testing.assert_array_equal(
                out["benchmark_subset/source_frame_indices"][:],
                np.asarray(bundle["frame_indices"])[np.arange(12) % 5],
            )

    def test_ambiguous_mask_orientation_is_refused(self):
        invalid = np.zeros(SHAPE, dtype=bool)
        invalid[0, 0] = True
        mask = np.zeros(SHAPE, dtype=bool)
        mask[0, 0] = mask[-1, 0] = True  # covers it as-is and flipped
        self.assertEqual(
            prepare_subset.choose_transform(mask, invalid, "auto")[0], "none"
        )
        mask[0, 0] = False
        mask[-1, -1] = True  # flipud and rot180 both cover, identity does not
        with self.assertRaisesRegex(ValueError, "Cannot determine mask orientation"):
            prepare_subset.choose_transform(mask, invalid, "auto")
        self.assertEqual(prepare_subset.even_indices(625, 32)[-1], 624)

    def test_subset_dataset_kind_flips_masks_and_zeroes(self):
        bundle = self.subset(self.root / "bundle", "--mask", str(self.mask_file))
        spec = dict(
            kind="subset",
            subset=bundle["subset"],
            poni="tiny.poni",
            mask="tiny.mask.tif",
            n_splitting=3,
            source_sha256=bundle["source_sha256"],
        )
        definition = dict(generator_version=1, datasets={"tiny-subset": spec})
        with patch("fixtures.manifest", return_value=definition):
            meta = fixtures.prepare(
                self.root / "fixture",
                "tiny-subset",
                self.root / "unused",
                self.root,
                subset_dir=self.root / "bundle",
            )
        frames = np.load(self.root / "fixture/frames.npy")
        mask = np.load(self.root / "fixture/mask.npy")
        self.assertEqual(frames.shape, (4, *SHAPE))
        self.assertEqual(meta["original_orientation"], 2)
        # Orientation 2 flips rows.
        expected_mask = self.mask_in_frames[::-1]
        np.testing.assert_array_equal(mask != 0, expected_mask)
        for position, index in enumerate(bundle["frame_indices"]):
            expected = self.data[index][::-1].astype(np.float64)
            expected[expected_mask] = 0
            np.testing.assert_array_equal(frames[position], expected)
        self.assertEqual(meta["subset"]["source_frame_indices"], [0, 3, 6, 9])
        self.assertIsNone(meta["beam_pixel_mask"])

        spec["source_sha256"] = "0" * 64
        with patch("fixtures.manifest", return_value=definition):
            with self.assertRaisesRegex(ValueError, "not cut from the scan"):
                fixtures.prepare(
                    self.root / "fixture2",
                    "tiny-subset",
                    self.root / "bundle",
                    self.root,
                )


class BeamPixelTests(unittest.TestCase):
    def integrator(self, poni1, poni2, rot1=0.0, rot2=0.0):
        from pyFAI.detectors import Detector
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator

        return AzimuthalIntegrator(
            dist=0.05,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            wavelength=1e-10,
            detector=Detector(1e-4, 1e-4, max_shape=SHAPE, orientation=3),
        )

    def test_pixels_containing_or_touching_the_point(self):
        inside = self.integrator(12.4e-4, 16.7e-4)
        self.assertEqual(fixtures.beam_pixels(inside, SHAPE, "poni")[1], [(12, 16)])
        corner = self.integrator(12e-4, 16e-4)
        self.assertEqual(
            fixtures.beam_pixels(corner, SHAPE, "poni")[1],
            [(11, 15), (11, 16), (12, 15), (12, 16)],
        )

    def test_direct_beam_pixel_is_the_minimum_two_theta_pixel(self):
        ai = self.integrator(12.4e-4, 16.7e-4, rot1=0.02, rot2=-0.015)
        _, poni_pixels = fixtures.beam_pixels(ai, SHAPE, "poni")
        _, direct = fixtures.beam_pixels(ai, SHAPE, "direct")
        tth = ai.center_array(SHAPE, unit="2th_rad")
        minimum = tuple(int(v) for v in np.unravel_index(np.argmin(tth), SHAPE))
        self.assertEqual(direct, [minimum])
        self.assertNotEqual(poni_pixels, direct)

    def test_mask_beam_pixels_updates_mask_and_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "frames.npy", np.ones((2, *SHAPE)))
            mask = np.zeros(SHAPE, dtype=np.uint8)
            record = fixtures.mask_beam_pixels(
                directory, self.integrator(12.4e-4, 16.7e-4), mask, "poni"
            )
            frames = np.load(directory / "frames.npy")
        self.assertEqual(record["pixels"], [[12, 16]])
        self.assertEqual(mask[12, 16], 1)
        self.assertTrue(np.all(frames[:, 12, 16] == 0))
        self.assertEqual(frames.sum(), 2 * (SHAPE[0] * SHAPE[1] - 1))


class ProcessModeTests(unittest.TestCase):
    def test_threads_are_per_process_and_openmp_pyfai_gets_the_budget(self):
        self.assertEqual(case_threads("pyfai-full-omp", 8)["OMP_NUM_THREADS"], "8")
        self.assertEqual(case_threads("pyfai-full-omp", 8, 2)["OMP_NUM_THREADS"], "4")
        self.assertEqual(case_threads("pyfai-full", 8, 2)["OMP_NUM_THREADS"], "1")
        self.assertEqual(case_threads("pyfai-full-opencl", 8)["OMP_NUM_THREADS"], "1")
        self.assertEqual(
            case_threads("dioptrin-pixel_split", 8, 4)["RAYON_NUM_THREADS"], "2"
        )
        self.assertEqual(case_threads("azint", 8, 2)["OMP_NUM_THREADS"], "4")
        with self.assertRaises(ValueError):
            case_threads("azint", 6, 4)

    def test_process_shares_are_contiguous_and_complete(self):
        shares = [process_share(32, 4, rank) for rank in range(4)]
        self.assertEqual(sum(shares, []), list(range(32)))
        self.assertEqual(process_share(8, 1, None), list(range(8)))
        with self.assertRaises(ValueError):
            process_share(10, 4, 0)

    def test_options_and_skips(self):
        self.assertEqual(opencl_target("gpu"), {"target_type": "gpu"})
        self.assertEqual(opencl_target("1:0"), {"target": (1, 0)})
        self.assertEqual(opencl_target("any"), {})
        with self.assertRaises(ValueError):
            opencl_target("nvidia")
        self.assertEqual(binding_commands("none", 3), [[], [], []])
        probes = dict(
            pyfai_openmp=dict(openmp=False, module="x.so"),
            opencl=dict(available=False, reason="pyopencl not importable"),
        )
        skipped = skipped_variants(["pyfai-full-omp", "pyfai-full-opencl"], probes)
        self.assertEqual(
            [s["variant"] for s in skipped], ["pyfai-full-omp", "pyfai-full-opencl"]
        )
        probes = dict(pyfai_openmp=dict(openmp=True), opencl=dict(available=True))
        self.assertEqual(skipped_variants(["pyfai-full-omp"], probes), [])

    def test_two_process_run_with_beam_pixel_mask(self):
        """End to end: pyFAI rows in one and two processes, direct beam masked."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "compare.py"),
                    "--suite",
                    "quick",
                    "--workloads",
                    "frames",
                    "--variants",
                    "pyfai-bbox",
                    "--workers",
                    "2",
                    "--processes",
                    "2",
                    "--frames",
                    "4",
                    "--seconds",
                    "0.01",
                    "--mask-beam-pixel",
                    "direct",
                    "--offline",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                sorted((r["workers"], r["processes"]) for r in rows),
                [("1", "1")] * 2 + [("2", "1")] * 2 + [("2", "2")] * 2,
            )
            self.assertTrue(all(r["self_check"] == "passed" for r in rows))
            self.assertTrue(all(r["mask_beam_pixel"] == "direct" for r in rows))
            split = json.loads(
                next(
                    (output / "cases").glob("*-w2-p2-poisson-t0.result.json")
                ).read_text()
            )
            self.assertEqual(split["passes_per_block"] * 4, split["block_frames"][0])
            self.assertEqual(len(split["rank_results"]), 2)
            meta = json.loads((output / "synthetic-small/fixture.json").read_text())
            self.assertEqual(meta["beam_pixel_mask"]["mode"], "direct")
            self.assertIn("Processes", (output / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
