"""2D (cake) cases: azimuth mapping, convention check, metrics, memory gate."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import (
    AZIMUTH_RANGE_DEG,
    CONVENTION_BANDS,
    Unsupported,
    azint_azimuth_axis,
    cake_metrics,
    check_convention,
    convention_rows,
    estimate_memory_mib,
    parse_cake,
    roll_azint,
)

HERE = Path(__file__).resolve().parents[1]


def band_cake(n_azimuthal, n_radial=5):
    """Ideal pattern/flat ratio cake on the harness rows [-180, 180)."""
    step = 360.0 / n_azimuthal
    centres = AZIMUTH_RANGE_DEG[0] + step * (np.arange(n_azimuthal) + 0.5)
    rows = np.ones(n_azimuthal)
    for low, high, value in CONVENTION_BANDS:
        rows[(centres > low) & (centres < high)] = value
    return np.repeat(rows[:, None], n_radial, axis=1)


class CakeTests(unittest.TestCase):
    def test_parse_cake(self):
        self.assertEqual(parse_cake("2000x360"), (2000, 360))
        self.assertEqual(parse_cake("1000X72"), (1000, 72))
        for bad in ("2000", "0x10", "ax3"):
            with self.assertRaises(ValueError):
                parse_cake(bad)

    def test_azint_rows_map_onto_minus_180_to_180(self):
        n = 8
        # azint row j covers [45 j, 45 (j + 1)); store its centre in the row.
        azint_centres = 45.0 * (np.arange(n) + 0.5)
        cake = np.repeat(azint_centres[:, None], 3, axis=1)
        rolled = roll_azint(cake, n)
        wrapped = np.where(rolled >= 180, rolled - 360, rolled)
        harness_centres = -180 + 45.0 * (np.arange(n) + 0.5)
        np.testing.assert_allclose(wrapped[:, 0], harness_centres)
        np.testing.assert_allclose(
            azint_azimuth_axis(azint_centres, n), harness_centres
        )
        with self.assertRaises(Unsupported):
            roll_azint(cake[:7], 7)

    def test_convention_check_rejects_flips_and_rotations(self):
        n = 36
        bright, medium, dark = convention_rows(n)
        self.assertEqual(bright, list(range(20, 26)))  # 20..80 degrees
        self.assertEqual(medium, [3, 4, 5])  # -150..-120 degrees
        good = band_cake(n)
        flat = np.ones_like(good)
        candidates = {
            "same": good,
            "mirrored": good[::-1],  # chi -> -chi
            "rotated": np.roll(good, n // 2, axis=0),  # chi -> chi + 180
            "quarter": np.roll(good, n // 4, axis=0),
        }
        report = check_convention(candidates, dict.fromkeys(candidates, flat), n)
        self.assertTrue(report["same"]["passed"])
        for name in ("mirrored", "rotated", "quarter"):
            self.assertFalse(report[name]["passed"], name)

    def test_cake_metrics_separate_the_seam_rows(self):
        reference = np.full((6, 4), 10.0)
        reference[2, 0] = 0.0  # empty in the reference only
        variant = reference.copy()
        variant[2, 0] = 5.0
        variant[0, 1] = 11.0  # seam row, +10 %
        variant[3, 2] = 10.05  # interior, +0.5 %
        metrics = cake_metrics(variant, reference)
        self.assertEqual(metrics["populated_in_both"], 23)
        self.assertEqual(metrics["populated_only_in_variant"], 1)
        self.assertEqual(metrics["populated_only_in_reference"], 0)
        self.assertEqual(metrics["all"]["bins_over_1pct"], 1)
        self.assertEqual(metrics["seam_rows"]["bins"], 8)
        self.assertEqual(metrics["seam_rows"]["bins_over_1pct"], 1)
        self.assertEqual(metrics["interior"]["bins_over_1pct"], 0)
        self.assertAlmostEqual(metrics["interior"]["max_relative_difference"], 0.005)

    def test_memory_estimate_scales_with_engines_and_processes(self):
        meta = dict(shape=[1000, 1000], distinct_frames=4)
        estimates = {
            "100x10": dict(polygon_nnz=3e6, subpixel_nnz={"10": 2e6}),
        }
        case = dict(
            variant="pyfai-full",
            workers=1,
            processes=1,
            workload="frames",
            frames=8,
            n_radial=100,
            n_azimuthal=10,
            errors=False,
            cake="100x10",
        )
        one = estimate_memory_mib(case, meta, estimates)
        four = estimate_memory_mib(dict(case, workers=4), meta, estimates)
        self.assertEqual(one["estimated_engines"], 1)
        self.assertEqual(four["estimated_engines"], 4)  # one CSR engine per thread
        self.assertGreater(four["estimated_memory_mib"], one["estimated_memory_mib"])
        shared = estimate_memory_mib(
            dict(case, variant="dioptrin-pixel_split", workers=4), meta, estimates
        )
        self.assertEqual(shared["estimated_engines"], 1)
        split = estimate_memory_mib(
            dict(case, variant="dioptrin-pixel_split", workers=4, processes=2),
            meta,
            estimates,
        )
        self.assertEqual(split["estimated_engines"], 2)
        sub = estimate_memory_mib(
            dict(case, variant="azint", subpixels=10), meta, estimates
        )
        self.assertEqual(sub["estimated_table_nnz"], 2_000_000)


class CakeRunTests(unittest.TestCase):
    def run_compare(self, output, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(HERE / "compare.py"),
                "--suite",
                "quick",
                "--workloads",
                "frames",
                "--dimensions",
                "2d",
                "--frames",
                "4",
                "--seconds",
                "0.01",
                "--offline",
                "--allow-unverified-build",
                "--output",
                str(output),
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_cakes_end_to_end(self):
        """Polygon clipping, pyFAI full and azint cakes, checked and compared."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.run_compare(
                output,
                "--variants",
                "dioptrin-pixel_split",
                "dioptrin-supersampled",
                "pyfai-full",
                "azint",
                "--cakes",
                "32x8",
                "32x7",
                "--subpixels",
                "3",
                "--workers",
                "1,2",
            )
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(r["dimension"] == "2d" for r in rows))
            self.assertTrue(all(r["self_check"] == "passed" for r in rows))
            self.assertEqual(
                {r["subpixels"] for r in rows if r["variant"] == "azint"}, {"3"}
            )
            self.assertTrue(
                any(r["table_nnz"] for r in rows if r["variant"] == "pyfai-full")
            )
            # azint cannot place odd azimuthal counts on [-180, 180): skipped, not faked.
            skipped = json.loads((output / "skipped.json").read_text())
            odd = [s for s in skipped if "32x7-azint" in s.get("case", "")]
            self.assertTrue(odd and all("even n_azimuthal" in s["reason"] for s in odd))
            self.assertIn(
                "32x7", {r["n_radial"] + "x" + r["n_azimuthal"] for r in rows}
            )
            preflight = json.loads(
                (output / "synthetic-small" / "preflight2d.json").read_text()
            )
            self.assertTrue(preflight["convention"]["passed"])
            numerical = json.loads((output / "numerical.json").read_text())
            pyfai = next(
                e
                for e in numerical
                if e["variant"] == "pyfai-full"
                and e["cake"] == "32x8"
                and e["reference"] == "dioptrin-pixel_split"
                and e["error_mode"] == "poisson"
            )
            self.assertLess(pyfai["max_radial_axis_difference_deg"], 1e-4)
            self.assertIn("seam_rows", pyfai["intensity"])
            self.assertIn("sigma", pyfai)
            # Supersampled cakes have their own radial grid; the common-grid
            # companion (MultiIntegrator) lines up with everyone else.
            grids = {
                e["grid"]: e
                for e in numerical
                if e["variant"] == "azint"
                and e["reference"] == "dioptrin-supersampled"
                and e["cake"] == "32x8"
                and e["error_mode"] == "intensity"
            }
            self.assertEqual(set(grids), {"native", "common"})
            self.assertLess(grids["common"]["max_radial_axis_difference_deg"], 1e-9)
            self.assertGreater(grids["native"]["max_radial_axis_difference_deg"], 1e-6)
            self.assertEqual(
                preflight["multiintegrator_check"]["max_relative_difference"], 0.0
            )
            self.assertIn("## 2D cakes", (output / "summary.md").read_text())
            # Full reference stacks are removed; first cakes stay for numerical.json.
            self.assertFalse(list((output / "cases").glob("*.result.npz")))
            self.assertTrue(list((output / "cases").glob("*.cake0.npz")))

    def test_memory_gate_skips_instead_of_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.run_compare(
                output,
                "--variants",
                "pyfai-bbox",
                "--cakes",
                "32x8",
                "--workers",
                "1",
                "--max-memory-fraction",
                "0.000001",
            )
            skipped = json.loads((output / "skipped.json").read_text())
            self.assertEqual(len(skipped), 2)  # intensity and Poisson
            self.assertTrue(
                all("estimated peak memory" in s["reason"] for s in skipped)
            )
            self.assertFalse(list((output / "cases").glob("*.result.json")))


if __name__ == "__main__":
    unittest.main()
