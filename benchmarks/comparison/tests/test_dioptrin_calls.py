"""CPU Dioptrin 1D call (--dioptrin-1d-path) and result conversion in timing."""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare
from compare import (
    DIOPTRIN_1D_PATHS,
    DioptrinOutput,
    dioptrin_converter,
    dioptrin_has_result_arrays,
    dioptrin_result_conversion,
    setup_case,
    setup_case_2d,
    summarize,
)

HERE = Path(__file__).resolve().parents[1]


class ListResult:
    """IntegrationResult of a build without the accessors (lists, as 0.15.2)."""

    def __init__(self, radial, intensity, errors):
        self.radial, self.intensity, self.errors = radial, intensity, errors
        self.reads = 0

    def __getattribute__(self, name):
        if name in ("intensity", "errors"):
            object.__getattribute__(self, "__dict__")["reads"] += 1
        return object.__getattribute__(self, name)


class ArrayResult:
    """IntegrationResult with the zero-copy accessors; list reads fail."""

    def __init__(self, radial, intensity, errors):
        self.radial_array = np.asarray(radial, dtype=np.float64)
        self.intensity_array = np.asarray(intensity, dtype=np.float64)
        self.errors_array = None if errors is None else np.asarray(errors, dtype=np.float64)
        self.azimuthal_array = None
        self.azimuthal_std_array = None

    def __getattr__(self, name):
        raise AssertionError(f"list attribute {name} read")


class Item:
    """BatchResult stand-in; counts reads of .result (a clone in 0.15.2)."""

    def __init__(self, index, result):
        self.filename = f"array[{index}]"
        self._result = result
        self.reads = 0

    def is_ok(self):
        return True

    @property
    def result(self):
        self.reads += 1
        return self._result


class FakeIntegrator:
    calls = []
    result_type = ListResult

    def __init__(self, poni, method, unit, n_ss, polarization_factor, radial_range):
        self.radial_range = radial_range
        self.mask = None

    def set_mask(self, mask):
        self.mask = np.asarray(mask)

    def batch1d(self, images, n, errors=False, **kwargs):
        FakeIntegrator.calls.append(kwargs)
        items = []
        for index, image in enumerate(images):
            values = np.bincount(np.arange(image.size) % n, image.ravel(), n)
            items.append(
                Item(
                    index,
                    FakeIntegrator.result_type(
                        list(np.linspace(*self.radial_range, n)),
                        list(values),
                        list(np.sqrt(values)) if errors else None,
                    ),
                )
            )
        FakeIntegrator.items = items
        return items


def write_fixture(directory, frames=4, shape=(6, 5)):
    rng = np.random.default_rng(1)
    np.save(directory / "frames.npy", rng.uniform(1.0, 100.0, size=(frames, *shape)))
    np.save(directory / "mask.npy", np.zeros(shape, dtype=np.uint8))
    (directory / "fixture.json").write_text(
        json.dumps(dict(dataset="toy", radial_range=[0.5, 20.0], n_splitting=1))
    )
    (directory / "geometry.poni").write_text("fake")


class DioptrinCallTests(unittest.TestCase):
    def setUp(self):
        FakeIntegrator.calls = []
        FakeIntegrator.result_type = ListResult
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = Path(self.directory.name)
        write_fixture(self.fixture)

    def tearDown(self):
        self.directory.cleanup()

    def setup(self, path, errors=True, workers=1, allow_list=True, rayon=None):
        case = dict(
            fixture=str(self.fixture),
            workload="frames",
            frames=4,
            bins=7,
            workers=workers,
            processes=1,
            errors=errors,
            variant="dioptrin-pixel_split",
            dioptrin_1d_path=path,
            allow_list_conversion=allow_list,
        )
        module = SimpleNamespace(
            Integrator=FakeIntegrator,
            rayon_num_threads=lambda: workers if rayon is None else rayon,
        )
        with patch.dict(sys.modules, {"dioptrin": module}):
            return setup_case(case)

    def test_global_is_the_default_call_and_split_passes_the_budget(self):
        self.assertEqual(DIOPTRIN_1D_PATHS, ("global", "split"))
        for path, expected in (
            ("global", {}),
            (None, {}),  # a case without the field uses the default
            ("split", dict(num_workers=3, num_threads=1)),
        ):
            with self.subTest(path=path):
                FakeIntegrator.calls = []
                state = self.setup(path, workers=3)
                self.assertTrue(FakeIntegrator.calls)
                self.assertTrue(all(c == expected for c in FakeIntegrator.calls))
                FakeIntegrator.calls = []
                state["run"]()
                self.assertEqual(FakeIntegrator.calls, [expected])
        with self.assertRaisesRegex(ValueError, "Unknown Dioptrin 1D path"):
            self.setup("nested")

    def test_list_conversion_happens_inside_the_timed_call(self):
        for errors in (False, True):
            with self.subTest(errors=errors):
                state = self.setup("global", errors=errors)
                self.assertEqual(state["info"]["dioptrin_result_conversion"], "list")
                results = state["run"]()
                items = FakeIntegrator.items
                # Every BatchResult unwrapped once, every list read once, in run().
                self.assertEqual([i.reads for i in items], [1] * 4)
                self.assertEqual(
                    [i._result.reads for i in items], [2 if errors else 1] * 4
                )
                for output in results:
                    self.assertIsInstance(output, DioptrinOutput)
                    self.assertIsInstance(output.intensity, np.ndarray)
                    self.assertEqual(output.intensity.dtype, np.float64)
                    self.assertEqual(output.intensity.shape, (7,))
                    if errors:
                        self.assertEqual(output.sigma.dtype, np.float64)
                    else:
                        self.assertIsNone(output.sigma)
                state["validate"](results)
                # Validation reuses the arrays: no second unwrap or conversion.
                self.assertEqual([i.reads for i in items], [1] * 4)
                self.assertEqual(
                    [i._result.reads for i in items], [2 if errors else 1] * 4
                )

    def test_array_accessors_are_used_without_copies(self):
        FakeIntegrator.result_type = ArrayResult
        state = self.setup("global")
        self.assertEqual(state["info"]["dioptrin_result_conversion"], "array")
        results = state["run"]()
        for item, output in zip(FakeIntegrator.items, results):
            self.assertIs(output.intensity, item._result.intensity_array)
            self.assertIs(output.sigma, item._result.errors_array)
        state["validate"](results)

    def test_list_conversion_is_refused_unless_allowed(self):
        for path in ("global", "split"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(RuntimeError, "--allow-list-conversion"):
                    self.setup(path, allow_list=False)
        FakeIntegrator.result_type = ArrayResult
        state = self.setup("global", allow_list=False)
        self.assertEqual(state["info"]["dioptrin_result_conversion"], "array")

    def test_rayon_pool_must_match_the_budget(self):
        # The only enforcement of the thread count on the global path.
        FakeIntegrator.result_type = ArrayResult
        for path in ("global", "split"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(RuntimeError, "Rayon pool"):
                    self.setup(path, workers=2, rayon=4)
        self.setup("global", workers=2, rayon=2)

    def test_rayon_pool_must_match_the_budget_2d(self):
        frame = np.ones((2, 2))
        inputs = (
            self.fixture, {"n_splitting": 1}, np.zeros_like(frame), {0: frame},
            [0], [frame], [0], 1,
        )
        case = dict(
            n_radial=2,
            n_azimuthal=1,
            workers=4,
            processes=2,  # 2 threads per process
            errors=False,
            variant="dioptrin-pixel_split",
            radial_range_2d=(0.5, 2.5),
            dioptrin_radial_axis=[1.0, 2.0],
        )
        for threads in (4, 1):
            with self.subTest(threads=threads):
                module = SimpleNamespace(
                    Integrator=Mock(), rayon_num_threads=lambda: threads
                )
                with patch.dict(sys.modules, {"dioptrin": module}), patch(
                    "compare.load_inputs", return_value=inputs
                ), self.assertRaisesRegex(RuntimeError, "Rayon pool"):
                    setup_case_2d(case, rank=0)
        # With the matching pool the check passes (the Mock engine then
        # fails further on, for another reason).
        module = SimpleNamespace(Integrator=Mock(), rayon_num_threads=lambda: 2)
        with patch.dict(sys.modules, {"dioptrin": module}), patch(
            "compare.load_inputs", return_value=inputs
        ), self.assertRaises(Exception) as caught:
            setup_case_2d(case, rank=0)
        self.assertNotIn("Rayon pool", str(caught.exception))

    def test_validation_rejects_wrong_order_dtype_and_missing_sigma(self):
        state = self.setup("global")
        results = state["run"]()
        results[0].filename = "array[2]"
        with self.assertRaisesRegex(AssertionError, "array\\[2\\]"):
            state["validate"](results)
        results = state["run"]()
        results[1].intensity = results[1].intensity.astype(np.float32)
        with self.assertRaisesRegex(AssertionError, "float64"):
            state["validate"](results)
        results = state["run"]()
        results[2].sigma = None
        with self.assertRaisesRegex(AssertionError, "Poisson"):
            state["validate"](results)
        # The unmodified outputs still pass against the same reference.
        state["validate"](state["run"]())

    def test_converter_contract(self):
        flat = ListResult([0.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 6)
        output = dioptrin_converter("list", True, (2, 3))(flat, "array[0]")
        self.assertEqual(output.intensity.shape, (2, 3))
        self.assertEqual(output.sigma.shape, (2, 3))
        self.assertEqual(output.filename, "array[0]")
        self.assertEqual(dioptrin_result_conversion(flat), "list")
        shaped = ArrayResult([0.0], np.ones((2, 3)), None)
        self.assertEqual(dioptrin_result_conversion(shaped), "array")
        output = dioptrin_converter("array", True, (2, 3))(shaped)
        self.assertIs(output.intensity, shaped.intensity_array)
        self.assertIsNone(output.sigma)
        with self.assertRaises(ValueError):
            dioptrin_converter("copy", False)


class DioptrinSummaryTests(unittest.TestCase):
    def write_row(self, output, variant, scale, path=None, conversion=None):
        how = f"-{path}" if path not in (None, "global") else ""
        result = output / "cases" / f"toy-repeated-{variant}{how}-w1-intensity-t0.result.json"
        np.savez_compressed(
            result.with_suffix(".npz"),
            radial=np.linspace(1.0, 2.0, 5),
            intensity=np.full((1, 5), 3.0 * scale),
            sigma=np.array([]),
        )
        result.write_text(
            json.dumps(
                dict(
                    dataset="toy",
                    workload="repeated",
                    variant=variant,
                    workers=1,
                    processes=1,
                    errors=False,
                    error_mode="intensity",
                    trial=0,
                    frames_per_s=10.0 * scale,
                    result=str(result),
                    dioptrin_1d_path=path,
                    dioptrin_result_conversion=conversion,
                )
            )
        )

    def test_columns_summary_and_same_path_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "cases").mkdir()
            (output / "environment.json").write_text("{}")
            self.write_row(output, "dioptrin-pixel_split", 1.0, "global", "list")
            self.write_row(output, "dioptrin-pixel_split", 2.0, "split", "list")
            self.write_row(output, "dioptrin-pyfai_full", 1.0, "split", "list")
            self.write_row(output, "pyfai-full", 1.0)
            self.write_row(output, "dioptrin-pixel_split-metal", 1.0, None, "array")
            summarize(output)
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            by = {(r["variant"], r["dioptrin_1d_path"]): r for r in rows}
            self.assertEqual(len(by), 5)
            self.assertEqual(by["pyfai-full", ""]["dioptrin_result_conversion"], "")
            self.assertEqual(
                by["dioptrin-pixel_split-metal", ""]["dioptrin_result_conversion"],
                "array",
            )
            self.assertEqual(
                by["dioptrin-pixel_split", "split"]["dioptrin_result_conversion"], "list"
            )
            summary = (output / "summary.md").read_text()
            self.assertIn("| dioptrin-pixel_split (global) |", summary)
            self.assertIn("| dioptrin-pixel_split (split) |", summary)
            self.assertIn("| pyfai-full |", summary)
            self.assertIn("num_workers=budget, num_threads=1", summary)
            self.assertIn("conversion: array (zero-copy accessors), list", summary)
            self.assertIn("**WARNING: LIST CONVERSION", summary)
            numerical = [
                e for e in json.loads((output / "numerical.json").read_text())
                if "comparison" not in e
            ]
            reference = {
                (e["variant"], e["dioptrin_1d_path"]): e["reference_dioptrin_1d_path"]
                for e in numerical
            }
            # Each path against its own reference; other rows against global.
            self.assertEqual(reference["dioptrin-pixel_split", "split"], "split")
            self.assertEqual(reference["dioptrin-pyfai_full", "split"], "split")
            self.assertEqual(reference["pyfai-full", None], "global")
            split = next(
                e for e in numerical
                if e["variant"] == "dioptrin-pixel_split" and e["dioptrin_1d_path"] == "split"
            )
            self.assertEqual(split["intensity_relative_l2"], 0.0)


class ListConversionGateTests(unittest.TestCase):
    """A build without the accessors is refused before any case runs."""

    class Probed(Exception):
        pass

    def main(self, *extra):
        argv = [
            "compare.py", "--suite", "quick", "--offline",
            "--allow-unverified-build", "--variants", "dioptrin-pixel_split",
            "--processes", "1,2", *extra,
        ]
        stderr = io.StringIO()
        with patch.object(sys, "argv", argv), patch(
            "compare.dioptrin_has_result_arrays", return_value=False
        ), patch(
            "compare.probe_variants", side_effect=self.Probed
        ), patch("sys.stderr", stderr):
            try:
                compare.main()
            except (SystemExit, self.Probed) as error:
                return error, stderr.getvalue()
        self.fail("main returned")

    def test_refused_without_the_flag(self):
        error, stderr = self.main()
        self.assertIsInstance(error, SystemExit)
        self.assertEqual(error.code, 2)
        self.assertIn("no IntegrationResult *_array accessors", stderr)
        self.assertIn("--allow-list-conversion", stderr)

    def test_flag_warns_and_continues(self):
        error, stderr = self.main("--allow-list-conversion")
        self.assertIsInstance(error, self.Probed)  # reached the variant probes
        self.assertIn("WARNING LIST CONVERSION", stderr)

    def test_detection_on_the_class(self):
        for module, expected in (
            (SimpleNamespace(), False),
            (SimpleNamespace(IntegrationResult=type("R", (), {})), False),
            (SimpleNamespace(IntegrationResult=ArrayResult([0.0], [0.0], None)), True),
        ):
            with patch.dict(sys.modules, {"dioptrin": module}):
                self.assertIs(dioptrin_has_result_arrays(), expected)


class DioptrinRunTests(unittest.TestCase):
    def test_quick_run_names_and_records_both_paths(self):
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
                    "--frames",
                    "4",
                    "--seconds",
                    "0.01",
                    "--offline",
                    "--allow-unverified-build",
                    "--variants",
                    "dioptrin-pixel_split",
                    "pyfai-full",
                    "--dioptrin-1d-path",
                    "global",
                    "split",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            environment = json.loads((output / "environment.json").read_text())
            self.assertEqual(
                environment["arguments"]["dioptrin_1d_path"], ["global", "split"]
            )
            self.assertEqual(json.loads((output / "failures.json").read_text()), [])
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            dioptrin = [r for r in rows if r["variant"] == "dioptrin-pixel_split"]
            # w1, w2 x intensity, Poisson x global, split.
            self.assertEqual(len(dioptrin), 8)
            self.assertEqual(
                sorted(r["dioptrin_1d_path"] for r in dioptrin),
                ["global"] * 4 + ["split"] * 4,
            )
            self.assertTrue(all(r["dioptrin_result_conversion"] for r in dioptrin))
            pyfai = [r for r in rows if r["variant"] == "pyfai-full"]
            self.assertEqual(len(pyfai), 4)
            self.assertTrue(
                all(
                    r["dioptrin_1d_path"] == "" and r["dioptrin_result_conversion"] == ""
                    for r in pyfai
                )
            )
            cases = output / "cases"
            split = sorted(p.name for p in cases.glob("*-dioptrin-pixel_split-split-w*.result.json"))
            default = sorted(p.name for p in cases.glob("*-dioptrin-pixel_split-w*.result.json"))
            self.assertEqual(len(split), 4)
            self.assertEqual(len(default), 4)
            for name in split:
                record = json.loads((cases / name).read_text())
                self.assertEqual(record["dioptrin_1d_path"], "split")
                if "-w2-" in name:  # checked against the split one-worker case
                    self.assertIn("-dioptrin-pixel_split-split-w1-", record["reference"])
            # Both calls integrate the same frames to the same bits.
            for mode in ("intensity", "poisson"):
                a, b = (
                    np.load(next(cases.glob(f"*-dioptrin-pixel_split{how}-w1-{mode}-t0.result.npz")))
                    for how in ("", "-split")
                )
                for key in a.files:
                    self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)


if __name__ == "__main__":
    unittest.main()
