"""Dioptrin CUDA variants, checked without a GPU through a fake dioptrin module."""

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare
from compare import (
    CUDA_NO_API,
    CUDA_PRECISION,
    CUDA_VARIANTS,
    DEFAULT_VARIANTS,
    GPU_VARIANTS,
    SEQUENTIAL_VARIANTS,
    VARIANTS,
    case_threads,
    cuda_probe,
    dioptrin_method,
    probe_cuda_child,
    probe_variants,
    setup_case,
    skipped_variants,
    summarize,
)
import setup

HERE = Path(__file__).resolve().parents[1]


class FakeIntegrator:
    """CPU stand-in: bins pixels by flat index modulo n (NumPy, float64)."""

    perturb = 0.0  # relative change applied by the fake device plan
    batch_perturb = 0.0  # extra relative change in plan.batch1d only
    calls = []

    def __init__(self, poni, method, unit, n_ss, polarization_factor, radial_range):
        self.method = method
        self.radial_range = radial_range
        self.mask = None

    def set_mask(self, mask):
        self.mask = np.asarray(mask)

    def integrate1d(self, image, n, errors=False):
        FakeIntegrator.calls.append("cpu")
        return self._bin(image, n, errors)

    def _bin(self, image, n, errors=False):
        image = np.asarray(image, dtype=np.float64)
        valid = (self.mask == 0).ravel().astype(float)
        index = np.arange(image.size) % n
        counts = np.bincount(index, valid, n)
        signal = np.bincount(index, image.ravel() * valid, n)
        return SimpleNamespace(
            radial=list(np.linspace(*self.radial_range, n)),
            intensity=list(signal / counts),
            errors=list(np.sqrt(signal) / counts) if errors else None,
            azimuthal_std=None,
        )

    def prepare_cuda1d(self, shape, n_points, *, precision, device=0, max_batch_bytes=None):
        FakeIntegrator.calls.append(("prepare", tuple(shape), n_points, precision, device))
        return FakePlan(self, tuple(shape), n_points, precision, device)


class FakePlan:
    def __init__(self, ai, shape, n, precision, device):
        self.ai, self.shape, self.n_points = ai, shape, n
        self.precision, self.device = precision, device
        self.device_name = "Fake GPU"
        self.max_batch_frames = 64
        self.table_entries = 123

    def _one(self, image, errors, scale):
        if image.shape != self.shape or image.dtype not in (np.float64, np.float32):
            raise ValueError("bad frame")
        cpu = self.ai._bin(image, self.n_points, errors)
        # float32 round trip, as the device's float32 storage would do.
        rounded = lambda v: list(  # noqa: E731
            np.float32(np.asarray(v)).astype(np.float64) * (1 + scale)
        )
        return SimpleNamespace(
            radial=cpu.radial,
            intensity=rounded(cpu.intensity),
            errors=rounded(cpu.errors) if errors else None,
            azimuthal_std=None,
        )

    def integrate1d(self, image, *, errors=False):
        FakeIntegrator.calls.append("integrate1d")
        return self._one(image, errors, FakeIntegrator.perturb)

    def batch1d(self, images, *, errors=False):
        FakeIntegrator.calls.append(("batch1d", len(images)))
        scale = FakeIntegrator.perturb + FakeIntegrator.batch_perturb
        return [self._one(image, errors, scale) for image in images]


def fake_cuda_info(device=0):
    return dict(
        compiled=True,
        available=True,
        reason=None,
        device=device,
        device_name="Fake GPU",
        compute_capability=(8, 9),
        total_memory_bytes=16 * 2**30,
        driver_version=13040,
        nvrtc_version=(13, 0),
        bindings_version=(13, 4),
    )


def fake_dioptrin(**overrides):
    module = SimpleNamespace(
        Integrator=FakeIntegrator,
        rayon_num_threads=lambda: 1,
        cuda_info=fake_cuda_info,
        cuda_available=lambda device=0: True,
    )
    for key, value in overrides.items():
        setattr(module, key, value)
    return module


def write_fixture(directory, frames=4, shape=(6, 5)):
    rng = np.random.default_rng(1)
    stack = rng.uniform(1.0, 100.0, size=(frames, *shape))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0, 0] = 1
    np.save(directory / "frames.npy", stack)
    np.save(directory / "mask.npy", mask)
    (directory / "fixture.json").write_text(
        json.dumps(dict(dataset="toy", radial_range=[0.5, 20.0], n_splitting=1))
    )
    (directory / "geometry.poni").write_text("fake")
    return stack


class CudaVariantTests(unittest.TestCase):
    def test_method_parsing(self):
        self.assertEqual(dioptrin_method("dioptrin-pixel_split-cuda"), "pixel_split")
        self.assertEqual(dioptrin_method("dioptrin-pyfai_full-cuda-batch"), "pyfai_full")
        self.assertEqual(dioptrin_method("dioptrin-pyfai_full"), "pyfai_full")
        self.assertEqual(dioptrin_method("dioptrin-supersampled"), "supersampled")
        with self.assertRaises(ValueError):
            dioptrin_method("pyfai-full")
        self.assertEqual(
            {compare.library(v) for v in CUDA_VARIANTS}, {"dioptrin"}
        )
        self.assertEqual(CUDA_PRECISION, "f32_compensated")

    def test_variant_lists(self):
        for variant in CUDA_VARIANTS:
            self.assertIn(variant, VARIANTS)
            self.assertIn(variant, GPU_VARIANTS)
            self.assertIn(variant, SEQUENTIAL_VARIANTS)
        # The default list is exactly the non-CUDA variants, in their order;
        # the CUDA rows are selectable but opt-in.
        self.assertEqual(
            DEFAULT_VARIANTS,
            (
                "dioptrin-pixel_split",
                "dioptrin-supersampled",
                "dioptrin-pyfai_full",
                "azint",
                "azint-cached",
                "pyfai-full",
                "pyfai-bbox",
                "pyfai-full-omp",
                "pyfai-full-opencl",
            ),
        )
        self.assertEqual(VARIANTS, DEFAULT_VARIANTS + tuple(CUDA_VARIANTS))
        with (
            patch.object(sys, "argv", ["compare.py", "--list-datasets"]),
            patch("argparse.ArgumentParser.parse_args", autospec=True) as parse,
        ):
            parse.side_effect = SystemExit
            with self.assertRaises(SystemExit):
                compare.main()
            parser = parse.call_args.args[0]
        defaults = parser.parse_args([])
        self.assertEqual(tuple(defaults.variants), DEFAULT_VARIANTS)
        chosen = parser.parse_args(["--variants", "dioptrin-pyfai_full-cuda"])
        self.assertEqual(chosen.variants, ["dioptrin-pyfai_full-cuda"])

    def test_threads_at_budget_one(self):
        for variant in CUDA_VARIANTS:
            threads = case_threads(variant, 1)
            self.assertEqual(threads["RAYON_NUM_THREADS"], "1")
            self.assertEqual(threads["OMP_NUM_THREADS"], "1")
            self.assertEqual(threads["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(threads, case_threads("pyfai-full-opencl", 1))

    def test_skip_messages(self):
        variants = ["dioptrin-pyfai_full-cuda", "dioptrin-pixel_split-cuda-batch", "pyfai-full"]
        not_compiled = dict(compiled=False, available=False, reason="feature 'cuda' off", device=0)
        skipped = skipped_variants(variants, dict(cuda=not_compiled))
        self.assertEqual(
            [s["variant"] for s in skipped],
            ["dioptrin-pyfai_full-cuda", "dioptrin-pixel_split-cuda-batch"],
        )
        self.assertEqual(
            skipped[0]["reason"],
            "Dioptrin was built without the 'cuda' feature; rerun setup.py "
            "--with-cuda (environment.py --with-cuda / WITH_CUDA=1)",
        )
        unavailable = dict(compiled=True, available=False, reason="no CUDA driver", device=1)
        skipped = skipped_variants(variants, dict(cuda=unavailable))
        self.assertEqual(skipped[0]["reason"], "CUDA unavailable on device 1: no CUDA driver")
        old = dict(compiled=False, available=False, reason=CUDA_NO_API, device=0)
        skipped = skipped_variants(variants, dict(cuda=old))
        self.assertTrue(skipped[0]["reason"].startswith("Dioptrin was built without"))
        self.assertIn(CUDA_NO_API, skipped[0]["reason"])
        available = dict(fake_cuda_info(), compute_capability=[8, 9])
        self.assertEqual(skipped_variants(variants, dict(cuda=available)), [])
        # Runs without CUDA variants need no CUDA probe.
        self.assertEqual(skipped_variants(["pyfai-full"], {}), [])

    def test_probe_on_old_and_new_wheels(self):
        with patch.dict(sys.modules, {"dioptrin": SimpleNamespace()}):
            self.assertEqual(
                cuda_probe(0),
                dict(compiled=False, available=False, reason=CUDA_NO_API, device=0),
            )
        with patch.dict(sys.modules, {"dioptrin": fake_dioptrin()}):
            info = cuda_probe(2)
        self.assertEqual(info["compute_capability"], [8, 9])
        self.assertEqual(info["nvrtc_version"], [13, 0])
        self.assertEqual(info["device"], 2)
        json.dumps(info, allow_nan=False)

        class PanicException(BaseException):
            pass

        def panics(device=0):
            raise PanicException("libnvrtc not found")

        with patch.dict(sys.modules, {"dioptrin": fake_dioptrin(cuda_info=panics)}):
            info = cuda_probe(0)
        self.assertFalse(info["available"])
        self.assertIn("libnvrtc", info["reason"])
        self.assertTrue(skipped_variants(["dioptrin-pyfai_full-cuda"], dict(cuda=info))[0]["reason"].startswith("CUDA unavailable on device 0"))


class CudaProbeChildTests(unittest.TestCase):
    """probe_cuda_child against real child processes standing in for compare.py."""

    def child(self, body):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "child.py"
        script.write_text("import json, os, signal, sys, time\n" + body)
        return patch.object(compare, "__file__", str(script))

    def test_non_zero_exit_skips_only_cuda(self):
        body = (
            "if '--probe-cuda' in sys.argv:\n"
            "    sys.stderr.write('NVRTC exploded\\n'); sys.exit(3)\n"
            "assert '--cuda-device' not in sys.argv\n"
            "print(json.dumps(dict(pyfai_openmp=dict(openmp=True),"
            " opencl=dict(available=True))))\n"
        )
        with self.child(body):
            probes = probe_variants(
                ["pyfai-full-opencl", "dioptrin-pyfai_full-cuda"], "gpu", 2
            )
        self.assertTrue(probes["opencl"]["available"])
        cuda = probes["cuda"]
        self.assertFalse(cuda["available"])
        self.assertTrue(cuda["probe_failed"])
        self.assertEqual(cuda["device"], 2)
        self.assertIn("exit status 3", cuda["reason"])
        self.assertIn("NVRTC exploded", cuda["reason"])
        skipped = skipped_variants(
            ["pyfai-full-opencl", "dioptrin-pyfai_full-cuda"], probes
        )
        self.assertEqual([s["variant"] for s in skipped], ["dioptrin-pyfai_full-cuda"])
        self.assertIn("exit status 3", skipped[0]["reason"])
        json.dumps(probes, allow_nan=False)

    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_signal_is_reported(self):
        with self.child("os.kill(os.getpid(), signal.SIGSEGV)\n"):
            cuda = probe_cuda_child(0)
        self.assertTrue(cuda["probe_failed"])
        self.assertIn("SIGSEGV", cuda["reason"])
        self.assertIn(f"exit status {-signal.SIGSEGV}", cuda["reason"])

    def test_timeout_is_reported(self):
        with self.child("time.sleep(60)\n"):
            started = time.monotonic()
            cuda = probe_cuda_child(1, timeout=0.5)
        self.assertLess(time.monotonic() - started, 30)
        self.assertFalse(cuda["available"])
        self.assertTrue(cuda["probe_failed"])
        self.assertIn("timed out after 0.5 s", cuda["reason"])
        reason = skipped_variants(["dioptrin-pixel_split-cuda"], dict(cuda=cuda))[0]["reason"]
        self.assertIn("timed out", reason)

    def test_success_and_no_cuda_request(self):
        info = compare.jsonable(fake_cuda_info(4))
        with self.child(
            "assert sys.argv[1:] == ['--probe-cuda', '--cuda-device', '4']\n"
            f"print('noise'); print(json.dumps(json.loads({json.dumps(info)!r})))\n"
        ):
            self.assertEqual(probe_cuda_child(4), info)
        with patch("compare.subprocess.run") as run:
            self.assertEqual(probe_variants(["pyfai-full", "azint"], "gpu"), {})
            run.assert_not_called()


class CudaSetupCaseTests(unittest.TestCase):
    def setUp(self):
        FakeIntegrator.perturb = 0.0
        FakeIntegrator.batch_perturb = 0.0
        FakeIntegrator.calls = []
        self.directory = tempfile.TemporaryDirectory()
        self.fixture = Path(self.directory.name)
        self.stack = write_fixture(self.fixture)

    def tearDown(self):
        self.directory.cleanup()

    def case(self, variant, errors):
        return dict(
            fixture=str(self.fixture),
            workload="frames",
            frames=4,
            bins=7,
            workers=1,
            processes=1,
            errors=errors,
            variant=variant,
            cuda_device=0,
        )

    def setup(self, variant, errors=True):
        with patch.dict(sys.modules, {"dioptrin": fake_dioptrin()}):
            return setup_case(self.case(variant, errors))

    def test_frame_and_batch_rows_run_validate_and_record(self):
        for variant in CUDA_VARIANTS:
            for errors in (False, True):
                with self.subTest(variant=variant, errors=errors):
                    FakeIntegrator.calls = []
                    state = self.setup(variant, errors)
                    method, kind = CUDA_VARIANTS[variant]
                    self.assertEqual(
                        FakeIntegrator.calls[0], ("prepare", (6, 5), 7, CUDA_PRECISION, 0)
                    )
                    FakeIntegrator.calls = []
                    results = state["run"]()
                    self.assertEqual(len(results), 4)
                    if kind == "frame":
                        self.assertEqual(FakeIntegrator.calls, ["integrate1d"] * 4)
                    else:
                        self.assertEqual(FakeIntegrator.calls, [("batch1d", 4)])
                    state["validate"](results)
                    cuda = state["info"]["cuda"]
                    json.dumps(cuda, allow_nan=False)
                    self.assertEqual(cuda["precision"], CUDA_PRECISION)
                    self.assertEqual(cuda["device_name"], "Fake GPU")
                    self.assertEqual(cuda["compute_capability"], [8, 9])
                    self.assertEqual(cuda["driver_version"], 13040)
                    self.assertEqual(cuda["nvrtc_version"], [13, 0])
                    self.assertEqual(cuda["max_batch_frames"], 64)
                    self.assertEqual(cuda["table_entries"], 123)
                    self.assertEqual(cuda["input_dtype"], "float64")
                    agreement = cuda["cpu_agreement"]
                    self.assertEqual(agreement["frames_checked"], 4)
                    self.assertTrue(agreement["radial_exactly_equal"])
                    # float32 rounding of the outputs: below 2^-24 relative.
                    self.assertGreater(agreement["intensity"]["max_rel_nonzero"], 0)
                    self.assertLess(agreement["intensity"]["max_rel_nonzero"], 6e-8)
                    self.assertEqual("sigma" in agreement, errors)
                    self.assertEqual(
                        cuda["batch_bit_identical"], True if kind == "batch" else None
                    )

    def test_check_then_warm_run_before_timing(self):
        """Order: warm run, setup_ready, CPU agreement check, one more warm run,
        then return (timing starts after setup_case returns)."""
        events = []
        clock = time.perf_counter

        def tick():
            events.append("clock")
            return clock()

        for variant in ("dioptrin-pyfai_full-cuda", "dioptrin-pixel_split-cuda-batch"):
            with self.subTest(variant=variant):
                FakeIntegrator.calls = events
                events.clear()
                with patch("compare.time.perf_counter", side_effect=tick):
                    state = self.setup(variant)
                run = ["integrate1d"] * 4 if "batch" not in variant else [("batch1d", 4)]
                # The last clock reading is setup_ready.
                ready = len(events) - 1 - events[::-1].index("clock")
                self.assertEqual(events[ready - len(run):ready], run)
                after = events[ready + 1:]
                self.assertIn("cpu", after)  # the agreement check
                last_cpu = len(after) - 1 - after[::-1].index("cpu")
                # After the CPU integrations: for batch rows the batch identity
                # check (batch1d, then integrate1d per frame); then the warm run.
                check = [] if "batch" not in variant else [("batch1d", 4)] + ["integrate1d"] * 4
                self.assertEqual(after[last_cpu + 1:], check + run)
                self.assertGreater(state["setup_ready"], 0)
                self.assertNotIn("cpu", events[:ready])

    def test_frames_reach_the_plan_unchanged(self):
        seen = []
        original = FakePlan.integrate1d

        def spy(plan, image, *, errors=False):
            seen.append(image)
            return original(plan, image, errors=errors)

        with patch.object(FakePlan, "integrate1d", spy):
            state = self.setup("dioptrin-pyfai_full-cuda")
            seen.clear()
            state["run"]()
        self.assertEqual(len(seen), 4)
        for index, image in enumerate(seen):
            self.assertEqual(image.dtype, np.float64)
            np.testing.assert_array_equal(image, self.stack[index])

    def test_cpu_agreement_fails_beyond_tolerance(self):
        FakeIntegrator.perturb = 1e-6  # within 1e-5
        state = self.setup("dioptrin-pixel_split-cuda")
        self.assertGreater(
            state["info"]["cuda"]["cpu_agreement"]["intensity"]["max_rel_nonzero"], 9e-7
        )
        FakeIntegrator.perturb = 1e-4
        for variant in ("dioptrin-pixel_split-cuda", "dioptrin-pyfai_full-cuda-batch"):
            with self.subTest(variant=variant), self.assertRaisesRegex(
                AssertionError, "differs from CPU integrate1d"
            ):
                self.setup(variant)

    def test_batch_identity_is_enforced(self):
        FakeIntegrator.batch_perturb = 1e-15  # far inside the CPU tolerance
        with self.assertRaisesRegex(AssertionError, "batch1d differs"):
            self.setup("dioptrin-pyfai_full-cuda-batch")
        # Frame rows do not call batch1d.
        self.assertIsNone(self.setup("dioptrin-pyfai_full-cuda")["info"]["cuda"]["batch_bit_identical"])


class CudaSummaryTests(unittest.TestCase):
    def write_rows(self, output, variants):
        (output / "cases").mkdir()
        radial = np.linspace(1.0, 2.0, 5)
        for variant, scale in variants:
            path = output / "cases" / f"toy-repeated-{variant}-w1-intensity-t0.result.json"
            np.savez_compressed(
                path.with_suffix(".npz"),
                radial=radial,
                intensity=np.full((1, 5), 3.0 * scale),
                sigma=np.array([]),
            )
            path.write_text(
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
                        frames_per_s=1.0,
                        result=str(path),
                    )
                )
            )

    def test_cpu_only_summary_keeps_its_wording(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_rows(
                output,
                [("dioptrin-pixel_split", 1.0), ("pyfai-full", 1.0), ("pyfai-full-opencl", 1.0)],
            )
            (output / "environment.json").write_text("{}")
            summarize(output)
            summary = (output / "summary.md").read_text()
            numerical = json.loads((output / "numerical.json").read_text())
        self.assertTrue(summary.startswith(
            "# CPU integration benchmark\n\n1D, preloaded float64 inputs; generation, "
            "decoding, and output writing excluded.\nRepeated reuses one image;"
        ))
        self.assertNotIn("GPU", summary)  # pyfai-full-opencl alone: unchanged
        references = [e for e in numerical if "comparison" not in e]
        self.assertEqual([sorted(e) for e in references], [sorted(numerical[0])] * 3)
        self.assertNotIn("also_counterpart", numerical[0])
        self.assertNotIn("comparison", numerical[0])

    def test_counterparts_and_device_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_rows(
                output,
                (
                    ("dioptrin-pixel_split", 1.0),
                    ("dioptrin-pyfai_full", 1.0),
                    ("dioptrin-pyfai_full-cuda", 1.0 + 1e-7),
                    ("dioptrin-pixel_split-cuda", 1.0 + 2e-7),
                    ("pyfai-full", 1.0),
                    ("pyfai-full-opencl", 1.0 + 1e-6),
                ),
            )
            (output / "environment.json").write_text(
                json.dumps(dict(variant_probes=dict(cuda=fake_cuda_info())))
            )
            summarize(output)
            numerical = json.loads((output / "numerical.json").read_text())
            references = [e for e in numerical if "comparison" not in e]
            self.assertEqual({e["reference"] for e in references}, {"dioptrin-pixel_split"})
            self.assertEqual(len(references), 6)
            # One entry per (variant, reference): no duplicate for pixel_split-cuda,
            # whose counterpart is the reference itself.
            pairs = [(e["variant"], e["reference"]) for e in numerical]
            self.assertEqual(len(pairs), len(set(pairs)))
            marked = {e["variant"] for e in references if e.get("also_counterpart")}
            self.assertEqual(marked, {"dioptrin-pixel_split-cuda"})
            counterparts = {
                (e["variant"], e["reference"]): e
                for e in numerical
                if e.get("comparison") == "counterpart"
            }
            # The batch rows are absent: no entry.
            self.assertEqual(
                set(counterparts),
                {
                    ("dioptrin-pyfai_full-cuda", "dioptrin-pyfai_full"),
                    ("pyfai-full-opencl", "pyfai-full"),
                },
            )
            self.assertAlmostEqual(
                counterparts[("dioptrin-pyfai_full-cuda", "dioptrin-pyfai_full")][
                    "intensity_relative_l2"
                ],
                1e-7,
                delta=1e-12,
            )
            summary = (output / "summary.md").read_text()
            self.assertTrue(summary.startswith("# CPU and GPU integration benchmark\n"))
            self.assertIn(
                "CUDA and OpenCL rows (dioptrin-pixel_split-cuda, "
                "dioptrin-pyfai_full-cuda, pyfai-full-opencl)\nuse one host thread",
                summary,
            )
            self.assertIn(
                "Dioptrin CUDA device: Fake GPU (compute 8.9), driver 13.4, NVRTC 13.0, "
                "precision f32_compensated",
                summary,
            )


class CudaSchedulingTests(unittest.TestCase):
    """main() with a CUDA-capable probe and recorded (not executed) cases."""

    def test_cuda_cases_only_at_one_worker_one_process_and_in_1d(self):
        cases, preflight_variants = [], []
        real_run = subprocess.run

        def record(case_path):
            case = json.loads(case_path.read_text())
            cases.append(case)
            # Stand-in one-worker reference, so later cases are scheduled.
            Path(case["result"]).with_suffix(".npz").write_bytes(b"")

        def fake_processes(case, case_path, env, prefixes):
            record(case_path)
            return "recorded"

        def fake_run(command, **kwargs):
            command = [str(c) for c in command]
            if "--preflight-2d" in command:
                directory = Path(command[command.index("--preflight-2d") + 1])
                start = command.index("--variants") + 1
                preflight_variants.extend(command[start:])
                (directory / "preflight2d.json").write_text(
                    json.dumps(
                        dict(
                            radial_range_2d=[0.5, 20.0],
                            convention=dict(passed=True, libraries={}),
                            dioptrin_radial_axes={
                                "pixel_split": {"8": list(np.linspace(1, 2, 8))}
                            },
                            table_estimates={
                                "8x4": dict(polygon_nnz=10.0, subpixel_nnz={"10": 10.0})
                            },
                        )
                    )
                )
                return subprocess.CompletedProcess(command, 0)
            if "--case" in command:
                record(Path(command[command.index("--case") + 1]))
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)  # platform/hardware/git queries

        probes = dict(cuda=fake_cuda_info(), opencl=dict(available=False))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "compare.py", "--suite", "quick", "--workloads", "frames",
                "--dimensions", "1d", "2d", "--cakes", "8x4", "--workers", "1,2",
                "--processes", "1,2", "--trials", "1", "--offline",
                "--allow-unverified-build", "--cuda-device", "3", "--output", str(output),
                "--variants", "dioptrin-pixel_split", "dioptrin-pyfai_full-cuda",
                "dioptrin-pixel_split-cuda-batch",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("compare.probe_variants", return_value=probes) as probe,
                patch("compare.subprocess.run", side_effect=fake_run),
                patch("compare.run_processes", side_effect=fake_processes),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                compare.main()
            probe.assert_called_once()
            self.assertEqual(probe.call_args.args[2], 3)
            environment = json.loads((output / "environment.json").read_text())
            self.assertEqual(environment["variant_probes"]["cuda"]["device_name"], "Fake GPU")
            self.assertEqual(json.loads((output / "skipped.json").read_text()), [])
        self.assertEqual(preflight_variants, ["dioptrin-pixel_split"])
        cuda = [c for c in cases if c["variant"] in CUDA_VARIANTS]
        self.assertEqual(len(cuda), 4)  # two variants x intensity/Poisson
        for case in cuda:
            self.assertEqual((case["workers"], case["processes"]), (1, 1))
            self.assertNotEqual(case.get("dimension"), "2d")
            self.assertEqual(case["cuda_device"], 3)
            self.assertEqual(case["threads"]["RAYON_NUM_THREADS"], "1")
        cpu = [c for c in cases if c["variant"] == "dioptrin-pixel_split"]
        self.assertTrue(any(c.get("dimension") == "2d" for c in cpu))
        self.assertTrue(any(c["workers"] == 2 for c in cpu))
        self.assertTrue(any(c["processes"] == 2 for c in cpu))
        self.assertTrue(all("cuda_device" not in c for c in cpu))


class CudaBuildTests(unittest.TestCase):
    def test_setup_builds_python_and_cuda_features(self):
        command = setup.maturin_command("py", "wheels", with_cuda=True)
        self.assertEqual(command[command.index("--features") + 1], "python,cuda")
        command = setup.maturin_command("py", "wheels", locked=["--locked"])
        self.assertEqual(command[command.index("--features") + 1], "python")
        self.assertIn("--locked", command)
        self.assertEqual(setup.cargo_features(True), ["python", "cuda"])

    def test_sbatch_has_the_cuda_switch(self):
        text = (HERE / "maxwell.sbatch").read_text()
        self.assertIn("WITH_CUDA", text)
        self.assertIn("cache_args+=(--with-cuda)", text)
        subprocess.run(["bash", "-n", str(HERE / "maxwell.sbatch")], check=True)

    @unittest.skipIf(sys.platform == "win32", "environment cache is POSIX only")
    def test_environment_key_and_probe(self):
        from environment import (
            build_options,
            cache_key,
            fingerprint,
            probe_environment,
            setup_arguments,
        )

        self.assertFalse(build_options()["with_cuda"])
        self.assertTrue(build_options(with_cuda=True)["with_cuda"])
        self.assertEqual(setup_arguments(build_options(with_cuda=True)), ["--with-cuda"])
        self.assertEqual(
            setup_arguments(build_options(True, True, with_cuda=True)),
            ["--azint-fast-math", "--with-opencl", "--with-cuda"],
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"CC": "gcc", "CXX": "g++"}),
        ):
            root = Path(temporary) / "repo"
            (root / "src").mkdir(parents=True)
            for name, text in (("src/lib.rs", "x"), ("Cargo.toml", "c"), ("pyproject.toml", "p")):
                (root / name).write_text(text)
            recipe = root / "benchmarks/comparison"
            recipe.mkdir(parents=True)
            for name in ("environment.py", "setup.py", "requirements.txt", "provenance.py"):
                (recipe / name).write_text(name)
            with (
                patch("environment.cpu_identity", return_value={"model": "a"}),
                patch("environment.subprocess.check_output", return_value="gcc a"),
                patch("environment.sysconfig.get_config_var", return_value="abi"),
                patch("environment.platform.libc_ver", return_value=("libc", "1")),
            ):
                plain = fingerprint(root, "stable", build_options())
                cuda = fingerprint(root, "stable", build_options(with_cuda=True))
            self.assertTrue(cuda["build_options"]["with_cuda"])
            self.assertNotEqual(cache_key(plain), cache_key(cuda))

            generation = Path(temporary) / "generation"
            for name in ("venv/bin/python", "venv/benchmark-build.json", "build/python-packages.txt"):
                (generation / name).parent.mkdir(parents=True, exist_ok=True)
                (generation / name).write_text("")
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 1, "", "stop")

            with patch("environment.subprocess.run", side_effect=fake_run):
                for options in (build_options(), build_options(with_cuda=True)):
                    probe_environment(generation, root, options=options)
        plain_check, cuda_check = (c[c.index("-c") + 1] for c in commands)
        self.assertNotIn("cuda_info", plain_check)
        self.assertIn("dioptrin.cuda_info()['compiled']", cuda_check)
        compile(cuda_check, "<check>", "exec")


if __name__ == "__main__":
    unittest.main()
