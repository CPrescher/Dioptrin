"""Experimental CUDA 1D plans: API, errors and input checks without a GPU;
agreement with the CPU on a usable CUDA device.

The device tests skip when ``dioptrin.cuda_available()`` is false (default
wheels, machines without an NVIDIA driver or NVRTC). Set
``DIOPTRIN_REQUIRE_CUDA=1`` to turn those skips into failures. CUDA is
initialised lazily, when the first test that needs to know runs, not when the
module is imported.
"""
import ast
import functools
import gc
import os
from pathlib import Path
import tempfile
import threading
import unittest

import dioptrin
import numpy as np

PONI = dict(pixel1=1e-4, pixel2=1e-4, distance=.1,
            poni1=.0016, poni2=.0032, wavelength=1e-10)
SHAPE = (32, 64)
NEW_NAMES = {"Cuda1dPlan", "CudaUnavailableError", "cuda_available", "cuda_info"}
INFO_KEYS = {"compiled", "available", "reason", "device", "device_name",
             "compute_capability", "total_memory_bytes", "driver_version",
             "nvrtc_version", "bindings_version"}
NOT_COMPILED = ("dioptrin was built without CUDA support; build with: "
                "maturin build --release --features python,cuda")
REQUIRE = os.environ.get("DIOPTRIN_REQUIRE_CUDA") == "1"


@functools.lru_cache(maxsize=None)
def available():
    """Whether device 0 is usable; asked once, on first use."""
    return dioptrin.cuda_available()


def integrator(method="pixel_split", **options):
    options.setdefault("polarization_factor", .965)
    return dioptrin.Integrator.from_poni_dict(PONI, method=method, **options)


class CudaApiTests(unittest.TestCase):
    """Run on every build, with or without the feature or a device."""

    def test_names_in_stub_and_module(self):
        stub = Path(__file__).resolve().parents[1] / "dioptrin.pyi"
        tree = ast.parse(stub.read_text())
        exported = next(
            ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "__all__" for t in node.targets)
        )
        self.assertLessEqual(NEW_NAMES, set(exported))
        defined = {node.name for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertLessEqual(set(exported), defined)
        self.assertEqual(set(dioptrin.__all__), set(exported))
        for name in exported:
            self.assertTrue(hasattr(dioptrin, name), name)
        self.assertTrue(hasattr(dioptrin.Integrator, "prepare_cuda1d"))

    def test_info_and_available(self):
        info = dioptrin.cuda_info()
        self.assertEqual(set(info), INFO_KEYS)
        self.assertIsInstance(info["compiled"], bool)
        self.assertIsInstance(info["available"], bool)
        self.assertEqual(info["device"], 0)
        self.assertEqual(dioptrin.cuda_available(), info["available"])
        self.assertEqual(dioptrin.cuda_info(device=0)["available"], info["available"])
        if info["available"]:
            self.assertIsNone(info["reason"])
            self.assertIsInstance(info["device_name"], str)
            self.assertIsInstance(info["compute_capability"], tuple)
        else:
            self.assertIsInstance(info["reason"], str)
            self.assertTrue(info["reason"])
        if info["compiled"]:
            self.assertIsInstance(info["bindings_version"], tuple)
        else:
            self.assertFalse(info["available"])
            self.assertIsNone(info["bindings_version"])
            self.assertIn("cuda", info["reason"])
        # A device that does not exist is reported, not raised.
        other = dioptrin.cuda_info(4096)
        self.assertEqual(other["device"], 4096)
        self.assertFalse(other["available"])
        self.assertFalse(dioptrin.cuda_available(4096))
        with self.assertRaises(OverflowError):
            dioptrin.cuda_info(-1)

    def test_exception_type(self):
        self.assertTrue(issubclass(dioptrin.CudaUnavailableError, RuntimeError))

    def test_text_signatures(self):
        self.assertEqual(dioptrin.Cuda1dPlan.integrate1d.__text_signature__,
                         "($self, image, *, errors=False)")
        self.assertEqual(dioptrin.Cuda1dPlan.batch1d.__text_signature__,
                         "($self, images, *, errors=False)")
        self.assertEqual(
            dioptrin.Integrator.prepare_cuda1d.__text_signature__,
            '($self, shape, n_points, *, precision="f32_compensated", device=0, '
            'max_batch_bytes=None)')

    def test_out_of_range_device(self):
        """Without the feature, without a driver, or with a device: an
        ordinal that does not exist is CudaUnavailableError."""
        with self.assertRaises(dioptrin.CudaUnavailableError):
            integrator().prepare_cuda1d(SHAPE, 40, device=4096)

    def test_cpu_non_finite_semantics(self):
        """The CPU results the device test of non-finite pixels compares with:
        each kind of non-finite bin occurs, NaN and -inf pixels add nothing
        to the variance, a +inf pixel makes it +inf."""
        image, mask = non_finite_image()
        for method in ("counting", "pixel_split", "pyfai_full"):
            with self.subTest(method=method):
                ai = integrator(method)
                ai.set_mask(mask)
                result = ai.integrate1d(image, 40, errors=True)
                intensity = np.array(result.intensity)
                errors = np.array(result.errors)
                for kind in (np.isnan, np.isposinf, np.isneginf):
                    self.assertTrue(kind(intensity).any(), kind.__name__)
                np.testing.assert_array_equal(np.isposinf(errors), np.isposinf(intensity))
                self.assertFalse(np.isnan(errors).any())
                self.assertFalse(np.isneginf(errors).any())

    def test_cpu_ignores_masked_array_mask(self):
        """docs/cuda.md says the mask of an np.ma.MaskedArray is ignored, as
        on the CPU."""
        image = np.random.default_rng(5).uniform(1, 400, SHAPE)
        masked = np.ma.masked_array(image, mask=image > 200)
        ai = integrator()
        self.assertEqual(ai.integrate1d(masked, 40).intensity,
                         ai.integrate1d(image, 40).intensity)

    def test_configuration_errors(self):
        ai = integrator()
        with self.assertRaises(ValueError):
            integrator(mask_mode="dynamic").prepare_cuda1d(SHAPE, 40)
        for method in ("histogram", "supersampled"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                integrator(method).prepare_cuda1d(SHAPE, 40)
        with self.assertRaises(ValueError):
            ai.prepare_cuda1d(SHAPE, 0)
        with self.assertRaises(ValueError):
            ai.prepare_cuda1d((0, 64), 40)
        with self.assertRaises(ValueError):
            ai.prepare_cuda1d(SHAPE, 40, precision="f16")
        with self.assertRaises(TypeError):
            ai.prepare_cuda1d(SHAPE, 40, precision=32)
        with self.assertRaises(ValueError):
            ai.prepare_cuda1d(SHAPE, 40, max_batch_bytes=0)
        with self.assertRaises(OverflowError):
            ai.prepare_cuda1d(SHAPE, 40, device=-1)
        masked = integrator()
        masked.set_mask(np.zeros((16, 64), np.uint8))
        with self.assertRaises(ValueError):
            masked.prepare_cuda1d(SHAPE, 40)

    def test_valid_configuration_without_cuda(self):
        if available():
            self.skipTest("a CUDA device is usable")
        info = dioptrin.cuda_info()
        for method in ("counting", "pixel_split", "pyfai_full"):
            for precision in ("f32", "f32_compensated", "f64"):
                with self.subTest(method=method, precision=precision):
                    with self.assertRaises(dioptrin.CudaUnavailableError) as raised:
                        integrator(method).prepare_cuda1d(SHAPE, 40, precision=precision)
                    self.assertIsInstance(raised.exception, RuntimeError)
                    if info["compiled"]:
                        self.assertIn(info["reason"], str(raised.exception))
                    else:
                        self.assertEqual(str(raised.exception), NOT_COMPILED)


class CudaInputTests(unittest.TestCase):
    """The argument checks and borrows of Cuda1dPlan.integrate1d/batch1d, run
    through a private hook so that they are covered without a device."""

    # Private, outside __all__: on the extension module, which maturin places
    # at dioptrin.dioptrin when it generates a package around the stub.
    check = staticmethod(getattr(dioptrin, "dioptrin", dioptrin)._cuda_check_input)

    def test_accepted_inputs(self):
        image = np.ones(SHAPE)
        self.assertEqual(self.check(image, SHAPE, False), (1, "float64"))
        self.assertEqual(self.check(image.astype(np.float32), SHAPE, False), (1, "float32"))
        self.assertEqual(self.check([image] * 3, SHAPE, True), (3, "float64"))
        self.assertEqual(self.check((image, image), SHAPE, True), (2, "float64"))
        self.assertEqual(self.check([], SHAPE, True), (0, "float64"))
        stack = np.ones((5, *SHAPE), np.float32)
        self.assertEqual(self.check(stack, SHAPE, True), (5, "float32"))
        self.assertEqual(self.check(np.ones((0, *SHAPE)), SHAPE, True), (0, "float64"))
        # Read-only arrays and slices of an existing array are borrowed too.
        readonly = np.ones(SHAPE)
        readonly.flags.writeable = False
        self.assertEqual(self.check(readonly, SHAPE, False), (1, "float64"))
        self.assertEqual(self.check([readonly, readonly], SHAPE, True), (2, "float64"))
        self.assertEqual(self.check(list(np.ones((3, *SHAPE))), SHAPE, True), (3, "float64"))

    def test_read_only_memmap_frames(self):
        """The benchmark harness passes read-only np.memmap frames."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, np.random.default_rng(3).uniform(1, 9, (4, *SHAPE)))
            stack = np.load(path, mmap_mode="r")
            self.assertIsInstance(stack, np.memmap)
            self.assertFalse(stack.flags.writeable)
            frames = [stack[i % 4] for i in range(6)]
            self.assertEqual(self.check(frames[0], SHAPE, False), (1, "float64"))
            self.assertEqual(self.check(frames, SHAPE, True), (6, "float64"))
            self.assertEqual(self.check(stack, SHAPE, True), (4, "float64"))
            # The CPU batch accepts the same frames.
            cpu = integrator().batch1d(frames, 40)
            self.assertTrue(all(r.is_ok() for r in cpu))
            del stack, frames

    def test_rejected_inputs(self):
        image = np.ones(SHAPE)
        for bad in (image.astype(np.int64), image.astype(np.float16),
                    image.astype(">f8"), image.astype(np.complex128)):
            with self.subTest(dtype=bad.dtype), self.assertRaisesRegex(TypeError, "dtype"):
                self.check(bad, SHAPE, False)
            with self.subTest(dtype=bad.dtype), self.assertRaisesRegex(TypeError, "dtype"):
                self.check([image, bad], SHAPE, True)
        with self.assertRaisesRegex(TypeError, "same dtype"):
            self.check([image, image.astype(np.float32)], SHAPE, True)
        with self.assertRaises(TypeError):
            self.check(image.tolist(), SHAPE, False)
        with self.assertRaises(TypeError):
            self.check([image.tolist()], SHAPE, True)
        with self.assertRaises(TypeError):
            self.check(iter([image]), SHAPE, True)
        with self.assertRaisesRegex(ValueError, "C contiguous"):
            self.check(np.asfortranarray(image), SHAPE, False)
        with self.assertRaisesRegex(ValueError, "C contiguous"):
            self.check([image, np.ones((32, 128))[:, ::2]], SHAPE, True)
        with self.assertRaisesRegex(ValueError, "C contiguous"):
            self.check(np.ones((*SHAPE, 3)).transpose(2, 0, 1), SHAPE, True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.check(image[:16], SHAPE, False)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.check([image, image[:, :32].copy()], SHAPE, True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.check(np.ones((2, 16, 64)), SHAPE, True)
        with self.assertRaises(ValueError):
            self.check(np.ones((1, *SHAPE)), SHAPE, False)
        with self.assertRaises(ValueError):
            self.check(image, SHAPE, True)

    def test_zero_sized_plan_shape(self):
        """Plans never have a zero side; the hook reports one as ValueError,
        not a division by zero."""
        for images, shape, batch in ((np.zeros((1, 0, 5)), (0, 5), True),
                                     (np.zeros((2, 5, 0)), (5, 0), True),
                                     (np.zeros((0, 5)), (0, 5), False),
                                     ([np.zeros((0, 5))], (0, 5), True)):
            with self.subTest(shape=np.shape(images), batch=batch):
                with self.assertRaisesRegex(ValueError, "non-zero"):
                    self.check(images, shape, batch)

    def test_unaligned_arrays(self):
        image = unaligned(SHAPE)
        self.assertTrue(image.flags.c_contiguous)
        self.assertFalse(image.flags.aligned)
        with self.assertRaisesRegex(ValueError, "^image is not aligned") as raised:
            self.check(image, SHAPE, False)
        self.assertNotIn("images", str(raised.exception))
        with self.assertRaisesRegex(ValueError, r"^images\[1\] is not aligned"):
            self.check([np.ones(SHAPE), image], SHAPE, True)
        with self.assertRaisesRegex(ValueError, "^images is not aligned"):
            self.check(unaligned((2, *SHAPE)), SHAPE, True)
        # An aligned copy is accepted.
        self.assertEqual(self.check(image.copy(), SHAPE, False), (1, "float64"))

    def test_numpy_scalars(self):
        for scalar in (np.float64(1.0), np.float32(1.0)):
            name = type(scalar).__name__
            with self.subTest(scalar=name):
                with self.assertRaisesRegex(TypeError, f"numpy scalar of type numpy.{name}"):
                    self.check(scalar, SHAPE, False)
                with self.assertRaisesRegex(TypeError, f"numpy scalar of type numpy.{name}"):
                    self.check(scalar, SHAPE, True)
                with self.assertRaisesRegex(TypeError, rf"^images\[0\] .*numpy scalar"):
                    self.check([scalar], SHAPE, True)

    def test_masked_array_is_accepted(self):
        image = np.ones(SHAPE)
        masked = np.ma.masked_array(image, mask=np.ones(SHAPE, bool))
        self.assertEqual(self.check(masked, SHAPE, False), (1, "float64"))


def unaligned(shape):
    """A C-contiguous float64 array whose data starts one byte into its
    buffer, so numpy marks it as not aligned."""
    count = int(np.prod(shape))
    buffer = np.zeros(count * 8 + 1, np.uint8)
    array = buffer[1:].view(np.float64).reshape(shape)
    array[...] = 1.0
    return array


def non_finite_image():
    """Finite pixels in [1, 400) plus NaN, +inf and -inf pixels in different
    radial regions, and a NaN pixel that the returned mask hides."""
    image = np.random.default_rng(42).uniform(1, 400, SHAPE)
    image[3, 5] = np.nan
    image[20, 40] = np.inf
    image[28, 10] = -np.inf
    image[10, 60] = np.nan
    mask = np.zeros(SHAPE, np.uint8)
    mask[10, 60] = 1
    return image, mask


class CudaDeviceTests(unittest.TestCase):
    """Need a usable CUDA device. Skipped without one, unless
    DIOPTRIN_REQUIRE_CUDA=1 makes every test here fail instead."""

    @classmethod
    def setUpClass(cls):
        if not REQUIRE and not available():
            raise unittest.SkipTest("CUDA is not available: "
                                    + str(dioptrin.cuda_info()["reason"]))

    def frames(self, n=6, dtype=np.float64):
        rng = np.random.default_rng(42)
        return rng.uniform(1, 400, (n, *SHAPE)).astype(dtype)

    def assert_close(self, gpu, cpu, rtol):
        cpu = np.asarray(cpu)
        np.testing.assert_allclose(gpu, cpu, rtol=rtol, atol=rtol * np.max(np.abs(cpu)))

    def test_matches_cpu(self):
        frames = self.frames()
        mask = (frames[0] > 380).astype(np.uint8)
        for method in ("counting", "pixel_split", "pyfai_full"):
            for unit, radial_range in (("2th_deg", (0.2, 1.8)), ("q_A^-1", None)):
                ai = integrator(method, unit=unit, radial_range=radial_range)
                ai.set_mask(mask)
                for precision, rtol in (("f32", 2e-6), ("f32_compensated", 2e-6), ("f64", 1e-12)):
                    plan = ai.prepare_cuda1d(SHAPE, 40, precision=precision)
                    self.assertEqual(plan.precision, precision)
                    self.assertEqual(plan.shape, SHAPE)
                    self.assertEqual(plan.n_points, 40)
                    self.assertGreater(plan.max_batch_frames, 0)
                    self.assertGreater(plan.table_entries, 0)
                    for dtype in (np.float64, np.float32):
                        for errors in (False, True):
                            with self.subTest(method=method, unit=unit, precision=precision,
                                              dtype=dtype.__name__, errors=errors):
                                image = frames[0].astype(dtype)
                                reference_input = image
                                if precision != "f64":
                                    reference_input = image.astype(np.float32)
                                cpu = ai.integrate1d(reference_input.astype(np.float64),
                                                     40, errors=errors)
                                gpu = plan.integrate1d(image, errors=errors)
                                self.assertEqual(gpu.radial, cpu.radial)
                                self.assertEqual(plan.radial, cpu.radial)
                                self.assert_close(gpu.intensity, cpu.intensity, rtol)
                                self.assertIsNone(gpu.azimuthal_std)
                                if errors:
                                    self.assert_close(gpu.errors, cpu.errors, rtol)
                                else:
                                    self.assertIsNone(gpu.errors)

    def test_default_precision_and_info(self):
        plan = integrator().prepare_cuda1d(SHAPE, 40)
        self.assertEqual(plan.precision, "f32_compensated")
        self.assertEqual(plan.device, 0)
        info = dioptrin.cuda_info(plan.device)
        self.assertEqual(plan.device_name, info["device_name"])
        self.assertIn("Cuda1dPlan", repr(plan))

    def test_batch_is_bit_identical(self):
        frames = self.frames(300)
        plan = integrator("pyfai_full").prepare_cuda1d(
            SHAPE, 40, max_batch_bytes=64 * (SHAPE[0] * SHAPE[1] * 12 + 40 * 16))
        self.assertLess(plan.max_batch_frames, len(frames))
        single = [plan.integrate1d(f, errors=True) for f in frames]
        for batch in (plan.batch1d(list(frames), errors=True),
                      plan.batch1d(tuple(frames), errors=True),
                      plan.batch1d(frames, errors=True)):
            self.assertEqual(len(batch), len(frames))
            for a, b in zip(batch, single):
                self.assertEqual(a.intensity, b.intensity)
                self.assertEqual(a.errors, b.errors)
                self.assertEqual(a.radial, b.radial)
        self.assertEqual(plan.batch1d([]), [])

    def test_read_only_memmap_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, self.frames(4))
            stack = np.load(path, mmap_mode="r")
            frames = [stack[i % 4] for i in range(6)]
            plan = integrator().prepare_cuda1d(SHAPE, 40)
            batch = plan.batch1d(frames)
            single = [plan.integrate1d(f) for f in frames]
            self.assertEqual([r.intensity for r in batch], [r.intensity for r in single])
            del stack, frames

    def test_input_errors(self):
        plan = integrator().prepare_cuda1d(SHAPE, 40)
        image = np.ones(SHAPE)
        with self.assertRaises(TypeError):
            plan.integrate1d(image.astype(np.int64))
        with self.assertRaises(TypeError):
            plan.batch1d([image, image.astype(np.float32)])
        with self.assertRaises(ValueError):
            plan.integrate1d(np.asfortranarray(image))
        with self.assertRaises(ValueError):
            plan.integrate1d(image[:16])
        with self.assertRaisesRegex(ValueError, "^image is not aligned"):
            plan.integrate1d(unaligned(SHAPE))
        with self.assertRaisesRegex(TypeError, "numpy scalar"):
            plan.integrate1d(np.float64(1.0))
        with self.assertRaisesRegex(TypeError, "numpy scalar"):
            plan.batch1d(np.float64(1.0))

    def test_snapshot_and_fully_masked(self):
        image = self.frames(1)[0]
        ai = integrator()
        plan = ai.prepare_cuda1d(SHAPE, 40)
        before = plan.integrate1d(image).intensity
        ai.set_mask(np.ones(SHAPE, np.uint8))
        ai.set_polarization_factor(.1)
        self.assertEqual(plan.integrate1d(image).intensity, before)
        masked = ai.prepare_cuda1d(SHAPE, 40)
        self.assertEqual(masked.integrate1d(image).intensity, [0.0] * 40)

    def test_threads_share_a_plan(self):
        frames = self.frames(8)
        plan = integrator().prepare_cuda1d(SHAPE, 40)
        expected = [plan.integrate1d(f).intensity for f in frames]
        results = {}

        def run(index):
            for _ in range(5):
                results[index] = plan.integrate1d(frames[index]).intensity

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(frames))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([results[i] for i in range(len(frames))], expected)

    def assert_identical(self, batch, single):
        self.assertEqual(len(batch), len(single))
        for a, b in zip(batch, single):
            self.assertEqual(a.intensity, b.intensity)
            self.assertEqual(a.errors, b.errors)
            self.assertEqual(a.radial, b.radial)

    def test_launch_split_with_default_budget(self):
        """With the default budget, 300 and 512 frames fit in one chunk, so a
        chunk is split into launches of at most 256 frames (one partial and
        two full ones)."""
        frames = self.frames(512)
        plan = integrator("pyfai_full").prepare_cuda1d(SHAPE, 40)
        # One frame is 2048 * 12 + 40 * 16 = 25216 bytes; 512 of them need
        # 12.9 MB of the min(1 GiB, free / 2) budget.
        self.assertGreaterEqual(plan.max_batch_frames, 512)
        single = [plan.integrate1d(f, errors=True) for f in frames]
        for count in (300, 512):
            with self.subTest(frames=count):
                batch = plan.batch1d(list(frames[:count]), errors=True)
                self.assertEqual(len(batch), count)
                self.assert_identical(batch, single[:count])
                batch = plan.batch1d(frames[:count], errors=True)
                self.assertEqual(len(batch), count)
                self.assert_identical(batch, single[:count])

    def test_batch_dtypes_match_single(self):
        """float32 lists and stacks, and float64 stacks, through every
        precision: bit-identical to integrate1d on the same frames."""
        frames64 = self.frames(10)
        frames32 = frames64.astype(np.float32)
        for precision in ("f32", "f32_compensated", "f64"):
            plan = integrator().prepare_cuda1d(SHAPE, 40, precision=precision)
            for frames in (frames32, frames64):
                with self.subTest(precision=precision, dtype=frames.dtype.name):
                    single = [plan.integrate1d(f, errors=True) for f in frames]
                    self.assert_identical(plan.batch1d(list(frames), errors=True), single)
                    self.assert_identical(plan.batch1d(frames, errors=True), single)

    def test_budget_below_one_frame(self):
        """The documented frame size: pixels * (4 + 8) for the float32
        precisions (storage plus float64 staging), pixels * 8 for f64, plus
        n_points * 16 for the results."""
        pixels = SHAPE[0] * SHAPE[1]
        frames = self.frames(3)
        ai = integrator()
        for precision, per_pixel in (("f32", 12), ("f32_compensated", 12), ("f64", 8)):
            frame = pixels * per_pixel + 40 * 16
            with self.subTest(precision=precision):
                for budget in (1, frame - 1):
                    with self.assertRaisesRegex(ValueError, "max_batch_bytes"):
                        ai.prepare_cuda1d(SHAPE, 40, precision=precision,
                                          max_batch_bytes=budget)
                plan = ai.prepare_cuda1d(SHAPE, 40, precision=precision,
                                         max_batch_bytes=frame)
                self.assertEqual(plan.max_batch_frames, 1)
                self.assert_identical(plan.batch1d(frames),
                                      [plan.integrate1d(f) for f in frames])
                plan = ai.prepare_cuda1d(SHAPE, 40, precision=precision,
                                         max_batch_bytes=2 * frame + frame // 2)
                self.assertEqual(plan.max_batch_frames, 2)

    def test_non_finite_pixels_match_cpu(self):
        """NaN, +inf and -inf pixels give the same non-finite bins as the
        CPU, and the finite bins agree within the precision's tolerance."""
        base, mask = non_finite_image()
        for method in ("counting", "pixel_split", "pyfai_full"):
            ai = integrator(method)
            ai.set_mask(mask)
            for precision, rtol in (("f32", 2e-6), ("f32_compensated", 2e-6), ("f64", 1e-12)):
                plan = ai.prepare_cuda1d(SHAPE, 40, precision=precision)
                for dtype in (np.float64, np.float32):
                    image = base.astype(dtype)
                    reference = image if precision == "f64" else image.astype(np.float32)
                    for errors in (False, True):
                        with self.subTest(method=method, precision=precision,
                                          dtype=dtype.__name__, errors=errors):
                            cpu = ai.integrate1d(reference.astype(np.float64), 40,
                                                 errors=errors)
                            gpu = plan.integrate1d(image, errors=errors)
                            self.assert_non_finite(gpu.intensity, cpu.intensity, rtol, True)
                            if errors:
                                # A split weight w stored as float32 can be so
                                # small that w * w is 0 in float32, and 0 * inf
                                # is NaN where the CPU's float64 gives +inf.
                                # Only finiteness is compared there.
                                exact = precision == "f64" or method == "counting"
                                self.assert_non_finite(gpu.errors, cpu.errors, rtol, exact)
                            else:
                                self.assertIsNone(gpu.errors)

    def assert_non_finite(self, gpu, cpu, rtol, exact):
        gpu = np.asarray(gpu)
        cpu = np.asarray(cpu)
        if exact:
            for kind in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(kind(gpu), kind(cpu), kind.__name__)
        else:
            np.testing.assert_array_equal(np.isfinite(gpu), np.isfinite(cpu), "isfinite")
        finite = np.isfinite(cpu)
        self.assert_close(gpu[finite], cpu[finite], rtol)

    def test_plan_outlives_integrator(self):
        image = self.frames(1)[0]
        ai = integrator()
        plan = ai.prepare_cuda1d(SHAPE, 40)
        expected = plan.integrate1d(image, errors=True)
        radial = plan.radial
        del ai
        gc.collect()
        result = plan.integrate1d(image, errors=True)
        self.assertEqual(result.intensity, expected.intensity)
        self.assertEqual(result.errors, expected.errors)
        self.assertEqual(plan.radial, radial)
        self.assert_identical(plan.batch1d([image, image], errors=True), [expected] * 2)

    def test_masked_array_mask_is_ignored(self):
        image = self.frames(1)[0]
        masked = np.ma.masked_array(image, mask=image > 200)
        plan = integrator().prepare_cuda1d(SHAPE, 40)
        self.assertEqual(plan.integrate1d(masked).intensity, plan.integrate1d(image).intensity)

    def test_compensated_accumulation_is_tight(self):
        """Counting weights are exactly 1, so f32_compensated loses nothing
        to weight rounding: with float32-exact input its sums are double-word
        sums of positive terms, relative error below about 3e-11 even in the
        worst case, against the CPU's float64 sums. Plain float32 sums of
        these log-uniform values over 1e-3..1e5, about 256 per bin, are
        typically off by 1e-7; that is not asserted."""
        rng = np.random.default_rng(7)
        image = np.exp(rng.uniform(np.log(1e-3), np.log(1e5), SHAPE))
        reference = image.astype(np.float32).astype(np.float64)
        ai = integrator("counting")
        cpu = ai.integrate1d(reference, 8, errors=True)
        plan = ai.prepare_cuda1d(SHAPE, 8, precision="f32_compensated")
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__):
                gpu = plan.integrate1d(image.astype(dtype), errors=True)
                np.testing.assert_allclose(gpu.intensity, cpu.intensity, rtol=1e-9, atol=0)
                np.testing.assert_allclose(gpu.errors, cpu.errors, rtol=1e-9, atol=0)

    def test_native_integer_dtypes_and_pinned_batch(self):
        rng = np.random.default_rng(19)
        base = rng.integers(0, 5000, size=(5, *SHAPE), dtype=np.uint32)
        ai = integrator("pyfai_full")
        refs = [ai.integrate1d(f.astype(np.float64), 40, errors=True) for f in base]
        for precision, rtol in (("f32", 1e-5), ("f32_compensated", 1e-6), ("f64", 1e-12)):
            plan = ai.prepare_cuda1d(SHAPE, 40, precision=precision)
            for dtype in (np.uint16, np.uint32, np.int32):
                with self.subTest(precision=precision, dtype=dtype.__name__):
                    arr = np.ascontiguousarray(base, dtype=dtype)
                    single = [plan.integrate1d(f, errors=True) for f in arr]
                    pageable = plan.batch1d(arr, errors=True)
                    pinned = plan.alloc_pinned_batch(len(arr), dtype=dtype)
                    self.assertEqual(pinned.shape, arr.shape)
                    self.assertEqual(pinned.dtype, np.dtype(dtype))
                    self.assertTrue(pinned.flags.c_contiguous and pinned.flags.writeable)
                    pinned[:] = arr
                    from_pinned = plan.batch1d(pinned, errors=True)
                    self.assert_identical(pageable, single)
                    self.assert_identical(from_pinned, single)
                    for gpu, cpu in zip(from_pinned, refs):
                        self.assert_close(gpu.intensity, cpu.intensity, rtol)
                        self.assert_close(gpu.errors, cpu.errors, rtol)
        with self.assertRaises(TypeError):
            plan.alloc_pinned_batch(2, dtype=np.int64)
        with self.assertRaises(TypeError):
            plan.alloc_pinned_batch(2, dtype=">f4")

    def test_batch1d_bslz4_matches_batch1d_and_rejects_corrupt(self):
        import struct

        seen_offsets = {"eq1": False, "short": False, "long": False}

        def lz4_compress_block(payload: bytes) -> bytes:
            """Minimal valid LZ4 block compressor that emits real matches
            across offset == 1, 1 < offset < 32, and offset >= 32."""
            n = len(payload)
            out = bytearray()
            anchor = 0
            ip = 0
            # Candidate lookback distances covering all three decoder paths.
            candidates = (1, 2, 3, 4, 7, 13, 32, 64, 128, 256)

            def emit_len(val: int) -> None:
                while val >= 255:
                    out.append(255)
                    val -= 255
                out.append(val)

            while ip + 12 < n:
                best_len = 3
                best_off = 0
                for off in candidates:
                    if ip < off:
                        continue
                    ml = 0
                    limit = n - 5 - ip
                    while ml < limit and payload[ip + ml] == payload[ip - off + ml]:
                        ml += 1
                    if ml > best_len:
                        best_len = ml
                        best_off = off
                if best_off == 0:
                    ip += 1
                    continue
                if best_off == 1:
                    seen_offsets["eq1"] = True
                elif best_off < 32:
                    seen_offsets["short"] = True
                else:
                    seen_offsets["long"] = True
                lit_len = ip - anchor
                match_extra = best_len - 4
                token = (min(lit_len, 15) << 4) | min(match_extra, 15)
                out.append(token)
                if lit_len >= 15:
                    emit_len(lit_len - 15)
                out.extend(payload[anchor:ip])
                out.extend(struct.pack("<H", best_off))
                if match_extra >= 15:
                    emit_len(match_extra - 15)
                ip += best_len
                anchor = ip

            # Trailing literals (required by LZ4 block format).
            lit_len = n - anchor
            out.append(min(lit_len, 15) << 4)
            if lit_len >= 15:
                emit_len(lit_len - 15)
            out.extend(payload[anchor:n])
            return bytes(out)

        def encode_bslz4(arr: np.ndarray, block_bytes: int = 8192) -> bytes:
            raw = arr.tobytes()
            elem_bytes = arr.dtype.itemsize
            n_elems = arr.size
            block_elems = block_bytes // elem_bytes
            out = bytearray(struct.pack(">QI", len(raw), block_bytes))

            def bshuf_block(elems: np.ndarray) -> bytes:
                # Pack bits per bit-plane (LSB first within each byte).
                bits = np.unpackbits(
                    elems.view(np.uint8).reshape(-1, elem_bytes),
                    axis=1,
                    bitorder="little",
                )  # shape (N, elem_bytes * 8)
                planes = np.packbits(bits.T, axis=1, bitorder="little")
                return planes.tobytes()

            flat = arr.ravel()
            pos = 0
            while pos + block_elems <= n_elems:
                comp = lz4_compress_block(bshuf_block(flat[pos : pos + block_elems]))
                out.extend(struct.pack(">I", len(comp)))
                out.extend(comp)
                pos += block_elems
            rem_elems = n_elems - pos
            rem_comp = (rem_elems // 8) * 8
            if rem_comp > 0:
                comp = lz4_compress_block(bshuf_block(flat[pos : pos + rem_comp]))
                out.extend(struct.pack(">I", len(comp)))
                out.extend(comp)
                pos += rem_comp
            if pos < n_elems:
                out.extend(flat[pos:].tobytes())
            return bytes(out)

        rng = np.random.default_rng(23)
        ai = integrator("pyfai_full")
        plan = ai.prepare_cuda1d(SHAPE, 40, precision="f32_compensated")
        for dtype, name in ((np.uint16, "uint16"), (np.uint32, "uint32"), (np.int32, "int32")):
            with self.subTest(dtype=name):
                frames = rng.integers(0, 4000, size=(5, *SHAPE), dtype=dtype)
                # Structured patterns so LZ4 emits offset==1, 1<offset<32, and offset>=32 matches.
                half_r = SHAPE[0] // 2
                frames[0, :half_r, :] = 0
                pat24 = rng.integers(1, 4000, size=24, dtype=dtype)
                frames[1, :half_r, :] = np.resize(pat24, (half_r, SHAPE[1]))
                frames[2, half_r : 2 * half_r, :] = frames[2, :half_r, :]
                chunks = [encode_bslz4(f) for f in frames]
                expected = plan.batch1d(frames, errors=True)
                actual = plan.batch1d_bslz4(chunks, dtype=name, errors=True)
                self.assert_identical(actual, expected)
        self.assertTrue(all(seen_offsets.values()), f"missing LZ4 offset coverage: {seen_offsets}")
        self.assertEqual(plan.batch1d_bslz4([], dtype="uint32"), [])
        with self.assertRaises(ValueError):
            plan.batch1d_bslz4([b"short"], dtype="uint32")
        # Over-bound comp_len must be rejected on host.
        over_bound = 8192 + 8192 // 255 + 17
        bad_chunk = struct.pack(">QII", SHAPE[0] * SHAPE[1] * 4, 8192, over_bound) + b"\x00" * over_bound
        with self.assertRaises(ValueError):
            plan.batch1d_bslz4([bad_chunk], dtype="uint32")
        with self.assertRaises(TypeError):
            plan.batch1d_bslz4([b"0" * 16], dtype="float32")


if __name__ == "__main__":
    unittest.main()
