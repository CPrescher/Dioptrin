"""Persistent cache invalidation, publication, and concurrent construction."""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import subprocess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment import (
    cache_key,
    cpu_identity,
    ensure_cached,
    fingerprint,
    probe_environment,
    sanitized_build_environment,
)


def concurrent_request(root, barrier, queue):
    root = Path(root)

    def build(generation):
        with (root / "build-count").open("a") as stream:
            stream.write("build\n")
        time.sleep(0.2)
        (generation / "complete").write_text("ok")

    barrier.wait(timeout=15)
    result = ensure_cached(
        root,
        {"cpu": "same", "source": "same"},
        build,
        lambda generation: (generation / "complete").is_file(),
    )
    queue.put(result)


def concurrent_warm_request(root, barrier, queue):
    def validate(generation):
        barrier.wait(timeout=10)
        return (generation / "complete").is_file()

    def build(generation):
        raise AssertionError("warm request unexpectedly rebuilt")

    queue.put(ensure_cached(Path(root), {"cpu": "warm"}, build, validate))


class EnvironmentTests(unittest.TestCase):
    def test_native_cpu_identity_ignores_counts_but_rejects_mixed_instruction_sets(
        self,
    ):
        block = "processor: {processor}\nvendor_id: Intel\ncpu family: 6\nmodel: 85\nmodel name: Xeon\nstepping: 7\nflags: {flags}\n"
        info = (
            block.format(processor=0, flags="avx avx2")
            + "\n"
            + block.format(processor=1, flags="avx avx2")
        )
        with (
            patch("environment.Path.is_file", return_value=True),
            patch("environment.Path.read_text", return_value=info) as read,
            patch(
                "environment.os.sched_getaffinity", return_value={0, 1}, create=True
            ) as affinity,
        ):
            both = cpu_identity()
            affinity.return_value = {1}
            self.assertEqual(cpu_identity(), both)
            affinity.return_value = {0, 1}
            read.return_value = (
                block.format(processor=0, flags="avx avx2")
                + "\n"
                + block.format(processor=1, flags="avx")
            )
            with self.assertRaisesRegex(RuntimeError, "homogeneous"):
                cpu_identity()

    def test_reuse_repair_and_force_preserve_active_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builds = []

            def build(generation):
                builds.append(generation)
                (generation / "complete").write_text("ok")

            def validate(generation):
                return (generation / "complete").is_file()

            first = ensure_cached(root, {"cpu": "a"}, build, validate)
            second = ensure_cached(root, {"cpu": "a"}, build, validate)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["python"], second["python"])
            self.assertEqual(len(builds), 1)
            forced = ensure_cached(root, {"cpu": "a"}, build, validate, rebuild=True)
            self.assertNotEqual(first["generation"], forced["generation"])
            self.assertTrue(Path(first["generation"]).is_dir())
            (Path(forced["generation"]) / "complete").unlink()
            repaired = ensure_cached(root, {"cpu": "a"}, build, validate)
            self.assertFalse(repaired["reused"])
            self.assertEqual(len(builds), 3)

    def test_failed_build_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fail(generation):
                raise RuntimeError("failed setup")

            with self.assertRaisesRegex(RuntimeError, "failed setup"):
                ensure_cached(root, {"cpu": "a"}, fail, lambda generation: False)
            self.assertEqual(list(root.glob("*/ready.json")), [])
            self.assertEqual(list(root.glob("*/env-*")), [])
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                ensure_cached(
                    root,
                    {"cpu": "a"},
                    lambda generation: None,
                    lambda generation: False,
                )
            self.assertEqual(list(root.glob("*/ready.json")), [])
            self.assertEqual(list(root.glob("*/env-*")), [])

    def test_failed_forced_build_keeps_previous_ready_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def valid(generation):
                return (generation / "complete").is_file()

            first = ensure_cached(
                root,
                {"cpu": "same"},
                lambda generation: (generation / "complete").write_text("ok"),
                valid,
            )

            def fail(generation):
                (generation / "partial").write_text("incomplete build")
                raise RuntimeError("failed partway")

            with self.assertRaisesRegex(RuntimeError, "failed partway"):
                ensure_cached(root, {"cpu": "same"}, fail, valid, rebuild=True)
            again = ensure_cached(root, {"cpu": "same"}, fail, valid)
            self.assertTrue(again["reused"])
            self.assertEqual(again["python"], first["python"])
            self.assertEqual(len(list(root.glob("*/env-*"))), 1)
            self.assertTrue((Path(first["generation"]) / "published.json").is_file())

    def test_warm_validation_is_concurrent_without_build_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_cached(
                root,
                {"cpu": "warm"},
                lambda generation: (generation / "complete").write_text("ok"),
                lambda generation: (generation / "complete").is_file(),
            )
            context = multiprocessing.get_context("spawn")
            barrier, queue = context.Barrier(2), context.Queue()
            processes = [
                context.Process(
                    target=concurrent_warm_request, args=(temporary, barrier, queue)
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(r["reused"] for r in results))

    def test_unkeyed_build_overrides_are_removed(self):
        unexpected = {
            "CARGO_ENCODED_RUSTFLAGS": "bad",
            "CARGO_PROFILE_RELEASE_LTO": "false",
            "CARGO_BUILD_RUSTFLAGS": "bad",
            "RUSTUP_TOOLCHAIN": "nightly",
            "MATURIN_PYTHON": "wrong",
            "PYO3_CONFIG_FILE": "wrong",
            "PIP_INDEX_URL": "wrong",
            "PYTHONPATH": "wrong",
        }
        with patch.dict(
            os.environ,
            unexpected | {"CC": "gcc", "CFLAGS": "-O2", "HTTPS_PROXY": "proxy"},
        ):
            env = sanitized_build_environment()
            self.assertTrue(set(unexpected).isdisjoint(env))
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
            self.assertEqual(env["CC"], "gcc")
            self.assertEqual(env["CFLAGS"], "-O2")
            self.assertEqual(env["HTTPS_PROXY"], "proxy")

    def test_probe_timeout_and_failure_report_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "venv/bin/python",
                "venv/benchmark-build.json",
                "build/python-packages.txt",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            with patch(
                "environment.subprocess.run",
                side_effect=subprocess.TimeoutExpired("probe", 0.01),
            ):
                result = probe_environment(root, root, timeout=0.01)
                self.assertFalse(result)
                self.assertIn("timed out", result.reason)
            with patch(
                "environment.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "stale binary", ""),
            ):
                result = probe_environment(root, root)
                self.assertFalse(result)
                self.assertIn("stale binary", result.reason)

    def test_parallel_jobs_build_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            barrier, queue = context.Barrier(2), context.Queue()
            processes = [
                context.Process(
                    target=concurrent_request, args=(temporary, barrier, queue)
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(Path(temporary, "build-count").read_text(), "build\n")
            self.assertEqual(sorted(r["reused"] for r in results), [False, True])
            self.assertEqual(results[0]["python"], results[1]["python"])

    def test_cpu_python_compiler_source_and_recipe_changes_invalidate(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"CC": "gcc", "CXX": "g++"}),
        ):
            root = Path(temporary) / "repo"
            (root / "src").mkdir(parents=True)
            (root / "src/lib.rs").write_text("library")
            (root / "Cargo.toml").write_text("cargo")
            (root / "pyproject.toml").write_text("python")
            recipe = root / "benchmarks/comparison"
            recipe.mkdir(parents=True)
            for name in (
                "environment.py",
                "setup.py",
                "requirements.txt",
                "provenance.py",
            ):
                (recipe / name).write_text(name)
            with (
                patch(
                    "environment.cpu_identity",
                    return_value={"model": "a", "flags": ["avx2"]},
                ) as cpu,
                patch(
                    "environment.subprocess.check_output", return_value="gcc version a"
                ) as compiler,
                patch(
                    "environment.sysconfig.get_config_var", return_value="abi-a"
                ) as abi,
                patch("environment.platform.libc_ver", return_value=("libc", "1")),
            ):
                original = cache_key(fingerprint(root, "stable"))
                cpu.return_value = {"model": "b", "flags": ["avx2"]}
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                cpu.return_value = {"model": "a", "flags": ["avx2"]}
                compiler.return_value = "gcc version b"
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                compiler.return_value = "gcc version a"
                abi.return_value = "abi-b"
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                abi.return_value = "abi-a"
                self.assertNotEqual(original, cache_key(fingerprint(root, "nightly")))
                with patch("environment.sys.executable", str(root / "another-python")):
                    self.assertNotEqual(
                        original, cache_key(fingerprint(root, "stable"))
                    )
                with patch("environment.sys.base_prefix", str(root / "another-prefix")):
                    self.assertNotEqual(
                        original, cache_key(fingerprint(root, "stable"))
                    )
                with patch.dict(os.environ, {"LDFLAGS": "-Wl,new"}):
                    self.assertNotEqual(
                        original, cache_key(fingerprint(root, "stable"))
                    )
                config = root.parent / ".cargo/config.toml"
                config.parent.mkdir()
                config.write_text('[build]\nrustflags=["-Copt-level=2"]\n')
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                config.unlink()
                (root / "build.rs").write_text("fn main() {}")
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                (root / "build.rs").unlink()
                (root / "src/lib.rs").write_text("changed library")
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                (root / "src/lib.rs").write_text("library")
                (recipe / "requirements.txt").write_text("changed dependency")
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))


if __name__ == "__main__":
    unittest.main()
