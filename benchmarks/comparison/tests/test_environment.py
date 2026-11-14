"""Persistent cache invalidation, publication, and concurrent construction."""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment import cache_key, cpu_identity, ensure_cached, fingerprint


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
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                ensure_cached(
                    root,
                    {"cpu": "a"},
                    lambda generation: None,
                    lambda generation: False,
                )
            self.assertEqual(list(root.glob("*/ready.json")), [])

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
            root = Path(temporary)
            (root / "src").mkdir()
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
                (root / "src/lib.rs").write_text("changed library")
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))
                (root / "src/lib.rs").write_text("library")
                (recipe / "requirements.txt").write_text("changed dependency")
                self.assertNotEqual(original, cache_key(fingerprint(root, "stable")))


if __name__ == "__main__":
    unittest.main()
