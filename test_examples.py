#!/usr/bin/env python3
"""
Test script to verify all Python examples run correctly.
Used in CI to catch breaking changes.
"""

import sys
import subprocess
from pathlib import Path


def run_example(script_path: Path) -> bool:
    """Run an example script and return True if successful."""
    print(f"\n{'='*70}")
    print(f"Testing: {script_path.name}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per example
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"❌ FAILED with exit code {result.returncode}")
            print("STDERR:", result.stderr)
            return False

        print(f"✅ PASSED")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Timeout after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all example tests."""
    examples_dir = Path(__file__).parent / "python_examples"

    # Automatically discover all Python scripts in python_examples/
    # Skip __init__.py and any scripts starting with underscore
    examples = sorted([
        f.name for f in examples_dir.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    ])

    if not examples:
        print("❌ No example scripts found in python_examples/")
        return 1

    print("Testing Python Examples")
    print("="*70)
    print(f"Found {len(examples)} example(s) to test")
    print()

    # Check dependencies
    try:
        import dioptrin
        print("✓ dioptrin imported successfully")
    except ImportError:
        print("❌ dioptrin not installed. Run: maturin develop --features python")
        return 1

    try:
        import h5py
        print("✓ h5py imported successfully")
    except ImportError:
        print("⚠️  h5py not installed. HDF5 examples will be skipped.")
        print("   Install with: pip install h5py")

    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError:
        print("❌ numpy not installed. Run: pip install numpy")
        return 1

    print()

    results = {}

    # Run all examples from examples/ directory
    for example in examples:
        example_path = examples_dir / example
        if example_path.exists():
            results[example] = run_example(example_path)
        else:
            print(f"⚠️  Example not found: {example_path}")
            results[example] = None

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"{status:10} {name}")

    print()
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        print("\n❌ Some examples failed!")
        return 1
    elif passed == 0:
        print("\n⚠️  No examples were run successfully")
        return 1
    else:
        print("\n✅ All examples passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
