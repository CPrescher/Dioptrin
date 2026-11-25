"""plot_scaling.py keeps the Dioptrin 1D calls apart."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import plot_scaling
except ImportError:  # matplotlib is optional for the harness
    plot_scaling = None


@unittest.skipIf(plot_scaling is None, "matplotlib not installed")
class PlotScalingTests(unittest.TestCase):
    def write(self, cases, name, variant, workers, rate, path="absent"):
        row = dict(
            dataset="pilatus",
            workload="repeated",
            variant=variant,
            workers=workers,
            frames=8,
            error_mode="poisson",
            frames_per_s=rate,
        )
        if path != "absent":
            row["dioptrin_1d_path"] = path
        (cases / f"{name}.result.json").write_text(json.dumps(row))

    def plotted(self, directory, **kwargs):
        """(variant label, workers, median rate) of every errorbar drawn."""
        drawn = []
        original = plot_scaling.plt.Axes.errorbar

        def record(ax, x, y, *args, **kw):
            drawn.extend(zip(x, y))
            return original(ax, x, y, *args, **kw)

        with tempfile.TemporaryDirectory() as out, patch.object(
            plot_scaling.plt.Axes, "errorbar", record
        ):
            plot_scaling.plot_cpu(directory, Path(out), "poisson", **kwargs)
            names = sorted(p.name for p in Path(out).iterdir())
        return sorted(drawn), names

    def test_paths_are_not_pooled_and_old_rows_still_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cases = directory / "cases"
            cases.mkdir()
            for w, rate in ((1, 10.0), (2, 20.0)):
                self.write(cases, f"g{w}", "dioptrin-pyfai_full", w, rate, "global")
                self.write(cases, f"s{w}", "dioptrin-pyfai_full", w, 100 * rate, "split")
                self.write(cases, f"p{w}", "pyfai-full", w, rate / 10, None)
            drawn, names = self.plotted(directory)
            self.assertEqual(drawn, [(1, 1.0), (1, 10.0), (2, 2.0), (2, 20.0)])
            self.assertIn("cpu-scaling-poisson.png", names)
            drawn, names = self.plotted(directory, dioptrin_1d_path="split")
            self.assertEqual(drawn, [(1, 1.0), (1, 1000.0), (2, 2.0), (2, 2000.0)])
            self.assertIn("cpu-scaling-poisson-split.png", names)

            # A run recorded before the field existed: Dioptrin rows are global.
            for path in cases.iterdir():
                path.unlink()
            for w, rate in ((1, 10.0), (2, 20.0)):
                self.write(cases, f"g{w}", "dioptrin-pyfai_full", w, rate)
                self.write(cases, f"p{w}", "pyfai-full", w, rate / 10)
            drawn, _ = self.plotted(directory)
            self.assertEqual(drawn, [(1, 1.0), (1, 10.0), (2, 2.0), (2, 20.0)])


if __name__ == "__main__":
    unittest.main()
