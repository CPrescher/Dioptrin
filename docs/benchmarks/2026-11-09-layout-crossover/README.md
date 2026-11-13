# Layout benchmark utilities

Keep new run metadata, raw timings, fitted models, tables, and plots outside
version control. Use an external run directory, or an ignored directory such
as `target/benchmarks/layout-crossover`.

`analyze.py RUN_DIRECTORY` reads `inputs.json` and `results.jsonl` from that
directory and writes its summaries, fitted model, and plots alongside them.
The input manifest supplies `bins` and an `images` list with each image's `id`
and `pixels`; the timing records supply the layout samples for each case.
The analysis requires NumPy and Matplotlib.

`predict.py --pixels PIXELS --bins BINS` prints the broad production estimate;
optional flags are `--threads`, `--poisson`, and `--batch`. The production
selector uses this estimate rather than a fitted benchmark model.

The in-memory integration comparison remains reproducible from the repository
root with the bundled CeO2 fixture:

```sh
mkdir -p target/benchmarks/consolidation
RAYON_NUM_THREADS=8 cargo run --release --example benchmark_consolidated \
  > target/benchmarks/consolidation/results.json
```

Record the machine, thread allocation, input/settings, setup costs, cache state,
and timing scope when reporting results in a PR. Keep frozen numerical reference
fixtures with the tests; they serve a different purpose from benchmark timings.
