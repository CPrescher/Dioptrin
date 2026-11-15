#!/usr/bin/env python3
"""Compare the criterion benches of two revisions and print a table.

    python3 scripts/bench_compare.py v0.12.0 HEAD --benches methods detectors multi
    RAYON_NUM_THREADS=1 python3 scripts/bench_compare.py main HEAD \\
        --benches methods plans --filter 'methods/build1d/|plans/fixed1d/' --rounds 3

Each revision is checked out into its own temporary git worktree. The bench
sources (``benches/`` and the ``[[bench]]`` entries) are taken from this
checkout for both revisions, so revisions older than the benches can be timed
with the same code; a revision whose API differs from what the benches call
fails to build, and then only benches that compile there can be compared.
Both revisions are built first (one shared target directory, the bench
binaries copied out), then the binaries run in rounds, alternating which goes
first (A B, B A, A B, ...), each round saving a criterion baseline per
revision under one ``CRITERION_HOME``. The table gives, per benchmark, the
median over rounds of criterion's median time for each revision, B/A, and
the spread between rounds (or criterion's 95% interval with one round).

Limits: criterion cannot interleave two binaries sample by sample, so drift
in machine load between runs shows up as a difference; more rounds and a
quiet machine help, and the spread column says how much the rounds disagreed.
Use ``RAYON_NUM_THREADS=1`` for kernel costs without thread scheduling noise.
Run from a clean shell: the script passes the environment on unchanged.
Revisions are commits: uncommitted changes to the library are not included
(only this checkout's benches are used, modified or not).

Outputs go to ``target/bench-compare/<timestamp>/`` (``comparison.md``,
``comparison.json``, the criterion data and per-run logs). The worktrees and
the shared target directory are removed afterwards unless ``--keep`` is given.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Every target. methods, detectors and multi (and the older build, integrate,
# integrator) also compile against v0.12.0; azimuthal, plans and massif need
# 0.13 APIs, and batch needs process_streaming (after 0.13).
DEFAULT_BENCHES = ["methods", "azimuthal", "detectors", "plans", "multi", "batch", "massif"]


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def load():
    """One-minute load average, or None where the platform has none."""
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def overlay_benches(worktree, benches):
    """Copy this checkout's benches/ into the worktree and declare each bench."""
    target = worktree / "benches"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / "benches", target)
    manifest = worktree / "Cargo.toml"
    text = manifest.read_text()
    for name in benches:
        if not re.search(r'\[\[bench\]\]\s*\nname\s*=\s*"%s"' % re.escape(name), text):
            text += f'\n[[bench]]\nname = "{name}"\nharness = false\n'
    manifest.write_text(text)
    # The benches read fixtures from test_data/. Fill in files the revision
    # lacks, or has only as Git LFS pointers (no git-lfs installed), from
    # this checkout.
    for source in (ROOT / "test_data").iterdir():
        dest = worktree / "test_data" / source.name
        if not source.is_file():
            continue
        if not dest.exists() or dest.read_bytes()[:40].startswith(b"version https://git-lfs"):
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)


def build(worktree, benches, target_dir, bin_dir):
    """Build the bench binaries of one worktree and copy them to bin_dir."""
    cmd = ["cargo", "bench", "--no-run", "--message-format=json"]
    for name in benches:
        cmd += ["--bench", name]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    print("+", " ".join(cmd), f"(in {worktree})", flush=True)
    proc = subprocess.run(cmd, cwd=worktree, env=env, check=True, stdout=subprocess.PIPE, text=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") and "bench" in msg["target"]["kind"]:
            name = msg["target"]["name"]
            if name in benches:
                dest = bin_dir / name
                shutil.copy2(msg["executable"], dest)
                found[name] = dest
    missing = set(benches) - set(found)
    if missing:
        sys.exit(f"no executable for {sorted(missing)} in {worktree}")
    return found


def collect(criterion_home):
    """{full_id: {baseline: estimates}} from a criterion directory."""
    data = {}
    for bench_json in criterion_home.rglob("benchmark.json"):
        baseline = bench_json.parent.name
        estimates = bench_json.parent / "estimates.json"
        if baseline in ("new", "base", "change") or not estimates.exists():
            continue
        info = json.loads(bench_json.read_text())
        # The title is the full ID, with " #2" etc. appended when an older
        # target registers the same ID twice.
        full_id = info.get("title") or info["full_id"]
        data.setdefault(full_id, {})[baseline] = json.loads(estimates.read_text())
    return data


def load_range(runs):
    loads = [r[k] for r in runs for k in ("load_start", "load_end") if r[k] is not None]
    return f"{min(loads)} to {max(loads)}" if loads else "unknown"


def fmt_time(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def summarise(data, labels, rounds):
    rows = []
    for full_id in sorted(data):
        entry = {"id": full_id}
        for label in labels:
            ests = [data[full_id].get(f"{label}-r{r}") for r in range(1, rounds + 1)]
            ests = [e for e in ests if e]
            if not ests:
                entry[label] = None
                continue
            medians = [e["median"]["point_estimate"] for e in ests]
            mid = median(medians)
            if len(ests) > 1:
                lo, hi = min(medians), max(medians)
            else:
                ci = ests[0]["median"]["confidence_interval"]
                lo, hi = ci["lower_bound"], ci["upper_bound"]
            entry[label] = {"median_ns": mid, "low_ns": lo, "high_ns": hi, "rounds": medians}
        a, b = (entry[labels[0]], entry[labels[1]])
        if a and b:
            entry["ratio"] = b["median_ns"] / a["median_ns"]
            entry["overlap"] = a["low_ns"] <= b["high_ns"] and b["low_ns"] <= a["high_ns"]
        rows.append(entry)
    return rows


def table(rows, labels, rounds):
    spread = "range over rounds" if rounds > 1 else "95% CI of median"
    lines = [
        f"| benchmark | {labels[0]} | {labels[1]} | B/A | {labels[0]} {spread} | {labels[1]} {spread} | |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        a, b = row[labels[0]], row[labels[1]]
        if not (a and b):
            have = labels[0] if a else labels[1] if b else "neither"
            lines.append(f"| `{row['id']}` | | | | | | only in {have} |")
            continue
        flag = "overlaps" if row["overlap"] else ("**slower**" if row["ratio"] > 1 else "**faster**")
        lines.append(
            f"| `{row['id']}` | {fmt_time(a['median_ns'])} | {fmt_time(b['median_ns'])} | {row['ratio']:.3f} | "
            f"{fmt_time(a['low_ns'])} – {fmt_time(a['high_ns'])} | {fmt_time(b['low_ns'])} – {fmt_time(b['high_ns'])} | {flag} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rev_a", help="baseline revision (any git revision)")
    parser.add_argument("rev_b", help="revision to compare with it")
    parser.add_argument("--benches", nargs="+", default=DEFAULT_BENCHES,
                        help=f"bench targets (default: {' '.join(DEFAULT_BENCHES)})")
    parser.add_argument("--filter", default=None, help="criterion filter (a regex on benchmark IDs)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="alternating rounds per revision (default: 3; odd counts give a true median)")
    parser.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"))
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: target/bench-compare/<timestamp>)")
    parser.add_argument("--keep", action="store_true", help="keep the worktrees and the target directory")
    parser.add_argument("--report-only", action="store_true",
                        help="only rebuild the table from an existing --out directory")
    args = parser.parse_args()
    labels = args.labels
    if labels[0] == labels[1] or args.rounds < 1:
        parser.error("labels must differ and --rounds must be positive")

    if args.report_only and args.out is None:
        parser.error("--report-only needs --out")
    out = (args.out or ROOT / "target" / "bench-compare" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    criterion_home = out / "criterion"
    if not args.report_only:
        if out.exists() and any(out.iterdir()):
            parser.error(f"{out} is not empty")
        out.mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir()
        meta = {"revisions": {}, "benches": args.benches, "filter": args.filter, "rounds": args.rounds,
                "labels": labels, "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
                "bench_source": run(["git", "rev-parse", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE,
                                    text=True).stdout.strip(),
                "bench_source_dirty": bool(run(["git", "status", "--porcelain", "benches", "Cargo.toml"], cwd=ROOT,
                                               stdout=subprocess.PIPE, text=True).stdout.strip()),
                "runs": []}
        worktrees, binaries = {}, {}
        try:
            for label, rev in zip(labels, (args.rev_a, args.rev_b)):
                commit = run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=ROOT,
                             stdout=subprocess.PIPE, text=True).stdout.strip()
                meta["revisions"][label] = {"rev": rev, "commit": commit}
                wt = out / f"src-{label}"
                run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=ROOT)
                worktrees[label] = wt
                overlay_benches(wt, args.benches)
                binaries[label] = build(wt, args.benches, out / "target", out / f"bin-{label}")
            if not args.keep:
                shutil.rmtree(out / "target", ignore_errors=True)
            env = dict(os.environ, CRITERION_HOME=str(criterion_home))
            for r in range(1, args.rounds + 1):
                order = labels if r % 2 else labels[::-1]
                for label in order:
                    for name in args.benches:
                        cmd = [str(binaries[label][name]), "--bench", "--save-baseline", f"{label}-r{r}"]
                        if args.filter:
                            cmd.append(args.filter)
                        start_load = load()
                        log = out / "logs" / f"{label}-r{r}-{name}.log"
                        print(f"+ round {r} {label} {name} (load {start_load})", flush=True)
                        with open(log, "w") as handle:
                            subprocess.run(cmd, cwd=worktrees[label], env=env, check=True,
                                           stdout=handle, stderr=subprocess.STDOUT)
                        meta["runs"].append({"round": r, "label": label, "bench": name,
                                             "load_start": start_load, "load_end": load()})
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
        finally:
            if not args.keep:
                for wt in worktrees.values():
                    subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT)
                shutil.rmtree(out / "target", ignore_errors=True)

    meta = json.loads((out / "meta.json").read_text())
    rows = summarise(collect(criterion_home), meta["labels"], meta["rounds"])
    revs = meta["revisions"]
    header = (f"{meta['labels'][0]} = {revs[meta['labels'][0]]['rev']} ({revs[meta['labels'][0]]['commit'][:10]}), "
              f"{meta['labels'][1]} = {revs[meta['labels'][1]]['rev']} ({revs[meta['labels'][1]]['commit'][:10]}); "
              f"benches from {meta['bench_source'][:10]}{' (modified)' if meta['bench_source_dirty'] else ''}; "
              f"{meta['rounds']} round(s); RAYON_NUM_THREADS={meta['RAYON_NUM_THREADS'] or 'unset'}; "
              f"one-minute load during runs {load_range(meta['runs'])}")
    text = header + "\n\n" + table(rows, meta["labels"], meta["rounds"]) + "\n"
    (out / "comparison.md").write_text(text)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    print()
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
