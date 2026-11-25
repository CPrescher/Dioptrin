#!/bin/bash
# Dioptrin 0.16.0 paper rerun on the MacBook (M4 Pro, Metal).
# Protocol: docs/benchmarks/2026-11-25-paper-0.16.0/README.md
#
# Usage, from anywhere inside the checkout:
#   bash benchmarks/comparison/paper-0.16.0/run-macos.sh             # all passes
#   bash benchmarks/comparison/paper-0.16.0/run-macos.sh --quick     # quick check only
#   bash benchmarks/comparison/paper-0.16.0/run-macos.sh --dry-run   # print commands
#
# The passes run in order; each writes its own directory under
# benchmark-results/ and its own <name>.log. A pass whose result directory
# already exists is skipped (compare.py refuses to write into an existing
# directory), so after an interruption delete the unfinished pass's directory
# and start the script again. A failed pass does not stop the next one.
# Start and exit lines with time stamps go to the terminal and to
# benchmark-results/run-macos-0.16.0.log.
#
# The three passes, their options and output names are those of the run started
# on 25 November at commit 8483c77 from an equivalent untracked script.
# compare.py re-executes itself with -E -s, so calling the venv python without
# those flags (as here and in that run) gives the same interpreter flags as the
# Windows script.
set -u

quick=0
dry_run=0
for arg in "$@"; do
    case $arg in
        --quick) quick=1 ;;
        --dry-run) dry_run=1 ;;
        -h | --help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg (use --quick, --dry-run)" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/../../.." || exit 1
PY=.venv-benchmark/bin/python
CMP=benchmarks/comparison/compare.py
OUT=benchmark-results
RUNLOG=$OUT/run-macos-0.16.0.log
CPU="dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full"
ALL="$CPU azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp pyfai-full-opencl \
 dioptrin-pixel_split-metal dioptrin-pyfai_full-metal \
 dioptrin-pixel_split-metal-batch dioptrin-pyfai_full-metal-batch"
GPU="--opencl-device gpu --metal-device 0 --gpu-precision f32_compensated"
CAKES="--dioptrin-2d-path batch --dioptrin-2d-batch-split nested"

if [[ ! -x $PY ]]; then
    if [[ $dry_run == 1 ]]; then
        echo "note: $PY not found; printing the commands only" >&2
    else
        echo "error: $PY not found in $PWD." >&2
        echo "Build the environment first:" >&2
        echo "  python3 benchmarks/comparison/setup.py --with-metal --with-opencl --azint-fast-math" >&2
        exit 1
    fi
fi
[[ $dry_run == 1 ]] || mkdir -p "$OUT"

log() {
    echo "$*"
    [[ $dry_run == 1 ]] || echo "$*" >> "$RUNLOG"
}

run() {
    local name=$1
    shift
    if [[ $dry_run == 1 ]]; then
        echo "$PY $CMP --output $OUT/$name $* > $OUT/$name.log 2>&1"
        return
    fi
    if [[ -e $OUT/$name ]]; then
        log "=== $name skipped: $OUT/$name already exists $(date)"
        return
    fi
    log "=== $name start $(date)"
    "$PY" "$CMP" --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
    log "=== $name exit $? $(date)"
}

# Word splitting of $CPU, $ALL, $GPU and $CAKES is intended below.
# shellcheck disable=SC2086
if [[ $quick == 1 ]]; then
    # Installation check (a few minutes): every variant of the passes below on
    # the small synthetic dataset at 1 and 2 workers.
    run 2026-11-25-macos-quick --suite quick --dimensions 1d 2d $CAKES $GPU --variants $ALL
    exit 0
fi

# shellcheck disable=SC2086
{
    # 1. Call-path ablation, Dioptrin CPU only (1 trial): default batch1d against
    #    the split call, and one integrate2d per frame against batch2d (pass 2).
    run 2026-11-25-macos-ablation-paths --dimensions 1d 2d \
        --dioptrin-1d-path global split --dioptrin-2d-path sequential --variants $CPU
    # 2. Primary float64 pass, 3 trials of 3 x 1 s blocks.
    run 2026-11-25-macos-f64 --dimensions 1d 2d --trials 3 --seconds 1.0 $CAKES $GPU --variants $ALL
    # 3. float32 input pass (suite defaults: 1 trial, 3 x 0.3 s).
    run 2026-11-25-macos-f32 --dimensions 1d 2d --input-dtype float32 $CAKES $GPU --variants $ALL
}
[[ $dry_run == 1 ]] || log "=== all done $(date)"
