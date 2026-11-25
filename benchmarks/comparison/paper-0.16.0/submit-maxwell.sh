#!/bin/bash
# Dioptrin 0.16.0 paper rerun on DESY Maxwell (EPYC 9475F + H200 NVL).
# Protocol: docs/benchmarks/2026-11-25-paper-0.16.0/README.md
#
# Run on a login node from the root of the Dioptrin checkout:
#   bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh quick   # quick check job only
#   bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh all     # every job
#   bash benchmarks/comparison/paper-0.16.0/submit-maxwell.sh all --dry-run
#
# "all" submits the quick check too unless its result directory already exists,
# then the nine benchmark jobs. The jobs are independent and may run at the
# same time; environment.py serialises the build of a shared environment with
# a file lock, so jobs that start together do not build it twice.
#
# Environment variables:
#   PARTITION            Slurm partition (default: upex)
#   CONSTRAINT           node constraint (default: H200&GPUx1)
#   RESULTS              results root (default: $PWD/benchmark-results)
#   PYFAI_PATCHED_WHEEL  path to the patched pyFAI 2026.9.0 wheel for the
#                        2026-11-25-maxwell-f64-2d-pyfai-patched job; when unset
#                        or missing, that job is skipped and the rest submitted.
#
# A job whose result directory already exists is skipped (compare.py refuses
# to write into an existing directory). Submitted job ids are printed and
# appended to $RESULTS/maxwell/jobs-2026-11-25.txt.
set -euo pipefail

usage() {
    sed -n '2,25p' "$0" >&2
    exit 2
}

mode=
dry_run=0
for arg in "$@"; do
    case $arg in
        quick | all) mode=$arg ;;
        --dry-run) dry_run=1 ;;
        -h | --help) usage ;;
        *)
            echo "unknown argument: $arg" >&2
            usage
            ;;
    esac
done
[[ -n $mode ]] || usage

if [[ ! -f benchmarks/comparison/maxwell.sbatch ]]; then
    echo "error: run this from the root of the Dioptrin checkout" \
        "(benchmarks/comparison/maxwell.sbatch not found in $PWD)" >&2
    exit 1
fi

PARTITION=${PARTITION:-upex}
CONSTRAINT=${CONSTRAINT:-H200&GPUx1}
R=${RESULTS:-$PWD/benchmark-results}
case $R in
    /*) ;;
    *) R=$PWD/$R ;;
esac

# maxwell.sbatch reads these from the submission environment (sbatch exports
# the whole environment by default). PYFAI_WHEEL is set only for the patched
# job, so a value left in the login shell must not reach the other jobs.
export WITH_OPENCL=1 WITH_CUDA=1 AZINT_FAST_MATH=1
unset PYFAI_WHEEL SUITE
for name in REBUILD_ENV ENV_CACHE_ROOT RESULT_ROOT EXTRA_MODULES BUILD_MODULES \
    PYTHON_BIN RUST_TOOLCHAIN KEEP_BUILD; do
    if [[ -n ${!name:-} ]]; then
        echo "note: $name=${!name} is set and will reach the jobs" >&2
    fi
done

SB=(sbatch --parsable "--partition=$PARTITION" "--constraint=$CONSTRAINT"
    "--output=$R/maxwell/dioptrin-%j.out")
SCRIPT=benchmarks/comparison/maxwell.sbatch
CPU=(dioptrin-pixel_split dioptrin-supersampled dioptrin-pyfai_full)
CUDA=(dioptrin-pixel_split-cuda dioptrin-pyfai_full-cuda
    dioptrin-pixel_split-cuda-batch dioptrin-pyfai_full-cuda-batch)
ALL1D=("${CPU[@]}" azint azint-cached pyfai-full pyfai-bbox pyfai-full-omp
    pyfai-full-opencl "${CUDA[@]}")
# Stock pyfai-full / pyfai-full-omp are left out of the 2D jobs; the patched
# job measures them.
ALL2D=("${CPU[@]}" azint azint-cached pyfai-bbox pyfai-full-opencl "${CUDA[@]}")
GPU=(--opencl-device gpu --cuda-device 0 --gpu-precision f32_compensated)
CAKES=(--dioptrin-2d-path batch --dioptrin-2d-batch-split nested)
LONG=(--trials 3 --seconds 1.0)

if [[ $dry_run == 0 ]]; then
    mkdir -p "$R/maxwell"
else
    echo "# exported to every job: WITH_OPENCL=1 WITH_CUDA=1 AZINT_FAST_MATH=1"
fi
JOBLOG=$R/maxwell/jobs-2026-11-25.txt

# submit NAME TIME [VAR=value ...] -- COMPARE-ARGS...
# compare.py's --output comes after maxwell.sbatch's own --output, so the
# results land in $R/NAME; maxwell/<job-id>-*/ keeps the job and build logs.
submit() {
    local name=$1 time=$2
    shift 2
    local envs=()
    while [[ $1 != -- ]]; do
        envs+=("$1")
        shift
    done
    shift
    local cmd=()
    if [[ ${#envs[@]} -gt 0 ]]; then
        cmd=(env "${envs[@]}")
    fi
    cmd+=("${SB[@]}" "--time=$time" "$SCRIPT" "$@" --output "$R/$name")
    if [[ -e $R/$name ]]; then
        echo "skip $name: $R/$name already exists"
        return
    fi
    if [[ $dry_run == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        echo
        return
    fi
    local id
    if ! id=$("${cmd[@]}"); then
        echo "error: sbatch failed for $name; jobs above were submitted" >&2
        exit 1
    fi
    echo "$id $name"
    printf '%s %s %s\n' "$id" "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOBLOG"
}

if [[ $mode == quick || ! -e $R/2026-11-25-maxwell-quick ]]; then
    # Installation check: every variant on the small synthetic dataset at 1
    # and 2 workers; also builds and caches the environment.
    submit 2026-11-25-maxwell-quick 00:30:00 -- \
        --suite quick --dimensions 1d 2d "${CAKES[@]}" "${GPU[@]}" --variants "${ALL1D[@]}"
fi
if [[ $mode == quick ]]; then
    exit 0
fi

# Primary float64 jobs (3 trials of 3 x 1 s). submit appends --output after
# --variants; argparse ends the variant list at the next option.
submit 2026-11-25-maxwell-f64-1d 12:00:00 -- \
    --dimensions 1d "${LONG[@]}" "${GPU[@]}" --variants "${ALL1D[@]}"
submit 2026-11-25-maxwell-f64-2d-fine 12:00:00 -- \
    --dimensions 2d --cakes 2000x360 "${LONG[@]}" "${CAKES[@]}" "${GPU[@]}" --variants "${ALL2D[@]}"
submit 2026-11-25-maxwell-f64-2d-coarse 12:00:00 -- \
    --dimensions 2d --cakes 1000x72 "${LONG[@]}" "${CAKES[@]}" "${GPU[@]}" --variants "${ALL2D[@]}"

# pyFAI full-splitting cakes with the patched wheel (the stock wheel fails
# its self-consistency check with more than one worker).
if [[ -z ${PYFAI_PATCHED_WHEEL:-} ]]; then
    echo "skip 2026-11-25-maxwell-f64-2d-pyfai-patched: PYFAI_PATCHED_WHEEL is not set"
elif [[ ! -f $PYFAI_PATCHED_WHEEL ]]; then
    echo "skip 2026-11-25-maxwell-f64-2d-pyfai-patched: $PYFAI_PATCHED_WHEEL not found"
else
    case $PYFAI_PATCHED_WHEEL in
        /*) wheel=$PYFAI_PATCHED_WHEEL ;;
        *) wheel=$PWD/$PYFAI_PATCHED_WHEEL ;;
    esac
    submit 2026-11-25-maxwell-f64-2d-pyfai-patched 12:00:00 "PYFAI_WHEEL=$wheel" -- \
        --dimensions 2d "${LONG[@]}" --opencl-device gpu \
        --variants pyfai-full pyfai-full-omp pyfai-full-opencl
fi

# Optional passes (suite defaults, 1 trial).
submit 2026-11-25-maxwell-ablation-paths 04:00:00 -- \
    --dimensions 1d 2d --dioptrin-1d-path global split --dioptrin-2d-path sequential \
    --variants "${CPU[@]}"
submit 2026-11-25-maxwell-f32-1d 06:00:00 -- \
    --dimensions 1d --input-dtype float32 "${GPU[@]}" --variants "${ALL1D[@]}"
submit 2026-11-25-maxwell-f32-2d 08:00:00 -- \
    --dimensions 2d --input-dtype float32 "${CAKES[@]}" "${GPU[@]}" --variants "${ALL2D[@]}"
submit 2026-11-25-maxwell-cuda-f64 04:00:00 -- \
    --dimensions 1d 2d --workers 1 --cuda-device 0 --gpu-precision f64 "${CAKES[@]}" \
    --variants "${CUDA[@]}"
