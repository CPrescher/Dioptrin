#!/bin/bash
#SBATCH --job-name=dioptrin-compare
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=dioptrin-%j.out

# Submit from a clone's root. Extra arguments are forwarded to compare.py:
# sbatch benchmarks/comparison/maxwell.sbatch --suite quick --workers 1,2
set -euo pipefail
unset LD_PRELOAD PYTHONPATH
export PYTHONNOUSERSITE=1
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
set +u
module purge
read -r -a build_modules <<< "${BUILD_MODULES:-maxwell mamba}"
module load "${build_modules[@]}"
set -u
unset PYTHONPATH
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Dioptrin root}"
repo_dir="$PWD"
test -f benchmarks/comparison/setup.py
run_id="${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}-$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${RESULT_ROOT:-$repo_dir/benchmark-results/maxwell}/$run_id"
mkdir -p "$result_dir"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/dioptrin-${SLURM_JOB_ID}.XXXXXXXX")
finish() {
    status=$?
    printf '%s\n' "$status" > "$result_dir/exit-code.txt"
    if [[ $status -eq 0 && ${KEEP_BUILD:-0} == 0 ]]; then
        rm -rf -- "$build_dir"
    else
        printf 'Temporary build files at %s (node %s; subject to node cleanup)\n' "$build_dir" "$(hostname)"
    fi
    printf 'Results: %s\n' "$result_dir"
}
trap finish EXIT
exec > >(tee "$result_dir/job.log") 2>&1
hostname > "$result_dir/hostname.txt"
lscpu > "$result_dir/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
module list 2> "$result_dir/modules.txt"
git rev-parse HEAD > "$result_dir/source-commit.txt"
git status --short > "$result_dir/source-status.txt"

# Keep Python environments on shared storage; native code is reused only for
# matching CPU/OS/Python/compiler/source/recipe fingerprints. Rust tools and
# Cargo output remain node-local and are installed only on a cache miss.
cache_args=(--cache-root "${ENV_CACHE_ROOT:-$repo_dir/.benchmark-environments}"
    --build-root "$build_dir" --artifacts "$result_dir/build"
    --receipt "$result_dir/environment.json" --rust-toolchain "${RUST_TOOLCHAIN:-stable}")
if [[ ${REBUILD_ENV:-0} == 1 ]]; then
    cache_args+=(--rebuild)
fi
"${PYTHON_BIN:-python3}" benchmarks/comparison/environment.py "${cache_args[@]}"
environment_python=$("${PYTHON_BIN:-python3}" -E -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$result_dir/environment.json")
"$environment_python" -u benchmarks/comparison/compare.py \
    --output "$result_dir/comparison" --suite "${SUITE:-standard}" "$@"
