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
# Environment switches: AZINT_FAST_MATH=1 (default; 0 builds the default azint),
# WITH_OPENCL=1 installs pyopencl for pyfai-full-opencl, EXTRA_MODULES adds
# site modules (e.g. a GPU driver/OpenCL stack) after BUILD_MODULES. For a GPU
# node pass your GPU partition: sbatch --partition=<gpu-partition> ...
# PYFAI_WHEEL selects a local pyFAI wheel for a separate cached environment.
# WITH_CUDA=1 builds Dioptrin with the 'cuda' feature (separate cached
# environment) for the dioptrin-*-cuda variants. The build needs no CUDA
# toolkit; at run time NVRTC must be found, so put a CUDA toolkit module no
# newer than the node's driver (nvidia-smi "CUDA Version") into EXTRA_MODULES.
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
if [[ -n "${EXTRA_MODULES:-}" ]]; then
    read -r -a extra_modules <<< "$EXTRA_MODULES"
    module load "${extra_modules[@]}"
fi
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
# Like-for-like azint build by default (see README, "azint build flags").
if [[ ${AZINT_FAST_MATH:-1} == 1 ]]; then
    cache_args+=(--azint-fast-math)
fi
if [[ ${WITH_OPENCL:-0} == 1 ]]; then
    cache_args+=(--with-opencl)
    command -v nvidia-smi >/dev/null && nvidia-smi > "$result_dir/nvidia-smi.txt" || true
    ls /etc/OpenCL/vendors > "$result_dir/opencl-vendors.txt" 2>&1 || true
fi
if [[ ${WITH_CUDA:-0} == 1 ]]; then
    cache_args+=(--with-cuda)
    command -v nvidia-smi >/dev/null && nvidia-smi > "$result_dir/nvidia-smi.txt" || true
    printf '%s\n' "${LD_LIBRARY_PATH:-}" > "$result_dir/ld-library-path.txt"
fi
if [[ -n "${PYFAI_WHEEL:-}" ]]; then
    cache_args+=(--pyfai-wheel "$PYFAI_WHEEL")
fi
command -v numactl >/dev/null && numactl --hardware > "$result_dir/numactl.txt" || true
"${PYTHON_BIN:-python3}" benchmarks/comparison/environment.py "${cache_args[@]}"
environment_python=$("${PYTHON_BIN:-python3}" -E -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$result_dir/environment.json")
# With --cpus-per-task the benchmark runs as a job step restricted to that many
# CPUs; the binding options are passed explicitly so the step gets whole cores
# (an exclusive node would otherwise let srun pick hardware threads freely):
# - job submitted with --hint=nomultithread: N cores, one hardware thread each.
#   srun does not reliably inherit sbatch's --hint, so it is repeated here when
#   the environment (SLURM_HINT) or the job record (threads per core 1,
#   "ReqB:S:C:T=...:1") shows it. --hint excludes --cpu-bind.
# - otherwise: N CPUs counted as hardware threads, bound as whole cores.
# Untested outside Slurm (the job-record format is version dependent): check
# launch.txt and the affinity in environment.json; if the hint was missed,
# submit with SLURM_HINT=nomultithread in the environment. Without
# --cpus-per-task the launch is unchanged.
launch=()
if [[ -n ${SLURM_CPUS_PER_TASK:-} ]]; then
    launch=(srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK")
    if [[ ${SLURM_HINT:-} == nomultithread ]] ||
        grep -Eq 'ReqB:S:C:T=[^ ]*:1( |$)' "$result_dir/slurm-job.txt"; then
        launch+=(--hint=nomultithread)
    else
        launch+=(--cpu-bind=cores)
    fi
fi
# ${a[@]+"${a[@]}"}: an empty array is not "unbound" under set -u (bash < 4.4).
printf '%s\n' "${launch[*]+${launch[*]}}" > "$result_dir/launch.txt"
${launch[@]+"${launch[@]}"} "$environment_python" -E -s -u benchmarks/comparison/compare.py \
    --output "$result_dir/comparison" --suite "${SUITE:-standard}" "$@"
