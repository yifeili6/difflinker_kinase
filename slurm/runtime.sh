#!/usr/bin/env bash
set -Eeuo pipefail
export KG_PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the revision compute project}"
[[ -f "$KG_PROJECT_ROOT/.kinase-revision" ]] || exit 2
export KG_RUNTIME_ENV="${KINASE_RUNTIME_ENV:?Set KINASE_RUNTIME_ENV to the recorded environment prefix}"
export KG_PYTHON="$KG_RUNTIME_ENV/bin/python"
[[ -x "$KG_PYTHON" ]] || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export PYTHONPATH="$KG_PROJECT_ROOT/src:$KG_PROJECT_ROOT/vendor/DiffLinker"
export PATH="$KG_RUNTIME_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$KG_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export KG_JOB_RUNTIME="$KG_PROJECT_ROOT/cache/job-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-${SLURM_ARRAY_TASK_ID}}"
export XDG_CACHE_HOME="$KG_JOB_RUNTIME/xdg"
export TORCH_HOME="$KG_JOB_RUNTIME/torch"
export MPLCONFIGDIR="$KG_JOB_RUNTIME/matplotlib"
export TMPDIR="$KG_JOB_RUNTIME/tmp"
export WANDB_MODE=offline CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR"
cd "$KG_PROJECT_ROOT"
printf 'project=%s\npython=%s\nhost=%s\njob=%s\n' "$KG_PROJECT_ROOT" "$KG_PYTHON" "$(hostname -f)" "$SLURM_JOB_ID"
