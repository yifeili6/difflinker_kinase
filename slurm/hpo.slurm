#!/usr/bin/env bash
#SBATCH --job-name=kr-hpo
#SBATCH --partition=dgx,A6000-lm,A100,A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/slurm/runtime.sh"
"$KG_PYTHON" scripts/source_manifest.py --check
study_id=anchor-corrected-20261001-v1
trial_label="$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
trial_config="outputs/hpo/studies/$study_id/configs/trial-$trial_label.yml"
run_id="hpo-$study_id-$trial_label"
runtime="outputs/hpo/studies/$study_id/runtime/trial-$trial_label"
[[ ! -e "models/runs/hpo_trial_$trial_label/$run_id/training_result.json" ]] || {
    echo 'A completed trial already exists; use a new study ID for another training run.' >&2
    exit 2
}
"$KG_PYTHON" -m kinase_gap.hpo materialize --search configs/hpo_grid.yml \
    --study-id "$study_id" --trial-index "$SLURM_ARRAY_TASK_ID" --output "$trial_config"
mkdir -p "$runtime"
cd "$runtime"
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 "$KG_PYTHON" -u -m kinase_gap.train \
    --config "$trial_config" --run-id "$run_id"
