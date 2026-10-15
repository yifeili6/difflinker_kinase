#!/usr/bin/env bash
#SBATCH --job-name=kr-generate
#SBATCH --partition=A100
#SBATCH --exclude=cardiff.ks.uiuc.edu,portia.ks.uiuc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-371%12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/slurm/runtime.sh"
sha256sum --check artifacts/generation-source.sha256 > "$KG_JOB_RUNTIME/source-check.txt"
srun "$KG_PYTHON" -u scripts/run_generation_with_stereo.py shard \
    --selection outputs/hpo/studies/anchor-corrected-20261001-v1/selection.json \
    --run-id anchor-corrected-20261001-v1
