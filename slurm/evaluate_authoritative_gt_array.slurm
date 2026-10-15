#!/usr/bin/env bash
#SBATCH --job-name=kr-auth-gt
#SBATCH --partition=A6000,A6000-lm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-41%24
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/slurm/runtime.sh"
sha256sum --check artifacts/authoritative-evaluation-source.sha256 > "$KG_JOB_RUNTIME/authoritative-source-check.txt"
srun "$KG_PYTHON" -u scripts/run_authoritative_evaluation.py ground-truth --config configs/evaluation_authoritative_v2.yml
