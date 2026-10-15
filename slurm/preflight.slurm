#!/usr/bin/env bash
#SBATCH --job-name=kr-preflight
#SBATCH --partition=dgx,A6000-lm,A100,A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/slurm/runtime.sh"
srun "$KG_PYTHON" -u scripts/preflight.py --require-cuda
