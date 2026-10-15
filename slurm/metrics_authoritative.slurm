#!/usr/bin/env bash
#SBATCH --job-name=kr-auth-metrics
#SBATCH --partition=A6000,A6000-lm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/slurm/runtime.sh"
sha256sum --check artifacts/authoritative-evaluation-source.sha256 > "$KG_JOB_RUNTIME/authoritative-source-check.txt"
sha256sum --check artifacts/metrics-source.sha256 > "$KG_JOB_RUNTIME/metrics-source-check.txt"
run_root=outputs/evaluation_authoritative_v2/anchor-corrected-20261001-v1
srun "$KG_PYTHON" -u -m kinase_gap.metrics \
    --candidates "$run_root/evaluations.csv" \
    --train-table data/processed/klifs_v1/KLIFS_train_table.csv \
    --validation-table data/processed/klifs_v1/KLIFS_val_table.csv \
    --test-table data/processed/klifs_v1/KLIFS_test_table.csv \
    --gt-evaluation "$run_root/gt_evaluations.csv" \
    --output-dir "$run_root/metrics" \
    --smiles-column canonical_smiles --n-jobs "$SLURM_CPUS_PER_TASK" \
    --device cpu --batch-size 512 --bootstrap-seed 20260724 \
    --bootstrap-samples 2000 --confidence 0.95
