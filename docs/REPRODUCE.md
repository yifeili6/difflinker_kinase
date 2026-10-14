# Reproduce the complete workflow

Run the preparation and consolidation commands on a Linux workstation with access to the shared Slurm filesystem. Submit training, generation, molecule evaluation and aggregate metrics to Slurm. Use one Bash session for the commands below.

These commands create a new study from the pretrained initialization. For generation with an archived selected checkpoint, follow [Model usage](MODEL_USAGE.md), including `scripts/copy_archived_segment_model_bundle.py`. [Checkpoint continuation](CHECKPOINT_CONTINUATION.md) describes the archived execution records for trials 4–7: model, optimizer, counters, stopping state and the prior best checkpoint are restored; the 100-epoch total budget and patience-20 rule continue across segments; each segment initializes its global and data-loader random generators with seed 42.

The experiment uses eight learning-rate/batch-size combinations. Each materialized HPO configuration has a maximum of **100 epochs**, early-stopping patience **20**, seed **42**, and one validation-ranked checkpoint. `configs/finetune_anchors.yml` supplies the common model settings; `configs/hpo_grid.yml` supplies this training budget. Generation uses 62 contexts, six linker sizes and 128 samples per condition: **47,616 attempts in 372 shards**. Ground-truth evaluation covers **5,292 rows in 42 shards**.

```text
Archived structural inputs and prepared splits
                    |
          Fresh workspace and CPU caches
                    |
              CUDA preflight
              /            \
       Eight HPO trials    42 GT evaluation shards
              |
       Validation-ranked checkpoint
              |
       372 generation shards
              |
       372 evaluation shards
              \            /
               Consolidation
                    |
       Metrics, structures and diagnostics
                    |
          Validated run manifest
                    |
            Publication figures
```

## 1. Create the environment and a new workspace

Complete steps 1–2 of the [README](../README.md), choosing the new-training route. The checkout's `scripts/create_from_archive.py` reads the Zenodo archive, installs the distributed execution files in a new experiment directory and creates empty output directories. It preserves the archive.

Continue below from the new experiment directory with `KINASE_RUNTIME_ENV` set to the recorded environment. Choose a shared environment prefix and experiment directory visible to the compute nodes. [Dataset reconstruction](DATASET_RECONSTRUCTION.md) describes rebuilding the prepared inputs from native structures.

## 2. Load the environment and build tensor inputs

Activate the environment created in step 1 and build the CPU tensor inputs:

```bash
source reproduction/activate.sh
"$KG_PYTHON" -B scripts/build_tensor_caches.py
"$KG_PYTHON" scripts/summarize_dataset_population.py \
  --dataset-dir data/processed/klifs_v1 \
  --output provenance/population_summary.json
```

Activation sets `KG_PYTHON` to `$KINASE_RUNTIME_ENV/bin/python` and configures the source imports and shared libraries. The expected CPU-cache row counts are **34,256 train**, **3,701 validation**, and **5,292 test**.

Set the experiment paths and freeze the selection rule:

```bash
export STUDY=anchor-corrected-20261001-v1
export SELECTION="outputs/hpo/studies/$STUDY/selection.json"
export GENERATION="outputs/generation/$STUDY"
export EVALUATION="outputs/evaluation_authoritative_v2/$STUDY"
export DATASET=data/processed/klifs_v1
export REPORT="$EVALUATION/report"
export STRUCTURAL="$REPORT/structural"
export FIGURES="figures/$STUDY"
"$KG_PYTHON" scripts/select_hpo_outcomes.py freeze --study-id "$STUDY"
```

Each fresh workspace uses these recorded relative paths. The selector considers every declared configuration and ranks finite models that complete the specified budget or early-stopping rule by validation loss, with trial index as the tie-breaker.

## 3. Submit preflight, HPO and ground-truth evaluation

Choose a cluster partition providing one NVIDIA RTX A6000 with 48 GB GPU memory per training task. The generation launcher below uses one NVIDIA A100 GPU per task. The following partition names apply to the recorded Slurm cluster; use the corresponding partitions on another cluster. Host-memory requests remain in the supplied Slurm scripts. The training submission below requests a 48-hour scheduling allowance; its configuration retains the 100-epoch cap and recorded early-stopping rule.

```bash
export KINASE_GPU_PARTITIONS=A6000,A6000-lm
export KINASE_GENERATION_PARTITIONS=A100
export KINASE_CPU_PARTITIONS=A6000,A6000-lm,A100
PREFLIGHT_JOB=$(sbatch --parsable --partition="$KINASE_GPU_PARTITIONS" slurm/preflight.slurm)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
HPO_JOB=$(sbatch --parsable --partition="$KINASE_GPU_PARTITIONS" --time=2-00:00:00 \
  --dependency="afterok:$PREFLIGHT_JOB" slurm/hpo.slurm)
export HPO_JOB=${HPO_JOB%%;*}
GT_JOB=$(sbatch --parsable --partition="$KINASE_CPU_PARTITIONS" \
  --dependency="afterok:$PREFLIGHT_JOB" slurm/evaluate_authoritative_gt_array.slurm)
export GT_JOB=${GT_JOB%%;*}
```

Record the eight exact array-task identifiers:

```bash
"$KG_PYTHON" - <<'PY'
import json, os
from pathlib import Path

attempts = [{"trial_index": i, "job_id": f"{os.environ['HPO_JOB']}_{i}"} for i in range(8)]
ledger = {"schema_version": 1, "study_id": os.environ["STUDY"], "attempts": attempts}
with Path("reproduction/hpo_attempts.json").open("x") as handle:
    json.dump(ledger, handle, indent=2)
with Path("reproduction/active_hpo_tasks.json").open("x") as handle:
    json.dump({str(row["trial_index"]): row["job_id"] for row in attempts}, handle, indent=2)
PY
sacct -X -j "$HPO_JOB,$GT_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
```

`logs/` contains scheduler output. Each training log directory contains the materialized configuration, software/GPU metadata, metric history and checkpoint records.

## 4. Select the model

Inspect the trial status using the recorded task map and attempt ledger:

```bash
"$KG_PYTHON" scripts/select_hpo_outcomes.py status --study-id "$STUDY" \
  --active-tasks reproduction/active_hpo_tasks.json \
  --attempt-ledger reproduction/hpo_attempts.json
```

The command reports completed and terminal numerical outcomes and lists pending task IDs. Exit status 3 denotes pending work. The selector's `register-numerical` mode records a terminal `FoundNaNException` using its exact task ID and corresponding `.err`/`.out` files; `--help` lists its arguments. Outcome records retain checkpoint hashes and validation observations.

When `ready` is true, select and initialize the generation plan:

```bash
"$KG_PYTHON" scripts/select_hpo_outcomes.py select --study-id "$STUDY" \
  --active-tasks reproduction/active_hpo_tasks.json \
  --attempt-ledger reproduction/hpo_attempts.json
"$KG_PYTHON" scripts/plot_hpo_learning_curves.py \
  --selection "$SELECTION" \
  --output "outputs/hpo/studies/$STUDY/learning_curves"
SELECTED_CHECKPOINT=$("$KG_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["selected_checkpoint"])' "$SELECTION")
"$KG_PYTHON" scripts/validate_publication_model.py \
  --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" \
  --config configs/generation.yml \
  --output provenance/publication_model_validation.json
"$KG_PYTHON" scripts/run_generation_with_stereo.py preflight \
  --selection "$SELECTION" --run-id "$STUDY"
"$KG_PYTHON" scripts/run_generation_with_stereo.py initialize \
  --selection "$SELECTION" --run-id "$STUDY"
```

The selected checkpoint, selection JSON, ranking and trial-outcome table identify the model used by every generation task.

The learning-curve folder contains training and validation plots for every declared trial, the original metric rows, epoch-aligned plotting data and input checksums. The common optimizer step aligns the recorded training and validation epoch means. To rebuild the archived parent-and-continuation curves, use the separate command in [Checkpoint continuation](CHECKPOINT_CONTINUATION.md).

## 5. Submit generation and molecule evaluation

`slurm/generate_with_stereo_array.slurm` calls `scripts/run_generation_with_stereo.py`, which verifies and loads `scripts/serialization_patch.py` before using the recorded sampler. On the recorded cluster, the generation launcher uses the A100 partition and excludes Cardiff and Portia, leaving Surtur and Puck as its eligible nodes. Set the corresponding partition and node options for another cluster.

Every candidate keeps its attempt identifier and sampled coordinates. For candidates that pass sanitization, the writer records molecular identity, atom and bond stereo, and an embedded molecule with full-precision coordinates. Existing double-bond assignments are encoded in neighboring bond directions when needed. Export checks retain the same molecular graph, atom order and coordinates.

`scripts/run_authoritative_evaluation.py` uses `configs/evaluation_authoritative_v2.yml`. For each generated candidate that passed sanitization, it validates the saved SDF against its embedded molecule and checks that molecule against the generation identity and full-precision coordinate array. PoseBusters and the molecular filters receive this verified molecular graph and pose. The evaluation contract records the reader and source checksums.

```bash
GEN_JOB=$(sbatch --parsable --partition="$KINASE_GENERATION_PARTITIONS" slurm/generate_with_stereo_array.slurm)
export GEN_JOB=${GEN_JOB%%;*}
EVAL_JOB=$(sbatch --parsable --partition="$KINASE_CPU_PARTITIONS" \
  --dependency="aftercorr:$GEN_JOB" slurm/evaluate_authoritative_array.slurm)
export EVAL_JOB=${EVAL_JOB%%;*}
sacct -X -j "$GEN_JOB,$EVAL_JOB,$GT_JOB" --format=JobID,State,ExitCode,Elapsed
```

`aftercorr` starts each generated-molecule evaluation task after its corresponding generation task completes. Each shard retains 128 attempt records, an SDF, full-precision sampled coordinates and a completion record.

After the generation, generated-evaluation and GT arrays complete, consolidate on the workstation:

```bash
"$KG_PYTHON" scripts/run_generation_with_stereo.py finalize \
  --selection "$SELECTION" --run-id "$STUDY"
"$KG_PYTHON" scripts/run_authoritative_evaluation.py finalize \
  --config configs/evaluation_authoritative_v2.yml --threads 4
```

The finalizers verify all 372 generation/evaluation shards and all 42 GT shards before publishing the combined tables.

## 6. Compute aggregate metrics

```bash
METRICS_JOB=$(sbatch --parsable --partition="$KINASE_CPU_PARTITIONS" \
  --dependency="afterok:$GEN_JOB:$EVAL_JOB:$GT_JOB" slurm/metrics_authoritative.slurm)
METRICS_JOB=${METRICS_JOB%%;*}
sacct -X -j "$METRICS_JOB" --format=JobID,State,ExitCode,Elapsed
```

`$EVALUATION/metrics/` contains the survivor table, funnel counts, reference definitions, generated and GT metrics, provenance and checksums. [Data and evaluation](DATA_AND_EVALUATION.md) defines the populations and denominators.

## 7. Build structural materials

After the metrics job completes:

```bash
"$KG_PYTHON" scripts/audit_receptor_contacts.py run \
  --survivors "$EVALUATION/metrics/final_survivors.csv" \
  --generation-dir "$GENERATION" \
  --contexts data/manifests/klifs_v1/contexts.csv \
  --dataset "$DATASET" \
  --output "$EVALUATION/receptor_diagnostics"

"$KG_PYTHON" -m kinase_gap.report build \
  --candidate-evaluation "$EVALUATION/metrics/final_survivors.csv" \
  --gt-evaluation "$EVALUATION/gt_evaluations.csv" \
  --funnel "$EVALUATION/metrics/funnel.csv" \
  --candidate-sdf "$GENERATION/candidates.sdf" \
  --output-dir "$REPORT" --deduplicate-canonical-smiles

"$KG_PYTHON" -m kinase_gap.structural_bundle build \
  --selection-json "$REPORT/ranking/selection.json" \
  --contexts data/manifests/klifs_v1/contexts.csv \
  --snapshot data/raw/klifs/klifs_2025-10-22_cached_2025-11-12 \
  --fragment-sdf "$DATASET/KLIFS_test_frag.sdf" \
  --output-dir "$STRUCTURAL"

"$KG_PYTHON" -m kinase_gap.structural_bundle verify \
  --bundle-dir "$STRUCTURAL" \
  --selection-json "$REPORT/ranking/selection.json" \
  --contexts data/manifests/klifs_v1/contexts.csv \
  --snapshot data/raw/klifs/klifs_2025-10-22_cached_2025-11-12 \
  --fragment-sdf "$DATASET/KLIFS_test_frag.sdf"
```

The receptor audit writes per-pose and per-atom distance tables for every final survivor and its matched native context. It uses source receptor coordinates, full-precision ligand coordinates, RDKit van der Waals radii and a distance threshold of 0.75 times the sum of atom radii. `receptor_diagnostics/done.json` records the completed populations and output checksums.

The report includes property distributions, survival plots, plotting tables and selected SDFs. The structural bundle supplies aligned receptor/ligand files, retained-fragment masks and similarity-map data. The verification command confirms the bundle checksums and its current source inputs.

## 8. Prepare contact files and render structures

Build the native/generated contact files and their distance tables:

```bash
"$KG_PYTHON" scripts/package_structural_inputs.py \
  --structural-dir "$STRUCTURAL" --output "$FIGURES/contact_inputs"
"$KG_PYTHON" scripts/structural_distance_qc.py \
  --inputs "$FIGURES/contact_inputs" --output "$FIGURES/distance_qc"
```

Each contact entry contains the complex PDB, receptor PDB, ligand SDF, atom mapping and ligand dictionary files. Install Ghostscript using the workstation's package manager so `gs` and `ps2pdf` are on `PATH`. Automatic LigPlot rendering uses `gs` for PDF and PNG export. Check the executable, then set the installed LigPlus directory and run:

```bash
command -v gs
command -v ps2pdf
gs --version
```

```bash
export LIGPLUS_DIR="${LIGPLUS_DIR:?Set the installed LigPlus directory}"
"$KG_PYTHON" scripts/run_ligplot_inputs.py \
  --inputs "$FIGURES/contact_inputs" --ligplus "$LIGPLUS_DIR" \
  --output "$FIGURES/ligplot"
"$KG_PYTHON" scripts/assemble_contact_figures.py \
  --contacts "$FIGURES/ligplot" --output "$FIGURES/contacts"
```

[Figure reproduction](FIGURE_REPRODUCTION.md) gives the LigPlot GUI steps for the same PDB/SDF inputs, ligand identifier and contact settings.

Create the recorded PyMOL environment at a new prefix, then render overlays:

```bash
export KINASE_PYMOL_ENV="$PWD/../kinase-pymol"
"${KINASE_CONDA:-conda}" create --yes --prefix "$KINASE_PYMOL_ENV" \
  --file environment/pymol-explicit-lock.txt
env LD_LIBRARY_PATH="$KINASE_PYMOL_ENV/lib" \
  "$KINASE_PYMOL_ENV/bin/python" scripts/render_structural_overlays.py \
  --structural-dir "$STRUCTURAL" --image-python "$KG_PYTHON" \
  --output "$FIGURES/overlays"
```

The output contains PNG/PDF views and editable PyMOL sessions. [Model usage](MODEL_USAGE.md) describes the checkpoint inputs and molecular identity loaders.

## 9. Finalize the run manifest and build publication figures

Validate the completed scientific records, model, structures and contact files, then write the new run manifest. The command below includes the automatic drawings from step 8. The `--ligplot`, `--contact-figures` and `--overlays` arguments are optional drawing directories; use your corresponding folders when preparing diagrams manually.

```bash
"$KG_PYTHON" scripts/finalize_reproduction.py \
  --project-root . --study-id "$STUDY" \
  --generation "$GENERATION" --evaluation "$EVALUATION" \
  --report "$REPORT" --structural "$STRUCTURAL" --dataset "$DATASET" \
  --selection "$SELECTION" \
  --model-validation provenance/publication_model_validation.json \
  --population provenance/population_summary.json \
  --contact-inputs "$FIGURES/contact_inputs" \
  --structural-contacts "$FIGURES/distance_qc" \
  --receptor-diagnostics "$EVALUATION/receptor_diagnostics" \
  --hpo-learning-curves "outputs/hpo/studies/$STUDY/learning_curves" \
  --ligplot "$FIGURES/ligplot" --contact-figures "$FIGURES/contacts" \
  --overlays "$FIGURES/overlays" \
  --output provenance/run_manifest.json

"$KG_PYTHON" scripts/build_publication_assets.py \
  --evaluation-dir "$EVALUATION" --dataset-dir "$DATASET" \
  --generation-dir "$GENERATION" --structural-dir "$STRUCTURAL" \
  --run-manifest provenance/run_manifest.json \
  --output "$FIGURES/publication"
```

`provenance/run_manifest.json` binds the selected model, source inputs, completed generation/evaluation populations, structural verification and diagnostic files by checksum. The publication folder contains the property figures, filtering plots, workflow and molecule illustrations, their exact data tables, and `figure_manifest.json`. The files remain associated with this new workspace and study.
