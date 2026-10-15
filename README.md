# Kinase linker generation with DiffLinker

Training, sampling and analysis code for *Designing Novel Kinase Inhibitors with Generative AI*. DiffLinker generates chemical linkers between ligand fragments in a kinase binding pocket.

Saved models and datasets are hosted on **Zenodo** in the data archive accompanying the paper. Download that archive using the record cited in the paper. It contains the prepared structures, pretrained and fine-tuned checkpoints, generated molecules, analysis tables and dependency wheels. This repository contains the source, settings and instructions.

## 1. Set up

Use Linux x86-64 with Conda. Training and generation require an NVIDIA GPU; the recorded environments use Python 3.10 and CUDA 11.3. Extract the Zenodo archive beside the checkout:

```bash
git clone --branch main --single-branch \
  https://github.com/yifeili6/difflinker_kinase.git kinase-code
cd kinase-code
```

```text
project/
  kinase-code/                 # this checkout
  kinase-linker-generation/    # extracted Zenodo archive
  environments/               # parent for the new environment
```

From `kinase-code/`, verify the code's archive correspondence and create the recorded environment. The environment command reads the dependency wheels from the archive.

```bash
python3 -B scripts/create_from_archive.py \
  --archive ../kinase-linker-generation --verify-only
mkdir -p ../environments
export KINASE_CODE="$PWD"
export KINASE_ARCHIVE="$(cd ../kinase-linker-generation && pwd -P)"
export KINASE_RUNTIME_ENV="$(cd ../environments && pwd -P)/kinase-paper"
bash "$KINASE_ARCHIVE/scripts/create_environment.sh" "$KINASE_RUNTIME_ENV"
export KG_PYTHON="$KINASE_RUNTIME_ENV/bin/python"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

Set `KINASE_CONDA` to the Conda executable if it is not on `PATH`. Keep the environment accessible from each compute node.

## 2. Choose a starting point

**Use the saved selected model.** Follow [Model usage](docs/MODEL_USAGE.md), starting at “Prepare a workspace for the saved model.” This creates a separate workspace, copies the selected checkpoint and its recorded dependencies from Zenodo, and shows how to generate molecules. Keep the archive's original source identities for this route.

**Train a new model.** Create a fresh experiment directory from the checkout:

```bash
"$KG_PYTHON" -B scripts/create_from_archive.py \
  --archive "$KINASE_ARCHIVE" --destination ../kinase-run
cd ../kinase-run
source reproduction/activate.sh
```

Continue with step 2 in [Reproduce the complete workflow](docs/REPRODUCE.md). It covers tensor inputs, eight HPO trials, checkpoint selection, molecular generation, evaluation and figure generation. Add `--dry-run` to the workspace command to inspect the copy plan. The destination must be new.

The recorded HPO search uses seed 42, a **100-epoch maximum per trial** and early-stopping patience 20. `configs/hpo_grid.yml` supplies this budget and overrides the shared training defaults. Generation uses 62 contexts, six linker lengths and 128 samples per condition: **47,616 attempts**. Use cluster partitions available to you when submitting the supplied Slurm jobs.

## 3. Evaluate molecules

Use `scripts/run_authoritative_evaluation.py` with `configs/evaluation_authoritative_v2.yml`; the complete commands are in steps 5–6 of [Reproduce the complete workflow](docs/REPRODUCE.md). Evaluation checks the saved molecular identity and coordinates, applies PoseBusters, Lipinski and MOSES filters, and writes the analysis tables. [Data and evaluation](docs/DATA_AND_EVALUATION.md) defines the populations and filter settings.

## 4. Rebuild figures

[Figure reproduction](docs/FIGURE_REPRODUCTION.md) covers property plots, filter survival, molecular illustrations, protein–ligand overlays and LigPlot+/HBPLUS diagrams. [Property figures](docs/PROPERTY_FIGURES.md) gives the final-size plotting commands. [Contact panels](docs/CONTACT_PANEL_PREPARATION.md) explains the editable drawing files and contact labels. Read plotting tables and structural inputs from Zenodo and choose new output directories.

## Files

| Directory | Purpose |
|---|---|
| `src/kinase_gap/` | Preparation, training, selection, generation, evaluation and analysis |
| `vendor/DiffLinker/` | Diffusion model, molecular assembly and PAINS SMARTS filter definitions |
| `configs/` | Dataset, training search, generation and evaluation settings |
| `scripts/` | Workspace preparation and calculation/plot entrypoints |
| `slurm/` | GPU and CPU batch launchers |
| `environment/` | Conda and pip package specifications |
| `docs/` | Detailed calculation and figure instructions |
| `licenses/` | Component attribution and license notices |

`source_manifest.json` is read by the workspace builder to match each distributed archive-derived file with its original Zenodo counterpart. [Portable paths](docs/PORTABILITY.md) explains the workspace and archive relationship. [Dataset reconstruction](docs/DATASET_RECONSTRUCTION.md) describes rebuilding inputs from native structures. [Checkpoint continuation](docs/CHECKPOINT_CONTINUATION.md) describes the source and records copied from Zenodo to reconstruct the joined training history.

## Citation and license

See [references.bib](references.bib) for software and data citations. DiffLinker retains its [MIT license](vendor/DiffLinker/LICENSE); see [third-party notices](licenses/THIRD_PARTY_NOTICES.md) for attribution. The original pretrained checkpoint is distributed through [DiffLinker Models](https://doi.org/10.5281/zenodo.10988017). The kinase datasets, fine-tuned models and paper results belong to the separate companion Zenodo archive.
