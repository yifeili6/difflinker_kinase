# Models and calculations

Run the following commands from the extracted package directory.

## Environment

Create the recorded Python environment and install the evaluation packages:

```bash
conda create --prefix .environment --file environment/explicit-lock.txt
conda activate ./.environment
python -m pip install --no-deps -r environment/pip-requirements.txt
python -m pip install --no-deps -r environment/evaluation-pip-requirements.txt
export PYTHONPATH=src:vendor/DiffLinker
export PYTHONDONTWRITEBYTECODE=1
```

`environment/environment.yml` provides the package versions in a readable Conda specification. The explicit lock records the Linux x86-64 build used for the calculations. Python 3.10.5, PyTorch 1.11.0, PyTorch Lightning 1.6.3 and RDKit 2022.03.5 are the core versions.

## Generate molecules

`models/kinase.ckpt` contains the selected fine-tuned model. `models/pretrained.ckpt` contains the pretrained DiffLinker initialization. Both retain the original network tensors. Runtime paths use relative locations, and the inference files contain model weights and architecture settings.

The example in `data/example/` contains two fragments of the 5LQF ligand and the corresponding protein pocket. `context.json` preserves fragment atom properties and attachment indices; `pocket.npz` stores pocket coordinates and element types. `fragments.sdf` and `native_ligand.mol2` provide viewable structures.

```bash
python scripts/predict.py --checkpoint models/kinase.ckpt --output outputs/prediction --samples 3 --device cpu
```

Use `--device cuda` on a workstation with a compatible NVIDIA GPU. The command performs all 500 diffusion steps. Each output directory contains `molecules.sdf`, `molecules.csv`, `coordinates.npz` and `prediction.json`. The table records molecular sanitization and connectivity for each attempt. The NPZ file preserves sampled coordinates; the SDF includes molecular identity metadata. Choose a new output directory for each command.

To use the pretrained model:

```bash
python scripts/predict.py --checkpoint models/pretrained.ckpt --output outputs/pretrained --samples 2 --device cpu
```

The input fragment atom order, anchor indices and pocket coordinates must share one coordinate frame. `context.json` is the editable input schema; its base64 fragment is an RDKit molecule serialized with all atom properties. The helpers in `kinase_gap.chemistry_io` and `kinase_gap.generate` build the same representation from the prepared KLIFS tables.

## Evaluate saved molecules

```bash
python scripts/evaluate_sdf.py --input outputs/prediction/molecules.sdf --output outputs/prediction/evaluation.csv
```

The evaluator applies PoseBusters molecule checks, Lipinski criteria, MOSES and PAINS filters. It records QED, synthetic accessibility and molecular descriptors. Analysis tables in the archive retain the identifiers used to connect candidates, structures and figure inputs.

To analyze the archived generated molecules, use `--input data/generated/candidates.sdf` and a new CSV path under `outputs/`. `data/generated/candidates.csv` contains the corresponding generation records. `data/analysis/` contains the saved molecule evaluations and summary tables used by the figure scripts.

## Prepare public training structures

KLIFS provides processed structures through its [download interface](https://klifs.net/faq.php) and [API](https://klifs.net/swagger_v2/). Search the human and mouse entries, download the MOL2 structure package and the result table, and keep the ZIP file. Extract its species/kinase directories and `overview.csv` into `data/raw/klifs/local/`. Save the ZIP as `KLIFS_download.zip` and the result table as `KLIFS_export.csv` in that directory.

The study used a KLIFS export dated 2025-10-22, cached on 2025-11-12. `data/dataset_metadata/study_dataset.json` records its ZIP, overview and export hashes, selection filters, split seed and row counts. `structure_splits.csv` identifies each curated structure by species, kinase, PDB, chain and alternate model, with its assigned split, ligand-file hash and emitted fragmentation count. These metadata identify the source structures; the training structures are obtained from KLIFS. For the recorded snapshot, keep the directory name in `configs/data_klifs_v1.yml` and run preparation with that configuration. The following commands record and prepare a newly downloaded package:

```bash
python scripts/configure_dataset.py --snapshot data/raw/klifs/local --output configs/data_local.yml
python -m kinase_gap.prepare --config configs/data_local.yml
python scripts/build_tensor_caches.py
```

The preparation settings in `configs/data_klifs_v1.yml` record the study's structure filters, fragmentation rules, grouping keys, seed and selected test targets. `configure_dataset.py` records hashes for the downloaded files. Preparation writes the train, validation and test tables, fragment/linker structures, pocket arrays and split manifests under `data/`.

## Fine-tune

```bash
python -m kinase_gap.train --config configs/train.yml --run-id training_01
```

`configs/train.yml` records the selected settings: learning rate 0.00005, batch size 32, seed 42, a maximum of 100 epochs, and early-stopping patience 20. `configs/trial-00.yml` through `trial-07.yml` contain the eight study configurations. Run each configuration with a separate run identifier to repeat the search. The training entrypoint loads pretrained tensors strictly, starts a new AdamW optimizer, monitors `loss/val`, and writes the best checkpoint and epoch metrics.

After training, the generation engine accepts the selected checkpoint and the study generation configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --config configs/generation.yml --checkpoint models/kinase.ckpt --run-id sampling_01
```

The full protocol uses 62 fragment contexts, linker sizes 8 through 13 and 128 samples per condition. The saved selected model is supplied on Zenodo together with datasets, prediction examples and figure materials. The GitHub repository supplies the code and instructions.
