# Kinase ligand generation with DiffLinker

Training, molecular generation, evaluation and figure code for the kinase ligand study. The model links supplied ligand fragments inside a protein pocket and evaluates the generated molecules using PoseBusters, Lipinski criteria, MOSES filters and molecular descriptors.

## Get the model and calculation data

Saved models and calculation datasets are hosted on Zenodo in the archive accompanying the paper. Download the record cited in the paper and extract `kinase_paper_release_20261015` beside this repository. From the repository directory, copy its model and data directories into the matching locations:

```bash
cp -R ../kinase_paper_release_20261015/models .
cp -R ../kinase_paper_release_20261015/data .
cp -R ../kinase_paper_release_20261015/figures/. figures/
```

The complete Zenodo package can also be used directly. It contains the same code with the model and figure inputs already in place. Training structures are obtained from the public KLIFS database using the recorded structure identifiers and preparation settings.

## Run a prediction

Create the environment using [MODEL_WORKFLOW.md](MODEL_WORKFLOW.md). Then run:

```bash
python scripts/predict.py --checkpoint models/kinase.ckpt --output outputs/prediction --samples 3 --device cpu
python scripts/evaluate_sdf.py --input outputs/prediction/molecules.sdf --output outputs/prediction/evaluation.csv
```

The example uses a 5LQF fragment pair and protein pocket and performs all 500 diffusion steps. Outputs include molecular structures, sampled coordinates and evaluation tables. Use `--device cuda` for a compatible GPU.

## Training and figures

[MODEL_WORKFLOW.md](MODEL_WORKFLOW.md) provides the commands for public-data preparation, fine-tuning, the eight training configurations and full sampling. [figures/README.md](figures/README.md) lists plotting notebooks, VMD setup files and LigPlot input instructions. Raw plot data and structures are supplied by the Zenodo archive.

| Directory | Purpose |
|---|---|
| `src/kinase_gap/` | Dataset preparation, chemistry, training, generation and evaluation |
| `vendor/DiffLinker/` | DiffLinker implementation and required scoring resources |
| `scripts/` | Command-line entrypoints |
| `configs/` | Data, model and sampling settings |
| `environment/` | Recorded dependencies |
| `figures/` | Plot notebooks and structural visualization setups |

## License and citations

The code uses the included [MIT license](LICENSE). Third-party notices are in [licenses/](licenses/). Cite the study, its Zenodo record and the DiffLinker method when using the code. Software and dataset references are listed in `references.bib`.
