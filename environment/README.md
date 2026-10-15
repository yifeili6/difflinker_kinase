# Reproduction environment

The archive records Python 3.10, PyTorch 1.11.0, PyTorch Lightning 1.6.3, RDKit 2022.03.5 and the evaluation packages used for the reported calculations. `explicit-lock.txt` identifies the exact Conda builds. The pip requirement files identify the included wheels by SHA-256. The Zenodo archive's `environment/runtime.json` records the execution environment, and `pymol-explicit-lock.txt` supplies the visualization environment.

From the archive root, create an environment at a new absolute path:

```bash
export KINASE_RUNTIME_ENV="$PWD/../kinase-runtime"
bash scripts/create_environment.sh "$KINASE_RUNTIME_ENV"
export PATH="$KINASE_RUNTIME_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/vendor/DiffLinker"
```

Set `KINASE_CONDA` to select the Conda executable. The creation script installs the recorded packages and verifies their versions. The main environment supplies CUDA 11.3 user-space libraries; GPU jobs use the NVIDIA driver installed on the selected compute node.

For new calculations, create a separate experiment directory with the checkout's `scripts/create_from_archive.py`, as shown in the README. Set `KINASE_RUNTIME_ENV` to an environment visible from the allocated nodes, then source the experiment's `reproduction/activate.sh`. The Slurm launchers read this variable and record the selected device and software through the preflight and run metadata.

See [Complete workflow](../docs/REPRODUCE.md) for workspace creation and verification, [Dataset reconstruction](../docs/DATASET_RECONSTRUCTION.md) for CPU preparation, and [Model usage](../docs/MODEL_USAGE.md) for training and generation.
