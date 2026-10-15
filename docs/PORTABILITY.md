# Portable paths and archive inputs

`source_manifest.json` records the original archive checksum and distributed checksum for each included archive-derived file. `scripts/create_from_archive.py --verify-only` checks both copies. The manifest covers code, configuration and documentation; it does not contain models or study data.

The fresh-workspace command reads the verified Zenodo archive, copies its required inputs into a new directory, and installs the applicable distributed files. It renews source seals and the workspace input inventory for that new directory. The archive is not edited. The workspace builder requires a destination that does not already exist.

The runtime path adaptations are:

| File | Path handling |
|---|---|
| `slurm/runtime.sh` | Uses `SLURM_SUBMIT_DIR`, requires `KINASE_RUNTIME_ENV`, and preserves the caller's `PATH` |
| `scripts/rebuild_corrected_dataset.py` | Takes the source archive through `--source`; default outputs derive from the current project |
| `scripts/preflight.py` | Checks the recorded packages and CUDA without requiring the original cluster's mount names |

Documentation uses the companion archive for data and checkpoints and a separate directory for each experiment. Scientific configuration, molecular graph handling, diffusion, filtering and model architecture retain their archived implementation.

For inference with the saved selected model, follow [Model usage](MODEL_USAGE.md). The archived model-copy helper verifies and copies the checkpoint and its recorded continuation dependencies, including the original source files named by their checksums. [Checkpoint continuation](CHECKPOINT_CONTINUATION.md) gives the joined-learning-curve command. Those byte-matched continuation records and source versions are obtained from Zenodo when that workspace is created.

For new training, use the checkout's `scripts/create_from_archive.py` command in the [README](../README.md), then follow step 2 of [Reproduce the complete workflow](REPRODUCE.md). Set `KINASE_RUNTIME_ENV` before submitting Slurm jobs and submit from the experiment root.
