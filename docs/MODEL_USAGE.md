# Model usage

DiffLinker predicts a linker from a pair of fixed ligand fragments, their atom types, the fragment attachment atoms and protein pocket coordinates. The pocket contains protein residues within 10 Å of the native ligand. The input atom order is shared by the fragment coordinates, atom features and anchor flags.

## Checkpoint contents

The pretrained model is the full-pocket, anchor-conditioned DiffLinker checkpoint distributed as `pockets_difflinker_full_fc_pdb_excluded.ckpt`. Its archive name and SHA-256 are recorded in `models/pretrained/checkpoint_manifest.json`. The fine-tuned checkpoint uses the same network architecture and records its selected training configuration, validation score and initialization checkpoint.

The architecture uses an EGNN denoising network with nine ligand element classes: C, N, O, F, P, S, Cl, Br and I. Atom features and three context indicators enter a 128-dimensional representation. The recorded configuration defines the message-passing layers, graph construction, coordinate normalization and diffusion schedule. Read the checkpoint and configuration together when reconstructing the network.

## Environment

The archived environment specification records Python, PyTorch, PyTorch Lightning, RDKit and the evaluation packages. Install the supplied exact package specifications for the environment assigned to the stage. GPU generation uses the CUDA runtime listed in that environment and an NVIDIA driver compatible with it. Property calculations and figure rendering use the evaluation and visualization environments.

From the archive root, create the recorded Python 3.10 environment at a new absolute prefix. Conda must be available; set `KINASE_CONDA` to its executable path when needed. The creation script installs the exact package lock and locally archived wheels. Slurm launchers use this environment through `KINASE_RUNTIME_ENV`.

```bash
export KINASE_ARCHIVE="$PWD"
export KINASE_RUNTIME_ENV="${KINASE_RUNTIME_ENV:?Set the recorded environment prefix}"
bash "$KINASE_ARCHIVE/scripts/create_environment.sh" "$KINASE_RUNTIME_ENV"
export KG_PYTHON="$KINASE_RUNTIME_ENV/bin/python"
export PATH="$KINASE_RUNTIME_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$KINASE_ARCHIVE/src:$KINASE_ARCHIVE/vendor/DiffLinker"
"$KINASE_RUNTIME_ENV/bin/python" -B "$KINASE_ARCHIVE/scripts/verify_archive.py" "$KINASE_ARCHIVE"
```

For an environment already created from these specifications, set `KINASE_RUNTIME_ENV` to its prefix and apply the exports above. Use its absolute Python path for the commands below.

Load checkpoint files supplied with this archive, whose identities can be verified against the checksum manifest. PyTorch checkpoint loading uses Python object serialization.

## Inspect configuration and inputs

The run manifest, `provenance/run_manifest.json`, names the selected checkpoint and the generation configuration. The generation entrypoint reads the checkpoint from its `--selection` record and verifies the recorded checkpoint checksum. The configuration supplies the test context table, row-aligned fragment SDF, pocket tensor data, linker lengths and random seed. Each context identifies the source structure, fragmentation pattern, two anchor indices and retained fragment atom count.

A linker-size setting specifies the number of generated heavy atoms. The retained fragment coordinates stay in the receptor coordinate frame. The model centers coordinates according to its recorded `center_of_mass` setting before sampling and restores the frame afterward.

## Prepare a workspace for the saved model

With `KINASE_ARCHIVE` and `KINASE_RUNTIME_ENV` set above, choose a new directory outside the archive:

```bash
export KINASE_MODEL_WORK="$KINASE_ARCHIVE/../kinase-saved-model-experiment"
"$KG_PYTHON" -B "$KINASE_ARCHIVE/scripts/create_authoritative_workspace.py" \
  --source "$KINASE_ARCHIVE" --destination "$KINASE_MODEL_WORK"
"$KG_PYTHON" -B "$KINASE_ARCHIVE/scripts/copy_archived_segment_model_bundle.py" \
  --archive "$KINASE_ARCHIVE" --workspace "$KINASE_MODEL_WORK"
source "$KINASE_MODEL_WORK/reproduction/activate.sh"
```

The workspace contains the native inputs, prepared dataset, code and environment specifications. The model-bundle helper adds the archived checkpoints, HPO configurations, selection records, training logs and checkpoint-continuation records, verifying every copied file against the archive manifest. Its `reproduction/archived_model/bundle.json` identifies the files used by the commands below. The archived `run_manifest.json` and `artifact_manifest.json` remain under `reproduction/archived_model/` as reference records for the source run.

The bundle preserves the parent checkpoints, continuation checkpoints, restoration receipts and raw metric tables at their recorded relative paths. Its `operation_prefixes` field names the included execution-record directories. [Checkpoint continuation](CHECKPOINT_CONTINUATION.md) describes these records and gives the command for plotting the joined training history.

Generation reads the prepared fragment and pocket records directly. CPU tensor caches are built when preparing training, as described below.

## Generate molecules

Inspect the command options provided by the archived version:

```bash
"$KINASE_RUNTIME_ENV/bin/python" scripts/run_generation_with_stereo.py --help
"$KINASE_RUNTIME_ENV/bin/python" -m kinase_gap.train --help
```

On an allocated GPU node with one visible GPU, generate 128 samples for condition 0 using the experiment workspace's model-bundle record:

```bash
"$KINASE_RUNTIME_ENV/bin/python" - <<'PY'
import datetime
import json
import subprocess
import sys

record = json.load(open("reproduction/archived_model/bundle.json"))
assert record["status"] == "copied"
run_id = "generation_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
entrypoint = [sys.executable, "scripts/run_generation_with_stereo.py"]
arguments = [
    "--config", record["generation_config"],
    "--selection", record["selection"],
    "--run-id", run_id,
]
subprocess.run(entrypoint + ["preflight"] + arguments, check=True)
subprocess.run(entrypoint + ["shard"] + arguments + ["--shard-index", "0"], check=True)
PY
```

The supplied configuration defines 372 conditions across the target panel. The example writes condition 0 under the new run identifier. The [complete workflow](REPRODUCE.md) submits all 372 conditions as a Slurm array. A serial full-panel run invokes `shard` for indices 0 through 371 and then invokes `finalize` with the same selection and run identifier.

The entrypoint uses the recorded sampler and the accompanying `scripts/serialization_patch.py` writer. For candidates that pass sanitization, atom and bond stereo assignments are saved in the SDF record. Existing double-bond stereo is encoded in neighboring bond directions when needed, with exact molecular-identity and coordinate checks.

Each generation shard contains candidate identifiers, molecular SDF records, tabular sampling metadata and saved coordinates. Once all conditions complete, the consolidated candidate table maps every attempted sample to its context and output record. Retain every attempted sample when calculating overall validity and filter survival.

## Train with the selected settings

The model bundle includes the selected materialized trial configuration and its pretrained initialization checkpoint. This command starts a new fine-tuning run from that initialization with the recorded optimizer, seed, 100-epoch maximum and early-stopping patience 20. The archived model's parent and continuation histories are described in [Checkpoint continuation](CHECKPOINT_CONTINUATION.md).

Build the CPU tensor inputs once in the experiment workspace. If they were built using the provenance guide, continue with GPU validation below.

```bash
"$KG_PYTHON" -B scripts/build_tensor_caches.py
```

On an allocated GPU node, validate the current runtime and start training:

```bash
"$KG_PYTHON" - <<'PY'
import datetime
import json
import subprocess
import sys

bundle = json.load(open("reproduction/archived_model/bundle.json"))
assert bundle["status"] == "copied"
config = bundle["training_config"]
subprocess.run([
    sys.executable, "scripts/preflight.py", "--config", config, "--require-cuda",
], check=True)
run_id = "training_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
subprocess.run([
    sys.executable, "-m", "kinase_gap.train", "--config", config, "--run-id", run_id,
], check=True)
PY
```

For a complete new hyperparameter search, create a separate fresh workspace and follow [Reproduce the complete workflow](REPRODUCE.md). That route starts from the pretrained initialization, evaluates all declared trial configurations and selects a checkpoint using validation loss.

## Evaluate and inspect

The `paths.evaluation` entry in `provenance/run_manifest.json` identifies the evaluation directory used for this release. Its tables retain `candidate_id`, `canonical_smiles`, the individual PoseBusters checks, molecular properties and cumulative filter flags. `scripts/run_authoritative_evaluation.py` loads generated molecules from their verified embedded molecular graphs and full-precision coordinate arrays. Open selected SDF files together with their corresponding receptor PDB in a molecular viewer to inspect the archived coordinates. [Data and evaluation](DATA_AND_EVALUATION.md) defines the populations used for each reported quantity.

For native and generated ligand chemistry, use the supplied SDF loader:

```bash
"$KINASE_RUNTIME_ENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from kinase_gap.structural_bundle import parse_sdf_bytes

archive = Path(os.environ["KINASE_ARCHIVE"])
record = json.loads((archive / "provenance/run_manifest.json").read_text())
path = archive / record["paths"]["structural"] / "targets/5lqf/reference/native_reference.sdf"
native = parse_sdf_bytes(path.read_bytes(), label=str(path))[0]
PY
```

The loader validates molecular identity and coordinates against the embedded RDKit molecule and restores its exact atom properties, hydrogen representation and full-precision coordinates. Chemistry sidecars preserve the same atom and bond assignments. Retained fragment atoms keep their source stereochemistry, while newly generated atoms receive assignments from the sampled pose.
