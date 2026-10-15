# Checkpoint continuation

## Training records

A logical HPO trial is defined by its materialized configuration, trial index and validation objective. Trials 4–7 have parent and continuation execution records under `recovery_operations/a100_relocation_20261003T150300Z/` in the Zenodo archive. The model-copy command below supplies those records and their original source files to the new experiment workspace. The original configuration specifies a maximum of 100 total epochs, early-stopping patience 20, a minimum improvement of 0.0001 and seed 42.

At each continuation, the worker restores the model tensors, optimizer tensors and parameter groups, completed epoch, global step, early-stopping history and validation-ranked best checkpoint. A startup callback compares these values with the captured checkpoint before the first new optimizer step. The prior best checkpoint remains eligible alongside checkpoints produced by the continuation. Each execution segment initializes its global and data-loader random generators with seed 42.

| Record | Contents |
|---|---|
| `provenance/run_manifest.json` | Selected model, scientific outputs and the `checkpoint_continuation` references |
| HPO `selection.json` | Logical trial outcomes, checkpoint ranking, each continued trial's `segment_lineage`, checkpoint-bound `metric_step_alignment` and the `learning_curve_renderer` file identity |
| `operations/segment_map.json` within the operation directory | Trial-to-contract mapping |
| Per-trial snapshot receipt | Parent last/best checkpoints, metric CSV, metadata and the completed-epoch cutoff |
| Per-trial allocation and activation receipts | Device/runtime identity and the checkpoint transfer |
| `runtime/pre_optimizer_state_verified.json` within each trial's operation records | Model/optimizer state digests, restored counters, stopping history and retained best checkpoint |
| Per-trial completion receipt | Continuation result, checkpoint and metric-file identities |

The source configuration remains attached to the logical trial. The continuation's training metadata records the checkpoint supplied through `--resume-checkpoint`; its restoration receipt supplies the state checks. The archive manifest identifies the bytes of each record and model file.

## Use the saved model and its history

Create the environment and fresh experiment workspace using [Model usage](MODEL_USAGE.md). Its model-copy command is:

```bash
"$KG_PYTHON" -B "$KINASE_ARCHIVE/scripts/copy_archived_segment_model_bundle.py" \
  --archive "$KINASE_ARCHIVE" --workspace "$KINASE_MODEL_WORK"
```

The helper verifies the completed archive and workspace input manifests, then copies checkpoints, HPO selection/configuration records, training logs and their continuation evidence. It preserves relative paths and publishes `reproduction/archived_model/bundle.json`. The `checkpoint`, `selection`, `generation_config` and `training_config` fields provide the paths for model execution; `operation_prefixes` identifies the copied continuation directories.

Use the generation commands in [Model usage](MODEL_USAGE.md) with the bundle's checkpoint and selection record. For a new fine-tuning study from the pretrained initialization, follow [Reproduce the complete workflow](REPRODUCE.md).

## Rebuild joined learning curves

After copying the model bundle, activate the experiment workspace and run:

```bash
source "$KINASE_MODEL_WORK/reproduction/activate.sh"
"$KG_PYTHON" - <<'PY_CURVES'
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

bundle = json.loads(Path("reproduction/archived_model/bundle.json").read_text())
assert bundle["status"] == "copied"
selection_path = Path(bundle["selection"])
selection = json.loads(selection_path.read_text())
assert selection["status"] == "complete"
renderer = selection["learning_curve_renderer"]
plotter = Path(renderer["path"])
assert plotter.stat().st_size == renderer["bytes"]
assert hashlib.sha256(plotter.read_bytes()).hexdigest() == renderer["sha256"]
output = "reproduction/learning_curves_" + datetime.datetime.now(
    datetime.timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")
subprocess.run([
    sys.executable, "-B", str(plotter),
    "--selection", str(selection_path), "--output", output,
], check=True)
PY_CURVES
```

The plotter joins the parent history through the captured checkpoint with the subsequent continuation history. It uses the parent and terminal checkpoint counters to express recorded metric steps on a cumulative optimizer-step axis. The selection record stores the verified offset for each continuation. Each raw CSV row retains its recorded step, source log, execution-segment identifier and checkpoint-interval status; plotting tables also record the cumulative step. The cumulative step aligns training and validation epoch means. The output includes `metric_rows.csv`, `learning_curve_points.csv`, `trial_summary.csv`, PNG/PDF/SVG figures, provenance and a completion record. The selected checkpoint's validation value is checked against the joined trace.
