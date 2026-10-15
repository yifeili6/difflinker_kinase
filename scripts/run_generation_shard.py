#!/usr/bin/env python3
"""Run one disjoint generation shard from a hash-verified HPO selection."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kinase_gap import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "initialize", "shard", "finalize"))
    parser.add_argument("--selection", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/generation.yml")
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    selection_path = generate.project_path(args.selection, must_exist=True)
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete":
        raise ValueError("HPO selection is incomplete")
    winner = selection["winner"]
    checkpoint = winner["selected_checkpoint"]
    if generate.sha256_file(generate.project_path(checkpoint, must_exist=True)) != winner["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash differs from the HPO selection")
    forwarded = ["--config", args.config, "--checkpoint", checkpoint,
                 "--selection", args.selection, "--run-id", args.run_id]
    if args.mode == "shard":
        index = args.shard_index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        forwarded.extend(["--shard-index", str(index)])
    else:
        forwarded.append({"preflight": "--preflight-only", "initialize": "--initialize-only",
                          "finalize": "--finalize-only"}[args.mode])
    return generate.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
