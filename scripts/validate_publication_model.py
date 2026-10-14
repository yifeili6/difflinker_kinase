#!/usr/bin/env python3
"""Verify the selected checkpoint by finite-state and strict CPU model loading."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", default="configs/generation.yml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from kinase_gap.generate import instantiate_model, load_checkpoint_payload, read_config, validate_checkpoint
    selection = json.loads(args.selection.read_text())
    winner = selection["winner"]
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if selection.get("status") != "complete" or digest != winner["selected_checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the completed HPO selection")
    if not math.isfinite(float(winner["objective"])):
        raise ValueError("Selected validation objective is non-finite")
    payload = load_checkpoint_payload(args.checkpoint.resolve())
    state = payload["state_dict"]
    if not state or any(not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("Selected checkpoint contains an empty, non-tensor or non-finite model state")
    if int(payload["epoch"]) != winner["best_epoch"] or int(payload["global_step"]) != winner["global_step"]:
        raise ValueError("Selected checkpoint epoch/step differs from the HPO record")
    config = read_config(args.config)
    information = validate_checkpoint(payload, args.checkpoint.resolve(), config)
    model = instantiate_model(payload, config, torch.device("cpu"))
    if any(tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise ValueError("Strictly loaded model is not finite and entirely on CPU")
    packages = {}
    for name in ("torch", "pytorch-lightning", "rdkit", "posebusters", "molsets", "numpy", "pandas", "matplotlib", "scipy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"status": "validated", "checkpoint": information, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              "strict_state_dict": True, "finite_model_state": True, "device": "cpu",
              "state_tensors": len(state), "state_elements": sum(tensor.numel() for tensor in state.values()),
              "software": {"python": platform.python_version(), **packages}}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": "validated", "sha256": digest, "device": "cpu", "state_tensors": len(state)}))


if __name__ == "__main__":
    main()
