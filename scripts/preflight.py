#!/usr/bin/env python3
"""Fail-closed validation for the environment, data, checkpoint, and CUDA job."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(root: Path, value: str, label: str, must_exist: bool = True) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be root-relative: {value}")
    result = (root / path).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the project root: {value}") from exc
    if must_exist and not result.exists():
        raise FileNotFoundError(f"Missing {label}: {value}")
    return result


def count_sdf(path: Path, chem: Any) -> int:
    return sum(mol is not None for mol in chem.SDMolSupplier(str(path), sanitize=False))


def move_to_device(value: Any, device: str) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def all_finite(value: Any) -> bool:
    """Recursively validate tensor and scalar model diagnostics."""
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, numbers.Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    raise TypeError(f"Unsupported model output type in smoke test: {type(value).__name__}")


def environment_report(require_cuda: bool) -> Dict[str, Any]:
    import Bio
    import networkx
    import numpy
    import pandas
    import pytorch_lightning
    import rdkit
    import scipy
    import sklearn
    import torch
    import torchmetrics
    import yaml

    executable = str(Path(sys.executable).resolve())
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false")
    report: Dict[str, Any] = {
        "python_executable": executable,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
        "pytorch_lightning": pytorch_lightning.__version__,
        "torchmetrics": torchmetrics.__version__,
        "networkx": networkx.__version__,
        "biopython": Bio.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    expected = {
        "python": "3.10.5",
        "numpy": "1.22.4",
        "pandas": "1.4.4",
        "scipy": "1.8.1",
        "scikit_learn": "1.1.2",
        "rdkit": "2022.03.5",
        "torch": "1.11.0",
        "pytorch_lightning": "1.6.3",
        "torchmetrics": "0.9.3",
        "networkx": "2.8.6",
        "biopython": "1.79",
        "pyyaml": "6.0.3",
        "cuda_runtime": "11.3",
    }
    mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Pinned runtime version mismatch: {mismatches}")
    report["pinned_versions_verified"] = expected
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return report


def verify_checksum_manifest(root: Path, manifest_relative: str) -> Dict[str, Any]:
    manifest = relative_path(root, manifest_relative, "checksum manifest")
    checked = 0
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            expected, value = raw.split(None, 1)
        except ValueError as exc:
            raise ValueError(f"Malformed {manifest.name} line {number}") from exc
        value = value.lstrip("*")
        path = relative_path(root, value, f"{manifest.name} entry")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {value}: {observed}")
        checked += 1
    if checked == 0:
        raise ValueError(f"Checksum manifest is empty: {manifest_relative}")
    return {
        "manifest": manifest_relative,
        "manifest_sha256": sha256_file(manifest),
        "files_verified": checked,
    }


def verify_source_manifest(root: Path) -> Dict[str, Any]:
    script = root / "scripts/source_manifest.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Source-manifest verification failed:\n{result.stdout}")
    manifest = root / "artifacts/source.sha256"
    return {
        "manifest": str(manifest.relative_to(root)),
        "manifest_sha256": sha256_file(manifest),
        "files_verified": sum(1 for line in manifest.read_text().splitlines() if line.strip()),
    }


def verify_file_manifest(data_dir: Path) -> Dict[str, Any]:
    manifest = data_dir / "files.sha256.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Processed-data hash manifest is missing: {manifest}")
    checked = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = data_dir / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest input is missing: {path.name}")
            if path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"Size mismatch for {path.name}")
            observed = sha256_file(path)
            if observed != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {observed}")
            checked += 1
    if checked == 0:
        raise ValueError("Processed-data hash manifest is empty")
    return {"manifest": manifest.name, "files_verified": checked, "sha256": sha256_file(manifest)}


def verify_dataset(root: Path, config: Mapping[str, Any], chem: Any, pandas: Any) -> Dict[str, Any]:
    data_dir = relative_path(root, str(config["data"]), "data")
    report_path = data_dir / "dataset_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Dataset report is missing: {report_path}")
    with report_path.open() as handle:
        dataset_report = json.load(handle)
    if dataset_report.get("dataset_version") != "klifs_v1":
        raise ValueError("Training config does not point at the prepared klifs_v1 dataset")
    preparation_config = root / "configs/data_klifs_v1.yml"
    if not preparation_config.is_file():
        raise FileNotFoundError("Dataset preparation config is missing")
    preparation_config_sha256 = sha256_file(preparation_config)
    if dataset_report.get("configuration_sha256") != preparation_config_sha256:
        raise ValueError(
            "Processed dataset was not built from the current configs/data_klifs_v1.yml"
        )

    graph_path = root / "provenance/prepared_graph_validation.json"
    with graph_path.open() as handle:
        graph_report = json.load(handle)
    if graph_report.get("passed") is not True or graph_report.get("failures"):
        raise ValueError("Independent parent-graph validation did not pass")
    if set(graph_report.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("Independent parent-graph validation must cover all three splits")
    for value, expected_hash in graph_report["inputs"].items():
        if sha256_file(relative_path(root, value, "graph validation input")) != expected_hash:
            raise ValueError(f"Independent parent-graph validation is stale for {value}")

    split_rows: Dict[str, Any] = {}
    all_tables = {}
    for split, prefix in (("train", "KLIFS_train"), ("validation", "KLIFS_val"), ("test", "KLIFS_test")):
        files = {
            "table": data_dir / f"{prefix}_table.csv",
            "mol": data_dir / f"{prefix}_mol.sdf",
            "frag": data_dir / f"{prefix}_frag.sdf",
            "link": data_dir / f"{prefix}_link.sdf",
            "pockets": data_dir / f"{prefix}_pockets.pkl",
        }
        missing = [path.name for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {split} dataset artifacts: {', '.join(missing)}")
        table = pandas.read_csv(files["table"])
        if table.empty:
            raise ValueError(f"{split} table is empty")
        expected = len(table)
        counts = {name: count_sdf(files[name], chem) for name in ("mol", "frag", "link")}
        with files["pockets"].open("rb") as handle:
            pocket_count = len(pickle.load(handle))
        if set(counts.values()) != {expected} or pocket_count != expected:
            raise ValueError(
                f"{split} alignment failure: table={expected}, sdf={counts}, pockets={pocket_count}"
            )
        if not ((table.anchor_1 >= 0) & (table.anchor_1 < table.fragments_size)).all():
            raise ValueError(f"{split} contains invalid anchor_1 indices")
        if not ((table.anchor_2 >= 0) & (table.anchor_2 < table.fragments_size)).all():
            raise ValueError(f"{split} contains invalid anchor_2 indices")
        if (table.anchor_1 == table.anchor_2).any():
            raise ValueError(f"{split} contains duplicate anchors")
        if not (table.pocket_cutoff_angstrom == 10.0).all():
            raise ValueError(f"{split} was not built with the declared 10 Å pocket cutoff")
        all_tables[split] = table
        split_rows[split] = {"rows": expected, "sdf_counts": counts, "pockets": pocket_count}

    leakage = {}
    for key in ("pdb", "canonical_smiles", "inchikey", "murcko_scaffold"):
        leakage[key] = {}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = set(all_tables[first][key].astype(str)) & set(all_tables[second][key].astype(str))
            leakage[key][f"{first}_{second}"] = len(overlap)
            if overlap:
                raise ValueError(f"Forbidden {first}/{second} leakage in {key}: {len(overlap)} values")
    overlap_report = root / "data/manifests/klifs_v1/moad_pdb_overlap.json"
    if not overlap_report.is_file():
        raise FileNotFoundError("Binding MOAD overlap audit has not been generated")
    with overlap_report.open() as handle:
        overlap_payload = json.load(handle)
    forced_targets = {
        value.lower()
        for value in dataset_report["configuration"]["split"]["forced_test_pdbs"]
    }
    if set(overlap_payload.get("manuscript_target_pdbs", [])) != forced_targets:
        raise ValueError("Binding MOAD audit is not bound to the manuscript target registry")
    moad_path = relative_path(
        root, str(overlap_payload.get("moad_metadata", "")), "MOAD metadata"
    )
    if sha256_file(moad_path) != overlap_payload.get("moad_metadata_sha256"):
        raise ValueError("Binding MOAD metadata hash does not match its audit report")
    for label in ("contexts", "targets"):
        audited_path = relative_path(
            root, str(overlap_payload.get(label, "")), f"MOAD audit {label}"
        )
        if sha256_file(audited_path) != overlap_payload.get(f"{label}_sha256"):
            raise ValueError(f"Binding MOAD audit is stale for {label}.csv")
    return {
        "data": str(data_dir.relative_to(root)),
        "preparation_config": str(preparation_config.relative_to(root)),
        "preparation_config_sha256": preparation_config_sha256,
        "dataset_report_sha256": sha256_file(report_path),
        "parent_graph_validation": str(graph_path.relative_to(root)),
        "parent_graph_validation_sha256": sha256_file(graph_path),
        "files": verify_file_manifest(data_dir),
        "splits": split_rows,
        "split_group_overlap": leakage,
        "moad_pdb_overlap_report": str(overlap_report.relative_to(root)),
        "moad_pdb_overlap_sha256": sha256_file(overlap_report),
    }


def verify_model(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    require_cuda: bool,
) -> Dict[str, Any]:
    import torch

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "vendor" / "DiffLinker"))
    from kinase_gap import train as training

    deterministic_warn_only = config.get("deterministic_warn_only")
    if not isinstance(deterministic_warn_only, bool):
        raise TypeError("deterministic_warn_only must be an explicit YAML boolean")
    training.configure_torch_determinism(warn_only=deterministic_warn_only)

    data_dir = relative_path(root, str(config["data"]), "data")
    checkpoint_value = config.get("init_checkpoint")
    if not checkpoint_value or config.get("resume_checkpoint"):
        raise ValueError("Prepared submission requires one init_checkpoint and no resume_checkpoint")
    checkpoint_path = relative_path(root, str(checkpoint_value), "init_checkpoint")
    checkpoint_manifest = root / "models/pretrained/checkpoint_manifest.json"
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError("Checkpoint provenance manifest is missing")
    with checkpoint_manifest.open() as handle:
        checkpoint_payload = json.load(handle)
    matches = [
        item
        for item in checkpoint_payload.get("files", [])
        if item.get("filename") == checkpoint_path.name
    ]
    if len(matches) != 1:
        raise ValueError("Configured checkpoint is not uniquely recorded in its manifest")
    checkpoint_record = matches[0]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != checkpoint_record.get("sha256"):
        raise ValueError("Configured checkpoint hash does not match its provenance manifest")
    if checkpoint_path.stat().st_size != int(checkpoint_record.get("bytes", -1)):
        raise ValueError("Configured checkpoint size does not match its provenance manifest")
    device = "cuda:0" if require_cuda else "cpu"
    model = training.build_model(config, data_dir, device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload is not a mapping")
    training.validate_checkpoint_metadata(payload, config, model)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)

    # Building the datasets here creates portable CPU caches and verifies one
    # representative collated batch before the Slurm submission is allowed.
    model.setup(stage="fit")
    train_batch = next(iter(model.train_dataloader()))
    val_batch = next(iter(model.val_dataloader()))
    batch_report = {
        "train_shape": list(train_batch["positions"].shape),
        "validation_shape": list(val_batch["positions"].shape),
    }
    if require_cuda:
        model = model.to(device)
        model.train_dataset = model.train_dataset
        with torch.no_grad():
            for label, batch, training_mode in (
                ("train", train_batch, True),
                ("validation", val_batch, False),
            ):
                outputs = model.forward(move_to_device(batch, device), training=training_mode)
                if not all_finite(outputs):
                    raise ValueError(f"Non-finite model output in {label} smoke batch")

    hparams = training.checkpoint_hparams(payload)
    return {
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest": str(checkpoint_manifest.relative_to(root)),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_hyperparameters": {
            key: hparams.get(key) for key in training.CHECKPOINT_METADATA_KEYS
        },
        "strict_state_load": True,
        "smoke_batches": batch_report,
        "cuda_forward_pass": bool(require_cuda),
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project root; defaults to the root containing this script")
    parser.add_argument("--config", default="configs/finetune_anchors.yml")
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else SCRIPT_ROOT.resolve()
    if root != SCRIPT_ROOT.resolve():
        raise ValueError(f"--root does not match the project containing this script: {root}")
    report: Dict[str, Any] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "environment": environment_report(args.require_cuda),
        "environment_artifacts": verify_checksum_manifest(
            root, "artifacts/environment.sha256"
        ),
    }
    if not args.environment_only:
        import pandas
        import yaml
        from rdkit import Chem

        config_path = relative_path(root, args.config, "config")
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        for key in ("data", "checkpoint_dir", "log_dir", "init_checkpoint"):
            if config.get(key):
                relative_path(root, str(config[key]), f"config:{key}", must_exist=key in {"data", "init_checkpoint"})
        report["config"] = str(config_path.relative_to(root))
        report["config_sha256"] = sha256_file(config_path)
        report["source"] = verify_source_manifest(root)
        report["dataset"] = verify_dataset(root, config, Chem, pandas)
        report["model"] = verify_model(root, config_path, config, args.require_cuda)

    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / ("preflight.cuda.json" if args.require_cuda else "preflight.json")
    with output.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"Preflight passed; report written to {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
