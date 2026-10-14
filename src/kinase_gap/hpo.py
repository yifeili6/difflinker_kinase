#!/usr/bin/env python3
"""Materialize, validate, and select the prospective kinase HPO study.

The manuscript does not define a training hyperparameter search.  This module
therefore implements an explicit prospective protocol while preserving the
test-only status of manuscript targets.  Trial configurations are generated
under ``outputs/`` so the sealed source manifest remains unchanged at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
OBJECTIVE_NAME = "loss/val"
OBJECTIVE_MODE = "min"


def project_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Path must be project-root-relative: {value}")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes the project root: {value}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required project path does not exist: {value}")
    return resolved


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Publish a byte-for-byte verified copy without exposing partial output."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_value)
    try:
        open_descriptor = descriptor
        descriptor = -1
        with os.fdopen(open_descriptor, "wb") as output_handle, source.open(
            "rb"
        ) as input_handle:
            shutil.copyfileobj(input_handle, output_handle, length=4 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        observed_sha256 = sha256_file(temporary)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "Temporary selected checkpoint failed SHA-256 verification"
            )
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {relative(path)}")
    return payload


def load_json_mapping(path: Path, *, label: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} is not a mapping: {relative(path)}")
    return payload


def normalized_project_file(value: Any, *, label: str) -> Path:
    """Resolve a required file and reject aliases or non-string path values."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty project-relative string")
    path = project_path(value, must_exist=True)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {value}")
    if value != relative(path):
        raise ValueError(f"{label} must be a normalized project-relative path: {value}")
    return path


def validate_study_id(value: str) -> str:
    if value in {".", ".."} or not STUDY_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Study ID may contain only letters, numbers, dot, underscore, and "
            "hyphen, and may not be '.' or '..'"
        )
    return value


def load_search(search_value: str) -> Tuple[Path, Dict[str, Any]]:
    search_path = project_path(search_value, must_exist=True)
    search = load_yaml(search_path)
    if search.get("schema_version") != 1:
        raise ValueError("Unsupported HPO schema_version")

    objective = search.get("objective")
    if not isinstance(objective, Mapping):
        raise TypeError("HPO objective must be a mapping")
    if objective.get("name") != OBJECTIVE_NAME or objective.get("mode") != OBJECTIVE_MODE:
        raise ValueError(
            f"HPO objective must be {OBJECTIVE_NAME!r} in {OBJECTIVE_MODE!r} mode"
        )

    base_value = search.get("base_config")
    if not isinstance(base_value, str):
        raise TypeError("base_config must be a project-relative string")
    project_path(base_value, must_exist=True)

    budget = search.get("budget")
    if not isinstance(budget, Mapping):
        raise TypeError("HPO budget must be a mapping")
    max_epochs = int(budget.get("max_epochs", 0))
    patience = int(budget.get("early_stopping_patience", -1))
    min_delta = float(budget.get("early_stopping_min_delta", -1))
    if max_epochs < 1 or patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("Invalid HPO epoch or early-stopping budget")
    if int(budget.get("save_top_k", 0)) != 1:
        raise ValueError("HPO must save exactly one validation-ranked checkpoint per trial")

    seed = int(search.get("seed", -1))
    if seed < 0:
        raise ValueError("HPO seed must be a non-negative integer")

    trials = search.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("HPO trials must be a non-empty list")
    seen = set()
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping) or set(trial) != {"lr", "batch_size"}:
            raise ValueError(f"Trial {index} must contain exactly lr and batch_size")
        lr = float(trial["lr"])
        batch_size = int(trial["batch_size"])
        if not math.isfinite(lr) or lr <= 0 or batch_size < 1:
            raise ValueError(f"Invalid parameters in trial {index}")
        key = (lr, batch_size)
        if key in seen:
            raise ValueError(f"Duplicate HPO parameters in trial {index}: {key}")
        seen.add(key)
    return search_path, search


def trial_identity(study_id: str, index: int) -> Tuple[str, str]:
    validate_study_id(study_id)
    return f"hpo_trial_{index:02d}", f"hpo-{study_id}-{index:02d}"


def expected_trial_config(
    search_path: Path,
    search: Mapping[str, Any],
    study_id: str,
    index: int,
) -> Dict[str, Any]:
    """Reconstruct the exact immutable configuration declared for one trial."""
    validate_study_id(study_id)
    trials = search["trials"]
    if index < 0 or index >= len(trials):
        raise IndexError(f"Trial index {index} is outside 0..{len(trials) - 1}")

    run_name, run_id = trial_identity(study_id, index)
    base_path = project_path(str(search["base_config"]), must_exist=True)
    config = load_yaml(base_path)
    trial = trials[index]
    budget = search["budget"]
    config.update(
        {
            "run_name": run_name,
            "seed": int(search["seed"]),
            "max_epochs": int(budget["max_epochs"]),
            "early_stopping_patience": int(budget["early_stopping_patience"]),
            "early_stopping_min_delta": float(
                budget["early_stopping_min_delta"]
            ),
            "save_top_k": int(budget["save_top_k"]),
            "lr": float(trial["lr"]),
            "batch_size": int(trial["batch_size"]),
            "enable_progress_bar": False,
            "_hpo": {
                "schema_version": 1,
                "study_id": study_id,
                "trial_index": index,
                "run_id": run_id,
                "search_config": relative(search_path),
                "search_config_sha256": sha256_file(search_path),
                "base_config": relative(base_path),
                "base_config_sha256": sha256_file(base_path),
                "objective": dict(search["objective"]),
            },
        }
    )
    return config


def materialize_trial(
    search_path: Path,
    search: Mapping[str, Any],
    study_id: str,
    index: int,
    output_path: Path,
) -> Dict[str, Any]:
    trials = search["trials"]
    run_name, run_id = trial_identity(study_id, index)
    config = expected_trial_config(search_path, search, study_id, index)
    trial = trials[index]

    expected_parent = project_path(f"outputs/hpo/studies/{study_id}/configs")
    try:
        output_path.resolve().relative_to(expected_parent.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Materialized trial config must be under {relative(expected_parent)}"
        ) from exc

    text = yaml.safe_dump(config, sort_keys=True)
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(
                f"Refusing to replace a different materialized config: {relative(output_path)}"
            )
    else:
        atomic_write_text(output_path, text)
    return {
        "study_id": study_id,
        "trial_index": index,
        "run_name": run_name,
        "run_id": run_id,
        "config": relative(output_path),
        "config_sha256": sha256_file(output_path),
        "parameters": {
            "lr": float(trial["lr"]),
            "batch_size": int(trial["batch_size"]),
        },
    }


def read_training_result(path: Path) -> Dict[str, Any]:
    result = load_json_mapping(path, label="Training result")
    if result.get("schema_version") != 1:
        raise ValueError(f"Unsupported training-result schema: {relative(path)}")
    if result.get("status") != "completed":
        raise ValueError(f"Trial did not report successful completion: {relative(path)}")
    if result.get("objective") != OBJECTIVE_NAME or result.get("objective_mode") != OBJECTIVE_MODE:
        raise ValueError(f"Trial objective mismatch: {relative(path)}")
    try:
        score = float(result.get("best_objective"))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Trial objective is not numeric: {relative(path)}"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(f"Trial objective is not finite: {relative(path)}")
    return result


def require_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} {key} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} {key} must be at least {minimum}")
    return value


def validate_materialized_config(
    *,
    result: Mapping[str, Any],
    search_path: Path,
    search: Mapping[str, Any],
    study_id: str,
    index: int,
) -> Tuple[Path, Dict[str, Any], str]:
    """Bind a result to the exact trial declared by the search specification."""
    config_path = normalized_project_file(
        result.get("config"),
        label=f"Trial {index} config",
    )
    configs_dir = project_path(f"outputs/hpo/studies/{study_id}/configs")
    try:
        config_relative = config_path.resolve().relative_to(configs_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Trial {index} config must be under {relative(configs_dir)}"
        ) from exc
    if config_relative == Path("."):
        raise ValueError(f"Trial {index} config path names the configs directory")

    config_sha256 = sha256_file(config_path)
    if result.get("config_sha256") != config_sha256:
        raise ValueError(f"Trial {index} config hash mismatch: {relative(config_path)}")

    expected_config = expected_trial_config(
        search_path=search_path,
        search=search,
        study_id=study_id,
        index=index,
    )
    observed_config = load_yaml(config_path)
    if observed_config != expected_config:
        differing_keys = sorted(
            key
            for key in set(observed_config) | set(expected_config)
            if observed_config.get(key) != expected_config.get(key)
        )
        raise ValueError(
            f"Trial {index} materialized config differs from the declared search "
            f"at keys: {', '.join(differing_keys)}"
        )

    expected_hpo = expected_config["_hpo"]
    if result.get("hpo") != expected_hpo:
        raise ValueError(f"Trial {index} result HPO metadata mismatch")
    return config_path, observed_config, config_sha256


def validate_run_metadata(
    *,
    result: Mapping[str, Any],
    config_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    run_name: str,
    run_id: str,
    index: int,
) -> Tuple[Path, str]:
    """Verify both result-to-metadata and metadata-to-config provenance links."""
    metadata_path = normalized_project_file(
        result.get("run_metadata"),
        label=f"Trial {index} run metadata",
    )
    log_dir_value = config.get("log_dir")
    if not isinstance(log_dir_value, str) or not log_dir_value:
        raise TypeError(f"Trial {index} log_dir must be a non-empty string")
    expected_metadata_path = (
        project_path(log_dir_value) / run_name / run_id / "run_metadata.json"
    )
    if metadata_path.resolve() != expected_metadata_path.resolve():
        raise ValueError(
            f"Trial {index} run metadata is outside its expected run log directory"
        )

    metadata_sha256 = sha256_file(metadata_path)
    if result.get("run_metadata_sha256") != metadata_sha256:
        raise ValueError(
            f"Trial {index} run metadata hash mismatch: {relative(metadata_path)}"
        )
    metadata = load_json_mapping(metadata_path, label=f"Trial {index} run metadata")
    if metadata.get("run_id") != run_id:
        raise ValueError(f"Trial {index} run metadata identity mismatch")
    if metadata.get("config") != relative(config_path):
        raise ValueError(f"Trial {index} run metadata config path mismatch")
    if metadata.get("config_sha256") != config_sha256:
        raise ValueError(f"Trial {index} run metadata config hash mismatch")

    snapshot_path = metadata_path.parent / "config.snapshot.yml"
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Trial {index} config snapshot is missing: {relative(snapshot_path)}"
        )
    if load_yaml(snapshot_path) != dict(config):
        raise ValueError(f"Trial {index} config snapshot mismatch")

    for mode in ("init_checkpoint", "resume_checkpoint"):
        configured_value = config.get(mode)
        expected_value = None
        expected_sha256 = None
        if configured_value:
            checkpoint = normalized_project_file(
                configured_value,
                label=f"Trial {index} configured {mode}",
            )
            expected_value = relative(checkpoint)
            expected_sha256 = sha256_file(checkpoint)
        if metadata.get(mode) != expected_value:
            raise ValueError(f"Trial {index} run metadata {mode} path mismatch")
        if metadata.get(f"{mode}_sha256") != expected_sha256:
            raise ValueError(f"Trial {index} run metadata {mode} hash mismatch")
    return metadata_path, metadata_sha256


def validate_training_budget(
    *,
    result: Mapping[str, Any],
    search: Mapping[str, Any],
    index: int,
) -> Tuple[int, int]:
    """Reject completed results that do not reflect the declared trial budget."""
    budget = search["budget"]
    max_epochs = int(budget["max_epochs"])
    best_epoch = require_integer(
        result, "best_epoch", label=f"Trial {index} result"
    )
    best_global_step = require_integer(
        result, "global_step", label=f"Trial {index} result"
    )
    trainer_epoch = require_integer(
        result,
        "trainer_current_epoch",
        label=f"Trial {index} result",
        minimum=1,
    )
    trainer_global_step = require_integer(
        result, "trainer_global_step", label=f"Trial {index} result"
    )
    if best_epoch >= max_epochs:
        raise ValueError(f"Trial {index} best epoch exceeds its declared budget")
    if trainer_epoch > max_epochs:
        raise ValueError(f"Trial {index} terminal epoch exceeds its declared budget")
    if best_epoch >= trainer_epoch:
        raise ValueError(f"Trial {index} best epoch is not before terminal epoch")
    if best_global_step > trainer_global_step:
        raise ValueError(f"Trial {index} best step exceeds its terminal step")

    early_stopping = result.get("early_stopping")
    if not isinstance(early_stopping, Mapping):
        raise TypeError(f"Trial {index} early_stopping result must be a mapping")
    if set(early_stopping) != {
        "enabled",
        "patience",
        "min_delta",
        "stopped_epoch",
    }:
        raise ValueError(f"Trial {index} early-stopping metadata fields mismatch")
    if early_stopping.get("enabled") is not True:
        raise ValueError(f"Trial {index} did not enable declared early stopping")
    if early_stopping.get("patience") != int(
        budget["early_stopping_patience"]
    ):
        raise ValueError(f"Trial {index} early-stopping patience mismatch")
    min_delta = early_stopping.get("min_delta")
    if isinstance(min_delta, bool):
        raise TypeError(f"Trial {index} early-stopping min_delta is not numeric")
    try:
        observed_min_delta = float(min_delta)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Trial {index} early-stopping min_delta is not numeric"
        ) from exc
    if observed_min_delta != float(budget["early_stopping_min_delta"]):
        raise ValueError(f"Trial {index} early-stopping min_delta mismatch")

    stopped_epoch = early_stopping.get("stopped_epoch")
    if stopped_epoch is None:
        if trainer_epoch != max_epochs:
            raise ValueError(
                f"Trial {index} ended before its budget without early stopping"
            )
    else:
        if isinstance(stopped_epoch, bool) or not isinstance(stopped_epoch, int):
            raise TypeError(f"Trial {index} stopped_epoch must be an integer or null")
        if stopped_epoch < 1 or stopped_epoch >= trainer_epoch:
            raise ValueError(f"Trial {index} stopped_epoch is inconsistent")
    return best_epoch, best_global_step


def select_winner(
    search_path: Path,
    search: Mapping[str, Any],
    study_id: str,
) -> Dict[str, Any]:
    validate_study_id(study_id)
    search_sha256 = sha256_file(search_path)
    if load_yaml(search_path) != dict(search):
        raise ValueError("In-memory HPO search does not match its search configuration")
    trial_records: List[Dict[str, Any]] = []
    for index, trial in enumerate(search["trials"]):
        run_name, run_id = trial_identity(study_id, index)
        run_dir = project_path(f"models/runs/{run_name}/{run_id}", must_exist=True)
        result_path = run_dir / "training_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Missing successful result for HPO trial {index}: {relative(result_path)}"
            )
        result = read_training_result(result_path)
        if result.get("run_name") != run_name or result.get("run_id") != run_id:
            raise ValueError(f"Run identity mismatch in {relative(result_path)}")

        config_path, config, config_sha256 = validate_materialized_config(
            result=result,
            search_path=search_path,
            search=search,
            study_id=study_id,
            index=index,
        )
        metadata_path, metadata_sha256 = validate_run_metadata(
            result=result,
            config_path=config_path,
            config=config,
            config_sha256=config_sha256,
            run_name=run_name,
            run_id=run_id,
            index=index,
        )
        best_epoch, best_global_step = validate_training_budget(
            result=result,
            search=search,
            index=index,
        )

        checkpoint = normalized_project_file(
            result.get("best_model_path"),
            label=f"Trial {index} best checkpoint",
        )
        try:
            checkpoint.resolve().relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Trial checkpoint escapes its run directory: {relative(checkpoint)}"
            ) from exc
        observed_sha = sha256_file(checkpoint)
        if observed_sha != result.get("best_model_sha256"):
            raise ValueError(f"Trial checkpoint hash mismatch: {relative(checkpoint)}")

        trial_records.append(
            {
                "trial_index": index,
                "run_name": run_name,
                "run_id": run_id,
                "parameters": {
                    "lr": float(trial["lr"]),
                    "batch_size": int(trial["batch_size"]),
                },
                "seed": int(search["seed"]),
                "budget": {
                    "max_epochs": int(search["budget"]["max_epochs"]),
                    "early_stopping_patience": int(
                        search["budget"]["early_stopping_patience"]
                    ),
                    "early_stopping_min_delta": float(
                        search["budget"]["early_stopping_min_delta"]
                    ),
                    "save_top_k": int(search["budget"]["save_top_k"]),
                },
                "objective": float(result["best_objective"]),
                "best_epoch": best_epoch,
                "global_step": best_global_step,
                "checkpoint": relative(checkpoint),
                "checkpoint_sha256": observed_sha,
                "config": relative(config_path),
                "config_sha256": config_sha256,
                "run_metadata": relative(metadata_path),
                "run_metadata_sha256": metadata_sha256,
                "training_result": relative(result_path),
                "training_result_sha256": sha256_file(result_path),
            }
        )

    if sha256_file(search_path) != search_sha256:
        raise RuntimeError("HPO search configuration changed during winner selection")
    for record in trial_records:
        for path_key, hash_key, label in (
            ("config", "config_sha256", "materialized config"),
            ("run_metadata", "run_metadata_sha256", "run metadata"),
            ("training_result", "training_result_sha256", "training result"),
            ("checkpoint", "checkpoint_sha256", "checkpoint"),
        ):
            path = project_path(record[path_key], must_exist=True)
            if sha256_file(path) != record[hash_key]:
                raise RuntimeError(
                    f"Trial {record['trial_index']} {label} changed during "
                    "winner selection"
                )
    ranked = sorted(trial_records, key=lambda item: (item["objective"], item["trial_index"]))
    winner = ranked[0]
    source_checkpoint = project_path(winner["checkpoint"], must_exist=True)
    selected_dir = project_path(f"models/selected/hpo-{study_id}")
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = selected_dir / "best.ckpt"
    atomic_copy_verified(
        source_checkpoint,
        selected_checkpoint,
        winner["checkpoint_sha256"],
    )
    selected_sha = sha256_file(selected_checkpoint)
    if selected_sha != winner["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint copy failed SHA-256 verification")

    generated_at = datetime.now(timezone.utc).isoformat()
    selection = {
        "schema_version": 1,
        "status": "complete",
        "selected_at_utc": generated_at,
        "study_id": study_id,
        "search_config": relative(search_path),
        "search_config_sha256": search_sha256,
        "objective": OBJECTIVE_NAME,
        "objective_mode": OBJECTIVE_MODE,
        "seed": int(search["seed"]),
        "budget": {
            "max_epochs": int(search["budget"]["max_epochs"]),
            "early_stopping_patience": int(
                search["budget"]["early_stopping_patience"]
            ),
            "early_stopping_min_delta": float(
                search["budget"]["early_stopping_min_delta"]
            ),
            "save_top_k": int(search["budget"]["save_top_k"]),
        },
        "tie_breaker": "lowest trial_index",
        "trial_count": len(trial_records),
        "winner": {
            **winner,
            "selected_checkpoint": relative(selected_checkpoint),
            "selected_checkpoint_sha256": selected_sha,
        },
        "trials": sorted(trial_records, key=lambda item: item["trial_index"]),
        "ranking": [item["trial_index"] for item in ranked],
    }

    study_dir = project_path(f"outputs/hpo/studies/{study_id}")
    selection_path = study_dir / "selection.json"
    atomic_write_text(selection_path, json.dumps(selection, indent=2, sort_keys=True) + "\n")

    rows = [
        {
            "rank": rank,
            "trial_index": item["trial_index"],
            "lr": item["parameters"]["lr"],
            "batch_size": item["parameters"]["batch_size"],
            "objective": item["objective"],
            "best_epoch": item["best_epoch"],
            "run_id": item["run_id"],
            "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["checkpoint_sha256"],
        }
        for rank, item in enumerate(ranked, 1)
    ]
    fieldnames = list(rows[0])
    csv_lines: List[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_lines.append(buffer.getvalue())
    atomic_write_text(study_dir / "ranking.csv", "".join(csv_lines))
    return selection


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser("count", help="print the number of trials")
    count_parser.add_argument("--search", default="configs/hpo_grid.yml")

    materialize_parser = subparsers.add_parser(
        "materialize", help="write one immutable runtime trial config"
    )
    materialize_parser.add_argument("--search", default="configs/hpo_grid.yml")
    materialize_parser.add_argument("--study-id", required=True)
    materialize_parser.add_argument("--trial-index", required=True, type=int)
    materialize_parser.add_argument("--output", required=True)

    select_parser = subparsers.add_parser(
        "select", help="validate all successful trials and freeze the winner"
    )
    select_parser.add_argument("--search", default="configs/hpo_grid.yml")
    select_parser.add_argument("--study-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    search_path, search = load_search(args.search)
    if args.command == "count":
        print(len(search["trials"]))
        return 0
    if args.command == "materialize":
        study_id = validate_study_id(args.study_id)
        output = project_path(args.output)
        record = materialize_trial(
            search_path=search_path,
            search=search,
            study_id=study_id,
            index=args.trial_index,
            output_path=output,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        selection = select_winner(search_path, search, validate_study_id(args.study_id))
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
