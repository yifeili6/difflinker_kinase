#!/usr/bin/env python3
"""Record terminal HPO outcomes and select a completed finite model."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kinase_gap import hpo

POLICY = {"all_declared_trials_must_be_terminal": True,
          "eligible_outcome": "completed_fixed_budget_or_prescribed_early_stopping",
          "required_objective": "finite_loss/val", "ranking": "minimum_objective_then_lowest_trial_index",
          "terminal_numerical_exception": "src.utils.FoundNaNException", "numerical_failure_eligible": False,
          "numerical_failure_retry": False, "retain_partial_checkpoints_logs_and_observations": True}
JOB_PATTERN = re.compile(r"^[0-9]+_([0-9]+)$")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def fingerprint(path):
    return {"path": hpo.relative(path), "bytes": path.stat().st_size, "sha256": hpo.sha256_file(path)}


def checked_fingerprint(record):
    path = hpo.normalized_project_file(record["path"], label="Recorded outcome input")
    if fingerprint(path) != record:
        raise ValueError(f"Recorded outcome input changed: {record['path']}")
    return path


def study_directory(study):
    hpo.validate_study_id(study)
    return hpo.project_path(f"outputs/hpo/studies/{study}")


def load_protocol(study, search_path, search):
    path = study_directory(study) / "selection_protocol_v2.json"
    protocol = hpo.load_json_mapping(path, label="Selection protocol")
    if (protocol.get("schema_version") != 1 or protocol.get("study_id") != study
            or protocol.get("policy") != POLICY or protocol.get("declared_trial_count") != len(search["trials"])
            or protocol.get("search_config_sha256") != hpo.sha256_file(search_path)
            or protocol.get("search_config") != hpo.relative(search_path)):
        raise ValueError("Selection protocol does not match the frozen search and outcome policy")
    return path, protocol


def freeze_protocol(study, search_path, search):
    path = study_directory(study) / "selection_protocol_v2.json"
    if path.exists():
        load_protocol(study, search_path, search)
        return fingerprint(path)
    write_json_new(path, {"schema_version": 1, "study_id": study, "recorded_utc": now(),
                         "search_config": hpo.relative(search_path), "search_config_sha256": hpo.sha256_file(search_path),
                         "declared_trial_count": len(search["trials"]), "policy": POLICY,
                         "selector": "scripts/select_hpo_outcomes.py"})
    return fingerprint(path)


def accounting(job_ids):
    for job in job_ids:
        if not JOB_PATTERN.fullmatch(job):
            raise ValueError(f"Use an exact Slurm array-task ID: {job}")
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(sorted({job.split('_')[0] for job in job_ids})),
               "--format=JobID,State,ExitCode,NodeList,AllocTRES,Start,End"]
    raw = subprocess.check_output(command, text=True)
    result = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) >= 7 and parts[0] in job_ids:
            result[parts[0]] = dict(zip(("job_id", "state", "exit_code", "node_list", "allocated_resources", "start", "end"), parts[:7]))
            result[parts[0]]["state"] = parts[1].split()[0].rstrip("+")
    return {"recorded_utc": now(), "command": command, "raw": raw, "tasks": result}


def metadata_task(metadata):
    array_job, index = metadata.get("slurm_array_job_id"), metadata.get("slurm_array_task_id")
    if array_job is None or index is None:
        raise ValueError("Run metadata does not identify its Slurm array task")
    return f"{array_job}_{index}"


def checkpoint_state(path):
    import torch
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", {})
    finite = bool(state) and all(not torch.is_tensor(value) or bool(torch.isfinite(value).all()) for value in state.values())
    return {"epoch": payload.get("epoch"), "global_step": payload.get("global_step"), "finite_model_state": finite}


def configuration_binding(study, index, search_path, search):
    run_name, run_id = hpo.trial_identity(study, index)
    config_path = hpo.project_path(f"outputs/hpo/studies/{study}/configs/trial-{index:02d}.yml", must_exist=True)
    config = hpo.load_yaml(config_path)
    if config != hpo.expected_trial_config(search_path, search, study, index):
        raise ValueError(f"Trial {index} config differs from the frozen search")
    metadata_path = hpo.project_path(f"{config['log_dir']}/{run_name}/{run_id}/run_metadata.json", must_exist=True)
    binding = {"config": hpo.relative(config_path), "config_sha256": hpo.sha256_file(config_path),
               "hpo": config["_hpo"], "run_metadata": hpo.relative(metadata_path),
               "run_metadata_sha256": hpo.sha256_file(metadata_path)}
    hpo.validate_materialized_config(result=binding, search_path=search_path, search=search, study_id=study, index=index)
    hpo.validate_run_metadata(result=binding, config_path=config_path, config=config,
                             config_sha256=binding["config_sha256"], run_name=run_name, run_id=run_id, index=index)
    return run_name, run_id, config_path, config, metadata_path


def register_numerical(study, index, job_id, stderr, stdout, search_path, search, *, scheduler=None):
    protocol_path, _ = load_protocol(study, search_path, search)
    if not JOB_PATTERN.fullmatch(job_id) or int(job_id.split("_")[1]) != index:
        raise ValueError("Slurm array-task index differs from the trial index")
    run_name, run_id, config_path, config, metadata_path = configuration_binding(study, index, search_path, search)
    metadata = hpo.load_json_mapping(metadata_path, label="Run metadata")
    if metadata_task(metadata) != job_id:
        raise ValueError("Registered task differs from the current run metadata")
    scheduler = scheduler or accounting([job_id])
    terminal = scheduler["tasks"].get(job_id, {})
    if terminal.get("state") != "FAILED" or terminal.get("exit_code") in (None, "0:0"):
        raise ValueError("Numerical registration requires sacct FAILED and a nonzero task exit code")
    stderr_path = hpo.normalized_project_file(stderr, label="Numerical exception stderr")
    stdout_path = hpo.normalized_project_file(stdout, label="Numerical exception stdout")
    if not stderr_path.name.endswith(f"-{job_id}.err") or not stdout_path.name.endswith(f"-{job_id}.out"):
        raise ValueError("Outcome log filenames must identify the exact Slurm array task")
    error_text = stderr_path.read_text(errors="replace")
    if not re.search(r"(?m)^src\.utils\.FoundNaNException:", error_text):
        raise ValueError("stderr does not contain the terminal src.utils.FoundNaNException traceback")
    if "Traceback (most recent call last):" not in error_text:
        raise ValueError("Numerical exception traceback is incomplete")
    run_dir = hpo.project_path(f"{config['checkpoint_dir']}/{run_name}/{run_id}", must_exist=True)
    if (run_dir / "training_result.json").exists():
        raise ValueError("A completed training result exists for this trial")
    metrics_path = metadata_path.parent / "metrics.csv"
    observations, finite_values = [], []
    if metrics_path.is_file():
        for row in csv.DictReader(metrics_path.open()):
            text = row.get("loss/val", "")
            if text:
                value = float(text)
                observations.append({"epoch": row.get("epoch"), "step": row.get("step"),
                                     "loss_val_text": text, "finite": math.isfinite(value)})
                if math.isfinite(value):
                    finite_values.append(value)
    partial = []
    for path in sorted(run_dir.glob("*.ckpt")):
        partial.append(dict(fingerprint(path), **checkpoint_state(path)))
    record = {"schema_version": 1, "study_id": study, "trial_index": index, "outcome": "numerical_failure",
              "eligible": False, "recorded_utc": now(), "job_id": job_id,
              "exception_type": POLICY["terminal_numerical_exception"], "protocol": fingerprint(protocol_path),
              "config": fingerprint(config_path), "run_metadata": fingerprint(metadata_path),
              "stderr": fingerprint(stderr_path), "stdout": fingerprint(stdout_path), "scheduler": scheduler,
              "metrics": fingerprint(metrics_path) if metrics_path.is_file() else None,
              "validation_observations": observations, "best_partial_validation": min(finite_values) if finite_values else None,
              "partial_checkpoints": partial}
    path = study_directory(study) / "terminal_outcomes" / f"trial-{index:02d}.json"
    write_json_new(path, record)
    return dict(fingerprint(path), outcome="numerical_failure", partial_checkpoints=len(partial))


def completed_record(study, index, job_id, search_path, search):
    run_name, run_id = hpo.trial_identity(study, index)
    run_dir = hpo.project_path(f"models/runs/{run_name}/{run_id}", must_exist=True)
    result_path = run_dir / "training_result.json"
    result = hpo.read_training_result(result_path)
    if result.get("run_name") != run_name or result.get("run_id") != run_id:
        raise ValueError(f"Trial {index} run identity differs from the declared trial")
    config_path, config, config_sha = hpo.validate_materialized_config(
        result=result, search_path=search_path, search=search, study_id=study, index=index)
    metadata_path, metadata_sha = hpo.validate_run_metadata(
        result=result, config_path=config_path, config=config, config_sha256=config_sha,
        run_name=run_name, run_id=run_id, index=index)
    metadata = hpo.load_json_mapping(metadata_path, label="Completed run metadata")
    if metadata_task(metadata) != job_id:
        raise ValueError(f"Trial {index} completed result differs from its actual task ID")
    best_epoch, global_step = hpo.validate_training_budget(result=result, search=search, index=index)
    checkpoint = hpo.normalized_project_file(result["best_model_path"], label="Completed best checkpoint")
    checkpoint.relative_to(run_dir)
    checkpoint_sha = hpo.sha256_file(checkpoint)
    if checkpoint_sha != result["best_model_sha256"]:
        raise ValueError(f"Trial {index} completed checkpoint hash changed")
    state = checkpoint_state(checkpoint)
    if not state["finite_model_state"] or state["epoch"] != best_epoch or state["global_step"] != global_step:
        raise ValueError(f"Trial {index} completed checkpoint state or training counters differ")
    return {"trial_index": index, "outcome": "completed", "eligible": True, "job_id": job_id,
            "run_name": run_name, "run_id": run_id, "parameters": dict(search["trials"][index]),
            "seed": int(search["seed"]), "budget": dict(search["budget"]), "objective": float(result["best_objective"]),
            "best_epoch": best_epoch, "global_step": global_step, "checkpoint": hpo.relative(checkpoint),
            "checkpoint_sha256": checkpoint_sha, "config": hpo.relative(config_path), "config_sha256": config_sha,
            "run_metadata": hpo.relative(metadata_path), "run_metadata_sha256": metadata_sha,
            "training_result": hpo.relative(result_path), "training_result_sha256": hpo.sha256_file(result_path)}


def numerical_record(study, index, job_id, search_path, search):
    path = study_directory(study) / "terminal_outcomes" / f"trial-{index:02d}.json"
    record = hpo.load_json_mapping(path, label="Terminal numerical outcome")
    if (record.get("study_id") != study or record.get("trial_index") != index or record.get("job_id") != job_id
            or record.get("outcome") != "numerical_failure" or record.get("eligible") is not False
            or record.get("exception_type") != POLICY["terminal_numerical_exception"]):
        raise ValueError("Terminal numerical outcome identity differs")
    for field in ("protocol", "config", "run_metadata", "stderr", "stdout"):
        checked_fingerprint(record[field])
    if record.get("metrics"):
        checked_fingerprint(record["metrics"])
    for partial in record["partial_checkpoints"]:
        checked_fingerprint({key: partial[key] for key in ("path", "bytes", "sha256")})
    run_name, run_id, config_path, config, metadata_path = configuration_binding(study, index, search_path, search)
    protocol_path, _ = load_protocol(study, search_path, search)
    if record["config"] != fingerprint(config_path) or record["run_metadata"] != fingerprint(metadata_path) or record["protocol"] != fingerprint(protocol_path):
        raise ValueError("Numerical outcome configuration/metadata/protocol binding changed")
    if metadata_task(hpo.load_json_mapping(metadata_path, label="Numerical run metadata")) != job_id:
        raise ValueError("Current run metadata differs from the terminal numerical outcome")
    if (hpo.project_path(f"{config['checkpoint_dir']}/{run_name}/{run_id}") / "training_result.json").exists():
        raise ValueError("Terminal numerical trial now has a completed result")
    return {"trial_index": index, "outcome": "numerical_failure", "eligible": False, "job_id": job_id,
            "run_name": run_name, "run_id": run_id, "parameters": dict(search["trials"][index]),
            "seed": int(search["seed"]), "budget": dict(search["budget"]), "objective": None,
            "best_partial_validation": record["best_partial_validation"], "exception_type": record["exception_type"],
            "terminal_outcome": fingerprint(path), "config": hpo.relative(config_path),
            "config_sha256": hpo.sha256_file(config_path), "run_metadata": hpo.relative(metadata_path),
            "run_metadata_sha256": hpo.sha256_file(metadata_path), "partial_checkpoints": record["partial_checkpoints"]}


def load_attempt_ledger(path, study, trial_count):
    if path.suffix != ".jsonl":
        ledger = hpo.load_json_mapping(path, label="Hardware attempt ledger")
        if ledger.get("schema_version") != 1 or ledger.get("study_id") != study:
            raise ValueError("Hardware attempt ledger identifies a different study")
        return ledger.get("attempts", [])
    attempts = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        submission = json.loads(line)
        command = submission.get("command", [])
        if not any(str(value).endswith("slurm/hpo.slurm") for value in command):
            continue
        indices = submission.get("trial_indices")
        if indices is None:
            array = next((str(value).split("=", 1)[1] for value in command if str(value).startswith("--array=")), None)
            if array is None:
                indices = list(range(trial_count))
            else:
                indices = []
                for token in array.split("%", 1)[0].split(","):
                    if "-" in token:
                        lower, upper = map(int, token.split("-", 1))
                        indices.extend(range(lower, upper + 1))
                    else:
                        indices.append(int(token))
        for index in indices:
            attempts.append({"trial_index": index, "job_id": f"{submission['job_id']}_{index}",
                             "submission_line": line_number, "submission": submission})
    return attempts


def inspect_outcomes(study, search_path, search, ledger_path, *, active_tasks_path=None, scheduler=None):
    protocol_path, protocol = load_protocol(study, search_path, search)
    attempts = load_attempt_ledger(ledger_path, study, len(search["trials"]))
    latest, seen = {}, set()
    for attempt in attempts:
        index, job_id = attempt["trial_index"], attempt["job_id"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(search["trials"]):
            raise ValueError("Attempt ledger has an invalid trial index")
        if not JOB_PATTERN.fullmatch(job_id) or int(job_id.split("_")[1]) != index or job_id in seen:
            raise ValueError("Attempt ledger must contain distinct exact Slurm array-task IDs")
        seen.add(job_id)
        latest[index] = job_id
    if not attempts:
        raise ValueError("Hardware attempt ledger is empty")
    if active_tasks_path is not None:
        active = hpo.load_json_mapping(active_tasks_path, label="Active HPO tasks")
        if set(active) != {str(index) for index in range(len(search["trials"]))}:
            raise ValueError("Active-task map must identify every declared trial")
        for index, job_id in active.items():
            if latest.get(int(index)) != job_id:
                raise ValueError(f"Active trial {index} differs from the latest ledger attempt")
        latest = {int(index): job_id for index, job_id in active.items()}
    for attempt in attempts:
        if latest.get(attempt["trial_index"]) == attempt["job_id"]:
            continue
        stderr_path = hpo.project_path(attempt.get("stderr", f"logs/kr-hpo-{attempt['job_id']}.err"))
        if stderr_path.is_file() and re.search(r"(?m)^src\.utils\.FoundNaNException:", stderr_path.read_text(errors="replace")):
            raise ValueError(f"Trial {attempt['trial_index']} has an earlier terminal numerical exception and a later attempt")
    scheduler = scheduler or accounting(list(seen))
    records, pending = [], []
    for index in range(len(search["trials"])):
        job_id = latest.get(index)
        task = scheduler["tasks"].get(job_id, {})
        if task.get("state") == "COMPLETED" and task.get("exit_code") == "0:0":
            failure_path = study_directory(study) / "terminal_outcomes" / f"trial-{index:02d}.json"
            if failure_path.exists():
                raise ValueError(f"Trial {index} has a terminal numerical outcome and a later completed task")
            records.append(completed_record(study, index, job_id, search_path, search))
        elif task.get("state") == "FAILED":
            failure_path = study_directory(study) / "terminal_outcomes" / f"trial-{index:02d}.json"
            if not failure_path.is_file():
                raise ValueError(f"Trial {index} terminal task {job_id} needs verified numerical-outcome registration")
            records.append(numerical_record(study, index, job_id, search_path, search))
        elif task.get("state") in {"CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}:
            raise RuntimeError(f"Active trial {index} task {job_id} requires scheduler handling: {task['state']}")
        elif task.get("state") == "COMPLETED":
            raise RuntimeError(f"Active trial {index} task {job_id} has an inconsistent completion exit code")
        else:
            pending.append({"trial_index": index, "job_id": job_id, "state": task.get("state", "not_recorded")})
    return {"protocol": fingerprint(protocol_path), "protocol_recorded_utc": protocol["recorded_utc"],
            "attempt_ledger": fingerprint(ledger_path),
            "active_tasks": fingerprint(active_tasks_path) if active_tasks_path is not None else None,
            "attempts": attempts, "scheduler": scheduler,
            "trials": records, "pending": pending, "ready": not pending}


def rank_eligible(outcomes):
    if not outcomes["ready"]:
        raise RuntimeError("Selection awaits all declared terminal outcomes")
    ranked = sorted((record for record in outcomes["trials"] if record["eligible"]),
                    key=lambda record: (record["objective"], record["trial_index"]))
    if not ranked:
        raise ValueError("No completed finite trial is eligible for selection")
    if any(not math.isfinite(record["objective"]) for record in ranked):
        raise ValueError("Eligible validation objective must be finite")
    return ranked


def publish_selection(study, search_path, search, outcomes):
    ranked = rank_eligible(outcomes)
    study_dir = study_directory(study)
    selection_path = study_dir / "selection.json"
    if selection_path.exists():
        selection = hpo.load_json_mapping(selection_path, label="Existing selection")
        if (selection.get("protocol") != outcomes["protocol"] or selection.get("attempt_ledger") != outcomes["attempt_ledger"]
                or selection.get("active_tasks") != outcomes.get("active_tasks")
                or selection.get("selector_sha256") != hpo.sha256_file(Path(__file__))
                or selection["winner"]["checkpoint_sha256"] != ranked[0]["checkpoint_sha256"]):
            raise ValueError("Existing selection belongs to different verified inputs")
        checkpoint = hpo.project_path(selection["winner"]["selected_checkpoint"], must_exist=True)
        if hpo.sha256_file(checkpoint) != selection["winner"]["selected_checkpoint_sha256"]:
            raise ValueError("Existing selected checkpoint changed")
        return selection
    winner = ranked[0]
    selected_dir = hpo.project_path(f"models/selected/hpo-{study}")
    selected_dir.mkdir(parents=True, exist_ok=False)
    selected = selected_dir / "best.ckpt"
    with hpo.project_path(winner["checkpoint"], must_exist=True).open("rb") as original, selected.open("xb") as copied:
        shutil.copyfileobj(original, copied, length=4 * 1024 * 1024)
    if hpo.sha256_file(selected) != winner["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint copy hash changed")
    hardware_path = study_dir / "hardware_attempts.json"
    write_json_new(hardware_path, {"schema_version": 1, "study_id": study, "attempt_ledger": outcomes["attempt_ledger"],
                                   "active_tasks": outcomes.get("active_tasks"), "attempts": outcomes["attempts"], "scheduler": outcomes["scheduler"]})
    trial_rows = [{"trial_index": row["trial_index"], "outcome": row["outcome"], "eligible": int(row["eligible"]),
                   "job_id": row["job_id"], "lr": row["parameters"]["lr"], "batch_size": row["parameters"]["batch_size"],
                   "objective": row["objective"], "best_partial_validation": row.get("best_partial_validation"),
                   "exception_type": row.get("exception_type")} for row in outcomes["trials"]]
    with (study_dir / "trial_outcomes.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader(); writer.writerows(trial_rows)
    with (study_dir / "ranking.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank", "trial_index", "lr", "batch_size", "objective", "best_epoch", "run_id", "checkpoint", "checkpoint_sha256"))
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow(dict(rank=rank, trial_index=row["trial_index"], lr=row["parameters"]["lr"], batch_size=row["parameters"]["batch_size"],
                                 **{key: row[key] for key in ("objective", "best_epoch", "run_id", "checkpoint", "checkpoint_sha256")}))
    selection = {"schema_version": 1, "status": "complete", "selected_at_utc": now(), "study_id": study,
                 "search_config": hpo.relative(search_path), "search_config_sha256": hpo.sha256_file(search_path),
                 "objective": hpo.OBJECTIVE_NAME, "objective_mode": hpo.OBJECTIVE_MODE, "seed": int(search["seed"]),
                 "budget": dict(search["budget"]), "tie_breaker": "lowest trial_index", "trial_count": len(search["trials"]),
                 "eligible_trial_count": len(ranked), "terminal_numerical_trial_count": len(outcomes["trials"]) - len(ranked),
                 "selector": "scripts/select_hpo_outcomes.py", "selector_sha256": hpo.sha256_file(Path(__file__)),
                 "validator_source_sha256": hpo.sha256_file(Path(hpo.__file__)), "protocol": outcomes["protocol"],
                 "protocol_recorded_utc": outcomes["protocol_recorded_utc"], "attempt_ledger": outcomes["attempt_ledger"],
                 "active_tasks": outcomes.get("active_tasks"),
                 "hardware_attempts": fingerprint(hardware_path), "trial_outcomes": fingerprint(study_dir / "trial_outcomes.csv"),
                 "winner": dict(winner, selected_checkpoint=hpo.relative(selected), selected_checkpoint_sha256=hpo.sha256_file(selected)),
                 "trials": outcomes["trials"], "ranking": [row["trial_index"] for row in ranked]}
    write_json_new(selection_path, selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "register-numerical", "status", "select"))
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--search", default="configs/hpo_grid.yml")
    parser.add_argument("--attempt-ledger")
    parser.add_argument("--active-tasks")
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--job-id")
    parser.add_argument("--stderr")
    parser.add_argument("--stdout")
    args = parser.parse_args()
    search_path, search = hpo.load_search(args.search)
    if args.mode == "freeze":
        print(json.dumps(freeze_protocol(args.study_id, search_path, search), indent=2)); return 0
    if args.mode == "register-numerical":
        if args.trial_index is None or not all((args.job_id, args.stderr, args.stdout)):
            parser.error("register-numerical requires --trial-index --job-id --stderr --stdout")
        print(json.dumps(register_numerical(args.study_id, args.trial_index, args.job_id, args.stderr, args.stdout, search_path, search), indent=2)); return 0
    if not args.attempt_ledger:
        parser.error("status/select requires --attempt-ledger")
    ledger = hpo.normalized_project_file(args.attempt_ledger, label="Hardware attempt ledger")
    active_tasks = hpo.normalized_project_file(args.active_tasks, label="Active HPO tasks") if args.active_tasks else None
    outcomes = inspect_outcomes(args.study_id, search_path, search, ledger, active_tasks_path=active_tasks)
    if args.mode == "status" or not outcomes["ready"]:
        print(json.dumps({"ready": outcomes["ready"], "declared_trial_count": len(search["trials"]),
                          "completed_trial_count": sum(row["outcome"] == "completed" for row in outcomes["trials"]),
                          "numerical_failure_count": sum(row["outcome"] == "numerical_failure" for row in outcomes["trials"]),
                          "terminal_trials": len(outcomes["trials"]), "pending": outcomes["pending"],
                          "pending_job_ids": [row["job_id"] for row in outcomes["pending"]]}, indent=2))
        return 0 if outcomes["ready"] else 3
    selection = publish_selection(args.study_id, search_path, search, outcomes)
    print(json.dumps({"selection": hpo.relative(study_directory(args.study_id) / "selection.json"),
                      "winner_trial_index": selection["winner"]["trial_index"], "eligible_trial_count": selection["eligible_trial_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
