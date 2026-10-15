#!/usr/bin/env python3
"""Fail-closed, resumable evaluation of generated and ground-truth molecules.

The evaluator deliberately operates on one small source shard at a time.  It
validates the immutable generation metadata and hashes before reading a
molecule, preserves every source row (including unreadable SDF records), and
publishes a shard only after both its CSV and completion metadata are durable.

The manuscript filtering order is represented explicitly:

``PoseBusters(mol) -> Lipinski -> MOSES medicinal chemistry + PAINS``.

Descriptors and individual filter results are computed for every sanitizable
molecule, while ``after_posebusters``, ``after_lipinski``, and ``after_2d``
record the sequential funnel.  Expensive optional dependencies are imported
only when a real shard is evaluated; tests can inject small deterministic
callables.

All configured paths must be relative to the ``kinase-gap`` project root.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import numbers
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
POSEBUSTERS_CONFIG = "mol"
POSEBUSTERS_CHECKS = (
    "mol_pred_loaded",
    "sanitization",
    "all_atoms_connected",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
)
GENERATION_FIELDS = (
    "candidate_id",
    "run_id",
    "shard_index",
    "context_index",
    "replicate",
    "source_uuid",
    "fragmentation_id",
    "structure_id",
    "kinase",
    "pdb",
    "anchor_1",
    "anchor_2",
    "fragments_size",
    "linker_size",
    "num_atoms",
    "shard_seed",
    "batch_seed",
    "nan_retries",
    "valid",
    "connected",
    "smiles",
    "in_memory_sanitize_pass",
    "postprocessing_error",
    "anchor_attachment_pass",
    "fragment_linker_bond_count",
    "fragment_linker_edges_json",
    "retained_stereo_pass",
    "retained_stereo_errors_json",
    "retained_nonanchor_explicit_h_match",
    "retained_nonanchor_total_h_match",
    "retained_nonanchor_h_changes_json",
)
GT_SOURCE_FIELDS = (
    "gt_id",
    "row_index",
    "uuid",
    "fragmentation_id",
    "molecule_name",
    "structure_id",
    "kinase",
    "pdb",
    "ligand_code",
    "linker_size",
    "source_smiles",
)
EVALUATION_FIELDS = (
    "evaluation_schema_version",
    "input_sdf_parsed",
    "rdkit_sanitize_pass",
    "canonical_smiles",
    *(f"posebusters_{check}" for check in POSEBUSTERS_CHECKS),
    "posebusters_pass",
    "posebusters_diagnostic_status",
    "posebusters_missing_checks",
    "posebusters_full_report_json",
    "formal_charge",
    "charged_atom_count",
    "neutral_charge_pass",
    "qed",
    "logp",
    "molecular_weight",
    "hbd",
    "hba",
    "rotatable_bond_count",
    "ring_count",
    "hetero_ring_count",
    "aromatic_ring_count",
    "heteroaromatic_ring_count",
    "fused_ring_count",
    "sa_score",
    "lipinski_hbd_pass",
    "lipinski_hba_pass",
    "lipinski_mw_pass",
    "lipinski_logp_pass",
    "lipinski_pass",
    "pains_pass",
    "pains_match_count",
    "pains_matches",
    "moses_filter_pass",
    "after_posebusters",
    "after_lipinski",
    "after_2d",
    "final_pass",
    "evaluation_error",
)
GENERATED_EVALUATION_FIELDS = GENERATION_FIELDS + EVALUATION_FIELDS
GT_EVALUATION_FIELDS = GT_SOURCE_FIELDS + EVALUATION_FIELDS
FUNNEL_FIELDS = (
    "scope",
    "pdb",
    "context_index",
    "linker_size",
    "stage",
    "passed",
    "total",
    "rate",
)
FUNNEL_STAGES = (
    ("generated", None),
    ("sdf_parsed", "input_sdf_parsed"),
    ("rdkit_sanitized", "rdkit_sanitize_pass"),
    ("posebusters", "after_posebusters"),
    ("lipinski", "after_lipinski"),
    ("2d", "after_2d"),
    ("evaluation_error_free", "evaluation_error"),
)


PoseBustersRunner = Callable[[Chem.Mol], Any]
MoleculeBoolRunner = Callable[[Chem.Mol], bool]
SAScorer = Callable[[Chem.Mol], float]
PainsRunner = Callable[[Chem.Mol], Any]


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated, project-root-relative evaluation contract."""

    path: Path
    payload: Dict[str, Any]
    generation_run: Path
    output_root: Path
    run_id: str
    expected_shards: int
    samples_per_shard: int
    expected_candidates: int
    posebusters_config: str
    pains_smarts: Optional[Path]
    pains_catalog: Optional[str]
    gt_table: Optional[Path]
    gt_molecules: Optional[Path]
    gt_expected_candidates: Optional[int]
    gt_samples_per_shard: Optional[int]

    @property
    def run_root(self) -> Path:
        return self.output_root / self.run_id

    @property
    def gt_expected_shards(self) -> Optional[int]:
        if self.gt_expected_candidates is None or self.gt_samples_per_shard is None:
            return None
        return math.ceil(self.gt_expected_candidates / self.gt_samples_per_shard)


@dataclass(frozen=True)
class EvaluationDependencies:
    """Dependency-injected molecule evaluators and their provenance."""

    posebusters: PoseBustersRunner
    moses_filter: MoleculeBoolRunner
    sa_score: SAScorer
    pains: PainsRunner
    versions: Mapping[str, str]


@dataclass(frozen=True)
class PainsResult:
    passed: bool
    match_count: int
    matches: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SDFRecord:
    index: int
    title: str
    properties: Mapping[str, str]
    mol_block: str
    authoritative_molecule: Optional[Chem.Mol] = None


@dataclass(frozen=True)
class GenerationRun:
    run_id: str
    contract_sha256: str
    done: Mapping[str, Any]
    done_sha256: str
    request_sha256: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_id(value: str) -> str:
    if value in {".", ".."} or not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Run ID may contain only letters, numbers, dot, underscore, and hyphen: "
            f"{value!r}"
        )
    return value


def project_path(value: str, *, must_exist: bool = False) -> Path:
    """Resolve a project-relative path and reject absolute or escaping paths."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Path must be project-root-relative: {value}")
    root = PROJECT_ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
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


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON mapping: {relative(path)}")
    return payload


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], Tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return rows, fields


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(fields), lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def count_csv_records(path: Path, expected_fields: Sequence[str]) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ValueError(f"CSV schema changed: {relative(path)}")
        return sum(1 for _ in reader)


def _optional_path(payload: Mapping[str, Any], key: str) -> Optional[Path]:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return project_path(str(value), must_exist=True)


def read_config(config_value: str) -> EvaluationConfig:
    config_path = project_path(config_value, must_exist=True)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Evaluation configuration must be a YAML mapping: {config_value}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation schema_version")

    required = {
        "generation_run",
        "output_root",
        "run_id",
        "expected_shards",
        "samples_per_shard",
        "expected_candidates",
        "posebusters_config",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Evaluation configuration is missing keys: {', '.join(missing)}")

    generation_run = project_path(str(payload["generation_run"]))
    output_root = project_path(str(payload["output_root"]))
    if generation_run.exists() and (not generation_run.is_dir() or generation_run.is_symlink()):
        raise ValueError("generation_run must be a real directory")
    if output_root == generation_run or output_root in generation_run.parents:
        raise ValueError("Evaluation output_root may not contain or equal generation_run")
    if generation_run in output_root.parents:
        raise ValueError("Evaluation output_root may not be inside generation_run")

    run_id = validate_run_id(str(payload["run_id"]))
    expected_shards = int(payload["expected_shards"])
    samples_per_shard = int(payload["samples_per_shard"])
    expected_candidates = int(payload["expected_candidates"])
    if expected_shards <= 0 or samples_per_shard <= 0 or expected_candidates <= 0:
        raise ValueError("Evaluation cardinalities must be positive")
    if expected_shards * samples_per_shard != expected_candidates:
        raise ValueError(
            "expected_candidates must equal expected_shards * samples_per_shard"
        )
    posebusters_config = str(payload["posebusters_config"])
    if posebusters_config != POSEBUSTERS_CONFIG:
        raise ValueError("posebusters_config must remain exactly 'mol'")

    pains_smarts = _optional_path(payload, "pains_smarts")
    pains_catalog_value = payload.get("pains_catalog")
    pains_catalog = (
        str(pains_catalog_value).lower()
        if pains_catalog_value not in (None, "")
        else None
    )
    if pains_smarts is None and pains_catalog is None:
        pains_catalog = "rdkit"
    if (pains_smarts is None) == (pains_catalog is None):
        raise ValueError("Configure exactly one of pains_smarts or pains_catalog")
    if pains_catalog not in (None, "rdkit"):
        raise ValueError("The only supported pains_catalog is 'rdkit'")

    gt_table = _optional_path(payload, "gt_table")
    gt_molecules = _optional_path(payload, "gt_molecules")
    gt_expected_raw = payload.get("gt_expected_candidates")
    gt_shard_raw = payload.get("gt_samples_per_shard")
    gt_values_present = (
        gt_table is not None,
        gt_molecules is not None,
        gt_expected_raw not in (None, ""),
        gt_shard_raw not in (None, ""),
    )
    if any(gt_values_present) and not all(gt_values_present):
        raise ValueError(
            "GT evaluation requires gt_table, gt_molecules, "
            "gt_expected_candidates, and gt_samples_per_shard together"
        )
    gt_expected_candidates = (
        int(gt_expected_raw) if gt_expected_raw not in (None, "") else None
    )
    gt_samples_per_shard = (
        int(gt_shard_raw) if gt_shard_raw not in (None, "") else None
    )
    if gt_expected_candidates is not None and gt_expected_candidates <= 0:
        raise ValueError("gt_expected_candidates must be positive")
    if gt_samples_per_shard is not None and gt_samples_per_shard <= 0:
        raise ValueError("gt_samples_per_shard must be positive")

    return EvaluationConfig(
        path=config_path,
        payload=dict(payload),
        generation_run=generation_run,
        output_root=output_root,
        run_id=run_id,
        expected_shards=expected_shards,
        samples_per_shard=samples_per_shard,
        expected_candidates=expected_candidates,
        posebusters_config=posebusters_config,
        pains_smarts=pains_smarts,
        pains_catalog=pains_catalog,
        gt_table=gt_table,
        gt_molecules=gt_molecules,
        gt_expected_candidates=gt_expected_candidates,
        gt_samples_per_shard=gt_samples_per_shard,
    )


def _validate_output_record(
    *,
    owner: Path,
    outputs: Mapping[str, Any],
    name: str,
    verify_hash: bool,
    expected_records: Optional[int] = None,
    expected_fields: Optional[Sequence[str]] = None,
) -> None:
    path = owner / name
    record = outputs.get(name)
    if not path.is_file() or not isinstance(record, Mapping):
        raise ValueError(f"Declared output is missing: {relative(path)}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"Declared output byte count changed: {relative(path)}")
    if verify_hash and record.get("sha256") != sha256_file(path):
        raise ValueError(f"Declared output SHA-256 changed: {relative(path)}")
    if expected_records is not None:
        if name.endswith(".sdf"):
            observed_records = count_sdf_records(path)
        elif expected_fields is not None:
            observed_records = count_csv_records(path, expected_fields)
        else:
            raise ValueError("CSV record validation requires expected_fields")
        if int(record.get("records", -1)) != observed_records:
            raise ValueError(f"Declared output record count changed: {relative(path)}")
        if observed_records != expected_records:
            raise ValueError(
                f"Output record count changed: {relative(path)} "
                f"({observed_records} != {expected_records})"
            )


def validate_generation_run(
    config: EvaluationConfig, *, verify_combined_hashes: bool, allow_incomplete: bool = False
) -> GenerationRun:
    done_path = config.generation_run / "done.json"
    request_path = config.generation_run / "request.json"
    if not request_path.is_file():
        raise FileNotFoundError("Generation request is not initialized")
    request = load_json(request_path)
    plan = load_json(config.generation_run / "contract.json")
    plan_without_hash = {key: value for key, value in plan.items() if key != "contract_sha256"}
    if sha256_payload(plan_without_hash) != plan.get("contract_sha256"):
        raise ValueError("Frozen generation plan content/hash mismatch")
    for key in ("contract_sha256", "checkpoint", "condition_shards", "expected_candidates"):
        if request.get(key) != plan.get(key):
            raise ValueError(f"Generation request/plan {key} mismatch")
    if request["condition_shards"] != config.expected_shards or request["expected_candidates"] != config.expected_candidates:
        raise ValueError("Generation request differs from evaluation cardinalities")
    run_id = validate_run_id(str(request["run_id"]))
    if not done_path.is_file():
        if not allow_incomplete:
            raise FileNotFoundError("Generation is not fully finalized")
        return GenerationRun(run_id, request["contract_sha256"], {}, "", sha256_file(request_path))
    done = load_json(done_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "condition_shards": config.expected_shards,
        "candidate_count": config.expected_candidates,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise ValueError(
                f"Generation done.json mismatch: {key}={done.get(key)!r}, expected={value!r}"
            )
    run_id = str(done.get("run_id", ""))
    contract_sha256 = str(done.get("contract_sha256", ""))
    if not RUN_ID_PATTERN.fullmatch(run_id) or not re.fullmatch(
        r"[0-9a-f]{64}", contract_sha256
    ):
        raise ValueError("Generation done.json has an invalid run or contract identity")
    if request.get("run_id") != run_id:
        raise ValueError("Generation request/done run IDs differ")
    if request.get("contract_sha256") != contract_sha256:
        raise ValueError("Generation request/done contracts differ")
    if int(request.get("condition_shards", -1)) != config.expected_shards:
        raise ValueError("Generation request shard count differs from evaluation config")
    if int(request.get("expected_candidates", -1)) != config.expected_candidates:
        raise ValueError("Generation request candidate count differs from evaluation config")

    outputs = done.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Generation done.json is missing its outputs mapping")
    for name in ("candidates.sdf", "candidates.csv", "provenance.json"):
        _validate_output_record(
            owner=config.generation_run,
            outputs=outputs,
            name=name,
            verify_hash=verify_combined_hashes,
            expected_records=(
                config.expected_candidates
                if verify_combined_hashes and name != "provenance.json"
                else None
            ),
            expected_fields=GENERATION_FIELDS if name == "candidates.csv" else None,
        )
    return GenerationRun(
        run_id=run_id,
        contract_sha256=contract_sha256,
        done=done,
        done_sha256=sha256_file(done_path),
        request_sha256=sha256_file(request_path),
    )


def read_sdf_records(path: Path) -> List[SDFRecord]:
    """Read exact SD records without sanitizing or dropping malformed molecules."""
    text = path.read_text(encoding="utf-8")
    raw_records: List[str] = []
    current: List[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() == "$$$$":
            if not current:
                raise ValueError(f"Empty SDF record: {relative(path)}")
            raw_records.append("".join(current))
            current = []
        else:
            current.append(line)
    if "".join(current).strip():
        raise ValueError(f"SDF has an unterminated final record: {relative(path)}")

    parsed: List[SDFRecord] = []
    property_header = re.compile(r"^>\s*<([^>]+)>")
    for index, record in enumerate(raw_records):
        lines = record.splitlines()
        if not lines:
            raise ValueError(f"Empty SDF record {index}: {relative(path)}")
        title = lines[0].strip()
        properties: Dict[str, str] = {}
        for line_index, line in enumerate(lines):
            match = property_header.match(line)
            if not match:
                continue
            key = match.group(1)
            if key in properties:
                raise ValueError(
                    f"Duplicate SDF property {key!r} in record {index}: {relative(path)}"
                )
            properties[key] = (
                lines[line_index + 1].strip() if line_index + 1 < len(lines) else ""
            )
        parsed.append(
            SDFRecord(
                index=index,
                title=title,
                properties=properties,
                mol_block=record,
            )
        )
    return parsed


def count_sdf_records(path: Path) -> int:
    """Count complete SD records without loading a potentially large combined file."""
    count = 0
    trailing_non_whitespace = False
    with path.open("rb") as handle:
        for line in handle:
            if line.strip() == b"$$$$":
                count += 1
                trailing_non_whitespace = False
            elif line.strip():
                trailing_non_whitespace = True
    if trailing_non_whitespace:
        raise ValueError(f"SDF has an unterminated final record: {relative(path)}")
    return count


def _require_int(row: Mapping[str, str], field: str, *, context: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer {field!r} in {context}") from exc


def validate_generation_shard(
    config: EvaluationConfig,
    generation: GenerationRun,
    shard_index: int,
) -> Tuple[List[Dict[str, str]], List[SDFRecord], Mapping[str, Any]]:
    if not 0 <= shard_index < config.expected_shards:
        raise ValueError(
            f"shard_index must be in 0..{config.expected_shards - 1}, found {shard_index}"
        )
    shard_dir = config.generation_run / "shards" / f"{shard_index:04d}"
    if not shard_dir.is_dir() or shard_dir.is_symlink():
        raise FileNotFoundError(f"Generation shard is missing: {relative(shard_dir)}")
    done_path = shard_dir / "done.json"
    if not done_path.is_file():
        raise FileNotFoundError(f"Generation shard is incomplete: {relative(shard_dir)}")
    done = load_json(done_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": generation.run_id,
        "contract_sha256": generation.contract_sha256,
        "shard_index": shard_index,
        "sample_count": config.samples_per_shard,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise ValueError(
                f"Generation shard metadata mismatch ({key}): {relative(done_path)}"
            )
    outputs = done.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Generation shard has no outputs mapping: {relative(done_path)}")
    _validate_output_record(
        owner=shard_dir,
        outputs=outputs,
        name="samples.csv",
        verify_hash=True,
        expected_records=config.samples_per_shard,
        expected_fields=GENERATION_FIELDS,
    )
    _validate_output_record(
        owner=shard_dir,
        outputs=outputs,
        name="molecules.sdf",
        verify_hash=True,
        expected_records=config.samples_per_shard,
    )
    rows, fields = read_csv(shard_dir / "samples.csv")
    if fields != GENERATION_FIELDS:
        raise ValueError(f"Generation shard CSV schema changed: {relative(shard_dir)}")
    records = read_sdf_records(shard_dir / "molecules.sdf")
    if len(rows) != config.samples_per_shard or len(records) != len(rows):
        raise ValueError(f"Generation shard SDF/CSV cardinality mismatch: {relative(shard_dir)}")

    seen = set()
    shard_metadata = {
        "context_index": done.get("context_index"),
        "source_uuid": done.get("source_uuid"),
        "fragmentation_id": done.get("fragmentation_id"),
        "structure_id": done.get("structure_id"),
        "kinase": done.get("kinase"),
        "pdb": done.get("pdb"),
        "linker_size": done.get("linker_size"),
        "shard_seed": done.get("shard_seed"),
    }
    for replicate, (row, record) in enumerate(zip(rows, records)):
        context = f"generation shard {shard_index}, row {replicate}"
        if _require_int(row, "shard_index", context=context) != shard_index:
            raise ValueError(f"CSV shard index mismatch in {context}")
        if _require_int(row, "replicate", context=context) != replicate:
            raise ValueError(f"CSV replicate order is not contiguous in {context}")
        if row["run_id"] != generation.run_id:
            raise ValueError(f"CSV run ID mismatch in {context}")
        expected_id = f"{generation.run_id}-{shard_index:04d}-{replicate:03d}"
        if row["candidate_id"] != expected_id or expected_id in seen:
            raise ValueError(f"Candidate identity mismatch or duplicate in {context}")
        seen.add(expected_id)
        if record.title != expected_id:
            raise ValueError(f"SDF title/CSV candidate mismatch in {context}")
        if record.properties.get("candidate_id") != expected_id:
            raise ValueError(f"SDF candidate_id property mismatch in {context}")
        for field, expected_value in shard_metadata.items():
            if expected_value is None:
                raise ValueError(f"Generation shard done.json omits {field}")
            if str(row[field]).lower() != str(expected_value).lower():
                raise ValueError(f"CSV/done {field} mismatch in {context}")
            if field in record.properties and str(record.properties[field]).lower() != str(
                row[field]
            ).lower():
                raise ValueError(f"SDF/CSV {field} mismatch in {context}")
    coordinates_path = shard_dir / "coordinates.npz"
    _validate_output_record(owner=shard_dir, outputs=outputs, name="coordinates.npz", verify_hash=True)
    verified_records = []
    with np.load(coordinates_path, allow_pickle=False) as saved:
        if len(saved["candidate_ids"]) != len(rows) or saved["coordinates"].shape[0] != len(rows):
            raise ValueError("Full-precision coordinate count differs from generation attempts")
        for index, (row, record) in enumerate(zip(rows, records)):
            xyz = saved["coordinates"][index]
            symbols = list(saved["symbols"][index])
            if str(saved["candidate_ids"][index]) != row["candidate_id"] or xyz.shape != (int(row["num_atoms"]), 3) or len(symbols) != int(row["num_atoms"]) or not np.isfinite(xyz).all():
                raise ValueError("Full-precision coordinates do not match the candidate record")
            authoritative = None
            if str(row["in_memory_sanitize_pass"]) == "1":
                authoritative = Chem.MolFromMolBlock(record.mol_block, sanitize=True, removeHs=False)
                if authoritative is None or [atom.GetSymbol() for atom in authoritative.GetAtoms()] != symbols:
                    raise ValueError("Valid candidate SDF element/order mismatch")
                from .chemistry_io import restore_native_identity
                from .fragmentation import canonical_isomeric_identity
                for key, value in record.properties.items():
                    if key.startswith("kinase_"):
                        authoritative.SetProp(key, value)
                authoritative = restore_native_identity(authoritative)
                if canonical_isomeric_identity(authoritative) != canonical_isomeric_identity(Chem.MolFromSmiles(row["smiles"])):
                    raise ValueError("Valid candidate SDF identity differs from in-memory molecule")
                if not np.allclose(authoritative.GetConformer().GetPositions(), xyz, atol=0.000051, rtol=0):
                    raise ValueError("Candidate SDF coordinates exceed permitted rounding")
                for atom_index, point in enumerate(xyz):
                    authoritative.GetConformer().SetAtomPosition(atom_index, point)
            verified_records.append(SDFRecord(record.index, record.title, record.properties, record.mol_block, authoritative))
    return rows, verified_records, done


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _load_moses_functions() -> Tuple[MoleculeBoolRunner, SAScorer]:
    modules = [
        importlib.import_module("moses.metrics"),
        importlib.import_module("moses.metrics.metrics"),
        importlib.import_module("moses.metrics.utils"),
    ]
    filter_function = next(
        (
            getattr(module, "mol_passes_filters")
            for module in modules
            if callable(getattr(module, "mol_passes_filters", None))
        ),
        None,
    )
    sa_function = next(
        (
            getattr(module, "SA")
            for module in modules
            if callable(getattr(module, "SA", None))
        ),
        None,
    )
    if filter_function is None or sa_function is None:
        raise ImportError(
            "The installed MOSES/molsets package does not expose "
            "mol_passes_filters and SA"
        )
    return filter_function, sa_function


def load_pains_smarts(path: Path) -> Tuple[Tuple[str, str, Chem.Mol], ...]:
    patterns: List[Tuple[str, str, Chem.Mol]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            smarts = row[0].strip()
            if line_number == 1 and smarts.lower() in {"smarts", "pattern"}:
                continue
            name = row[1].strip() if len(row) > 1 and row[1].strip() else f"line_{line_number}"
            pattern = Chem.MolFromSmarts(smarts, mergeHs=True)
            if pattern is None:
                raise ValueError(
                    f"Invalid PAINS SMARTS at {relative(path)}:{line_number}"
                )
            patterns.append((smarts, name, pattern))
    if not patterns:
        raise ValueError(f"PAINS SMARTS file contains no patterns: {relative(path)}")
    return tuple(patterns)


def build_pains_runner(config: EvaluationConfig) -> PainsRunner:
    if config.pains_smarts is not None:
        patterns = load_pains_smarts(config.pains_smarts)

        def check_smarts(molecule: Chem.Mol) -> PainsResult:
            matches = tuple(
                name for _smarts, name, pattern in patterns if molecule.HasSubstructMatch(pattern)
            )
            return PainsResult(
                passed=not matches, match_count=len(matches), matches=matches
            )

        return check_smarts

    if config.pains_catalog == "rdkit":
        filter_module = importlib.import_module("rdkit.Chem.FilterCatalog")
        params = filter_module.FilterCatalogParams()
        params.AddCatalog(filter_module.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = filter_module.FilterCatalog(params)

        def check_catalog(molecule: Chem.Mol) -> PainsResult:
            entries = catalog.GetMatches(molecule)
            names = tuple(sorted(str(entry.GetDescription()) for entry in entries))
            return PainsResult(passed=not names, match_count=len(names), matches=names)

        return check_catalog
    raise AssertionError("PAINS configuration was not validated")


def load_dependencies(config: EvaluationConfig) -> EvaluationDependencies:
    """Lazily instantiate PoseBusters, MOSES, SA, and PAINS evaluators."""
    # PoseBusters 0.2.10 imports this newer RDKit symbol even though the
    # molecule-only configuration never calls protein/ligand alignment.  The
    # frozen RDKit 2022.03.5 stack lacks it.  Supply a fail-closed in-memory
    # symbol so the module can import, without changing the installed package.
    # If a future code path unexpectedly attempts alignment, it stops loudly.
    ensure_posebusters_alignment_import_compatibility()

    posebusters_module = importlib.import_module("posebusters")
    buster_class = getattr(posebusters_module, "PoseBusters", None)
    if buster_class is None:
        raise ImportError("The installed posebusters package has no PoseBusters class")
    buster = buster_class(config=config.posebusters_config)
    threads = int(os.environ.get("KINASE_EVALUATION_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if threads < 1:
        raise ValueError("PoseBusters thread count must be positive")
    energy_module = importlib.import_module("posebusters.modules.energy_ratio")
    original = getattr(energy_module, "_kinase_uncapped_get_average_energy", energy_module.get_average_energy)
    energy_module._kinase_uncapped_get_average_energy = original

    def capped_average_energy(inchi, n_confs=50, num_threads=0):
        return original(inchi, n_confs=n_confs, num_threads=threads)

    energy_module.get_average_energy = capped_average_energy

    def run_posebusters(molecule: Chem.Mol) -> Any:
        return buster.bust(mol_pred=molecule, full_report=True)

    moses_filter, sa_score = _load_moses_functions()
    return EvaluationDependencies(
        posebusters=run_posebusters,
        moses_filter=moses_filter,
        sa_score=sa_score,
        pains=build_pains_runner(config),
        versions={
            "python": sys.version,
            "rdkit": getattr(__import__("rdkit"), "__version__", "unknown"),
            "posebusters": _distribution_version("posebusters"),
            "molsets": _distribution_version("molsets", "moses"),
            "posebusters_energy_threads": str(threads),
            "posebusters_full_report": "true",
        },
    )


def ensure_posebusters_alignment_import_compatibility() -> bool:
    """Install an in-memory, fail-closed symbol needed by PoseBusters 0.2.10.

    Returns ``True`` only when the compatibility symbol was added.  No installed
    package or source file is modified.
    """
    from rdkit.Chem import rdMolAlign

    if hasattr(rdMolAlign, "GetBestAlignmentTransform"):
        return False

    def unsupported_alignment(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "GetBestAlignmentTransform is unavailable in frozen RDKit; "
            "PoseBusters config='mol' must not invoke alignment"
        )

    rdMolAlign.GetBestAlignmentTransform = unsupported_alignment
    return True


def normalize_binary(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return int(value)
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, bool):
            return int(scalar)
        if isinstance(scalar, numbers.Integral) and int(scalar) in (0, 1):
            return int(scalar)
    raise TypeError(f"{label} must be a boolean or binary integer, found {value!r}")


def normalize_posebusters_binary(value: Any, *, label: str) -> int:
    """Normalize a PoseBusters check, treating unavailable checks as failed.

    PoseBusters 0.2.10 emits NaN for downstream geometry/energy checks when an
    earlier sanitization check fails.  Such a molecule does not pass the
    molecule-only suite, but the missing value is expected data rather than an
    evaluator exception.
    """
    if value is None or type(value).__name__ == "NAType":
        return 0
    if isinstance(value, numbers.Real):
        try:
            if math.isnan(float(value)):
                return 0
        except (TypeError, ValueError):
            pass
    return normalize_binary(value, label=label)


def normalize_posebusters_report(report: Any) -> Dict[str, int]:
    if isinstance(report, Mapping):
        row = dict(report)
    elif hasattr(report, "iloc") and hasattr(report, "__len__"):
        if len(report) != 1:
            raise ValueError(f"PoseBusters returned {len(report)} rows for one molecule")
        converted = report.iloc[0].to_dict()
        if not isinstance(converted, dict):
            raise TypeError("PoseBusters row did not convert to a mapping")
        row = converted
    else:
        raise TypeError("PoseBusters result must be a mapping or one-row table")
    missing = [check for check in POSEBUSTERS_CHECKS if check not in row]
    if missing:
        raise ValueError(f"PoseBusters mol report omits checks: {', '.join(missing)}")
    return {
        check: normalize_posebusters_binary(
            row[check], label=f"PoseBusters {check}"
        )
        for check in POSEBUSTERS_CHECKS
    }


def normalize_pains_result(result: Any) -> PainsResult:
    if isinstance(result, PainsResult):
        normalized = result
    elif isinstance(result, bool):
        normalized = PainsResult(
            passed=result, match_count=0 if result else 1, matches=()
        )
    elif isinstance(result, Mapping):
        passed = bool(normalize_binary(result.get("passed"), label="PAINS passed"))
        matches_raw = result.get("matches", ())
        if isinstance(matches_raw, str):
            matches = (matches_raw,) if matches_raw else ()
        elif isinstance(matches_raw, Sequence):
            matches = tuple(str(value) for value in matches_raw)
        else:
            raise TypeError("PAINS matches must be a string or sequence")
        count = int(result.get("match_count", len(matches)))
        normalized = PainsResult(passed=passed, match_count=count, matches=matches)
    else:
        raise TypeError("PAINS evaluator must return PainsResult, bool, or mapping")
    if normalized.match_count < 0:
        raise ValueError("PAINS match_count must be non-negative")
    if normalized.passed != (normalized.match_count == 0):
        raise ValueError("PAINS pass flag and match_count are inconsistent")
    if normalized.matches and len(normalized.matches) != normalized.match_count:
        raise ValueError("PAINS match names and match_count are inconsistent")
    return normalized


def ring_descriptors(molecule: Chem.Mol) -> Dict[str, int]:
    rings = [set(map(int, ring)) for ring in Chem.GetSymmSSSR(molecule)]
    aromatic = [
        ring
        for ring in rings
        if ring and all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    ]
    heteroaromatic = [
        ring
        for ring in aromatic
        if any(molecule.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in ring)
    ]
    hetero = [
        ring
        for ring in rings
        if any(molecule.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in ring)
    ]
    fused_neighbors = {
        index: {
            other_index
            for other_index, other in enumerate(rings)
            if other_index != index and len(ring.intersection(other)) >= 2
        }
        for index, ring in enumerate(rings)
    }
    fused_system_count = 0
    visited = set()
    for start, neighbors in fused_neighbors.items():
        if start in visited or not neighbors:
            continue
        fused_system_count += 1
        pending = [start]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(fused_neighbors[current] - visited)
    return {
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "hetero_ring_count": len(hetero),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "heteroaromatic_ring_count": len(heteroaromatic),
        # Count connected fused ring systems, where rings are adjacent when
        # they share a bond (at least two atoms).
        "fused_ring_count": fused_system_count,
    }


def molecular_descriptors(molecule: Chem.Mol) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "qed": float(QED.qed(molecule)),
        "logp": float(Descriptors.MolLogP(molecule)),
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "charged_atom_count": sum(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()),
        "neutral_charge_pass": int(all(atom.GetFormalCharge() == 0 for atom in molecule.GetAtoms())),
        "hbd": int(Lipinski.NumHDonors(molecule)),
        "hba": int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bond_count": int(
            rdMolDescriptors.CalcNumRotatableBonds(molecule, strict=True)
        ),
    }
    result.update(ring_descriptors(molecule))
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Descriptor {key} is non-finite")
    return result


def lipinski_flags(descriptors: Mapping[str, Any]) -> Dict[str, int]:
    """Return the exact four manuscript Lipinski comparisons."""
    component = {
        "lipinski_hbd_pass": int(int(descriptors["hbd"]) <= 5),
        "lipinski_hba_pass": int(int(descriptors["hba"]) <= 10),
        "lipinski_mw_pass": int(float(descriptors["molecular_weight"]) < 500.0),
        "lipinski_logp_pass": int(float(descriptors["logp"]) < 5.0),
    }
    component["lipinski_pass"] = int(all(component.values()))
    return component


def empty_evaluation_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {field: "" for field in EVALUATION_FIELDS}
    row["evaluation_schema_version"] = SCHEMA_VERSION
    for field in (
        "input_sdf_parsed",
        "rdkit_sanitize_pass",
        *(f"posebusters_{check}" for check in POSEBUSTERS_CHECKS),
        "posebusters_pass",
        "after_posebusters",
        "after_lipinski",
        "after_2d",
        "final_pass",
    ):
        row[field] = 0
    return row


def _error_text(stage: str, error: BaseException) -> str:
    message = " ".join(str(error).split())
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{stage}:{type(error).__name__}:{message}"


def parse_sdf_molecule(record: SDFRecord) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    if record.authoritative_molecule is not None:
        return Chem.Mol(record.authoritative_molecule), None
    try:
        molecule = Chem.MolFromMolBlock(
            record.mol_block,
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )
    except Exception as exc:
        return None, _error_text("sdf_parse", exc)
    if molecule is None:
        return None, "sdf_parse:ParseFailure:RDKit returned no molecule"
    return molecule, None


def _full_report_json(report):
    if hasattr(report, "to_json"):
        return report.to_json(orient="records")
    def clean(value):
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
    return json.dumps(clean(report), sort_keys=True, allow_nan=False)


def evaluate_record(
    source_row: Mapping[str, Any],
    record: SDFRecord,
    dependencies: EvaluationDependencies,
    *,
    stage_attempts: Optional[Dict[str, int]] = None,
    stage_errors: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Evaluate one aligned source row, retaining all failures explicitly."""
    output = dict(source_row)
    output.update(empty_evaluation_row())
    errors: List[str] = []
    attempts = stage_attempts if stage_attempts is not None else {}
    failures = stage_errors if stage_errors is not None else {}

    def attempt(stage: str) -> None:
        attempts[stage] = attempts.get(stage, 0) + 1

    def failed(stage: str, error: BaseException) -> None:
        failures[stage] = failures.get(stage, 0) + 1
        errors.append(_error_text(stage, error))

    raw_molecule, parse_error = parse_sdf_molecule(record)
    if raw_molecule is None:
        errors.append(parse_error or "sdf_parse:ParseFailure:unknown")
        output["evaluation_error"] = "|".join(errors)
        return output
    output["input_sdf_parsed"] = 1
    if str(source_row.get("in_memory_sanitize_pass", "")) == "0":
        output["posebusters_diagnostic_status"] = "known_invalid_before_serialization"
        output["posebusters_full_report_json"] = "{}"
        output["evaluation_error"] = str(source_row.get("postprocessing_error") or "generation:invalid_in_memory")
        return output

    attempt("posebusters")
    try:
        raw_report = dependencies.posebusters(Chem.Mol(raw_molecule))
        output["posebusters_full_report_json"] = _full_report_json(raw_report)
        raw_values = raw_report.iloc[0].to_dict() if hasattr(raw_report, "iloc") else dict(raw_report)
        missing_checks = [check for check in POSEBUSTERS_CHECKS
                          if check not in raw_values or str(raw_values[check]).lower() in ("nan", "none", "<na>")]
        output["posebusters_missing_checks"] = ";".join(missing_checks)
        output["posebusters_diagnostic_status"] = "incomplete_outputs" if missing_checks else "completed"
        report = normalize_posebusters_report(raw_report)
    except Exception as exc:
        output["posebusters_diagnostic_status"] = "exception"
        failed("posebusters", exc)
    else:
        for check, value in report.items():
            output[f"posebusters_{check}"] = value
        output["posebusters_pass"] = int(all(report.values()))
    output["after_posebusters"] = output["posebusters_pass"]

    sanitized = Chem.Mol(raw_molecule)
    attempt("rdkit_sanitize")
    try:
        Chem.SanitizeMol(sanitized)
        from .fragmentation import canonical_isomeric_identity
        canonical_smiles = canonical_isomeric_identity(sanitized)
        if not canonical_smiles:
            raise ValueError("canonical SMILES is empty")
    except Exception as exc:
        failed("rdkit_sanitize", exc)
        output["evaluation_error"] = "|".join(errors)
        return output
    output["rdkit_sanitize_pass"] = 1
    output["canonical_smiles"] = canonical_smiles

    attempt("descriptors")
    try:
        descriptors = molecular_descriptors(sanitized)
    except Exception as exc:
        failed("descriptors", exc)
        descriptors = None
    if descriptors is not None:
        output.update(descriptors)
        output.update(lipinski_flags(descriptors))

    attempt("sa_score")
    try:
        sa_score = float(dependencies.sa_score(Chem.Mol(sanitized)))
        if not math.isfinite(sa_score):
            raise ValueError("SA score is non-finite")
    except Exception as exc:
        failed("sa_score", exc)
    else:
        output["sa_score"] = sa_score

    attempt("pains")
    try:
        pains = normalize_pains_result(dependencies.pains(Chem.Mol(sanitized)))
    except Exception as exc:
        failed("pains", exc)
    else:
        output["pains_pass"] = int(pains.passed)
        output["pains_match_count"] = pains.match_count
        output["pains_matches"] = ";".join(sorted(pains.matches))

    attempt("moses_filter")
    try:
        moses_pass = normalize_binary(
            dependencies.moses_filter(Chem.Mol(sanitized)),
            label="MOSES medicinal chemistry filter",
        )
    except Exception as exc:
        failed("moses_filter", exc)
    else:
        output["moses_filter_pass"] = moses_pass

    lipinski_pass = output["lipinski_pass"] == 1
    output["after_lipinski"] = int(
        output["after_posebusters"] == 1 and lipinski_pass
    )
    output["after_2d"] = int(
        output["after_lipinski"] == 1
        and output["pains_pass"] == 1
        and output["moses_filter_pass"] == 1
    )
    output["final_pass"] = output["after_2d"]
    output["evaluation_error"] = "|".join(errors)
    return output


def _fail_on_systemic_stage_errors(
    attempts: Mapping[str, int], errors: Mapping[str, int]
) -> None:
    dependency_stages = {
        "posebusters",
        "descriptors",
        "sa_score",
        "pains",
        "moses_filter",
    }
    systemic = [
        stage
        for stage, count in attempts.items()
        if stage in dependency_stages
        and count > 0
        and errors.get(stage, 0) == count
    ]
    if systemic:
        raise RuntimeError(
            "Every attempted molecule failed in evaluation stage(s): "
            + ", ".join(sorted(systemic))
        )


def evaluation_contract(
    config: EvaluationConfig, generation: GenerationRun
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": relative(config.path),
        "config_sha256": sha256_file(config.path),
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "chemistry_source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                    for name in ("fragmentation.py", "chemistry_io.py")},
        "generation_run": relative(config.generation_run),
        "generation_run_id": generation.run_id,
        "generation_contract_sha256": generation.contract_sha256,
        "generation_request_sha256": generation.request_sha256,
        "expected_shards": config.expected_shards,
        "samples_per_shard": config.samples_per_shard,
        "expected_candidates": config.expected_candidates,
        "posebusters_config": config.posebusters_config,
        "posebusters_checks": list(POSEBUSTERS_CHECKS),
        "posebusters_full_reports": True,
        "posebusters_energy": {"seed": 42, "conformers": 50, "ratio_threshold": 100,
                               "thread_policy": "KINASE_EVALUATION_THREADS or allocated SLURM CPUs, otherwise 1"},
        "charge": {"record_total_formal_charge": True, "record_charged_atom_count": True,
                   "neutral_charge_pass": "Every atom has formal charge zero; also applied within MOSES filter"},
        "lipinski": {
            "hbd": "<=5",
            "hba": "<=10",
            "molecular_weight": "<500",
            "logp": "<5",
        },
        "pains": {
            "source": (
                relative(config.pains_smarts)
                if config.pains_smarts is not None
                else config.pains_catalog
            ),
            "sha256": (
                sha256_file(config.pains_smarts)
                if config.pains_smarts is not None
                else None
            ),
        },
    }
    if config.gt_table is not None and config.gt_molecules is not None:
        payload["ground_truth"] = gt_input_contract(config)
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def gt_chemistry_path(config):
    if config.gt_molecules is None:
        raise ValueError("Ground truth molecules are not configured")
    return config.gt_molecules.with_name(config.gt_molecules.name.replace("_mol.sdf", "_chemistry.jsonl"))


def gt_input_contract(config):
    sidecar = gt_chemistry_path(config)
    return {"table": relative(config.gt_table), "table_sha256": sha256_file(config.gt_table),
            "molecules": relative(config.gt_molecules), "molecules_sha256": sha256_file(config.gt_molecules),
            "chemistry": relative(sidecar), "chemistry_sha256": sha256_file(sidecar),
            "molecule_authority": "native parent RDKit binary with original stereochemistry and full precision coordinates",
            "expected_candidates": config.gt_expected_candidates,
            "samples_per_shard": config.gt_samples_per_shard, "expected_shards": config.gt_expected_shards}


def gt_evaluation_contract(config):
    payload = {"schema_version": SCHEMA_VERSION, "kind": "ground_truth_evaluation",
               "run_id": config.run_id, "config": relative(config.path),
               "config_sha256": sha256_file(config.path),
               "evaluation_source_sha256": sha256_file(Path(__file__)),
               "chemistry_source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                           for name in ("fragmentation.py", "chemistry_io.py")},
               "ground_truth": gt_input_contract(config),
               "posebusters_config": config.posebusters_config,
               "posebusters_checks": list(POSEBUSTERS_CHECKS), "posebusters_full_reports": True,
               "posebusters_energy": {"seed": 42, "conformers": 50, "ratio_threshold": 100},
               "lipinski": {"hbd": "<=5", "hba": "<=10", "molecular_weight": "<500", "logp": "<5"},
               "pains_sha256": sha256_file(config.pains_smarts) if config.pains_smarts else None}
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def ensure_gt_run_contract(config, contract):
    config.run_root.mkdir(parents=True, exist_ok=True)
    path = config.run_root / "gt_request.json"
    with (config.run_root / ".gt_request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if load_json(path) != contract:
                raise ValueError("Ground-truth evaluation contract changed")
        else:
            atomic_write_json(path, contract)


def ensure_run_contract(
    config: EvaluationConfig,
    generation: GenerationRun,
    contract: Mapping[str, Any],
) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_root = config.run_root
    if run_root.exists() and (not run_root.is_dir() or run_root.is_symlink()):
        raise ValueError(f"Evaluation run path is not a real directory: {relative(run_root)}")
    run_root.mkdir(parents=True, exist_ok=True)
    request_path = run_root / "request.json"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.run_id,
        "contract_sha256": contract["contract_sha256"],
        "generation_run_id": generation.run_id,
        "generation_contract_sha256": generation.contract_sha256,
        "expected_shards": config.expected_shards,
        "expected_candidates": config.expected_candidates,
    }

    def validate_existing_request() -> None:
        # Hundreds of array tasks can enter here together.  The task that wins
        # O_EXCL creates the directory entry before it finishes and fsyncs the
        # JSON payload, so losers must tolerate that short publication window.
        observed = None
        last_error: Optional[BaseException] = None
        for _attempt in range(100):
            try:
                observed = load_json(request_path)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.05)
                continue
            break
        if observed is None:
            raise RuntimeError(
                f"Timed out waiting for evaluation request publication: "
                f"{relative(request_path)}"
            ) from last_error
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(
                    f"Evaluation run ID is bound to a different contract ({key} mismatch)"
                )

    if request_path.exists():
        validate_existing_request()
        return

    request = {**expected, "created_at_utc": utc_now()}
    text = json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{request_path.name}.tmp.", dir=str(run_root)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Linking a fully fsynced temporary file publishes an immutable
            # first-writer-wins request without exposing a zero-byte JSON file.
            os.link(temporary, request_path)
        except FileExistsError:
            validate_existing_request()
        else:
            fsync_directory(run_root)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_shard_directory(run_root: Path, shard_index: int) -> Path:
    return run_root / "shards" / f"{shard_index:04d}"


def validate_completed_evaluation_shard(
    config: EvaluationConfig,
    contract: Mapping[str, Any],
    shard_index: int,
) -> Optional[Mapping[str, Any]]:
    shard_dir = evaluation_shard_directory(config.run_root, shard_index)
    if not shard_dir.exists():
        return None
    if not shard_dir.is_dir() or shard_dir.is_symlink():
        raise ValueError(f"Evaluation shard path is not a real directory: {relative(shard_dir)}")
    done_path = shard_dir / "done.json"
    if not done_path.is_file():
        raise FileExistsError(
            f"Refusing to replace incomplete evaluation shard: {relative(shard_dir)}"
        )
    done = load_json(done_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": config.run_id,
        "contract_sha256": contract["contract_sha256"],
        "shard_index": shard_index,
        "record_count": config.samples_per_shard,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise ValueError(
                f"Completed evaluation shard metadata mismatch ({key}): "
                f"{relative(done_path)}"
            )
    outputs = done.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Evaluation shard has no outputs mapping: {relative(done_path)}")
    _validate_output_record(
        owner=shard_dir,
        outputs=outputs,
        name="evaluations.csv",
        verify_hash=True,
        expected_records=config.samples_per_shard,
        expected_fields=GENERATED_EVALUATION_FIELDS,
    )
    return done


def evaluate_generation_shard(
    config: EvaluationConfig,
    generation: GenerationRun,
    contract: Mapping[str, Any],
    shard_index: int,
    dependencies: EvaluationDependencies,
) -> Mapping[str, Any]:
    completed = validate_completed_evaluation_shard(config, contract, shard_index)
    if completed is not None:
        print(f"evaluation shard={shard_index:04d} status=skipped", flush=True)
        return completed

    source_rows, records, generation_done = validate_generation_shard(
        config, generation, shard_index
    )
    attempts: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    evaluated = [
        evaluate_record(
            source_row,
            record,
            dependencies,
            stage_attempts=attempts,
            stage_errors=errors,
        )
        for source_row, record in zip(source_rows, records)
    ]
    _fail_on_systemic_stage_errors(attempts, errors)
    if len(evaluated) != config.samples_per_shard:
        raise AssertionError("Evaluation dropped a source candidate")

    shards_root = config.run_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    final_dir = evaluation_shard_directory(config.run_root, shard_index)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{shard_index:04d}.tmp.", dir=str(shards_root))
    )
    try:
        csv_path = temporary / "evaluations.csv"
        written = write_csv(csv_path, GENERATED_EVALUATION_FIELDS, evaluated)
        if written != config.samples_per_shard:
            raise RuntimeError("Evaluation CSV serialization count mismatch")
        output_record = {
            "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "records": written,
        }
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "run_id": config.run_id,
            "contract_sha256": contract["contract_sha256"],
            "shard_index": shard_index,
            "record_count": written,
            "source_generation_shard": {
                "path": relative(
                    config.generation_run / "shards" / f"{shard_index:04d}"
                ),
                "done_sha256": sha256_file(
                    config.generation_run / "shards" / f"{shard_index:04d}" / "done.json"
                ),
                "samples_csv_sha256": generation_done["outputs"]["samples.csv"][
                    "sha256"
                ],
                "molecules_sdf_sha256": generation_done["outputs"]["molecules.sdf"][
                    "sha256"
                ],
            },
            "counts": evaluation_counts(evaluated),
            "stage_attempts": attempts,
            "stage_errors": errors,
            "software": dict(dependencies.versions),
            "runtime": {"host": socket.getfqdn(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
                        "evaluation_threads": os.environ.get("KINASE_EVALUATION_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))},
            "outputs": {"evaluations.csv": output_record},
        }
        atomic_write_json(temporary / "done.json", done)
        fsync_directory(temporary)
        os.replace(temporary, final_dir)
        fsync_directory(shards_root)
        print(
            f"evaluation shard={shard_index:04d} status=complete "
            f"records={written} final={done['counts']['after_2d']}",
            flush=True,
        )
        return done
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def evaluation_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(rows),
        "sdf_parsed": sum(int(row["input_sdf_parsed"]) for row in rows),
        "rdkit_sanitized": sum(int(row["rdkit_sanitize_pass"]) for row in rows),
        "posebusters": sum(int(row["after_posebusters"]) for row in rows),
        "lipinski": sum(int(row["after_lipinski"]) for row in rows),
        "after_2d": sum(int(row["after_2d"]) for row in rows),
        "final_pass": sum(int(row["final_pass"]) for row in rows),
        "evaluation_error": sum(bool(row["evaluation_error"]) for row in rows),
    }


def _funnel_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scopes: List[Tuple[str, str, str, str, Sequence[Mapping[str, Any]]]] = [
        ("overall", "all", "all", "all", rows)
    ]
    linker_sizes = sorted(
        {str(row["linker_size"]) for row in rows},
        key=lambda value: int(value),
    )
    scopes.extend(
        (
            "linker_size",
            "all",
            "all",
            linker_size,
            [row for row in rows if str(row["linker_size"]) == linker_size],
        )
        for linker_size in linker_sizes
    )
    pdbs = sorted({str(row["pdb"]).lower() for row in rows})
    scopes.extend(
        (
            "pdb",
            pdb,
            "all",
            "all",
            [row for row in rows if str(row["pdb"]).lower() == pdb],
        )
        for pdb in pdbs
    )
    scopes.extend(
        (
            "pdb_linker",
            pdb,
            "all",
            linker_size,
            [
                row
                for row in rows
                if str(row["pdb"]).lower() == pdb
                and str(row["linker_size"]) == linker_size
            ],
        )
        for pdb in pdbs
        for linker_size in linker_sizes
    )
    contexts = sorted({str(row["context_index"]) for row in rows}, key=int)
    scopes.extend(
        (
            "context",
            str(scoped[0]["pdb"]).lower(),
            context_index,
            "all",
            scoped,
        )
        for context_index in contexts
        for scoped in (
            [row for row in rows if str(row["context_index"]) == context_index],
        )
    )
    output: List[Dict[str, Any]] = []
    for scope, pdb, context_index, linker_size, scoped_rows in scopes:
        total = len(scoped_rows)
        for stage, field in FUNNEL_STAGES:
            if field is None:
                passed = total
            elif field == "evaluation_error":
                passed = sum(not row[field] for row in scoped_rows)
            else:
                passed = sum(int(row[field]) for row in scoped_rows)
            output.append(
                {
                    "scope": scope,
                    "pdb": pdb,
                    "context_index": context_index,
                    "linker_size": linker_size,
                    "stage": stage,
                    "passed": passed,
                    "total": total,
                    "rate": passed / total if total else 0.0,
                }
            )
    return output


def _verify_combined_done(
    config: EvaluationConfig, contract: Mapping[str, Any]
) -> bool:
    done_path = config.run_root / "done.json"
    if not done_path.exists():
        return False
    done = load_json(done_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": config.run_id,
        "contract_sha256": contract["contract_sha256"],
        "record_count": config.expected_candidates,
        "shard_count": config.expected_shards,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise ValueError(f"Combined evaluation metadata mismatch: {key}")
    outputs = done.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Combined evaluation has no outputs mapping")
    for name in ("evaluations.csv", "funnel.csv", "funnel.json", "provenance.json"):
        _validate_output_record(
            owner=config.run_root,
            outputs=outputs,
            name=name,
            verify_hash=True,
            expected_records=(
                config.expected_candidates if name == "evaluations.csv" else None
            ),
            expected_fields=(
                GENERATED_EVALUATION_FIELDS if name == "evaluations.csv" else None
            ),
        )
    return True


def _atomic_publish_file(temporary: Path, final: Path) -> None:
    fsync_file(temporary)
    os.replace(temporary, final)
    fsync_directory(final.parent)


def consolidate_generation(
    config: EvaluationConfig,
    generation: GenerationRun,
    contract: Mapping[str, Any],
) -> bool:
    if _verify_combined_done(config, contract):
        print("Combined evaluation outputs are already complete and verified.", flush=True)
        return True

    completed: List[Mapping[str, Any]] = []
    for shard_index in range(config.expected_shards):
        done = validate_completed_evaluation_shard(config, contract, shard_index)
        if done is not None:
            completed.append(done)
    if len(completed) != config.expected_shards:
        print(
            f"Evaluation progress: {len(completed)}/{config.expected_shards} shards "
            "complete; combined outputs not published.",
            flush=True,
        )
        return False

    all_rows: List[Dict[str, str]] = []
    expected_candidate_index = 0
    for shard_index in range(config.expected_shards):
        path = (
            evaluation_shard_directory(config.run_root, shard_index)
            / "evaluations.csv"
        )
        rows, fields = read_csv(path)
        if fields != GENERATED_EVALUATION_FIELDS:
            raise ValueError(f"Evaluation shard schema changed: {relative(path)}")
        for replicate, row in enumerate(rows):
            expected_id = (
                f"{generation.run_id}-{shard_index:04d}-{replicate:03d}"
            )
            if row["candidate_id"] != expected_id:
                raise ValueError(
                    f"Evaluation ordering/identity changed at global row "
                    f"{expected_candidate_index}"
                )
            expected_candidate_index += 1
            all_rows.append(row)
    if len(all_rows) != config.expected_candidates:
        raise RuntimeError(
            f"Consolidated evaluation count mismatch: {len(all_rows)} != "
            f"{config.expected_candidates}"
        )
    if len({row["candidate_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Consolidated evaluation contains duplicate candidate IDs")

    funnel_rows = _funnel_rows(all_rows)
    funnel_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.run_id,
        "denominator_policy": (
            "Every generated candidate remains in every rate denominator; sequential "
            "stages are represented by after_posebusters, after_lipinski, and after_2d."
        ),
        "overall": evaluation_counts(all_rows),
        "by_linker_size": {
            linker_size: evaluation_counts(
                [row for row in all_rows if row["linker_size"] == linker_size]
            )
            for linker_size in sorted(
                {row["linker_size"] for row in all_rows}, key=int
            )
        },
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "run_id": config.run_id,
        "contract": dict(contract),
        "contract_sha256": contract["contract_sha256"],
        "generation": {
            "path": relative(config.generation_run),
            "run_id": generation.run_id,
            "contract_sha256": generation.contract_sha256,
            "done_sha256": generation.done_sha256,
        },
        "record_count": len(all_rows),
        "shard_count": len(completed),
        "ordering": "ascending generation shard_index, then ascending replicate",
        "candidate_identity": (
            "<generation-run-id>-<four-digit-shard>-<three-digit-replicate>"
        ),
        "shards": [
            {
                "shard_index": index,
                "path": relative(evaluation_shard_directory(config.run_root, index)),
                "done_sha256": sha256_file(
                    evaluation_shard_directory(config.run_root, index) / "done.json"
                ),
            }
            for index in range(config.expected_shards)
        ],
    }

    temporary_paths = {
        name: config.run_root / f".{name}.tmp.{os.getpid()}"
        for name in ("evaluations.csv", "funnel.csv", "funnel.json", "provenance.json")
    }
    try:
        written = write_csv(
            temporary_paths["evaluations.csv"], GENERATED_EVALUATION_FIELDS, all_rows
        )
        if written != config.expected_candidates:
            raise RuntimeError("Combined evaluation CSV serialization count mismatch")
        write_csv(temporary_paths["funnel.csv"], FUNNEL_FIELDS, funnel_rows)
        temporary_paths["funnel.json"].write_text(
            json.dumps(funnel_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fsync_file(temporary_paths["funnel.json"])
        temporary_paths["provenance.json"].write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fsync_file(temporary_paths["provenance.json"])
        for name, temporary in temporary_paths.items():
            _atomic_publish_file(temporary, config.run_root / name)
        outputs = {
            name: {
                "bytes": (config.run_root / name).stat().st_size,
                "sha256": sha256_file(config.run_root / name),
                **(
                    {"records": config.expected_candidates}
                    if name == "evaluations.csv"
                    else {}
                ),
            }
            for name in temporary_paths
        }
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "run_id": config.run_id,
            "contract_sha256": contract["contract_sha256"],
            "shard_count": config.expected_shards,
            "record_count": config.expected_candidates,
            "counts": evaluation_counts(all_rows),
            "outputs": outputs,
        }
        atomic_write_json(config.run_root / "done.json", done)
        print(
            f"Evaluation complete: shards={config.expected_shards} "
            f"records={config.expected_candidates} final={done['counts']['after_2d']}",
            flush=True,
        )
        return True
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)


def validate_gt_inputs(
    config: EvaluationConfig,
) -> Tuple[List[Dict[str, str]], Tuple[str, ...], List[SDFRecord]]:
    if (
        config.gt_table is None
        or config.gt_molecules is None
        or config.gt_expected_candidates is None
        or config.gt_samples_per_shard is None
    ):
        raise ValueError("Ground-truth inputs are not configured")
    rows, fields = read_csv(config.gt_table)
    required = {
        "uuid",
        "fragmentation_id",
        "molecule_name",
        "structure_id",
        "split",
        "kinase",
        "pdb",
        "ligand_code",
        "linker_size",
        "canonical_smiles",
    }
    if not required.issubset(fields):
        raise ValueError(
            "Ground-truth table is missing fields: "
            + ", ".join(sorted(required - set(fields)))
        )
    records = read_sdf_records(config.gt_molecules)
    if len(rows) != config.gt_expected_candidates or len(records) != len(rows):
        raise ValueError(
            "Ground-truth table/SDF count mismatch: "
            f"table={len(rows)}, sdf={len(records)}, "
            f"expected={config.gt_expected_candidates}"
        )
    for row_index, (row, record) in enumerate(zip(rows, records)):
        if _require_int(row, "uuid", context=f"GT row {row_index}") != row_index:
            raise ValueError(f"Ground-truth UUID is not row-aligned at row {row_index}")
        if row["split"] != "test":
            raise ValueError(f"Ground-truth row is not test-only at row {row_index}")
        if record.title != row["fragmentation_id"]:
            raise ValueError(f"Ground-truth SDF/table identity mismatch at row {row_index}")
    sidecars = [json.loads(line) for line in gt_chemistry_path(config).read_text().splitlines() if line.strip()]
    if len(sidecars) != len(rows):
        raise ValueError("Ground-truth chemistry sidecar row count differs")
    verified_records = []
    for row_index, (row, record, sidecar) in enumerate(zip(rows, records, sidecars)):
        for key in ("uuid", "fragmentation_id", "ligand_sha256", "canonical_smiles"):
            if str(sidecar[key]) != str(row[key]):
                raise ValueError(f"Ground-truth chemistry sidecar {key} differs at row {row_index}")
        authoritative = Chem.Mol(base64.b64decode(sidecar["parent_molecule_binary_base64"], validate=True))
        Chem.SanitizeMol(authoritative)
        from .fragmentation import canonical_isomeric_identity
        if canonical_isomeric_identity(authoritative) != canonical_isomeric_identity(Chem.MolFromSmiles(row["canonical_smiles"])):
            raise ValueError(f"Authoritative ground-truth identity mismatch at row {row_index}")
        molecule = Chem.MolFromMolBlock(record.mol_block, sanitize=True, removeHs=False)
        if molecule is None:
            raise ValueError(f"Ground-truth record does not sanitize at row {row_index}")
        atom_keys = lambda mol: [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge()) for a in mol.GetAtoms()]
        if atom_keys(molecule) != atom_keys(authoritative):
            raise ValueError(f"Ground-truth atom order/charge changed at row {row_index}")
        if Chem.MolToSmiles(molecule, isomericSmiles=False) != Chem.MolToSmiles(authoritative, isomericSmiles=False):
            raise ValueError(f"Ground-truth constitution changed at row {row_index}")
        if not np.allclose(molecule.GetConformer().GetPositions(), authoritative.GetConformer().GetPositions(), atol=0.000051, rtol=0):
            raise ValueError(f"Ground-truth native coordinates changed at row {row_index}")
        verified_records.append(SDFRecord(record.index, record.title, record.properties, record.mol_block, authoritative))
    return rows, fields, verified_records


def _gt_source_row(row: Mapping[str, str], row_index: int) -> Dict[str, Any]:
    return {
        "gt_id": f"gt-{row_index:05d}",
        "row_index": row_index,
        "uuid": row["uuid"],
        "fragmentation_id": row["fragmentation_id"],
        "molecule_name": row["molecule_name"],
        "structure_id": row["structure_id"],
        "kinase": row["kinase"],
        "pdb": row["pdb"].lower(),
        "ligand_code": row["ligand_code"],
        "linker_size": row["linker_size"],
        "source_smiles": row["canonical_smiles"],
    }


def gt_shard_directory(run_root: Path, shard_index: int) -> Path:
    return run_root / "gt_shards" / f"{shard_index:04d}"


def _gt_shard_bounds(config: EvaluationConfig, shard_index: int) -> Tuple[int, int]:
    if config.gt_expected_shards is None or config.gt_samples_per_shard is None:
        raise ValueError("Ground-truth inputs are not configured")
    if not 0 <= shard_index < config.gt_expected_shards:
        raise ValueError(
            f"GT shard_index must be in 0..{config.gt_expected_shards - 1}, "
            f"found {shard_index}"
        )
    start = shard_index * config.gt_samples_per_shard
    end = min(
        start + config.gt_samples_per_shard,
        int(config.gt_expected_candidates),
    )
    return start, end


def validate_completed_gt_shard(
    config: EvaluationConfig,
    contract: Mapping[str, Any],
    shard_index: int,
) -> Optional[Mapping[str, Any]]:
    start, end = _gt_shard_bounds(config, shard_index)
    shard_dir = gt_shard_directory(config.run_root, shard_index)
    if not shard_dir.exists():
        return None
    if not shard_dir.is_dir() or shard_dir.is_symlink():
        raise ValueError(f"GT shard path is not a real directory: {relative(shard_dir)}")
    done_path = shard_dir / "done.json"
    if not done_path.is_file():
        raise FileExistsError(f"Refusing to replace incomplete GT shard: {relative(shard_dir)}")
    done = load_json(done_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": config.run_id,
        "contract_sha256": contract["contract_sha256"],
        "shard_index": shard_index,
        "row_start": start,
        "row_end": end,
        "record_count": end - start,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise ValueError(f"Completed GT shard metadata mismatch ({key})")
    outputs = done.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"GT shard has no outputs mapping: {relative(done_path)}")
    _validate_output_record(
        owner=shard_dir,
        outputs=outputs,
        name="evaluations.csv",
        verify_hash=True,
        expected_records=end - start,
        expected_fields=GT_EVALUATION_FIELDS,
    )
    return done


def evaluate_gt_shard(
    config: EvaluationConfig,
    contract: Mapping[str, Any],
    shard_index: int,
    dependencies: EvaluationDependencies,
) -> Mapping[str, Any]:
    completed = validate_completed_gt_shard(config, contract, shard_index)
    if completed is not None:
        print(f"GT evaluation shard={shard_index:04d} status=skipped", flush=True)
        return completed
    rows, _fields, records = validate_gt_inputs(config)
    start, end = _gt_shard_bounds(config, shard_index)
    attempts: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    evaluated = [
        evaluate_record(
            _gt_source_row(rows[row_index], row_index),
            records[row_index],
            dependencies,
            stage_attempts=attempts,
            stage_errors=errors,
        )
        for row_index in range(start, end)
    ]
    _fail_on_systemic_stage_errors(attempts, errors)
    if len(evaluated) != end - start:
        raise AssertionError("GT evaluation dropped a source row")

    shards_root = config.run_root / "gt_shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    final_dir = gt_shard_directory(config.run_root, shard_index)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{shard_index:04d}.tmp.", dir=str(shards_root))
    )
    try:
        csv_path = temporary / "evaluations.csv"
        written = write_csv(csv_path, GT_EVALUATION_FIELDS, evaluated)
        output_record = {
            "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "records": written,
        }
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "run_id": config.run_id,
            "contract_sha256": contract["contract_sha256"],
            "shard_index": shard_index,
            "row_start": start,
            "row_end": end,
            "record_count": written,
            "source": {
                "table": relative(config.gt_table),  # type: ignore[arg-type]
                "table_sha256": sha256_file(config.gt_table),  # type: ignore[arg-type]
                "molecules": relative(config.gt_molecules),  # type: ignore[arg-type]
                "molecules_sha256": sha256_file(config.gt_molecules),  # type: ignore[arg-type]
            },
            "counts": evaluation_counts(evaluated),
            "stage_attempts": attempts,
            "stage_errors": errors,
            "software": dict(dependencies.versions),
            "runtime": {"host": socket.getfqdn(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
                        "evaluation_threads": os.environ.get("KINASE_EVALUATION_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))},
            "outputs": {"evaluations.csv": output_record},
        }
        atomic_write_json(temporary / "done.json", done)
        fsync_directory(temporary)
        os.replace(temporary, final_dir)
        fsync_directory(shards_root)
        print(
            f"GT evaluation shard={shard_index:04d} status=complete "
            f"records={written} final={done['counts']['after_2d']}",
            flush=True,
        )
        return done
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def consolidate_gt(
    config: EvaluationConfig,
    contract: Mapping[str, Any],
) -> bool:
    if (
        config.gt_expected_shards is None
        or config.gt_expected_candidates is None
    ):
        raise ValueError("Ground-truth inputs are not configured")
    done_path = config.run_root / "gt_done.json"
    if done_path.exists():
        done = load_json(done_path)
        expected = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_id": config.run_id,
            "contract_sha256": contract["contract_sha256"],
            "shard_count": config.gt_expected_shards,
            "record_count": config.gt_expected_candidates,
        }
        for key, value in expected.items():
            if done.get(key) != value:
                raise ValueError(f"Combined GT metadata mismatch: {key}")
        outputs = done.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError("Combined GT output mapping is missing")
        _validate_output_record(
            owner=config.run_root,
            outputs=outputs,
            name="gt_evaluations.csv",
            verify_hash=True,
            expected_records=config.gt_expected_candidates,
            expected_fields=GT_EVALUATION_FIELDS,
        )
        _validate_output_record(
            owner=config.run_root,
            outputs=outputs,
            name="gt_provenance.json",
            verify_hash=True,
        )
        print("Combined GT evaluation is already complete and verified.", flush=True)
        return True

    completed = [
        validate_completed_gt_shard(config, contract, index)
        for index in range(config.gt_expected_shards)
    ]
    if any(done is None for done in completed):
        count = sum(done is not None for done in completed)
        print(
            f"GT evaluation progress: {count}/{config.gt_expected_shards} shards "
            "complete; combined output not published.",
            flush=True,
        )
        return False

    all_rows: List[Dict[str, str]] = []
    for shard_index in range(config.gt_expected_shards):
        path = gt_shard_directory(config.run_root, shard_index) / "evaluations.csv"
        rows, fields = read_csv(path)
        if fields != GT_EVALUATION_FIELDS:
            raise ValueError(f"GT evaluation shard schema changed: {relative(path)}")
        for row in rows:
            expected_id = f"gt-{len(all_rows):05d}"
            if row["gt_id"] != expected_id or int(row["row_index"]) != len(all_rows):
                raise ValueError(f"GT evaluation identity/order changed at {expected_id}")
            all_rows.append(row)
    if len(all_rows) != config.gt_expected_candidates:
        raise RuntimeError("Combined GT record count mismatch")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "run_id": config.run_id,
        "contract_sha256": contract["contract_sha256"],
        "record_count": len(all_rows),
        "shard_count": config.gt_expected_shards,
        "duplicates": "retained exactly in KLIFS test table row order",
        "identity": "gt-<five-digit-table-row-index>",
        "counts": evaluation_counts(all_rows),
        "source": dict(contract["ground_truth"]),
        "shards": [
            {
                "shard_index": index,
                "path": relative(gt_shard_directory(config.run_root, index)),
                "done_sha256": sha256_file(
                    gt_shard_directory(config.run_root, index) / "done.json"
                ),
            }
            for index in range(config.gt_expected_shards)
        ],
    }
    csv_temporary = config.run_root / f".gt_evaluations.csv.tmp.{os.getpid()}"
    provenance_temporary = config.run_root / f".gt_provenance.json.tmp.{os.getpid()}"
    try:
        write_csv(csv_temporary, GT_EVALUATION_FIELDS, all_rows)
        provenance_temporary.write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fsync_file(provenance_temporary)
        _atomic_publish_file(csv_temporary, config.run_root / "gt_evaluations.csv")
        _atomic_publish_file(
            provenance_temporary, config.run_root / "gt_provenance.json"
        )
        outputs = {
            "gt_evaluations.csv": {
                "bytes": (config.run_root / "gt_evaluations.csv").stat().st_size,
                "sha256": sha256_file(config.run_root / "gt_evaluations.csv"),
                "records": config.gt_expected_candidates,
            },
            "gt_provenance.json": {
                "bytes": (config.run_root / "gt_provenance.json").stat().st_size,
                "sha256": sha256_file(config.run_root / "gt_provenance.json"),
            },
        }
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "run_id": config.run_id,
            "contract_sha256": contract["contract_sha256"],
            "shard_count": config.gt_expected_shards,
            "record_count": config.gt_expected_candidates,
            "counts": evaluation_counts(all_rows),
            "outputs": outputs,
        }
        atomic_write_json(done_path, done)
        print(
            f"GT evaluation complete: shards={config.gt_expected_shards} "
            f"records={config.gt_expected_candidates} "
            f"final={done['counts']['after_2d']}",
            flush=True,
        )
        return True
    finally:
        csv_temporary.unlink(missing_ok=True)
        provenance_temporary.unlink(missing_ok=True)


def preflight_report(
    config: EvaluationConfig,
    generation: GenerationRun,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    gt_summary: Optional[Dict[str, Any]] = None
    if config.gt_table is not None:
        gt_rows, _gt_fields, gt_records = validate_gt_inputs(config)
        gt_summary = {
            "table": relative(config.gt_table),
            "molecules": relative(config.gt_molecules),  # type: ignore[arg-type]
            "records": len(gt_rows),
            "sdf_records": len(gt_records),
            "shards": config.gt_expected_shards,
            "samples_per_shard": config.gt_samples_per_shard,
            "duplicates_retained": True,
        }
    return {
        "status": "preflight_passed",
        "writes_performed": False,
        "config": relative(config.path),
        "config_sha256": sha256_file(config.path),
        "run_id": config.run_id,
        "generation": {
            "path": relative(config.generation_run),
            "run_id": generation.run_id,
            "contract_sha256": generation.contract_sha256,
            "done_sha256": generation.done_sha256,
            "shards": config.expected_shards,
            "samples_per_shard": config.samples_per_shard,
            "candidates": config.expected_candidates,
        },
        "ground_truth": gt_summary,
        "protocol": {
            "posebusters_config": config.posebusters_config,
            "posebusters_checks": list(POSEBUSTERS_CHECKS),
            "lipinski": dict(contract["lipinski"]),
            "pains": dict(contract["pains"]),
            "funnel_order": ["posebusters", "lipinski", "2d"],
        },
        "contract_sha256": contract["contract_sha256"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "gt-preflight", "consolidate", "gt-consolidate"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/evaluation.yml")
    for name in ("shard", "gt-shard"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/evaluation.yml")
        child.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    dependencies: Optional[EvaluationDependencies] = None,
) -> int:
    args = parse_args(argv)
    config = read_config(args.config)
    if args.command in ("gt-preflight", "gt-shard", "gt-consolidate"):
        contract = gt_evaluation_contract(config)
        if args.command == "gt-preflight":
            rows, _, _ = validate_gt_inputs(config)
            print(json.dumps({"status": "preflight_passed", "gt_records": len(rows), "contract": contract}, sort_keys=True))
            return 0
        ensure_gt_run_contract(config, contract)
        if args.command == "gt-consolidate":
            return 0 if consolidate_gt(config, contract) else 2
        resolved_dependencies = dependencies or load_dependencies(config)
        evaluate_gt_shard(config, contract, args.shard_index, resolved_dependencies)
        return 0
    generation = validate_generation_run(
        config, verify_combined_hashes=args.command in ("preflight", "consolidate"),
        allow_incomplete=args.command == "shard"
    )
    contract = evaluation_contract(config, generation)

    if args.command == "preflight":
        report = preflight_report(config, generation, contract)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    ensure_run_contract(config, generation, contract)
    if args.command == "consolidate":
        return 0 if consolidate_generation(config, generation, contract) else 2
    if args.command == "gt-consolidate":
        return 0 if consolidate_gt(config, contract) else 2

    resolved_dependencies = dependencies or load_dependencies(config)
    if args.command == "shard":
        evaluate_generation_shard(
            config,
            generation,
            contract,
            args.shard_index,
            resolved_dependencies,
        )
        return 0
    if args.command == "gt-shard":
        evaluate_gt_shard(
            config,
            contract,
            args.shard_index,
            resolved_dependencies,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
