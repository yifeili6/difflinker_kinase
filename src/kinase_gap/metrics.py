"""Deterministic aggregation of the kinase-gap candidate evaluation.

This module deliberately keeps molecule-level evaluation separate from aggregate
statistics.  Its input is the consolidated CSV produced by
``kinase_gap.evaluate``.  The three sequential survivor columns are treated as
the source of truth:

``after_posebusters`` -> ``after_lipinski`` -> ``after_2d``.

MOSES is an optional runtime dependency and is imported only when no metric or
descriptor function is injected. The default engine calls MOSES' declared
metric primitives and caches reference/generated intermediates; it deliberately
does not invoke MOSES 0.3.1's inseparable, unrelated scaffold-test branch.
This keeps the aggregation logic testable in the locked training environment,
where MOSES is intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

from .identity import IDENTITY_FIELDS, normalize_smiles_identity, with_normalized_identity


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_COLUMNS = (
    ("input", None),
    ("posebusters", "after_posebusters"),
    ("lipinski", "after_lipinski"),
    ("two_d", "after_2d"),
)

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "linker_size",
    "pdb",
    "context_index",
    "canonical_smiles",
    "after_posebusters",
    "after_lipinski",
}

FINAL_FLAG_COLUMNS = ("after_2d", "final_pass")

REQUIRED_GT_METRIC_COLUMNS = {
    "gt_id",
    "row_index",
    "linker_size",
    "pdb",
    "canonical_smiles",
    "after_posebusters",
    "after_lipinski",
}

FUNNEL_FIELDS = (
    "scope",
    "group_value",
    "linker_size",
    "pdb",
    "context_index",
    "stage",
    "stage_order",
    "count",
    "input_count",
    "conditional_denominator",
    "conditional_rate",
    "cumulative_rate",
    "conditional_ci_low",
    "conditional_ci_high",
    "cumulative_ci_low",
    "cumulative_ci_high",
)

METRIC_FIELDS = (
    "reference_label",
    "reference_description",
    "scope",
    "group_value",
    "linker_size",
    "generated_count",
    "generated_unique_count",
    "reference_count",
    "train_count",
    "novelty",
    "novelty_ci_low",
    "novelty_ci_high",
    "unique_at_1000",
    "unique_at_all",
    "valid",
    "fcd_test",
    "fcd_test_defined",
    "fcd_test_undefined_reason",
    "snn_test",
    "frag_test",
    "intdiv",
    "intdiv2",
    "filters",
    "w1_logp",
    "w1_sa",
    "w1_qed",
    "w1_molecular_weight",
)

MOSES_KEYS = {
    "valid": ("valid", "Valid"),
    "fcd_test": ("FCD/Test",),
    "snn_test": ("SNN/Test",),
    "frag_test": ("Frag/Test",),
    "intdiv": ("IntDiv",),
    "intdiv2": ("IntDiv2",),
    "filters": ("Filters",),
}

DESCRIPTOR_KEYS = ("logp", "sa", "qed", "molecular_weight")

METRIC_DEFINITIONS: Dict[str, Any] = {
    "chemical_identity": (
        "SMILES are parsed and converted to canonical isomeric identities before "
        "deduplication, uniqueness, novelty, and metric computation. Source "
        "canonical_smiles and candidate IDs are retained; normalized_identity and "
        "identity_normalization_status record the identity used for statistics."
    ),
    "population_weighting": (
        "Generated and GT final-survivor rows retain their original order and "
        "multiplicity for FCD, SNN, fragment similarity, internal diversity, filters, "
        "and descriptor distributions. Full-molecule reference identities are unique. "
        "Novelty uses unique survivor identities; uniqueness uses survivor row counts."
    ),
    "funnel": {
        "input": "All generated candidate records in the consolidated CSV.",
        "posebusters": (
            "Candidates with after_posebusters=1. Conditional denominator: input."
        ),
        "lipinski": (
            "Candidates with after_lipinski=1. Conditional denominator: "
            "PoseBusters survivors."
        ),
        "two_d": (
            "Candidates with after_2d=1 after the MOSES/PAINS stage. Conditional "
            "denominator: Lipinski survivors."
        ),
        "conditional_rate": "stage count divided by the immediately preceding stage.",
        "cumulative_rate": "stage count divided by the original input count.",
        "confidence_interval": (
            "Deterministic percentile interval from a binomial bootstrap with the "
            "recorded seed and number of replicates."
        ),
    },
    "reference_sets": {
        "primary_test": (
            "Corrected primary comparison: unique full-molecule canonical SMILES "
            "from the leakage-safe KLIFS test split only."
        ),
        "legacy_val_test_sensitivity": (
            "Sensitivity analysis matching the legacy script's KLIF_ValTest default: "
            "the union of unique validation and test full-molecule canonical SMILES."
        ),
    },
    "novelty": (
        "Fraction of unique final-survivor canonical SMILES absent from the unique "
        "KLIFS training full-molecule SMILES."
    ),
    "unique_at_1000": (
        "Fraction of distinct canonical SMILES among the first min(1000, N) final "
        "survivors in stable candidate-file order."
    ),
    "unique_at_all": (
        "Fraction of distinct canonical SMILES among all final survivors in a cohort."
    ),
    "moses": {
        "implementation": (
            "MOSES 0.3.1 metric primitives with cached intermediate statistics. "
            "The MOSES scaffold-test panel is not computed because KLIFS has no "
            "declared scaffold-test comparator and get_all_metrics cannot disable "
            "its bundled 176k-molecule scaffold benchmark."
        ),
        "valid": "Raw MOSES validity result.",
        "fcd_test": (
            "Raw MOSES FCD/Test result. JSON null and a blank CSV value mean "
            "undefined because a generated or reference population contains fewer "
            "than two molecules; the adjacent status fields record the reason."
        ),
        "snn_test": "Raw MOSES SNN/Test result.",
        "frag_test": "Raw MOSES Frag/Test result.",
        "intdiv": "Raw MOSES IntDiv result.",
        "intdiv2": "Raw MOSES IntDiv2 result.",
        "filters": "Raw MOSES Filters result.",
    },
    "wasserstein": (
        "Empirical Wasserstein-1 distance between final-survivor and reference-test "
        "distributions for RDKit/MOSES logP, SA, QED, and molecular weight."
    ),
}


MetricFunction = Callable[..., Mapping[str, Any]]
DescriptorFunction = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class ReferenceSet:
    """A named training/test reference pair for aggregate metrics."""

    label: str
    description: str
    train_smiles: Tuple[str, ...]
    test_smiles: Tuple[str, ...]
    test_sources: Tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _strict_binary(value: Any, column: str, candidate_id: str) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return 1
    if text in {"0", "false"}:
        return 0
    raise ValueError(
        f"Candidate {candidate_id!r} has non-binary {column} value {value!r}"
    )


def _optional_binary(value: Any, column: str, candidate_id: str) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return _strict_binary(value, column, candidate_id)


def _final_flag(row: Mapping[str, Any], candidate_id: str) -> int:
    """Read the final-stage flag, requiring exact agreement when aliased."""

    present = [column for column in FINAL_FLAG_COLUMNS if column in row]
    if not present:
        raise ValueError(
            f"Candidate {candidate_id!r} has neither after_2d nor final_pass"
        )
    parsed = {
        column: _strict_binary(row[column], column, candidate_id)
        for column in present
    }
    if len(set(parsed.values())) != 1:
        raise ValueError(
            f"Candidate {candidate_id!r} has inconsistent after_2d/final_pass aliases"
        )
    return next(iter(parsed.values()))


def _strict_int(value: Any, column: str, candidate_id: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Candidate {candidate_id!r} has empty {column}")
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Candidate {candidate_id!r} has invalid integer {column}={value!r}"
        ) from exc
    if str(parsed) != text and text not in {f"+{parsed}", f"-{abs(parsed)}"}:
        raise ValueError(
            f"Candidate {candidate_id!r} has non-canonical integer "
            f"{column}={value!r}"
        )
    return parsed


def read_candidate_csv(path: Path) -> Tuple[Tuple[str, ...], List[Dict[str, str]]]:
    """Read and strictly validate a consolidated candidate CSV."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Candidate CSV has no header: {path}")
        fields = tuple(reader.fieldnames)
        if len(fields) != len(set(fields)):
            raise ValueError(f"Candidate CSV has duplicate column names: {path}")
        missing = REQUIRED_CANDIDATE_COLUMNS.difference(fields)
        if missing:
            raise ValueError(
                f"Candidate CSV is missing required columns: {sorted(missing)}"
            )
        if not set(FINAL_FLAG_COLUMNS).intersection(fields):
            raise ValueError(
                "Candidate CSV must contain either after_2d or final_pass"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Candidate CSV contains no records: {path}")
    validate_candidate_rows(rows)
    rows = [with_normalized_identity(row, require_valid=bool(_final_flag(row, row["candidate_id"]))) for row in rows]
    fields += tuple(field for field in IDENTITY_FIELDS if field not in fields)
    return fields, rows


def read_gt_evaluation_csv(
    path: Path,
) -> Tuple[Tuple[str, ...], List[Dict[str, str]]]:
    """Read GT evaluations and adapt their identity for shared metric logic.

    Rows and duplicate SMILES remain in exact file order. Only two in-memory
    aliases are added: ``candidate_id=gt_id`` and ``context_index=row_index``.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"GT evaluation CSV has no header: {path}")
        fields = tuple(reader.fieldnames)
        if len(fields) != len(set(fields)):
            raise ValueError(f"GT evaluation CSV has duplicate columns: {path}")
        missing = REQUIRED_GT_METRIC_COLUMNS.difference(fields)
        if missing:
            raise ValueError(
                f"GT evaluation CSV is missing required columns: {sorted(missing)}"
            )
        if not set(FINAL_FLAG_COLUMNS).intersection(fields):
            raise ValueError(
                "GT evaluation CSV must contain either after_2d or final_pass"
            )
        source_rows = [dict(row) for row in reader]
    if not source_rows:
        raise ValueError(f"GT evaluation CSV contains no records: {path}")

    rows: List[Dict[str, str]] = []
    for position, source in enumerate(source_rows):
        gt_id = str(source["gt_id"]).strip()
        row_index = _strict_int(source["row_index"], "row_index", gt_id)
        if row_index != position:
            raise ValueError(
                f"GT evaluation order changed at position {position}: "
                f"row_index={row_index}"
            )
        expected_id = f"gt-{position:05d}"
        if gt_id != expected_id:
            raise ValueError(
                f"GT evaluation identity changed at position {position}: "
                f"{gt_id!r} != {expected_id!r}"
            )
        row = dict(source)
        row["candidate_id"] = gt_id
        row["context_index"] = str(row_index)
        rows.append(row)
    validate_candidate_rows(rows)
    rows = [with_normalized_identity(row, require_valid=bool(_final_flag(row, row["candidate_id"]))) for row in rows]
    fields += tuple(field for field in IDENTITY_FIELDS if field not in fields)
    return fields, rows


def validate_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate identity and monotonic sequential-stage invariants."""

    if not rows:
        raise ValueError("Candidate rows must not be empty")
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        missing = REQUIRED_CANDIDATE_COLUMNS.difference(row)
        if missing:
            raise ValueError(
                f"Candidate row {row_number} is missing columns: {sorted(missing)}"
            )
        candidate_id = str(row["candidate_id"]).strip()
        if not candidate_id:
            raise ValueError(f"Candidate row {row_number} has an empty candidate_id")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)

        linker_size = _strict_int(row["linker_size"], "linker_size", candidate_id)
        context_index = _strict_int(
            row["context_index"], "context_index", candidate_id
        )
        if linker_size <= 0 or context_index < 0:
            raise ValueError(
                f"Candidate {candidate_id!r} has invalid grouping identifiers"
            )
        if not str(row["pdb"]).strip():
            raise ValueError(f"Candidate {candidate_id!r} has an empty pdb")

        after_posebusters = _strict_binary(
            row["after_posebusters"], "after_posebusters", candidate_id
        )
        after_lipinski = _strict_binary(
            row["after_lipinski"], "after_lipinski", candidate_id
        )
        after_2d = _final_flag(row, candidate_id)
        if after_lipinski > after_posebusters or after_2d > after_lipinski:
            raise ValueError(
                f"Candidate {candidate_id!r} violates sequential funnel monotonicity"
            )

        # When component flags are present, require exact agreement with the
        # cumulative flags. Empty component fields mean "not evaluable" and fail
        # closed to zero.
        if "posebusters_pass" in row:
            component = _optional_binary(
                row.get("posebusters_pass"), "posebusters_pass", candidate_id
            )
            if after_posebusters != component:
                raise ValueError(
                    f"Candidate {candidate_id!r} has inconsistent PoseBusters flags"
                )
        if "lipinski_pass" in row:
            component = _optional_binary(
                row.get("lipinski_pass"), "lipinski_pass", candidate_id
            )
            expected = after_posebusters & component
            if after_lipinski != expected:
                raise ValueError(
                    f"Candidate {candidate_id!r} has inconsistent Lipinski flags"
                )
        if "moses_filter_pass" in row:
            component = _optional_binary(
                row.get("moses_filter_pass"), "moses_filter_pass", candidate_id
            )
            expected = after_lipinski & component
            if after_2d != expected:
                raise ValueError(
                    f"Candidate {candidate_id!r} has inconsistent 2D-filter flags"
                )

        if after_2d and not str(row["canonical_smiles"]).strip():
            raise ValueError(
                f"Final survivor {candidate_id!r} has no canonical_smiles"
            )


def select_final_survivors(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return final sequential survivors in stable input order."""

    validate_candidate_rows(rows)
    survivors: List[Dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if _final_flag(row, candidate_id):
            output = dict(row)
            if "final_pass" in output and _strict_binary(
                output["final_pass"], "final_pass", candidate_id
            ) != 1:
                raise ValueError(
                    f"Candidate {candidate_id!r} has conflicting final_pass"
                )
            output["final_pass"] = 1
            survivors.append(output)
    return survivors


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    *,
    seed: int,
    samples: int = 2000,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Return a deterministic percentile interval for a binary proportion."""

    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Invalid successes/total for proportion")
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between zero and one")
    if total == 0:
        return None, None
    probability = successes / total
    if successes in {0, total}:
        return probability, probability
    rng = np.random.default_rng(seed)
    draws = rng.binomial(total, probability, size=samples) / total
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _rate(successes: int, total: int) -> Optional[float]:
    return None if total == 0 else successes / total


def _group_specs(
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[Tuple[str, str, List[Mapping[str, Any]]]]:
    yield "overall", "all", list(rows)
    specifications = (
        ("linker_size", lambda row: _strict_int(
            row["linker_size"], "linker_size", str(row["candidate_id"])
        )),
        ("pdb", lambda row: str(row["pdb"]).strip().lower()),
        ("context_index", lambda row: _strict_int(
            row["context_index"], "context_index", str(row["candidate_id"])
        )),
    )
    for scope, getter in specifications:
        grouped: Dict[Any, List[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(getter(row), []).append(row)
        for key in sorted(grouped):
            yield scope, str(key), grouped[key]

    joint: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        key = (
            str(row["pdb"]).strip().lower(),
            _strict_int(row["linker_size"], "linker_size", candidate_id),
        )
        joint.setdefault(key, []).append(row)
    for (pdb, linker_size), group_rows in sorted(joint.items()):
        yield "pdb_linker_size", f"{pdb}|{linker_size}", group_rows


def build_funnel(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260724,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> List[Dict[str, Any]]:
    """Build long-form funnel summaries overall and by requested groupings."""

    validate_candidate_rows(rows)
    output: List[Dict[str, Any]] = []
    for scope, group_value, group_rows in _group_specs(rows):
        counts = {"input": len(group_rows)}
        for stage, column in STAGE_COLUMNS[1:]:
            assert column is not None
            counts[stage] = sum(
                (
                    _final_flag(row, str(row["candidate_id"]))
                    if column == "after_2d"
                    else _strict_binary(
                        row[column], column, str(row["candidate_id"])
                    )
                )
                for row in group_rows
            )
        if not (
            counts["input"]
            >= counts["posebusters"]
            >= counts["lipinski"]
            >= counts["two_d"]
        ):
            raise ValueError(f"Non-monotonic aggregate funnel for {scope}={group_value}")

        previous_count = counts["input"]
        for stage_order, (stage, _) in enumerate(STAGE_COLUMNS):
            count = counts[stage]
            conditional_denominator = (
                counts["input"] if stage == "input" else previous_count
            )
            conditional = _rate(count, conditional_denominator)
            cumulative = _rate(count, counts["input"])
            conditional_ci = bootstrap_proportion_ci(
                count,
                conditional_denominator,
                seed=_stable_seed(
                    bootstrap_seed, scope, group_value, stage, "conditional"
                ),
                samples=bootstrap_samples,
                confidence=confidence,
            )
            cumulative_ci = bootstrap_proportion_ci(
                count,
                counts["input"],
                seed=_stable_seed(
                    bootstrap_seed, scope, group_value, stage, "cumulative"
                ),
                samples=bootstrap_samples,
                confidence=confidence,
            )
            record: Dict[str, Any] = {
                "scope": scope,
                "group_value": group_value,
                "linker_size": (
                    group_value
                    if scope == "linker_size"
                    else (
                        _strict_int(
                            group_rows[0]["linker_size"],
                            "linker_size",
                            str(group_rows[0]["candidate_id"]),
                        )
                        if scope == "pdb_linker_size"
                        else None
                    )
                ),
                "pdb": (
                    group_value
                    if scope == "pdb"
                    else (
                        str(group_rows[0]["pdb"]).strip().lower()
                        if scope == "pdb_linker_size"
                        else None
                    )
                ),
                "context_index": group_value if scope == "context_index" else None,
                "stage": stage,
                "stage_order": stage_order,
                "count": count,
                "input_count": counts["input"],
                "conditional_denominator": conditional_denominator,
                "conditional_rate": conditional,
                "cumulative_rate": cumulative,
                "conditional_ci_low": conditional_ci[0],
                "conditional_ci_high": conditional_ci[1],
                "cumulative_ci_low": cumulative_ci[0],
                "cumulative_ci_high": cumulative_ci[1],
            }
            output.append(record)
            previous_count = count
    return output


def read_reference_smiles(
    path: Path, *, smiles_column: str = "canonical_smiles"
) -> Tuple[str, ...]:
    """Read, validate, and deterministically deduplicate full-molecule SMILES."""

    if not path.is_file():
        raise FileNotFoundError(path)
    values = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Reference table has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Reference table has duplicate columns: {path}")
        if smiles_column not in reader.fieldnames:
            raise ValueError(
                f"Reference table {path} has no {smiles_column!r} column"
            )
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            smiles = str(row.get(smiles_column, "")).strip()
            if not smiles:
                raise ValueError(
                    f"Reference table {path} row {row_number} has empty "
                    f"{smiles_column}"
                )
            try:
                values.add(normalize_smiles_identity(smiles))
            except ValueError as exc:
                raise ValueError(f"Reference table {path} row {row_number}: {exc}") from exc
    if row_count == 0 or not values:
        raise ValueError(f"Reference table contains no SMILES: {path}")
    return tuple(sorted(values))


def build_reference_sets(
    train_table: Path,
    test_table: Path,
    *,
    validation_table: Optional[Path] = None,
    smiles_column: str = "canonical_smiles",
    require_disjoint: bool = True,
) -> Tuple[ReferenceSet, ...]:
    """Build the corrected primary and optional legacy-sensitivity references."""

    train = read_reference_smiles(train_table, smiles_column=smiles_column)
    test = read_reference_smiles(test_table, smiles_column=smiles_column)
    if require_disjoint and set(train).intersection(test):
        raise ValueError("KLIFS train and test full-molecule references overlap")
    references = [
        ReferenceSet(
            label="primary_test",
            description=METRIC_DEFINITIONS["reference_sets"]["primary_test"],
            train_smiles=train,
            test_smiles=test,
            test_sources=("test",),
        )
    ]
    if validation_table is not None:
        validation = read_reference_smiles(
            validation_table, smiles_column=smiles_column
        )
        legacy_test = tuple(sorted(set(validation).union(test)))
        if require_disjoint and set(train).intersection(legacy_test):
            raise ValueError(
                "KLIFS train and legacy validation+test references overlap"
            )
        references.append(
            ReferenceSet(
                label="legacy_val_test_sensitivity",
                description=METRIC_DEFINITIONS["reference_sets"][
                    "legacy_val_test_sensitivity"
                ],
                train_smiles=train,
                test_smiles=legacy_test,
                test_sources=("validation", "test"),
            )
        )
    return tuple(references)


def uniqueness_and_novelty(
    smiles: Sequence[str], train_smiles: Sequence[str]
) -> Dict[str, Any]:
    """Compute explicitly defined uniqueness and novelty values."""

    if not smiles:
        return {
            "generated_count": 0,
            "generated_unique_count": 0,
            "unique_at_1000": None,
            "unique_at_all": None,
            "novelty": None,
            "novel_count": 0,
        }
    if any(not str(value).strip() for value in smiles):
        raise ValueError("Generated canonical SMILES must be non-empty")
    canonical = [normalize_smiles_identity(value) for value in smiles]
    first = canonical[: min(1000, len(canonical))]
    unique = set(canonical)
    train = {normalize_smiles_identity(value) for value in train_smiles}
    novel_count = len(unique.difference(train))
    return {
        "generated_count": len(canonical),
        "generated_unique_count": len(unique),
        "unique_at_1000": len(set(first)) / len(first),
        "unique_at_all": len(unique) / len(canonical),
        "novelty": novel_count / len(unique),
        "novel_count": novel_count,
    }


class _MosesRequiredMetricEngine:
    """Compute the declared MOSES panel without an unrelated scaffold panel.

    MOSES 0.3.1's :func:`get_all_metrics` cannot disable its scaffold-test
    branch: ``test_scaffolds=None`` loads the packaged 176k-molecule MOSES
    scaffold benchmark. Passing the KLIFS test molecules as ``test_scaffolds``
    is scientifically incorrect and also duplicates every expensive
    reference/generated ChemNet calculation. This adapter uses MOSES' own
    metric primitives and caches their intermediate representations.
    """

    def __init__(self, *, n_jobs: int, device: str, batch_size: int) -> None:
        try:
            from moses.metrics.metrics import (
                FCDMetric,
                FragMetric,
                SNNMetric,
                fraction_passes_filters,
                internal_diversity,
                remove_invalid,
            )
            from moses.metrics.utils import get_mol
            from moses.utils import mapper
        except ImportError as exc:
            raise RuntimeError(
                "MOSES is required for aggregate chemical metrics. Install the "
                "hash-pinned evaluation environment or inject metric_fn."
            ) from exc

        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self._FCDMetric = FCDMetric
        self._FragMetric = FragMetric
        self._SNNMetric = SNNMetric
        self._fraction_passes_filters = fraction_passes_filters
        self._internal_diversity = internal_diversity
        self._remove_invalid = remove_invalid
        self._get_mol = get_mol
        self._mapper = mapper
        # Loading ChemNet once matters on CPU and ensures each unique
        # population is embedded exactly once.
        self._fcd = FCDMetric(
            n_jobs=n_jobs, device=device, batch_size=batch_size
        )
        self._snn = SNNMetric(n_jobs=n_jobs, device=device, batch_size=batch_size)
        self._frag = FragMetric(
            n_jobs=n_jobs, device=device, batch_size=batch_size
        )
        self._reference_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._generated_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def _molecules(self, smiles: Sequence[str]) -> List[Any]:
        molecules = self._mapper(self.n_jobs)(self._get_mol, list(smiles))
        if any(molecule is None for molecule in molecules):
            raise ValueError("MOSES could not parse a metric-panel SMILES")
        return molecules

    def _reference(self, smiles: Tuple[str, ...]) -> Dict[str, Any]:
        cached = self._reference_cache.get(smiles)
        if cached is not None:
            return cached
        molecules = self._molecules(smiles)
        cached = {
            "fcd": (
                self._fcd.precalc(list(smiles))
                if len(smiles) >= 2
                else None
            ),
            "snn": self._snn.precalc(molecules),
            "frag": self._frag.precalc(molecules),
        }
        self._reference_cache[smiles] = cached
        return cached

    def _generated(self, smiles: Tuple[str, ...]) -> Dict[str, Any]:
        cached = self._generated_cache.get(smiles)
        if cached is not None:
            return cached
        canonical = tuple(
            self._remove_invalid(
                list(smiles), canonize=True, n_jobs=self.n_jobs
            )
        )
        if not canonical:
            raise ValueError("MOSES metric panel contains no valid molecules")
        molecules = self._molecules(canonical)
        snn = self._snn.precalc(molecules)
        cached = {
            "valid": len(canonical) / len(smiles),
            "fcd": (
                self._fcd.precalc(list(canonical))
                if len(canonical) >= 2
                else None
            ),
            "snn": snn,
            "frag": self._frag.precalc(molecules),
            "intdiv": self._internal_diversity(
                molecules,
                self.n_jobs,
                device=self.device,
                gen_fps=snn["fps"],
            ),
            "intdiv2": self._internal_diversity(
                molecules,
                self.n_jobs,
                device=self.device,
                gen_fps=snn["fps"],
                p=2,
            ),
            "filters": self._fraction_passes_filters(
                molecules, self.n_jobs
            ),
        }
        self._generated_cache[smiles] = cached
        return cached

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        generated_smiles = tuple(str(value) for value in kwargs["gen"])
        reference_smiles = tuple(str(value) for value in kwargs["test"])
        if not generated_smiles or not reference_smiles:
            raise ValueError("MOSES generated and test populations must be non-empty")
        reference = self._reference(reference_smiles)
        generated = self._generated(generated_smiles)
        fcd = (
            None
            if reference["fcd"] is None or generated["fcd"] is None
            else self._fcd.metric(reference["fcd"], generated["fcd"])
        )
        return {
            "valid": generated["valid"],
            "FCD/Test": fcd,
            "SNN/Test": self._snn.metric(reference["snn"], generated["snn"]),
            "Frag/Test": self._frag.metric(reference["frag"], generated["frag"]),
            "IntDiv": generated["intdiv"],
            "IntDiv2": generated["intdiv2"],
            "Filters": generated["filters"],
        }


def _load_moses_metric_function(
    *, n_jobs: int, device: str, batch_size: int
) -> MetricFunction:
    return _MosesRequiredMetricEngine(
        n_jobs=n_jobs, device=device, batch_size=batch_size
    )


def _load_moses_descriptor_function() -> DescriptorFunction:
    try:
        from moses.metrics.utils import QED, SA, logP, weight
        from moses.utils import get_mol
    except ImportError as exc:
        raise RuntimeError(
            "MOSES is required for descriptor distributions. Install the "
            "hash-pinned evaluation environment or inject descriptor_fn."
        ) from exc

    def descriptors(smiles: str) -> Mapping[str, Any]:
        molecule = get_mol(smiles)
        if molecule is None:
            raise ValueError(f"MOSES could not parse reference SMILES: {smiles!r}")
        return {
            "logp": logP(molecule),
            "sa": SA(molecule),
            "qed": QED(molecule),
            "molecular_weight": weight(molecule),
        }

    return descriptors


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _json_safe(value: Any, label: str = "metric") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{label}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, label) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item(), label)
    if isinstance(value, float):
        return _finite_float(value, label)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} is not JSON-serializable: {type(value).__name__}")


def _raw_metric(
    raw: Mapping[str, Any],
    aliases: Sequence[str],
    label: str,
    *,
    allow_undefined: bool = False,
) -> Optional[float]:
    for alias in aliases:
        if alias in raw:
            if raw[alias] is None and allow_undefined:
                return None
            return _finite_float(raw[alias], label)
    raise ValueError(f"MOSES result is missing required metric {aliases[0]!r}")


def _descriptor_values(
    smiles: Sequence[str],
    descriptor_fn: DescriptorFunction,
    cache: Dict[str, Dict[str, float]],
) -> Dict[str, List[float]]:
    output = {key: [] for key in DESCRIPTOR_KEYS}
    for text in smiles:
        canonical = str(text).strip()
        if not canonical:
            raise ValueError("Cannot compute descriptors for an empty SMILES")
        if canonical not in cache:
            raw = descriptor_fn(canonical)
            missing = set(DESCRIPTOR_KEYS).difference(raw)
            if missing:
                raise ValueError(
                    f"Descriptor function omitted keys for {canonical!r}: "
                    f"{sorted(missing)}"
                )
            cache[canonical] = {
                key: _finite_float(raw[key], f"descriptor.{key}")
                for key in DESCRIPTOR_KEYS
            }
        for key in DESCRIPTOR_KEYS:
            output[key].append(cache[canonical][key])
    return output


def _metric_cohorts(
    survivors: Sequence[Mapping[str, Any]],
) -> Iterable[Tuple[str, str, Optional[int], List[Mapping[str, Any]]]]:
    yield "overall", "all", None, list(survivors)
    by_size: Dict[int, List[Mapping[str, Any]]] = {}
    for row in survivors:
        size = _strict_int(
            row["linker_size"], "linker_size", str(row["candidate_id"])
        )
        by_size.setdefault(size, []).append(row)
    for size in sorted(by_size):
        yield "linker_size", str(size), size, by_size[size]


def compute_metric_panels(
    survivors: Sequence[Mapping[str, Any]],
    references: Sequence[ReferenceSet],
    *,
    metric_fn: Optional[MetricFunction] = None,
    descriptor_fn: Optional[DescriptorFunction] = None,
    n_jobs: int = 1,
    device: str = "cpu",
    batch_size: int = 512,
    bootstrap_seed: int = 20260724,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute flattened metric panels and lossless raw MOSES records."""

    if not references:
        raise ValueError("At least one reference set is required")
    if n_jobs <= 0 or batch_size <= 0:
        raise ValueError("n_jobs and batch_size must be positive")
    metric_callable = metric_fn or _load_moses_metric_function(
        n_jobs=n_jobs, device=device, batch_size=batch_size
    )
    descriptor_callable = descriptor_fn or _load_moses_descriptor_function()
    descriptor_cache: Dict[str, Dict[str, float]] = {}
    flattened: List[Dict[str, Any]] = []
    raw_panels: List[Dict[str, Any]] = []

    for reference in references:
        if not reference.train_smiles or not reference.test_smiles:
            raise ValueError(f"Reference {reference.label!r} is empty")
        reference = ReferenceSet(reference.label, reference.description,
                                 tuple(sorted({normalize_smiles_identity(value) for value in reference.train_smiles})),
                                 tuple(sorted({normalize_smiles_identity(value) for value in reference.test_smiles})),
                                 reference.test_sources)
        reference_descriptors = _descriptor_values(
            reference.test_smiles, descriptor_callable, descriptor_cache
        )
        for scope, group_value, linker_size, cohort in _metric_cohorts(survivors):
            generated_smiles = [
                normalize_smiles_identity(row.get("canonical_smiles", "")) for row in cohort
            ]
            if any(not smiles for smiles in generated_smiles):
                raise ValueError(
                    f"Final-survivor cohort {scope}={group_value} contains empty SMILES"
                )
            derived = uniqueness_and_novelty(
                generated_smiles, reference.train_smiles
            )
            if not generated_smiles:
                raw: Dict[str, Any] = {}
                extracted = {key: None for key in MOSES_KEYS}
                wasserstein = {key: None for key in DESCRIPTOR_KEYS}
                fcd_undefined_reason = "empty_generated_cohort"
            else:
                raw_result = metric_callable(
                    gen=list(generated_smiles),
                    train=list(reference.train_smiles),
                    test=list(reference.test_smiles),
                    n_jobs=n_jobs,
                    device=device,
                    batch_size=batch_size,
                )
                if not isinstance(raw_result, Mapping):
                    raise TypeError("MOSES metric function must return a mapping")
                raw = _json_safe(raw_result, "raw_moses")
                extracted = {}
                for output_key, aliases in MOSES_KEYS.items():
                    extracted[output_key] = _raw_metric(
                        raw,
                        aliases,
                        output_key,
                        allow_undefined=(output_key == "fcd_test"),
                    )
                if extracted["fcd_test"] is None:
                    generated_small = len(generated_smiles) < 2
                    reference_small = len(reference.test_smiles) < 2
                    if not generated_small and not reference_small:
                        raise ValueError(
                            "fcd_test is undefined despite both populations "
                            "containing at least two molecules"
                        )
                    if generated_small and reference_small:
                        fcd_undefined_reason = (
                            "generated_and_reference_populations_have_fewer_than_"
                            "two_molecules"
                        )
                    elif generated_small:
                        fcd_undefined_reason = (
                            "generated_population_has_fewer_than_two_molecules"
                        )
                    else:
                        fcd_undefined_reason = (
                            "reference_population_has_fewer_than_two_molecules"
                        )
                else:
                    fcd_undefined_reason = None
                generated_descriptors = _descriptor_values(
                    generated_smiles, descriptor_callable, descriptor_cache
                )
                wasserstein = {
                    key: _finite_float(
                        wasserstein_distance(
                            generated_descriptors[key], reference_descriptors[key]
                        ),
                        f"wasserstein.{key}",
                    )
                    for key in DESCRIPTOR_KEYS
                }

            novelty_ci = bootstrap_proportion_ci(
                int(derived["novel_count"]),
                int(derived["generated_unique_count"]),
                seed=_stable_seed(
                    bootstrap_seed,
                    reference.label,
                    scope,
                    group_value,
                    "novelty",
                ),
                samples=bootstrap_samples,
                confidence=confidence,
            )
            record = {
                "reference_label": reference.label,
                "reference_description": reference.description,
                "scope": scope,
                "group_value": group_value,
                "linker_size": linker_size,
                "generated_count": derived["generated_count"],
                "generated_unique_count": derived["generated_unique_count"],
                "reference_count": len(reference.test_smiles),
                "train_count": len(reference.train_smiles),
                "novelty": derived["novelty"],
                "novelty_ci_low": novelty_ci[0],
                "novelty_ci_high": novelty_ci[1],
                "unique_at_1000": derived["unique_at_1000"],
                "unique_at_all": derived["unique_at_all"],
                **extracted,
                "fcd_test_defined": int(extracted["fcd_test"] is not None),
                "fcd_test_undefined_reason": fcd_undefined_reason,
                "w1_logp": wasserstein["logp"],
                "w1_sa": wasserstein["sa"],
                "w1_qed": wasserstein["qed"],
                "w1_molecular_weight": wasserstein["molecular_weight"],
            }
            flattened.append(record)
            raw_panels.append(
                {
                    "reference_label": reference.label,
                    "scope": scope,
                    "group_value": group_value,
                    "linker_size": linker_size,
                    "generated_count": len(generated_smiles),
                    "fcd_test_defined": bool(
                        extracted["fcd_test"] is not None
                    ),
                    "fcd_test_undefined_reason": fcd_undefined_reason,
                    "raw_moses": raw,
                }
            )
    return flattened, raw_panels


def compute_gt_metric_panels(
    gt_candidate_csv: Path,
    references: Sequence[ReferenceSet],
    *,
    metric_fn: Optional[MetricFunction] = None,
    descriptor_fn: Optional[DescriptorFunction] = None,
    n_jobs: int = 1,
    device: str = "cpu",
    batch_size: int = 512,
    bootstrap_seed: int = 20260724,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compute final-survivor metrics for GT without deduplicating GT rows."""

    _fields, rows = read_gt_evaluation_csv(Path(gt_candidate_csv))
    survivors = select_final_survivors(rows)
    flattened, raw_panels = compute_metric_panels(
        survivors,
        references,
        metric_fn=metric_fn,
        descriptor_fn=descriptor_fn,
        n_jobs=n_jobs,
        device=device,
        batch_size=batch_size,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    metadata = {
        "candidate_count": len(rows),
        "final_survivor_count": len(survivors),
        "sample_duplicates": "retained in exact GT table row order",
        "reference_test_duplicates": "removed by canonical_smiles",
    }
    return flattened, raw_panels, metadata


def annotate_survivors(
    survivors: Sequence[Mapping[str, Any]], train_smiles: Sequence[str]
) -> List[Dict[str, Any]]:
    """Add deterministic primary novelty and first-occurrence flags."""

    train = {normalize_smiles_identity(value) for value in train_smiles}
    seen = set()
    output: List[Dict[str, Any]] = []
    for row in survivors:
        candidate_id = str(row["candidate_id"])
        smiles = normalize_smiles_identity(row.get("canonical_smiles", ""))
        if not smiles:
            raise ValueError(f"Final survivor {candidate_id!r} has empty SMILES")
        record = with_normalized_identity(row, require_valid=True)
        for reserved in ("novel", "unique_first"):
            if reserved in record:
                raise ValueError(
                    f"Input candidate schema already contains reserved column {reserved!r}"
                )
        record["final_pass"] = 1
        record["novel"] = int(smiles not in train)
        record["unique_first"] = int(smiles not in seen)
        seen.add(smiles)
        output.append(record)
    return output


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields), lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue()


def _json_text(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _rdkit_version() -> str:
    """Return RDKit's runtime version even for conda-only installations."""

    try:
        from rdkit import rdBase

        version = str(rdBase.rdkitVersion).strip()
    except ImportError:
        version = ""
    if not version:
        version = (
            _package_version("rdkit")
            or _package_version("rdkit-pypi")
            or ""
        )
    if not version:
        raise RuntimeError("Unable to determine the runtime RDKit version")
    return version


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_aggregation(
    candidate_csv: Path,
    train_table: Path,
    test_table: Path,
    output_dir: Path,
    *,
    validation_table: Optional[Path] = None,
    gt_candidate_csv: Optional[Path] = None,
    smiles_column: str = "canonical_smiles",
    metric_fn: Optional[MetricFunction] = None,
    descriptor_fn: Optional[DescriptorFunction] = None,
    n_jobs: int = 1,
    device: str = "cpu",
    batch_size: int = 512,
    bootstrap_seed: int = 20260724,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run aggregation and atomically publish all tabular/JSON artifacts."""

    candidate_csv = Path(candidate_csv)
    train_table = Path(train_table)
    test_table = Path(test_table)
    output_dir = Path(output_dir)
    validation_table = (
        Path(validation_table) if validation_table is not None else None
    )
    gt_candidate_csv = (
        Path(gt_candidate_csv) if gt_candidate_csv is not None else None
    )
    input_fields, rows = read_candidate_csv(candidate_csv)
    references = build_reference_sets(
        train_table,
        test_table,
        validation_table=validation_table,
        smiles_column=smiles_column,
    )
    survivors = select_final_survivors(rows)
    survivors = annotate_survivors(survivors, references[0].train_smiles)
    funnel = build_funnel(
        rows,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    # Share the cached MOSES engine across generated and GT populations.
    aggregation_metric_fn = metric_fn or _load_moses_metric_function(
        n_jobs=n_jobs, device=device, batch_size=batch_size
    )
    metrics, raw_panels = compute_metric_panels(
        survivors,
        references,
        metric_fn=aggregation_metric_fn,
        descriptor_fn=descriptor_fn,
        n_jobs=n_jobs,
        device=device,
        batch_size=batch_size,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    gt_metrics: Optional[List[Dict[str, Any]]] = None
    gt_raw_panels: Optional[List[Dict[str, Any]]] = None
    gt_metadata: Optional[Dict[str, Any]] = None
    if gt_candidate_csv is not None:
        gt_metrics, gt_raw_panels, gt_metadata = compute_gt_metric_panels(
            gt_candidate_csv,
            references,
            metric_fn=aggregation_metric_fn,
            descriptor_fn=descriptor_fn,
            n_jobs=n_jobs,
            device=device,
            batch_size=batch_size,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )

    artifacts = {
        "final_survivors.csv",
        "funnel.csv",
        "metrics.csv",
        "metrics.json",
        "references.json",
        "provenance.json",
        "files.sha256.csv",
        "done.json",
    }
    if gt_candidate_csv is not None:
        artifacts.update({"gt_metrics.csv", "gt_metrics.json"})
    existing = sorted(name for name in artifacts if (output_dir / name).exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing aggregate artifacts: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Completion is always published last. Removing it first ensures an
        # interrupted replacement cannot masquerade as a complete result.
        (output_dir / "done.json").unlink(missing_ok=True)

    survivor_fields = tuple(input_fields) + tuple(
        field
        for field in ("final_pass", "novel", "unique_first")
        if field not in input_fields
    )
    reference_payload = {
        "schema_version": SCHEMA_VERSION,
        "smiles_column": smiles_column,
        "references": [
            {
                "label": reference.label,
                "description": reference.description,
                "test_sources": list(reference.test_sources),
                "train_count": len(reference.train_smiles),
                "test_count": len(reference.test_smiles),
            }
            for reference in references
        ],
    }
    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "definitions": METRIC_DEFINITIONS,
        "panels": raw_panels,
    }
    gt_metrics_payload = (
        {
            "schema_version": SCHEMA_VERSION,
            "population": "ground_truth_final_survivors",
            "definitions": METRIC_DEFINITIONS,
            "sampling": gt_metadata,
            "panels": gt_raw_panels,
        }
        if gt_candidate_csv is not None
        else None
    )
    input_paths = [candidate_csv, train_table, test_table]
    if validation_table is not None:
        input_paths.append(validation_table)
    if gt_candidate_csv is not None:
        input_paths.append(gt_candidate_csv)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "aggregation_source": _display_path(Path(__file__)),
        "aggregation_source_sha256": sha256_file(Path(__file__)),
        "inputs": {
            _display_path(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "candidate_count": len(rows),
        "final_survivor_count": len(survivors),
        "ground_truth": gt_metadata,
        "candidate_identity_normalization": {
            status: sum(row["identity_normalization_status"] == status for row in rows)
            for status in ("unchanged", "normalized", "empty", "invalid")
        },
        "bootstrap": {
            "method": "binomial_percentile",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "confidence": confidence,
        },
        "moses_call": {
            "implementation": "required_moses_primitives_cached",
            "scaffold_test": None,
            "n_jobs": n_jobs,
            "device": device,
            "batch_size": batch_size,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "rdkit": _rdkit_version(),
            "molsets": _package_version("molsets"),
            "fcd_torch": _package_version("fcd-torch"),
        },
    }

    publish = {
        "final_survivors.csv": _csv_text(survivors, survivor_fields),
        "funnel.csv": _csv_text(funnel, FUNNEL_FIELDS),
        "metrics.csv": _csv_text(metrics, METRIC_FIELDS),
        "metrics.json": _json_text(metrics_payload),
        "references.json": _json_text(reference_payload),
        "provenance.json": _json_text(provenance),
    }
    if gt_candidate_csv is not None:
        assert gt_metrics is not None
        assert gt_metrics_payload is not None
        publish["gt_metrics.csv"] = _csv_text(gt_metrics, METRIC_FIELDS)
        publish["gt_metrics.json"] = _json_text(gt_metrics_payload)
    for name in sorted(publish):
        _atomic_write_text(output_dir / name, publish[name])

    manifest_rows = []
    for name in sorted(publish):
        path = output_dir / name
        manifest_rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _atomic_write_text(
        output_dir / "files.sha256.csv",
        _csv_text(manifest_rows, ("path", "bytes", "sha256")),
    )
    manifest_path = output_dir / "files.sha256.csv"
    completed_files = {
        row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in manifest_rows
    }
    completed_files["files.sha256.csv"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    done = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "candidate_count": len(rows),
        "final_survivor_count": len(survivors),
        "funnel_records": len(funnel),
        "metric_records": len(metrics),
        "reference_labels": [reference.label for reference in references],
        "files": completed_files,
    }
    if gt_metadata is not None:
        done["gt_candidate_count"] = gt_metadata["candidate_count"]
        done["gt_final_survivor_count"] = gt_metadata["final_survivor_count"]
    _atomic_write_text(output_dir / "done.json", _json_text(done))
    return done


def _resolve_project_path(value: str, *, must_exist: bool = False) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"CLI paths must be project-root relative: {value}")
    root = PROJECT_ROOT.resolve()
    resolved = (root / raw).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"CLI path escapes the project root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the sequential kinase-gap candidate evaluation"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--validation-table")
    parser.add_argument(
        "--gt-candidates",
        "--gt-evaluation",
        dest="gt_candidates",
        help=(
            "Optional consolidated GT evaluation CSV; GT sample duplicates are "
            "retained while reference SMILES remain deduplicated"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    done = run_aggregation(
        candidate_csv=_resolve_project_path(args.candidates, must_exist=True),
        train_table=_resolve_project_path(args.train_table, must_exist=True),
        test_table=_resolve_project_path(args.test_table, must_exist=True),
        validation_table=(
            _resolve_project_path(args.validation_table, must_exist=True)
            if args.validation_table
            else None
        ),
        gt_candidate_csv=(
            _resolve_project_path(args.gt_candidates, must_exist=True)
            if args.gt_candidates
            else None
        ),
        output_dir=_resolve_project_path(args.output_dir),
        smiles_column=args.smiles_column,
        n_jobs=args.n_jobs,
        device=args.device,
        batch_size=args.batch_size,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        overwrite=args.overwrite,
    )
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
