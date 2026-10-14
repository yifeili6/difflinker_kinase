#!/usr/bin/env python3
"""Deterministic, project-root-scoped DiffLinker training entry point.

This wrapper deliberately distinguishes two operations that the historical
scripts conflated:

* ``--init-checkpoint`` loads model weights strictly and starts a new optimizer
  and epoch counter (fine-tuning).
* ``--resume-checkpoint`` delegates the complete checkpoint to Lightning and
  restores model, optimizer, scheduler, epoch, and global-step state.

All configured paths must be relative to the kinase-gap project root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "DiffLinker"
if not VENDOR_ROOT.is_dir():
    raise RuntimeError(f"Vendored DiffLinker source is missing: {VENDOR_ROOT}")
sys.path.insert(0, str(VENDOR_ROOT))

from src.const import GEOM_NUMBER_OF_ATOM_TYPES, NUMBER_OF_ATOM_TYPES  # noqa: E402
from src.datasets import collate, get_dataloader  # noqa: E402
from src.lightning import DDPM  # noqa: E402
from src.utils import disable_rdkit_logging  # noqa: E402


PATH_KEYS = ("data", "checkpoint_dir", "log_dir")
CHECKPOINT_METADATA_KEYS = (
    "in_node_nf",
    "n_dims",
    "context_node_nf",
    "hidden_nf",
    "activation",
    "n_layers",
    "attention",
    "tanh",
    "norm_constant",
    "inv_sublayers",
    "sin_embedding",
    "normalization_factor",
    "aggregation_method",
    "diffusion_steps",
    "diffusion_noise_schedule",
    "diffusion_noise_precision",
    "diffusion_loss_type",
    "normalize_factors",
    "include_charges",
    "model",
    "normalization",
    "data_augmentation",
    "inpainting",
    "anchors_context",
    "center_of_mass",
    "graph_type",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
OBJECTIVE_NAME = "loss/val"
OBJECTIVE_MODE = "min"


def project_path(value: str, *, must_exist: bool = False) -> Path:
    """Resolve a root-relative path and reject absolute or escaping paths."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are forbidden; use a project-relative path: {value}")

    root = PROJECT_ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes the project root: {value}") from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required project path does not exist: {value}")
    return resolved


def read_config(config_value: str) -> Tuple[Path, Dict[str, Any]]:
    config_path = project_path(config_value, must_exist=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must be a YAML mapping: {config_value}")

    missing = [key for key in PATH_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"Configuration is missing path keys: {', '.join(missing)}")
    return config_path, config


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Run ID may contain only letters, numbers, dot, underscore, and hyphen: "
            f"{run_id!r}"
        )
    return run_id


def is_geom_prefix(prefix: str) -> bool:
    return any(token in prefix for token in ("geom", "MOAD", "KLIF"))


def dataset_stem(prefix: str) -> str:
    """Return the on-disk stem used by MOADDataset for ``name.mode``."""
    if "." in prefix:
        stem, _ = prefix.split(".", maxsplit=1)
        return stem
    return prefix


def validate_dataset_files(config: Mapping[str, Any], data_dir: Path) -> None:
    for split_key in ("train_data_prefix", "val_data_prefix"):
        prefix = str(config[split_key])
        if "." not in prefix:
            raise ValueError(
                f"{split_key}={prefix!r} is not pocket-conditioned; expected a prefix such as "
                "KLIFS_train.full"
            )
        stem = dataset_stem(prefix)
        required = (
            data_dir / f"{stem}_table.csv",
            data_dir / f"{stem}_frag.sdf",
            data_dir / f"{stem}_link.sdf",
            data_dir / f"{stem}_pockets.pkl",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing files for {split_key}={prefix!r} under "
                f"{data_dir.relative_to(PROJECT_ROOT)}: {', '.join(missing)}"
            )


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_torch_determinism(*, warn_only: bool) -> None:
    """Enable deterministic kernels while allowing explicitly warned fallbacks.

    DiffLinker's EGNN aggregation uses CUDA ``scatter_add``.  PyTorch 1.11 has
    no deterministic CUDA implementation for that operation, so strict mode
    aborts before training begins.  Warn-only mode keeps deterministic
    implementations enabled wherever PyTorch provides them and makes every
    unavoidable nondeterministic fallback visible in the job log.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path, *, label: str) -> str:
    """Return a normalized project-relative path or reject an escaping path."""
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes the project root: {resolved}") from exc


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_early_stopping_callback(
    config: Mapping[str, Any],
) -> Optional[callbacks.EarlyStopping]:
    """Build the optional validation-loss early-stopping policy."""
    patience_value = config.get("early_stopping_patience")
    min_delta_value = config.get("early_stopping_min_delta")
    if patience_value is None and min_delta_value is None:
        return None
    if patience_value is None or min_delta_value is None:
        raise ValueError(
            "early_stopping_patience and early_stopping_min_delta must be set together"
        )
    if isinstance(patience_value, bool) or not isinstance(patience_value, int):
        raise TypeError("early_stopping_patience must be an integer")
    if patience_value < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    if isinstance(min_delta_value, bool):
        raise TypeError("early_stopping_min_delta must be a finite non-negative number")
    try:
        min_delta = float(min_delta_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "early_stopping_min_delta must be a finite non-negative number"
        ) from exc
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError(
            "early_stopping_min_delta must be a finite non-negative number"
        )
    return callbacks.EarlyStopping(
        monitor=OBJECTIVE_NAME,
        min_delta=min_delta,
        patience=patience_value,
        mode=OBJECTIVE_MODE,
        strict=True,
        check_finite=True,
        check_on_train_epoch_end=False,
    )


class KinaseDDPM(DDPM):
    """Thin training adapter around the vendored model.

    It adds deterministic DataLoaders and logs the actual validation objective
    with batch-size-aware epoch reduction.  The upstream epoch-end validation
    hook is intentionally bypassed because it uses an unweighted mean and runs
    expensive molecule sampling during fitting.
    """

    def __init__(self, *args: Any, data_seed: int, num_workers: int, **kwargs: Any) -> None:
        self.data_seed = int(data_seed)
        self.num_workers = int(num_workers)
        super().__init__(*args, **kwargs)

    def train_dataloader(self, collate_fn=collate):
        generator = torch.Generator()
        generator.manual_seed(self.data_seed)
        return get_dataloader(
            self.train_dataset,
            self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker if self.num_workers else None,
        )

    def val_dataloader(self, collate_fn=collate):
        return get_dataloader(
            self.val_dataset,
            self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker if self.num_workers else None,
        )

    def training_step(self, data, *args):
        metrics = super().training_step(data, *args)
        self.log(
            "loss/train",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(data["positions"].shape[0]),
        )
        return metrics

    def training_epoch_end(self, training_step_outputs):
        # ``loss/train`` is already reduced correctly by Lightning.
        return None

    def validation_step(self, data, *args):
        metrics = super().validation_step(data, *args)
        self.log(
            "loss/val",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(data["positions"].shape[0]),
        )
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        # Avoid the upstream unweighted aggregation and generation-time metrics.
        return None


def model_dimensions(config: Mapping[str, Any]) -> Tuple[int, int, bool]:
    prefix = str(config["train_data_prefix"])
    number_of_atoms = GEOM_NUMBER_OF_ATOM_TYPES if is_geom_prefix(prefix) else NUMBER_OF_ATOM_TYPES
    include_charges = bool(config.get("include_charges", False))
    in_node_nf = number_of_atoms + int(include_charges)
    anchors_context = not bool(config.get("remove_anchors_context", False))
    context_node_nf = 2 if anchors_context else 1
    if "." in prefix:
        context_node_nf += 1
    return in_node_nf, context_node_nf, anchors_context


def build_model(config: Mapping[str, Any], data_dir: Path, runtime_device: str) -> KinaseDDPM:
    in_node_nf, context_node_nf, anchors_context = model_dimensions(config)
    log_iterations = config.get("log_iterations")
    if log_iterations is not None:
        log_iterations = int(log_iterations)

    return KinaseDDPM(
        data_path=str(data_dir),
        train_data_prefix=str(config["train_data_prefix"]),
        val_data_prefix=str(config["val_data_prefix"]),
        in_node_nf=in_node_nf,
        n_dims=3,
        context_node_nf=context_node_nf,
        hidden_nf=int(config.get("nf", 128)),
        activation=str(config.get("activation", "silu")),
        n_layers=int(config.get("n_layers", 6)),
        attention=bool(config.get("attention", False)),
        tanh=bool(config.get("tanh", False)),
        norm_constant=float(config.get("norm_constant", 1e-6)),
        inv_sublayers=int(config.get("inv_sublayers", 2)),
        sin_embedding=bool(config.get("sin_embedding", False)),
        normalization_factor=float(config.get("normalization_factor", 100)),
        aggregation_method=str(config.get("aggregation_method", "sum")),
        diffusion_steps=int(config.get("diffusion_steps", 500)),
        diffusion_noise_schedule=str(config.get("diffusion_noise_schedule", "polynomial_2")),
        diffusion_noise_precision=float(config.get("diffusion_noise_precision", 1e-5)),
        diffusion_loss_type=str(config.get("diffusion_loss_type", "l2")),
        normalize_factors=list(config.get("normalize_factors", [1, 4, 10])),
        include_charges=bool(config.get("include_charges", False)),
        lr=float(config.get("lr", 2e-4)),
        batch_size=int(config.get("batch_size", 32)),
        torch_device=runtime_device,
        model=str(config.get("model", "egnn_dynamics")),
        test_epochs=int(config.get("test_epochs", 20)),
        n_stability_samples=int(config.get("n_stability_samples", 5)),
        normalization=str(config.get("normalization", "batch_norm")),
        log_iterations=log_iterations,
        samples_dir=None,
        data_augmentation=bool(config.get("data_augmentation", False)),
        center_of_mass=str(config.get("center_of_mass", "anchors")),
        inpainting=bool(config.get("inpainting", False)),
        anchors_context=anchors_context,
        graph_type=config.get("graph_type"),
        data_seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
    )


def checkpoint_hparams(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    hparams = payload.get("hyper_parameters", {})
    if isinstance(hparams, Mapping):
        return hparams
    if hasattr(hparams, "__dict__"):
        return vars(hparams)
    return {}


def validate_checkpoint_metadata(
    payload: Mapping[str, Any], config: Mapping[str, Any], model: KinaseDDPM
) -> None:
    hparams = checkpoint_hparams(payload)
    expected = {
        key: getattr(model.hparams, key, config.get(key))
        for key in CHECKPOINT_METADATA_KEYS
    }

    mismatches = []
    for key in CHECKPOINT_METADATA_KEYS:
        if key not in hparams or hparams[key] is None:
            continue
        if hparams[key] != expected[key]:
            mismatches.append(f"{key}: checkpoint={hparams[key]!r}, config={expected[key]!r}")
    if mismatches:
        raise ValueError("Checkpoint metadata is incompatible:\n  " + "\n  ".join(mismatches))


def inspect_resume_checkpoint(
    model: KinaseDDPM, config: Mapping[str, Any], checkpoint_path: Path
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise TypeError("A full resume requires a Lightning checkpoint with a state_dict")
    for required in ("optimizer_states", "epoch", "global_step"):
        if required not in payload:
            raise ValueError(f"Resume checkpoint is missing full-state field: {required}")
    validate_checkpoint_metadata(payload, config, model)


def resolve_checkpoint_modes(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Tuple[Optional[Path], Optional[Path]]:
    if args.init_checkpoint is not None or args.resume_checkpoint is not None:
        init_value = args.init_checkpoint
        resume_value = args.resume_checkpoint
    else:
        init_value = config.get("init_checkpoint")
        resume_value = config.get("resume_checkpoint")

    if init_value and resume_value:
        raise ValueError("Choose exactly one of init_checkpoint and resume_checkpoint")
    if not init_value and not resume_value and not bool(config.get("allow_from_scratch", False)):
        raise ValueError(
            "No checkpoint was selected. Set init_checkpoint for fine-tuning, "
            "resume_checkpoint for a full resume, or explicitly set allow_from_scratch: true."
        )

    init_path = project_path(str(init_value), must_exist=True) if init_value else None
    resume_path = project_path(str(resume_value), must_exist=True) if resume_value else None
    return init_path, resume_path


def runtime_device(config: Mapping[str, Any]) -> Tuple[str, str, int]:
    accelerator = str(config.get("accelerator", "gpu"))
    devices = int(config.get("devices", 1))
    if devices != 1:
        raise ValueError("This prepared workflow currently supports exactly one device per job")
    if accelerator not in {"gpu", "cpu"}:
        raise ValueError("accelerator must be 'gpu' or 'cpu'")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU training was requested, but torch.cuda.is_available() is false")
        return accelerator, "cuda:0", devices
    return accelerator, "cpu", devices


def write_run_metadata(
    run_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    run_id: str,
    init_checkpoint: Optional[Path],
    resume_checkpoint: Optional[Path],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.snapshot.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)

    upstream_file = VENDOR_ROOT / "UPSTREAM_COMMIT"
    upstream_commit = upstream_file.read_text(encoding="utf-8").strip() if upstream_file.exists() else None
    data_dir = project_path(str(config["data"]), must_exist=True)
    fingerprint_paths = {
        "source_manifest": PROJECT_ROOT / "artifacts/source.sha256",
        "environment_manifest": PROJECT_ROOT / "artifacts/environment.sha256",
        "dataset_report": data_dir / "dataset_report.json",
        "dataset_files_manifest": data_dir / "files.sha256.csv",
        "moad_overlap_report": PROJECT_ROOT
        / "data/manifests/klifs_v1/moad_pdb_overlap.json",
        "checkpoint_manifest": PROJECT_ROOT
        / "models/pretrained/checkpoint_manifest.json",
    }
    missing_fingerprints = [
        str(path.relative_to(PROJECT_ROOT))
        for path in fingerprint_paths.values()
        if not path.is_file()
    ]
    if missing_fingerprints:
        raise FileNotFoundError(
            f"Run provenance inputs are missing: {', '.join(missing_fingerprints)}"
        )
    artifact_fingerprints = {
        label: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
        for label, path in fingerprint_paths.items()
    }
    cuda_preflight = PROJECT_ROOT / "artifacts/preflight.cuda.json"
    if os.environ.get("SLURM_JOB_ID") and not cuda_preflight.is_file():
        raise FileNotFoundError("Slurm training requires artifacts/preflight.cuda.json")
    if cuda_preflight.is_file():
        shutil.copy2(cuda_preflight, run_dir / "preflight.cuda.snapshot.json")
        artifact_fingerprints["cuda_preflight"] = {
            "path": "preflight.cuda.snapshot.json",
            "sha256": sha256_file(cuda_preflight),
        }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "init_checkpoint": (
            str(init_checkpoint.relative_to(PROJECT_ROOT)) if init_checkpoint is not None else None
        ),
        "init_checkpoint_sha256": (
            sha256_file(init_checkpoint) if init_checkpoint is not None else None
        ),
        "resume_checkpoint": (
            str(resume_checkpoint.relative_to(PROJECT_ROOT)) if resume_checkpoint is not None else None
        ),
        "resume_checkpoint_sha256": (
            sha256_file(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "difflinker_upstream_commit": upstream_commit,
        "artifact_fingerprints": artifact_fingerprints,
        "python": sys.version,
        "torch": torch.__version__,
        "pytorch_lightning": __import__("pytorch_lightning").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "host": __import__("platform").node(),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_training_result(
    *,
    run_checkpoint_dir: Path,
    run_log_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    run_name: str,
    run_id: str,
    checkpoint_callback: callbacks.ModelCheckpoint,
    trainer: Trainer,
    early_stopping_callback: Optional[callbacks.EarlyStopping],
) -> Path:
    """Validate the winning checkpoint and atomically publish a completed run."""
    score_value = checkpoint_callback.best_model_score
    if score_value is None:
        raise RuntimeError(f"Training completed without a {OBJECTIVE_NAME} score")
    if isinstance(score_value, torch.Tensor):
        if score_value.numel() != 1:
            raise ValueError("Best validation objective is not scalar")
        best_objective = float(score_value.detach().cpu().item())
    else:
        best_objective = float(score_value)
    if not math.isfinite(best_objective):
        raise ValueError("Best validation objective is not finite")

    path_value = checkpoint_callback.best_model_path
    if not path_value:
        raise RuntimeError("Training completed without a best checkpoint")
    best_model_path = Path(path_value)
    if not best_model_path.is_absolute():
        best_model_path = PROJECT_ROOT / best_model_path
    best_model_path = best_model_path.resolve()
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {best_model_path}")
    try:
        best_model_path.relative_to(run_checkpoint_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Best checkpoint is outside its run directory: {best_model_path}"
        ) from exc

    checkpoint_payload = torch.load(best_model_path, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("Best checkpoint payload is not a mapping")
    for key in ("epoch", "global_step"):
        if key not in checkpoint_payload:
            raise ValueError(f"Best checkpoint is missing {key}")
    best_epoch = int(checkpoint_payload["epoch"])
    best_global_step = int(checkpoint_payload["global_step"])
    if best_epoch < 0 or best_global_step < 0:
        raise ValueError("Best checkpoint has a negative epoch or global step")

    run_metadata_path = run_log_dir / "run_metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata is missing: {run_metadata_path}")

    early_stopping = {
        "enabled": early_stopping_callback is not None,
        "patience": (
            int(early_stopping_callback.patience)
            if early_stopping_callback is not None
            else None
        ),
        "min_delta": (
            float(config["early_stopping_min_delta"])
            if early_stopping_callback is not None
            else None
        ),
        "stopped_epoch": (
            int(early_stopping_callback.stopped_epoch)
            if early_stopping_callback is not None
            and early_stopping_callback.stopped_epoch > 0
            else None
        ),
    }
    hpo_metadata = config.get("_hpo")
    if hpo_metadata is not None and not isinstance(hpo_metadata, Mapping):
        raise TypeError("_hpo metadata must be a mapping")

    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "run_id": run_id,
        "objective": OBJECTIVE_NAME,
        "objective_mode": OBJECTIVE_MODE,
        "best_objective": best_objective,
        "best_epoch": best_epoch,
        # This is the step stored with the validation-ranked checkpoint. The
        # trainer's terminal step is recorded separately below.
        "global_step": best_global_step,
        "best_model_path": project_relative(
            best_model_path, label="best checkpoint"
        ),
        "best_model_sha256": sha256_file(best_model_path),
        "config": project_relative(config_path, label="training config"),
        "config_sha256": sha256_file(config_path),
        "run_metadata": project_relative(
            run_metadata_path, label="run metadata"
        ),
        "run_metadata_sha256": sha256_file(run_metadata_path),
        "trainer_current_epoch": int(trainer.current_epoch),
        "trainer_global_step": int(trainer.global_step),
        "early_stopping": early_stopping,
        "hpo": dict(hpo_metadata) if hpo_metadata is not None else None,
    }
    output_path = run_checkpoint_dir / "training_result.json"
    atomic_write_json(output_path, result)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_anchors.yml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--init-checkpoint")
    checkpoint_group.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--run-id",
        help="Stable run label; defaults to SLURM_JOB_ID, then 'manual' outside Slurm",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, model construction, and checkpoint compatibility without fitting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path, config = read_config(args.config)
    data_dir = project_path(str(config["data"]), must_exist=True)
    checkpoint_root = project_path(str(config["checkpoint_dir"]))
    log_root = project_path(str(config["log_dir"]))
    validate_dataset_files(config, data_dir)

    run_id = validate_run_id(args.run_id or os.environ.get("SLURM_JOB_ID", "manual"))
    run_name = validate_run_id(str(config.get("run_name", "klifs_finetune_anchors")))
    init_checkpoint, resume_checkpoint = resolve_checkpoint_modes(args, config)
    early_stopping_callback = build_early_stopping_callback(config)

    seed = int(config.get("seed", 42))
    deterministic_warn_only = config.get("deterministic_warn_only")
    if not isinstance(deterministic_warn_only, bool):
        raise TypeError("deterministic_warn_only must be a YAML boolean")
    seed_everything(seed, workers=True)
    configure_torch_determinism(warn_only=deterministic_warn_only)
    disable_rdkit_logging()

    accelerator, torch_device, devices = runtime_device(config)
    model = build_model(config, data_dir, torch_device)
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"Checkpoint is not a mapping: {init_checkpoint}")
        validate_checkpoint_metadata(payload, config, model)
        state_dict = payload.get("state_dict", payload)
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Checkpoint has no state_dict mapping: {init_checkpoint}")
        # Strict mode is essential: the historical no-anchor checkpoint has a
        # 12-wide embedding, whereas the anchor-conditioned model requires 13.
        model.load_state_dict(state_dict, strict=True)
    elif resume_checkpoint is not None:
        inspect_resume_checkpoint(model, config, resume_checkpoint)

    if args.preflight_only:
        print("Preflight passed; no training was started.")
        return 0

    run_log_dir = log_root / run_name / run_id
    run_checkpoint_dir = checkpoint_root / run_name / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        run_log_dir,
        config_path,
        config,
        run_id,
        init_checkpoint,
        resume_checkpoint,
    )

    csv_logger = loggers.CSVLogger(
        save_dir=str(log_root),
        name=run_name,
        version=run_id,
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=str(run_checkpoint_dir),
        filename="epoch{epoch:04d}-step{step:08d}",
        auto_insert_metric_name=False,
        monitor="loss/val",
        mode="min",
        save_top_k=int(config.get("save_top_k", 3)),
        save_last=True,
        every_n_epochs=1,
    )
    learning_rate_callback = callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer_callbacks = [checkpoint_callback, learning_rate_callback]
    if early_stopping_callback is not None:
        trainer_callbacks.append(early_stopping_callback)

    trainer = Trainer(
        max_epochs=int(config.get("max_epochs", 1000)),
        accelerator=accelerator,
        devices=devices,
        precision=int(config.get("precision", 32)),
        deterministic=True,
        benchmark=False,
        logger=csv_logger,
        callbacks=trainer_callbacks,
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 2)),
        enable_progress_bar=bool(config.get("enable_progress_bar", True)),
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        gradient_clip_val=float(config.get("gradient_clip_val", 0.0)),
        accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
    )
    # Lightning 1.6 resets deterministic algorithms to strict mode while the
    # Trainer is constructed.  Re-apply the recorded project policy before any
    # validation or training step executes.
    configure_torch_determinism(warn_only=deterministic_warn_only)
    trainer.fit(
        model=model,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )
    training_result_path = write_training_result(
        run_checkpoint_dir=run_checkpoint_dir,
        run_log_dir=run_log_dir,
        config_path=config_path,
        config=config,
        run_name=run_name,
        run_id=run_id,
        checkpoint_callback=checkpoint_callback,
        trainer=trainer,
        early_stopping_callback=early_stopping_callback,
    )
    print(
        "Training result written: "
        f"{project_relative(training_result_path, label='training result')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
