#!/usr/bin/env python3
"""Copy archived inputs into a new, independently writable experiment workspace."""
import argparse
import csv
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREES = ("src", "vendor", "configs", "environment", "scripts", "slurm",
         "data/manifests/klifs_v1", "data/raw", "models/pretrained")
OPTIONAL_TREES = ("docs", "licenses", "tests")
SEALS = ("source.sha256", "generation-source.sha256", "evaluation-source.sha256",
         "metrics-source.sha256", "reporting-source.sha256", "environment.sha256")
ARCHIVES = ("training-source.tar.gz", "generation-source.tar.gz", "evaluation-source.tar.gz",
            "metrics-source.tar.gz", "reporting-source.tar.gz")
FIXED = (".kinase-revision", "provenance/prepared_graph_validation.json",
         "provenance/curated_source_files.jsonl", "data/processed/klifs_v1/files.sha256.csv",
         "data/processed/klifs_v1/PREPARED.json")
IGNORED = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}


def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a contained relative path: {relative}")
    resolved = (root / path).resolve(strict=True)
    if not is_within(resolved, root):
        raise ValueError(f"Input points outside the source workspace: {relative}")
    if not resolved.is_file():
        raise ValueError(f"Input must be a regular file: {relative}")
    return resolved


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_files(source):
    paths = set(FIXED)
    for name in SEALS:
        paths.add(f"artifacts/{name}")
    for name in ARCHIVES:
        if (source / "artifacts" / name).is_file():
            paths.add(f"artifacts/{name}")
    for name in ("README.md",):
        if (source / name).is_file():
            paths.add(name)
    for relative in TREES + OPTIONAL_TREES:
        directory = source / relative
        if not directory.is_dir():
            if relative in OPTIONAL_TREES:
                continue
            raise FileNotFoundError(f"Required input directory: {relative}")
        if directory.is_symlink():
            raise ValueError(f"Input directories must be ordinary directories: {relative}")
        for parent, directories, names in os.walk(directory, followlinks=False):
            directories[:] = sorted(name for name in directories if name not in IGNORED)
            for name in directories:
                if (Path(parent) / name).is_symlink():
                    raise ValueError(f"Input directory is a symbolic link: {Path(parent) / name}")
            for name in sorted(names):
                if name in IGNORED or name.endswith((".pyc", ".pyo")):
                    continue
                path = (Path(parent) / name).relative_to(source).as_posix()
                if path == "scripts/advance_revision.py":
                    continue
                paths.add(path)
    manifest = source / "data/processed/klifs_v1/files.sha256.csv"
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            paths.add((Path("data/processed/klifs_v1") / row["path"]).as_posix())
    for relative in paths:
        contained(source, relative)
    return sorted(paths)


def verify_copies(root, records):
    hashes = {row["path"]: row["sha256"] for row in records}
    for name in SEALS:
        for line in (root / "artifacts" / name).read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(None, 1)
            relative = relative.lstrip("*")
            if hashes.get(relative) != expected:
                raise ValueError(f"Source seal differs for {relative} ({name})")
    with (root / "data/processed/klifs_v1/files.sha256.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            relative = (Path("data/processed/klifs_v1") / row["path"]).as_posix()
            if hashes.get(relative) != row["sha256"]:
                raise ValueError(f"Prepared data hash differs for {relative}")
            if (root / relative).stat().st_size != int(row["bytes"]):
                raise ValueError(f"Prepared data size differs for {relative}")
    checkpoint_manifest = json.loads((root / "models/pretrained/checkpoint_manifest.json").read_text())
    for entry in checkpoint_manifest["files"]:
        relative = f"models/pretrained/{entry['filename']}"
        if hashes.get(relative) != entry["sha256"]:
            raise ValueError(f"Pretrained checkpoint hash differs for {relative}")


ACTIVATE = '''# Source this file after setting KINASE_RUNTIME_ENV to the environment prefix.
export KG_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${KINASE_RUNTIME_ENV:?Set KINASE_RUNTIME_ENV to the recorded environment prefix}"
export KG_PYTHON="$KINASE_RUNTIME_ENV/bin/python"
export PATH="$KINASE_RUNTIME_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$KINASE_RUNTIME_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export PYTHONPATH="$KG_PROJECT_ROOT/src:$KG_PROJECT_ROOT/vendor/DiffLinker"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$KG_PROJECT_ROOT/cache/matplotlib-local"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
mkdir -p "$MPLCONFIGDIR"
cd "$KG_PROJECT_ROOT"
'''


def create_workspace(source, destination, *, dry_run=False):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Choose a destination that does not exist: {destination}")
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if is_within(destination, source):
        raise ValueError("Choose a destination outside the source archive")
    paths = plan_files(source)
    total_bytes = sum(contained(source, path).stat().st_size for path in paths)
    plan = {"source": str(source), "destination": str(destination),
            "input_files": len(paths), "input_bytes": total_bytes}
    if dry_run:
        return dict(plan, status="planned")
    destination.mkdir(mode=0o755)
    records = []
    for relative in paths:
        original = contained(source, relative)
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with original.open("rb") as source_handle, copied.open("xb") as destination_handle:
            for block in iter(lambda: source_handle.read(4 * 1024 * 1024), b""):
                destination_handle.write(block)
                digest.update(block)
        copied.chmod(stat.S_IMODE(original.stat().st_mode) & ~0o222)
        records.append({"path": relative, "bytes": copied.stat().st_size, "sha256": digest.hexdigest()})
    verify_copies(destination, records)
    for relative in ("outputs", "logs", "cache", "models/runs", "reproduction"):
        (destination / relative).mkdir(parents=True, exist_ok=True)
    with (destination / "reproduction/activate.sh").open("x") as handle:
        handle.write(ACTIVATE)
    with (destination / "reproduction/inputs.sha256.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    result = dict(plan, schema_version=1, status="ready",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  study_id="anchor-corrected-20261001-v1",
                  copied_input_policy="Independent read-only files; fresh output and runtime directories",
                  input_manifest_sha256=file_hash(destination / "reproduction/inputs.sha256.csv"))
    with (destination / "reproduction/workspace.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_workspace(args.source, args.destination, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
