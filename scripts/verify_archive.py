#!/usr/bin/env python3
"""Verify a published archive against provenance/artifact_manifest.json."""
import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

MANIFEST = "provenance/artifact_manifest.json"


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError(f"Expected a normalized relative archive path: {value!r}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(archive):
    root = Path(archive).resolve(strict=True)
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink() or root not in manifest_path.resolve().parents:
        raise ValueError("Artifact manifest must be a regular archive file")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise ValueError("Artifact manifest files must be a list")
    declared, observed, hashes, failures = set(), set(), {}, []
    for row in rows:
        relative = safe_relative(row["path"])
        if relative == MANIFEST or relative in declared:
            raise ValueError(f"Duplicate or self-referential artifact entry: {relative}")
        declared.add(relative)
        if type(row["bytes"]) is not int or row["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid artifact size or checksum: {relative}")
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                failures.append(relative + ": symbolic links are not archive files")
                continue
            mode = path.stat().st_mode
            if stat.S_ISREG(mode):
                if relative != MANIFEST:
                    observed.add(relative)
            elif not stat.S_ISDIR(mode):
                failures.append(relative + ": unsupported file type")
    for relative in sorted(declared - observed):
        failures.append(relative + ": declared file is missing")
    for relative in sorted(observed - declared):
        failures.append(relative + ": file is absent from the artifact manifest")
    if failures:
        raise ValueError("\n".join(failures))
    for row in rows:
        path = root / row["path"]
        hashes[row["path"]] = sha256(path)
        if hashes[row["path"]] != row["sha256"] or path.stat().st_size != row["bytes"]:
            failures.append(row["path"] + ": checksum or size mismatch")
    run_relative = "provenance/run_manifest.json"
    if run_relative not in declared:
        failures.append("Run manifest is absent from the archive")
    else:
        run = json.loads((root / run_relative).read_text())
        if run.get("status") != "complete":
            failures.append("Run manifest does not record a completed run")
        checkpoint = safe_relative(run["model"]["checkpoint"])
        expected = run["model"]["sha256"]
        if checkpoint not in declared:
            failures.append("Selected checkpoint is absent from the artifact manifest")
        elif hashes[checkpoint] != expected:
            failures.append("Selected checkpoint differs from model.sha256")
    if failures:
        raise ValueError("\n".join(failures))
    return {"files": len(rows), "bytes": sum(row["bytes"] for row in rows), "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.archive)
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(f"Verified {result['files']} files ({result['bytes']} bytes)")


if __name__ == "__main__":
    main()
