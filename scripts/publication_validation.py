"""Portable checksum and completion validation for publication artifacts."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}


def project_path(root, name):
    name = str(name)
    relative = PurePosixPath(name)
    if not name or not relative.parts or relative.is_absolute() or str(relative) != name or ".." in relative.parts:
        raise ValueError(f"Expected a normalized relative path: {name!r}")
    result = Path(root).joinpath(*relative.parts)
    for item in [result, *result.parents]:
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"Symlink in publication input: {item}")
    return result


def verify_file(path, expected):
    if not expected.get("sha256"):
        raise ValueError(f"Missing expected checksum: {path}")
    observed = fingerprint(path)
    if observed["sha256"] != expected["sha256"] or ("bytes" in expected and observed["bytes"] != expected["bytes"]):
        raise ValueError(f"Checksum or size mismatch: {path}")
    return observed


def done(root, name, expected=None):
    path = project_path(root, name)
    record = load(path)
    if record.get("status") != "complete":
        raise ValueError(f"Incomplete stage: {name}")
    for key, value in (expected or {}).items():
        if record.get(key) != value:
            raise ValueError(f"Unexpected {key} in {name}: {record.get(key)!r}")
    outputs = record.get("outputs", record.get("files"))
    if not outputs:
        raise ValueError(f"Completion record has no output checksums: {name}")
    for filename, information in outputs.items():
        verify_file(project_path(path.parent, filename), information)
    return record


def shards(root, provenance, count):
    records = provenance["shards"]
    if len(records) != count or sorted(r["shard_index"] for r in records) != list(range(count)):
        raise ValueError("Shard completion indexes differ from the declared count")
    if len({r["path"] for r in records}) != count:
        raise ValueError("Duplicate shard path")
    for record in records:
        name = record["path"] + "/done.json"
        verify_file(project_path(root, name), {"sha256": record["done_sha256"]})
        done(root, name, {"shard_index": record["shard_index"]})


def files(root, directory):
    result = {}
    base = project_path(root, directory)
    if not base.is_dir():
        raise ValueError(f"Missing publication directory: {directory}")
    for path in sorted(base.rglob("*")):
        relative = str(path.relative_to(root))
        project_path(root, relative)
        if path.is_file():
            result[relative] = fingerprint(path)
    if not result:
        raise ValueError(f"Empty publication directory: {directory}")
    return result


def descriptions(root, record):
    """Verify every nested {path, sha256, bytes?} input description."""
    if isinstance(record, dict):
        if 'path' in record and 'sha256' in record:
            verify_file(project_path(root, record['path']), record)
        else:
            for value in record.values():
                descriptions(root, value)
    elif isinstance(record, list):
        for value in record:
            descriptions(root, value)


def execution_seals(root):
    names = ('source.sha256', 'generation-source.sha256', 'evaluation-source.sha256',
             'metrics-source.sha256', 'reporting-source.sha256', 'environment.sha256')
    verified, seals = {}, {}
    for name in names:
        path = project_path(root, 'artifacts/' + name)
        seals[name] = fingerprint(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            digest, filename = line.split(None, 1)
            filename = filename.lstrip('*')
            if filename in verified and verified[filename] != digest:
                raise ValueError(f'Conflicting execution checksums: {filename}')
            if filename not in verified:
                verify_file(project_path(root, filename), {'sha256': digest})
                verified[filename] = digest
    return {'seals': seals, 'files': verified, 'verified_file_count': len(verified)}
