#!/usr/bin/env python3
"""Create a fresh experiment workspace with the recorded evaluation sources."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re

BUILDER = 'scripts/create_reproduction_workspace.py'
BUILDER_SHA = '27aae51c69680e2f86513a76ba4f745fdfefa6575a2f49668f07e4281085d884'
SEAL = 'artifacts/authoritative-evaluation-source.sha256'
SEAL_SHA = 'f1f449c5c596dc2f8cd4a60d1508dd655a9e3fad4e5e87f41b52fc84f71ddbcd'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked(source, name, expected):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
        raise ValueError('Source member must be a contained relative path')
    path = source / name
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Source member traverses a symbolic link: ' + name)
    if not path.is_file() or digest(path) != expected:
        raise ValueError('Source member checksum differs: ' + name)
    return path


def create(source, destination, *, dry_run=False):
    source = Path(source).absolute()
    builder = checked(source, BUILDER, BUILDER_SHA)
    seal = checked(source, SEAL, SEAL_SHA)
    members = {}
    for line in seal.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
        if match is None or match[2] in members:
            raise ValueError('Malformed or duplicate evaluation source checksum')
        checked(source, match[2], match[1])
        members[match[2]] = match[1]
    if not members:
        raise ValueError('Evaluation source manifest is empty')
    spec = importlib.util.spec_from_file_location('recorded_workspace_builder', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEALS = (*module.SEALS, Path(SEAL).name)
    module.FIXED = (*module.FIXED, *sorted(members))
    result = module.create_workspace(source, destination, dry_run=dry_run)
    result['authoritative_evaluation_sources'] = {
        'seal': SEAL, 'sha256': SEAL_SHA, 'members': len(members),
        'included_in_input_manifest': not dry_run,
    }
    if not dry_run:
        target = Path(destination).absolute()
        checked(target, SEAL, SEAL_SHA)
        for name, expected in members.items():
            checked(target, name, expected)
        receipt = {
            'schema_version': 1, 'status': 'verified',
            'source': str(source), 'workspace': str(target),
            'builder': {'path': BUILDER, 'sha256': BUILDER_SHA},
            'seal': {'path': SEAL, 'sha256': SEAL_SHA},
            'members': members,
            'input_manifest_sha256': digest(target / 'reproduction/inputs.sha256.csv'),
        }
        with (target / 'reproduction/authoritative_sources.json').open('x') as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.destination, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()
