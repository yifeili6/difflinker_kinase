#!/usr/bin/env python3
"""Verify archive and package integrity, then create a fresh portable study workspace."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import stat
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVES = {
    'source.sha256': 'training-source.tar.gz',
    'generation-source.sha256': 'generation-source.tar.gz',
    'evaluation-source.sha256': 'evaluation-source.tar.gz',
    'metrics-source.sha256': 'metrics-source.tar.gz',
    'reporting-source.sha256': 'reporting-source.tar.gz',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def records():
    return json.loads((ROOT / 'source_manifest.json').read_text())['files']


def verify(source):
    rows = records()
    for row in rows:
        name = Path(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Manifest path must be relative: ' + str(name))
        for root, prefix in ((ROOT, 'packaged'), (source, 'archive_source')):
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(str(path))
            if path.stat().st_size != row[prefix + '_bytes'] or digest(path) != row[prefix + '_sha256']:
                raise ValueError(prefix + ' integrity differs: ' + str(path))
    adapted = sum(row['archive_source_sha256'] != row['packaged_sha256'] for row in rows)
    return {'verified_archive_source_files': len(rows),
            'verified_packaged_files': len(rows),
            'unchanged_files': len(rows) - adapted, 'portability_adaptations': adapted}


def write_owned_copy(path, data):
    """Replace a copy made in this call's new workspace, preserving its mode."""
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a newly copied regular file: ' + str(path))
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        path.write_bytes(data)
    finally:
        path.chmod(mode)


def install_portable_sources(target):
    """Adapt copied execution files; preserve archive records in a separate receipt."""
    evidence = target / 'reproduction/portable_source_adaptation'
    evidence.mkdir(exist_ok=False)
    changes = []
    # Historical continuation source is kept in the code release for inspection.
    # Only files selected by the archive's fresh-study builder are installed here.
    for row in records():
        if row['archive_source_sha256'] == row['packaged_sha256']:
            continue
        path = target / row['path']
        if not path.exists():
            continue
        if digest(path) != row['archive_source_sha256']:
            raise ValueError('Fresh copy differs from the archive: ' + row['path'])
        saved = evidence / 'archive_source' / row['path']
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(saved))
        write_owned_copy(path, (ROOT / row['path']).read_bytes())
        changes.append(row)

    seals = []
    for path in sorted((target / 'artifacts').glob('*.sha256')):
        entries = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(None, 1)
            name = name.lstrip('*')
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid source-seal path: ' + name)
            entries.append((digest(target / relative), name))
        content = ''.join(value + '  ' + name + '\n' for value, name in entries).encode()
        if content == path.read_bytes():
            continue
        saved = evidence / 'archive_source_seals' / path.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(saved))
        old = digest(path)
        write_owned_copy(path, content)
        seals.append({'path': str(path.relative_to(target)),
                      'archive_sha256': old, 'portable_sha256': digest(path)})
        archive_name = SOURCE_ARCHIVES.get(path.name)
        if archive_name and (target / 'artifacts' / archive_name).is_file():
            archive_path = target / 'artifacts' / archive_name
            backup = evidence / 'archive_source_archives' / archive_name
            backup.parent.mkdir(parents=True, exist_ok=True)
            archive_path.rename(backup)
            with tarfile.open(str(archive_path), 'x:gz') as archive:
                for _, name in entries:
                    archive.add(str(target / name), arcname=name, recursive=False)

    inventory = target / 'reproduction/inputs.sha256.csv'
    saved_inventory = evidence / 'archive_inputs.sha256.csv'
    shutil.copyfile(str(inventory), str(saved_inventory))
    with inventory.open(newline='') as stream:
        inputs = list(csv.DictReader(stream))
    for row in inputs:
        path = target / row['path']
        row['bytes'] = path.stat().st_size
        row['sha256'] = digest(path)
    with inventory.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('path', 'bytes', 'sha256'))
        writer.writeheader()
        writer.writerows(inputs)
    workspace_path = target / 'reproduction/workspace.json'
    shutil.copyfile(str(workspace_path), str(evidence / 'archive_workspace.json'))
    workspace = json.loads(workspace_path.read_text())
    workspace.update(source_kind='portable_code_with_archive_inputs',
                     input_manifest_sha256=digest(inventory),
                     input_bytes=sum(int(row['bytes']) for row in inputs))
    workspace_path.write_text(json.dumps(workspace, indent=2) + '\n')
    receipt = {'schema_version': 1, 'source_manifest_sha256': digest(ROOT / 'source_manifest.json'),
               'installed_adaptations': changes, 'renewed_source_seals': seals,
               'archive_input_inventory': str(saved_inventory.relative_to(target)),
               'portable_input_inventory': str(inventory.relative_to(target)),
               'portable_input_inventory_sha256': digest(inventory),
               'purpose': 'Fresh training study; archived checkpoint replay uses the separate archive-preserving model workspace.'}
    (evidence / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path,
                        help='Root of the extracted companion Zenodo archive')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-only', action='store_true')
    group.add_argument('--destination', type=Path,
                       help='A new study directory under an existing parent')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = args.archive.resolve(strict=True)
    verification = verify(source)
    if args.verify_only:
        print(json.dumps(dict(status='verified', **verification), indent=2))
        return
    target = args.destination.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('Choose a destination that does not exist: ' + str(target))
    if not target.parent.is_dir():
        raise FileNotFoundError('Create or choose an existing destination parent first')
    helper = source / 'scripts/create_authoritative_workspace.py'
    spec = importlib.util.spec_from_file_location('archived_workspace_builder', str(helper))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.create(source, target, dry_run=args.dry_run)
    result['source_verification'] = verification
    if args.dry_run:
        result['portable_adaptations'] = [row['path'] for row in records() if row['adaptation']]
    else:
        result['portable_source_receipt'] = install_portable_sources(target)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
