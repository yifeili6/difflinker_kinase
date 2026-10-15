#!/usr/bin/env python3
"""Copy verified archived models and their checkpoint-continuation records."""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BLOCK = 4 * 1024 * 1024
MANIFEST = 'provenance/artifact_manifest.json'
RUN = 'provenance/run_manifest.json'
BUNDLE = 'reproduction/archived_model'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Expected a normalized relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value or value == '.':
        raise ValueError('Unsafe path: ' + repr(value))
    return value


def contained(root, name):
    path = root / relative(name)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError('Symbolic link in input or destination: ' + str(item))
    if root not in path.resolve().parents:
        raise ValueError('Path escapes root: ' + name)
    return path


def fingerprint(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('Expected an ordinary file: ' + str(path))
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def read(path):
    with path.open() as handle:
        return json.load(handle)


def check(root, name, expected):
    path = contained(root, name)
    observed = fingerprint(path)
    if observed != {key: expected[key] for key in ('bytes', 'sha256')}:
        raise ValueError('File differs from its manifest: ' + name)
    return path


def write_json_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def operation_bundle(archive, records, selection, run):
    """Include the manifest-listed public operation trees and actual lineage files.

    Execution checksums and source_fingerprints_at_capture are historical records.
    Copied parent_* artifacts and committed receipt links bind the archived files.
    """
    required, prefixes, visited = set(), set(), set()

    def bound(info):
        name = relative(info["path"])
        expected = records.get(name)
        if expected is None or expected["sha256"] != info["sha256"] or (
                "bytes" in info and expected["bytes"] != info["bytes"]):
            raise ValueError("Archived segment binding differs: " + name)
        path = check(archive, name, expected)
        required.add(name)
        if name.startswith("recovery_operations/"):
            parts = PurePosixPath(name).parts
            if len(parts) < 3:
                raise ValueError("Segment evidence must lie inside its operation directory")
            prefixes.add("/".join(parts[:2]) + "/")
        return path

    def receipt(info, kind):
        path = bound(info)
        if (info["path"], kind) in visited:
            return
        visited.add((info["path"], kind))
        data = read(path)
        if kind == "map":
            if data.get("study_id") != selection["study_id"] or not data.get("segments"):
                raise ValueError("Archived segment map identifies a different study")
            for item in data["segments"].values():
                receipt(item["contract"], "contract")
        elif kind == "contract":
            if data.get("kind") != "kinase_hpo_segment" or data.get("study_id") != selection["study_id"]:
                raise ValueError("Archived segment contract identity differs")
            bound(data["config"])
            bound(data["parent"]["run_metadata"])
        elif kind == "completion":
            if data.get("kind") != "kinase_segment_completion" or data.get("status") != "completed":
                raise ValueError("Archived segment completion is incomplete")
            for key, child_kind in (("contract", "contract"), ("snapshot", "snapshot"),
                                    ("activation", "activation"), ("allocation", "allocation")):
                receipt(data[key], child_kind)
            for key in ("training_result", "run_metadata", "metrics", "best_checkpoint", "last_checkpoint",
                        "state_verification", "config_snapshot"):
                bound(data[key])
        elif kind == "snapshot":
            if data.get("kind") != "kinase_segment_snapshot":
                raise ValueError("Archived snapshot kind differs")
            for key in ("parent_last", "parent_best", "parent_metrics", "parent_run_metadata"):
                bound(data[key])
            receipt(data["contract"], "contract")
        elif kind == "activation":
            if data.get("kind") != "kinase_segment_activation":
                raise ValueError("Archived activation kind differs")
            bound(data["parent_terminal"]["evidence"])
            receipt(data["contract"], "contract")
            receipt(data["snapshot"], "snapshot")
            receipt(data["allocation"], "allocation")
        elif kind == "allocation":
            if data.get("kind") != "kinase_segment_allocation":
                raise ValueError("Archived allocation kind differs")
            receipt(data["contract"], "contract")

    if selection.get("segment_map"):
        receipt(selection["segment_map"], "map")
    for trial in selection.get("trials", []):
        lineage = trial.get("segment_lineage")
        if lineage:
            if not selection.get("segment_map"):
                raise ValueError("Segmented selection lacks its segment map")
            receipt(lineage["contract"], "contract")
            receipt(lineage["completion"], "completion")
            bound(lineage["parent_metrics"])
            bound(lineage["segment_metrics"])
    continuation = run.get("checkpoint_continuation")
    if continuation:
        if not selection.get("segment_map") or continuation["segment_map"] != selection["segment_map"]:
            raise ValueError("Run and selection segment maps differ")
        for key in ("handoff", "source_bindings"):
            if key in continuation:
                bound(continuation[key])
    names = required | {name for name in records if any(name.startswith(prefix) for prefix in prefixes)}
    return names, sorted(prefixes)


def plan(archive, workspace):
    archive = Path(archive).resolve(strict=True)
    workspace = Path(workspace).resolve(strict=True)
    if archive == workspace or archive in workspace.parents or workspace in archive.parents:
        raise ValueError('Archive and workspace must be separate directory trees')
    bundle = contained(workspace, BUNDLE)
    if bundle.exists():
        raise FileExistsError('Archived model bundle already exists: ' + str(bundle))
    manifest_path = contained(archive, MANIFEST)
    manifest_info = fingerprint(manifest_path)
    records = {}
    for item in read(manifest_path)['files']:
        name = relative(item['path'])
        if name == MANIFEST or name in records:
            raise ValueError('Duplicate or self-referential artifact path')
        if (type(item['bytes']) is not int or item['bytes'] < 0
                or not isinstance(item['sha256'], str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Invalid artifact fingerprint')
        records[name] = {key: item[key] for key in ('bytes', 'sha256')}
    run_path = check(archive, RUN, records[RUN])
    run = read(run_path)
    if run.get('status') != 'complete':
        raise ValueError('Archived run is incomplete')
    selection_name = relative(run['model']['hpo_selection'])
    selection_path = check(archive, selection_name, records[selection_name])
    if records[selection_name]['sha256'] != run['model']['hpo_selection_sha256']:
        raise ValueError('Run manifest selection binding differs')
    selection = read(selection_path)
    if selection.get('status') != 'complete' or selection.get('study_id') != run['run_id']:
        raise ValueError('HPO selection and archived run differ')
    checkpoint = relative(run['model']['checkpoint'])
    winner = selection['winner']
    if (checkpoint != winner['selected_checkpoint']
            or records[checkpoint]['sha256'] != run['model']['sha256']
            or run['model']['sha256'] != winner['selected_checkpoint_sha256']):
        raise ValueError('Selected checkpoint bindings differ')
    generation_config = relative(run['paths']['generation_config'])
    training_config = relative(winner['config'])
    if records[training_config]['sha256'] != winner['config_sha256']:
        raise ValueError('Selected trial configuration binding differs')

    workspace_record = read(contained(workspace, 'reproduction/workspace.json'))
    inventory = contained(workspace, 'reproduction/inputs.sha256.csv')
    if workspace_record.get('status') != 'ready' or sha(inventory) != workspace_record['input_manifest_sha256']:
        raise ValueError('Workspace input inventory binding differs')
    existing = set()
    with inventory.open(newline='') as handle:
        for item in csv.DictReader(handle):
            name = relative(item['path'])
            if name in existing:
                raise ValueError('Duplicate workspace input path')
            expected = {'bytes': int(item['bytes']), 'sha256': item['sha256']}
            if records.get(name) != expected:
                raise ValueError('Workspace input belongs to a different archive: ' + name)
            check(workspace, name, expected)
            existing.add(name)
    if not existing or workspace_record['input_files'] != len(existing):
        raise ValueError('Workspace input count differs')

    study_prefix = 'outputs/hpo/studies/' + str(selection['study_id']) + '/'
    required = {checkpoint, selection_name, generation_config, training_config}
    for field in ('protocol', 'attempt_ledger', 'active_tasks', 'hardware_attempts', 'trial_outcomes'):
        item = selection.get(field)
        if item is not None:
            name = relative(item['path'])
            if records[name]['sha256'] != item['sha256']:
                raise ValueError('HPO provenance binding differs: ' + field)
            required.add(name)
    if run['model'].get('validation'):
        required.add(relative(run['model']['validation']))
    operation_names, operation_prefixes = operation_bundle(archive, records, selection, run)
    prefixes = ('models/', 'logs/general/', study_prefix)
    names = required | {name for name in records if name.startswith(prefixes)
                        or (name.startswith('logs/kr-hpo-') and name.endswith(('.out', '.err')))}
    names.update(operation_names)
    files = []
    for name in sorted(names):
        check(archive, name, records[name])
        target = contained(workspace, name)
        if target.exists() and name not in existing:
            raise FileExistsError('Destination already exists: ' + name)
        files.append({'path': name, **records[name], 'action': 'verified_workspace_input' if name in existing else 'copy'})
    for name in (generation_config, training_config, checkpoint, selection_name):
        if name not in names:
            raise ValueError('Required model input is absent')
    return {'schema_version': 1, 'archive': str(archive), 'workspace': str(workspace),
            'archive_artifact_manifest': manifest_info, 'archive_run_manifest': records[RUN],
            'workspace_inventory': fingerprint(inventory), 'study_id': selection['study_id'],
            'checkpoint': checkpoint, 'selection': selection_name, 'generation_config': generation_config,
            'training_config': training_config, 'operation_prefixes': operation_prefixes, 'files': files}


def copy_file_new(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    value, size = hashlib.sha256(), 0
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        for block in iter(lambda: incoming.read(BLOCK), b''):
            value.update(block)
            size += len(block)
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if {'bytes': size, 'sha256': value.hexdigest()} != expected or fingerprint(destination) != expected:
        raise ValueError('Copied file differs: ' + str(destination))


def copy_bundle(archive, workspace, dry_run=False):
    record = plan(archive, workspace)
    if dry_run:
        return dict(record, status='planned')
    archive, workspace = Path(record['archive']), Path(record['workspace'])
    bundle = contained(workspace, BUNDLE)
    bundle.mkdir(parents=True)
    write_json_new(bundle / 'copy_plan.json', record)
    for item in record['files']:
        expected = {key: item[key] for key in ('bytes', 'sha256')}
        source = check(archive, item['path'], expected)
        if item['action'] == 'copy':
            target = contained(workspace, item['path'])
            copy_file_new(source, target, expected)
        else:
            check(workspace, item['path'], expected)
    for name, expected in ((MANIFEST, record['archive_artifact_manifest']), (RUN, record['archive_run_manifest'])):
        source = check(archive, name, expected)
        copy_file_new(source, bundle / Path(name).name, expected)
    for item in record['files']:
        check(workspace, item['path'], item)
    check(archive, MANIFEST, record['archive_artifact_manifest'])
    check(workspace, 'reproduction/inputs.sha256.csv', record['workspace_inventory'])
    result = {'schema_version': 1, 'status': 'copied', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_archive': str(archive), 'archived_study_id': record['study_id'],
              'historical_run_manifest': BUNDLE + '/run_manifest.json',
              'historical_artifact_manifest': BUNDLE + '/artifact_manifest.json',
              'checkpoint': record['checkpoint'], 'selection': record['selection'],
              'generation_config': record['generation_config'], 'training_config': record['training_config'],
              'operation_prefixes': record['operation_prefixes'],
              'copy_plan_sha256': sha(bundle / 'copy_plan.json'), 'files': record['files']}
    write_json_new(bundle / 'bundle.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = copy_bundle(args.archive, args.workspace, args.dry_run)
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'checkpoint': result['checkpoint'], 'selection': result['selection']}, indent=2))


if __name__ == '__main__':
    main()
