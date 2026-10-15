from __future__ import annotations
import argparse
import csv
import hashlib
import fcntl
import inspect
import json
import math
import os
import pickle
import random
import re
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import yaml
from rdkit import Chem, Geometry
from kinase_gap.chemistry_io import reconstruct_fragment, sanitized_complete_molecule, sdf_record, attachment_diagnostics
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / 'vendor' / 'DiffLinker'
if not VENDOR_ROOT.is_dir():
    raise RuntimeError(f'Vendored DiffLinker source is missing: {VENDOR_ROOT}')
sys.path.insert(0, str(VENDOR_ROOT))
from src import const
from src.datasets import MOADDataset, collate, get_one_hot, parse_molecule
from src.lightning import DDPM
from src.molecule_builder import get_bond_order
from src.utils import FoundNaNException, disable_rdkit_logging
SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile('^[A-Za-z0-9_.-]+$')
EXPECTED_LINKER_SIZES = (8, 9, 10, 11, 12, 13)
EXPECTED_CONTEXTS = 62
EXPECTED_SAMPLES_PER_CONTEXT = 128
EXPECTED_BATCH_SIZE = 16
EXPECTED_N_STEPS = 500
EXPECTED_NAN_RETRIES = 5
SHARD_FILE_NAMES = ('molecules.sdf', 'samples.csv', 'coordinates.npz', 'done.json')
COMBINED_FILE_NAMES = ('candidates.sdf', 'candidates.csv', 'provenance.json', 'done.json')
CSV_FIELDS = ('candidate_id', 'run_id', 'shard_index', 'context_index', 'replicate', 'source_uuid', 'fragmentation_id', 'structure_id', 'kinase', 'pdb', 'anchor_1', 'anchor_2', 'fragments_size', 'linker_size', 'num_atoms', 'shard_seed', 'batch_seed', 'nan_retries', 'valid', 'connected', 'smiles', 'in_memory_sanitize_pass', 'postprocessing_error', 'anchor_attachment_pass', 'fragment_linker_bond_count', 'fragment_linker_edges_json', 'retained_stereo_pass', 'retained_stereo_errors_json', 'retained_nonanchor_explicit_h_match', 'retained_nonanchor_total_h_match', 'retained_nonanchor_h_changes_json')
REQUIRED_TABLE_FIELDS = {'uuid', 'fragmentation_id', 'structure_id', 'split', 'kinase', 'pdb', 'fragments_size', 'anchor_1', 'anchor_2', 'pocket_full_size', 'pocket_cutoff_angstrom'}
REQUIRED_CHECKPOINT_HPARAMS = {'train_data_prefix': 'KLIFS_train.full', 'val_data_prefix': 'KLIFS_val.full', 'in_node_nf': 9, 'context_node_nf': 3, 'include_charges': False, 'anchors_context': True, 'center_of_mass': 'fragments', 'graph_type': 'FC-10A-4A', 'diffusion_steps': EXPECTED_N_STEPS}
DETERMINISM_POLICY = 'Fixed seeds are derived for each context, linker size and batch; PyTorch deterministic algorithms run with the recorded warn_only setting.'

@dataclass(frozen=True)
class GenerationConfig:
    path: Path
    payload: Dict[str, Any]
    data_dir: Path
    test_prefix: str
    targets_manifest: Path
    output_root: Path
    representative_structures: Tuple[str, ...]
    target_pdbs: Tuple[str, ...]
    linker_sizes: Tuple[int, ...]
    samples_per_context: int
    seed: int
    n_steps: int
    batch_size: int
    max_nan_retries: int
    deterministic_warn_only: bool

@dataclass
class GenerationContext:
    context_index: int
    row_index: int
    row: Dict[str, str]
    fragment: Chem.Mol
    pocket: Dict[str, Any]
    item: Dict[str, Any]

    @property
    def fragmentation_id(self) -> str:
        return self.row['fragmentation_id']

    @property
    def structure_id(self) -> str:
        return self.row['structure_id']

@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    context_index: int
    linker_size_index: int
    linker_size: int
    shard_seed: int
    context: GenerationContext

    @property
    def shard_name(self) -> str:
        return f'{self.shard_index:04d}'

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def project_path(value: str, *, must_exist: bool=False) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f'Path must be project-root-relative: {value}')
    root = PROJECT_ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Path escapes the project root: {value}') from exc
    if must_exist and (not resolved.exists()):
        raise FileNotFoundError(f'Required project path does not exist: {value}')
    return resolved

def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

def validate_run_id(value: str) -> str:
    if value in {'.', '..'}:
        raise ValueError("Run ID may not be '.' or '..'")
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f'Run ID may contain only letters, numbers, dot, underscore, and hyphen: {value!r}')
    return value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def count_sdf_records(path: Path) -> int:
    count = 0
    with path.open('rb') as handle:
        for line in handle:
            if line.strip() == b'$$$$':
                count += 1
    return count

def count_csv_records(path: Path) -> int:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f'Candidate CSV schema changed: {relative(path)}')
        return sum((1 for _ in reader))

def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()

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
    (descriptor, temporary_name) = tempfile.mkstemp(prefix=f'.{path.name}.tmp.', dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')

def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f'Expected a JSON mapping: {relative(path)}')
    return payload

def read_config(config_value: str) -> GenerationConfig:
    config_path = project_path(config_value, must_exist=True)
    with config_path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f'Generation configuration must be a YAML mapping: {config_value}')
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported generation schema_version')
    required = {'data', 'test_data_prefix', 'targets_manifest', 'output_root', 'representative_structures', 'target_pdbs', 'expected_contexts', 'linker_sizes', 'samples_per_context', 'expected_candidates', 'seed', 'n_steps', 'batch_size', 'max_nan_retries', 'deterministic_warn_only'}
    missing = sorted((key for key in required if key not in payload))
    if missing:
        raise ValueError(f"Generation configuration is missing keys: {', '.join(missing)}")
    data_dir = project_path(str(payload['data']), must_exist=True)
    targets_manifest = project_path(str(payload['targets_manifest']), must_exist=True)
    output_root = project_path(str(payload['output_root']))
    expected_output_parent = project_path('outputs/generation')
    if output_root != expected_output_parent:
        raise ValueError('output_root must remain exactly outputs/generation')
    prefix = str(payload['test_data_prefix'])
    if prefix != 'KLIFS_test.full':
        raise ValueError('test_data_prefix must remain exactly KLIFS_test.full')
    representatives_raw = payload['representative_structures']
    if not isinstance(representatives_raw, list) or not representatives_raw:
        raise TypeError('representative_structures must be a non-empty list')
    representatives = tuple((str(value) for value in representatives_raw))
    if len(representatives) != 9 or len(set(representatives)) != len(representatives):
        raise ValueError('representative_structures must contain exactly nine unique IDs')
    target_pdbs_raw = payload['target_pdbs']
    if not isinstance(target_pdbs_raw, list):
        raise TypeError('target_pdbs must be a list')
    target_pdbs = tuple((str(value).lower() for value in target_pdbs_raw))
    if len(target_pdbs) != 9 or len(set(target_pdbs)) != len(target_pdbs):
        raise ValueError('target_pdbs must contain exactly nine unique PDB accessions')
    linker_sizes_raw = payload['linker_sizes']
    if not isinstance(linker_sizes_raw, list):
        raise TypeError('linker_sizes must be a list')
    linker_sizes = tuple((int(value) for value in linker_sizes_raw))
    if linker_sizes != EXPECTED_LINKER_SIZES:
        raise ValueError(f'linker_sizes must remain exactly {list(EXPECTED_LINKER_SIZES)}')
    expected_contexts = int(payload['expected_contexts'])
    samples_per_context = int(payload['samples_per_context'])
    expected_candidates = int(payload['expected_candidates'])
    batch_size = int(payload['batch_size'])
    n_steps = int(payload['n_steps'])
    max_nan_retries = int(payload['max_nan_retries'])
    seed = int(payload['seed'])
    deterministic_warn_only = payload['deterministic_warn_only']
    if expected_contexts != EXPECTED_CONTEXTS:
        raise ValueError(f'expected_contexts must remain {EXPECTED_CONTEXTS}')
    if samples_per_context != EXPECTED_SAMPLES_PER_CONTEXT:
        raise ValueError(f'samples_per_context must remain {EXPECTED_SAMPLES_PER_CONTEXT}')
    if batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f'batch_size must remain {EXPECTED_BATCH_SIZE}')
    if n_steps != EXPECTED_N_STEPS:
        raise ValueError(f'n_steps must remain {EXPECTED_N_STEPS}')
    if max_nan_retries != EXPECTED_NAN_RETRIES:
        raise ValueError(f'max_nan_retries must remain {EXPECTED_NAN_RETRIES}')
    if not isinstance(deterministic_warn_only, bool) or not deterministic_warn_only:
        raise ValueError('deterministic_warn_only must be the YAML boolean true')
    if seed < 0:
        raise ValueError('seed must be non-negative')
    calculated_candidates = expected_contexts * len(linker_sizes) * samples_per_context
    if expected_candidates != calculated_candidates or expected_candidates != 47616:
        raise ValueError(f'expected_candidates must equal 62 * 6 * 128 = 47616, found {expected_candidates}')
    return GenerationConfig(path=config_path, payload=dict(payload), data_dir=data_dir, test_prefix=prefix, targets_manifest=targets_manifest, output_root=output_root, representative_structures=representatives, target_pdbs=target_pdbs, linker_sizes=linker_sizes, samples_per_context=samples_per_context, seed=seed, n_steps=n_steps, batch_size=batch_size, max_nan_retries=max_nan_retries, deterministic_warn_only=deterministic_warn_only)

def dataset_stem(prefix: str) -> str:
    if '.' not in prefix:
        raise ValueError(f'Generation requires a pocket-conditioned prefix: {prefix}')
    (stem, pocket_mode) = prefix.split('.', maxsplit=1)
    if pocket_mode != 'full':
        raise ValueError('Generation protocol requires full-pocket tensors')
    return stem

def dataset_paths(config: GenerationConfig) -> Dict[str, Path]:
    stem = dataset_stem(config.test_prefix)
    paths = {'table': config.data_dir / f'{stem}_table.csv', 'fragments': config.data_dir / f'{stem}_frag.sdf', 'linkers': config.data_dir / f'{stem}_link.sdf', 'molecules': config.data_dir / f'{stem}_mol.sdf', 'pockets': config.data_dir / f'{stem}_pockets.pkl', 'chemistry': config.data_dir / f'{stem}_chemistry.jsonl', 'dataset_report': config.data_dir / 'dataset_report.json', 'files_manifest': config.data_dir / 'files.sha256.csv'}
    missing = [relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generation data inputs are missing: {', '.join(missing)}")
    return paths

def verify_dataset_hashes(paths: Mapping[str, Path]) -> Dict[str, str]:
    manifest_path = paths['files_manifest']
    with manifest_path.open('r', encoding='utf-8', newline='') as handle:
        rows = {row['path']: row for row in csv.DictReader(handle) if row.get('path') and row.get('sha256')}
    observed: Dict[str, str] = {}
    for label in ('table', 'fragments', 'linkers', 'molecules', 'pockets', 'chemistry'):
        path = paths[label]
        entry = rows.get(path.name)
        if entry is None:
            raise ValueError(f'Dataset hash manifest omits {path.name}')
        if int(entry['bytes']) != path.stat().st_size:
            raise ValueError(f'Dataset byte count changed: {path.name}')
        digest = sha256_file(path)
        if digest != entry['sha256']:
            raise ValueError(f'Dataset SHA-256 changed: {path.name}')
        observed[label] = digest
    observed['dataset_report'] = sha256_file(paths['dataset_report'])
    observed['files_manifest'] = sha256_file(paths['files_manifest'])
    return observed

def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], Tuple[str, ...]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return (rows, fields)

def validate_target_registry(config: GenerationConfig) -> Dict[str, Dict[str, str]]:
    (rows, fields) = read_csv_rows(config.targets_manifest)
    required = {'structure_id', 'pdb', 'split'}
    if not required.issubset(fields):
        raise ValueError(f'Targets manifest is missing columns: {sorted(required - set(fields))}')
    by_structure: Dict[str, Dict[str, str]] = {}
    for row in rows:
        structure_id = row['structure_id']
        if structure_id in by_structure:
            raise ValueError(f'Duplicate structure in targets manifest: {structure_id}')
        by_structure[structure_id] = row
    missing = sorted(set(config.representative_structures) - set(by_structure))
    if missing:
        raise ValueError('Representative structures are not declared manuscript targets: ' + ', '.join(missing))
    representative_pdbs = set()
    for structure_id in config.representative_structures:
        row = by_structure[structure_id]
        if row['split'] != 'test':
            raise ValueError(f'Representative target is not test-only: {structure_id}')
        representative_pdbs.add(row['pdb'].lower())
    if representative_pdbs != set(config.target_pdbs):
        raise ValueError(f'Representative PDB set differs from the frozen target_pdbs: {sorted(representative_pdbs)}')
    return by_structure

def load_selected_rows(config: GenerationConfig, table_path: Path, targets: Mapping[str, Mapping[str, str]]) -> List[Tuple[int, Dict[str, str]]]:
    (rows, fields) = read_csv_rows(table_path)
    if not REQUIRED_TABLE_FIELDS.issubset(fields):
        raise ValueError('Test table is missing columns: ' + ', '.join(sorted(REQUIRED_TABLE_FIELDS - set(fields))))
    by_structure: Dict[str, List[Tuple[int, Dict[str, str]]]] = {structure_id: [] for structure_id in config.representative_structures}
    seen_fragmentations = set()
    for (row_index, row) in enumerate(rows):
        if int(row['uuid']) != row_index:
            raise ValueError(f"Test UUIDs must be contiguous and row-aligned: row={row_index}, uuid={row['uuid']}")
        if row['structure_id'] not in by_structure:
            continue
        if row['split'] != 'test':
            raise ValueError(f"Selected generation row is not test-only: {row['fragmentation_id']}")
        target = targets[row['structure_id']]
        if row['pdb'].lower() != target['pdb'].lower():
            raise ValueError(f"Target/table PDB mismatch for {row['fragmentation_id']}")
        if row['pdb'].lower() not in set(config.target_pdbs):
            raise ValueError(f"Selected row is outside target_pdbs: {row['fragmentation_id']}")
        if row['fragmentation_id'] in seen_fragmentations:
            raise ValueError(f"Duplicate fragmentation ID: {row['fragmentation_id']}")
        seen_fragmentations.add(row['fragmentation_id'])
        by_structure[row['structure_id']].append((row_index, row))
    selected: List[Tuple[int, Dict[str, str]]] = []
    for structure_id in config.representative_structures:
        structure_rows = sorted(by_structure[structure_id], key=lambda item: int(item[1]['uuid']))
        if not structure_rows:
            raise ValueError(f'Representative structure has no fragmentation rows: {structure_id}')
        selected.extend(structure_rows)
    if len(selected) != EXPECTED_CONTEXTS:
        counts = {structure_id: len(by_structure[structure_id]) for structure_id in config.representative_structures}
        raise ValueError(f'Representative selection must yield exactly {EXPECTED_CONTEXTS} rows; found {len(selected)} ({counts})')
    return selected

def pocket_arrays(pocket: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if 'full_coord' not in pocket or 'full_types' not in pocket:
        raise ValueError('Pocket record is missing full_coord or full_types')
    coordinates = np.asarray(pocket['full_coord'], dtype=np.float32)
    types = np.asarray(pocket['full_types'], dtype=object)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('Pocket coordinates must have shape N x 3')
    if len(coordinates) != len(types) or len(coordinates) == 0:
        raise ValueError('Pocket coordinates and atom types are not aligned')
    if not np.isfinite(coordinates).all():
        raise ValueError('Pocket coordinates contain non-finite values')
    unsupported = sorted(set(map(str, types)) - set(const.GEOM_ATOM2IDX))
    if unsupported:
        raise ValueError(f'Pocket contains unsupported atom types: {unsupported}')
    cutoff = float(pocket.get('cutoff_angstrom', math.nan))
    if not math.isclose(cutoff, 10.0, rel_tol=0.0, abs_tol=1e-08):
        raise ValueError(f'Pocket cutoff must be exactly 10.0 Å, found {cutoff}')
    return (coordinates, types)

def build_generation_item(row: Mapping[str, str], fragment: Chem.Mol, pocket: Mapping[str, Any]) -> Dict[str, Any]:
    if fragment is None or fragment.GetNumConformers() != 1:
        raise ValueError(f"Fragment record is missing one conformer: {row['fragmentation_id']}")
    (fragment_positions, fragment_one_hot, fragment_charges) = parse_molecule(fragment, is_geom=True)
    fragment_positions = np.asarray(fragment_positions, dtype=np.float32)
    fragment_one_hot = np.asarray(fragment_one_hot, dtype=np.float32)
    fragment_charges = np.asarray(fragment_charges, dtype=np.float32)
    if not np.isfinite(fragment_positions).all():
        raise ValueError(f"Fragment coordinates are non-finite: {row['fragmentation_id']}")
    fragment_size = fragment.GetNumAtoms()
    if fragment_size != int(row['fragments_size']):
        raise ValueError(f"Fragment atom count mismatch: {row['fragmentation_id']}")
    components = Chem.GetMolFrags(fragment, asMols=False, sanitizeFrags=False)
    if len(components) != 2:
        raise ValueError(f"Expected exactly two retained fragments: {row['fragmentation_id']}")
    anchor_indices = (int(row['anchor_1']), int(row['anchor_2']))
    if len(set(anchor_indices)) != 2 or any((anchor < 0 or anchor >= fragment_size for anchor in anchor_indices)):
        raise ValueError(f"Invalid anchor indices: {row['fragmentation_id']}")
    component_for_atom = {atom_index: component_index for (component_index, component) in enumerate(components) for atom_index in component}
    if component_for_atom[anchor_indices[0]] == component_for_atom[anchor_indices[1]]:
        raise ValueError(f"Anchors must occur on different fragments: {row['fragmentation_id']}")
    (pocket_positions, pocket_types) = pocket_arrays(pocket)
    if len(pocket_positions) != int(row['pocket_full_size']):
        raise ValueError(f"Pocket atom count mismatch: {row['fragmentation_id']}")
    if not math.isclose(float(row['pocket_cutoff_angstrom']), 10.0, rel_tol=0.0, abs_tol=1e-08):
        raise ValueError(f"Table pocket cutoff changed: {row['fragmentation_id']}")
    pocket_one_hot = np.asarray([get_one_hot(str(atom), const.GEOM_ATOM2IDX) for atom in pocket_types], dtype=np.float32)
    pocket_charges = np.asarray([const.GEOM_CHARGES[str(atom)] for atom in pocket_types], dtype=np.float32)
    positions = np.concatenate([fragment_positions, pocket_positions], axis=0)
    one_hot = np.concatenate([fragment_one_hot, pocket_one_hot], axis=0)
    charges = np.concatenate([fragment_charges, pocket_charges], axis=0)
    anchors = np.zeros(len(positions), dtype=np.float32)
    anchors[list(anchor_indices)] = 1.0
    fragment_only_mask = np.concatenate([np.ones(fragment_size, dtype=np.float32), np.zeros(len(pocket_positions), dtype=np.float32)])
    pocket_mask = 1.0 - fragment_only_mask
    fragment_mask = np.ones(len(positions), dtype=np.float32)
    linker_mask = np.zeros(len(positions), dtype=np.float32)
    return {'uuid': str(row['uuid']), 'name': row['fragmentation_id'], 'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT), 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT), 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT), 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT), 'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT), 'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT), 'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT), 'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT), 'num_atoms': len(positions)}

def load_contexts(config: GenerationConfig, paths: Mapping[str, Path], selected_rows: Sequence[Tuple[int, Dict[str, str]]]) -> List[GenerationContext]:
    with paths['pockets'].open('rb') as handle:
        pockets = pickle.load(handle)
    if not isinstance(pockets, list):
        raise TypeError('KLIFS test pockets must be a row-aligned list')
    fragments = Chem.SDMolSupplier(str(paths['fragments']), sanitize=False, removeHs=True)
    (table_rows, _) = read_csv_rows(paths['table'])
    if len(fragments) != len(table_rows) or len(pockets) != len(table_rows):
        raise ValueError(f'Test table, fragment SDF, and pocket pickle are not row-aligned: table={len(table_rows)}, fragments={len(fragments)}, pockets={len(pockets)}')
    contexts: List[GenerationContext] = []
    for (context_index, (row_index, row)) in enumerate(selected_rows):
        carrier = fragments[row_index]
        if carrier is None:
            raise ValueError(f'Unreadable fragment SDF record at row {row_index}')
        if carrier.HasProp('_Name') and carrier.GetProp('_Name') != row['fragmentation_id']:
            raise ValueError(f'Fragment SDF name mismatch at row {row_index}')
        fragment = reconstruct_fragment(row, PROJECT_ROOT)
        if [a.GetSymbol() for a in carrier.GetAtoms()] != [a.GetSymbol() for a in fragment.GetAtoms()]:
            raise ValueError(f'Fragment carrier element/order mismatch at row {row_index}')
        if not np.allclose(carrier.GetConformer().GetPositions(), fragment.GetConformer().GetPositions(), atol=5.1e-05, rtol=0):
            raise ValueError(f'Fragment carrier coordinate mismatch at row {row_index}')
        pocket = pockets[row_index]
        if not isinstance(pocket, Mapping):
            raise TypeError(f'Pocket row {row_index} is not a mapping')
        item = build_generation_item(row, fragment, pocket)
        contexts.append(GenerationContext(context_index=context_index, row_index=row_index, row=row, fragment=Chem.Mol(fragment), pocket=dict(pocket), item=item))
    return contexts

def checkpoint_hparams(payload: Mapping[str, Any]) -> Dict[str, Any]:
    hparams = payload.get('hyper_parameters')
    if isinstance(hparams, Mapping):
        return dict(hparams)
    if hasattr(hparams, '__dict__'):
        return dict(vars(hparams))
    raise ValueError('Checkpoint is missing a hyper_parameters mapping')

def load_checkpoint_payload(checkpoint: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location='cpu')
    if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), Mapping):
        raise TypeError('Generation requires a Lightning checkpoint with a state_dict')
    return payload

def validate_checkpoint(payload: Mapping[str, Any], checkpoint: Path, config: GenerationConfig) -> Dict[str, Any]:
    hparams = checkpoint_hparams(payload)
    mismatches = []
    for (key, expected) in REQUIRED_CHECKPOINT_HPARAMS.items():
        observed = hparams.get(key)
        if observed != expected:
            mismatches.append(f'{key}: checkpoint={observed!r}, expected={expected!r}')
    if mismatches:
        raise ValueError('Checkpoint is incompatible with the frozen generation protocol:\n  ' + '\n  '.join(mismatches))
    if 'test' in str(hparams['train_data_prefix']).lower() or 'test' in str(hparams['val_data_prefix']).lower():
        raise ValueError('Checkpoint reports use of test data during fitting')
    if config.n_steps > int(hparams['diffusion_steps']):
        raise ValueError('Generation n_steps exceeds the trained diffusion schedule')
    return {'path': relative(checkpoint), 'sha256': sha256_file(checkpoint), 'epoch': int(payload.get('epoch', -1)), 'global_step': int(payload.get('global_step', -1)), 'train_data_prefix': hparams['train_data_prefix'], 'val_data_prefix': hparams['val_data_prefix'], 'graph_type': hparams['graph_type'], 'anchors_context': hparams['anchors_context'], 'diffusion_steps': hparams['diffusion_steps']}

def configure_torch_determinism(*, warn_only: bool) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def derive_seed(base_seed: int, *parts: Any) -> int:
    material = '|'.join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False) % (2 ** 63 - 1)

def build_shards(config: GenerationConfig, contexts: Sequence[GenerationContext]) -> List[ShardSpec]:
    shards: List[ShardSpec] = []
    for context in contexts:
        for (size_index, linker_size) in enumerate(config.linker_sizes):
            shard_index = context.context_index * len(config.linker_sizes) + size_index
            shard_seed = derive_seed(config.seed, 'shard-v1', context.fragmentation_id, linker_size)
            shards.append(ShardSpec(shard_index=shard_index, context_index=context.context_index, linker_size_index=size_index, linker_size=linker_size, shard_seed=shard_seed, context=context))
    expected = EXPECTED_CONTEXTS * len(EXPECTED_LINKER_SIZES)
    if len(shards) != expected or [shard.shard_index for shard in shards] != list(range(expected)):
        raise AssertionError('Shard construction is not contiguous and deterministic')
    return shards

def selected_rows_report(contexts: Sequence[GenerationContext]) -> List[Dict[str, Any]]:
    return [{'context_index': context.context_index, 'source_uuid': int(context.row['uuid']), 'fragmentation_id': context.fragmentation_id, 'structure_id': context.structure_id, 'kinase': context.row['kinase'], 'pdb': context.row['pdb'].lower(), 'anchor_1': int(context.row['anchor_1']), 'anchor_2': int(context.row['anchor_2']), 'fragments_size': int(context.row['fragments_size']), 'pocket_full_size': int(context.row['pocket_full_size']), 'split': context.row['split']} for context in contexts]

def build_contract(config: GenerationConfig, contexts: Sequence[GenerationContext], checkpoint_info: Mapping[str, Any], data_hashes: Mapping[str, str]) -> Dict[str, Any]:
    plan = {'schema_version': SCHEMA_VERSION, 'chemistry_protocol': 'raw_parent_fragment_v2_kekulized_complete_molecules', 'source_hashes': {name: sha256_file(PROJECT_ROOT / 'src/kinase_gap' / name) for name in ('generate.py', 'fragmentation.py', 'chemistry_io.py')}, 'config': relative(config.path), 'config_sha256': sha256_file(config.path), 'checkpoint': dict(checkpoint_info), 'data_inputs': {label: {'path': relative(dataset_paths(config)[label]), 'sha256': digest} for (label, digest) in data_hashes.items()}, 'targets_manifest': {'path': relative(config.targets_manifest), 'sha256': sha256_file(config.targets_manifest)}, 'representative_rows': selected_rows_report(contexts), 'representative_context_count': len(contexts), 'linker_sizes': list(config.linker_sizes), 'samples_per_condition': config.samples_per_context, 'condition_shards': len(contexts) * len(config.linker_sizes), 'expected_candidates': len(contexts) * len(config.linker_sizes) * config.samples_per_context, 'base_seed': config.seed, 'n_steps': config.n_steps, 'batch_size': config.batch_size, 'max_nan_retries': config.max_nan_retries, 'deterministic_warn_only': config.deterministic_warn_only, 'selection_policy': 'All valid two-cut fragmentation rows for the nine explicit representative test-only manuscript target structures, in configuration order and then UUID order.', 'linker_size_policy': 'Explicit exhaustive size sweep over 8, 9, 10, 11, 12, and 13 atoms.'}
    plan['contract_sha256'] = sha256_payload(plan)
    return plan

def derive_run_id(checkpoint: Path, checkpoint_sha256: str) -> str:
    stem = re.sub('[^A-Za-z0-9_.-]+', '-', checkpoint.stem).strip('.-') or 'checkpoint'
    return validate_run_id(f'generation-{stem}-{checkpoint_sha256[:12]}')

def validate_single_visible_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('Actual generation requires CUDA, but CUDA is unavailable')
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f'Actual generation requires exactly one visible GPU; torch.cuda.device_count() returned {count}. Set CUDA_VISIBLE_DEVICES to select one GPU.')
    return torch.device('cuda:0')

def instantiate_model(payload: Mapping[str, Any], config: GenerationConfig, device: torch.device) -> DDPM:
    hparams = checkpoint_hparams(payload)
    signature = inspect.signature(DDPM.__init__)
    kwargs: Dict[str, Any] = {}
    missing = []
    for (name, parameter) in signature.parameters.items():
        if name == 'self':
            continue
        if name in hparams:
            kwargs[name] = hparams[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise ValueError(f'Checkpoint lacks required DDPM hyperparameters: {missing}')
    kwargs['data_path'] = str(config.data_dir)
    kwargs['val_data_prefix'] = config.test_prefix
    kwargs['torch_device'] = str(device)
    model = DDPM(**kwargs)
    incompatible = model.load_state_dict(payload['state_dict'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'Strict checkpoint loading failed: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}')
    model.edm.T = config.n_steps
    model.val_dataset = MOADDataset(data=[])
    model.torch_device = str(device)
    model.eval().to(device)
    if any((parameter.device != device for parameter in model.parameters())):
        raise RuntimeError('Not every model parameter was moved to the generation GPU')
    return model

def move_batch_to_device(data: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for (key, value) in data.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=False)
        elif isinstance(value, list) and value and all((torch.is_tensor(item) for item in value)):
            moved[key] = [item.to(device=device, non_blocking=False) for item in value]
        else:
            moved[key] = value
    required_tensors = {'positions', 'one_hot', 'charges', 'anchors', 'fragment_only_mask', 'pocket_mask', 'fragment_mask', 'linker_mask', 'atom_mask', 'edge_mask'}
    for key in required_tensors:
        if not torch.is_tensor(moved.get(key)) or moved[key].device != device:
            raise RuntimeError(f'Batch tensor was not moved to {device}: {key}')
    return moved

def restore_fragment_frame(chain_final: torch.Tensor, node_mask: torch.Tensor, input_positions: torch.Tensor, fragment_only_mask: torch.Tensor) -> torch.Tensor:
    fragment_count = fragment_only_mask.sum(dim=1, keepdim=True)
    if torch.any(fragment_count <= 0):
        raise ValueError('Cannot restore coordinates without retained fragment atoms')
    fragment_com = (input_positions * fragment_only_mask).sum(dim=1, keepdim=True) / fragment_count
    restored = chain_final.clone()
    restored[..., :3] = restored[..., :3] + fragment_com * node_mask
    return restored

def extend_pocket_mask(pocket_mask: torch.Tensor, generated_node_mask: torch.Tensor) -> torch.Tensor:
    if pocket_mask.ndim != 3 or generated_node_mask.ndim != 3:
        raise ValueError('Pocket and generated node masks must be rank-three tensors')
    if pocket_mask.shape[0] != generated_node_mask.shape[0]:
        raise ValueError('Pocket and generated node masks have different batch sizes')
    added_nodes = generated_node_mask.shape[1] - pocket_mask.shape[1]
    if added_nodes < 0:
        raise ValueError('Generated node mask is shorter than the conditioning input')
    if added_nodes == 0:
        return pocket_mask
    padding = torch.zeros((pocket_mask.shape[0], added_nodes, pocket_mask.shape[2]), dtype=pocket_mask.dtype, device=pocket_mask.device)
    return torch.cat([pocket_mask, padding], dim=1)

def ligand_features(model: DDPM, restored: torch.Tensor, node_mask: torch.Tensor, pocket_mask: torch.Tensor, batch_index: int) -> Tuple[np.ndarray, List[str]]:
    keep = node_mask[batch_index].squeeze().bool() & ~pocket_mask[batch_index].squeeze().bool()
    coordinates = restored[batch_index, keep, :model.n_dims].detach().cpu().numpy().astype(np.float64)
    categorical = restored[batch_index, keep, model.n_dims:model.n_dims + model.num_classes]
    atom_indices = categorical.argmax(dim=-1).detach().cpu().tolist()
    symbols = [const.GEOM_IDX2ATOM[int(index)] for index in atom_indices]
    if len(coordinates) != len(symbols) or not np.isfinite(coordinates).all():
        raise ValueError('Generated ligand features are malformed')
    return (coordinates, symbols)

def _ordered_symbols(left: str, right: str) -> Tuple[str, str]:
    left_index = const.GEOM_ATOM2IDX[left]
    right_index = const.GEOM_ATOM2IDX[right]
    return (left, right) if left_index <= right_index else (right, left)

def build_candidate_molecule(fragment: Chem.Mol, coordinates: np.ndarray, symbols: Sequence[str], candidate_id: str) -> Chem.Mol:
    fragment_size = fragment.GetNumAtoms()
    if len(symbols) < fragment_size or len(coordinates) != len(symbols):
        raise ValueError('Generated atom count is inconsistent with the source fragments')
    source_symbols = [atom.GetSymbol() for atom in fragment.GetAtoms()]
    if list(symbols[:fragment_size]) != source_symbols:
        raise ValueError('Diffusion output changed a retained fragment atom type')
    base = Chem.Mol(fragment)
    base.RemoveAllConformers()
    editable = Chem.RWMol(base)
    for symbol in symbols[fragment_size:]:
        editable.AddAtom(Chem.Atom(symbol))
    for atom_index in range(fragment_size, len(symbols)):
        for neighbor_index in range(atom_index):
            if editable.GetBondBetweenAtoms(atom_index, neighbor_index) is not None:
                continue
            distance = float(np.linalg.norm(coordinates[atom_index] - coordinates[neighbor_index]))
            (left, right) = _ordered_symbols(str(symbols[atom_index]), str(symbols[neighbor_index]))
            order = get_bond_order(left, right, distance)
            if order > 0:
                editable.AddBond(atom_index, neighbor_index, const.BOND_DICT[int(order)])
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(symbols))
    for (atom_index, (x, y, z)) in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, Geometry.Point3D(float(x), float(y), float(z)))
    molecule.AddConformer(conformer, assignId=True)
    molecule.SetProp('_Name', candidate_id)
    return molecule

def molecule_quality(molecule: Chem.Mol) -> Tuple[bool, bool, str]:
    valid = True
    sanitized = Chem.Mol(molecule)
    try:
        sanitized = sanitized_complete_molecule(sanitized)
    except Exception:
        valid = False
    try:
        connected = len(Chem.GetMolFrags(molecule, sanitizeFrags=False)) == 1
    except Exception:
        connected = False
    try:
        if valid:
            from .fragmentation import canonical_isomeric_identity
            smiles = canonical_isomeric_identity(sanitized)
        else:
            smiles = Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        smiles = ''
    return (valid, connected, smiles)

def set_candidate_properties(molecule: Chem.Mol, row: Mapping[str, Any]) -> None:
    for key in CSV_FIELDS:
        molecule.SetProp(key, str(row[key]))

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

def shard_directory(run_root: Path, shard: ShardSpec) -> Path:
    return run_root / 'shards' / shard.shard_name

def validate_completed_shard(shard_dir: Path, shard: ShardSpec, contract_sha256: str, expected_samples: int) -> Optional[Dict[str, Any]]:
    if not shard_dir.exists():
        return None
    if not shard_dir.is_dir() or shard_dir.is_symlink():
        raise ValueError(f'Shard path is not a real directory: {shard_dir}')
    done_path = shard_dir / 'done.json'
    if not done_path.is_file():
        raise FileExistsError(f'Refusing to replace an incomplete existing shard: {relative(shard_dir)}')
    done = load_json(done_path)
    expected_values = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'shard_index': shard.shard_index, 'fragmentation_id': shard.context.fragmentation_id, 'linker_size': shard.linker_size, 'sample_count': expected_samples, 'contract_sha256': contract_sha256, 'shard_seed': shard.shard_seed}
    for (key, expected) in expected_values.items():
        if done.get(key) != expected:
            raise ValueError(f'Completed shard metadata mismatch for {relative(shard_dir)}: {key}={done.get(key)!r}, expected={expected!r}')
    outputs = done.get('outputs')
    if not isinstance(outputs, Mapping):
        raise ValueError(f'Shard done.json has no outputs mapping: {relative(done_path)}')
    for name in ('molecules.sdf', 'samples.csv', 'coordinates.npz'):
        path = shard_dir / name
        record = outputs.get(name)
        if not path.is_file() or not isinstance(record, Mapping):
            raise ValueError(f'Completed shard output is missing: {relative(path)}')
        if int(record.get('bytes', -1)) != path.stat().st_size:
            raise ValueError(f'Completed shard byte count changed: {relative(path)}')
        if record.get('sha256') != sha256_file(path):
            raise ValueError(f'Completed shard hash changed: {relative(path)}')
        if name.endswith('.npz'):
            with np.load(path, allow_pickle=False) as saved:
                observed_records = len(saved['candidate_ids'])
                if saved['coordinates'].shape[0] != observed_records or not np.isfinite(saved['coordinates']).all():
                    raise ValueError('Saved full-precision coordinates are malformed')
        else:
            observed_records = count_sdf_records(path) if name.endswith('.sdf') else count_csv_records(path)
        if int(record.get('records', -1)) != observed_records:
            raise ValueError(f'Completed shard record metadata changed: {relative(path)}')
        if observed_records != expected_samples:
            raise ValueError(f'Completed shard record count changed: {relative(path)}')
    return done

def generate_shard(model: DDPM, device: torch.device, config: GenerationConfig, shard: ShardSpec, run_id: str, run_root: Path, contract_sha256: str, checkpoint_sha256: str) -> Dict[str, Any]:
    final_dir = shard_directory(run_root, shard)
    completed = validate_completed_shard(final_dir, shard, contract_sha256, config.samples_per_context)
    if completed is not None:
        print(f'shard={shard.shard_index:04d} status=skipped fragmentation={shard.context.fragmentation_id} size={shard.linker_size}', flush=True)
        return completed
    shards_root = run_root / 'shards'
    shards_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{shard.shard_name}.tmp.', dir=str(shards_root)))
    sdf_path = temporary / 'molecules.sdf'
    csv_path = temporary / 'samples.csv'
    writer = sdf_path.open('w', encoding='utf-8')
    coordinate_path = temporary / 'coordinates.npz'
    (coordinate_rows, symbol_rows, candidate_ids) = ([], [], [])
    rows: List[Dict[str, Any]] = []
    retry_total = 0
    try:
        for batch_start in range(0, config.samples_per_context, config.batch_size):
            batch_count = min(config.batch_size, config.samples_per_context - batch_start)
            cpu_batch = collate([shard.context.item] * batch_count)
            data = move_batch_to_device(cpu_batch, device)

            def fixed_size(_data: Mapping[str, Any]) -> torch.Tensor:
                return torch.full((_data['positions'].shape[0],), fill_value=shard.linker_size, dtype=const.TORCH_INT, device=device)
            chain = None
            node_mask = None
            successful_attempt = None
            successful_batch_seed = None
            for attempt in range(config.max_nan_retries):
                batch_seed = derive_seed(shard.shard_seed, 'batch-v1', batch_start, attempt)
                seed_all(batch_seed)
                try:
                    with torch.inference_mode():
                        (chain, node_mask) = model.sample_chain(data, sample_fn=fixed_size, keep_frames=1)
                    successful_attempt = attempt
                    successful_batch_seed = batch_seed
                    break
                except FoundNaNException:
                    retry_total += 1
                    torch.cuda.empty_cache()
            if chain is None or node_mask is None:
                raise RuntimeError(f'Generation failed after {config.max_nan_retries} NaN attempts: shard={shard.shard_index}, batch_start={batch_start}')
            restored = restore_fragment_frame(chain_final=chain[0], node_mask=node_mask, input_positions=data['positions'], fragment_only_mask=data['fragment_only_mask'])
            generated_pocket_mask = extend_pocket_mask(data['pocket_mask'], node_mask)
            for batch_index in range(batch_count):
                replicate = batch_start + batch_index
                candidate_id = f'{run_id}-{shard.shard_index:04d}-{replicate:03d}'
                (coordinates, symbols) = ligand_features(model=model, restored=restored, node_mask=node_mask, pocket_mask=generated_pocket_mask, batch_index=batch_index)
                expected_atoms = shard.context.fragment.GetNumAtoms() + shard.linker_size
                if len(symbols) != expected_atoms:
                    raise RuntimeError(f'Pocket exclusion or linker sizing failed for {candidate_id}: atoms={len(symbols)}, expected={expected_atoms}')
                coordinate_rows.append(coordinates.copy())
                symbol_rows.append(list(symbols))
                candidate_ids.append(candidate_id)
                postprocessing_error = ''
                try:
                    molecule = build_candidate_molecule(fragment=shard.context.fragment, coordinates=coordinates, symbols=symbols, candidate_id=candidate_id)
                    (valid, connected, smiles) = molecule_quality(molecule)
                    if not valid:
                        try:
                            sanitized_complete_molecule(molecule)
                        except Exception as exc:
                            postprocessing_error = f'sanitize:{type(exc).__name__}:{str(exc)}'
                except Exception as exc:
                    molecule = Chem.Mol()
                    molecule.SetProp('_Name', candidate_id)
                    (valid, connected, smiles) = (False, False, '')
                    postprocessing_error = f'assembly:{type(exc).__name__}:{str(exc)}'
                sanitized = sanitized_complete_molecule(molecule) if valid else None
                diagnostics = attachment_diagnostics(shard.context.fragment, molecule, (int(shard.context.row['anchor_1']), int(shard.context.row['anchor_2'])), sanitized=sanitized)
                row: Dict[str, Any] = {**diagnostics, 'candidate_id': candidate_id, 'run_id': run_id, 'shard_index': shard.shard_index, 'context_index': shard.context_index, 'replicate': replicate, 'source_uuid': int(shard.context.row['uuid']), 'fragmentation_id': shard.context.fragmentation_id, 'structure_id': shard.context.structure_id, 'kinase': shard.context.row['kinase'], 'pdb': shard.context.row['pdb'].lower(), 'anchor_1': int(shard.context.row['anchor_1']), 'anchor_2': int(shard.context.row['anchor_2']), 'fragments_size': shard.context.fragment.GetNumAtoms(), 'linker_size': shard.linker_size, 'num_atoms': len(symbols), 'shard_seed': shard.shard_seed, 'batch_seed': successful_batch_seed, 'nan_retries': successful_attempt, 'valid': int(valid), 'connected': int(connected), 'smiles': smiles, 'in_memory_sanitize_pass': int(valid), 'postprocessing_error': ' '.join(postprocessing_error.split())}
                set_candidate_properties(molecule, row)
                writer.write(sdf_record(molecule, valid=valid, properties=row))
                rows.append(row)
            del chain, node_mask, restored, generated_pocket_mask, data, cpu_batch
        writer.close()
        writer = None
        np.savez_compressed(coordinate_path, coordinates=np.stack(coordinate_rows), symbols=np.asarray(symbol_rows), candidate_ids=np.asarray(candidate_ids))
        fsync_file(coordinate_path)
        if len(rows) != config.samples_per_context:
            raise RuntimeError(f'Shard candidate count mismatch: {len(rows)} != {config.samples_per_context}')
        fsync_file(sdf_path)
        write_csv(csv_path, rows)
        sdf_records = count_sdf_records(sdf_path)
        csv_records = count_csv_records(csv_path)
        if sdf_records != config.samples_per_context or csv_records != len(rows):
            raise RuntimeError(f'Shard serialization count mismatch: SDF={sdf_records}, CSV={csv_records}, expected={len(rows)}')
        outputs = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_file(path), 'records': sdf_records if path.suffix == '.sdf' else csv_records} for path in (sdf_path, csv_path, coordinate_path)}
        done = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'completed_at_utc': utc_now(), 'run_id': run_id, 'shard_index': shard.shard_index, 'context_index': shard.context_index, 'source_uuid': int(shard.context.row['uuid']), 'fragmentation_id': shard.context.fragmentation_id, 'structure_id': shard.context.structure_id, 'kinase': shard.context.row['kinase'], 'pdb': shard.context.row['pdb'].lower(), 'linker_size': shard.linker_size, 'sample_count': len(rows), 'valid_count': sum((int(row['valid']) for row in rows)), 'connected_count': sum((int(row['connected']) for row in rows)), 'shard_seed': shard.shard_seed, 'nan_retry_count': retry_total, 'postprocessing_error_count': sum((bool(row['postprocessing_error']) for row in rows)), 'anchor_attachment_pass_count': sum((int(row['anchor_attachment_pass']) for row in rows)), 'retained_stereo_failure_count': sum((row['retained_stereo_pass'] == 0 for row in rows)), 'coordinate_precision': 'native sampled tensors cast losslessly to float64 before SDF serialization', 'software': {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'rdkit': __import__('rdkit').__version__}, 'runtime': {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'gpu_capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}, 'contract_sha256': contract_sha256, 'checkpoint_sha256': checkpoint_sha256, 'outputs': outputs}
        atomic_write_json(temporary / 'done.json', done)
        fsync_directory(temporary)
        os.replace(temporary, final_dir)
        fsync_directory(shards_root)
        print(f"shard={shard.shard_index:04d} status=complete fragmentation={shard.context.fragmentation_id} size={shard.linker_size} samples={len(rows)} valid={done['valid_count']} connected={done['connected_count']}", flush=True)
        return done
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

def ensure_run_contract(run_root: Path, run_id: str, contract: Mapping[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / '.request.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _ensure_run_contract_locked(run_root, run_id, contract)

def _ensure_run_contract_locked(run_root, run_id, contract):
    contract_path = run_root / 'contract.json'
    if contract_path.exists():
        if load_json(contract_path) != dict(contract):
            raise ValueError('Frozen generation contract differs')
    else:
        atomic_write_json(contract_path, contract)
    request_path = run_root / 'request.json'
    expected = {'schema_version': SCHEMA_VERSION, 'run_id': run_id, 'contract_sha256': contract['contract_sha256'], 'config': contract['config'], 'config_sha256': contract['config_sha256'], 'checkpoint': contract['checkpoint'], 'expected_candidates': contract['expected_candidates'], 'condition_shards': contract['condition_shards']}
    if request_path.exists():
        observed = load_json(request_path)
        for (key, value) in expected.items():
            if observed.get(key) != value:
                raise ValueError(f'Run ID {run_id!r} is already bound to a different generation contract ({key} mismatch)')
        return
    request = {**expected, 'created_at_utc': utc_now()}
    atomic_write_json(request_path, request)

def combined_is_valid(run_root: Path, contract_sha256: str, expected_candidates: int) -> bool:
    done_path = run_root / 'done.json'
    if not done_path.exists():
        return False
    done = load_json(done_path)
    expected = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'contract_sha256': contract_sha256, 'candidate_count': expected_candidates}
    for (key, value) in expected.items():
        if done.get(key) != value:
            raise ValueError(f'Combined generation done.json mismatch: {key}')
    outputs = done.get('outputs')
    if not isinstance(outputs, Mapping):
        raise ValueError('Combined done.json is missing outputs')
    for name in ('candidates.sdf', 'candidates.csv', 'provenance.json'):
        path = run_root / name
        record = outputs.get(name)
        if not path.is_file() or not isinstance(record, Mapping):
            raise ValueError(f'Combined output is missing: {relative(path)}')
        if int(record.get('bytes', -1)) != path.stat().st_size:
            raise ValueError(f'Combined output byte count changed: {relative(path)}')
        if record.get('sha256') != sha256_file(path):
            raise ValueError(f'Combined output hash changed: {relative(path)}')
        if name == 'candidates.sdf':
            observed_records = count_sdf_records(path)
            if int(record.get('records', -1)) != observed_records:
                raise ValueError('Combined SDF record metadata changed')
            if observed_records != expected_candidates:
                raise ValueError('Combined SDF record count changed')
        elif name == 'candidates.csv':
            observed_records = count_csv_records(path)
            if int(record.get('records', -1)) != observed_records:
                raise ValueError('Combined CSV record metadata changed')
            if observed_records != expected_candidates:
                raise ValueError('Combined CSV record count changed')
    return True

def publish_combined_file(temporary: Path, final: Path) -> None:
    fsync_file(temporary)
    os.replace(temporary, final)
    fsync_directory(final.parent)

def consolidate_if_complete(run_root: Path, run_id: str, config: GenerationConfig, shards: Sequence[ShardSpec], contract: Mapping[str, Any], checkpoint_info: Mapping[str, Any]) -> bool:
    expected_candidates = len(shards) * config.samples_per_context
    if combined_is_valid(run_root, str(contract['contract_sha256']), expected_candidates):
        print('Combined generation outputs are already complete and verified.', flush=True)
        return True
    completed: List[Tuple[ShardSpec, Dict[str, Any]]] = []
    for shard in shards:
        done = validate_completed_shard(shard_directory(run_root, shard), shard, str(contract['contract_sha256']), config.samples_per_context)
        if done is not None:
            completed.append((shard, done))
    if len(completed) != len(shards):
        print(f'Generation progress: {len(completed)}/{len(shards)} condition shards complete; combined outputs not published yet.', flush=True)
        return False
    sdf_temporary = run_root / f'.candidates.sdf.tmp.{os.getpid()}'
    csv_temporary = run_root / f'.candidates.csv.tmp.{os.getpid()}'
    provenance_temporary = run_root / f'.provenance.json.tmp.{os.getpid()}'
    try:
        candidate_count = 0
        valid_count = 0
        connected_count = 0
        nan_retry_count = 0
        with sdf_temporary.open('wb') as output_sdf, csv_temporary.open('w', encoding='utf-8', newline='') as output_csv:
            csv_writer = csv.DictWriter(output_csv, fieldnames=CSV_FIELDS, lineterminator='\n')
            csv_writer.writeheader()
            for (shard, done) in completed:
                shard_dir = shard_directory(run_root, shard)
                with (shard_dir / 'molecules.sdf').open('rb') as input_sdf:
                    shutil.copyfileobj(input_sdf, output_sdf)
                with (shard_dir / 'samples.csv').open('r', encoding='utf-8', newline='') as input_csv:
                    reader = csv.DictReader(input_csv)
                    if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                        raise ValueError(f"Shard CSV schema changed: {relative(shard_dir / 'samples.csv')}")
                    rows_in_shard = 0
                    for row in reader:
                        csv_writer.writerow(row)
                        rows_in_shard += 1
                if rows_in_shard != config.samples_per_context:
                    raise ValueError(f'Shard CSV row count changed: {relative(shard_dir)}')
                candidate_count += rows_in_shard
                valid_count += int(done['valid_count'])
                connected_count += int(done['connected_count'])
                nan_retry_count += int(done['nan_retry_count'])
            output_sdf.flush()
            os.fsync(output_sdf.fileno())
            output_csv.flush()
            os.fsync(output_csv.fileno())
        if candidate_count != expected_candidates:
            raise RuntimeError(f'Combined candidate count mismatch: {candidate_count} != {expected_candidates}')
        combined_sdf_records = count_sdf_records(sdf_temporary)
        combined_csv_records = count_csv_records(csv_temporary)
        if combined_sdf_records != expected_candidates or combined_csv_records != expected_candidates:
            raise RuntimeError(f'Combined serialization count mismatch: SDF={combined_sdf_records}, CSV={combined_csv_records}, expected={expected_candidates}')
        upstream_path = VENDOR_ROOT / 'UPSTREAM_COMMIT'
        provenance = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'completed_at_utc': utc_now(), 'run_id': run_id, 'contract': dict(contract), 'contract_sha256': contract['contract_sha256'], 'checkpoint': dict(checkpoint_info), 'candidate_count': candidate_count, 'condition_shards': len(shards), 'valid_count': valid_count, 'connected_count': connected_count, 'nan_retry_count': nan_retry_count, 'software': {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'rdkit': getattr(Chem, '__version__', None) or __import__('rdkit').__version__, 'difflinker_upstream_commit': upstream_path.read_text(encoding='utf-8').strip() if upstream_path.is_file() else None, 'generation_source': relative(Path(__file__)), 'generation_source_sha256': sha256_file(Path(__file__))}, 'runtime': {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cuda_device_count': torch.cuda.device_count(), 'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, 'determinism': {'base_seed': config.seed, 'seed_derivation': 'SHA-256(base seed, fragmentation ID, linker size, batch start, retry)', 'algorithms_enabled': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'policy': DETERMINISM_POLICY}, 'output_policy': {'pocket_atoms_in_candidate_outputs': False, 'coordinate_frame': 'Generated ligand coordinates translated back by the original retained-fragment center of mass.', 'fragment_bonds': 'Copied from verified native-parent fragment reconstruction.', 'fragment_properties': 'Native atom identity, isotope, formal charge, explicit hydrogens and source-defined stereo retained.', 'stereochemistry': 'Retained-source neighbor-order parity; generated atom stereo from sampled 3D coordinates; unresolved retained correspondence recorded.', 'coordinates_npz': 'Every sampled ligand coordinate and element retained at native tensor precision in each shard.', 'diagnostics': 'Declared-anchor topology and retained nonanchor explicit/total hydrogen counts recorded for every attempt.', 'generated_bonds': 'Distance-inferred with the vendored DiffLinker bond thresholds for bonds involving at least one generated linker atom.'}, 'shards': [{'shard_index': shard.shard_index, 'path': relative(shard_directory(run_root, shard)), 'done_sha256': sha256_file(shard_directory(run_root, shard) / 'done.json')} for (shard, _) in completed]}
        provenance_text = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'
        with provenance_temporary.open('w', encoding='utf-8', newline='') as handle:
            handle.write(provenance_text)
            handle.flush()
            os.fsync(handle.fileno())
        publish_combined_file(sdf_temporary, run_root / 'candidates.sdf')
        publish_combined_file(csv_temporary, run_root / 'candidates.csv')
        publish_combined_file(provenance_temporary, run_root / 'provenance.json')
        outputs = {name: {'bytes': (run_root / name).stat().st_size, 'sha256': sha256_file(run_root / name), **({'records': candidate_count} if name in {'candidates.sdf', 'candidates.csv'} else {})} for name in ('candidates.sdf', 'candidates.csv', 'provenance.json')}
        done = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'completed_at_utc': utc_now(), 'run_id': run_id, 'contract_sha256': contract['contract_sha256'], 'condition_shards': len(shards), 'candidate_count': candidate_count, 'valid_count': valid_count, 'connected_count': connected_count, 'nan_retry_count': nan_retry_count, 'outputs': outputs}
        atomic_write_json(run_root / 'done.json', done)
        print(f'Generation complete: shards={len(shards)} candidates={candidate_count} valid={valid_count} connected={connected_count}', flush=True)
        return True
    finally:
        sdf_temporary.unlink(missing_ok=True)
        csv_temporary.unlink(missing_ok=True)
        provenance_temporary.unlink(missing_ok=True)

def preflight_report(config: GenerationConfig, contexts: Sequence[GenerationContext], shards: Sequence[ShardSpec], checkpoint_info: Mapping[str, Any], data_hashes: Mapping[str, str], contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {'status': 'preflight_passed', 'preflight_device': 'cpu', 'config': relative(config.path), 'config_sha256': sha256_file(config.path), 'checkpoint': dict(checkpoint_info), 'dataset_hashes': dict(data_hashes), 'selection': {'split': 'test', 'target_only': True, 'representative_structures': list(config.representative_structures), 'target_pdbs': list(config.target_pdbs), 'fragment_rows': len(contexts), 'rows': selected_rows_report(contexts)}, 'protocol': {'condition_shards': len(shards), 'linker_sizes': list(config.linker_sizes), 'samples_per_condition': config.samples_per_context, 'expected_candidates': len(shards) * config.samples_per_context, 'n_steps': config.n_steps, 'batch_size': config.batch_size, 'base_seed': config.seed}, 'leakage_controls': {'all_generation_rows_are_test': all((context.row['split'] == 'test' for context in contexts)), 'checkpoint_train_prefix': checkpoint_info['train_data_prefix'], 'checkpoint_validation_prefix': checkpoint_info['val_data_prefix'], 'test_prefix_absent_from_checkpoint_fit': True}, 'contract_sha256': contract['contract_sha256'], 'writes_performed': False}

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/generation.yml')
    parser.add_argument('--selection', help='Hash-verified HPO selection manifest, relative to project root')
    parser.add_argument('--checkpoint', required=True, help='Selected HPO checkpoint, as a path relative to the project root')
    parser.add_argument('--run-id', help='Stable output identity. If omitted, it is deterministically derived from the checkpoint filename and SHA-256.')
    parser.add_argument('--shard-index', type=int, help='Generate exactly one 0-based condition shard; default processes all shards')
    parser.add_argument('--preflight-only', action='store_true', help='Validate the complete generation contract on CPU without writing outputs')
    parser.add_argument('--initialize-only', action='store_true', help='Freeze the generation request on CPU before submitting an array')
    parser.add_argument('--finalize-only', action='store_true', help='Validate all condition shards and publish combined outputs on CPU')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = parse_args(argv)
    disable_rdkit_logging()
    config = read_config(args.config)
    paths = dataset_paths(config)
    data_hashes = verify_dataset_hashes(paths)
    targets = validate_target_registry(config)
    selected_rows = load_selected_rows(config, paths['table'], targets)
    contexts = load_contexts(config, paths, selected_rows)
    shards = build_shards(config, contexts)
    checkpoint = project_path(args.checkpoint, must_exist=True)
    checkpoint_payload = load_checkpoint_payload(checkpoint)
    checkpoint_info = validate_checkpoint(checkpoint_payload, checkpoint, config)
    if args.selection:
        selection_path = project_path(args.selection, must_exist=True)
        selection = load_json(selection_path)
        if selection.get('status') != 'complete' or selection['winner']['selected_checkpoint_sha256'] != checkpoint_info['sha256']:
            raise ValueError('Checkpoint is not the frozen HPO winner')
        checkpoint_info['hpo_selection'] = {'path': relative(selection_path), 'sha256': sha256_file(selection_path), 'trial_index': selection['winner']['trial_index'], 'objective': selection['winner']['objective']}
    contract = build_contract(config=config, contexts=contexts, checkpoint_info=checkpoint_info, data_hashes=data_hashes)
    if args.shard_index is not None and (not 0 <= args.shard_index < len(shards)):
        raise ValueError(f'--shard-index must be in 0..{len(shards) - 1}, found {args.shard_index}')
    if args.preflight_only:
        report = preflight_report(config=config, contexts=contexts, shards=shards, checkpoint_info=checkpoint_info, data_hashes=data_hashes, contract=contract)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    run_id = validate_run_id(args.run_id or derive_run_id(checkpoint, str(checkpoint_info['sha256'])))
    run_root = config.output_root / run_id
    ensure_run_contract(run_root, run_id, contract)
    if args.initialize_only:
        return 0
    if args.finalize_only:
        with (run_root / '.finalize.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            complete = consolidate_if_complete(run_root, run_id, config, shards, contract, checkpoint_info)
        return 0 if complete else 2
    device = validate_single_visible_gpu()
    configure_torch_determinism(warn_only=config.deterministic_warn_only)
    seed_all(config.seed)
    selected_shards = [shards[args.shard_index]] if args.shard_index is not None else list(shards)
    pending = [shard for shard in selected_shards if validate_completed_shard(shard_directory(run_root, shard), shard, str(contract['contract_sha256']), config.samples_per_context) is None]
    if pending:
        model = instantiate_model(payload=checkpoint_payload, config=config, device=device)
        del checkpoint_payload
        for shard in selected_shards:
            generate_shard(model=model, device=device, config=config, shard=shard, run_id=run_id, run_root=run_root, contract_sha256=str(contract['contract_sha256']), checkpoint_sha256=str(checkpoint_info['sha256']))
    else:
        print('Requested shard set is already complete and verified.', flush=True)
    if args.shard_index is not None:
        return 0
    consolidate_if_complete(run_root=run_root, run_id=run_id, config=config, shards=shards, contract=contract, checkpoint_info=checkpoint_info)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
