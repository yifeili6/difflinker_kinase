from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from kinase_gap.fragmentation import MMPA_PATTERN, build_coordinate_components, enumerate_two_cut_mmpa
from kinase_gap.mol2 import SUPPORTED_ELEMENTS, pocket_from_protein_mol2
PREFIX = {'train': 'KLIFS_train', 'validation': 'KLIFS_val', 'test': 'KLIFS_test'}

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def root_path(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f'{label} must be relative to the project root: {relative}')
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'{label} escapes the project root: {relative}') from exc
    return resolved

def sha256_file(path: Path, block_size: int=4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()

def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def normalize_alt(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    text = str(value).strip()
    return '-' if not text or text.lower() == 'nan' else text

def normalize_species(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith('HUMAN'):
        return 'HUMAN'
    if text.startswith('MOUSE'):
        return 'MOUSE'
    return text

def safe_id(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '-', value.strip()).strip('-') or 'unknown'

def parse_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return result

def load_ligand(path: Path) -> Tuple[Chem.Mol, str]:
    mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=True)
    mode = 'sanitized'
    if mol is None:
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError('RDKit could not parse MOL2')
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mode = 'sanitize_fallback'
    else:
        mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() != 1:
        raise ValueError(f'Expected one ligand conformer, found {mol.GetNumConformers()}')
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if coords.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError('Ligand has missing or non-finite coordinates')
    return (mol, mode)

def rejection(ledger: List[Dict[str, object]], stage: str, reason: str, structure_id: str='', detail: object='', fragmentation_id: str='') -> None:
    ledger.append({'stage': stage, 'reason': reason, 'structure_id': structure_id, 'fragmentation_id': fragmentation_id, 'detail': str(detail).replace('\n', ' ')[:1000]})

def verify_snapshot(snapshot: Path, config: Mapping[str, object]) -> Dict[str, object]:
    expected = config['snapshot']
    files = {'archive': snapshot / 'KLIFS_download.zip', 'overview': snapshot / 'overview.csv', 'export': snapshot / 'KLIFS_export.csv'}
    hashes = {}
    for (key, path) in files.items():
        if not path.is_file():
            raise FileNotFoundError(f'Snapshot is missing {path.name}')
        hashes[key] = sha256_file(path)
        wanted = str(expected[f'{key}_sha256'])
        if hashes[key] != wanted:
            raise ValueError(f'Snapshot hash mismatch for {path.name}: {hashes[key]} != {wanted}')
    return {'snapshot_name': snapshot.name, 'database_export_date': str(expected['database_export_date']), 'local_cache_date': str(expected['local_cache_date']), 'files': {key: {'path': path.name, 'sha256': hashes[key], 'bytes': path.stat().st_size} for (key, path) in files.items()}}

def curate_contexts(root: Path, snapshot: Path, config: Mapping[str, object], ledger: List[Dict[str, object]]) -> List[Dict[str, object]]:
    overview = pd.read_csv(snapshot / 'overview.csv', keep_default_na=False)
    overview = overview.sort_values(['species', 'kinase', 'pdb', 'alt', 'chain'], kind='mergesort')
    allowed_species = set(config['species'])
    allowed_elements = set(config['allowed_elements'])
    forced_test_pdbs = {str(value).lower() for value in config['split']['forced_test_pdbs']}
    accepted: List[Dict[str, object]] = []
    seen_ids = set()
    for row in overview.to_dict(orient='records'):
        species = normalize_species(row['species'])
        if species not in allowed_species:
            continue
        kinase = str(row['kinase']).strip()
        pdb = str(row['pdb']).strip().lower()
        alt = normalize_alt(row['alt'])
        chain = str(row['chain']).strip()
        alt_id = '-' if alt == '-' else safe_id(alt)
        structure_id = '__'.join([species.lower(), safe_id(kinase), safe_id(pdb), 'alt' + alt_id, 'chain' + safe_id(chain)])
        if structure_id in seen_ids:
            rejection(ledger, 'metadata', 'duplicate_structure_id', structure_id)
            continue
        seen_ids.add(structure_id)
        ligand_code = str(row.get('orthosteric_PDB', '')).strip()
        if not ligand_code or ligand_code == '-':
            rejection(ledger, 'metadata', 'no_orthosteric_ligand', structure_id)
            continue
        quality = parse_float(row.get('qualityscore'))
        resolution = parse_float(row.get('resolution'))
        quality_failed = not np.isfinite(quality) or quality < float(config['quality_score_min'])
        resolution_failed = not np.isfinite(resolution) or resolution > float(config['resolution_max_angstrom'])
        target_metadata_exception = pdb in forced_test_pdbs and bool(config.get('allow_forced_test_metadata_exceptions', False)) and (quality_failed or resolution_failed)
        if quality_failed and (not target_metadata_exception):
            rejection(ledger, 'metadata', 'quality_below_threshold', structure_id, quality)
            continue
        if resolution_failed and (not target_metadata_exception):
            rejection(ledger, 'metadata', 'resolution_above_threshold', structure_id, resolution)
            continue
        directory = snapshot / species / kinase / f'{pdb}_alt{alt}_chain{chain}'
        ligand_path = directory / 'ligand.mol2'
        protein_path = directory / 'protein.mol2'
        pocket_path = directory / 'pocket.mol2'
        missing = [path.name for path in (ligand_path, protein_path, pocket_path) if not path.is_file()]
        if missing:
            rejection(ledger, 'coordinates', 'missing_coordinate_file', structure_id, ','.join(missing))
            continue
        try:
            (mol, parse_mode) = load_ligand(ligand_path)
        except Exception as exc:
            rejection(ledger, 'ligand', 'ligand_parse_failed', structure_id, exc)
            continue
        elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
        unsupported = sorted(set(elements) - allowed_elements)
        if unsupported:
            rejection(ledger, 'ligand', 'unsupported_element', structure_id, ','.join(unsupported))
            continue
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms < int(config['heavy_atoms_min']) or heavy_atoms > int(config['heavy_atoms_max']):
            rejection(ledger, 'ligand', 'heavy_atom_count_outside_range', structure_id, heavy_atoms)
            continue
        rings = int(mol.GetRingInfo().NumRings())
        if rings < int(config['rings_min']):
            rejection(ledger, 'ligand', 'too_few_rings', structure_id, rings)
            continue
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchikey = Chem.inchi.MolToInchiKey(mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        if not scaffold:
            rejection(ledger, 'ligand', 'empty_murcko_scaffold', structure_id)
            continue
        accepted.append({'structure_id': structure_id, 'species': species, 'kinase': kinase, 'pdb': pdb, 'alternate_model': alt, 'chain': chain, 'ligand_code': ligand_code, 'qualityscore': quality, 'resolution_angstrom': resolution, 'target_metadata_exception': target_metadata_exception, 'canonical_smiles': canonical_smiles, 'inchikey': inchikey, 'murcko_scaffold': scaffold, 'elements': ';'.join(elements), 'heavy_atoms': heavy_atoms, 'rings': rings, 'formal_charge': int(Chem.GetFormalCharge(mol)), 'ligand_parse_mode': parse_mode, 'ligand_path': ligand_path.relative_to(root).as_posix(), 'protein_path': protein_path.relative_to(root).as_posix(), 'cached_pocket_path': pocket_path.relative_to(root).as_posix(), 'ligand_sha256': sha256_file(ligand_path)})
    return accepted

class UnionFind:

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        (a, b) = (self.find(left), self.find(right))
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def assign_splits(contexts: List[Dict[str, object]], config: Mapping[str, object]) -> None:
    split_cfg = config['split']
    uf = UnionFind(len(contexts))
    for key in split_cfg['group_keys']:
        first: Dict[object, int] = {}
        for (idx, context) in enumerate(contexts):
            value = context[key]
            if value in first:
                uf.union(idx, first[value])
            else:
                first[value] = idx
    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(contexts)):
        components[uf.find(idx)].append(idx)
    target_pdbs = {str(pdb).lower() for pdb in split_cfg['forced_test_pdbs']}
    assignments: Dict[int, str] = {}
    current = Counter()
    for (root_idx, members) in components.items():
        if any((str(contexts[idx]['pdb']).lower() in target_pdbs for idx in members)):
            assignments[root_idx] = 'test'
            current['test'] += len(members)
    total = len(contexts)
    desired = {'train': total * float(split_cfg['train_fraction']), 'validation': total * float(split_cfg['validation_fraction']), 'test': total * float(split_cfg['test_fraction'])}
    seed = str(split_cfg['seed'])
    pending = [item for item in components.items() if item[0] not in assignments]
    pending.sort(key=lambda item: (-len(item[1]), hashlib.sha256((seed + '|' + min((contexts[i]['structure_id'] for i in item[1]))).encode()).hexdigest()))
    order = ('train', 'validation', 'test')
    for (root_idx, members) in pending:
        scores = {name: (desired[name] - current[name]) / max(desired[name], 1.0) for name in order}
        choice = max(order, key=lambda name: (scores[name], -order.index(name)))
        assignments[root_idx] = choice
        current[choice] += len(members)
    for (root_idx, members) in components.items():
        member_ids = sorted((str(contexts[idx]['structure_id']) for idx in members))
        group_id = 'grp_' + hashlib.sha256('|'.join(member_ids).encode()).hexdigest()[:16]
        for idx in members:
            contexts[idx]['group_id'] = group_id
            contexts[idx]['split'] = assignments[root_idx]
            contexts[idx]['forced_target_group'] = assignments[root_idx] == 'test' and any((str(contexts[j]['pdb']).lower() in target_pdbs for j in members))
    assert_no_leakage(contexts, ('pdb', 'canonical_smiles', 'inchikey', 'murcko_scaffold'))

def assert_no_leakage(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> None:
    for key in keys:
        owners: Dict[object, str] = {}
        for row in rows:
            (value, split) = (row[key], str(row['split']))
            previous = owners.setdefault(value, split)
            if previous != split:
                raise AssertionError(f'Split leakage for {key}={value}: {previous} and {split}')

def set_name(mol: Chem.Mol, name: str) -> Chem.Mol:
    copy = Chem.Mol(mol)
    copy.SetProp('_Name', name)
    return copy

def expand_contexts(root: Path, contexts: List[Dict[str, object]], config: Mapping[str, object], build_dir: Path, ledger: List[Dict[str, object]]) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, int]]:
    writers = {}
    tables: Dict[str, List[Dict[str, object]]] = {name: [] for name in PREFIX}
    pockets: Dict[str, List[Dict[str, object]]] = {name: [] for name in PREFIX}
    for (split, prefix) in PREFIX.items():
        split_writers = {}
        for kind in ('mol', 'frag', 'link'):
            writer = Chem.SDWriter(str(build_dir / f'{prefix}_{kind}.sdf'))
            writer.SetKekulize(False)
            split_writers[kind] = writer
        writers[split] = split_writers
    context_fragment_counts: Dict[str, int] = {}
    ordered_contexts = sorted(contexts, key=lambda row: str(row['structure_id']))
    for (context_index, context) in enumerate(ordered_contexts, start=1):
        if context_index == 1 or context_index % 250 == 0:
            print(f'Expanding KLIFS context {context_index}/{len(ordered_contexts)}', flush=True)
        structure_id = str(context['structure_id'])
        split = str(context['split'])
        ligand_path = root / str(context['ligand_path'])
        protein_path = root / str(context['protein_path'])
        try:
            (mol, _) = load_ligand(ligand_path)
            pocket = pocket_from_protein_mol2(protein_path, mol.GetConformer().GetPositions(), float(config['pocket_cutoff_angstrom']), config['allowed_elements'])
        except Exception as exc:
            rejection(ledger, 'pocket', 'pocket_construction_failed', structure_id, exc)
            context_fragment_counts[structure_id] = 0
            continue
        fragmentations = enumerate_two_cut_mmpa(mol, min_fragment_atoms=int(config['mmpa']['minimum_fragment_atoms']), min_linker_atoms=int(config['mmpa']['minimum_linker_atoms']))
        if not fragmentations:
            rejection(ledger, 'mmpa', 'no_valid_two_cut_fragmentation', structure_id)
            context_fragment_counts[structure_id] = 0
            continue
        emitted = 0
        protein_sha = sha256_file(protein_path)
        for fragmentation in fragmentations:
            fragmentation_id = structure_id + '__mmpa_' + fragmentation.signature
            try:
                (fragments, linker, anchors) = build_coordinate_components(mol, fragmentation)
                if not np.isfinite(fragments.GetConformer().GetPositions()).all():
                    raise ValueError('Non-finite fragment coordinates')
                if not np.isfinite(linker.GetConformer().GetPositions()).all():
                    raise ValueError('Non-finite linker coordinates')
            except Exception as exc:
                rejection(ledger, 'fragment_coordinates', 'coordinate_transfer_failed', structure_id, exc, fragmentation_id)
                continue
            uuid = len(tables[split])
            named_mol = set_name(mol, fragmentation_id)
            named_fragments = set_name(fragments, fragmentation_id)
            named_linker = set_name(linker, fragmentation_id)
            writers[split]['mol'].write(named_mol)
            writers[split]['frag'].write(named_fragments)
            writers[split]['link'].write(named_linker)
            pockets[split].append(pocket)
            tables[split].append({'uuid': uuid, 'fragmentation_id': fragmentation_id, 'molecule_name': structure_id, 'structure_id': structure_id, 'split': split, 'group_id': context['group_id'], 'forced_target_group': context['forced_target_group'], 'target_metadata_exception': context['target_metadata_exception'], 'species': context['species'], 'kinase': context['kinase'], 'pdb': context['pdb'], 'alternate_model': context['alternate_model'], 'chain': context['chain'], 'ligand_code': context['ligand_code'], 'molecule': context['canonical_smiles'], 'canonical_smiles': context['canonical_smiles'], 'inchikey': context['inchikey'], 'murcko_scaffold': context['murcko_scaffold'], 'fragments': Chem.MolToSmiles(fragments, canonical=True, isomericSmiles=True), 'linker': Chem.MolToSmiles(linker, canonical=True, isomericSmiles=True), 'mmpa_fragments_with_dummies': fragmentation.fragments_smiles, 'mmpa_linker_with_dummies': fragmentation.linker_smiles, 'anchor_1': anchors[0], 'anchor_2': anchors[1], 'pocket_full_size': len(pocket['full_types']), 'pocket_bb_size': len(pocket['bb_types']), 'pocket_residue_count': pocket['residue_count'], 'pocket_cutoff_angstrom': pocket['cutoff_angstrom'], 'molecule_size': mol.GetNumAtoms(), 'fragments_size': fragments.GetNumAtoms(), 'linker_size': linker.GetNumAtoms(), 'ligand_sha256': context['ligand_sha256'], 'protein_sha256': protein_sha})
            emitted += 1
        context_fragment_counts[structure_id] = emitted
        if emitted == 0:
            rejection(ledger, 'fragment_coordinates', 'no_fragmentation_emitted', structure_id)
    for split in PREFIX:
        for writer in writers[split].values():
            writer.close()
        prefix = PREFIX[split]
        pd.DataFrame(tables[split]).to_csv(build_dir / f'{prefix}_table.csv', index=False)
        with (build_dir / f'{prefix}_pockets.pkl').open('wb') as handle:
            pickle.dump(pockets[split], handle, protocol=4)
    return (tables, context_fragment_counts)

def count_sdf(path: Path) -> int:
    return sum((mol is not None for mol in Chem.SDMolSupplier(str(path), sanitize=False)))

def validate_built_dataset(build_dir: Path, tables: Mapping[str, Sequence[Mapping[str, object]]]) -> Dict[str, object]:
    report = {'splits': {}, 'leakage': {}}
    all_rows: List[Mapping[str, object]] = []
    allowed = set(SUPPORTED_ELEMENTS)
    for (split, prefix) in PREFIX.items():
        rows = tables[split]
        expected = len(rows)
        counts = {kind: count_sdf(build_dir / f'{prefix}_{kind}.sdf') for kind in ('mol', 'frag', 'link')}
        with (build_dir / f'{prefix}_pockets.pkl').open('rb') as handle:
            pocket_rows = pickle.load(handle)
        if set(counts.values()) != {expected} or len(pocket_rows) != expected:
            raise AssertionError(f'Row alignment failed for {split}: {expected}, {counts}, pockets={len(pocket_rows)}')
        for row in rows:
            if not 0 <= int(row['anchor_1']) < int(row['fragments_size']):
                raise AssertionError(f"anchor_1 outside fragment for {row['fragmentation_id']}")
            if not 0 <= int(row['anchor_2']) < int(row['fragments_size']):
                raise AssertionError(f"anchor_2 outside fragment for {row['fragmentation_id']}")
            if int(row['anchor_1']) == int(row['anchor_2']):
                raise AssertionError(f"Duplicate anchors for {row['fragmentation_id']}")
        for pocket in pocket_rows:
            if set(pocket['full_types']) - allowed:
                raise AssertionError('Pocket contains an unsupported element')
            coords = np.asarray(pocket['full_coord'], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3 or (not np.isfinite(coords).all()):
                raise AssertionError('Pocket coordinates are malformed')
        report['splits'][split] = {'fragmentation_rows': expected, 'contexts': len({row['structure_id'] for row in rows}), 'pdbs': len({row['pdb'] for row in rows}), 'ligands': len({row['inchikey'] for row in rows}), 'scaffolds': len({row['murcko_scaffold'] for row in rows}), 'aligned_file_counts': counts}
        all_rows.extend(rows)
    assert_no_leakage(all_rows, ('pdb', 'canonical_smiles', 'inchikey', 'murcko_scaffold'))
    for key in ('pdb', 'canonical_smiles', 'inchikey', 'murcko_scaffold'):
        sets = {split: {row[key] for row in tables[split]} for split in PREFIX}
        report['leakage'][key] = {'train_validation': len(sets['train'] & sets['validation']), 'train_test': len(sets['train'] & sets['test']), 'validation_test': len(sets['validation'] & sets['test'])}
    return report

def write_hash_manifest(directory: Path, output: Path) -> None:
    rows = []
    for path in sorted((p for p in directory.iterdir() if p.is_file() and p != output)):
        rows.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    pd.DataFrame(rows).to_csv(output, index=False)

def run(root: Path, config_path: Path, force: bool=False) -> None:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if config['mmpa']['smarts'] != MMPA_PATTERN or int(config['mmpa']['cuts']) != 2:
        raise ValueError('The code and data contract disagree on the MMPA rule')
    snapshot = root_path(root, config['snapshot_dir'], 'snapshot_dir')
    output = root_path(root, config['output_dir'], 'output_dir')
    manifest_dir = root_path(root, config['manifest_dir'], 'manifest_dir')
    if output.exists() and any(output.iterdir()) and (not force):
        raise FileExistsError(f'Output is not empty: {output}. Use --force to rebuild it deliberately.')
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_manifest = verify_snapshot(snapshot, config)
    ledger: List[Dict[str, object]] = []
    contexts = curate_contexts(root, snapshot, config, ledger)
    if not contexts:
        raise RuntimeError('No KLIFS contexts passed curation')
    print(f'Curated {len(contexts)} KLIFS contexts; assigning leakage-safe splits', flush=True)
    assign_splits(contexts, config)
    build_dir = Path(tempfile.mkdtemp(prefix='.klifs-v1-build-', dir=str(output.parent)))
    try:
        (tables, fragmentation_counts) = expand_contexts(root, contexts, config, build_dir, ledger)
        for context in contexts:
            context['emitted_fragmentations'] = fragmentation_counts.get(str(context['structure_id']), 0)
        validation = validate_built_dataset(build_dir, tables)
        contexts_df = pd.DataFrame(contexts).sort_values('structure_id')
        contexts_df.to_csv(manifest_dir / 'contexts.csv', index=False)
        contexts_df[['structure_id', 'split', 'group_id', 'forced_target_group', 'target_metadata_exception', 'pdb', 'canonical_smiles', 'inchikey', 'murcko_scaffold', 'emitted_fragmentations']].to_csv(manifest_dir / 'splits.csv', index=False)
        target_pdbs = {str(pdb).lower() for pdb in config['split']['forced_test_pdbs']}
        contexts_df[contexts_df.pdb.str.lower().isin(target_pdbs)].to_csv(manifest_dir / 'targets.csv', index=False)
        pd.DataFrame(ledger).sort_values(['stage', 'reason', 'structure_id', 'fragmentation_id'], kind='mergesort').to_csv(manifest_dir / 'rejections.csv', index=False)
        snapshot_manifest['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        snapshot_manifest['configuration_sha256'] = sha256_file(config_path)
        with (manifest_dir / 'snapshot.json').open('w') as handle:
            json.dump(snapshot_manifest, handle, indent=2, sort_keys=True)
        report = {'dataset_version': config['dataset_version'], 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'configuration': config, 'configuration_sha256': sha256_file(config_path), 'accepted_contexts': len(contexts), 'accepted_contexts_by_split': dict(Counter((row['split'] for row in contexts))), 'forced_target_metadata_exceptions': sum((bool(row['target_metadata_exception']) for row in contexts)), 'emitted_fragmentations': sum((len(rows) for rows in tables.values())), 'rejections_by_stage_reason': {f'{stage}:{reason}': count for ((stage, reason), count) in sorted(Counter(((row['stage'], row['reason']) for row in ledger)).items())}, 'validation': validation}
        with (build_dir / 'dataset_report.json').open('w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        write_hash_manifest(build_dir, build_dir / 'files.sha256.csv')
        if force and output.exists():
            for existing in output.iterdir():
                if existing.is_dir():
                    shutil.rmtree(existing)
                else:
                    existing.unlink()
        for built in build_dir.iterdir():
            shutil.move(str(built), str(output / built.name))
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/data_klifs_v1.yml')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    RDLogger.DisableLog('rdApp.warning')
    root = project_root()
    config_path = root_path(root, args.config, 'config')
    run(root, config_path, force=args.force)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
