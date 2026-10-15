from __future__ import annotations
import csv
import hashlib
import base64
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from rdkit import Chem

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

@lru_cache(maxsize=8)
def _manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return {row['structure_id']: row for row in csv.DictReader(handle)}

def reconstruct_fragment(row, project_root):
    from .fragmentation import Fragmentation, build_coordinate_components
    from .prepare import load_ligand
    root = Path(project_root)
    context = _manifest(str(root / 'data/manifests/klifs_v1/contexts.csv'))[row['structure_id']]
    relative = Path(context['ligand_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Raw ligand path must be a contained project-relative path')
    path = root / relative
    expected = row.get('ligand_sha256') or context['ligand_sha256']
    if file_sha256(path) != expected:
        raise ValueError(f"Raw ligand hash mismatch: {row['structure_id']}")
    (parent, _) = load_ligand(path)
    decomposition = Fragmentation(row['mmpa_linker_with_dummies'], row['mmpa_fragments_with_dummies'], row['fragmentation_id'].split('__mmpa_')[-1])
    (fragment, _linker, anchors) = build_coordinate_components(parent, decomposition)
    if tuple(map(int, (row['anchor_1'], row['anchor_2']))) != tuple(anchors):
        raise ValueError(f"Raw fragment anchor order mismatch: {row['fragmentation_id']}")
    if fragment.GetNumAtoms() != int(row['fragments_size']):
        raise ValueError('Raw fragment atom count differs from the corrected table')
    fragment.SetProp('_Name', row['fragmentation_id'])
    return fragment

def sanitized_complete_molecule(molecule):
    result = Chem.Mol(molecule)
    Chem.SanitizeMol(result)
    if result.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(result, confId=0, replaceExistingTags=True)
    parent_indices = {atom.GetIntProp('_ParentAtomIndex'): atom.GetIdx() for atom in result.GetAtoms() if atom.HasProp('_ParentAtomIndex')}
    errors = []

    def substituted_neighbors(atom):
        original = [int(value) for value in atom.GetProp('_ParentNeighborOrder').split(',') if value]
        current = list(atom.GetNeighbors())
        replacements = [neighbor for neighbor in current if not neighbor.HasProp('_ParentAtomIndex')]
        missing = [index for index in original if index not in parent_indices]
        if len(replacements) != len(missing) or len(missing) > 1:
            raise ValueError('native cut-neighbor correspondence is not unique')
        observed = [neighbor.GetIntProp('_ParentAtomIndex') if neighbor.HasProp('_ParentAtomIndex') else missing[0] for neighbor in current]
        if set(observed) != set(original) or len(observed) != len(original):
            raise ValueError('native neighbor graph changed')
        translation = {source: neighbor.GetIdx() for (source, neighbor) in zip(observed, current)}
        return (original, observed, translation)
    for atom in result.GetAtoms():
        if not atom.HasProp('_ParentChiralTag'):
            continue
        native_tag = Chem.ChiralType(atom.GetIntProp('_ParentChiralTag'))
        atom.SetChiralTag(native_tag)
        if native_tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        try:
            (original, observed, _) = substituted_neighbors(atom)
            order = [original.index(value) for value in observed]
            if sum((order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))) % 2:
                atom.InvertChirality()
        except ValueError as exc:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            errors.append({'atom': atom.GetIdx(), 'reason': str(exc)})
    for bond in result.GetBonds():
        if not bond.HasProp('_ParentStereo'):
            continue
        bond.SetBondDir(Chem.BondDir(bond.GetIntProp('_ParentBondDir')))
        native_stereo = Chem.BondStereo(bond.GetIntProp('_ParentStereo'))
        old_refs = [int(value) for value in bond.GetProp('_ParentStereoAtoms').split(',') if value]
        try:
            if old_refs:
                translated = []
                for (source, endpoint) in zip(old_refs, (bond.GetBeginAtom(), bond.GetEndAtom())):
                    if source in parent_indices:
                        translated.append(parent_indices[source])
                    else:
                        translated.append(substituted_neighbors(endpoint)[2][source])
                bond.SetStereoAtoms(*translated)
            bond.SetStereo(native_stereo)
        except (ValueError, KeyError, RuntimeError) as exc:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            errors.append({'bond': [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()], 'reason': str(exc)})
    Chem.AssignStereochemistry(result, cleanIt=False, force=True)
    if parent_indices:
        result.SetProp('kinase_retained_stereo_errors', json.dumps(errors, separators=(',', ':')))
        result.SetProp('kinase_stereo_policy', 'retained source stereo with neighbor-order parity; generated atom stereo from sampled coordinates')
    return result

def checked_mol_block(molecule):
    if molecule.HasProp('kinase_native_stereo'):
        return checked_native_mol_block(restore_native_identity(molecule))
    result = sanitized_complete_molecule(molecule)
    if any((atom.HasProp('_ParentAtomIndex') for atom in result.GetAtoms())):
        return checked_native_mol_block(result)
    block = Chem.MolToMolBlock(result, includeStereo=True, confId=0, kekulize=True)
    loaded = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=True)
    if loaded is None:
        raise ValueError('Complete-molecule export failed RDKit readback')
    expected = Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)
    observed = Chem.MolToSmiles(loaded, canonical=True, isomericSmiles=True)
    if expected != observed:
        raise ValueError(f'Complete-molecule identity changed on SDF export: {expected} -> {observed}')
    if [a.GetSymbol() for a in result.GetAtoms()] != [a.GetSymbol() for a in loaded.GetAtoms()]:
        raise ValueError('Complete-molecule atom order changed on SDF export')
    if result.GetNumConformers():
        maximum = np.abs(result.GetConformer().GetPositions() - loaded.GetConformer().GetPositions()).max(initial=0.0)
        if maximum > 5.1e-05:
            raise ValueError('Complete-molecule coordinates changed beyond SDF precision')
    return (block, result)

def attachment_diagnostics(fragment, molecule, anchors, *, sanitized=None):
    size = fragment.GetNumAtoms()
    edges = []
    for bond in molecule.GetBonds():
        (left, right) = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if (left < size) != (right < size):
            if left >= size:
                (left, right) = (right, left)
            edges.append([left, right, str(bond.GetBondType())])
    (errors, hydrogen_changes) = ([], [])
    if sanitized is not None:
        errors = json.loads(sanitized.GetProp('kinase_retained_stereo_errors')) if sanitized.HasProp('kinase_retained_stereo_errors') else []
        for source in fragment.GetAtoms():
            if source.GetIdx() in anchors:
                continue
            observed = sanitized.GetAtomWithIdx(source.GetIdx())
            source_total = source.GetIntProp('_ParentTotalHs') if source.HasProp('_ParentTotalHs') else source.GetTotalNumHs()
            if source.GetNumExplicitHs() != observed.GetNumExplicitHs() or source_total != observed.GetTotalNumHs():
                hydrogen_changes.append({'atom': source.GetIdx(), 'element': source.GetSymbol(), 'source_explicit_h': source.GetNumExplicitHs(), 'candidate_explicit_h': observed.GetNumExplicitHs(), 'source_total_h': source_total, 'candidate_total_h': observed.GetTotalNumHs()})
    return {'anchor_attachment_pass': int(len(edges) == 2 and sorted((edge[0] for edge in edges)) == sorted(map(int, anchors))), 'fragment_linker_bond_count': len(edges), 'fragment_linker_edges_json': json.dumps(edges, separators=(',', ':')), 'retained_stereo_pass': int(not errors) if sanitized is not None else '', 'retained_stereo_errors_json': json.dumps(errors, separators=(',', ':')) if sanitized is not None else '', 'retained_nonanchor_explicit_h_match': int(all((item['source_explicit_h'] == item['candidate_explicit_h'] for item in hydrogen_changes))) if sanitized is not None else '', 'retained_nonanchor_total_h_match': int(all((item['source_total_h'] == item['candidate_total_h'] for item in hydrogen_changes))) if sanitized is not None else '', 'retained_nonanchor_h_changes_json': json.dumps(hydrogen_changes, separators=(',', ':')) if sanitized is not None else ''}

def native_stereo_record(molecule):
    return {'atom_chiral_tags': [int(atom.GetChiralTag()) for atom in molecule.GetAtoms()], 'atom_neighbor_order': [[neighbor.GetIdx() for neighbor in atom.GetNeighbors()] for atom in molecule.GetAtoms()], 'bonds': [{'begin': bond.GetBeginAtomIdx(), 'end': bond.GetEndAtomIdx(), 'stereo': int(bond.GetStereo()), 'stereo_atoms': list(bond.GetStereoAtoms()), 'bond_dir': int(bond.GetBondDir())} for bond in molecule.GetBonds()]}

def restore_native_identity(molecule):
    if not molecule.HasProp('kinase_native_stereo'):
        return molecule
    result = Chem.Mol(molecule)
    record = json.loads(result.GetProp('kinase_native_stereo'))
    if len(record['atom_chiral_tags']) != result.GetNumAtoms():
        raise ValueError('Native stereochemistry atom count mismatch')
    for (atom, tag) in zip(result.GetAtoms(), record['atom_chiral_tags']):
        atom.SetChiralTag(Chem.ChiralType(tag))
        if 'atom_neighbor_order' in record:
            original = record['atom_neighbor_order'][atom.GetIdx()]
            observed = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
            if set(original) != set(observed):
                raise ValueError('Native stereochemistry neighbor graph changed')
            order = [original.index(value) for value in observed]
            odd = sum((order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))) % 2
            if odd and atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                atom.InvertChirality()
    for item in record['bonds']:
        bond = result.GetBondBetweenAtoms(item['begin'], item['end'])
        if bond is None:
            raise ValueError('Native stereochemistry bond is absent')
        bond.SetBondDir(Chem.BondDir(item['bond_dir']))
        if item['stereo_atoms']:
            bond.SetStereoAtoms(*item['stereo_atoms'])
        bond.SetStereo(Chem.BondStereo(item['stereo']))
    Chem.AssignStereochemistry(result, cleanIt=False, force=True)
    from .fragmentation import canonical_isomeric_identity
    if result.HasProp('kinase_native_smiles') and canonical_isomeric_identity(result) != canonical_isomeric_identity(Chem.MolFromSmiles(result.GetProp('kinase_native_smiles'))):
        raise ValueError('Native stereochemistry metadata did not restore source identity')
    return result

def checked_native_mol_block(molecule):
    result = Chem.Mol(molecule)
    Chem.SanitizeMol(result)
    result.SetProp('kinase_native_smiles', Chem.MolToSmiles(result))
    result.SetProp('kinase_native_stereo', json.dumps(native_stereo_record(result), separators=(',', ':')))
    if result.HasProp('kinase_native_molecule_binary_base64'):
        result.ClearProp('kinase_native_molecule_binary_base64')
    binary = result.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result.SetProp('kinase_native_molecule_binary_base64', base64.b64encode(binary).decode())
    block = Chem.MolToMolBlock(result, includeStereo=True, confId=0, kekulize=True)
    loaded = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=True)
    if loaded is None or Chem.MolToSmiles(loaded, isomericSmiles=False) != Chem.MolToSmiles(result, isomericSmiles=False):
        raise ValueError('Native ligand constitutional identity changed on SDF export')
    for key in ('kinase_native_smiles', 'kinase_native_stereo'):
        loaded.SetProp(key, result.GetProp(key))
    restored = restore_native_identity(loaded)
    from .fragmentation import canonical_isomeric_identity
    if canonical_isomeric_identity(restored) != canonical_isomeric_identity(result):
        raise ValueError('Native ligand isomeric identity did not survive stereo metadata roundtrip')
    atom_key = lambda atom: (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetTotalNumHs())
    if [atom_key(a) for a in restored.GetAtoms()] != [atom_key(a) for a in result.GetAtoms()]:
        raise ValueError('Native atom order or chemistry changed on SDF export')
    if not np.allclose(restored.GetConformer().GetPositions(), result.GetConformer().GetPositions(), atol=5.1e-05, rtol=0):
        raise ValueError('Native coordinates changed beyond SDF precision')
    return (block, result)

def sdf_record(molecule, *, valid, properties):
    if valid:
        (block, result) = checked_mol_block(molecule)
    else:
        result = Chem.Mol(molecule)
        try:
            block = Chem.MolToMolBlock(result, includeStereo=True, kekulize=False)
        except Exception:
            result = Chem.Mol()
            result.SetProp('_Name', str(properties['candidate_id']))
            block = Chem.MolToMolBlock(result, kekulize=False)
    chunks = [block.rstrip('\r\n')]
    properties = dict(properties)
    for key in result.GetPropNames(includePrivate=False, includeComputed=False):
        if key.startswith('kinase_'):
            properties[key] = result.GetProp(key)
    for (key, value) in properties.items():
        text = str(value).replace('\n', ' ').replace('\r', ' ')
        chunks.extend([f'>  <{key}>', text, ''])
    chunks.extend(['$$$$', ''])
    return '\n'.join(chunks)
