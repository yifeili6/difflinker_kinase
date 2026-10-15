from __future__ import annotations
import hashlib
import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem.rdMMPA import FragmentMol
MMPA_PATTERN = '[#6+0;!$(*=,#[!#6])]!@!=!#[*]'

@dataclass(frozen=True)
class Fragmentation:
    linker_smiles: str
    fragments_smiles: str
    signature: str

def _dummy_count(mol: Chem.Mol) -> int:
    return sum((atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()))

def _non_dummy_atoms(mol: Chem.Mol) -> int:
    return sum((atom.GetAtomicNum() != 0 for atom in mol.GetAtoms()))

def _canonical_dummy_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse MMPA component: {smiles}')
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def enumerate_two_cut_mmpa(mol: Chem.Mol, min_fragment_atoms: int=5, min_linker_atoms: int=3) -> List[Fragmentation]:
    parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    query_mol = Chem.MolFromSmiles(parent)
    raw_results = FragmentMol(query_mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=MMPA_PATTERN, resultsAsMols=False)
    accepted = {}
    for (linker_raw, fragments_raw) in raw_results:
        if not linker_raw or not fragments_raw:
            continue
        linker = Chem.MolFromSmiles(linker_raw)
        fragment_parts = [Chem.MolFromSmiles(part) for part in fragments_raw.split('.')]
        if linker is None or len(fragment_parts) != 2 or any((part is None for part in fragment_parts)):
            continue
        if _dummy_count(linker) != 2 or any((_dummy_count(part) != 1 for part in fragment_parts)):
            continue
        if _non_dummy_atoms(linker) < min_linker_atoms:
            continue
        if any((_non_dummy_atoms(part) < min_fragment_atoms for part in fragment_parts)):
            continue
        linker_smi = _canonical_dummy_smiles(linker_raw)
        fragment_smis = sorted((_canonical_dummy_smiles(part) for part in fragments_raw.split('.')))
        fragments_smi = '.'.join(fragment_smis)
        key = (linker_smi, fragments_smi)
        signature = hashlib.sha256((parent + '|' + linker_smi + '|' + fragments_smi).encode()).hexdigest()[:20]
        accepted[key] = Fragmentation(linker_smi, fragments_smi, signature)
    return [accepted[key] for key in sorted(accepted)]

def _remove_dummy_atoms(component: Chem.Mol, expected: int, mark_anchor: bool) -> Chem.Mol:
    dummies = [atom for atom in component.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != expected:
        raise ValueError(f'Expected {expected} dummy atoms, found {len(dummies)}')
    anchors = []
    for dummy in dummies:
        neighbors = list(dummy.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError('Every MMPA dummy atom must have exactly one neighbor')
        anchors.append(neighbors[0].GetIdx())
    editable = Chem.RWMol(component)
    for dummy in sorted(dummies, key=lambda atom: atom.GetIdx(), reverse=True):
        editable.RemoveAtom(dummy.GetIdx())
    result = editable.GetMol()
    for atom in result.GetAtoms():
        atom.SetProp('_Anchor', '0')
        atom.SetIntProp('_ExitCount', 0)
    for anchor in anchors:
        removed_before = sum((dummy.GetIdx() < anchor for dummy in dummies))
        atom = result.GetAtomWithIdx(anchor - removed_before)
        atom.SetIntProp('_ExitCount', atom.GetIntProp('_ExitCount') + 1)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if mark_anchor:
        anchor = anchors[0]
        removed_before = sum((dummy.GetIdx() < anchor for dummy in dummies))
        result.GetAtomWithIdx(anchor - removed_before).SetProp('_Anchor', '1')
    return result

def _conformer_for_match(parent: Chem.Mol, match: Sequence[int]) -> Chem.Conformer:
    positions = parent.GetConformer().GetPositions()
    conformer = Chem.Conformer(len(match))
    for (idx, parent_idx) in enumerate(match):
        (x, y, z) = positions[parent_idx]
        conformer.SetAtomPosition(idx, Geometry.Point3D(float(x), float(y), float(z)))
    return conformer

def _first_partition(parent: Chem.Mol, components: Sequence[Chem.Mol]) -> Tuple[Tuple[int, ...], ...]:
    if len(components) != 3:
        raise ValueError('Exactly two fragments and one linker are required')
    match_lists = []
    for component in components:
        matches = sorted(parent.GetSubstructMatches(component, uniquify=False, useChirality=False, maxMatches=1000000))
        if not matches:
            raise ValueError('A fragment/linker component does not match the parent molecule')
        if len(matches) == 1000000:
            raise ValueError('Symmetry match enumeration reached its explicit safety bound')
        match_lists.append(matches)
    parent_atoms = set(range(parent.GetNumAtoms()))

    def search(index: int, used: set, chosen: List[Tuple[int, ...]]):
        if index == len(match_lists):
            if used != parent_atoms:
                return None
            owner = {atom: part for (part, match) in enumerate(chosen) for atom in match}
            cut_edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in parent.GetBonds() if owner[bond.GetBeginAtomIdx()] != owner[bond.GetEndAtomIdx()]]
            if len(cut_edges) != 2:
                return None
            attachment_counts = {}
            pairs = []
            for (left, right) in cut_edges:
                pairs.append(tuple(sorted((owner[left], owner[right]))))
                for atom in (left, right):
                    attachment_counts[atom] = attachment_counts.get(atom, 0) + 1
            if sorted(pairs) != [(0, 2), (1, 2)]:
                return None
            expected = {}
            for (component, match) in zip(components, chosen):
                for atom in component.GetAtoms():
                    count = atom.GetIntProp('_ExitCount')
                    if count:
                        expected[match[atom.GetIdx()]] = count
            if attachment_counts != expected:
                return None
            return tuple(chosen)
        for match in match_lists[index]:
            match_set = set(match)
            if used.isdisjoint(match_set):
                result = search(index + 1, used | match_set, chosen + [match])
                if result is not None:
                    return result
        return None
    partition = search(0, set(), [])
    if partition is None:
        raise ValueError('No disjoint fragment/linker matching reconstructs the parent')
    return partition

def _parent_component(parent: Chem.Mol, query: Chem.Mol, match: Sequence[int]) -> Chem.Mol:
    keep = set(match)
    editable = Chem.RWMol(parent)
    for index in reversed(range(parent.GetNumAtoms())):
        if index not in keep:
            editable.RemoveAtom(index)
    retained_order = sorted(keep)
    result = Chem.RenumberAtoms(editable.GetMol(), [retained_order.index(i) for i in match])
    result.RemoveAllConformers()
    result.AddConformer(_conformer_for_match(parent, match), assignId=True)
    for (index, parent_index) in enumerate(match):
        atom = result.GetAtomWithIdx(index)
        atom.SetIntProp('_ParentAtomIndex', int(parent_index))
        atom.SetIntProp('_ParentTotalHs', parent.GetAtomWithIdx(parent_index).GetTotalNumHs())
        atom.SetIntProp('_ParentChiralTag', int(parent.GetAtomWithIdx(parent_index).GetChiralTag()))
        atom.SetProp('_ParentNeighborOrder', ','.join((str(a.GetIdx()) for a in parent.GetAtomWithIdx(parent_index).GetNeighbors())))
        atom.SetProp('_Anchor', query.GetAtomWithIdx(index).GetProp('_Anchor'))
        atom.SetIntProp('_ExitCount', query.GetAtomWithIdx(index).GetIntProp('_ExitCount'))
        if atom.GetSymbol() != parent.GetAtomWithIdx(parent_index).GetSymbol():
            raise AssertionError('Parent element identity changed')
    for bond in result.GetBonds():
        source = parent.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()])
        bond.SetIntProp('_ParentBondIndex', source.GetIdx())
        bond.SetIntProp('_ParentStereo', int(source.GetStereo()))
        bond.SetIntProp('_ParentBondDir', int(source.GetBondDir()))
        bond.SetProp('_ParentStereoAtoms', ','.join(map(str, source.GetStereoAtoms())))
    return result

def validate_native_reassembly(parent: Chem.Mol, fragments: Chem.Mol, linker: Chem.Mol) -> str:
    atoms = {}
    internal_bonds = {}
    for component in (fragments, linker):
        for atom in component.GetAtoms():
            index = atom.GetIntProp('_ParentAtomIndex')
            if index in atoms:
                raise ValueError('A parent atom is present in both components')
            atoms[index] = atom
        for bond in component.GetBonds():
            internal_bonds[bond.GetIntProp('_ParentBondIndex')] = bond
    if set(atoms) != set(range(parent.GetNumAtoms())):
        raise ValueError('Components do not cover the native parent')
    editable = Chem.RWMol()
    for index in range(parent.GetNumAtoms()):
        atom = Chem.Atom(atoms[index])
        atom.SetChiralTag(Chem.ChiralType(atom.GetIntProp('_ParentChiralTag')))
        editable.AddAtom(atom)
    cut_bonds = []
    for source in parent.GetBonds():
        copied = internal_bonds.get(source.GetIdx())
        if copied is None:
            cut_bonds.append(source)
            copied = source
        editable.AddBond(source.GetBeginAtomIdx(), source.GetEndAtomIdx(), copied.GetBondType())
        bond = editable.GetBondBetweenAtoms(source.GetBeginAtomIdx(), source.GetEndAtomIdx())
        bond.SetIsAromatic(copied.GetIsAromatic())
        bond.SetIsConjugated(copied.GetIsConjugated())
        bond.SetBondDir(Chem.BondDir(copied.GetIntProp('_ParentBondDir')) if copied.HasProp('_ParentBondDir') else copied.GetBondDir())
    if len(cut_bonds) != 2:
        raise ValueError('Native reassembly requires exactly two cut bonds')
    result = editable.GetMol()
    for atom in result.GetAtoms():
        original = [int(i) for i in atom.GetProp('_ParentNeighborOrder').split(',') if i]
        observed = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if set(original) != set(observed):
            raise ValueError('Native neighbor identities changed during reassembly')
        order = [original.index(i) for i in observed]
        odd = sum((order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))) % 2
        if odd and atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atom.InvertChirality()
    for source in parent.GetBonds():
        copied = internal_bonds.get(source.GetIdx())
        stereo = copied.GetIntProp('_ParentStereo') if copied is not None else int(source.GetStereo())
        indices = [int(i) for i in copied.GetProp('_ParentStereoAtoms').split(',') if i] if copied is not None else list(source.GetStereoAtoms())
        bond = result.GetBondWithIdx(source.GetIdx())
        if indices:
            bond.SetStereoAtoms(*indices)
        bond.SetStereo(Chem.BondStereo(stereo))
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result, cleanIt=False, force=True)
    identity = canonical_isomeric_identity(result)
    if identity != canonical_isomeric_identity(parent):
        raise ValueError('Native component reassembly changed isomeric identity')
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)

def canonical_isomeric_identity(molecule: Chem.Mol) -> str:
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError('Molecule cannot be represented by isomeric SMILES')
    return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)

def build_coordinate_components(parent: Chem.Mol, fragmentation: Fragmentation) -> Tuple[Chem.Mol, Chem.Mol, Tuple[int, int]]:
    if parent.GetNumConformers() != 1:
        raise ValueError('Parent ligand must have exactly one conformer')
    fragment_components = [_remove_dummy_atoms(Chem.MolFromSmiles(smiles), expected=1, mark_anchor=True) for smiles in fragmentation.fragments_smiles.split('.')]
    linker = _remove_dummy_atoms(Chem.MolFromSmiles(fragmentation.linker_smiles), expected=2, mark_anchor=False)
    components = fragment_components + [linker]
    matches = _first_partition(parent, components)
    components = [_parent_component(parent, component, match) for (component, match) in zip(components, matches)]
    fragment_components = components[:2]
    linker = components[2]
    fragments = Chem.CombineMols(fragment_components[0], fragment_components[1])
    anchor_indices = tuple((atom.GetIdx() for atom in fragments.GetAtoms() if atom.HasProp('_Anchor') and atom.GetProp('_Anchor') == '1'))
    if len(anchor_indices) != 2 or anchor_indices[0] == anchor_indices[1]:
        raise ValueError(f'Expected exactly two distinct anchors, found {anchor_indices}')
    if fragments.GetNumAtoms() + linker.GetNumAtoms() != parent.GetNumAtoms():
        raise ValueError('Fragments plus linker do not contain every parent atom')
    return (fragments, linker, (int(anchor_indices[0]), int(anchor_indices[1])))
