from __future__ import annotations
import io, json, base64
from typing import Any, List
import numpy as np
from rdkit import Chem

def canonical_isomeric_identity(molecule: Chem.Mol) -> str:
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError('Molecule cannot be represented by isomeric SMILES')
    return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)

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
    if result.HasProp('kinase_native_smiles') and canonical_isomeric_identity(result) != canonical_isomeric_identity(Chem.MolFromSmiles(result.GetProp('kinase_native_smiles'))):
        raise ValueError('Native stereochemistry metadata did not restore source identity')
    return result

def finite_coordinates(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or (not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a finite N x 3 coordinate matrix')
    return array

def molecule_coordinates(molecule: Chem.Mol, *, label: str) -> np.ndarray:
    if molecule is None:
        raise ValueError(f'{label} did not parse')
    if molecule.GetNumConformers() != 1:
        raise ValueError(f'{label} must have exactly one conformer, found {molecule.GetNumConformers()}')
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(f'{label} conformer has the wrong shape')
    if not np.isfinite(coordinates).all():
        raise ValueError(f'{label} has non-finite coordinates')
    if molecule.GetNumAtoms() == 0:
        raise ValueError(f'{label} has no atoms')
    return coordinates

def parse_sdf_bytes(payload: bytes, *, label: str, sanitize: bool=True) -> List[Chem.Mol]:
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(payload), sanitize=sanitize, removeHs=False, strictParsing=True)
    molecules: List[Chem.Mol] = []
    for (index, molecule) in enumerate(supplier):
        if molecule is None:
            raise ValueError(f'{label} has an unreadable record at index {index}')
        molecule = restore_native_identity(molecule)
        if molecule.HasProp('kinase_native_molecule_binary_base64'):
            authoritative = Chem.Mol(base64.b64decode(molecule.GetProp('kinase_native_molecule_binary_base64'), validate=True))
            Chem.SanitizeMol(authoritative)
            atom_keys = lambda mol: [(atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge()) for atom in mol.GetAtoms()]
            if atom_keys(molecule) != atom_keys(authoritative) or canonical_isomeric_identity(molecule) != canonical_isomeric_identity(authoritative):
                raise ValueError(f'{label} embedded molecule identity differs from the SDF')
            if not np.allclose(molecule.GetConformer().GetPositions(), authoritative.GetConformer().GetPositions(), atol=5.1e-05, rtol=0):
                raise ValueError(f'{label} embedded coordinates differ from the SDF')
            for key in molecule.GetPropNames(includePrivate=False, includeComputed=False):
                authoritative.SetProp(key, molecule.GetProp(key))
            authoritative.SetProp('_Name', molecule.GetProp('_Name'))
            molecule = authoritative
        molecule_coordinates(molecule, label=f'{label} record {index}')
        molecules.append(molecule)
    if not molecules:
        raise ValueError(f'{label} contains no molecules')
    return molecules
