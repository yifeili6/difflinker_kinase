from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
SUPPORTED_ELEMENTS = ('C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P')

@dataclass(frozen=True)
class Mol2Atom:
    atom_id: int
    name: str
    element: str
    coord: Tuple[float, float, float]
    residue_id: int
    residue_name: str

def element_from_tripose_type(atom_type: str) -> str:
    token = atom_type.split('.', 1)[0].strip()
    upper = token.upper()
    if upper == 'CL':
        return 'Cl'
    if upper == 'BR':
        return 'Br'
    if upper in {'C', 'O', 'N', 'F', 'S', 'I', 'P', 'H'}:
        return upper
    return token[:1].upper() + token[1:].lower()

def read_mol2_atoms(path: Path) -> List[Mol2Atom]:
    atoms: List[Mol2Atom] = []
    in_atoms = False
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == '@<TRIPOS>ATOM':
                in_atoms = True
                continue
            if line.startswith('@<TRIPOS>'):
                if in_atoms:
                    break
                continue
            if not in_atoms or not line:
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f'Malformed MOL2 ATOM record in {path}: {raw_line.rstrip()}')
            atoms.append(Mol2Atom(atom_id=int(fields[0]), name=fields[1], element=element_from_tripose_type(fields[5]), coord=(float(fields[2]), float(fields[3]), float(fields[4])), residue_id=int(fields[6]), residue_name=fields[7]))
    if not atoms:
        raise ValueError(f'No MOL2 atoms found in {path}')
    return atoms

def pocket_from_protein_mol2(protein_path: Path, ligand_coords: np.ndarray, cutoff: float, supported_elements: Sequence[str]=SUPPORTED_ELEMENTS) -> Dict[str, object]:
    atoms = read_mol2_atoms(protein_path)
    residues: Dict[Tuple[int, str], List[Mol2Atom]] = {}
    for atom in atoms:
        residues.setdefault((atom.residue_id, atom.residue_name), []).append(atom)
    ligand_coords = np.asarray(ligand_coords, dtype=np.float64)
    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3 or (not np.isfinite(ligand_coords).all()):
        raise ValueError('Ligand coordinates must be a finite N x 3 array')
    selected: List[Tuple[int, str]] = []
    cutoff_sq = float(cutoff) ** 2
    for residue_key in sorted(residues):
        heavy = [atom for atom in residues[residue_key] if atom.element != 'H']
        if not heavy:
            continue
        coords = np.asarray([atom.coord for atom in heavy], dtype=np.float64)
        delta = coords[:, None, :] - ligand_coords[None, :, :]
        if np.any(np.sum(delta * delta, axis=-1) <= cutoff_sq):
            selected.append(residue_key)
    full_coord: List[List[float]] = []
    full_types: List[str] = []
    bb_coord: List[List[float]] = []
    bb_types: List[str] = []
    unsupported = 0
    supported = set(supported_elements)
    for residue_key in selected:
        for atom in sorted(residues[residue_key], key=lambda item: item.atom_id):
            if atom.element == 'H':
                continue
            if atom.element not in supported:
                unsupported += 1
                continue
            coord = list(atom.coord)
            full_coord.append(coord)
            full_types.append(atom.element)
            if atom.name.upper() in {'N', 'CA', 'C', 'O'}:
                bb_coord.append(coord)
                bb_types.append(atom.element)
    if not full_coord:
        raise ValueError('Pocket contains no supported heavy atoms')
    return {'full_coord': full_coord, 'full_types': full_types, 'bb_coord': bb_coord, 'bb_types': bb_types, 'cutoff_angstrom': float(cutoff), 'residue_count': len(selected), 'unsupported_heavy_atoms_removed': unsupported}
