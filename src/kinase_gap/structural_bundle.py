#!/usr/bin/env python3
"""Build an auditable, prospective structural-analysis bundle.

This post-report stage consumes the deterministic ``prospective_chemistry_only``
selection emitted by :mod:`kinase_gap.report` and preserves its recorded ranking.

The build performs only CPU-local operations:

* validate the selection CSV/SDF files and their recorded SHA-256 hashes;
* resolve every candidate through the pinned KLIFS context manifest;
* align each source protein frame to one declared target reference using a
  proper-rotation C-alpha Kabsch fit;
* transform the candidate, its native ligand, and its *source* receptor
  together, preserving all source-frame contact distances;
* publish aligned SDF/PDB inputs, fragment masks, numerical attribution
  payloads, deterministic RDKit atom maps/grids, a PyMOL script, and provenance.

Separate rendering scripts execute PyMOL and LigPlot+. ``ligplot-check``
validates an explicit, hash-pinned tool configuration and the contact-ready inputs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, QED
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SELECTION_LABEL = "prospective_chemistry_only"
BUNDLE_LABEL = "prospective_structural_followup"
DEFAULT_REFERENCES = {
    "5lqf": "human__CDC2__5lqf__altA__chainA",
    "4ag8": "human__KDR__4ag8__alt-__chainA",
}
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_USE_CHIRALITY = True
TANIMOTO_METRIC = "Tanimoto"
PDB_LIGAND_CHAIN = "Z"
PDB_LIGAND_RESIDUE = 900
PDB_LIGAND_NAME = "LIG"
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PDBAtom:
    """One coordinate-bearing PDB record."""

    line: str
    serial: int
    atom_name: str
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str
    element: str
    coordinate: np.ndarray

    @property
    def ca_key(self) -> Tuple[int, str, str]:
        return (self.residue_number, self.insertion_code, self.residue_name)


@dataclass(frozen=True)
class Alignment:
    """Row-vector rigid transform: ``x_reference = x_source @ R + t``."""

    rotation: np.ndarray
    translation: np.ndarray
    source_centroid: np.ndarray
    reference_centroid: np.ndarray
    matched_ca_atoms: int
    source_ca_atoms: int
    reference_ca_atoms: int
    coverage: float
    rmsd_before: float
    rmsd_after: float
    max_after: float
    determinant: float
    singular_values: np.ndarray

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        points = finite_coordinates(coordinates, name="coordinates")
        return points @ self.rotation + self.translation

    def as_dict(self) -> Dict[str, Any]:
        return {
            "convention": "row vectors: x_reference = x_source @ rotation + translation",
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "source_centroid": self.source_centroid.tolist(),
            "reference_centroid": self.reference_centroid.tolist(),
            "matched_ca_atoms": self.matched_ca_atoms,
            "source_ca_atoms": self.source_ca_atoms,
            "reference_ca_atoms": self.reference_ca_atoms,
            "coverage": self.coverage,
            "rmsd_before_angstrom": self.rmsd_before,
            "rmsd_after_angstrom": self.rmsd_after,
            "max_residual_after_angstrom": self.max_after,
            "rotation_determinant": self.determinant,
            "singular_values": self.singular_values.tolist(),
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    write_bytes(path, canonical_json(payload))


def read_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.read_bytes()


def read_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def project_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"CLI paths must be project-root-relative: {value}")
    root = PROJECT_ROOT.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"CLI path escapes the project root: {value}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_component(value: str) -> str:
    raw = str(value).strip()
    slug = SAFE_NAME.sub("_", raw).strip("._-") or "item"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{suffix}"


def validate_identifier(value: str, *, label: str) -> str:
    text = str(value).strip()
    if IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise ValueError(
            f"{label} must use only ASCII letters, digits, dot, underscore, or dash"
        )
    return text


def finite_coordinates(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite N x 3 coordinate matrix")
    return array


def molecule_coordinates(molecule: Chem.Mol, *, label: str) -> np.ndarray:
    if molecule is None:
        raise ValueError(f"{label} did not parse")
    if molecule.GetNumConformers() != 1:
        raise ValueError(
            f"{label} must have exactly one conformer, found "
            f"{molecule.GetNumConformers()}"
        )
    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(f"{label} conformer has the wrong shape")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{label} has non-finite coordinates")
    if molecule.GetNumAtoms() == 0:
        raise ValueError(f"{label} has no atoms")
    return coordinates


def transform_molecule(
    molecule: Chem.Mol,
    alignment: Alignment,
    *,
    label: str,
) -> Chem.Mol:
    result = Chem.Mol(molecule)
    coordinates = alignment.transform(molecule_coordinates(result, label=label))
    conformer = result.GetConformer()
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    return result


def parse_pdb_atoms(payload: bytes, *, label: str) -> List[PDBAtom]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not ASCII PDB text") from exc
    atoms: List[PDBAtom] = []
    seen_serials = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"{label}:{line_number} has a short atom record")
        try:
            serial = int(line[6:11])
            residue_number = int(line[22:26])
            coordinate = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(
                f"{label}:{line_number} has invalid PDB numeric fields"
            ) from exc
        if serial in seen_serials:
            raise ValueError(f"{label} has duplicate atom serial {serial}")
        seen_serials.add(serial)
        if not np.isfinite(coordinate).all():
            raise ValueError(f"{label}:{line_number} has non-finite coordinates")
        atom_name = line[12:16].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            element = re.sub(r"[^A-Za-z]", "", atom_name)[:2].title()
        atoms.append(
            PDBAtom(
                line=line,
                serial=serial,
                atom_name=atom_name,
                residue_name=line[17:20].strip(),
                chain=line[21:22].strip(),
                residue_number=residue_number,
                insertion_code=line[26:27].strip(),
                element=element,
                coordinate=coordinate,
            )
        )
    if not atoms:
        raise ValueError(f"{label} contains no coordinate records")
    return atoms


def protein_atoms(atoms: Iterable[PDBAtom]) -> List[PDBAtom]:
    result = [atom for atom in atoms if atom.line.startswith("ATOM  ")]
    if not result:
        raise ValueError("Complex PDB contains no protein ATOM records")
    return result


def ca_coordinates(
    atoms: Sequence[PDBAtom], *, label: str
) -> Dict[Tuple[int, str, str], np.ndarray]:
    result: Dict[Tuple[int, str, str], np.ndarray] = {}
    for atom in atoms:
        if atom.atom_name != "CA":
            continue
        if atom.ca_key in result:
            raise ValueError(
                f"{label} has ambiguous C-alpha residue key {atom.ca_key}"
            )
        result[atom.ca_key] = atom.coordinate
    if not result:
        raise ValueError(f"{label} has no C-alpha atoms")
    return result


def compute_alignment(
    source_atoms: Sequence[PDBAtom],
    reference_atoms: Sequence[PDBAtom],
    *,
    min_ca_atoms: int = 20,
    min_ca_coverage: float = 0.80,
    max_ca_rmsd: float = 2.0,
) -> Alignment:
    """Fit a proper source-to-reference Kabsch transform."""

    if min_ca_atoms < 4:
        raise ValueError("min_ca_atoms must be at least 4")
    if not 0.0 < min_ca_coverage <= 1.0:
        raise ValueError("min_ca_coverage must be in (0, 1]")
    if not math.isfinite(max_ca_rmsd) or max_ca_rmsd <= 0:
        raise ValueError("max_ca_rmsd must be a positive finite number")

    source = ca_coordinates(source_atoms, label="source protein")
    reference = ca_coordinates(reference_atoms, label="reference protein")
    keys = sorted(set(source).intersection(reference))
    if len(keys) < min_ca_atoms:
        raise ValueError(
            f"Only {len(keys)} matched C-alpha atoms; at least "
            f"{min_ca_atoms} are required"
        )
    coverage = len(keys) / min(len(source), len(reference))
    if coverage + 1e-12 < min_ca_coverage:
        raise ValueError(
            f"C-alpha coverage {coverage:.6f} is below {min_ca_coverage:.6f}"
        )

    source_points = np.vstack([source[key] for key in keys])
    reference_points = np.vstack([reference[key] for key in keys])
    source_centroid = source_points.mean(axis=0)
    reference_centroid = reference_points.mean(axis=0)
    source_centered = source_points - source_centroid
    reference_centered = reference_points - reference_centroid
    covariance = source_centered.T @ reference_centered
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)
    if np.linalg.matrix_rank(source_centered, tol=1e-8) < 3:
        raise ValueError("C-alpha coordinates are rank-deficient for 3D alignment")
    rotation = u_matrix @ v_transpose
    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ v_transpose
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-8:
        raise ValueError(f"Kabsch fit did not produce a proper rotation: {determinant}")
    translation = reference_centroid - source_centroid @ rotation
    aligned = source_points @ rotation + translation
    before_distances = np.linalg.norm(source_points - reference_points, axis=1)
    after_distances = np.linalg.norm(aligned - reference_points, axis=1)
    rmsd_before = float(np.sqrt(np.mean(np.square(before_distances))))
    rmsd_after = float(np.sqrt(np.mean(np.square(after_distances))))
    max_after = float(after_distances.max())
    if rmsd_after > max_ca_rmsd + 1e-12:
        raise ValueError(
            f"C-alpha RMSD {rmsd_after:.6f} exceeds {max_ca_rmsd:.6f} Å"
        )
    return Alignment(
        rotation=rotation,
        translation=translation,
        source_centroid=source_centroid,
        reference_centroid=reference_centroid,
        matched_ca_atoms=len(keys),
        source_ca_atoms=len(source),
        reference_ca_atoms=len(reference),
        coverage=float(coverage),
        rmsd_before=rmsd_before,
        rmsd_after=rmsd_after,
        max_after=max_after,
        determinant=determinant,
        singular_values=singular_values,
    )


def rewrite_pdb_atom(atom: PDBAtom, coordinate: np.ndarray) -> str:
    x, y, z = [float(value) for value in coordinate]
    if any(value < -999.999 or value > 9999.999 for value in (x, y, z)):
        raise ValueError("Transformed coordinate exceeds the PDB fixed-width field")
    line = atom.line.ljust(80)
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}".rstrip()


def transformed_protein_pdb(
    atoms: Sequence[PDBAtom],
    alignment: Alignment,
    *,
    remarks: Sequence[str],
) -> bytes:
    proteins = protein_atoms(atoms)
    coordinates = alignment.transform(
        np.vstack([atom.coordinate for atom in proteins])
    )
    lines = [f"REMARK 950 {text}"[:80] for text in remarks]
    lines.extend(
        rewrite_pdb_atom(atom, coordinate)
        for atom, coordinate in zip(proteins, coordinates)
    )
    lines.extend(["TER", "END"])
    return ("\n".join(lines) + "\n").encode("ascii")


def contact_ready_pdb(
    atoms: Sequence[PDBAtom],
    alignment: Alignment,
    ligand: Chem.Mol,
    *,
    candidate_id: str,
    source_structure_id: str,
) -> bytes:
    """Create protein+ligand PDB input without changing protonation."""

    proteins = protein_atoms(atoms)
    protein_coordinates = alignment.transform(
        np.vstack([atom.coordinate for atom in proteins])
    )
    serials = [atom.serial for atom in proteins]
    if len(serials) != len(set(serials)):
        raise ValueError("Protein atom serials are not unique")
    next_serial = max(serials) + 1
    ligand_end = next_serial + ligand.GetNumAtoms() - 1
    if next_serial <= 0 or ligand_end > 99999:
        raise ValueError("Contact PDB atom serials exceed PDB limits")
    coordinates = molecule_coordinates(ligand, label=candidate_id)

    lines = [
        f"REMARK 950 CANDIDATE_ID {candidate_id}"[:80],
        f"REMARK 950 SOURCE_STRUCTURE_ID {source_structure_id}"[:80],
        "REMARK 950 COORDINATES RIGIDLY ALIGNED WITH SOURCE RECEPTOR",
        "REMARK 950 PROTONATION UNCHANGED; CONTACT-ANALYSIS INPUT ONLY",
    ]
    lines.extend(
        rewrite_pdb_atom(atom, coordinate)
        for atom, coordinate in zip(proteins, protein_coordinates)
    )
    lines.append("TER")
    ligand_serials: Dict[int, int] = {}
    for index, (atom, coordinate) in enumerate(zip(ligand.GetAtoms(), coordinates)):
        serial = next_serial + index
        ligand_serials[index] = serial
        element = atom.GetSymbol()
        atom_name = f"{element}{index + 1}"[:4]
        x, y, z = [float(value) for value in coordinate]
        if any(value < -999.999 or value > 9999.999 for value in (x, y, z)):
            raise ValueError("Ligand coordinate exceeds PDB fixed-width field")
        lines.append(
            f"HETATM{serial:5d} {atom_name:<4s} {PDB_LIGAND_NAME:>3s} "
            f"{PDB_LIGAND_CHAIN}{PDB_LIGAND_RESIDUE:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
        )
    for atom in ligand.GetAtoms():
        bonded = sorted(
            ligand_serials[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
        )
        if not bonded:
            continue
        source_serial = ligand_serials[atom.GetIdx()]
        for start in range(0, len(bonded), 4):
            lines.append(
                f"CONECT{source_serial:5d}"
                + "".join(f"{serial:5d}" for serial in bonded[start : start + 4])
            )
    lines.append("END")
    return ("\n".join(lines) + "\n").encode("ascii")


def max_pairwise_distance_change(
    protein_source: np.ndarray,
    ligand_source: np.ndarray,
    alignment: Alignment,
) -> float:
    source = np.linalg.norm(
        protein_source[:, None, :] - ligand_source[None, :, :], axis=2
    )
    protein_aligned = alignment.transform(protein_source)
    ligand_aligned = alignment.transform(ligand_source)
    aligned = np.linalg.norm(
        protein_aligned[:, None, :] - ligand_aligned[None, :, :], axis=2
    )
    return float(np.max(np.abs(source - aligned)))


def read_csv_bytes(payload: bytes, *, label: str) -> Tuple[List[Dict[str, str]], List[str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"{label} has missing or duplicate column names")
    rows = [dict(row) for row in reader]
    if any(None in row for row in rows):
        raise ValueError(f"{label} has rows wider than its header")
    return rows, fields


def require_text(row: Mapping[str, str], field: str, *, label: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"{label} has an empty {field!r} field")
    return value


def require_integer(
    row: Mapping[str, str],
    field: str,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    value = require_text(row, field, label=label)
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        raise ValueError(f"{label} has a non-integer {field!r}: {value!r}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{label} has {field!r} below {minimum}: {parsed}")
    return parsed


def parse_sdf_bytes(
    payload: bytes,
    *,
    label: str,
    sanitize: bool = True,
) -> List[Chem.Mol]:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(payload),
        sanitize=sanitize,
        removeHs=False,
        strictParsing=True,
    )
    molecules: List[Chem.Mol] = []
    for index, molecule in enumerate(supplier):
        if molecule is None:
            raise ValueError(f"{label} has an unreadable record at index {index}")
        from .chemistry_io import restore_native_identity
        molecule = restore_native_identity(molecule)
        if molecule.HasProp("kinase_native_molecule_binary_base64"):
            from .fragmentation import canonical_isomeric_identity
            authoritative = Chem.Mol(base64.b64decode(molecule.GetProp("kinase_native_molecule_binary_base64"), validate=True))
            Chem.SanitizeMol(authoritative)
            atom_keys = lambda mol: [(atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge()) for atom in mol.GetAtoms()]
            if atom_keys(molecule) != atom_keys(authoritative) or canonical_isomeric_identity(molecule) != canonical_isomeric_identity(authoritative):
                raise ValueError(f"{label} embedded molecule identity differs from the SDF")
            if not np.allclose(molecule.GetConformer().GetPositions(), authoritative.GetConformer().GetPositions(), atol=0.000051, rtol=0):
                raise ValueError(f"{label} embedded coordinates differ from the SDF")
            for key in molecule.GetPropNames(includePrivate=False, includeComputed=False):
                authoritative.SetProp(key, molecule.GetProp(key))
            authoritative.SetProp("_Name", molecule.GetProp("_Name"))
            molecule = authoritative
        molecule_coordinates(molecule, label=f"{label} record {index}")
        molecules.append(molecule)
    if not molecules:
        raise ValueError(f"{label} contains no molecules")
    return molecules


def molecule_identifier(molecule: Chem.Mol, *, label: str) -> str:
    title = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
    property_id = (
        molecule.GetProp("candidate_id").strip()
        if molecule.HasProp("candidate_id")
        else ""
    )
    if title and property_id and title != property_id:
        raise ValueError(
            f"{label} title/candidate_id mismatch: {title!r} != {property_id!r}"
        )
    identifier = property_id or title
    if not identifier:
        raise ValueError(f"{label} has neither title nor candidate_id")
    return identifier


def mol_to_sdf_bytes(molecule: Chem.Mol, *, title: str, native_identity: bool = False) -> bytes:
    """Serialize one molecule deterministically without a timestamp."""

    result = Chem.Mol(molecule)
    result.SetProp("_Name", title)
    from .chemistry_io import checked_mol_block, checked_native_mol_block
    block, result = (checked_native_mol_block(result) if native_identity else checked_mol_block(result))
    block = block.rstrip("\r\n")
    chunks = [block]
    for name in sorted(result.GetPropNames(includePrivate=False, includeComputed=False)):
        value = result.GetProp(name)
        if "\x00" in value:
            raise ValueError(f"SDF property {name!r} contains a NUL byte")
        chunks.extend([f">  <{name}>", value, ""])
    chunks.extend(["$$$$", ""])
    return "\n".join(chunks).encode("utf-8")


def load_ligand(payload: bytes, *, suffix: str, label: str) -> Chem.Mol:
    extension = suffix.lower()
    if extension == ".mol2":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} MOL2 is not UTF-8") from exc
        molecule = Chem.MolFromMol2Block(
            text, sanitize=True, removeHs=True, cleanupSubstructures=True
        )
    elif extension in {".sdf", ".sd"}:
        molecules = parse_sdf_bytes(payload, label=label, sanitize=True)
        if len(molecules) != 1:
            raise ValueError(f"{label} must contain exactly one SDF record")
        molecule = Chem.RemoveHs(molecules[0])
    elif extension == ".mol":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} MOL is not UTF-8") from exc
        molecule = Chem.MolFromMolBlock(text, sanitize=True, removeHs=True)
    elif extension == ".pdb":
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} PDB is not ASCII") from exc
        molecule = Chem.MolFromPDBBlock(text, sanitize=True, removeHs=True)
    else:
        raise ValueError(f"Unsupported ligand format for {label}: {suffix}")
    if molecule is None:
        raise ValueError(f"{label} could not be parsed and sanitized")
    molecule_coordinates(molecule, label=label)
    return molecule


def load_snapshot_hashes(snapshot_dir: Path) -> Dict[str, str]:
    payload = read_bytes(snapshot_dir / "files.sha256", label="snapshot files.sha256")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("snapshot files.sha256 is not ASCII") from exc
    hashes: Dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(
                f"snapshot files.sha256:{line_number} has invalid syntax"
            )
        digest, relative = match.groups()
        if relative in hashes:
            raise ValueError(f"Duplicate snapshot hash entry: {relative}")
        candidate = (snapshot_dir / relative).resolve()
        try:
            candidate.relative_to(snapshot_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Snapshot hash path escapes snapshot: {relative}") from exc
        hashes[relative] = digest
    if not hashes:
        raise ValueError("snapshot files.sha256 is empty")
    return hashes


def verified_snapshot_asset(
    path_text: str,
    *,
    snapshot_dir: Path,
    snapshot_hashes: Mapping[str, str],
    label: str,
) -> Tuple[Path, bytes, str]:
    path = Path(path_text)
    if path.is_absolute():
        raise ValueError(f"{label} must be project-root-relative")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(snapshot_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} is outside the pinned snapshot: {path_text}") from exc
    expected = snapshot_hashes.get(relative)
    if expected is None:
        raise ValueError(f"{label} is absent from snapshot files.sha256: {relative}")
    payload = read_bytes(resolved, label=label)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return resolved, payload, observed


def load_context_manifest(payload: bytes) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    rows, fields = read_csv_bytes(payload, label="context manifest")
    required = {
        "structure_id",
        "pdb",
        "ligand_path",
        "protein_path",
        "cached_pocket_path",
        "ligand_sha256",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Context manifest is missing columns: {missing}")
    result: Dict[str, Dict[str, str]] = {}
    for index, row in enumerate(rows):
        identifier = require_text(row, "structure_id", label=f"context row {index}")
        validate_identifier(identifier, label=f"context row {index} structure_id")
        if identifier in result:
            raise ValueError(f"Duplicate context structure_id: {identifier}")
        result[identifier] = row
    return result, fields


def context_assets(
    row: Mapping[str, str],
    *,
    snapshot_dir: Path,
    snapshot_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    structure_id = require_text(row, "structure_id", label="context")
    assets: Dict[str, Any] = {}
    for name, field in (
        ("ligand", "ligand_path"),
        ("protein", "protein_path"),
        ("pocket", "cached_pocket_path"),
    ):
        path, payload, digest = verified_snapshot_asset(
            require_text(row, field, label=structure_id),
            snapshot_dir=snapshot_dir,
            snapshot_hashes=snapshot_hashes,
            label=f"{structure_id} {name}",
        )
        assets[name] = {"path": path, "bytes": payload, "sha256": digest}
    if assets["ligand"]["sha256"] != require_text(
        row, "ligand_sha256", label=structure_id
    ):
        raise ValueError(f"{structure_id} ligand hash disagrees with context manifest")
    ligand_relative = Path(require_text(row, "ligand_path", label=structure_id))
    complex_relative = ligand_relative.parent / "complex.pdb"
    path, payload, digest = verified_snapshot_asset(
        complex_relative.as_posix(),
        snapshot_dir=snapshot_dir,
        snapshot_hashes=snapshot_hashes,
        label=f"{structure_id} complex PDB",
    )
    assets["complex_pdb"] = {"path": path, "bytes": payload, "sha256": digest}
    return assets


def validate_selection(
    selection_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, str]]], Dict[str, List[Chem.Mol]]]:
    selection_payload = read_bytes(selection_path, label="selection JSON")
    selection = read_json_bytes(selection_payload, label="selection JSON")
    if not isinstance(selection, dict):
        raise ValueError("selection JSON must be an object")
    if selection.get("selection_label") != SELECTION_LABEL:
        raise ValueError(
            f"Selection must be labelled {SELECTION_LABEL!r}, found "
            f"{selection.get('selection_label')!r}"
        )
    if selection.get("candidate_source") != "recorded_generation_run":
        raise ValueError("Selection must identify its recorded generation source")
    targets = selection.get("targets")
    outputs = selection.get("outputs")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Selection has no target mapping")
    if not isinstance(outputs, dict):
        raise ValueError("Selection has no output hash mapping")

    rows_by_target: Dict[str, List[Dict[str, str]]] = {}
    molecules_by_target: Dict[str, List[Chem.Mol]] = {}
    directory = selection_path.parent
    for raw_target, target_payload in sorted(targets.items()):
        target = str(raw_target).strip().lower()
        if target != raw_target or not target:
            raise ValueError(f"Selection target key is not canonical lowercase: {raw_target!r}")
        validate_identifier(target, label="selection target")
        if not isinstance(target_payload, dict):
            raise ValueError(f"Selection target {target!r} must be an object")
        expected_ids = target_payload.get("selected_candidate_ids")
        if not isinstance(expected_ids, list) or not all(
            isinstance(value, str) and value.strip() for value in expected_ids
        ):
            raise ValueError(f"Selection target {target!r} has invalid candidate IDs")
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(f"Selection target {target!r} has duplicate candidate IDs")

        csv_name = f"selected_{target}.csv"
        sdf_name = f"selected_{target}.sdf"
        csv_payload = read_bytes(directory / csv_name, label=csv_name)
        sdf_payload = read_bytes(directory / sdf_name, label=sdf_name)
        for name, payload in ((csv_name, csv_payload), (sdf_name, sdf_payload)):
            record = outputs.get(name)
            if not isinstance(record, dict) or not SHA256_PATTERN.fullmatch(
                str(record.get("sha256", ""))
            ):
                raise ValueError(f"Selection has no valid recorded hash for {name}")
            observed = sha256_bytes(payload)
            if observed != record["sha256"]:
                raise ValueError(
                    f"{name} SHA-256 mismatch: expected {record['sha256']}, "
                    f"observed {observed}"
                )

        rows, fields = read_csv_bytes(csv_payload, label=csv_name)
        required = {
            "candidate_id",
            "pdb",
            "structure_id",
            "fragmentation_id",
            "source_uuid",
            "anchor_1",
            "anchor_2",
            "fragments_size",
            "linker_size",
            "num_atoms",
        }
        missing = sorted(required.difference(fields))
        if missing:
            raise ValueError(f"{csv_name} is missing columns: {missing}")
        row_ids = [
            validate_identifier(
                require_text(row, "candidate_id", label=f"{csv_name} row {index}"),
                label=f"{csv_name} row {index} candidate_id",
            )
            for index, row in enumerate(rows)
        ]
        if row_ids != expected_ids:
            raise ValueError(
                f"{csv_name} candidate order differs from selection JSON: "
                f"{row_ids} != {expected_ids}"
            )
        molecules = parse_sdf_bytes(sdf_payload, label=sdf_name, sanitize=True) if expected_ids else []
        if not expected_ids and sdf_payload.strip():
            raise ValueError(f"Empty target selection has nonempty SDF: {sdf_name}")
        molecule_ids = [
            molecule_identifier(molecule, label=f"{sdf_name} record {index}")
            for index, molecule in enumerate(molecules)
        ]
        if molecule_ids != expected_ids:
            raise ValueError(
                f"{sdf_name} candidate order differs from selection JSON: "
                f"{molecule_ids} != {expected_ids}"
            )
        recorded_ids = outputs[sdf_name].get("candidate_ids")
        if recorded_ids is not None and recorded_ids != expected_ids:
            raise ValueError(f"{sdf_name} recorded candidate IDs disagree with selection")
        if int(outputs[csv_name].get("records", -1)) != len(rows):
            raise ValueError(f"{csv_name} recorded row count is wrong")
        if int(outputs[sdf_name].get("records", -1)) != len(molecules):
            raise ValueError(f"{sdf_name} recorded molecule count is wrong")

        for index, (row, molecule) in enumerate(zip(rows, molecules)):
            label = f"{csv_name} row {index}"
            for diagnostic in ("anchor_attachment_pass", "retained_stereo_pass",
                               "retained_nonanchor_explicit_h_match", "retained_nonanchor_total_h_match"):
                if str(row.get(diagnostic, "")).lower() not in ("1", "true", "1.0"):
                    raise ValueError(f"Selected candidate does not satisfy {diagnostic}: {label}")
            if require_text(row, "pdb", label=label).lower() != target:
                raise ValueError(f"{label} PDB does not match target {target}")
            fragment_size = require_integer(row, "fragments_size", label=label, minimum=1)
            atom_count = require_integer(row, "num_atoms", label=label, minimum=2)
            require_integer(row, "source_uuid", label=label, minimum=0)
            require_integer(row, "linker_size", label=label, minimum=1)
            anchor_1 = require_integer(row, "anchor_1", label=label, minimum=0)
            anchor_2 = require_integer(row, "anchor_2", label=label, minimum=0)
            if atom_count != molecule.GetNumAtoms():
                raise ValueError(f"{label} num_atoms does not match selected SDF")
            if fragment_size >= atom_count:
                raise ValueError(f"{label} fragment size leaves no generated linker atoms")
            if (
                anchor_1 >= fragment_size
                or anchor_2 >= fragment_size
                or anchor_1 == anchor_2
            ):
                raise ValueError(f"{label} has invalid retained-fragment anchors")
            for field in ("structure_id", "fragmentation_id"):
                expected = validate_identifier(
                    require_text(row, field, label=label),
                    label=f"{label} {field}",
                )
                if molecule.HasProp(field) and molecule.GetProp(field).strip() != expected:
                    raise ValueError(f"{label}/{field} disagrees with selected SDF")
        rows_by_target[target] = rows
        molecules_by_target[target] = molecules

    selection["_validated_sha256"] = sha256_bytes(selection_payload)
    return selection, rows_by_target, molecules_by_target


def load_selected_fragments(
    fragment_sdf: Path,
    rows_by_target: Mapping[str, Sequence[Mapping[str, str]]],
) -> Tuple[Dict[int, Chem.Mol], str]:
    payload = read_bytes(fragment_sdf, label="test fragment SDF")
    wanted = {
        require_integer(row, "source_uuid", label="candidate", minimum=0)
        for rows in rows_by_target.values()
        for row in rows
    }
    from .chemistry_io import reconstruct_fragment
    table_path = fragment_sdf.with_name(fragment_sdf.name.replace("_frag.sdf", "_table.csv"))
    with table_path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    found: Dict[int, Chem.Mol] = {}
    for index in sorted(wanted):
        if index >= len(table) or int(table[index]["uuid"]) != index:
            raise ValueError(f"Corrected fragment table has no aligned UUID {index}")
        molecule = reconstruct_fragment(table[index], PROJECT_ROOT)
        molecule_coordinates(molecule, label=f"raw parent fragment {index}")
        found[index] = molecule
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"Test fragment SDF is missing source UUIDs: {missing}")
    return found, sha256_bytes(payload)


def validate_fragment_prefix(
    candidate: Chem.Mol,
    fragment: Chem.Mol,
    row: Mapping[str, str],
    *,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    candidate_id = require_text(row, "candidate_id", label="candidate")
    fragmentation_id = require_text(row, "fragmentation_id", label=candidate_id)
    title = fragment.GetProp("_Name").strip() if fragment.HasProp("_Name") else ""
    if title != fragmentation_id:
        raise ValueError(
            f"{candidate_id} fragment title mismatch: {title!r} != {fragmentation_id!r}"
        )
    size = require_integer(row, "fragments_size", label=candidate_id, minimum=1)
    if fragment.GetNumAtoms() != size:
        raise ValueError(f"{candidate_id} fragment atom count mismatch")
    components = Chem.GetMolFrags(fragment, asMols=False, sanitizeFrags=False)
    if len(components) != 2:
        raise ValueError(f"{candidate_id} retained mask must contain two fragments")
    anchors = (
        require_integer(row, "anchor_1", label=candidate_id, minimum=0),
        require_integer(row, "anchor_2", label=candidate_id, minimum=0),
    )
    component_by_atom = {
        atom_index: component_index
        for component_index, component in enumerate(components)
        for atom_index in component
    }
    if component_by_atom[anchors[0]] == component_by_atom[anchors[1]]:
        raise ValueError(f"{candidate_id} anchors occur on the same fragment")
    candidate_coordinates = molecule_coordinates(candidate, label=candidate_id)
    fragment_coordinates = molecule_coordinates(fragment, label=fragmentation_id)
    candidate_symbols = [
        candidate.GetAtomWithIdx(index).GetSymbol() for index in range(size)
    ]
    fragment_symbols = [atom.GetSymbol() for atom in fragment.GetAtoms()]
    if candidate_symbols != fragment_symbols:
        raise ValueError(f"{candidate_id} retained-fragment atom symbols changed")
    fragment_bonds = sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            str(bond.GetBondType()),
        )
        for bond in fragment.GetBonds()
    )
    candidate_fragment_bonds = sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            str(bond.GetBondType()),
        )
        for bond in candidate.GetBonds()
        if bond.GetBeginAtomIdx() < size and bond.GetEndAtomIdx() < size
    )
    if fragment_bonds != candidate_fragment_bonds:
        raise ValueError(f"{candidate_id} retained-fragment bonds changed")
    differences = np.linalg.norm(
        candidate_coordinates[:size] - fragment_coordinates, axis=1
    )
    maximum = float(differences.max())
    if maximum > tolerance + 1e-12:
        raise ValueError(
            f"{candidate_id} retained-fragment coordinates differ by "
            f"{maximum:.6f} Å (limit {tolerance:.6f})"
        )
    from .chemistry_io import attachment_diagnostics, sanitized_complete_molecule
    from .fragmentation import canonical_isomeric_identity
    verified = sanitized_complete_molecule(candidate)
    if canonical_isomeric_identity(verified) != canonical_isomeric_identity(candidate):
        raise ValueError(f"{candidate_id} retained/source stereochemistry check changed identity")
    chemistry = attachment_diagnostics(fragment, candidate, anchors, sanitized=verified)
    for field in ("anchor_attachment_pass", "retained_stereo_pass",
                  "retained_nonanchor_explicit_h_match", "retained_nonanchor_total_h_match"):
        if chemistry[field] != 1:
            raise ValueError(f"{candidate_id} selected fragment verification failed: {field}")
    generated = list(range(size, candidate.GetNumAtoms()))
    return {
        "fragment_verification": chemistry,
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "fragmentation_id": fragmentation_id,
        "source_uuid": require_integer(row, "source_uuid", label=candidate_id),
        "atom_index_base": 0,
        "retained_fragment_atom_indices": list(range(size)),
        "generated_linker_atom_indices": generated,
        "anchor_atom_indices": list(anchors),
        "retained_fragment_atom_symbols": fragment_symbols,
        "maximum_coordinate_difference_angstrom": maximum,
        "coordinate_tolerance_angstrom": tolerance,
        "attribution_display_policy": (
            "retained fragment atoms are masked; only generated linker atoms are colored"
        ),
    }


def morgan_fingerprint(molecule: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        MORGAN_RADIUS,
        nBits=MORGAN_BITS,
        useChirality=MORGAN_USE_CHIRALITY,
    )


def morgan_similarity_weights(
    reference: Chem.Mol,
    candidate: Chem.Mol,
) -> Tuple[float, List[float]]:
    """Return RDKit SimilarityMaps' Morgan/Tanimoto atom weights."""

    reference_copy = Chem.Mol(reference)
    candidate_copy = Chem.Mol(candidate)
    baseline = float(
        DataStructs.TanimotoSimilarity(
            morgan_fingerprint(reference_copy),
            morgan_fingerprint(candidate_copy),
        )
    )

    def fingerprint(molecule: Chem.Mol, atomId: int = -1):
        return SimilarityMaps.GetMorganFingerprint(
            molecule,
            atomId=atomId,
            radius=MORGAN_RADIUS,
            fpType="bv",
            nBits=MORGAN_BITS,
            useChirality=MORGAN_USE_CHIRALITY,
        )

    weights = [
        float(value)
        for value in SimilarityMaps.GetAtomicWeightsForFingerprint(
            reference_copy,
            candidate_copy,
            fingerprint,
            metric=DataStructs.TanimotoSimilarity,
        )
    ]
    if len(weights) != candidate.GetNumAtoms() or not all(
        math.isfinite(value) for value in weights
    ):
        raise ValueError("Morgan similarity attribution returned invalid weights")
    return baseline, weights


def qed_atom_deletion_weights(molecule: Chem.Mol) -> Tuple[float, List[float], List[float]]:
    """Compute an explicitly perturbational, non-additive QED deletion heuristic.

    Each atom and all incident bonds are removed from a kekulized copy.  RDKit
    sanitization then supplies implicit hydrogens to newly exposed valences.
    All disconnected components are retained.  A positive weight means the
    original molecule's QED was higher than the atom-deleted molecule's QED.
    """

    baseline_molecule = Chem.Mol(molecule)
    Chem.SanitizeMol(baseline_molecule)
    baseline = float(QED.qed(baseline_molecule))
    weights: List[float] = []
    deleted_scores: List[float] = []
    for atom_index in range(baseline_molecule.GetNumAtoms()):
        edited = Chem.Mol(baseline_molecule)
        try:
            Chem.Kekulize(edited, clearAromaticFlags=True)
            editable = Chem.RWMol(edited)
            editable.RemoveAtom(atom_index)
            deleted = editable.GetMol()
            if deleted.GetNumAtoms() == 0:
                raise ValueError("atom deletion produced an empty molecule")
            Chem.SanitizeMol(deleted)
            score = float(QED.qed(deleted))
        except Exception as exc:
            raise ValueError(
                f"QED atom deletion failed at atom {atom_index}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"QED atom deletion returned non-finite atom {atom_index}")
        deleted_scores.append(score)
        weights.append(baseline - score)
    return baseline, weights, deleted_scores


def attribution_payload(
    *,
    candidate_id: str,
    method: str,
    weights: Sequence[float],
    fragment_indices: Sequence[int],
    baseline: float,
    deleted_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    masked = set(int(value) for value in fragment_indices)
    display_weights = [
        0.0 if index in masked else float(weight)
        for index, weight in enumerate(weights)
    ]
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "method": method,
        "atom_index_base": 0,
        "raw_atom_weights": [float(value) for value in weights],
        "display_atom_weights": display_weights,
        "masked_retained_fragment_atom_indices": sorted(masked),
        "baseline_score": float(baseline),
    }
    if deleted_scores is not None:
        payload["atom_deleted_qed_scores"] = [
            float(value) for value in deleted_scores
        ]
    return payload


def weight_color(weight: float, scale: float) -> Tuple[float, float, float]:
    ratio = max(-1.0, min(1.0, weight / scale)) if scale > 0 else 0.0
    if ratio >= 0:
        return (1.0, 1.0 - 0.70 * ratio, 1.0 - 0.70 * ratio)
    magnitude = -ratio
    return (1.0 - 0.70 * magnitude, 1.0 - 0.70 * magnitude, 1.0)


def draw_atom_weights(
    molecule: Chem.Mol,
    weights: Sequence[float],
    *,
    masked_indices: Sequence[int],
    scale: float,
    legend: str,
    width: int = 520,
    height: int = 400,
) -> Tuple[bytes, bytes]:
    if len(weights) != molecule.GetNumAtoms():
        raise ValueError("Drawing weights do not match molecule atom count")
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    drawing_molecule = Chem.Mol(molecule)
    drawing_molecule.RemoveAllConformers()
    AllChem.Compute2DCoords(drawing_molecule, canonOrient=True, clearConfs=True)
    masked = set(int(value) for value in masked_indices)
    highlighted = [
        index for index in range(drawing_molecule.GetNumAtoms()) if index not in masked
    ]
    colors = {
        index: weight_color(float(weights[index]), scale) for index in highlighted
    }
    radii = {
        index: 0.22 + 0.18 * min(1.0, abs(float(weights[index])) / scale)
        for index in highlighted
    }

    def render(drawer):
        options = drawer.drawOptions()
        options.clearBackground = True
        options.padding = 0.08
        options.addAtomIndices = True
        drawer.DrawMolecule(
            drawing_molecule,
            legend=legend,
            highlightAtoms=highlighted,
            highlightAtomColors=colors,
            highlightAtomRadii=radii,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    png = render(rdMolDraw2D.MolDraw2DCairo(width, height))
    svg = render(rdMolDraw2D.MolDraw2DSVG(width, height))
    if isinstance(svg, str):
        svg = svg.encode("utf-8")
    if not isinstance(png, bytes) or not png.startswith(b"\x89PNG"):
        raise ValueError("RDKit did not produce a PNG attribution map")
    if not isinstance(svg, bytes) or b"<svg" not in svg:
        raise ValueError("RDKit did not produce an SVG attribution map")
    return png, svg


def make_png_grid(
    images: Sequence[bytes],
    *,
    columns: int = 3,
    background: str = "white",
) -> bytes:
    if not images:
        raise ValueError("Cannot make an empty image grid")
    from PIL import Image

    opened = [Image.open(io.BytesIO(payload)).convert("RGB") for payload in images]
    width = max(image.width for image in opened)
    height = max(image.height for image in opened)
    rows = int(math.ceil(len(opened) / columns))
    grid = Image.new("RGB", (columns * width, rows * height), background)
    for index, image in enumerate(opened):
        x = (index % columns) * width + (width - image.width) // 2
        y = (index // columns) * height + (height - image.height) // 2
        grid.paste(image, (x, y))
    output = io.BytesIO()
    grid.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def target_overlay_pml(
    target: str,
    candidate_entries: Sequence[Tuple[str, str]],
) -> bytes:
    """Emit a deterministic, unexecuted PyMOL script using relative paths."""

    lines = [
        "# Protein-aligned molecular views of the selected generated ligands.",
        "# Run this script in PyMOL to render the recorded views.",
        "reinitialize",
        "load reference/protein_reference.pdb, target_protein",
        "load reference/native_reference.sdf, native_reference",
        "hide everything",
        "show cartoon, target_protein",
        "color gray80, target_protein",
        "show sticks, native_reference",
        "color green, native_reference",
    ]
    candidate_objects: List[str] = []
    for index, (candidate_id, relative_directory) in enumerate(candidate_entries, start=1):
        object_name = f"candidate_{index:02d}"
        candidate_objects.append(object_name)
        lines.extend(
            [
                f"# {candidate_id}",
                f"load {relative_directory}/candidate_aligned.sdf, {object_name}",
                f"show sticks, {object_name}",
                f"color cyan, {object_name}",
            ]
        )
    lines.extend(
        [
            "select native_contact_residues, byres (target_protein within 4.0 of native_reference)",
            "show sticks, native_contact_residues",
            "color wheat, native_contact_residues",
            "set stick_radius, 0.18",
            "set cartoon_transparency, 0.25",
            "orient native_reference",
            "zoom native_reference, 12",
            f"# target={target} candidates={len(candidate_objects)}",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    write_bytes(path, stream.getvalue().encode("utf-8"))


def artifact_inventory(root: Path) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "manifest.sha256"}:
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def build_bundle(
    *,
    selection_json: Path,
    contexts_csv: Path,
    snapshot_dir: Path,
    fragment_sdf: Path,
    output_dir: Path,
    references: Mapping[str, str],
    min_ca_atoms: int = 20,
    min_ca_coverage: float = 0.80,
    max_ca_rmsd: float = 2.0,
    fragment_coordinate_tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Build the structural bundle atomically and return its manifest."""

    inputs = [selection_json, contexts_csv, snapshot_dir, fragment_sdf]
    for path in inputs:
        if not path.is_absolute():
            raise ValueError(f"Library build paths must be absolute: {path}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Structural bundle output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    selection, rows_by_target, molecules_by_target = validate_selection(selection_json)
    normalized_references = {
        validate_identifier(str(target).strip().lower(), label="reference target"):
        validate_identifier(str(structure).strip(), label="reference structure_id")
        for target, structure in references.items()
    }
    if set(normalized_references) != set(rows_by_target):
        raise ValueError(
            "References must exactly cover selected targets: "
            f"{sorted(normalized_references)} != {sorted(rows_by_target)}"
        )
    if any(not value for value in normalized_references.values()):
        raise ValueError("Reference structure IDs must be non-empty")

    context_payload = read_bytes(contexts_csv, label="context manifest")
    contexts, _ = load_context_manifest(context_payload)
    snapshot_hashes = load_snapshot_hashes(snapshot_dir)
    fragments, fragment_sdf_sha = load_selected_fragments(
        fragment_sdf, rows_by_target
    )
    asset_cache: Dict[str, Dict[str, Any]] = {}

    def get_assets(structure_id: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
        if structure_id not in contexts:
            raise ValueError(f"Unknown source context structure_id: {structure_id}")
        if structure_id not in asset_cache:
            asset_cache[structure_id] = context_assets(
                contexts[structure_id],
                snapshot_dir=snapshot_dir,
                snapshot_hashes=snapshot_hashes,
            )
        return contexts[structure_id], asset_cache[structure_id]

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=str(output_dir.parent))
    )
    try:
        contact_rows: List[Dict[str, Any]] = []
        target_manifest: Dict[str, Any] = {}
        for target in sorted(rows_by_target):
            # Selection targets have already passed validate_identifier(), so
            # preserve that canonical identifier as the public bundle path.
            # The renderer consumes the same target keys from manifest.json.
            target_directory = staging / "targets" / target
            reference_id = normalized_references[target]
            reference_row, reference_assets = get_assets(reference_id)
            if require_text(reference_row, "pdb", label=reference_id).lower() != target:
                raise ValueError(
                    f"Reference {reference_id} PDB does not match target {target}"
                )
            reference_atoms = protein_atoms(
                parse_pdb_atoms(
                    reference_assets["complex_pdb"]["bytes"],
                    label=f"{reference_id} complex PDB",
                )
            )
            reference_ligand = load_ligand(
                reference_assets["ligand"]["bytes"],
                suffix=reference_assets["ligand"]["path"].suffix,
                label=f"{reference_id} native ligand",
            )
            identity = compute_alignment(
                reference_atoms,
                reference_atoms,
                min_ca_atoms=min_ca_atoms,
                min_ca_coverage=min_ca_coverage,
                max_ca_rmsd=max_ca_rmsd,
            )
            reference_directory = target_directory / "reference"
            write_bytes(
                reference_directory / "protein_reference.pdb",
                transformed_protein_pdb(
                    reference_atoms,
                    identity,
                    remarks=(
                        f"TARGET {target}",
                        f"REFERENCE_STRUCTURE_ID {reference_id}",
                        "CANONICAL ALIGNMENT REFERENCE",
                    ),
                ),
            )
            reference_ligand.SetProp("structure_id", reference_id)
            reference_ligand.SetProp("frame", "canonical_reference")
            write_bytes(
                reference_directory / "native_reference.sdf",
                mol_to_sdf_bytes(reference_ligand, title=f"{target}_native_reference", native_identity=True),
            )

            candidate_work: List[Dict[str, Any]] = []
            candidate_manifest: List[Dict[str, Any]] = []
            similarity_scale_values: List[float] = []
            qed_scale_values: List[float] = []
            for row, candidate in zip(
                rows_by_target[target], molecules_by_target[target]
            ):
                candidate_id = require_text(row, "candidate_id", label=target)
                structure_id = require_text(row, "structure_id", label=candidate_id)
                source_row, source_assets = get_assets(structure_id)
                if require_text(source_row, "pdb", label=structure_id).lower() != target:
                    raise ValueError(
                        f"{candidate_id} source context PDB does not match target {target}"
                    )
                source_uuid = require_integer(
                    row, "source_uuid", label=candidate_id, minimum=0
                )
                fragment_mask = validate_fragment_prefix(
                    candidate,
                    fragments[source_uuid],
                    row,
                    tolerance=fragment_coordinate_tolerance,
                )
                source_atoms = protein_atoms(
                    parse_pdb_atoms(
                        source_assets["complex_pdb"]["bytes"],
                        label=f"{structure_id} complex PDB",
                    )
                )
                alignment = compute_alignment(
                    source_atoms,
                    reference_atoms,
                    min_ca_atoms=min_ca_atoms,
                    min_ca_coverage=min_ca_coverage,
                    max_ca_rmsd=max_ca_rmsd,
                )
                source_native = load_ligand(
                    source_assets["ligand"]["bytes"],
                    suffix=source_assets["ligand"]["path"].suffix,
                    label=f"{structure_id} native ligand",
                )
                source_candidate_coordinates = molecule_coordinates(
                    candidate, label=candidate_id
                )
                source_protein_coordinates = np.vstack(
                    [atom.coordinate for atom in source_atoms]
                )
                contact_distance_change = max_pairwise_distance_change(
                    source_protein_coordinates,
                    source_candidate_coordinates,
                    alignment,
                )
                if contact_distance_change > 1e-9:
                    raise ValueError(
                        f"{candidate_id} rigid transform changed contact distances by "
                        f"{contact_distance_change:.12g} Å"
                    )
                aligned_candidate = transform_molecule(
                    candidate, alignment, label=candidate_id
                )
                aligned_native = transform_molecule(
                    source_native, alignment, label=f"{structure_id} native ligand"
                )
                for molecule, role in (
                    (aligned_candidate, "generated_candidate"),
                    (aligned_native, "source_native_ligand"),
                ):
                    molecule.SetProp("candidate_id", candidate_id)
                    molecule.SetProp("source_structure_id", structure_id)
                    molecule.SetProp("reference_structure_id", reference_id)
                    molecule.SetProp("coordinate_frame", "canonical_reference")
                    molecule.SetProp("structural_followup_label", BUNDLE_LABEL)
                    molecule.SetProp("structural_role", role)

                similarity, similarity_weights = morgan_similarity_weights(
                    source_native, candidate
                )
                qed_score, qed_weights, deleted_scores = qed_atom_deletion_weights(
                    candidate
                )
                similarity_payload = attribution_payload(
                    candidate_id=candidate_id,
                    method="RDKit SimilarityMaps Morgan/Tanimoto atom removal",
                    weights=similarity_weights,
                    fragment_indices=fragment_mask[
                        "retained_fragment_atom_indices"
                    ],
                    baseline=similarity,
                )
                similarity_payload["fingerprint"] = {
                    "type": "Morgan bit vector",
                    "radius": MORGAN_RADIUS,
                    "n_bits": MORGAN_BITS,
                    "use_chirality": MORGAN_USE_CHIRALITY,
                    "metric": TANIMOTO_METRIC,
                    "reference": "source native ligand for this exact KLIFS context",
                }
                qed_payload = attribution_payload(
                    candidate_id=candidate_id,
                    method="QED single-atom deletion perturbation",
                    weights=qed_weights,
                    fragment_indices=fragment_mask[
                        "retained_fragment_atom_indices"
                    ],
                    baseline=qed_score,
                    deleted_scores=deleted_scores,
                )
                qed_payload["interpretation"] = (
                    "Perturbational and non-additive: positive = original QED exceeds "
                    "QED after atom+incident-bond deletion, sanitization, implicit-H "
                    "capping, and retention of all resulting components."
                )
                similarity_scale_values.extend(
                    abs(value)
                    for index, value in enumerate(similarity_weights)
                    if index
                    not in set(fragment_mask["retained_fragment_atom_indices"])
                )
                qed_scale_values.extend(
                    abs(value)
                    for index, value in enumerate(qed_weights)
                    if index
                    not in set(fragment_mask["retained_fragment_atom_indices"])
                )
                candidate_work.append(
                    {
                        "row": row,
                        "candidate_id": candidate_id,
                        "structure_id": structure_id,
                        "alignment": alignment,
                        "source_atoms": source_atoms,
                        "aligned_candidate": aligned_candidate,
                        "aligned_native": aligned_native,
                        "fragment_mask": fragment_mask,
                        "similarity": similarity_payload,
                        "qed": qed_payload,
                        "contact_distance_change": contact_distance_change,
                        "source_assets": source_assets,
                    }
                )

            similarity_scale = max(similarity_scale_values, default=1.0) or 1.0
            qed_scale = max(qed_scale_values, default=1.0) or 1.0
            similarity_pngs: List[bytes] = []
            qed_pngs: List[bytes] = []
            pml_entries: List[Tuple[str, str]] = []
            for item in candidate_work:
                candidate_id = item["candidate_id"]
                structure_id = item["structure_id"]
                candidate_component = safe_component(candidate_id)
                relative_directory = f"candidates/{candidate_component}"
                candidate_directory = target_directory / relative_directory
                write_bytes(
                    candidate_directory / "candidate_aligned.sdf",
                    mol_to_sdf_bytes(
                        item["aligned_candidate"], title=candidate_id
                    ),
                )
                write_bytes(
                    candidate_directory / "native_source_aligned.sdf",
                    mol_to_sdf_bytes(
                        item["aligned_native"],
                        title=f"{candidate_id}_source_native",
                        native_identity=True,
                    ),
                )
                write_bytes(
                    candidate_directory / "protein_aligned.pdb",
                    transformed_protein_pdb(
                        item["source_atoms"],
                        item["alignment"],
                        remarks=(
                            f"CANDIDATE_ID {candidate_id}",
                            f"SOURCE_STRUCTURE_ID {structure_id}",
                            f"REFERENCE_STRUCTURE_ID {reference_id}",
                        ),
                    ),
                )
                contact_path = candidate_directory / "complex_contact_ready.pdb"
                write_bytes(
                    contact_path,
                    contact_ready_pdb(
                        item["source_atoms"],
                        item["alignment"],
                        item["aligned_candidate"],
                        candidate_id=candidate_id,
                        source_structure_id=structure_id,
                    ),
                )
                alignment_payload = item["alignment"].as_dict()
                alignment_payload.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "candidate_id": candidate_id,
                        "source_structure_id": structure_id,
                        "reference_structure_id": reference_id,
                        "maximum_receptor_ligand_pairwise_distance_change_angstrom": item[
                            "contact_distance_change"
                        ],
                    }
                )
                write_json(candidate_directory / "alignment.json", alignment_payload)
                write_json(
                    candidate_directory / "fragment_mask.json",
                    item["fragment_mask"],
                )
                item["similarity"]["target_display_scale"] = similarity_scale
                item["qed"]["target_display_scale"] = qed_scale
                write_json(
                    candidate_directory / "morgan_similarity_weights.json",
                    item["similarity"],
                )
                write_json(
                    candidate_directory / "qed_atom_deletion_weights.json",
                    item["qed"],
                )
                masked = item["fragment_mask"]["retained_fragment_atom_indices"]
                similarity_png, similarity_svg = draw_atom_weights(
                    item["aligned_candidate"],
                    item["similarity"]["raw_atom_weights"],
                    masked_indices=masked,
                    scale=similarity_scale,
                    legend=(
                        f"{candidate_id} | Morgan/Tanimoto "
                        f"{item['similarity']['baseline_score']:.3f}"
                    ),
                )
                qed_png, qed_svg = draw_atom_weights(
                    item["aligned_candidate"],
                    item["qed"]["raw_atom_weights"],
                    masked_indices=masked,
                    scale=qed_scale,
                    legend=(
                        f"{candidate_id} | QED "
                        f"{item['qed']['baseline_score']:.3f}"
                    ),
                )
                write_bytes(
                    candidate_directory / "morgan_similarity_map.png",
                    similarity_png,
                )
                write_bytes(
                    candidate_directory / "morgan_similarity_map.svg",
                    similarity_svg,
                )
                write_bytes(
                    candidate_directory / "qed_atom_deletion_map.png", qed_png
                )
                write_bytes(
                    candidate_directory / "qed_atom_deletion_map.svg", qed_svg
                )
                similarity_pngs.append(similarity_png)
                qed_pngs.append(qed_png)
                pml_entries.append((candidate_id, relative_directory))
                contact_relative = contact_path.relative_to(staging).as_posix()
                contact_rows.append(
                    {
                        "target": target,
                        "candidate_id": candidate_id,
                        "source_structure_id": structure_id,
                        "reference_structure_id": reference_id,
                        "contact_pdb": contact_relative,
                        "contact_pdb_sha256": sha256_file(contact_path),
                        "protonation_policy": "unchanged",
                        "status": "input_only_not_executed",
                    }
                )
                candidate_manifest.append(
                    {
                        "candidate_id": candidate_id,
                        "source_structure_id": structure_id,
                        "source_uuid": require_integer(
                            item["row"], "source_uuid", label=candidate_id
                        ),
                        "fragmentation_id": require_text(
                            item["row"], "fragmentation_id", label=candidate_id
                        ),
                        "directory": (
                            Path("targets")
                            / target
                            / relative_directory
                        ).as_posix(),
                        "source_assets": {
                            name: {
                                "path": portable_path(asset["path"]),
                                "sha256": asset["sha256"],
                            }
                            for name, asset in sorted(
                                item["source_assets"].items()
                            )
                        },
                        "alignment": alignment_payload,
                        "morgan_tanimoto": item["similarity"]["baseline_score"],
                        "qed": item["qed"]["baseline_score"],
                        "fragment_verification": {field: item["row"][field] for field in (
                            "anchor_attachment_pass", "retained_stereo_pass", "retained_nonanchor_explicit_h_match",
                            "retained_nonanchor_total_h_match", "fragment_linker_edges_json",
                            "retained_stereo_errors_json", "retained_nonanchor_h_changes_json")},
                    }
                )

            if similarity_pngs:
                write_bytes(target_directory / "morgan_similarity_grid.png", make_png_grid(similarity_pngs))
            if qed_pngs:
                write_bytes(target_directory / "qed_atom_deletion_grid.png", make_png_grid(qed_pngs))
            write_bytes(
                target_directory / "overlay.pml",
                target_overlay_pml(target, pml_entries),
            )
            target_manifest[target] = {
                "reference_structure_id": reference_id,
                "reference_assets": {
                    name: {
                        "path": portable_path(asset["path"]),
                        "sha256": asset["sha256"],
                    }
                    for name, asset in sorted(reference_assets.items())
                },
                "candidate_count": len(candidate_manifest),
                "similarity_display_scale": similarity_scale,
                "qed_deletion_display_scale": qed_scale,
                "candidates": candidate_manifest,
                "rendering": {
                    "pymol_script_emitted": True,
                    "pymol_executed": False,
                    "camera_definition": "saved_pymol_commands",
                },
            }

        contact_fields = (
            "target",
            "candidate_id",
            "source_structure_id",
            "reference_structure_id",
            "contact_pdb",
            "contact_pdb_sha256",
            "protonation_policy",
            "status",
        )
        contact_csv = staging / "contact_inputs.csv"
        write_csv(contact_csv, contact_fields, contact_rows)
        ligplot_contract = {
            "schema_version": SCHEMA_VERSION,
            "stage": "external_contact_diagram_preflight_only",
            "execution_supported": False,
            "external_tools_executed": False,
            "contact_inputs": {
                "path": "contact_inputs.csv",
                "sha256": sha256_file(contact_csv),
                "records": len(contact_rows),
            },
            "preparation": {
                "protein": "source KLIFS receptor, rigidly aligned only",
                "ligand": "prospectively selected generated candidate, rigidly aligned only",
                "protonation": "unchanged from pinned inputs",
            },
            "required_tool_config": {
                "mode": "required",
                "tools": {
                    name: {
                        "path": "project-root-relative explicit executable",
                        "sha256": "64 lowercase hexadecimal characters",
                    }
                    for name in ("ligplot_plus", "hbplus", "dimplot")
                },
            },
            "policy": (
                "No PATH search and no execution. Run ligplot-check to validate an "
                "explicit hash-pinned configuration before designing a separately "
                "authorized external execution stage."
            ),
        }
        write_json(staging / "ligplot_contract.json", ligplot_contract)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle_label": BUNDLE_LABEL,
            "selection_label": SELECTION_LABEL,
            "candidate_selection_source": "recorded_generation_run",
            "structural_figure_source": "recorded_coordinates",
            "stage_relation": (
                "post-report structural analysis using the recorded candidate ranking"
            ),
            "inputs": {
                "selection_json": {
                    "path": portable_path(selection_json),
                    "sha256": selection["_validated_sha256"],
                },
                "contexts_csv": {
                    "path": portable_path(contexts_csv),
                    "sha256": sha256_bytes(context_payload),
                },
                "snapshot_files_sha256": {
                    "path": portable_path(snapshot_dir / "files.sha256"),
                    "sha256": sha256_file(snapshot_dir / "files.sha256"),
                },
                "fragment_sdf": {
                    "path": portable_path(fragment_sdf),
                    "sha256": fragment_sdf_sha,
                },
            },
            "methods": {
                "alignment": {
                    "algorithm": "proper-rotation C-alpha Kabsch",
                    "minimum_matched_ca_atoms": min_ca_atoms,
                    "minimum_ca_coverage": min_ca_coverage,
                    "maximum_ca_rmsd_angstrom": max_ca_rmsd,
                    "shared_transform_policy": (
                        "source receptor, source native ligand, and candidate use "
                        "the same source-to-reference transform"
                    ),
                },
                "fragment_mask": {
                    "source": "row-aligned pinned KLIFS test fragment SDF",
                    "prefix_coordinate_tolerance_angstrom": (
                        fragment_coordinate_tolerance
                    ),
                },
                "morgan_similarity": {
                    "implementation": "RDKit SimilarityMaps",
                    "radius": MORGAN_RADIUS,
                    "n_bits": MORGAN_BITS,
                    "use_chirality": MORGAN_USE_CHIRALITY,
                    "metric": TANIMOTO_METRIC,
                },
                "qed_atom_deletion": {
                    "implementation": "RDKit QED single-atom deletion heuristic",
                    "definition": (
                        "baseline QED minus QED after atom and incident-bond "
                        "deletion from a kekulized copy, sanitization with implicit-H "
                        "capping, retaining all components"
                    ),
                    "interpretation": "perturbational, non-additive",
                },
            },
            "software": {
                "python_component": "kinase_gap.structural_bundle",
                "rdkit_version": rdBase.rdkitVersion,
                "numpy_version": np.__version__,
                "pymol_available_or_executed": False,
                "ligplot_plus_available_or_executed": False,
            },
            "targets": target_manifest,
            "artifacts": artifact_inventory(staging),
        }
        write_json(staging / "manifest.json", manifest)
        manifest_hash = sha256_file(staging / "manifest.json")
        write_bytes(
            staging / "manifest.sha256",
            f"{manifest_hash}  manifest.json\n".encode("ascii"),
        )
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle_manifest(bundle_dir: Path) -> Dict[str, Any]:
    manifest_payload = read_bytes(bundle_dir / "manifest.json", label="bundle manifest")
    sidecar_payload = read_bytes(
        bundle_dir / "manifest.sha256", label="bundle manifest SHA-256"
    )
    try:
        sidecar = sidecar_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("bundle manifest SHA-256 is not ASCII") from exc
    expected_line = f"{sha256_bytes(manifest_payload)}  manifest.json\n"
    if sidecar != expected_line:
        raise ValueError("bundle manifest SHA-256 sidecar does not match manifest.json")
    manifest = read_json_bytes(manifest_payload, label="bundle manifest")
    if not isinstance(manifest, dict) or manifest.get("bundle_label") != BUNDLE_LABEL:
        raise ValueError("Not a prospective structural-followup bundle")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Bundle manifest has no artifact inventory")
    seen = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("Bundle artifact record is not an object")
        relative = str(record.get("path", ""))
        if not relative or relative in seen:
            raise ValueError(f"Invalid or duplicate bundle artifact path: {relative!r}")
        seen.add(relative)
        path = (bundle_dir / relative).resolve()
        try:
            path.relative_to(bundle_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Bundle artifact escapes its root: {relative}") from exc
        payload = read_bytes(path, label=f"bundle artifact {relative}")
        if sha256_bytes(payload) != record.get("sha256"):
            raise ValueError(f"Bundle artifact SHA-256 mismatch: {relative}")
        if len(payload) != record.get("bytes"):
            raise ValueError(f"Bundle artifact byte count mismatch: {relative}")
    return manifest


def verify_bundle_inputs(
    *,
    bundle_dir: Path,
    selection_json: Path,
    contexts_csv: Path,
    snapshot_dir: Path,
    fragment_sdf: Path,
) -> Dict[str, Any]:
    """Verify an existing bundle and bind it to the current declared inputs."""

    manifest = verify_bundle_manifest(bundle_dir)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Bundle manifest has no input provenance")
    current = {
        "selection_json": sha256_file(selection_json),
        "contexts_csv": sha256_file(contexts_csv),
        "snapshot_files_sha256": sha256_file(snapshot_dir / "files.sha256"),
        "fragment_sdf": sha256_file(fragment_sdf),
    }
    for name, observed in current.items():
        record = inputs.get(name)
        if not isinstance(record, dict) or record.get("sha256") != observed:
            raise ValueError(f"Bundle input SHA-256 mismatch: {name}")
    return manifest


def check_ligplot_contract(
    *,
    bundle_dir: Path,
    config_path: Path,
) -> Dict[str, Any]:
    """Validate contact inputs and an explicit tool config without execution."""

    verify_bundle_manifest(bundle_dir)
    contract_payload = read_bytes(
        bundle_dir / "ligplot_contract.json", label="LigPlot contract"
    )
    contract = read_json_bytes(contract_payload, label="LigPlot contract")
    if not isinstance(contract, dict):
        raise ValueError("LigPlot contract must be an object")
    if contract.get("execution_supported") is not False:
        raise ValueError("LigPlot contract must prohibit execution")
    contact_record = contract.get("contact_inputs")
    if not isinstance(contact_record, dict):
        raise ValueError("LigPlot contract has no contact-input record")
    contact_relative = str(contact_record.get("path", ""))
    contact_path = (bundle_dir / contact_relative).resolve()
    try:
        contact_path.relative_to(bundle_dir.resolve())
    except ValueError as exc:
        raise ValueError("LigPlot contact-input path escapes the bundle") from exc
    contact_payload = read_bytes(contact_path, label="LigPlot contact-input table")
    if sha256_bytes(contact_payload) != contact_record.get("sha256"):
        raise ValueError("LigPlot contact-input table hash mismatch")
    rows, fields = read_csv_bytes(contact_payload, label="LigPlot contact-input table")
    required_fields = {
        "candidate_id",
        "contact_pdb",
        "contact_pdb_sha256",
        "status",
    }
    if not required_fields.issubset(fields):
        raise ValueError("LigPlot contact-input table is missing required columns")
    if len(rows) != contact_record.get("records"):
        raise ValueError("LigPlot contact-input record count mismatch")
    seen_candidates = set()
    for index, row in enumerate(rows):
        candidate_id = require_text(
            row, "candidate_id", label=f"contact-input row {index}"
        )
        if candidate_id in seen_candidates:
            raise ValueError(f"Duplicate LigPlot contact candidate: {candidate_id}")
        seen_candidates.add(candidate_id)
        if row.get("status") != "input_only_not_executed":
            raise ValueError(f"Contact input {candidate_id} has an unsafe status")
        relative = require_text(
            row, "contact_pdb", label=f"contact-input row {index}"
        )
        path = (bundle_dir / relative).resolve()
        try:
            path.relative_to(bundle_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Contact PDB escapes bundle: {relative}") from exc
        if sha256_file(path) != row.get("contact_pdb_sha256"):
            raise ValueError(f"Contact PDB hash mismatch: {candidate_id}")

    config_payload = read_bytes(config_path, label="LigPlot tool config")
    config = read_json_bytes(config_payload, label="LigPlot tool config")
    if not isinstance(config, dict):
        raise ValueError("LigPlot tool config must be an object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("LigPlot tool config schema version is unsupported")
    mode = config.get("mode")
    if mode == "disabled":
        if set(config).difference({"schema_version", "mode", "reason"}):
            raise ValueError("Disabled LigPlot config contains unsupported fields")
        return {
            "schema_version": SCHEMA_VERSION,
            "ready": False,
            "mode": "disabled",
            "reason": str(config.get("reason", "explicitly disabled")),
            "contact_inputs": len(rows),
            "external_tools_executed": False,
            "config_sha256": sha256_bytes(config_payload),
        }
    if mode != "required":
        raise ValueError("LigPlot tool config mode must be 'disabled' or 'required'")
    if set(config) != {"schema_version", "mode", "tools"}:
        raise ValueError("Required LigPlot config has unsupported or missing fields")
    tools = config.get("tools")
    required_tools = {"ligplot_plus", "hbplus", "dimplot"}
    if not isinstance(tools, dict) or set(tools) != required_tools:
        raise ValueError(
            "Required LigPlot config must define exactly ligplot_plus, hbplus, dimplot"
        )
    validated_tools: Dict[str, Any] = {}
    for name in sorted(required_tools):
        record = tools[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"LigPlot tool {name} must contain only path and sha256")
        expected = str(record["sha256"])
        if SHA256_PATTERN.fullmatch(expected) is None:
            raise ValueError(f"LigPlot tool {name} has an invalid SHA-256")
        executable = project_path(str(record["path"]), must_exist=True)
        if not executable.is_file():
            raise ValueError(f"LigPlot tool {name} is not a regular file")
        if not executable.stat().st_mode & (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        ):
            raise ValueError(f"LigPlot tool {name} is not executable")
        observed = sha256_file(executable)
        if observed != expected:
            raise ValueError(
                f"LigPlot tool {name} SHA-256 mismatch: "
                f"expected {expected}, observed {observed}"
            )
        validated_tools[name] = {
            "path": portable_path(executable),
            "sha256": observed,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": True,
        "mode": "required",
        "contact_inputs": len(rows),
        "tools": validated_tools,
        "external_tools_executed": False,
        "config_sha256": sha256_bytes(config_payload),
        "next_action": (
            "A separate, explicitly authorized execution stage is still required; "
            "this checker does not run LigPlot+, HBPLUS, or DIMPlot."
        ),
    }


def parse_references(values: Sequence[str]) -> Dict[str, str]:
    if not values:
        return dict(DEFAULT_REFERENCES)
    references: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Reference must be TARGET=STRUCTURE_ID: {value!r}")
        target, structure_id = value.split("=", 1)
        target = target.strip().lower()
        structure_id = structure_id.strip()
        if not target or not structure_id:
            raise ValueError(f"Reference must be TARGET=STRUCTURE_ID: {value!r}")
        validate_identifier(target, label="reference target")
        validate_identifier(structure_id, label="reference structure_id")
        if target in references:
            raise ValueError(f"Duplicate reference target: {target}")
        references[target] = structure_id
    return references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate a prospective structural-analysis bundle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="build the CPU-only post-report structural bundle",
    )
    build.add_argument("--selection-json", required=True)
    build.add_argument("--contexts", required=True)
    build.add_argument("--snapshot", required=True)
    build.add_argument("--fragment-sdf", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="TARGET=STRUCTURE_ID",
        help="repeat for each target; defaults to the pinned 5lqf/4ag8 contexts",
    )
    build.add_argument("--min-ca-atoms", type=int, default=20)
    build.add_argument("--min-ca-coverage", type=float, default=0.80)
    build.add_argument("--max-ca-rmsd", type=float, default=2.0)
    build.add_argument("--fragment-coordinate-tolerance", type=float, default=1e-3)

    check = subparsers.add_parser(
        "ligplot-check",
        help="validate, but never execute, an explicit LigPlot tool contract",
    )
    check.add_argument("--bundle-dir", required=True)
    check.add_argument("--config", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="verify a completed bundle and its current declared inputs",
    )
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--selection-json", required=True)
    verify.add_argument("--contexts", required=True)
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--fragment-sdf", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_bundle(
            selection_json=project_path(args.selection_json, must_exist=True),
            contexts_csv=project_path(args.contexts, must_exist=True),
            snapshot_dir=project_path(args.snapshot, must_exist=True),
            fragment_sdf=project_path(args.fragment_sdf, must_exist=True),
            output_dir=project_path(args.output_dir),
            references=parse_references(args.reference),
            min_ca_atoms=args.min_ca_atoms,
            min_ca_coverage=args.min_ca_coverage,
            max_ca_rmsd=args.max_ca_rmsd,
            fragment_coordinate_tolerance=args.fragment_coordinate_tolerance,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "bundle_label": manifest["bundle_label"],
                    "targets": sorted(manifest["targets"]),
                    "external_tools_executed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "ligplot-check":
        result = check_ligplot_contract(
            bundle_dir=project_path(args.bundle_dir, must_exist=True),
            config_path=project_path(args.config, must_exist=True),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "verify":
        manifest = verify_bundle_inputs(
            bundle_dir=project_path(args.bundle_dir, must_exist=True),
            selection_json=project_path(args.selection_json, must_exist=True),
            contexts_csv=project_path(args.contexts, must_exist=True),
            snapshot_dir=project_path(args.snapshot, must_exist=True),
            fragment_sdf=project_path(args.fragment_sdf, must_exist=True),
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "bundle_label": manifest["bundle_label"],
                    "targets": sorted(manifest["targets"]),
                    "current_inputs_match": True,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
