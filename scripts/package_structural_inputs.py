#!/usr/bin/env python3
"""Create coordinate-preserving native and generated ligand contact inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def copy(source, target):
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def load_molecule(path):
    from kinase_gap.structural_bundle import parse_sdf_bytes
    molecules = parse_sdf_bytes(path.read_bytes(), label=str(path), sanitize=True)
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"Expected one sanitized molecule: {path}")
    mol = molecules[0]
    if mol.GetNumConformers() != 1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError(f"Expected one finite conformer: {path}")
    return mol


def write_sdf(mol, path):
    from kinase_gap.fragmentation import canonical_isomeric_identity
    original = canonical_isomeric_identity(mol)
    if mol.HasProp("kinase_native_smiles"):
        from kinase_gap.chemistry_io import checked_native_mol_block
        block, serialized = checked_native_mol_block(mol)
    else:
        serialized = Chem.Mol(mol)
        block = Chem.MolToMolBlock(serialized, includeStereo=True, kekulize=True)
    with path.open("x") as handle:
        handle.write(block.rstrip() + "\n")
        for name in sorted(serialized.GetPropNames(includePrivate=False, includeComputed=False)):
            handle.write(f">  <{name}>\n{serialized.GetProp(name)}\n\n")
        handle.write("$$$$\n")
    readback = load_molecule(path)
    if canonical_isomeric_identity(readback) != original:
        raise ValueError(f"Chemical identity changed in SDF round trip: {path}")
    if [a.GetSymbol() for a in readback.GetAtoms()] != [a.GetSymbol() for a in mol.GetAtoms()]:
        raise ValueError(f"Atom ordering changed in SDF round trip: {path}")
    delta = float(np.max(np.abs(mol.GetConformer().GetPositions() - readback.GetConformer().GetPositions())))
    if delta > 0.00006:
        raise ValueError(f"SDF coordinate change exceeds four-decimal precision: {path}")
    return original


def write_complex(protein, mol, output):
    protein_lines = [line for line in protein.read_text().splitlines() if line.startswith("ATOM  ")]
    if not protein_lines:
        raise ValueError(f"No receptor ATOM records: {protein}")
    if any(line[21:22] == "Z" and int(line[22:26]) == 900 for line in protein_lines):
        raise ValueError("Receptor contains residue Z900 reserved for the ligand")
    first_serial = max(int(line[6:11]) for line in protein_lines) + 1
    if first_serial + mol.GetNumAtoms() >= 100000:
        raise ValueError("PDB atom serial field overflow")
    lines = ["REMARK 950 LIGAND LIG CHAIN Z RESIDUE 900", "REMARK 950 COORDINATES PRESERVED FROM ALIGNED INPUT SDF"] + protein_lines + ["TER"]
    coordinates = mol.GetConformer().GetPositions()
    atom_records, bond_records = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        name = f"{atom.GetSymbol()}{i + 1}"
        if len(name) > 4:
            raise ValueError("PDB ligand atom name exceeds four characters")
        serial = first_serial + i
        x, y, z = coordinates[i]
        name_field = " " + name.ljust(3) if len(atom.GetSymbol()) == 1 and len(name) < 4 else name.ljust(4)
        lines.append(f"HETATM{serial:5d} {name_field} LIG Z 900    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom.GetSymbol():>2s}")
        atom_records.append({"sdf_atom_index": i, "pdb_atom_name": name, "pdb_serial": serial,
                             "element": atom.GetSymbol(), "formal_charge": atom.GetFormalCharge(),
                             "explicit_hydrogens": atom.GetNumExplicitHs(), "implicit_hydrogens": atom.GetNumImplicitHs(),
                             "aromatic": atom.GetIsAromatic(), "x": float(x), "y": float(y), "z": float(z)})
    for atom in mol.GetAtoms():
        neighbours = sorted(first_serial + a.GetIdx() for a in atom.GetNeighbors())
        for start in range(0, len(neighbours), 4):
            lines.append(f"CONECT{first_serial + atom.GetIdx():5d}" + "".join(f"{n:5d}" for n in neighbours[start:start + 4]))
    for bond in mol.GetBonds():
        bond_records.append({"begin_sdf_index": bond.GetBeginAtomIdx(), "end_sdf_index": bond.GetEndAtomIdx(),
                             "bond_type": str(bond.GetBondType()), "aromatic": bond.GetIsAromatic()})
    lines.append("END")
    with output.open("x", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")
    pdb_xyz = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines if line.startswith("HETATM")])
    if not np.allclose(pdb_xyz, coordinates, atol=0.00051, rtol=0):
        raise ValueError("PDB ligand coordinates changed beyond PDB precision")
    return {"index_base": 0, "ligand_residue": {"name": "LIG", "chain": "Z", "number": 900},
            "atoms": atom_records, "bonds": bond_records,
            "pdb_coordinate_precision_angstrom": 0.001, "sdf_coordinate_precision_angstrom": 0.0001}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "kinase-abr" in str(args.structural_dir.resolve()):
        raise ValueError("Excluded source path")
    manifest_path = args.structural_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for target, target_info in sorted(manifest["targets"].items()):
        target_path = args.structural_dir / "targets" / target
        inputs = [("native", "native", target_path / "reference/protein_reference.pdb", target_path / "reference/native_reference.sdf", None)]
        for i, record in enumerate(target_info["candidates"], start=1):
            source = (args.structural_dir / record["directory"]).resolve()
            if args.structural_dir.resolve() not in source.parents:
                raise ValueError("Candidate directory escapes structural root")
            inputs.append((f"candidate_{i:02d}", record["candidate_id"], source / "protein_aligned.pdb", source / "candidate_aligned.sdf", source))
        for label, candidate_id, protein, ligand, source in inputs:
            out = args.output / target / label
            out.mkdir(parents=True)
            molecule = load_molecule(ligand)
            smiles = write_sdf(molecule, out / "ligand.sdf")
            copy(protein, out / "receptor.pdb")
            mapping = write_complex(protein, molecule, out / "complex.pdb")
            mapping.update({"candidate_id": candidate_id, "target": target, "canonical_smiles": smiles,
                            "input_ligand_sha256": sha256(ligand), "input_receptor_sha256": sha256(protein)})
            dump(out / "atom_mapping.json", mapping)
            from run_ligplot_inputs import write_ligand_definitions
            write_ligand_definitions(molecule, out)
            if source:
                for name in ("fragment_mask.json", "alignment.json"):
                    copy(source / name, out / name)
            records.append({"target": target, "panel": label, "candidate_id": candidate_id,
                            "complex_pdb": str((out / "complex.pdb").relative_to(args.output)),
                            "ligand_sdf": str((out / "ligand.sdf").relative_to(args.output)),
                            "ligand_name": "LIG", "ligand_chain": "Z", "ligand_residue": 900,
                            "complex_sha256": sha256(out / "complex.pdb"), "ligand_sha256": sha256(out / "ligand.sdf")})
    with (args.output / "contact_inputs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    copy(manifest_path, args.output / "source_structural_manifest.json")
    copy(Path(__file__), args.output / Path(__file__).name)
    dump(args.output / "manifest.json", {"schema_version": 1, "source_structural_manifest_sha256": sha256(manifest_path),
                                        "records": records, "coordinate_policy": "Use the aligned source coordinates and source protonation state",
                                        "chemical_identity_check": "Canonical isomeric SMILES equality after kekulized SDF round trip with native identity properties restored"})


if __name__ == "__main__":
    main()
