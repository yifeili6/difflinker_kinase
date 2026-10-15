#!/usr/bin/env python3
"""Tabulate receptor–ligand heavy-atom distances independently of filtering."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem, rdBase
from package_structural_inputs import load_molecule, sha256


def receptor_atoms(path):
    records = []
    periodic = Chem.GetPeriodicTable()
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        symbol = line[76:78].strip()
        if not symbol:
            name = line[12:16].strip().lstrip("0123456789")
            symbol = name[0]
        symbol = symbol.capitalize()
        if symbol in ("H", "D"):
            continue
        number = periodic.GetAtomicNumber(symbol)
        if not number:
            raise ValueError(f"Unknown receptor element {symbol}")
        records.append({"serial": int(line[6:11]), "atom": line[12:16].strip(),
                        "residue": line[17:20].strip(), "chain": line[21:22],
                        "residue_number": line[22:26].strip(), "insertion_code": line[26:27].strip(),
                        "element": symbol, "vdw_radius": periodic.GetRvdw(number),
                        "coordinates": [float(line[30:38]), float(line[38:46]), float(line[46:54])]})
    if not records:
        raise ValueError(f"No receptor heavy atoms: {path}")
    return records


def inspect(protein, molecule, overlap_scale, contact_radius):
    atoms = receptor_atoms(protein)
    periodic = Chem.GetPeriodicTable()
    indices = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]
    positions = molecule.GetConformer().GetPositions()[indices]
    distances = cdist(positions, np.array([a["coordinates"] for a in atoms]))
    ligand_radii = np.array([periodic.GetRvdw(molecule.GetAtomWithIdx(i).GetAtomicNum()) for i in indices])
    radii_sum = ligand_radii[:, None] + np.array([a["vdw_radius"] for a in atoms])[None, :]
    ratios = distances / radii_sum
    close = distances <= contact_radius
    rows, nearest = [], []
    for ligand_row, index in enumerate(indices):
        neighbour = int(np.argmin(distances[ligand_row]))
        nearest.append({"ligand_atom_index": index, "nearest_distance_angstrom": float(distances[ligand_row, neighbour]),
                        "nearest_receptor_atom_serial": atoms[neighbour]["serial"],
                        "nearest_receptor_residue": atoms[neighbour]["residue"],
                        "nearest_receptor_chain": atoms[neighbour]["chain"],
                        "nearest_receptor_residue_number": atoms[neighbour]["residue_number"],
                        "minimum_distance_vdw_ratio": float(ratios[ligand_row].min())})
        for receptor_index in np.flatnonzero(close[ligand_row]):
            atom = atoms[receptor_index]
            rows.append({"ligand_atom_index": index, "ligand_element": molecule.GetAtomWithIdx(index).GetSymbol(),
                         "receptor_serial": atom["serial"], "receptor_atom": atom["atom"], "receptor_element": atom["element"],
                         "receptor_residue": atom["residue"], "receptor_chain": atom["chain"], "receptor_residue_number": atom["residue_number"],
                         "distance_angstrom": float(distances[ligand_row, receptor_index]),
                         "vdw_radius_sum_angstrom": float(radii_sum[ligand_row, receptor_index]),
                         "distance_vdw_ratio": float(ratios[ligand_row, receptor_index]),
                         "below_overlap_threshold": bool(ratios[ligand_row, receptor_index] < overlap_scale)})
    summary = {"ligand_heavy_atoms": len(indices), "receptor_heavy_atoms": len(atoms),
               "minimum_distance_angstrom": float(distances.min()), "minimum_distance_vdw_ratio": float(ratios.min()),
               "pairs_within_contact_radius": int(close.sum()), "pairs_below_overlap_threshold": int((ratios < overlap_scale).sum()),
               "ligand_atoms_below_overlap_threshold": int((ratios < overlap_scale).any(axis=1).sum())}
    return summary, rows, nearest


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-scale", type=float, default=0.75)
    parser.add_argument("--contact-radius", type=float, default=6.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.inputs / "contact_inputs.csv").open() as handle:
        entries = list(csv.DictReader(handle))
    summaries = []
    for entry in entries:
        source = (args.inputs / entry["complex_pdb"]).resolve().parent
        if args.inputs.resolve() not in source.parents:
            raise ValueError("Input path escapes package")
        ligand, receptor = source / "ligand.sdf", source / "receptor.pdb"
        summary, contacts, nearest = inspect(receptor, load_molecule(ligand), args.overlap_scale, args.contact_radius)
        output = args.output / entry["target"] / entry["panel"]
        output.mkdir(parents=True)
        write_csv(output / "contacts_within_radius.csv", contacts)
        write_csv(output / "nearest_receptor_atoms.csv", nearest)
        summary.update({"target": entry["target"], "panel": entry["panel"], "candidate_id": entry["candidate_id"],
                        "ligand_sha256": sha256(ligand), "receptor_sha256": sha256(receptor)})
        summaries.append(summary)
    write_csv(args.output / "summary.csv", summaries)
    with (args.output / "method.json").open("x") as handle:
        json.dump({"method": "All receptor–ligand heavy-atom distances", "radii": "RDKit periodic table van der Waals radii",
                   "rdkit_version": rdBase.rdkitVersion, "overlap_rule": "distance < overlap_scale * (ligand_radius + receptor_radius)",
                   "overlap_scale": args.overlap_scale, "contact_table_radius_angstrom": args.contact_radius,
                   "stage": "Post-filter structural diagnostic"}, handle, indent=2)


if __name__ == "__main__":
    main()
