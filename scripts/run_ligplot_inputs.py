#!/usr/bin/env python3
"""Render contact packages with explicit ligand chemistry and HBPLUS geometry."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import re
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Lipinski
from package_structural_inputs import load_molecule, sha256


def atom_name(atom):
    name = atom.GetSymbol() + str(atom.GetIdx() + 1)
    return " " + name.ljust(3) if len(atom.GetSymbol()) == 1 and len(name) < 4 else name.ljust(4)


def write_ligand_definitions(molecule, directory):
    """Write CCD-format bonds and explicit SMARTS-derived HBPLUS atom roles."""
    mol = Chem.Mol(molecule)
    donor_atoms = {match[0] for match in Lipinski._HDonors(mol)}
    acceptor_atoms = {match[0] for match in Lipinski._HAcceptors(mol)}
    names = [atom_name(atom) for atom in mol.GetAtoms()]
    rc = ["-u LIG"]
    for start in range(0, len(names), 10):
        rc.append('-M LIG "' + "".join(names[start:start + 10]) + '"')
    bonds = [names[b.GetBeginAtomIdx()] + names[b.GetEndAtomIdx()] + ":" for b in mol.GetBonds()]
    for start in range(0, len(bonds), 5):
        rc.append('-T LIG "' + "".join(bonds[start:start + 5]) + '"')
    roles = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        donors = atom.GetTotalNumHs() if index in donor_atoms else 0
        # HBPLUS -1 denotes a pi-face acceptor normal to a ring plane; a
        # pyridine-like N is an ordinary lone-pair acceptor, including in rings.
        acceptors = (2 if atom.GetSymbol() in ("O", "S") else 1) if index in acceptor_atoms else 0
        rc.append(f'-E LIG "{names[index]}" {donors}')
        rc.append(f'-e LIG "{names[index]}" {acceptors}')
        roles.append({"atom_index": index, "pdb_atom_name": names[index].strip(),
                      "donor_hydrogens": donors, "acceptor": index in acceptor_atoms,
                      "atom_is_aromatic": atom.GetIsAromatic(), "hbplus_acceptor_capacity": acceptors})
    (directory / "ligand_definitions.rc").write_text("\n".join(rc) + "\n")
    with (directory / "ligand_atom_roles.json").open("x") as handle:
        json.dump({"method": "RDKit Lipinski.HDonorSmarts and HAcceptorSmarts", "rdkit_version": rdBase.rdkitVersion,
                   "donor_smarts": Chem.MolToSmarts(Lipinski.HDonorSmarts),
                   "acceptor_smarts": Chem.MolToSmarts(Lipinski.HAcceptorSmarts), "atoms": roles}, handle, indent=2)
    extended = Chem.AddHs(mol, addCoords=True)
    Chem.Kekulize(extended, clearAromaticFlags=False)
    lines = ["data_LIG", "_chem_comp.id LIG", "_chem_comp.name 'Archived kinase ligand'", "_chem_comp.type 'NON-POLYMER'",
             "_chem_comp.pdbx_release_status REL", "loop_"]
    fields = ["comp_id", "atom_id", "alt_atom_id", "type_symbol", "charge", "pdbx_aromatic_flag", "pdbx_leaving_atom_flag", "pdbx_stereo_config",
              "model_Cartn_x", "model_Cartn_y", "model_Cartn_z", "pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal", "pdbx_ordinal"]
    lines.extend("_chem_comp_atom." + field for field in fields)
    for index, atom in enumerate(extended.GetAtoms()):
        name = atom_name(atom).strip()
        xyz = [f"{v:.3f}" for v in extended.GetConformer().GetPositions()[index]]
        lines.append(" ".join(["LIG", name, name, atom.GetSymbol(), str(atom.GetFormalCharge()), "Y" if atom.GetIsAromatic() else "N", "N", "N"] + xyz + xyz + [str(index + 1)]))
    lines.extend(["#", "loop_"])
    lines.extend("_chem_comp_bond." + field for field in ["comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag", "pdbx_stereo_config", "pdbx_ordinal"])
    for index, bond in enumerate(extended.GetBonds()):
        names2 = [atom_name(extended.GetAtomWithIdx(i)).strip() for i in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]
        order = {"SINGLE": "SING", "DOUBLE": "DOUB", "TRIPLE": "TRIP"}.get(str(bond.GetBondType()))
        if order is None:
            raise ValueError(f"Unsupported ligand bond order: {bond.GetBondType()}")
        lines.append(" ".join(["LIG"] + names2 + [order, "Y" if bond.GetIsAromatic() else "N", "N", str(index + 1)]))
    (directory / "ligand.cif").write_text("\n".join(lines + ["#"]) + "\n")


def parse_hhb(line):
    if len(line) < 81 or line[9:10] != "/" or line[29:30] != "/":
        raise ValueError(f"Unexpected HBPLUS long-format record: {line!r}")
    result = {"donor_chain": line[10], "donor_residue_number": line[11:15].strip(), "donor_residue": line[16:19], "donor_atom": line[19:23].strip(),
              "acceptor_chain": line[30], "acceptor_residue_number": line[31:35].strip(), "acceptor_residue": line[36:39], "acceptor_atom": line[39:43].strip(),
              "donor_acceptor_angstrom": float(line[44:49]), "donor_hydrogen_acceptor_degrees": float(line[60:65]),
              "hydrogen_acceptor_angstrom": float(line[65:70]), "hydrogen_acceptor_antecedent_degrees": float(line[70:75]),
              "donor_acceptor_antecedent_degrees": float(line[75:81])}
    result["complete_geometry"] = all(result[key] >= 0 for key in ["donor_acceptor_angstrom", "donor_hydrogen_acceptor_degrees", "hydrogen_acceptor_angstrom", "hydrogen_acceptor_antecedent_degrees", "donor_acceptor_antecedent_degrees"])
    result["figure_pass"] = result["complete_geometry"] and result["donor_acceptor_angstrom"] <= 3.35 and result["hydrogen_acceptor_angstrom"] <= 2.70 and all(result[key] >= 90 for key in ["donor_hydrogen_acceptor_degrees", "hydrogen_acceptor_antecedent_degrees", "donor_acceptor_antecedent_degrees"])
    return result


def run(command, directory, label):
    result = subprocess.run([str(arg) for arg in command], cwd=directory, input="", text=True, capture_output=True, timeout=180)
    with (directory / f"{label}.log").open("x") as handle:
        handle.write(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}: {directory}")
    return {"arguments": [str(arg) for arg in command], "returncode": result.returncode}


def process_one(source, output, tools):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("complex.pdb", "ligand.sdf", "atom_mapping.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    mol = load_molecule(output / "ligand.sdf")
    write_ligand_definitions(mol, output)
    exe = tools / "lib/exe_linux64"
    commands = []
    commands.append(run([exe / "hbadd", output / "complex.pdb", output / "ligand.cif", "-wkdir", str(output) + "/"], output, "hbadd"))
    for label, options in [("hbplus_hb", ["-h", "2.70", "-d", "3.35"]), ("hbplus_nb", ["-h", "2.90", "-d", "3.90", "-N"])]:
        commands.append(run([exe / "hbplus", "-L", "-f", output / "ligand_definitions.rc"] + options + [output / "complex.pdb", "-wkdir", str(output) + "/"], output, label))
    contacts, retained = [], []
    for line in (output / "complex.hhb").read_text().splitlines(keepends=True):
        if not line.strip():
            continue
        parsed = parse_hhb(line)
        if parsed["figure_pass"]:
            retained.append(line)
        if (parsed["donor_chain"] == "Z" and parsed["donor_residue"] == "LIG") or (parsed["acceptor_chain"] == "Z" and parsed["acceptor_residue"] == "LIG"):
            contacts.append(parsed)
    shutil.copyfile(output / "complex.pdb", output / "figure.pdb")
    shutil.copyfile(output / "complex.nnb", output / "figure.nnb")
    with (output / "figure.hhb").open("x") as handle:
        handle.writelines(retained)
    parameters = (tools / "lib/params/ligplot.prm").read_text()
    for label in ("Print: Key to symbols", "Print: Atom names", "Print: Filename as title"):
        lines = parameters.splitlines(keepends=True)
        matches = [i for i, line in enumerate(lines) if label in line]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one LigPlot display setting: {label}")
        index = matches[0]
        lines[index] = "N" + lines[index][1:]
        parameters = "".join(lines)
    (output / "display.prm").write_text(parameters)
    commands.append(run([exe / "ligplot", output / "figure.pdb", "900", "900", "Z", "-wkdir", str(output) + "/",
                         "-prm", output / "display.prm", "-ctype", "1"], output, "ligplot"))
    if not (output / "ligplot.ps").is_file():
        raise RuntimeError(f"LigPlot produced no PostScript file: {output}")
    commands.append(run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-sOutputFile=" + str(output / "diagram.pdf"), output / "ligplot.ps"], output, "pdf_export"))
    commands.append(run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300", "-sOutputFile=" + str(output / "diagram.png"), output / "ligplot.ps"], output, "png_export"))
    with (output / "hydrogen_bonds.json").open("x") as handle:
        json.dump(contacts, handle, indent=2)
    with (output / "execution.json").open("x") as handle:
        ligplot_version = re.search(r"Running LIGPLOT v[^\s]+", (output / "ligplot.log").read_text())
        hbplus_version = re.search(r"HBPLUS Hydrogen Bond Calculator v [\d.]+", (output / "hbplus_hb.log").read_text())
        json.dump({"commands": commands, "ligand_input_sha256": sha256(source / "ligand.sdf"), "complex_input_sha256": sha256(source / "complex.pdb"),
                   "hydrogen_bond_policy": "Recorded HBPLUS distance and angle thresholds, with all geometry values defined",
                   "ligand_roles": "RDKit Lipinski donor and acceptor SMARTS", "ligplot_binary_sha256": sha256(exe / "ligplot"),
                   "hbplus_binary_sha256": sha256(exe / "hbplus"), "hbadd_binary_sha256": sha256(exe / "hbadd"),
                   "ligplot_version": ligplot_version.group() if ligplot_version else None,
                   "hbplus_version": hbplus_version.group() if hbplus_version else None,
                   "thresholds": {"maximum_donor_acceptor_angstrom": 3.35, "maximum_hydrogen_acceptor_angstrom": 2.7,
                                  "minimum_dha_h_aa_d_aa_degrees": 90, "nonbonded_h_parameter": 2.9, "nonbonded_d_parameter": 3.9}}, handle, indent=2)
    return contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ligplus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.inputs = args.inputs.resolve()
    args.ligplus = args.ligplus.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    distribution = args.ligplus.parent / "download_manifest.json"
    if distribution.is_file():
        shutil.copyfile(distribution, args.output / "tool_distribution.json")
    with (args.inputs / "contact_inputs.csv").open() as handle:
        entries = list(csv.DictReader(handle))
    summary = []
    for entry in entries:
        source = (args.inputs / entry["complex_pdb"]).resolve().parent
        if args.inputs.resolve() not in source.parents:
            raise ValueError("Contact input escapes package")
        contacts = process_one(source, args.output / entry["target"] / entry["panel"], args.ligplus)
        summary.append({"target": entry["target"], "panel": entry["panel"], "candidate_id": entry["candidate_id"],
                        "raw_hydrogen_bonds": len(contacts), "figure_hydrogen_bonds": sum(c["figure_pass"] for c in contacts)})
    with (args.output / "contact_summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


if __name__ == "__main__":
    main()
