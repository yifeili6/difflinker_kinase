#!/usr/bin/env python3
"""Measure receptor–ligand heavy-atom distances for generated and native poses."""
import argparse
import base64
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import scipy
from scipy.spatial import cKDTree
from rdkit import Chem, rdBase
from kinase_gap.fragmentation import canonical_isomeric_identity
from kinase_gap.structural_bundle import (load_context_manifest, molecule_coordinates,
                                         molecule_identifier, mol_to_sdf_bytes, parse_sdf_bytes)

META_FIELDS = ["population", "candidate_id", "source_uuid", "fragmentation_id", "structure_id",
               "pdb", "kinase", "context_index", "linker_size", "canonical_smiles"]
MEASURE_FIELDS = ["ligand_heavy_atoms", "receptor_heavy_atoms", "minimum_distance_angstrom",
                  "minimum_distance_vdw_ratio", "pairs_within_contact_radius",
                  "pairs_below_overlap_threshold", "ligand_atoms_below_overlap_threshold"]
NEAREST_FIELDS = META_FIELDS[:8] + ["ligand_atom_index", "ligand_element", "nearest_distance_angstrom",
                 "nearest_receptor_atom_serial", "nearest_receptor_atom", "nearest_receptor_residue",
                 "nearest_receptor_residue_number", "minimum_distance_vdw_ratio",
                 "pairs_below_overlap_threshold"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path):
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, fields, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def vdw_radius(number):
    if not 1 <= number <= 118:
        raise ValueError("An atom has no defined chemical element")
    radius = float(Chem.GetPeriodicTable().GetRvdw(number))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("An atom has no positive finite van der Waals radius")
    return radius


def receptor_atoms(path):
    """Read source MOL2 atom types and coordinates without assigning protein bonds."""
    records, serials, section, declared_count, observed_count = [], set(), None, None, 0
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            if section == "@<TRIPOS>MOLECULE":
                if declared_count is not None:
                    raise ValueError("Protein MOL2 contains more than one molecule")
                declared_count = int(lines[index + 2].split()[0])
            continue
        if section != "@<TRIPOS>ATOM" or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError("Incomplete protein MOL2 atom record")
        serial = int(fields[0])
        if serial in serials:
            raise ValueError("Duplicate protein MOL2 atom serial")
        serials.add(serial)
        observed_count += 1
        coordinates = np.array([float(value) for value in fields[2:5]], dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError("Protein MOL2 contains nonfinite coordinates")
        symbol = fields[5].split(".")[0].capitalize()
        symbol = "H" if symbol in ("D", "T") else symbol
        try:
            number = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
        except RuntimeError as exc:
            raise ValueError("Unknown protein MOL2 element: " + symbol) from exc
        radius = vdw_radius(number)
        if number == 1:
            continue
        records.append({"serial": serial, "atom": fields[1], "element": symbol,
                        "residue_number": fields[6], "residue": fields[7],
                        "vdw_radius": radius, "coordinates": coordinates})
    if declared_count != observed_count or not records:
        raise ValueError("Protein MOL2 atom count disagrees with its header or has no heavy atoms")
    return records


class ReceptorIndex:
    """Exact nearest-distance and radius-normalized queries with radius-group trees."""
    def __init__(self, atoms):
        self.atoms = atoms
        self.coordinates = np.asarray([atom["coordinates"] for atom in atoms], dtype=float)
        self.radii = np.asarray([atom["vdw_radius"] for atom in atoms], dtype=float)
        if (self.coordinates.shape != (len(atoms), 3) or not len(atoms)
                or not np.isfinite(self.coordinates).all() or not np.isfinite(self.radii).all()
                or np.any(self.radii <= 0)):
            raise ValueError("Receptor requires finite coordinates and positive radii")
        self.tree = cKDTree(self.coordinates)
        self.radius_trees = [(radius, cKDTree(self.coordinates[self.radii == radius]))
                             for radius in sorted(set(self.radii))]

    def inspect(self, molecule, overlap_scale=0.75, contact_radius=6.0):
        if not np.isfinite(overlap_scale) or overlap_scale <= 0 or not np.isfinite(contact_radius) or contact_radius <= 0:
            raise ValueError("Distance criteria must be finite and positive")
        all_positions = molecule_coordinates(molecule, label="ligand")
        indices = []
        for atom in molecule.GetAtoms():
            vdw_radius(atom.GetAtomicNum())
            if atom.GetAtomicNum() > 1:
                indices.append(atom.GetIdx())
        if not indices:
            raise ValueError("Ligand has no heavy atoms")
        positions = all_positions[indices]
        radii = np.array([vdw_radius(molecule.GetAtomWithIdx(i).GetAtomicNum()) for i in indices])
        distances, neighbours = self.tree.query(positions, k=1)
        minimum_ratios = np.minimum.reduce([tree.query(positions, k=1)[0] / (radii + radius)
                                            for radius, tree in self.radius_trees])
        candidates = self.tree.query_ball_point(positions, overlap_scale * (radii + self.radii.max()))
        counts = np.array([int(np.sum(np.linalg.norm(self.coordinates[near] - position, axis=1)
                                     < overlap_scale * (radius + self.radii[near])))
                           for position, radius, near in zip(positions, radii, candidates)])
        contact_count = sum(len(near) for near in self.tree.query_ball_point(positions, contact_radius))
        nearest = []
        for i, index in enumerate(indices):
            receptor = self.atoms[int(neighbours[i])]
            nearest.append({"ligand_atom_index": index, "ligand_element": molecule.GetAtomWithIdx(index).GetSymbol(),
                            "nearest_distance_angstrom": float(distances[i]),
                            "nearest_receptor_atom_serial": receptor["serial"], "nearest_receptor_atom": receptor["atom"],
                            "nearest_receptor_residue": receptor["residue"], "nearest_receptor_residue_number": receptor["residue_number"],
                            "minimum_distance_vdw_ratio": float(minimum_ratios[i]),
                            "pairs_below_overlap_threshold": int(counts[i])})
        summary = {"ligand_heavy_atoms": len(indices), "receptor_heavy_atoms": len(self.atoms),
                   "minimum_distance_angstrom": float(distances.min()), "minimum_distance_vdw_ratio": float(minimum_ratios.min()),
                   "pairs_within_contact_radius": int(contact_count), "pairs_below_overlap_threshold": int(counts.sum()),
                   "ligand_atoms_below_overlap_threshold": int(np.count_nonzero(counts))}
        return summary, nearest


def iter_sdf_records(path):
    block = []
    with path.open("rb") as handle:
        for line in handle:
            block.append(line)
            if line.strip() == b"$$$$":
                yield b"".join(block)
                block = []
    if b"".join(block).strip():
        raise ValueError("SDF has an unterminated final record")


def identity(molecule):
    return canonical_isomeric_identity(molecule)


def smiles_identity(value):
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError("Input contains an invalid molecular identity")
    return identity(molecule)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[("overall", "all")].append(row)
        groups[("pdb", row["pdb"])].append(row)
        if row["linker_size"] != "":
            groups[("linker_size", str(row["linker_size"]))].append(row)
    result = []
    for (group, value), members in sorted(groups.items()):
        record = {"group": group, "value": value, "records": len(members),
                  "records_with_pairs_below_overlap_threshold": sum(row["pairs_below_overlap_threshold"] > 0 for row in members)}
        for field in MEASURE_FIELDS[2:]:
            values = np.asarray([row[field] for row in members], dtype=float)
            record[field] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}
        result.append(record)
    return result


def run(args):
    started = time.monotonic()
    if (not np.isfinite(args.overlap_scale) or args.overlap_scale <= 0
            or not np.isfinite(args.contact_radius) or args.contact_radius <= 0):
        raise ValueError("Distance criteria must be finite and positive")
    dataset, contexts_path, output = resolve(args.dataset), resolve(args.contexts), resolve(args.output)
    table_path, chemistry_path = dataset / "KLIFS_test_table.csv", dataset / "KLIFS_test_chemistry.jsonl"
    table_rows = read_csv(table_path)
    table = {row["uuid"]: row for row in table_rows}
    if len(table) != len(table_rows):
        raise ValueError("Test table has duplicate UUIDs")
    contexts, _ = load_context_manifest(contexts_path.read_bytes())
    inputs = {"contexts": describe(contexts_path), "test_table": describe(table_path), "native_chemistry": describe(chemistry_path)}
    survivors, generation_rows = [], []
    if args.mode == "run":
        survivor_path, generation = resolve(args.survivors), resolve(args.generation_dir)
        inputs.update({"survivors": describe(survivor_path), "generation_done": describe(generation / "done.json")})
        metrics_done_path = survivor_path.parent / "done.json"
        metrics_done = json.loads(metrics_done_path.read_text())
        inputs["metrics_done"] = describe(metrics_done_path)
        if metrics_done.get("status") != "complete":
            raise ValueError("Metrics must be complete before receptor auditing")
        if any(inputs["survivors"][key] != metrics_done["files"][survivor_path.name][key] for key in ("sha256", "bytes")):
            raise ValueError("Final-survivor table differs from its metrics completion record")
        done = json.loads((generation / "done.json").read_text())
        if done.get("status") != "complete":
            raise ValueError("Generation must be complete before receptor auditing")
        for name in ("candidates.csv", "candidates.sdf", "provenance.json"):
            observed = describe(generation / name)
            expected = done["outputs"][name]
            if any(observed[key] != expected[key] for key in ("sha256", "bytes")):
                raise ValueError("Generation completion hash mismatch: " + name)
            inputs[name] = observed
        generation_rows = read_csv(generation / "candidates.csv")
        if len(generation_rows) != done["candidate_count"]:
            raise ValueError("Generation candidate count differs from its completion record")
        generated = {row["candidate_id"]: row for row in generation_rows}
        if len(generated) != len(generation_rows):
            raise ValueError("Generation contains duplicate candidate IDs")
        survivors = read_csv(survivor_path)
        if len(survivors) != metrics_done["final_survivor_count"] or len(generation_rows) != metrics_done["candidate_count"]:
            raise ValueError("Metrics and generation population counts disagree")
        if len({row["candidate_id"] for row in survivors}) != len(survivors):
            raise ValueError("Survivor table contains duplicate candidate IDs")
        for row in survivors:
            if row.get("final_pass") != "1":
                raise ValueError("Every input row must be a final survivor")
            source = generated[row["candidate_id"]]
            for key in ("source_uuid", "fragmentation_id", "structure_id", "pdb", "context_index", "linker_size"):
                if row[key] != source[key]:
                    raise ValueError("Survivor/generation metadata mismatch: " + key)
            for key in ("fragmentation_id", "structure_id", "pdb"):
                if row[key] != table[row["source_uuid"]][key]:
                    raise ValueError("Survivor/native context metadata mismatch: " + key)
            if source["valid"] != "1":
                raise ValueError("A final survivor is marked invalid by generation")
        native_ids = sorted({row["source_uuid"] for row in survivors}, key=int)
    else:
        native_ids = []
        for pdb in args.pdb:
            matches = [row["uuid"] for row in table_rows if row["pdb"].lower() == pdb.lower()]
            if not matches:
                raise ValueError("Requested native PDB is absent from test data: " + pdb)
            native_ids.append(min(matches, key=int))
        if len(native_ids) != len(set(native_ids)):
            raise ValueError("Native pilot PDB arguments must be distinct")
    chemistry = {}
    wanted = set(native_ids)
    with chemistry_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            uuid = str(record["uuid"])
            if uuid in wanted:
                if uuid in chemistry:
                    raise ValueError("Duplicate requested UUID in native chemistry sidecar")
                chemistry[uuid] = record
    if set(chemistry) != wanted:
        raise ValueError("Native chemistry coverage is incomplete")
    receptors, receptor_sources, native_molecules = {}, {}, {}
    for uuid in native_ids:
        row, record = table[uuid], chemistry[uuid]
        context = contexts[row["structure_id"]]
        for key in ("fragmentation_id", "ligand_sha256"):
            if record[key] != row[key]:
                raise ValueError("Native chemistry/table mismatch: " + key)
        if context["ligand_sha256"] != row["ligand_sha256"] or context["pdb"] != row["pdb"]:
            raise ValueError("Native/context manifest mismatch")
        molecule = Chem.Mol(base64.b64decode(record["parent_molecule_binary_base64"], validate=True))
        if molecule is None:
            raise ValueError("Native molecule binary did not load")
        Chem.SanitizeMol(molecule)
        if identity(molecule) != smiles_identity(row["canonical_smiles"]) or identity(molecule) != smiles_identity(record["canonical_smiles"]):
            raise ValueError("Native identity differs from its table or chemistry sidecar")
        serialized = mol_to_sdf_bytes(molecule, title="native_" + uuid, native_identity=True)
        native_molecules[uuid] = parse_sdf_bytes(serialized, label="native_" + uuid)[0]
        structure = row["structure_id"]
        if structure not in receptors:
            protein = resolve(context["protein_path"])
            source = describe(protein)
            if source["sha256"] != row["protein_sha256"]:
                raise ValueError("Source receptor differs from the prepared native context")
            receptors[structure] = ReceptorIndex(receptor_atoms(protein))
            receptor_sources[structure] = source
        elif receptor_sources[structure]["sha256"] != row["protein_sha256"]:
            raise ValueError("Native fragmentations disagree on their source receptor")
    output.mkdir(parents=True, exist_ok=False)
    generated_results, native_results = [], []
    with (output / "nearest_receptor_atoms.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEAREST_FIELDS, extrasaction="ignore")
        writer.writeheader()

        def audit(row, molecule, population):
            meta = {field: row.get(field, "") for field in META_FIELDS}
            meta["population"] = population
            measure, nearest = receptors[row["structure_id"]].inspect(molecule, args.overlap_scale, args.contact_radius)
            for atom in nearest:
                writer.writerow(dict(meta, **atom))
            return dict(meta, **measure)

        for uuid in native_ids:
            row = dict(table[uuid], candidate_id="native_" + uuid, source_uuid=uuid, linker_size="")
            native_results.append(audit(row, native_molecules[uuid], "native_context"))
        if args.mode == "run":
            by_id = {row["candidate_id"]: row for row in survivors}
            observed = set()
            records = iter_sdf_records(generation / "candidates.sdf")
            for row in generation_rows:
                try:
                    payload = next(records)
                except StopIteration as exc:
                    raise ValueError("Generation SDF has fewer records than its table") from exc
                candidate_id = row["candidate_id"]
                if payload.splitlines()[0].decode().strip() != candidate_id:
                    raise ValueError("Generation SDF/table candidate order differs")
                if candidate_id not in by_id:
                    continue
                molecule = parse_sdf_bytes(payload, label=candidate_id)[0]
                if not molecule.HasProp("kinase_native_molecule_binary_base64"):
                    raise ValueError("Survivor lacks its authoritative molecule binary")
                if molecule_identifier(molecule, label=candidate_id) != candidate_id:
                    raise ValueError("Survivor SDF identifier mismatch")
                survivor = by_id[candidate_id]
                if identity(molecule) != smiles_identity(survivor["canonical_smiles"]):
                    raise ValueError("Survivor evaluation/SDF identity mismatch")
                generated_results.append(audit(survivor, molecule, "generated_final_survivor"))
                observed.add(candidate_id)
            if next(records, None) is not None or observed != set(by_id):
                raise ValueError("Generation SDF or final-survivor coverage mismatch")
    write_csv(output / "candidate_diagnostics.csv", META_FIELDS + MEASURE_FIELDS, generated_results)
    write_csv(output / "native_diagnostics.csv", META_FIELDS + MEASURE_FIELDS, native_results)
    method = {"radii": "RDKit periodic table van der Waals radii", "overlap_scale": args.overlap_scale,
              "overlap_rule": "distance < overlap_scale * (ligand_radius + receptor_radius)",
              "contact_radius_angstrom": args.contact_radius, "contact_rule": "distance <= contact_radius",
              "coordinates": "Source protein MOL2 and full-precision authoritative ligand molecule binary in the source frame",
              "algorithm": "KD-tree nearest distances; one tree per receptor radius for exact normalized minima; exact distances for radius-query neighbours",
              "population": "Each final-survivor record and one native parent per represented source_uuid",
              "stage": "Post-filter receptor–ligand distance diagnostic"}
    summary = {"schema_version": 1, "mode": args.mode, "method": method, "generated_records": len(generated_results),
               "native_contexts": len(native_results), "receptor_structures": len(receptors),
               "generated": summarize(generated_results), "native": summarize(native_results)}
    write_json(output / "summary.json", summary)
    sources = {name: describe(ROOT / name) for name in ("scripts/audit_receptor_contacts.py", "src/kinase_gap/structural_bundle.py",
               "src/kinase_gap/chemistry_io.py", "src/kinase_gap/fragmentation.py")}
    provenance = {"schema_version": 1, "inputs": inputs, "receptors": receptor_sources, "sources": sources,
                  "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                                  "numpy": np.__version__, "scipy": scipy.__version__},
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started}
    write_json(output / "provenance.json", provenance)
    outputs = {path.name: describe(path) for path in sorted(output.iterdir())}
    completion = {"schema_version": 1, "status": "complete", "mode": args.mode, "generated_records": len(generated_results),
                  "native_contexts": len(native_results), "outputs": outputs}
    write_json(output / "done.json", completion)
    print(json.dumps({key: value for key, value in completion.items() if key != "outputs"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "native-pilot"):
        command = modes.add_parser(mode)
        command.add_argument("--contexts", required=True)
        command.add_argument("--dataset", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--overlap-scale", type=float, default=0.75)
        command.add_argument("--contact-radius", type=float, default=6.0)
        if mode == "run":
            command.add_argument("--survivors", required=True)
            command.add_argument("--generation-dir", required=True)
        else:
            command.add_argument("--pdb", action="append", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
