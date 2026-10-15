"""Independently check serialized coordinates, cut topology and full-ligand chemistry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from rdkit import Chem

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atom_key(mol, index):
    p = mol.GetConformer().GetAtomPosition(index)
    return mol.GetAtomWithIdx(index).GetAtomicNum(), round(p.x, 4), round(p.y, 4), round(p.z, 4)

def source_stereo(mol):
    if not mol.HasProp("kinase_native_stereo"):
        return mol
    record = json.loads(mol.GetProp("kinase_native_stereo"))
    assert len(record["atom_chiral_tags"]) == mol.GetNumAtoms()
    result = Chem.Mol(mol)
    for atom, tag in zip(result.GetAtoms(), record["atom_chiral_tags"]):
        atom.SetChiralTag(Chem.ChiralType(tag))
        if "atom_neighbor_order" in record:
            original = record["atom_neighbor_order"][atom.GetIdx()]
            observed = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
            assert set(original) == set(observed), "source stereo neighbor identity"
            order = [original.index(value) for value in observed]
            inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i+1, len(order)))
            if inversions % 2 and tag in (int(Chem.ChiralType.CHI_TETRAHEDRAL_CW), int(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)):
                atom.InvertChirality()
    for item in record["bonds"]:
        bond = result.GetBondBetweenAtoms(item["begin"], item["end"])
        if item["stereo_atoms"]:
            bond.SetStereoAtoms(*item["stereo_atoms"])
        bond.SetStereo(Chem.BondStereo(item["stereo"]))
        if "bond_dir" in item:
            bond.SetBondDir(Chem.BondDir(item["bond_dir"]))
    Chem.AssignStereochemistry(result, cleanIt=False, force=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/klifs_v1"))
    parser.add_argument("--output", type=Path, default=Path("provenance/prepared_graph_validation.json"))
    parser.add_argument("--splits", choices=("train", "val", "test"), nargs="+", default=("train", "val", "test"))
    args = parser.parse_args()
    result = {"splits": {}, "inputs": {}}
    failures = []
    for split in args.splits:
        stem = args.data/f"KLIFS_{split}"
        paths = {kind: Path(str(stem)+f"_{kind}.sdf") for kind in ("mol", "frag", "link")}
        table = Path(str(stem)+"_table.csv")
        rows = list(csv.DictReader(table.open()))
        suppliers = {kind: Chem.SDMolSupplier(str(path), sanitize=(kind == "mol"), removeHs=False)
                     for kind, path in paths.items()}
        assert all(len(supplier) == len(rows) for supplier in suppliers.values())
        passed = 0
        long_cut_lengths = []
        for index, row in enumerate(rows):
            try:
                parent, fragments, linker = (suppliers[k][index] for k in ("mol", "frag", "link"))
                assert all(mol is not None for mol in (parent, fragments, linker)), "unreadable SDF"
                stereo_smiles = Chem.MolToSmiles(source_stereo(parent), canonical=True, isomericSmiles=True)
                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(stereo_smiles), canonical=True, isomericSmiles=True)
                expected = Chem.MolToSmiles(Chem.MolFromSmiles(row["canonical_smiles"]), canonical=True, isomericSmiles=True)
                assert canonical == expected, "full ligand chemical identity"
                keys = {atom_key(parent, i): i for i in range(parent.GetNumAtoms())}
                assert len(keys) == parent.GetNumAtoms(), "ambiguous identical atom coordinates"
                frag_map = [keys[atom_key(fragments, i)] for i in range(fragments.GetNumAtoms())]
                link_map = [keys[atom_key(linker, i)] for i in range(linker.GetNumAtoms())]
                assert len(set(frag_map + link_map)) == len(frag_map + link_map) == parent.GetNumAtoms(), "non-partition coordinates"
                parts = Chem.GetMolFrags(fragments)
                assert len(parts) == 2, "retained fragment components"
                owners = {frag_map[i]: part for part, atoms in enumerate(parts) for i in atoms}
                owners.update({i: 2 for i in link_map})
                cuts = [b for b in parent.GetBonds() if owners[b.GetBeginAtomIdx()] != owners[b.GetEndAtomIdx()]]
                pairs = sorted(tuple(sorted((owners[b.GetBeginAtomIdx()], owners[b.GetEndAtomIdx()]))) for b in cuts)
                assert pairs == [(0, 2), (1, 2)], "fragment-linker-fragment topology"
                ends = {i for b in cuts for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) if owners[i] != 2}
                anchors = {frag_map[int(row["anchor_1"])], frag_map[int(row["anchor_2"])]}
                assert len(anchors) == 2 and anchors == ends, "anchor cut endpoints"
                for b in cuts:
                    conf = parent.GetConformer()
                    length = (conf.GetAtomPosition(b.GetBeginAtomIdx()) - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()
                    if length > 1.95:
                        long_cut_lengths.append({"uuid": row["uuid"], "bond_atoms": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()], "angstrom": length})
                passed += 1
            except Exception as exc:
                failures.append({"split": split, "uuid": row["uuid"], "error": str(exc)})
        result["splits"][split] = {"rows": len(rows), "passed": passed, "cut_bonds_longer_than_1_95_A": long_cut_lengths}
        for path in list(paths.values()) + [table]:
            result["inputs"][str(path)] = sha256(path)
    result["failures"] = failures
    result["passed"] = not failures
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"passed": result["passed"], "rows": sum(s["rows"] for s in result["splits"].values()), "failures": len(failures)}))
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
