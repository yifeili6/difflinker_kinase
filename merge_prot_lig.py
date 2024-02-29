import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import argparse
import os, sys, pathlib, shutil

parser = argparse.ArgumentParser()
parser.add_argument("--merged_pdb_dir", type=str, default=None)
args = parser.parse_args()

def load_prot_lig(args: argparse.ArgumentParser, prot: str, lig: str):
    assert prot.endswith(".pdb") and lig.endswith(".sdf")
    protU = mda.Universe(prot)
  
    mol = Chem.SDMolSupplier(lig, removeHs=False, sanitize=True)[0]
    ligU = mda.Universe(mol)
    ligU.add_TopologyAttr("resname", ["LIG"])
    ligU.add_TopologyAttr("chainID", ["L"] * len(ligU.atoms))
  
    mergeU = mda.Merge(protU.atoms, ligU.atoms)
    mergeU.add_TopologyAttr("resname", np.concatenate([protU.residues.resnames, ligU.residues.resnames]))
    mergeU.add_TopologyAttr("chainID", np.concatenate([protU.atoms.chainIDs, ligU.atoms.chainIDs]))

    directory = args.merged_pdb_dir
    path_and_name = os.path.join(directory, "prot_lig.pdb")
    
    mergeU.atoms.write(path_and_name)

if __name__ == "__main__":
    args.merged_pdb_dir = "."
    load_prot_lig(args, "data_docking/complex/processed_klif_wl/proteins/1yol_chainA_protein.pdb", "data_docking/result_difflinker/s8/1yol_chainA_2_s8_12_KLIF_ValTest_frag.sdf")
    # git pull && python -m merge_prot_lig --merged_pdb_dir [directory_to_save]
