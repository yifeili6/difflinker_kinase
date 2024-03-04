import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import argparse
import os, sys, pathlib, shutil
import warnings
import glob
from typing import *

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--merged_pdb_dir", type=str, default="data_docking/result_hydrogenated")
args = parser.parse_args()

def load_prot_lig(args: argparse.ArgumentParser, prot: str, lig: str):
    assert prot.endswith(".pdb") and lig.endswith(".sdf")
    protU = mda.Universe(prot)
  
    mol = Chem.SDMolSupplier(lig, removeHs=False, sanitize=True)[0]
    ligU = mda.Universe(mol)
    ligU.add_TopologyAttr("resname", ["LIG"])
    ligU.add_TopologyAttr("chainID", ["L"] * len(ligU.atoms))
  
    mergeU = mda.Merge(protU.atoms, ligU.atoms)
    mergeU.add_TopologyAttr("resid", np.concatenate([protU.residues.resids, protU.residues.resids[-1:] + 1]))
    mergeU.add_TopologyAttr("resname", np.concatenate([protU.residues.resnames, ligU.residues.resnames]))
    mergeU.add_TopologyAttr("chainID", np.concatenate([protU.atoms.chainIDs, ligU.atoms.chainIDs]))

    prot_segment = mergeU.add_Segment(segid='PRO')
    lig_segment = mergeU.add_Segment(segid='LIG')
    prot_atoms = mergeU.select_atoms(f'resid {mergeU.residues.resids[0]}:{mergeU.residues.resids[-2]}')
    prot_atoms.residues.segments = prot_segment
    lig_atoms = mergeU.select_atoms(f'resid {mergeU.residues.resids[-1]}')
    lig_atoms.residues.segments = lig_segment
    
    directory = args.merged_pdb_dir
    complex_name = os.path.basename(lig).splitext()[0] + ".pdb"
    path_and_name = os.path.join(directory, complex_name)
    
    mergeU.atoms.write(path_and_name)

if __name__ == "__main__":
    lig_list: List[str] = glob.glob(os.path.join(args.merged_pdb_dir, "*.sdf"))
    for one_lig in lig_list:
        one_prot = "_".join(os.path.basename(one_lig).split("_")[:3]) + "_protein.pdb" if "alt" in os.path.basename(one_lig) else "_".join(os.path.basename(one_lig).split("_")[:2]) + "_protein.pdb" #match the protein prefix!
        load_prot_lig(args, f"data_docking/complex/processed_klif_wl/proteins/{one_prot}", one_lig)
    # git pull && python -m merge_prot_lig --merged_pdb_dir [directory_to_save]
