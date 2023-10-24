import sys
sys.path.append('../../')

import argparse
import os
import subprocess

from rdkit import Chem
from tqdm import tqdm
from src.utils import disable_rdkit_logging

import functools
import multiprocessing
import ray

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=True):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol

def get_relevant_ligands(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ligands = []
    for frag in frags:
        if 10 < frag.GetNumAtoms() <= 40:
            ligands.append(frag)
    return ligands


def run(input_dir, proteins_dir, ligands_dir):
    os.makedirs(proteins_dir, exist_ok=True)
    os.makedirs(ligands_dir, exist_ok=True)

    fnames = [fname for fname in tqdm(os.listdir(input_dir)) if fname.endswith(".bio1") or fname.endswith(".pdb")]
    return fnames

@ray.remote
def process_one_file(input_dir, proteins_dir, ligands_dir, fname):
    # assert fname.endswith('.bio1')

    pdb_code = fname.split('.')[0]
    input_path = os.path.join(input_dir, fname)
    temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')
    temp_path_1 = os.path.join(proteins_dir, f'{pdb_code}_temp_1.pdb')
    temp_path_2 = os.path.join(proteins_dir, f'{pdb_code}_temp_2.pdb')
    temp_path_3 = os.path.join(proteins_dir, f'{pdb_code}_temp_3.pdb')

    out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
    out_ligands_path = os.path.join(ligands_dir, f'{pdb_code}_ligands.pdb')

    subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
    subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
    subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

    subprocess.run(f'pdb_selhetatm {temp_path_1} > {temp_path_2}', shell=True)
    subprocess.run(f'pdb_delelem -H {temp_path_2} > {temp_path_3}', shell=True)
    subprocess.run(f'pdb_delelem -X {temp_path_3} > {out_ligands_path}', shell=True)

    os.remove(temp_path_0)
    os.remove(temp_path_1)
    os.remove(temp_path_2)
    os.remove(temp_path_3)

    try:
        mol = Chem.MolFromPDBFile(out_ligands_path, sanitize=False)
        os.remove(out_ligands_path)
    except Exception as e:
        print(f'Problem reading ligands PDB={pdb_code}: {e}')
        os.remove(out_ligands_path)
        # continue

    try:
        ligands = get_relevant_ligands(mol)
    except Exception as e:
        print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
        # continue

    try:
        for i, lig in enumerate(ligands):
            out_ligand_path = os.path.join(ligands_dir, f'{pdb_code}_{i}.mol')
            Chem.MolToMolFile(lig, out_ligand_path)
    except Exception as e:
        print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
        # continue

def process_one_file_noray(input_dir, proteins_dir, ligands_dir, fname):
    # assert fname.endswith('.bio1')

    pdb_code = fname.split('.')[0]
    input_path = os.path.join(input_dir, fname)
    temp_path_0 = os.path.join(proteins_dir, f'{pdb_code}_temp_0.pdb')
    temp_path_1 = os.path.join(proteins_dir, f'{pdb_code}_temp_1.pdb')
    temp_path_2 = os.path.join(proteins_dir, f'{pdb_code}_temp_2.pdb')
    temp_path_3 = os.path.join(proteins_dir, f'{pdb_code}_temp_3.pdb')

    out_protein_path = os.path.join(proteins_dir, f'{pdb_code}_protein.pdb')
    out_ligands_path = os.path.join(ligands_dir, f'{pdb_code}_ligands.pdb')

    subprocess.run(f'pdb_selmodel -1 {input_path} > {temp_path_0}', shell=True)
    subprocess.run(f'pdb_delelem -H {temp_path_0} > {temp_path_1}', shell=True)
    subprocess.run(f'pdb_delhetatm {temp_path_1} > {out_protein_path}', shell=True)

    subprocess.run(f'pdb_selhetatm {temp_path_1} > {temp_path_2}', shell=True)
    subprocess.run(f'pdb_delelem -H {temp_path_2} > {temp_path_3}', shell=True)
    subprocess.run(f'pdb_delelem -X {temp_path_3} > {out_ligands_path}', shell=True)

    os.remove(temp_path_0)
    os.remove(temp_path_1)
    os.remove(temp_path_2)
    os.remove(temp_path_3)

    try:
        # mol = Chem.MolFromPDBFile(out_ligands_path, sanitize=False)
        mol = read_molecule(out_ligands_path)
        os.remove(out_ligands_path)
    except Exception as e:
        print(f'Problem reading ligands PDB={pdb_code}: {e}')
        os.remove(out_ligands_path)
        # continue

    try:
        ligands = get_relevant_ligands(mol)
    except Exception as e:
        print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
        # continue

    try:
        for i, lig in enumerate(ligands):
            out_ligand_path = os.path.join(ligands_dir, f'{pdb_code}_{i}.mol')
            Chem.MolToMolFile(lig, out_ligand_path)
    except Exception as e:
        print(f'Problem getting relevant ligands PDB={pdb_code}: {e}')
        # continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--proteins-dir', action='store', type=str, required=True)
    parser.add_argument('--ligands-dir', action='store', type=str, required=True)
    args = parser.parse_args()

    disable_rdkit_logging()
    fnames = run(input_dir=args.in_dir, proteins_dir=args.proteins_dir, ligands_dir=args.ligands_dir)

    # process_one_file_ = lambda fname: functools.partial(process_one_file, input_dir=args.in_dir, proteins_dir=args.proteins_dir, ligands_dir=args.ligands_dir)(fname=fname)
    
    # with multiprocessing.Pool(os.cpu_count()-1) as mpool:
    #     result = mpool.map(process_one_file_, fnames) #.get()
    input_dir, proteins_dir, ligands_dir = ray.put(args.in_dir), ray.put(args.proteins_dir), ray.put(args.ligands_dir)
    results = [process_one_file.remote(input_dir, proteins_dir, ligands_dir, fname) for fname in fnames]
    results = ray.get(results)
