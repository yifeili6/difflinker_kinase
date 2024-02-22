from posebusters import PoseBusters
from pathlib import Path
from rdkit import Chem
import argparse
from typing import List
import os
import pandas as pd
import glob
import curtsies.fmtfuncs as cf
import warnings
from rdkit import rdBase
import moses
import numpy as np
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
import pickle
from moses.metrics.utils import mapper, mol_passes_filters
from tqdm.auto import tqdm
import time
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
import networkx

rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
# parser.add_argument("--filenames", "-f", nargs="*", help="xyz file names")
parser.add_argument("--gen", "-g", type=str, default="data_docking/result_difflinker", help="generated ligand SDF directory")
parser.add_argument("--train", "-t", type=str, default="datasets/KLIF_train_table.csv", help="train dataset")
parser.add_argument("--valtest", "-vt", type=str, default="datasets/KLIF_ValTest_table.csv", help="ValTest dataset")
parser.add_argument("--size_prefix", "-f", nargs="*", help="size prefiexs e.g. s11, s21")
args = parser.parse_args()

warnings.simplefilter('ignore')

def get_posebuster_stats(size_prefixes: List[str]):
    return_good_mols = []
    return_good_files = []
    total_file_counter = 0
    current_file_counter = 0

    assert size_prefixes.__len__() == 1, "for paper's sake, let's use only one size evaluation!"
    
    for size_prefix in size_prefixes:
        print(f"Size {size_prefix.strip('s')} is chosen...")
        s = time.time()
        try:
            filenames = sorted(glob.glob(os.path.join("data_docking/result_difflinker", size_prefix, "*.sdf")))
            # filenames = list(filter(lambda inp: "5lqf" in inp, filenames)) ###WIP: Will delete soon
            # pred_files = [Chem.MolFromXYZFile(os.path.join("data_docking/result_difflinker", filename)) if filename.endswith("xyz") else Chem.SDMolSupplier(os.path.join("data_docking/result_difflinker", filename))[0] for filename in filenames] 
            pred_files_ = [(filename, Chem.SDMolSupplier(filename, sanitize=False, removeHs=True)[0]) for filename in tqdm(filenames)] 
            print(cf.red(f"None location: {np.where(np.array(pred_files_)[:, 1] == None)[0]}"))
            pred_files = [p[1] for p in pred_files_ if p[1] is not None] #List[MOL]
            pred_file_names = [p[0] for p in pred_files_ if p[1] is not None] #List[str]

            buster = PoseBusters(config="mol")
            df = buster.bust(pred_files, None, None, full_report=False)
            # print(df.columns)
            # print(df.values)
            Df = pd.DataFrame(data=df.values, columns=df.columns.tolist())
            print(cf.on_green(f"s{size_prefix} is pose busted"))
            Df["Total Pass"] = Df.all(axis=1)
            print("Result:\n", Df)
            Df.to_pickle(os.path.join("data_docking/result_difflinker", size_prefix, "posebuster.pickle"))

            good_mols = np.array(pred_files)[Df.values[:,-1].astype(bool)]
            good_mol_files = np.array(pred_file_names)[Df.values[:,-1].astype(bool)]
            return_good_mols.extend(good_mols.tolist())
            return_good_files.extend(good_mol_files.tolist())
            # df.drop(index=["molecule", "file"], inplace=True)
            # print(df)
            current_file_counter += len(return_good_mols)
            total_file_counter += len(pred_files_)

        except Exception as e:
            # raise RuntimeError from e #cannot continue if raising errors!
            print(cf.on_red(f"Something happend... skipping!!!\n {e}"))
            continue
        e = time.time()
        print("Size", size_prefix.strip("s"), " : ", e-s, "seconds taken!")

    print(cf.on_yellow(f"PoseBuster retained {current_file_counter/total_file_counter*100} % valid molecules"))        
    # print(return_good_smiles)
    return return_good_mols, return_good_files, total_file_counter

def get_moses_stats(gen=None, k=None, n_jobs=os.cpu_count()-1,
                    device='cuda', batch_size=512, pool=None,
                    test=None, test_scaffolds=None,
                    ptest=None, ptest_scaffolds=None,
                    train=None, files=None, file_counter_from_posebuster=None, size_prefix=None):
                        
    assert len(gen) == len(files), "gen and files must match in length!" ##files variable is not actively used here!
                        
    if isinstance(gen, list):
        if isinstance(gen[0], str):
            pass
        else:
            gen: List[str] = [Chem.MolToSmiles(g) for g in gen if g is not None]
    else:
        gen = glob.glob(gen + "/*.sdf")
        # print(gen)
        gen: List[str] = [Chem.SDMolSupplier(g)[0] for g in gen]
        gen: List[str] = [Chem.MolToSmiles(g) for g in gen if g is not None]
        
    train = pd.read_csv(train).molecule.drop_duplicates().tolist()
    test = pd.read_csv(test).molecule.drop_duplicates().tolist()
    test_scaffolds = pd.read_csv(test_scaffolds).molecule.drop_duplicates().tolist()

    passes = mapper(n_jobs)(mol_passes_filters, gen)
    gen = np.array(gen)[np.array(passes).astype(bool)]
    files = np.array(files)[np.array(passes).astype(bool)]

    metrics = moses.get_all_metrics(gen=gen, k=k, n_jobs=n_jobs,
                    device=device, batch_size=batch_size, pool=pool,
                    test=test, test_scaffolds=test_scaffolds,
                    ptest=ptest, ptest_scaffolds=ptest_scaffolds,
                    train=train)

    with open(os.path.join("data_docking/result_difflinker", size_prefix, "moses.pickle"), "wb") as f:
        pickle.dump(metrics, f)
                        
    assert metrics["Filters"] == 1.0, "Filters must be 1.0 since we already apply mole_passes_filter!"
    print(cf.on_blue("MOSES metrics"))    
    print(metrics)
    print(cf.on_yellow(f"MOSES retained {len(gen)/file_counter_from_posebuster*100} % valid molecules"))        
                        
    return gen.tolist(), files.tolist()

def get_lipinski(gen: List[str], files: List[str], file_counter_from_posebuster: int, size_prefix: str):
    """
    Source:
    https://gist.github.com/strets123/fdc4db6d450b66345f46
    """
    class SmilesError(Exception): pass
    
    def log_partition_coefficient(smiles):
        '''
        Returns the octanol-water partition coefficient given a molecule SMILES 
        string
        '''
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            raise SmilesError('%s returns a None molecule' % smiles)
            
        return Crippen.MolLogP(mol)
        
    def lipinski_trial(smiles):
        '''
        Returns which of Lipinski's rules a molecule has failed, or an empty list
        
        Lipinski's rules are:
        Hydrogen bond donors <= 5
        Hydrogen bond acceptors <= 10
        Molecular weight < 500 daltons
        logP < 5
        '''
        passed = []
        failed = []
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise Exception('%s is not a valid SMILES string' % smiles)
        
        num_hdonors = Lipinski.NumHDonors(mol)
        num_hacceptors = Lipinski.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Crippen.MolLogP(mol)
        
        failed = []
        
        if num_hdonors > 5:
            failed.append('Over 5 H-bond donors, found %s' % num_hdonors)
        else:
            passed.append('Found %s H-bond donors' % num_hdonors)
            
        if num_hacceptors > 10:
            failed.append('Over 10 H-bond acceptors, found %s' \
            % num_hacceptors)
        else:
            passed.append('Found %s H-bond acceptors' % num_hacceptors)
            
        if mol_weight >= 500:
            failed.append('Molecular weight over 500, calculated %s'\
            % mol_weight)
        else:
            passed.append('Molecular weight: %s' % mol_weight)
            
        if mol_logp >= 5:
            failed.append('Log partition coefficient over 5, calculated %s' \
            % mol_logp)
        else:
            passed.append('Log partition coefficient: %s' % mol_logp)
        
        return passed, failed
        
    def lipinski_pass(smiles):
        '''
        Wraps around lipinski trial, but returns a simple pass/fail True/False
        '''
        passed, failed = lipinski_trial(smiles)
        if failed:
            return False
        else:
            return True
            
    if isinstance(gen, list):
        if isinstance(gen[0], str):
            pass
        else:
            gen: List[str] = [Chem.MolToSmiles(g) for g in gen if g is not None]
    else:
        gen = glob.glob(gen + "/*.sdf")
        # print(gen)
        gen: List[str] = [Chem.SDMolSupplier(g)[0] for g in gen]
        gen: List[str] = [Chem.MolToSmiles(g) for g in gen if g is not None]

    assert len(gen) == len(files), "gen and files must match in length!"

    lipinski_results = [lipinski_pass(smiles) for smiles in gen]
    with open(os.path.join("data_docking/result_difflinker", size_prefix, "lipinski.pickle"), "wb") as f:
        pickle.dump(lipinski_results, f)
        
    print(cf.yellow("Lipinski Rule of 5"))
    print(np.mean(lipinski_results))

    return_good_smiles = np.array(gen)[np.array(lipinski_results).astype(bool)]
    return_good_files = np.array(files)[np.array(lipinski_results).astype(bool)]

    print(cf.on_yellow(f"Lipinski's Rule of 5 retained {len(return_good_smiles)/file_counter_from_posebuster*100} % valid molecules"))        
    return return_good_smiles.tolist(), return_good_files.tolist()

def bonds_and_rings(gen: List[str]):
    def rotatable_bonds(gen):
        rot_bonds: List[int] = [CalcNumRotatableBonds(Chem.MolFromSmiles(g), True) for g in gen]
        return rot_bonds

    def symm_SSSR(gen):
        rings: List[List[tuple]] = [frozenset(Chem.GetSymmSSSR(Chem.MolFromSmiles(g))) for g in gen]
        return rings

    def fused_rings_SSSR(gen: List[str], Rings_list: List[List[tuple]]):
        assert len(gen) == len(Rings_list), "SMILES and Rings_list from SMILES must match"
        fused_list = []
        
        for Rings in Rings_list:
            G = networkx.Graph()
    
            L = len(Rings)
            for i, j in ((i, j) for i in range(L) for j in range(i + 1, L)):
                if len(Rings[i] & Rings[j]) >= 2:
                    G.add_edge(i, j)
            fused_rings = [
                 frozenset(j for i in ring_ids for j in Rings[i])
                for ring_ids in networkx.connected_components(G)
                ]
            fused_list.append(fused_rings)
        return fused_list
        
    def hetero_rings_SSSR(gen: List[str], Rings_list: List[List[tuple]]):
        assert len(gen) == len(Rings_list), "SMILES and Rings_list from SMILES must match"
        
        num_heteros = []
        for i, g in enumerate(gen):
            mol = Chem.MolFromSmiles(g)
            heteros_tracking = []
            Rings = Rings_list[i]
            for R in Rings:
                has_hetero = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in R)
                heteros_tracking.append(has_hetero)
            num_heteros.append(sum(heteros_tracking))
        return num_heteros
            
    def aromatic_rings_SSSR(gen: List[str], Rings_list: List[List[tuple]]):
        assert len(gen) == len(Rings_list), "SMILES and Rings_list from SMILES must match"
        
        num_aromatics = []
        for i, g in enumerate(gen):
            mol = Chem.MolFromSmiles(g)
            aromatics_tracking = []
            Rings = Rings_list[i]
            for R in Rings:
                is_arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in R)
                aromatics_tracking.append(is_arom)
            num_aromatics.append(sum(aromatics_tracking))
        return num_aromatics

    rot_bonds = rotatable_bonds(gen)
    rings = symm_SSSR(gen)
    fused_rings = fused_rings_SSSR(gen, rings)
    num_hetero_rings = hetero_rings_SSSR(gen, rings)
    num_aromatic_rings = aromatic_rings_SSSR(gen, rings)
    num_rings = list(map(lambda inp: len(inp), rings ))
    num_fused_rings = list(map(lambda inp: len(inp), fused_rings ))
    
    return rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings
    
    

if __name__ == "__main__":
    ###Current as of Feb 22, 2024
    ###3D
    # gen, files, file_counter = get_posebuster_stats(args.size_prefix) # filtration 1
    # if len(gen) !=0:
    #     pass
    # else:
    #     gen = args.gen
    # ###Drugness
    # gen, files = get_lipinski(gen, files, file_counter, args.size_prefix[0]) #filtration 2
    # ###2D
    # gen, files = get_moses_stats(gen=gen, files=files, train=args.train, test=args.valtest, test_scaffolds=args.valtest, file_counter_from_posebuster=file_counter, size_prefix=args.size_prefix[0]) # final filtration
    # print(files)

    gen = ["c1ccccc1", "c1cnccc1", "C1CCCCC1", "C1CNCCC1", "CCCCCC", "CCNCCC", "c1cccc2CC=CN2c1"]
    rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings = bonds_and_rings(gen)
    for prop in [rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings]:
        print(prop)
