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
from dataclasses import dataclass
# from merge_prot_lig import load_prot_lig #load_prot_lig(args: argparse.ArgumentParser, prot: str, lig: str)

rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
# parser.add_argument("--filenames", "-f", nargs="*", help="xyz file names")
parser.add_argument("--gen", "-g", type=str, default="data_docking/result_difflinker", help="generated ligand SDF directory")
parser.add_argument("--train", "-t", type=str, default="datasets/KLIF_train_table.csv", help="train dataset")
parser.add_argument("--valtest", "-vt", type=str, default="datasets/KLIF_ValTest_table.csv", help="ValTest dataset")
parser.add_argument("--size_prefix", "-pf", type=str, help="1 size prefix e.g. s11")
parser.add_argument("--size_prefixes", "-f", nargs="*", help="size prefixes e.g. s11, s21")
parser.add_argument("--run_analysis", "-ra", action="store_true", help="generting pickles!")
parser.add_argument("--turn_off_run_test", "-ra", action="store_false", help="for ValTest_mol.sdf!")
parser.add_argument("--merged_pdb_dir", "-mpdb", type=str, help="merge 1 PDB and 1 SDF")

args = parser.parse_args()

warnings.simplefilter('ignore')

def get_kinase_indices(kinase_names: List[str]) -> List[List[str]]:
    """Return corresponding indices of kinase indices"""
  
    df0=pd.read_csv("datasets/KLIF_train_table.csv", header=0)
    df1=pd.read_csv("datasets/KLIF_val_table.csv", header=0)
    df2=pd.read_csv("datasets/KLIF_test_table.csv", header=0)
    
    df=pd.concat([df1, df2], axis=0)
    df.reset_index(drop=True, inplace=True)
    
    indices = df.index[df.molecule_name.apply(lambda inp: inp[:4] in kinase_names)].tolist()
    info = df.loc[df.molecule_name.apply(lambda inp: inp[:4] in kinase_names), ["molecule_name", "anchor_1", "anchor_2", "linker_size"]]

    return indices, info

# def get_complete_stats(size_prefixes: List[str]):
#     for size_prefix in size_prefixes:
#         filenames = sorted(glob.glob(os.path.join("data_docking/result_difflinker", size_prefix, "*.pickle")))
#         for f in filenames:
#             data = pd.read_pickle(f):
#             if "posebuster" in os.path.basename(f):
#                 ...
#             if "lipinski" in os.path.basename(f):
#                 ...            
#             if "moses" in os.path.basename(f):
#                 ...
#             if "rings" in os.path.basename(f):
#                 ...

@dataclass
class Analyse_generation(object):
    args: argparse.ArgumentParser
    
    @staticmethod
    def get_posebuster_stats_for_test():
        return_good_mols = []
        return_good_files = []
        total_file_counter = 0
        current_file_counter = 0
    
        filenames = os.path.join("data_docking/datasets", "KLIF_test_mol.sdf")
        pred_files_ = [m for m in Chem.SDMolSupplier(filename, sanitize=False, removeHs=True)] 
        print(cf.red(f"None location: {np.where(np.array(pred_files_) == None)[0]}"))
        pred_files = [p for p in pred_files_ if p is not None] #List[MOL]
    
        buster = PoseBusters(config="mol")
        df = buster.bust(pred_files, None, None, full_report=False)
        # print(df.columns)
        # print(df.values)
        Df = pd.DataFrame(data=df.values, columns=df.columns.tolist())
        print(cf.on_green(f"s{size_prefix} is pose busted"))
        Df["Total Pass"] = Df.all(axis=1)
        print("Result:\n", Df)
        Df.to_pickle(os.path.join("data_docking/datasets", "posebuster.pickle"))
    
        good_mols = np.array(pred_files)[Df.values[:,-1].astype(bool)]
        return_good_mols.extend(good_mols.tolist())
        current_file_counter += len(return_good_mols)
        total_file_counter += len(pred_files_)
    
        print(cf.on_yellow(f"PoseBuster retained {current_file_counter/total_file_counter*100} % valid molecules"))        
        # print(return_good_smiles)
        return return_good_mols, total_file_counter
        
    @staticmethod
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
        
    @staticmethod
    def get_lipinski_for_test(gen: List[str], file_counter_from_posebuster: int):
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
    
        lipinski_results = [lipinski_pass(smiles) for smiles in gen]
        with open(os.path.join("data_docking/datasets", "lipinski.pickle"), "wb") as f:
            pickle.dump(lipinski_results, f)
            
        print(cf.yellow("Lipinski Rule of 5"))
        print(np.mean(lipinski_results))
    
        return_good_smiles = np.array(gen)[np.array(lipinski_results).astype(bool)]
    
        print(cf.on_yellow(f"Lipinski's Rule of 5 retained {len(return_good_smiles)/file_counter_from_posebuster*100} % valid molecules"))        
        return return_good_smiles.tolist()
        
    @staticmethod
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
        
    @staticmethod
    def get_moses_stats_for_test(gen=None, k=None, n_jobs=os.cpu_count()-1,
                        device='cuda', batch_size=512, pool=None,
                        test=None, test_scaffolds=None,
                        ptest=None, ptest_scaffolds=None,
                        train=None, files=None, file_counter_from_posebuster=None, size_prefix=None):
                                                    
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
    
        with open(os.path.join("data_docking/datasets", "moses.pickle"), "wb") as f:
            pickle.dump(metrics, f)
                            
        assert metrics["Filters"] == 1.0, "Filters must be 1.0 since we already apply mole_passes_filter!"
        print(cf.on_blue("MOSES metrics"))    
        print(metrics)
        print(cf.on_yellow(f"MOSES retained {len(gen)/file_counter_from_posebuster*100} % valid molecules"))        
                            
        return gen.tolist()
                            
    @staticmethod
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
                            
    @staticmethod
    def bonds_and_rings_for_test(gen: List[str]):
        assert len(gen) == len(files), "gen and files must have the same length"
        
        def rotatable_bonds(gen):
            rot_bonds: List[int] = [CalcNumRotatableBonds(Chem.MolFromSmiles(g), True) for g in gen]
            return rot_bonds
    
        def symm_SSSR(gen):
            rings: List[List[tuple]] = [list(frozenset(Chem.GetSymmSSSR(Chem.MolFromSmiles(g)))) for g in gen]
            return rings
    
        def fused_rings_SSSR(gen: List[str], Rings_list: List[List[tuple]]):
            assert len(gen) == len(Rings_list), "SMILES and Rings_list from SMILES must match"
            fused_list = []
            
            for Rings in Rings_list:
                G = networkx.Graph()
        
                L = len(Rings)
                # print(len(Rings))
                for i, j in ((i, j) for i in range(L) for j in range(i + 1, L)):
                    if len({_ for _ in Rings[i]} & {_ for _ in Rings[j]}) >= 2:
                        G.add_edge(i, j)
                # print(G.number_of_nodes())
                fused_rings = [
                     list(frozenset(j for i in ring_ids for j in Rings[i]))
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
            
        print(cf.on_green("Rings analysis starting..."))
    
        rot_bonds = rotatable_bonds(gen)
        rings = symm_SSSR(gen)
        fused_rings = fused_rings_SSSR(gen, rings)
        num_hetero_rings = hetero_rings_SSSR(gen, rings)
        num_aromatic_rings = aromatic_rings_SSSR(gen, rings)
        num_rings = list(map(lambda inp: len(inp), rings ))
        num_fused_rings = list(map(lambda inp: len(inp), fused_rings ))
    
        rings_results = np.stack([rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings], axis=0)
        with open(os.path.join("data_docking/datasets", "rings.pickle"), "wb") as f:
            pickle.dump(rings_results, f)
    
        gen = np.array(gen).reshape(-1,)
        with open(os.path.join("data_docking/datasets", "gen_and_files.pickle"), "wb") as f:
            pickle.dump(gen, f)
        return rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings
        
    @staticmethod
    def bonds_and_rings(gen: List[str], files: List[str], size_prefix: str):
        assert len(gen) == len(files), "gen and files must have the same length"
        
        def rotatable_bonds(gen):
            rot_bonds: List[int] = [CalcNumRotatableBonds(Chem.MolFromSmiles(g), True) for g in gen]
            return rot_bonds
    
        def symm_SSSR(gen):
            rings: List[List[tuple]] = [list(frozenset(Chem.GetSymmSSSR(Chem.MolFromSmiles(g)))) for g in gen]
            return rings
    
        def fused_rings_SSSR(gen: List[str], Rings_list: List[List[tuple]]):
            assert len(gen) == len(Rings_list), "SMILES and Rings_list from SMILES must match"
            fused_list = []
            
            for Rings in Rings_list:
                G = networkx.Graph()
        
                L = len(Rings)
                # print(len(Rings))
                for i, j in ((i, j) for i in range(L) for j in range(i + 1, L)):
                    if len({_ for _ in Rings[i]} & {_ for _ in Rings[j]}) >= 2:
                        G.add_edge(i, j)
                # print(G.number_of_nodes())
                fused_rings = [
                     list(frozenset(j for i in ring_ids for j in Rings[i]))
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
            
        print(cf.on_green("Rings analysis starting..."))
    
        rot_bonds = rotatable_bonds(gen)
        rings = symm_SSSR(gen)
        fused_rings = fused_rings_SSSR(gen, rings)
        num_hetero_rings = hetero_rings_SSSR(gen, rings)
        num_aromatic_rings = aromatic_rings_SSSR(gen, rings)
        num_rings = list(map(lambda inp: len(inp), rings ))
        num_fused_rings = list(map(lambda inp: len(inp), fused_rings ))
    
        rings_results = np.stack([rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings], axis=0)
        with open(os.path.join("data_docking/result_difflinker", size_prefix, "rings.pickle"), "wb") as f:
            pickle.dump(rings_results, f)
    
        gen = np.array(gen).reshape(-1,)
        files = np.array(files).reshape(-1,)
        gen_and_files = np.stack([gen, files], axis=1)
        with open(os.path.join("data_docking/result_difflinker", size_prefix, "gen_and_files.pickle"), "wb") as f:
            pickle.dump(gen_and_files, f)
        return rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings
        
    @staticmethod
    def collate_fn_for_test():
        # lipinski.pickle, posebuster.pickle, moses.pickle, rings.pickle
        all_passes = []
        all_passes.append(all([os.path.isfile(os.path.join(f"data_docking/datasets", one_file)) for one_file in ("lipinski.pickle", "posebuster.pickle", "moses.pickle", "rings.pickle")]))
        assert all(all_passes), "every file must exist!"
        
        DF_posebuster = []
        DF_lipinski = []
        DF_moses = []
        DF_rings_dist = []
        DF_rings = []
    
        one_file = "posebuster.pickle" #summarize
        df = pd.read_pickle(os.path.join(f"data_docking/datasets", one_file))
        DF_posebuster.append(pd.DataFrame([df["Total Pass"].sum() / len(df)], columns=["posebuster"]))
        
        one_file = "lipinski.pickle" #summarize
        df = pd.read_pickle(os.path.join(f"data_docking/datasets", one_file))
        DF_lipinski.append(pd.DataFrame([sum(df) / len(df)], columns=["lipinski"])) #--> List[float]
        
        one_file = "moses.pickle" #use original
        df = pd.read_pickle(os.path.join(f"data_docking/datasets", one_file))
        df = pd.DataFrame.from_records([df])
        DF_moses.append(df)
        
        one_file = "rings.pickle" #get distribution & Summary
        df = pd.read_pickle(os.path.join(f"data_docking/datasets", one_file))
        df = pd.DataFrame(df.T, columns=["rot_bonds", "num_rings", "num_fused_rings", "num_hetero_rings", "num_aromatic_rings"])
        DF_rings_dist.append(df)
    
        DF_rings_dist = pd.concat(DF_rings_dist, axis=0, ignore_index=True) #(nmols, 5) ;; for distribution!
        DF_lipinski, DF_posebuster, DF_moses = list(map(lambda inp: pd.concat(inp, axis=0, ignore_index=True), [DF_lipinski, DF_posebuster, DF_moses] ))
        DF = pd.concat( [DF_lipinski, DF_posebuster, DF_moses], axis=1 ) #num_size, columns
        
        return DF, DF_rings_dist
        
    @staticmethod
    def collate_fn():
        # lipinski.pickle, posebuster.pickle, moses.pickle, rings.pickle
        all_passes = []
        for snum in range(8, 14, 1):
            all_passes.append(all([os.path.isfile(os.path.join(f"data_docking/result_difflinker/s{snum}", one_file)) for one_file in ("lipinski.pickle", "posebuster.pickle", "moses.pickle", "rings.pickle")]))
        assert all(all_passes), "every file must exist!"
        
        DF_posebuster = []
        DF_lipinski = []
        DF_moses = []
        DF_rings_dist = []
        DF_rings = []
    
        for snum in range(8, 14, 1):
            one_file = "posebuster.pickle" #summarize
            df = pd.read_pickle(os.path.join(f"data_docking/result_difflinker/s{snum}", one_file))
            DF_posebuster.append(pd.DataFrame([df["Total Pass"].sum() / len(df)], columns=["posebuster"]))
            
            one_file = "lipinski.pickle" #summarize
            df = pd.read_pickle(os.path.join(f"data_docking/result_difflinker/s{snum}", one_file))
            DF_lipinski.append(pd.DataFrame([sum(df) / len(df)], columns=["lipinski"])) #--> List[float]
            
            one_file = "moses.pickle" #use original
            df = pd.read_pickle(os.path.join(f"data_docking/result_difflinker/s{snum}", one_file))
            df = pd.DataFrame.from_records([df])
            DF_moses.append(df)
            
            one_file = "rings.pickle" #get distribution & Summary
            df = pd.read_pickle(os.path.join(f"data_docking/result_difflinker/s{snum}", one_file))
            df = pd.DataFrame(df.T, columns=["rot_bonds", "num_rings", "num_fused_rings", "num_hetero_rings", "num_aromatic_rings"])
            # DF_rings.append(pd.DataFrame(df.mean(axis=0).values.reshape(1, -1), columns=df.columns))
            df["size"] = [snum] * len(df) #so we can do multi-index plot using SEABORN
            DF_rings_dist.append(df)
    
        DF_rings_dist = pd.concat(DF_rings_dist, axis=0, ignore_index=True) #(nmols, 5 + 1) ;; for distribution!
        DF_lipinski, DF_posebuster, DF_moses = list(map(lambda inp: pd.concat(inp, axis=0, ignore_index=True), [DF_lipinski, DF_posebuster, DF_moses] ))
        DF = pd.concat( [DF_lipinski, DF_posebuster, DF_moses], axis=1 ) #num_size, columns
        DF.rename(mapper=lambda inp: f"size_{inp + 8}", axis='index', inplace=True)
        
        return DF, DF_rings_dist
        
if __name__ == "__main__":
    ###Current as of Mar 1st, 2024
    if args.run_analysis:
        if not args.turn_off_run_test:
            print(cf.on_yellow(f"Running Analysis on GENERATED data with {args.size_prefix}!!!!"))
            ###3D
            gen, files, file_counter = Analyse_generation.get_posebuster_stats([args.size_prefix]) # filtration 1
            if len(gen) !=0:
                pass
            else:
                gen = args.gen
            ###Drugness
            gen, files = Analyse_generation.get_lipinski(gen, files, file_counter, args.size_prefix) #filtration 2
            ###2D
            gen, files = Analyse_generation.get_moses_stats(gen=gen, files=files, train=args.train, test=args.valtest, test_scaffolds=args.valtest, file_counter_from_posebuster=file_counter, size_prefix=args.size_prefix) # final filtration
            # print(files)
            # gen = ["c1ccccc1", "c1cnccc1", "C1CCCCC1", "C1CNCCC1", "CCCCCC", "CCNCCC", "c12ccccc1NC=C2", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "Cc1c(Nc2nccc(-c3cnccc3)n2)cc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1"]
            # files = np.arange(len(gen))
            rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings = bonds_and_rings(gen, files, args.size_prefix)
            # for prop in [rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings]:
            #     print(prop)
        else:
            print(cf.on_green(f"Running Analysis on TEST data!!!!"))
            ###3D
            gen, file_counter = Analyse_generation.get_posebuster_stats_for_test() # filtration 1
            ###Drugness
            gen = Analyse_generation.get_lipinski_for_test(gen, file_counter) #filtration 2
            ###2D
            gen = Analyse_generation.get_moses_stats_for_test(gen=gen,  train=args.train, test=args.valtest, test_scaffolds=args.valtest, file_counter_from_posebuster=file_counter) # final filtration
            rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings = bonds_and_rings(gen)
            
    else:
        if not args.turn_off_run_test:
            print(cf.on_blue(f"Concatenating GENERATED data statistics!!!!"))
            DF, DF_rings_dist = Analyse_generation.collate_fn()
            print(DF.loc[:, ["IntDiv", "IntDiv2"]])
            print(DF)
            print(DF_rings_dist.groupby("size").mean())
        else:
            print(cf.on_red(f"Concatenating TEST data statistics!!!!"))
            DF, DF_rings_dist = Analyse_generation.collate_fn_for_test()
            print(DF.loc[:, ["IntDiv", "IntDiv2"]])
            print(DF)
            print(DF_rings_dist.mean(axis=0))
        

