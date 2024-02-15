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
from moses.metrics.utils import mapper, mol_passes_filters

rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
# parser.add_argument("--filenames", "-f", nargs="*", help="xyz file names")
parser.add_argument("--gen", "-g", type=str, default="data_docking/result_difflinker", help="generated ligand SDF directory")
parser.add_argument("--train", "-t", type=str, default="datasets/KLIF_train_table.csv", help="train dataset")
parser.add_argument("--valtest", "-vt", type=str, default="datasets/KLIF_ValTest_table.csv", help="ValTest dataset")
parser.add_argument("--kinase_prefix_names", "-f", nargs="*", help="kinase file names to test")
args = parser.parse_args()

warnings.simplefilter('ignore')

def get_posebuster_stats(kinase_prefix_names: List[str]):
    return_good_mols = []
    return_good_files = []

    for kinase_file_prefix in kinase_prefix_names:
        try:
            filenames = sorted(glob.glob(os.path.join("data_docking/result_difflinker", kinase_file_prefix + "*.sdf")))
            # pred_files = [Chem.MolFromXYZFile(os.path.join("data_docking/result_difflinker", filename)) if filename.endswith("xyz") else Chem.SDMolSupplier(os.path.join("data_docking/result_difflinker", filename))[0] for filename in filenames] 
            pred_files_ = [(filename, Chem.SDMolSupplier(filename)[0]) for filename in filenames] 
            print(cf.red(f"None location: {np.where(np.array(pred_files_)[:, 1] == None)[0]}"))
            pred_files = [p[1] for p in pred_files_ if p[1] is not None] #List[MOL]
            pred_file_names = [p[0] for p in pred_files_ if p[1] is not None] #List[str]

            buster = PoseBusters(config="mol")
            df = buster.bust(pred_files, None, None, full_report=False)
            # print(df.columns)
            # print(df.values)
            Df = pd.DataFrame(data=df.values, columns=df.columns.tolist())
            print(cf.on_green(f"{kinase_file_prefix} is pose busted"))
            Df["Total Pass"] = Df.all(axis=1)
            print("Result:\n", Df)

            good_mols = np.array(pred_files)[Df.values[:,-1].astype(bool)]
            good_mol_files = np.array(pred_file_names)[Df.values[:,-1].astype(bool)]
            return_good_mols.extend(good_mols.tolist())
            return_good_files.extend(good_mol_files.tolist())
            # df.drop(index=["molecule", "file"], inplace=True)
            # print(df)
        except Exception as e:
            # raise RuntimeError from e #cannot continue if raising errors!
            print(cf.on_red(f"Something happend... skipping!!!\n {e}"))
            continue
    # print(return_good_smiles)
    return return_good_mols, return_good_files

def get_moses_stats(gen=None, k=None, n_jobs=os.cpu_count()-1,
                    device='cuda', batch_size=512, pool=None,
                    test=None, test_scaffolds=None,
                    ptest=None, ptest_scaffolds=None,
                    train=None, files=None):
                        
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
    
    print(cf.on_blue("MOSES metrics"))    
    print(metrics)
    return gen.tolist(), files.tolist()

def get_lipinski(gen: List[str], files: List[str]):
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
    
    print(cf.on_yellow("Lipinski Rule of 5"))
    print(np.mean(lipinski_results))

    return_good_smiles = np.array(gen)[np.array(lipinski_results).astype(bool)]
    return_good_files = np.array(files)[np.array(lipinski_results).astype(bool)]
    return return_good_smiles.tolist(), return_good_files.tolist()
    
if __name__ == "__main__":
    ###3D
    gen, files = get_posebuster_stats(args.kinase_prefix_names) # filtration 1
    if len(gen) !=0:
        pass
    else:
        gen = args.gen
    ###Drugness
    gen, files = get_lipinski(gen, files) #filtration 2
    ###2D
    gen, files = get_moses_stats(gen=gen, files=files, train=args.train, test=args.valtest, test_scaffolds=args.valtest) # final filtration
    print(files)

    ## Current as of [Feb 1st 2024]
    ## git pull && python -m posebuster_analysis --filenames output_0_2_KLIF_test_frag_.xyz output_0_3_KLIF_test_frag_.xyz
