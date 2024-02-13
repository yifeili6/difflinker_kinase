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
    retun_good_smiles = []
    for kinase_file_prefix in kinase_prefix_names:
        try:
            filenames = sorted(glob.glob(os.path.join("data_docking/result_difflinker", kinase_file_prefix + "*")))
            # pred_files = [Chem.MolFromXYZFile(os.path.join("data_docking/result_difflinker", filename)) if filename.endswith("xyz") else Chem.SDMolSupplier(os.path.join("data_docking/result_difflinker", filename))[0] for filename in filenames] 
            pred_files = [Chem.SDMolSupplier(filename)[0] for filename in filenames] 
            print(cf.red(f"None location: {np.where(np.array(pred_files) == None)[0]}"))
            pred_files = [p for p in pred_files if p is not None]
            buster = PoseBusters(config="mol")
            df = buster.bust(pred_files, None, None, full_report=False)
            # print(df.columns)
            # print(df.values)
            Df = pd.DataFrame(data=df.values, columns=df.columns.tolist())
            print(cf.on_green(f"{kinase_file_prefix} is pose busted"))
            Df["Total Pass"] = Df.all(axis=1)
            print("Result:\n", Df)
            good_smiles = np.array(pred_files)[Df.values[:,-1]]
            print(np.array(pred_files), Df.values[:,-1])

            retun_good_smiles.extend(good_smiles.tolist())
            # df.drop(index=["molecule", "file"], inplace=True)
            # print(df)
        except Exception as e:
            # raise RuntimeError from e #cannot continue if raising errors!
            print(cf.on_red(f"Something happend... skipping!!!\n {e}"))
            continue
    return retun_good_smiles

def get_moses_stats(gen, k=None, n_jobs=os.cpu_count()-1,
                    device='cuda', batch_size=512, pool=None,
                    test=None, test_scaffolds=None,
                    ptest=None, ptest_scaffolds=None,
                    train=None):
    if isinstance(gen, list):
        pass
    else:
        gen = glob.glob(gen + "/*.sdf")
        # print(gen)
        gen: List[str] = [Chem.SDMolSupplier(g)[0] for g in gen]
        gen: List[str] = [Chem.MolToSmiles(g) for g in gen if g is not None]
        
    train = pd.read_csv(train).molecule.drop_duplicates().tolist()
    test = pd.read_csv(test).molecule.drop_duplicates().tolist()
    test_scaffolds = pd.read_csv(test_scaffolds).molecule.drop_duplicates().tolist()

    metrics = moses.get_all_metrics(gen=gen, k=k, n_jobs=n_jobs,
                    device=device, batch_size=batch_size, pool=pool,
                    test=test, test_scaffolds=test_scaffolds,
                    ptest=ptest, ptest_scaffolds=ptest_scaffolds,
                    train=train)
    print(cf.on_blue("MOSES metrics"))    
    print(metrics)
    return metrics
    
if __name__ == "__main__":
    gen = get_posebuster_stats(args.kinase_prefix_names)
    if len(gen) !=0:
        pass
    else:
        gen = args.gen
    get_moses_stats(gen, train=args.train, test=args.valtest, test_scaffolds=args.valtest)

    ## Current as of [Feb 1st 2024]
    ## git pull && python -m posebuster_analysis --filenames output_0_2_KLIF_test_frag_.xyz output_0_3_KLIF_test_frag_.xyz
