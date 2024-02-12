from posebusters import PoseBusters
from pathlib import Path
from rdkit import Chem
import argparse
from typing import List
import os
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filenames", "-f", nargs="*", help="xyz file names")
parser.add_argument("--kinase_file_names", "-f", nargs="*", help="kinase file names to test")
args = parser.parse_args()

def get_posebuster_stats(kinase_file_names: List[str]):
    for kinase_file_prefix in kinase_file_names:
        filenames = sorted(glob.glob((os.path.join("data_docking/result_difflinker", kinase_file_prefix + "*")))
        # pred_files = [Chem.MolFromXYZFile(os.path.join("data_docking/result_difflinker", filename)) if filename.endswith("xyz") else Chem.SDMolSupplier(os.path.join("data_docking/result_difflinker", filename))[0] for filename in filenames] 
        pred_files = [Chem.SDMolSupplier(filename)[0] for filename in filenames] 
        buster = PoseBusters(config="mol")
        df = buster.bust(pred_files, None, None, full_report=False)
        # print(df.columns)
        # print(df.values)
        Df = pd.DataFrame(data=df.values, columns=df.columns.tolist())
        print(Df)
        # df.drop(index=["molecule", "file"], inplace=True)
        # print(df)

if __name__ == "__main__":
    get_posebuster_stats(args.kinase_file_names)

    ## Current as of [Feb 1st 2024]
    ## git pull && python -m posebuster_analysis --filenames output_0_2_KLIF_test_frag_.xyz output_0_3_KLIF_test_frag_.xyz
