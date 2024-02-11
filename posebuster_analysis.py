from posebusters import PoseBusters
from pathlib import Path
from rdkit import Chem
import argparse
from typing import List
import os

parser = argparse.ArgumentParser()
parser.add_argument("--filenames", "-f", nargs="*", help="xyz file names")
args = parser.parse_args()

def get_posebuster_stats(filenames: List[str]):
    pred_files = [Chem.MolFromXYZFile(os.path.join("data_docking/result_difflinker", filename)) if filename.endswith("xyz") else Chem.SDMolSupplier(os.path.join("data_docking/result_difflinker", filename))[0] for filename in filenames] 
    buster = PoseBusters(config="mol")
    df = buster.bust(pred_files, None, None, full_report=False)
    # print(df.columns)
    # print(df.values)
    print(df)

if __name__ == "__main__":
    get_posebuster_stats(args.filenames)

    ## Current as of [Feb 1st 2024]
    ## git pull && python -m posebuster_analysis --filenames output_0_2_KLIF_test_frag_.xyz output_0_3_KLIF_test_frag_.xyz
