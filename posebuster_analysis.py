from posebusters import PosBusters
from pathlib import Path
from rdkit import Chem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", "-f", type=str, help="xyz file name")
args = parse.parse_args()

def get_posebuster_stats(filename: str):
    pred_file = Chem.MolFromXYZFile(filename)
    buster = PoseBusters(config="mol")
    df = buster.bust([pred_file], None, None, full_report=False)
    print(df)

if __name__ == "__main__":
    get_posebuster_states(args.filename)

    ## Current as of [Feb 1st 2024]
    ## git pull && python -m posebuster_analysis --filename data_docking/result_difflinker/output_0_2_KLIF_test_frag_.xyz
