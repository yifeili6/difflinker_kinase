import pandas as pd
from typing import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kinase_names', nargs="*", type=str, required=True)

def get_kinase_indices(kinase_names: List[str]) -> List[List[str]]:
    """Return corresponding indices of kinase indices"""
  
    df0=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_train_table.csv", header=0)
    df1=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_val_table.csv", header=0)
    df2=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_test_table.csv", header=0)
    
    df=pd.concat([df1, df2], axis=0)
    df.reset_index(drop=True, inplace=True)
    
    indices = df.index[df.molecule_name.apply(lambda inp: inp[:4] in kinase_names)].tolist()
    info = df.loc[df.molecule_name.apply(lambda inp: inp[:4] in kinase_names), ["molecule_name", "anchor_1", "anchor_2", "linker_size"]]

    return indices, info

if __name__ == "__main__":
    args = parser.parse_args()
    indices, info = get_kinase_indices(args.kinase_names)
    print(indices)
    print(info)
