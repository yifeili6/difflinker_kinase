import pandas as pd
from typing import *

def get_kinase_indices(kinase_names: List[str]) -> List[List[str]]:
    """Return corresponding indices of kinase indices"""
  
    df0=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_train_table.csv", header=0)
    df1=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_val_table.csv", header=0)
    df2=pd.read_csv("/Scr/yifei6/difflinker_kinase/data_docking/complex/processed_klif_wl/KLIF_test_table.csv", header=0)
    
    df=pd.concat([df0, df1, df2], axis=0)
    df.reset_index(drop=True, inplace=True)
    
    indices = df.index[df.molecule_name.apply(lambda inp: inp.startswith('5lqf'))].tolist()

    return 

if __name__ == "__main__":
    get_kinase_indices(["5lqf"])
