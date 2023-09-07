import tensorflow as tf
import numpy as np
import mdtraj as md
import pandas as pd
import glob
import os

TRAIN_INDS_FN = "/project/bowmanlab/mdward/projects/rocklin-collab/training/train_inds_balanced_oversampling.npy"
VAL_INDS_FN = "/project/bowmanlab/mdward/projects/rocklin-collab/training/validation_inds.npy"
TEST_INDS_FN = "/project/bowmanlab/mdward/projects/rocklin-collab/training/test_inds.npy"

PROTEASE_DATA_DIR = "/project/bowmanlab/mdward/projects/rocklin-collab/protease-data/"
PROTEASE_DATA = "/project/bowmanlab/mdward/projects/rocklin-collab/protease-data/miniprotein_rd123456_newvec_stability_GJR_210203.csv"


def rocklin_dataset(batch_size):
    trainset = np.load(TRAIN_INDS_FN)
    valset = np.load(VAL_INDS_FN)
    testset = np.load(TEST_INDS_FN)
    
    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)
    
    output_types = (tf.float32, tf.int32, tf.float32, tf.float32)
    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)
    
    return trainset, valset, testset

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

def parse_batch(batch):
    df = pd.read_csv(PROTEASE_DATA)
    pdbs = []
    for ind in batch:
        pdb_fn = df.iloc[ind]['name']
        fn = glob.glob(os.path.join(PROTEASE_DATA_DIR,"rd*/%s" % pdb_fn))
        pdbs.append(md.load(fn))
    
    B = len(batch)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    
    for i, pdb in enumerate(pdbs):
        l = pdb.top.n_residues
        seq = [r.name for r in pdb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        
        pdb_inds = pdb.top.select("name N or name CA or name C or name O")
        pdb = pdb.atom_slice(pdb_inds)
        xyz = pdb.xyz.reshape(l,4,3)   
        
        # Pad to the maximum length in the batch
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                        'constant', constant_values=(np.nan, ))
    
    #Output targets are structures with stabilities > 1
    scores = df.iloc[batch]['stabilityscore'].values
    y = scores >= 1
    y = y.astype(int)
    
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)
        
    return X, S, y, mask

class DynamicLoader(): 
    def __init__(self, dataset, batch_size=32, shuffle=True): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def chunks(self,arr, chunk_size):
        """Yield successive chunk_size chunks from arr."""
        for i in range(0, len(arr), chunk_size):
            yield arr[i:i + chunk_size]
        
    def batch(self):
        dataset = self.dataset
        self.clusters = list(self.chunks(dataset,self.batch_size))

    def __iter__(self):
        self.batch()
        if self.shuffle: np.random.shuffle(self.clusters)
        N = len(self.clusters)
        for batch in self.clusters[:N]:
            yield parse_batch(batch)
