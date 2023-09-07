import tensorflow as tf
#tf.debugging.enable_check_numerics()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import numpy as np
import glob
import mdtraj as md
from util import load_checkpoint
from models import MQAModel

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

def process_struc(strucs):
    """Takes a list of single frame md.Trajectory objects
    """
    
    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    
    for i,prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l,4,3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                        'constant', constant_values=(np.nan, ))
        
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask

model = MQAModel(node_features=(8,50), edge_features=(1,32), hidden_dim=(16,100)) 
nn_path = "../models/1623688914_049" 
opt=tf.keras.optimizers.Adam()
load_checkpoint(model,opt,nn_path)
af_fns = glob.glob("/project/bowmanlab/mdward/ml_datasets/af_structures/*pdb*")
y_pred = []
pdb_fns = []
for i,fn in enumerate(af_fns):
    struc = md.load(fn)
    strucs = [struc]
    ##TODO: write a function to get rid of terminal disordered regions
    ## and save out the resulting pdb
    ## see /project/bowmanlab/mdward/projects/FAST-pocket-pred/AF_results/calc_pred_density.py
    ## to find a function for reading in uncertainties from PDBs.
    ## Will be important to write out the AF uncertainties to the new saved pdb file!
    X, S, mask = process_struc(strucs)
    prediction = model(X, S, mask, train=False, res_level=True)
    y_pred.extend(prediction.numpy().tolist())
    pdb_fns.extend([fn])
    if i % 100 == 0:
        print(i)

np.save("../models/AF_predictions.npy", y_pred)
np.save("../models/AF_fns.npy", pdb_fns)

#nn_path = '../models/{}_{}'
#BS = 2

#def get_model():
    #Emailed the lead author for what these values should be, these are good defaults.
#    model = MQAModel(node_features=(8,50), edge_features=(1,32), hidden_dim=(16,100))
#    opt=tf.keras.optimizers.Adam()
#    load_checkpoint(model,opt,nn_path)
#    return model

#def predict_loop(dataset):
#    y_pred = []
#    for batch in tqdm.tqdm(dataset):
#        X, S, y, meta, M = batch
#        prediction = model(X, S, M, train=False, res_level=True)
#        y_pred.extend(prediction.numpy().tolist())

#def main():
#    trainset, valset, testset = pockets_dataset(BS)# batch size = N proteins
#    model = make_model()
#    loop(testset, model, train=False, val=True)
