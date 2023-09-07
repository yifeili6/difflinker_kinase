import tensorflow as tf
import numpy as np
import mdtraj as md
import pandas as pd
import glob
import os
import time

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

DATA_DIR = "/project/bowmore/ameller/gvp/data/"

def determine_global_weights(filestem, positive_cutoff, negative_cutoff):
    y_train = np.load(os.path.join(DATA_DIR, f"5-fold-cv/y-train-{filestem}.npy"),allow_pickle=True)
    all_examples = np.concatenate(y_train)
    number_positive_examples = np.sum(all_examples >= positive_cutoff)
    number_negative_examples = np.sum(all_examples < negative_cutoff)
    total_examples = number_positive_examples + number_negative_examples
    positive_weight = 1 / number_positive_examples * (total_examples / 2.0)
    negative_weight = 1 / number_negative_examples * (total_examples / 2.0)
    return positive_weight, negative_weight

def pockets_dataset_fold(batch_size, filestem):
    #will have [(xtc,pdb,index,residue,1/0),...]
    X_train = np.load(os.path.join(DATA_DIR, f"5-fold-cv/X-train-{filestem}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"5-fold-cv/y-train-{filestem}.npy"),allow_pickle=True)
    trainset = list(zip(X_train,y_train))

    X_validate = np.load(os.path.join(DATA_DIR, f"5-fold-cv/X-val-{filestem}.npy"))
    y_validate = np.load(os.path.join(DATA_DIR, f"5-fold-cv/y-val-{filestem}.npy"),allow_pickle=True)
    valset = list(zip(X_validate,y_validate))

    X_test = np.load(os.path.join(DATA_DIR, f"5-fold-cv/X-test-{filestem}.npy"))
    y_test = np.load(os.path.join(DATA_DIR, f"5-fold-cv/y-test-{filestem}.npy"),allow_pickle=True)
    testset = list(zip(X_test, y_test))

    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)

    output_types = (tf.float32, tf.int32, tf.int32, tf.string, tf.float32)
    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)

    return trainset, valset, testset

def pockets_dataset(batch_size, filestem):
    #will have [(xtc,pdb,index,residue,1/0),...]
    X_train = np.load(os.path.join(DATA_DIR, f"train-test-split/X-train-{filestem}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"train-test-split/y-train-{filestem}.npy"),allow_pickle=True)
    trainset = list(zip(X_train,y_train))

    X_validate = np.load(os.path.join(DATA_DIR, f"train-test-split/X-val-{filestem}.npy"))
    y_validate = np.load(os.path.join(DATA_DIR, f"train-test-split/y-val-{filestem}.npy"),allow_pickle=True)
    valset = list(zip(X_validate,y_validate))

    X_test = np.load(os.path.join(DATA_DIR, f"train-test-split/X-test-{filestem}.npy"))
    y_test = np.load(os.path.join(DATA_DIR, f"train-test-split/y-test-{filestem}.npy"),allow_pickle=True)
    testset = list(zip(X_test, y_test))

    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)

    output_types = (tf.float32, tf.int32, tf.int32, tf.string, tf.float32)
    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)

    return trainset, valset, testset

def simulation_dataset(batch_size, filestem, use_tensors=True, y_type='int32', use_lm=False):
    '''
    for training models to locate cryptic pockets in xtals
    returns only training datasets from simulation data
    '''
    X_train = np.load(os.path.join(DATA_DIR, f"task2/X-train-{filestem}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"task2/y-train-{filestem}.npy"), allow_pickle=True)
    trainset = list(zip(X_train, y_train))

    trainset = DynamicLoader(trainset, batch_size, use_tensors=use_tensors, y_type=y_type, use_lm=use_lm)

    output_types = (tf.float32)

    if y_type == 'int32' and use_lm:
        output_types = (tf.float32, tf.float32, tf.int32, tf.string, tf.float32)
    elif y_type == 'int32' and not use_lm:
        output_types = (tf.float32, tf.int32, tf.int32, tf.string, tf.float32)
    elif y_type == 'float32' and use_lm:
        output_types = (tf.float32, tf.float32, tf.float32, tf.string, tf.float32)
    else:
        output_types = (tf.float32, tf.int32, tf.float32, tf.string, tf.float32)

    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)

    return trainset


def parse_batch(batch, use_tensors=True, y_type='int32', use_lm=False):
    # t0 = time.time()
    #Batch will have [(xtc,pdb,index,residue,1/0),...]
    pdbs = []
    #can parallelize to improve speed
    for ex in batch:
        x, y = ex
        pdb = md.load(x[1])
        prot_iis = pdb.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = pdb.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(batch)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    if use_tensors:
        # precomputed tensors include terminal sidechain atom
        X = np.zeros([B, L_max, 5, 3], dtype=np.float32)
    else:
        X = np.zeros([B, L_max, 4, 3], dtype=np.float32)

    if use_lm:
        S = np.zeros([B, L_max, 1280], dtype=np.float32)
    else:
        S = np.zeros([B, L_max], dtype=np.int32)

    # -1 so we can distinguish 0 pocket volume and padded indices later
    if y_type == 'int32':
        y = np.zeros([B, L_max], dtype=np.int32) - 1
    elif y_type == 'float32':
        y = np.zeros([B, L_max], dtype=np.float32) - 1

    meta = []
    for i, ex in enumerate(batch):
        x, targs = ex
        traj_fn, pdb_fn, traj_iis = x
        traj_iis = int(traj_iis)

        pdb = pdbs[i]
        l = pdb.top.n_residues

        if use_tensors:
            fn = 'X_tensor_' + traj_fn.split('/')[-1].split('.')[0] + '_frame' + str(traj_iis) + '.npy'
            X_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_X/{fn}'
            xyz = np.load(X_path)
        else:
            struc = md.load_frame(traj_fn, traj_iis, top=pdb_fn)
            prot_iis = struc.top.select("protein and (name N or name CA or name C or name O)")
            prot_bb = struc.atom_slice(prot_iis)
            xyz = prot_bb.xyz.reshape(l, 4, 3)

        if use_lm:
            fn = 'S_embedding_' + os.path.basename(pdb_fn).split('.')[0] + '.npy'
            S_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_S/{fn}'
            S[i, :l] = np.load(S_path)
        else:
            seq = [r.name for r in pdb.top.residues]
            S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)

        X[i] = np.pad(xyz, [[0, L_max - l], [0, 0], [0, 0]],
                      'constant', constant_values=(np.nan, ))
        y[i, :l] = targs
        meta.append(x)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    # t1 = time.time()
    # print(use_tensors, t1-t0)
    return X, S, y, meta, mask

class DynamicLoader(): 
    def __init__(self, dataset, batch_size=32, shuffle=True,
                 use_tensors=True, y_type='int32', use_lm=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_tensors = use_tensors
        self.y_type = y_type
        self.use_lm = use_lm

    def chunks(self,arr, chunk_size):
        """Yield successive chunk_size chunks from arr."""
        for i in range(0, len(arr), chunk_size):
            yield arr[i:i + chunk_size]

    def batch(self):
        dataset = self.dataset
        np.random.shuffle(dataset)
        self.clusters = list(self.chunks(dataset,self.batch_size))

    def __iter__(self):
        self.batch()
        if self.shuffle: np.random.shuffle(self.clusters)
        N = len(self.clusters)
        for batch in self.clusters[:N]:
            yield parse_batch(batch, use_tensors=self.use_tensors,
                              y_type=self.y_type, use_lm=self.use_lm)
