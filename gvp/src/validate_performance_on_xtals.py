from models import MQAModel
from util import load_checkpoint
import tensorflow as tf
import mdtraj as md
import numpy as np
import os
from glob import glob
from tensorflow import keras as keras
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

def process_strucs(strucs):
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

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l, 4, 3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask

def process_paths(apo_IDs, use_tensors=True, use_lm=False):
    """Takes a list of apo IDs to pdb files
    """
    pdb_dir = '/project/bowmore/ameller/projects/pocket_prediction/val_structures/'
    paths = [pdb_dir + e + '_clean_h.pdb' for e in apo_IDs]

    paths = [p if os.path.exists(p)
             else pdb_dir + apo_IDs[i].upper() + '_clean_h.pdb'
             for i, p in enumerate(paths)]

    strucs = [md.load(p) for p in paths]
    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])

    if use_tensors:
        X = np.zeros([B, L_max, 5, 3], dtype=np.float32)
    else:
        X = np.zeros([B, L_max, 4, 3], dtype=np.float32)

    if use_lm:
        S = np.zeros([B, L_max, 1280], dtype=np.float32)
    else:
        S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues

        if use_tensors:
            fn = f'X_tensor_{apo_IDs[i]}.npy'
            X_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_X/{fn}'
            if os.path.exists(X_path):
                xyz = np.load(X_path)
            else:
                # try capitalizing the apo PDB ID
                fn = f'X_tensor_{apo_IDs[i].upper()}.npy'
                X_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_X/{fn}'
                xyz = np.load(X_path)
        else:
            xyz = prot_bb.xyz.reshape(l, 4, 3)

        if use_lm:
            fn = f'S_embedding_{apo_IDs[i]}.npy'
            S_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_S/{fn}'
            if os.path.exists(S_path):
                S[i, :l] = np.load(S_path)
            else:
                fn = f'S_embedding_{apo_IDs[i].upper()}.npy'
                S_path = f'/project/bowmanlab/mdward/projects/FAST-pocket-pred/precompute_S/{fn}'
                S[i, :l] = np.load(S_path)
        else:
            seq = [r.name for r in prot_bb.top.residues]
            S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)

        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    if use_lm:
        S = tf.convert_to_tensor(S)

    return X, S, mask

# def predict(model, nn_path, strucs, opt=tf.keras.optimizers.Adam()):
#     load_checkpoint(model, opt, nn_path)
#     X, S, mask = process_struc(strucs)
#     # model.multiclass = True
#     prediction = model(X, S, mask, train=False, res_level=True)
#     return prediction


def predict_on_xtals(model, nn_path, X, S, mask, opt=tf.keras.optimizers.legacy.Adam()):
    load_checkpoint(model, opt, nn_path)
    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction

def assess_performance(predictions, true_labels, mask):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score

    # calculate AUC and PR-AUC for each protein
    assert mask.sum() == sum([len(y_true) for y_true in true_labels])

    protein_aucs = [roc_auc_score(y_true, preds[:len(y_true)])
                    for (y_true, preds) in zip(true_labels, predictions)]

    precisions = [precision_recall_curve(y_true, preds[:len(y_true)])[0]
                  for (y_true, preds) in zip(true_labels, predictions)]

    recalls = [precision_recall_curve(y_true, preds[:len(y_true)])[1]
               for (y_true, preds) in zip(true_labels, predictions)]

    protein_pr_aucs = [auc(recall, precision)
                       for (recall, precision) in zip(recalls, precisions)]

    return protein_aucs, protein_pr_aucs

if __name__ == '__main__':
    # Load val set
    # val_set = np.load('/project/bowmanlab/borowsky.jonathan/FAST-cs/new_pockets/labels/new_pocket_labels_validation.npy',
    #                   allow_pickle=True)

    # val_set = np.load('/project/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets/new_pockets/labels/'
    #                   'new_pocket_labels_validation_all1.npy', allow_pickle=True)
    # strucs = [md.load(p[0]) for p in val_set]

    all_labels = np.load('/project/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets/'
                         'all-GVP-project-ligand-resis.npy', allow_pickle=True)

    label_dictionary = {
        e[0][0].upper(): e[5]
        for e in all_labels[0]
    }
    # val_set_apo_ids = np.load('/project/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets/new_pockets/labels/validation_apo_ids_all.npy')
    val_set_apo_ids = np.load('../data/validation_apo_ids.npy')

    upper_val_set_apo_ids = [e.upper() for e in val_set_apo_ids]
    true_labels = [label_dictionary[e[:-1].upper()] for e in val_set_apo_ids]
                   # if e[0][0] + e[0][1] in val_set_apo_ids or
                   # e[0][0] + e[0][1] in upper_val_set_apo_ids]

    # Create model
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100

    # get NN directories
    nn_dirs = glob('/project/bowmanlab/ameller/gvp/task2/*/*')

    use_tensors = True

    protein_level_performance = False

    X, S, mask = process_paths(val_set_apo_ids, use_tensors=use_tensors)

    for nn_dir in nn_dirs:

        # determine number of compeleted epochs
        val_files = glob(f"{nn_dir}/val_pr_auc_*.npy")

        if len(val_files) == 0:
            continue

        # Determine network name
        index_filenames = glob(f"{nn_dir}/*.index")
        nn_id = os.path.basename(index_filenames[0]).split('_')[0]

        if 'sidechain' in nn_dir:
            ablate_sidechain_vectors = False
        else:
            ablate_sidechain_vectors = True

        if not use_tensors and 'sidechain' in nn_path:
            continue

        print(nn_dir)

        model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                         hidden_dim=(16, HIDDEN_DIM),
                         num_layers=NUM_LAYERS, dropout=DROPOUT_RATE,
                         ablate_sidechain_vectors=ablate_sidechain_vectors)


        # Determine which network to use (i.e. epoch with best AUC)
        pr_auc = []

        auc_metric = keras.metrics.AUC(name='auc')
        pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

        for epoch in tqdm(range(len(val_files))):
            nn_path = f"{nn_dir}/{nn_id}_{str(epoch).zfill(3)}"
            predictions = predict_on_xtals(model, nn_path, X, S, mask)

            if protein_level_performance:
                protein_aucs, protein_pr_aucs = assess_performance(predictions, true_labels, mask)
                np.save(os.path.join(nn_dir, f"val_protein_aucs_{epoch}.npy"), protein_aucs)
                np.save(os.path.join(nn_dir, f"val_protein_pr_aucs_{epoch}.npy"), protein_pr_aucs)

            # Pooled PR AUC and AUC
            auc_metric.reset_states()
            pr_auc_metric.reset_states()

            y_pred = predictions[mask.astype(bool)]
            y_true = np.concatenate(true_labels)

            auc_metric.update_state(y_true, y_pred)
            pr_auc_metric.update_state(y_true, y_pred)

            np.save(os.path.join(nn_dir, f"val_auc_{epoch}.npy"), auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_pr_auc_{epoch}.npy"), pr_auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_y_pred_{epoch}.npy"), y_pred)
            np.save(os.path.join(nn_dir, f"val_y_true_{epoch}.npy"), y_true)

            pr_auc.append(pr_auc_metric.result().numpy())

        print(pr_auc)
        best_epoch = np.argmax(pr_auc)
        nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"

        predictions = predict_on_xtals(model, nn_path, X, S, mask)

        np.save(f'{nn_dir}/full_val_set_y_pred.npy', predictions)

        if protein_level_performance:
            protein_aucs, protein_pr_aucs = assess_performance(predictions, true_labels, mask)
            np.save(f'{nn_dir}/full_val_set_protein_aucs.npy', protein_aucs)
            np.save(f'{nn_dir}/full_val_set_protein_pr_aucs.npy', protein_pr_aucs)

        np.save(f'{nn_dir}/full_val_set_y_true.npy', true_labels)



    # This is the path to the trained model. You should have something like
    # 1623688914_049.index and 1623688914_049.data-00000-of-00001 in the 'models' dir
    # INSERT BEST MODEL HERE
    # nn_dir = ("net_8-50_1-32_16-100_nl_4_lr_0.00029_dr_0.13225_b1_"
    #           "20epoch_feat_method_nearby-pv-procedure_rank_7_stride_1"
    #           "_pos116_neg20_window_40ns_test_TEM-1MY0-1BSQ-nsp5-il6-2OFV")
    # nn_path = (f"/project/bowmanlab/ameller/gvp/{nn_dir}/"
    #            "1636056869_018")

    # nn_path = "/project/bowmore/ameller/gvp/models/1623688914_049"
    # opt = tf.keras.optimizers.Adam()

    # prediction = predict(model, nn_path, strucs)
    # np.save(f'../data/val-set-predictions-{nn_dir}.npy', prediction)
