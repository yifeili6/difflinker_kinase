from models import MQAModel
from util import load_checkpoint
import tensorflow as tf
import mdtraj as md
import numpy as np
import os
from glob import glob
from tensorflow import keras as keras
from tqdm import tqdm
from validate_performance_on_xtals import process_strucs, process_paths, predict_on_xtals
from itertools import accumulate

def determine_optimal_threshold(y_pred, y_true):
    from sklearn import metrics

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    j_statistic = tpr - fpr

    return thresholds[j_statistic.argmax()]


def determine_recall_or_sensitivity(parsed_y_pred, parsed_y_true, threshold):
    assert len(parsed_y_pred) == len(parsed_y_true)

    protein_performance = []
    for i in range(len(parsed_y_pred)):
        # if protein is a source of negative examples
        if 0 in parsed_y_true[i]:
            assert 1 not in parsed_y_true[i]
            neg_prediction = parsed_y_pred[i] < threshold
            sensitivity = sum(neg_prediction) / len(parsed_y_true[i])
            protein_performance.append(sensitivity)
        else:
            assert 0 not in parsed_y_true[i]
            pos_prediction = parsed_y_pred[i] >= threshold
            recall = sum(pos_prediction) / len(parsed_y_true[i])
            protein_performance.append(recall)

    return protein_performance


if __name__ == '__main__':
    label_dictionary = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_label_dictionary.npy',
                               allow_pickle=True).item()

    # get NN directories
    nn_dirs = glob('/project/bowmanlab/ameller/gvp/task2/*/*')

    for nn_dir in nn_dirs:
        # if sidechain is in name, need to use tensors
        if 'sidechain' in nn_dir:
            continue

        # determine number of compeleted epochs
        val_files = glob(f"{nn_dir}/val_new_auc_*.npy")

        if len(val_files) == 0:
            continue

        aucs = []
        for epoch in range(len(val_files)):
            aucs.append(np.load(os.path.join(nn_dir, f"val_new_auc_{epoch}.npy")))

        best_epoch = np.argmax(aucs)
        print(aucs[best_epoch])

        print(nn_dir, best_epoch)

        y_pred = np.load(os.path.join(nn_dir, f"val_new_y_pred_{best_epoch}.npy"))
        y_true = np.load(os.path.join(nn_dir, f"val_new_y_true_{best_epoch}.npy"))

        optimal_threshold = determine_optimal_threshold(y_pred, y_true)
        np.save(os.path.join(nn_dir, 'val_new_optimal_threshold.npy'), optimal_threshold)

        print(optimal_threshold)

        lengths = [sum(l != 2) for l in label_dictionary.values()]
        parsed_y_pred = [y_pred[end - length:end] for length, end in zip(lengths, accumulate(lengths))]
        parsed_y_true = [y_true[end - length:end] for length, end in zip(lengths, accumulate(lengths))]

        protein_performance = determine_recall_or_sensitivity(parsed_y_pred, parsed_y_true, optimal_threshold)

        np.save(os.path.join(nn_dir, 'val_new_protein_performance.npy'), protein_performance)

        print([(p, round(v, 3)) for p, v in zip(label_dictionary.keys(), protein_performance)])

