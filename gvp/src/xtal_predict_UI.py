import tensorflow as tf
from models import MQAModel

from validate_performance_on_xtals import process_strucs, predict_on_xtals
import sys, os
import mdtraj as md
from glob import glob
import numpy as np

def make_predictions(strucs, model, nn_path):
    '''
    strucs : list of single frame MDTraj trajectories
    model : MQAModel corresponding to network in nn_path
    nn_path : path to checkpoint files
    '''
    X, S, mask = process_strucs(strucs)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions

# main method
if __name__ == '__main__':
    model_path = sys.argv[1]
    pdb_file_path = sys.argv[2]
    output_path = sys.argv[3]

    # TO DO - provide pdbs
    strucs = [md.load(pdb_file_path)]

    # create a MQA model
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100

    # MQA Model used for selected NN network
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

    predictions = make_predictions(strucs, model, model_path)

    np.savetxt(os.path.join(output_path,'predictions.txt'), predictions, fmt='%.4g', delimiter='\n')




