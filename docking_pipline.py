import subprocess
import shutil
import os 
# completed_process = subprocess.run(["ls", "-l"], capture_output=True, text=True)
# print(f"Return code: {completed_process.returncode}")
# print(f"Have {len(completed_process.stdout)} bytes in stdout:\n{completed_process.stdout}")

#### use f-pocket to predict pocket 
# fpocket -f data_docking/protein/1ADE.pdb -o 
'''reference: https://github.com/Discngine/fpocket'''


protein_path = 'data_docking/protein'
output_path = 'data_docking/result_fpocket' 

protein_name = '1ADE.pdb'
outfile_name = protein_name.split('.')[0] +'_out' 

if 0:
    try:
        if not os.path.exists(os.path.join(protein_path, outfile_name)):
            # Run the command and wait for it to complete
            completed_process = subprocess.run(["fpocket", "-f", os.path.join(protein_path, protein_name)], check=True, capture_output=True, text=True)
            
            # Print the return code
            print(f"Return code: {completed_process.returncode}")
            # Print captured stdout and stderr, if any
            print(f"Output: {completed_process.stdout}")
            print(f"Error: {completed_process.stderr}")

            # Move the output file to the desired location
            shutil.move(os.path.join(protein_path, outfile_name), 
                        os.path.join(output_path))

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


## Use gvp predict pocket 
if 1: 
    import torch
    import torch_geometric
    import tensorflow as tf
    from gvp.src.models import MQAModel
    import numpy as np
    from glob import glob
    import mdtraj as md
    import os

    from gvp.src.validate_performance_on_xtals import process_strucs, predict_on_xtals

    # esure working directory to be the same as where your Python script is located, regardless of where you run it from
    # script_path = os.path.abspath(__file__)
    # script_dir = os.path.dirname(script_path)
    # os.chdir(script_dir)

    def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
        '''
            pdb_paths : list of pdb paths
            model : MQAModel corresponding to network in nn_path
            nn_path : path to checkpoint files
        '''
        strucs = [md.load(s) for s in pdb_paths]
        X, S, mask = process_strucs(strucs)
        if debug:
            np.save(f'{output_basename}_X.npy', X)
            np.save(f'{output_basename}_S.npy', S)
            np.save(f'{output_basename}_mask.npy', mask)
        predictions = predict_on_xtals(model, nn_path, X, S, mask)
        return predictions

    # TO DO - provide input pdb(s), output name, and output folder
    # strucs = [
    #     '../../data_docking/protein/1ADE.pdb'
    # ]
    strucs = [os.path.join(protein_path, protein_name)]
    #output_name = '1ADE'
    #output_folder = '../../data_docking/result_gvp'
    output_folder = 'data_docking/result_gvp'
    # debugging mode can be turned on to output protein features and sequence
    debug = False

    # Load MQA Model used for selected NN network

    nn_path = "./gvp/models/pocketminer"
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                        hidden_dim=(16, HIDDEN_DIM),
                        num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)


    if debug:
        output_basename = f'{output_folder}/{outfile_name}'
        predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
    else:
        predictions = make_predictions(strucs, model, nn_path)

    # output filename can be modified here
    np.save(f'{output_folder}/{outfile_name}-preds.npy', predictions)
    np.savetxt(os.path.join(output_folder,f'{outfile_name}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')

