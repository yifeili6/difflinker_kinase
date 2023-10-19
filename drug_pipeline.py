import os 

# f-pocket related import
import subprocess
import shutil

# gvp related import 
import torch
import torch_geometric
import tensorflow as tf
from gvp.src.models import MQAModel
import numpy as np
from glob import glob
import mdtraj as md
from gvp.src.validate_performance_on_xtals import process_strucs, predict_on_xtals
import ast

# diff dock related import

# completed_process = subprocess.run(["ls", "-l"], capture_output=True, text=True)
# print(f"Return code: {completed_process.returncode}")
# print(f"Have {len(completed_process.stdout)} bytes in stdout:\n{completed_process.stdout}")

#### use f-pocket to predict pocket 
# fpocket -f data_docking/protein/1ADE.pdb -o 
'''reference: https://github.com/Discngine/fpocket'''

class PocketPrediction:
    def __init__(self, 
                 protein_path = 'data_docking/protein', 
                 ligand_path = 'data_docking/ligand',
                 outpath_fpocket = 'data_docking/result_fpocket' , 
                 outpath_vina = 'data_docking/result_vina' , 
                 outpath_diffdock = 'data_docking/result_diffdock',
                 outpath_difflinker = 'data_docking/result_difflinker',
                 outpath_gvp = 'data_docking/result_gvp', 
                 nn_path_gvp = "./gvp/models/pocketminer",
                 vina_script_path = "./vina/runVina.sh"):

        # input path, files
        self.protein_path   = protein_path
        self.ligand_path   = ligand_path
        self.vina_script_path = vina_script_path
                     
        self.protein_path_pdb_files      = glob(f"{protein_path}/*.pdb")
        self.ligand_path_pdb_files      = glob(f"{ligand_path}/*.pdb")
        self.pdb_files      = [os.path.basename(f) for f in self.protein_path_pdb_files]
        self.ligand_files      = [os.path.basename(f) for f in self.ligand_path_pdb_files]

        # output path, files
        self.outpath_fpocket = outpath_fpocket
        self.outpath_vina = outpath_vina
        self.outpath_gvp     = outpath_gvp
        self.outpath_diffdock = outpath_diffdock
        self.outpath_difflinker = outpath_difflinker

        self.outfile_files  = [pdb_file.split('.')[0] +'_out' for pdb_file in self.pdb_files]

        # model
        self.nn_path_gvp     = nn_path_gvp

    @property
    def predict_all_with_vina(self, ):
        try:
            # if not os.path.exists(os.path.join(self.outpath_vina, outfile_name)):
                # Run the command and wait for it to complete
            for pdb in self.protein_path_pdb_files:
                # center = 
                # size =  --center {ast.literal_eval(center)} --size {ast.literal_eval(size)}
                completed_process = subprocess.run([f"{self.vina_script_path}", "-l", f"{self.ligand_path}", "-r", f"{pdb}", "-o", f"{self.outpath_vina}"], check=True, capture_output=True, text=True)
                print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
                print(f"Output: {completed_process.stdout}")
                print(f"Output: {completed_process.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

    def predict_1_with_fpocket(self, protein_path, protein_name, outpath_fpocket, outfile_name):
        try:
            if not os.path.exists(os.path.join(outpath_fpocket, outfile_name)):
                # Run the command and wait for it to complete
                completed_process = subprocess.run(["fpocket", "-f", os.path.join(protein_path, protein_name)], check=True, capture_output=True, text=True)
                print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
                print(f"Output: {completed_process.stdout}")
                print(f"Output: {completed_process.stderr}")
                # Move the output file to the desired location
                shutil.move(os.path.join(protein_path, outfile_name), 
                            os.path.join(outpath_fpocket))

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            
    @property
    def predict_all_with_fpocket(self,): 
        protein_path    = self.protein_path
        outpath_fpocket = self.outpath_fpocket
        pdb_files       = self.pdb_files
        outfile_files   = self.outfile_files
        for protein_name, outfile_name in zip(pdb_files, outfile_files):
            self.predict_1_with_fpocket(protein_path, protein_name, outpath_fpocket, outfile_name)

    @property
    def predict_all_with_gvp(self, debug=False, output_basename=None):
        '''
            protein_path_pdb_files : list of pdb paths
            model : MQAModel corresponding to network in nn_path_gvp
            nn_path_gvp : path to checkpoint files
        '''
        protein_path_pdb_files = self.protein_path_pdb_files
        nn_path_gvp = self.nn_path_gvp
        outpath_gvp = self.outpath_gvp  
        outfile_files = 'concat_gvp_results'

        model = self.mqa_model()

        strucs = [md.load(s) for s in protein_path_pdb_files]
        X, S, mask = process_strucs(strucs)
        if debug:
            output_basename = f'{outpath_gvp}/{outfile_files}'
            np.save(f'{output_basename}_X.npy', X)
            np.save(f'{output_basename}_S.npy', S)
            np.save(f'{output_basename}_mask.npy', mask)
        predictions = predict_on_xtals(model, nn_path_gvp, X, S, mask)
        
        # save predictions
        # output filename can be modified here
        np.save(f'{outpath_gvp}/{outfile_files}-preds.npy', predictions)
        np.savetxt(os.path.join(outpath_gvp,f'{outfile_files}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')
        print('done')

        return predictions


    def mqa_model(self, DROPOUT_RATE = 0.1, NUM_LAYERS = 4, HIDDEN_DIM = 100):
        # Load MQA Model used for selected NN network
        model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                            hidden_dim=(16, HIDDEN_DIM),
                            num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
        return model

    def predict_1_with_diffdock(self, outpath_diffdock, protein_path, protein_name, ligand_path, ligand_name):
        try:
            outfile_name = os.path.splitext(protein_name)[0] + "_" + os.path.splitext(ligand_name)[0]
            # if not os.path.exists(os.path.join(outpath_diffdock, outfile_name)):
            # Run the command and wait for it to complete
            completed_process = subprocess.run([f"git pull && python -m inference --protein_path {os.path.join(protein_path, protein_name)} --ligand_description {os.path.join(ligand_path, ligand_name)} --complex_name {outfile_name} --out_dir {outpath_diffdock} --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise"], shell=True, capture_output=True, text=True)                
            print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
            print(f"Output: {completed_process.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            
    @property
    def predict_all_with_diffdock(self,): 
        protein_path    = self.protein_path
        pdb_files       = self.pdb_files
        ligand_path     = self.ligand_path
        ligand_files    = self.ligand_files
        outfile_files   = self.outfile_files
        outpath_diffdock = self.outpath_diffdock

        # completed_process = subprocess.run(["#!/bin/bash", "pushd", "DiffDock"], text=True, capture_output=True)
        os.chdir("DiffDock")
        outpath_diffdock = os.path.join("..", outpath_diffdock)
        protein_path = os.path.join("..", protein_path)
        ligand_path = os.path.join("..", ligand_path)

        for protein_name, ligand_name in zip(pdb_files, ligand_files):
            self.predict_1_with_diffdock(outpath_diffdock, protein_path, protein_name, ligand_path, ligand_name)
        # completed_process = subprocess.run(["popd"], text=True, capture_output=True)
        os.chdir("..")

    def predict_1_with_difflinker(self, outpath_difflinker, protein_path, protein_name):
        try:
            # if not os.path.exists(os.path.join(outpath_diffdock, outfile_name)):
            # Run the command and wait for it to complete
            completed_process = subprocess.run([f"git pull && python -W ignore generate_with_protein.py --fragments ../DiffLinkerMOAD/processed/MOAD_test_frag.sdf --protein {os.path.join(protein_path, protein_name)} --model models/pocket_difflinker_fullpocket_no_anchors.ckpt --linker_size models/geom_size_gnn.ckpt --output {outpath_difflinker} --n_samples 15 --n_steps 2000"], shell=True, capture_output=True, text=True)                
            print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
            print(f"Output: {completed_process.stdout}")
            print(f"Output: {completed_process.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            
    @property
    def predict_all_with_difflinker(self,): 
        protein_path    = self.protein_path
        pdb_files       = self.pdb_files
        outpath_difflinker = self.outpath_difflinker

        for protein_name in pdb_files:
            self.predict_1_with_diffdock(outpath_difflinker, protein_path, protein_name)
        
if __name__ == '__main__':
    pred = PocketPrediction()
    # pred.predict_all_with_vina
    # pred.predict_all_with_fpocket
    # pred.predict_all_with_gvp
    # pred.predict_all_with_diffdock
    predict_all_with_difflinker

# git pull && python -m inference --protein_path ../data_docking/protein/1ADE.pdb --ligand ../data_docking/ligand/benzene.mol2 --out_dir ../data_docking/result_diffdock --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise
