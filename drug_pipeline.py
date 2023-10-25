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
import pathlib
import ray 

import MDAnalysis as mda
# https://zenodo.org/records/4740366 ;; Natural ligands

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
                 vina_script_path = "./vina/runVina.sh",
                 complex_path = "data_docking/complex",
                 processed_path = "data_docking/processed_complex",
                 vina = None,
                 autodock_python = None,
                 autodock_tools_path = None):

        # input path, files
        self.protein_path   = protein_path
        self.ligand_path   = ligand_path
        self.vina_script_path = vina_script_path
        self.vina = vina
        self.autodock_python = autodock_python
        self.autodock_tools_path = autodock_tools_path
                     
        self.protein_path_pdb_files      = glob(f"{protein_path}/*.pdb")
        self.protein_path_pdb_beta_files      = glob(f"{outpath_gvp}/*.pdb")

        self.ligand_path_pdb_files      = glob(f"{ligand_path}/*.pdb")
        self.pdb_files      = [os.path.basename(f) for f in self.protein_path_pdb_files]
        self.ligand_files      = [os.path.basename(f) for f in self.ligand_path_pdb_files]
        self.pdb_beta_files = [os.path.basename(f) for f in self.protein_path_pdb_beta_files]
                     
        # output path, files
        self.outpath_fpocket = outpath_fpocket
        self.outpath_vina = outpath_vina
        self.outpath_gvp     = outpath_gvp
        self.outpath_diffdock = outpath_diffdock
        self.outpath_difflinker = outpath_difflinker
        self.complex_path = complex_path
        self.processed_path = processed_path

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

    @property
    def convert_gvp_output_to_universe(self, ):
        protein_path_pdb_files = self.protein_path_pdb_files
        outpath_gvp = self.outpath_gvp  
        outfile_files = 'concat_gvp_results'
        scores = np.load(f'{outpath_gvp}/{outfile_files}-preds.npy')
        assert len(protein_path_pdb_files) == scores.shape[0], "Number of PDBs and scores must match!"

        for pdb_file, beta in zip(protein_path_pdb_files, scores):
            pdb = mda.Universe(pdb_file)
            natoms_per_residue = np.array([res.atoms.__len__() for res in pdb.residues])
            beta = beta[beta > 0.]
            betas_for_atoms = np.repeat(beta, natoms_per_residue)
            pdb.add_TopologyAttr('tempfactors', betas_for_atoms)
            _, basename = os.path.dirname(pdb_file), os.path.basename(pdb_file)
            basename, extension = os.path.splitext(basename)
            basename += "_beta"
            basename += extension
            pdb.atoms.write(os.path.join(outpath_gvp, basename))

    @property
    def extract_universe_betas_for_vinadock(self, ):
        outpath_gvp = self.outpath_gvp  
        protein_path_beta_pdb_files = glob(f"{outpath_gvp}/*.pdb")
        vina, autodock_python, autodock_tools_path = self.vina, self.autodock_python, self.autodock_tools_path
        
        try:
            print(protein_path_beta_pdb_files)
            for pdb_file in protein_path_beta_pdb_files:
                pdb = mda.Universe(pdb_file)
                threshold = np.quantile(pdb.atoms.tempfactors, 0.95)
                hit_atoms = np.where(pdb.atoms.tempfactors > threshold)[0]
                hit_atoms = " ".join(hit_atoms.astype(str).tolist()) #https://gitlab.com/-/ide/project/hyunp2/protTransVAE/edit/main/-/analysis.py
                ag = pdb.select_atoms(f"index {hit_atoms}")
                center_coords = ag.center_of_mass()
                xyz_size = np.array([15, 15, 15])
                
                center = f"{center_coords[0]:3.3f} {center_coords[1]:3.3f} {center_coords[2]:3.3f}"
                size = f"{xyz_size[0]:3.3f} {xyz_size[1]:3.3f} {xyz_size[2]:3.3f}"
    
                basename = os.path.basename(pdb_file)
                pdb_name = basename.split("_")[0] #Cuz we are reading with _beta.pdb suffix
                print(center, size)
                # size =  --center {ast.literal_eval(center)} --size {ast.literal_eval(size)}
                if (autodock_tools_path is None) and (autodock_python is None) and (vina is None):
                    completed_process = subprocess.run([f"{self.vina_script_path} -l {self.ligand_path} -r {pdb_file} --center {center} --size {size} -o {self.outpath_vina}"], shell=True, check=True, capture_output=True, text=True)
                else:
                    completed_process = subprocess.run([f"{self.vina_script_path} -l {self.ligand_path} -r {pdb_file} --center {center} --size {size} -o {self.outpath_vina} -v {vina} -py {autodock_python} -t {autodock_tools_path}"], shell=True, check=True, capture_output=True, text=True)
                print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
                print(f"Output: {completed_process.stdout}")
                print(f"Output: {completed_process.stderr}")
        except Exception as e:
            print(e)
            
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
        pdb_beta_files  = self.pdb_beta_files
        outpath_gvp     = self.outpath_gvp  
        ligand_path     = self.ligand_path
        ligand_files    = self.ligand_files
        outfile_files   = self.outfile_files
        outpath_diffdock = self.outpath_diffdock

        # completed_process = subprocess.run(["#!/bin/bash", "pushd", "DiffDock"], text=True, capture_output=True)
        os.chdir("DiffDock")
        outpath_diffdock = os.path.join("..", outpath_diffdock)
        outpath_gvp = os.path.join("..", outpath_gvp)
        ligand_path = os.path.join("..", ligand_path)

        for protein_name, ligand_name in zip(pdb_beta_files, ligand_files):
            self.predict_1_with_diffdock(outpath_diffdock, outpath_gvp, protein_name, ligand_path, ligand_name)
        # completed_process = subprocess.run(["popd"], text=True, capture_output=True)
        os.chdir("..")

    def make_1_prot_ligand_complex_for_difflinker(self, protein_path, protein_name, ligand_path, ligand_name, complex_path):
        #For vina!
        pathlib.Path(os.path.join(ligand_path, os.path.splitext(ligand_name)[0])).mkdir(exist_ok=True)
        os.system(f"obabel -ipdbqt {os.path.join(ligand_path, ligand_name)} -opdb -O {os.path.join(ligand_path, os.path.splitext(ligand_name)[0], ligand_name.replace('.pdbqt', '.pdb'))} -m") 
        top1_lig = os.path.join(os.path.join(ligand_path, os.path.splitext(ligand_name)[0]), sorted(os.listdir(f"{os.path.join(ligand_path, os.path.splitext(ligand_name)[0])}"), key=str)[0])

        l = mda.Universe(top1_lig)
        l.add_TopologyAttr("record_type", ["HETATM"] * l.atoms.__len__())
        r = mda.Universe(os.path.join(protein_path, protein_name))
        r.add_TopologyAttr("record_type", ["ATOM"] * r.atoms.__len__())
        c = mda.Merge(l.atoms, r.atoms)
        pathlib.Path(complex_path).mkdir(exist_ok=True)
        c.atoms.write(os.path.join(complex_path, os.path.splitext(protein_name)[0] + "_" + os.path.splitext(ligand_name)[0] + ".pdb"))

    def process_prot_ligand_complex_for_difflinker(self, complex_path, processed_path):
        from data.pocket.clean_and_split import run, process_one_file, process_one_file_noray
        pathlib.Path(processed_path).mkdir(exist_ok=True)

        proteins_dir = os.path.join(processed_path, "proteins")
        ligands_dir = os.path.join(processed_path, "ligands")
        fnames = run(input_dir=complex_path, proteins_dir=proteins_dir, ligands_dir=ligands_dir)
        print(fnames)
        [process_one_file_noray(complex_path, proteins_dir, ligands_dir, fname) for fname in fnames]
        
        # input_dir, proteins_dir, ligands_dir = ray.put(complex_path), ray.put(proteins_dir), ray.put(ligands_dir)
        # results = [process_one_file.remote(input_dir, proteins_dir, ligands_dir, fname) for fname in fnames]
        # results = ray.get(results)
        # ray.shutdown()

    def generate_fragmentation_for_difflinker(self, processed_path):
        from data.pocket.generate_fragmentation_and_conformers import run
        pathlib.Path(processed_path).mkdir(exist_ok=True)
        
        ligands_dir = os.path.join(processed_path, "ligands")
        out_fragmentations = os.path.join(processed_path, "generated_splits.csv")
        out_conformers = os.path.join(processed_path, "generated_conformers.sdf")

        run(
            ligands_dir=ligands_dir,
            output_table=out_fragmentations,
            output_conformers=out_conformers)

    def prepare_dataset_for_difflinker(self, processed_path):
        from data.pocket.prepare_dataset import run
        pathlib.Path(processed_path).mkdir(exist_ok=True)
        
        proteins_dir = os.path.join(processed_path, "proteins")
        ligands_dir = os.path.join(processed_path, "ligands")
        out_fragmentations = os.path.join(processed_path, "generated_splits.csv")
        out_conformers = os.path.join(processed_path, "generated_conformers.sdf")
        out_mol_sdf = os.path.join(processed_path, "Custom_mol.sdf")
        out_frag_sdf = os.path.join(processed_path, "Custom_frag.sdf")
        out_link_sdf = os.path.join(processed_path, "Custom_linker.sdf")
        out_pockets_pkl = os.path.join(processed_path, "Custom_pockets.pkl")
        out_table = os.path.join(processed_path, "Custom_table.csv")
        
        run(
            table_path=out_fragmentations,
            sdf_path=out_conformers,
            proteins_path=proteins_dir,
            out_mol_path=out_mol_sdf,
            out_frag_path=out_frag_sdf,
            out_link_path=out_link_sdf,
            out_pockets_path=out_pockets_pkl,
            out_table_path=out_table)
    
    def predict_1_with_difflinker(self, outpath_difflinker, protein_path, protein_name):
        try:
            # if not os.path.exists(os.path.join(outpath_diffdock, outfile_name)):
            # Run the command and wait for it to complete   
            print(protein_path, protein_name, outpath_difflinker)
            # completed_process = subprocess.run([f"git pull && python -W ignore generate_with_protein.py --fragments ../DiffLinkerMOAD/processed/MOAD_test_frag.sdf  --protein {os.path.join(protein_path, protein_name)} --model models/pocket_difflinker_fullpocket_no_anchors.ckpt --linker_size models/geom_size_gnn.ckpt --output {outpath_difflinker} --n_samples 1 --n_steps 2000"], shell=True, capture_output=True, text=True)                
            completed_process = subprocess.run([f"git pull && python -W ignore generate_with_protein.py --fragments data_docking/processed_complex/Custom_frag.sdf  --protein {os.path.join(protein_path, protein_name)} --model models/pocket_difflinker_fullpocket_no_anchors.ckpt --linker_size models/geom_size_gnn.ckpt --output {outpath_difflinker} --n_samples 1 --n_steps 2000"], shell=True, capture_output=True, text=True)                

            print(f"Return code: {completed_process.returncode}") #an exit status of 0 indicates that it ran successfully
            print(f"Output: {completed_process.stdout}")
            print(f"Output: {completed_process.stderr}")

        # except subprocess.CalledProcessError as e:
        except ValueError as e:
            print(f"An error occurred: {e}")
            print(f"This may happen because ligand and protein pocket distance is too far!")

    @property
    def predict_all_with_difflinker(self,): 
        protein_path    = self.protein_path
        pdb_files       = self.pdb_files
        outpath_difflinker = self.outpath_difflinker

        for protein_name in pdb_files:
            self.predict_1_with_difflinker(outpath_difflinker, protein_path, protein_name)

    
        
if __name__ == '__main__':
    pred = PocketPrediction()
    pred.predict_all_with_vina
    # pred.predict_all_with_fpocket
    # pred.predict_all_with_gvp
    # pred.predict_all_with_diffdock
    # pred.predict_all_with_difflinker
    # pred.convert_gvp_output_to_universe
    # pred.extract_universe_betas_for_vinadock
    # pred.make_1_prot_ligand_complex_for_difflinker(pred.protein_path, "1ADE.pdb", os.path.join(pred.outpath_vina, "1ADE_beta/vina"), "oxo.pdbqt", pred.complex_path)
    # pred.process_prot_ligand_complex_for_difflinker(pred.complex_path, pred.processed_path)
    # pred.generate_fragmentation_for_difflinker(pred.processed_path)
    # pred.prepare_dataset_for_difflinker(pred.processed_path)
    # pred.predict_1_with_difflinker(pred.outpath_difflinker, os.path.join(pred.processed_path, "proteins"), "1ADE_oxo_protein.pdb")
    # pred.predict_1_with_difflinker(pred.outpath_difflinker, pred.protein_path, "1ADE.pdb")

# git pull && python -m inference --protein_path ../data_docking/protein/1ADE.pdb --ligand ../data_docking/ligand/benzene.mol2 --out_dir ../data_docking/result_diffdock --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise
