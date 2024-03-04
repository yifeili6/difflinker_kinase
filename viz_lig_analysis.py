import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import os
from torch_geometric.data import DataLoader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
import wandb
import PIL
from typing import *


def img_for_mol(mol, atom_weights=[], bond_weights: Union[None, List]=[], start_idx: int=0, edge_index: torch.LongTensor=None, use_custom_draw: bool=True, new_edge_index: torch.LongTensor=None):
  """
    https://gitlab.com/hyunp2/argonne_gnn_gitlab/-/blob/main/train/explainer.py?ref_type=heads
  """
    highlight_kwargs = {}
    
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        } #DICT: {atm_num: color_val}
        
        
        bonds = []
        for bond in mol.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bonds_zip = list(zip(*bonds)) #2, Edges
        bonds_zip = np.array(bonds_zip).T #edges, 2
        # print(bonds_zip, edge_index.t())
        
        if new_edge_index != None:
            # print(bonds_zip, new_edge_index)
            bw_list = []
            for bond in bonds_zip:
                idx_of_geomedge = (bond == new_edge_index.t().detach().cpu().numpy()).all(axis=-1).tolist().index(True)
                w = bond_weights[idx_of_geomedge]  
                bw_list.append(w) #num_real_bonds
            bond_colors = {
                i: plt_colors.to_rgba(bw_list[i]) for i in range(len(bw_list))
            }
        else:
            bw_list = []
            bond_colors = {}
                # all_bonds.append( (bond == new_edge_index).all(axis=-1) ) 
            # all_bonds = np.array(all_bonds).T.any(axis=-1) #(real_bonds, torch_geom_bonds) - > (torch_geom_bonds, real_bonds) -> Boolean array of (torch_geom_bonds, )


        """
        # np.isin(edge_index.detach().cpu().numpy().T, bonds_zip)

        if not isinstance(bond_weights, type(None)):
            bond_colors = {
                i: plt_colors.to_rgba(bond_weights[i]) for i in range(len(bond_weights))
            }
        else:
            bond_weights = []
            bond_colors = {}
        """
        if new_edge_index != None:
            highlight_kwargs = {
                'highlightAtoms': [],
                'highlightAtomColors': {},
                'highlightBonds': list(range(len(bw_list))),
                'highlightBondColors': bond_colors
            }
        else:
            highlight_kwargs = {
                'highlightAtoms': list(range(len(atom_weights))),
                'highlightAtomColors': atom_colors,
                'highlightBonds': list(range(len(bw_list))),
                'highlightBondColors': bond_colors
            }

        # print(highlight_kwargs)
    # print(bond_weights, mol.GetNumBonds())
    

    #########################
    #METHOD 1 for DRAWING MOL (DrawMolecule)
    #########################
    if use_custom_draw:
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
        drawer.drawOptions().addStereoAnnotation=True
        drawer.atomHighlightsAreCircles = True
        drawer.fillHighlights=False
        drawer.SetFontSize(1)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol, **highlight_kwargs)
                            # highlightAtoms=list(range(len(atom_weights))),
                            # highlightBonds=[],
                            # highlightAtomColors=atom_colors)
        # PrepareAndDrawMolecule #https://www.rdkit.org/docs/GettingStartedInPython.html?highlight=maccs#:~:text=%2C%20500)-,%3E%3E%3E%20rdMolDraw2D.PrepareAndDrawMolecule(d%2C%20mol%2C%20highlightAtoms%3Dhit_ats%2C,-...%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20highlightAtomColors%3D
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('svg:', '')
        svg2png(bytestring=svg, write_to='tmp.png', dpi=100)
        img = imread('tmp.png')
        os.remove('tmp.png')


if __name__ == "__main__":
    ###Current as of Mar 1st, 2024
    if args.run_analysis:
        if not args.turn_off_run_test:
            print(cf.on_yellow(f"Running Analysis on GENERATED data with {args.size_prefix}!!!!"))
            ###3D
            gen, files, file_counter = Analyse_generation.get_posebuster_stats([args.size_prefix]) # filtration 1
            if len(gen) !=0:
                pass
            else:
                gen = args.gen
            ###Drugness
            gen, files = Analyse_generation.get_lipinski(gen, files, file_counter, args.size_prefix) #filtration 2
            ###2D
            gen, files = Analyse_generation.get_moses_stats(gen=gen, files=files, train=args.train, test=args.valtest, test_scaffolds=args.valtest, file_counter_from_posebuster=file_counter, size_prefix=args.size_prefix) # final filtration
            # print(files)
            # gen = ["c1ccccc1", "c1cnccc1", "C1CCCCC1", "C1CNCCC1", "CCCCCC", "CCNCCC", "c12ccccc1NC=C2", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "Cc1c(Nc2nccc(-c3cnccc3)n2)cc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1"]
            # files = np.arange(len(gen))
            rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings = bonds_and_rings(gen, files, args.size_prefix)
            # for prop in [rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings]:
            #     print(prop)
        else:
            print(cf.on_green(f"Running Analysis on TEST data!!!!"))
            ###3D
            gen, file_counter = Analyse_generation.get_posebuster_stats_for_test() # filtration 1
            ###Drugness
            gen = Analyse_generation.get_lipinski_for_test(gen, file_counter) #filtration 2
            ###2D
            gen = Analyse_generation.get_moses_stats_for_test(gen=gen,  train=args.train, test=args.valtest, test_scaffolds=args.valtest, file_counter_from_posebuster=file_counter) # final filtration
            rot_bonds, num_rings, num_fused_rings, num_hetero_rings, num_aromatic_rings = Analyse_generation.bonds_and_rings_for_test(gen)
            
    else:
        if not args.turn_off_run_test:
            print(cf.on_blue(f"Concatenating GENERATED data statistics!!!!"))
            DF, DF_rings_dist = Analyse_generation.collate_fn()
            print(DF.loc[:, ["IntDiv", "IntDiv2"]])
            print(DF)
            print(DF_rings_dist.groupby("size").mean())
            DF = Analyse_generation.get_non_wass_stats()
            print(DF)
        else:
            print(cf.on_red(f"Concatenating TEST data statistics!!!!"))
            DF, DF_rings_dist = Analyse_generation.collate_fn_for_test()
            print(DF.loc[:, ["IntDiv", "IntDiv2"]])
            print(DF)
            print(DF_rings_dist.mean(axis=0))
            DF = Analyse_generation.get_non_wass_stats_for_test()
            print(DF)
