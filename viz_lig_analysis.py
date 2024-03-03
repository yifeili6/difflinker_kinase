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
