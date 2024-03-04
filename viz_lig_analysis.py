import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import os
from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
import wandb
import PIL
from typing import *
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
from gen_lig_analysis import Analyse_generation
import pathlib

rdDepictor.SetPreferCoordGen(True)
IPythonConsole.ipython_3d = True

rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument("--turn_off_run_test", "-to", action="store_false", help="for ValTest_mol.sdf!")
args = parser.parse_args()

def findMCS(ms: List[Chem.Mol], qry: Chem.Mol):
    """
      https://greglandrum.github.io/rdkit-blog/posts/2022-06-23-3d-mcs.html
      https://greglandrum.github.io/rdkit-blog/posts/2023-10-27-mcswhatsnew.html
      https://www.blopig.com/blog/2023/06/customising-mcs-mapping-in-rdkit/
    """
    # ms: 5lqf_altB_chainA_3_25/27/55/60/81/82_KLIF_ValTest_frag
  
    ps = rdFMCS.MCSParameters()
    ps.AtomCompareParameters.MaxDistance = 1.0
    ps.AtomTyper = rdFMCS.AtomCompare.CompareAny
    res = rdFMCS.FindMCS(ms, ps)
    qry = Chem.MolFromSmarts(res.smartsString)
    
    matches = [x.GetSubstructMatch(qry) for x in ms] 
    
    conf = Chem.Conformer(qry.GetNumAtoms())
    for i, mi in enumerate(matches[0]):
        conf.SetAtomPosition(i, ms[0].GetConformer().GetAtomPosition(mi))
    qry.AddConformer(conf)
    for m in ms:
        rdDepictor.GenerateDepictionMatching2DStructure(m, qry) # constrained coord to qry

    pathlib.Path("data_docking/result_images").mkdir(exist_ok=True)
    img = Draw.MolsToGridImage(ms, highlightAtomLists=matches, molsPerRow=3, subImgSize=(200,200), legends=[x.GetProp("_Name") for x in ms)    
    img.save('data_docking/result_images/cdk2_molgrid.png')    

def plot_properties(args: argparse.ArgumentParser):
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
    plot_properties(args)
    root = "data_docking/result_hydrogenated"
    test_ms = [Chem.SDMolSupplier(os.path.join(root, f"5lqf_altB_chainA_3_{num}_KLIF_ValTest_frag.sdf"), removeHs=False, sanitize=False)[0] for num in [25, 27, 55, 60, 81, 82] ]
    query = Chem.SDMolSupplier(os.path.join(root, f"5lqf_altB_chainA_3_GT_KLIF_ValTest_frag.sdf"), removeHs=False, sanitize=False)[0]
    findMCS(test_ms, query)
