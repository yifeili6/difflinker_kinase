import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import curtsies.fmtfuncs as cf
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import os, shutil
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
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from gen_lig_analysis import Analyse_generation
from moses.metrics.utils import QED, SA
import pathlib
import copy
import tempfile
import warnings

warnings.simplefilter("ignore")

rdDepictor.SetPreferCoordGen(True)
IPythonConsole.ipython_3d = True

rdBase.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument("--turn_off_run_test", "-to", action="store_false", help="for ValTest_mol.sdf!")
args = parser.parse_args()

def edit_ligand(ligand, num_frag_atoms: int=16):
    atoms = ligand.GetAtoms()
    bonds = ligand.GetBonds()
    eligand = Chem.RWMol(copy.deepcopy(ligand))

    for bond in bonds:
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        if max(source_idx, target_idx) < num_frag_atoms:
            eligand.RemoveBond(source_idx, target_idx)
    
    for atom in atoms:
        if atom.GetIdx() < num_frag_atoms:
            eligand.RemoveAtom(atom.GetIdx())
    
    return eligand.GetMol()

def findMCS(ms: List[Chem.Mol]):
    """
      https://greglandrum.github.io/rdkit-blog/posts/2022-06-23-3d-mcs.html
      https://greglandrum.github.io/rdkit-blog/posts/2023-10-27-mcswhatsnew.html
      https://www.blopig.com/blog/2023/06/customising-mcs-mapping-in-rdkit/
    """
    # ms: 5lqf_altB_chainA_3_25/27/55/60/81/82_KLIF_ValTest_frag
  
    ps = rdFMCS.MCSParameters()
    # ps.AtomCompareParameters.MaxDistance = 1.0
    # ps.AtomTyper = rdFMCS.AtomCompare.CompareAny
    res = rdFMCS.FindMCS(ms, ps)
    qry = Chem.MolFromSmarts(res.smartsString)
    
    matches = [x.GetSubstructMatch(qry) for x in ms] 
    
    conf = Chem.Conformer(qry.GetNumAtoms())
    for i, mi in enumerate(matches[0]):
        conf.SetAtomPosition(i, ms[0].GetConformer().GetAtomPosition(mi))
    qry.AddConformer(conf)
    for m in ms:
        rdDepictor.GenerateDepictionMatching2DStructure(m, qry) # constrained coord to qry!

    pathlib.Path("data_docking/result_images").mkdir(exist_ok=True)
    img = Draw.MolsToGridImage(ms, highlightAtomLists=matches, molsPerRow=3, subImgSize=(200,200), legends=[x.GetProp("_Name") for x in ms], returnPNG=False)    
    if os.path.isfile('data_docking/result_images/cdk2_molgrid.png'):
        pathlib.Path('data_docking/result_images/cdk2_molgrid.png').unlink()
        img.save('data_docking/result_images/cdk2_molgrid.png')    
    else:
        img.save('data_docking/result_images/cdk2_molgrid.png')    

def merge_properties(args: argparse.ArgumentParser):
    if not args.turn_off_run_test:
        print(cf.on_blue(f"Concatenating GENERATED data statistics!!!!"))
        DF0, DF_rings_dist = Analyse_generation.collate_fn()
        DF = Analyse_generation.get_non_wass_stats()
        DF1 = pd.concat([DF_rings_dist.drop(columns="size"), DF], axis=1)
    else:
        print(cf.on_red(f"Concatenating TEST data statistics!!!!"))
        DF0, DF_rings_dist = Analyse_generation.collate_fn_for_test()
        DF = Analyse_generation.get_non_wass_stats()
        DF1 = pd.concat([DF_rings_dist.drop(columns="size"), DF], axis=1)
    return DF0, DF1

def remove_one_atom_qed(mol: Chem.Mol, query_num_atoms: int) -> List[Chem.Mol]:
    # assert remove_idx >= query_num_atoms, "remove_idx must be equal or larger than query_num_atoms; index starts from 0"
    num_atoms = mol.GetNumAtoms()
    original_qed = QED(mol)
    contribs = []
    
    for idx in range(query_num_atoms, num_atoms, 1):
        eligand = Chem.RWMol(copy.deepcopy(mol))
        atom = eligand.GetAtomWithIdx(idx)
        neigh = [x.GetIdx() for x in atom.GetNeighbors()]
        [eligand.RemoveBond(idx, x) for x in neigh]
        eligand.RemoveAtom(idx)
        # print(idx)
        try:
            qed = QED(eligand.GetMol())
        except:
            qed = 0
        qed_diff = qed - original_qed
        contribs.append(qed_diff)
    contribs = [0] * query_num_atoms + contribs
    
    return contribs

def remove_one_atom_sa(mol: Chem.Mol, query_num_atoms: int) -> List[Chem.Mol]:
    # assert remove_idx >= query_num_atoms, "remove_idx must be equal or larger than query_num_atoms; index starts from 0"
    num_atoms = mol.GetNumAtoms()
    original_qed = SA(mol)
    contribs = []
    print(original_qed)
    for idx in range(query_num_atoms, num_atoms, 1):
        eligand = Chem.RWMol(copy.deepcopy(mol))
        atom = eligand.GetAtomWithIdx(idx)
        neigh = [x.GetIdx() for x in atom.GetNeighbors()]
        [eligand.RemoveBond(idx, x) for x in neigh]
        eligand.RemoveAtom(idx)
        # print(idx)
        try:
            qed = SA(eligand.GetMol())
        except:
            qed = 10
        qed_diff = qed - original_qed
        print(qed_diff)
        contribs.append(qed_diff)
    contribs = [0] * query_num_atoms + contribs
    
    return contribs

def img_for_mol(mol: Chem.Mol, qry: Chem.Mol, query_num_atoms: int=None, contribution: str= "atomic", atom_weights=[], bond_weights: Union[None, List]=[], start_idx: int=0, edge_index: torch.LongTensor=None, use_custom_draw: bool=False, new_edge_index: torch.LongTensor=None):
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
        # PrepareAndDrawMolecule #https://www.rdkit.org/docs/GettingStartedInPython.html?highlight=maccs#:~:text=%2C%20500)-,%3E%3E%3E%20rdMolDraw2D.PrepareAndDrawMolecule(d%2C%20mol%2C%20highlightAtoms%3Dhit_ats%2C,-...%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20highlightAtomColors%3D
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('svg:', '')
        with tempfile.TemporaryFile() as fp:
            svg2png(bytestring=svg, write_to=fp.name, dpi=100)
            img = imread(fp.name)
    else:
        import io
        from rdkit.Chem.Draw import SimilarityMaps
        # drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
        
        drawer = Draw.MolDraw2DCairo(280, 280)

        
        assert contribution in ["atomic", "lipinski", "qed", "sa"], "Not a correct keyword!"
        if contribution == "atomic":
            atom_weights = SimilarityMaps.GetAtomicWeightsForFingerprint(qry, mol, SimilarityMaps.GetMorganFingerprint)
        elif contribution == "qed":
            atom_weights = remove_one_atom_qed(mol, query_num_atoms)
        # elif contribution == "sa":
        #     atom_weights = remove_one_atom_sa(mol, qry_numa)
            
        atom_weights = np.array(atom_weights)
        atom_weights = atom_weights / np.sum(atom_weights**2)**(0.5) #normalize
        atom_weights[:query_num_atoms] = 0
        atom_weights = atom_weights.tolist()
        img = SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=None) #http://rdkit.blogspot.com/2020/01/similarity-maps-with-new-drawing-code.html#:~:text=SimilarityMaps.GetSimilarityMapFromWeights(atorvastatin%2Clist(mean_chgs)%2Cdraw2d%3Dd)

        with tempfile.TemporaryDirectory() as fp:
            # print(fp.name)
            img.savefig(os.path.join(fp, "img.png"), bbox_inches='tight')
            img = imread(os.path.join(fp, "img.png"))
        drawer.FinishDrawing()
    return img

def plot_similarity_maps(ms: List[Chem.Mol], qry: Chem.Mol, query_num_atoms: int, contribution: str="atomic"):
    """
        ms: list of generated mols
        qry: gt mol
        query_num_atoms: number of atoms in fragments
        contribution: atomic or qed
    """
    fig, axes = plt.subplots(len(ms)//3, 3, figsize=(10,7))
    imgs = [img_for_mol(m , qry, query_num_atoms=query_num_atoms, contribution=contribution) for m in ms]
    [ax.imshow(img) for ax, img in zip(axes.flatten(), imgs)]
    
    filename = f"{contribution.lower()}_molgrid.png" 
    fig.tight_layout()
    if os.path.isfile(f'data_docking/result_images/{filename}'):
        pathlib.Path(f'data_docking/result_images/{filename}').unlink()
        fig.savefig(f'data_docking/result_images/{filename}')    
    else:
        fig.savefig(f'data_docking/result_images/{filename}')    

def plot_maps():
    root_h = "data_docking/result_hydrogenated" #for hydrogenated dir; both sdf and PDB (for ligprot)
    root_d = "datasets" #for GT dir
    files = os.listdir(root_h) #Manually saved GT/GenAI hydrogenated ligands!
    files_base = [os.path.basename(one_file) for one_file in files]
    files_set = set()
    
    for f in files_base:
        header = "_".join(f.split("_")[:3]) if "alt" in f else "_".join(f.split("_")[:2])
        files_set.add(header)
    files_set = list(files_set) #list of unique file headers

    for file_header in files_set:
        #for each unique file_header from the files_set, get the sdf files (ABS dir)
        files_selected = list(map(lambda inp: os.path.join(root_h, inp), list(filter(lambda inp: inp.startswith(file_header) & inp.endswith(".sdf"), files_base )) ))
        files_selected.sort() #inplace

        # test_ms = [Chem.SDMolSupplier(os.path.join(root_h, f"5lqf_altB_chainA_3_{num}_KLIF_ValTest_frag.sdf"), removeHs=True, sanitize=True)[0] for num in [25, 27, 55, 60, 81, 82] ]
        test_ms = [Chem.SDMolSupplier(one_file, removeHs=True, sanitize=True)[0] for one_file in files_selected if not "GT" in os.path.basename(one_file) ]
        test_ms = [Chem.RemoveHs(m) for m in test_ms]
        
        query = Chem.SDMolSupplier(files_selected[-1], removeHs=True, sanitize=True)[0] #GT is last in sorted order
        query = Chem.RemoveHs(query)

        df = pd.read_csv(os.path.join(root_d, "KLIF_test_table.csv")).molecule_name
        idx = int(df.index[df.apply(lambda inp: inp.startswith(file_header))][0]) #fine a SMILES molecule name with this prefix so that we can choose corresponding fragment
        # print(idx)
        qry_numa = Chem.SDMolSupplier(os.path.join(root_d, f"KLIF_test_frag.sdf"), removeHs=True, sanitize=True)[idx].GetNumAtoms()
        print(qry_numa)
        for contribution in ["qed", "atomic"]:
            plot_similarity_maps(test_ms, query, query_num_atoms=qry_numa, contribution=contribution)

def plot_by_group(df: pd.DataFrame):
    #https://github.com/hyunp2/ghp_mof/blob/main/analyze_linkers.py
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # distribution of SCscore and SAscore
    sns.set_theme()
    sns.set_style("white")
    kwargs = dict(bins=50, stacked=True)
    
    df.drop_duplicates(subset='SMILES', keep='first', inplace=True)
    df.rename(columns={"rot_bonds": "num_rot_bonds"}, inplace=True)
    metric_name = list(df.columns)
    metric_name.remove("SMILES")
    metric_name.remove("size")
    metric_name.remove("files")
    
    index = 0
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    palette = sns.color_palette('pastel', 6)
    
    for i, metric in enumerate(metric_name):
        row_num = int(index / 3)
        col_num = index % 3
        for idx, n_atoms in enumerate(range(8, 14, 1)):
            if metric.startswith("num_"):
                data = df.loc[df.index[df.loc[:, "size"].apply(lambda inp: inp == n_atoms)]]
                ax[row_num][col_num].spines[['left','right', 'bottom']].set_visible(False)
                # ax[row_num][col_num].hist(data.loc[:, metric].values.reshape(-1, ), **kwargs, label=n_atoms)
                sns.barplot(x="size", y=metric,
                            data=data, ax=ax[row_num][col_num], palette=[palette[idx]], label=n_atoms)
            else:
                data = df.loc[df.index[df.loc[:, "size"].apply(lambda inp: inp == n_atoms)]]
                ax[row_num][col_num].spines[['left','right', 'bottom']].set_visible(False)
                # sns.kdeplot(data=data.loc[:, metric].values.reshape(-1, ), label=n_atoms, ax=ax[row_num][col_num])
                # Draw a nested boxplot to show bills by day and time
                sns.violinplot(x="size", y=metric,
                            data=data, ax=ax[row_num][col_num], palette=[palette[idx]], label=n_atoms)
        ax[row_num][col_num].set_xlabel(metric)
        ax[row_num][col_num].set_ylabel('Count')
        ax[row_num][col_num].set_title(f"{metric} distribution", weight='bold', fontsize=13.5)  
        index+=1
    handles, labels = ax[row_num][col_num].get_legend_handles_labels()
    
    fig.subplots_adjust(top=0.8)
    fig.suptitle("5LQF AltB ChainA", y=1.2, fontsize=15)
    fig.tight_layout()
    lgd = fig.legend(handles, labels, loc='lower center', ncol=9, bbox_to_anchor=(0.5, -0.05), framealpha=0, edgecolor='gray')
    fig.savefig(f'./test_.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    print(f'Plotted distribution of metrics')
    
if __name__ == "__main__":
    ###Current as of Mar 1st, 2024
    DF0, DF1 = merge_properties(args)
    print(DF0, DF1)
    print(DF1.columns)
    DF0.to_csv("latex_table.csv")
    plot_by_group(DF1)
    # plot_maps()
    
    ####WARNING####
    ####DO Manually,
    #Choose only molecule_name ending with ONLY '_0' in KLIF_test_frag.sdf file
    #b.c. we redefined the _0/1/2 from num of ligands to num of fragmentation types
    


    # test_ms = [edit_ligand(m) for m in test_ms]
    # print(Chem.MolToSmiles(qry).split("."))
    # qry = [Chem.MolFromSmarts(q) for q in Chem.MolToSmiles(qry).split(".")]
    # matches = [x.GetSubstructMatch(qry) for x in test_ms] 
    # print(matches)
        
    # am = {}
    # for test_m in test_ms:
    #     am[test_m.GetProp("_Name")] = {}
    #     for idx, atom in enumerate(test_m.GetAtoms()):
    #         # print(atom.GetIdx(), atom.GetAtomMapNum())
    #         atom.SetAtomMapNum(idx)
    #         am[test_m.GetProp("_Name")][atom.GetAtomMapNum()] = (atom.GetIdx(), atom.GetAtomMapNum(), atom.GetSymbol())
    # print(am)
        
    # findMCS(test_ms)
