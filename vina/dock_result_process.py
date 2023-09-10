import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys

def convert_docking_output_files_to_pdb(path):
    os.system("pushd {} && obabel -ipdbqt *.pdbqt -opdb -m && popd".format(path))

def convertUniToDict(uni,newNames = [],add_idx = True):
    allCoords = dict()
    atom_types = dict()
    for i,atom in enumerate(uni.atoms):
        if not newNames:
            name = atom.name
        elif newNames[i].startswith(atom.name[0]):
            name = newNames[i]
        else:
            print('Atom names improperly mapped.')
            sys.exit(1)
        if add_idx:
            if name not in atom_types:
                atom_types[name] = 1
            name_type = name
            if not name.endswith('*') and not name == 'OXT' and not name[-1].isdigit():
                name = name + str(atom_types[name])
            atom_types[name_type] += 1
        coord = atom.position
        allCoords[name] = coord
    return allCoords

def modifyPDBFiles(filePath):
    #Use openbabel to convert autodock .out files from pdbqt to pdb format before using this function.
    #Keep the .out and the .pdb files in the same directory under the same name except for the file extension.
    #This function is also used to modify dock6 files that have been converted from mol2 to pdb format for
    #input into MDAnalysis.Universe
    with open(filePath, "r") as f:
        fileLines = f.readlines()
    fileToWrite = open(filePath, "w")
    for line in fileLines:
        if not line.startswith("MODEL") and not line.startswith("CONECT") and not line.startswith("END"):
            fileToWrite.write(line)
    fileToWrite.close()
    return

def make_names_unique(names):
    name_types = dict()
    unique_names = []
    for name in names:
        if name not in name_types:
            name_types[name] = 1
        name_type = name
        if not name.endswith('*') and not name == 'OXT' and not name[-1].isdigit():
            name = name + str(name_types[name])
        unique_names.append(name)
        name_types[name_type] += 1
    return unique_names

def extractAutodockScores(path_to_out_file):
    #This function gets scores from autodock output files.
    scores_list = []
    with open(path_to_out_file) as f:
        file_lines = f.readlines()
    for line in file_lines:
        if line.split()[0] == "REMARK" and line.split()[1] == "VINA":
            scores_list.append(line.split()[3])
    return scores_list

def appendVinaDataToDataframe(vina_path: "./1v74/vina/"):
    dockingTool = "vina"
    df_list = []
    for file in sorted(os.listdir(vina_path)):
        if file.endswith(".pdbqt"):
            convert_docking_output_files_to_pdb(vina_path)
            # scores = extractAutodockScores(os.path.join(vina_path, file))
            modifyPDBFiles(os.path.join(vina_path, file[:-6] + ".pdb"))

if __name__ == "__main__":
    appendVinaDataToDataframe(vina_path="vina/1v74/vina")
