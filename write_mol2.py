import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem

# http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf ##see for more explanations for SYBYL!
# https://github.com/jensengroup/xyz2mol/blob/master/xyz2mol.py ## see for better conversion!

############
#####WIP####
############

def filter_mol2_atoms(at: "MDAnalysis atom name", atom: "RDKIT mol atom instance"):
    if at in ["C", "N", "O", "S", "P"]:
        if atom.GetHybridization().name.lower() == "sp3":
            return ".3"
        elif atom.GetHybridization().name.lower() == "sp2":
            return ".2"
        elif atom.GetHybridization().name.lower() == "sp":
            return ".1"
        elif atom.GetIsAromatic():
            return ".ar"
    else:
        return ""

def filter_mol2_bonds(bo: "MDAnalysis order"):
    if bo in [1, 2, 3]:
        return int(bo)
    else:
        return "ar"

def encode_block(filename, obj, mol):
    """
    Parameters
    ----------
    obj : AtomGroup or Universe
    """
    try:
        mol.GetConformer()
    except Exception as e:
        print(e)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMoecule(mol)
        
    # Issue 2717
    obj = obj.atoms
    traj = obj.universe.trajectory
    ts = traj.ts

    molecule = []
    # try:
    #     molecule = ts.data['molecule']
    # except KeyError:
    #     raise_from(NotImplementedError(
    #         "MOL2Writer cannot currently write non MOL2 data"),
    #         None)

    # Need to remap atom indices to 1 based in this selection
    mapping = {a: i for i, a in enumerate(obj.atoms, start=1)}

    # Grab only bonds between atoms in the obj
    # ie none that extend out of it
    bondgroup = obj.bonds.atomgroup_intersection(obj, strict=True)
    bonds = sorted((b[0], b[1], b.order) for b in bondgroup)
    bond_lines = ["@<TRIPOS>BOND"]
    bond_lines.extend("{0:>5} {1:>5} {2:>5} {3:>2}"
                      "".format(bid,
                                mapping[atom1],
                                mapping[atom2],
                                filter_mol2_bonds(order))
                      for bid, (atom1, atom2, order)in enumerate(
                              bonds, start=1))
    bond_lines.append("\n")
    bond_lines = "\n".join(bond_lines)

    atom_lines = ["@<TRIPOS>ATOM"]
    atom_lines.extend("{0:>4} {1:>4} {2:>13.4f} {3:>9.4f} {4:>9.4f}"
                      " {5:>4} {6} {7} {8:>7.4f}"
                      "".format(mapping[a],
                                a.name,
                                a.position[0],
                                a.position[1],
                                a.position[2],
                                a.type[0] + filter_mol2_atoms(a.type[0], at), ######WIP!
                                a.resid,
                                # a.resname,
                                # a.charge)
                                "lig",
                                0)
                      for a, at in zip(obj.atoms, mol.GetAtoms()))
    atom_lines.append("\n")
    atom_lines = "\n".join(atom_lines)

    # try:
    #     substructure = ["@<TRIPOS>SUBSTRUCTURE\n"] + ts.data['substructure']
    # except KeyError:
    #     substructure = ""
    substructure = ["@<TRIPOS>SUBSTRUCTURE\n"] + ["1 **** 1 TEMP 0 **** **** 0 ROOT"]

    # check_sums = molecule[1].split()
    # check_sums[0], check_sums[1] = str(len(obj.atoms)), str(len(bondgroup))

    # prevent behavior change between repeated calls
    # see gh-2678
    # molecule_0_store = molecule[0]
    # molecule_1_store = molecule[1]

    # molecule[1] = "{0}\n".format(" ".join(check_sums))

    molecule.insert(0, "@<TRIPOS>MOLECULE\n")
    molecule.insert(1, "LIG\n")
    molecule.insert(2, f"{len(obj.atoms)} {len(bonds)} {0} {0} {1}\n")
    molecule.insert(3, "SMALL\n")
    molecule.insert(4, "USER_CHARGES\n\n")
    
    return_val = ("".join(molecule) + atom_lines +
                  bond_lines + "".join(substructure))

    # molecule[0] = molecule_0_store
    # molecule[1] = molecule_1_store

    f = open(filename, "w")
    f.write(return_val)
    
    return return_val
