import MDAnalysis as mda

def encode_block(obj):
    """
    Parameters
    ----------
    obj : AtomGroup or Universe
    """
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
                                order)
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
                                a.type,
                                a.resid,
                                # a.resname,
                                # a.charge)
                                "lig",
                                0)
                      for a in obj.atoms)
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
    molecule.insert(4, "USER_CHARGES\n")
    
    return_val = ("".join(molecule) + atom_lines +
                  bond_lines + "".join(substructure))

    # molecule[0] = molecule_0_store
    # molecule[1] = molecule_1_store
    return return_val
