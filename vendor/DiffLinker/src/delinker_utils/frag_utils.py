import csv
import networkx as nx
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from src.delinker_utils import sascorer

def read_triples_file(filename):
    """Returns array containing smiles strings of molecules"""
    (smiles, names) = ([], [])
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0:3])
    return smiles

def remove_dummys(smi_string):
    return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), True)[0]))

def sa_filter(results, verbose=True):
    count = 0
    total = 0
    for (processed, res) in enumerate(results):
        total += len(res)
        for m in res:
            if calc_mol_props(m[1])[1] < calc_mol_props(m[0])[1]:
                count += 1
        if verbose:
            if processed % 10 == 0:
                print('\rProcessed %d' % processed, end='')
    print('\r', end='')
    return count / total

def ring_check_res(res, clean_frag):
    check = True
    gen_mol = Chem.MolFromSmiles(res[1])
    linker = Chem.DeleteSubstructs(gen_mol, clean_frag)
    ssr = Chem.GetSymmSSSR(linker)
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and (bond.GetEndAtomIdx() in ring):
                    check = False
    return check

def ring_filter(results, verbose=True):
    count = 0
    total = 0
    du = Chem.MolFromSmiles('*')
    for (processed, res) in enumerate(results):
        total += len(res)
        for m in res:
            clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(m[0]), du, Chem.MolFromSmiles('[H]'), True)[0])
            if ring_check_res(m, clean_frag):
                count += 1
        if verbose:
            if processed % 10 == 0:
                print('\rProcessed %d' % processed, end='')
    print('\r', end='')
    return count / total

def check_ring_filter(linker):
    check = True
    ssr = Chem.GetSymmSSSR(linker)
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and (bond.GetEndAtomIdx() in ring):
                    check = False
    return check

def check_pains(mol, pains_smarts):
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True

def calc_2d_filters(toks, pains_smarts):
    try:
        frags = Chem.MolFromSmiles(toks[2])
        linker = Chem.MolFromSmiles(toks[1])
        full_mol = Chem.MolFromSmiles(toks[0])
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frags, du, Chem.MolFromSmiles('[H]'), True)[0])
        res = []
        if len(full_mol.GetSubstructMatch(clean_frag)) > 0:
            if calc_sa_score_mol(full_mol) < calc_sa_score_mol(frags):
                res.append(True)
            else:
                res.append(False)
            if check_ring_filter(linker):
                res.append(True)
            else:
                res.append(False)
            if check_pains(full_mol, pains_smarts):
                res.append(True)
            else:
                res.append(False)
        return res
    except:
        return [False, False, False]

def calc_filters_2d_dataset(results, pains_smarts_loc, n_cores=1):
    with open(pains_smarts_loc, 'r') as f:
        pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]
    with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
        filters_2d = parallel((delayed(calc_2d_filters)([toks[2], toks[4], toks[1]], pains_smarts) for toks in results))
    return filters_2d

def calc_mol_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Error passing: %s' % smiles)
        return None
    qed = Chem.QED.qed(mol)
    sas = sascorer.calculateScore(mol)
    ri = mol.GetRingInfo()
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 6:
            nMacrocycles += 1
    prop_array = [qed, sas]
    return prop_array

def calc_sa_score_mol(mol, verbose=False):
    if mol is None:
        if verbose:
            print('Error passing: %s' % mol)
        return None
    return sascorer.calculateScore(mol)

def get_linker(full_mol, clean_frag, starting_point):
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    if len(matches) == 0:
        print('No matches')
        return ''
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if linker_len == 0:
        return ''
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    poss_linker = []
    if len(matches) > 0:
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            linker_bonds = []
            atoms_joined_to_linker = []
            for idx_to_delete in sorted(match, reverse=True):
                nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
                intersect = set(nei).intersection(set(linker_atoms))
                if len(intersect) == 1:
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete, list(intersect)[0]).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
                elif len(intersect) > 1:
                    for idx_nei in list(intersect):
                        linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete, idx_nei).GetIdx())
                        atoms_joined_to_linker.append(idx_to_delete)
            if len(set(atoms_joined_to_linker)) != 2:
                continue
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    fragmented_mol = Chem.FragmentOnBonds(mol_to_break, linker_bonds)
                    linker_to_return = Chem.Mol(fragmented_mol)
                    qp = Chem.AdjustQueryParameters()
                    qp.makeDummiesQueries = True
                    for f in starting_point.split('.'):
                        qfrag = Chem.AdjustQueryProperties(Chem.MolFromSmiles(f), qp)
                        linker_to_return = AllChem.DeleteSubstructs(linker_to_return, qfrag, onlyFrags=True)
                    if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 2:
                        Chem.Kekulize(linker_to_return, clearAromaticFlags=True)
                        if len(Chem.rdmolops.GetMolFrags(linker_to_return)) > 1:
                            for frag in Chem.MolToSmiles(linker_to_return).split('.'):
                                if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == linker_len:
                                    return frag
                        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(linker_to_return)))
                    else:
                        fragmented_mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragmented_mol), sanitize=False)
                        linker_to_return = AllChem.DeleteSubstructs(fragmented_mol, Chem.MolFromSmiles(starting_point))
                        poss_linker.append(Chem.MolToSmiles(linker_to_return))
    if len(poss_linker) == 1:
        return poss_linker[0]
    elif len(poss_linker) == 0:
        print('FAIL:', Chem.MolToSmiles(full_mol), Chem.MolToSmiles(clean_frag), starting_point)
        return ''
    else:
        print('More than one poss linker. ', poss_linker)
        return poss_linker[0]

def get_linker_v2(full_mol, clean_frag):
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    if len(matches) == 0:
        print('No matches')
        return ''
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if linker_len == 0:
        return ''
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    poss_linker = []
    if len(matches) > 0:
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            linker_bonds = []
            atoms_joined_to_linker = []
            for idx_to_delete in sorted(match, reverse=True):
                nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
                intersect = set(nei).intersection(set(linker_atoms))
                if len(intersect) == 1:
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete, list(intersect)[0]).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
                elif len(intersect) > 1:
                    for idx_nei in list(intersect):
                        linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete, idx_nei).GetIdx())
                        atoms_joined_to_linker.append(idx_to_delete)
            if len(set(atoms_joined_to_linker)) != 2:
                continue
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 2:
                    Chem.Kekulize(linker, clearAromaticFlags=True)
                    if len(Chem.rdmolops.GetMolFrags(linker)) > 1:
                        for frag in Chem.MolToSmiles(linker).split('.'):
                            if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == linker_len:
                                return frag
                    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(linker)))
                else:
                    poss_linker.append(Chem.MolToSmiles(linker))
    if len(poss_linker) == 1:
        return poss_linker[0]
    elif len(poss_linker) == 0:
        print('FAIL:', Chem.MolToSmiles(full_mol), Chem.MolToSmiles(clean_frag))
        return ''
    else:
        print('More than one poss linker. ', poss_linker)
        return poss_linker[0]

def unique(results):
    total_dupes = 0
    total = 0
    for res in results:
        original_num = len(res)
        test_data = set(res)
        new_num = len(test_data)
        total_dupes += original_num - new_num
        total += original_num
    return 1 - total_dupes / float(total)

def check_recovered_original_mol_with_idx(results):
    outcomes = []
    rec_idx = []
    for res in results:
        success = False
        orig_mol = Chem.MolFromSmiles(res[0][0][0])
        Chem.RemoveStereochemistry(orig_mol)
        orig_mol = Chem.MolToSmiles(Chem.RemoveHs(orig_mol))
        for m in res:
            gen_mol = Chem.MolFromSmiles(m[0][2])
            Chem.RemoveStereochemistry(gen_mol)
            gen_mol = Chem.MolToSmiles(Chem.RemoveHs(gen_mol))
            if gen_mol == orig_mol:
                success = True
                rec_idx.append(m[1])
        if not success:
            outcomes.append(False)
        else:
            outcomes.append(True)
    return (outcomes, rec_idx)

def topology_from_rdkit(rdkit_molecule):
    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        topology.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())
    for bond in rdkit_molecule.GetBonds():
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
    return topology
