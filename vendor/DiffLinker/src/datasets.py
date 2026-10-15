import os
import hashlib
import numpy as np
import pandas as pd
import pickle
import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const
from pdb import set_trace

def input_fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.basename(path).encode())
        with open(path, 'rb') as handle:
            for block in iter(lambda : handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()

def cache_is_current(dataset_path, fingerprint):
    fingerprint_path = f'{dataset_path}.inputs.sha256'
    if not os.path.exists(dataset_path) or not os.path.exists(fingerprint_path):
        return False
    with open(fingerprint_path) as handle:
        return handle.read().strip() == fingerprint

def save_cpu_cache(data, dataset_path, fingerprint):
    temporary_path = f'{dataset_path}.tmp.{os.getpid()}'
    torch.save(data, temporary_path)
    os.replace(temporary_path, dataset_path)
    with open(f'{dataset_path}.inputs.sha256', 'w') as handle:
        handle.write(f'{fingerprint}\n')

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def parse_molecule(mol, is_geom):
    one_hot = []
    charges = []
    atom2idx = const.GEOM_ATOM2IDX if is_geom else const.ATOM2IDX
    charges_dict = const.GEOM_CHARGES if is_geom else const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return (positions, np.array(one_hot), np.array(charges))

class ZincDataset(Dataset):

    def __init__(self, data_path, prefix, device):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        inputs = [os.path.join(data_path, f'{prefix}_table.csv'), os.path.join(data_path, f'{prefix}_frag.sdf'), os.path.join(data_path, f'{prefix}_link.sdf')]
        fingerprint = input_fingerprint(inputs)
        if cache_is_current(dataset_path, fingerprint):
            self.data = torch.load(dataset_path, map_location='cpu')
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = ZincDataset.preprocess(data_path, prefix, 'cpu')
            save_cpu_cache(self.data, dataset_path, fingerprint)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')
        is_geom = 'geom' in prefix or 'MOAD' in prefix or 'KLIF' in prefix
        is_multifrag = 'multifrag' in prefix
        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path)), total=len(table))
        for ((_, row), fragments, linker) in generator:
            uuid = row['uuid']
            name = row['molecule']
            (frag_pos, frag_one_hot, frag_charges) = parse_molecule(fragments, is_geom=is_geom)
            (link_pos, link_one_hot, link_charges) = parse_molecule(linker, is_geom=is_geom)
            positions = np.concatenate([frag_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)
            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1
            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(link_charges)])
            data.append({'uuid': uuid, 'name': name, 'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device), 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device), 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device), 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device), 'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device), 'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device), 'num_atoms': len(positions)})
        return data

class MOADDataset(Dataset):

    def __init__(self, data=None, data_path=None, prefix=None, device=None):
        assert data is not None or all((x is not None for x in (data_path, prefix, device)))
        if data is not None:
            self.data = data
            return
        if '.' in prefix:
            (prefix, pocket_mode) = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]
        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        inputs = [os.path.join(data_path, f'{prefix}_table.csv'), os.path.join(data_path, f'{prefix}_frag.sdf'), os.path.join(data_path, f'{prefix}_link.sdf'), os.path.join(data_path, f'{prefix}_pockets.pkl')]
        fingerprint = input_fingerprint(inputs)
        if cache_is_current(dataset_path, fingerprint):
            self.data = torch.load(dataset_path, map_location='cpu')
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = self.preprocess(data_path, prefix, pocket_mode, 'cpu')
            save_cpu_cache(self.data, dataset_path, fingerprint)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')
        is_geom = True
        is_multifrag = 'multifrag' in prefix
        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)
        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path), pockets), total=len(table))
        for ((_, row), fragments, linker, pocket_data) in generator:
            uuid = row['uuid']
            name = row['molecule']
            (frag_pos, frag_one_hot, frag_charges) = parse_molecule(fragments, is_geom=is_geom)
            (link_pos, link_one_hot, link_charges) = parse_molecule(linker, is_geom=is_geom)
            pocket_pos = pocket_data[f'{pocket_mode}_coord']
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f'{pocket_mode}_types']:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)
            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, pocket_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, pocket_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)
            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1
            fragment_only_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(pocket_charges), np.zeros_like(link_charges)])
            pocket_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(pocket_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.zeros_like(pocket_charges), np.ones_like(link_charges)])
            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.ones_like(pocket_charges), np.zeros_like(link_charges)])
            data.append({'uuid': uuid, 'name': name, 'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device), 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device), 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device), 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device), 'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT, device=device), 'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device), 'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device), 'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device), 'num_atoms': len(positions)})
        return data

class OptimisedMOADDataset(MOADDataset):

    def __len__(self):
        return len(self.data['fragmentation_level_data'])

    def __getitem__(self, item):
        fragmentation_level_data = self.data['fragmentation_level_data'][item]
        protein_level_data = self.data['protein_level_data'][fragmentation_level_data['name']]
        return {**fragmentation_level_data, **protein_level_data}

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        print('Preprocessing optimised version of the dataset')
        protein_level_data = {}
        fragmentation_level_data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')
        is_geom = True
        is_multifrag = 'multifrag' in prefix
        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)
        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path), pockets), total=len(table))
        for ((_, row), fragments, linker, pocket_data) in generator:
            uuid = row['uuid']
            name = row['molecule']
            (frag_pos, frag_one_hot, frag_charges) = parse_molecule(fragments, is_geom=is_geom)
            (link_pos, link_one_hot, link_charges) = parse_molecule(linker, is_geom=is_geom)
            pocket_pos = pocket_data[f'{pocket_mode}_coord']
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f'{pocket_mode}_types']:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)
            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, pocket_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, pocket_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)
            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1
            fragment_only_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(pocket_charges), np.zeros_like(link_charges)])
            pocket_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(pocket_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.zeros_like(pocket_charges), np.ones_like(link_charges)])
            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.ones_like(pocket_charges), np.zeros_like(link_charges)])
            fragmentation_level_data.append({'uuid': uuid, 'name': name, 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device), 'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT, device=device), 'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device), 'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device), 'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device)})
            protein_level_data[name] = {'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device), 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device), 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device), 'num_atoms': len(positions)}
        return {'fragmentation_level_data': fragmentation_level_data, 'protein_level_data': protein_level_data}

def collate(batch):
    out = {}
    for (i, data) in enumerate(batch):
        for (key, value) in data.items():
            out.setdefault(key, []).append(value)
    for (key, value) in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')
    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]
    (batch_size, n_nodes) = atom_mask.size()
    if 'pocket_mask' in batch[0].keys():
        batch_mask = torch.cat([torch.ones(n_nodes, dtype=const.TORCH_INT) * i for i in range(batch_size)]).to(atom_mask.device)
        out['edge_mask'] = batch_mask
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
        edge_mask *= diag_mask
        out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]
    return out

def collate_with_fragment_edges(batch):
    out = {}
    for (i, data) in enumerate(batch):
        for (key, value) in data.items():
            out.setdefault(key, []).append(value)
    for (key, value) in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')
    frag_mask = out['fragment_mask']
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    (batch_size, n_nodes) = frag_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    (rows, cols, bonds) = ([], [], [])
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    edges = [torch.LongTensor(rows).to(frag_mask.device), torch.LongTensor(cols).to(frag_mask.device)]
    out['edges'] = edges
    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]
    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]
    return out

def collate_with_fragment_without_pocket_edges(batch):
    out = {}
    for (i, data) in enumerate(batch):
        for (key, value) in data.items():
            out.setdefault(key, []).append(value)
    for (key, value) in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')
    frag_mask = out['fragment_only_mask']
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    (batch_size, n_nodes) = frag_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    (rows, cols, bonds) = ([], [], [])
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    edges = [torch.LongTensor(rows).to(frag_mask.device), torch.LongTensor(cols).to(frag_mask.device)]
    out['edges'] = edges
    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]
    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]
    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False, generator=None, num_workers=0, worker_init_fn=None):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle, generator=generator, num_workers=num_workers, worker_init_fn=worker_init_fn)

def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)

def create_templates_for_linker_generation(data, linker_sizes):
    decoupled_data = []
    for (i, linker_size) in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data['fragment_mask'][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for (k, v) in data.items():
            if k == 'num_atoms':
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                fill_value = 1 if k == 'linker_mask' else 0
                template = create_template(v[i], fragment_size, linker_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template
        decoupled_data.append(data_dict)
    return collate(decoupled_data)
