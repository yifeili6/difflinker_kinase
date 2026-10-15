from pathlib import Path
import argparse
import base64
import csv
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinase_gap.generate import build_generation_item, instantiate_model, move_batch_to_device, restore_fragment_frame, extend_pocket_mask, ligand_features, build_candidate_molecule, molecule_quality, seed_all
from kinase_gap.chemistry_io import sdf_record
from kinase_gap.stereo import install
from src.datasets import collate

def contained_path(value, base=ROOT):
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Use a relative path inside the package')
    return base / relative

def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate linker coordinates and molecule records from two fragments and a protein pocket.')
    parser.add_argument('--checkpoint', default='models/kinase.ckpt')
    parser.add_argument('--input', default='data/example/context.json')
    parser.add_argument('--output', default='outputs/prediction')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--linker-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.linker_size < 1 or args.threads < 1:
        raise ValueError('Samples, linker size and threads must be positive')
    checkpoint = contained_path(args.checkpoint)
    context_path = contained_path(args.input)
    destination = contained_path(args.output)
    if destination.exists():
        raise FileExistsError('Choose a new output directory: ' + args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(args.seed)
    context = json.loads(context_path.read_text())
    row = context['row']
    fragment = Chem.Mol(base64.b64decode(context['fragment_binary_base64']))
    arrays = np.load(contained_path(context['pocket'], context_path.parent), allow_pickle=False)
    pocket = {'full_coord':arrays['coordinates'], 'full_types':arrays['elements'], 'cutoff_angstrom':10.0}
    item = build_generation_item(row, fragment, pocket)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    payload = torch.load(checkpoint, map_location='cpu')
    config = SimpleNamespace(data_dir=Path(args.input).parent, test_prefix='KLIFS_test.full', n_steps=500)
    model = instantiate_model(payload, config, device)
    data = move_batch_to_device(collate([item] * args.samples), device)
    def fixed_size(batch):
        return torch.full((batch['positions'].shape[0],), args.linker_size, dtype=torch.int64, device=device)
    start = time.monotonic()
    with torch.inference_mode():
        chain, node_mask = model.sample_chain(data, sample_fn=fixed_size, keep_frames=1)
    restored = restore_fragment_frame(chain_final=chain[0], node_mask=node_mask, input_positions=data['positions'], fragment_only_mask=data['fragment_only_mask'])
    pocket_mask = extend_pocket_mask(data['pocket_mask'], node_mask)
    install()
    records, coordinates, symbols_out, blocks = [], [], [], []
    for index in range(args.samples):
        positions, elements = ligand_features(model=model, restored=restored, node_mask=node_mask, pocket_mask=pocket_mask, batch_index=index)
        if not np.isfinite(positions).all():
            raise ValueError('Nonfinite coordinates were sampled')
        identifier = 'sample_' + str(index + 1).zfill(4)
        molecule = build_candidate_molecule(fragment=fragment, coordinates=positions, symbols=elements, candidate_id=identifier)
        valid, connected, smiles = molecule_quality(molecule)
        properties = {'candidate_id':identifier, 'source_context':row['fragmentation_id'], 'linker_atoms':args.linker_size, 'seed':args.seed, 'sample_index':index, 'valid':int(valid), 'connected':int(connected), 'smiles':smiles}
        for key, value in properties.items():
            molecule.SetProp(key, str(value))
        blocks.append(sdf_record(molecule, valid=valid, properties=properties))
        records.append(properties)
        coordinates.append(positions)
        symbols_out.append(elements)
    destination.mkdir(parents=True)
    (destination / 'molecules.sdf').write_text(''.join(blocks))
    with (destination / 'molecules.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(destination / 'coordinates.npz', coordinates=np.asarray(coordinates), elements=np.asarray(symbols_out, dtype=str))
    summary = {'checkpoint':args.checkpoint, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'input':args.input, 'input_sha256':hashlib.sha256(context_path.read_bytes()).hexdigest(), 'seed':args.seed, 'samples':args.samples, 'diffusion_steps':500, 'linker_atoms':args.linker_size, 'fragment_atoms':fragment.GetNumAtoms(), 'pocket_atoms':len(pocket['full_coord']), 'device':args.device, 'valid_molecules':sum(r['valid'] for r in records), 'connected_molecules':sum(r['connected'] for r in records), 'seconds':round(time.monotonic() - start, 3), 'output':args.output}
    (destination / 'prediction.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
