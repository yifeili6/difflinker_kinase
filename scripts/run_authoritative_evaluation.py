#!/usr/bin/env python3
"""Evaluate saved molecules using their authoritative chemical graph and coordinates."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/kinase_gap/evaluate.py').is_file())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from rdkit import Chem
from kinase_gap import evaluate
from kinase_gap.fragmentation import canonical_isomeric_identity
from kinase_gap.structural_bundle import parse_sdf_bytes

PINNED_SOURCES={
    'src/kinase_gap/evaluate.py':'9ad1cbf40eb14466b0213b50e16a6d192e0ace6050c2a4d97e67b6e040bb7b26',
    'src/kinase_gap/chemistry_io.py':'b9aac220cdd62e1768f91a86ebbc6227e305e323fac357d7af07abc8836234b8',
    'src/kinase_gap/fragmentation.py':'a4b542bd39868cf40435df30f4b4dcc9d76e34af727752d3c77c9266c79321ac',
    'src/kinase_gap/structural_bundle.py':'fa4c79f4ba7abeba427150a210d1a2530f296a58dbb478fbb353082b7ec29792',
}


def describe(path):
    path=Path(path)
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':evaluate.sha256_file(path)}


def source_contract():
    for name,digest in PINNED_SOURCES.items():
        if evaluate.sha256_file(ROOT/name)!=digest:
            raise ValueError('Scientific source checksum differs: '+name)
    return {'schema_version':1,'method':'Authoritative embedded molecule binary and exact saved coordinates',
            'helper':describe(Path(__file__).resolve()),'sources':dict(PINNED_SOURCES),
            'identity_checks':'Generation SMILES, binary identity and repeated sanitization agree',
            'coordinate_checks':'Binary coordinates and saved NPZ coordinates agree exactly'}


def chemical_state(molecule):
    atoms=[(a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),a.GetNumExplicitHs(),
            a.GetTotalNumHs(),a.GetNumRadicalElectrons(),int(a.GetChiralTag()),a.GetIsAromatic())
           for a in molecule.GetAtoms()]
    bonds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType()),b.GetIsAromatic(),
            int(b.GetStereo()),tuple(b.GetStereoAtoms())) for b in molecule.GetBonds()]
    return atoms,bonds


def validated_authoritative_record(row,record):
    if str(row['in_memory_sanitize_pass'])!='1':
        return record
    if record.authoritative_molecule is None:
        raise ValueError('Valid generation record has no verified coordinate carrier')
    if 'kinase_native_molecule_binary_base64' not in record.properties:
        raise ValueError('Valid generation record has no authoritative molecule binary')
    payload=record.mol_block.rstrip()+'\n'
    for key,value in record.properties.items():
        payload+='\n>  <'+key+'>\n'+value+'\n'
    payload+='\n$$$$\n'
    molecules=parse_sdf_bytes(payload.encode(),label=row['candidate_id'])
    if len(molecules)!=1:
        raise ValueError('Expected one authoritative generation molecule')
    molecule=molecules[0]
    source=record.authoritative_molecule
    keys=lambda mol:[(a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge()) for a in mol.GetAtoms()]
    if keys(molecule)!=keys(source):
        raise ValueError('Authoritative molecule atom order, isotope or charge differs')
    expected=canonical_isomeric_identity(Chem.MolFromSmiles(row['smiles']))
    if canonical_isomeric_identity(molecule)!=expected:
        raise ValueError('Authoritative molecule differs from generation identity')
    coordinates=source.GetConformer().GetPositions()
    if not np.isfinite(coordinates).all() or not np.array_equal(molecule.GetConformer().GetPositions(),coordinates):
        raise ValueError('Authoritative binary and saved NPZ coordinates differ')
    before=chemical_state(molecule)
    probe=Chem.Mol(molecule)
    for _ in range(3):
        Chem.SanitizeMol(probe)
        if canonical_isomeric_identity(probe)!=expected or chemical_state(probe)!=before:
            raise ValueError('Authoritative chemical graph changes during sanitization')
        if not np.array_equal(probe.GetConformer().GetPositions(),coordinates):
            raise ValueError('Authoritative pose changes during sanitization')
    for index,point in enumerate(coordinates):molecule.GetConformer().SetAtomPosition(index,point)
    return replace(record,authoritative_molecule=molecule)


def install():
    if getattr(evaluate,'_authoritative_binary_installed',False):return
    extension=source_contract()
    original_validate=evaluate.validate_generation_shard
    original_contract=evaluate.evaluation_contract
    def validate(config,generation,index):
        rows,records,done=original_validate(config,generation,index)
        return rows,[validated_authoritative_record(row,record) for row,record in zip(rows,records)],done
    def contract(config,generation):
        result=original_contract(config,generation)
        result.pop('contract_sha256')
        result['authoritative_molecule_loading']=extension
        result['contract_sha256']=evaluate.sha256_payload(result)
        return result
    evaluate.validate_generation_shard=validate
    evaluate.evaluation_contract=contract
    evaluate._authoritative_binary_installed=True


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def verify_record(root,record):
    raw=Path(record['path'])
    if raw.is_absolute() or '..' in raw.parts:raise ValueError('Unsafe reused input path')
    path=root/raw
    if any(p.is_symlink() for p in [path,*path.parents] if p!=root and root in p.parents):
        raise ValueError('Reused input traverses a symbolic link')
    if not path.is_file() or path.stat().st_size!=record['bytes'] or evaluate.sha256_file(path)!=record['sha256']:
        raise ValueError('Reused input checksum differs: '+str(path))
    return path


def reuse_ground_truth(config,source_config):
    old=evaluate.read_config(source_config)
    if old.run_root==config.run_root:raise ValueError('Ground truth reuse requires a new output root')
    old_science=dict(old.payload);new_science=dict(config.payload)
    old_science.pop('output_root');new_science.pop('output_root')
    if old_science!=new_science:raise ValueError('GT scientific configuration differs')
    old_contract=evaluate.gt_evaluation_contract(old)
    if evaluate.load_json(old.run_root/'gt_request.json')!=old_contract:
        raise ValueError('Original GT request differs from the active unchanged evaluator')
    if not (old.run_root/'gt_done.json').is_file():raise ValueError('Original GT is not complete')
    evaluate.consolidate_gt(old,old_contract)
    source_files=[old.run_root/'gt_request.json',old.run_root/'gt_done.json']
    source_files += [old.run_root/name for name in evaluate.load_json(old.run_root/'gt_done.json')['outputs']]
    for index in range(old.gt_expected_shards):
        marker=evaluate.validate_completed_gt_shard(old,old_contract,index)
        if marker is None:raise ValueError('Original GT shard is missing')
        folder=old.run_root/'gt_shards'/f'{index:04d}'
        source_files += [folder/'done.json']+[folder/name for name in marker['outputs']]
    marker_path=config.run_root/'ground_truth_reuse.json'
    records=[{'source':describe(p),'destination':str((config.run_root/p.relative_to(old.run_root)).relative_to(ROOT))} for p in source_files]
    receipt={'schema_version':1,'status':'complete','source_config':describe(old.path),
             'destination_config':describe(config.path),'helper':describe(Path(__file__).resolve()),
             'ground_truth_contract':old_contract,'files':records,
             'method':'Reuse completed ground-truth bytes after verifying unchanged scientific inputs and evaluator'}
    if marker_path.exists():
        if evaluate.load_json(marker_path)!=receipt:raise ValueError('GT reuse receipt differs')
        for row in records:verify_record(ROOT,{'path':row['destination'],**{k:row['source'][k] for k in ('bytes','sha256')}})
        return receipt
    config.run_root.mkdir(parents=True,exist_ok=True)
    for row in records:
        source=verify_record(ROOT,row['source']);destination=ROOT/row['destination']
        if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    for row in records:
        destination=ROOT/row['destination'];destination.parent.mkdir(parents=True,exist_ok=True)
        with verify_record(ROOT,row['source']).open('rb') as src,destination.open('xb') as dest:
            shutil.copyfileobj(src,dest);dest.flush();os.fsync(dest.fileno())
        verify_record(ROOT,{'path':row['destination'],**{k:row['source'][k] for k in ('bytes','sha256')}})
    write_new(marker_path,receipt)
    return receipt


def finalize_gt(config):
    reuse=config.run_root/'ground_truth_reuse.json'
    if reuse.exists():
        record=evaluate.load_json(reuse)
        source=verify_record(ROOT,record['source_config'])
        reuse_ground_truth(config,str(source.relative_to(ROOT)))
        return 0 if evaluate.consolidate_gt(config,record['ground_truth_contract']) else 2
    return evaluate.main(['gt-consolidate','--config',str(config.path.relative_to(ROOT))])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','validate-shard','generated','ground-truth','gt-preflight','finalize','gt-finalize','reuse-ground-truth'])
    parser.add_argument('--config',default='configs/evaluation_authoritative_v2.yml')
    parser.add_argument('--source-config',default='configs/evaluation.yml')
    parser.add_argument('--shard-index',type=int)
    parser.add_argument('--threads',type=int)
    args=parser.parse_args(argv)
    threads=args.threads or int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    if threads<1:raise ValueError('Thread count must be positive')
    os.environ['KINASE_EVALUATION_THREADS']=str(threads);os.environ['CUDA_VISIBLE_DEVICES']=''
    install();config=evaluate.read_config(args.config)
    if args.mode=='reuse-ground-truth':
        result=reuse_ground_truth(config,args.source_config)
        print(json.dumps({'status':result['status'],'files':len(result['files']),'ground_truth_contract':result['ground_truth_contract']['contract_sha256']}));return 0
    if args.mode=='validate-shard':
        if args.shard_index is None:raise ValueError('Validation requires a shard index')
        generation=evaluate.validate_generation_run(config,verify_combined_hashes=False)
        rows,records,_=evaluate.validate_generation_shard(config,generation,args.shard_index)
        print(json.dumps({'status':'validated','records':len(records),'authoritative_records':sum(r.authoritative_molecule is not None for r in records),'shard_index':args.shard_index}));return 0
    if args.mode=='gt-finalize':return finalize_gt(config)
    if args.mode=='finalize':
        first=evaluate.main(['consolidate','--config',args.config])
        return first or finalize_gt(config)
    command={'preflight':'preflight','generated':'shard','ground-truth':'gt-shard','gt-preflight':'gt-preflight'}[args.mode]
    forwarded=[command,'--config',args.config]
    if args.mode in ['generated','ground-truth']:
        index=args.shard_index if args.shard_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        forwarded+=['--shard-index',str(index)]
    return evaluate.main(forwarded)


if __name__=='__main__':raise SystemExit(main())
