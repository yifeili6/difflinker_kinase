#!/usr/bin/env python3
"""Audit native graph mappings and build a separate coordinate-faithful dataset."""
from __future__ import annotations
import argparse
import base64
import collections
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinase_gap.fragmentation import Fragmentation, build_coordinate_components, validate_native_reassembly, canonical_isomeric_identity
from kinase_gap.prepare import load_ligand

SPLITS = {'train':'KLIFS_train', 'validation':'KLIFS_val', 'test':'KLIFS_test'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def same_coordinates(left, right):
    return np.array_equal(left.GetConformer().GetPositions(), right.GetConformer().GetPositions())

def native_map(mol):
    return [atom.GetIntProp('_ParentAtomIndex') for atom in mol.GetAtoms()]

def old_graph_state(parent, fragment, linker, anchors):
    pos = parent.GetConformer().GetPositions()
    def map_points(mol):
        coordinates = mol.GetConformer().GetPositions()
        distances = np.sum((coordinates[:, None] - pos[None, :]) ** 2, axis=2)
        mapping = np.argmin(distances, axis=1)
        if np.max(np.min(distances, axis=1)) > 1e-6 or len(set(mapping)) != len(mapping):
            raise ValueError('Source coordinate mapping is not one-to-one')
        return list(map(int, mapping))
    fm, lm = map_points(fragment), map_points(linker)
    parts = [{fm[i] for i in group} for group in Chem.GetMolFrags(fragment)] + [set(lm)]
    owner = {atom: part for part, group in enumerate(parts) for atom in group}
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in parent.GetBonds()
             if owner[b.GetBeginAtomIdx()] != owner[b.GetEndAtomIdx()]]
    pairs = sorted(tuple(sorted((owner[i], owner[j]))) for i,j in edges)
    marked = all(any(n.GetIdx() in parts[-1] for n in parent.GetAtomWithIdx(fm[a]).GetNeighbors()) for a in anchors)
    return {'source_anchor_graph_valid':marked,
            'source_partition_graph_valid':pairs == [(0,2),(1,2)],
            'source_fragment_parent_indices':fm, 'source_linker_parent_indices':lm}

def assert_native_graph(parent, fragment, linker, anchors):
    fm, lm = native_map(fragment), native_map(linker)
    if len(fm+lm) != len(set(fm+lm)) or set(fm+lm) != set(range(parent.GetNumAtoms())):
        raise ValueError('Corrected components do not partition the parent atoms')
    expected = parent.GetConformer().GetPositions()
    for mol, match in ((fragment,fm),(linker,lm)):
        if not np.array_equal(mol.GetConformer().GetPositions(), expected[match]):
            raise ValueError('Native coordinate identity failed')
        if [a.GetSymbol() for a in mol.GetAtoms()] != [parent.GetAtomWithIdx(i).GetSymbol() for i in match]:
            raise ValueError('Native element identity failed')
    state = old_graph_state(parent, fragment, linker, anchors)
    if not state['source_anchor_graph_valid'] or not state['source_partition_graph_valid']:
        raise ValueError('Corrected graph validation failed')
    validate_native_reassembly(parent, fragment, linker)
    return fm, lm

def tensor_component(molecule):
    exported = Chem.Mol(molecule)
    for atom in exported.GetAtoms():
        # Cap each severed single bond on this tensor-only export. In particular,
        # an aromatic N bearing a cut substituent becomes [nH] in its carrier.
        atom.SetNumExplicitHs(atom.GetIntProp('_ParentTotalHs') + atom.GetIntProp('_ExitCount'))
        atom.SetNoImplicit(True)
    Chem.SanitizeMol(exported)
    block = Chem.MolToMolBlock(exported, kekulize=True)
    loaded = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    if loaded is None:
        raise ValueError('Component export is not sanitizable')
    if [a.GetSymbol() for a in loaded.GetAtoms()] != [a.GetSymbol() for a in molecule.GetAtoms()]:
        raise ValueError('Component export changed atom identities/order')
    if not np.allclose(loaded.GetConformer().GetPositions(), molecule.GetConformer().GetPositions(), atol=5.1e-5, rtol=0):
        raise ValueError('Component export changed coordinates beyond SDF precision')
    return exported

def atom_properties(mol):
    return [{'index':a.GetIdx(), 'parent_index':a.GetIntProp('_ParentAtomIndex'),
             'element':a.GetSymbol(), 'formal_charge':a.GetFormalCharge(),
             'explicit_h':a.GetNumExplicitHs(), 'no_implicit':a.GetNoImplicit(),
             'aromatic':a.GetIsAromatic(), 'isotope':a.GetIsotope(),
             'chiral_tag':int(a.GetChiralTag()), 'radical_electrons':a.GetNumRadicalElectrons(),
             'native_chiral_tag':a.GetIntProp('_ParentChiralTag'),
             'native_neighbor_order':[int(i) for i in a.GetProp('_ParentNeighborOrder').split(',') if i],
             'anchor':a.GetProp('_Anchor') == '1', 'cut_count':a.GetIntProp('_ExitCount')}
            for a in mol.GetAtoms()]

def bond_properties(mol):
    return [{'begin':b.GetBeginAtomIdx(), 'end':b.GetEndAtomIdx(),
             'type':str(b.GetBondType()), 'aromatic':b.GetIsAromatic(),
             'conjugated':b.GetIsConjugated(), 'stereo':int(b.GetStereo()),
             'bond_dir':int(b.GetBondDir()),'native_bond_dir':b.GetIntProp('_ParentBondDir'),
             'native_stereo':b.GetIntProp('_ParentStereo'),
             'native_stereo_atoms':[int(i) for i in b.GetProp('_ParentStereoAtoms').split(',') if i],
             'stereo_atoms':list(b.GetStereoAtoms())} for b in mol.GetBonds()]

def native_stereo(mol):
    return {'atom_chiral_tags':[int(a.GetChiralTag()) for a in mol.GetAtoms()],
            'atom_neighbor_order':[[n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()],
            'bonds':[{'begin':b.GetBeginAtomIdx(),'end':b.GetEndAtomIdx(),
                      'stereo':int(b.GetStereo()),'stereo_atoms':list(b.GetStereoAtoms()),
                      'bond_dir':int(b.GetBondDir())}
                     for b in mol.GetBonds()]}

def restored_stereo(mol, record):
    result=Chem.Mol(mol)
    for atom,tag in zip(result.GetAtoms(),record['atom_chiral_tags']):
        atom.SetChiralTag(Chem.ChiralType(tag))
        original=record['atom_neighbor_order'][atom.GetIdx()]
        observed=[n.GetIdx() for n in atom.GetNeighbors()]
        if set(original)!=set(observed):raise ValueError('Native stereo neighbor graph changed')
        order=[original.index(i) for i in observed]
        odd=sum(order[i]>order[j] for i in range(len(order)) for j in range(i+1,len(order)))%2
        if odd and atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atom.InvertChirality()
    for item in record['bonds']:
        bond=result.GetBondBetweenAtoms(item['begin'],item['end'])
        bond.SetBondDir(Chem.BondDir(item['bond_dir']))
        if item['stereo_atoms']:
            bond.SetStereoAtoms(*item['stereo_atoms'])
        bond.SetStereo(Chem.BondStereo(item['stereo']))
    Chem.AssignStereochemistry(result,cleanIt=False,force=True)
    return result

def parent_roundtrip(parent):
    loaded=Chem.MolFromMolBlock(Chem.MolToMolBlock(parent,kekulize=True),sanitize=True,removeHs=False)
    if loaded is None or Chem.MolToSmiles(loaded,isomericSmiles=False)!=Chem.MolToSmiles(parent,isomericSmiles=False):
        raise ValueError('Native ligand SDF constitutional identity roundtrip failed')
    for source,exported in zip(parent.GetAtoms(),loaded.GetAtoms()):
        for method in ('GetAtomicNum','GetIsotope','GetFormalCharge','GetTotalNumHs'):
            if getattr(source,method)()!=getattr(exported,method)():
                raise ValueError('Native SDF changed atom chemistry: '+method)
    inferred_changed=Chem.MolToSmiles(loaded)!=Chem.MolToSmiles(parent)
    restored=restored_stereo(loaded,native_stereo(parent))
    if canonical_isomeric_identity(restored)!=canonical_isomeric_identity(parent):
        raise ValueError('Native stereo record did not restore exact isomeric identity')
    binary=parent.ToBinary(Chem.PropertyPickleOptions.AllProps)
    copied=Chem.Mol(binary)
    if Chem.MolToSmiles(copied)!=Chem.MolToSmiles(parent) or not same_coordinates(copied,parent):
        raise ValueError('Native binary chemistry sidecar failed exact roundtrip')
    return inferred_changed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='Project containing the original fragment tables and structural inputs')
    parser.add_argument('--audit-dir',type=Path,default=ROOT/'provenance/dataset_audit')
    parser.add_argument('--audit-name',default='dataset_mapping_v1')
    parser.add_argument('--prior-audit-rows',type=Path,
                        help='Reuse previously passed graph/cap checks; every build row is revalidated')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--build-workers',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/klifs_v1')
    args=parser.parse_args()
    source=args.source.resolve(); args.audit_dir.mkdir(parents=True,exist_ok=True)
    out=args.output.resolve()
    out.relative_to(ROOT.resolve())
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    manifest={r['structure_id']:r for r in read_csv(source/'data/manifests/klifs_v1/contexts.csv')}
    srcdata=source/'data/processed/klifs_v1'
    rows_path=args.audit_dir/(args.audit_name+'_rows.jsonl')
    errors_path=args.audit_dir/(args.audit_name+'_errors.jsonl')
    totals={}; input_hashes={}; cache={}; stereo_changed={}; total_errors=0
    sourcecode=sha(ROOT/'src/kinase_gap/fragmentation.py')
    prior={}
    if args.prior_audit_rows:
        args.prior_audit_rows.resolve().relative_to(args.audit_dir.resolve())
        prior={r['fragmentation_id']:r for r in map(json.loads,args.prior_audit_rows.read_text().splitlines())}
    RDLogger.DisableLog('rdApp.*')
    with rows_path.open('x') as audit, errors_path.open('x') as errors:
        for split,prefix in SPLITS.items():
            table=read_csv(srcdata/(prefix+'_table.csv'))
            input_hashes[prefix+'_table.csv']=sha(srcdata/(prefix+'_table.csv'))
            counts=collections.Counter(rows=len(table))
            original_frag=Chem.SDMolSupplier(str(srcdata/(prefix+'_frag.sdf')),sanitize=False,removeHs=False)
            original_link=Chem.SDMolSupplier(str(srcdata/(prefix+'_link.sdf')),sanitize=False,removeHs=False)
            if len(original_frag)!=len(table) or len(original_link)!=len(table):
                raise ValueError('Source row counts differ')
            for idx,row in enumerate(table):
                try:
                    context=manifest[row['structure_id']]
                    path=source/context['ligand_path']
                    if row['structure_id'] not in cache:
                        if sha(path)!=row['ligand_sha256']:
                            raise ValueError('Raw ligand hash differs from source table')
                        parent,_=load_ligand(path)
                        stereo_changed[row['structure_id']]=parent_roundtrip(parent)
                        cache[row['structure_id']]=parent
                    parent=cache[row['structure_id']]
                    if row['fragmentation_id'] in prior:
                        record=dict(prior[row['fragmentation_id']])
                        if (record['ligand_sha256']!=row['ligand_sha256'] or record['uuid']!=int(row['uuid'])
                            or record['structure_id']!=row['structure_id'] or record['split']!=split):
                            raise ValueError('Prior validated mapping identity differs')
                        record['sdf_inferred_stereo_differs']=stereo_changed[row['structure_id']]
                        counts['prior_graph_checks_reused']+=1
                    else:
                        spec=Fragmentation(row['mmpa_linker_with_dummies'],row['mmpa_fragments_with_dummies'],row['fragmentation_id'].split('__mmpa_')[-1])
                        frag,link,anchors=build_coordinate_components(parent,spec)
                        fm,lm=assert_native_graph(parent,frag,link,anchors)
                        tensor_component(frag);tensor_component(link)
                        old=old_graph_state(parent,original_frag[idx],original_link[idx],tuple(int(row[k]) for k in ('anchor_1','anchor_2')))
                        changed=fm!=old['source_fragment_parent_indices'] or lm!=old['source_linker_parent_indices']
                        record={'split':split,'uuid':int(row['uuid']),'fragmentation_id':row['fragmentation_id'],
                            'structure_id':row['structure_id'],'ligand_sha256':row['ligand_sha256'],
                            'fragment_parent_indices':fm,'linker_parent_indices':lm,'anchors':list(anchors),
                            'mapping_changed':changed,'sdf_inferred_stereo_differs':stereo_changed[row['structure_id']],**old}
                    audit.write(json.dumps(record,separators=(',',':'))+'\n')
                    counts['validated']+=1;counts['mapping_changed']+=record['mapping_changed']
                    counts['source_anchor_graph_invalid']+=not record['source_anchor_graph_valid']
                    counts['source_partition_graph_invalid']+=not record['source_partition_graph_valid']
                    counts['sdf_inferred_stereo_differs']+=stereo_changed[row['structure_id']]
                except Exception as exc:
                    total_errors+=1; counts['errors']+=1
                    errors.write(json.dumps({'split':split,'uuid':row['uuid'],'id':row['fragmentation_id'],'error':type(exc).__name__+': '+str(exc)})+'\n');errors.flush()
                if idx%1000==0:
                    print(json.dumps({'stage':'audit','split':split,'row':idx,'counts':dict(counts)}),flush=True)
            totals[split]=dict(counts)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'hostname':os.uname().nodename,
             'mapping_module_sha256':sourcecode,'input_table_sha256':input_hashes,'counts':totals,
             'source_contexts':len(cache),'total_errors':total_errors,'audit_rows_sha256':sha(rows_path),
             'prior_validated_mapping_rows':({'path':str(args.prior_audit_rows),'sha256':sha(args.prior_audit_rows)} if args.prior_audit_rows else None),
             'invariants':['complete disjoint parent atom partition','exactly two fragment-linker cut bonds',
                           'dummy-neighbor cut endpoint correspondence','symmetry-complete matching',
                           'native element and coordinate identity','native parent stereochemistry preservation',
                           'parent and tensor-component SDF sanitization roundtrip',
                           'source stereo record and binary sidecar restore exact native isomeric identity']}
    write_json(args.audit_dir/(args.audit_name+'.json'),summary)
    print(json.dumps(summary),flush=True)
    if total_errors:
        raise RuntimeError(f'Audit failed on {total_errors} rows; no dataset created')
    if not args.build:
        return
    out.mkdir(parents=True)
    checks=list(map(json.loads,rows_path.read_text().splitlines()))
    check_by_id={r['fragmentation_id']:r for r in checks}
    reports={}
    def write_split(split,prefix):
        table=read_csv(srcdata/(prefix+'_table.csv'))
        with (out/(prefix+'_table.csv')).open('x',newline='') as tf, (out/(prefix+'_chemistry.jsonl')).open('x') as cf:
            tablewriter=csv.DictWriter(tf,fieldnames=list(table[0]));tablewriter.writeheader()
            writers={kind:Chem.SDWriter(str(out/(prefix+'_'+kind+'.sdf'))) for kind in ('mol','frag','link')}
            try:
                for idx,row in enumerate(table):
                    parent=cache[row['structure_id']]
                    spec=Fragmentation(row['mmpa_linker_with_dummies'],row['mmpa_fragments_with_dummies'],row['fragmentation_id'].split('__mmpa_')[-1])
                    frag,link,anchors=build_coordinate_components(parent,spec)
                    fm,lm=assert_native_graph(parent,frag,link,anchors)
                    check=check_by_id[row['fragmentation_id']]
                    if fm!=check['fragment_parent_indices'] or lm!=check['linker_parent_indices']:
                        raise ValueError('Build mapping differs from verified audit')
                    row['anchor_1'],row['anchor_2']=anchors
                    for kind,mol in (('mol',parent),('frag',tensor_component(frag)),('link',tensor_component(link))):
                        named=Chem.Mol(mol);named.SetProp('_Name',row['fragmentation_id'])
                        if kind=='mol':
                            named.SetProp('kinase_native_smiles',Chem.MolToSmiles(parent))
                            named.SetProp('kinase_native_stereo',json.dumps(native_stereo(parent),separators=(',',':')))
                        writers[kind].write(named)
                    tablewriter.writerow(row)
                    cf.write(json.dumps({'uuid':int(row['uuid']),'fragmentation_id':row['fragmentation_id'],
                        'ligand_sha256':row['ligand_sha256'],'anchors':list(anchors),
                        'canonical_smiles':Chem.MolToSmiles(parent),
                        'parent_molecule_binary_base64':base64.b64encode(parent.ToBinary(Chem.PropertyPickleOptions.AllProps)).decode(),
                        'fragment_atoms':atom_properties(frag),'fragment_bonds':bond_properties(frag),
                        'linker_atoms':atom_properties(link),'linker_bonds':bond_properties(link)},separators=(',',':'))+'\n')
                    if idx%2000==0:print(json.dumps({'stage':'build','split':split,'row':idx}),flush=True)
            finally:
                for writer in writers.values():writer.close()
        shutil.copyfile(srcdata/(prefix+'_pockets.pkl'),out/(prefix+'_pockets.pkl'))
        if sha(out/(prefix+'_pockets.pkl'))!=sha(srcdata/(prefix+'_pockets.pkl')):
            raise ValueError('Pocket copy hash mismatch')
    jobs=[]
    context=multiprocessing.get_context('fork')
    for split,prefix in SPLITS.items():
        table=read_csv(srcdata/(prefix+'_table.csv'))
        reports[split]={'fragmentation_rows':len(table),'structures':len({r['structure_id'] for r in table}),
                       'pdbs':len({r['pdb'] for r in table}),'molecules':len({r['canonical_smiles'] for r in table})}
        job=context.Process(target=write_split,args=(split,prefix));job.start();jobs.append(job)
        if len(jobs)>=args.build_workers:
            for job in jobs:
                job.join()
                if job.exitcode!=0:raise RuntimeError('A dataset split export failed')
            jobs=[]
    for job in jobs:
        job.join()
        if job.exitcode!=0:raise RuntimeError('A dataset split export failed')
    report=json.loads((srcdata/'dataset_report.json').read_text())
    report['dataset_version']='klifs_v1'
    report['configuration_sha256']=sha(ROOT/'configs/data_klifs_v1.yml')
    report['coordinate_construction']={'version':'native_graph_20261001','matching':'symmetry-complete parent-graph mapping with native stereochemistry',
        'cut_bonds':2,'parent_sdf':'kekulized; constitutional identity and native-stereo property roundtrips verified',
        'native_parent_chemistry':'RDKit all-properties binary with canonical isomeric identity and native coordinates in per-row sidecar',
        'component_sdf':'sanitized coordinate and element carriers for tensor loading',
        'candidate_chemistry':'native parent atoms and bonds reconstructed by fragmentation_id',
        'native_chemistry_sidecars':'KLIFS_{train,val,test}_chemistry.jsonl',
        'all_rows_graph_validated':True,'all_source_row_ids_and_splits_preserved':True,
        'pockets':'byte-identical source whole-residue pocket arrays',
        'mapping_module_sha256':sourcecode,'preparation_script_sha256':sha(Path(__file__)),
        'audit_rows_sha256':sha(rows_path)}
    report['prepared_splits']=reports
    write_json(out/'dataset_report.json',report)
    files=sorted(p for p in out.iterdir() if p.is_file())
    with (out/'files.sha256.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for path in files:writer.writerow({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
    write_json(out/'PREPARED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'splits':reports,
        'dataset_manifest_sha256':sha(out/'files.sha256.csv'),'audit_summary_sha256':sha(args.audit_dir/(args.audit_name+'.json'))})
    print(json.dumps({'prepared':str(out),'splits':reports}),flush=True)

if __name__=='__main__':main()
