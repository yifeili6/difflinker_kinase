from pathlib import Path
import argparse
import csv
import re
import sys
from types import SimpleNamespace
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinase_gap.evaluate import load_dependencies, SDFRecord, evaluate_record
from kinase_gap.structural_bundle import parse_sdf_bytes

def local(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use a path relative to the package root')
    return ROOT / path

def sdf_records(source):
    current = []
    index = 0
    with source.open() as stream:
        for line in stream:
            if line.strip() != '$$$$':
                current.append(line)
                continue
            text = ''.join(current)
            lines = text.splitlines()
            properties = {}
            for row, value in enumerate(lines):
                match = re.match(r'^>\s*<([^>]+)>', value)
                if match:
                    properties[match.group(1)] = lines[row + 1].strip() if row + 1 < len(lines) else ''
            yield SDFRecord(index=index, title=lines[0].strip(), properties=properties, mol_block=text)
            index += 1
            current = []
    if ''.join(current).strip():
        raise ValueError('SDF contains an unterminated final record')

def main():
    parser = argparse.ArgumentParser(description='Evaluate saved molecules using the paper\'s geometry, Lipinski, MOSES and descriptor settings.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = local(args.input), local(args.output)
    if target.exists():
        raise FileExistsError(args.output)
    config = SimpleNamespace(posebusters_config='mol', pains_smarts=ROOT / 'vendor/DiffLinker/resources/wehi_pains.csv', pains_catalog=None)
    dependencies = load_dependencies(config)
    records = sdf_records(source)
    result = []
    for index, record in enumerate(records):
        props = dict(record.properties)
        source_row = {'candidate_id':props.get('candidate_id', 'molecule_' + str(index + 1)), 'in_memory_sanitize_pass':props.get('valid', ''), 'smiles':props.get('smiles', '')}
        if 'kinase_native_molecule_binary_base64' in props:
            payload = record.mol_block.rstrip() + '\n'
            for key, value in props.items():
                payload += '\n>  <' + key + '>\n' + value + '\n'
            payload += '\n$$$$\n'
            molecule = parse_sdf_bytes(payload.encode(), label=source_row['candidate_id'])[0]
            from dataclasses import replace
            record = replace(record, authoritative_molecule=molecule)
        result.append(evaluate_record(source_row, record, dependencies))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print('Wrote ' + str(len(result)) + ' molecule records to ' + args.output)

if __name__ == '__main__':
    main()
