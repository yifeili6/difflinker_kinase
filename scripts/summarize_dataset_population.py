import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem, rdBase

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from kinase_gap.fragmentation import canonical_isomeric_identity
    (tables, summary, identities, seen) = ([], {}, {}, {})
    for (split, prefix) in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        path = args.dataset_dir / f'KLIFS_{prefix}_table.csv'
        table = pd.read_csv(path, keep_default_na=False, dtype=str)
        if table.uuid.duplicated().any() or table.fragmentation_id.duplicated().any():
            raise ValueError(f'Repeated row identifier: {split}')
        for value in table.canonical_smiles.unique():
            if value not in seen:
                molecule = Chem.MolFromSmiles(value)
                if molecule is None:
                    raise ValueError(f'Unparsable dataset identity: {split}')
                seen[value] = canonical_isomeric_identity(molecule)
        table['normalized_identity'] = table.canonical_smiles.map(seen)
        identities[split] = set(table.normalized_identity)
        summary[split] = {'fragmentation_rows': len(table), 'structures': table.structure_id.nunique(), 'pdbs': table.pdb.str.lower().nunique(), 'canonical_smiles_strings': table.canonical_smiles.nunique(), 'canonical_isomeric_molecules': table.normalized_identity.nunique(), 'kinases': table.kinase.nunique(), 'species_kinase_pairs': len(table[['species', 'kinase']].drop_duplicates()), 'scaffolds': table.murcko_scaffold.nunique(), 'source': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    result = {'schema_version': 1, 'population': 'Emitted fragmentation rows and their source structures', 'identity': 'RDKit SMILES parse followed by canonical isomeric normalization', 'rdkit_version': rdBase.rdkitVersion, 'splits': summary, 'total': {'fragmentation_rows': len(combined), 'structures': combined.structure_id.nunique(), 'pdbs': combined.pdb.str.lower().nunique(), 'canonical_isomeric_molecules': combined.normalized_identity.nunique(), 'kinases': combined.kinase.nunique(), 'species_kinase_pairs': len(combined[['species', 'kinase']].drop_duplicates())}, 'identity_intersections': {f'{a}_{b}': len(identities[a] & identities[b]) for (a, b) in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result['splits']))
if __name__ == '__main__':
    main()
