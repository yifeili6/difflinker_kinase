# Dataset reconstruction

The archive contains the native structures, curated split assignments, fragmentation specifications and coordinate preparation code needed to reconstruct the model inputs. Run the commands below from the archive root. Preparation uses CPUs; `--build-workers 3` exports the three splits concurrently.

## 1. Create the recorded environment

Choose a new absolute directory for the environment and create it with the supplied package specifications:

```bash
export KINASE_REBUILD_ENV="$PWD/reconstruction_env"
bash scripts/create_environment.sh "$KINASE_REBUILD_ENV"
export PATH="$KINASE_REBUILD_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$KINASE_REBUILD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/vendor/DiffLinker"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

The preparation environment uses Python 3.10 and RDKit 2022.03.5. `environment/explicit-lock.txt` records exact Conda packages, and the pip requirement files identify the archived wheels by hash. An existing environment created from these specifications can be used by setting `KINASE_REBUILD_ENV` to its prefix and applying the exports above.

## 2. Locate the source records

The reconstruction command takes the archive root as `--source`. It reads these relative paths:

| Path | Content used during reconstruction |
| --- | --- |
| `data/manifests/klifs_v1/contexts.csv` | Structure identifiers, ligand paths, source hashes and split assignments |
| `data/raw/klifs/klifs_2025-10-22_cached_2025-11-12/` | Native ligand MOL2 files referenced by the context manifest |
| `data/processed/klifs_v1/KLIFS_{train,val,test}_table.csv` | Ordered row identifiers, MMPA fragment/linker patterns and attachment indices |
| `data/processed/klifs_v1/KLIFS_{train,val,test}_{frag,link}.sdf` | Published coordinate records used to compare atom mappings |
| `data/processed/klifs_v1/KLIFS_{train,val,test}_pockets.pkl` | Row-aligned protein pocket arrays |
| `data/processed/klifs_v1/dataset_report.json` | Source curation, split and preparation metadata |
| `configs/data_klifs_v1.yml` | Data configuration recorded in the preparation provenance |

`provenance/curated_source_files.jsonl` lists source file sizes and SHA-256 values. The context manifest and row tables identify each native ligand by SHA-256. Reconstruction verifies each ligand against its recorded hash before using it. The snapshot metadata records the KLIFS export date of 2025-10-22 and local cache date of 2025-11-12.

The prepared dataset has the following population:

| Split | Fragmentation rows | Native structures | PDB entries |
| --- | ---: | ---: | ---: |
| Train | 34,256 | 5,696 | 2,807 |
| Validation | 3,701 | 665 | 339 |
| Test | 5,292 | 788 | 381 |
| Total | 43,249 | 7,149 | 3,527 |

Each row preserves its UUID, fragmentation identifier and split assignment. One native structure can contribute several two-cut fragmentation patterns.

## 3. Reconstruct the coordinate files

Use a new output directory under the archive root and a new provenance directory. The command below derives a unique directory name from the current UTC time:

```bash
export KINASE_REBUILD_ID="reconstruction_$(date -u +%Y%m%dT%H%M%S%NZ)"
export KINASE_REBUILD_DATA="data/processed/$KINASE_REBUILD_ID"
export KINASE_REBUILD_PROVENANCE="provenance/$KINASE_REBUILD_ID"

python -B scripts/rebuild_corrected_dataset.py \
  --source . \
  --output "$KINASE_REBUILD_DATA" \
  --audit-dir "$KINASE_REBUILD_PROVENANCE" \
  --audit-name native_graph \
  --build --build-workers 3
```

The command first verifies every row, then writes the prepared files. `--build-workers 1` runs the split exports sequentially. The output directory must be new and located within the archive root. Verification records use new filenames in the specified provenance directory.

For each MMPA pattern, `src/kinase_gap/fragmentation.py` finds a complete, disjoint assignment of parent atoms to two retained fragments and one linker. The accepted assignment has exactly two native bonds joining the linker to the two fragments. The attachment atoms correspond to the MMPA dummy-neighbor endpoints. Matching includes equivalent assignments arising from molecular symmetry. Reassembling the components with the two original bonds verifies the native chemical identity, including isotope, charge and stereochemistry.

Fragment and linker coordinates retain the native atom coordinates and the order used by the model. Attachment indices are zero-based indices in the combined retained-fragment molecule. Protein pocket arrays are copied byte-for-byte from the published prepared dataset.

## 4. Verify the reconstructed dataset

Run the independent serialized-file validator on the new directory:

```bash
python -B scripts/validate_prepared_graphs.py \
  --data "$KINASE_REBUILD_DATA" \
  --output "$KINASE_REBUILD_PROVENANCE/prepared_graph_validation.json"
```

It checks all three splits for full-ligand chemical identity, coordinate and element correspondence, complete atom partitions, two fragment-linker bonds and attachment endpoints. The expected result is `passed: true`, 43,249 rows and zero failed checks.

Verify the generated checksum manifest:

```bash
python -B - <<'PY'
import csv
import hashlib
import os
from pathlib import Path

directory = Path(os.environ['KINASE_REBUILD_DATA'])
with (directory / 'files.sha256.csv').open() as handle:
    records = list(csv.DictReader(handle))
for record in records:
    path = directory / record['path']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    assert path.stat().st_size == int(record['bytes']), path
    assert digest.hexdigest() == record['sha256'], path
print(f'Verified {len(records)} prepared files')
PY
```

## 5. Build CPU tensor inputs

Create tensor caches from the reconstructed files with the archived DiffLinker dataset loader:

```bash
python -B - <<'PY'
import json
import os
from pathlib import Path
from src.datasets import MOADDataset

data = Path(os.environ['KINASE_REBUILD_DATA'])
counts = {}
for split in ('train', 'val', 'test'):
    dataset = MOADDataset(data_path=str(data), prefix=f'KLIFS_{split}.full', device='cpu')
    assert all(int((row['anchors'] == 1).sum()) == 2 for row in dataset)
    assert all(bool((row['anchors'] <= row['fragment_only_mask']).all()) for row in dataset)
    assert all(row['positions'].device.type == 'cpu' for row in dataset)
    counts[split] = len(dataset)
    del dataset
record = Path(os.environ['KINASE_REBUILD_PROVENANCE']) / 'tensor_cache_counts.json'
with record.open('x') as handle:
    json.dump(counts, handle, indent=2)
print(json.dumps(counts))
PY
```

The expected counts are `train: 34256`, `val: 3701` and `test: 5292`. For a training experiment using these reconstructed files, create a new training configuration with its `data` field set to the relative path stored in `KINASE_REBUILD_DATA`. [Model usage](MODEL_USAGE.md) describes checkpoint loading, fine-tuning and generation.

## Output records and molecular identity

Each split produces an ordered CSV table; complete-parent, fragment and linker SDF files; a pocket pickle; and a chemistry JSONL sidecar. All files share the row order in the split table. Complete-parent SDF records contain native stereo properties. Fragment and linker SDF records use chemically capped components as coordinate and element carriers for tensor loading.

`KLIFS_{train,val,test}_chemistry.jsonl` preserves the authoritative native parent as an RDKit all-properties binary, encoded in base64. It also records native atom indices, attachment flags, atom properties, bond properties and neighbor order for each component. Read an authoritative parent with the recorded RDKit version:

```python
import base64
import json
from rdkit import Chem

with open('data/processed/klifs_v1/KLIFS_test_chemistry.jsonl') as handle:
    row = json.loads(next(handle))
parent = Chem.Mol(base64.b64decode(row['parent_molecule_binary_base64']))
```

Candidate assembly derives retained-fragment chemistry from this native parent. The exact native graph defines atom charge, isotope, hydrogen representation and stereochemistry; the SDF records supply the corresponding pose and ordered tensor inputs. [Data and evaluation](DATA_AND_EVALUATION.md) describes the downstream chemical and structural checks.

`dataset_report.json` retains the original curation timestamp in `created_at_utc`. `PREPARED.json` records the current coordinate preparation time in `created_utc`. The preparation records include hashes of the mapping module, preparation script, verification rows and prepared files. A new reconstruction receives its own time and provenance hashes while preserving the row identities, split membership and native coordinate inputs.
