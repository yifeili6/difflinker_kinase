# Kinase DiffLinker source

The package prepares KLIFS protein-ligand contexts, fine-tunes the anchor-conditioned DiffLinker model, generates molecular linkers and evaluates molecular structures. Paths in the supplied configurations resolve relative to the project root containing this package.

| Module | Operation |
| --- | --- |
| `prepare.py`, `fragmentation.py`, `mol2.py` | Native structures, pocket atoms, two-cut partitions and attachment atoms |
| `train.py`, `hpo.py` | Pretrained initialization, fine-tuning and validation-based checkpoint selection |
| `generate.py`, `chemistry_io.py` | Conditional sampling and assembly with native fragment chemistry |
| `evaluate.py`, `identity.py`, `metrics.py` | Chemical identity, filtering and population metrics |
| `report.py`, `structural_bundle.py` | Plot tables and protein-ligand visualization inputs |

Run the entry points from a reproduction workspace using the recorded environment:

```bash
python -m kinase_gap.train --help
python -m kinase_gap.generate --help
python -m kinase_gap.evaluate --help
python -m kinase_gap.metrics --help
```

The training configuration selects the prepared split prefixes and an initialization checkpoint. `init_checkpoint` loads the model weights strictly and starts an optimizer at epoch zero. `--resume-checkpoint` restores the optimizer, epoch and global step from a saved training checkpoint. `max_epochs` specifies the total target epoch for that run.

The fine-tuning model has nine ligand element classes, three context indicators and a 128-dimensional hidden representation. Checkpoint selection uses the recorded minimum, batch-size-weighted validation objective. Configuration files and run metadata specify the seed, precision, deterministic-kernel policy, optimization budget and saved checkpoint identities.

Read [Complete workflow](../../docs/REPRODUCE.md), [Model usage](../../docs/MODEL_USAGE.md) and [Data and evaluation](../../docs/DATA_AND_EVALUATION.md) for the input and output contracts.
