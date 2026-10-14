"""Seal and verify the exact source files used for fine-tuning and its preflight."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FIXED = (
    "src/kinase_gap/__init__.py", "src/kinase_gap/train.py", "src/kinase_gap/hpo.py",
    "configs/finetune_anchors.yml", "configs/hpo_grid.yml", "configs/data_klifs_v1.yml",
    "scripts/preflight.py", "scripts/source_manifest.py", "scripts/build_tensor_caches.py",
    "scripts/validate_prepared_graphs.py", ".kinase-revision",
    "slurm/runtime.sh", "slurm/preflight.slurm", "slurm/hpo.slurm",
    "vendor/DiffLinker/UPSTREAM_COMMIT", "vendor/DiffLinker/LICENSE",
)

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def paths():
    return sorted(set(FIXED) | {str(p.relative_to(ROOT)) for p in (ROOT/"vendor/DiffLinker/src").rglob("*")
                               if p.is_file() and "__pycache__" not in p.parts})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = ROOT/"artifacts/source.sha256"
    content = "".join(f"{digest(ROOT/p)}  {p}\n" for p in paths())
    if args.write:
        with manifest.open("x") as handle:
            handle.write(content)
        with tarfile.open(ROOT/"artifacts/training-source.tar.gz", "x:gz") as archive:
            for path in paths():
                archive.add(ROOT/path, arcname=path, recursive=False)
    else:
        assert manifest.read_text() == content, "Training source has changed since it was sealed"
    print(json.dumps({"source_files": len(paths()), "sha256": digest(manifest), "verified": True}))

if __name__ == "__main__":
    main()
