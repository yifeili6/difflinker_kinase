from pathlib import Path
import argparse
import datetime
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[1]

def local(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use a path relative to the package root')
    return ROOT / path

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='Record a downloaded KLIFS structure package for dataset preparation.')
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--config', default='configs/data_klifs_v1.yml')
    parser.add_argument('--output', default='configs/data_local.yml')
    parser.add_argument('--export-date', default=datetime.date.today().isoformat())
    args = parser.parse_args()
    snapshot = local(args.snapshot)
    output = local(args.output)
    if output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(local(args.config).read_text())
    files = {'archive':'KLIFS_download.zip', 'overview':'overview.csv', 'export':'KLIFS_export.csv'}
    config['snapshot_dir'] = args.snapshot
    config['snapshot'] = {'database_export_date':args.export_date, 'local_cache_date':datetime.date.today().isoformat(), **{key + '_sha256':sha256(snapshot / name) for key,name in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(args.output)

if __name__ == '__main__':
    main()
