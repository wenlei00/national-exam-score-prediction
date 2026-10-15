"""Explicit input allowlist. Never opens 2026 observed scores or registrations."""
from pathlib import Path
import hashlib
import json
import shutil
from settings import config, resolve_path, input_path

ROOT = Path(__file__).resolve().parents[1]

def inputs():
    records = []
    for item in config()['historical_inputs']:
        year = int(item['year'])
        if year > 2025 or item['kind'] not in ('jobs', 'scores'):
            raise ValueError('Frozen workflow accepts historical labels only through 2025')
        records.append((year, item['kind'], resolve_path(item['path'])))
    records.append((2026, 'jobs', input_path('target_2026')))
    return records

def ingest():
    output = ROOT / 'data/raw'
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    manifest = []
    for year, kind, source in inputs():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = output / source.name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing raw file differs: {target}')
        if not target.exists():
            shutil.copy2(source, target)
        manifest.append(dict(year=year, kind=kind, source=str(source), path=str(target), sha256=digest))
    (ROOT / 'data/input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def inspect():
    import pandas as pd
    manifest = ingest()
    report = []
    for record in manifest:
        book = pd.ExcelFile(record['path'])
        for sheet in book.sheet_names:
            top = pd.read_excel(book, sheet_name=sheet, header=None, nrows=6, dtype=str).fillna('')
            info = dict(year=record['year'], kind=record['kind'], file=Path(record['path']).name,
                        sheet=sheet, top=top.values.tolist())
            report.append(info)
            print(json.dumps(info, ensure_ascii=False), flush=True)
    (ROOT / 'data/processed/inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    inspect()
