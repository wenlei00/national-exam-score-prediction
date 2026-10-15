"""Portable configuration; local paths are never committed."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def default_config():
    history = []
    for year in range(2019, 2026):
        suffix = '职位表.xls' if year == 2022 else '岗位表.xls'
        history.append(dict(year=year, kind='jobs', path=f'data/raw/{year}年国考{suffix}'))
        history.append(dict(year=year, kind='scores', path=f'data/raw/{year}国考岗位分数线.xlsx'))
    return dict(historical_inputs=history,
                target_2026='data/raw/中央机关及其直属机构2026年度考试录用公务员招考简章_北体转载.xls',
                actual_2026='data/raw/2026面试名单.xlsx',
                target_2027='data/raw/中央机关及其直属机构2027年度考试录用公务员招考简章.xls')

def config():
    configured = os.environ.get('EXAM_CONFIG')
    path = Path(configured) if configured else ROOT / 'config.local.json'
    result = default_config()
    if path.exists():
        result.update(json.loads(path.read_text(encoding='utf-8')))
    elif configured:
        raise FileNotFoundError(path)
    return result

def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def input_path(name):
    return resolve_path(config()[name])
