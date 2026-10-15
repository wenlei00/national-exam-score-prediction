"""Primary forecast: reuse the frozen model and exclude all 2026 labels."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from inventory import ROOT
from settings import input_path
from data_processing import clean, identifier, JOB_FIELDS
from features import build_features, ALL_FEATURES

def run():
    out = ROOT / 'results/frozen_2027'
    out.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / 'results/run_summary.json').read_text(encoding='utf-8'))
    path = ROOT / 'results/models/final_model_2019_2025.joblib'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != frozen['model_sha256']:
        raise ValueError('Frozen model hash changed')
    history = pd.read_csv(ROOT / 'data/processed/job_panel.csv', dtype={'部门代码': str, '职位代码': str}, low_memory=False)
    history = history.loc[history.year <= 2025].copy()
    frames = []
    source = input_path('target_2027')
    for sheet in pd.ExcelFile(source).sheet_names:
        top = pd.read_excel(source, sheet_name=sheet, header=None, nrows=5, dtype=str).fillna('')
        if top.empty:
            continue
        header = next(i for i, row in enumerate(top.values.tolist()) if '职位代码' in row)
        d = pd.read_excel(source, sheet_name=sheet, header=header, dtype=str).fillna('').map(clean)
        for col, width in [('部门代码', 6), ('职位代码', 12)]:
            d[col] = d[col].map(lambda value: identifier(value, width))
            if not d[col].str.fullmatch(r'\d{' + str(width) + r'}').all():
                raise ValueError(f'Invalid {col}')
        frames.append(d)
    target = pd.concat(frames, ignore_index=True)
    if target.duplicated(['部门代码', '职位代码']).any():
        raise ValueError('Duplicate target keys')
    target['year'] = 2027
    target['min_interview_score'] = np.nan
    target['train_eligible'] = False
    panel = pd.concat([history, target], ignore_index=True)
    assert not panel.year.eq(2026).any()
    f = build_features(panel)
    future = f.loc[f.year == 2027]
    model = joblib.load(path)
    prediction = model.predict(future[ALL_FEATURES])
    assert np.isfinite(prediction).all() and len(prediction) == len(target)
    result = target[JOB_FIELDS].copy()
    result.insert(0, '招录年度', 2027)
    result['预测最低进面分'] = np.round(prediction, 2)
    result['分数口径分组'] = future.score_group.to_numpy()
    result['模型训练截止年度'] = 2025
    result['历史特征标签截止年度'] = 2025
    result['同部门同类别历史样本数'] = future.department_history_n.astype(int).to_numpy()
    result.to_csv(out / '2027_prediction_frozen.csv', index=False, encoding='utf-8-sig')
    exact = target[['部门代码', '职位代码']].copy()
    exact['predicted_score'] = prediction
    exact.to_csv(out / '2027_prediction_frozen_exact.csv', index=False, encoding='utf-8-sig')
    summary = dict(target_year=2027, target_jobs=len(result), training_label_cutoff=2025,
                   historical_feature_label_cutoff=2025, labels_2026_used=False,
                   labels_2027_used=False, model_sha256=digest,
                   prediction_sha256=hashlib.sha256((out / '2027_prediction_frozen.csv').read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
