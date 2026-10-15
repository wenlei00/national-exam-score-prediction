"""Meaningful invariants: no outcome leakage, correct joins and complete prediction keys."""
import json
import numpy as np
import pandas as pd
from inventory import ROOT
from features import build_features,ALL_FEATURES,HISTORY_NUM

def verify():
    panel=pd.read_csv(ROOT/'data/processed/job_panel.csv',dtype={'部门代码':str,'职位代码':str},low_memory=False)
    assert not panel.duplicated(['year','部门代码','职位代码']).any()
    assert panel.loc[panel.train_eligible,'year'].max()==2025
    assert not panel.loc[panel.year==2026,'min_interview_score'].notna().any()
    # Change all labels in 2025; features for predicting 2025 must remain unchanged.
    small=panel[panel.year<=2025].groupby('year').head(150).reset_index(drop=True)
    original=build_features(small)
    perturbed=small.copy()
    perturbed.loc[perturbed.year==2025,'min_interview_score']=999
    changed=build_features(perturbed)
    pd.testing.assert_frame_equal(original.loc[original.year==2025,ALL_FEATURES],changed.loc[changed.year==2025,ALL_FEATURES])
    assert original.loc[original.year==2019,HISTORY_NUM[::2]].isna().all().all()
    result=pd.read_csv(ROOT/'results/2026_prediction.csv',dtype={'部门代码':str,'职位代码':str})
    target=pd.read_csv(ROOT/'data/processed/jobs_2026_verified.csv',dtype={'部门代码':str,'职位代码':str})
    assert len(result)==len(target)==20714
    assert set(map(tuple,result[['部门代码','职位代码']].values))==set(map(tuple,target[['部门代码','职位代码']].values))
    assert result['预测最低进面分'].notna().all()
    assert np.isfinite(result['预测最低进面分']).all()
    assert result['部门代码'].str.len().eq(6).all()
    assert result['职位代码'].str.len().eq(12).all()
    tests=['unique annual job keys','2026 outcomes absent','same-year target mutation cannot change input features',
           '2019 has no future historical features','prediction keys exactly match verified 2026 input',
           'all predictions finite','identifier lengths preserved']
    (ROOT/'results/verification.json').write_text(json.dumps({'passed':tests,'count':len(tests)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS:', '; '.join(tests))

if __name__=='__main__': verify()
