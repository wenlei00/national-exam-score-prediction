"""Apply frozen model to verified target jobs. Never reads target-year results."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from inventory import ROOT
from features import build_features,ALL_FEATURES,CAT,NUM,HISTORY_NUM
from train import baseline

def predict_verified():
    history=pd.read_csv(ROOT/'data/processed/job_panel.csv',dtype={'部门代码':str,'职位代码':str},low_memory=False)
    history=history[history.year<=2025].copy()
    target=pd.read_csv(ROOT/'data/processed/jobs_2026_verified.csv',dtype={'部门代码':str,'职位代码':str},low_memory=False).fillna('')
    target['min_interview_score']=np.nan; target['train_eligible']=False
    combined=pd.concat([history,target],ignore_index=True)
    f=build_features(combined)
    f.to_csv(ROOT/'data/processed/model_features.csv',index=False,encoding='utf-8-sig')
    x=f[f.year==2026].copy()
    model=joblib.load(ROOT/'results/models/final_model_2019_2025.joblib')
    pred=baseline(f[f.eligible],x) if isinstance(model,dict) else model.predict(x[ALL_FEATURES])
    assert len(pred)==len(target) and np.isfinite(pred).all()
    ref=pd.read_csv(ROOT/'results/2025_error_reference.csv').set_index('score_group')
    raw={}
    for c in ['部门代码','职位代码','部门名称','用人司局','招考职位','工作地点','考试类别','职位属性','招考人数','专业','学历','学位','政治面貌','基层工作最低年限','备注']:
        raw[c]=target[c].to_numpy()
    result=pd.DataFrame(raw)
    result.insert(0,'招录年度',2026)
    result['预测最低进面分']=np.round(pred,2)
    result['分数口径分组']=x.score_group.to_numpy()
    radii=np.array([ref.loc[g,'radius'] if g in ref.index and ref.loc[g,'n']>=100 else np.nan for g in x.score_group])
    result['历史误差参考下限']=np.round(pred-radii,2)
    result['历史误差参考上限']=np.round(pred+radii,2)
    result['同司局同类别历史样本数']=x.unit_history_n.astype(int).to_numpy()
    result['同部门同类别历史样本数']=x.department_history_n.astype(int).to_numpy()
    result['提示']=np.where(np.isnan(radii),'同类验证样本不足100条；未提供误差范围','历史误差范围不保证2026覆盖率')
    result.loc[result['同部门同类别历史样本数']==0,'提示']+='；无同部门同类别近期历史样本'
    assert not result.duplicated(['招录年度','部门代码','职位代码']).any()
    path=ROOT/'results/2026_prediction.csv'
    result.to_csv(path,index=False,encoding='utf-8-sig')
    # Save exact unrounded predictions for later external evaluation.
    exact=pd.DataFrame({'year':2026,'department_code':target['部门代码'],'position_code':target['职位代码'],'predicted_score':pred})
    exact.to_csv(ROOT/'results/2026_prediction_exact.csv',index=False,encoding='utf-8-sig')
    x.to_csv(ROOT/'data/processed/features_2026_verified.csv',index=False,encoding='utf-8-sig')
    summary=json.loads((ROOT/'results/run_summary.json').read_text(encoding='utf-8'))
    summary.update(prediction_2026_status='complete',reason='',target_2026_n=len(result),
                   target_source='data/verified_target_manifest.json',prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   target_2026_scores_read=False)
    (ROOT/'results/run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # TreeSHAP contributions sampled from ordinary jobs: avoid mixing different score scales in importance.
    if not isinstance(model,dict) and summary['chosen_model']=='lightgbm':
        sample=x[x.score_group=='普通职位'].sample(n=min(2000,int((x.score_group=='普通职位').sum())),random_state=2026)
        transformed=model.named_steps['preprocess'].transform(sample[ALL_FEATURES])
        contributions=model.named_steps['model'].predict(transformed,pred_contrib=True)
        if hasattr(contributions,'toarray'): contributions=contributions.toarray()
        np.testing.assert_allclose(contributions.sum(axis=1),model.predict(sample[ALL_FEATURES]),rtol=1e-6,atol=1e-6)
        names=model.named_steps['preprocess'].get_feature_names_out()
        grouped={c:np.zeros(len(sample)) for c in ALL_FEATURES}
        for i,name in enumerate(names):
            base=name.split('__',1)[1]
            if base.startswith('missingindicator_'): base=base[len('missingindicator_'):]
            field=next((c for c in sorted(ALL_FEATURES,key=len,reverse=True) if base==c or base.startswith(c+'_')),None)
            if field: grouped[field]+=contributions[:,i]
        importance=pd.DataFrame([dict(feature=c,mean_absolute_shap=float(np.mean(np.abs(v))),sample_n=len(sample)) for c,v in grouped.items()]).sort_values('mean_absolute_shap',ascending=False)
        importance.to_csv(ROOT/'results/ordinary_jobs_shap.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return result

if __name__=='__main__': predict_verified()
