"""2019-2022 -> 2023 and 2019-2023 -> 2024 model selection; 2025 temporal test.

2026 prediction is blocked if target inputs are identical to 2025 source jobs.
"""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from inventory import ROOT
from features import build_features,CAT,NUM,HISTORY_NUM,ALL_FEATURES

def estimator(name):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    transform=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=10),CAT),
                                 ('numeric',numeric,NUM+HISTORY_NUM)])
    if name=='ridge': model=Ridge(alpha=100,solver='lsqr')
    elif name=='random_forest': model=RandomForestRegressor(n_estimators=100,max_depth=16,min_samples_leaf=10,max_features=0.7,n_jobs=4,random_state=2026)
    elif name=='lightgbm': model=LGBMRegressor(n_estimators=400,learning_rate=0.04,num_leaves=31,max_depth=-1,min_child_samples=40,
                                             reg_lambda=5,verbosity=-1,n_jobs=4,random_state=2026)
    else: raise ValueError(name)
    return Pipeline([('preprocess',transform),('model',model)])

def metrics(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))

def baseline(train,test):
    latest=train[train.year==train.year.max()]
    med=latest.groupby(['exam_category','score_group']).target.median()
    return np.array([med.get((r.exam_category,r.score_group),latest.target.median()) for r in test.itertuples()])

def run():
    out=ROOT/'results'; (out/'models').mkdir(exist_ok=True)
    panel=pd.read_csv(ROOT/'data/processed/job_panel.csv',dtype={'部门代码':str,'职位代码':str},low_memory=False)
    f=build_features(panel)
    f.to_csv(ROOT/'data/processed/model_features.csv',index=False,encoding='utf-8-sig')
    history=f[f.eligible].copy()
    candidates=['ridge','random_forest','lightgbm']
    protocol={'target':'岗位最低进面分数（原文件口径）','training_years':[2019,2020,2021,2022,2023,2024,2025],
              'selection_validation_years':[2023,2024],'temporal_test_year':2025,'prediction_year':2026,
              'no_random_split':True,'candidate_models':candidates,'selection_metric':'equal-year average MAE',
              'features':ALL_FEATURES,'annual_difficulty_known':False,
              'history_window_years':3,'labels_from_2026_read':False,
              'interval_rule':'absolute 2025 test residual 90th percentile by score_group (n>=100), descriptive not guaranteed coverage'}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for year in [2023,2024]:
        train=history[history.year<year]; test=history[history.year==year]
        for name in ['category_median']+candidates:
            print(f'validation {year} {name}: train={len(train)}, valid={len(test)}',flush=True)
            if name=='category_median': pred=baseline(train,test)
            else:
                model=estimator(name); model.fit(train[ALL_FEATURES],train.target); pred=model.predict(test[ALL_FEATURES])
            row=dict(stage='selection',year=year,model=name,train_n=len(train),test_n=len(test),**metrics(test.target,pred))
            rows.append(row); print(row,flush=True)
            pd.DataFrame(rows).to_csv(out/'model_comparison.csv',index=False,encoding='utf-8-sig')
    ranking=pd.DataFrame(rows).groupby('model').mae.mean().sort_values()
    # Keep baseline eligible for selection; complex models must earn their use.
    chosen=ranking.index[0]
    print('chosen',chosen,ranking.to_dict(),flush=True)
    train=history[history.year<2025]; test=history[history.year==2025]
    if chosen=='category_median': pred=baseline(train,test); model=None
    else:
        model=estimator(chosen); model.fit(train[ALL_FEATURES],train.target); pred=model.predict(test[ALL_FEATURES])
        joblib.dump(model,out/'models/temporal_test_model.joblib')
    row=dict(stage='temporal_test',year=2025,model=chosen,train_n=len(train),test_n=len(test),**metrics(test.target,pred))
    rows.append(row)
    basepred=baseline(train,test)
    rows.append(dict(stage='temporal_test',year=2025,model='category_median',train_n=len(train),test_n=len(test),**metrics(test.target,basepred)))
    results=panel.loc[test.index,['year','部门代码','职位代码','部门名称','用人司局','招考职位','工作地点','学历','招考人数','min_interview_score']].copy()
    results['score_group']=test.score_group
    results['predicted_score']=pred
    results['error']=pred-test.target.to_numpy()
    results['absolute_error']=np.abs(results.error)
    results.to_csv(out/'2025_temporal_test_predictions.csv',index=False,encoding='utf-8-sig')
    diagnostics=[]
    for col in ['score_group','province','department_family']:
        for value,index in test.groupby(col).groups.items():
            pos=test.index.get_indexer(index)
            diagnostics.append(dict(group_field=col,group=value,n=len(index),**metrics(test.loc[index].target,pred[pos])))
    pd.DataFrame(diagnostics).to_csv(out/'2025_group_metrics.csv',index=False,encoding='utf-8-sig')
    q=results.groupby('score_group').absolute_error.agg(n='size',radius=lambda s:np.quantile(s,0.9,method='higher'))
    q.to_csv(out/'2025_error_reference.csv',encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out/'model_comparison.csv',index=False,encoding='utf-8-sig')
    print('final fit',chosen,len(history),flush=True)
    if chosen=='category_median': final_model=dict(kind='category_median',history=history)
    else:
        final_model=estimator(chosen); final_model.fit(history[ALL_FEATURES],history.target)
    joblib.dump(final_model,out/'models/final_model_2019_2025.joblib')
    audit=json.loads((ROOT/'data/processed/data_audit.json').read_text(encoding='utf-8'))
    blocked=audit['target_2026_identical_to_2025']==audit['target_2026_jobs']
    status={'chosen_model':chosen,'validation_mean_mae':float(ranking[chosen]),'test_2025':row,
            'final_training_n':len(history),'prediction_2026_status':'blocked_input_year' if blocked else 'ready',
            'reason':'2026-labeled inputs are all identical to 2025 jobs' if blocked else '',
            'model_sha256':hashlib.sha256((out/'models/final_model_2019_2025.joblib').read_bytes()).hexdigest()}
    (out/'run_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    if (ROOT/'data/verified_target_manifest.json').exists():
        from model import predict_verified
        predict_verified()
    elif blocked:
        print('No 2026 predictions produced: input-year validation failed.',flush=True)
    else:
        target=f[f.year==2026]
        predictions=baseline(history,target) if chosen=='category_median' else final_model.predict(target[ALL_FEATURES])
        export=panel.loc[target.index].drop(columns=['min_interview_score'],errors='ignore').copy()
        export['predicted_score']=predictions
        export.to_csv(out/'2026_prediction.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__': run()
