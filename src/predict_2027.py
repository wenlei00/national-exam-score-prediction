"""Refit fixed LightGBM method with 2019-2026 labels; predict all supplied 2027 jobs.

Keeps 2026 frozen assets and independent evaluation unchanged. No 2027 outcomes read.
"""
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import joblib
from inventory import ROOT
from data_processing import clean,identifier,JOB_FIELDS
from features import build_features,ALL_FEATURES
from train import estimator
from settings import input_path

SOURCE=input_path('target_2027')
KEY=['部门代码','职位代码']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    output=ROOT/'results/prediction_2027'; output.mkdir(exist_ok=True)
    data=ROOT/'data/processed/2027'; data.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'results/run_summary.json').read_text(encoding='utf-8'))
    protected={ROOT/'results/2026_prediction.csv':frozen['prediction_sha256'],
               ROOT/'results/models/final_model_2019_2025.joblib':frozen['model_sha256']}
    for p,v in protected.items(): assert sha(p)==v,'Old frozen asset changed'
    target_path=ROOT/'data/raw'/SOURCE.name
    if target_path.exists(): assert sha(target_path)==sha(SOURCE),'Existing 2027 source differs'
    else: shutil.copy2(SOURCE,target_path)
    frames=[]; sheets=[]; dictionary=[]
    for sheet in pd.ExcelFile(target_path).sheet_names:
        top=pd.read_excel(target_path,sheet_name=sheet,header=None,nrows=5,dtype=str).fillna('')
        if top.empty:
            sheets.append(dict(sheet=sheet,rows=0,status='empty_sheet_skipped')); continue
        header=next((i for i,row in enumerate(top.values.tolist()) if '职位代码' in row),None)
        if header is None: raise ValueError(f'Unexpected nonempty sheet: {sheet}')
        d=pd.read_excel(target_path,sheet_name=sheet,header=header,dtype=str).fillna('').map(clean)
        d.columns=[clean(c) for c in d.columns]
        assert all(c in d.columns for c in JOB_FIELDS)
        for c,width in [('部门代码',6),('职位代码',12)]: d[c]=d[c].map(lambda v:identifier(v,width))
        invalid=~(d['部门代码'].str.fullmatch(r'\d{6}')&d['职位代码'].str.fullmatch(r'\d{12}'))
        assert not invalid.any(),f'Invalid keys: {sheet}'
        assert pd.to_numeric(d['招考人数'],errors='coerce').gt(0).all()
        for field in JOB_FIELDS:
            dictionary.append(dict(sheet=sheet,field=field,rows=len(d),missing=int(d[field].eq('').sum()),unique=int(d[field].nunique())))
        d['year']=2027; d['source_file']=SOURCE.name; d['source_sheet']=sheet
        d['source_row']=np.arange(len(d))+header+2
        frames.append(d); sheets.append(dict(sheet=sheet,rows=len(d),status='loaded'))
    target=pd.concat(frames,ignore_index=True)
    assert not target.duplicated(KEY).any(),'Duplicate target keys need review'
    target['min_interview_score']=np.nan; target['train_eligible']=False
    target.to_csv(data/'jobs_2027.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dictionary).to_csv(data/'data_dictionary_2027.csv',index=False,encoding='utf-8-sig')
    history=pd.read_csv(ROOT/'data/processed/job_panel.csv',dtype={c:str for c in KEY},low_memory=False)
    history=history[history.year<=2025].copy()
    jobs2026=pd.read_csv(ROOT/'data/processed/jobs_2026_verified.csv',dtype={c:str for c in KEY},low_memory=False)
    actual=pd.read_csv(ROOT/'results/evaluation_2026/actual_scores_by_job.csv',dtype={c:str for c in KEY})
    joined=jobs2026.merge(actual,on=KEY,how='left',validate='one_to_one')
    names=joined['部门名称'].eq(joined['实际部门名称'])&joined['用人司局'].eq(joined['实际用人司局'])&joined['招考职位'].eq(joined['实际职位名称'])
    joined['min_interview_score']=joined['实际最低进面分']
    joined['train_eligible']=joined['标签有效'].eq(True)&names&joined['实际最低进面分'].notna()
    joined['year']=2026
    combined=pd.concat([history,joined,target],ignore_index=True)
    assert not combined.duplicated(['year']+KEY).any()
    combined.to_csv(data/'panel_2019_2027.csv',index=False,encoding='utf-8-sig')
    f=build_features(combined)
    train=f[f.eligible].copy(); future=f[f.year==2027].copy()
    assert train.year.max()==2026 and future.target.isna().all() and not future.eligible.any()
    assert len(future)==len(target)
    f.to_csv(data/'model_features_2019_2027.csv',index=False,encoding='utf-8-sig')
    model=estimator('lightgbm')
    print('Refit fixed LightGBM:',len(train),'training jobs;',len(future),'target jobs',flush=True)
    model.fit(train[ALL_FEATURES],train.target)
    values=model.predict(future[ALL_FEATURES])
    assert np.isfinite(values).all()
    model_path=output/'model_2019_2026.joblib'; joblib.dump(model,model_path)
    previous_eval=pd.read_csv(ROOT/'results/evaluation_2026/2026_prediction_vs_actual.csv',low_memory=False)
    ref=previous_eval.groupby('分数口径分组')['绝对误差'].agg(n='size',radius=lambda s:np.quantile(s,.9,method='higher'))
    radii=np.array([ref.loc[g,'radius'] if g in ref.index and ref.loc[g,'n']>=100 else np.nan for g in future.score_group])
    export=target[KEY+['部门名称','用人司局','招考职位','工作地点','考试类别','职位属性','招考人数','专业','学历','学位','政治面貌','基层工作最低年限','服务基层项目工作经历','面试人员比例','备注']].copy()
    export.insert(0,'招录年度',2027)
    export['省份']=future.province.to_numpy(); export['部门系统']=future.department_family.to_numpy()
    export['分数口径分组']=future.score_group.to_numpy()
    export['预测最低进面分']=np.round(values,2)
    export['历史误差参考下限']=np.round(values-radii,2); export['历史误差参考上限']=np.round(values+radii,2)
    export['同司局同类别历史样本数']=future.unit_history_n.astype(int).to_numpy()
    export['同部门同类别历史样本数']=future.department_history_n.astype(int).to_numpy()
    export['提示']=np.where(np.isnan(radii),'同类独立检验样本不足100；未生成参考范围','参考范围来自2026旧模型误差，不保证2027覆盖率')
    export.loc[export['同部门同类别历史样本数']==0,'提示']+='；无同部门同类别近期历史样本'
    export.to_csv(output/'2027_prediction.csv',index=False,encoding='utf-8-sig')
    exact=target[KEY].copy(); exact.insert(0,'year',2027); exact['predicted_score']=values
    exact.to_csv(output/'2027_prediction_exact.csv',index=False,encoding='utf-8-sig')
    # Summary by scale first; avoid averaging unlike score definitions together.
    overview=export.groupby(['分数口径分组','省份']).agg(岗位数=('职位代码','size'),
        计划招录人数=('招考人数',lambda s:pd.to_numeric(s).sum()),预测分中位数=('预测最低进面分','median'),
        最低预测分=('预测最低进面分','min'),最高预测分=('预测最低进面分','max')).reset_index()
    overview.to_csv(output/'2027_province_summary.csv',index=False,encoding='utf-8-sig')
    source2026=jobs2026.copy().fillna('').map(clean)
    compare=target.fillna('').merge(source2026,on=KEY,suffixes=('_2027','_2026'),how='left',indicator=True)
    compare['core_identical']=compare.apply(lambda r:all(r[c+'_2027']==r[c+'_2026'] for c in JOB_FIELDS if c not in KEY),axis=1)
    compare[KEY+['_merge','core_identical']].to_csv(data/'2027_vs_2026_audit.csv',index=False,encoding='utf-8-sig')
    assert compare.core_identical.sum()<len(target)*.5,'Target looks like wrong year'
    # Verify same-year outcome mutations cannot alter 2027 feature inputs.
    small=combined.groupby('year').head(80).reset_index(drop=True)
    original=build_features(small)
    changed=small.copy(); changed.loc[changed.year==2027,'min_interview_score']=999
    altered=build_features(changed)
    pd.testing.assert_frame_equal(original.loc[original.year==2027,ALL_FEATURES],altered.loc[altered.year==2027,ALL_FEATURES])
    for p,v in protected.items(): assert sha(p)==v,'Frozen 2026 asset changed during refit'
    readback=pd.read_csv(output/'2027_prediction.csv',dtype={c:str for c in KEY})
    assert len(readback)==len(target) and not readback.duplicated(KEY).any()
    assert set(map(tuple,readback[KEY].values))==set(map(tuple,target[KEY].values))
    assert readback['预测最低进面分'].notna().all()
    years=train.groupby('year').size().to_dict()
    summary=dict(target_year=2027,source=str(SOURCE),source_sha256=sha(SOURCE),sheets=sheets,
        target_jobs=len(target),target_recruits=int(pd.to_numeric(target['招考人数']).sum()),
        core_identical_to_2026=int(compare.core_identical.sum()),training_n=len(train),
        training_by_year={str(k):int(v) for k,v in years.items()},model='LightGBM',
        parameter_source='same estimator(lightgbm) as previous frozen method; no new tuning',
        prediction_sha256=sha(output/'2027_prediction.csv'),model_sha256=sha(model_path),
        no_2027_outcomes_read=True,original_2026_assets_unchanged=True,
        no_recent_department_history=int((export['同部门同类别历史样本数']==0).sum()),
        error_reference='2026 independent frozen-model absolute error 90th percentile, groups n>=100',
        note='2026 now included in training; earlier independent 2026 evaluation belongs to earlier model only')
    (output/'run_summary_2027.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# 2027 年度全部岗位进面分预测

已对用户提供的招考简章中全部 {len(target):,} 个岗位（计划招录 {summary['target_recruits']:,} 人）生成预测。
范围为文件包含的岗位；未另行断言该文件覆盖之后的公告调整。原文件含 4 个有数据工作表及 1 个空白 Sheet1，空白页已跳过。
本次目标是 2027 年度招录。岗位年度按用户提供的文件及备注中的年度标记识别。与 2026 年相比，核心字段完全一致的岗位只有 {summary['core_identical_to_2026']} 个，没有出现上一次学历筛选表整表误标的情况。

## 模型与数据

沿用此前按年度验证选出的 LightGBM 算法、参数和特征处理，加入已核验的 2026 年首批进面结果后重新训练，不进行新一轮调参。
训练共 {len(train):,} 条年度岗位记录，覆盖 2019—2026 年，其中 2026 年可用记录 {years[2026]:,} 条。
报名人数和过审人数未作为输入，2027 年真实进面结果未读取。

历史同部门、同司局、同城市和同考试类别分数特征严格只使用预测年度之前三年可用数据；2027 年历史特征因此可使用 2024—2026 年结果。
没有把同一岗位代码自动作为跨年稳定职位标识，历史司局统计表示同类记录而非岗位逐一对应。
各年度类别编码、专业与备注关键词特征等沿用现有版本，仍有近似编码的局限，未进行个人资格筛选。

## 结果文件

- [全部岗位预测](2027_prediction.csv)：保留全部岗位、资格原文、预测分、分数组、历史误差参考范围与历史样本数。
- [各省分组汇总](2027_province_summary.csv)：按分数口径分组与省份汇总岗位数、招录人数及预测分分布。
- `2027_prediction_exact.csv` 保存未四舍五入预测，供之后独立检验。
- `run_summary_2027.json` 保存输入、模型和预测的校验值及训练记录数。

当前 {summary['no_recent_department_history']} 个岗位没有同部门同类别近期历史样本，表中已提示。普通岗位、公安、金融监管、证监和非通用语等保留原数据分数口径，不能直接按预测分跨组比较。

## 精度和参考范围

上一个模型在 2026 年独立检验中，普通岗位 MAE 5.26 分、RMSE 6.53 分，平均低估 2.32 分。这些数值属于旧模型的历史检验，不能当作本次重训模型或 2027 年的已验证误差。
本版通过重训吸收新增年度数据，未将旧模型的平均偏差直接机械加到所有预测中。
表内参考范围为当前预测加减 2026 年旧模型同组绝对误差的第 90 百分位，不足 100 个检验样本的组不生成范围。重训已改变模型，范围仅作历史误差参考，不保证 2027 年覆盖率，不能称为已校准置信区间。
试卷难度、考生水平、报考选择与年度政策变化可能使实际结果发生系统性变化；本版未单独建模尚未观测的年度难度。
历史无分数岗位未填成零，本模型预测的是有可观测进面分数时的数值，没有预测是否无人入围、取消或需要调剂。

## 验证与版本保留

已核验目标键唯一且完整、全部预测有限、代码文本长度、目标年标签不参与特征，并核验原 2026 模型及预测 SHA-256 不变。
2026 年首批真实结果现已用于本版训练。此前冻结预测及独立检验仍保留，但不能再把本版在 2026 年上的拟合分数作为独立检验成绩。
没有按 2027 年预测结果排序后删除岗位，所有原表岗位均输出。

在项目独立环境中运行 `src/predict_2027.py` 可复现。输入与结果独立保存，不改写原 2026 预测或旧训练面板。
'''
    (output/'2027预测说明.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='对照实验：将2026年纳入训练，不能继续称2026为本版独立测试集')
    parser.add_argument('--include-2026-training',action='store_true')
    if not parser.parse_args().include_2026_training:
        parser.error('需要显式指定 --include-2026-training；主版本运行 cli.py predict-2027')
    run()
