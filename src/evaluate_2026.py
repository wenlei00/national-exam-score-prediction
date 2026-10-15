"""Evaluate frozen predictions against explicitly supplied first-batch 2026 interview list.

Never calls training or modifies frozen models, predictions or training datasets.
Candidate names and admission-ticket numbers are not imported.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from inventory import ROOT
from data_processing import identifier,clean
from settings import input_path

SOURCE=input_path('actual_2026')
KEY=['部门代码','职位代码']

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(d):
    error=d['预测值']-d['实际最低进面分']
    ae=error.abs()
    sst=((d['实际最低进面分']-d['实际最低进面分'].mean())**2).sum()
    return dict(n=len(d),mae=float(ae.mean()),rmse=float(np.sqrt((error**2).mean())),
        bias=float(error.mean()),r2=float(1-(error**2).sum()/sst) if sst>0 else None,
        median_absolute_error=float(ae.median()),p90_absolute_error=float(ae.quantile(.9)),
        within_3=float((ae<=3).mean()),within_5=float((ae<=5).mean()),within_10=float((ae<=10).mean()),
        actual_mean=float(d['实际最低进面分'].mean()),predicted_mean=float(d['预测值'].mean()))

def mdtable(df):
    def fmt(x): return f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def run():
    output=ROOT/'results/evaluation_2026'; output.mkdir(exist_ok=True)
    frozen=json.loads((ROOT/'results/run_summary.json').read_text(encoding='utf-8'))
    paths={ROOT/'results/2026_prediction.csv':frozen['prediction_sha256'],
           ROOT/'results/models/final_model_2019_2025.joblib':frozen['model_sha256']}
    for path,expected in paths.items():
        assert digest(path)==expected,f'Frozen asset changed before evaluation: {path}'
    prediction=pd.read_csv(ROOT/'results/2026_prediction.csv',dtype={c:str for c in KEY})
    exact=pd.read_csv(ROOT/'results/2026_prediction_exact.csv',dtype={'department_code':str,'position_code':str})
    exact=exact.rename(columns={'department_code':'部门代码','position_code':'职位代码','predicted_score':'预测值'})
    prediction=prediction.merge(exact[KEY+['预测值']],on=KEY,validate='one_to_one')
    assert np.max(np.abs(prediction['预测值']-prediction['预测最低进面分']))<=0.005001
    columns=['招录机关','部门代码','用人司局','招考职位','职位代码','最低面试分数']
    records=[]
    for sheet in pd.ExcelFile(SOURCE).sheet_names:
        d=pd.read_excel(SOURCE,sheet_name=sheet,usecols=columns,dtype=str).fillna('').map(clean)
        d['source_sheet']=sheet
        for c,width in [('部门代码',6),('职位代码',12)]: d[c]=d[c].map(lambda v:identifier(v,width))
        records.append(d)
    raw=pd.concat(records,ignore_index=True)
    invalid=~(raw['部门代码'].str.fullmatch(r'\d{6}') & raw['职位代码'].str.fullmatch(r'\d{12}'))
    raw.loc[invalid].to_csv(output/'invalid_keys.csv',index=False,encoding='utf-8-sig')
    raw=raw.loc[~invalid].copy()
    raw['actual']=pd.to_numeric(raw['最低面试分数'],errors='coerce')
    raw['valid_score']=raw.actual.between(.001,200)
    grouped=raw.groupby(KEY,sort=False).agg(实际最低进面分=('actual','first'),
        分数不同值数=('actual','nunique'),原名单记录数=('actual','size'),分数均有效=('valid_score','all'),
        实际部门名称=('招录机关','first'),实际用人司局=('用人司局','first'),实际职位名称=('招考职位','first'),
        部门名称不同值数=('招录机关','nunique'),司局名称不同值数=('用人司局','nunique'),职位名称不同值数=('招考职位','nunique')).reset_index()
    grouped['标签有效']=grouped['分数不同值数'].eq(1)&grouped['分数均有效']&grouped['部门名称不同值数'].eq(1)&grouped['司局名称不同值数'].eq(1)&grouped['职位名称不同值数'].eq(1)
    grouped.to_csv(output/'actual_scores_by_job.csv',index=False,encoding='utf-8-sig')
    grouped.loc[~grouped['标签有效']].to_csv(output/'conflicting_or_invalid_labels.csv',index=False,encoding='utf-8-sig')
    matched=prediction.merge(grouped,on=KEY,how='outer',validate='one_to_one',indicator=True)
    matched['名称一致']=matched['部门名称'].eq(matched['实际部门名称'])&matched['用人司局'].eq(matched['实际用人司局'])&matched['招考职位'].eq(matched['实际职位名称'])
    matched.loc[matched['_merge']=='left_only'].to_csv(output/'predicted_jobs_absent_from_first_batch.csv',index=False,encoding='utf-8-sig')
    matched.loc[matched['_merge']=='right_only'].to_csv(output/'actual_jobs_without_prediction.csv',index=False,encoding='utf-8-sig')
    matched.loc[(matched['_merge']=='both')&~matched['名称一致']].to_csv(output/'name_mismatches.csv',index=False,encoding='utf-8-sig')
    eligible=matched['_merge'].eq('both')&matched['标签有效'].eq(True)&matched['名称一致']
    evaluated=matched.loc[eligible].copy()
    evaluated['误差_预测减实际']=evaluated['预测值']-evaluated['实际最低进面分']
    evaluated['绝对误差']=evaluated['误差_预测减实际'].abs()
    evaluated['实际落入历史误差范围']=np.where(evaluated['历史误差参考下限'].notna(),
        evaluated['实际最低进面分'].between(evaluated['历史误差参考下限'],evaluated['历史误差参考上限']).astype(float),np.nan)
    feat=pd.read_csv(ROOT/'data/processed/features_2026_verified.csv',usecols=['province','department_family'])
    assert len(feat)==len(prediction)
    feature_keys=prediction[KEY].copy(); feature_keys[['province','department_family']]=feat.values
    evaluated=evaluated.merge(feature_keys,on=KEY,validate='one_to_one')
    evaluated.to_csv(output/'2026_prediction_vs_actual.csv',index=False,encoding='utf-8-sig')
    groups=[]
    for field in ['分数口径分组','考试类别','province','department_family']:
        for value,g in evaluated.groupby(field):
            interval=g['实际落入历史误差范围'].dropna()
            groups.append(dict(group_field=field,group=value,**metrics(g),reference_range_n=len(interval),
                               reference_range_coverage=float(interval.mean()) if len(interval) else None))
    group_metrics=pd.DataFrame(groups)
    group_metrics.to_csv(output/'group_metrics.csv',index=False,encoding='utf-8-sig')
    evaluated.nlargest(100,'绝对误差').to_csv(output/'largest_errors_top100.csv',index=False,encoding='utf-8-sig')
    # Independent baseline: same rule fixed before 2026 outcomes, using only 2025 outcomes.
    history=pd.read_csv(ROOT/'data/processed/model_features.csv',low_memory=False)
    last=history[(history.year==2025)&history.eligible.eq(True)]
    med=last.groupby(['exam_category','score_group']).target.median()
    target_features=pd.read_csv(ROOT/'data/processed/features_2026_verified.csv',usecols=['exam_category','score_group'])
    base=prediction[KEY].copy()
    base['baseline']=np.array([med.get((r.exam_category,r.score_group),last.target.median()) for r in target_features.itertuples()])
    check=evaluated.merge(base,on=KEY,validate='one_to_one')
    baseline_check=check.copy(); baseline_check['预测值']=baseline_check.baseline
    comparison=[]
    for scope,subset in [('全部岗位',check),('普通职位',check[check['分数口径分组']=='普通职位'])]:
        comparison.append(dict(scope=scope,model='frozen_lightgbm',**metrics(subset)))
        b=subset.copy(); b['预测值']=b.baseline
        comparison.append(dict(scope=scope,model='previous_year_category_median',**metrics(b)))
    pd.DataFrame(comparison).to_csv(output/'baseline_comparison.csv',index=False,encoding='utf-8-sig')
    previous=pd.read_csv(ROOT/'results/2025_group_metrics.csv')
    previous=previous[previous.group_field=='score_group'][['group','n','mae','rmse','bias']] if 'bias' in previous.columns else previous[previous.group_field=='score_group'][['group','n','mae','rmse']]
    compare_year=previous.merge(group_metrics[group_metrics.group_field=='分数口径分组'][['group','n','mae','rmse','bias']],on='group',how='outer',suffixes=('_2025','_2026'))
    compare_year.to_csv(output/'2025_vs_2026_metrics.csv',index=False,encoding='utf-8-sig')
    overall=metrics(evaluated)
    summary=dict(source=str(SOURCE),source_sha256=digest(SOURCE),first_batch_only=True,source_rows=len(raw),
        invalid_key_rows=int(invalid.sum()),actual_unique_jobs=len(grouped),invalid_or_conflicting_jobs=int((~grouped['标签有效']).sum()),
        prediction_jobs=len(prediction),matched_keys=int(matched['_merge'].eq('both').sum()),
        missing_first_batch_jobs=int(matched['_merge'].eq('left_only').sum()),actual_without_prediction=int(matched['_merge'].eq('right_only').sum()),
        name_mismatches=int((matched['_merge'].eq('both')&~matched['名称一致']).sum()),evaluated_jobs=len(evaluated),
        frozen_prediction_sha256=frozen['prediction_sha256'],frozen_model_sha256=frozen['model_sha256'],
        exact_prediction_sha256=digest(ROOT/'results/2026_prediction_exact.csv'),retrained=False,overall=overall,
        ordinary=metrics(evaluated[evaluated['分数口径分组']=='普通职位']))
    for path,expected in paths.items(): assert digest(path)==expected,'Frozen asset changed during evaluation'
    (output/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    make_report(evaluated,group_metrics,pd.DataFrame(comparison),compare_year,summary,output)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def make_report(d,g,comparison,year_comparison,s,out):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    ordinary=d[d['分数口径分组']=='普通职位']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),dpi=150)
    axes[0].hexbin(ordinary['实际最低进面分'],ordinary['预测值'],gridsize=45,mincnt=1,cmap='Blues')
    lo=min(ordinary['实际最低进面分'].min(),ordinary['预测值'].min()); hi=max(ordinary['实际最低进面分'].max(),ordinary['预测值'].max())
    axes[0].plot([lo,hi],[lo,hi],color='orange',linewidth=1)
    axes[0].set(title='2026年首批名单：普通岗位',xlabel='实际最低进面分',ylabel='冻结预测分数')
    axes[1].hist(ordinary['误差_预测减实际'],bins=50,color='#32688d'); axes[1].axvline(0,color='orange')
    axes[1].set(title='误差分布',xlabel='预测减实际（分）',ylabel='岗位数')
    fig.tight_layout(); fig.savefig(out/'ordinary_errors.png'); plt.close(fig)
    compact=g[g.group_field=='分数口径分组'][['group','n','mae','rmse','bias','r2','within_5','within_10','reference_range_coverage']].copy()
    compact.columns=['分组','岗位数','MAE','RMSE','偏差_预测减实际','R²','误差不超过5分比例','误差不超过10分比例','历史参考范围覆盖率']
    report=f'''# 2026 年冻结预测独立检验

本次使用用户新提供的首批面试名单检验上一轮固定预测，未重训、未改参数、未修正预测年度整体分数。模型及预测文件在读取实际结果前后校验 SHA-256 一致。
预测分数使用上轮未四舍五入版本，并检查其与当时冻结的两位小数文件逐项相符（差值不超过 0.005001）。精确预测文件此前未单独登记校验值，本次记录其校验值，不能把本次记录的值表述为此前已登记。

## 匹配与样本

- 原名单记录：{s['source_rows']:,} 行；每行不等于一个岗位，最低线在同岗位人员间重复。
- 按部门代码＋职位代码合并：{s['actual_unique_jobs']:,} 个岗位。
- 原预测：{s['prediction_jobs']:,} 个岗位；岗位键匹配 {s['matched_keys']:,} 个。
- 预测有、首批名单无：{s['missing_first_batch_jobs']:,} 个，保留为未检验，不当成零分。
- 名单有、预测无：{s['actual_without_prediction']:,} 个。
- 分数或名称冲突：{s['invalid_or_conflicting_jobs']:,} 个；岗位名称不一致：{s['name_mismatches']:,} 个。
- 按键、分数、部门/司局/职位名称核验后，实际参与评估 {s['evaluated_jobs']:,} 个岗位。

仅导入岗位和最低线字段，没有导入姓名或准考证号。实际分数另存于评估目录，没有并入训练面板。

## 结果

整体 MAE={s['overall']['mae']:.3f}、RMSE={s['overall']['rmse']:.3f}、平均偏差（预测减实际）={s['overall']['bias']:.3f} 分。
整体指标混合不同分数口径，普通岗位的结果更适合与个人公共科目总分对照。普通岗位 MAE={s['ordinary']['mae']:.3f}、RMSE={s['ordinary']['rmse']:.3f}、平均偏差={s['ordinary']['bias']:.3f} 分。
MAE 是绝对误差的平均值，不表示每个岗位误差都不超过这一数值。

{mdtable(compact)}

负偏差表示整体偏低估，正偏差表示整体偏高估。各分组原始分数口径不同，不能以专业类分数的 MAE 较低就认定这些组更容易准确预测。

## 与原定简单基线比较

基线使用 2025 年同考试类别、同分数组的中位数，未用 2026 年实际结果拟合。按完全相同岗位集合比较：

{mdtable(comparison[['scope','model','n','mae','rmse','bias','r2']])}

## 与 2025 年时间测试比较

{mdtable(year_comparison)}

跨年度误差变化包含试卷难度、考生构成、报考选择、岗位结构以及数据口径等多种因素。这里不能据差异单独判定其中哪一项导致了变化。首批名单缺少的岗位也不能自动视为模型漏预测或实际无人报考。

## 历史参考范围

上轮范围由 2025 年同组绝对误差的第 90 百分位构成。本次分组覆盖率见上表。它是跨年实际检验的结果，不能事后根据 2026 年成绩扩大范围，再宣称此前有相应覆盖率。

## 后续改进的边界

本轮检验已揭示的误差可用于提出下一版方案。若之后用 2026 年结果调参、重训或校准，则 2026 年应记为开发数据，不能继续当作那一版模型的未见测试集。原模型、原预测及本次检验结果均保留。

明细见 [逐岗位预测与实际对照](2026_prediction_vs_actual.csv)，大误差清单见 [绝对误差最大的100个岗位](largest_errors_top100.csv)。本次只检验首批名单，调剂、补录或后续变动不在本文件覆盖范围内。

![2026 年普通岗位预测与误差](ordinary_errors.png)
'''
    (out/'2026独立检验报告.md').write_text(report,encoding='utf-8')

if __name__=='__main__': run()
