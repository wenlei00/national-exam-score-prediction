"""Read explicitly allowed inputs, audit keys, retain lineage, never read 2026 outcomes."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from inventory import ROOT, ingest

KEY = ['year', '部门代码', '职位代码']
JOB_FIELDS = ['部门代码', '部门名称', '用人司局', '机构性质', '招考职位', '职位属性', '职位分布',
              '职位简介', '职位代码', '机构层级', '考试类别', '招考人数', '专业', '学历', '学位',
              '政治面貌', '基层工作最低年限', '服务基层项目工作经历',
              '是否在面试阶段组织专业能力测试', '面试人员比例', '工作地点', '落户地点', '备注']

def clean(value):
    return re.sub(r'\s+', ' ', str(value)).strip() if pd.notna(value) else ''

def identifier(value, width):
    text = clean(value)
    if re.fullmatch(r'\d+\.0+', text): text = text.split('.')[0]
    return text.zfill(width) if text.isdigit() else text

def read_inputs():
    manifest = ingest()
    job_tables, score_tables, dictionary = [], [], []
    for record in manifest:
        book = pd.ExcelFile(record['path'])
        for sheet in book.sheet_names:
            top = pd.read_excel(book, sheet_name=sheet, header=None, nrows=6, dtype=str).fillna('')
            header = next(i for i, row in enumerate(top.values.tolist()) if '职位代码' in row)
            df = pd.read_excel(book, sheet_name=sheet, header=header, dtype=str).fillna('')
            df.columns = [clean(c) for c in df.columns]
            df = df.map(clean)
            df['source_row'] = np.arange(len(df)) + header + 2
            for col, width in [('部门代码', 6), ('职位代码', 12)]: df[col] = df[col].map(lambda v: identifier(v, width))
            valid = df['部门代码'].str.fullmatch(r'\d{6}') & df['职位代码'].str.fullmatch(r'\d{12}')
            if (~valid).any():
                df.loc[~valid].to_csv(ROOT/'data/processed'/f"invalid_keys_{record['year']}_{record['kind']}_{len(dictionary)}.csv", index=False, encoding='utf-8-sig')
            df = df.loc[valid].copy()
            for col in df.columns:
                if col=='source_row': continue
                dictionary.append(dict(year=record['year'], kind=record['kind'], file=Path(record['path']).name,
                                       sheet=sheet, field=col, rows=len(df), missing=int((df[col]=='').sum()),
                                       unique=int(df[col].nunique()), sample=' / '.join(df.loc[df[col]!='',col].unique()[:3])[:400]))
            df['year'] = record['year']
            df['source_file'] = Path(record['path']).name
            df['source_sheet'] = sheet
            (job_tables if record['kind']=='jobs' else score_tables).append(df)
            print(record['year'],record['kind'],sheet,len(df),flush=True)
    pd.DataFrame(dictionary).to_csv(ROOT/'data/processed/data_dictionary.csv', index=False, encoding='utf-8-sig')
    return pd.concat(job_tables,ignore_index=True), pd.concat(score_tables,ignore_index=True)

def process():
    out=ROOT/'data/processed'
    jobs, scores=read_inputs()
    jobs.to_csv(out/'jobs_source_rows.csv',index=False,encoding='utf-8-sig')
    scores.to_csv(out/'scores_source_rows.csv',index=False,encoding='utf-8-sig')
    conflicts=jobs.groupby(KEY)[JOB_FIELDS[1:]].nunique().gt(1).any(axis=1)
    bad_keys=conflicts[conflicts].reset_index()[KEY]
    conflicting=jobs.merge(bad_keys,on=KEY,how='inner')
    conflicting.to_csv(out/'job_conflicts.csv',index=False,encoding='utf-8-sig')
    if len(bad_keys): raise ValueError(f'{len(bad_keys)} conflicting job keys; do not silently choose records')
    lineage=jobs.groupby(KEY).agg(source_files=('source_file',lambda s:';'.join(sorted(set(s)))),
                                 source_rows=('source_row',lambda s:';'.join(map(str,s))),
                                 source_count=('source_file','size')).reset_index()
    jobs=jobs.drop_duplicates(KEY).merge(lineage,on=KEY,validate='one_to_one')
    scores['min_interview_score']=pd.to_numeric(scores['最低面试分数'],errors='coerce')
    scores['score_valid']=scores['min_interview_score'].between(0.001,200)
    score_groups=scores.groupby(KEY).agg(min_interview_score=('min_interview_score','first'),
        distinct_score_count=('min_interview_score','nunique'), score_valid=('score_valid','all'),
        score_source_rows=('source_row',lambda s:';'.join(map(str,s))), score_record_count=('source_row','size'),
        score_department=('招录机关','first'),score_unit=('用人司局','first'),score_title=('招考职位','first')) .reset_index()
    score_groups['score_usable']=score_groups['score_valid'] & (score_groups['distinct_score_count']==1)
    score_groups.to_csv(out/'scores_by_job.csv',index=False,encoding='utf-8-sig')
    score_groups.loc[~score_groups['score_usable']].to_csv(out/'score_conflicts_or_invalid.csv',index=False,encoding='utf-8-sig')
    panel=jobs.merge(score_groups,on=KEY,how='left',validate='one_to_one',indicator=True)
    panel['label_status']=np.where(panel['_merge']=='left_only','missing_score',np.where(panel['score_usable']==True,'usable','conflicting_or_invalid'))
    panel['name_match']=(panel['用人司局']==panel['score_unit']) & (panel['招考职位']==panel['score_title'])
    panel['train_eligible']=(panel['year']<=2025)&(panel['label_status']=='usable')&panel['name_match']
    panel.loc[(panel['year']<=2025)&~panel['train_eligible']].to_csv(out/'excluded_training_jobs.csv',index=False,encoding='utf-8-sig')
    unmatched=score_groups.merge(jobs[KEY],on=KEY,how='left',indicator=True).query('_merge=="left_only"')
    unmatched.to_csv(out/'scores_without_jobs.csv',index=False,encoding='utf-8-sig')
    panel.drop(columns='_merge').to_csv(out/'job_panel.csv',index=False,encoding='utf-8-sig')
    audit=[]
    for year,grp in panel.groupby('year'):
        raw=jobs.query('year==@year')
        audit.append(dict(year=int(year),unique_jobs=len(grp),recruit_count=int(pd.to_numeric(raw['招考人数']).sum()),
                          source_rows=int(raw['source_count'].sum()),usable_labels=int(grp['train_eligible'].sum()),
                          missing_labels=int((grp['label_status']=='missing_score').sum()),
                          name_mismatches=int(((grp['label_status']=='usable')&~grp['name_match']).sum()),
                          score_conflicts=int((grp['label_status']=='conflicting_or_invalid').sum())))
    # Compare versions without assuming that a filename proves its recruitment year.
    compare=jobs.query('year==2026').merge(jobs.query('year==2025'),on=['部门代码','职位代码'],suffixes=('_2026','_2025'),how='left',indicator=True)
    compare['all_core_fields_identical']=compare.apply(lambda r: all(r[c+'_2026']==r[c+'_2025'] for c in JOB_FIELDS if c not in ['部门代码','职位代码']),axis=1)
    compare[['部门代码','职位代码','_merge','all_core_fields_identical']].to_csv(out/'2026_vs_2025_version_audit.csv',index=False,encoding='utf-8-sig')
    summary=dict(years=audit,score_rows=len(scores),score_unique_jobs=len(score_groups),unmatched_score_jobs=len(unmatched),
                 target_2026_jobs=len(compare),target_2026_keys_in_2025=int((compare['_merge']=='both').sum()),
                 target_2026_identical_to_2025=int(compare['all_core_fields_identical'].sum()))
    (out/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return panel

if __name__=='__main__': process()
