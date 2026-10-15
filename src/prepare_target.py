"""Validate publicly posted 2026 job attachment without accessing any 2026 outcomes."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from inventory import ROOT
from data_processing import clean,identifier,JOB_FIELDS
from settings import input_path

TARGET_NAME=input_path('target_2026').name
SOURCE_PAGE='https://jy.bsu.edu.cn/front/showContent.jspa?channelId=695&contentId=104685'
SOURCE_URL='https://jy.bsu.edu.cn/uploadfile/front/bsu/default/BA4776F0C0A2584AFDA129D68E090667.xls'

def prepare():
    path=input_path('target_2026')
    book=pd.ExcelFile(path)
    frames=[]
    for sheet in book.sheet_names:
        d=pd.read_excel(book,sheet_name=sheet,header=1,dtype=str).fillna('').map(clean)
        for col,width in [('部门代码',6),('职位代码',12)]: d[col]=d[col].map(lambda x:identifier(x,width))
        assert d['部门代码'].str.fullmatch(r'\d{6}').all()
        assert d['职位代码'].str.fullmatch(r'\d{12}').all()
        d['year']=2026; d['source_file']=TARGET_NAME; d['source_sheet']=sheet
        d['source_row']=range(3,len(d)+3)
        frames.append(d)
    target=pd.concat(frames,ignore_index=True)
    assert not target.duplicated(['部门代码','职位代码']).any()
    old=pd.read_csv(ROOT/'data/processed/job_panel.csv',dtype={'部门代码':str,'职位代码':str},low_memory=False)
    year2025=old[old.year==2025]
    comparison=target.merge(year2025,on=['部门代码','职位代码'],how='left',suffixes=('_2026','_2025'),indicator=True)
    comparison['all_core_fields_identical']=comparison.apply(lambda r:all(r[c+'_2026']==r[c+'_2025'] for c in JOB_FIELDS if c not in ['部门代码','职位代码']),axis=1)
    identical=int(comparison.all_core_fields_identical.sum())
    assert identical<len(target)*0.5, 'Target still looks like a previous-year copy'
    target.to_csv(ROOT/'data/processed/jobs_2026_verified.csv',index=False,encoding='utf-8-sig')
    dictionary_path=ROOT/'data/processed/data_dictionary.csv'
    dictionary=pd.read_csv(dictionary_path).fillna('')
    dictionary=dictionary.loc[dictionary['file']!=TARGET_NAME].copy()
    dictionary.loc[(dictionary['year']==2026)&(dictionary['file']!=TARGET_NAME),'kind']='quarantined_wrong_year'
    rows=[]
    for sheet,grp in target.groupby('source_sheet'):
        for field in JOB_FIELDS:
            rows.append(dict(year=2026,kind='jobs_verified',file=TARGET_NAME,sheet=sheet,field=field,
                             rows=len(grp),missing=int(grp[field].eq('').sum()),unique=int(grp[field].nunique()),
                             sample=' / '.join(grp.loc[grp[field]!='',field].unique()[:3])[:400]))
    pd.concat([dictionary,pd.DataFrame(rows)],ignore_index=True).to_csv(dictionary_path,index=False,encoding='utf-8-sig')
    comparison[['部门代码','职位代码','_merge','all_core_fields_identical']].to_csv(ROOT/'data/processed/2026_verified_vs_2025.csv',index=False,encoding='utf-8-sig')
    info={'year':2026,'kind':'jobs','path':str(path),'source_page':SOURCE_PAGE,'source_url':SOURCE_URL,
          'source_description':'北京体育大学2025年10月14日发布的2026年度招考公告转载附件；国家公务员局直连未能访问',
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'unique_jobs':len(target),
          'recruit_count':int(pd.to_numeric(target['招考人数']).sum()),'keys_in_2025':int((comparison['_merge']=='both').sum()),
          'core_identical_to_2025':identical,'observed_scores_read':False}
    (ROOT/'data/verified_target_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest_path=ROOT/'data/input_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for item in manifest:
        item['status']=('target_input' if Path(item['path']).name==TARGET_NAME else 'quarantined_wrong_year') if item['year']==2026 else 'historical_input'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Canonical panel now contains the verified target, never the mislabeled subset.
    if not (ROOT/'data/processed/jobs_2026_rejected_wrong_year.csv').exists() and old.loc[old.year==2026,'source_file'].ne(TARGET_NAME).any():
        old.loc[old.year==2026].to_csv(ROOT/'data/processed/jobs_2026_rejected_wrong_year.csv',index=False,encoding='utf-8-sig')
    target['min_interview_score']=float('nan')
    target['train_eligible']=False
    target['label_status']='target_unobserved'
    target['name_match']=False
    pd.concat([old.loc[old.year<=2025],target],ignore_index=True).to_csv(ROOT/'data/processed/job_panel.csv',index=False,encoding='utf-8-sig')
    audit=json.loads((ROOT/'data/processed/data_audit.json').read_text(encoding='utf-8'))
    rejected=next(r for r in audit['years'] if r['year']==2026)
    if audit.get('target_2026_identical_to_2025')==rejected['unique_jobs']:
        audit['rejected_input_2026']=audit.get('rejected_input_2026',rejected)
    audit['years']=[r for r in audit['years'] if r['year']!=2026]+[dict(year=2026,unique_jobs=len(target),
                     recruit_count=info['recruit_count'],source_rows=len(target),usable_labels=0,
                     missing_labels=len(target),name_mismatches=0,score_conflicts=0)]
    audit.update(target_2026_jobs=len(target),target_2026_keys_in_2025=info['keys_in_2025'],target_2026_identical_to_2025=identical)
    (ROOT/'data/processed/data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': prepare()
