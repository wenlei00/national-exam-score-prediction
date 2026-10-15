"""Features from job tables and strictly earlier-year observed outcomes."""
import re
import numpy as np
import pandas as pd

PROVINCES=['北京市','天津市','上海市','重庆市','河北省','山西省','辽宁省','吉林省','黑龙江省','江苏省','浙江省',
           '安徽省','福建省','江西省','山东省','河南省','湖北省','湖南省','广东省','海南省','四川省','贵州省',
           '云南省','陕西省','甘肃省','青海省','内蒙古自治区','广西壮族自治区','西藏自治区','宁夏回族自治区','新疆维吾尔自治区']

def category(text):
    if '行政执法' in text: return '行政执法'
    if '省级以上' in text or '中央机关' in text: return '中央省级综合管理'
    if '市' in text: return '市地以下综合管理'
    return text or '未知'

def score_group(text):
    if '公安' in text: return '公安专业职位'
    if any(w in text for w in ['银保监','金融监管']): return '金融监管专业职位'
    if '证监' in text: return '证监专业职位'
    if '非通用语' in text and not text.startswith('普通'): return '非通用语职位'
    if text.startswith('普通'): return '普通职位'
    return '其他特殊专业职位'

def location(text):
    province=next((p for p in PROVINCES if text.startswith(p)), '未知')
    if province=='未知': return province,text or '未知'
    remaining=text[len(province):]
    if province in ['北京市','天津市','上海市','重庆市']: return province,province
    m=re.match(r'(.+?(?:市|自治州|地区|盟))',remaining)
    return province,province+(m.group(1) if m else '')

def department_family(text):
    for words,label in [(['税务'],'税务'),(['海关'],'海关'),(['海事'],'海事'),(['金融监管','银保监'],'金融监管'),
                        (['证券','证监'],'证监'),(['铁路公安'],'铁路公安'),(['边防','移民'],'移民边检'),
                        (['统计'],'统计'),(['气象'],'气象'),(['邮政'],'邮政'),(['铁路'],'铁路'),(['审计'],'审计')]:
        if any(w in text for w in words): return label
    return '其他'

CAT=['department','department_family','exam_category','score_group','province','city','institution_level','institution_type',
     'education','degree','political_status','grassroots','service_project','professional_test','distribution']
NUM=['recruit_count','interview_ratio','year','fresh_restriction_mentioned','male_restriction_mentioned','female_restriction_mentioned',
     'household_restriction_mentioned','major_unrestricted','major_text_length','major_option_proxy','remarks_length']
MAJORS={'economics':['经济','金融','财政'],'law':['法学','法律'],'literature':['文学','中文','新闻'],
        'management':['管理','会计','财务','审计'],'computer':['计算机','软件','数据','人工智能'],
        'engineering':['工程','工学'],'statistics':['统计'],'science':['理学','数学','物理','化学'],
        'medical':['医学','药学'],'agriculture':['农学','农业'],'maritime':['航海','轮机']}
NUM += ['major_'+c for c in MAJORS]
HISTORY_GROUPS={'department_history':['department','exam_category','score_group'],
                'unit_history':['department','用人司局','exam_category','score_group'],
                'city_history':['city','exam_category','score_group'],
                'category_history':['exam_category','score_group']}
HISTORY_NUM=[p+s for p in HISTORY_GROUPS for s in ['_mean','_n']]

def build_features(panel):
    d=panel.copy().fillna('')
    f=pd.DataFrame(index=d.index)
    for col,raw in [('department','部门名称'),('education','学历'),('degree','学位'),('political_status','政治面貌'),
                    ('grassroots','基层工作最低年限'),('service_project','服务基层项目工作经历'),
                    ('professional_test','是否在面试阶段组织专业能力测试'),('distribution','职位分布'),
                    ('institution_level','机构层级'),('institution_type','机构性质')]:
        f[col]=d[raw].replace('', '未知')
    f['department_family']=d['部门名称'].map(department_family)
    f['exam_category']=d['考试类别'].map(category)
    f['score_group']=d['职位属性'].map(score_group)
    places=d['工作地点'].map(location)
    f['province']=places.map(lambda x:x[0]); f['city']=places.map(lambda x:x[1])
    f['recruit_count']=pd.to_numeric(d['招考人数'],errors='coerce')
    f['interview_ratio']=pd.to_numeric(d['面试人员比例'].str.extract(r'(\d+(?:\.\d+)?)\s*[:：]')[0],errors='coerce')
    f['year']=d['year'].astype(int)
    remarks=d['备注']
    # These are mention indicators, not definitive qualification decisions.
    f['fresh_restriction_mentioned']=remarks.str.contains(r'限.*应届|应届.*限|应届.*报考',regex=True).astype(int)
    for sex,label in [('male','男性'),('female','女性')]:
        f[sex+'_restriction_mentioned']=remarks.str.contains(r'限'+label+'|仅限'+label+'|'+label+'报考',regex=True).astype(int)
    f['household_restriction_mentioned']=remarks.str.contains('户籍|生源',regex=True).astype(int)
    major=d['专业']
    f['major_unrestricted']=major.str.contains('不限|无限制').astype(int)
    f['major_text_length']=major.str.len()
    f['major_option_proxy']=major.str.count(r'[、,，;；]')+1
    f['remarks_length']=remarks.str.len()
    for name,words in MAJORS.items(): f['major_'+name]=major.str.contains('|'.join(words)).astype(int)
    f['用人司局']=d['用人司局']
    f['target']=pd.to_numeric(d['min_interview_score'],errors='coerce')
    f['eligible']=d['train_eligible'].eq(True)
    for year in sorted(f.year.unique()):
        # At most three prior years; no current/future labels enter these aggregates.
        history=f[(f.year<year)&(f.year>=year-3)&f.eligible]
        rows=f.index[f.year==year]
        for prefix,columns in HISTORY_GROUPS.items():
            stats=history.groupby(columns).target.agg(['mean','count']).reset_index()
            current=f.loc[rows,columns].merge(stats,on=columns,how='left',validate='many_to_one')
            f.loc[rows,prefix+'_mean']=current['mean'].to_numpy()
            f.loc[rows,prefix+'_n']=current['count'].fillna(0).to_numpy()
    return f

ALL_FEATURES=CAT+NUM+HISTORY_NUM
