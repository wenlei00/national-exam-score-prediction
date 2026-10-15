"""Build research report and reproducible static figures from completed outputs."""
import json
import os
import numpy as np
import pandas as pd
from inventory import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import score_group

def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f'{x:.3f}'
        return str(x).replace('|','/').replace('\n',' ')
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows+=['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def run():
    out=ROOT/'results'; plots=out/'plots'
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['figure.dpi']=150
    summary=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((ROOT/'data/processed/data_audit.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'data/verified_target_manifest.json').read_text(encoding='utf-8'))
    comparisons=pd.read_csv(out/'model_comparison.csv')
    groups=pd.read_csv(out/'2025_group_metrics.csv')
    tests=pd.read_csv(out/'2025_temporal_test_predictions.csv')
    panel=pd.read_csv(ROOT/'data/processed/job_panel.csv',low_memory=False)
    panel=panel[panel.train_eligible==True].copy()
    panel['score_group']=panel['职位属性'].map(score_group)
    ordinary=panel[panel.score_group=='普通职位']
    fig,ax=plt.subplots(figsize=(9,4))
    trends=ordinary.groupby('year').min_interview_score.agg(['mean','median'])
    ax.plot(trends.index,trends['mean'],marker='o',label='均值')
    ax.plot(trends.index,trends['median'],marker='o',label='中位数')
    ax.set(title='普通岗位历史最低进面分（有匹配分数的岗位）',xlabel='招录年度',ylabel='原始分数')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(plots/'historical_trends.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    t=tests[tests.score_group=='普通职位']
    axes[0].hexbin(t.min_interview_score,t.predicted_score,gridsize=45,mincnt=1,cmap='Blues')
    lo=min(t.min_interview_score.min(),t.predicted_score.min()); hi=max(t.min_interview_score.max(),t.predicted_score.max())
    axes[0].plot([lo,hi],[lo,hi],color='orange',linewidth=1)
    axes[0].set(title='2025年普通岗位时间测试',xlabel='实际最低进面分',ylabel='预测最低进面分')
    axes[1].hist(t.error,bins=45,color='#32688d')
    axes[1].axvline(0,color='orange'); axes[1].set(title='误差分布',xlabel='预测减实际（分）',ylabel='岗位数量')
    fig.tight_layout(); fig.savefig(plots/'2025_test_errors.png'); plt.close(fig)
    selection=comparisons[comparisons.stage=='selection']
    fig,ax=plt.subplots(figsize=(8,4))
    pivot=selection.pivot(index='model',columns='year',values='mae')
    pivot.plot.bar(ax=ax,rot=0)
    ax.set(title='按年份验证的模型误差',xlabel='模型',ylabel='MAE（分；混合原始分数口径）')
    fig.tight_layout(); fig.savefig(plots/'model_comparison.png'); plt.close(fig)
    data_table=pd.DataFrame(audit['years'])[['year','unique_jobs','recruit_count','usable_labels','missing_labels']]
    data_table.columns=['年度','唯一岗位','招录人数','可训练标签','缺少标签或目标年未观测']
    group_table=groups[groups.group_field=='score_group'][['group','n','mae','rmse','r2']]
    group_table.columns=['岗位分组','测试岗位数','MAE','RMSE','R²']
    averages=selection.groupby('model')[['mae','rmse']].mean().reset_index()
    text=f'''# 国考最低进面分预测：2026 年第一版

已生成 2026 年 {summary['target_2026_n']:,} 个岗位的预测。预测文件为 [2026_prediction.csv](2026_prediction.csv)。
招录年度与考试举行年份分开：本次预测目标是 2026 年度招录，不是 2027 年度招录。
模型输入不含报名人数或过审人数，未读取 2026 年实际进面分。最后一次拟合使用 2019—2025 年 {summary['final_training_n']:,} 条年度岗位记录。

## 输入核验及替换来源

用户提供的三个“2026”学历筛选表共 34,132 行，去重后 20,245 个岗位、招录 38,762 人。全部 20,245 个岗位的 23 个核心字段（含键）与用户提供的 2025 年岗位表一致。其内容被隔离保存，不用作 2026 年预测输入。去重核验未发现相同岗位键的核心字段冲突。

为完成预测，另从[北京体育大学 2025 年 10 月 14 日转载的 2026 年招考公告]({source['source_page']})下载了完整附件：[招考简章]({source['source_url']})。
该附件有 20,714 个唯一岗位、招录 38,119 人，与 2025 年共有 11,301 个岗位键，但没有核心字段完全一致的岗位。下载来源、校验值见 `data/verified_target_manifest.json`。
国家公务员局直连未能访问，因此本次使用高校转载附件，不把它表述为直接从国家公务员局下载。仅下载岗位表，不下载目标年度成绩。

## 数据清洗

岗位键为“年度＋六位部门代码＋十二位职位代码”，代码按文本保留。只在同年度匹配岗位与分数，不把跨年相同代码自动当成同一岗位。
分数表共 611,550 行，合并为 106,866 个年度岗位，重复行的最低分均一致，没有靠取最小值消除冲突。
岗位表与分数表全部按键对应，但有 1 条 2019 年职位名称存在 `I` 与 `|` 的文字差异，未自动修正，暂排除；另有 3,970 个历史岗位缺少进面分，未填成零。最终可训练记录为 106,865 条。

{table(data_table)}

字段字典见 `data/processed/data_dictionary.csv`。隔离的输入标为 `quarantined_wrong_year`，补充的目标表标为 `jobs_verified`。

## 训练、选择与测试

1. 2019—2022 训练，2023 验证；2019—2023 训练，2024 验证。
2. 比较上一年度同考试类别/分组中位数基线、岭线性回归、随机森林和 LightGBM。参数在观察 2025 年预测误差之前确定。线性模型使用岭正则，未声称复现论文中的 OLS 显著性检验。
3. 按 2023、2024 两年度 MAE 等权平均选定 LightGBM。未按随机行切分，也未使用 2025 年选择模型或调参。
4. 用 2019—2024 训练后的选定模型检验 2025 年；最后用 2019—2025 重训，并冻结模型与预测文件校验值。

验证期两年度平均指标：

{table(averages)}

2025 年时间测试整体 MAE={summary['test_2025']['mae']:.3f}，RMSE={summary['test_2025']['rmse']:.3f}，R²={summary['test_2025']['r2']:.3f}。这是测试过的历史年份误差，不是已知的 2026 年误差。
整体指标混合不同分数口径，整体高 R² 很大程度上反映区分不同分数组，不能理解为普通岗位预测解释了 93.4% 的差异。各分组表现如下：

{table(group_table)}

普通岗位 MAE 约 4.95 分、R² 约 0.498。金融监管与证监分组的 R² 不高于零，模型在这些组内解释分数差异的能力弱。非通用语分组仅 73 个测试样本、误差较大，应单独核验分数口径，不作精确分数承诺。

## 特征与解释

使用地区、部门/系统、考试类别、职位属性、学历学位、政治面貌、基层经历、服务基层项目、面试比例、招录人数以及专业/备注的文本规则特征。
专业类别使用关键词、文本长度及分隔符数量近似编码，并未建立完整专业资格本体；备注中的性别、应届和户籍仅作提及标记，不等于已完成资格审查。
部门、司局、城市、考试类别的历史分数均值与样本数，严格只从预测年度之前三年的有标签记录生成。没有把同年度真实进面分用于特征。
模型不依赖年度岗位代码的跨年稳定性。司局历史分组使用部门、司局名称、考试类别和分数组，仅作为历史同类统计，不代表已经确认跨年岗位一一对应。

普通岗位抽样 2,000 条的 TreeSHAP 分组贡献见 `ordinary_jobs_shap.csv`。类别虚拟变量的贡献按原始字段求和后再取绝对值均值。
模型以全体训练样本作为解释背景，分数组字段的贡献仍包含普通岗位与专业岗位的基准差异，不能把它视为普通岗位内部变化的首要原因。其余贡献中包含服务基层条件、备注长度、历史同类分数及招录规模。备注长度是限制复杂度的代理，也可能吸收数据编排习惯，不能作因果解释。
SHAP 是模型预测贡献，不是政策权重或因果效应。

## 如何使用预测表

- `预测最低进面分`保留两位小数用于展示，未四舍五入版本单独保存，以供之后的外部检验。
- `历史误差参考下限/上限`是点预测加减 2025 年同分组绝对误差的第 90 百分位数。少于 100 个测试样本的组不生成范围。这不是经证明可覆盖 2026 年的 90% 预测区间。
- `同司局/同部门同类别历史样本数`反映近期历史支持程度，无历史同类样本的岗位有提示。
- 普通岗位与专业岗位保留原始分数口径，不能直接跨分组比较分数高低。
- 预测针对有可观测进面线的岗位。缺少历史标签的岗位可能有无人入围、调剂或来源不全等原因，本版没有预测岗位取消或进面线是否存在。
- 试卷难度、考生水平与报考选择的年度变化尚未观测，可能让一整年出现系统性偏差。没有报名数据的报名开始前预测尤其需要保留这一限制。

## 检查与复现

已核验年度岗位键唯一、代码长度保留、目标年真实成绩缺失、全部目标键完整输出、预测数值有限。
还进行了标签扰动检查：改变全部 2025 年标签不会改变用于预测 2025 年的特征。
输入文件逐份记录 SHA-256，最终模型与预测 SHA-256 保存于 `run_summary.json`。

在项目独立环境运行：

```powershell
.\\.venv\\Scripts\\python.exe src/data_processing.py
.\\.venv\\Scripts\\python.exe src/prepare_target.py
.\\.venv\\Scripts\\python.exe src/train.py
.\\.venv\\Scripts\\python.exe src/verify.py
.\\.venv\\Scripts\\python.exe src/report.py
```

目标附件已保存在 `data/raw`，上述流程不会重新下载网络数据。用户文件来源采用显式清单，未读取同目录里的其他年度结果。
收到你保留的 2026 年结果后，应先按岗位键和分数口径核验，再计算冻结预测的误差；不能先根据真实结果修改模型，再把修改后的成绩当作独立测试。

## 图表

![普通岗位历史趋势](plots/historical_trends.png)

![2025 年普通岗位预测与误差](plots/2025_test_errors.png)

![模型验证比较](plots/model_comparison.png)
'''
    (out/'2026预测说明.md').write_text(text,encoding='utf-8')
    print('Report and three figures saved.',flush=True)

if __name__=='__main__': run()
