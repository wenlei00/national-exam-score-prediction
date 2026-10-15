# 国考岗位最低进面分预测

一个从岗位表、进面名单到时间验证和预测的数据科学项目。使用岗位资格条件、地区、部门及历史同类岗位特征，研究报名开始前的最低进面分预测。

**主流程严格保留2026作为独立检验集：模型训练标签截至2025年，预测2027时也不把2026真实分数用于历史特征。** 加入2026重训的版本仅作为显式选择的对照实验。

## 已完成的研究

- 核验2019—2025年度岗位与进面分，得到106,865条可训练年度岗位记录。
- 按年份验证，比较同类别中位数基线、岭回归、随机森林与LightGBM。
- 固定预测后再检验2026首批名单，匹配20,227个岗位；487个预测岗位不在首批名单内，未填零。
- 原冻结模型在2026普通岗位上的MAE为5.26分、RMSE为6.53分，平均低估2.32分。
- 已对用户提供的2027表内全部19,426个岗位生成冻结主版本预测；2027实际精度尚未验证。

指标和适用范围见[模型说明](docs/MODEL_CARD.md)。这个项目不预测个人上岸概率，不提供保证分数线。不同专业考试岗位保留原始口径，不能直接横向比较分数。

## 快速开始

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

将 `config.example.json` 复制为 `config.local.json`，按[数据约定](docs/DATA.md)配置本地输入。要求Python 3.12；仓库不分发原始Excel、人员名单、模型文件或逐岗位结果。

```bash
python src/cli.py audit
python src/cli.py build-panel
python src/cli.py prepare-target
python src/cli.py train-frozen
python src/cli.py evaluate-2026
python src/cli.py predict-2027
```

在已完成的本地项目中，不必重跑训练，直接 `python src/cli.py predict-2027` 即可使用现有冻结模型。
命令拒绝覆盖已存在的冻结模型；要重新训练，请使用新的独立项目副本。

主版本结果：`results/frozen_2027/2027_prediction_frozen.csv`。旧的、加入2026重训的结果位于另一目录，其检验边界见[复现说明](docs/REPRODUCIBILITY.md)。

可选对照实验需要显式标志：

```bash
python src/cli.py refit-2027 --include-2026-training
```

## 目录

```text
src/                 数据读取、清洗、特征、训练、评估和命令入口
notebooks/           查看阶段数据和历史结果
config.example.json  可移植的输入配置示例
requirements.txt     固定的研究环境依赖
tests/               合成数据的时间边界和数据规则测试
docs/                数据口径、模型边界和复现方法
.github/workflows/   自动测试配置
```

`data/`、`results/`、本地配置和虚拟环境均由Git忽略。文档仅保留汇总指标和自行生成的研究图表，不含人员字段。数据年度和来源需要独立核验，不能只相信文件名。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

CI使用合成数据运行规则测试，不下载真实数据，也不把合成测试通过当作真实预测精度证明。

## 研究参考

[论文方法评审](研究参考评审.md)记录了所参考OLS研究的变量设计和需核验问题。本项目重新估计模型，不照搬其系数或把SHAP解释为因果权重。
