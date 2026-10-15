# 复现流程与版本

Python 3.12，依赖见根目录 requirements.txt。原始数据不随仓库分发，完整复现须自行配置真实输入；CI仅使用合成数据核验研究约束，不声称CI复现了真实指标。

## 冻结实验

1. 使用 config.example.json 建立本地 config.local.json。
2. `python src/cli.py audit`：读取显式输入并记录字段及校验值，不读取2026成绩。
3. `python src/cli.py build-panel`：按年度岗位合并历史分数。
4. `python src/cli.py prepare-target`：验证2026岗位表及年度，建立预测输入。
5. `python src/cli.py train-frozen`：按年份选模，保存原模型、2026预测及校验值。已存在冻结模型时命令拒绝覆盖；需要重新研究时使用独立目录。
6. `python src/cli.py evaluate-2026`：此时才读取2026首批名单，不重训。
7. `python src/cli.py predict-2027`：使用原冻结模型，参数及历史特征均不使用2026标签。

第5步会保存原始预测；第6步在读取实际结果前后校验原预测及模型哈希。实际标签保存在独立评估目录，不并入冻结训练面板。
主版本2027结果写入 `results/frozen_2027/2027_prediction_frozen.csv`。

## 可选重训对照

只有显式选择时才运行：

```bash
python src/cli.py refit-2027 --include-2026-training
```

它使用2019—2026数据拟合，输出到 `results/prediction_2027/`，与主版本分别保存。
2026此时是该版开发数据；不能引用原冻结模型的独立检验成绩来证明此版精度。
直接运行旧脚本 `predict_2027.py` 同样需要此标志，防止误把重训当作默认预测。

## 测试范围

`python -m unittest discover -s tests -v` 使用合成数据检查：

- 字符串代码的前导零与不截断规则；
- 当年标签或未来标签扰动不影响当年/过去年份特征；
- 历史统计只看此前三年；
- 无标签/不参与训练的记录不进入历史分数统计；
- 冻结配置拒绝2026及以后标签进入训练清单；
- 默认路径可在另一台机器配置。

真实数据上的岗位数量、缺失、名称冲突和哈希验证由阶段脚本完成。输入格式或模型发生变化后需要重新验证受影响部分；合成测试不证明预测准确。
