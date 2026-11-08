### 当前任务

- 根据 main 给出的实验指令，实现未来数月径流预测的 Python 候选，按指令中的调参策略逐步执行，完成内部检查和外部验证，并报告最佳结果。
- 方法、参数范围和执行步骤以本次实验指令为准。实现细节和执行错误由你处理；指令之外的改进想法写入 summary，由 main 安排后续实验。

### 数据

${task_context}

- 公开训练表为 `input/train.csv`。日期边界、站点顺序、预测长度、内部验证区间和首个允许训练起报月见上文。
- 每次评测先使用允许的训练段拟合一次，再逐月提供截至起报月的真实历史，预测其后的连续月份。最终测试在更长的训练段上重新拟合选定方案。

### 候选接口

- 每个候选是能够在新进程独立执行的 Python 文件，必要辅助函数写在同一文件中，顶层定义依赖、参数列表和以下接口：

```python
candidates = [...]  # 非空列表，每项为完整方案的参数字典


def fit(train_data, num_months, candidates):
    return state


def predict(state, history_data, target_months):
    return predictions
```

- 顶层 `candidates` 是准备比较的整批方案。`fit` 的 `candidates` 是本次实际执行的方案；验证传入整批，最终测试只传入 `[选中的参数]`。
- 每项参数完整描述可独立执行的方案，包括可调特征、模型参数、成员组合、融合权重和随机种子。使用 JSON 支持的类型，浮点数必须有限。验证确定的轮数、权重和校准系数保存在代码或参数中。
- `train_data` 和 `history_data` 是 pandas.DataFrame，第一列 `month` 为 `YYYY-MM` 字符串，其余列为站点径流 float64，与训练 CSV 列顺序一致；按月递增、连续、无缺失，径流有限且为正。
- `train_data` 是允许用于拟合的数据；`num_months` 是预测月份数；`state` 保存模型、统计量、共享特征和参数，供随后各次预测复用。
- `history_data` 包含截至本次起报月的完整真实观测，末行是起报月；`target_months` 是其后连续 `num_months` 个月的 `list[str]`。起报月推进后，历史中加入新发生的真实观测。
- `predict` 返回 numpy.ndarray 组成的列表，与传给 `fit` 的参数顺序一一对应。每个数组形状为 `[len(target_months), 站点数]`，按目标月份和 CSV 站点顺序排列；预测必须是原始径流尺度的有限非负值。只有一个方案时也返回长度为 1 的列表。
- 整批与单参数执行必须遵循相同训练和预测规则，不能依赖参数在整批中的位置或其他方案的存在。
- 构造滞后和自然月特征时，可以先把 `month` 换成整数月序（如 `int(m[:4]) * 12 + int(m[5:7]) - 1`），再按位置取值，避免日期类型和字符串拼接出错。

### 参考代码

${references}

- 参考文件按上面的路径读取。根据本次指令复用其方法、参数或成员实现，在自己的 `solution/` 下编写可以独立执行的候选。
- 融合时把需要的成员实现包含在候选文件内，并记录所用成员参数和权重；最终执行不能依赖参考文件继续存在。

### 评分

- `score = 1 - mean_station(mean_records(abs(prediction - truth) / truth))`，越高越好。各站先汇总全部预测记录，再等权平均；同一目标月份在不同起报和提前期下分别计分。
- 这是相对误差：同样的绝对误差在低流量月份影响更大；高估的误差没有上限，低估的误差最多为 1。
- 转换目标或使用其他训练损失时，预测仍需还原到原始径流尺度。融合在原始尺度上先组合预测再评分，成员分数的加权平均不代表融合分数。

### 训练要求

- 同一候选中，各方案通过 `fit` 和 `predict` 共享数据处理、成员训练和基础预测；只改变融合权重、缩放或截断时复用成员预测。最终单参数重训只准备选中方案需要的成员。
- 模型和预处理统计量只能使用传入的 `train_data` 拟合。训练样本的输入和目标都在该训练段内，样本起报不早于任务给出的首个允许起报月；训练与预测使用同一特征定义。
- `predict` 使用固定训练结果和方案参数，按当前历史构造输入。需要早停时，先在训练段内部留出数据选轮，再将确定轮数纳入方案；校准残差须来自未拟合对应目标的模型。
- 每个随机成员明确设置种子，使整批与单独执行一致。
- 正式执行仅携带候选代码和参数，不能依赖临时模型、缓存、参考文件或 `analysis`；候选加载与 `fit`、`predict` 期间不向标准输出打印。搜索和分析脚本可以打印所需结果。

### 内部验证

- 工作区中的 `analysis.py` 是只读的公开内部验证工具，直接 `from analysis import calculate_internal_validation_score` 即可，导入成功时没有输出。
- 内部验证使用上文给出的内部验证区间（公开训练表的最后一段）：按上文的内部训练截止月拟合一次，再逐月提供真实历史，只计算目标全部位于验证段的完整窗口。可以反复运行，出错时能看到完整报错。
- `calculate_internal_validation_score(path, selected_params=None)` 默认比较文件中整批参数，指定参数字典时只评测该方案。返回 `results`（每项含 `params`、`score` 和逐年分数 `by_year`）、`best_params` 和 `best_score`。逐年分数用于检查改进是否在各年份都成立。

```bash
python - <<'PY'
from analysis import calculate_internal_validation_score
result = calculate_internal_validation_score('solution/seasonal_ridge.py')
for item in result['results']:
    print(item['params'], round(item['score'], 4), {k: round(v, 4) for k, v in item['by_year'].items()})
print(result['best_params'])
PY
```

### 外部验证与提交

- 工作目录为 `/workspace`，Python 环境和命令已经配置，使用现有依赖即可，无需激活 Conda 或安装软件包。`input/` 为只读数据；候选代码写入 `solution/`，用方法名命名，例如 `solution/seasonal_ridge.py`；临时文件放 `/tmp`。
- 提交外部验证：

```bash
bench evaluate --candidate solution/seasonal_ridge.py
```

- 返回保存代码的 `file`、该文件外部验证最高的 `score`、对应的 `best_params`，以及系统对该参数重新计算的内部分数 `internal_score`；失败时返回简短说明。外部验证不提供报错细节和逐年结果，调试请用内部验证。
- 每次提交等评测结束后再修改或提交下一版。保存的代码可以通过返回的 `file` 路径读取。

### 执行方式

- 按调参策略的步骤依次执行：每完成一步，根据结果判断是否进入下一步；满足停止条件就结束，不自行增加指令之外的比较。
- 指令要求外部验证的节点按要求提交；指令没有指定时，至少用外部验证确认最终版本。
- 最终报告已提交且外部验证分数最高的版本及参数；指令另有选择标准时按指令执行。

### 输出

最终答复返回一个 JSON 对象，包含以下字段：

- `file`：最佳外部验证返回的保存代码路径，与 `bench evaluate` 的返回值一致。没有成功评估时填写已有候选路径，尚未生成代码时填 `null`。
- `validation_score`：该候选的外部验证分数，没有成功评估时填 `null`。
- `best_params`：照抄该次外部验证返回的 `best_params`，没有成功评估时填 `null`。
- `internal_score`：该次外部验证返回的 `internal_score`，没有则填 `null`。
- `summary`：本次方案、执行到了调参策略的哪一步、各步主要结果（内部分数及逐年是否稳定、外部分数）、结论和后续建议；未执行的步骤说明原因，尚未获得结果时说明已完成内容和当前问题。内容简洁。

```json
{"file":"submissions/2/seasonal_ridge.py","validation_score":0.47,"best_params":{"alpha":10.0},"internal_score":0.52,"summary":"…"}
```

### 本次实验指令

${instruction}

按上述步骤完成实现、调参和外部验证后，按输出格式报告最佳已评估结果。实现过程中发现的其他改进方向写入 summary，交由 main 决定后续安排。
