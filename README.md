### 月度径流预测：主 agent 与并发 subagent

- 主 agent 在同一个 Pi 会话中提出具体方案和调参策略，接收执行结果、安排改进或融合并选择已有候选；工具仅开放文件读取和目录查看。subagent 按 instruction 实现方案和执行调参，其他改进想法交回主 agent。
- 每轮最多三个 subagent，每个使用新 Pi 会话、独立工作区和固定的 12 核 CPU。
- 最多启动三批 subagent；第三批返回后，主 agent 在原会话中提交最终选择，不再启动第四批。正常跑满三批时主 agent 有四次决策（初始计划、两次后续计划、最终选择）。每个 subagent 在 480 秒软截止时收到一次收尾指令，额外收尾上限由 `worker_finish_timeout` 指定（默认 180 秒）；搜索总上限仍为 1 小时。
- 正式验证不限次数。正常结束可选择已验证候选；没有选择或搜索超时，自动使用全程正式验证最高的代码和参数。`result.json` 的 `selected_by` 记录最终候选来自主 agent 的选择（`main`）还是兜底（`fallback`）。
- 最终测试另受 `candidate_timeout` 限制，在搜索后执行一次；总耗时可能超过 1 小时一个测试的时间。
- 不设置 200G 内存限制，不在提示词中暴露 CPU 和时间配额；主 agent 会收到当前规划阶段及最终提交指令。CPU affinity 限定可用核，不代表宿主独占核。

### 安装

- 使用已固定依赖的 huawei Python 环境与已有模型服务，Pi 源码位于 `packages/pi`；安装脚本不安装或更新 Python 包。
- 在实际执行机器运行；需要 root 启动器权限、可用的 bubblewrap user/mount/network namespace 和 libseccomp。
- 安装过程下载依赖并构建 Pi，不启动模型服务或实验。

```bash
bash install.sh
cp .env.example .env
```

- 编辑 `.env` 中的模型地址、名称和 API key。支持 OpenAI Chat Completions 兼容服务，key 仅由宿主持有。
- `PI_RUNTIME_ROOT` 可指定本机运行目录，安装与运行时保持一致；默认 `.runtime/`。共享项目建议指定机器本地目录，例如 `/var/lib/runtime/runoff-pi`。
- `HUAWEI_PYTHON` 可指定系统依赖检查使用的 Python；同时修改 `config.yaml` 中的 `python`，使两者一致。
- Pi 依赖在运行目录中构建，不在 `packages/pi` 写入 node_modules/dist。
- `bash install.sh` 自动补装缺失的系统依赖（git、curl、tar、xz-utils、bubblewrap、libseccomp2、ca-certificates），普通用户仅在 APT 步骤通过 sudo 提权，可能需要输入密码。系统需支持 apt-get，Python 环境及其依赖需预先就绪。
- 需要代理时，在运行脚本前导出 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`，或对应小写变量；大小写同时设置时以小写为准。脚本自动启用 `NODE_USE_ENV_PROXY=1`，将代理传递给下载、构建和 sudo 下的 APT，不写入全局配置。代理地址使用实际可用的 HTTP(S) 代理地址。
- Node 版本不足或不支持环境代理时自动下载本地 Node；`--install-system-deps` 参数保留兼容，已不再需要。

### 运行

乌江径流可直接使用专用脚本（先完成上述安装及 `.env` 配置）：

```bash
bash scripts/run_wujiang.sh                 # 启动完整方法搜索、验证及最终测试
bash scripts/run_wujiang.sh --repeat 5      # 同一批次下串行执行 5 次独立搜索
bash scripts/run_wujiang.sh verify-data     # 仅校验乌江数据
bash scripts/run_wujiang.sh doctor          # 检查运行环境
bash scripts/run_wujiang.sh --config config.yaml
```

- 脚本和默认配置使用项目内的 `data/wujiang/` 数据集，运行与测试均不依赖外部任务仓库。默认预测未来 3 个月（由配置中的 `horizon` 控制）。
- 可从任意目录使用脚本的绝对路径调用；`--config` 等相对路径仍相对于调用目录。
- Python 默认与 `install.sh` 一致，可用 `HUAWEI_PYTHON` 指定；候选运行使用 `config.yaml` 的 `python`，更换环境时两处需保持一致。输出保存在项目的 `runs/` 下。

也可直接调用框架入口：

```bash
python run.py verify-data
python run.py doctor
python run.py run

# 切换到另一份现有月度径流数据包
python run.py verify-data --task /path/to/another/task
python run.py run --task /path/to/another/task

# 独立验证已有候选，不启动 agent
python run.py evaluate --candidate examples/seasonal_mean.py
```

- 默认任务为乌江 plus，H=3。任务配置直接填写数据目录，不需要注册适配器。
- 读取 `benchmark.yaml` 的时间划分、`public/train.csv` 的字段、`private/observations.csv` 的完整历史，以及训练窗口中的首个允许起报月。
- 仅公开训练数据进入 agent；候选沙箱逐月收到合法历史前缀，私有观测表与评分始终留在宿主。
- 站点数量、顺序、年月范围、数据说明由任务生成。评分统一为各站等权的 `1-MAPE`，只计算完整预测窗口。
- `doctor` 检查启动权限、CPU 数量并实际启动一个隔离探针；不支持隔离时直接报错。

### 角色与交接

- system prompt 按角色各一句任务描述（`harness/pi.py` 的 `SYSTEM_PROMPTS`），任务规则和数据都在 user 提示词中。
- `prompts/main.md`：方法设计、分步调参指令（每步比较内容、推进条件和停止条件）、两种验证的含义、结果分析、参考文件选择和最终候选选择。
- `prompts/subagent_finish.md`：软截止后的收尾指令，优先提交当前代码或返回已有结果；未完成时说明方案、进展和问题。
- `prompts/subagent.md`：数据与接口说明、本次 instruction、参考文件路径、内部验证（含逐年分数）与外部验证用法、按 main 的步骤执行、简短 JSON 交付（说明执行到哪一步）。
- 主 agent 全程复用同一个 Pi 进程和上下文，不自动压缩或按轮次重建。首次发送完整任务提示词，之后只追加新一批 subagent 的交付结果；不回灌 subagent 的完整调试对话。
- 最后一批结果之后追加两个换行 `\n\n` 和“请综合各轮实验结果，提交最终结果：候选文件路径和对应参数。”的收尾指令。三批执行上限由调度代码控制，即使返回新任务也不会启动第四批。
- 主工作区以只读方式挂载，框架持续发布各轮交付文件，主 agent 在同一会话中可读取最新结果。正常结束、提前选择、空任务、超时或异常时均清理主会话进程和 API 转发；主会话失败不会重启为空上下文，已有正式验证结果仍参与最终兜底选择。
- 主 agent 输出 `tasks`，每项包含直观的 `name`、具体 `instruction` 和 `files`。框架把文件及相邻结果说明复制到下一轮工作区的 `references/`。
- 交给主 agent 的每项结果以任务名 `name` 开头。subagent 返回 `file`、`validation_score`、`best_params`、`internal_score` 和 `summary`。外部分数、参数和内部分数都以宿主记录为准（内部分数由宿主对最佳参数重算）；文字分析是 subagent 的报告。
- 主 agent 的 JSON 无法读取或字段不符时，在同一会话追加一次说明原因的重发请求；subagent 的最终报告同样重发一次。某一轮只有思考、没有正文（如输出长度用满）时，追加一次“接着当前进展继续”。重发仍失败时：主 agent 按原规则结束搜索并兜底选择；subagent 保留宿主分数，并把原文截取到 `summary`。
- 正常报告后，代码和结果被复制到主工作区的 `results/roundN/方法名/`。融合任务也从这些文件开始，最终交付自包含单文件。
- 软截止通过 Pi steering 排入一次收尾消息，当前模型输出和工具调用继续执行，评估服务保持可用；注入内容记录在 `prompts.jsonl`。收尾上限或搜索总上限到达后才结束会话，最终测试另行限时。
- 硬超时优先回收已有正式验证最佳结果；没有成功提交则保留 `solution/` 下最新编辑的 Python 文件并注明无验证结果；两种情况都附上 subagent 最后一条回复的末尾，供下一轮了解进展。报告的文件或参数与所有评测记录都不一致时，改交该实验外部验证最高的版本并注明。参考文件路径不存在时跳过该文件并在 subagent 提示中注明，不中止实验。正常返回时也可用 `file: null` 交回方案和问题说明；已有正式结果仍由宿主保留。

### 候选接口

```python
candidates = [{"alpha": 1.0}, {"alpha": 10.0}]

def fit(train_data, num_months, candidates):
    return state

def predict(state, history_data, target_months):
    return predictions
```

- 文件按方法命名，例如 `seasonal_ridge.py`、`lag_lightgbm.py`、`ridge_tree_blend.py`。
- 数据是 `month` 字符串列和各站 float64 径流列；预测为每个参数方案一个 `[H, 站点数]` 数组，有限且非负。
- 每次评测拟合一次，随后逐月预测；最终测试只传入选中的一个参数字典，在扩展训练历史上重新拟合。
- subagent 工作区中的只读 `analysis.py` 只提供 `calculate_internal_validation_score`，使用公开训练表最后 36 个月，返回每组参数的分数和逐年分数 `by_year`。
- `bench evaluate --candidate solution/seasonal_ridge.py` 返回 `file/score/best_params/internal_score`（内部分数为宿主对 `best_params` 的公开内部验证）；代码副本位于只读 `submissions/`，源文件之后的修改不改变已有结果。
- 每次提交执行前复制候选。公开内部检查可显示调试错误；私有验证错误只返回简短说明（文件路径问题、超时、预测形状或数值等宿主侧检查），详细日志留在宿主。agent 沙箱设置 `PYTHONSAFEPATH=1`，模块只经 `PYTHONPATH` 查找。
- 候选正式执行禁用网络与外部程序执行，可使用同进程线程进行数值计算。线程默认值由 `native_threads` 设置；模型代码可以在其 12 核范围内自行安排计算。

### 目录

```text
data/wujiang/          乌江数据、时间划分及各预测长度的训练窗口
scripts/run_wujiang.sh 乌江径流运行入口
harness/run.py          三轮调度、文件交接、超时回收与最终选择
harness/evaluate.py     宿主候选评测与每个 subagent 的提交入口
harness/forecasting.py  任务加载、滚动窗口、评分与候选进程协议
harness/analysis.py     公开内部验证分数
harness/pi.py           Pi RPC 会话与最终 JSON 提取
harness/gateway.py      每会话模型 API 转发
harness/sandbox.py      隔离进程与 CPU 分配
harness/shared_proc.py  独立 UID 和后代进程清理
harness/seccomp.py      隔离内的执行限制
harness/relay.py        隔离内的 API 连接
```

- 运行生成 `runs/YYYYMMDD_HHMMSS/repeat_001/` 等目录。`--repeat` 默认 1，每次独立执行完整搜索与最终测试，各有完整时间预算；同一秒的批次目录重名时报错，不覆盖已有结果。
- 每次目录保留 `config.json`、`input/`、`main/`、`workers/roundN/方法名/`、`final/` 和原有 `result.json`。`result.json` 只追加 `start`、`finish`、`duration`：起止时间到秒并带时区，耗时以秒计，覆盖初始化、搜索、最终测试和清理。
- 批次外层仅有 `summary.json`，字段为 `repeat`、`completed`、`validation_mean`、`validation_std`、`test_mean`、`test_std`。只汇总成功运行，标准差采用样本标准差；没有成功结果时均值为 `null`，不足两次成功时标准差为 `null`。普通失败记录后继续，手动中断停止批次；未全部成功时命令退出码非零。
- 历史 10 次运行按前后各 5 次归入 `20261107_013847`、`20261107_135434`。各次 `start` 从原目录名恢复（原运行时区 UTC+08:00），缺少准确记录的 `finish` 和 `duration` 为 `null`。
- 主 agent 全程只有 `main/trajectory.jsonl`，各 subagent 分别写入 `workers/roundN/方法名/trajectory.jsonl`。轨迹按消息完成事件追加并立即刷新，不记录逐 token 增量。
- 每个会话保留首次输入 `prompt.txt`、全部发送输入 `prompts.jsonl` 和 `stderr.log`；主 agent 的各轮结果反馈及最终提交指令都在 `main/prompts.jsonl` 中。每次验证保留对应代码、参数与分数。
- 不复制旧版运行结果、不使用任务哈希或额外消息编号。

### 验证

```bash
python -m unittest discover -s tests -v
bash -n install.sh
```

- 测试覆盖滚动边界、单次 fit、最终单参数重训、三轮文件交接、并发、超时、API 转发和安装流程。
- 安装测试使用本地替代工具，不下载软件；集成测试使用真实数据与候选进程，但替代模型服务和隔离启动。
- 环境禁止本地 socket 时相关测试明确跳过。真实 Pi + bwrap + 模型端到端运行需在目标执行机器完成。

### 本地版本管理

- 使用标准 `.git` 仓库，通过 `git status`、`git add`、`git commit` 管理项目。
- `runs/`、`results/`、`.runtime/`、缓存、`.env` 和本机凭据配置不纳入版本管理。
- `packages/pi` 保留为 Git 子模块，主仓库记录源码提交。新机器使用 `git clone --recurse-submodules <仓库地址>`，已有克隆可运行 `git submodule update --init --recursive`，再执行 `bash install.sh`。
