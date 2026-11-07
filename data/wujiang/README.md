# 乌江月径流数据

本目录保存乌江 monthly-runoff-forecast_plus 任务所需的独立数据副本。
观测 CSV、训练窗口和数据说明从原数据包原样复制；`benchmark.yaml`
仅保留本框架读取的预测长度和时间划分。运行时无需访问原数据包或其代码。

- `public/train.csv`：1951-05 至 2010-12 的公开训练数据，包含 9 个坝址。
- `private/observations.csv`：1951-05 至 2023-04 的完整观测，仅供宿主评测使用。
- `public/h{3,6,14}/train_windows.csv`：各预测长度允许的训练起报窗口。
- `public/description.md`：数据含义及单位说明。
- `benchmark.yaml`：支持的预测长度，以及训练、验证和测试时间划分。

验证目标月份为 2011-01 至 2015-12，测试目标月份为 2016-01 至 2023-04。
默认预测长度为 3 个月，可在项目 `config.yaml` 中修改 `horizon`。

在项目根目录运行 `bash scripts/run_wujiang.sh verify-data` 可校验数据。
