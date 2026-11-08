# CAMELS-IND 月径流数据

由 `scripts/build_camels_tasks.py` 从 `data/data_20261106/data/runoff/` 生成，格式与 `data/wujiang/` 一致。

- 站点（2）：5002 Biligundulu (Cauvery)、12026 Hoshangabad (Narmada)；12025 与 12026 同河上下游、高度重复，只保留 12026。
- `public/train.csv`：1980-01 至 2008-12 的公开训练数据。
- `private/observations.csv`：1980-01 至 2017-12 的完整观测，仅供宿主评测使用。
- `public/h{3,6,14}/train_windows.csv`：各预测长度允许的训练起报窗口。
- `benchmark.yaml`：支持的预测长度，以及训练、验证和测试时间划分。

验证目标月份为 2009-01 至 2012-12，测试目标月份为 2013-01 至 2017-12。
内部验证目标月份为 2006-01 至 2008-12（`benchmark.yaml` 的 `internal_validation_targets`），位于公开训练段内。

站点筛选：去掉提供方估算或受影响比例超过 10%、低于中位数 10% 的月份超过 3%、季节中位数基线 1-MAPE 低于 0.1、年径流趋势超过每 10 年 20% 的站点，以及缺测导致无法组成公共连续区间的站点。
