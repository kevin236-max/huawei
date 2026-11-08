# CAMELS-US 月径流预测

TASK_ID=camels-us-monthly-runoff-forecast

预测 3 个站点的月平均流量，单位 m³/s（由 CAMELS-US 日流量按自然月求平均）；日期使用自然年月 YYYY-MM。

公开训练数据覆盖 1980-01 至 2000-09。方法与参数选择使用宿主外部验证反馈，公开目录不提供内部验证划分。每次预测遵守 training_cutoff 与顺序披露历史的协议。
