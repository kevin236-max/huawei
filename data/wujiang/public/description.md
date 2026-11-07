# 乌江月径流预测 plus

TASK_ID=wujiang-monthly-runoff-forecast_plus

预测 9 个坝址的月径流“半还原”系列。原始单位未注明，保留数值尺度；日期使用自然年月 YYYY-MM。

公开训练数据覆盖 1951-05 至 2010-12。方法与参数选择使用宿主外部验证反馈，公开目录不提供内部验证划分。每次预测遵守 training_cutoff 与顺序披露历史的协议。
