"""Build CAMELS monthly runoff task packages in the data/wujiang layout.

Source: data/data_20261106/data/runoff/CAMELS_*_monthly_m3s.csv (monthly mean m3/s).
Stations and splits were chosen by data quality first (provider flags, low-flow share,
trend, duplicate gauges), then by matching flow distribution across splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/data_20261106/data/runoff"
HORIZONS = [3, 6, 14]
HISTORY_MONTHS = 12  # first origin = start + 11 months, as in data/wujiang

TASKS = {
    "camels_us": dict(
        source="US", name="CAMELS-US", stations=["02300700", "06154410", "07299670"],
        train=("1980-01", "2000-09"), internal=("1997-10", "2000-09"),
        validation=("2000-10", "2004-09"), test=("2004-10", "2009-09"),
        notes="02300700 Bullfrog Creek (FL)、06154410 Little Peoples Creek (MT)、07299670 Groesbeck Creek (TX)，互不嵌套。"),
    "camels_ind": dict(
        source="IND", name="CAMELS-IND", stations=["5002", "12026"],
        train=("1980-01", "2008-12"), internal=("2006-01", "2008-12"),
        validation=("2009-01", "2012-12"), test=("2013-01", "2017-12"),
        notes="5002 Biligundulu (Cauvery)、12026 Hoshangabad (Narmada)；12025 与 12026 同河上下游、高度重复，只保留 12026。"),
    "camels_aus": dict(
        source="AUS", name="CAMELS-AUS", stations=["401210", "405214", "405263"],
        train=("1975-06", "2002-10"), internal=("1999-11", "2002-10"),
        validation=("2002-11", "2006-10"), test=("2006-11", "2011-10"),
        notes="401210 Snowy Creek、405214 Delatite River、405263 Goulburn 支流，均位于 Murray-Darling 流域；验证与测试段处于千年干旱期。"),
}


def months(start, end):
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def build(task_id, spec):
    flows = pd.read_csv(SOURCE / f"CAMELS_{spec['source']}_monthly_m3s.csv", dtype={"month": str})
    flows = flows.set_index("month")[spec["stations"]]
    start, train_end = spec["train"]
    full = flows.loc[start:spec["test"][1]]
    if full.index.tolist() != months(start, spec["test"][1]):
        raise ValueError(f"{task_id}: months not continuous")
    values = full.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{task_id}: observations must be finite and positive")

    out = ROOT / "data" / task_id
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    (out / "private").mkdir(parents=True)
    (out / "public").mkdir()
    observations = full.reset_index()
    observations.to_csv(out / "private/observations.csv", index=False)
    observations.loc[observations.month <= train_end].to_csv(out / "public/train.csv", index=False)

    first_origin = pd.Period(start, freq="M") + (HISTORY_MONTHS - 1)
    for horizon in HORIZONS:
        last_origin = pd.Period(train_end, freq="M") - horizon
        rows = []
        for origin in pd.period_range(first_origin, last_origin, freq="M"):
            rows.append(dict(window_id=f"train_h{horizon}_{origin}", origin_month=str(origin), horizon=horizon,
                             target_start=str(origin + 1), target_end=str(origin + horizon)))
        (out / f"public/h{horizon}").mkdir()
        pd.DataFrame(rows).to_csv(out / f"public/h{horizon}/train_windows.csv", index=False)

    benchmark = dict(
        horizons=HORIZONS,
        splits=dict(
            train_observations=[start, train_end],
            internal_validation_targets=list(spec["internal"]),
            validation_targets=list(spec["validation"]),
            final_fit_observations=[start, spec["validation"][1]],
            test_targets=list(spec["test"]),
            incomplete_horizon_policy="exclude_origin",
            invariant="every supervised training target must be at or before the fit cutoff",
        ),
    )
    (out / "benchmark.yaml").write_text(yaml.safe_dump(benchmark, sort_keys=False, allow_unicode=True))

    n = len(spec["stations"])
    (out / "public/description.md").write_text(
        f"# {spec['name']} 月径流预测\n\n"
        f"TASK_ID={task_id.replace('_', '-')}-monthly-runoff-forecast\n\n"
        f"预测 {n} 个站点的月平均流量，单位 m³/s（由 {spec['name']} 日流量按自然月求平均）；"
        f"日期使用自然年月 YYYY-MM。\n\n"
        f"公开训练数据覆盖 {start} 至 {train_end}。方法与参数选择使用宿主外部验证反馈，"
        f"公开目录不提供内部验证划分。每次预测遵守 training_cutoff 与顺序披露历史的协议。\n")
    (out / "README.md").write_text(
        f"# {spec['name']} 月径流数据\n\n"
        f"由 `scripts/build_camels_tasks.py` 从 `data/data_20261106/data/runoff/` 生成，格式与 `data/wujiang/` 一致。\n\n"
        f"- 站点（{n}）：{spec['notes']}\n"
        f"- `public/train.csv`：{start} 至 {train_end} 的公开训练数据。\n"
        f"- `private/observations.csv`：{start} 至 {spec['test'][1]} 的完整观测，仅供宿主评测使用。\n"
        f"- `public/h{{3,6,14}}/train_windows.csv`：各预测长度允许的训练起报窗口。\n"
        f"- `benchmark.yaml`：支持的预测长度，以及训练、验证和测试时间划分。\n\n"
        f"验证目标月份为 {spec['validation'][0]} 至 {spec['validation'][1]}，"
        f"测试目标月份为 {spec['test'][0]} 至 {spec['test'][1]}。\n"
        f"内部验证目标月份为 {spec['internal'][0]} 至 {spec['internal'][1]}"
        f"（`benchmark.yaml` 的 `internal_validation_targets`），位于公开训练段内。\n\n"
        f"站点筛选：去掉提供方估算或受影响比例超过 10%、低于中位数 10% 的月份超过 3%、"
        f"季节中位数基线 1-MAPE 低于 0.1、年径流趋势超过每 10 年 20% 的站点，"
        f"以及缺测导致无法组成公共连续区间的站点。\n")
    return out


if __name__ == "__main__":
    for task_id, spec in TASKS.items():
        print(build(task_id, spec))
