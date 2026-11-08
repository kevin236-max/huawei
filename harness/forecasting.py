"""Monthly rolling forecasts: plain task data, windows, scores, and sandbox worker."""
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_task(path, horizon=3):
    """Read an existing monthly task directory without executing task code."""
    import yaml

    path = Path(path).resolve()
    benchmark = yaml.safe_load((path / "benchmark.yaml").read_text())
    if horizon not in benchmark["horizons"]:
        raise ValueError(f"Unsupported horizon: {horizon}")
    train = pd.read_csv(path / "public/train.csv", dtype={"month": str})
    history = pd.read_csv(path / "private/observations.csv", dtype={"month": str})
    splits = benchmark["splits"]
    stations = train.columns[1:].tolist()
    if not stations or train.columns[0] != "month" or list(history.columns) != list(train.columns):
        raise ValueError("Train and observations must have matching month and station columns")
    for frame, bounds in ((train, splits["train_observations"]),
                          (history, [splits["train_observations"][0], splits["test_targets"][1]])):
        expected = pd.period_range(*bounds, freq="M").astype(str).tolist()
        if frame.month.tolist() != expected:
            raise ValueError("Observation months must be continuous and match declared bounds")
        frame[stations] = frame[stations].astype(float)
        values = frame[stations].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Observations must be finite and positive")
    if not np.array_equal(train[stations].to_numpy(), history.iloc[:len(train)][stations].to_numpy()):
        raise ValueError("Public train must equal the corresponding observation history")
    description_path = path / "public/description.md"
    description = path.name
    if description_path.exists():
        after_title = False
        for line in description_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("# "):
                after_title = True
            elif after_title and line and not line.startswith(("#", ">", "TASK_ID=")):
                description = line.replace("public/train.csv", "公开训练表")
                break
    first_origin = pd.read_csv(path / f"public/h{horizon}/train_windows.csv", nrows=1).origin_month.iloc[0]
    splits["internal_validation_targets"] = internal_targets(splits, first_origin, horizon)
    return dict(path=path, train=train, history=history, stations=stations,
                splits=splits, description=description,
                horizon=horizon, first_origin=first_origin)


def internal_targets(splits, first_origin, horizon):
    """Validate the declared internal validation target months inside the public train span."""
    bounds = splits.get("internal_validation_targets")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("benchmark.yaml splits must declare internal_validation_targets: [YYYY-MM, YYYY-MM]")
    try:
        start, end = (pd.Period(str(month), freq="M") for month in bounds)
    except ValueError as error:
        raise ValueError(f"internal_validation_targets must use YYYY-MM months: {bounds}") from error
    train_start, train_end = (pd.Period(month, freq="M") for month in splits["train_observations"])
    if not train_start < start <= end <= train_end:
        raise ValueError("internal_validation_targets must lie inside train_observations, after its first month")
    if (end - start).n + 1 < horizon:
        raise ValueError("internal_validation_targets must cover at least one complete forecast window")
    if pd.Period(first_origin, freq="M") + horizon >= start:
        raise ValueError("Internal fit rows before internal_validation_targets need one complete training window")
    return [str(start), str(end)]


def windows(data, start, end, horizon):
    """Yield (history, target months, truth), with every target inside [start, end]."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    for first in pd.period_range(start, pd.Period(end, freq="M") - horizon + 1, freq="M"):
        targets = pd.period_range(first, periods=horizon, freq="M").astype(str).tolist()
        target_data = data.loc[data.month.isin(targets)]
        history = data.loc[data.month < targets[0]].copy()
        if target_data.month.tolist() != targets:
            raise ValueError(f"Missing or unordered target observations: {targets}")
        if history.empty:
            raise ValueError(f"No history before target month {targets[0]}")
        yield history, targets, target_data.iloc[:, 1:].to_numpy(dtype=float)


def score(predictions, truth):
    """Return one station-macro 1-MAPE per parameter, without clipping scores."""
    predictions, truth = np.asarray(predictions, dtype=float), np.asarray(truth, dtype=float)
    if (predictions.ndim != 4 or truth.ndim != 3
            or predictions.shape[0] != truth.shape[0]
            or predictions.shape[2:] != truth.shape[1:]
            or not all(predictions.shape)):
        raise ValueError("Expected predictions [origins, params, horizon, stations] and matching truth")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    if not np.isfinite(truth).all() or (truth <= 0).any():
        raise ValueError("Truth must be finite and positive")
    errors = np.abs(predictions - truth[:, None, :, :]) / truth[:, None, :, :]
    return (1 - errors.mean(axis=(0, 2)).mean(axis=1)).tolist()


def public_settings(task):
    return dict(horizon=task["horizon"], internal_validation_targets=task["splits"]["internal_validation_targets"],
                first_origin=task["first_origin"])


def task_context(task):
    splits, train = task["splits"], task["train"]
    internal_start, internal_end = splits["internal_validation_targets"]
    internal_cutoff = str(pd.Period(internal_start, freq="M") - 1)
    return "\n".join([
        task["description"],
        f"Stations ({len(task['stations'])}): {', '.join(task['stations'])}",
        f"Train: {' to '.join(splits['train_observations'])}; fit cutoff: {train.month.iloc[-1]}.",
        f"Validation targets: {' to '.join(splits['validation_targets'])}.",
        f"Final fit: {' to '.join(splits['final_fit_observations'])}.",
        f"Test targets: {' to '.join(splits['test_targets'])}.",
        f"Internal fit cutoff: {internal_cutoff}; internal targets: {internal_start} to {internal_end}.",
        f"Horizon: {task['horizon']}; first training origin: {task['first_origin']}.",
        "CSV preview:", train.head(3).to_csv(index=False).strip(),
    ])


def fit_candidate(path, train, horizon, params=None):
    """Load and fit candidate code. Call only inside a candidate sandbox."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate", path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(candidate)
        params = candidate.candidates if params is None else params
        if not isinstance(params, list) or not params or not all(isinstance(p, dict) for p in params):
            raise ValueError("candidates must be a nonempty list of parameter dictionaries")
        json.dumps(params, allow_nan=False)
        state = candidate.fit(train, horizon, params)
    return candidate, state, params


def predict_candidate(candidate, state, history, targets, params):
    """Invoke candidate prediction and check its public array contract."""
    with redirect_stdout(sys.stderr):
        output = candidate.predict(state, history, targets)
    if not isinstance(output, list) or len(output) != len(params):
        raise ValueError("predict must return one array per parameter dictionary")
    predictions = np.asarray(output, dtype=float)
    expected = (len(params), len(targets), history.shape[1] - 1)
    if predictions.shape != expected:
        raise ValueError(f"Prediction shape {predictions.shape}; expected {expected}")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    return predictions


def serve(path, selected_path=None):
    """JSONL worker entry point; the host launches this inside its sandbox."""
    request = json.loads(sys.stdin.readline())
    params = request.get("params")
    if params is None and selected_path is not None:
        selected = json.loads(Path(selected_path).read_text())
        params = [selected] if isinstance(selected, dict) else selected
    train = pd.DataFrame(request["train"])
    train = train.astype({column: float for column in train.columns[1:]})
    horizon = request["horizon"]
    candidate, state, params = fit_candidate(path, train, horizon, params)
    print(json.dumps(dict(candidates=params), allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        history = pd.DataFrame(request["history"], columns=train.columns)
        history = history.astype({column: float for column in train.columns[1:]})
        targets = request["targets"]
        if len(targets) != horizon:
            raise ValueError("Target count must equal horizon")
        predictions = predict_candidate(candidate, state, history, targets, params)
        print(json.dumps(dict(predictions=predictions.tolist()), allow_nan=False), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("candidate")
    parser.add_argument("selected", nargs="?")
    arguments = parser.parse_args()
    serve(arguments.candidate, arguments.selected)
