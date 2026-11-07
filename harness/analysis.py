"""Public internal validation; available inside each candidate workspace."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .forecasting import fit_candidate, predict_candidate, score, windows
else:
    from forecasting import fit_candidate, predict_candidate, score, windows


def calculate_internal_validation_score(path, selected_params=None):
    """Fit once before the final 36 public months, then score complete rolling windows."""
    settings = json.loads(Path("input/task.json").read_text())
    data = pd.read_csv("input/train.csv", dtype={"month": str})
    data = data.astype({column: float for column in data.columns[1:]})
    horizon = settings["horizon"]
    validation_months = settings["internal_validation_months"]
    if len(data) <= validation_months:
        raise ValueError("Internal validation needs training observations before its target months")
    train = data.iloc[:-validation_months].copy()
    start, end = data.month.iloc[-validation_months], data.month.iloc[-1]
    params = None if selected_params is None else [selected_params]
    candidate, state, params = fit_candidate(path, train, horizon, params)
    predictions, truth = [], []
    for history, targets, actual in windows(data, start, end, horizon):
        predictions.append(predict_candidate(candidate, state, history, targets, params))
        truth.append(actual)
    scores = score(predictions, truth)
    best_index = int(np.argmax(scores))
    return dict(results=[dict(params=p, score=s) for p, s in zip(params, scores)],
                best_index=best_index, best_params=params[best_index], best_score=scores[best_index])
