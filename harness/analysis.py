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
    """Fit once before the final 36 public months, then score complete rolling windows.

    Each result also carries ``by_year``: the same score restricted to records whose
    target month falls in that calendar year, to check whether a gain is consistent.
    """
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
    predictions, truth, years = [], [], []
    for history, targets, actual in windows(data, start, end, horizon):
        predictions.append(predict_candidate(candidate, state, history, targets, params))
        truth.append(actual)
        years.append([month[:4] for month in targets])
    scores = score(predictions, truth)
    predictions, truth, years = np.asarray(predictions, dtype=float), np.asarray(truth, dtype=float), np.asarray(years)
    # Relative error per (origin, param, lead, station); a year averages its own records per station.
    errors = np.abs(predictions - truth[:, None]) / truth[:, None]
    by_year = []
    for index in range(len(params)):
        yearly = {}
        for year in sorted(set(years.ravel())):
            mask = years == year
            yearly[year] = float(1 - errors[:, index][mask].mean(axis=0).mean())
        by_year.append(yearly)
    best_index = int(np.argmax(scores))
    return dict(results=[dict(params=p, score=s, by_year=y) for p, s, y in zip(params, scores, by_year)],
                best_index=best_index, best_params=params[best_index], best_score=scores[best_index])
