"""Compare shared seasonal means and the latest observed runoff."""
import numpy as np

candidates = [dict(season_weight=weight) for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]


def fit(train_df, horizon, params):
    seasonal = train_df.iloc[:, 1:].groupby(train_df.month.str[-2:]).mean()
    return dict(seasonal=seasonal, params=params)


def predict(state, history_df, target_months):
    seasonal = state["seasonal"].loc[[month[-2:] for month in target_months]].to_numpy(dtype=float)
    latest = history_df.iloc[-1, 1:].to_numpy(dtype=float)
    persistence = np.repeat(latest[None, :], len(target_months), axis=0)
    return [p["season_weight"] * seasonal + (1 - p["season_weight"]) * persistence
            for p in state["params"]]
