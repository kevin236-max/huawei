import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import pandas as pd
import yaml

from harness import forecasting


ROOT = Path(__file__).resolve().parents[1]


def observations(start="2000-01", periods=48, stations=2):
    return pd.DataFrame({
        "month": pd.period_range(start, periods=periods, freq="M").astype(str),
        **{f"station_{i}": np.full(periods, (i + 1) * 10.0) for i in range(stations)},
    })


class ForecastingTests(unittest.TestCase):
    def test_existing_task_packages_load_for_every_declared_horizon(self):
        manifests = sorted((ROOT / "data").glob("*/benchmark.yaml"))
        if not manifests:
            self.skipTest("Local task data is not present in this project's data directory")
        for manifest in manifests:
            for horizon in yaml.safe_load(manifest.read_text())["horizons"]:
                with self.subTest(path=manifest.parent.name, horizon=horizon):
                    task = forecasting.load_task(manifest.parent, horizon)
                    self.assertEqual(task["stations"], task["train"].columns[1:].tolist())
                    for split in ("validation", "test"):
                        start, end = task["splits"][split + "_targets"]
                        history, targets, truth = next(forecasting.windows(task["history"], start, end, horizon))
                        self.assertEqual(history.month.iloc[-1], str(pd.Period(start, freq="M") - 1))
                        self.assertEqual(targets[0], start)
                        self.assertEqual(truth.shape, (horizon, len(task["stations"])))
                        last_start = str(pd.Period(end, freq="M") - horizon + 1)
                        final = list(forecasting.windows(task["history"], last_start, end, horizon))
                        self.assertEqual(len(final), 1)
                        self.assertEqual(final[0][1][-1], end)

    def test_windows_start_before_first_target_and_exclude_incomplete_horizons(self):
        data = observations(periods=8)
        windows = list(forecasting.windows(data, "2000-04", "2000-08", 3))
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0][0].month.tolist(), ["2000-01", "2000-02", "2000-03"])
        self.assertEqual(windows[0][1], ["2000-04", "2000-05", "2000-06"])
        self.assertEqual(windows[-1][0].month.iloc[-1], "2000-05")
        self.assertEqual(windows[-1][1], ["2000-06", "2000-07", "2000-08"])
        np.testing.assert_array_equal(windows[0][2], [[10.0, 20.0]] * 3)

    def test_windows_fail_instead_of_silently_reducing_scores_for_missing_months(self):
        data = observations(periods=8).drop(index=4)
        with self.assertRaisesRegex(ValueError, "target"):
            list(forecasting.windows(data, "2000-04", "2000-08", 3))

    def test_station_macro_score_selects_batch_and_does_not_clip(self):
        truth = np.array([[[10.0, 1000.0]], [[20.0, 2000.0]]])
        predictions = np.array([
            [[[10.0, 1000.0]], [[20.0, 1000.0]], [[40.0, 1000.0]]],
            [[[20.0, 2000.0]], [[40.0, 2000.0]], [[80.0, 2000.0]]],
        ])
        self.assertEqual(forecasting.score(predictions, truth), [1.0, 0.5, -0.5])

    def test_score_rejects_wrong_shape_or_invalid_predictions(self):
        truth = np.ones((2, 3, 2))
        for predictions in (np.ones((2, 1, 2, 2)), np.full((2, 1, 3, 2), -1.0),
                            np.full((2, 1, 3, 2), np.nan)):
            with self.subTest(shape=predictions.shape), self.assertRaises(ValueError):
                forecasting.score(predictions, truth)

    def test_load_task_uses_directory_dates_and_station_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public/h3").mkdir(parents=True)
            (root / "private").mkdir()
            data = observations("1987-03", 66, 4)
            data.iloc[:48].to_csv(root / "public/train.csv", index=False)
            data.to_csv(root / "private/observations.csv", index=False)
            (root / "benchmark.yaml").write_text(textwrap.dedent("""\
                horizons: [3, 6]
                splits:
                  train_observations: ['1987-03', '1991-02']
                  internal_validation_targets: ['1989-03', '1991-02']
                  validation_targets: ['1991-03', '1991-11']
                  final_fit_observations: ['1987-03', '1991-11']
                  test_targets: ['1991-12', '1992-08']
                """))
            (root / "public/h3/train_windows.csv").write_text(
                "origin_month,target_start\n1988-02,1988-03\n")
            (root / "public/description.md").write_text("# Basin data\n\nFour stations in another basin.\n")
            task = forecasting.load_task(root)
            self.assertEqual(task["stations"], ["station_0", "station_1", "station_2", "station_3"])
            self.assertEqual(task["train"].month.iloc[-1], "1991-02")
            self.assertEqual(len(task["history"]), 66)
            self.assertEqual(task["splits"]["test_targets"], ["1991-12", "1992-08"])
            self.assertEqual(forecasting.public_settings(task), {
                "horizon": 3, "internal_validation_targets": ["1989-03", "1991-02"], "first_origin": "1988-02"})
            context = forecasting.task_context(task)
            for value in ("Four stations", "1991-02", "1991-03", "1992-08", "1988-02", "station_3",
                          "Internal fit cutoff: 1989-02; internal targets: 1989-03 to 1991-02"):
                self.assertIn(value, context)

            manifest = (root / "benchmark.yaml").read_text()
            for bounds in (None, "['1987-03', '1991-02']", "['1989-03', '1991-03']",
                           "['1991-02', '1991-02']", "['1988-04', '1991-02']", "['1989-13', '1991-02']"):
                with self.subTest(internal=bounds):
                    lines = [line for line in manifest.splitlines() if "internal_validation_targets" not in line]
                    if bounds is not None:
                        lines.insert(3, f"  internal_validation_targets: {bounds}")
                    (root / "benchmark.yaml").write_text("\n".join(lines) + "\n")
                    with self.assertRaises(ValueError):
                        forecasting.load_task(root)
            (root / "benchmark.yaml").write_text(manifest)

            (root / "public/description.md").write_text(
                "> old access policy\n\n# Another basin\n\nTASK_ID=old-name\n\n"
                "Monthly discharge in cubic metres per second.\n"
                "Old H=3 interface with public/rolling_protocol.py.\n\n## Old security\nDo legacy things.\n")
            brief = forecasting.load_task(root)
            context = forecasting.task_context(brief)
            self.assertIn("Monthly discharge in cubic metres per second.", context)
            for stale in ("TASK_ID=", "old access", "Old H=3", "rolling_protocol", "##", "Do legacy"):
                self.assertNotIn(stale, context)
            brief["history"].iloc[-1, 1] = 123456789.0
            self.assertNotIn("123456789", forecasting.task_context(brief))

            for failure in ("missing_month", "bad_value", "wrong_columns", "wrong_train"):
                with self.subTest(failure=failure):
                    invalid = data.copy()
                    public = data.iloc[:48].copy()
                    if failure == "missing_month":
                        invalid = invalid.drop(index=50)
                    elif failure == "bad_value":
                        invalid.iloc[50, 1] = -1.0
                    elif failure == "wrong_columns":
                        invalid = invalid.rename(columns={"station_0": "different_station"})
                    else:
                        public.iloc[0, 1] = 11.0
                    invalid.to_csv(root / "private/observations.csv", index=False)
                    public.to_csv(root / "public/train.csv", index=False)
                    with self.assertRaises(ValueError):
                        forecasting.load_task(root)

    def test_worker_fits_once_and_accepts_only_selected_final_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidate = directory / "method.py"
            candidate.write_text(textwrap.dedent("""\
                import numpy as np
                candidates = [{'scale': 1}, {'scale': 2}]
                def fit(train, horizon, params):
                    print('candidate diagnostic')
                    with open('fits.txt', 'a') as file:
                        file.write(train.month.iloc[-1] + '\\n')
                    return dict(cutoff=train.month.iloc[-1], params=params)
                def predict(state, history, targets):
                    assert state['cutoff'] == '2000-03'
                    return [np.full((len(targets), history.shape[1]-1),
                                    len(history) * p['scale']) for p in state['params']]
                """))
            selected = directory / "selected.json"
            selected.write_text(json.dumps({"scale": 7}))
            data = observations(periods=8)
            requests = [dict(train=data.iloc[:3].to_dict("records"), horizon=3, params=None)]
            requests.extend(dict(history=history.to_dict("records"), targets=targets)
                            for history, targets, _ in forecasting.windows(data, "2000-04", "2000-08", 3))
            result = subprocess.run(
                [sys.executable, str(ROOT / "harness/forecasting.py"), "--serve", str(candidate), str(selected)],
                input="".join(json.dumps(message) + "\n" for message in requests),
                text=True, capture_output=True, cwd=directory, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(responses[0], {"candidates": [{"scale": 7}]})
            self.assertEqual([response["predictions"][0][0][0] for response in responses[1:]], [21, 28, 35])
            self.assertEqual((directory / "fits.txt").read_text(), "2000-03\n")
            self.assertIn("candidate diagnostic", result.stderr)

    def test_worker_rejects_wrong_prediction_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "wrong.py"
            candidate.write_text("candidates=[{}]\ndef fit(*args): return None\ndef predict(*args): return [[[1.0]]]\n")
            data = observations(periods=5)
            requests = [dict(train=data.iloc[:2].to_dict("records"), horizon=3, params=None),
                        dict(history=data.iloc[:2].to_dict("records"), targets=["2000-03", "2000-04", "2000-05"])]
            result = subprocess.run([sys.executable, str(ROOT / "harness/forecasting.py"), "--serve", str(candidate)],
                                    input="".join(json.dumps(message) + "\n" for message in requests),
                                    text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shape", result.stderr)

    def test_worker_supports_serialized_candidate_defined_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "dataclass_method.py"
            candidate.write_text(textwrap.dedent("""\
                from __future__ import annotations
                from dataclasses import dataclass
                import pickle
                import numpy as np
                @dataclass
                class Model:
                    value: float
                candidates = [{}]
                def fit(train, horizon, params):
                    return pickle.loads(pickle.dumps(Model(5.0)))
                def predict(state, history, targets):
                    return [np.full((len(targets), history.shape[1]-1), state.value)]
                """))
            data = observations(periods=5)
            requests = [dict(train=data.iloc[:2].to_dict("records"), horizon=3, params=None),
                        dict(history=data.iloc[:2].to_dict("records"), targets=["2000-03", "2000-04", "2000-05"])]
            result = subprocess.run([sys.executable, str(ROOT / "harness/forecasting.py"), "--serve", str(candidate)],
                                    input="".join(json.dumps(message) + "\n" for message in requests),
                                    text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.splitlines()[1])["predictions"], [[[5.0, 5.0]] * 3])

    def test_example_reuses_fitted_seasonal_mean_with_current_persistence(self):
        data = observations(periods=25)
        data.loc[24, ["station_0", "station_1"]] = [30.0, 60.0]
        requests = [dict(train=data.iloc[:24].to_dict("records"), horizon=3, params=None),
                    dict(history=data.to_dict("records"), targets=["2002-02", "2002-03", "2002-04"])]
        result = subprocess.run([sys.executable, str(ROOT / "harness/forecasting.py"), "--serve",
                                 str(ROOT / "examples/seasonal_mean.py")],
                                input="".join(json.dumps(message) + "\n" for message in requests),
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        predictions = json.loads(result.stdout.splitlines()[1])["predictions"]
        self.assertEqual([prediction[0][0] for prediction in predictions], [30, 25, 20, 15, 10])
        np.testing.assert_array_equal(predictions[-1], [[10.0, 20.0]] * 3)

    def test_internal_validation_selects_batch_without_refitting_on_rolling_history(self):
        from harness import analysis

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "input").mkdir()
            data = observations(periods=48)
            data.to_csv(workspace / "input/train.csv", index=False, float_format="%.0f")
            (workspace / "input/task.json").write_text(json.dumps({
                "horizon": 3, "internal_validation_targets": ["2001-01", "2003-12"], "first_origin": "2000-03"}))
            candidate = workspace / "method.py"
            candidate.write_text(textwrap.dedent("""\
                import numpy as np
                candidates = [{'scale': 1}, {'scale': 2}]
                def fit(train, horizon, params):
                    assert train.month.iloc[-1] == '2000-12'
                    assert len(train) == 12
                    assert (train.iloc[:, 1:].dtypes == 'float64').all()
                    with open('fits.txt', 'a') as file:
                        file.write(str(len(params)) + '\\n')
                    return dict(train=train.copy(), params=params)
                def predict(state, history, targets):
                    assert len(state['train']) == 12
                    assert history.month.iloc[-1] < targets[0]
                    assert history.month.iloc[-1] <= '2003-09'
                    with open('origins.txt', 'a') as file:
                        file.write(history.month.iloc[-1] + '\\n')
                    return [np.repeat(history.iloc[-1:, 1:].to_numpy(dtype=float), len(targets), axis=0) * p['scale']
                            for p in state['params']]
                """))
            previous = Path.cwd()
            os.chdir(workspace)
            self.addCleanup(os.chdir, previous)
            result = analysis.calculate_internal_validation_score(candidate)
            self.assertEqual([(r["params"], r["score"]) for r in result["results"]],
                             [({"scale": 1}, 1.0), ({"scale": 2}, 0.0)])
            # Per-year scores use the same formula on that year's target records.
            self.assertEqual(result["results"][0]["by_year"], {"2001": 1.0, "2002": 1.0, "2003": 1.0})
            self.assertEqual(result["results"][1]["by_year"], {"2001": 0.0, "2002": 0.0, "2003": 0.0})
            self.assertEqual(result["best_params"], {"scale": 1})
            self.assertEqual(result["best_index"], 0)
            self.assertEqual(result["best_score"], 1.0)
            self.assertNotIn("records", result)
            self.assertEqual((workspace / "fits.txt").read_text(), "2\n")
            origins = (workspace / "origins.txt").read_text().splitlines()
            self.assertEqual((len(origins), origins[0], origins[-1]), (34, "2000-12", "2003-09"))
            selected = analysis.calculate_internal_validation_score(candidate, {"scale": 2})
            self.assertEqual([(r["params"], r["score"]) for r in selected["results"]], [({"scale": 2}, 0.0)])
            self.assertEqual(selected["best_params"], {"scale": 2})
            self.assertEqual((workspace / "fits.txt").read_text(), "2\n1\n")
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
