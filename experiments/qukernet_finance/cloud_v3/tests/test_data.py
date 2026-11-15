"""Protocol tests use constructed data only, never reported as experiments."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cloud_v3.data import (FEATURES, calendar_split, select_universe, full_training_mrmr,
                           prepare_window, save_window, load_window, build_features)
from cloud_v2.data80 import mrmr_select


def fixture():
    calendar = pd.date_range("2019-01-02", periods=500, freq="B")
    symbols = [f"sh60{i:04d}" for i in range(30)] + [f"sz00{i:04d}" for i in range(30)]
    metadata = [{"symbol": s, "exchange": s[:2], "observed_valid_dates": calendar.strftime("%Y-%m-%d").tolist()}
                for s in symbols]
    rows = [{"symbol": s, "date": d, "open": 10., "high": 12., "low": 9., "close": 11., "volume": 1000.}
            for s in symbols for d in calendar]
    return pd.DataFrame(rows), calendar, metadata


def constructed_features(raw, calendar):
    rng = np.random.default_rng(550)
    tables = []
    for k, symbol in enumerate(sorted(raw.symbol.unique())):
        n = len(calendar)
        values = rng.normal(size=(n, 80))
        table = pd.DataFrame(values, columns=FEATURES)
        table.insert(0, "j", np.arange(n))
        table.insert(0, "date", calendar)
        table.insert(0, "symbol", symbol)
        table["feature_valid"] = True
        table["history_complete"] = True
        table["bar_valid"] = True
        table["y_raw"] = np.where(np.arange(n) % 2, -.01, .01)
        table["target_valid"] = np.arange(n) + 6 < n
        table["label_start_j"] = np.arange(n) + 1
        table["label_end_j"] = np.arange(n) + 6
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.calendar, cls.universe = fixture()
        chosen, _, _, _ = select_universe(cls.universe, cls.calendar, latest_date=cls.calendar[-1])
        cls.frame = constructed_features(cls.raw[cls.raw.symbol.isin(chosen)], cls.calendar)
        # Ensure the constructed selected kernel sample contains both classes;
        # feature-only date/security sampling itself remains independent of y.
        from cloud_v2.data80 import balanced_sample
        _, split = calendar_split(cls.calendar, cls.calendar[-1])
        selected, _ = balanced_sample(cls.frame[cls.frame.j.isin(split["train_days"])], np.asarray(split["train_days"]), 16, 20261115)
        cls.frame.loc[selected.index, "y_raw"] = np.array([-.01, .01] * 8)
        with patch("cloud_v3.data.build_features", return_value=cls.frame):
            cls.window = prepare_window(cls.raw, cls.calendar, cls.universe, latest_date=cls.calendar[-1])

    def prepare(self, frame=None, universe=None):
        with patch("cloud_v3.data.build_features", return_value=self.frame if frame is None else frame):
            return prepare_window(self.raw, self.calendar, self.universe if universe is None else universe, latest_date=self.calendar[-1])

    def test_boundaries_and_midpoint_dates(self):
        dates, split = calendar_split(self.calendar, self.calendar[-1])
        self.assertEqual(len(split["outer_days"]), 63)
        self.assertEqual(len(split["valid_days"]), 57)
        self.assertEqual(split["validation_start"], dates[split["v_start_j"]].strftime("%Y-%m-%d"))
        self.assertEqual(split["evaluation_start"], dates[split["o_start_j"]].strftime("%Y-%m-%d"))
        self.assertLess(max(split["train_days"]) + 6, split["v_start_j"])
        self.assertLess(max(split["valid_days"]) + 6, split["o_start_j"])
        self.assertEqual(len(set(split["outer_selected_days"])), 10)
        self.assertTrue(np.all(np.diff(split["outer_selected_days"]) >= 6))

    def test_unavailable_validation_records_have_structured_phase(self):
        changed = self.frame.copy()
        _, split = calendar_split(self.calendar, self.calendar[-1])
        changed.loc[changed.j.isin(split["valid_days"]), "feature_valid"] = False
        with self.assertRaises(ValueError) as caught:
            self.prepare(changed)
        detail = caught.exception.preparation_audit
        self.assertEqual(detail["code"], "fixed_sample_unavailable")
        self.assertEqual(detail["phase"], "valid")
        self.assertEqual(detail["feature_valid_pool_rows"], 0)
        self.assertTrue(all(r["dates_with_two_feature_valid_securities"] == 0 for r in detail["date_bins"]))

    def test_exchange_allocation_fifty_and_future_independence(self):
        first = select_universe(self.universe, self.calendar, latest_date=self.calendar[-1])
        changed = copy.deepcopy(self.universe)
        v_start = first[2][first[3]["v_start_j"]]
        for row in changed:
            row["observed_valid_dates"] = [d for d in row["observed_valid_dates"] if pd.Timestamp(d) < v_start]
        second = select_universe(changed, self.calendar, latest_date=self.calendar[-1])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[1]["allocation"], {"bj": 0, "sh": 25, "sz": 25})
        self.assertEqual(len(set(r["symbol"] for r in first[1]["O_assignments"])), 50)

    def test_exact_data_shapes_and_full_T_mapping(self):
        w = self.window
        self.assertEqual(w["x_train"].shape, (16, 8))
        self.assertEqual(w["x_valid"].shape, (16, 8))
        self.assertEqual(w["x_test"].shape, (50, 8))
        self.assertGreater(len(w["x_full_train"]), 16)
        self.assertEqual(w["preprocessing"]["scaler"]["fit_rows"], len(w["x_full_train"]))
        self.assertEqual(w["preprocessing"]["mrmr"]["fit_rows"], len(w["x_full_train"]))
        self.assertEqual(len({r["symbol"] for r in w["meta_test"]}), 50)
        self.assertEqual(len({r["date"] for r in w["meta_test"]}), 10)

    def test_O_missing_does_not_reassign_stock_or_date(self):
        changed = self.frame.copy()
        assignment = self.window["universe"]["O_assignments"][0]
        changed.loc[(changed.symbol == assignment["symbol"]) & (changed.j == assignment["j"]), "feature_valid"] = False
        with self.assertRaisesRegex(ValueError, "coverage failed"):
            self.prepare(changed)

    def test_minor_class_failure_is_not_resampled(self):
        changed = self.frame.copy()
        t = self.window["audit"]["split"]["train_days"]
        changed.loc[changed.j.isin(t), "y_raw"] = -.01
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            self.prepare(changed)

    def test_future_values_cannot_fit_preprocessing(self):
        changed = self.frame.copy()
        v0 = self.window["audit"]["split"]["v_start_j"]
        changed.loc[changed.j >= v0, FEATURES] += 1000.
        changed.loc[changed.j >= v0, "y_raw"] *= -1
        other = self.prepare(changed)
        self.assertEqual(other["preprocessing_hash"], self.window["preprocessing_hash"])
        np.testing.assert_array_equal(other["x_train"], self.window["x_train"])

    def test_mrmr_matches_existing_objective(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(150, 80))
        y = (x[:, 2] > 0).astype(int)
        actual, _ = full_training_mrmr(x, y)
        expected, _ = mrmr_select(x, y, count=8, task="classification")
        self.assertEqual(actual, expected)

    def test_private_snapshot_roundtrip_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = save_window(self.window, Path(temp) / "private")
            restored = load_window(paths["npz"])
            np.testing.assert_array_equal(restored["x_full_train"], self.window["x_full_train"])
            self.assertEqual(restored["preprocessing_hash"], self.window["preprocessing_hash"])
            with self.assertRaises(FileExistsError):
                save_window(self.window, Path(temp) / "private")
            with Path(paths["npz"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_window(paths["npz"])

    def test_actual_feature_engine_requires121_consecutive_bars(self):
        rng = np.random.default_rng(8)
        dates = self.calendar[:140]
        close = 10 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
        open_ = close * np.exp(rng.normal(0, .005, len(dates)))
        raw = pd.DataFrame({"symbol": "sh600000", "date": dates, "open": open_,
                            "high": np.maximum(open_, close) * 1.01,
                            "low": np.minimum(open_, close) * .99, "close": close,
                            "volume": rng.integers(100, 10000, len(dates))})
        result = build_features(raw, dates)
        self.assertFalse(result.history_complete.iloc[119])
        self.assertTrue(result.history_complete.iloc[120])
        missing = build_features(raw.drop(index=50), dates)
        self.assertFalse(missing.history_complete.iloc[120:].any())


if __name__ == "__main__":
    unittest.main()
