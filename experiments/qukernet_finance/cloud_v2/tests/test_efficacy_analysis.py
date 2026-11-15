"""Offline tests for classical selection and date-block comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cloud_v2 import efficacy_analysis as ea


def fixture():
    rng = np.random.default_rng(1701)
    out = {"audit": {"quarter": "2022Q1", "task": "classification"}}
    for split, n, year in (("train", 16, 2020), ("valid", 16, 2021), ("test", 6, 2022)):
        x = rng.normal(0, .3, (n, 40))
        y = (np.arange(n) % 2).astype(int)
        x[:, 0] += 2*y-1
        out["x_"+split], out["y_"+split] = x, y
        out["meta_"+split] = [{"date": f"{year}-01-{1+i//2:02d}", "symbol": f"private{i}"} for i in range(n)]
    return out


def records(correct=True, invert=False):
    return [{"date": f"2022-01-{1+i//2:02d}", "row_id": i,
             "target": i % 2, "prediction": i % 2 if correct else 1-i % 2}
            for i in range(12)]


class MetricTests(unittest.TestCase):
    def test_date_weighting_differs_from_row_weighting(self):
        metric = ea.classification_metrics([0, 1, 1, 1], [0, 0, 0, 0], ["a", "b", "b", "b"], [0]*4)
        self.assertEqual(metric["date_equal_accuracy"], .5)
        self.assertEqual(metric["balanced_accuracy"], .5)
        self.assertEqual(metric["confusion_counts"], [[1, 0], [3, 0]])
        self.assertEqual(metric["roc_auc"], .5)
    def test_missing_class_is_explicit(self):
        metric = ea.classification_metrics([0, 0], [0, 1], ["a", "b"], [0., 1.])
        self.assertIsNone(metric["roc_auc"])
        self.assertIsNone(metric["recall_class_1"])
        self.assertIsNone(metric["balanced_accuracy"])
    def test_malformed_values_rejected(self):
        with self.assertRaises(ValueError):
            ea.classification_metrics([0, 1], [0, np.nan], ["a", "b"])


class BaselineTests(unittest.TestCase):
    def test_only_V_selected_RBF_is_computed_on_O(self):
        calls = []
        original = ea._gram
        def logged(left, right, kind, gamma=None):
            if len(left) == 6 and kind == "rbf":
                calls.append(gamma)
            return original(left, right, kind, gamma)
        with tempfile.TemporaryDirectory() as temp, patch.object(ea, "_gram", side_effect=logged):
            summary = ea.baseline(fixture(), Path(temp)/"run")
            self.assertEqual(calls, [summary["selection"]["rbf"]["gamma"]])
            text = json.dumps(summary)
            self.assertNotIn("private", text)
            self.assertNotIn("x_train", text)
            self.assertNotIn(temp, text)
            self.assertEqual(summary["models"]["majority"]["predicted_class"], 0)
            saved = json.loads((Path(temp)/"run/classical_baselines_predictions.json").read_text())
            self.assertEqual(len(saved["models"]["rbf"]["records"]), 6)
            with self.assertRaises(FileExistsError):
                ea.baseline(fixture(), Path(temp)/"run")
    def test_O_targets_cannot_change_gamma(self):
        first, second = fixture(), fixture()
        second["y_test"] = 1-second["y_test"]
        with tempfile.TemporaryDirectory() as temp:
            a = ea.baseline(first, Path(temp)/"a")
            b = ea.baseline(second, Path(temp)/"b")
        self.assertEqual(a["selection"], b["selection"])
    def test_tie_selects_smallest_gamma(self):
        original = ea.evaluate_validation
        def tied(*args, **kwargs):
            result = original(*args, **kwargs)
            result.update(selection_loss=-.5, validation_accuracy=.5)
            return result
        with tempfile.TemporaryDirectory() as temp, patch.object(ea, "evaluate_validation", side_effect=tied):
            summary = ea.baseline(fixture(), Path(temp)/"run")
        self.assertEqual(summary["selection"]["rbf"]["gamma"], .1)
    def test_2025_rejected(self):
        window = fixture()
        window["meta_test"][0]["date"] = "2025-01-01"
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            ea.baseline(window, Path(temp)/"run")


class BootstrapTests(unittest.TestCase):
    def test_identical_arms_zero_and_reproducible(self):
        rows = {q: records() for q in ea.QUARTERS}
        first = ea.paired_date_block_bootstrap(rows, rows, repeats=100)
        self.assertEqual(first, ea.paired_date_block_bootstrap(rows, rows, repeats=100))
        self.assertTrue(first["coverage"]["complete"])
        for item in first["blocks"].values():
            self.assertEqual(item["delta_date_equal_accuracy"]["ci95"], [0., 0.])
            self.assertEqual(item["delta_balanced_accuracy"]["ci95"], [0., 0.])
    def test_known_difference_and_missing_quarters_not_zero_imputed(self):
        a, b = {"2022Q1": records()}, {"2022Q1": records(False)}
        result = ea.paired_date_block_bootstrap(a, b, repeats=100)
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(len(result["coverage"]["excluded"]), 2)
        self.assertEqual(result["blocks"]["2"]["delta_date_equal_accuracy"]["estimate"], 1.)
    def test_incomplete_pairs_are_excluded_as_whole_quarter(self):
        a, b = {"2022Q1": records()}, {"2022Q1": records()[:-1]}
        result = ea.paired_date_block_bootstrap(a, b, repeats=100)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["blocks"], {})
    def test_changed_target_is_not_a_pair(self):
        a, b = {"2022Q1": records()}, {"2022Q1": records()}
        b["2022Q1"][0]["target"] = 1
        result = ea.paired_date_block_bootstrap(a, b, repeats=100)
        self.assertEqual(result["status"], "unavailable")
    def test_quarters_are_equal_weight_despite_different_rows(self):
        a = {"2022Q1": records(), "2023Q1": records()[:6]}
        b = {"2022Q1": records(False), "2023Q1": records()[:6]}
        result = ea.paired_date_block_bootstrap(a, b, repeats=100)
        self.assertEqual(result["blocks"]["2"]["delta_date_equal_accuracy"]["estimate"], .5)
    def test_single_class_BA_undefined_not_zero(self):
        rows = records()
        for row in rows:
            row["target"] = 0
        result = ea.paired_date_block_bootstrap({"2022Q1": rows}, {"2022Q1": rows}, repeats=100)
        self.assertIsNone(result["blocks"]["2"]["delta_balanced_accuracy"]["estimate"])
        self.assertEqual(result["blocks"]["2"]["delta_balanced_accuracy"]["undefined_replicates"], 100)


if __name__ == "__main__":
    unittest.main()
