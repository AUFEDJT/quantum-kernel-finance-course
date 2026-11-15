"""Offline control-flow and quota checks; fake probabilities are not physics evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cloud_v2 import circuits as qc
from cloud_v2 import search
from cloud_v3 import study


def toy_window():
    rng = np.random.default_rng(867)
    window = {}
    metadata = {"audit": {"task": "classification", "window_id": "analytic-fixture"}}
    for split, n in (("train", 16), ("valid", 16), ("test", 50)):
        window["x_"+split] = rng.uniform(-1.2, 1.2, (n, 8))
        window["y_"+split] = np.arange(n) % 2
        per_date = 5 if split == "test" else 8
        month = {"train": "01", "valid": "02", "test": "03"}[split]
        metadata["meta_"+split] = [{"symbol": f"sh{i:06d}", "date": f"2024-{month}-{i//per_date+1:02d}"} for i in range(n)]
    window["x_full_train"] = rng.uniform(-1.2, 1.2, (48, 8))
    window["y_full_train"] = np.arange(48) % 2
    metadata["universe"] = {"selected_symbols": [r["symbol"] for r in metadata["meta_test"]],
                            "O_assignments": [dict(r) for r in metadata["meta_test"]]}
    window["metadata"] = metadata
    return window


class FakeProvider:
    def __init__(self, fail_at=None, constant_one=False):
        self.calls = []
        self.fail_at = fail_at
        self.constant_one = constant_one

    def execution_identity(self):
        return {"kind": "mock_control_flow_only", "is_mock": True}

    def probability(self, specs, shots, context, deadline):
        self.calls.append({"count": len(specs), "shots": shots, "context": dict(context),
                           "request_keys": [s["metadata"]["request_key"] for s in specs]})
        if len(self.calls) == self.fail_at:
            raise TimeoutError("injected unknown result")
        result = []
        for spec in specs:
            value = .1+.8*(int(hashlib.sha256(spec["qcis"].encode()).hexdigest()[:12], 16) % 1000)/1000
            result.append({"circuit_id": spec["circuit_id"], "shots": shots,
                           "p0": 1. if self.constant_one else value})
        return result


class FakeScorer:
    def predict(self, layouts):
        return np.asarray([int(qc.digest(v)[:10], 16)/16**10 for v in layouts])


def fake_trainer(layouts, labels, maximum_depth, seed):
    assert len(layouts) == len(labels) == 4 and maximum_depth == 3 and seed == 20261115
    return FakeScorer(), {"test_only": True, "state": "not a trained scientific predictor"}


class CompletedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)/"study"
        cls.window = toy_window()
        cls.provider = FakeProvider()
        cls.result = study.run_study(cls.window, cls.root, cls.provider, predictor_trainer=fake_trainer)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_success_quota_and_batch_size(self):
        calls = self.provider.calls
        self.assertEqual(sum(c["count"] for c in calls), 5288)
        self.assertEqual(len(calls), 270)
        self.assertTrue(all(1 <= c["count"] <= 20 and c["shots"] == 512 for c in calls))
        self.assertEqual(self.result["circuits"], {"qukernet": 3044, "random": 2244})
        self.assertEqual(self.result["request_shots"], 2707456)
        self.assertEqual(self.result["execution_kind"], "mock_control_flow_only")
        self.assertEqual(study.budget_upper()["common_per_arm"], 1444)

    def test_no_initial_or_original_validation_quantum_calls(self):
        operations = [c["context"]["operation"] for c in self.provider.calls]
        self.assertFalse(any("initial_valid" in o or "selected/valid" in o for o in operations))
        valid_calls = [c for c in self.provider.calls if c["context"]["operation"].startswith("endpoint/valid/")]
        self.assertEqual(sum(c["count"] for c in valid_calls), 512)

    def test_stages_alternate_and_o_control_is_last(self):
        status = json.loads((self.root/"study_status.json").read_text())
        events = [(e["phase"], e["arm"]) for e in status["events"]]
        self.assertEqual(events[:4], [("labels/0", "qukernet"), ("labels/0", "random"),
                                      ("labels/1", "qukernet"), ("labels/1", "random")])
        self.assertEqual(events[-3:], [("endpoint_O", "qukernet"), ("endpoint_O", "random"), ("initial_O", "qukernet")])
        self.assertEqual([v for v in events if v[0].startswith("optimization/")],
                         [("optimization/step0", "qukernet"), ("optimization/step0", "random"),
                          ("optimization/step1", "qukernet"), ("optimization/step1", "random")])

    def test_shared_frozen_balanced_batches_and_gradient_quota(self):
        frozen = json.loads((self.root/"frozen_batches.json").read_text())
        for step, batch in enumerate(frozen["indices"]):
            self.assertEqual(len(set(batch)), 4)
            self.assertEqual(sorted(self.window["y_train"][batch].tolist()), [0, 0, 1, 1])
            for arm in study.ARMS:
                trace = json.loads((self.root/arm/f"tuning/step{step}.json").read_text())
                self.assertEqual(trace["batch_indices"], batch)
                self.assertTrue(trace["both_overlap_ends"])
        gradient = [c for c in self.provider.calls if "/gradient/" in c["context"]["operation"]]
        self.assertEqual(sum(c["count"] for c in gradient), 4*48)

    def test_mask_initialization_saved_and_reloaded(self):
        for arm in study.ARMS:
            frozen = json.loads((self.root/arm/"selected_and_initialization.json").read_text())
            endpoint = json.loads((self.root/arm/"tuning/endpoint.json").read_text())
            self.assertEqual(len(frozen["mask"]), 2)
            self.assertEqual(frozen["mask"], endpoint["mask"])
            expected = qc.initial_parameters(self.window["x_train"], frozen["layout"], frozen["mask"])
            np.testing.assert_array_equal(expected, frozen["initial_parameters"])
            self.assertEqual(endpoint["initial_parameters"], frozen["initial_parameters"])

    def test_method_operation_ids_are_stable_and_disjoint(self):
        ids = [c["context"]["operation_id"] for c in self.provider.calls]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("compact_pilot_v3/analytic-fixture/") for i in ids))
        self.assertTrue(any("/qukernet/20261115/" in i for i in ids))
        self.assertTrue(any("/random/20261115/" in i for i in ids))

    def test_completed_reload_has_no_provider_calls(self):
        provider = FakeProvider()
        loaded = study.run_study(self.window, self.root, provider, predictor_trainer=fake_trainer)
        self.assertEqual(loaded, self.result)
        self.assertEqual(provider.calls, [])

    def test_baseline_solver_distinction_and_bootstrap_present(self):
        classic = json.loads((self.root/"classical/results.json").read_text())
        note = classic["selection"]["full_train_linear_solver"]
        self.assertEqual(note["estimator"], "LinearSVC")
        self.assertEqual(note["loss"], "squared_hinge")
        self.assertFalse(note["dual"])
        self.assertIsInstance(note["converged"], bool)
        self.assertEqual(set(self.result["paired_date_bootstrap"]), set(self.result["date_equal_accuracy_differences"]))
        for value in self.result["paired_date_bootstrap"].values():
            self.assertEqual(value["repetitions"], 10000)
            self.assertEqual(set(value["blocks"]), {"1", "2", "3"})
        self.assertTrue(all(v["available"] for v in self.result["comparison_availability"].values()))

    def test_O_label_flip_leaves_quantum_search_and_updates_unchanged(self):
        window = toy_window()
        window["y_test"] = 1-window["y_test"]
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            other = Path(tmp)/"flipped"
            study.run_study(window, other, provider, predictor_trainer=fake_trainer)
            self.assertEqual([c["request_keys"] for c in provider.calls],
                             [c["request_keys"] for c in self.provider.calls])
            for arm in study.ARMS:
                for relative in ("selected_and_initialization.json", "tuning/endpoint.json", "tuning/step0.json", "tuning/step1.json"):
                    first = json.loads((self.root/arm/relative).read_text())
                    second = json.loads((other/arm/relative).read_text())
                    self.assertEqual(first, second)


class RejectionTests(unittest.TestCase):
    def test_comparison_requires_fifty_matching_rows_at_both_ends(self):
        rows = [{"row_id": i, "date": "2024-01-%02d" % (i//5+1), "target": i%2,
                 "prediction": i%2} for i in range(50)]
        states = {"qukernet": {"endpoint_O": {"records": rows}},
                  "random": {"endpoint_O": {"records": rows[:49]}}}
        key = "qukernet_endpoint_minus_random_endpoint"
        self.assertFalse(study.comparison_availability(states)[key]["available"])
        states["random"]["endpoint_O"]["records"] = [dict(r) for r in rows]
        self.assertTrue(study.comparison_availability(states)[key]["available"])
        states["random"]["endpoint_O"]["records"][0]["target"] = 1
        self.assertEqual(study.comparison_availability(states)[key]["reason"], "row_date_target_mismatch")

    def test_protocol_id_namespaces_cloud_operations(self):
        window, provider = toy_window(), FakeProvider(fail_at=1)
        window["metadata"]["protocol_id"] = "qlib-missing-fixture-v1"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                study.run_study(window, Path(tmp)/"private", provider, predictor_trainer=fake_trainer)
            self.assertIn("/qlib-missing-fixture-v1/", provider.calls[0]["context"]["operation_id"])

    def test_unknown_attempt_is_not_automatically_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, provider, window = Path(tmp)/"private", FakeProvider(fail_at=2), toy_window()
            with self.assertRaises(TimeoutError):
                study.run_study(window, root, provider, predictor_trainer=fake_trainer)
            count = len(provider.calls)
            with self.assertRaisesRegex(RuntimeError, "Incomplete/unknown"):
                study.run_study(window, root, provider, predictor_trainer=fake_trainer)
            self.assertEqual(len(provider.calls), count)
            status = json.loads((root/"study_status.json").read_text())
            self.assertEqual(status["state"], "stopped")
            self.assertTrue(status["no_automatic_resume"])
            availability = json.loads((root/"comparisons_availability.json").read_text())
            self.assertTrue(all(not v["available"] for v in availability.values()))

    def test_constant_kernel_stops_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(constant_one=True)
            with self.assertRaises(search.ScientificInvalid):
                study.run_study(toy_window(), Path(tmp)/"private", provider, predictor_trainer=fake_trainer)
            self.assertEqual(sum(c["count"] for c in provider.calls), 120)
            self.assertTrue(all("/qukernet/" in c["context"]["operation_id"] for c in provider.calls))

    def test_invalid_dimensions_and_unsupported_seed_spend_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            window = toy_window()
            window["x_test"] = window["x_test"][:49]
            with self.assertRaises(ValueError):
                study.run_study(window, Path(tmp)/"private", provider, predictor_trainer=fake_trainer)
            with self.assertRaises(ValueError):
                study.run_study(toy_window(), Path(tmp)/"private", provider, seed=1, predictor_trainer=fake_trainer)
            self.assertEqual(provider.calls, [])

    def test_single_minority_record_cannot_form_batch(self):
        window = toy_window()
        window["y_train"] = np.asarray([0]*15+[1])
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            with self.assertRaises(search.ScientificInvalid):
                study.run_study(window, Path(tmp)/"private", provider, predictor_trainer=fake_trainer)
            self.assertEqual(provider.calls, [])

    def test_duplicate_stock_or_changed_fixed_date_rejected_before_cloud(self):
        for kind in ("duplicate_stock", "changed_date", "changed_assignment"):
            window = toy_window()
            if kind == "duplicate_stock":
                window["metadata"]["meta_test"][1]["symbol"] = window["metadata"]["meta_test"][0]["symbol"]
            elif kind == "changed_date":
                window["metadata"]["meta_test"][1]["date"] = "2024-03-20"
            else:
                window["metadata"]["universe"]["O_assignments"][0]["date"] = "2024-03-20"
            with tempfile.TemporaryDirectory() as tmp:
                provider = FakeProvider()
                with self.assertRaises(ValueError):
                    study.run_study(window, Path(tmp)/"private", provider, predictor_trainer=fake_trainer)
                self.assertEqual(provider.calls, [])

    def test_effective_dimension_is_recorded_without_inventing_eight(self):
        window = toy_window()
        window["metadata"]["preprocessing"] = {"scaler": {"zero_iqr_columns": [1]}}
        for split in ("train", "valid", "test", "full_train"):
            window["x_"+split][:, 1] = 0
        arrays, metadata, _ = study._admit(window)
        quality = study.feature_quality(arrays, metadata)
        self.assertEqual(quality["input_columns"], 8)
        self.assertEqual(quality["zero_iqr_columns"], [1])
        self.assertEqual(quality["splits"]["train"]["varying_columns"], 7)
        self.assertLessEqual(quality["splits"]["train"]["centered_matrix_rank"], 7)

    def test_O_labels_do_not_choose_rbf_gamma(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for flip in (False, True):
                window = toy_window()
                if flip:
                    window["y_test"] = 1-window["y_test"]
                arrays, metadata, _ = study._admit(window)
                store = search.Checkpoints(Path(tmp)/str(flip), {"fixture": True, "flip": flip})
                results.append(study.classical_baselines(arrays, metadata, store)["selection"])
            self.assertEqual(results[0], results[1])

    def test_block_bootstrap_requires_pairing_and_enough_dates(self):
        one = [{"row_id": i, "date": "2024-01-01", "target": i%2, "prediction": i%2} for i in range(4)]
        result = study.paired_date_bootstrap(one, one)
        self.assertEqual(result["status"], "insufficient_dates")
        self.assertEqual(result["blocks"], {})
        bad = [dict(r) for r in one]
        bad[0]["target"] = 1-bad[0]["target"]
        with self.assertRaises(ValueError):
            study.paired_date_bootstrap(one, bad)


if __name__ == "__main__":
    unittest.main()
