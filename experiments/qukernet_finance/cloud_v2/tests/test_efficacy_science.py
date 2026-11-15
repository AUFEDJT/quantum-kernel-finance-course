"""Offline control-flow/science checks. Mock values are never cloud evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_v2 import circuits as c, search as s, experiments as e
from test_search import ExplicitMockProvider, mock_factory


class EfficacyScienceTests(unittest.TestCase):
    def test_frozen_profile_and_exact_budget(self):
        profile = e.profile("efficacy8_n16_512_v1")
        self.assertEqual((profile["labels"], profile["predictor_train_labels"], profile["shots"]), (16, 12, 512))
        self.assertIsNone(profile["kernel_seconds"])
        self.assertEqual(profile["configuration_deadline_scope"].split(";")[0],
                         "persisted wall deadline for the three optimization steps only")
        for size, expected in ((220, 15744), (209, 15392)):
            budget = e.dry_run("efficacy8_n16_512_v1", size)["budget_upper_bound"]
            self.assertEqual(budget["probability_circuits_per_arm_seed"], expected)
            self.assertEqual(budget["nominal_shots_per_arm_seed"], expected*512)
        legacy = e.dry_run("pilot8", 220)["budget_upper_bound"]
        self.assertEqual(legacy["probability_circuits_per_arm_seed"], 11200)

    def test_exact_masks_reproducible_and_legacy_unchanged(self):
        for depth in range(1, 6):
            layout = c.sample_unique(1, 8, 40, [depth], 49)[0]
            for config in range(2):
                mask = c.make_mask(layout, 20261110, config, 2, exact_parameters=2)
                self.assertEqual(len(mask), 2)
                self.assertEqual(len(set(mask)), 2)
                self.assertEqual(mask, c.make_mask(layout, 20261110, config, 2, exact_parameters=2))
                allowed = c.valid_positions(layout)
                rng = np.random.default_rng(np.random.SeedSequence([20261110, 8, 40, depth, config]))
                n = int(rng.integers(1, 3))
                legacy = sorted(int(v) for v in rng.choice(allowed, size=n, replace=False))
                self.assertEqual(c.make_mask(layout, 20261110, config, 2), legacy)

    def test_matrix_can_exceed_300_while_every_batch_is_bounded(self):
        clock = [1000.]
        limits = []
        class SlowMock(ExplicitMockProvider):
            def probability(self, specs, shots, context=None, deadline=None):
                limits.append(deadline-clock[0])
                if deadline-clock[0] < 200:
                    raise TimeoutError("test batch exceeds deadline")
                clock[0] += 200
                return super().probability(specs, shots, context=context, deadline=deadline)
        with tempfile.TemporaryDirectory() as directory, patch.object(s.time, "monotonic", side_effect=lambda: clock[0]):
            kernel = s.CloudKernel(SlowMock(), s.Checkpoints(Path(directory)/"out", {"test": "batch_budget"}),
                                   shots=16, kernel_seconds=None)
            layout = c.sample_unique(1, 2, 2, [1], 8)[0]
            result = kernel.matrix(np.ones((12, 2)), None, layout)
            self.assertEqual(result.shape, (12, 12))
            self.assertGreater(clock[0]-1000, 300)
            self.assertEqual(limits, [300.]*4)

    def test_timer_deadline_persists_and_cannot_be_renewed(self):
        clock = [1000.]
        with tempfile.TemporaryDirectory() as directory, patch.object(s.time, "time", side_effect=lambda: clock[0]):
            store = s.Checkpoints(Path(directory)/"out", {"test": "timer"})
            timer = s.OptimizationTimer(store, "config0", 1800.)
            original = timer.record["deadline_epoch"]
            clock[0] = 1900.
            restored = s.OptimizationTimer(store, "config0", 1800.)
            self.assertEqual(restored.record["deadline_epoch"], original)
            clock[0] = 2801.
            with self.assertRaisesRegex(TimeoutError, "will not reset"):
                restored.deadline()
            with self.assertRaisesRegex(ValueError, "budget changed"):
                s.OptimizationTimer(store, "config0", 3600.)

    def test_optimization_timer_excludes_initial_final_and_validation(self):
        clock = [1000.]
        class TimedKernel:
            def __init__(self, store):
                self.store = store
            def matrix(self, left, right, layout, mask=(), parameters=(), key="", deadline=None):
                clock[0] += 700 if "/step" not in key else 100
                if deadline is not None and clock[0] > deadline:
                    raise TimeoutError("test kernel exhausted optimization budget")
                if right is None:
                    return np.eye(len(left))*.8+np.ones((len(left), len(left)))*.2
                return np.full((len(left), len(right)), .2)
            def derivatives(self, x, layout, mask, parameters, key, deadline=None):
                clock[0] += 100
                return np.zeros((len(mask), len(x), len(x)))
        with tempfile.TemporaryDirectory() as directory, patch.object(s.time, "time", side_effect=lambda: clock[0]), \
                patch.object(s.time, "monotonic", side_effect=lambda: clock[0]):
            store = s.Checkpoints(Path(directory)/"out", {"test": "scope"})
            kernel = TimedKernel(store)
            layout = c.sample_unique(1, 2, 2, [1], 8)[0]
            x = np.arange(32).reshape(16, 2)/100
            y = np.array([0, 1]*8)
            tuned = s.finetune(kernel, x, y, layout, [0, 1], 20261110, 0,
                              "classification", 3, 8, .2, "config0", 1800.,
                              initial_validation=(x, y, [str(i) for i in range(16)]))
            self.assertEqual(tuned["optimization_timer"]["elapsed_seconds"], 600)
            self.assertGreater(tuned["seconds_this_invocation"], 1800)
            self.assertEqual(tuned["optimization_timer"]["started_epoch"], 2400)
            self.assertIn("initial_evaluation", tuned)

    def test_complete_paired_chain_selects_post_and_reuses_no_cloud_after_resume(self):
        rng = np.random.default_rng(15)
        window = {"x_"+split: rng.normal(size=(4, 4)) for split in ("train", "valid", "test")}
        window.update({"y_"+split: np.array([0, 1, 0, 1]) for split in ("train", "valid", "test")})
        window["metadata"] = {"audit": {"task": "classification", "quarter": "unit_test_quarter"},
                              **{"meta_"+split: [{"date": str(i//2)} for i in range(4)]
                                 for split in ("train", "valid", "test")}}
        profile = {"stage": "mock_efficacy_only", "qubits": 2, "features": 4, "depths": [1, 2],
                   "n_train": 4, "n_valid": 4, "labels": 8, "predictor_train_labels": 6,
                   "candidate_pool": 16, "top_k": 2, "configurations": 2, "steps": 2,
                   "batch_size": 4, "max_parameters": 2, "exact_parameters": 2,
                   "learning_rate": .2, "shots": 16, "paired_initialization_control": True,
                   "kernel_seconds": None}
        evaluate = s.evaluate_validation
        calls, operation_ids = [], []
        # Initial V would choose the LAST configuration; post V chooses FIRST.
        forced_losses = {1: -.1, 2: -.9, 3: -.2, 4: -.4, 6: -.3, 7: -.3, 8: -.99, 9: -.2}
        def scored(*args, **kwargs):
            row = evaluate(*args, **kwargs)
            index = len(calls); calls.append(index)
            if index in forced_losses:
                row["selection_loss"] = forced_losses[index]
            return row
        class ContextMock(ExplicitMockProvider):
            def probability(self, specs, shots, context=None, deadline=None):
                operation_ids.append(context["operation_id"])
                return super().probability(specs, shots, context=context, deadline=deadline)
        provider = ContextMock()
        with tempfile.TemporaryDirectory() as directory, patch.object(s, "evaluate_validation", side_effect=scored), \
                patch.object(s, "source_hash", return_value="offline_fixture_frozen_source"):
            out = Path(directory)/"out"
            result = s.run_search(window, profile, provider, out, 500, predictor_factory=mock_factory)
            self.assertEqual(result["workflow_status"], "complete")
            self.assertEqual(result["valid_configurations"], 4)
            final = json.loads((out/"final_candidates.json").read_text())
            winner = min(final, key=lambda r: (r["selection_loss"], r["layout_hash"], r["configuration"]))
            initial_best = min(final, key=lambda r: r["initial_evaluation"]["selection_loss"])
            self.assertNotEqual((winner["layout_index"], winner["configuration"]),
                                (initial_best["layout_index"], initial_best["configuration"]))
            paired = json.loads((out/"paired_initial_predictions.json").read_text())
            self.assertEqual(paired["selected_layout_hash"], winner["layout_hash"])
            self.assertEqual(paired["selected_configuration"], winner["configuration"])
            self.assertEqual(paired["initial_parameters"], winner["initial_parameters"])
            self.assertTrue(all(len(r["mask"]) == 2 for r in final))
            self.assertEqual(len(operation_ids), len(set(operation_ids)))
            self.assertTrue(all(v.startswith("mock_efficacy_only/unit_test_quarter/qukernet/500/") for v in operation_ids))
            before = provider.calls
            restored = s.run_search(window, profile, provider, out, 500, predictor_factory=mock_factory)
            self.assertEqual(restored, result)
            self.assertEqual(provider.calls, before)


if __name__ == "__main__":
    unittest.main()
