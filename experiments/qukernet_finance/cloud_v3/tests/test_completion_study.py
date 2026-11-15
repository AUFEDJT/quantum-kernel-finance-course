"""Constructed continuation tests: fake probabilities, no cloud/financial data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from cloud_v2 import circuits as qc
from cloud_v2.ledger import request_identity
from cloud_v3 import study, completion_study as completion
from cloud_v3.tests.test_study import toy_window, fake_trainer


class RecordingProvider:
    def __init__(self, stop_after_o_batches=None, fail_at=None):
        self.stop_after_o_batches = stop_after_o_batches
        self.fail_at = fail_at
        self.calls, self.entries, self.o_calls = [], [], 0

    def execution_identity(self):
        return {"kind": "mock_control_flow_only", "is_mock": True}

    def probability(self, specs, shots, context, deadline):
        operation = context["operation"]
        is_o = operation.startswith(("endpoint/test/", "initial/test/"))
        if is_o and self.stop_after_o_batches is not None and self.o_calls >= self.stop_after_o_batches:
            raise RuntimeError("analytic controlled boundary stop before reservation")
        if self.fail_at is not None and len(self.calls) + 1 == self.fail_at:
            raise TimeoutError("analytic provider failure; no replay permitted")
        index = len(self.calls)
        jobs = [f"analytic-job-{index}-{j}" for j in range(len(specs))]
        self.calls.append({"operation": operation, "operation_id": context["operation_id"],
                           "count": len(specs), "shots": shots})
        self.o_calls += int(is_o)
        self.entries.append({"attempt_id": f"analytic-attempt-{index}", "state": "validated",
            "execution_window_id": "parent-window", "operation_id": context["operation_id"],
            "circuits": len(specs), "shots_per_circuit": shots, "requested_shots": len(specs)*shots,
            "request_identity": request_identity(specs, shots, "tianyan_sw", "main"), "job_ids": jobs})
        result = []
        for spec, job in zip(specs, jobs):
            value = .1+.8*(int(hashlib.sha256(spec["qcis"].encode()).hexdigest()[:12], 16) % 1000)/1000
            result.append({"circuit_id": spec["circuit_id"], "id": spec["id"], "shots": shots,
                           "job_id": job, "p0": value})
        return result


def snapshot(provider):
    preflight = {"attempt_id": "analytic-preflight", "operation_id": "analytic-preflight-operation",
                 "execution_window_id": "parent-window", "state": "validated", "circuits": 4,
                 "shots_per_circuit": 512, "requested_shots": 2048}
    return {"schema_version": 1, "entries": [preflight]+copy.deepcopy(provider.entries),
            "execution_windows": [{"window_id": "parent-window", "state": "closed", "outcome": "operator_stop",
                                   "backend_cancelled": False}]}


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.window = toy_window()
        cls.parents = {}
        for cached in (0, 5, 40):
            path = cls.root / f"parent-{cached}"
            provider = RecordingProvider(stop_after_o_batches=cached)
            with cls._assert_raises_static(RuntimeError):
                study.run_study(cls.window, path, provider, predictor_trainer=fake_trainer)
            cls.parents[cached] = (path, snapshot(provider))

    @staticmethod
    def _assert_raises_static(kind):
        return unittest.TestCase().assertRaises(kind)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self, cached=0, path=None, ledger=None, window=None):
        parent, original = self.parents[cached]
        parent = parent if path is None else path
        return completion.build_completion_plan(parent, window or self.window,
            copy.deepcopy(original) if ledger is None else ledger, parent_window_id="parent-window",
            parent_file_hashes=completion.freeze_parent_file_hashes(parent))

    def test_plan_uses_original_operations_and_exact_remaining_quota(self):
        for cached in (0, 5, 40):
            plan = self.plan(cached)
            self.assertEqual(len(plan.allowed_operations), 120-cached)
            self.assertEqual(plan.parent_recovery["reused_O_circuits"], cached*20)
            self.assertEqual(plan.parent_recovery["new_O_circuits"], (120-cached)*20)
            self.assertEqual(len(plan.parent_recovery["unreserved_intents"]), 1)
            for op, value in plan.allowed_operations.items():
                self.assertTrue("/endpoint/test/batch" in op or "/qukernet/20261115/initial/test/batch" in op)
                self.assertEqual(value["circuits"], 20)
                self.assertEqual(value["shots_per_circuit"], 512)

    def test_runs_only_remaining_o_batches_and_parent_is_byte_identical(self):
        parent, ledger = self.parents[5]
        before = completion.freeze_parent_file_hashes(parent)
        plan = self.plan(5)
        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"continuation"/"study"
            report = completion.run_completion(plan, output, provider,
                ledger_snapshot=ledger, execution_identity=provider.execution_identity())
            self.assertEqual(len(provider.calls), 115)
            self.assertEqual(sum(c["count"] for c in provider.calls), 2300)
            self.assertEqual(report["continuation"]["reused_O_circuits"], 100)
            self.assertEqual(report["new_probability_circuits"], 2300)
            self.assertEqual(report["circuits"], {"qukernet":3044,"random":2244})
            self.assertTrue(all(v["available"] for v in report["comparison_availability"].values()))
            for c in provider.calls:
                self.assertTrue(c["operation"].startswith(("endpoint/test/", "initial/test/")))
            self.assertEqual(provider.calls[0]["operation"], "endpoint/test/batch5")
            self.assertIn("/random/", provider.calls[35]["operation_id"])
            self.assertEqual(provider.calls[-1]["operation"], "initial/test/batch39")
            for relative in plan.parent_recovery["copied_parent_files"]:
                self.assertEqual(completion.file_sha256(output/relative), completion.file_sha256(parent/relative))
            count = len(provider.calls)
            with self.assertRaisesRegex(completion.CompletionBlocked, "already exists"):
                completion.run_completion(plan, output, provider, ledger_snapshot=ledger,
                                          execution_identity=provider.execution_identity())
            self.assertEqual(len(provider.calls), count)
        self.assertEqual(completion.freeze_parent_file_hashes(parent), before)

    def test_complete_parent_phase_is_recomputed_from_cache_without_calls(self):
        plan = self.plan(40)
        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"new-study"
            report = completion.run_completion(plan, output, provider,
                ledger_snapshot=self.parents[40][1], execution_identity=provider.execution_identity())
            self.assertEqual(len(provider.calls), 80)
            self.assertIn("/random/", provider.calls[0]["operation_id"])
            saved = json.loads((self.parents[40][0]/"qukernet/endpoint_predictions.json").read_text())
            reused = json.loads((output/"qukernet/endpoint_predictions.json").read_text())
            self.assertEqual(saved, reused)
            self.assertEqual(report["continuation"]["reused_O_circuits"], 800)

    def test_unresolved_account_attempt_blocks_even_unrelated_window(self):
        ledger = copy.deepcopy(self.parents[0][1])
        ledger["entries"].append({"state": "observation_unknown", "execution_window_id": "other"})
        with self.assertRaisesRegex(completion.CompletionBlocked, "unresolved"):
            self.plan(ledger=ledger)

    def test_reserved_unknown_intent_never_reissued(self):
        plan = self.plan(0)
        ledger = copy.deepcopy(self.parents[0][1])
        op = next(iter(plan.allowed_operations))
        ledger["entries"].append({"state":"submission_unknown", "operation_id":op})
        with self.assertRaisesRegex(completion.CompletionBlocked, "unresolved"):
            self.plan(ledger=ledger)

    def test_validated_operation_without_complete_cache_blocks(self):
        plan = self.plan(0); ledger = copy.deepcopy(self.parents[0][1])
        op, allow = next(iter(plan.allowed_operations.items()))
        ledger["entries"].append({**allow, "operation_id":op, "state":"validated", "execution_window_id":"parent-window",
                                  "requested_shots":10240,"attempt_id":"invented","job_ids":[str(i) for i in range(20)]})
        with self.assertRaisesRegex(completion.CompletionBlocked, "complete reusable checkpoint"):
            self.plan(ledger=ledger)

    def test_parent_must_be_operator_stop_and_input_unchanged(self):
        ledger = copy.deepcopy(self.parents[0][1]); ledger["execution_windows"][0]["outcome"] = "client_stopped"
        with self.assertRaisesRegex(completion.CompletionBlocked, "controlled"):
            self.plan(ledger=ledger)
        window = copy.deepcopy(self.window); window["x_test"][0,0] += .1
        with self.assertRaisesRegex(completion.CompletionBlocked, "identity changed"):
            self.plan(window=window)

    def test_pinned_source_and_file_changes_rejected(self):
        parent, ledger = self.parents[0]
        with self.assertRaisesRegex(completion.CompletionBlocked, "sources differ"):
            completion.build_completion_plan(parent, self.window, ledger, parent_window_id="parent-window",
                parent_file_hashes=completion.freeze_parent_file_hashes(parent), expected_source_sha256="0"*64)
        with tempfile.TemporaryDirectory() as tmp:
            copyroot = Path(tmp)/"copy"; shutil.copytree(parent, copyroot)
            hashes = completion.freeze_parent_file_hashes(copyroot)
            target = copyroot/"qukernet/endpoint_model_bundle.json"
            target.write_text(target.read_text()+" ")
            with self.assertRaisesRegex(completion.CompletionBlocked, "changed after freezing"):
                completion.build_completion_plan(copyroot, self.window, ledger, parent_window_id="parent-window", parent_file_hashes=hashes)

    def test_original_optimization_timer_cannot_be_reset_or_extended(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/"copy"; shutil.copytree(self.parents[0][0], parent)
            timer = parent/"qukernet/timers"/(qc.digest("tuning")+".json")
            value = json.loads(timer.read_text()); value["seconds"] = 3600
            timer.write_text(json.dumps(value))
            with self.assertRaisesRegex(completion.CompletionBlocked, "1800"):
                self.plan(path=parent)

    def test_cached_values_and_jobs_are_independently_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/"copy"; shutil.copytree(self.parents[5][0], parent)
            record = parent/"qukernet/checkpoints"/(qc.digest("endpoint/test/batch0")+".json")
            value = json.loads(record.read_text()); value["result"]["p0"][0] += .01
            value["result_sha256"] = qc.digest(value["result"])
            record.write_text(json.dumps(value))
            with self.assertRaisesRegex(completion.CompletionBlocked, "probability vector"):
                self.plan(5, path=parent)
        ledger = copy.deepcopy(self.parents[5][1])
        for entry in ledger["entries"]:
            if "/endpoint/test/batch0" in entry.get("operation_id", ""):
                entry["job_ids"][0] = "different-analytic-job"
        with self.assertRaisesRegex(completion.CompletionBlocked, "task IDs"):
            self.plan(5, ledger=ledger)

    def test_plan_mutation_after_freezing_blocks_before_any_provider_call(self):
        plan = self.plan(0)
        plan.phases[0]["batches"][0]["specs"][0]["qcis"] += "\n"
        provider = RecordingProvider()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"new-study"
            with self.assertRaisesRegex(completion.CompletionBlocked, "plan or frozen"):
                completion.run_completion(plan, output, provider, ledger_snapshot=self.parents[0][1],
                                          execution_identity=provider.execution_identity())
            self.assertFalse(output.exists())
        self.assertEqual(provider.calls, [])

    def test_partial_child_output_has_no_fabricated_predictions_and_cannot_restart(self):
        provider = RecordingProvider(fail_at=3); plan = self.plan(0)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"new-study"
            with self.assertRaises(TimeoutError):
                completion.run_completion(plan, output, provider, ledger_snapshot=self.parents[0][1],
                                          execution_identity=provider.execution_identity())
            self.assertFalse((output/"qukernet/endpoint_predictions.json").exists())
            value=json.loads((output/"comparisons_availability.json").read_text())
            self.assertTrue(all(not v["available"] for v in value.values()))
            with self.assertRaisesRegex(completion.CompletionBlocked, "already exists"):
                completion.run_completion(plan, output, provider, ledger_snapshot=self.parents[0][1],
                                          execution_identity=provider.execution_identity())
            self.assertEqual(len(provider.calls), 2)


if __name__ == "__main__":
    unittest.main()
