"""Offline main-window invariants; synthetic readings, no SDK/network/credentials."""
import copy
import hashlib
import tempfile
import unittest
from unittest import mock

from cloud_v2.cloud import TianyanCloudAdapter, UnknownSubmission, InvalidResult
from cloud_v2.ledger import (PersistentLedger, BudgetExceeded, DuplicateSubmission,
                            LedgerError, HARD_LIMITS, MAIN_POLICY)
from cloud_v2.preflight import PreflightBlocked
from test_authorized_window_integration import (VirtualClock, twelve_specs, fixture_evidence,
                                                DeadlineTransport)


class MainTransport(DeadlineTransport):
    def submit(self, specs, shots, backend, aid, deadline):
        result = super().submit(specs, shots, backend, aid, deadline)
        self.ids = ["offline-main-%d-%d" % (self.submissions, i) for i in range(len(specs))]
        return self.ids if len(result) == len(specs) else self.ids[:1]

    def item(self, i):
        result = super().item(i)
        if self.mode == "invalid_rows":
            result["resultStatus"][1][0] = False
        return result


class MainWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = VirtualClock()
        self.scope = "offline-main-protocol-account"
        self.ledger = PersistentLedger(self.temp.name, self.scope, clock=self.clock.wall)
        for n in (8, 12):
            e = self.ledger.reserve(n, 128, "preflight", "old-" + str(n), {})
            self.ledger.update(e["attempt_id"], "submitted", job_ids=["old-%d-%d" % (n, i) for i in range(n)])
            self.ledger.update(e["attempt_id"], "validated")
        self.old = copy.deepcopy(self.ledger.snapshot())
        self.clock.advance(HARD_LIMITS["seconds"] + 10)
        self.wid = "three-quarter-main-fixture"
        self.protocol = "c" * 64
        self.ledger.authorize_main_window(self.wid, self.protocol, "Offline authorized fixture")
        self.transport = MainTransport(self.ledger, self.clock)
        self.provider = self.provider_for()
        self.specs = twelve_specs()[:2]

    def tearDown(self):
        self.temp.cleanup()

    def provider_for(self, **kwargs):
        proof = fixture_evidence(self.ledger, self.clock)
        proof["capabilities"]["approved_main_shots_limit"] = 512
        return TianyanCloudAdapter(self.transport, self.ledger, proof,
            sleep=self.clock.sleep, monotonic=self.clock.mono, execution_window_id=self.wid,
            execution_protocol_sha256=kwargs.pop("protocol", self.protocol), **kwargs)

    def run_batch(self, op="2024Q1/qukernet/seed/labels/0/batch0", **kwargs):
        return self.provider.probability(self.specs, 512, context={"operation_id": op}, **kwargs)

    def test_fixed_policy_and_old_counters_preserved(self):
        self.assertEqual((MAIN_POLICY["circuits"], MAIN_POLICY["shots"], MAIN_POLICY["max_attempts"]),
                         (93760, 48005120, 4804))
        self.assertIsNone(self.ledger.window(self.wid)["started_epoch"])
        self.run_batch()
        after = self.ledger.snapshot()
        for k in ("started_epoch", "deadline_epoch", "limits"):
            self.assertEqual(after[k], self.old[k])
        self.assertEqual(after["entries"][:2], self.old["entries"])
        self.assertEqual(sum(e["circuits"] for e in after["entries"]), 22)
        self.assertEqual(sum(e["requested_shots"] for e in after["entries"]), 2560 + 1024)
        self.assertEqual(self.ledger.window(self.wid)["state"], "running")

    def test_multiple_batches_and_legal_different_method_same_qcis(self):
        self.run_batch()
        start = self.ledger.window(self.wid)["started_epoch"]
        self.clock.advance(21)
        self.run_batch(op="2024Q1/random/seed/labels/0/batch0")
        self.assertEqual(self.transport.submissions, 2)
        self.assertEqual(self.ledger.window(self.wid)["started_epoch"], start)
        self.assertEqual(self.ledger.window(self.wid)["deadline_epoch"], start + 72 * 3600)
        self.ledger.close_main_window(self.wid, "completed", "All registered phases terminal")
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")

    def test_missing_and_wrong_protocol_binding_rejected(self):
        with self.assertRaises(PreflightBlocked):
            self.provider_for(protocol=None)
        with self.assertRaises(PreflightBlocked):
            self.provider_for(protocol="d" * 64)
        self.assertEqual(self.transport.submissions, 0)

    def test_initial_backend_check_precedes_clock(self):
        before = self.clock.wall()
        self.transport.costs["backend"] = 7
        self.run_batch()
        self.assertEqual(self.ledger.window(self.wid)["started_epoch"], before + 7)

    def test_duplicate_operation_after_restart_is_not_resent(self):
        self.run_batch()
        self.ledger = PersistentLedger(self.temp.name, self.scope, clock=self.clock.wall)
        self.transport.ledger = self.ledger
        self.provider = self.provider_for()
        with self.assertRaises(DuplicateSubmission):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")

    def test_changed_contents_under_same_operation_cannot_resubmit(self):
        self.run_batch()
        self.specs[0]["qcis"] = self.specs[0]["qcis"].replace("0.101000", "0.301000")
        self.specs[0]["sha256"] = hashlib.sha256(self.specs[0]["qcis"].encode()).hexdigest()
        with self.assertRaisesRegex(DuplicateSubmission, "content changed"):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 1)

    def test_operation_cannot_be_replayed_by_new_window_id(self):
        self.run_batch()
        self.ledger.close_main_window(self.wid, "completed", "Fixture completed")
        self.wid = "new-id-does-not-reset-operation"
        self.ledger.authorize_main_window(self.wid, self.protocol, "Fixture fresh authorization")
        self.provider = self.provider_for()
        with self.assertRaises(DuplicateSubmission):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 1)

    def test_empty_results_stop_entire_window_and_no_further_queries(self):
        self.transport.mode = "empty"
        self.transport.costs["submit"] = 250
        before = self.clock.mono()
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.clock.mono() - before, 300)
        queries = self.transport.queries
        self.provider = self.provider_for()
        with self.assertRaises(LedgerError):
            self.run_batch(op="different-op")
        self.assertEqual(self.transport.queries, queries)
        self.assertEqual(self.transport.submissions, 1)
        self.assertFalse(self.ledger.window(self.wid)["backend_cancelled"])

    def test_uncertain_submit_consumes_budget_and_closes(self):
        self.transport.mode = "submit_timeout"
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.ledger.snapshot()["entries"][-1]["requested_shots"], 1024)
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")
        self.assertEqual(self.transport.queries, 0)

    def test_invalid_structure_and_optional_callback_close(self):
        self.transport.mode = "invalid_rows"
        with self.assertRaises(InvalidResult):
            self.run_batch()
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")

    def test_optional_callback_does_not_replace_structure_check(self):
        called = []
        self.provider = self.provider_for(result_validator=lambda rows: called.append(len(rows)))
        self.run_batch()
        self.assertEqual(called, [2])

    def test_wrong_shots_or_missing_operation_closes_without_reservation(self):
        with self.assertRaises(BudgetExceeded):
            self.provider.probability(self.specs, 128, context={"operation_id": "wrong-shots"})
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])
        self.assertEqual(self.transport.submissions, 0)

    def test_missing_operation_id_is_fail_closed(self):
        with self.assertRaises(PreflightBlocked):
            self.provider.probability(self.specs, 512)
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")
        self.assertEqual(self.transport.submissions, 0)

    def test_ledger_cannot_mark_result_validated_after_persisted_300_seconds(self):
        meta = {"backend": "tianyan_sw", "protocol_sha256": self.protocol,
                "circuits": self.specs, "context": {"operation_id": "expired-result"}}
        entry = self.ledger.reserve(2, 512, "main", "expired-result", meta, window_id=self.wid)
        self.ledger.update(entry["attempt_id"], "submitted", job_ids=["expired-a", "expired-b"])
        self.clock.advance(300)
        with self.assertRaises(BudgetExceeded):
            self.ledger.update(entry["attempt_id"], "validated")
        self.assertEqual(self.ledger.snapshot()["entries"][-1]["state"], "submitted")

    def test_pre_reservation_failure_closes_without_starting_clock(self):
        self.transport.backend_fee = 2
        with self.assertRaises(PreflightBlocked):
            self.run_batch()
        w = self.ledger.window(self.wid)
        self.assertEqual(w["state"], "closed")
        self.assertIsNone(w["started_epoch"])
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])

    def test_main_72h_deadline_is_not_extended_by_next_batch(self):
        self.run_batch()
        end = self.ledger.window(self.wid)["deadline_epoch"]
        self.clock.advance(end - self.clock.wall())
        with self.assertRaises(PreflightBlocked):
            self.run_batch(op="next")
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.ledger.window(self.wid)["deadline_epoch"], end)

    def test_remaining_global_seconds_bound_last_batch(self):
        self.run_batch()
        end = self.ledger.window(self.wid)["deadline_epoch"]
        self.clock.advance(end - self.clock.wall() - 9)
        self.transport.mode = "empty"
        with self.assertRaises(UnknownSubmission):
            self.run_batch(op="last")
        self.assertAlmostEqual(self.clock.wall(), end)
        self.assertEqual(self.ledger.window(self.wid)["state"], "closed")

    def test_small_fixture_window_caps_are_enforced_without_real_submissions(self):
        # Patch only the offline policy constant to exercise cap boundaries with
        # two fixtures rather than generating 4,804 filesystem-heavy attempts.
        self.ledger.close_main_window(self.wid, "not_started", "Use a separate small fixture ledger")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(MAIN_POLICY, {"circuits": 2, "shots": 1024, "max_attempts": 1}):
                ledger = PersistentLedger(directory, "reduced-fixture", clock=self.clock.wall)
                ledger.authorize_main_window("small", self.protocol, "Offline fixture")
                meta = {"backend": "tianyan_sw", "protocol_sha256": self.protocol,
                        "circuits": self.specs, "context": {"operation_id": "one"}}
                entry = ledger.reserve(2, 512, "main", "one", meta, window_id="small")
                ledger.update(entry["attempt_id"], "submitted", job_ids=["a", "b"])
                ledger.update(entry["attempt_id"], "validated")
                meta = copy.deepcopy(meta); meta["context"]["operation_id"] = "two"
                with self.assertRaises(BudgetExceeded):
                    ledger.reserve(2, 512, "main", "two", meta, window_id="small")


if __name__ == "__main__":
    unittest.main()
