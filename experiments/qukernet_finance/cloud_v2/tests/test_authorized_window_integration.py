"""Independent single-batch authorization tests; synthetic fixtures only.

No credential access, SDK login, network request or cloud submission occurs.
All clock advances are virtual. The mock does not model quantum probabilities.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cloud_v2.cloud import TianyanCloudAdapter, UnknownSubmission, InvalidResult
from cloud_v2.ledger import (
    PersistentLedger, LedgerError, HARD_LIMITS, request_identity, utc,
)
from cloud_v2.preflight import PreflightBlocked, INTERFACE


class VirtualClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def wall(self):
        return 1_800_000_000.0 + self.t

    def mono(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


def twelve_specs():
    result = []
    for i in range(12):
        qcis = "\n".join(
            ["RY Q%d %.6f" % (q, .1 + i / 100 + q / 1000) for q in range(1, 9)]
            + ["CZ Q%d Q%d" % (q, q + 1) for q in range(1, 8)]
            + ["M Q%d" % q for q in range(1, 9)]
        )
        result.append({"id": "mock-eight-qubit-%02d" % i, "qcis": qcis,
                       "sha256": hashlib.sha256(qcis.encode()).hexdigest(),
                       "qubits": 8, "measured_qubits": list(range(1, 9))})
    return result


def fixture_evidence(ledger, clock):
    return {
        "account_scope_sha256": ledger.account_hash, "backend": "tianyan_sw",
        "canonical_ledger_path_sha256": hashlib.sha256(str(ledger.path.resolve()).encode()).hexdigest(),
        "interface": INTERFACE, "authorized": True, "free": True,
        "zero_real_device_quota": True, "no_untracked_inflight_jobs": True,
        "source_kind": "test_fixture", "source_locator": "offline-window-test",
        "source_sha256": "a" * 64, "observed_utc": utc(clock.wall() - 1),
        "valid_until_utc": utc(clock.wall() + 400_000),
        "capabilities": {
            "backend": "tianyan_sw", "running": True, "max_qubits": 36,
            "server_max_shots": None, "approved_probe_shots_limit": 128,
            "approved_main_shots_limit": 128, "client_max_circuit_characters": 20_000,
            "server_max_circuit_characters": None,
            "gate_validation_basis": "installed_cqlib_1.3.11_offline_parser",
            "supported_gates": ["H", "I", "CX", "CZ", "RX", "RY", "RZ", "M"],
            "observed_utc": utc(clock.wall() - 1),
            "valid_until_utc": utc(clock.wall() + 400_000),
            "source_locator": "offline-window-capabilities", "source_sha256": "b" * 64,
        },
    }


class DeadlineTransport:
    is_mock = True
    sdk_version = "1.3.11-offline-window-mock"

    def __init__(self, ledger, clock):
        self.ledger, self.clock = ledger, clock
        self.mode = "success"
        self.costs = {"backend": 0., "submit": 0., "query": 0., "circuit": 0.}
        self.calls = []
        self.submissions = 0
        self.queries = 0
        self.circuits_read = 0
        self.backend_fee = 1
        self.ids = []
        self.specs = []
        self.reservation_was_durable = False

    def spend(self, stage, deadline):
        self.calls.append((stage, self.clock.mono(), deadline))
        remaining = deadline - self.clock.mono()
        if remaining <= 0:
            raise TimeoutError("Offline transport deadline")
        cost = self.costs[stage]
        if cost >= remaining:
            self.clock.advance(remaining)
            raise TimeoutError("Offline transport exhausted remaining time")
        self.clock.advance(cost)

    def backend_snapshot(self, deadline):
        self.spend("backend", deadline)
        return {"code": 0, "data": [{"code": "tianyan_sw", "isToll": self.backend_fee, "status": 0}]}

    def submit(self, specs, shots, backend, aid, deadline):
        self.submissions += 1
        self.specs, self.shots = specs, shots
        self.ids = ["offline-new-job-%02d" % i for i in range(len(specs))]
        entry = self.ledger.snapshot()["entries"][-1]
        self.reservation_was_durable = entry["attempt_id"] == aid and entry["state"] == "submission_pending"
        self.spend("submit", deadline)
        if self.mode == "submit_timeout":
            raise TimeoutError("Offline uncertain submission")
        if self.mode == "partial_ids":
            return self.ids[:1]
        return self.ids

    def item(self, i):
        # Synthetic all-zero rows allow structure validation without pretending
        # to reproduce the quantum distribution of the synthetic circuits.
        return {"experimentTaskId": self.ids[i], "status": 0,
                "resultStatus": [list(range(1, 9))] + [[0] * 8 for _ in range(self.shots)],
                "probability": {"0" * 8: 1.0}}

    def query(self, ids, deadline):
        self.queries += 1
        self.spend("query", deadline)
        items = [self.item(i) for i in range(len(ids))]
        if self.mode == "empty":
            items = []
        elif self.mode == "partial_forever":
            items = items[:1]
        return {"code": 0, "data": {"experimentResultModelList": items}}

    def returned_circuit(self, jid, deadline):
        self.circuits_read += 1
        self.spend("circuit", deadline)
        return [{"experimentTaskId": jid, "qcis": self.specs[self.ids.index(jid)]["qcis"]}]


class AuthorizedWindowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = VirtualClock()
        self.account = "offline-integration-account"
        self.ledger = PersistentLedger(self.temp.name, self.account, clock=self.clock.wall)
        old = self.ledger.reserve(8, 128, "preflight", "historical-eight-fixture", {})
        self.ledger.update(old["attempt_id"], "submitted", job_ids=["old-%d" % i for i in range(8)])
        self.ledger.update(old["attempt_id"], "validated", evidence_file="offline-historical-fixture")
        self.old = self.ledger.snapshot()
        self.clock.advance(HARD_LIMITS["seconds"] + 10)
        self.specs = twelve_specs()
        self.window_id = "offline-new-72-hour-single-batch"
        self.authorize()
        self.transport = DeadlineTransport(self.ledger, self.clock)
        self.validator_called = 0
        self.provider = self.new_provider()

    def tearDown(self):
        self.temp.cleanup()

    def authorize(self):
        return self.ledger.authorize_window(
            self.window_id, "c" * 64,
            request_identity(self.specs, 128, "tianyan_sw", "preflight"),
            "Offline fixture authorization only; no real cloud execution",
        )

    def validate(self, out):
        self.validator_called += 1
        if len(out) != 12 or any(x["zero_count"] != 128 for x in out):
            raise InvalidResult("Synthetic acceptance fixture mismatch")

    def new_provider(self):
        return TianyanCloudAdapter(
            self.transport, self.ledger, fixture_evidence(self.ledger, self.clock),
            observation_seconds=300, sleep=self.clock.sleep, monotonic=self.clock.mono,
            execution_window_id=self.window_id, result_validator=self.validate,
        )

    def run_batch(self, **kwargs):
        return self.provider.probability(self.specs, 128, phase="preflight", **kwargs)

    def entry(self):
        return self.ledger.snapshot()["entries"][-1]

    def window(self):
        return next(w for w in self.ledger.snapshot()["execution_windows"] if w["window_id"] == self.window_id)

    def assert_old_preserved_and_new_reserved(self):
        value = self.ledger.snapshot()
        self.assertEqual(value["started_epoch"], self.old["started_epoch"])
        self.assertEqual(value["deadline_epoch"], self.old["deadline_epoch"])
        self.assertEqual(value["limits"], self.old["limits"])
        self.assertEqual(value["entries"][:-1], self.old["entries"])
        self.assertEqual(sum(e["circuits"] for e in value["entries"]), 20)
        self.assertEqual(sum(e["requested_shots"] for e in value["entries"]), 2560)
        self.assertEqual(self.entry()["observation_deadline_epoch"] - self.entry()["reserved_epoch"], 300)

    def test_success_preserves_old_window_counters_and_closes_new_scope(self):
        out = self.run_batch()
        self.assertEqual(len(out), 12)
        self.assertTrue(self.transport.reservation_was_durable)
        self.assertEqual(self.validator_called, 1)
        self.assertEqual(self.entry()["state"], "validated")
        self.assertEqual(self.window()["state"], "closed")
        self.assert_old_preserved_and_new_reserved()
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.run_batch(context={"repetition": "changed-context-cannot-reopen"})
        self.assertEqual(self.transport.submissions, 1)

    def test_expired_legacy_window_remains_unusable_without_new_id(self):
        self.provider.execution_window_id = None
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])

    def test_new_window_begins_at_reservation_after_readonly_backend_check(self):
        before = self.clock.wall()
        self.transport.costs["backend"] = 37
        self.run_batch()
        self.assertEqual(self.entry()["reserved_epoch"], before + 37)
        self.assertEqual(self.window()["started_epoch"], before + 37)
        self.assertEqual(self.window()["deadline_epoch"], before + 37 + 72 * 3600)
        submit = next(c for c in self.transport.calls if c[0] == "submit")
        self.assertAlmostEqual(submit[2] - submit[1], 300)

    def test_empty_results_stop_at_total_300_seconds_including_submission(self):
        self.transport.mode = "empty"
        self.transport.costs["submit"] = 250
        before = self.clock.mono()
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.clock.mono() - before, 300)
        self.assertEqual(self.entry()["state"], "observation_unknown")
        self.assertEqual(self.window()["state"], "closed")
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.validator_called, 0)
        self.assert_old_preserved_and_new_reserved()

    def test_partial_results_are_saved_and_do_not_extend_time(self):
        self.transport.mode = "partial_forever"
        before = self.clock.mono()
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.clock.mono() - before, 300)
        folder = self.ledger.root / self.entry()["attempt_id"]
        queries = sorted(folder.glob("query_*.json"))
        self.assertTrue(queries)
        self.assertEqual(len(json.loads(queries[-1].read_text())["data"]["experimentResultModelList"]), 1)
        self.assertEqual(len(self.entry()["job_ids"]), 12)
        self.assertEqual(self.validator_called, 0)

    def test_submission_timeout_retains_full_reservation_and_closes(self):
        self.transport.mode = "submit_timeout"
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.entry()["state"], "submission_unknown")
        self.assertEqual(self.transport.queries, 0)
        self.assertEqual(self.window()["state"], "closed")
        self.assert_old_preserved_and_new_reserved()

    def test_partial_submission_ids_are_retained_without_retry(self):
        self.transport.mode = "partial_ids"
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.entry()["partial_returned_ids"], ["offline-new-job-00"])
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.transport.queries, 0)
        self.assertEqual(self.window()["state"], "closed")

    def test_query_and_all_returned_circuit_reads_share_submission_deadline(self):
        self.transport.costs.update(submit=200, query=95, circuit=1)
        before = self.clock.mono()
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.clock.mono() - before, 300)
        self.assertEqual(self.entry()["state"], "observation_unknown")
        self.assertLess(self.transport.circuits_read, 12)
        self.assertEqual(self.validator_called, 0)
        deadlines = [c[2] for c in self.transport.calls if c[0] != "backend"]
        self.assertEqual(len(set(deadlines)), 1)

    def test_statistical_acceptance_must_finish_within_same_deadline(self):
        def slow_validator(out):
            self.validator_called += 1
            self.clock.advance(301)
        self.provider.result_validator = slow_validator
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        self.assertEqual(self.validator_called, 1)
        self.assertNotEqual(self.entry()["state"], "validated")
        self.assertEqual(self.window()["state"], "closed")

    def test_statistical_rejection_is_not_validated_or_retried(self):
        def reject(out):
            raise InvalidResult("Offline frozen acceptance interval mismatch")
        self.provider.result_validator = reject
        with self.assertRaises(InvalidResult):
            self.run_batch()
        self.assertEqual(self.entry()["state"], "received_invalid")
        self.assertEqual(self.window()["state"], "closed")
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.run_batch(context={"new": "context"})
        self.assertEqual(self.transport.submissions, 1)

    def test_restart_cannot_extend_deadline_poll_or_repeat(self):
        self.transport.mode = "empty"
        with self.assertRaises(UnknownSubmission):
            self.run_batch()
        prior = copy.deepcopy(self.entry())
        queries = self.transport.queries
        self.clock.advance(900)
        self.ledger = PersistentLedger(self.temp.name, self.account, clock=self.clock.wall)
        self.transport.ledger = self.ledger
        self.provider = self.new_provider()
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.run_batch(context={"restart": True})
        self.assertEqual(self.entry(), prior)
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.transport.queries, queries)

    def test_changed_circuit_content_is_outside_single_batch_authorization(self):
        self.specs[0]["qcis"] = self.specs[0]["qcis"].replace("0.101000", "0.301000")
        self.specs[0]["sha256"] = hashlib.sha256(self.specs[0]["qcis"].encode()).hexdigest()
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])

    def test_changed_shots_cannot_consume_authorization(self):
        with self.assertRaises((LedgerError, PreflightBlocked)):
            self.provider.probability(self.specs, 64, phase="preflight")
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])

    def test_paid_backend_stops_before_reservation(self):
        self.transport.backend_fee = 2
        with self.assertRaises(PreflightBlocked):
            self.run_batch()
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])
        self.assertEqual(self.window()["state"], "authorized")

    def test_new_window_requires_explicit_statistical_acceptance_callback(self):
        with self.assertRaises(PreflightBlocked):
            TianyanCloudAdapter(
                self.transport, self.ledger, fixture_evidence(self.ledger, self.clock),
                observation_seconds=300, sleep=self.clock.sleep, monotonic=self.clock.mono,
                execution_window_id=self.window_id,
            )
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], self.old["entries"])


if __name__ == "__main__":
    unittest.main()
