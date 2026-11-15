"""Pre-reservation closure never starts a clock, erases history, or permits retries."""
import tempfile
import unittest

from cloud_v2.ledger import PersistentLedger, DuplicateSubmission, LedgerError, request_identity


class UnstartedWindowCloseTests(unittest.TestCase):
    def test_close_before_reservation_is_persistent_and_cannot_close_reserved(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = PersistentLedger(root, "offline-close-test", clock=lambda: 1_800_000_000.0)
            original = ledger.snapshot()
            specs = [{"id": str(i), "sha256": format(i, "064x"), "qubits": 8} for i in range(12)]
            fingerprint = request_identity(specs, 128, "tianyan_sw", "preflight")
            ledger.authorize_window("closed-unused", "a" * 64, fingerprint, "offline fixture authorization")
            ledger.close_unstarted_window("closed-unused", "Live backend is not free")
            restored = PersistentLedger(root, "offline-close-test", clock=lambda: 1_800_000_010.0)
            now = restored.snapshot()
            for key, value in original.items():
                self.assertEqual(now[key], value)
            window = now["execution_windows"][0]
            self.assertEqual(window["outcome"], "preflight_blocked_before_reservation")
            self.assertIsNone(window["started_epoch"])
            self.assertIsNone(window["deadline_epoch"])
            self.assertFalse(window["backend_cancelled"])
            restored.close_unstarted_window("closed-unused", "Repeated close is idempotent")
            self.assertEqual(restored.snapshot(), now)
            with self.assertRaises(DuplicateSubmission):
                restored.effective_deadline("closed-unused")
            with self.assertRaises(DuplicateSubmission):
                restored.authorize_window("replacement", "a" * 64, fingerprint, "no repeat")

            specs[0]["sha256"] = "f" * 64
            fingerprint2 = request_identity(specs, 128, "tianyan_sw", "preflight")
            restored.authorize_window("reserved", "b" * 64, fingerprint2, "separate fixture authorization")
            restored.reserve(12, 128, "preflight", "context-dependent-fingerprint", {
                "circuits": specs, "backend": "tianyan_sw"}, window_id="reserved")
            before = restored.snapshot()
            with self.assertRaises(LedgerError):
                restored.close_unstarted_window("reserved", "Cannot erase an existing attempt")
            self.assertEqual(restored.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
