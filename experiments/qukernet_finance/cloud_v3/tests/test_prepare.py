"""Admission and failure-control tests; constructed inputs never run as real."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from cloud_v3.prepare import (evidence_gate, latest_date_gate, prepare_inputs, stable_read,
                              read_calendar_bytes, freeze_checkpoint, diagnose_checkpoint,
                              selected_coverage_diagnostics)
from cloud_v3.data import select_universe


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, evidence):
        calendar = self.root / "calendar.csv"
        calendar.write_text("date\n2025-01-02\n", encoding="utf-8")
        provenance = self.root / "evidence.json"
        provenance.write_text(json.dumps(evidence), encoding="utf-8")
        return {"source_root": self.root / "not_available", "calendar_csv": calendar,
                "calendar_date_column": "date", "calendar_evidence_json": provenance,
                "output": self.root / "private", "data_role": "real_required"}

    def test_fixture_cannot_be_admitted_as_real(self):
        args = self.arguments({"data_role": "synthetic_fixture", "status": "verified_market_calendar", "evidence": "constructed"})
        with patch("cloud_v3.prepare.scan_source") as scanner:
            result = prepare_inputs(**args)
        scanner.assert_not_called()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stage"], "calendar_evidence")
        self.assertEqual(result["cloud_submissions"], 0)

    def test_unverified_calendar_blocks_before_source(self):
        args = self.arguments({"data_role": "real_market_data", "status": "unverified", "source_kind": "index_proxy", "evidence": "none"})
        with patch("cloud_v3.prepare.scan_source") as scanner:
            result = prepare_inputs(**args)
        scanner.assert_not_called()
        self.assertEqual(result["stage"], "calendar_evidence")

    def test_proxy_is_not_silently_upgraded(self):
        evidence = {"data_role": "real_market_data", "status": "explicitly_accepted_diagnostic_proxy",
                    "source_kind": "index_observation_proxy", "evidence": "date coverage review"}
        with self.assertRaises(ValueError):
            evidence_gate(evidence, "real_required")
        evidence["acceptance_note"] = "Accepted only for this price diagnostic; not official exchange coverage"
        evidence_gate(evidence, "real_required")
        self.assertEqual(evidence["status"], "explicitly_accepted_diagnostic_proxy")

    def test_stale_or_future_D_is_rejected(self):
        for latest in ("1909-01-01", "2024-12-31", "2028-01-01"):
            with self.assertRaises(ValueError):
                latest_date_gate(latest, today="2026-11-15")
        self.assertEqual(latest_date_gate("2025-01-02", today="2026-11-15"), "2025-01-02")

    def test_path_guard_precedes_input_reads_and_writes(self):
        args = self.arguments({"data_role": "synthetic_fixture"})
        args["output"] = args["source_root"] / "cache"
        with patch("cloud_v3.prepare.stable_read") as reader:
            result = prepare_inputs(**args)
        reader.assert_not_called()
        self.assertEqual(result["stage"], "output_path_guard")
        self.assertFalse(args["source_root"].exists())

    def test_provider_error_message_not_recorded_or_retried(self):
        evidence = {"data_role": "real_market_data", "status": "verified_market_calendar",
                    "source_kind": "exchange_calendar", "evidence": "admission-test evidence only"}
        args = self.arguments(evidence)
        with patch("cloud_v3.prepare.scan_source", side_effect=OSError("private-sensitive-detail")) as scanner:
            result = prepare_inputs(**args)
        self.assertEqual(scanner.call_count, 1)
        self.assertEqual(result["error_type"], "OSError")
        self.assertNotIn("private-sensitive-detail", json.dumps(result))
        saved = list(args["output"].glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertNotIn("private-sensitive-detail", saved[0].read_text())

    def test_calendar_identity_changes_with_content(self):
        path = self.root / "calendar.csv"
        path.write_text("date\n2025-01-02\n")
        _, before = stable_read(path)
        path.write_text("date\n2025-01-03\n")
        _, after = stable_read(path)
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_calendar_provider_intro_and_chinese_header(self):
        payload = "供应商描述\n交易日期,开盘价\n2025-01-02,10\n2025-01-03,11\n".encode("gb18030")
        dates, skipped = read_calendar_bytes(payload, "交易日期")
        self.assertEqual(skipped, 1)
        self.assertEqual(dates.strftime("%Y-%m-%d").tolist(), ["2025-01-02", "2025-01-03"])

    def checkpoint_fixture(self):
        evidence = {"data_role": "real_market_data", "status": "verified_market_calendar",
                    "source_kind": "exchange_calendar", "evidence": "admission-test evidence only"}
        args = self.arguments(evidence)
        dates = pd.bdate_range("2024-01-02", periods=500)
        args["calendar_csv"].write_text("date\n" + "\n".join(dates.strftime("%Y-%m-%d")) + "\n")
        records = [{"symbol": f"sh60{i:04d}", "exchange": "sh",
                    "observed_valid_dates": dates.strftime("%Y-%m-%d").tolist(),
                    "identity": {"filename": f"sh60{i:04d}.csv", "bytes": 1, "mtime_ns": 1, "sha256": "a"*64}}
                   for i in range(60)]
        manifest = {"latest_date": dates[-1].strftime("%Y-%m-%d"), "securities": records,
                    "calendar_provenance": evidence}
        symbols, universe, _, _ = select_universe(manifest, dates, latest_date=dates[-1])
        panel = pd.DataFrame([{"symbol": s, "date": d, "open": 10., "high": 12., "low": 9.,
                               "close": 11., "volume": 1000.} for s in symbols for d in dates])
        used = [r["identity"] for r in records if r["symbol"] in symbols]
        checkpoint = freeze_checkpoint(manifest, universe,
                                       calendar_identity=stable_read(args["calendar_csv"])[1],
                                       evidence_identity=stable_read(args["calendar_evidence_json"])[1],
                                       calendar_date_column="date", calendar_evidence=evidence)
        return args, dates, manifest, universe, panel, used, checkpoint

    def test_failure_keeps_frozen_sources_and_all_fifty_gap_evidence(self):
        args, dates, manifest, universe, panel, used, _ = self.checkpoint_fixture()
        with patch("cloud_v3.prepare.scan_source", return_value=manifest), \
             patch("cloud_v3.prepare.read_selected_panel", return_value=(panel, used)), \
             patch("cloud_v3.prepare.prepare_window", side_effect=ValueError("private-sensitive-value")):
            result = prepare_inputs(**args)
        self.assertEqual(result["stage"], "feature_and_sample_preparation")
        self.assertEqual(set(result["artifacts"]), {"checkpoint", "selected_coverage"})
        checkpoint = json.loads((args["output"] / result["artifacts"]["checkpoint"]).read_text())
        self.assertEqual(checkpoint["universe"]["O_assignments"], universe["O_assignments"])
        self.assertNotIn("observed_valid_dates", json.dumps(checkpoint))
        self.assertNotIn("private-sensitive-value", json.dumps(result))
        diagnostics = json.loads((args["output"] / result["artifacts"]["selected_coverage"]).read_text())
        self.assertEqual(len(diagnostics["records"]), 50)

    def test_selected_gap_reports_date_without_reassigning(self):
        _, dates, manifest, universe, panel, _, _ = self.checkpoint_fixture()
        assignment = universe["O_assignments"][0]
        absent = dates[assignment["j"]-5]
        panel = panel[~(panel.symbol.eq(assignment["symbol"]) & panel.date.eq(absent))]
        report = selected_coverage_diagnostics(panel, dates, universe, manifest["latest_date"])
        row = next(r for r in report["records"] if r["symbol"] == assignment["symbol"])
        self.assertEqual(row["evaluation_date"], assignment["date"])
        self.assertEqual(row["evaluation_missing_history_dates"], [absent.strftime("%Y-%m-%d")])
        self.assertEqual(report["evaluation_history_failures"], 1)

    def test_checkpoint_diagnosis_is_one_selected_read_and_no_scan(self):
        args, _, _, _, panel, used, checkpoint = self.checkpoint_fixture()
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        with patch("cloud_v3.prepare.scan_source") as scanner, \
             patch("cloud_v3.prepare.read_selected_panel", return_value=(panel, used)) as reader:
            result = diagnose_checkpoint(checkpoint_path=path, source_root=args["source_root"],
                                         calendar_csv=args["calendar_csv"],
                                         calendar_evidence_json=args["calendar_evidence_json"], output=args["output"])
        scanner.assert_not_called()
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(len(reader.call_args.args[1]), 50)
        self.assertEqual(result["status"], "diagnosed")

    def test_checkpoint_calendar_change_blocks_before_any_source_read(self):
        args, _, _, _, _, _, checkpoint = self.checkpoint_fixture()
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        args["calendar_csv"].write_text(args["calendar_csv"].read_text() + "2026-01-02\n")
        with patch("cloud_v3.prepare.read_selected_panel") as reader:
            with self.assertRaisesRegex(ValueError, "version"):
                diagnose_checkpoint(checkpoint_path=path, source_root=args["source_root"],
                                    calendar_csv=args["calendar_csv"],
                                    calendar_evidence_json=args["calendar_evidence_json"], output=args["output"])
        reader.assert_not_called()

    def test_known_source_identifier_is_saved_without_provider_message(self):
        args = self.arguments({"data_role": "real_market_data", "status": "verified_market_calendar",
                               "source_kind": "exchange_calendar", "evidence": "admission-test evidence only"})
        error = ValueError("private-sensitive-detail")
        error.source_audit = {"code": "source_dates_outside_calendar", "filename": "sh600000.csv", "dates": ["2025-01-03"]}
        with patch("cloud_v3.prepare.scan_source", side_effect=error):
            result = prepare_inputs(**args)
        self.assertEqual(result["source_diagnostic"]["filename"], "sh600000.csv")
        self.assertEqual(result["source_diagnostic"]["dates"], ["2025-01-03"])
        self.assertNotIn("private-sensitive-detail", json.dumps(result))

    def test_explicit_cutoff_is_forwarded_and_invalid_value_stops_before_scan(self):
        evidence = {"data_role": "real_market_data", "status": "verified_market_calendar",
                    "source_kind": "exchange_calendar", "evidence": "admission-test evidence only"}
        args = self.arguments(evidence)
        with patch("cloud_v3.prepare.scan_source", side_effect=ValueError("bounded stop")) as scanner:
            result = prepare_inputs(**args, cutoff_date="2026-03-25")
        self.assertEqual(scanner.call_args.kwargs["cutoff_date"], pd.Timestamp("2026-03-25"))
        self.assertEqual(result["stage"], "source_inventory")
        with patch("cloud_v3.prepare.scan_source") as scanner:
            result = prepare_inputs(**args, cutoff_date="2026-03-25T12:00:00")
        scanner.assert_not_called()
        self.assertEqual(result["stage"], "requested_cutoff_gate")

    def test_checkpoint_retains_independent_interval_and_full_source_hashes(self):
        args, _, manifest, universe, _, _, _ = self.checkpoint_fixture()
        manifest["requested_cutoff_date"] = manifest["latest_date"]
        manifest["used_interval"] = {"start": "2019-01-01", "end": manifest["latest_date"], "rows": 30000, "sha256": "b"*64}
        for row in manifest["securities"]:
            row["used_interval"] = {"start": "2019-01-01", "end": manifest["latest_date"], "rows": 500, "sha256": "c"*64}
        checkpoint = freeze_checkpoint(manifest, universe,
                                       calendar_identity=stable_read(args["calendar_csv"])[1],
                                       evidence_identity=stable_read(args["calendar_evidence_json"])[1],
                                       calendar_date_column="date", calendar_evidence=manifest["calendar_provenance"])
        row = checkpoint["selected_source_manifest"]["securities"][0]
        self.assertEqual(row["identity"]["sha256"], "a"*64)
        self.assertEqual(row["used_interval"]["sha256"], "c"*64)
        self.assertEqual(checkpoint["used_interval"]["sha256"], "b"*64)
        self.assertEqual(checkpoint["selected_source_manifest"]["requested_cutoff_date"], manifest["latest_date"])

    def test_diagnostic_cutoff_conflict_blocks_without_reading_selected_sources(self):
        args, _, _, _, _, _, checkpoint = self.checkpoint_fixture()
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        with patch("cloud_v3.prepare.read_selected_panel") as reader:
            with self.assertRaisesRegex(ValueError, "cutoff differs"):
                diagnose_checkpoint(checkpoint_path=path, source_root=args["source_root"],
                                    calendar_csv=args["calendar_csv"], calendar_evidence_json=args["calendar_evidence_json"],
                                    output=args["output"], cutoff_date="2026-03-25")
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
