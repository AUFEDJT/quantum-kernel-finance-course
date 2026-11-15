"""Synthetic temporary-file tests; no laboratory data or cloud access."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from cloud_v3.sources import (checked_calendar, read_stock_csv, read_selected_panel,
                              scan_source, source_files, validate_output_path,
                              public_source_root, checked_cutoff)
from cloud_v3.data import select_universe, build_features


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "provider" / "stock-trading-data"
        self.source.mkdir(parents=True)
        self.calendar = pd.date_range("2019-01-02", periods=3, freq="B")

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, symbol="sh600000", *, intro=True, encoding="gb18030"):
        path = self.source / (symbol + ".csv")
        body = "股票代码,交易日期,开盘价,最高价,最低价,收盘价,成交量\n"
        body += "".join(f"{symbol},{d:%Y-%m-%d},10,12,9,11,1000\n" for d in self.calendar)
        path.write_text(("供应商描述行\n" if intro else "") + body, encoding=encoding)
        return path

    def test_reads_gb_and_utf8_without_modifying_source(self):
        for symbol, encoding, intro in (("sh600000", "gb18030", True), ("sz000001", "utf-8-sig", False)):
            path = self.write_csv(symbol, intro=intro, encoding=encoding)
            before = path.read_bytes()
            frame, record = read_stock_csv(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(record["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame.symbol.unique().tolist(), [symbol])

    def test_scan_uses_content_dates_not_mtime(self):
        self.write_csv()
        manifest = scan_source(self.source.parent, self.calendar)
        self.assertEqual(manifest["latest_date"], "2019-01-04")
        panel, records = read_selected_panel(self.source.parent, ["sh600000"], manifest=manifest)
        self.assertEqual(len(panel), 3)
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest["source_writes"], 0)

    def test_changed_source_cannot_mix_snapshots(self):
        path = self.write_csv()
        manifest = scan_source(self.source.parent, self.calendar)
        path.write_text(path.read_text(encoding="gb18030").replace("1000", "1001"), encoding="gb18030")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            read_selected_panel(self.source.parent, ["sh600000"], manifest=manifest)

    def test_calendar_mismatch_rejected(self):
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "calendar"):
            scan_source(self.source.parent, self.calendar[:-1])

    def test_identifier_mismatch_rejected(self):
        path = self.write_csv()
        path.write_text(path.read_text(encoding="gb18030").replace("sh600000", "sh600001"), encoding="gb18030")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            read_stock_csv(path)

    def test_output_guard_never_writes_source_or_network(self):
        for output in (self.source / "cache", "//server/share/result"):
            with self.assertRaises(ValueError):
                validate_output_path(output, [self.source])
        self.assertFalse((self.source / "cache").exists())
        approved = validate_output_path(self.root / "private", [self.source])
        self.assertFalse(approved.exists())

    def test_only_stock_csv_inventory(self):
        self.write_csv()
        for name in ("sh900001.csv", "sz200001.csv", "sh510300.csv", "supplier.py"):
            (self.source / name).write_text("not read")
        self.assertEqual([p.name for p in source_files(self.source.parent)], ["sh600000.csv"])

    def test_local_course_output_remains_forbidden(self):
        with self.assertRaises(ValueError):
            validate_output_path(public_source_root() / "private-data")

    def test_deployed_snapshot_keeps_sibling_private_directory(self):
        code = self.root / "remote" / "code_compact_20261115"
        private = self.root / "remote" / "private"
        ledger_file = code / "qukernet_finance" / "cloud_v2" / "ledger.py"
        with patch("cloud_v2.ledger.__file__", str(ledger_file)):
            self.assertEqual(public_source_root(), code.resolve())
            self.assertEqual(validate_output_path(private, [self.source]), private.resolve())
            with self.assertRaises(ValueError):
                validate_output_path(code / "cache", [self.source])
            with self.assertRaises(ValueError):
                validate_output_path(self.source / "cache", [self.source])
        self.assertFalse(private.exists())

    def test_duplicate_calendar_rejected(self):
        with self.assertRaises(ValueError):
            checked_calendar([self.calendar[0], self.calendar[0]])

    def test_fixed_cutoff_filters_before_latest_and_calendar_checks(self):
        self.write_csv()
        manifest = scan_source(self.source.parent, self.calendar[:2], cutoff_date="2019-01-03")
        self.assertEqual(manifest["latest_date"], "2019-01-03")
        self.assertEqual(manifest["requested_cutoff_date"], "2019-01-03")
        self.assertEqual(manifest["securities"][0]["observed_valid_dates"], ["2019-01-02", "2019-01-03"])
        self.assertEqual(manifest["used_interval"]["rows"], 2)
        self.assertEqual(manifest["securities"][0]["identity"]["rows"], 3)

    def test_cutoff_requires_observed_date_without_rollback(self):
        self.write_csv()
        expanded_calendar = self.calendar.union(pd.DatetimeIndex(["2019-01-07"]))
        with self.assertRaisesRegex(ValueError, "no observed source"):
            scan_source(self.source.parent, expanded_calendar, cutoff_date="2019-01-07")
        with self.assertRaisesRegex(ValueError, "absent.*calendar"):
            scan_source(self.source.parent, expanded_calendar, cutoff_date="2019-01-06")

    def test_cutoff_dates_must_be_normalized(self):
        for value in ("2019-01-03T12:00:00", "2019-01-03T00:00:00Z", "2018-12-31", "NaT"):
            with self.assertRaises(ValueError):
                checked_cutoff(value)

    def test_post_cutoff_change_preserves_interval_panel_and_features(self):
        dates = pd.bdate_range("2025-08-01", periods=160)
        rng = np.random.default_rng(90)
        close = 10 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
        frame = pd.DataFrame({"symbol": "sh600000", "date": dates, "open": close * .999,
                              "high": close * 1.01, "low": close * .99, "close": close,
                              "volume": rng.integers(100, 10000, len(dates))})
        path = self.source / "sh600000.csv"
        frame.to_csv(path, index=False)
        cutoff = dates[149]
        first = scan_source(self.source.parent, dates, cutoff_date=cutoff)
        panel1, _ = read_selected_panel(self.source.parent, ["sh600000"], manifest=first)
        frame.loc[frame.date > cutoff, ["open", "high", "low", "close"]] *= 1.7
        frame.loc[frame.date > cutoff, "volume"] *= 3
        frame.to_csv(path, index=False)
        second = scan_source(self.source.parent, dates, cutoff_date=cutoff)
        panel2, _ = read_selected_panel(self.source.parent, ["sh600000"], manifest=second)
        self.assertNotEqual(first["securities"][0]["identity"]["sha256"], second["securities"][0]["identity"]["sha256"])
        self.assertEqual(first["used_interval"], second["used_interval"])
        pd.testing.assert_frame_equal(panel1, panel2)
        pd.testing.assert_frame_equal(build_features(panel1, dates[:150]), build_features(panel2, dates[:150]))
        with self.assertRaisesRegex(RuntimeError, "changed"):
            read_selected_panel(self.source.parent, ["sh600000"], manifest=first)

    def test_cutoff_retains_complete_calendar_and_detects_missing_bar(self):
        self.write_csv()
        path = self.source / "sh600000.csv"
        lines = path.read_text(encoding="gb18030").splitlines()
        path.write_text("\n".join(line for line in lines if "2019-01-03" not in line) + "\n", encoding="gb18030")
        manifest = scan_source(self.source.parent, self.calendar, cutoff_date="2019-01-04")
        self.assertEqual(manifest["calendar_sha256"], hashlib.sha256("\n".join(self.calendar.strftime("%Y-%m-%d")).encode()).hexdigest())
        panel, _ = read_selected_panel(self.source.parent, ["sh600000"], manifest=manifest)
        self.assertEqual(self.calendar.difference(panel.date).strftime("%Y-%m-%d").tolist(), ["2019-01-03"])

    def test_selection_and_assignments_are_deterministic_after_future_change(self):
        dates = pd.bdate_range("2024-01-02", periods=500)
        symbols = [f"sh60{i:04d}" for i in range(30)] + [f"sz00{i:04d}" for i in range(30)]
        self.calendar = dates
        for symbol in symbols:
            self.write_csv(symbol, intro=False, encoding="utf-8")
        cutoff = dates[479]
        first = scan_source(self.source.parent, dates, cutoff_date=cutoff)
        for symbol in symbols:
            path = self.source / (symbol + ".csv")
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.replace(",1000", ",1234") if any(d in line for d in dates[480:].strftime("%Y-%m-%d")) else line for line in lines) + "\n", encoding="utf-8")
        second = scan_source(self.source.parent, dates, cutoff_date=cutoff)
        one = select_universe(first, dates, latest_date=cutoff)
        two = select_universe(second, dates, latest_date=cutoff)
        self.assertEqual(one[:2], two[:2])
        self.assertEqual(one[3], two[3])
        self.assertEqual(first["used_interval"], second["used_interval"])

    def test_selected_read_cannot_override_frozen_cutoff(self):
        self.write_csv()
        manifest = scan_source(self.source.parent, self.calendar, cutoff_date="2019-01-03")
        with self.assertRaisesRegex(ValueError, "preserve.*cutoff"):
            read_selected_panel(self.source.parent, ["sh600000"], manifest=manifest, latest_date="2019-01-04")


if __name__ == "__main__":
    unittest.main()
