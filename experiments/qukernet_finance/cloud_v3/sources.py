"""Read-only stock CSV access; source data and manifests are private artifacts.

No API, supplier program, archive extraction, credentials or cloud execution is
used here. Calendar dates must be supplied from a separately checked calendar.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from ..cloud_v2.ledger import public_source_root
except ImportError:
    from cloud_v2.ledger import public_source_root

FIELDS = {"股票代码": "symbol", "股票名称": "name", "交易日期": "date",
          "日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
          "收盘价": "close", "成交量": "volume", "前收盘价": "prev_close", "成交额": "amount"}
REQUIRED = {"symbol", "date", "open", "high", "low", "close", "volume"}
STOCK_NAME = re.compile(r"(?:sh(?:60|68)\d{4}|sz(?:00|30)\d{4}|bj\d{6})\.csv", re.I)


def is_network_path(value):
    value = str(value)
    if value.startswith(("\\\\", "//")):
        return True
    path = Path(value)
    if os.name == "nt" and path.drive:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 4
    return False


def validate_output_path(output, source_roots=(), *, public_root=None):
    """Pure guard; never creates or probes a source with a write operation."""
    if is_network_path(output):
        raise ValueError("Outputs cannot be on network storage")
    target = Path(output).resolve()
    if is_network_path(target):
        raise ValueError("Resolved output cannot be network storage")
    public_root = public_source_root() if public_root is None else public_root
    for root in [*source_roots, public_root]:
        if root is None:
            continue
        root = Path(root).resolve()
        if target == root or root in target.parents:
            raise ValueError("Private output overlaps a read-only source or public course")
    return target


def checked_calendar(calendar):
    values = calendar["date"] if isinstance(calendar, pd.DataFrame) else calendar
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if dates.empty or dates.hasnans or dates.has_duplicates or dates.tz is not None:
        raise ValueError("Calendar must contain unique nonmissing naive market dates")
    if not dates.equals(dates.normalize()):
        raise ValueError("Calendar requires daily dates")
    return dates.sort_values()


def checked_cutoff(value):
    """Parse an explicit daily boundary without silently normalizing a time."""
    if value is None:
        return None
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tz is not None or date != date.normalize() or date < pd.Timestamp("2019-01-01"):
        raise ValueError("Cutoff must be a normalized date on or after 2019-01-01")
    return date


def used_interval_identity(frame, *, end):
    """Hash normalized model-source rows separately from the full-file hash.

    Only source columns consumed by feature/label construction are included.
    A subsequent independent audit can therefore distinguish newer file
    contents from changes to the frozen historical input interval.
    """
    columns = [name for name in ("symbol", "date", "open", "high", "low", "close",
                                "volume", "prev_close", "amount") if name in frame]
    used = frame.loc[frame.date.between("2019-01-01", end), columns].copy()
    used = used.sort_values(["symbol", "date"]).reset_index(drop=True)
    used["date"] = used.date.dt.strftime("%Y-%m-%d")
    # Fixed serialization; NA stays explicit rather than becoming a fabricated
    # observation. Full original bytes remain covered by identity.sha256.
    payload = used.to_csv(index=False, float_format="%.17g", na_rep="NA", lineterminator="\n").encode("utf-8")
    return {"schema": "normalized_model_source_rows_v1", "start": "2019-01-01",
            "end": pd.Timestamp(end).strftime("%Y-%m-%d"), "rows": len(used),
            "columns": columns, "sha256": hashlib.sha256(payload).hexdigest()}


def valid_bars(frame):
    p = frame[["open", "high", "low", "close", "volume"]]
    return (np.isfinite(p).all(axis=1) & (p[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (p.volume >= 0) & (p.high >= p[["open", "close", "low"]].max(axis=1))
            & (p.low <= p[["open", "close", "high"]].min(axis=1)))


def source_directory(root):
    root = Path(root).resolve()
    candidate = root if root.name == "stock-trading-data" else root / "stock-trading-data"
    if not candidate.is_dir():
        raise FileNotFoundError("Read-only stock-trading-data directory unavailable")
    if candidate.is_symlink():
        raise ValueError("Source directory symlink is not admitted")
    return candidate


def source_files(root):
    directory = source_directory(root)
    with os.scandir(directory) as entries:
        files = [Path(e.path) for e in entries
                 if e.is_file(follow_symlinks=False) and STOCK_NAME.fullmatch(e.name)]
    return sorted(files, key=lambda p: p.name.lower())


def read_stock_csv(path, *, max_bytes=64 * 1024**2):
    """Read one bounded provider CSV, checking stability and identifier agreement."""
    path = Path(path)
    if path.is_symlink() or not STOCK_NAME.fullmatch(path.name):
        raise ValueError("Only plain A-share CSV files are admitted")
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError("CSV exceeds bounded-reader size limit")
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed while reading")
    try:
        content = payload.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        content = payload.decode("gb18030", errors="strict")
        encoding = "gb18030"
    lines = content.splitlines()
    if not lines:
        raise ValueError("Empty source CSV")
    first = [cell.strip().lstrip("\ufeff") for cell in next(csv.reader([lines[0]]))]
    header = 0 if "date" in first or "交易日期" in first or "日期" in first else 1
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), dtype=str)
    frame.columns = [FIELDS.get(str(c).strip().lstrip("\ufeff"), str(c).strip()) for c in frame.columns]
    if frame.columns.duplicated().any() or not REQUIRED.issubset(frame.columns):
        raise ValueError("Missing or duplicate normalized CSV columns")
    symbol = path.stem.lower()
    supplied = frame.symbol.astype(str).str.strip().str.lower()
    if not supplied.isin([symbol, symbol[2:]]).all():
        raise ValueError("CSV symbol disagrees with filename")
    frame["symbol"] = symbol
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any() or frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Invalid source daily date")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate source security-date key")
    for column in ("open", "high", "low", "close", "volume", "prev_close", "amount"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    identity = {"source_path": str(path.resolve()), "filename": path.name,
                "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                "sha256": hashlib.sha256(payload).hexdigest(), "encoding": encoding,
                "header_skip": header, "rows": len(frame)}
    return frame, identity


def scan_source(root, calendar, *, calendar_provenance=None, cutoff_date=None):
    """Inventory actual CSV contents; do not select on future completeness.

    D is the latest observed CSV date that is also a supplied market date.
    It does not assert that every security is complete through D.
    """
    dates = checked_calendar(calendar)
    cutoff = checked_cutoff(cutoff_date)
    if cutoff is not None and cutoff not in dates:
        raise ValueError("Requested cutoff is absent from the supplied market calendar")
    files = source_files(root)
    if not files:
        raise ValueError("No eligible A-share source filenames")
    records, latest = [], None
    for path in files:
        try:
            frame, identity = read_stock_csv(path)
        except Exception as error:
            error.source_audit = {"code": "source_read_failed", "filename": path.name}
            raise
        recent = frame[frame.date >= pd.Timestamp("2019-01-01")]
        if cutoff is not None:
            recent = recent.loc[recent.date <= cutoff]
        if recent.empty:
            continue
        if not recent.date.isin(dates).all():
            error = ValueError("Source date is absent from the supplied market calendar")
            error.source_audit = {"code": "source_dates_outside_calendar", "filename": path.name,
                                  "dates": recent.loc[~recent.date.isin(dates), "date"].dt.strftime("%Y-%m-%d").tolist()}
            raise error
        observed = recent.date.max()
        latest = observed if latest is None else max(latest, observed)
        good = recent.loc[valid_bars(recent), "date"]
        record = {"symbol": path.stem.lower(), "exchange": path.stem[:2].lower(),
                        "observed_valid_dates": good.dt.strftime("%Y-%m-%d").tolist(),
                        "first_observed_date": recent.date.min().strftime("%Y-%m-%d"),
                        "last_observed_date": observed.strftime("%Y-%m-%d"), "identity": identity}
        if cutoff is not None:
            record["used_interval"] = used_interval_identity(recent, end=cutoff)
        records.append(record)
    if latest is None:
        raise ValueError("No source records from 2019 onward")
    if cutoff is not None and latest != cutoff:
        raise ValueError("Requested cutoff has no observed source record; automatic rollback is forbidden")
    manifest = {"schema": "cloud_v3_source_scan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(Path(root).resolve()), "latest_date": latest.strftime("%Y-%m-%d"),
            "latest_date_basis": "maximum actual CSV date, validated against supplied calendar; not mtime",
            "securities": records, "all_scanned_files": len(files), "source_writes": 0,
            "calendar_provenance": calendar_provenance or {"status": "unverified", "source_kind": "unspecified"},
            "calendar_sha256": hashlib.sha256("\n".join(dates.strftime("%Y-%m-%d")).encode()).hexdigest(),
            "universe_scope": "source-visible A-share files; completeness and delisted-security coverage require separate audit"}
    if cutoff is not None:
        manifest["requested_cutoff_date"] = cutoff.strftime("%Y-%m-%d")
        manifest["latest_date_basis"] = "maximum actual CSV date inside the requested historical interval; exact cutoff observation required"
        manifest["used_interval"] = {"start": "2019-01-01", "end": cutoff.strftime("%Y-%m-%d"),
                                      "rows": sum(r["used_interval"]["rows"] for r in records),
                                      "sha256": hashlib.sha256(json.dumps(
                                          [{"symbol": r["symbol"], "used_interval": r["used_interval"]} for r in records],
                                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return manifest


def read_selected_panel(root, symbols, *, manifest, latest_date=None):
    directory = source_directory(root)
    symbols = list(symbols)
    if len(symbols) != len(set(symbols)):
        raise ValueError("Duplicate requested security")
    expected_records = {r["symbol"]: r for r in manifest["securities"]}
    expected = {symbol: r["identity"] for symbol, r in expected_records.items()}
    end = pd.Timestamp(latest_date or manifest["latest_date"])
    cutoff = checked_cutoff(manifest.get("requested_cutoff_date"))
    if cutoff is not None and end != cutoff:
        raise ValueError("Selected read must preserve the frozen cutoff")
    frames, used = [], []
    for symbol in symbols:
        if symbol not in expected:
            raise ValueError("Selected symbol not present in frozen source scan")
        filename = symbol + ".csv"
        try:
            frame, identity = read_stock_csv(directory / filename)
        except Exception as error:
            error.source_audit = {"code": "selected_source_read_failed", "filename": filename}
            raise
        if any(identity[k] != expected[symbol][k] for k in ("bytes", "mtime_ns", "sha256")):
            error = RuntimeError("Selected source changed since frozen source scan")
            error.source_audit = {"code": "selected_source_version_changed", "filename": filename}
            raise error
        selected = frame[frame.date.between("2019-01-01", end)].copy()
        if "used_interval" in expected_records[symbol]:
            interval = used_interval_identity(selected, end=end)
            if interval != expected_records[symbol]["used_interval"]:
                raise RuntimeError("Selected source interval disagrees with frozen scan")
            identity["used_interval"] = interval
        frames.append(selected)
        used.append(identity)
    if not frames:
        raise ValueError("No selected securities")
    return pd.concat(frames, ignore_index=True), used
