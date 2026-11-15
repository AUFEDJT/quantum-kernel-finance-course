"""Explicit, read-only-source preparation CLI. Does not start cloud work.

Run from experiments/qukernet_finance with ``python -m cloud_v3.prepare``.
Every input path is explicit. Only private snapshot and audit outputs are
written; an unavailable source is a recorded stop, never a data substitution.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from .data import prepare_window, save_window, select_universe, digest, calendar_split
from .sources import (checked_calendar, read_selected_panel, scan_source,
                      validate_output_path, valid_bars, STOCK_NAME, checked_cutoff)


def stable_read(path):
    path = Path(path)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during read")
    return payload, {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                     "sha256": hashlib.sha256(payload).hexdigest()}


def _decode(payload):
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode("gb18030", errors="strict")


def read_calendar_bytes(payload, date_column):
    lines = _decode(payload).splitlines()
    if not lines:
        raise ValueError("Empty calendar CSV")
    header = None
    for index, line in enumerate(lines[:2]):
        cells = [cell.strip().lstrip("\ufeff") for cell in next(csv.reader([line]))]
        if date_column in cells:
            header = index
            break
    if header is None:
        raise ValueError("Requested calendar date column is absent from the first two lines")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), dtype=str)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate calendar columns")
    return checked_calendar(frame[date_column]), header


def evidence_gate(evidence, data_role):
    if data_role != "real_required" or not isinstance(evidence, dict):
        raise ValueError("Real-source evidence is required")
    serialized = json.dumps(evidence, ensure_ascii=False).lower()
    if any(marker in serialized for marker in ("synthetic", "fixture", "mock_data", "test_data", "模拟数据", "构造数据", "测试夹具")):
        raise ValueError("Constructed data cannot be admitted as real source evidence")
    if evidence.get("data_role") != "real_market_data":
        raise ValueError("Evidence must explicitly identify real market data")
    if evidence.get("status") not in ("verified_market_calendar", "explicitly_accepted_diagnostic_proxy"):
        raise ValueError("Unverified calendar cannot prepare the real-data pilot")
    if not evidence.get("evidence") or not evidence.get("source_kind"):
        raise ValueError("Calendar provenance is incomplete")
    if evidence["status"] == "explicitly_accepted_diagnostic_proxy" and not evidence.get("acceptance_note"):
        raise ValueError("Proxy calendar acceptance is not recorded")


def latest_date_gate(value, *, today=None):
    latest = pd.Timestamp(value)
    today = pd.Timestamp(today or datetime.now().date())
    if latest != latest.normalize() or latest <= pd.Timestamp("2024-12-31") or latest > today:
        raise ValueError("Latest observed date is stale, malformed, or in the future")
    return latest.strftime("%Y-%m-%d")


def _save_audit(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _private_source_context(error):
    """Only locally defined codes and normalized file/date identifiers escape."""
    context = getattr(error, "source_audit", {})
    codes = {"source_read_failed", "selected_source_read_failed",
             "source_dates_outside_calendar", "selected_source_version_changed"}
    if context.get("code") not in codes or not STOCK_NAME.fullmatch(context.get("filename", "")):
        return None
    result = {"code": context["code"], "filename": context["filename"]}
    if "dates" in context:
        dates = pd.DatetimeIndex(pd.to_datetime(context["dates"], errors="raise"))
        result["dates"] = dates.strftime("%Y-%m-%d").tolist()
    return result


def freeze_checkpoint(manifest, universe, *, calendar_identity, evidence_identity,
                      calendar_date_column, calendar_evidence):
    """Compact provenance only: no prices, features, returns or labels."""
    identities = [{"symbol": r["symbol"], "exchange": r["exchange"],
                   "identity": {k: r["identity"][k] for k in
                                ("filename", "bytes", "mtime_ns", "sha256")},
                   **({"used_interval": r["used_interval"]} if "used_interval" in r else {})}
                  for r in manifest["securities"]]
    chosen = set(universe["selected_symbols"])
    checkpoint = {"schema": "cloud_v3_preparation_checkpoint_v1",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "latest_date": manifest["latest_date"],
                  "source_manifest_sha256": digest(manifest),
                  "source_identities": identities,
                  "selected_source_manifest": {"latest_date": manifest["latest_date"],
                                               "securities": [r for r in identities if r["symbol"] in chosen]},
                  "calendar_identity": calendar_identity,
                  "calendar_evidence_identity": evidence_identity,
                  "calendar_date_column": calendar_date_column,
                  "calendar_evidence": calendar_evidence, "universe": universe,
                  "reuse_scope": "Frozen source identities and stock/date pairs; diagnostic reuse must rehash the 50 sources and calendar. This is not admission of a refreshed full source universe.",
                  "source_writes": 0, "cloud_submissions": 0}
    if "requested_cutoff_date" in manifest:
        checkpoint["requested_cutoff_date"] = manifest["requested_cutoff_date"]
        checkpoint["used_interval"] = manifest["used_interval"]
        checkpoint["selected_source_manifest"]["requested_cutoff_date"] = manifest["requested_cutoff_date"]
    checkpoint["sha256"] = digest(checkpoint)
    return checkpoint


def selected_coverage_diagnostics(panel, calendar, universe, latest_date):
    """Inspect all fifty once in memory without selecting on gaps or labels."""
    dates, split = calendar_split(calendar, latest_date)
    selected = universe["selected_symbols"]
    if len(selected) != 50 or len(set(selected)) != 50:
        raise ValueError("Frozen fifty-security set is required")
    assignments = {r["symbol"]: r for r in universe["O_assignments"]}
    if set(assignments) != set(selected):
        raise ValueError("Frozen stock/date assignment mismatch")
    records = []
    for symbol in selected:
        group = panel.loc[panel.symbol.eq(symbol)].copy()
        if group.empty or group.date.duplicated().any():
            raise ValueError("Missing or duplicated selected source observations")
        observed = pd.DatetimeIndex(group.date)
        good = pd.DatetimeIndex(group.loc[valid_bars(group), "date"])
        invalid = observed.difference(good)
        j = assignments[symbol]["j"]
        history = dates[max(0, j-120):j+1]
        endpoints = dates[[j+1, j+6]]
        historical = pd.Series(dates.isin(good), index=dates).rolling(121, min_periods=121).sum().eq(121)
        observed_span = dates[dates >= observed.min()]
        records.append({"symbol": symbol, "filename": symbol + ".csv",
                        "missing_reason": "Unverified: suspension or incomplete source are possible; absence is not classified as provider omission.",
                        "first_observed": observed.min().strftime("%Y-%m-%d"),
                        "last_observed": observed.max().strftime("%Y-%m-%d"),
                        "missing_dates_since_first_observation": observed_span.difference(observed).strftime("%Y-%m-%d").tolist(),
                        "invalid_bar_dates": invalid.strftime("%Y-%m-%d").tolist(),
                        "validation_dates_with_complete_121_bar_history": int(historical.iloc[split["valid_days"]].sum()),
                        "evaluation_date": assignments[symbol]["date"],
                        "evaluation_history_start": history[0].strftime("%Y-%m-%d"),
                        "evaluation_missing_history_dates": history.difference(observed).strftime("%Y-%m-%d").tolist(),
                        "evaluation_invalid_history_dates": history.intersection(invalid).strftime("%Y-%m-%d").tolist(),
                        "evaluation_history_complete": bool(len(history) == 121 and history.isin(good).all()),
                        "label_endpoint_dates": endpoints.strftime("%Y-%m-%d").tolist(),
                        "missing_or_invalid_label_endpoints": endpoints.difference(good).strftime("%Y-%m-%d").tolist()})
    return {"schema": "cloud_v3_selected_date_coverage_v1", "latest_date": latest_date,
            "universe_sha256": universe["sha256"], "selected_stocks": 50,
            "evaluation_history_failures": sum(not r["evaluation_history_complete"] for r in records),
            "records": records, "source_writes": 0, "cloud_submissions": 0,
            "interpretation": "Missing records and invalid OHLCV, not inferred suspensions. A complete history alone does not guarantee all eighty derived features are finite. No source values are filled or dates reassigned."}


def diagnose_checkpoint(*, checkpoint_path, source_root, calendar_csv,
                        calendar_evidence_json, output, cutoff_date=None):
    """One selected-source pass; no full inventory scan or cloud execution."""
    destination = validate_output_path(output, [source_root, calendar_csv,
                                              calendar_evidence_json, checkpoint_path])
    checkpoint_bytes, checkpoint_identity = stable_read(checkpoint_path)
    checkpoint = json.loads(_decode(checkpoint_bytes))
    value = dict(checkpoint)
    fingerprint = value.pop("sha256", None)
    if checkpoint.get("schema") != "cloud_v3_preparation_checkpoint_v1" or digest(value) != fingerprint:
        raise ValueError("Checkpoint checksum mismatch")
    cutoff = checked_cutoff(cutoff_date)
    if cutoff is not None and cutoff.strftime("%Y-%m-%d") != checkpoint["latest_date"]:
        raise ValueError("Requested cutoff differs from frozen checkpoint")
    universe = dict(checkpoint["universe"])
    universe_hash = universe.pop("sha256", None)
    if digest(universe) != universe_hash:
        raise ValueError("Frozen universe checksum mismatch")
    calendar_bytes, calendar_identity = stable_read(calendar_csv)
    evidence_bytes, evidence_identity = stable_read(calendar_evidence_json)
    if calendar_identity != checkpoint["calendar_identity"] or evidence_identity != checkpoint["calendar_evidence_identity"]:
        raise ValueError("Calendar or evidence version differs from checkpoint")
    evidence_gate(json.loads(_decode(evidence_bytes)), "real_required")
    calendar, _ = read_calendar_bytes(calendar_bytes, checkpoint["calendar_date_column"])
    panel, used = read_selected_panel(source_root, checkpoint["universe"]["selected_symbols"],
                                     manifest=checkpoint["selected_source_manifest"],
                                     latest_date=checkpoint["latest_date"])
    result = selected_coverage_diagnostics(panel, calendar, checkpoint["universe"], checkpoint["latest_date"])
    if stable_read(calendar_csv)[1] != calendar_identity or stable_read(calendar_evidence_json)[1] != evidence_identity:
        raise RuntimeError("Calendar or evidence changed during diagnosis")
    result["checkpoint_identity"] = checkpoint_identity
    result["verified_source_identities"] = [{k: r[k] for k in ("filename", "bytes", "mtime_ns", "sha256")} for r in used]
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = destination / ("selected_coverage_" + stamp + ".json")
    _save_audit(path, result)
    return {"status": "diagnosed", "audit_file": path.name, "selected_stocks": 50,
            "evaluation_history_failures": result["evaluation_history_failures"],
            "source_writes": 0, "cloud_submissions": 0}


def prepare_inputs(*, source_root, calendar_csv, calendar_date_column,
                   calendar_evidence_json, output, data_role="real_required", cutoff_date=None,
                   protocol=None, frozen_samples=None):
    """Return a safe receipt; errors are typed and never expose source contents."""
    if protocol is not None or frozen_samples is not None:
        return prepare_missing_inputs(source_root=source_root, calendar_csv=calendar_csv,
                                      calendar_date_column=calendar_date_column,
                                      calendar_evidence_json=calendar_evidence_json, output=output,
                                      data_role=data_role, cutoff_date=cutoff_date,
                                      protocol=protocol, frozen_samples=frozen_samples)
    stage, destination = "output_path_guard", None
    artifacts = {}
    diagnostics = None
    started = datetime.now(timezone.utc).isoformat()
    try:
        destination = validate_output_path(output, [source_root, calendar_csv, calendar_evidence_json])
        if (destination / "latest50.npz").exists() or (destination / "latest50.json").exists():
            raise FileExistsError("Frozen output exists")
        stage = "requested_cutoff_gate"
        cutoff = checked_cutoff(cutoff_date)
        if cutoff is not None:
            latest_date_gate(cutoff)
        stage = "calendar_evidence"
        evidence_bytes, evidence_before = stable_read(calendar_evidence_json)
        evidence = json.loads(_decode(evidence_bytes))
        evidence_gate(evidence, data_role)
        stage = "calendar_read"
        calendar_bytes, calendar_before = stable_read(calendar_csv)
        calendar, calendar_header_skip = read_calendar_bytes(calendar_bytes, calendar_date_column)
        stage = "source_inventory"
        manifest = scan_source(source_root, calendar, calendar_provenance=evidence, cutoff_date=cutoff)
        manifest["data_role"] = "real_market_data"
        stage = "latest_date_gate"
        latest = latest_date_gate(manifest["latest_date"])
        stage = "universe_freeze"
        symbols, universe, _, _ = select_universe(manifest, calendar, latest_date=latest)
        stage = "preparation_checkpoint"
        checkpoint = freeze_checkpoint(manifest, universe, calendar_identity=calendar_before,
                                       evidence_identity=evidence_before,
                                       calendar_date_column=calendar_date_column,
                                       calendar_evidence=evidence)
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        checkpoint_name = "preparation_checkpoint_" + stamp + ".json"
        _save_audit(destination / checkpoint_name, checkpoint)
        artifacts["checkpoint"] = checkpoint_name
        stage = "selected_source_read"
        panel, used = read_selected_panel(source_root, symbols, manifest=manifest, latest_date=latest)
        if not panel.date.isin(calendar).all():
            raise ValueError("Selected-security date union is not contained in calendar")
        stage = "calendar_stability_check"
        _, calendar_after = stable_read(calendar_csv)
        _, evidence_after = stable_read(calendar_evidence_json)
        if calendar_before != calendar_after or evidence_before != evidence_after:
            raise RuntimeError("Calendar or provenance changed during source reading")
        selected_dates = pd.DatetimeIndex(panel.date.unique()).sort_values()
        relevant_calendar = calendar[(calendar >= pd.Timestamp("2019-01-01")) & (calendar <= pd.Timestamp(latest))]
        coverage = {"calendar_first": relevant_calendar.min().strftime("%Y-%m-%d"),
                    "calendar_last": relevant_calendar.max().strftime("%Y-%m-%d"),
                    "calendar_dates": len(relevant_calendar), "selected_union_dates": len(selected_dates),
                    "selected_union_first": selected_dates.min().strftime("%Y-%m-%d"),
                    "selected_union_last": selected_dates.max().strftime("%Y-%m-%d"),
                    "calendar_dates_without_selected_observation": relevant_calendar.difference(selected_dates).strftime("%Y-%m-%d").tolist(),
                    "per_security": [{"symbol": str(symbol), "rows": len(group),
                                      "first_date": group.date.min().strftime("%Y-%m-%d"),
                                      "last_date": group.date.max().strftime("%Y-%m-%d")}
                                     for symbol, group in panel.groupby("symbol", sort=True)],
                    "interpretation": "observed data/calendar agreement; does not certify official exchange coverage"}
        manifest["calendar_input"] = {"identity_before": calendar_before, "identity_after": calendar_after,
                                      "date_column": calendar_date_column, "evidence_identity": evidence_before,
                                      "header_skip": calendar_header_skip,
                                      "selected_date_union_matches_calendar": True,
                                      "calendar_days": len(calendar), "provenance": evidence,
                                      "coverage_audit": coverage}
        manifest["selected_source_identities"] = used
        stage = "selected_coverage_diagnostics"
        diagnostics = selected_coverage_diagnostics(panel, calendar, universe, latest)
        coverage_name = "selected_coverage_" + stamp + ".json"
        _save_audit(destination / coverage_name, diagnostics)
        artifacts["selected_coverage"] = coverage_name
        stage = "feature_and_sample_preparation"
        window = prepare_window(panel, calendar, manifest, latest_date=latest)
        if window["universe"]["sha256"] != universe["sha256"]:
            raise RuntimeError("Universe changed during preparation")
        # Check the external calendar again after all CPU preparation.
        _, final_calendar = stable_read(calendar_csv)
        _, final_evidence = stable_read(calendar_evidence_json)
        if final_calendar != calendar_before or final_evidence != evidence_before:
            raise RuntimeError("Calendar or provenance changed during feature preparation")
        stage = "private_snapshot_write"
        receipt = save_window(window, destination, source_roots=[source_root], source_manifest=manifest)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _save_audit(destination / ("source_calendar_audit_" + stamp + ".json"), manifest)
        result = {"status": "prepared", "data_role": "real_market_data", "latest_date": latest,
                  "rows": window["audit"]["rows"], "selected_stocks": 50,
                  "calendar_status": evidence["status"], "cloud_submissions": 0,
                  "source_writes": 0, "started_utc": started, "snapshot": receipt}
        if cutoff is not None:
            result["requested_cutoff_date"] = cutoff.strftime("%Y-%m-%d")
            result["used_interval"] = manifest["used_interval"]
        result["artifacts"] = artifacts
        _save_audit(destination / ("preparation_receipt_" + stamp + ".json"), result)
        return result
    except Exception as error:
        # Do not serialize exception messages; provider errors can contain
        # internal source paths. Preserve only type and our fixed stage name.
        result = {"status": "blocked", "stage": stage, "error_type": type(error).__name__,
                  "data_role": data_role, "cloud_submissions": 0, "source_writes": 0,
                  "started_utc": started, "automatic_retry": False, "artifacts": artifacts}
        context = _private_source_context(error)
        if context is not None:
            result["source_diagnostic"] = context
        preparation_context = getattr(error, "preparation_audit", None)
        if (isinstance(preparation_context, dict) and preparation_context.get("code") in
                {"fixed_sample_unavailable", "training_minority_class_too_small", "frozen_outer_coverage_invalid"}):
            result["preparation_diagnostic"] = preparation_context
        if diagnostics is not None:
            result["coverage_summary"] = {
                "selected_stocks": diagnostics["selected_stocks"],
                "evaluation_history_failures": diagnostics["evaluation_history_failures"],
                "validation_complete_history_counts": {
                    r["symbol"]: r["validation_dates_with_complete_121_bar_history"] for r in diagnostics["records"]},
                "interpretation": "Independent coverage evidence; not a claim that coverage caused the first preparation error."}
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _save_audit(destination / ("preparation_blocked_" + stamp + ".json"), result)
        return result


def prepare_missing_inputs(*, source_root, calendar_csv, calendar_date_column,
                           calendar_evidence_json, output, data_role, cutoff_date,
                           protocol, frozen_samples):
    """Use immutable old sample keys and sources; no repeated stock or row selection."""
    from .data import MISSING_PROTOCOL_ID
    destination, stage = None, "output_path_guard"
    started = datetime.now(timezone.utc).isoformat()
    try:
        if protocol is None or frozen_samples is None:
            raise ValueError("Both explicit protocol and frozen-samples paths are required")
        destination = validate_output_path(output, [source_root, calendar_csv, calendar_evidence_json,
                                                   protocol, frozen_samples])
        if (destination / "latest50.npz").exists() or (destination / "latest50.json").exists():
            raise FileExistsError("Frozen output exists")
        stage = "missing_protocol_admission"
        protocol_bytes, protocol_identity = stable_read(protocol)
        approved = Path(__file__).with_name("protocol_cutoff_20260325_qlib_missing_v1.json")
        if protocol_bytes != approved.read_bytes():
            raise ValueError("Explicit protocol differs from approved version")
        specification = json.loads(_decode(protocol_bytes))
        if specification.get("protocol_id") != MISSING_PROTOCOL_ID:
            raise ValueError("Missing policy protocol mismatch")
        latest = latest_date_gate(specification["data"]["cutoff_date"])
        if cutoff_date is not None and checked_cutoff(cutoff_date) != pd.Timestamp(latest):
            raise ValueError("CLI cutoff differs from protocol")
        stage = "frozen_sample_admission"
        frozen_bytes, frozen_identity = stable_read(frozen_samples)
        frozen = json.loads(_decode(frozen_bytes))
        verify = dict(frozen)
        if verify.pop("sha256", None) != digest(verify):
            raise ValueError("Frozen sample record checksum mismatch")
        stage = "calendar_evidence"
        evidence_bytes, evidence_identity = stable_read(calendar_evidence_json)
        calendar_bytes, calendar_identity = stable_read(calendar_csv)
        if (evidence_identity != frozen["calendar_evidence_identity"]
                or calendar_identity != frozen["calendar_identity"]):
            raise ValueError("Original calendar or calendar evidence changed")
        evidence = json.loads(_decode(evidence_bytes))
        evidence_gate(evidence, data_role)
        calendar, _ = read_calendar_bytes(calendar_bytes, calendar_date_column)
        manifest = frozen["selected_source_manifest"]
        if manifest["calendar_provenance"] != evidence:
            raise ValueError("Source calendar provenance differs")
        stage = "selected_source_read"
        panel, used = read_selected_panel(source_root, frozen["universe"]["selected_symbols"],
                                         manifest=manifest, latest_date=latest)
        stage = "missing_feature_and_sample_preparation"
        window = prepare_window(panel, calendar, manifest, latest_date=latest,
                                protocol=specification, frozen_samples=frozen)
        stage = "input_stability_check"
        for path, identity in ((calendar_csv, calendar_identity), (calendar_evidence_json, evidence_identity),
                               (protocol, protocol_identity), (frozen_samples, frozen_identity)):
            if stable_read(path)[1] != identity:
                raise RuntimeError("A frozen preparation input changed")
        stage = "private_snapshot_write"
        receipt = save_window(window, destination, source_roots=[source_root], source_manifest=manifest)
        result = {"status": "prepared", "data_role": "real_market_data", "protocol_id": MISSING_PROTOCOL_ID,
                  "latest_date": latest, "rows": window["audit"]["rows"], "selected_stocks": 50,
                  "calendar_status": evidence["status"], "cloud_submissions": 0, "source_writes": 0,
                  "started_utc": started, "snapshot": receipt,
                  "frozen_samples_sha256": window["audit"]["frozen_samples_sha256"],
                  "frozen_sample_record_sha256": frozen["sha256"],
                  "preprocessing_sha256": window["preprocessing_hash"],
                  "missingness": window["audit"]["missingness"],
                  "raw_observations_filled": 0, "labels_filled": 0,
                  "preparation_inputs": {"protocol": protocol_identity, "frozen_samples": frozen_identity,
                                         "calendar": calendar_identity, "calendar_evidence": evidence_identity},
                  "verified_source_file_count": len(used), "automatic_retry": False}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _save_audit(destination / ("preparation_receipt_"+stamp+".json"), result)
        return result
    except Exception as error:
        result = {"status": "blocked", "stage": stage, "error_type": type(error).__name__,
                  "data_role": data_role, "cloud_submissions": 0, "source_writes": 0,
                  "started_utc": started, "automatic_retry": False}
        context = _private_source_context(error)
        if context is not None:
            result["source_diagnostic"] = context
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _save_audit(destination / ("preparation_blocked_"+stamp+".json"), result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--calendar-csv", required=True)
    parser.add_argument("--calendar-date-column", required=True)
    parser.add_argument("--calendar-evidence-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-role", choices=["real_required"], required=True)
    parser.add_argument("--cutoff-date", help="Exact historical cutoff (YYYY-MM-DD); require observations on this market date and ignore later rows for preparation")
    parser.add_argument("--protocol", help="Explicit approved versioned missing-policy protocol")
    parser.add_argument("--frozen-samples", help="Immutable old sample-key record; required with --protocol")
    parser.add_argument("--diagnostic-checkpoint", help="Only recheck the frozen fifty sources and save gap evidence; do not scan or prepare models")
    args = parser.parse_args(argv)
    inputs = vars(args)
    checkpoint_path = inputs.pop("diagnostic_checkpoint")
    if checkpoint_path:
        if args.protocol is not None or args.frozen_samples is not None:
            parser.error("Checkpoint diagnostics cannot be combined with a new preparation protocol")
        try:
            result = diagnose_checkpoint(checkpoint_path=checkpoint_path, source_root=args.source_root,
                                         calendar_csv=args.calendar_csv,
                                         calendar_evidence_json=args.calendar_evidence_json,
                                         output=args.output, cutoff_date=args.cutoff_date)
        except Exception as error:
            result = {"status": "blocked", "stage": "checkpoint_diagnosis", "error_type": type(error).__name__,
                      "source_writes": 0, "cloud_submissions": 0, "automatic_retry": False}
            context = _private_source_context(error)
            if context is not None:
                result["source_diagnostic"] = context
    else:
        result = prepare_inputs(**inputs)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] in ("prepared", "diagnosed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
