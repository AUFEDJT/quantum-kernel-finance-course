"""Explicit, single-start compact-pilot entry point; disabled unless --execute.

The private manifest binds every deployed runtime source, protocol, source/data
snapshot, reviewed authentication helper, fee evidence, and the existing account
ledger. Credentials arrive only as one JSON object on stdin. There is no resume,
fork, automatic reconciliation, or backend cancellation. Tests inject services
through a Python-only seam; the command line has no mock or gate-bypass option.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

SCHEMA = "compact_pilot_v3_launch_manifest_v1"
AUTHORIZATION = "Explicit compact 50-stock pilot: 5292 circuits, 512 shots, three hours"
CACHE_LIMIT_BYTES = 10 * 1024**3
QLIB_PROTOCOL_ID = "qukernet-50stocks-3h-cutoff20260325-qlib-missing-v1"
REGISTERED_PROTOCOLS = {
    "qukernet-50stocks-3h-20261115-v1": "protocol.json",
    "qukernet-50stocks-3h-cutoff20260325-20261115-v2": "protocol_cutoff_20260325.json",
    QLIB_PROTOCOL_ID: "protocol_cutoff_20260325_qlib_missing_v1.json",
}


class LaunchBlocked(RuntimeError):
    pass


class DataAdmissionFailure(LaunchBlocked):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LaunchBlocked("A frozen SHA256 digest is required")
    return value


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise LaunchBlocked("Manifest paths must be relative to deployment root")
    result = (root / relative).resolve()
    if result == root or root not in result.parents:
        raise LaunchBlocked("Manifest path escapes deployment root")
    return result


def _private(path):
    """Read-only path check: no mkdir or network-share write probe."""
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")):
        raise LaunchBlocked("Network output is forbidden")
    p = Path(path).resolve()
    source = Path(__file__).resolve().parents[1]
    for ancestor in (source, *source.parents):
        if ancestor.name == "experiments":
            source = ancestor.parent
            break
        if ancestor.name == "code" or ancestor.name.startswith("code_"):
            source = ancestor
            break
    if p == source or source in p.parents or p in source.parents:
        raise LaunchBlocked("Manifest and outputs must be outside runtime/course sources")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(p.anchor) == 4:
            raise LaunchBlocked("Mapped network output is forbidden")
    return p


def required_runtime_files():
    """Bind loaded sources, not merely arbitrary files named in a manifest."""
    package = Path(__file__).resolve().parents[1]
    paths = {p.resolve() for name in ("cloud_v2", "cloud_v3")
             for p in (package / name).glob("*.py")}
    # The existing output guard is a direct transitive dependency.
    from cloud_v2 import search  # establishes the reviewed feasibility import path
    from feasibility import safe_io
    paths.add(Path(safe_io.__file__).resolve())
    paths.add(Path(search.modeling.__file__).resolve())
    return paths


def frozen_preflight():
    """Four analytic QCIS inputs and simultaneous 99% binomial checks."""
    from scipy.stats import binom
    from cloud_v2.circuits import specification
    from cloud_v2.ledger import digest

    zero = [{"gate": "RY", "wires": [q], "angle": 0.} for q in range(8)]
    flip = [dict(g) for g in zero]
    flip[2] = {"gate": "RX", "wires": [2], "angle": math.pi}
    bell = [dict(g) for g in zero]
    bell[0] = {"gate": "RY", "wires": [0], "angle": math.pi / 2}
    bell.append({"gate": "CNOT", "wires": [0, 1]})
    angles = [.17 + .09*q for q in range(8)]
    general = [{"gate": "RY", "wires": [q], "angle": a} for q, a in enumerate(angles)]
    general += [{"gate": "CNOT", "wires": [q, q+1]} for q in range(7)]
    p = math.prod(math.cos(a/2)**2 for a in angles)
    checks = []
    for name, gates, prob in (("zero", zero, 1.), ("flip_Q3", flip, 0.),
                               ("bell_Q1_Q2", bell, .5), ("ry_chain", general, p)):
        spec = specification(gates, 8, "compact_pilot_v3/analytic/" + name)
        lo, hi = (int(binom.ppf(t, 512, prob)) for t in (.01/8, 1-.01/8))
        checks.append({"name": name, "spec": spec, "expected_p0": prob,
                       "zero_count_interval_inclusive": [lo, hi]})
    value = {"shots": 512, "familywise_alpha": .01, "interval": "Bonferroni binomial quantiles",
             "checks": checks, "statement": "Statistical anomaly check, not a proof of numerical accuracy"}
    return {**value, "sha256": digest(value)}


def _analytic_validator(frozen, ledger_root):
    from cloud_v2.cloud import validate_raw_result, _decode
    from cloud_v2.ledger import digest

    def validate(results):
        if len(results) != 4:
            raise LaunchBlocked("Four analytic results required")
        for result, check in zip(results, frozen["checks"]):
            if result["id"] != check["spec"]["id"] or result["shots"] != 512:
                raise LaunchBlocked("Analytic result identity mismatch")
            lo, hi = check["zero_count_interval_inclusive"]
            if not lo <= result["zero_count"] <= hi:
                raise LaunchBlocked("Analytic probability outside frozen check interval")
            folder = Path(result["evidence"]["directory"]).resolve()
            if ledger_root.resolve() not in folder.parents:
                raise LaunchBlocked("Raw evidence is outside the account ledger")
            item = None
            for path in sorted(folder.glob("query_*.json"), reverse=True):
                envelope = json.loads(path.read_text(encoding="utf-8"))
                hits = [r for r in envelope.get("data", {}).get("experimentResultModelList", [])
                        if str(r.get("experimentTaskId")) == result["job_id"] and r.get("resultStatus") is not None]
                if hits:
                    if len(hits) != 1:
                        raise LaunchBlocked("Ambiguous raw analytic evidence")
                    item = hits[0]
                    break
            if item is None:
                raise LaunchBlocked("Raw analytic shots unavailable")
            checked = validate_raw_result(item, result["job_id"], check["spec"], 512)
            if any(checked[k] != result[k] for k in ("zero_count", "raw_rows_sha256", "measurement_columns")):
                raise LaunchBlocked("Raw analytic evidence differs from validated result")
            rows = _decode(item["resultStatus"])
            if digest(rows) != result["raw_rows_sha256"]:
                raise LaunchBlocked("Raw analytic checksum mismatch")
            for row in rows[1:]:
                bits = dict(zip(rows[0], row))
                name = check["name"]
                if name == "zero" and any(row):
                    raise LaunchBlocked("Zero-state bit check failed")
                if name == "flip_Q3" and any(bits[q] != int(q == 3) for q in range(1, 9)):
                    raise LaunchBlocked("Single-bit position check failed")
                if name == "bell_Q1_Q2" and (bits[1] != bits[2] or any(bits[q] for q in range(3, 9))):
                    raise LaunchBlocked("Bell bit correlation check failed")
    return validate


def _validate_cutoff(metadata, protocol, source_manifest):
    """Bind an explicitly approved historical cutoff without changing legacy runs."""
    from cloud_v3.data import digest

    data = (protocol or {}).get("data", {})
    cutoff = data.get("cutoff_date")
    if cutoff is None:
        return
    from datetime import date
    try:
        if not isinstance(cutoff, str) or date.fromisoformat(cutoff).isoformat() != cutoff:
            raise ValueError
        fixed = data["fixed_dates"]
        expected_dates = fixed["evaluation_dates"]
        if (len(expected_dates) != 10 or sorted(set(expected_dates)) != expected_dates
                or any(date.fromisoformat(v).isoformat() != v for v in expected_dates)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise LaunchBlocked("Malformed approved cutoff/date specification") from None
    audit, universe = metadata["audit"], metadata["universe"]
    if (audit.get("latest_date") != cutoff or audit["split"].get("D") != cutoff
            or data.get("latest_date") != cutoff
            or universe.get("qualification_anchor") != fixed["qualification_anchor"]
            or audit["split"].get("validation_start") != fixed["validation_signal_start"]
            or audit["split"].get("evaluation_start") != fixed["evaluation_signal_start"]
            or audit.get("outer_dates") != expected_dates
            or sorted({r["date"] for r in metadata["meta_test"]}) != expected_dates):
        raise LaunchBlocked("Frozen data cutoff or approved evaluation dates differ from protocol")
    for split, bounds in (("train", "train"), ("full_train", "train"),
                          ("valid", "validation"), ("test", "evaluation")):
        lo, hi = fixed[bounds + "_signal_start"], fixed[bounds + "_signal_end"]
        if any(not lo <= r["date"] <= hi or not data["start"] <= r["label_start"] <= r["label_end"] <= cutoff
               for r in metadata["meta_" + split]):
            raise LaunchBlocked("Sample or label lies outside the approved historical interval")
    if (not isinstance(source_manifest, dict) or digest(source_manifest) != audit.get("source_manifest_sha256")
            or source_manifest.get("requested_cutoff_date") != cutoff
            or source_manifest.get("latest_date") != cutoff):
        raise LaunchBlocked("Cutoff source manifest missing or differs from prepared data")
    used = source_manifest.get("used_interval", {})
    records = source_manifest.get("securities", [])
    if (used.get("start") != data["start"] or used.get("end") != cutoff
            or not records or len({r.get("symbol") for r in records}) != len(records)
            or not set(universe["selected_symbols"]).issubset({r.get("symbol") for r in records})):
        raise LaunchBlocked("Frozen source interval or stock coverage differs from protocol")
    for record in records:
        interval = record.get("used_interval", {})
        if (interval.get("start") != data["start"] or interval.get("end") != cutoff
                or type(interval.get("rows")) is not int or interval["rows"] <= 0
                or not data["start"] <= record.get("first_observed_date", "") <= record.get("last_observed_date", "") <= cutoff
                or any(not data["start"] <= d <= cutoff for d in record.get("observed_valid_dates", []))):
            raise LaunchBlocked("A source security includes an inconsistent historical interval")
        _hash(interval.get("sha256"))
    aggregate = hashlib.sha256(json.dumps(
        [{"symbol": r["symbol"], "used_interval": r["used_interval"]} for r in records],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (used.get("rows") != sum(r["used_interval"]["rows"] for r in records)
            or used.get("sha256") != aggregate
            or max(r["last_observed_date"] for r in records) != cutoff):
        raise LaunchBlocked("Source used-interval digest or observed cutoff does not match")


def _validate_missing_policy(window, metadata, protocol, frozen_samples):
    """Require the fitted training-only policy and unchanged frozen 16/16 keys."""
    if (protocol or {}).get("protocol_id") != QLIB_PROTOCOL_ID:
        return
    from cloud_v3.data import digest
    from cloud_v2.features80 import FEATURES
    audit = metadata["audit"]
    if (metadata.get("schema_version") != "cloud_v3_qlib_missing_data_v1"
            or metadata.get("protocol_id") != QLIB_PROTOCOL_ID
            or audit.get("protocol_id") != QLIB_PROTOCOL_ID
            or metadata.get("missing_policy") != protocol["data"].get("missing_policy")
            or not isinstance(metadata.get("missing_policy"), dict)
            or audit.get("history_span_minimum") != 121
            or audit.get("row_eligibility") != "current_valid_bar_and_121_market_session_span_since_first_valid_bar"):
        raise LaunchBlocked("Fitted missing-data policy or calendar-span audit differs from protocol")
    imputer = metadata.get("preprocessing", {}).get("imputer", {})
    n = len(window["y_full_train"])
    if (imputer.get("method") != "training_feature_median" or imputer.get("fit_split") != "full_train"
            or imputer.get("fit_rows") != n or imputer.get("feature_names") != list(FEATURES)):
        raise LaunchBlocked("Feature imputer was not fitted on the frozen full-training feature table")
    medians, finite, missing = (imputer.get(k) for k in ("medians", "finite_counts", "missing_counts"))
    if any(not isinstance(v, list) or len(v) != 80 for v in (medians, finite, missing)):
        raise LaunchBlocked("Incomplete fitted imputer statistics")
    for i, (value, observed, absent) in enumerate(zip(medians, finite, missing)):
        if (type(observed) is not int or type(absent) is not int or min(observed, absent) < 0
                or observed + absent != n or (observed == 0 and value is not None)
                or (observed > 0 and (type(value) not in (int, float) or not math.isfinite(value)))):
            raise LaunchBlocked("Invalid training-only feature imputation statistics")
    selected = metadata["preprocessing"].get("selected_features", [])
    if len(selected) != 8 or len(set(selected)) != 8 or any(s not in FEATURES or finite[list(FEATURES).index(s)] == 0 for s in selected):
        raise LaunchBlocked("Selected features do not have observed training support")
    import numpy as np
    indices = metadata["preprocessing"].get("selected_indices")
    if indices != [list(FEATURES).index(s) for s in selected]:
        raise LaunchBlocked("Selected feature names and source-column indices differ")
    for phase in ("train", "valid", "test", "full_train"):
        count = len(window["y_" + phase])
        all_mask = np.asarray(window.get("missing_mask80_" + phase))
        mask = np.asarray(window.get("missing_mask_" + phase))
        support = np.asarray(window.get("support_counts_" + phase))
        if (all_mask.dtype != np.dtype(bool) or all_mask.shape != (count, 80)
                or mask.dtype != np.dtype(bool) or mask.shape != (count, 8)
                or not np.array_equal(mask, all_mask[:, indices])
                or support.shape != (count, 80) or not np.issubdtype(support.dtype, np.integer)
                or np.any(support < 0) or np.any(support > 120)):
            raise LaunchBlocked("Missing-feature masks or diagnostic support counts are inconsistent")
        note = audit.get("missingness", {}).get(phase, {})
        if (note.get("rows") != count or note.get("missing_cells_80") != int(all_mask.sum())
                or note.get("selected_missing_cells") != int(mask.sum())
                or note.get("rows_with_any_missing_80") != int(all_mask.any(axis=1).sum())
                or note.get("raw_features_missing_by_column") != all_mask.sum(axis=0).tolist()):
            raise LaunchBlocked("Saved missingness audit differs from the frozen snapshot arrays")
        if phase == "full_train" and (missing != all_mask.sum(axis=0).tolist()
                or finite != (~all_mask).sum(axis=0).tolist()):
            raise LaunchBlocked("Fitted imputer counts differ from the full-training missingness mask")
    keys = {split: metadata["meta_" + split] for split in ("train", "valid")}
    stored = metadata.get("frozen_samples", {})
    key_hash = digest(keys)
    if (stored.get("train") != keys["train"] or stored.get("valid") != keys["valid"]
            or stored.get("sha256") != key_hash or audit.get("frozen_samples_sha256") != key_hash):
        raise LaunchBlocked("Training/validation keys differ from the frozen 32-sample identity")
    if not isinstance(frozen_samples, dict):
        raise LaunchBlocked("Private frozen 32-sample record is required")
    record = dict(frozen_samples)
    record_hash = record.pop("sha256", None)
    if (record_hash != digest(record) or record_hash != audit.get("frozen_sample_record_sha256")
            or record.get("train") != keys["train"] or record.get("valid") != keys["valid"]):
        raise LaunchBlocked("External frozen sample record differs from the model snapshot")


def _validate_data(window, protocol=None, source_manifest=None, frozen_samples=None):
    from cloud_v3.study import _admit, feature_quality
    try:
        arrays, admitted_metadata, _ = _admit(window)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise DataAdmissionFailure("Frozen data failed study admission: " + type(exc).__name__) from None
    quality = feature_quality(arrays, admitted_metadata)
    if quality["splits"]["full_train"]["varying_columns"] != 8 or quality["zero_iqr_columns"]:
        raise DataAdmissionFailure("Fewer than eight nondegenerate full-training features; no replacement")
    metadata = window.get("metadata", window)
    rows = metadata["meta_test"]
    if len({r["symbol"] for r in rows}) != 50 or sorted(Counter(r["date"] for r in rows).values()) != [5]*10:
        raise LaunchBlocked("Fifty distinct stocks across ten five-stock dates required")
    audit = metadata["audit"]
    if audit.get("boundaries_verified") is not True:
        raise LaunchBlocked("Data boundary audit missing")
    split = audit["split"]
    v0, o0, last = split["validation_start"], split["evaluation_start"], audit["latest_date"]
    for name, boundary in (("train", v0), ("full_train", v0), ("valid", o0)):
        if any(r["label_end"] >= boundary for r in metadata["meta_" + name]):
            raise LaunchBlocked("Label end crosses a split boundary")
    if any(r["label_end"] > last or r["date"] < o0 for r in rows):
        raise LaunchBlocked("Outer labels are not mature in the evaluation window")
    _validate_cutoff(metadata, protocol, source_manifest)
    _validate_missing_policy(window, metadata, protocol, frozen_samples)
    return quality


@dataclass
class Bundle:
    root: Path
    manifest_path: Path
    manifest: dict
    manifest_hash: str
    files: dict
    window: dict
    evidence: dict
    frozen: dict
    ledger: object
    output: Path


def _cache_bytes(bundle):
    """Count this run's data/output/new attempt evidence, not old account logs."""
    roots = [bundle.output, _inside(bundle.root, bundle.manifest["data_npz"]),
             _inside(bundle.root, bundle.manifest["data_npz"]).with_suffix(".json")]
    roots += [bundle.ledger.root / e["attempt_id"] for e in bundle.ledger.snapshot()["entries"]
              if e.get("execution_window_id") == bundle.manifest["window_id"]]
    total, seen = 0, set()
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*") if root.exists() else []
        for path in candidates:
            if path.is_symlink():
                raise LaunchBlocked("Run cache must not follow filesystem links")
            if path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                total += path.stat().st_size
    return total


def _check_cache(bundle):
    size = _cache_bytes(bundle)
    if size >= CACHE_LIMIT_BYTES:
        raise LaunchBlocked("Private pilot cache reached the 10 GiB stop threshold")
    return size


def _validate_account_grant(grant, ui, account_hash, ui_hash, now):
    """Check a current official UI grant's scope/validity, never label it a live API read."""
    from datetime import datetime
    if (grant.get("schema") != "compact_account_ui_grant_v1"
            or grant.get("account_scope_sha256") != account_hash or grant.get("ui_sha256") != ui_hash
            or grant.get("evidence_mode") != "official_ui_snapshot_validity_check"
            or grant.get("live_account_api") is not False):
        raise LaunchBlocked("Account grant identity or evidence mode differs from the frozen authority")
    if (ui.get("source") != "authenticated_official_account_dashboard"
            or ui.get("source_url") != "https://qc.zdxlz.com/laboratory/#/dashboard?lang=zh"
            or ui.get("backend") != "tianyan_sw" or ui.get("type") != "simulator"
            or ui.get("grant") != "free_gift" or ui.get("remaining_time") != "unlimited"
            or type(ui.get("account_allowed_qubits")) is not int or ui["account_allowed_qubits"] < 8
            or ui.get("raw_credentials_recorded") is not False):
        raise LaunchBlocked("Official account grant does not cover the authorized free eight-qubit simulator task")
    try:
        stamps = [datetime.fromisoformat(ui[k]) for k in ("observed_at","grant_start","grant_expiry")]
        if any(v.tzinfo is None for v in stamps):raise ValueError
        observed, start, expiry = [v.timestamp() for v in stamps]
        if not start <= observed <= now < expiry:raise ValueError
    except (KeyError, TypeError, ValueError):
        raise LaunchBlocked("Official account grant is not currently within its observed validity interval") from None
    return {"evidence_mode":"official_ui_snapshot_validity_check","live_account_api":False,
        "account_scope_sha256":account_hash,"ui_sha256":ui_hash,"ui_observed_at":ui["observed_at"],
        "checked_epoch":now,"grant_expiry":ui["grant_expiry"],"backend":"tianyan_sw",
        "account_allowed_qubits":ui["account_allowed_qubits"],"requested_qubits":8,
        "remaining_seconds":"not_applicable_unlimited","cash_balance":"unknown",
        "limitation":"Grant snapshot validity is checked per batch; no fresh account permission API response"}


def _account_grant_check(bundle):
    if "current_account_grant" not in bundle.manifest:
        return None  # Legacy protocols retain their existing explicit evidence gate.
    paths = {key:_private(_inside(bundle.root,bundle.manifest[key])) for key in
             ("current_account_grant","current_account_ui")}
    if any(p not in bundle.files or sha(p) != bundle.files[p] for p in paths.values()):
        raise LaunchBlocked("Current account grant/UI source missing or changed")
    return _validate_account_grant(json.loads(paths["current_account_grant"].read_text(encoding="utf-8")),
        json.loads(paths["current_account_ui"].read_text(encoding="utf-8")),bundle.ledger.account_hash,
        bundle.files[paths["current_account_ui"]],bundle.ledger.clock())


class _CacheBoundProvider:
    """Stop before the next reservation when the private cache cap is reached."""
    def __init__(self, provider, bundle):
        self.provider, self.bundle = provider, bundle

    def __getattr__(self, name):
        return getattr(self.provider, name)

    @property
    def result_validator(self):
        return self.provider.result_validator

    @result_validator.setter
    def result_validator(self, value):
        self.provider.result_validator = value

    def probability(self, *args, **kwargs):
        _check_cache(self.bundle)
        grant = _account_grant_check(self.bundle)
        if grant is not None:
            # Bind each reservation to the checked proof without changing circuit IDs or quota.
            if len(args) > 3:
                positional=list(args);context=dict(positional[3] or {})
                context["account_grant_check"]=grant;positional[3]=context;args=tuple(positional)
            else:
                kwargs=dict(kwargs);context=dict(kwargs.get("context") or {})
                context["account_grant_check"]=grant;kwargs["context"]=context
        result = self.provider.probability(*args, **kwargs)
        _check_cache(self.bundle)
        return result


def verify_manifest(root, manifest, manifest_sha256, *, _runtime_files=None, _clock=time.time, _mock=False):
    """Read/check all required private inputs before any authentication or intent."""
    from cloud_v3.ledger import PersistentLedger, TERMINAL
    from cloud_v2.preflight import validate_evidence
    from cloud_v3.data import load_window

    root = Path(root).resolve()
    manifest_path = _private(_inside(root, manifest))
    if sha(manifest_path) != _hash(manifest_sha256):
        raise LaunchBlocked("Manifest digest mismatch")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise LaunchBlocked("Launch manifest schema mismatch")
    files = {}
    for name, fingerprint in value["files"].items():
        p = _inside(root, name)
        if p in files or not p.is_file() or sha(p) != _hash(fingerprint):
            raise LaunchBlocked("Frozen file missing, aliased, or changed")
        files[p] = fingerprint
    required = required_runtime_files() if _runtime_files is None else set(_runtime_files)
    if not required.issubset(files):
        raise LaunchBlocked("Loaded runtime has sources absent from the frozen manifest")
    paths = {key: _inside(root, value[key]) for key in
             ("data_npz", "protocol", "auth_helper", "fee_evidence")}
    for p in [*paths.values(), paths["data_npz"].with_suffix(".json")]:
        if p not in files:
            raise LaunchBlocked("Required input is not hash-bound")
    for key in ("data_npz", "auth_helper", "fee_evidence"):
        _private(paths[key])
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    cutoff = protocol.get("data", {}).get("cutoff_date")
    if cutoff is not None and (not isinstance(cutoff, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff)):
        raise LaunchBlocked("Malformed cutoff-specific protocol identifier")
    registered = REGISTERED_PROTOCOLS.get(protocol.get("protocol_id"))
    if registered is None:
        raise LaunchBlocked("Protocol version is not explicitly registered")
    current_protocol = Path(__file__).with_name(registered)
    if not current_protocol.is_file() or sha(paths["protocol"]) != sha(current_protocol) or protocol.get("schema") != "compact_pilot_v3":
        raise LaunchBlocked("Protocol is not the current frozen compact protocol")
    frozen = frozen_preflight()
    if value.get("preflight_sha256") != frozen["sha256"]:
        raise LaunchBlocked("Analytic specification/acceptance rules changed")
    output = _private(_inside(root, value["output"]))
    if output.exists() and any(output.iterdir()):
        raise LaunchBlocked("Prior run output exists; automatic restart is forbidden")
    scope = value["account_scope"]
    if not isinstance(scope, str) or not scope.strip():
        raise LaunchBlocked("Stable account scope required")
    ledger_base = _private(Path(value["account_ledger_root"]))
    if not Path(value["account_ledger_root"]).is_absolute():
        raise LaunchBlocked("Canonical account ledger root must be absolute")
    account = hashlib.sha256(scope.encode()).hexdigest()
    ledger_path = ledger_base / ("account-" + account) / "ledger.json"
    if not ledger_path.is_file() or sha(ledger_path) != _hash(value["ledger_sha256_before"]):
        raise LaunchBlocked("Existing canonical account ledger missing or changed")
    window = load_window(paths["data_npz"])
    sidecar = json.loads(paths["data_npz"].with_suffix(".json").read_text(encoding="utf-8"))
    frozen_samples = None
    if protocol.get("protocol_id") == QLIB_PROTOCOL_ID:
        frozen_path = _private(_inside(root, value.get("frozen_samples")))
        if frozen_path not in files:
            raise LaunchBlocked("Frozen sample record is not hash-bound")
        frozen_samples = json.loads(frozen_path.read_text(encoding="utf-8"))
    _validate_data(window, protocol, sidecar.get("source_manifest"), frozen_samples)
    # Constructor must find an existing ledger; no fresh account budget is allowed.
    ledger = PersistentLedger(ledger_base, scope, clock=_clock)
    history = ledger.snapshot()
    if any(e["state"] not in TERMINAL for e in history["entries"]) or any(w["state"] != "closed" for w in history.get("execution_windows", [])):
        raise LaunchBlocked("Historical unknown tasks or open windows block launch")
    wid = value["window_id"]
    if not isinstance(wid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", wid) or any(w["window_id"] == wid for w in history.get("execution_windows", [])):
        raise LaunchBlocked("New unique execution window ID required")
    if protocol.get("protocol_id") == QLIB_PROTOCOL_ID and not wid.startswith(QLIB_PROTOCOL_ID + "-"):
        raise LaunchBlocked("New Qlib protocol requires its own explicitly versioned execution window")
    evidence = json.loads(paths["fee_evidence"].read_text(encoding="utf-8"))
    validate_evidence(evidence, account, now=_clock(), mock=_mock,
                      ledger_path_hash=hashlib.sha256(str(ledger.path.resolve()).encode()).hexdigest())
    bundle = Bundle(root, manifest_path, value, manifest_sha256, files, window, evidence, frozen, ledger, output)
    if protocol.get("protocol_id") == QLIB_PROTOCOL_ID and not all(k in value for k in ("current_account_grant","current_account_ui")):
        raise LaunchBlocked("Current official account UI grant must be separately hash-bound")
    _account_grant_check(bundle)
    _check_cache(bundle)
    return bundle


def _load_authentication(path):
    spec = importlib.util.spec_from_file_location("compact_reviewed_authentication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "authenticate", None)):
        raise LaunchBlocked("Reviewed authentication helper lacks authenticate")
    return module


def _resource_limits():
    if os.name != "posix" or not hasattr(os, "sched_getaffinity"):
        raise LaunchBlocked("Live execution requires the reviewed Linux host")
    import resource
    limit = 32 * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])


@dataclass
class _Services:
    authenticate: object
    adapter: object
    study: object
    deadline_guard: object
    monotonic: object = time.monotonic


def _production_services(bundle):
    from cloud_v2.cloud import Cqlib1311Transport, _hard_request_deadline
    from cloud_v3.ledger import TianyanCloudAdapter
    from cloud_v3.study import run_study
    helper = _load_authentication(_inside(bundle.root, bundle.manifest["auth_helper"]))
    def adapter(platform, validator):
        return TianyanCloudAdapter(Cqlib1311Transport(platform), bundle.ledger, bundle.evidence,
            execution_window_id=bundle.manifest["window_id"], result_validator=validator,
            execution_protocol_sha256=bundle.files[_inside(bundle.root, bundle.manifest["protocol"])])
    return _Services(helper.authenticate, adapter, run_study, _hard_request_deadline)


def _run_verified(bundle, credential, services):
    """Single process only. Cleanup never polls or submits, even after SIGINT."""
    from cloud_v3.ledger import atomic_json, MAIN_POLICY, TERMINAL
    ledger, value, output = bundle.ledger, bundle.manifest, bundle.output
    if not isinstance(credential, str) or not credential.strip():
        raise LaunchBlocked("Credential must be provided in memory")
    if sha(bundle.manifest_path) != bundle.manifest_hash or any(sha(p) != h for p, h in bundle.files.items()):
        raise LaunchBlocked("Frozen inputs changed after verification")
    if sha(ledger.path) != value["ledger_sha256_before"]:
        raise LaunchBlocked("Account ledger changed after verification")
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "launch_intent.json"
    with intent.open("x", encoding="utf-8") as stream:
        json.dump({"manifest_sha256": bundle.manifest_hash, "window_id": value["window_id"],
                   "automatic_restart": False, "state": "intent_persisted"}, stream)
        stream.flush(); os.fsync(stream.fileno())
    atomic_json(output / "analytic_preflight_frozen.json", bundle.frozen)
    status = {"state": "starting", "window_id": value["window_id"],
              "manifest_sha256": bundle.manifest_hash, "backend_cancelled": False,
              "automatic_restart": False, "automatic_query_after_stop": False}
    authorized = False
    error = None
    try:
        platform = services.authenticate(credential, services.monotonic() + 30)
        protocol_id = json.loads(_inside(bundle.root, value["protocol"]).read_text(encoding="utf-8"))["protocol_id"]
        ledger.authorize_main_window(value["window_id"], bundle.files[_inside(bundle.root, value["protocol"])],
                                     AUTHORIZATION + "; protocol=" + protocol_id)
        authorized = True
        provider = _CacheBoundProvider(services.adapter(platform, _analytic_validator(bundle.frozen, ledger.root)), bundle)
        probe = provider.probability([c["spec"] for c in bundle.frozen["checks"]], 512, phase="main",
            context={"phase": "main", "operation_id": "compact_pilot_v3/" +
                     protocol_id + "/analytic_preflight",
                     "purpose": "four analytic probes", "preflight_sha256": bundle.frozen["sha256"]})
        atomic_json(output / "analytic_preflight_results.json", probe)
        deadline_epoch = ledger.effective_deadline(value["window_id"])
        if deadline_epoch is None or deadline_epoch <= ledger.clock():
            raise TimeoutError("Global compact window expired")
        # The initial cloud batch has its own <=300s persisted bound. From here
        # the remaining absolute budget also covers classical baselines and I/O.
        provider.result_validator = None
        with services.deadline_guard(services.monotonic() + deadline_epoch - ledger.clock()):
            result = services.study(bundle.window, output / "study", provider,
                                    execution_identity=provider.execution_identity())
            entries = [e for e in ledger.snapshot()["entries"] if e.get("execution_window_id") == value["window_id"]]
            counts = {"circuits": sum(e["circuits"] for e in entries), "shots": sum(e["requested_shots"] for e in entries), "batches": len(entries)}
            if (result.get("workflow_status") != "complete" or counts != {"circuits": MAIN_POLICY["circuits"], "shots": MAIN_POLICY["shots"], "batches": MAIN_POLICY["max_attempts"]}
                    or any(e["state"] != "validated" for e in entries) or ledger.clock() >= deadline_epoch):
                raise LaunchBlocked("Complete workflow does not match the frozen quota/deadline")
            status.update(state="completed", counts=counts)
    except BaseException as exc:
        error = exc
        status.update(state="client_stopped", error_type=type(exc).__name__)
    finally:
        # No external exception strings or platform/auth objects are persisted.
        if authorized:
            for entry in ledger.snapshot()["entries"]:
                if entry.get("execution_window_id") != value["window_id"]:
                    continue
                target = {"submission_pending": "submission_unknown", "submitted": "observation_unknown"}.get(entry["state"])
                if target:
                    ledger.update(entry["attempt_id"], target, client_cleanup=True)
            ledger.close_main_window(value["window_id"], "completed" if status["state"] == "completed" else "client_stopped",
                "Client completed" if status["state"] == "completed" else "Client stopped; background tasks not cancelled; no automatic queries")
            status["window"] = ledger.window(value["window_id"])
        status["unresolved_attempt_ids"] = [e["attempt_id"] for e in ledger.snapshot()["entries"] if e["state"] not in TERMINAL]
        atomic_json(output / "ledger_after.json", ledger.snapshot())
        atomic_json(output / "runner_status.json", status)
    if error is not None:
        raise LaunchBlocked("Execution stopped; inspect private runner_status.json") from None
    return status


def main(argv=None, *, stdin=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "disabled", "submitted_circuits": 0, "required": "--execute plus frozen private manifest; credential only on stdin"}))
        return 0
    if not all((args.root, args.manifest, args.manifest_sha256)):
        parser.error("--execute requires --root, --manifest, and --manifest-sha256")
    # Set thread pools before lazily importing numerical packages.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    try:
        _resource_limits()
        bundle = verify_manifest(args.root, args.manifest, args.manifest_sha256)
        request = json.loads((sys.stdin if stdin is None else stdin).read(16385))
        if set(request) != {"credential"}:
            raise LaunchBlocked("stdin must contain only a credential object")
        credential = request.pop("credential")
        result = _run_verified(bundle, credential, _production_services(bundle))
        print(json.dumps({"state": result["state"], "backend_cancelled": False}))
        return 0
    except BaseException as exc:
        print(json.dumps({"state": "blocked_or_stopped", "error_type": type(exc).__name__, "automatic_restart": False, "backend_cancelled": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
