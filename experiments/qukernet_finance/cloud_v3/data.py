"""Frozen latest-window preparation for the 50-stock cloud pilot.

This module performs classical data preparation only. It never accesses a
cloud API. Source access is explicit in sources.py; private outputs cannot be
written inside the public course or a source directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..cloud_v2.features80 import FEATURES, FEATURE_DICTIONARY, build_features
    from ..cloud_v2.data80 import balanced_sample, fit_scaler, apply_scaler
except ImportError:
    from cloud_v2.features80 import FEATURES, FEATURE_DICTIONARY, build_features
    from cloud_v2.data80 import balanced_sample, fit_scaler, apply_scaler
from .sources import checked_calendar, validate_output_path, REQUIRED

SEED = 20261115


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def calendar_split(calendar, latest_date):
    dates = checked_calendar(calendar)
    end = pd.Timestamp(latest_date)
    if end != end.normalize() or end not in dates:
        raise ValueError("D must be an observed normalized market date")
    dates = dates[(dates >= pd.Timestamp("2019-01-01")) & (dates <= end)]
    d = len(dates) - 1
    m = d - 6
    v0, o0 = m - 125, m - 62
    if v0 < 128:
        raise ValueError("Insufficient history for qualification, features and purged splits")
    return dates, {"D": end.strftime("%Y-%m-%d"), "d": d, "last_mature_j": m,
                   "v_start_j": v0, "o_start_j": o0,
                   "validation_start": dates[v0].strftime("%Y-%m-%d"),
                   "evaluation_start": dates[o0].strftime("%Y-%m-%d"),
                   "train_days": list(range(v0 - 6)),
                   "valid_days": list(range(v0, o0 - 6)),
                   "outer_days": list(range(o0, m + 1)),
                   "outer_selected_days": [o0 + ((2 * r + 1) * 63) // 20 for r in range(10)],
                   "boundary_rule": "T label end < V start; V label end < O start; O label end <= D"}


def select_universe(universe_metadata, calendar, *, latest_date, seed=SEED):
    dates, split = calendar_split(calendar, latest_date)
    if isinstance(universe_metadata, dict):
        if universe_metadata.get("latest_date") != split["D"]:
            raise ValueError("D disagrees with frozen source inventory")
        records = universe_metadata["securities"]
    else:
        records = list(universe_metadata)
    if len({r["symbol"] for r in records}) != len(records):
        raise ValueError("Duplicate universe metadata security")
    anchor = dates[split["v_start_j"] - 1]
    eligible, qualification = [], []
    for row in sorted(records, key=lambda r: r["symbol"]):
        symbol = str(row["symbol"])
        exchange = str(row["exchange"])
        if exchange not in ("sh", "sz", "bj") or not symbol.startswith(exchange):
            raise ValueError("Invalid universe exchange")
        valid = pd.DatetimeIndex(pd.to_datetime(row["observed_valid_dates"], errors="raise"))
        if valid.hasnans or valid.has_duplicates or valid.tz is not None:
            raise ValueError("Invalid universe observed-date metadata")
        before = valid[(valid < dates[split["v_start_j"]]) & (valid >= dates[0])]
        if not before.isin(dates).all():
            raise ValueError("Qualification dates absent from calendar")
        admitted = len(before) >= 120 and anchor in before
        qualification.append({"symbol": symbol, "exchange": exchange,
                              "valid_days_before_V": len(before),
                              "valid_bar_on_pre_V_anchor": anchor in before, "eligible": bool(admitted)})
        if admitted:
            eligible.append((symbol, exchange))
    if len(eligible) < 50:
        raise ValueError("Fewer than 50 securities qualify without future screening")
    groups = {e: sorted(s for s, x in eligible if x == e) for e in ("bj", "sh", "sz")}
    numerator = {e: 50 * len(v) for e, v in groups.items()}
    allocation = {e: numerator[e] // len(eligible) for e in groups}
    remaining = 50 - sum(allocation.values())
    remainder_order = sorted(groups, key=lambda e: (-(numerator[e] % len(eligible)), e))
    for exchange in remainder_order[:remaining]:
        allocation[exchange] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for exchange in sorted(groups):
        if allocation[exchange]:
            chosen.extend(rng.choice(groups[exchange], allocation[exchange], replace=False).tolist())
    chosen = sorted(chosen)
    if len(chosen) != 50 or len(set(chosen)) != 50:
        raise RuntimeError("Universe allocation failure")
    assignment_rng = np.random.default_rng(np.random.SeedSequence([seed, 503]))
    permutation = assignment_rng.permutation(chosen).tolist()
    assignments = [{"symbol": symbol, "j": int(j), "date": dates[j].strftime("%Y-%m-%d")}
                   for r, j in enumerate(split["outer_selected_days"])
                   for symbol in permutation[5 * r:5 * r + 5]]
    frozen = {"seed": int(seed), "selected_symbols": chosen, "allocation": allocation,
              "eligible_exchange_counts": {e: len(v) for e, v in groups.items()},
              "qualification_anchor": anchor.strftime("%Y-%m-%d"),
              "qualification": qualification, "O_assignments": assignments,
              "scope": "fixed source-visible universe qualified before V; no future completeness screen",
              "qualifying_history": "at least 120 valid observed days since 2019 and a valid bar at pre-V anchor",
              "feature_history": "separate requirement: 121 consecutive valid market bars per feature row",
              "remainder_tie": "exchange code alphabetical", "assignment_seed_stream": 503}
    frozen["sha256"] = digest(frozen)
    return chosen, frozen, dates, split


def _mi(a, b, na, nb):
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)
    joint /= len(a)
    independent = joint.sum(1)[:, None] * joint.sum(0)[None, :]
    use = joint > 0
    return float(np.sum(joint[use] * np.log(joint[use] / independent[use])))


def full_training_mrmr(x, y, count=8, *, feature_names=None):
    """Same 10-quantile empirical-MI objective, lazy redundancy evaluation.

    All fitted quantities use the eligible full T pool; only the selected
    columns need pairwise redundancy, avoiding an unnecessary full 80² table.
    """
    x, y = np.asarray(x, float), np.asarray(y, int)
    names = list(FEATURES if feature_names is None else feature_names)
    if (x.shape != (len(y), len(names)) or len(names) != len(set(names))
            or not len(y) or not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}):
        raise ValueError("Invalid full-T feature-selection input")
    kept, removed = [], []
    for j, name in enumerate(names):
        reason = "constant_on_full_T" if np.ptp(x[:, j]) == 0 else None
        duplicate = next((k for k in kept if np.array_equal(x[:, j], x[:, k])), None)
        if reason or duplicate is not None:
            removed.append({"feature": name, "reason": reason or "exact_duplicate_on_full_T"})
        else:
            kept.append(j)
    if len(kept) < count:
        raise ValueError("Fewer than eight nonconstant nonduplicate full-T features")
    discrete, edges, sizes = {}, {}, {}
    for j in kept:
        cuts = np.unique(np.quantile(x[:, j], np.arange(1, 10) / 10))
        cuts = cuts[(cuts > x[:, j].min()) & (cuts < x[:, j].max())]
        edges[names[j]] = cuts.tolist()
        discrete[j] = np.searchsorted(cuts, x[:, j], side="right")
        sizes[j] = len(cuts) + 1
    relevance = {j: _mi(discrete[j], y, sizes[j], 2) for j in kept}
    selected, steps, cache = [], [], {}
    for _ in range(count):
        scores = {}
        for j in kept:
            if j in selected:
                continue
            redundancy = []
            for k in selected:
                pair = tuple(sorted((j, k)))
                if pair not in cache:
                    cache[pair] = _mi(discrete[j], discrete[k], sizes[j], sizes[k])
                redundancy.append(cache[pair])
            scores[j] = relevance[j] - (float(np.mean(redundancy)) if redundancy else 0.0)
        highest = max(scores.values())
        winner = next(j for j in kept if j in scores and scores[j] >= highest - 1e-12)
        selected.append(winner)
        steps.append({"feature": names[winner], "score_nats": scores[winner]})
    return selected, {"method": "10-quantile empirical MI relevance minus mean redundancy",
                      "fit_rows": len(y), "fit_split": "all eligible matured T records",
                      "removed": removed, "feature_bin_edges": edges, "steps": steps,
                      "tie_rule": "declared feature order within 1e-12 nats"}


def _meta(rows, calendar):
    return [{"symbol": str(row.symbol), "date": calendar[int(row.j)].strftime("%Y-%m-%d"),
             "label_start": calendar[int(row.j) + 1].strftime("%Y-%m-%d"),
             "label_end": calendar[int(row.j) + 6].strftime("%Y-%m-%d")}
            for row in rows.itertuples()]


def _balanced_with_audit(pool, days, dates, seed, phase):
    try:
        return balanced_sample(pool, np.asarray(days), 16, seed)
    except ValueError as error:
        bins = []
        for number, box in enumerate(np.array_split(np.asarray(days), 8)):
            counts = pool[pool.j.isin(box)].groupby("j").size()
            bins.append({"bin": number, "first_date": dates[box[0]].strftime("%Y-%m-%d"),
                         "last_date": dates[box[-1]].strftime("%Y-%m-%d"),
                         "dates_with_two_feature_valid_securities": int((counts >= 2).sum())})
        error.preparation_audit = {"code": "fixed_sample_unavailable", "phase": phase,
                                   "feature_valid_pool_rows": len(pool), "date_bins": bins}
        raise


def prepare_window(panel, calendar, universe_metadata, *, latest_date, seed=SEED,
                   protocol=None, frozen_samples=None):
    if protocol is not None or frozen_samples is not None:
        return prepare_missing_window(panel, calendar, universe_metadata, latest_date=latest_date,
                                      seed=seed, protocol=protocol, frozen_samples=frozen_samples)
    calendar_provenance = {"status": "constructed_or_in_memory_input", "scope": "caller-supplied; not an assertion of real-data provenance"}
    if isinstance(universe_metadata, dict):
        calendar_provenance = universe_metadata.get("calendar_provenance", {})
        accepted = ("verified_market_calendar", "explicitly_accepted_diagnostic_proxy")
        if calendar_provenance.get("status") not in accepted or not calendar_provenance.get("evidence"):
            raise ValueError("Market-calendar quality gate is not satisfied")
        if (calendar_provenance["status"] == "explicitly_accepted_diagnostic_proxy"
                and not calendar_provenance.get("acceptance_note")):
            raise ValueError("Diagnostic calendar proxy requires explicit acceptance note")
    chosen, universe, dates, split = select_universe(universe_metadata, calendar, latest_date=latest_date, seed=seed)
    raw = panel.copy(deep=True)
    if not REQUIRED.issubset(raw.columns):
        raise ValueError("Missing panel columns")
    raw["date"] = pd.to_datetime(raw.date, errors="raise")
    if (raw.duplicated(["symbol", "date"]).any() or raw.date.isna().any()
            or raw.date.dt.tz is not None or not raw.date.eq(raw.date.dt.normalize()).all()
            or raw.symbol.isna().any()):
        raise ValueError("Duplicate or missing panel key")
    raw = raw[raw.symbol.isin(chosen) & raw.date.between(dates[0], dates[-1])].copy()
    if set(raw.symbol) != set(chosen) or not raw.date.isin(dates).all():
        raise ValueError("Selected panel securities or market dates mismatch")
    frame = build_features(raw, dates)
    t_pool = frame[frame.j.isin(split["train_days"]) & frame.feature_valid].copy()
    v_pool = frame[frame.j.isin(split["valid_days"]) & frame.feature_valid].copy()
    train, t_trace = _balanced_with_audit(t_pool, split["train_days"], dates, seed, "train")
    valid, v_trace = _balanced_with_audit(v_pool, split["valid_days"], dates, seed + 1, "valid")
    target = (train.y_raw.to_numpy() > 0).astype(np.int64)
    if np.bincount(target, minlength=2).min() < 2:
        error = ValueError("Kernel training has fewer than two rows in a class; no resampling")
        error.preparation_audit = {"code": "training_minority_class_too_small", "phase": "train",
                                   "class_counts": np.bincount(target, minlength=2).tolist()}
        raise error
    indexed = frame.set_index(["symbol", "j"], drop=False)
    outer = indexed.loc[[(r["symbol"], r["j"]) for r in universe["O_assignments"]]].copy()
    outer = outer.reset_index(drop=True).sort_values(["j", "symbol"])
    if len(outer) != 50 or outer.symbol.nunique() != 50 or not outer.feature_valid.all() or not outer.target_valid.all():
        error = ValueError("Frozen 50-stock O coverage failed; no replacement or date reassignment")
        error.preparation_audit = {"code": "frozen_outer_coverage_invalid", "phase": "test",
                                   "rows": len(outer), "distinct_stocks": outer.symbol.nunique(),
                                   "feature_invalid_rows": int((~outer.feature_valid).sum()),
                                   "target_invalid_rows": int((~outer.target_valid).sum())}
        raise error
    full_train = t_pool[t_pool.target_valid].sort_values(["j", "symbol"]).copy()
    full_y = (full_train.y_raw.to_numpy() > 0).astype(np.int64)
    selected, mrmr = full_training_mrmr(full_train[FEATURES].to_numpy(), full_y)
    names = [FEATURES[j] for j in selected]
    scaler = fit_scaler(full_train[names].to_numpy())
    preprocessing = {"selected_features": names, "selected_indices": selected, "mrmr": mrmr,
                     "scaler": scaler, "feature_dictionary_sha256": digest(FEATURE_DICTIONARY),
                     "fit_scope": "all eligible full T; same mapping for all methods; kernel fit uses only 16 rows"}
    arrays, meta = {}, {}
    for name, rows in (("train", train), ("valid", valid), ("test", outer), ("full_train", full_train)):
        arrays["x_" + name] = apply_scaler(rows[names].to_numpy(), scaler)
        arrays["y_" + name] = (rows.y_raw.to_numpy() > 0).astype(np.int64)
        arrays["y_raw_" + name] = rows.y_raw.to_numpy(dtype=float)
        meta["meta_" + name] = _meta(rows, dates)
        if not np.isfinite(arrays["x_" + name]).all():
            raise ValueError("Nonfinite transformed features")
    if not (train.label_end_j.max() < split["v_start_j"] and
            full_train.label_end_j.max() < split["v_start_j"] and
            valid.label_end_j.max() < split["o_start_j"] and outer.label_end_j.max() <= split["d"]):
        raise RuntimeError("Label boundary violation")
    audit = {"window_id": "latest50_" + split["D"].replace("-", ""), "latest_date": split["D"],
             "task": "classification", "calendar_provenance": calendar_provenance,
             "seed": int(seed), "rows": {k: len(arrays["y_" + k]) for k in ("train", "valid", "test", "full_train")},
             "split": split, "sampling": {"train": t_trace, "valid": v_trace},
             "class_counts": {k: np.bincount(arrays["y_" + k], minlength=2).tolist() for k in ("train", "valid", "test", "full_train")},
             "outer_dates": sorted(outer.date.dt.strftime("%Y-%m-%d").unique().tolist()),
             "boundaries_verified": True, "outer_coverage": "50 distinct stocks once each; 10 dates x5",
             "classification_rule": "open[t+6]/open[t+1]-1 >0; otherwise 0",
             "financial_scope": "D0 raw-price historical pilot, not net returns or causal effect",
             "target_missing_full_T_rows": int((~t_pool.target_valid).sum()),
             "feature_history": "121 consecutive valid market bars, no filling",
             "source_manifest_sha256": digest(universe_metadata)}
    metadata = {**meta, "audit": audit, "preprocessing": preprocessing,
                "preprocessing_hash": digest(preprocessing), "universe": universe,
                "schema_version": "cloud_v3_latest50_data_v1"}
    return {**arrays, **metadata, "metadata": metadata}


MISSING_PROTOCOL_ID = "qukernet-50stocks-3h-cutoff20260325-qlib-missing-v1"


def fit_missing_preprocessing(x, y):
    """Fit medians, exclusions, supervised selection and scaling on full T only."""
    from .features_missing import POLICY_ID, FEATURE_DICTIONARY as MISSING_DICTIONARY
    x, y = np.asarray(x, float), np.asarray(y, int)
    if x.ndim != 2 or x.shape != (len(y), 80) or not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Full training requires eighty features and two label classes")
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    medians, kept, removed, filled = [], [], [], {}
    for j, name in enumerate(FEATURES):
        if counts[j] == 0:
            medians.append(None)
            removed.append({"feature": name, "index": j, "reason": "all_missing_on_full_T"})
            continue
        median = float(np.median(x[finite[:, j], j]))
        medians.append(median)
        column = np.where(finite[:, j], x[:, j], median)
        reason = None
        if np.ptp(column) == 0:
            reason = "constant_after_training_median"
        elif np.quantile(column, .75) == np.quantile(column, .25):
            reason = "zero_iqr_after_training_median"
        elif any(np.array_equal(column, filled[k]) for k in kept):
            reason = "exact_duplicate_after_training_median"
        if reason:
            removed.append({"feature": name, "index": j, "reason": reason})
        else:
            kept.append(j)
            filled[j] = column
    if len(kept) < 8:
        raise ValueError("Fewer than eight nondegenerate training features after missing treatment")
    candidate = np.column_stack([filled[j] for j in kept])
    selected_local, mrmr = full_training_mrmr(candidate, y, feature_names=[FEATURES[j] for j in kept])
    selected = [kept[j] for j in selected_local]
    selected_values = np.column_stack([filled[j] for j in selected])
    scaler = fit_scaler(selected_values)
    if scaler["zero_iqr_columns"]:
        raise ValueError("Selected features have zero robust scale")
    return {"missing_policy_id": POLICY_ID, "selected_features": [FEATURES[j] for j in selected],
            "selected_indices": selected,
            "imputer": {"method": "training_feature_median", "fit_split": "full_train", "fit_rows": len(y),
                        "feature_names": list(FEATURES), "medians": medians,
                        "finite_counts": counts.tolist(), "missing_counts": (len(y)-counts).tolist(),
                        "eligible_indices": kept, "excluded_features": removed,
                        "fit_order": "finite median -> exclude all-missing/constant/zero-IQR/duplicate -> mRMR8 -> robust angle scaler"},
            "mrmr": mrmr, "scaler": scaler, "feature_dictionary_sha256": digest(MISSING_DICTIONARY),
            "fit_scope": "all eligible matured full T only; V and O only transform; fixed kernel fit uses original sixteen keys"}


def apply_missing_preprocessing(x, preprocessing):
    """Apply frozen training-only values; do not fit any quantity on this input."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 80:
        raise ValueError("Expected eighty raw feature columns")
    selected = preprocessing["selected_indices"]
    medians = preprocessing["imputer"]["medians"]
    if len(selected) != 8 or len(set(selected)) != 8 or any(medians[j] is None for j in selected):
        raise ValueError("Invalid selected training medians")
    missing80 = ~np.isfinite(x)
    values = np.where(missing80[:, selected], np.asarray([medians[j] for j in selected]), x[:, selected])
    angles = apply_scaler(values, preprocessing["scaler"])
    if not np.isfinite(angles).all():
        raise ValueError("Nonfinite imputed/scaled model features")
    return {"angles": angles, "missing_mask": missing80[:, selected], "missing_mask80": missing80}


def prepare_missing_window(panel, calendar, source_manifest, *, latest_date, seed=SEED,
                           protocol, frozen_samples):
    from .features_missing import build_features as build_missing_features, POLICY_ID
    if (not isinstance(protocol, dict) or protocol.get("protocol_id") != MISSING_PROTOCOL_ID
            or not isinstance(frozen_samples, dict) or frozen_samples.get("schema") != "cloud_v3_frozen_sample_keys_v1"):
        raise ValueError("Explicit approved missing protocol and frozen sample record required")
    frozen = dict(frozen_samples)
    frozen_hash = frozen.pop("sha256", None)
    if frozen_hash != digest(frozen) or frozen.get("protocol_id") != MISSING_PROTOCOL_ID:
        raise ValueError("Frozen sample checksum/protocol mismatch")
    policy = protocol["data"]["missing_policy"]
    if policy.get("id") != POLICY_ID or str(latest_date) != protocol["data"]["cutoff_date"] or int(seed) != protocol["seed"]:
        raise ValueError("Missing policy, cutoff or seed mismatch")
    universe = dict(frozen["universe"])
    universe_hash = universe.pop("sha256", None)
    if universe_hash != digest(universe):
        raise ValueError("Frozen universe checksum mismatch")
    universe["sha256"] = universe_hash
    chosen = universe["selected_symbols"]
    if len(chosen) != 50 or len(set(chosen)) != 50:
        raise ValueError("Fifty frozen stock identifiers required")
    dates, split = calendar_split(calendar, latest_date)
    if source_manifest != frozen["selected_source_manifest"]:
        raise ValueError("Frozen source manifest changed before model preparation")
    raw = panel.copy(deep=True)
    raw["date"] = pd.to_datetime(raw.date, errors="raise")
    if (not REQUIRED.issubset(raw.columns) or raw.duplicated(["symbol", "date"]).any()
            or raw.date.isna().any() or raw.date.dt.tz is not None
            or not raw.date.eq(raw.date.dt.normalize()).all()):
        raise ValueError("Invalid selected source keys")
    raw = raw[raw.date.between(dates[0], dates[-1]) & raw.symbol.isin(chosen)].copy()
    if set(raw.symbol) != set(chosen) or not raw.date.isin(dates).all():
        raise ValueError("Frozen panel/date coverage mismatch")
    frame = build_missing_features(raw, dates)
    indexed = frame.set_index(["symbol", "date"], drop=False)
    rows_by_split = {}
    for phase in ("train", "valid"):
        keys = frozen[phase]
        if len(keys) != 16 or len({(r["symbol"], r["date"]) for r in keys}) != 16:
            raise ValueError("Frozen controlled sample keys must contain sixteen distinct rows")
        rows = indexed.loc[[(r["symbol"], pd.Timestamp(r["date"])) for r in keys]].copy().reset_index(drop=True)
        if _meta(rows, dates) != keys:
            raise ValueError("Frozen labels/date keys disagree with calendar")
        if not rows.row_eligible.all() or not rows.target_valid.all() or not rows.j.isin(split[phase+"_days"]).all():
            raise ValueError("Frozen controlled sample failed bar, history-span or mature-label admission")
        rows_by_split[phase] = rows
    outer = indexed.loc[[(r["symbol"], pd.Timestamp(r["date"])) for r in universe["O_assignments"]]].copy()
    outer = outer.reset_index(drop=True).sort_values(["j", "symbol"])
    if len(outer) != 50 or outer.symbol.nunique() != 50 or not outer.row_eligible.all() or not outer.target_valid.all():
        raise ValueError("Fixed outer rows require current valid bars and actual matured labels")
    train = rows_by_split["train"]
    valid = rows_by_split["valid"]
    if np.bincount((train.y_raw.to_numpy()>0).astype(int), minlength=2).min() < 2:
        raise ValueError("Fixed training samples require two examples of each class")
    full_train = frame[frame.j.isin(split["train_days"]) & frame.row_eligible & frame.target_valid].sort_values(["j", "symbol"]).copy()
    if not (train.label_end_j.max() < split["v_start_j"] and full_train.label_end_j.max() < split["v_start_j"]
            and valid.label_end_j.max() < split["o_start_j"] and outer.label_end_j.max() <= split["d"]):
        raise ValueError("Mature label or purged time boundary violated")
    preprocessing = fit_missing_preprocessing(full_train[FEATURES].to_numpy(), (full_train.y_raw.to_numpy()>0).astype(int))
    arrays, meta, missingness = {}, {}, {}
    for phase, rows in (("train", train), ("valid", valid), ("test", outer), ("full_train", full_train)):
        transformed = apply_missing_preprocessing(rows[FEATURES].to_numpy(), preprocessing)
        arrays["x_"+phase] = transformed["angles"]
        arrays["y_"+phase] = (rows.y_raw.to_numpy()>0).astype(np.int64)
        arrays["y_raw_"+phase] = rows.y_raw.to_numpy(float)
        arrays["missing_mask_"+phase] = transformed["missing_mask"]
        arrays["missing_mask80_"+phase] = transformed["missing_mask80"]
        support = rows[["support__"+name for name in FEATURES]].to_numpy(dtype=np.int16)
        arrays["support_counts_"+phase] = support
        meta["meta_"+phase] = _meta(rows, dates)
        mask, selected_mask = transformed["missing_mask80"], transformed["missing_mask"]
        missingness[phase] = {"rows": len(rows), "rows_with_any_missing_80": int(mask.any(axis=1).sum()),
                              "missing_cells_80": int(mask.sum()), "missing_rate_80": float(mask.mean()),
                              "selected_missing_cells": int(selected_mask.sum()), "selected_missing_rate": float(selected_mask.mean()),
                              "raw_features_missing_by_column": mask.sum(axis=0).tolist(),
                              "support_count_min": support.min(axis=0).tolist(),
                              "support_count_median": np.median(support, axis=0).tolist(),
                              "support_semantics": "finite primitive observation counts, not confidence or effective sample size"}
    sample_binding = {"train": meta["meta_train"], "valid": meta["meta_valid"]}
    sample_binding["sha256"] = digest(sample_binding)
    audit = {"window_id": "latest50_20260325_qlib_missing_v1_"+sample_binding["sha256"][:12],
             "protocol_id": MISSING_PROTOCOL_ID, "latest_date": split["D"], "task": "classification", "seed": int(seed),
             "calendar_provenance": source_manifest["calendar_provenance"],
             "rows": {k:len(arrays["y_"+k]) for k in ("train", "valid", "test", "full_train")},
             "split": split, "sampling": frozen["sampling"], "sampling_rule": "original strict sample keys frozen before missing-policy transformation",
             "class_counts": {k:np.bincount(arrays["y_"+k], minlength=2).tolist() for k in ("train", "valid", "test", "full_train")},
             "outer_dates": sorted(outer.date.dt.strftime("%Y-%m-%d").unique().tolist()),
             "boundaries_verified": True, "outer_coverage": "50 distinct stocks once each; 10 dates x5",
             "classification_rule": "open[t+6]/open[t+1]-1 >0; otherwise 0; labels never imputed",
             "financial_scope": "D0 raw-price historical pilot, not net returns or causal effect",
             "row_eligibility": "current_valid_bar_and_121_market_session_span_since_first_valid_bar",
             "history_span_minimum": 121, "source_manifest_sha256": digest(source_manifest),
             "frozen_samples_sha256": sample_binding["sha256"], "frozen_sample_record_sha256": frozen_hash,
             "missingness": missingness, "raw_observations_filled": 0, "labels_filled": 0}
    metadata = {**meta, "audit": audit, "preprocessing": preprocessing, "preprocessing_hash": digest(preprocessing),
                "universe": universe, "frozen_samples": sample_binding, "missing_policy": policy,
                "protocol_id": MISSING_PROTOCOL_ID, "schema_version": "cloud_v3_qlib_missing_data_v1"}
    return {**arrays, **metadata, "metadata": metadata}


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes()).hexdigest()


def save_window(window, output, *, source_roots=(), source_manifest=None, stem="latest50"):
    roots = list(source_roots)
    if source_manifest is not None:
        if digest(source_manifest) != window["metadata"]["audit"]["source_manifest_sha256"]:
            raise ValueError("Source manifest disagrees with prepared window")
        if isinstance(source_manifest, dict) and source_manifest.get("source_root"):
            roots.append(source_manifest["source_root"])
    destination = validate_output_path(output, roots)
    if not stem or Path(stem).name != stem or not all(c.isalnum() or c in "_-" for c in stem):
        raise ValueError("Unsafe snapshot stem")
    npz_path, json_path = destination / (stem + ".npz"), destination / (stem + ".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("Frozen snapshots are not overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in window.items() if isinstance(v, np.ndarray)}
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    record = {"schema": "cloud_v3_private_snapshot_v1", "metadata": window["metadata"],
              "source_manifest": source_manifest,
              "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
              "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": _array_hash(v)} for k, v in arrays.items()}}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {"npz": str(npz_path), "json": str(json_path), "npz_sha256": record["npz_sha256"],
            "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest()}


def load_window(path):
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["npz_sha256"]:
        raise ValueError("Snapshot file checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if set(arrays) != set(record["arrays"]):
        raise ValueError("Snapshot array keys mismatch")
    for key, value in arrays.items():
        if _array_hash(value) != record["arrays"][key]["sha256"]:
            raise ValueError("Snapshot array checksum mismatch")
    metadata = record["metadata"]
    if digest(metadata["preprocessing"]) != metadata["preprocessing_hash"]:
        raise ValueError("Preprocessing checksum mismatch")
    return {**arrays, **metadata, "metadata": metadata}
