"""Classical controls and exploratory paired evaluation for frozen stock windows.

No authentication, cloud provider, submission or quantum simulator is used here.
2022--2024 are development replays, not a new untouched financial holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .search import (ScientificInvalid, evaluate_validation, predict_bundle,
                     OutputRoot)

QUARTERS = ("2022Q1", "2023Q1", "2024Q1")
GAMMAS = (.1, 1., 10.)
SEED = 20261110


def _dates(values):
    return np.asarray([str(v["date"]) if isinstance(v, dict) else str(v)
                       for v in values])


def classification_metrics(y, prediction, dates, decision=None):
    """Each observed date receives equal mass, split equally over its rows.

    BA is the mean of recalls derived from the date-weighted confusion matrix;
    it is not an average of per-date BA (a date may contain only one class).
    AUC uses the same row weights. Missing-class recalls/BA/AUC are explicit null.
    """
    y, prediction, dates = np.asarray(y), np.asarray(prediction), _dates(dates)
    if (y.ndim != 1 or prediction.shape != y.shape or dates.shape != y.shape
            or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all()
            or not set(np.unique(y)).issubset({0, 1})
            or not set(np.unique(prediction)).issubset({0, 1})):
        raise ValueError("Finite aligned binary targets/predictions and dates required")
    unique, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = 1. / (len(unique) * counts[inverse])
    confusion, raw = np.zeros((2, 2)), np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (y.astype(int), prediction.astype(int)), weights)
    np.add.at(raw, (y.astype(int), prediction.astype(int)), 1)
    recall = [float(confusion[c, c] / confusion[c].sum())
              if confusion[c].sum() > 0 else None for c in (0, 1)]
    auc = None
    if decision is not None:
        decision = np.asarray(decision, float)
        if decision.shape != y.shape or not np.isfinite(decision).all():
            raise ValueError("Finite aligned decisions required")
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, decision, sample_weight=weights))
    return {"rows": len(y), "dates": len(unique),
            "date_equal_accuracy": float(np.trace(confusion)),
            "balanced_accuracy": float(np.mean(recall)) if None not in recall else None,
            "recall_class_0": recall[0], "recall_class_1": recall[1],
            "confusion_counts": raw.tolist(),
            "confusion_date_equal_mass": confusion.tolist(), "roc_auc": auc,
            "weighting": "Each date has mass 1/D; its n_d rows each have mass 1/(D*n_d)",
            "confusion_axes": "rows=true 0/1, columns=predicted 0/1",
            "auc_missing_class_or_decisions": auc is None}


def _gram(left, right, kind, gamma=None):
    if kind == "linear":
        return left @ right.T
    if kind != "rbf" or gamma not in GAMMAS:
        raise ValueError("Unregistered classical kernel")
    distance = np.maximum(np.sum(left*left, axis=1)[:, None]
                          + np.sum(right*right, axis=1)[None, :] - 2*left@right.T, 0.)
    return np.exp(-gamma*distance)


def _records(y, prediction, dates, decision=None):
    return [{"row_id": i, "date": str(date), "target": int(y[i]),
             "prediction": int(prediction[i]),
             "decision": None if decision is None else float(decision[i])}
            for i, date in enumerate(_dates(dates))]


def _private_root(output):
    here = Path(__file__).resolve()
    code_root = next((p for p in here.parents if p.name == "code"), here.parent)
    public = next((p for p in here.parents if (p/"课程大纲.md").exists()), code_root)
    destination = Path(output).resolve()
    if destination == public or destination.is_relative_to(public):
        raise ValueError("Per-row financial results must remain outside public source")
    return OutputRoot(destination)


def baseline(window, output):
    """Fit frozen classical controls using the same admitted T/V/O window.

    The caller loads the checksum-verified window with experiments.load_window.
    SVC C=1 and the positive-spectrum train/cross map are shared with QSVC.
    Only the V-selected RBF gamma is evaluated on O. Output is never overwritten.
    """
    metadata = window.get("metadata", {k: v for k, v in window.items()
                                       if not isinstance(v, np.ndarray)})
    audit = metadata.get("audit", {})
    quarter = audit.get("quarter")
    if quarter not in QUARTERS or audit.get("task") != "classification":
        raise ValueError("Only registered classification development quarters are admitted")
    x, y, dates = {}, {}, {}
    for split in ("train", "valid", "test"):
        x[split] = np.asarray(window["x_"+split], float)/np.pi
        y[split] = np.asarray(window["y_"+split])
        dates[split] = _dates(metadata["meta_"+split])
        if (x[split].ndim != 2 or x[split].shape != (len(y[split]), 40)
                or not np.isfinite(x[split]).all() or not np.isfinite(y[split]).all()
                or dates[split].shape != y[split].shape
                or not set(np.unique(y[split])).issubset({0, 1})
                or any(int(d[:4]) not in range(2019, 2025) for d in dates[split])):
            raise ValueError("Same finite 40-feature admitted development window required")
    if len(x["train"]) != 16 or len(x["valid"]) != 16 or set(np.unique(y["train"])) != {0, 1}:
        raise ValueError("Registered 16/16 two-class training window required")
    out = _private_root(output)
    names = ("classical_baselines_summary.json", "classical_baselines_predictions.json",
             "classical_selection_before_O.json")
    if any(out.path(name).exists() for name in names):
        raise FileExistsError("Classical baseline outputs already exist; no overwrite")
    selected, selection = {}, {}
    # This entire loop uses only T and V arrays/targets. O kernels are below.
    for kind, gammas in (("linear", (None,)), ("rbf", GAMMAS)):
        trials, candidates = [], []
        for gamma in gammas:
            k = _gram(x["train"], x["train"], kind, gamma)
            b = _gram(x["valid"], x["train"], kind, gamma)
            try:
                fit = evaluate_validation(k, b, y["train"], y["valid"], dates["valid"], "classification")
                trials.append({"gamma": gamma, "status": "valid",
                               "validation_date_equal_accuracy": fit["validation_accuracy"]})
                candidates.append((fit["selection_loss"], 0. if gamma is None else gamma, fit))
            except ScientificInvalid as exc:
                trials.append({"gamma": gamma, "status": "scientific_invalid", "reason": str(exc)})
        if candidates:
            _, gamma_key, fit = min(candidates, key=lambda row: row[:2])
            gamma = None if kind == "linear" else gamma_key
            selected[kind] = (gamma, fit)
            selection[kind] = {"status": "selected", "gamma": gamma,
                               "validation_date_equal_accuracy": fit["validation_accuracy"], "grid": trials}
        else:
            selection[kind] = {"status": "scientific_invalid", "grid": trials}
    selection_record = {"quarter": quarter, "selection_split": "V only", "C": 1.,
                        "input_transform": "same training-selected/scaled angles divided by pi",
                        "rbf_gamma_grid": list(GAMMAS), "tie_rule": "smaller gamma for equal V accuracy",
                        "models": selection, "frozen_utc": datetime.now(timezone.utc).isoformat()}
    out.json(names[2], selection_record)
    models, private = {}, {"quarter": quarter, "models": {}}
    for kind, (gamma, fit) in selected.items():
        cross = _gram(x["test"], x["train"], kind, gamma)
        prediction, decision = predict_bundle(fit["bundle"], cross)
        models[kind] = {"status": "complete", "gamma": gamma,
                        "V": classification_metrics(y["valid"], fit["validation_predictions"], dates["valid"], fit["validation_decisions"]),
                        "O": classification_metrics(y["test"], prediction, dates["test"], decision)}
        private["models"][kind] = {"records": _records(y["test"], prediction, dates["test"], decision),
                                    "validation_records": _records(y["valid"], fit["validation_predictions"], dates["valid"], fit["validation_decisions"]),
                                    "bundle": fit["bundle"]}
    for kind in ("linear", "rbf"):
        if kind not in models:
            models[kind] = {"status": "scientific_invalid", "O": None}
    counts = np.bincount(y["train"].astype(int), minlength=2)
    majority = int(np.argmax(counts))
    for name, value in (("majority", majority), ("always0", 0), ("always1", 1)):
        vp, op = np.full(len(y["valid"]), value), np.full(len(y["test"]), value)
        models[name] = {"status": "complete", "predicted_class": value,
                        "V": classification_metrics(y["valid"], vp, dates["valid"], vp.astype(float)),
                        "O": classification_metrics(y["test"], op, dates["test"], op.astype(float))}
        private["models"][name] = {"records": _records(y["test"], op, dates["test"], op.astype(float)),
                                    "validation_records": _records(y["valid"], vp, dates["valid"], vp.astype(float))}
    summary = {"schema": "classical_efficacy_v1", "quarter": quarter,
               "execution_kind": "local_classical_computation", "cloud_submissions": 0,
               "financial_scope": "D0 raw-price development replay; not sealed holdout or trading returns",
               "C": 1., "input_transform": "z=x_angles/pi", "training_class_counts": counts.tolist(),
               "majority_tie_rule": "class 0", "selection": selection, "models": models,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.json(names[1], private)
    out.json(names[0], summary)
    return summary


def _validated_record_map(records):
    result = {}
    for row in records:
        key = (str(row["date"]), str(row["row_id"]))
        if key in result or row["target"] not in (0, 1) or row["prediction"] not in (0, 1):
            raise ValueError("Duplicate or nonbinary paired record")
        result[key] = row
    if not result:
        raise ValueError("Empty records")
    return result


def _balanced(confusion):
    totals = confusion.sum(axis=-1)
    valid = np.all(totals > 0, axis=-1)
    recalls = np.divide(np.diagonal(confusion, axis1=-2, axis2=-1), totals,
                        out=np.full_like(totals, np.nan, dtype=float), where=totals > 0)
    return np.mean(recalls, axis=-1), valid


def paired_date_block_bootstrap(arm_a, arm_b, expected_quarters=QUARTERS,
                               repeats=10000, seed=SEED):
    """Exploratory A-minus-B comparisons, equal-weight quarters and dates.

    arm_a/arm_b map quarter -> O records (row_id/date/target/prediction).
    Each included quarter requires exactly identical record keys and targets.
    Noncircular moving blocks of adjacent observed dates are sampled within
    quarter; concatenation is truncated to its original number of dates.
    Two-date blocks are primary, one and three are sensitivity analyses.
    """
    if type(repeats) is not int or repeats < 1 or type(seed) is not int:
        raise ValueError("Positive repetitions and integer seed required")
    expected = list(expected_quarters)
    if not expected or len(set(expected)) != len(expected) or any(q not in QUARTERS for q in expected):
        raise ValueError("Unique registered development quarters required")
    included, excluded, series = [], [], {}
    for quarter in expected:
        if quarter not in arm_a or quarter not in arm_b:
            excluded.append({"quarter": quarter, "reason": "missing_arm"})
            continue
        try:
            a, b = _validated_record_map(arm_a[quarter]), _validated_record_map(arm_b[quarter])
            if set(a) != set(b) or any(a[k]["target"] != b[k]["target"] for k in a):
                raise ValueError("Unpaired row keys or different targets")
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({"quarter": quarter, "reason": "incomplete_or_inconsistent_pair", "detail": str(exc)})
            continue
        dates = sorted({k[0] for k in a})
        if len(dates) < 3:
            excluded.append({"quarter": quarter, "reason": "fewer_than_three_dates_for_registered_blocks"})
            continue
        values = np.zeros((len(dates), 2, 2, 2))
        for i, date in enumerate(dates):
            keys = [k for k in a if k[0] == date]
            for side, rows in enumerate((a, b)):
                for key in keys:
                    values[i, side, rows[key]["target"], rows[key]["prediction"]] += 1/len(keys)
        series[quarter] = values
        included.append({"quarter": quarter, "rows": len(a), "dates": len(dates)})
    out = {"schema": "paired_date_bootstrap_v1", "difference": "A minus B",
           "status": "available" if series else "unavailable",
           "coverage": {"expected_quarters": expected, "included": included, "excluded": excluded,
                        "complete": len(included) == len(expected), "zero_imputation": False},
           "repetitions": repeats, "seed": seed, "quarter_weighting": "equal among complete paired included quarters",
           "interval": "95% percentile bootstrap, exploratory and not multiplicity adjusted",
           "limitations": ["Development replay, not a newly unseen financial holdout",
                          "Few dates and only three planned quarters limit bootstrap calibration",
                          "Does not represent seed, shot or model-selection uncertainty",
                          "Partial coverage does not estimate the missing-quarter contrast"], "blocks": {}}
    if not series:
        return out
    for length in (2, 1, 3):
        delta_acc, delta_ba, point_acc, point_ba = [], [], [], []
        for qi, (quarter, values) in enumerate(series.items()):
            count = len(values)
            # Quarter identifier, rather than its index among included quarters,
            # keeps its resampling stream fixed when a different quarter is absent.
            rng = np.random.default_rng(np.random.SeedSequence([seed, length, QUARTERS.index(quarter)]))
            starts = rng.integers(0, count-length+1, size=(repeats, int(np.ceil(count/length))))
            indices = (starts[:, :, None]+np.arange(length)).reshape(repeats, -1)[:, :count]
            conf = values[indices].mean(axis=1)
            accuracy = np.trace(conf, axis1=-2, axis2=-1)
            ba, valid = _balanced(conf)
            delta_acc.append(accuracy[:, 0]-accuracy[:, 1])
            delta_ba.append(np.where(valid.all(axis=-1), ba[:, 0]-ba[:, 1], np.nan))
            original = values.mean(axis=0)
            point_acc.append(float(np.trace(original[0])-np.trace(original[1])))
            original_ba, original_valid = _balanced(original)
            point_ba.append(float(original_ba[0]-original_ba[1]) if original_valid.all() else np.nan)
        def interval(draws, points):
            draws = np.mean(np.asarray(draws), axis=0)
            usable = np.isfinite(draws)
            return {"estimate": float(np.mean(points)) if np.isfinite(points).all() else None,
                    "ci95": np.quantile(draws[usable], [.025, .975]).tolist() if usable.any() else None,
                    "usable_replicates": int(usable.sum()), "undefined_replicates": int((~usable).sum())}
        out["blocks"][str(length)] = {"role": "primary" if length == 2 else "sensitivity",
            "delta_date_equal_accuracy": interval(delta_acc, point_acc),
            "delta_balanced_accuracy": interval(delta_ba, point_ba)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from .experiments import load_window
    summary = baseline(load_window(args.window), args.output)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
