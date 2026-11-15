"""One bounded 50-stock cloud pilot with paired initialization control.

The provider is the only source of quantum probabilities. Completed workflows
can be read back; an interrupted workflow requires explicit reconciliation and
is never automatically resumed or resubmitted. Account/window enforcement is
the responsibility of the injected durable ledger-backed provider.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import time
import warnings

import numpy as np

from cloud_v2 import circuits as qc
from cloud_v2 import search
from cloud_v2.efficacy_analysis import classification_metrics

STAGE = "compact_pilot_v3"
SEED = 20261115
ARMS = ("qukernet", "random")
PROFILE = {
    "stage": STAGE, "paired_initialization_control": True,
    "qubits": 8, "features": 8, "n_train": 16, "n_valid": 16, "n_test": 50,
    "depths": [1, 2, 3], "shots": 512, "labels": 6,
    "predictor_train_labels": 4, "candidate_pool": 256,
    "top_k": 1, "configurations": 1, "exact_parameters": 2,
    "steps": 2, "batch_size": 4, "learning_rate": .2,
    "batch_seconds": 300, "batch_limit": 20,
    "optimization_seconds": 1800, "kernel_seconds": None,
    "validation_use": "endpoint diagnostic only; no original-layout or initialization V",
    "test_order": ["qukernet_endpoint", "random_endpoint", "qukernet_initial"],
    "random_measurements": "7 independent training kernels, choose highest KTA",
    "shared_training_batches": "2 per class without replacement within each batch; frozen before cloud",
}


def budget_upper():
    """Exact successful-path quota; the four analytic preflight circuits are external."""
    train_pairs = 16 * 15 // 2
    step_pairs = 4 * 3 // 2
    gradient = step_pairs * 2 * 2 * 2  # two parameters, two ends, two signs
    common = 7 * train_pairs + train_pairs + 2 * (step_pairs + gradient) + train_pairs + 16 * 16
    q, r = common + 2 * 50 * 16, common + 50 * 16
    assert (common, q, r) == (1444, 3044, 2244)
    return {"common_per_arm": common, "qukernet": q, "random": r,
            "probability_circuits": q+r, "requested_shots": (q+r)*512,
            "qukernet_batches": 155, "random_batches": 115, "batches": 270,
            "analytic_preflight_included": False}


def _source_hash():
    return qc.digest({"study": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "cloud_v2": search.source_hash()})


def _dates(metadata, split, count):
    value = metadata.get("meta_"+split)
    if value is None or len(value) != count:
        raise ValueError("Complete aligned split dates are required")
    return [str(v["date"]) if isinstance(v, dict) else str(v) for v in value]


def _admit(window):
    arrays = {}
    metadata = window.get("metadata", window)
    for split, count in (("train", 16), ("valid", 16), ("test", 50)):
        x, y = np.asarray(window["x_"+split], float), np.asarray(window["y_"+split], float)
        if (x.shape != (count, 8) or y.shape != (count,) or not np.isfinite(x).all()
                or not np.isfinite(y).all() or not set(np.unique(y)).issubset({0., 1.})):
            raise ValueError("This pilot requires finite 16/16/50 arrays with eight features and binary labels")
        arrays["x_"+split], arrays["y_"+split] = x, y
        _dates(metadata, split, count)
    fx, fy = np.asarray(window["x_full_train"], float), np.asarray(window["y_full_train"], float)
    if (fx.ndim != 2 or fx.shape[1] != 8 or fy.shape != (len(fx),) or len(fx) < 16
            or not np.isfinite(fx).all() or not np.isfinite(fy).all()
            or set(np.unique(fy)) != {0., 1.}):
        raise ValueError("Full-training linear control requires the same eight-feature map and both classes")
    arrays.update(x_full_train=fx, y_full_train=fy)
    if any(np.sum(arrays["y_train"] == c) < 2 for c in (0, 1)):
        raise search.ScientificInvalid("Two distinct training examples per class are required; no replacement")
    audit = metadata.get("audit", {})
    if audit.get("task") != "classification":
        raise ValueError("Classification data audit required")
    window_id = audit.get("window_id") or audit.get("quarter") or audit.get("latest_date")
    if not window_id or "/" in str(window_id) or "\\" in str(window_id):
        raise ValueError("A fixed auditable market-window identifier is required")
    outer = metadata["meta_test"]
    if any(not isinstance(r, dict) or not isinstance(r.get("symbol"), str) or not r["symbol"] for r in outer):
        raise ValueError("Evaluation rows require explicit security identifiers")
    symbols = [r["symbol"] for r in outer]
    outer_dates = _dates(metadata, "test", 50)
    _, per_date = np.unique(outer_dates, return_counts=True)
    if len(set(symbols)) != 50 or len(per_date) != 10 or not np.all(per_date == 5):
        raise ValueError("Evaluation must contain fifty distinct securities once each, on ten dates of five rows")
    universe = metadata.get("universe", {})
    selected = universe.get("selected_symbols", [])
    assignments = universe.get("O_assignments", [])
    if (len(selected) != 50 or set(selected) != set(symbols) or len(assignments) != 50
            or {(str(r.get("symbol")), str(r.get("date"))) for r in assignments}
               != set(zip(symbols, outer_dates))):
        raise ValueError("Evaluation identifiers/dates differ from the frozen universe assignments")
    if len(np.unique(arrays["x_train"], axis=0)) < 2 or not np.any(np.ptp(arrays["x_train"], axis=0) > 0):
        raise search.ScientificInvalid("Transformed training features are constant; no replacement")
    return arrays, metadata, str(window_id)


def feature_quality(arrays, metadata):
    """Record effective variation; eight input columns need not be eight active dimensions."""
    result = {"input_columns": 8,
              "selected_feature_names": metadata.get("preprocessing", {}).get("selected_features", []),
              "zero_iqr_columns": metadata.get("preprocessing", {}).get("scaler", {}).get("zero_iqr_columns", []),
              "interpretation": "Column count is not a claim of eight varying or independent feature dimensions",
              "splits": {}}
    for split in ("train", "valid", "test", "full_train"):
        x = arrays["x_"+split]
        result["splits"][split] = {"rows": len(x), "varying_columns": int(np.count_nonzero(np.ptp(x, axis=0)>0)),
                                    "unique_rows": len(np.unique(x, axis=0)),
                                    "centered_matrix_rank": int(np.linalg.matrix_rank(x-x.mean(0)))}
    return result


def freeze_batches(labels, seed=SEED):
    y = np.asarray(labels)
    if y.shape != (16,) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Frozen batches require sixteen two-class labels")
    if any(np.sum(y == c) < 2 for c in (0, 1)):
        raise search.ScientificInvalid("Balanced batch cannot be formed without replacement")
    result = []
    for step in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), step, 8703]))
        batch = np.concatenate([rng.choice(np.flatnonzero(y == c), 2, replace=False) for c in (0, 1)])
        result.append(batch.astype(int).tolist())
    return result


def _records(y, prediction, dates, decisions):
    return [{"row_id": i, "date": date, "target": int(y[i]), "prediction": int(prediction[i]),
             "decision": float(decisions[i])} for i, date in enumerate(dates)]


def _model_result(model, x, y, dates):
    pred = model.predict(x)
    decision = model.decision_function(x)
    return {"metrics": classification_metrics(y, pred, dates, decision),
            "records": _records(y, pred, dates, decision)}


def classical_baselines(arrays, metadata, store):
    """No O target influences model/gamma selection. Full-T adds data, not features."""
    from sklearn.svm import SVC, LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    x, xv, xt = (arrays["x_"+s]/np.pi for s in ("train", "valid", "test"))
    y, yv, yt = (arrays["y_"+s] for s in ("train", "valid", "test"))
    vd, od = _dates(metadata, "valid", 16), _dates(metadata, "test", 50)
    fixed = {"linear": SVC(C=1., kernel="linear").fit(x, y)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        full_model = LinearSVC(C=1., dual=False, tol=1e-6, max_iter=10000,
                               random_state=SEED).fit(arrays["x_full_train"]/np.pi, arrays["y_full_train"])
    fixed["full_train_linear"] = full_model
    full_note = {"estimator": "LinearSVC", "C": 1., "dual": False, "tol": 1e-6,
                 "max_iter": 10000, "random_state": SEED, "penalty": "l2", "loss": "squared_hinge",
                 "n_iter": int(full_model.n_iter_),
                 "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                 "comparison_limit": "More training rows AND a different solver/loss from controlled linear SVC; not a data-size-only ablation"}
    trials, rbf_models = [], []
    for gamma in (.1, 1., 10.):
        fitted = SVC(C=1., kernel="rbf", gamma=gamma).fit(x, y)
        accuracy = classification_metrics(yv, fitted.predict(xv), vd)["date_equal_accuracy"]
        trials.append({"gamma": gamma, "validation_date_equal_accuracy": accuracy})
        rbf_models.append((-accuracy, gamma, fitted))
    _, gamma, fitted = min(rbf_models, key=lambda v: v[:2])
    fixed["rbf"] = fitted
    # Persist selection before looking at the O predictions/targets.
    selection = {"C": 1., "input_transform": "same selected/scaled angles divided by pi",
                 "rbf_grid": trials, "selected_gamma": gamma, "selection_split": "V only",
                 "full_train_linear_rows": len(arrays["y_full_train"]),
                 "full_train_linear_solver": full_note}
    store.write("classical/selection_before_O.json", selection)
    models = {}
    for name, model in fixed.items():
        models[name] = {"V": _model_result(model, xv, yv, vd),
                        "O": _model_result(model, xt, yt, od)}
    majority = int(np.argmax(np.bincount(y.astype(int), minlength=2)))
    for name, value in (("majority", majority), ("always0", 0), ("always1", 1)):
        models[name] = {}
        for split, label, dates in (("V", yv, vd), ("O", yt, od)):
            pred = np.full(len(label), value)
            models[name][split] = {"metrics": classification_metrics(label, pred, dates, pred.astype(float)),
                                   "records": _records(label, pred, dates, pred.astype(float))}
    result = {"status": "complete", "execution_kind": "classical_local", "selection": selection,
              "training_class_counts": np.bincount(y.astype(int), minlength=2).tolist(), "models": models}
    store.write("classical/results.json", result)
    return result


def _fit_without_validation(k, y):
    """Fit initial or endpoint SVC; no validation cross-kernel is needed to fit."""
    search.reject_constant_kernel(k)
    # Empty held-out block uses only the training spectrum; no new quantum calls.
    kp, _, mapping = search.spectral_classification_map(k, np.empty((0, len(y))))
    from sklearn.svm import SVC
    model = SVC(C=1., kernel="precomputed").fit(kp, y)
    bundle = {"kind": "svc_binary", "C": 1., "mapping": mapping, "classes": model.classes_.tolist(),
              "support": model.support_.tolist(), "dual": model.dual_coef_[0].tolist(),
              "intercept": float(model.intercept_[0])}
    manual, _ = search.predict_bundle(bundle, k)
    if not np.array_equal(manual, model.predict(kp)):
        raise RuntimeError("Serialized SVC predictions do not reload consistently")
    return bundle


def _evaluate(bundle, cross, y, dates):
    prediction, decision = search.predict_bundle(bundle, cross)
    return {"metrics": classification_metrics(y, prediction, dates, decision),
            "records": _records(y, prediction, dates, decision)}


def paired_date_bootstrap(records_a, records_b, seed=SEED):
    """One-window paired noncircular date-block accuracy differences, not IID stocks."""
    def keyed(records):
        result = {(str(r["date"]), str(r["row_id"])): r for r in records}
        if len(result) != len(records):
            raise ValueError("Duplicate date/row pairs")
        return result
    a, b = keyed(records_a), keyed(records_b)
    if not a or set(a) != set(b) or any(a[k]["target"] != b[k]["target"] for k in a):
        raise ValueError("Complete matching date/row keys and targets are required")
    dates = sorted({k[0] for k in a})
    delta = np.asarray([np.mean([int(a[k]["prediction"] == a[k]["target"])
                                 - int(b[k]["prediction"] == b[k]["target"])
                                 for k in a if k[0] == d]) for d in dates])
    out = {"status": "available" if len(dates) >= 3 else "insufficient_dates",
           "dates": len(dates), "rows": len(a), "difference": "A minus B",
           "estimate": float(delta.mean()), "repetitions": 10000, "seed": seed, "blocks": {},
           "interval": "Exploratory 95% percentile interval, not multiplicity adjusted",
           "limitations": ["Single market window and search seed", "Does not cover new shot draws or model-selection uncertainty",
                           "Paired rows are clustered by observed date, not independent market trials"]}
    if len(dates) < 3:
        out["reason"] = "Fewer than three observed dates cannot support all registered block lengths"
        return out
    for length in (2, 1, 3):
        rng = np.random.default_rng(np.random.SeedSequence([seed, length, 4821]))
        starts = rng.integers(0, len(dates)-length+1,
                             size=(10000, int(np.ceil(len(dates)/length))))
        indices = (starts[:, :, None]+np.arange(length)).reshape(10000, -1)[:, :len(dates)]
        distribution = delta[indices].mean(1)
        out["blocks"][str(length)] = {"role": "primary" if length == 2 else "sensitivity",
                                      "ci95": np.quantile(distribution, [.025, .975]).tolist(),
                                      "estimate": out["estimate"]}
    return out


def comparison_availability(states, classical=None):
    """A partial kernel or prediction file is never a complete 50-row comparison."""
    left = states.get("qukernet", {}).get("endpoint_O", {}).get("records", [])
    right = {
        "qukernet_endpoint_minus_same_initial": states.get("qukernet", {}).get("initial_O", {}).get("records", []),
        "qukernet_endpoint_minus_random_endpoint": states.get("random", {}).get("endpoint_O", {}).get("records", []),
        "qukernet_endpoint_minus_rbf": (classical or {}).get("models", {}).get("rbf", {}).get("O", {}).get("records", []),
    }

    def valid(records):
        if len(records) != 50:
            return False
        try:
            ids = [r["row_id"] for r in records]
            dates, counts = np.unique([str(r["date"]) for r in records], return_counts=True)
            return (all(type(i) is int for i in ids) and set(ids) == set(range(50))
                    and len(dates) == 10 and bool(np.all(counts == 5))
                    and all(r["target"] in (0, 1) and r["prediction"] in (0, 1) for r in records))
        except (KeyError, TypeError, ValueError):
            return False

    out = {}
    for name, records in right.items():
        complete = valid(left) and valid(records)
        paired = complete and {(r["row_id"], str(r["date"]), r["target"]) for r in left} == {
            (r["row_id"], str(r["date"]), r["target"]) for r in records}
        out[name] = {"available": bool(paired), "reason": "complete_matching_50_rows" if paired else
                     "incomplete_or_invalid_50_row_endpoint" if not complete else "row_date_target_mismatch",
                     "left_completed_rows": len(left), "right_completed_rows": len(records), "expected_rows": 50}
        if paired:
            by_id = {r["row_id"]: r for r in records}
            out[name]["estimate"] = float(np.mean([int(r["prediction"] == r["target"])
                - int(by_id[r["row_id"]]["prediction"] == r["target"]) for r in left]))
    return out


def run_study(window, output, provider, *, seed=SEED, execution_identity=None, predictor_trainer=None):
    """Run Q/R stages alternately with 5,288 successful-path probability circuits.

    predictor_trainer is a test-only seam. Production callers must omit it.
    The parent runner supplies the preflight, account ledger, three-hour window,
    process resources and post-run private/public export boundary.
    """
    if type(seed) is not int or seed != SEED:
        raise ValueError("The authorized pilot freezes seed 20261115")
    arrays, metadata, market_window = _admit(window)
    execution = execution_identity or (provider.execution_identity() if hasattr(provider, "execution_identity")
                                       else {"kind": "injected_test_provider", "is_mock": True})
    test_execution = search.is_test_execution(execution, predictor_trainer is not None)
    versions = {}
    for package in ("numpy", "scikit-learn", "torch", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    profile = dict(PROFILE)
    protocol_id = metadata.get("protocol_id")
    if protocol_id:
        if "/" in protocol_id or "\\" in protocol_id:
            raise ValueError("Protocol identifier must be a single operation namespace component")
        profile["stage"] = STAGE + "/" + protocol_id
    identity = {"schema": "compact_pilot_v3_study", "profile": profile, "seed": seed,
                "protocol_id": protocol_id,
                "market_window": market_window, "method": "paired_study", "execution_identity": execution,
                "source_sha256": _source_hash(), "package_versions": versions,
                "arrays_sha256": {k: search.array_hash(v) for k, v in arrays.items()},
                "metadata_sha256": qc.digest(metadata), "test_override": predictor_trainer is not None}
    root = search.Checkpoints(output, identity)
    completed = root.get("workflow/complete", identity)
    if completed is not None:
        return completed
    intent = root.intent("workflow/complete", identity)
    started = time.monotonic()
    status = {"state": "running", "started_utc": search.now(), "stage": "prepared",
              "completed_arms": 0, "quantum_effect_conclusion": None,
              "execution_kind": "mock_control_flow_only" if test_execution else "cloud_provider",
              "budget_upper": budget_upper()}
    root.write("study_status.json", status)
    kernels, arm_store, states = {}, {}, {arm: {} for arm in ARMS}
    events = []
    classical = None
    root.write("comparisons_availability.json", comparison_availability(states, classical))

    def progress(phase, arm):
        events.append({"phase": phase, "arm": arm, "finished_utc": search.now(),
                       "validated_probability_circuits": {a: k.invocation_calls for a, k in kernels.items()}})
        status.update(stage=phase, current_arm=arm, events=events,
                      validated_probability_circuits=sum(k.invocation_calls for k in kernels.values()))
        root.write("study_status.json", status)
        root.write("comparisons_availability.json", comparison_availability(states, classical))

    try:
        for arm in ARMS:
            arm_store[arm] = search.Checkpoints(Path(output)/arm, dict(identity, method=arm))
            kernels[arm] = search.CloudKernel(provider, arm_store[arm], 512, None, 20, 300.)
        x, y = arrays["x_train"], arrays["y_train"]
        quality = feature_quality(arrays, metadata)
        root.write("feature_quality.json", quality)
        batches = freeze_batches(y, seed)
        root.write("frozen_batches.json", {"seed": seed, "indices": batches,
                                          "labels": [y[b].astype(int).tolist() for b in batches],
                                          "shared_across_arms": True, "resampling_on_failure": False})
        classical = classical_baselines(arrays, metadata, root)
        qlabels = qc.sample_unique(6, 8, 8, (1, 2, 3), seed)
        qpool = qc.sample_unique(256, 8, 8, (1, 2, 3), seed+1,
                                 excluded=[qc.layout_code(v) for v in qlabels])
        rlabels = qc.sample_unique(7, 8, 8, (1, 2, 3), seed+2)
        layouts = {"qukernet": qlabels, "random": rlabels}
        for arm in ARMS:
            states[arm]["labels"], states[arm]["train_kernels"] = [], []
            arm_store[arm].write("layouts.json", {"label_layouts": layouts[arm],
                                                    "candidate_layouts": qpool if arm == "qukernet" else [],
                                                    "deduplication": "canonical circuit code; functional kernel equivalence reported separately"})

        def measure_label(arm, index):
            layout = layouts[arm][index]
            k = kernels[arm].matrix(x, None, layout, key=f"labels/{index}")
            search.reject_constant_kernel(k)
            value = search.alignment_and_gradient(k, y, "classification")[0]
            equivalent = [i for i, old in enumerate(states[arm]["train_kernels"])
                          if np.max(np.abs(k-old)) <= 1e-10]
            row = {"index": index, "layout": layout, "layout_hash": qc.digest(layout), "alignment": value,
                   "status": "valid", "measured_train_kernel_equivalent_to": equivalent}
            states[arm]["labels"].append(row)
            states[arm]["train_kernels"].append(k)
            arm_store[arm].write("labels.json", states[arm]["labels"])
            progress(f"labels/{index}", arm)

        # Alternate complete candidate kernels rather than finish one full arm first.
        for i in range(6):
            for arm in ARMS:
                measure_label(arm, i)
        trainer = predictor_trainer or search.train_predictor
        scorer, note = trainer(qlabels[:4], [r["alignment"] for r in states["qukernet"]["labels"][:4]], 3, seed)
        predictions = np.asarray(scorer.predict(qpool), float)
        if predictions.shape != (256,) or not np.isfinite(predictions).all():
            raise search.ScientificInvalid("Invalid candidate predictions; no replacement")
        selected = min(range(256), key=lambda i: (-predictions[i], qc.layout_code(qpool[i])))
        heldout = []
        for i in (4, 5):
            row = states["qukernet"]["labels"][i]
            equivalent = any(j < 4 for j in row["measured_train_kernel_equivalent_to"])
            heldout.append({"index": i, "alignment": row["alignment"],
                            "training_kernel_equivalent": equivalent,
                            "prediction": None if equivalent else float(scorer.predict([qlabels[i]])[0])})
        arm_store["qukernet"].write("predictor.json", {**note, "heldout": heldout,
                                                       "pool_predictions": predictions.tolist(), "selected_index": selected})
        qlayout = qpool[selected]
        qk = kernels["qukernet"].matrix(x, None, qlayout, key="selected/train")
        search.reject_constant_kernel(qk)
        states["qukernet"].update(layout=qlayout, original_train=qk,
                                  original_alignment=search.alignment_and_gradient(qk, y)[0])
        progress("selected/train", "qukernet")
        measure_label("random", 6)
        ri = min(range(7), key=lambda i: (-states["random"]["labels"][i]["alignment"], qc.layout_code(rlabels[i])))
        states["random"].update(layout=rlabels[ri], original_train=states["random"]["train_kernels"][ri],
                                original_alignment=states["random"]["labels"][ri]["alignment"])
        for arm in ARMS:
            state, store = states[arm], arm_store[arm]
            state["mask"] = qc.make_mask(state["layout"], seed, 0, 2, exact_parameters=2)
            state["initial_parameters"] = qc.initial_parameters(x, state["layout"], state["mask"])
            state["parameters"] = state["initial_parameters"].copy()
            store.write("selected_and_initialization.json", {"layout": state["layout"], "mask": state["mask"],
                "initial_parameters": state["initial_parameters"], "original_alignment": state["original_alignment"],
                "parameter_initialization": "training feature means at fixed replacement positions",
                "selection": "neural pool score" if arm == "qukernet" else "measured training KTA"})
            state["initial_train"] = kernels[arm].matrix(x, None, state["layout"], state["mask"],
                                                          state["parameters"], key="tuning/initial_train")
            state["initial_alignment"] = search.alignment_and_gradient(state["initial_train"], y)[0]
            state["initial_bundle"] = _fit_without_validation(state["initial_train"], y)
            store.write("initial_model_bundle.json", state["initial_bundle"])
            progress("initial_train", arm)
        timers = {arm: search.OptimizationTimer(arm_store[arm], "tuning", PROFILE["optimization_seconds"]) for arm in ARMS}
        for step, indices in enumerate(batches):
            for arm in ARMS:
                state, kernel = states[arm], kernels[arm]
                xb, yb = x[indices], y[indices]
                deadline = timers[arm].deadline()
                k = kernel.matrix(xb, None, state["layout"], state["mask"], state["parameters"],
                                  key=f"tuning/step{step}/base", deadline=deadline)
                derivatives = kernel.derivatives(xb, state["layout"], state["mask"], state["parameters"],
                                                 key=f"tuning/step{step}/gradient", deadline=deadline)
                value, gradient = search.alignment_and_gradient(k, yb, "classification", derivatives)
                if not np.isfinite(gradient).all():
                    raise search.ScientificInvalid("Nonfinite KTA gradient; no replacement")
                state["parameters"] = state["parameters"]+.2*gradient
                trace = {"step": step, "batch_indices": indices, "alignment_before": value,
                         "gradient": gradient, "parameters_after": state["parameters"],
                         "shared_batch": True, "both_overlap_ends": True}
                arm_store[arm].write(f"tuning/step{step}.json", trace)
                progress(f"optimization/step{step}", arm)
        for timer in timers.values():
            timer.finish()
        for arm in ARMS:
            state = states[arm]
            state["final_train"] = kernels[arm].matrix(x, None, state["layout"], state["mask"], state["parameters"], key="tuning/final_train")
            state["final_alignment"] = search.alignment_and_gradient(state["final_train"], y)[0]
            state["final_bundle"] = _fit_without_validation(state["final_train"], y)
            arm_store[arm].write("endpoint_model_bundle.json", state["final_bundle"])
            arm_store[arm].write("tuning/endpoint.json", {k: state[k] for k in ("layout", "mask", "initial_parameters", "parameters", "initial_alignment", "final_alignment")})
            progress("final_train", arm)
        for arm in ARMS:
            state = states[arm]
            bv = kernels[arm].matrix(arrays["x_valid"], x, state["layout"], state["mask"], state["parameters"], key="endpoint/valid")
            state["validation"] = _evaluate(state["final_bundle"], bv, arrays["y_valid"], _dates(metadata, "valid", 16))
            arm_store[arm].write("endpoint_validation.json", state["validation"])
            progress("endpoint_valid_diagnostic", arm)
        for arm in ARMS:
            state = states[arm]
            bo = kernels[arm].matrix(arrays["x_test"], x, state["layout"], state["mask"], state["parameters"], key="endpoint/test")
            state["endpoint_O"] = _evaluate(state["final_bundle"], bo, arrays["y_test"], _dates(metadata, "test", 50))
            arm_store[arm].write("endpoint_predictions.json", state["endpoint_O"])
            progress("endpoint_O", arm)
        state = states["qukernet"]
        bi = kernels["qukernet"].matrix(arrays["x_test"], x, state["layout"], state["mask"], state["initial_parameters"], key="initial/test")
        state["initial_O"] = _evaluate(state["initial_bundle"], bi, arrays["y_test"], _dates(metadata, "test", 50))
        arm_store["qukernet"].write("paired_initial_predictions.json", state["initial_O"])
        progress("initial_O", "qukernet")
        actual = {arm: kernels[arm].invocation_calls for arm in ARMS}
        upper = budget_upper()
        if any(actual[a] != upper[a] for a in ARMS):
            raise RuntimeError("Successful workflow does not match the frozen circuit quota")
        arm_summary = {a: {"original_KTA": states[a]["original_alignment"],
                           "initial_KTA": states[a]["initial_alignment"], "endpoint_KTA": states[a]["final_alignment"],
                           "V": states[a]["validation"]["metrics"], "O": states[a]["endpoint_O"]["metrics"],
                           "probability_circuits": actual[a]} for a in ARMS}
        arm_summary["qukernet"]["initial_O"] = states["qukernet"]["initial_O"]["metrics"]
        metric = "date_equal_accuracy"
        qacc = arm_summary["qukernet"]["O"][metric]
        comparisons = {
            "qukernet_endpoint_minus_same_initial": qacc-arm_summary["qukernet"]["initial_O"][metric],
            "qukernet_endpoint_minus_random_endpoint": qacc-arm_summary["random"]["O"][metric],
            "qukernet_endpoint_minus_rbf": qacc-classical["models"]["rbf"]["O"]["metrics"][metric]}
        qr = states["qukernet"]["endpoint_O"]["records"]
        uncertainty = {
            "qukernet_endpoint_minus_same_initial": paired_date_bootstrap(qr, states["qukernet"]["initial_O"]["records"], seed),
            "qukernet_endpoint_minus_random_endpoint": paired_date_bootstrap(qr, states["random"]["endpoint_O"]["records"], seed),
            "qukernet_endpoint_minus_rbf": paired_date_bootstrap(qr, classical["models"]["rbf"]["O"]["records"], seed)}
        root.write("paired_uncertainty.json", uncertainty)
        result = {"workflow_status": "complete", "execution_kind": status["execution_kind"],
                  "completed_arms": 2, "market_window": market_window, "seed": seed,
                  "arms": arm_summary, "classical": {k: {s: v[s]["metrics"] for s in ("V", "O")} for k,v in classical["models"].items()},
                  "date_equal_accuracy_differences": comparisons, "circuits": actual,
                  "paired_date_bootstrap": uncertainty,
                  "comparison_availability": comparison_availability(states, classical),
                  "feature_quality": quality,
                  "request_shots": sum(actual.values())*512, "elapsed_seconds": time.monotonic()-started,
                  "interpretation": "Single-window, single-seed small-sample pilot; endpoint V is diagnostic; no profit or quantum advantage claim",
                  "quantum_effect_conclusion": "Requires substantive interpretation of complete paired results, not workflow completion"}
        root.finish("workflow/complete", intent, result)
        status.update(state="complete", completed_arms=2, finished_utc=search.now(), elapsed_seconds=result["elapsed_seconds"])
        root.write("study_status.json", status)
        root.write("summary.json", result)
        return result
    except BaseException as exc:
        intent.update(status="failed_or_unknown", error_type=type(exc).__name__, stopped_utc=search.now())
        root.write(root.path_for("workflow/complete"), intent)
        status.update(state="stopped", error_type=type(exc).__name__, stopped_utc=search.now(),
                      no_automatic_resume=True, elapsed_seconds=time.monotonic()-started)
        root.write("study_status.json", status)
        root.write("comparisons_availability.json", comparison_availability(states, classical))
        raise
