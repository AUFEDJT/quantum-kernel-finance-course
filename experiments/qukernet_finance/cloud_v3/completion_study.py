"""Explicit evaluation-only completion of a safely paused compact pilot.

This module never trains a scorer, selects a layout, tunes parameters, measures a
training/validation kernel, resets an optimization timer, or rewrites its parent.
A launcher must independently authorize the remaining operations against the
canonical account ledger. No command-line cloud launcher is provided here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cloud_v2 import circuits as qc, search
from cloud_v2.ledger import request_identity, TERMINAL
from cloud_v3 import study

PHASES = (("qukernet", "endpoint", "endpoint/test", "endpoint_predictions.json"),
          ("random", "endpoint", "endpoint/test", "endpoint_predictions.json"),
          ("qukernet", "initial", "initial/test", "paired_initial_predictions.json"))


class CompletionBlocked(RuntimeError):
    pass


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_parent_file_hashes(parent_study):
    """Read-only inventory; call only after the parent is safely stopped."""
    root = Path(parent_study).resolve()
    if not root.is_dir():
        raise CompletionBlocked("Parent study directory is missing")
    result = {}
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink():
            raise CompletionBlocked("Parent evidence cannot contain symlinks")
        result[path.relative_to(root).as_posix()] = file_sha256(path)
    return result


class _Reader:
    def __init__(self, root, pinned):
        self.root, self.pinned, self.used = Path(root).resolve(), dict(pinned), {}

    def get(self, relative, *, optional=False):
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path.is_symlink():
            raise CompletionBlocked("Parent evidence path is unsafe")
        if not path.exists() and optional and relative not in self.pinned:
            return None
        if not path.is_file() or relative not in self.pinned:
            raise CompletionBlocked("Required parent evidence was not frozen")
        payload = path.read_bytes()
        value = hashlib.sha256(payload).hexdigest()
        if value != self.pinned[relative]:
            raise CompletionBlocked("Parent evidence changed after freezing")
        self.used[relative] = value
        return json.loads(payload.decode("utf-8"))


def _namespace(identity):
    return "/".join(str(v) for v in (identity["profile"]["stage"], identity["market_window"],
                                     identity["method"], identity["seed"]))


def _result_check(saved, expected_y, expected_dates):
    """Re-evaluate saved records without constructing or fitting any model."""
    if not isinstance(saved, dict) or len(saved.get("records", [])) != len(expected_y):
        raise CompletionBlocked("Saved predictions are incomplete")
    records = saved["records"]
    if [r.get("row_id") for r in records] != list(range(len(expected_y))):
        raise CompletionBlocked("Saved prediction row order changed")
    if ([str(r.get("date")) for r in records] != expected_dates
            or [r.get("target") for r in records] != np.asarray(expected_y, int).tolist()):
        raise CompletionBlocked("Saved prediction dates or targets changed")
    recalculated = study.classification_metrics(expected_y,
        np.asarray([r["prediction"] for r in records]), expected_dates,
        np.asarray([r["decision"] for r in records], float))
    if qc.digest(recalculated) != qc.digest(saved.get("metrics")):
        raise CompletionBlocked("Saved metrics do not match their complete records")


def _validate_bundle(bundle, n=16):
    if (bundle.get("kind") != "svc_binary" or bundle.get("C") != 1.
            or bundle.get("classes") != [0., 1.]):
        raise CompletionBlocked("Parent SVC configuration changed")
    basis = np.asarray(bundle.get("mapping", {}).get("basis"), float)
    support = np.asarray(bundle.get("support"))
    dual = np.asarray(bundle.get("dual"), float)
    if (basis.ndim != 2 or basis.shape[0] != n or not 1 <= basis.shape[1] <= n
            or support.ndim != 1 or not len(support) or not np.issubdtype(support.dtype, np.integer)
            or len(set(support.tolist())) != len(support) or np.any(support < 0) or np.any(support >= n)
            or dual.shape != support.shape or not np.isfinite(basis).all() or not np.isfinite(dual).all()
            or not np.isfinite(float(bundle.get("intercept", np.nan)))):
        raise CompletionBlocked("Invalid frozen SVC bundle")


def _spec_batches(x_test, x_train, layout, mask, parameters, key):
    pending = []
    for i in range(50):
        for j in range(16):
            gates = qc.overlap_gates(x_test[i], x_train[j], layout, mask, parameters)
            pending.append(qc.specification(gates, 8, f"{key}/{i}/{j}"))
            if len(pending) == 20:
                yield pending
                pending = []
    if pending:
        raise CompletionBlocked("The frozen 50 by 16 evaluation must form forty full batches")


def _provider_values(results, specs, *, expected_jobs=None):
    if not isinstance(results, list) or len(results) != len(specs):
        raise CompletionBlocked("Provider batch is incomplete")
    values, jobs = [], []
    for item, spec in zip(results, specs):
        p = float(item.get("p0", np.nan))
        if (item.get("circuit_id", item.get("id")) != spec["id"] or item.get("shots") != 512
                or not np.isfinite(p) or not 0 <= p <= 1):
            raise CompletionBlocked("Provider result identity, shots or probability mismatch")
        values.append(p); jobs.append(str(item.get("job_id", "")))
    if any(not j for j in jobs) or len(set(jobs)) != len(jobs):
        raise CompletionBlocked("Provider result task IDs are missing or duplicated")
    if expected_jobs is not None and jobs != [str(j) for j in expected_jobs]:
        raise CompletionBlocked("Cached results do not match the original cloud task IDs")
    return values


@dataclass
class CompletionPlan:
    parent_study: Path
    parent_window_id: str
    parent_file_hashes: dict
    parent_ledger_digest: str
    parent_identity: dict
    arrays: dict
    metadata: dict
    arms: dict
    classical: dict
    phases: list
    allowed_operations: dict
    parent_recovery: dict
    integrity_sha256: str


def _plan_integrity(identity, arrays, metadata, arms, phases, allowed, recovery):
    return qc.digest({"parent_identity": identity,
                      "arrays": {k: search.array_hash(v) for k, v in arrays.items()},
                      "metadata": metadata, "arms": arms, "phases": phases,
                      "allowed_operations": allowed, "parent_recovery": recovery})


def build_completion_plan(parent_study, window, ledger_snapshot, *, parent_window_id,
                          parent_file_hashes, expected_source_sha256=None):
    """Pure/read-only preparation; no provider calls and no output creation.

    The allowlist retains each ORIGINAL operation ID. An old failed/intent
    checkpoint is eligible only when that ID never appeared in the whole account
    ledger. A validated reservation requires its complete, hash-matched cache.
    """
    reader = _Reader(parent_study, parent_file_hashes)
    arrays, metadata, market_window = study._admit(window)
    identity = reader.get("identity.json")
    source = study._source_hash()
    if expected_source_sha256 is not None and source != expected_source_sha256:
        raise CompletionBlocked("Loaded frozen computation sources differ from the authorized source hash")
    if (identity.get("source_sha256") != source or identity.get("seed") != study.SEED
            or identity.get("method") != "paired_study" or identity.get("market_window") != market_window
            or identity.get("metadata_sha256") != qc.digest(metadata)
            or identity.get("arrays_sha256") != {k: search.array_hash(v) for k, v in arrays.items()}
            or identity.get("protocol_id") != metadata.get("protocol_id")):
        raise CompletionBlocked("Parent code, frozen data, metadata or task identity changed")
    expected_profile = dict(study.PROFILE)
    if metadata.get("protocol_id"):
        expected_profile["stage"] = study.STAGE + "/" + metadata["protocol_id"]
    if identity.get("profile") != expected_profile:
        raise CompletionBlocked("Parent profile differs from the unchanged pilot")
    entries = ledger_snapshot.get("entries", [])
    if any(e.get("state") not in TERMINAL for e in entries):
        raise CompletionBlocked("Any account-wide unresolved attempt blocks completion")
    windows = [w for w in ledger_snapshot.get("execution_windows", []) if w.get("window_id") == parent_window_id]
    if (len(windows) != 1 or windows[0].get("state") != "closed"
            or windows[0].get("outcome") != "operator_stop" or windows[0].get("backend_cancelled") is not False):
        raise CompletionBlocked("Parent must be a controlled closed boundary stop, not a failure or cancellation")
    parent_entries = [e for e in entries if e.get("execution_window_id") == parent_window_id]
    if not parent_entries or any(e.get("state") != "validated" for e in parent_entries):
        raise CompletionBlocked("All parent reservations must be validated")
    by_operation = {}
    for e in entries:
        op = e.get("operation_id")
        if op:
            if op in by_operation:
                raise CompletionBlocked("Duplicate account-wide operation IDs")
            by_operation[op] = e
    arms, copied = {}, {}
    for relative in ("classical/results.json", "classical/selection_before_O.json", "feature_quality.json", "frozen_batches.json"):
        copied[relative] = reader.get(relative)
    classical = copied["classical/results.json"]
    if classical.get("status") != "complete":
        raise CompletionBlocked("Parent classical controls are incomplete")
    for model in classical.get("models", {}).values():
        for split, phase in (("V", "valid"), ("O", "test")):
            _result_check(model[split], arrays["y_"+phase], study._dates(metadata, phase, len(arrays["y_"+phase])))
    if not all(k in classical.get("models", {}) for k in ("majority", "always0", "always1", "linear", "rbf", "full_train_linear")):
        raise CompletionBlocked("A registered classical control is missing")
    for arm in study.ARMS:
        arm_identity = reader.get(arm + "/identity.json")
        if arm_identity != dict(identity, method=arm):
            raise CompletionBlocked("Parent arm identity mismatch")
        values = {}
        for name in ("selected_and_initialization.json", "initial_model_bundle.json", "endpoint_model_bundle.json",
                     "tuning/endpoint.json", "endpoint_validation.json", "timers/"+qc.digest("tuning")+".json"):
            relative = arm + "/" + name
            values[name] = reader.get(relative); copied[relative] = values[name]
        initial, endpoint = values["selected_and_initialization.json"], values["tuning/endpoint.json"]
        if any(initial.get(k) != endpoint.get(k) for k in ("layout", "mask", "initial_parameters")):
            raise CompletionBlocked("Parent selected layout, replacement positions or initialization disagree")
        q, p, d = qc.validate_layout(endpoint["layout"])
        if (q, p) != (8, 8) or d not in (1, 2, 3) or len(endpoint["mask"]) != 2:
            raise CompletionBlocked("Parent circuit configuration differs from the registered pilot")
        for key in ("initial_parameters", "parameters"):
            params = np.asarray(endpoint[key], float)
            if params.shape != (2,) or not np.isfinite(params).all():
                raise CompletionBlocked("Parent parameters are invalid")
        for name in ("initial_model_bundle.json", "endpoint_model_bundle.json"):
            _validate_bundle(values[name])
        _result_check(values["endpoint_validation.json"], arrays["y_valid"], study._dates(metadata, "valid", 16))
        timer = values["timers/"+qc.digest("tuning")+".json"]
        if (timer.get("identity_hash") != qc.digest(arm_identity) or timer.get("key") != "tuning"
                or timer.get("seconds") != 1800 or timer.get("status") != "complete"
                or timer.get("deadline_epoch") != timer.get("started_epoch", 0)+1800
                or not 0 <= timer.get("elapsed_seconds", -1) < 1800):
            raise CompletionBlocked("Parent optimization did not finish within its original durable 1800-second timer")
        arms[arm] = {"identity": arm_identity, "initial": initial, "endpoint": endpoint,
                     "initial_bundle": values["initial_model_bundle.json"],
                     "endpoint_bundle": values["endpoint_model_bundle.json"],
                     "validation": values["endpoint_validation.json"]}
    phases, allowed, all_o_ops, unreserved_intents = [], {}, set(), []
    reused = 0
    for arm, checkpoint, key, filename in PHASES:
        a = arms[arm]; endpoint = a["endpoint"]
        parameters = endpoint["parameters" if checkpoint == "endpoint" else "initial_parameters"]
        phase = {"arm": arm, "checkpoint": checkpoint, "key": key, "filename": filename, "batches": []}
        phase["saved_predictions"] = reader.get(arm+"/"+filename, optional=True)
        for index, specs in enumerate(_spec_batches(arrays["x_test"], arrays["x_train"], endpoint["layout"], endpoint["mask"], parameters, key)):
            operation = f"{key}/batch{index}"
            op = _namespace(a["identity"]) + "/" + operation
            all_o_ops.add(op)
            request = {"specifications": specs, "shots": 512}
            relative = arm+"/checkpoints/"+qc.digest(operation)+".json"
            cache = reader.get(relative, optional=True)
            entry = by_operation.get(op)
            batch = {"operation_id": op, "operation": operation, "specs": specs,
                     "request_identity": request_identity(specs, 512, "tianyan_sw", "main"),
                     "request_hash": qc.digest(request), "parent_checkpoint": relative,
                     "cached_values": None, "cached_record": None}
            if cache is not None:
                if (cache.get("identity_hash") != qc.digest(a["identity"])
                        or cache.get("request_hash") != batch["request_hash"] or cache.get("key") != operation):
                    raise CompletionBlocked("Parent evaluation checkpoint request/source binding changed")
            if entry is not None:
                if (entry.get("execution_window_id") != parent_window_id or entry.get("state") != "validated"
                        or entry.get("request_identity") != batch["request_identity"] or entry.get("circuits") != 20
                        or entry.get("shots_per_circuit") != 512 or entry.get("requested_shots") != 10240):
                    raise CompletionBlocked("Existing operation is not a validated matching parent batch")
                if cache is None or cache.get("status") != "complete":
                    raise CompletionBlocked("A reserved batch has no complete reusable checkpoint; never resubmit")
                value = cache.get("result", {})
                if (cache.get("result_sha256") != qc.digest(value) or value.get("probability_circuits") != 20
                        or value.get("nominal_shots") != 10240):
                    raise CompletionBlocked("Cached evaluation payload hash or resource count mismatch")
                p0 = _provider_values(value.get("provider_results"), specs, expected_jobs=entry.get("job_ids", []))
                if value.get("p0") != p0:
                    raise CompletionBlocked("Cached probability vector differs from validated provider records")
                batch.update(cached_values=p0, cached_record=cache)
                reused += 20
            else:
                if cache is not None and cache.get("status") == "complete":
                    raise CompletionBlocked("Unbacked complete cache has no account reservation")
                if cache is not None:
                    unreserved_intents.append({"operation_id": op, "parent_checkpoint": relative,
                                              "parent_checkpoint_sha256": reader.used[relative],
                                              "reason": "operation_absent_from_entire_frozen_account_ledger"})
                allowed[op] = {"request_identity": batch["request_identity"], "circuits": 20,
                               "shots_per_circuit": 512, "backend": "tianyan_sw", "qubits": 8}
            phase["batches"].append(batch)
        if phase["saved_predictions"] is not None:
            if any(b["cached_values"] is None for b in phase["batches"]):
                raise CompletionBlocked("Saved full evaluation is missing a validated backing batch")
            _result_check(phase["saved_predictions"], arrays["y_test"], study._dates(metadata, "test", 50))
        phases.append(phase)
    base_entries = [e for e in parent_entries if e.get("operation_id") not in all_o_ops]
    if (len(base_entries) != 151 or sum(e["circuits"] for e in base_entries) != 2892
            or sum(e["requested_shots"] for e in base_entries) != 2892*512):
        raise CompletionBlocked("Parent has not completed exactly the preflight, search, tuning and validation stages")
    if len(parent_entries)+len(allowed) != 271 or sum(e["circuits"] for e in parent_entries)+20*len(allowed) != 5292:
        raise CompletionBlocked("Combined quota differs from the original two-arm plan")
    recovery = {"schema": "evaluation_only_parent_recovery_v1", "parent_window_id": parent_window_id,
                "parent_outcome": "operator_stop", "parent_identity_sha256": qc.digest(identity),
                "parent_source_sha256": source, "parent_metadata_sha256": qc.digest(metadata),
                "parent_ledger_digest": qc.digest(ledger_snapshot),
                "copied_parent_files": {k: reader.used[k] for k in copied},
                "read_parent_files": dict(reader.used), "unreserved_intents": unreserved_intents,
                "reused_O_circuits": reused, "new_O_circuits": len(allowed)*20,
                "new_O_batches": len(allowed), "new_O_shots": len(allowed)*20*512,
                "no_scorer_training": True, "no_gradient_updates": True,
                "no_training_or_validation_quantum_calls": True, "optimization_timers_preserved": True}
    recovery["parent_evidence_sha256"] = qc.digest(recovery["read_parent_files"])
    recovery["copied_values"] = copied
    return CompletionPlan(Path(parent_study).resolve(), parent_window_id, dict(parent_file_hashes),
                          qc.digest(ledger_snapshot), identity, arrays, metadata, arms, classical,
                          phases, allowed, recovery,
                          _plan_integrity(identity, arrays, metadata, arms, phases, allowed, recovery))


def run_completion(plan, output, provider, *, ledger_snapshot, execution_identity):
    """One explicit launch. Incomplete child output is never automatically resumed."""
    if not isinstance(plan, CompletionPlan) or qc.digest(ledger_snapshot) != plan.parent_ledger_digest:
        raise CompletionBlocked("Parent planning ledger snapshot changed")
    if _plan_integrity(plan.parent_identity, plan.arrays, plan.metadata, plan.arms,
                       plan.phases, plan.allowed_operations, plan.parent_recovery) != plan.integrity_sha256:
        raise CompletionBlocked("Completion plan or frozen model inputs changed")
    # Recheck immutable inputs before any output creation or cloud call.
    for relative, expected in plan.parent_recovery["read_parent_files"].items():
        if file_sha256(plan.parent_study/relative) != expected:
            raise CompletionBlocked("Parent evidence changed before completion launch")
    destination = Path(output).resolve()
    if destination == plan.parent_study or destination.is_relative_to(plan.parent_study) or plan.parent_study.is_relative_to(destination):
        raise CompletionBlocked("Continuation output must be separate from the parent study")
    if destination.exists() and any(destination.iterdir()):
        raise CompletionBlocked("Continuation output already exists; no automatic restart")
    identity = {"schema": "evaluation_only_completion_v1", "parent_identity_sha256": qc.digest(plan.parent_identity),
                "parent_window_id": plan.parent_window_id, "parent_evidence_sha256": plan.parent_recovery["parent_evidence_sha256"],
                "allowed_operations_sha256": qc.digest(plan.allowed_operations),
                "execution_identity": execution_identity, "source_sha256": file_sha256(__file__)}
    out = search.Checkpoints(destination, identity)
    recovery = {k: v for k, v in plan.parent_recovery.items() if k != "copied_values"}
    out.write("parent_recovery.json", recovery)
    for relative, value in plan.parent_recovery["copied_values"].items():
        payload = (plan.parent_study/relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != recovery["copied_parent_files"][relative]:
            raise CompletionBlocked("Parent copy changed after admission")
        out.out.path(relative).write_bytes(payload)
    out.write("allowed_operations.json", plan.allowed_operations)
    status = {"state": "running", "scope": "evaluation_only", "parent_window_id": plan.parent_window_id,
              "new_probability_circuits": 0, "reused_probability_circuits": 0,
              "execution_kind": "mock_control_flow_only" if search.is_test_execution(execution_identity) else "cloud_provider",
              "started_utc": search.now(), "automatic_restart": False}
    states = {a: {} for a in study.ARMS}
    out.write("study_status.json", status)
    out.write("comparisons_availability.json", study.comparison_availability(states, plan.classical))
    started = time.monotonic()
    try:
        for phase in plan.phases:
            arm = phase["arm"]
            probabilities = []
            for batch in phase["batches"]:
                if batch["cached_values"] is not None:
                    probabilities.extend(batch["cached_values"])
                    status["reused_probability_circuits"] += 20
                    out.write("reused_batches/"+qc.digest(batch["operation_id"])+".json",
                              {"operation_id": batch["operation_id"], "parent_checkpoint": batch["parent_checkpoint"],
                               "parent_checkpoint_sha256": plan.parent_recovery["read_parent_files"][batch["parent_checkpoint"]],
                               "status": "reused_validated_parent_cache", "probability_circuits": 20})
                    continue
                op = batch["operation_id"]
                if op not in plan.allowed_operations:
                    raise CompletionBlocked("Operation is outside the frozen completion allowlist")
                request = {"specifications": batch["specs"], "shots": 512}
                if (qc.digest(request) != batch["request_hash"]
                        or request_identity(batch["specs"],512,"tianyan_sw","main") != plan.allowed_operations[op]["request_identity"]):
                    raise CompletionBlocked("Completion operation request changed")
                marker = out.intent("evaluation/"+op, request)
                result = provider.probability(batch["specs"], 512, context={"phase": "main",
                    "operation": batch["operation"], "operation_id": op,
                    "run_identity_sha256": qc.digest(plan.arms[arm]["identity"]),
                    "continuation_parent_window_id": plan.parent_window_id}, deadline=time.monotonic()+300)
                values = _provider_values(result, batch["specs"])
                out.finish("evaluation/"+op, marker, {"p0": values, "provider_results": result,
                    "probability_circuits": 20, "nominal_shots": 10240, "operation_id": op})
                probabilities.extend(values)
                status["new_probability_circuits"] += 20
                out.write("study_status.json", status)
            if len(probabilities) != 800:
                raise CompletionBlocked("Only a complete 50 by 16 cross-kernel may be evaluated")
            cross = np.asarray(probabilities, float).reshape(50, 16)
            bundle = plan.arms[arm][phase["checkpoint"]+"_bundle"]
            evaluated = study._evaluate(bundle, cross, plan.arrays["y_test"], study._dates(plan.metadata, "test", 50))
            _result_check(evaluated, plan.arrays["y_test"], study._dates(plan.metadata, "test", 50))
            if phase["saved_predictions"] is not None and qc.digest(evaluated) != qc.digest(phase["saved_predictions"]):
                raise CompletionBlocked("Parent complete predictions disagree with recomputed cached kernel predictions")
            states[arm]["endpoint_O" if phase["checkpoint"] == "endpoint" else "initial_O"] = evaluated
            out.write(arm+"/"+phase["filename"], evaluated)
            status.update(stage="endpoint_O" if phase["checkpoint"] == "endpoint" else "initial_O", current_arm=arm)
            out.write("study_status.json", status)
            out.write("comparisons_availability.json", study.comparison_availability(states, plan.classical))
        availability = study.comparison_availability(states, plan.classical)
        if not all(v["available"] for v in availability.values()):
            raise CompletionBlocked("All three complete 50-row comparisons are required")
        q = states["qukernet"]["endpoint_O"]["records"]
        right = {"qukernet_endpoint_minus_same_initial": states["qukernet"]["initial_O"]["records"],
                 "qukernet_endpoint_minus_random_endpoint": states["random"]["endpoint_O"]["records"],
                 "qukernet_endpoint_minus_rbf": plan.classical["models"]["rbf"]["O"]["records"]}
        uncertainty = {name: study.paired_date_bootstrap(q, records, study.SEED) for name, records in right.items()}
        arms = {}
        for arm, a in plan.arms.items():
            arms[arm] = {"original_KTA": a["initial"]["original_alignment"],
                         "initial_KTA": a["endpoint"]["initial_alignment"], "endpoint_KTA": a["endpoint"]["final_alignment"],
                         "V": a["validation"]["metrics"], "O": states[arm]["endpoint_O"]["metrics"],
                         "probability_circuits": study.budget_upper()[arm]}
        arms["qukernet"]["initial_O"] = states["qukernet"]["initial_O"]["metrics"]
        result = {"workflow_status": "complete", "execution_kind": status["execution_kind"], "completed_arms": 2,
                  "market_window": plan.parent_identity["market_window"], "seed": study.SEED,
                  "arms": arms, "classical": {k: {s: v[s]["metrics"] for s in ("V", "O")} for k,v in plan.classical["models"].items()},
                  "date_equal_accuracy_differences": {name: v["estimate"] for name,v in availability.items()},
                  "comparison_availability": availability, "paired_date_bootstrap": uncertainty,
                  "circuits": {a: study.budget_upper()[a] for a in study.ARMS},
                  "request_shots": study.budget_upper()["requested_shots"],
                  "new_probability_circuits": status["new_probability_circuits"],
                  "requested_new_shots": status["new_probability_circuits"]*512,
                  "continuation": {"scope": "evaluation_only", "parent_window_id": plan.parent_window_id,
                      "parent_evidence_sha256": recovery["parent_evidence_sha256"],
                      "reused_O_circuits": status["reused_probability_circuits"], "new_O_circuits": status["new_probability_circuits"]},
                  "elapsed_seconds": time.monotonic()-started,
                  "interpretation": "User-authorized quota completion; original three-hour deadline retained as historical evidence. No profit or quantum advantage claim."}
        if (status["new_probability_circuits"] != plan.parent_recovery["new_O_circuits"]
                or status["reused_probability_circuits"] != plan.parent_recovery["reused_O_circuits"]):
            raise CompletionBlocked("Completion work and reuse differ from the frozen plan")
        out.write("paired_uncertainty.json", uncertainty)
        out.write("summary.json", result)
        status.update(state="complete", completed_arms=2, finished_utc=search.now())
        out.write("study_status.json", status)
        return result
    except BaseException as exc:
        status.update(state="stopped", error_type=type(exc).__name__, stopped_utc=search.now(),
                      no_automatic_resume=True, background_tasks_cancelled=False)
        out.write("study_status.json", status)
        out.write("comparisons_availability.json", study.comparison_availability(states, plan.classical))
        raise
