"""The registered three-quarter study. Importing this module submits nothing.

An authenticated, budget-gated provider must be explicitly injected. Private
sample-level evidence stays outside the course. This runner never reads 2025.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from .experiments import load_window, profile
from .ledger import atomic_json, private_directory
from .search import ScientificInvalid, run_search

STAGE = "efficacy8_n16_512_v1"
SEED = 20261110
ORDER = ("2024Q1", "2023Q1", "2022Q1")
METHODS = ("qukernet", "random")
LIMITS = {"seconds": 259200, "circuits": 93760, "shots": 48005120,
          "batches": 4804, "batch_size": 20, "batch_seconds": 300}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(n_observations):
    # 18 complete alignment kernels, original pre validation, four paired
    # configurations, 3 x 28 x (1+4*2) optimizer circuits per configuration,
    # and two O blocks for the same post-selected configuration.
    a, v, o = 120, 256, 16 * n_observations
    circuits = 18*a + 2*v + 4*(2*a + 2*v + 3*28*9) + 2*o
    return {"circuits": circuits, "shots": circuits*512,
            "batches": 806 if n_observations == 220 else 790}


def snapshot_totals(ledger, window_id):
    state = ledger.snapshot()
    entries = [e for e in state["entries"] if e.get("execution_window_id") == window_id]
    windows = [w for w in state.get("execution_windows", []) if w["window_id"] == window_id]
    return {"window": windows[0] if windows else None,
            "batches": len(entries),
            "circuits": sum(e["circuits"] for e in entries),
            "requested_shots": sum(e["requested_shots"] for e in entries),
            "validated_batches": sum(e["state"] == "validated" for e in entries),
            "last_attempt_id": entries[-1]["attempt_id"] if entries else None,
            "last_attempt_state": entries[-1]["state"] if entries else None}


class ObservedProvider:
    """Preserve provider semantics while recording progress and storage use."""
    def __init__(self, provider, ledger, window_id, output, private_cache_root):
        self.provider, self.ledger, self.window_id = provider, ledger, window_id
        self.output = Path(output)
        self.private_cache_root = Path(private_cache_root).resolve()
        if (self.private_cache_root not in self.output.resolve().parents or
                self.private_cache_root not in ledger.root.resolve().parents):
            raise ValueError("Cache root must cover results and canonical cloud records")
        self.current = None

    def execution_identity(self):
        return self.provider.execution_identity()

    def probability(self, specs, shots, **kwargs):
        # A single response is independently capped by the reviewed transport.
        # Leave 512 MiB below the private 10 GiB cap before each new operation.
        used = sum(p.stat().st_size for p in self.private_cache_root.rglob("*") if p.is_file())
        if used > 10*1024**3 - 512*1024**2:
            raise RuntimeError("Private cache headroom exhausted; no new submission")
        if shutil.disk_usage(self.output).free < 2*1024**3:
            raise RuntimeError("Disk reserve below 2 GiB; no new submission")
        started = time.monotonic()
        try:
            return self.provider.probability(specs, shots, **kwargs)
        finally:
            atomic_json(self.output/"progress.json", {
                "updated_utc": now(), "current_arm": self.current,
                "last_operation_id": (kwargs.get("context") or {}).get("operation_id"),
                "last_batch_client_seconds": time.monotonic()-started,
                "private_cache_bytes_before_batch": used,
                **snapshot_totals(self.ledger, self.window_id)})


def run_study(data_dir, output_dir, provider, ledger, window_id, private_cache_root):
    """Run the frozen six arms. Cloud failures propagate and stop the study."""
    from .efficacy_analysis import baseline
    output = private_directory(output_dir)
    data_dir = Path(data_dir).resolve()
    protocol = profile(STAGE)
    if (protocol["shots"], protocol["n_train"], protocol["n_valid"],
        protocol["qubits"], protocol["features"]) != (512, 16, 16, 8, 40):
        raise ValueError("Registered study profile differs from authorization")
    windows, sources = {}, {}
    for quarter in ORDER:
        path = data_dir/(quarter+"_classification_n016.npz")
        windows[quarter] = load_window(path)
        sources[quarter] = {"npz_sha256": sha_file(path),
                            "sidecar_sha256": sha_file(path.with_suffix(".json")),
                            "n_observations": len(windows[quarter]["x_test"])}
    if [sources[q]["n_observations"] for q in ORDER] != [220, 209, 220]:
        raise ValueError("Frozen quarterly row counts changed")
    state_path = output/"study_status.json"
    if state_path.exists():
        raise RuntimeError("A study invocation already exists; inspect it, do not restart")
    state = {"schema": "qukernet_efficacy_study_v1", "state": "prepared",
             "prepared_utc": now(), "stage": STAGE, "seed": SEED,
             "order": list(ORDER), "methods": list(METHODS), "limits": LIMITS,
             "sources": sources, "arms": [], "new_cloud_submissions": 0,
             "financial_time_status": "2022-2024 development replay, not an unseen holdout",
             "zero_update_scope": "same configuration chosen by post validation",
             "execution_location": "TianYan cloud full-amplitude simulator; classical work on Wuying"}
    atomic_json(state_path, state)
    # These independent baselines do not decide the quantum candidate pool.
    for quarter in ORDER:
        baseline(windows[quarter], output/"baselines"/quarter)
    observed = ObservedProvider(provider, ledger, window_id, output, private_cache_root)
    state.update(state="running", classical_baselines_completed=True)
    atomic_json(state_path, state)
    try:
        for quarter in ORDER:
            for method in METHODS:
                observed.current = {"quarter": quarter, "method": method, "seed": SEED}
                state.update(current_arm=observed.current, updated_utc=now())
                atomic_json(state_path, state)
                dest = output/"arms"/quarter/method
                before = snapshot_totals(ledger, window_id)
                record = {**observed.current, "started_utc": now(), "budget": budget(sources[quarter]["n_observations"])}
                try:
                    result = run_search(windows[quarter], protocol, observed, dest,
                                        SEED, "classification", method, source=data_dir)
                    record.update(status=result["workflow_status"],
                                  scientific_status=result["scientific_status"],
                                  valid_configurations=result["valid_configurations"],
                                  test_metrics=result["test_metrics"],
                                  paired_initial_test_metrics=result["paired_initial_test_metrics"],
                                  selected_initial_alignment=result["selected_initial_alignment"],
                                  selected_final_alignment=result["selected_final_alignment"])
                except ScientificInvalid as exc:
                    # No replacement candidates; preserve the failed arm and
                    # continue the predeclared order only while the cloud is known.
                    record.update(status="scientific_invalid", error_type=type(exc).__name__)
                except BaseException as exc:
                    record.update(status="stopped", error_type=type(exc).__name__)
                    raise
                finally:
                    after = snapshot_totals(ledger, window_id)
                    record.update(finished_utc=now(),
                                  reserved_circuits=after["circuits"]-before["circuits"],
                                  requested_shots=after["requested_shots"]-before["requested_shots"],
                                  batches=after["batches"]-before["batches"])
                    state["arms"].append(record)
                    atomic_json(state_path, state)
        state.update(state="completed", finished_utc=now(),
                     completed_arms=sum(a['status']=='complete' for a in state['arms']),
                     all_configurations_valid_arms=sum(a.get('scientific_status')=='complete' for a in state['arms']),
                     scientific_invalid_arms=sum(a['status']=='scientific_invalid' for a in state['arms']),
                     efficacy_conclusion='requires paired analysis; workflow completion is not efficacy')
    except BaseException as exc:
        state.update(state="stopped", error_type=type(exc).__name__, finished_utc=now(),
                     new_submissions_stopped=True, automatic_polling_stopped=True,
                     backend_cancelled=False)
        raise
    finally:
        state["resource_totals"] = snapshot_totals(ledger, window_id)
        atomic_json(state_path, state)
    return state


if __name__ == "__main__":
    print(json.dumps({"status": "execution_disabled", "stage": STAGE,
                      "limits": LIMITS, "quantum_backend": "tianyan_sw",
                      "requires": "explicit authenticated provider and registered main window"}, indent=2))
