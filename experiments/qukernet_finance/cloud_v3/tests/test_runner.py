"""Offline runner checks: no authentication, SSH, SDK, or cloud network calls."""
import copy
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "cloud_v2/tests"))
from cloud_v2 import ledger as legacy
from cloud_v2.cloud import validate_raw_result
from cloud_v3 import runner
from cloud_v3.data import save_window
from cloud_v3.ledger import PersistentLedger, atomic_json, digest
from test_authorized_window_integration import VirtualClock, fixture_evidence


def window_fixture():
    rng = np.random.default_rng(909)
    arrays, metadata = {}, {}
    for split, n in (("train", 16), ("valid", 16), ("test", 50), ("full_train", 20)):
        arrays["x_"+split] = rng.normal(size=(n, 8))
        arrays["y_"+split] = np.resize([0, 1], n)
        rows = []
        for i in range(n):
            date = "2024-01-%02d" % (i//5+1) if split == "test" else ("2023-10-02" if split == "valid" else "2023-01-03")
            end = "2024-01-20" if split == "test" else ("2023-10-11" if split == "valid" else "2023-01-12")
            rows.append({"symbol": "stock%03d" % i, "date": date, "label_start": date, "label_end": end})
        metadata["meta_"+split] = rows
    metadata["audit"] = {"window_id": "fixture_2024", "task": "classification",
                         "boundaries_verified": True, "latest_date": "2024-02-01",
                         "split": {"validation_start": "2023-10-01", "evaluation_start": "2024-01-01"}}
    metadata["preprocessing"] = {"fixture": True}
    from cloud_v3.data import digest as data_digest
    metadata["preprocessing_hash"] = data_digest(metadata["preprocessing"])
    metadata["universe"] = {"selected_symbols": [r["symbol"] for r in metadata["meta_test"]],
                            "O_assignments": copy.deepcopy(metadata["meta_test"])}
    return {**arrays, **metadata, "metadata": metadata}


def cutoff_fixture():
    """Synthetic data/provenance matching the approved dates; no source access."""
    from cloud_v3.data import digest as data_digest
    protocol = json.loads((CODE / "cloud_v3/protocol_cutoff_20260325.json").read_text(encoding="utf-8"))
    window = window_fixture()
    meta = window["metadata"]
    fixed = protocol["data"]["fixed_dates"]
    for split in ("train", "full_train", "valid", "test"):
        for i, row in enumerate(meta["meta_" + split]):
            day = (fixed["evaluation_dates"][i // 5] if split == "test" else
                   "2025-09-15" if split == "valid" else "2025-08-01")
            row.update(date=day, label_start=day,
                       label_end="2026-03-20" if split == "test" else "2025-09-24" if split == "valid" else "2025-08-12")
    meta["audit"].update(latest_date="2026-03-25", outer_dates=fixed["evaluation_dates"],
        split={"D": "2026-03-25", "validation_start": "2025-09-03", "evaluation_start": "2025-12-09"})
    meta["universe"].update(qualification_anchor="2025-09-02", O_assignments=copy.deepcopy(meta["meta_test"]))
    records = [{"symbol": s, "first_observed_date": "2019-01-02", "last_observed_date": "2026-03-25",
                "observed_valid_dates": ["2019-01-02", "2026-03-25"],
                "used_interval": {"start": "2019-01-01", "end": "2026-03-25", "rows": 100, "sha256": "a"*64}}
               for s in meta["universe"]["selected_symbols"]]
    manifest = {"requested_cutoff_date": "2026-03-25", "latest_date": "2026-03-25", "securities": records,
                "used_interval": {"start": "2019-01-01", "end": "2026-03-25", "rows": 5000,
                    "sha256": hashlib.sha256(json.dumps(
                        [{"symbol": r["symbol"], "used_interval": r["used_interval"]} for r in records],
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()}}
    meta["audit"]["source_manifest_sha256"] = data_digest(manifest)
    return window, protocol, manifest


def qlib_fixture():
    from cloud_v3.data import digest as data_digest
    from cloud_v2.features80 import FEATURES
    window, protocol, source = cutoff_fixture()
    protocol["protocol_id"] = runner.QLIB_PROTOCOL_ID
    protocol["data"]["missing_policy"] = {"mode": "fixture_training_median_only"}
    meta = window["metadata"]
    meta.update(protocol_id=runner.QLIB_PROTOCOL_ID, schema_version="cloud_v3_qlib_missing_data_v1",
                missing_policy=copy.deepcopy(protocol["data"]["missing_policy"]))
    meta["audit"].update(protocol_id=runner.QLIB_PROTOCOL_ID, history_span_minimum=121,
        row_eligibility="current_valid_bar_and_121_market_session_span_since_first_valid_bar")
    n = len(window["y_full_train"])
    meta["preprocessing"] = {"selected_features": list(FEATURES)[:8], "selected_indices": list(range(8)), "imputer": {
        "method": "training_feature_median", "fit_split": "full_train", "fit_rows": n,
        "feature_names": list(FEATURES), "medians": [0.]*80, "finite_counts": [n]*80,
        "missing_counts": [0]*80, "excluded_features": []}}
    meta["preprocessing_hash"] = data_digest(meta["preprocessing"])
    keys = {s: copy.deepcopy(meta["meta_"+s]) for s in ("train", "valid")}
    meta["frozen_samples"] = dict(keys, sha256=data_digest(keys))
    frozen = dict(keys, source_identity="synthetic")
    frozen["sha256"] = data_digest(frozen)
    meta["audit"].update(frozen_samples_sha256=data_digest(keys), frozen_sample_record_sha256=frozen["sha256"])
    meta["audit"]["missingness"] = {}
    for phase in ("train", "valid", "test", "full_train"):
        count = len(window["y_"+phase])
        window["missing_mask_"+phase] = np.zeros((count,8),dtype=bool)
        window["missing_mask80_"+phase] = np.zeros((count,80),dtype=bool)
        window["support_counts_"+phase] = np.full((count,80),120,dtype=np.int16)
        meta["audit"]["missingness"][phase] = {"rows":count,"missing_cells_80":0,
            "selected_missing_cells":0,"rows_with_any_missing_80":0,"raw_features_missing_by_column":[0]*80}
    return window, protocol, source, frozen


class FakeAdapter:
    """Four preflight rows are explicit constructed records, never real shots."""
    def __init__(self, bundle, validator, clock):
        self.bundle, self.ledger, self.result_validator, self.clock = bundle, bundle.ledger, validator, clock
        self.calls = 0

    def execution_identity(self):
        return {"execution_window_kind": "compact_pilot_v3", "fixture": True}

    def reserve(self, specs, op):
        ph = self.bundle.files[runner._inside(self.bundle.root, self.bundle.manifest["protocol"])]
        entry = self.ledger.reserve(len(specs), 512, "main", hashlib.sha256(op.encode()).hexdigest(),
            {"circuits": specs, "backend": "tianyan_sw", "protocol_sha256": ph, "context": {"operation_id": op}},
            window_id=self.bundle.manifest["window_id"])
        ids = [entry["attempt_id"] + "-%d" % i for i in range(len(specs))]
        self.ledger.update(entry["attempt_id"], "submitted", job_ids=ids)
        return entry, ids

    def probability(self, specs, shots, phase, context):
        self.calls += 1
        entry, ids = self.reserve(specs, context["operation_id"])
        folder = self.ledger.root / entry["attempt_id"]
        folder.mkdir()
        items, results = [], []
        for i, (spec, check, jid) in enumerate(zip(specs, self.bundle.frozen["checks"], ids)):
            nzero = round(check["expected_p0"]*512)
            zero = [0]*8
            other = ([0, 0, 1, 0, 0, 0, 0, 0] if i == 1 else [1, 1, 0, 0, 0, 0, 0, 0])
            rows = [list(range(1, 9))] + [zero[:]]*nzero + [other[:]]*(512-nzero)
            item = {"experimentTaskId": jid, "status": 0, "resultStatus": rows}
            items.append(item)
            result = {"id": spec["id"], "job_id": jid, **validate_raw_result(item, jid, spec, shots),
                      "evidence": {"directory": str(folder)}}
            results.append(result)
        atomic_json(folder / "query_00001.json", {"code": 0, "data": {"experimentResultModelList": items}})
        self.result_validator(results)
        self.ledger.update(entry["attempt_id"], "validated")
        return results


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clock = VirtualClock()
        self.scope = "offline-runner-account"
        prior = legacy.PersistentLedger(self.root / "ledger", self.scope, clock=self.clock.wall)
        self.ledger = PersistentLedger(self.root / "ledger", self.scope, clock=self.clock.wall)
        self.window = window_fixture()
        self.saved = save_window(self.window, self.root / "data", stem="fixture")
        (self.root / "protocol.json").write_bytes((CODE / "cloud_v3/protocol.json").read_bytes())
        (self.root / "auth.py").write_text("def authenticate(key,deadline):\n    raise AssertionError('never authenticate in tests')\n", encoding="utf-8")
        evidence = fixture_evidence(self.ledger, self.clock)
        evidence["capabilities"]["approved_main_shots_limit"] = 512
        atomic_json(self.root / "fee.json", evidence)
        self.manifest = {"schema": runner.SCHEMA, "files": {}, "data_npz": "data/fixture.npz",
                         "protocol": "protocol.json", "auth_helper": "auth.py", "fee_evidence": "fee.json",
                         "account_scope": self.scope, "account_ledger_root": str(self.root / "ledger"),
                         "ledger_sha256_before": runner.sha(prior.path), "output": "results",
                         "window_id": "fixture-v3", "preflight_sha256": runner.frozen_preflight()["sha256"]}
        for name in ("data/fixture.npz", "data/fixture.json", "protocol.json", "auth.py", "fee.json"):
            self.manifest["files"][name] = runner.sha(self.root / name)
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest()
        self.auth_calls = 0
        self.study_calls = 0
        self.adapter = None

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        atomic_json(self.manifest_path, self.manifest)
        self.manifest_hash = runner.sha(self.manifest_path)

    def verify(self, runtime=()):
        return runner.verify_manifest(self.root, "manifest.json", self.manifest_hash,
                                      _runtime_files=runtime, _clock=self.clock.wall, _mock=True)

    def services(self, bundle, study=None, auth_failure=False):
        def authenticate(key, end):
            self.auth_calls += 1
            self.assertEqual(key, "fixture-secret-do-not-log")
            if auth_failure:
                raise ValueError("fixture-secret-do-not-log")
            return object()
        def factory(platform, validator):
            self.adapter = FakeAdapter(bundle, validator, self.clock)
            return self.adapter
        @contextmanager
        def guard(end):
            self.guard_end = end
            yield
            if self.clock.mono() >= end:
                raise TimeoutError("mock global deadline")
        def incomplete(window, output, provider, **kwargs):
            self.study_calls += 1
            return {"workflow_status": "complete"}
        return runner._Services(authenticate, factory, study or incomplete, guard, self.clock.mono)

    def test_disabled_does_not_read_stdin_or_verify(self):
        class Forbidden:
            def read(self, *a): raise AssertionError("credential read")
        with patch.object(runner, "verify_manifest", side_effect=AssertionError("verification")), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main([], stdin=Forbidden()), 0)
        self.assertEqual(json.loads(stream.getvalue())["state"], "disabled")

    def test_malformed_stdin_never_loads_authentication_or_starts_clock(self):
        bundle = self.verify()
        argv = ["--execute", "--root", str(self.root), "--manifest", "manifest.json", "--manifest-sha256", self.manifest_hash]
        with patch.object(runner, "_resource_limits"), patch.object(runner, "verify_manifest", return_value=bundle), \
             patch.object(runner, "_production_services", side_effect=AssertionError("authentication helper must not load")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(argv, stdin=io.StringIO("{broken")), 2)
        self.assertEqual(self.ledger.snapshot().get("execution_windows", []), [])

    def test_analytic_probability_and_wrong_bit_position_rejected(self):
        bundle = self.verify()
        bundle.ledger.authorize_main_window("fixture-v3", bundle.files[(self.root / "protocol.json").resolve()], runner.AUTHORIZATION)
        validator = runner._analytic_validator(bundle.frozen, bundle.ledger.root)
        adapter = FakeAdapter(bundle, validator, self.clock)
        values = adapter.probability([c["spec"] for c in bundle.frozen["checks"]], 512, "main", {"operation_id": "analytic-test"})
        wrong_probability = copy.deepcopy(values)
        wrong_probability[2]["zero_count"] = 512
        with self.assertRaises(runner.LaunchBlocked): validator(wrong_probability)
        folder = Path(values[1]["evidence"]["directory"])
        envelope = json.loads((folder / "query_00001.json").read_text())
        item = envelope["data"]["experimentResultModelList"][1]
        item["resultStatus"][1:] = [[0, 0, 0, 1, 0, 0, 0, 0] for _ in range(512)]
        atomic_json(folder / "query_00001.json", envelope)
        values[1].update(validate_raw_result(item, values[1]["job_id"], bundle.frozen["checks"][1]["spec"], 512))
        with self.assertRaises(runner.LaunchBlocked): validator(values)

    def test_valid_manifest_is_read_only_and_does_not_start_clock(self):
        before = self.ledger.path.read_bytes()
        bundle = self.verify()
        self.assertEqual(bundle.ledger.path.read_bytes(), before)
        self.assertFalse((self.root / "results").exists())
        self.assertEqual(bundle.ledger.snapshot().get("execution_windows", []), [])

    def test_cache_limit_blocks_before_auth_or_reservation(self):
        with patch.object(runner, "CACHE_LIMIT_BYTES", 1):
            with self.assertRaises(runner.LaunchBlocked): self.verify()
        self.assertEqual(self.ledger.snapshot().get("execution_windows", []), [])
        bundle = self.verify()
        provider = FakeAdapter(bundle, lambda x: None, self.clock)
        wrapper = runner._CacheBoundProvider(provider, bundle)
        with patch.object(runner, "_cache_bytes", return_value=runner.CACHE_LIMIT_BYTES):
            with self.assertRaises(runner.LaunchBlocked): wrapper.probability([], 512, "main", {})
        self.assertEqual(provider.calls, 0)

    def test_current_account_grant_rejects_scope_expiry_bits_and_evidence_mode(self):
        from datetime import datetime
        ui={"source":"authenticated_official_account_dashboard",
            "source_url":"https://qc.zdxlz.com/laboratory/#/dashboard?lang=zh",
            "backend":"tianyan_sw","type":"simulator","grant":"free_gift","remaining_time":"unlimited",
            "account_allowed_qubits":36,"raw_credentials_recorded":False,
            "observed_at":"2026-11-15T12:00:00+08:00","grant_start":"2026-10-26T08:00:04+08:00",
            "grant_expiry":"2026-11-26T08:00:04+08:00"}
        grant={"schema":"compact_account_ui_grant_v1","account_scope_sha256":"a"*64,"ui_sha256":"b"*64,
            "evidence_mode":"official_ui_snapshot_validity_check","live_account_api":False}
        now=datetime.fromisoformat("2026-11-15T13:00:00+08:00").timestamp()
        check=runner._validate_account_grant(grant,ui,"a"*64,"b"*64,now)
        self.assertEqual(check['remaining_seconds'],'not_applicable_unlimited')
        self.assertFalse(check['live_account_api'])
        for field,value in (("backend","tianyan_s"),("account_allowed_qubits",4),
                            ("grant_expiry","2026-11-15T12:30:00+08:00"),("remaining_time",0)):
            changed=dict(ui);changed[field]=value
            with self.subTest(field=field),self.assertRaises(runner.LaunchBlocked):
                runner._validate_account_grant(grant,changed,"a"*64,"b"*64,now)
        for field,value in (("account_scope_sha256","c"*64),("ui_sha256","c"*64),("live_account_api",True)):
            changed=dict(grant);changed[field]=value
            with self.subTest(field=field),self.assertRaises(runner.LaunchBlocked):
                runner._validate_account_grant(changed,ui,"a"*64,"b"*64,now)

    def test_changed_ui_and_failed_provider_are_never_retried(self):
        bundle=self.verify()
        bundle.manifest.update(current_account_grant='grant.json',current_account_ui='ui.json')
        for name in ('grant.json','ui.json'):
            atomic_json(self.root/name,{})
            bundle.files[(self.root/name).resolve()]=runner.sha(self.root/name)
        atomic_json(self.root/'ui.json',{'changed':True})
        with self.assertRaises(runner.LaunchBlocked):runner._account_grant_check(bundle)
        class Failing:
            calls=0
            def probability(self,*a,**k):
                self.calls+=1
                raise TimeoutError('Synthetic HTTP failure')
        provider=Failing();wrapped=runner._CacheBoundProvider(provider,bundle)
        with patch.object(runner,'_account_grant_check',return_value={'live_account_api':False}):
            with self.assertRaises(TimeoutError):wrapped.probability([],512,context={'operation_id':'fixture'})
        self.assertEqual(provider.calls,1)

    def test_runtime_manifest_includes_top_level_modeling(self):
        self.assertIn((CODE / "modeling.py").resolve(), runner.required_runtime_files())

    def test_manifest_hash_and_missing_runtime_and_data_fail_before_auth(self):
        with self.assertRaises(runner.LaunchBlocked):
            runner.verify_manifest(self.root, "manifest.json", "0"*64)
        with self.assertRaises(runner.LaunchBlocked): self.verify(runtime={Path(__file__).resolve()})
        (self.root / "data/fixture.npz").unlink()
        with self.assertRaises(runner.LaunchBlocked): self.verify()
        self.assertEqual(self.auth_calls, 0)
        self.assertEqual(self.ledger.snapshot().get("execution_windows", []), [])

    def test_frozen_auth_helper_and_probe_changes_block(self):
        (self.root / "auth.py").write_text("modified", encoding="utf-8")
        with self.assertRaises(runner.LaunchBlocked): self.verify()
        self.manifest["files"]["auth.py"] = runner.sha(self.root / "auth.py")
        self.manifest["preflight_sha256"] = "0"*64
        self.write_manifest()
        with self.assertRaises(runner.LaunchBlocked): self.verify()

    def test_source_output_and_escape_are_blocked(self):
        with self.assertRaises(runner.LaunchBlocked): runner._private(CODE)
        with self.assertRaises(runner.LaunchBlocked): runner._private(CODE.parent)
        with self.assertRaises(runner.LaunchBlocked): runner._inside(self.root, "../escape")
        self.manifest["output"] = "../escape"
        self.write_manifest()
        with self.assertRaises(runner.LaunchBlocked): self.verify()

    def test_ledger_cannot_be_replaced_by_new_empty_budget(self):
        self.manifest["account_ledger_root"] = str(self.root / "new-ledger")
        self.write_manifest()
        with self.assertRaises(runner.LaunchBlocked): self.verify()
        self.assertFalse((self.root / "new-ledger").exists())

    def test_unresolved_historical_attempt_blocks_authorization(self):
        old = legacy.PersistentLedger(self.root / "ledger", self.scope, clock=self.clock.wall)
        old.reserve(1, 128, "preflight", "old-unknown", {})
        self.manifest["ledger_sha256_before"] = runner.sha(old.path)
        self.write_manifest()
        with self.assertRaises(runner.LaunchBlocked): self.verify()

    def test_data_coverage_and_time_leak_fail(self):
        duplicate = copy.deepcopy(self.window)
        duplicate["metadata"]["meta_test"][1]["symbol"] = duplicate["metadata"]["meta_test"][0]["symbol"]
        with self.assertRaises(runner.LaunchBlocked): runner._validate_data(duplicate)
        leaking = copy.deepcopy(self.window)
        leaking["metadata"]["meta_train"][0]["label_end"] = "2024-01-01"
        with self.assertRaises(runner.LaunchBlocked): runner._validate_data(leaking)

    def test_cutoff_binding_accepts_approved_dates_and_source_interval(self):
        window, protocol, source = cutoff_fixture()
        self.assertEqual(runner._validate_data(window, protocol, source)["splits"]["test"]["rows"], 50)

    def test_qlib_policy_and_frozen_sample_identity_required(self):
        window, protocol, source, frozen = qlib_fixture()
        self.assertEqual(runner._validate_data(window, protocol, source, frozen)["splits"]["test"]["rows"], 50)
        with self.assertRaises(runner.LaunchBlocked):
            runner._validate_data(window, protocol, source)
        for kind in ("fit_split", "fit_rows", "counts", "schema", "history_span", "policy", "keys"):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(window)
                meta = changed["metadata"]
                imputer = meta["preprocessing"]["imputer"]
                if kind == "fit_split": imputer["fit_split"] = "all_data"
                if kind == "fit_rows": imputer["fit_rows"] += 1
                if kind == "counts": imputer["missing_counts"][0] = 1
                if kind == "schema": meta["schema_version"] = "cloud_v3_data_v1"
                if kind == "history_span": meta["audit"]["history_span_minimum"] = 120
                if kind == "policy": meta["missing_policy"] = {}
                if kind == "keys": meta["frozen_samples"]["train"][0]["symbol"] = "changed"
                from cloud_v3.data import digest as data_digest
                meta["preprocessing_hash"] = data_digest(meta["preprocessing"])
                with self.assertRaises(runner.LaunchBlocked):
                    runner._validate_data(changed, protocol, source, frozen)

    def test_qlib_external_frozen_record_cannot_be_replaced(self):
        window, protocol, source, frozen = qlib_fixture()
        altered = copy.deepcopy(frozen)
        altered["train"][0]["symbol"] = "different"
        from cloud_v3.data import digest as data_digest
        altered.pop("sha256")
        altered["sha256"] = data_digest(altered)
        with self.assertRaises(runner.LaunchBlocked): runner._validate_data(window, protocol, source, altered)

    def test_qlib_mask_and_support_cannot_be_mixed_with_legacy_snapshot(self):
        for kind in ("missing_mask", "support", "mask_audit", "selected_indices"):
            window, protocol, source, frozen = qlib_fixture()
            if kind == "missing_mask": window.pop("missing_mask80_test")
            if kind == "support": window["support_counts_test"][0,0] = 121
            if kind == "mask_audit": window["missing_mask80_test"][0,20] = True
            if kind == "selected_indices": window["metadata"]["preprocessing"]["selected_indices"][0] = 79
            from cloud_v3.data import digest as data_digest
            window["metadata"]["preprocessing_hash"] = data_digest(window["metadata"]["preprocessing"])
            with self.subTest(kind=kind), self.assertRaises(runner.LaunchBlocked):
                runner._validate_data(window, protocol, source, frozen)

    def test_unregistered_protocol_is_blocked(self):
        protocol = json.loads((self.root/"protocol.json").read_text())
        protocol["protocol_id"] = "unapproved-v99"
        atomic_json(self.root/"protocol.json", protocol)
        self.manifest["files"]["protocol.json"] = runner.sha(self.root/"protocol.json")
        self.write_manifest()
        with self.assertRaises(runner.LaunchBlocked): self.verify()

    def test_cutoff_binding_rejects_wrong_latest_anchor_and_dates(self):
        original, protocol, source = cutoff_fixture()
        for section, key, value in (("audit", "latest_date", "2026-11-13"),
                                    ("universe", "qualification_anchor", "2025-09-01"),
                                    ("audit", "outer_dates", ["2025-12-11"]*10)):
            with self.subTest(key=key):
                window = copy.deepcopy(original)
                window["metadata"][section][key] = value
                with self.assertRaises(runner.LaunchBlocked): runner._validate_data(window, protocol, source)
        window = copy.deepcopy(original)
        window["metadata"]["meta_train"][0]["date"] = "2025-08-26"
        with self.assertRaises(runner.LaunchBlocked): runner._validate_data(window, protocol, source)

    def test_cutoff_binding_rejects_missing_or_inconsistent_source_interval(self):
        from cloud_v3.data import digest as data_digest
        original, protocol, source = cutoff_fixture()
        with self.assertRaises(runner.LaunchBlocked): runner._validate_data(original, protocol)
        for change in ("source_cutoff", "interval_end", "future_row", "used_hash", "missing_stock"):
            with self.subTest(change=change):
                window, changed = copy.deepcopy(original), copy.deepcopy(source)
                if change == "source_cutoff": changed["requested_cutoff_date"] = "2026-11-13"
                if change == "interval_end": changed["securities"][0]["used_interval"]["end"] = "2026-11-13"
                if change == "future_row": changed["securities"][0]["observed_valid_dates"].append("2026-03-26")
                if change == "used_hash": changed["used_interval"]["sha256"] = "b"*64
                if change == "missing_stock": changed["securities"].pop()
                # Even correctly re-hashed metadata cannot change the authorized cutoff.
                window["metadata"]["audit"]["source_manifest_sha256"] = data_digest(changed)
                with self.assertRaises(runner.LaunchBlocked): runner._validate_data(window, protocol, changed)

    def test_cutoff_protocol_manifest_is_verified_without_auth_or_reservation(self):
        window, protocol, source = cutoff_fixture()
        save_window(window, self.root / "data", source_manifest=source, stem="cutoff")
        (self.root / "cutoff_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        # Bind the exact versioned approved bytes, not just equivalent JSON content.
        (self.root / "cutoff_protocol.json").write_bytes((CODE / "cloud_v3/protocol_cutoff_20260325.json").read_bytes())
        self.manifest.update(data_npz="data/cutoff.npz", protocol="cutoff_protocol.json")
        for name in ("data/cutoff.npz", "data/cutoff.json", "cutoff_protocol.json"):
            self.manifest["files"][name] = runner.sha(self.root / name)
        self.write_manifest()
        before = self.ledger.path.read_bytes()
        bundle = self.verify()
        self.assertEqual(bundle.window["metadata"]["audit"]["latest_date"], "2026-03-25")
        self.assertEqual(self.ledger.path.read_bytes(), before)
        self.assertEqual(self.auth_calls, 0)
        self.assertFalse((self.root / "results").exists())

    def test_degenerate_full_training_columns_block_without_replacement(self):
        zero = copy.deepcopy(self.window)
        zero["x_full_train"][:, 3] = 0
        with self.assertRaises(runner.DataAdmissionFailure): runner._validate_data(zero)
        iqr = copy.deepcopy(self.window)
        iqr["metadata"]["preprocessing"]["scaler"] = {"zero_iqr_columns": [2]}
        with self.assertRaises(runner.DataAdmissionFailure): runner._validate_data(iqr)
        small = copy.deepcopy(self.window)
        small["x_train"][:, 2] = 0
        self.assertEqual(runner._validate_data(small)["splits"]["train"]["varying_columns"], 7)

    def test_auth_failure_is_redacted_and_does_not_start_window(self):
        bundle = self.verify()
        with self.assertRaises(runner.LaunchBlocked):
            runner._run_verified(bundle, "fixture-secret-do-not-log", self.services(bundle, auth_failure=True))
        self.assertEqual(bundle.ledger.snapshot().get("execution_windows", []), [])
        for path in (self.root / "results").glob("*.json"):
            self.assertNotIn("fixture-secret-do-not-log", path.read_text(encoding="utf-8"))
        with self.assertRaises(runner.LaunchBlocked): self.verify()

    def test_incomplete_quota_cannot_be_reported_complete(self):
        bundle = self.verify()
        with self.assertRaises(runner.LaunchBlocked):
            runner._run_verified(bundle, "fixture-secret-do-not-log", self.services(bundle))
        self.assertEqual(self.adapter.calls, 1)
        self.assertEqual(bundle.ledger.window("fixture-v3")["state"], "closed")
        status = json.loads((self.root / "results/runner_status.json").read_text())
        self.assertEqual(status["state"], "client_stopped")
        self.assertFalse(status["backend_cancelled"])

    def test_keyboard_interrupt_marks_pending_unknown_without_query(self):
        bundle = self.verify()
        def stop(window, output, provider, **kwargs):
            provider.reserve([bundle.frozen["checks"][0]["spec"]], "mock-interrupted-study")
            raise KeyboardInterrupt()
        with self.assertRaises(runner.LaunchBlocked):
            runner._run_verified(bundle, "fixture-secret-do-not-log", self.services(bundle, stop))
        entries = bundle.ledger.snapshot()["entries"]
        self.assertEqual([e["state"] for e in entries], ["validated", "observation_unknown"])
        self.assertEqual(self.adapter.calls, 1)
        self.assertTrue(json.loads((self.root / "results/runner_status.json").read_text())["unresolved_attempt_ids"])

    def test_global_deadline_includes_classical_phase(self):
        bundle = self.verify()
        def expensive_classical(window, output, provider, **kwargs):
            self.clock.advance(10801)
            raise TimeoutError("classical calculations expired")
        with self.assertRaises(runner.LaunchBlocked):
            runner._run_verified(bundle, "fixture-secret-do-not-log", self.services(bundle, expensive_classical))
        self.assertEqual(self.guard_end, 10800)
        self.assertEqual(len(bundle.ledger.snapshot()["entries"]), 1)
        self.assertEqual(bundle.ledger.window("fixture-v3")["state"], "closed")

    def test_same_bundle_cannot_launch_twice(self):
        bundle = self.verify()
        service = self.services(bundle, auth_failure=True)
        with self.assertRaises(runner.LaunchBlocked): runner._run_verified(bundle, "fixture-secret-do-not-log", service)
        with self.assertRaises(FileExistsError): runner._run_verified(bundle, "fixture-secret-do-not-log", service)
        self.assertEqual(self.auth_calls, 1)

    def test_complete_requires_exact_totals_and_preserves_history(self):
        bundle = self.verify()
        def finish(window, output, provider, **kwargs):
            # Synthetic quota fixture; no actual quantum computation is claimed.
            for i, n in enumerate([20]*258 + [10]*11 + [18]):
                specs = [{"id": f"fixture-{i}-{j}", "qubits": 8, "sha256": "a"*64} for j in range(n)]
                entry, _ = provider.reserve(specs, "fixture-study-%d" % i)
                provider.ledger.update(entry["attempt_id"], "validated")
            return {"workflow_status": "complete"}
        status = runner._run_verified(bundle, "fixture-secret-do-not-log", self.services(bundle, finish))
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["counts"], {"circuits": 5292, "shots": 2709504, "batches": 271})
        self.assertEqual(status["window"]["deadline_epoch"]-status["window"]["started_epoch"], 10800)
        self.assertEqual(status["unresolved_attempt_ids"], [])


if __name__ == "__main__":
    unittest.main()
