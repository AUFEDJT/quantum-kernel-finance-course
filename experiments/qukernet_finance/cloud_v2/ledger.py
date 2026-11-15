"""Persistent account-scoped reservation ledger; no network or credentials.

All clients for one account MUST use the same private ``account_ledger_root``.
An OS lock serializes cooperating processes on that host. This cannot police an
unrelated application or another host; preflight requires reconciliation of
those jobs. Rejected, uncertain and failed attempts all consume reservations.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

HARD_LIMITS = {"seconds": 72 * 3600, "shots": 10_000_000_000,
               "circuits": 80_000_000, "batch": 20,
               "preflight_circuits": 50, "preflight_shots": 20_000}
TERMINAL = {"validated", "terminal_failure", "submission_rejected"}
MAIN_POLICY = {"backend": "tianyan_sw", "phase": "main", "circuits": 93760,
               "shots": 48005120, "shots_per_circuit": 512, "qubits": 8,
               "max_attempts": 4804, "batch": 20, "seconds": 72 * 3600,
               "observation_seconds": 300}
_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LedgerError(RuntimeError): pass
class BudgetExceeded(LedgerError): pass
class UnresolvedTask(LedgerError): pass
class DuplicateSubmission(LedgerError): pass


def utc(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def request_identity(specs, shots, backend, phase):
    """Submission identity deliberately excludes caller context and run labels."""
    return digest({"circuits": [(s["id"], s["sha256"]) for s in specs],
                   "shots": shots, "backend": backend, "phase": phase})


def public_source_root():
    """Course checkout uses experiments/; deployed snapshot uses code/."""
    source = Path(__file__).resolve().parents[2]
    return source.parent if source.name == "experiments" else source


def private_directory(root):
    """Reject network/public destinations BEFORE any mkdir or lock-file write."""
    raw = os.fspath(root)
    if raw.startswith(("\\\\", "//")):
        raise LedgerError("Private output must be a local directory")
    path = Path(root).expanduser().resolve()
    public = public_source_root()
    if path == public or public in path.parents:
        raise LedgerError("Private output must be outside the public course tree")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 4:
            raise LedgerError("Mapped network output is forbidden")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8")
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("xb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            fd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


@contextmanager
def file_lock(path):
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        guard = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not guard.acquire(blocking=False):
        raise UnresolvedTask("Another account operation holds the global lock")
    handle = None
    locked = False
    try:
        handle = path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, BlockingIOError) as exc:
            raise UnresolvedTask("Another process holds the account lock") from exc
        yield
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        guard.release()


class PersistentLedger:
    def __init__(self, account_ledger_root, account_scope, limits=None, clock=time.time):
        if not isinstance(account_scope, str) or not account_scope.strip():
            raise LedgerError("A stable non-secret account scope is required")
        self.account_hash = hashlib.sha256(account_scope.encode()).hexdigest()
        base = private_directory(account_ledger_root)
        self.root = base / ("account-" + self.account_hash)
        if self.root.is_symlink() or self.root.resolve().parent != base:
            raise LedgerError("Account directory may not redirect outside the private ledger root")
        self.root.mkdir(exist_ok=True)
        self.path = self.root / "ledger.json"
        self.clock = clock
        self.limits = dict(HARD_LIMITS)
        for key, value in (limits or {}).items():
            if key not in HARD_LIMITS or type(value) is not int or not 0 < value <= HARD_LIMITS[key]:
                raise LedgerError("Limits may only reduce the approved positive integer caps")
            self.limits[key] = value
        with file_lock(self.root / "ledger.lock"):
            if self.path.exists():
                self._read()
            else:
                now = clock()
                atomic_json(self.path, {"schema_version": 1, "account_scope_sha256": self.account_hash,
                    "started_epoch": now, "started_utc": utc(now),
                    "deadline_epoch": now + self.limits["seconds"], "limits": self.limits,
                    "entries": [], "scope": "all cooperating clients using this canonical account ledger"})

    def _read(self):
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (value.get("schema_version") != 1 or value.get("account_scope_sha256") != self.account_hash
                or value.get("limits") != self.limits or not isinstance(value.get("entries"), list)):
            raise LedgerError("Ledger schema, account or frozen limits mismatch")
        if value.get("deadline_epoch") != value.get("started_epoch", 0) + self.limits["seconds"]:
            raise LedgerError("Ledger deadline was changed")
        ids = [e.get("attempt_id") for e in value["entries"]]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
            raise LedgerError("Duplicate or malformed ledger attempts")
        for e in value["entries"]:
            if (type(e.get("circuits")) is not int or e["circuits"] <= 0
                    or type(e.get("shots_per_circuit")) is not int or e["shots_per_circuit"] <= 0
                    or e.get("requested_shots") != e["circuits"] * e["shots_per_circuit"]):
                raise LedgerError("Malformed reserved resource counters")
        windows = value.get("execution_windows", [])
        if not isinstance(windows, list) or len({w.get("window_id") for w in windows}) != len(windows):
            raise LedgerError("Malformed execution windows")
        for w in windows:
            if w.get("kind") == "main_protocol_v1":
                self._validate_main_window(w, value["entries"])
                continue
            if (w.get("policy") != {"backend": "tianyan_sw", "phase": "preflight", "circuits": 12,
                    "shots_per_circuit": 128, "qubits": 8, "max_attempts": 1, "seconds": 72 * 3600,
                    "observation_seconds": 300}
                    or w.get("state") not in {"authorized", "reserved", "closed"}
                    or not isinstance(w.get("window_id"), str) or not w["window_id"]):
                raise LedgerError("Execution window policy mismatch")
            related = [e for e in value["entries"] if e.get("execution_window_id") == w["window_id"]]
            unstarted_closed = (w["state"] == "closed"
                                and w.get("outcome") == "preflight_blocked_before_reservation")
            if w["state"] == "authorized" or unstarted_closed:
                if (related or w.get("started_epoch") is not None or w.get("deadline_epoch") is not None
                        or w.get("attempt_id") is not None):
                    raise LedgerError("Unstarted window has an attempt or running clock")
            else:
                if (len(related) != 1 or w.get("attempt_id") != related[0]["attempt_id"]
                        or w.get("deadline_epoch") != w.get("started_epoch", 0) + w["policy"]["seconds"]
                        or related[0].get("reserved_epoch") != w["started_epoch"]
                        or related[0].get("observation_deadline_epoch") != w["started_epoch"] + 300):
                    raise LedgerError("Execution window clock or attempt binding changed")
        return value

    @staticmethod
    def _validate_main_window(w, entries):
        if (w.get("policy") != MAIN_POLICY or w.get("state") not in {"authorized", "running", "closed"}
                or not isinstance(w.get("window_id"), str) or not w["window_id"]
                or not isinstance(w.get("protocol_sha256"), str)
                or len(w["protocol_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in w["protocol_sha256"])):
            raise LedgerError("Main execution window policy or protocol binding changed")
        related = [e for e in entries if e.get("execution_window_id") == w["window_id"]]
        if not related:
            if w["state"] == "running" or w.get("started_epoch") is not None or w.get("deadline_epoch") is not None:
                raise LedgerError("Unstarted main window cannot have a running clock")
            return
        if (w["state"] == "authorized" or w.get("started_epoch") != related[0].get("reserved_epoch")
                or w.get("deadline_epoch") != w["started_epoch"] + MAIN_POLICY["seconds"]):
            raise LedgerError("Main window clock or first reservation changed")
        operations = []
        for e in related:
            meta = e.get("metadata", {})
            if (e.get("phase") != "main" or e["shots_per_circuit"] != 512 or e["circuits"] > 20
                    or meta.get("backend") != "tianyan_sw" or meta.get("protocol_sha256") != w["protocol_sha256"]
                    or len(meta.get("circuits", [])) != e["circuits"]
                    or any(s.get("qubits") != 8 for s in meta["circuits"])
                    or not isinstance(e.get("operation_id"), str) or not e["operation_id"].strip()
                    or e.get("observation_deadline_epoch") != e.get("reserved_epoch", 0) + 300
                    or e["reserved_epoch"] >= w["deadline_epoch"]
                    or e.get("request_identity") != request_identity(meta["circuits"], 512, "tianyan_sw", "main")):
                raise LedgerError("Main reservation constraints or logical operation binding changed")
            operations.append(e["operation_id"])
        if (len(set(operations)) != len(operations) or len(related) > MAIN_POLICY["max_attempts"]
                or sum(e["circuits"] for e in related) > MAIN_POLICY["circuits"]
                or sum(e["requested_shots"] for e in related) > MAIN_POLICY["shots"]):
            raise LedgerError("Main window reservations exceed frozen caps or repeat an operation")

    def authorize_main_window(self, window_id, protocol_sha256, authorization_note):
        """Register the fixed three-quarter protocol; start clocks only on first reserve."""
        if (not isinstance(window_id, str) or not window_id.strip()
                or not isinstance(authorization_note, str) or not authorization_note.strip()
                or not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64
                or any(c not in "0123456789abcdef" for c in protocol_sha256)):
            raise LedgerError("Main window requires an ID, SHA256 protocol binding and explicit authorization")
        declaration = {"kind": "main_protocol_v1", "window_id": window_id,
                       "protocol_sha256": protocol_sha256, "authorization_note": authorization_note,
                       "policy": dict(MAIN_POLICY)}
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            windows = data.setdefault("execution_windows", [])
            same = [w for w in windows if w["window_id"] == window_id]
            if same:
                if any(same[0].get(k) != v for k, v in declaration.items()):
                    raise LedgerError("Existing main authorization cannot be replaced")
                return same[0]
            if any(w["state"] != "closed" for w in windows):
                raise UnresolvedTask("An earlier execution window is still open")
            if any(e.get("state") not in TERMINAL for e in data["entries"]):
                raise UnresolvedTask("Unresolved historical reservations block new authorization")
            window = {**declaration, "state": "authorized", "authorized_utc": utc(self.clock()),
                      "started_epoch": None, "deadline_epoch": None}
            windows.append(window)
            atomic_json(self.path, data)
            return window

    def close_main_window(self, window_id, outcome, reason):
        """Stop a main protocol without cancelling or refunding its cloud tasks."""
        if not isinstance(outcome, str) or not outcome or not isinstance(reason, str) or not reason:
            raise LedgerError("Window closure requires an outcome and reason")
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            found = [w for w in data.get("execution_windows", []) if w["window_id"] == window_id]
            if len(found) != 1 or found[0].get("kind") != "main_protocol_v1":
                raise LedgerError("Main execution window not found")
            w = found[0]
            if w["state"] == "closed":
                return w
            related = [e for e in data["entries"] if e.get("execution_window_id") == window_id]
            if outcome == "completed" and (not related or any(e["state"] != "validated" for e in related)):
                raise UnresolvedTask("Cannot mark unstarted or unresolved main work complete")
            w.update(state="closed", outcome=outcome, reason=reason, closed_utc=utc(self.clock()),
                     client_execution_stopped=True, backend_cancelled=False)
            atomic_json(self.path, data)
            return w

    def window(self, window_id):
        found = [w for w in self.snapshot().get("execution_windows", []) if w["window_id"] == window_id]
        if len(found) != 1:
            raise LedgerError("Explicitly authorized execution window not found")
        return found[0]

    def authorize_window(self, window_id, request_file_sha256, request_fingerprint, authorization_note):
        """Append an explicit one-batch authorization; this does not start either clock.

        Caller must already have the user's authorization. This method cannot
        enlarge global budgets or reopen an earlier attempt. Repeating the exact
        declaration is idempotent; a differing declaration is rejected.
        """
        if (not isinstance(window_id, str) or not window_id.strip()
                or not isinstance(authorization_note, str) or not authorization_note.strip()):
            raise LedgerError("A window ID and explicit authorization note are required")
        for h in (request_file_sha256, request_fingerprint):
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise LedgerError("Frozen request digests must be SHA256 hex strings")
        declaration = {"window_id": window_id, "request_file_sha256": request_file_sha256,
            "request_fingerprint": request_fingerprint, "authorization_note": authorization_note,
            "policy": {"backend": "tianyan_sw", "phase": "preflight", "circuits": 12,
                       "shots_per_circuit": 128, "qubits": 8, "max_attempts": 1, "seconds": 72 * 3600,
                       "observation_seconds": 300}}
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            windows = data.setdefault("execution_windows", [])
            same = [w for w in windows if w["window_id"] == window_id]
            if same:
                if any(same[0].get(k) != v for k, v in declaration.items()):
                    raise LedgerError("Existing authorization differs; cannot replace it")
                return same[0]
            if any(w.get("request_fingerprint") == request_fingerprint for w in windows):
                raise DuplicateSubmission("Frozen request already has an authorized window")
            if any(w["state"] != "closed" for w in windows):
                raise UnresolvedTask("An existing authorization window is still open")
            if any(e.get("state") not in TERMINAL for e in data["entries"]):
                raise UnresolvedTask("Reconcile previous attempts before adding an execution window")
            window = {**declaration, "state": "authorized", "authorized_utc": utc(self.clock()),
                      "started_epoch": None, "deadline_epoch": None}
            windows.append(window)
            atomic_json(self.path, data)
            return window

    def close_unstarted_window(self, window_id, reason):
        """Close a pre-reservation failure without inventing an attempt or starting clocks."""
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError("Pre-reservation stop requires a reason")
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            found = [w for w in data.get("execution_windows", []) if w["window_id"] == window_id]
            if len(found) != 1:
                raise LedgerError("Explicitly authorized execution window not found")
            w = found[0]
            if (w.get("started_epoch") is not None or w.get("deadline_epoch") is not None
                    or w.get("attempt_id") is not None
                    or any(e.get("execution_window_id") == window_id for e in data["entries"])):
                raise LedgerError("A reserved window cannot be closed as unstarted")
            if w["state"] == "closed":
                if w.get("outcome") != "preflight_blocked_before_reservation":
                    raise LedgerError("Window already closed with a different outcome")
                return w
            if w["state"] != "authorized":
                raise LedgerError("Only an unstarted authorization may close without reservation")
            w.update(state="closed", closed_utc=utc(self.clock()), reason=reason,
                     outcome="preflight_blocked_before_reservation",
                     client_execution_stopped=True, backend_cancelled=False)
            atomic_json(self.path, data)
            return w

    def effective_deadline(self, window_id=None):
        """Read the selected clock without altering the legacy account window."""
        data = self.snapshot()
        if window_id is None:
            return data["deadline_epoch"]
        found = [w for w in data.get("execution_windows", []) if w["window_id"] == window_id]
        if len(found) != 1:
            raise LedgerError("Explicitly authorized execution window not found")
        if found[0].get("kind") == "main_protocol_v1":
            if found[0]["state"] == "closed":
                raise DuplicateSubmission("Main execution window is closed; do not submit or poll")
            return found[0]["deadline_epoch"]
        if found[0]["state"] != "authorized":
            raise DuplicateSubmission("Window already reserved or closed; do not resubmit")
        return found[0]["deadline_epoch"]

    @contextmanager
    def execution(self):
        """Hold across reservation, submission, polling and returned-circuit checks."""
        with file_lock(self.root / "execution.lock"):
            yield

    def snapshot(self):
        with file_lock(self.root / "ledger.lock"):
            return self._read()

    def reserve(self, circuits, shots, phase, fingerprint, metadata, window_id=None):
        if type(circuits) is not int or not 1 <= circuits <= self.limits["batch"]:
            raise BudgetExceeded("Empty batch or approved batch limit exceeded")
        if type(shots) is not int or shots <= 0 or phase not in ("main", "preflight"):
            raise LedgerError("Invalid shots or phase")
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            now = self.clock()
            window = None
            main_window = False
            operation_id = None
            if window_id is not None:
                found = [w for w in data.get("execution_windows", []) if w["window_id"] == window_id]
                if len(found) != 1:
                    raise LedgerError("Explicitly authorized execution window not found")
                window = found[0]
                main_window = window.get("kind") == "main_protocol_v1"
                if main_window:
                    if window["state"] not in {"authorized", "running"}:
                        raise DuplicateSubmission("Main execution window is closed; no retry")
                    operation_id = metadata.get("context", {}).get("operation_id")
                    if (phase != "main" or shots != 512 or metadata.get("backend") != "tianyan_sw"
                            or metadata.get("protocol_sha256") != window["protocol_sha256"]
                            or len(metadata.get("circuits", [])) != circuits
                            or any(s.get("qubits") != 8 for s in metadata.get("circuits", []))
                            or not isinstance(operation_id, str) or not operation_id.strip()):
                        raise BudgetExceeded("Main request differs from its frozen protocol or lacks a logical operation ID")
                    prior_operations = [e for e in data["entries"] if e.get("operation_id") == operation_id]
                    if prior_operations:
                        same_content = prior_operations[0].get("request_identity") == request_identity(
                            metadata["circuits"], shots, metadata["backend"], phase)
                        raise DuplicateSubmission("Logical operation already reserved; use prior evidence" if same_content
                                                  else "Logical operation content changed; resubmission forbidden")
                    previous = [e for e in data["entries"] if e.get("execution_window_id") == window_id]
                    if (len(previous) + 1 > MAIN_POLICY["max_attempts"]
                            or sum(e["circuits"] for e in previous) + circuits > MAIN_POLICY["circuits"]
                            or sum(e["requested_shots"] for e in previous) + circuits * shots > MAIN_POLICY["shots"]):
                        raise BudgetExceeded("Main protocol reservation cap exceeded")
                    if window.get("deadline_epoch") is not None and now >= window["deadline_epoch"]:
                        raise BudgetExceeded("Main protocol 72-hour execution window expired")
                elif window["state"] != "authorized":
                    raise DuplicateSubmission("Execution window already consumed; no retry")
                identity = request_identity(metadata.get("circuits", []), shots, metadata.get("backend"), phase)
                if not main_window and (circuits != 12 or shots != 128 or phase != "preflight"
                        or metadata.get("backend") != "tianyan_sw"
                        or len(metadata.get("circuits", [])) != 12
                        or any(s.get("qubits") != 8 for s in metadata.get("circuits", []))
                        or identity != window["request_fingerprint"]):
                    raise BudgetExceeded("Request differs from this one-batch authorization")
            if window is None and now >= data["deadline_epoch"]:
                raise BudgetExceeded("72-hour account execution window expired")
            if any(e.get("state") not in TERMINAL for e in data["entries"]):
                raise UnresolvedTask("Unresolved or invalid prior attempt blocks account-wide submission")
            if any(e.get("fingerprint") == fingerprint for e in data["entries"]):
                raise DuplicateSubmission("Request already reserved; do not resubmit")
            all_c = sum(e["circuits"] for e in data["entries"]) + circuits
            all_s = sum(e["requested_shots"] for e in data["entries"]) + circuits * shots
            if all_c > self.limits["circuits"] or all_s > self.limits["shots"]:
                raise BudgetExceeded("Account-wide requested resource cap exceeded")
            if phase == "preflight":
                prev = [e for e in data["entries"] if e["phase"] == "preflight"]
                if (sum(e["circuits"] for e in prev) + circuits > self.limits["preflight_circuits"]
                        or sum(e["requested_shots"] for e in prev) + circuits * shots > self.limits["preflight_shots"]):
                    raise BudgetExceeded("Preflight cap exceeded (included in global totals)")
            entry = {"attempt_id": uuid.uuid4().hex, "state": "submission_pending",
                "submission_started_utc": utc(now), "phase": phase,
                "circuits": circuits, "shots_per_circuit": shots, "requested_shots": circuits * shots,
                "fingerprint": fingerprint, "metadata": metadata,
                "history": [{"state": "submission_pending", "utc": utc(now)}]}
            if window is not None:
                entry.update(execution_window_id=window_id, reserved_epoch=now,
                             observation_deadline_epoch=now + 300)
                if main_window:
                    entry.update(operation_id=operation_id, request_identity=identity)
                    if window["state"] == "authorized":
                        window.update(state="running", started_epoch=now, started_utc=utc(now),
                                      deadline_epoch=now + 72 * 3600)
                else:
                    window.update(state="reserved", started_epoch=now, started_utc=utc(now),
                                  deadline_epoch=now + 72 * 3600, attempt_id=entry["attempt_id"])
            data["entries"].append(entry)
            atomic_json(self.path, data)  # Durable reservation BEFORE any submission call.
            return entry

    def update(self, attempt_id, state, **fields):
        allowed = {"submission_pending": {"submitted", "submission_unknown", "submission_rejected"},
                   "submitted": {"submitted", "validated", "terminal_failure", "observation_unknown", "received_invalid"}}
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            found = [e for e in data["entries"] if e["attempt_id"] == attempt_id]
            if len(found) != 1:
                raise LedgerError("Attempt missing from account ledger")
            e = found[0]
            if state not in allowed.get(e["state"], set()):
                raise LedgerError("Forbidden state transition; unresolved jobs require independent reconciliation")
            if e.get("execution_window_id") and state == "validated":
                bound_window = next(w for w in data["execution_windows"]
                                    if w["window_id"] == e["execution_window_id"])
                if (bound_window.get("kind") == "main_protocol_v1"
                        and self.clock() >= min(e["observation_deadline_epoch"], bound_window["deadline_epoch"])):
                    raise BudgetExceeded("Main result cannot become validated after its persisted deadline")
            protected = {"attempt_id", "fingerprint", "circuits", "shots_per_circuit", "requested_shots",
                         "submission_started_utc", "phase", "history", "metadata", "state",
                         "execution_window_id", "reserved_epoch", "observation_deadline_epoch",
                         "operation_id", "request_identity"}
            if protected.intersection(fields):
                raise LedgerError("Reserved counters or provenance cannot be overwritten")
            if "job_ids" in fields:
                previous = {jid for other in data["entries"] if other is not e for jid in other.get("job_ids", [])}
                if previous.intersection(fields["job_ids"]):
                    raise LedgerError("Server reused a prior task ID; independent reconciliation required")
            e.update(fields); e["state"] = state
            e["history"].append({"state": state, "utc": utc(self.clock())})
            if e.get("execution_window_id") and state not in {"submission_pending", "submitted"}:
                w = next(w for w in data["execution_windows"] if w["window_id"] == e["execution_window_id"])
                if w.get("kind") != "main_protocol_v1" or state != "validated":
                    w.update(state="closed", closed_utc=utc(self.clock()), outcome=state,
                             client_execution_stopped=True, backend_cancelled=False)
            atomic_json(self.path, data)
