"""Execute educational snippets in isolated processes; never run platform orders."""
import argparse
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tempfile
import traceback
from pathlib import Path
from inventory import classify, extract_workflows


def clean(text):
    text = re.sub(r"(?i)((?:token|password|passwd|secret|api_key)\s*[:=]\s*)[^\s,;}]+", r"\1[REDACTED]", str(text))
    text = re.sub(r"\b[a-fA-F0-9]{40,}\b", "[LONG_HEX_REDACTED]", text)
    return text


def alarm_handler(*_):
    raise TimeoutError("teaching cell exceeded the configured execution timeout")


def run_child(job, output, network, seconds):
    data = json.loads(Path(job).read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    (root / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(root / "tmp")
    tempfile.tempdir = str(root / "tmp")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(root / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    # Matplotlib may discover fonts through fc-list during its first import.
    # Perform library initialization before disabling child processes in snippets.
    import matplotlib.pyplot
    # Optional runtime-only alias for upstream SimHei requests on Linux.
    # The supplied font must already exist; no font download or source edit.
    if os.environ.get("WHALE_CJK_FONT"):
        from matplotlib.font_manager import FontEntry, fontManager
        font_path = Path(os.environ["WHALE_CJK_FONT"])
        if not font_path.is_file():
            raise FileNotFoundError("WHALE_CJK_FONT does not name an existing font")
        fontManager.ttflist.append(FontEntry(fname=str(font_path), name="SimHei"))
    if not network:
        def no_connect(*args, **kwargs):
            raise ConnectionError("network disabled for isolated original-source audit")
        socket.socket.connect = no_connect
        socket.create_connection = no_connect
    original_open = builtins.open
    def confined_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+"):
            target = Path(os.fsdecode(file)).resolve()
            if not target.is_relative_to(root):
                raise PermissionError("teaching write outside its isolated run directory")
        return original_open(file, mode, *args, **kwargs)
    builtins.open = confined_open
    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
            raise PermissionError("external process execution disabled in teaching source")
        if event in ("os.remove", "os.rmdir", "os.rename"):
            for arg in args[:2] if event == "os.rename" else args[:1]:
                if isinstance(arg, (str, bytes, os.PathLike)) and not Path(os.fsdecode(arg)).resolve().is_relative_to(root):
                    raise PermissionError("filesystem mutation outside isolated run directory")
    sys.addaudithook(audit)
    namespace = {"__name__": "__main__"}
    rows = []
    for loc, code, kind in data["cells"]:
        ident = data["source"] + "#" + loc
        check = classify(code)
        row = {"id": ident, "source": data["source"], "location": loc, "kind": kind, "flags": check["flags"], "action": "original source executed unchanged", "elapsed_seconds": 0, "target_verified": False}
        if "external_platform_or_order_api" in check["flags"]:
            row.update(status="blocked_platform", observed_error="platform-specific or order API intercepted before execution", action="preserved source; platform credentials and live-order APIs not invoked")
        elif "embedded_credential_assignment" in check["flags"]:
            row.update(status="blocked_credential", observed_error="credential-shaped literal detected; value not used or logged", action="requires sanitized teaching source or explicit credential-free adapter")
        elif re.search(r"(?:set_token\(|pro_api\(|ts\.pro_bar|\bauth\()", code):
            row.update(status="blocked_credential", observed_error="authenticated provider initialization detected; user-home credentials are not read", action="public equivalent endpoint will be probed separately; original authenticated experiment requires supplied fixture or authorized credentials")
        elif "subprocess_or_dynamic_execution" in check["flags"]:
            row.update(status="blocked_dynamic", observed_error="dynamic execution or subprocess call detected", action="requires per-cell manual semantic review")
        else:
            stream = io.StringIO()
            start = time.monotonic()
            before = set(namespace)
            files_before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.glob("*") if p.is_file()}
            try:
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(seconds)
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    exec(compile(code, ident, "exec"), namespace)
                row.update(status="executed", observed_result="source cell completed; target verification reported separately")
            except BaseException as exc:
                expected = bool(re.search(r"raise[s]? (?:a |an )?(?:exception|KeyError|IndexError)|will not work|will fail|引发|抛出|报错|raise an", code, re.I))
                row.update(status="expected_demo_error" if expected else "failed", observed_error=clean(type(exc).__name__ + ": " + str(exc)), traceback=clean(traceback.format_exc(limit=4)))
            finally:
                signal.alarm(0)
                row["elapsed_seconds"] = round(time.monotonic() - start, 4)
                row["stdout"] = clean(stream.getvalue()[-16000:])
                row["new_objects"] = {key: type(namespace[key]).__name__ for key in sorted(set(namespace)-before) if not key.startswith("_") and not re.search("token|password|secret", key, re.I)}
                files_after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.glob("*") if p.is_file()}
                row["written_files"] = [{"name": Path(name).name, "bytes": size, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()} for name,(size,mtime) in files_after.items() if files_before.get(name)!=(size,mtime)]
                if "matplotlib.pyplot" in sys.modules:
                    plt = sys.modules["matplotlib.pyplot"]
                    figs = []
                    for number in plt.get_fignums()[:12]:
                        dest = Path("figures") / (hashlib.sha256(ident.encode()).hexdigest()[:10] + f"-{number}.png")
                        dest.parent.mkdir(exist_ok=True)
                        try:
                            plt.figure(number).savefig(dest, dpi=90)
                            figs.append(str(dest))
                        except Exception as exc:
                            row["figure_warning"] = clean(str(exc))
                    row["figures"] = figs
                    plt.close("all")
        rows.append(row)
        # The private output may be outside cwd; use pre-captured open explicitly.
        with original_open(output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--only", default="")
    p.add_argument("--network", action="store_true")
    p.add_argument("--cell-timeout", type=int, default=25)
    p.add_argument("--child", type=Path)
    a = p.parse_args()
    if a.child:
        run_child(a.child, a.output, a.network, a.cell_timeout)
        return
    a.source = a.source.resolve()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    for source, cells in extract_workflows(a.source):
        if a.only and a.only not in source:
            continue
        label = hashlib.sha256(source.encode()).hexdigest()[:12]
        folder = a.output / label
        folder.mkdir(exist_ok=True)
        work = folder / "work"
        work.mkdir(exist_ok=True)
        for item in a.source.rglob("*.csv"):
            target = work / item.name
            if not target.exists():
                shutil.copy2(item, target)
        job = folder / "job.json"
        job.write_text(json.dumps({"source": source, "cells": cells}, ensure_ascii=False), encoding="utf-8")
        out = folder / "result.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(job), "--output", str(out), "--cell-timeout", str(a.cell_timeout)]
        if a.network:
            cmd.append("--network")
        t = time.monotonic()
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=max(120, len(cells) * (a.cell_timeout + 5)))
        (folder / "process.log").write_text(clean(proc.stdout + proc.stderr), encoding="utf-8")
        rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        for row in rows:
            row["evidence"] = label + "/result.json"
        results.extend(rows)
        (a.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"source": source, "cells": len(cells), "recorded": len(rows), "exit": proc.returncode, "seconds": round(time.monotonic()-t, 2)}, ensure_ascii=False), flush=True)
    summary = {}
    for row in results:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
