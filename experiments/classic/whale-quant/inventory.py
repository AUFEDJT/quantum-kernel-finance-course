"""Read-only source inventory for the frozen WhaleQuant teaching snapshot."""
import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path


def markdown_blocks(text):
    lines = text.splitlines()
    blocks, current, fence, start, language = [], [], None, None, None
    for i, line in enumerate(lines, 1):
        m = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if m:
            if fence is None:
                fence, language, start = m.group(1)[0], m.group(2).strip().lower(), i + 1
                current = []
            elif m.group(1)[0] == fence:
                if language in ("python", "py", "python3", "") and current:
                    blocks.append((start, "\n".join(current), "fenced"))
                fence, current = None, []
            continue
        if fence:
            current.append(line)
    # nbconvert exports code as four-space indentation, not fenced blocks.
    current, start = [], None
    fence = None
    for i, line in enumerate(lines + ["END"], 1):
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m:
            fence = None if fence else m.group(1)[0]
        if not fence and (line.startswith("    ") or (current and not line.strip())):
            if start is None:
                start = i
            current.append(line[4:] if line.startswith("    ") else "")
        elif current:
            code = "\n".join(current).strip()
            try:
                ast.parse(code)
                if code and not code.startswith(("http", "<")):
                    blocks.append((start, code, "indented"))
            except SyntaxError:
                pass
            current, start = [], None
    return sorted(blocks, key=lambda block: block[0])


def classify(code):
    reasons = []
    if re.search(r"(?:set_token\(|pro_api\(|ts\.pro_bar|ts\.get_hist_data|jqdatasdk|\bauth\()", code):
        reasons.append("market_provider_or_credentials")
    if re.search(r"biglearning|bigdatasource|bigtrader|from jqdata|\bM\.trade|order_target|order_value|order_target_percent", code):
        reasons.append("external_platform_or_order_api")
    if re.search(r"os\.system|subprocess\.|\!pip|\!wget|\!curl|requests\.get.*\.py|eval\(|exec\(", code):
        reasons.append("subprocess_or_dynamic_execution")
    if re.search(r"(?:mytoken|token|password|passwd)\s*=\s*['\"][^'\"]+", code, re.I):
        reasons.append("embedded_credential_assignment")
    if re.search(r"stock_data\.csv|toolkit_config|tsla_prc\.csv|StupidHead", code):
        reasons.append("known_missing_source_or_input")
    if re.search(r"yf\.download|ak\.|bs\.|baostock", code):
        reasons.append("public_market_network")
    try:
        tree = ast.parse(code)
        imports = sorted({n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))})
        syntax = "valid"
    except SyntaxError as exc:
        imports, syntax = [], f"syntax:{exc.msg}:{exc.lineno}"
    return {"flags": reasons, "imports": imports, "syntax": syntax}


def read_sources(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                name = "/".join(item.filename.split("/")[1:])
                if name.endswith((".ipynb", ".md", ".py", ".txt")):
                    yield name, z.read(item).decode("utf-8-sig")
    else:
        for item in sorted(path.rglob("*")):
            if item.is_file() and item.suffix in (".ipynb", ".md", ".py", ".txt"):
                yield item.relative_to(path).as_posix(), item.read_text(encoding="utf-8-sig")


def extract_workflows(path):
    for name, text in read_sources(path):
        if ".ipynb_checkpoints" in name:
            continue
        cells = []
        if name.endswith(".ipynb"):
            notebook = json.loads(text)
            for i, cell in enumerate(notebook.get("cells", []), 1):
                src = cell.get("source", [])
                src = "".join(src) if isinstance(src, list) else src
                if cell.get("cell_type") == "code" and src.strip():
                    cells.append((f"cell-{i}", src, "notebook"))
                elif cell.get("cell_type") == "markdown":
                    cells.extend((f"cell-{i}-line-{line}", code, kind) for line, code, kind in markdown_blocks(src))
        elif name.endswith(".md") and name.startswith("docs/ch"):
            cells = [(f"line-{line}", code, kind) for line, code, kind in markdown_blocks(text)]
        elif name.endswith(".py") and name.startswith("code/"):
            cells = [("module", text, "python")]
        if cells:
            yield name, cells


def build(path):
    records, workflows, seen = [], [], {}
    for name, cells in extract_workflows(path):
        workflows.append({"source": name, "count": len(cells)})
        for loc, code, kind in cells:
            normalized = "\n".join(line.rstrip() for line in code.strip().splitlines())
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            ident = f"{name}#{loc}"
            record = {"id": ident, "source": name, "location": loc, "kind": kind, "sha256": digest, "duplicate_of": seen.get(digest), "lines": len(code.splitlines()), **classify(code)}
            seen.setdefault(digest, ident)
            records.append(record)
    return {"snapshot": "828009d910db5eb2c06fa14f4439421b235bef44", "workflows": workflows, "blocks": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    inventory = build(a.source)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    flags = {}
    for item in inventory["blocks"]:
        for flag in item["flags"]:
            flags[flag] = flags.get(flag, 0) + 1
    print(json.dumps({"workflows": len(inventory["workflows"]), "blocks": len(inventory["blocks"]), "unique_code_blocks": len({b["sha256"] for b in inventory["blocks"]}), "flags": flags}, ensure_ascii=False))


if __name__ == "__main__":
    main()
