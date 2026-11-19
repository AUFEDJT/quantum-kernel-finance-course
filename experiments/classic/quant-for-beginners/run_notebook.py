"""Run one source Notebook in a fresh kernel, preserving errors and source identity."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-relative', required=True)
    p.add_argument('--variant', default='original')
    p.add_argument('--append')
    p.add_argument('--prelude', help='Explicit input adapter code, executed before provenance wrapping')
    p.add_argument('--skip-cells', default='')
    a = p.parse_args()
    source, output = Path(a.source).resolve(), Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['QFB_EVIDENCE_DIR'] = str(output / 'data_snapshots')
    os.environ['QFB_SOURCE_PATH'] = a.source_relative
    os.environ.update(TZ='Asia/Shanghai', MPLBACKEND='Agg', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    tools = str(Path(__file__).resolve().parent)
    nb = nbformat.read(source, as_version=4)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    skips = {int(x) for x in a.skip_cells.split(',') if x}
    orig_ids = []
    for i, cell in enumerate(nb.cells):
        cell.metadata['original_cell_number'] = i + 1
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            if i + 1 in skips:
                cell.source = '# Selected independent branch: alternative input cell omitted here.'
                cell.metadata['branch_omission'] = True
        orig_ids.append(i + 1)
    adapter_code = Path(a.prelude).read_text(encoding='utf8') if a.prelude else ''
    hook = nbformat.v4.new_code_cell(f'import sys\nsys.path.insert(0, {tools!r})\n{adapter_code}\nfrom provenance_hook import install\ninstall()')
    hook.metadata['instrumentation'] = 'Explicit input adapter plus API-return snapshots' if a.prelude else 'API-return snapshots only; original task code unchanged'
    nb.cells.insert(0, hook)
    if a.append:
        extra = nbformat.v4.new_code_cell(Path(a.append).read_text(encoding='utf8'))
        extra.metadata['extension'] = 'challenge or independent numerical checks'
        nb.cells.append(extra)
    nb.cells.append(nbformat.v4.new_code_cell("""
import json as _json
from pathlib import Path as _Path
_state={}
for _key,_value in list(globals().items()):
    if _key.startswith('_'): continue
    if isinstance(_value,(int,float,str,bool)):
        _state[_key]=_value
    elif isinstance(getattr(_value,'shape',None),tuple):
        _state[_key]={'type':type(_value).__name__,'shape':list(_value.shape)}
_Path(__import__('os').environ['QFB_EVIDENCE_DIR']).parent.joinpath('state_summary.json').write_text(_json.dumps(_state,ensure_ascii=False,default=str,indent=2),encoding='utf8')
"""))
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    def stop_if_setup_failed(cell, cell_index, **kwargs):
        if cell.metadata.get('instrumentation'):
            raise RuntimeError('Provenance setup or explicit input adapter failed; source cells were not executed')
    client = NotebookClient(nb, km=km, timeout=180, startup_timeout=90, allow_errors=True,
                            on_cell_error=stop_if_setup_failed,
                            interrupt_on_timeout=True, record_timing=True,
                            resources={'metadata': {'path': str(source.parent)}})
    summary = {'source_path': a.source_relative, 'source_commit': '8fbf8942afc0e80d01538ee545e72174e099e81a',
               'source_sha256': source_hash, 'variant': a.variant, 'started_utc': stamp(),
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}},
               'cells': [], 'skipped_branch_cells': sorted(skips)}
    try:
        client.execute()
    except BaseException as e:
        summary['execution_exception'] = {'type': type(e).__name__, 'message': str(e)[:3000]}
    finally:
        for cell in nb.cells:
            if cell.cell_type != 'code': continue
            errors = [{'type':o.get('ename'), 'message':o.get('evalue')} for o in cell.get('outputs',[]) if o.output_type=='error']
            summary['cells'].append({'original_cell':cell.metadata.get('original_cell_number'),
                                     'execution_count':cell.execution_count,
                                     'source_sha256':hashlib.sha256(cell.source.encode()).hexdigest(),
                                     'status':'not_run' if cell.execution_count is None else ('error' if errors else 'executed'),
                                     'errors':errors, 'output_count':len(cell.get('outputs',[])),
                                     'empty':not bool(cell.source.strip()),
                                     'timing':cell.metadata.get('execution',{}),
                                     'branch_omission':cell.metadata.get('branch_omission',False)})
        summary['finished_utc']=stamp()
        nbformat.write(nb, output / 'executed.ipynb')
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'source_path':a.source_relative,'variant':a.variant,
                          'errors':sum(c['status']=='error' for c in summary['cells']),
                          'executed':sum(c['status']=='executed' for c in summary['cells'])},ensure_ascii=False),flush=True)
        if summary.get('execution_exception'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
