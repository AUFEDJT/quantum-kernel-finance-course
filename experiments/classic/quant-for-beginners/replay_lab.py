"""Replay the explicitly dated, verified 600519 laboratory input contract.

This creates an attributed runtime copy and binds the two original datetime.now()
calls to the archived run date. It does not claim a fresh/current market run.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--lab-root',required=True)
    p.add_argument('--reference',required=True,help='The frozen original Tencent qfq CSV used only for validation')
    p.add_argument('--output',required=True);p.add_argument('--as-of',required=True,choices=['2026-11-19']);a=p.parse_args()
    tools=Path(__file__).resolve().parent;src=Path(a.source_root).resolve();out=Path(a.output).resolve()
    lab=Path(a.lab_root).resolve();reference=Path(a.reference).resolve()
    manifest=json.loads((tools/'compatibility_manifest.json').read_text(encoding='utf8'))
    item=next(x for x in manifest['files'] if x['source_path'].startswith('notebooks/phase1_intro/03_'))
    original=src/item['source_path']
    if hashlib.sha256(original.read_bytes()).hexdigest()!=item['source_sha256']:raise ValueError('Source differs from pinned upstream')
    if not (lab/'supplement/stock-trading-data-pro.csv').is_file() or not reference.is_file():raise FileNotFoundError('Private laboratory slice or original-provider reference is unavailable')
    out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(tools/'make_compat.py'),'--source-root',str(src),'--output',str(out/'compat')],check=True)
    source=out/'compat'/item['source_path'];nb=json.loads(source.read_text(encoding='utf8'));code=''.join(nb['cells'][12]['source'])
    if code.count('datetime.now()')!=2:raise ValueError('Original date expressions changed; review required')
    code=code.replace('datetime.now()',f"datetime.fromisoformat('{a.as_of}T12:00:00')")
    nb['cells'][12]['source']=code.splitlines(keepends=True)
    nb.setdefault('metadata',{})['qfb_archived_replay']={'as_of':a.as_of,'original_source_sha256':item['source_sha256'],'date_binding_cell':13,'purpose':'replay the archived verified interval, not current data acquisition'}
    source.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    (out/'replay_manifest.json').write_text(json.dumps({'as_of':a.as_of,'source_path':item['source_path'],'original_sha256':item['source_sha256'],'runtime_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'change':'Bind exactly two original datetime.now() calls in cell 13 to the explicit archived run date'},indent=2),encoding='utf8')
    env=os.environ.copy();env['QFB_LAB_ROOT']=str(lab);env['QFB_LAB_REFERENCE']=str(reference)
    subprocess.run([sys.executable,str(tools/'run_notebook.py'),'--source',str(source),'--source-relative',item['source_path'],'--output',str(out/'run'),'--variant','laboratory_archived_replay','--prelude',str(tools/'lab_prelude.py'),'--skip-cells','11','--append',str(tools/'lab_checks.py')],check=True,env=env)
    checks=json.loads((out/'run/lab_checks.json').read_text(encoding='utf8'))
    if checks['status']!='pass':raise RuntimeError('Laboratory assertions did not pass')
    print('Archived laboratory replay passed; no current-data claim is made.')
if __name__=='__main__':main()
