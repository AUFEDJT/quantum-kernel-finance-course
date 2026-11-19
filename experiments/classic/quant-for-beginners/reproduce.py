"""Run all original QFB workflows and bounded semantic compatibility reruns."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--skip-html',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--browser-executable',help='Optional Chrome/Chromium executable for the HTML driver');a=p.parse_args()
    src=Path(a.source_root).resolve();out=Path(a.output).resolve();tools=Path(__file__).resolve().parent
    notebooks=sorted(src.glob('notebooks/*/*.ipynb'))
    assert len(notebooks)==8,'Expected the pinned eight upstream notebooks'
    manifest=json.loads((tools/'compatibility_manifest.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        assert hashlib.sha256((src/item['source_path']).read_bytes()).hexdigest()==item['source_sha256'], 'Source does not match the pinned, verified upstream'
    plan=[]
    for i,n in enumerate(notebooks,1):
        plan.append([sys.executable,str(tools/'run_notebook.py'),'--source',str(n),'--source-relative',n.relative_to(src).as_posix(),'--output',str(out/f'runs/chapter{i:02}_original'),'--append',str(tools/'append_challenges.py')])
    plan.append([sys.executable,str(tools/'make_compat.py'),'--source-root',str(src),'--output',str(out/'compat')])
    for i in [3,6,7,8]:
        n=notebooks[i-1];args=[sys.executable,str(tools/'run_notebook.py'),'--source',str(out/'compat'/n.relative_to(src)),'--source-relative',n.relative_to(src).as_posix(),'--output',str(out/f'runs/chapter{i:02}_compatible'),'--variant','compatible','--append',str(tools/'append_challenges.py')]
        if i==3:args+=['--skip-cells','11']
        plan.append(args)
    if not a.skip_html:
        html_args=[sys.executable,str(tools/'run_html.py'),'--source-root',str(src),'--output',str(out/'html_results')]
        if a.browser_executable:html_args+=['--browser-executable',str(Path(a.browser_executable).resolve())]
        plan.append(html_args)
    if a.dry_run:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=False);statuses=[]
    for i,args in enumerate(plan):
        with (out/f'driver_{i:02}.log').open('w',encoding='utf8') as log:
            try:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1800);status={'step':i,'exit_code':r.returncode}
            except subprocess.TimeoutExpired:status={'step':i,'timeout_seconds':1800}
        statuses.append(status);(out/'orchestration.json').write_text(json.dumps(statuses,indent=2),encoding='utf8')
    print('Execution records are ready; API errors, empty inputs and task completion must be assessed from each summary, not the driver exit code.')
    if any(s.get('exit_code')!=0 for s in statuses):raise SystemExit(1)
if __name__=='__main__':main()
