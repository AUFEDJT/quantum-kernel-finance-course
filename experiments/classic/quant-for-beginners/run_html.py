"""Exercise source HTML controls in an actual browser; save private evidence."""
import argparse
import hashlib
import json
import threading
import functools
import shutil
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from playwright.sync_api import sync_playwright


def stamp(): return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--browser-executable',default=shutil.which('google-chrome'))
    a=ap.parse_args();root=Path(a.source_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    handler=functools.partial(SimpleHTTPRequestHandler,directory=str(root))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    results=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.browser_executable,headless=True,args=['--no-sandbox'])
        for source in sorted((root/'assets/interactive').glob('*.html')):
            errors=[];console=[];failed=[];responses=[];controls=[]
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda e:console.append({'type':e.type,'text':e.text}) if e.type in ['error','warning'] else None)
            page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':r.failure}))
            page.on('response',lambda r:responses.append({'url':r.url,'status':r.status}) if r.status>=400 else None)
            result={'source_path':source.relative_to(root).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'started_utc':stamp(),'controls':controls,'page_errors':errors,'console':console,'request_failures':failed,'http_failures':responses}
            try:
                page.goto(f'http://127.0.0.1:{server.server_port}/'+source.relative_to(root).as_posix(),wait_until='networkidle',timeout=45000)
                page.wait_for_timeout(800)
                result['title']=page.title();result['canvas_count']=page.locator('canvas').count()
                result['initial_body_text']=page.locator('body').inner_text()[:18000]
                page.screenshot(path=str(out/(source.stem+'_initial.png')),full_page=True)
                # Visit every tab explicitly so controls in initially hidden scenes are tested.
                tabs=page.locator('button.tab,[role="tab"]')
                scene_count=max(1,tabs.count())
                for scene in range(scene_count):
                  if tabs.count():
                    tabs.nth(scene).click();page.wait_for_timeout(150)
                  for i in range(page.locator('input,select,button').count()):
                    el=page.locator('input,select,button').nth(i)
                    meta=el.evaluate('(e)=>({tag:e.tagName,id:e.id,type:e.type,text:e.textContent,min:e.min,max:e.max,step:e.step,value:e.value})')
                    if el.evaluate('e=>e.matches("button.tab,[role=tab]")'):continue
                    if not el.is_visible():continue
                    action={'scene':scene,'control':meta}
                    try:
                        before=page.evaluate("""()=>Array.from(document.querySelectorAll('canvas')).map(c=>{let s=c.toDataURL();let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h+s.charCodeAt(i))|0;return h}).join(',')""")
                        before_text=page.locator('body').inner_text()
                        if not el.is_visible() or not el.is_enabled():action['status']='not_interactive'
                        elif meta['tag']=='SELECT':
                            opts=el.locator('option').all()
                            if len(opts)>1: el.select_option(index=1)
                            action['status']='changed'
                        elif meta['tag']=='BUTTON':
                            el.click(timeout=4000);action['status']='clicked'
                        elif meta['type'] in ['range','number']:
                            value=el.evaluate("""e=>{const lo=Number(e.min||0),hi=Number(e.max||100),step=Number(e.step||1);let v=lo+Math.round((hi-lo)/step/2)*step;if(v===Number(e.value))v=Math.min(hi,v+step);e.value=String(v);e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return e.value;}""")
                            action.update(status='changed',value=value)
                        elif meta['type'] in ['checkbox','radio']:
                            el.click(timeout=4000);action['status']='clicked'
                        elif meta['type'] in ['text','']:
                            action['status']='text_input_preserved'
                        else:action['status']='unhandled_control'
                        page.wait_for_timeout(150)
                        after=page.evaluate("""()=>Array.from(document.querySelectorAll('canvas')).map(c=>{let s=c.toDataURL();let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h+s.charCodeAt(i))|0;return h}).join(',')""")
                        action['canvas_changed']=before!=after
                        action['text_changed']=before_text!=page.locator('body').inner_text()
                    except Exception as e:action.update(status='error',error=str(e)[:1500])
                    controls.append(action)
                page.wait_for_timeout(500)
                result['final_body_text']=page.locator('body').inner_text()[:18000]
                result['body_changed']=result['initial_body_text']!=result['final_body_text']
                result['canvas_dimensions']=page.locator('canvas').evaluate_all('(xs)=>xs.map(e=>({width:e.width,height:e.height}))')
                page.screenshot(path=str(out/(source.stem+'_interacted.png')),full_page=True)
                result['reactive_controls']=sum(bool(c.get('canvas_changed') or c.get('text_changed')) for c in controls)
                result['status']='passed' if not errors and not failed and not responses and not any(c['status']=='error' for c in controls) and result['reactive_controls']>0 else 'issues_observed'
            except Exception as e:result.update(status='error',error=str(e)[:2500])
            result['finished_utc']=stamp();results.append(result);page.close()
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps({'html':source.name,'status':result['status'],'errors':len(errors),'controls':len(controls)},ensure_ascii=False),flush=True)
        browser.close()
    server.shutdown()
    if not results or any(r['status']!='passed' for r in results):
        raise SystemExit(1)


if __name__=='__main__':main()
