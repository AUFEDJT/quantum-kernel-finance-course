"""Private, opt-in provenance recording around unchanged public data calls."""
import functools
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ['QFB_EVIDENCE_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
COUNTER = 0


def stamp():
    return datetime.now(timezone.utc).isoformat()


def record_api(module, name, provider):
    original = getattr(module, name)
    if getattr(original, '_qfb_recording', False):
        return

    @functools.wraps(original)
    def recorded(*args, **kwargs):
        global COUNTER
        COUNTER += 1
        ident = f'{COUNTER:04d}_{provider}_{name}'
        event = {'id': ident, 'provider': provider, 'function': name,
                 'started_utc': stamp(),
                 'args': [str(a) for a in args],
                 'kwargs': {k: str(v) for k, v in kwargs.items()
                            if not any(s in k.lower() for s in ['token', 'key', 'secret', 'password', 'session'])}}
        event['upstream_provider'] = {'stock_us_daily':'Sina Finance','stock_zh_a_daily':'Sina Finance',
                                      'stock_us_hist_sina':'Sina Finance',
                                      'stock_zh_a_hist_tx':'Tencent Finance','stock_zh_a_hist':'Eastmoney',
                                      'download':'Yahoo Finance'}.get(name,provider)
        if os.environ.get('QFB_PROVIDER_OVERRIDE'):
            event['upstream_provider']=os.environ['QFB_PROVIDER_OVERRIDE']
            event['provider']='laboratory_adapter'
        try:
            from IPython import get_ipython
            event['execution_count']=get_ipython().execution_count
        except Exception:
            pass
        try:
            bound=inspect.signature(original).bind_partial(*args,**kwargs)
            bound.apply_defaults()
            event['effective_parameters']={k:str(v) for k,v in bound.arguments.items()
                if not any(s in k.lower() for s in ['token','key','secret','password','session'])}
        except (TypeError,ValueError):
            event['effective_parameters_status']='signature_unavailable'
        try:
            result = original(*args, **kwargs)
            if hasattr(result, 'to_csv'):
                p = ROOT / (ident + '.csv')
                result.to_csv(p, encoding='utf-8', index=True)
                event.update(snapshot=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             rows=len(result), columns=[str(x) for x in getattr(result, 'columns', [])],
                             dtypes=str(getattr(result, 'dtypes', '')), status='empty' if len(result) == 0 else 'returned')
                event['nonnull_counts']={str(k):int(v) for k,v in result.count().items()}
            else:
                event.update(status='returned_non_tabular', type=type(result).__name__)
            return result
        except BaseException as e:
            event.update(status='error', error_type=type(e).__name__, error=str(e)[:2000])
            raise
        finally:
            event['finished_utc'] = stamp()
            with (ROOT / 'acquisitions.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
    recorded._qfb_recording = True
    setattr(module, name, recorded)


def install():
    import socket
    socket.setdefaulttimeout(25)
    import yfinance as yf
    import akshare as ak
    record_api(yf, 'download', 'yahoo')
    for name in ['stock_us_daily', 'stock_us_hist_sina', 'stock_zh_a_hist_tx', 'stock_zh_a_daily', 'stock_zh_a_hist']:
        if hasattr(ak, name):
            record_api(ak, name, 'akshare')

