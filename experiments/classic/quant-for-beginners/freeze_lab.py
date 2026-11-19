"""Freeze minimum original-asset laboratory inputs; raw market rows remain private."""
import argparse, hashlib, io, json, re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

SYMBOLS = ['AAPL','AMD','AMZN','COIN','GOOGL','JPM','KO','META','MSFT','NFLX','NVDA','TSLA','SPY','XLE']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    start=pd.Timestamp('2021-10-10');end=pd.Timestamp('2026-11-19')
    manifest={'acquired_utc':datetime.now(timezone.utc).isoformat(),'source_root':str(root),'window_start':str(start.date()),'window_end_exclusive':str(end.date()),'files':[],'missing':[],'inventory':{}}
    def freeze(path,kind,symbol):
        raw=path.read_bytes();df=pd.read_csv(io.BytesIO(raw),encoding='gb18030',skiprows=1)
        record={'kind':kind,'symbol':symbol,'source_relative':path.relative_to(root).as_posix(),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'columns':list(df.columns),'source_rows':len(df)}
        if kind=='prices':
            dates=pd.to_datetime(df['交易日期']);mask=dates.ge(start)&dates.lt(end)
            record.update(source_date_min=str(dates.min().date()),source_date_max=str(dates.max().date()),duplicate_dates=int(dates.duplicated().sum()))
            test=df.assign(_date=dates).sort_values('_date')
            prev=pd.to_numeric(test['前收盘价'],errors='coerce');last=pd.to_numeric(test['收盘价'],errors='coerce').shift()
            changed=(prev-last).abs()>0.02
            record['preclose_vs_previous_close_discontinuities_in_window']=int((changed & test['_date'].ge(start)).sum())
        else:
            datecols=[c for c in ['除权日期','拆分生效日期'] if c in df]
            dates=[pd.to_datetime(df[c],errors='coerce') for c in datecols]
            mask=pd.Series(False,index=df.index)
            for d in dates:mask|=d.ge(start)&d.lt(end)
            record['date_columns']=datecols
            record['last_action_dates']={c:str(d.max().date()) if d.notna().any() else None for c,d in zip(datecols,dates)}
        snap=df.loc[mask].copy();target=out/kind/(symbol+'.csv');target.parent.mkdir(exist_ok=True)
        snap.to_csv(target,index=False,encoding='utf8');b=target.read_bytes()
        record.update(snapshot_relative=target.relative_to(out).as_posix(),snapshot_sha256=hashlib.sha256(b).hexdigest(),snapshot_rows=len(snap),snapshot_bytes=len(b))
        manifest['files'].append(record)
    price_dir=root/'stock-us-trading-data';action_dir=root/'stock-us-dividend-delivery-data'
    for symbol in SYMBOLS:
        names=list(price_dir.glob(symbol+'.*.csv'))
        if len(names)==1:freeze(names[0],'prices',symbol)
        else:manifest['missing'].append({'symbol':symbol,'kind':'prices','reason':'no exact active symbol match' if not names else 'ambiguous'})
        actions=list(action_dir.glob(symbol+'.*.csv'))
        if len(actions)==1:freeze(actions[0],'actions',symbol)
        else:manifest['missing'].append({'symbol':symbol,'kind':'actions','reason':'not present; absence alone does not prove no corporate actions'})
    freeze(root/'stock-trading-data'/'sh600519.csv','prices','sh600519')
    supplement=out/'supplement';supplement.mkdir(exist_ok=True);extra=[]
    for dirname in ['stock-trading-data-pro','stock-dividend-delivery']:
        path=root/dirname/'sh600519.csv';raw=path.read_bytes()
        df=pd.read_csv(io.BytesIO(raw),encoding='gb18030',skiprows=1,low_memory=False)
        if dirname=='stock-trading-data-pro':
            dates=pd.to_datetime(df['交易日期']);df=df.loc[dates.ge('2024-09-10')&dates.lt(end)]
        target=supplement/(dirname+'.csv');df.to_csv(target,index=False,encoding='utf8')
        extra.append({'source_relative':path.relative_to(root).as_posix(),'source_sha256':hashlib.sha256(raw).hexdigest(),'snapshot_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'snapshot_rows':len(df),'columns':list(df.columns)})
    (supplement/'manifest.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf8')
    for dirname in ['stock-us-main-index-data','stock-etf-trading-data']:
        paths=list((root/dirname).glob('*.csv'))
        manifest['inventory'][dirname]={'count':len(paths),'original_US_ETF_matches':[x.name for x in paths if re.search(r'(^|[._-])(SPY|XLE)([._-]|$)',x.name)],'index_names':[x.name for x in paths] if len(paths)<10 else None}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'files':len(manifest['files']),'missing':manifest['missing'],'coverage':[{'symbol':x['symbol'],'kind':x['kind'],'rows':x['snapshot_rows'],'last':x.get('source_date_max',x.get('last_action_dates')),'preclose_discontinuities':x.get('preclose_vs_previous_close_discontinuities_in_window')} for x in manifest['files']],'additional_inventory':manifest['inventory']},ensure_ascii=False))
if __name__=='__main__':main()
