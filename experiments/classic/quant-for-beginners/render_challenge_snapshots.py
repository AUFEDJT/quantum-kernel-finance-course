"""Render chapter 8 challenge figures from existing, hashed private inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 events=[json.loads(s) for s in (run/'data_snapshots/acquisitions.jsonl').read_text().splitlines()]
 challenges=json.loads((run/'challenges.json').read_text());records=[]
 symbols=['TSLA','NVDA','KO','SPY']
 for years in [1,3,5]:
  series={};inputs=[]
  for symbol in symbols:
   event=next(e for e in events if e['status']=='returned' and e['kwargs'].get('symbol')==symbol)
   path=run/'data_snapshots'/event['snapshot'];assert hashlib.sha256(path.read_bytes()).hexdigest()==event['sha256']
   d=pd.read_csv(path);d['date']=pd.to_datetime(d['date']);s=d.set_index('date').sort_index()['close']
   start=pd.Timestamp('2026-11-19')-pd.DateOffset(years=years,days=30);series[symbol]=s.loc[s.index>=start];inputs.append({'symbol':symbol,'sha256':event['sha256']})
  prices=pd.DataFrame(series).dropna();r=prices.pct_change().dropna()
  prior=next(x['result'] for x in challenges if x['id']==f'ch8_alternative_symbols_{years}y')
  assert len(r)==prior['rows']
  assert np.allclose(r.std()*np.sqrt(252),pd.Series(prior['annual_volatility']).reindex(r.columns),atol=1e-12,rtol=1e-12)
  files=[]
  def save(name):
   target=out/f'{years}y_{name}.png';plt.tight_layout();plt.savefig(target,dpi=120);plt.close();files.append({'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
  plt.figure(figsize=(6,5));sns.heatmap(r.corr(),annot=True,vmin=-1,vmax=1,cmap='RdBu_r');save('correlation')
  (r.std()*np.sqrt(252)).plot.bar(title=f'Annual volatility, {years} years');save('volatility')
  pd.DataFrame({'daily_equal_weight':r[['TSLA','NVDA','KO']].mean(axis=1),'SPY':r.SPY}).add(1).cumprod().plot(title=f'Equity, {years} years');save('equity')
  records.append({'years':years,'status':'passed','rows':len(r),'reproduces_challenge_volatility':True,'inputs':inputs,'figures':files})
 (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8');print(json.dumps({'statuses':[x['status'] for x in records],'figures':sum(len(x['figures']) for x in records)}))
if __name__=='__main__':main()
