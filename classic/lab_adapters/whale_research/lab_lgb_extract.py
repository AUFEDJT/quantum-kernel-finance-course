"""Build the minimum private input for WhaleQuant 8.2, keeping the upstream pool order."""
import concurrent.futures,csv,hashlib,json,zipfile,io,sys
from pathlib import Path
import pandas as pd
import numpy as np
share,repo=map(Path,sys.argv[1:3]);base=repo/'private/classic_lab/whale_research';out=base/'lgb_inputs.zip'
pool=pd.read_csv(next((repo/'classic/sources/whale-quant/notebook').glob('ch08*/stock/company_info.csv')),encoding='utf-8-sig')
pool.drop(columns=['index','symbol','fullname'],inplace=True);pool.dropna(inplace=True)
names=['交易日期','开盘价','最高价','最低价','收盘价','前收盘价','成交量','成交额']
def one(item):
 ix,code=item
 key=code[-2:].lower()+code[:6]+'.csv'
 p=share/'stock-trading-data'/key
 if not p.exists(): p=share/'stock-trading-data-pro'/key
 f=None
 for skip in (1,0):
  for enc in ('gb18030','utf-8-sig'):
   try:f=pd.read_csv(p,skiprows=skip,encoding=enc,usecols=names)[names];break
   except (UnicodeError,ValueError): pass
  if f is not None:break
 if f is None:raise ValueError('Could not map original daily fields')
 d=pd.to_datetime(f.pop(names[0]).astype(str),format='mixed')
 keep=d.between('2018-07-01','2023-06-30')
 v=np.column_stack((d[keep].dt.strftime('%Y%m%d').astype(np.int32).to_numpy(),f.loc[keep].to_numpy(dtype=np.float64)))
 if len(v) and (not np.isfinite(v).all() or np.any(v[:,1:6]<=0) or np.any(v[:,6:]<0) or len(np.unique(v[:,0]))!=len(v)): raise ValueError('Invalid daily price, amount, volume or duplicate date')
 buf=io.BytesIO();np.save(buf,v[np.argsort(v[:,0])],allow_pickle=False)
 return ix,code,buf.getvalue(),len(v)
manifest=[];issues=[]
with zipfile.ZipFile(out.with_suffix('.partial'),mode='w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z, concurrent.futures.ThreadPoolExecutor(max_workers=6) as exe:
 jobs=[exe.submit(one,row) for row in enumerate(pool.ts_code)]
 for i,j in enumerate(jobs):
  try:
   ix,code,payload,count=j.result(); z.writestr(f'{ix:04}.npy',payload);manifest.append({'id':ix,'symbol':code,'rows':count})
  except Exception as e:issues.append({'id':i,'type':type(e).__name__,'message':str(e)[:250]})
  if (i+1)%400==0:print('examined',i+1,'failures',len(issues),flush=True)
 z.writestr('pool.json',json.dumps(manifest))
out.with_suffix('.partial').replace(out)
r={'universe':len(pool),'source':'stock-trading-data; stock-trading-data-pro fallback','matched':len(manifest),'rows':sum(v['rows'] for v in manifest),'issues':issues,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'units':{'volume':'shares (convert to hands / 100)','amount':'CNY (convert to thousands / 1000)'},'window':['20180701','20230630']}
(base/'lgb_input_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
print({k:v for k,v in r.items() if k in ['universe','matched','rows','sha256']},'issues',len(issues))

