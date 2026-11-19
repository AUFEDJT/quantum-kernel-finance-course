from pathlib import Path
import pandas as pd,baostock as bs,json
import argparse
cli=argparse.ArgumentParser();cli.add_argument('workdir',type=Path);args=cli.parse_args()
work=args.workdir.resolve();out=work/'results';out.mkdir(parents=True,exist_ok=True)
l=bs.login();assert l.error_code=='0',l.error_msg
try:
 for code,lo,hi,key in [('sz.000001','2015-01-01','2026-11-18','wavelet'),('sh.600000','2018-07-01','2023-06-30','panel')]:
  rs=bs.query_history_k_data_plus(code,'date,open,high,low,close,preclose,volume,amount,tradestatus',start_date=lo,end_date=hi,frequency='d',adjustflag='3')
  assert rs.error_code=='0',rs.error_msg
  rows=[]
  while rs.next():rows.append(rs.get_row_data())
  df=pd.DataFrame(rows,columns=rs.fields);assert len(df)>100
  df.to_csv(out/(key+'_independent_quotes.csv'),index=False)
  df=df[df.tradestatus=='1'].copy()
  if key=='wavelet':
   a=pd.read_csv(work/'inputs/wavelet.csv');missing=sorted(set(df.date)-set(a.date));extra=sorted(set(a.date)-set(df.date))
   merged=a.merge(df[['date','close']],on='date',suffixes=('_lab','_check'))
   receipt={'active_days':len(df),'lab_days':len(a),'compare_through':df.date.max(),'missing_days':missing,'extra_days':extra,'close_disagreement_count':int((abs(merged.close_lab-merged.close_check.astype(float))>.0051).sum())}
   (out/'wavelet_coverage.json').write_text(json.dumps(receipt));print('coverage',len(missing),len(extra),len(df),df.date.max(),receipt['close_disagreement_count'])
  if key=='panel':
   import zipfile,io,numpy as np
   with zipfile.ZipFile(work/'lgb_inputs.zip') as z:
    item=next(x for x in json.loads(z.read('pool.json')) if x['symbol']=='600000.SH')
    arr=np.load(io.BytesIO(z.read(f"{item['id']:04}.npy")))
   lab=pd.DataFrame(arr,columns=['date','open','high','low','close','preclose','volume','amount']);lab.date=pd.to_datetime(lab.date.astype(int).astype(str)).dt.strftime('%Y-%m-%d')
   cols=['open','high','low','close','preclose','volume','amount'];c=lab.merge(df,on='date',suffixes=('','_ref'));match={}
   for col in cols:
    delta=abs(c[col]-pd.to_numeric(c[col+'_ref']));match[col]={'exact_or_rounded_fraction':float((delta<=({'volume':100.,'amount':1000.}.get(col,.0051))).mean()),'median_abs':float(delta.median())}
   val={'n':len(c),'ref':len(df),'lab':len(lab),'missing_dates':len(set(df.date)-set(lab.date)),'extra_dates':len(set(lab.date)-set(df.date)),'fields':match}
   (out/'panel_coverage.json').write_text(json.dumps(val));print('panel',len(c),len(df),len(lab),match)
finally:bs.logout()

