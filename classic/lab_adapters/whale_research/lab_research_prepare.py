"""Inspect the permitted market share and extract the subset required by the lesson."""
from pathlib import Path
import csv,hashlib,io,json,sys,datetime
import pandas as pd
share,repo,asof=Path(sys.argv[1]),Path(sys.argv[2]),pd.Timestamp(sys.argv[3])
out=repo/'private/classic_lab/whale_research';out.mkdir(parents=True,exist_ok=True)
records={}; manifest=[]
def read(p,rows=None):
 for enc in ('gb18030','utf-8-sig'):
  for skip in (1,0):
   try:
    frame=pd.read_csv(p,encoding=enc,skiprows=skip,nrows=rows)
    if {'日期','交易日期','candle_end_time','trade_date'}&set(frame):return frame
   except (UnicodeError,pd.errors.ParserError,ValueError):pass
 raise ValueError('unrecognized source schema: '+p.name)
def dump(key,public,**details):
 records[key]={'finding':public,**details}
def slice_file(p,lo,hi,needed):
 f=read(p)
 datecol=next(x for x in ('交易日期','candle_end_time','trade_date','日期') if x in f)
 day=pd.to_datetime(f[datecol].astype(str))
 meta={'from':day.min().strftime('%Y-%m-%d'),'until':day.max().strftime('%Y-%m-%d'),'fields':list(f)}
 s=f.loc[(day>=lo)&(day<=hi),needed].copy();s.insert(0,'date',day[(day>=lo)&(day<=hi)].dt.strftime('%Y-%m-%d').to_numpy());return s,meta
# A stock's unadjusted closes, required by the wavelet/LinearSVR experiment.
p=share/'stock-trading-data/sz000001.csv'
if p.exists():
 x,info=slice_file(p,pd.Timestamp('2015-01-01'),asof-pd.Timedelta(days=1),['收盘价'])
 pro=share/'stock-trading-data-pro/sz000001.csv'
 supplementation=0
 if pro.is_file():
  extra,pro_info=slice_file(pro,pd.Timestamp('2015-01-01'),asof-pd.Timedelta(days=1),['收盘价'])
  add=extra[~extra.date.isin(x.date)];supplementation=len(add)
  x=pd.concat([x,add],ignore_index=True).sort_values('date')
 info['supplemental_daily_closes']=supplementation
 # Compare with the actual latest exchange session if the calendar reaches this week.
 latest=x.date.max() if len(x) else None; previous_business_day=(asof-pd.offsets.BDay(1)).strftime('%Y-%m-%d')
 eligible=bool(len(x) and x.date.min()<='2015-01-09' and '2021-12-31' in x.date.to_list() and latest>=previous_business_day and x['收盘价'].gt(0).all() and not x.date.duplicated().any())
 dump('wavelet','数据覆盖与原始收盘价检查完成',eligible=eligible,expected_to=previous_business_day,**info)
 if eligible:x.rename(columns={'收盘价':'close'}).to_csv(out/'wavelet.csv',index=False)
else:dump('wavelet','未发现原标的日线文件',eligible=False)
# Check the identifiers that survive the source notebook's company_info.dropna().
pool_file=next((repo/'classic/sources/whale-quant/notebook').glob('ch08*/stock/company_info.csv'))
pool=pd.read_csv(pool_file,encoding='utf-8-sig').drop(columns=['index','symbol','fullname']).dropna()
symbols={x.rsplit('.',1)[1].lower()+x.split('.')[0] for x in pool.ts_code}
raw_dir=share/'stock-trading-data';available={p.stem.casefold() for p in raw_dir.glob('*.csv')}
pro_dir=share/'stock-trading-data-pro';available_pro={p.stem.casefold() for p in pro_dir.glob('*.csv')}
missing=sorted(symbols-(available|available_pro))
dump('lightgbm','核对上游原股票池在实验室日线目录的覆盖',pool=len(symbols),missing_count=len(missing),missing=missing,all_present=not missing)
idxdir=share/'stock-us-main-index-data'
idxs=sorted(p.name for p in idxdir.glob('*.csv'))
# SPY is an ETF; NASDAQ Composite is a different index.
sp=next((idxdir/n for n in idxs if Path(n).stem.upper() in {'SPX','GSPC','^GSPC','SPX.GI','GSPC.GI','INX','SPX.US','SP500','SP500.GI'}),None)
if sp:
 x,info=slice_file(sp,pd.Timestamp('2015-01-01'),pd.Timestamp('2022-12-30'),['close'])
 eligible=(len(x)>504 and x.date.min()=='2015-01-02' and x.date.max()=='2022-12-30' and not x.date.duplicated().any() and x.close.gt(0).all())
 if eligible:x.to_csv(out/'sp500.csv',index=False)
 dump('decomposition','核对标普500指数原标的及完整窗口',eligible=bool(eligible),**info)
else:dump('decomposition','实验室美股指数目录未发现标普500价格指数',eligible=False,index_candidates=idxs)
pro_path=pro_dir/'sz000001.csv'
if pro_path.exists():
 sample=read(pro_path); date=pd.to_datetime(sample['交易日期'].astype(str));fields=sample.columns.tolist()
 dump('portfolio','核对上证50时点成分、复权字段和数据截止日',latest=date.max().strftime('%Y-%m-%d'),member_flag='上证50成分股' in fields,adjusted_fields=[x for x in fields if any(y in x for y in ['复权','成分权重','调整日'])],fields=fields)
else:dump('portfolio','缺少股票扩展表',member_flag=False)
for p in [x for x in (out/'wavelet.csv',out/'sp500.csv') if x.is_file()]:manifest.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'admission.json').write_text(json.dumps({'asof':str(asof.date()),'checked':datetime.datetime.now(datetime.timezone.utc).isoformat(),'contracts':records,'snapshots':manifest},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:{a:b for a,b in v.items() if a in ('eligible','missing_count','pool','until','latest','all_present','member_flag')} for k,v in records.items()}))


