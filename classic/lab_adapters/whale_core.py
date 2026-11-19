"""Frozen-input laboratory adaptations of WhaleQuant chapters 3, 5 and 7.

Source-derived formulas and strategy: datawhalechina/whale-quant,
CC BY-NC-SA 4.0. This file does not fetch market data or invoke cloud services.
All row-level outputs belong in a private directory supplied explicitly.
"""
import argparse, hashlib, importlib.metadata, json, math, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

class ContractBlocked(Exception):
    pass

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_frame(base,entry):
    path=(base/entry['file']).resolve()
    if not path.is_relative_to(base.resolve()):raise ContractBlocked('input_path_escape')
    if digest(path)!=entry['sha256']:raise ContractBlocked('input_hash_mismatch')
    df=pd.read_csv(path)
    if 'datetime' in df:
        df['datetime']=pd.to_datetime(df['datetime'])
        if df['datetime'].duplicated().any():raise ContractBlocked('duplicate_datetime')
        df=df.sort_values('datetime').reset_index(drop=True)
    elif 'date' in df:
        df['date']=pd.to_datetime(df['date'])
        if df['date'].duplicated().any():raise ContractBlocked('duplicate_date')
        df=df.sort_values('date').reset_index(drop=True)
    return df

def prices(base,manifest,key,start=None,end=None):
    if key not in manifest['datasets']:raise ContractBlocked('required_dataset_unavailable: '+key)
    entry=manifest['datasets'][key]
    if entry.get('status')!='ready':raise ContractBlocked(entry.get('reason','dataset_not_ready'))
    df=load_frame(base,entry)
    if start:df=df[df.date>=pd.Timestamp(start)]
    if end:df=df[df.date<=pd.Timestamp(end)]
    if len(df)<2:raise ContractBlocked('insufficient_rows_in_original_window')
    if df.date.min()>pd.Timestamp(start)+pd.Timedelta(days=8) or df.date.max()<pd.Timestamp(end)-pd.Timedelta(days=8):raise ContractBlocked('original_date_window_not_covered')
    if not np.isfinite(df.close).all() or (df.close<=0).any():raise ContractBlocked('invalid_close')
    return df.reset_index(drop=True)

def save(df,out,name):
    df.to_csv(out/name,index=False)

def original_ma_logic(df):
    d=df.copy()
    d['MA5']=d.close.rolling(5).mean();d['MA20']=d.close.rolling(20).mean()
    d['signal']=np.where(d.MA5>d.MA20,1,-1)
    d['position']=d.signal.diff()
    d['return']=d.close.pct_change(fill_method=None)
    d['strategy_return']=d.position.shift()*d['return']
    d['cumulative_return']=(1+d.strategy_return).cumprod()
    return d

def metrics_case(base,m,out):
    keys=['sh600519','sh601398','sh601318','sh000300_index']
    frames=[prices(base,m,k,'2018-01-01','2022-12-31').set_index('date').close.rename(k) for k in keys]
    # Match the original three-stock dataframe: the first stock supplies its index.
    df=pd.DataFrame({keys[0]:frames[0]})
    for k,f in zip(keys[1:],frames[1:]):df[k]=f
    worth=df/df.iloc[0];total=worth.iloc[-1]-1
    raw=df/df.shift(1)-1
    sq=(raw.iloc[1:]-raw.mean())**2
    rets=df.ffill().apply(lambda x:x/x.shift(1)-1).iloc[1:]
    if rets.isna().any().any():raise ContractBlocked('unresolved_start_or_index_missingness')
    from scipy import stats
    rows=[]
    for key in keys:
        r=rets[key];ex=r-.03/250
        row=dict(series=key,n_returns=len(r),cumulative_price_return=float(total[key]),original_annual_return=float((1+total[key])**(252/1826)-1),original_volatility=float(np.sqrt(sq[key].sum()*252/(1826-1))),max_drawdown=round(float(((df[key].cummax()-df[key])/df[key].cummax()).max()),4),original_sharpe=float(np.sqrt(len(ex))*ex.mean()/ex.std()),diagnostic_annual_return=float((1+total[key])**(252/len(r))-1),diagnostic_volatility=float(r.std()*np.sqrt(252)),diagnostic_sharpe=float(np.sqrt(252)*ex.mean()/ex.std()))
        if key!=keys[-1]:
            fit=stats.linregress(rets[keys[-1]],r);excess=r-rets[keys[-1]]
            row.update(original_alpha=round(float(fit.intercept*250),3),original_beta=round(float(fit.slope),3),original_information_ratio=float(np.sqrt(len(excess))*excess.mean()/excess.std()),diagnostic_information_ratio=float(np.sqrt(252)*excess.mean()/excess.std()))
        rows.append(row)
    result=pd.DataFrame(rows);save(result,out,'metrics.csv');save(worth.reset_index(),out,'price_worth.csv')
    return dict(rows=len(df),series=4,metrics_file='metrics.csv',original_annualization_preserved=True,diagnostic_columns_separate=True)

def backtrader_case(base,m,out):
    import backtrader as bt
    df=prices(base,m,'sz000001','2018-01-01','2023-01-01').set_index('date')
    events=[];daily=[]
    class DoubleAverages(bt.Strategy):
        def __init__(self):
            self.order=None
            self.sma5=bt.indicators.SimpleMovingAverage(self.data,period=5)
            self.sma10=bt.indicators.SimpleMovingAverage(self.data,period=10)
        def notify_order(self,order):
            if order.status in [order.Submitted,order.Accepted]:return
            events.append(dict(date=self.data.datetime.date(0).isoformat(),status=order.getstatusname(),side='buy' if order.isbuy() else 'sell',size=order.executed.size,price=order.executed.price,commission=order.executed.comm))
            self.order=None
        def next(self):
            daily.append(dict(date=self.data.datetime.date(0).isoformat(),cash=self.broker.getcash(),value=self.broker.getvalue(),position=self.position.size))
            if self.order:return
            if not self.position:
                if self.sma5[0]>self.sma10[0]:self.order=self.buy()
            elif self.sma5[0]<self.sma10[0]:self.order=self.sell()
    df['openinterest']=0
    cerebro=bt.Cerebro();cerebro.addstrategy(DoubleAverages)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(10000.0);cerebro.addsizer(bt.sizers.FixedSize,stake=100)
    cerebro.broker.setcommission(commission=.01);cerebro.run()
    save(pd.DataFrame(events),out,'orders.csv');save(pd.DataFrame(daily),out,'daily_account.csv')
    value=float(cerebro.broker.getvalue())
    assert np.isfinite(value)
    assert all(abs(x['size'])==100 for x in events if x['status']=='Completed')
    return dict(rows=len(df),initial_cash=10000,final_value=value,orders_completed=sum(x['status']=='Completed' for x in events),orders_other=sum(x['status']!='Completed' for x in events),stake=100,commission=.01,market_order_execution='next_bar_open_default')

def run_one(case,base,m,out):
    cid=case['id'];evidence=m.get('case_evidence',{}).get(cid,{})
    if evidence.get('status')=='blocked':raise ContractBlocked(evidence['reason'])
    if cid.startswith('W03_'):
        key=evidence.get('dataset')
        if not key:raise ContractBlocked('interface_equivalence_not_established')
        entry=m['datasets'][key]
        if entry.get('status')!='ready':raise ContractBlocked(entry.get('reason','dataset_unavailable'))
        df=load_frame(base,entry)
        missing=[f for f in case['required_fields'] if f not in df.columns]
        save(df,out,'available_interface.csv')
        partial=bool(missing) or evidence.get('status')=='partial'
        return dict(rows=len(df),missing_fields=missing,status_override='partial' if partial else 'completed',original_contract_status='terminated' if partial else 'completed_with_lab_data',diagnostic_status='completed',required_field_coverage=not missing,provider_response_equivalence_claimed=False,scope_note=evidence.get('scope_note','Same target/window laboratory data; provider changed.'))
    if cid=='W05_ma_values':
        df=prices(base,m,'sz000001','2023-05-01','2023-06-30')
        for n in [5,10,20,30]:
            mean=df.close.rolling(n).mean()
            ref=np.array([np.nan if i+1<n else math.fsum(df.close.iloc[i-n+1:i+1])/n for i in range(len(df))])
            np.testing.assert_allclose(mean,ref,atol=1e-12,rtol=0,equal_nan=True)
            df['M'+str(n)]=mean.round(2)
        save(df.sort_values('date',ascending=False),out,'moving_averages.csv')
        return dict(rows=len(df),windows=[5,10,20,30],independent_manual_windows_passed=True)
    if cid=='W05_ma_strategy':
        df=prices(base,m,'sz000001','2021-01-01','2023-06-30')
        asc=original_ma_logic(df);desc=original_ma_logic(df.iloc[::-1].reset_index(drop=True))
        save(asc,out,'original_logic_ascending.csv');save(desc,out,'original_logic_descending.csv')
        diagnostic=df.copy();diagnostic['position']=(df.close.rolling(5).mean()>df.close.rolling(20).mean()).astype(int).shift(1).fillna(0)
        diagnostic['return']=df.close.pct_change(fill_method=None)*diagnostic.position
        diagnostic['worth']=(1+diagnostic['return']).cumprod();save(diagnostic,out,'separate_long_only_diagnostic.csv')
        assert set(asc.position.dropna().unique()).issubset({-2.,0.,2.})
        return dict(rows=len(df),original_logic_ascending_final=float(asc.cumulative_return.iloc[-1]),original_logic_descending_final=float(desc.cumulative_return.iloc[-1]),diagnostic_long_only_final=float(diagnostic.worth.iloc[-1]),upstream_logic_retained=True,input_order_sensitivity_explicit=True,warning='signal.diff is a change in signal, not a held position; these are code diagnostics, not validated trading returns.')
    if cid=='W05_macd':
        import talib
        df=prices(base,m,'sh000300_index','2015-01-01','2018-12-31')
        dif,dea,hist=talib.MACD(df.close.to_numpy(dtype=float))
        np.testing.assert_allclose(hist,dif-dea,atol=1e-12,equal_nan=True)
        sig=pd.Series(hist>0,index=df.index).shift(1).fillna(False).astype(int)
        df['dif']=dif;df['dea']=dea;df['macd']=hist;df['position']=sig
        df['strategy_return']=sig*df.close.pct_change(fill_method=None)
        df['strategy_worth']=(1+df.strategy_return).cumprod();df['benchmark_worth']=df.close/df.close.iloc[0]
        save(df,out,'macd.csv')
        return dict(rows=len(df),strategy_worth=float(df.strategy_worth.iloc[-1]),benchmark_worth=float(df.benchmark_worth.iloc[-1]),fees=0,macd_identity_passed=True)
    if cid=='W07_metrics':return metrics_case(base,m,out)
    if cid=='W07_backtrader':return backtrader_case(base,m,out)
    raise ContractBlocked('continuous_contract_identity_or_roll_adjustment_not_established')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--contracts',default=str(Path(__file__).with_name('whale_core_contracts.json')));p.add_argument('--case',action='append',help='Run only selected contract IDs')
    a=p.parse_args();base=Path(a.input_dir).resolve();out=Path(a.output_dir).resolve()
    if out==base or out.is_relative_to(base):raise ValueError('Output must be separate from immutable inputs')
    out.mkdir(parents=True,exist_ok=True)
    contracts=json.loads(Path(a.contracts).read_text(encoding='utf-8'))
    m=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    results=[]
    if a.case and not set(a.case).issubset({c['id'] for c in contracts['cases']}):raise ValueError('Unknown contract ID')
    for case in contracts['cases']:
        if a.case and case['id'] not in a.case:continue
        d=out/case['id'];d.mkdir(exist_ok=True)
        try:
            result=run_one(case,base,m,d);status=result.pop('status_override','completed')
            row=dict(id=case['id'],status=status,result=result)
        except ContractBlocked as exc:row=dict(id=case['id'],status='blocked',reason=str(exc))
        except Exception as exc:
            (d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            row=dict(id=case['id'],status='failed',reason=type(exc).__name__+': '+str(exc))
        results.append(row)
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(case['id']+': '+row['status'],flush=True)
    packages={}
    for name in ['numpy','pandas','scipy','TA-Lib','backtrader']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    (out/'run_receipt.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=digest(Path(__file__)),contract_sha256=digest(Path(a.contracts)),input_manifest_sha256=digest(base/'manifest.json'),packages=packages,status_counts={s:sum(r['status']==s for r in results) for s in ['completed','partial','blocked','failed']},source_write_operations=0),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
