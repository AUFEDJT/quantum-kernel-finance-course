"""Strict, asset-specific laboratory adapter for QFB's 600519 branch only.

The verified interval has cash adjustments, no share changes. Reconstruct the
provider's additive front adjustment from each ex-date reference close, then
require cent-for-cent agreement with the previously frozen original provider.
This is intentionally not a generic adjustment implementation for other stocks.
"""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd

def prepare(root, reference):
    root=Path(root);p=root/'supplement/stock-trading-data-pro.csv'
    x=pd.read_csv(p);x['date']=pd.to_datetime(x['交易日期']);x=x.set_index('date').sort_index()
    assert x.index.is_unique and (x['收盘价']>0).all() and (x['前收盘价']>0).all()
    delta=x['收盘价'].shift()-x['前收盘价']
    delta=delta.where(delta.abs()>0.02,0)
    assert (delta>=0).all(), 'Unexplained adjustment event: require manual corporate-action audit'
    future_adjustment=delta.iloc[::-1].cumsum().iloc[::-1]-delta
    x['close']=(x['收盘价']-future_adjustment).round(2)
    y=pd.read_csv(reference);y['date']=pd.to_datetime(y['date']);y=y.set_index('date')
    assert x.index.equals(y.index), 'Original interval has missing or additional trading dates'
    assert np.array_equal(x['close'].to_numpy(),y['close'].to_numpy()), 'Reconstructed adjustment differs from original provider'
    result={'status':'pass','source':'laboratory stock-trading-data-pro','rows':len(x),
            'source_slice_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'reference_snapshot_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
            'all_dates_equal':True,'all_rounded_close_values_equal':True,
            'rounding_decimals':2,'input_role':'laboratory prices; reference only validates, never fills missing values',
            'adjustment':'additive front-adjustment using future (previous actual close - ex-date reference close)',
            'scope':'sh600519 and this frozen interval only; not a generic stock adjustment method'}
    return x[['close']].reset_index(),result

def install():
    import akshare as ak
    root=Path(os.environ['QFB_LAB_ROOT']);reference=Path(os.environ['QFB_LAB_REFERENCE'])
    frame,report=prepare(root,reference)
    out=Path(os.environ['QFB_EVIDENCE_DIR']).parent
    out.joinpath('lab_adjustment_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    def fetch(symbol,start_date,end_date,adjust='qfq',**kwargs):
        assert symbol=='sh600519' and adjust=='qfq','Asset or adjustment outside verified laboratory contract'
        a,b=pd.Timestamp(start_date),pd.Timestamp(end_date)
        assert a==frame.date.min() and b==pd.Timestamp('2026-11-19'),'Frozen interval is not this requested interval'
        return frame.loc[frame.date.between(a,b)].copy()
    ak.stock_zh_a_hist_tx=fetch
    os.environ['QFB_PROVIDER_OVERRIDE']='laboratory stock-trading-data-pro; derived and validated qfq; no network'
