"""Bounded, source-linked recovery of WhaleQuant teaching experiments.

WhaleQuant-derived excerpts remain CC BY-NC-SA 4.0 with attribution to
datawhalechina/whale-quant at 828009d910db5eb2c06fa14f4439421b235bef44.
Alternate-source checks are explicitly distinguished from exact reproduction.
"""
import argparse, ast, contextlib, datetime, hashlib, importlib, io, json, math
import os, re, shutil, signal, sys, time, traceback, types, warnings
from pathlib import Path
from inventory import extract_workflows

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--only',default='')
a=p.parse_args()
SOURCE=a.source.resolve(); OUT=a.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLBACKEND']='Agg'; os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
os.environ['XDG_CACHE_HOME']=str(OUT/'cache')
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
warnings.filterwarnings('ignore',category=FutureWarning)
WORKFLOWS=dict(extract_workflows(SOURCE)); ROWS=[]
W3='docs/ch03_股票数据获取/ch03_股票数据获取.md'
W6='notebook/ch06_量化调仓策略/chapter 6.3-6.4.ipynb'
W7='docs/ch07_量化回测/ch07_量化回测.md'
W81='notebook/ch08_机器学习与量化策略/ch08-01.ipynb'
W83='docs/ch08_机器学习与量化策略/8.3_时间序列预测与量化择时.md'
W85='docs/ch08_机器学习与量化策略/8.5_机器学习量化策略基础实战.md'
W86='docs/ch08_机器学习与量化策略/8.6_深度学习量化策略基础实战.md'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe_message(text):
    return re.sub(r'(?i)((?:token|password|secret)\s*[:=]\s*)[^\s,;}]+',r'\1[REDACTED]',str(text))
def alarm(*_): raise TimeoutError('bounded recovery experiment exceeded timeout')

def run(ident,source,section,fn,seconds=180):
    if a.only and a.only not in ident:return
    folder=OUT/ident; folder.mkdir(exist_ok=True)
    prev=Path.cwd(); os.chdir(folder); stream=io.StringIO();start=time.monotonic()
    row={'id':ident,'source':source,'section':section,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        signal.signal(signal.SIGALRM,alarm);signal.alarm(seconds)
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream): result=fn(folder)
        row.update(status='target_verified',**result)
    except BaseException as exc:
        row.update(status='blocked',observed_error=safe_message(type(exc).__name__+': '+str(exc)),traceback=safe_message(traceback.format_exc(limit=5)))
    finally:
        signal.alarm(0);row['elapsed_seconds']=round(time.monotonic()-start,3)
        (folder/'execution.log').write_text(safe_message(stream.getvalue()),encoding='utf-8')
        for i,num in enumerate(plt.get_fignums()):
            plt.figure(num).savefig(folder/f'figure-{i}.png',dpi=90)
        plt.close('all');os.chdir(prev)
        row['evidence']=ident+'/execution.log'
        row['artifacts']=[{'path':str(f.relative_to(OUT)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in folder.rglob('*') if f.is_file()]
        ROWS.append(row);(OUT/'recovery_results.json').write_text(json.dumps(ROWS,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row[k] for k in ('id','status','elapsed_seconds')},ensure_ascii=False),flush=True)

def getcode(source,needle):
    return next(code for loc,code,kind in WORKFLOWS[source] if needle in code)
def definitions(source,needle,ns,include_classes=False):
    code=getcode(source,needle);tree=ast.parse(code)
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) or include_classes and isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=keep,type_ignores=[]),source,'exec'),ns)
    return [n.name for n in keep]

def baostock_original(folder):
    # Preserve the original 8 provider examples; only fix Windows-specific output paths.
    ns={'pd':pd,'np':np};items=[]
    for loc,code,kind in WORKFLOWS[W3]:
        if not ('import baostock' in code):continue
        repaired=re.sub(r"(['\"])D:[/\\]+",r'\1',code)
        start=time.monotonic()
        exec(compile(repaired,W3+'#'+loc,'exec'),ns)
        frames={k:v for k,v in ns.items() if isinstance(v,pd.DataFrame)}
        target=next((v for k,v in reversed(list(frames.items())) if len(v)>0),None)
        assert target is not None,'provider example produced no nonempty frame'
        dest=folder/(loc+'.csv');target.to_csv(dest,index=False)
        items.append({'location':loc,'rows':len(target),'columns':list(target),'sha256':sha(dest),'seconds':round(time.monotonic()-start,3)})
    assert len(items)==8,(len(items),items)
    return {'scope':'original provider, symbols, periods and formulas; output path adapted','action':'Windows drive output paths changed to isolated relative paths','observed_result':items,'target_verified':True}

def public_prices(folder):
    import baostock as bs
    records=[]; login=bs.login();assert login.error_code=='0',login.error_msg
    try:
        for code in ['sz.000001','sh.600519','sh.601398','sh.601318']:
            fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,isST'
            rs=bs.query_history_k_data_plus(code,fields,start_date='2015-01-01',end_date='2023-06-30',frequency='d',adjustflag='3')
            assert rs.error_code=='0',rs.error_msg
            rows=[]
            while rs.next():rows.append(rs.get_row_data())
            df=pd.DataFrame(rows,columns=rs.fields);assert len(df)>1000
            target=folder/(code+'.csv');df.to_csv(target,index=False)
            records.append({'symbol':code,'rows':len(df),'min_date':df.date.min(),'max_date':df.date.max(),'sha256':sha(target),'provider':'BaoStock','adjustflag':'3'})
    finally:bs.logout()
    return {'scope':'documented alternate-source fixtures; not Tushare output','action':'downloaded real historical prices without credentials; retained privately','observed_result':records,'target_verified':True}

def price(code):
    df=pd.read_csv(OUT/'W-DATA-ALTERNATE'/f'{code}.csv')
    df['date']=pd.to_datetime(df['date'])
    for col in df.columns.difference(['date','code']):df[col]=pd.to_numeric(df[col],errors='coerce')
    return df.set_index('date').sort_index()

def pandas_scripts(folder):
    fixtures=folder/'data';fixtures.mkdir(exist_ok=True)
    for f in (SOURCE/'data/ch3').glob('*.csv'):shutil.copy2(f,fixtures/f.name)
    config=types.ModuleType('toolkit_config');config.DATADIR=str(fixtures);sys.modules['toolkit_config']=config
    helper=types.ModuleType('mod_inside_lec');exec((SOURCE/'code/ch03/mod_inside_lec.py').read_text(encoding='utf-8'),helper.__dict__)
    statuses=[]
    for f in sorted((SOURCE/'code/ch03').glob('*.py')):
        if f.name=='__init__.py':continue
        source=f.read_text(encoding='utf-8');source=source.replace('from code import mod_inside_lec','import mod_inside_lec')
        sys.modules['mod_inside_lec']=helper
        ns={'__name__':'__main__'};rows=[]
        for stmt in ast.parse(source).body:
            try:
                exec(compile(ast.Module(body=[stmt],type_ignores=[]),str(f.name),'exec'),ns)
                rows.append({'line':stmt.lineno,'status':'executed'})
            except Exception as exc:
                rows.append({'line':stmt.lineno,'status':'observed_exception','error':type(exc).__name__+': '+str(exc)})
        statuses.append({'source':'code/ch03/'+f.name,'statements':rows})
    (folder/'script_statements.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2),encoding='utf-8')
    data=pd.read_csv(fixtures/'qan_prc_2020.csv',parse_dates=['Date']).set_index('Date')
    value=float(data.loc['2020-01-06':'2020-01-10','Close'].mean());assert abs(value-6.982)<0.001
    assert (fixtures/'new_file.txt').exists()
    return {'scope':'all 13 nonempty source scripts, each top-level statement attempted; intentional invalid-index examples preserved','action':'provided toolkit_config.DATADIR; fixed stdlib code import collision; continued after each demonstrated exception','observed_result':{'scripts':len(statuses),'weekly_average':value,'statement_counts':{s:sum(r['status']==s for x in statuses for r in x['statements']) for s in ['executed','observed_exception']}},'remaining':'tsla_prc.csv absent upstream; datetime exercises requiring that specific file remain blocked','target_verified':True}

def timing(folder):
    import talib
    df=price('sz.000001').loc['2023-06-01':'2023-06-30'].copy()
    supplied=pd.read_csv(SOURCE/'notebook/ch05_量化择时策略/result.csv')
    assert len(supplied)>10
    # Exact upstream result fixture remains intact; compute fresh indicators on sufficient real history.
    series=price('sz.000001')['close'].loc['2018':'2022']
    out=pd.DataFrame({'close':series,'MA5':series.rolling(5).mean(),'MA10':series.rolling(10).mean()})
    out['signal']=(out.MA5>out.MA10).astype(int)
    macd,signal,hist=talib.MACD(series.to_numpy(),fastperiod=12,slowperiod=26,signalperiod=9)
    out['MACD']=macd;out['MACD_signal']=(out.MACD>0).astype(int)
    out['strategy_return']=out.signal.shift(1).fillna(0)*out.close.pct_change().fillna(0)
    assert len(out)>1000 and out.MACD.notna().sum()>1000
    assert out.strategy_return.iloc[0]==0
    out.to_csv(folder/'timing-signals.csv')
    return {'scope':'same MA/MACD teaching calculation on explicitly identified alternate-provider real data','action':'original Tushare/JoinQuant authentication remains separately blocked; supplemental pure signal entry does not submit orders','observed_result':{'rows':len(out),'signal_changes':int(out.signal.diff().abs().sum()),'source_result_rows':len(supplied)},'target_verified':True}

def portfolio(folder):
    ns={'np':np,'pd':pd,'minimize':minimize}
    definitions(W6,'def get_weights',ns)
    df=pd.concat([price(c)['close'].rename(c) for c in ['sz.000001','sh.600519','sh.601398','sh.601318']],axis=1).loc['2022'].pct_change(fill_method=None).dropna()
    weights={}
    for target in ['sharp','rp','var']:
        w=ns['get_weights'](df,target=target,canshort=False)
        assert abs(w.sum()-1)<1e-6 and w.min()>-1e-6 and np.isfinite(w).all()
        weights[target]={k:float(v) for k,v in w.items()}
    pd.DataFrame(weights).to_csv(folder/'weights.csv')
    return {'scope':'original three SLSQP objective functions exercised on an explicitly supplemental four-stock matrix','action':'provided named four-stock return matrix, retaining original objectives and long-only sum-to-one constraints','observed_result':{'samples':len(df),'weights':weights},'remaining':'Notebook standalone inputs are absent; the original Markdown defines the complete universe and data window, evaluated separately in W06-ORIGINAL-AKSHARE','target_verified':True}

def backtrader_case(folder):
    import backtrader as bt
    ns={'bt':bt};definitions(W7,'class DoubleAverages',ns,True)
    cls=ns['DoubleAverages']
    data=price('sz.000001').loc['2018-01-01':'2022-12-31'][['open','high','low','close','volume']].copy();data['openinterest']=0
    c=bt.Cerebro();c.addstrategy(cls);c.adddata(bt.feeds.PandasData(dataname=data));c.broker.setcash(10000.0);c.addsizer(bt.sizers.FixedSize,stake=100);c.broker.setcommission(commission=0.01)
    c.addanalyzer(bt.analyzers.TradeAnalyzer,_name='trades')
    result=c.run();trades=result[0].analyzers.trades.get_analysis()
    assert len(result[0])==len(data) and np.isfinite(c.broker.getvalue())
    assert trades.total.total>0
    result={'bars':len(data),'completed_trades':int(trades.total.closed),'final_value':float(c.broker.getvalue()),'initial_value':10000.0,'commission':0.01}
    (folder/'backtest.json').write_text(json.dumps(result),encoding='utf-8')
    return {'scope':'original DoubleAverages strategy and engine settings; named alternate-provider raw OHLCV for original symbol and time span','action':'extracted strategy class; bound local PandasData instead of credentialed Tushare getter','observed_result':result,'target_verified':True}

def stupidhead_case(folder):
    import talib
    from hyperopt import tpe,hp,fmin,STATUS_OK,Trials
    from hyperopt.pyll.base import scope
    code=getcode(W7,'def CTA(')
    # This is the implementation already printed in the upstream Markdown.
    code=code.replace("HQDf[:idx].pos = HQDf[:idx].pos.fillna(method='ffill')","HQDf.loc[:idx, 'pos'] = HQDf.loc[:idx, 'pos'].ffill()")
    code=code.replace("'SimHei'","'DejaVu Sans'")
    header='# Recovered from datawhalechina/whale-quant ch07; CC BY-NC-SA 4.0.\n# Change: explicit .loc forward-fill assignment for pandas compatibility.\n'
    (folder/'StupidHead.py').write_text(header+code,encoding='utf-8')
    ns={'pd':pd,'np':np,'plt':plt,'talib':talib};exec(code,ns)
    definitions(W7,'def doubleMa',ns)
    data=pd.read_csv(SOURCE/'notebook/ch07_量化回测/T888_1d_j84jdLF4_M.csv',parse_dates=['date']).set_index('date').sort_index()
    result,stats=ns['CTA'](data.copy(),40,ns['doubleMa'],fast=5,slow=20)
    expected=(1+result.close.pct_change().fillna(0)*result.pos.shift(1).fillna(0)).cumprod()
    np.testing.assert_allclose(result.strategy_balance,expected,atol=1e-12)
    assert len(result)==len(data) and np.isfinite(result.strategy_balance).all()
    result.to_csv(folder/'stupidhead-ledger.csv');(folder/'stats.json').write_text(json.dumps({k:float(v) for k,v in stats.items()}),encoding='utf-8')
    # Preserve the full 100-evaluation budget from the tutorial, with integer periods required by TA-Lib.
    space={'HQDf':data.copy(),'loadBars':40,'func':ns['doubleMa'],'fast':scope.int(hp.quniform('fast',3,30,1)),'slow':scope.int(hp.quniform('slow',5,40,1))}
    trials,best=ns['hypeFun'](space,'sharpe_ratio')
    assert len(trials.trials)==100 and all(t['result']['status']=='ok' for t in trials.trials)
    (folder/'optimization.json').write_text(json.dumps({'trials':len(trials.trials),'best':best}),encoding='utf-8')
    return {'scope':'upstream embedded framework plus original T888 daily fixture and 100-evaluation optimization','action':'restored missing standalone module from Markdown; explicit pandas assignment; integer TA-Lib periods','observed_result':{'rows':len(result),'ledger_max_abs_error':float(np.max(np.abs(expected-result.strategy_balance))),'optimization_trials':len(trials.trials),'best':best},'caveat':'upstream Sharpe formula/risk-free units and in-sample parameter optimization are retained; values are not validated investment-performance evidence','target_verified':True}

def decomposition(folder):
    import yfinance as yf
    from statsmodels.tsa.seasonal import seasonal_decompose
    plt.style.use('seaborn-v0_8')
    data=yf.download('^GSPC',start='2015-01-01',end='2022-12-31',auto_adjust=False,progress=False,timeout=20)
    assert len(data)>504,'original Yahoo endpoint returned insufficient data'
    close=data['Close'];close=close.iloc[:,0] if isinstance(close,pd.DataFrame) else close
    result=seasonal_decompose(close,model='additive',period=252);result.plot()
    frame=pd.DataFrame({'observed':result.observed,'trend':result.trend,'seasonal':result.seasonal,'resid':result.resid})
    valid=frame.dropna();assert len(valid)>252
    np.testing.assert_allclose(valid.observed,valid.trend+valid.seasonal+valid.resid,atol=1e-8)
    frame.to_csv(folder/'decomposition.csv')
    return {'scope':'original yfinance symbol/time window and additive period=252 decomposition','action':'updated removed matplotlib seaborn style name; handled current yfinance column shape; explicit unadjusted Close','observed_result':{'rows':len(data),'decomposed_rows':len(valid),'snapshot_sha256':sha(folder/'decomposition.csv')},'target_verified':True}

def wavelet_case(folder):
    code=getcode(W81,'ts.pro_bar(ts_code=')
    # Select complete source block containing model fitting, not the initial download-only cell.
    code=next(c for loc,c,k in WORKFLOWS[W81] if 'regr.fit' in c and 'ts.pro_bar' in c)
    tree=ast.parse(code);kept=[]
    for node in tree.body:
        seg=ast.get_source_segment(code,node) or ''
        if any(s in seg for s in ['ts.set_token','ts.pro_api','ts.pro_bar','ts_token']):continue
        kept.append(node)
    frame=price('sz.000001').reset_index();frame['trade_date']=frame['date'].dt.strftime('%Y%m%d');frame=frame.sort_values('date',ascending=False)
    ns={'pd':pd,'df':frame};exec(compile(ast.Module(body=kept,type_ignores=[]),W81,'exec'),ns)
    assert len(ns['pred_test'])==len(ns['y_test']) and len(ns['pred_test'])>100
    assert np.isfinite(ns['pred_test']).all()
    pd.DataFrame({'prediction':ns['pred_test'],'target':ns['y_test']}).to_csv(folder/'wavelet-predictions.csv',index=False)
    return {'scope':'original db8 wavelet/30-lag/20211231 split/LinearSVR pipeline on explicitly alternate-provider prices ending 20230630','action':'supplied missing real-price input; removed authenticated download statements only','observed_result':{'train_rows':len(ns['y_train']),'test_rows':len(ns['y_test']),'predictions_finite':True},'caveat':'upstream scaler and wavelet operate on the full sequence before time split; execution verifies mechanics, not leakage-free prediction','target_verified':True}

def platform_callbacks(folder):
    from types import SimpleNamespace as NS
    events=[]
    def record(name):return lambda *args,**kwargs:events.append((name,args,kwargs))
    ns={'np':np,'pd':pd,'math':math,'g':NS(),'log':NS(info=lambda *_:None),'set_benchmark':record('benchmark'),'set_option':record('option'),'set_order_cost':record('cost'),'OrderCost':lambda **kw:kw,'run_daily':record('schedule'),'order_value':record('mock_buy'),'order_target':record('mock_target'),'get_trades':lambda:{}}
    definitions(W7,'def market_open',ns)
    ctx=NS(current_dt=datetime.datetime(2020,1,2),portfolio=NS(available_cash=100.0,positions={'002594.XSHE':NS(closeable_amount=1)}))
    ns['initialize'](ctx);ns['before_market_open'](ctx)
    ns['get_bars']=lambda s,count,**kw:pd.DataFrame({'close':np.arange(11-count,11,dtype=float)})
    ns['market_open'](ctx);assert any(e[0]=='mock_buy' for e in events)
    ns['get_bars']=lambda s,count,**kw:pd.DataFrame({'close':np.arange(11-count,11,dtype=float)[::-1] if count==10 else np.arange(1,6,dtype=float)})
    ns['market_open'](ctx);assert any(e[0]=='mock_target' for e in events)
    ns['after_market_close'](ctx)
    class DataSource:
        @staticmethod
        def write_df(x):return x
        @staticmethod
        def write_pickle(x):return x
    ns85={'pd':pd,'np':np,'DataSource':DataSource,'Outputs':lambda **kw:NS(**kw),'PerOrder':lambda **kw:kw}
    definitions(W85,'def m6_run_bigquant_run',ns85)
    train=pd.DataFrame({'f':[1.,2.,3.,4.]});test=pd.DataFrame({'f':[-99.,99.]})
    wrap=lambda x:NS(read=lambda:x,read_df=lambda:x,read_pickle=lambda:x)
    result=ns85['m6_run_bigquant_run'](wrap(train.copy()),wrap(['f']),wrap(test.copy()))
    assert result.data_2.f.min()>=train.f.quantile(.005) and result.data_2.f.max()<=train.f.quantile(.995)
    mock_orders=[]
    signals=pd.DataFrame({'date':['2020-01-02']*2,'instrument':['A','B'],'pred_label':[1.,0.]})
    context=NS(options={'data':wrap(signals)},extension={},set_commission=lambda *_:None,portfolio=NS(positions=[]),symbol=lambda x:x,order_target_percent=lambda *x:mock_orders.append(x))
    ns85['m4_initialize_bigquant_run'](context);data=NS(current_dt=datetime.datetime(2020,1,2),can_trade=lambda _:True)
    for _ in range(5):ns85['m4_handle_data_bigquant_run'](context,data)
    assert mock_orders==[('A',1.0)]
    ns86={'pd':pd,'np':np,'math':math,'DataSource':DataSource,'Outputs':lambda **kw:NS(**kw)}
    definitions(W86,'def m4_run_bigquant_run',ns86)
    reshaped=ns86['m4_run_bigquant_run'](wrap({'x':np.arange(24).reshape(2,12)}),wrap(['a','b','c']),None)
    assert reshaped.data_1['x'].shape==(2,3,4)
    ranked=ns86['m24_run_bigquant_run'](wrap(np.array([[.2],[.9]])),wrap(signals),None)
    assert ranked.data_1.instrument.tolist()==['B','A']
    imports={}
    for mod in ['jqdata','jqdatasdk','biglearning','bigdatasource','bigtrader']:
        try:importlib.import_module(mod);imports[mod]='available'
        except Exception as exc:imports[mod]=type(exc).__name__+': '+str(exc)
    (folder/'mock-events.json').write_text(json.dumps(events,default=str),encoding='utf-8')
    return {'scope':'original callback functions under explicit in-memory mocks; no brokerage or platform connection','action':'verified buy/sell branches, train-derived clipping, fifth-day rebalance, sequence reshape and rank ordering','observed_result':{'callback_assertions':7,'platform_import_probes':imports,'submitted_real_orders':0},'remaining':'platform-hosted dataset modules, SVC/LSTM training and full platform backtest remain unexecuted without their supported runtimes','target_verified':True}

def original_akshare(folder):
    import akshare as ak
    source='docs/ch06_量化调仓策略/ch06_量化调仓策略.md'
    raw=getcode(source,'def get_ret')
    # One prose paragraph was accidentally placed inside the Python fence.
    code='\n'.join(line for line in raw.splitlines() if not line.startswith('数据和模型都准备好后'))
    tree=ast.parse(code)
    ns={'ak':ak,'pd':pd,'np':np,'datetime':datetime,'plt':plt,'minimize':minimize}
    definitions(W6,'def get_weights',ns)
    steps=[]
    try:
        for node in tree.body:
            began=time.monotonic()
            try:
                if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='code':
                    # Each constituent request is independent. Keep all failures,
                    # and never silently optimize a reduced constituent universe.
                    history=[]
                    for stock in ns['stock_codes']:
                        try:
                            ret=ns['get_ret'](stock,start_date=ns['start_date'],end_date=ns['end_date'])
                            assert len(ret)>1,'original hfq endpoint returned insufficient rows'
                            ns['ret_list'].append(ret)
                            ret.to_csv(folder/('hfq-return-'+stock+'.csv'))
                            history.append({'symbol':stock,'status':'retrieved','rows':len(ret)})
                        except BaseException as exc:
                            history.append({'symbol':stock,'status':'blocked','error':safe_message(type(exc).__name__+': '+str(exc))})
                        (folder/'history-requests.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
                    assert all(row['status']=='retrieved' for row in history),f"original hfq requests failed for {sum(row['status']=='blocked' for row in history)}/{len(history)} constituents; original full universe not reduced"
                    steps.append({'line':node.lineno,'status':'executed','seconds':round(time.monotonic()-began,3)})
                    continue
                exec(compile(ast.Module(body=[node],type_ignores=[]),source,'exec'),ns)
                steps.append({'line':node.lineno,'status':'executed','seconds':round(time.monotonic()-began,3)})
            except BaseException as exc:
                steps.append({'line':node.lineno,'status':'blocked','error':safe_message(type(exc).__name__+': '+str(exc))})
                raise
        assert 'df_ret' in ns and not ns['df_ret'].empty,'original history returned no common daily-return rows'
        records=ns.get('records',[])
        assert len(records)>0,'original official constituent date produced zero rolling rebalancing dates'
        return {'scope':'original 000016 official constituents, official-date minus one year to run date, hfq history and original rolling objectives','action':'restored get_weights from the original Notebook and removed the one prose line accidentally embedded in the Markdown code fence','observed_result':{'constituents':len(ns['stock_codes']),'return_rows':len(ns['df_ret']),'rolling_rows':len(records)},'target_verified':True}
    finally:
        (folder/'steps.json').write_text(json.dumps(steps,ensure_ascii=False,indent=2),encoding='utf-8')
        for key,value in ns.items():
            if isinstance(value,pd.DataFrame):value.to_csv(folder/(key+'.csv'),index=True)
        inputs={key:str(ns[key]) for key in ('start_date','end_date','stock_codes') if key in ns}
        (folder/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')

run('W03-BAOSTOCK',W3,'3.3 technical and fundamental data',baostock_original)
run('W-DATA-ALTERNATE',W3,'documented supplemental data adapter',public_prices)
run('W03-PANDAS-SCRIPTS','code/ch03','file IO, Series, indexing, joins, groupby, dates',pandas_scripts)
run('W05-SIGNALS','docs/ch05_量化择时策略/ch05_量化择时策略.md','5.3 and 5.4',timing)
run('W06-OPTIMIZATION',W6,'6.3 and 6.4',portfolio)
run('W06-ORIGINAL-AKSHARE','docs/ch06_量化调仓策略/ch06_量化调仓策略.md','6.4 original constituent and hfq rolling pipeline',original_akshare,seconds=900)
run('W07-BACKTRADER',W7,'7.3',backtrader_case)
run('W07-STUPIDHEAD',W7,'7.5',stupidhead_case,seconds=900)
run('W08-DECOMPOSITION',W83,'1.1',decomposition,seconds=70)
run('W08-WAVELET',W81,'complete implementation cell-36',wavelet_case)
run('W-PLATFORM-CALLBACKS',[W7,W85,W86],'JoinQuant and BigQuant teaching callbacks',platform_callbacks)
