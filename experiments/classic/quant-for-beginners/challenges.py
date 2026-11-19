"""Executable responses to the fixed upstream's quantitative challenge prompts.

Choices left open by the source are explicit: TSLA (chapter 1), MSFT (chapter 2),
TSLA MA 5/20 and 10/30 (chapter 3), NVDA periods 1y/2y/5y (chapter 4).
These are extensions, not changes to the source Notebook's executed results.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def require_nonempty(obj, minimum=2):
    if obj is None or len(obj)<minimum:raise ValueError('Required original-provider observations unavailable or insufficient')
    if hasattr(obj,'isna') and np.asarray(obj.isna()).all():raise ValueError('All input observations missing')
    return obj


def yf_data(symbol,period):
    return require_nonempty(yf.download(symbol,period=period,progress=False,multi_level_index=False).dropna())


def mdd(r):
    nav=(1+r).cumprod();return float((nav/nav.cummax()-1).min())


def backtest(raw,short=5,long=20,size=1.0):
    c=require_nonempty(raw['Close'],long+1)
    ret=c.pct_change().fillna(0)
    sig=(c.rolling(short).mean()>c.rolling(long).mean()).astype(float)
    strategy=sig.shift(1).fillna(0)*ret*size
    return {'rows':len(c),'total_return':float((1+strategy).prod()-1),'maximum_drawdown':mdd(strategy),
            'buy_signals':int((sig.diff()>0).sum()),'sell_signals':int((sig.diff()<0).sum())},strategy


def run(g):
    source=os.environ['QFB_SOURCE_PATH'];out=Path(os.environ['QFB_EVIDENCE_DIR']).parent
    results=[]
    def task(name,fn):
        try:
            value=fn();results.append({'id':name,'status':'passed','result':value})
        except Exception as e:
            results.append({'id':name,'status':'blocked_or_failed','error_type':type(e).__name__,'reason':str(e)[:1500]})
        (out/'challenges.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    def savefig(name):
        plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=110);plt.close()
    if 'phase1_intro/01_' in source:
        def swapped():
            d=yf_data('TSLA','6mo');fig,axes=plt.subplots(2,1,sharex=True);axes[0].plot(d.index,d['Close']);axes[1].bar(d.index,d['Volume']);savefig('challenge_TSLA_price_volume')
            return {'symbol':'TSLA','last_close':float(d.Close.iloc[-1]),'sample_return':float(d.Close.iloc[-1]/d.Close.iloc[0]-1),'rows':len(d)}
        task('ch1_swap_symbol_price_volume',swapped)
        def check_simulations():
            assert np.isclose(g['kelly_fraction'],0.04)
            assert g['curves_A'].shape==(200,1001)
            return {'kelly_fraction':g['kelly_fraction'],'curves_shape':g['curves_A'].shape}
        task('ch1_simulation_invariants',check_simulations)
    elif 'phase1_intro/02_' in source:
        def alternative():
            d=yf_data('MSFT','1y');r=d.Close.pct_change().dropna();plt.figure();plt.hist(r,bins=30);savefig('challenge_MSFT_histogram')
            return {'symbol':'MSFT','rows':len(r),'daily_sample_std':float(r.std())}
        task('ch2_alternative_company_histogram',alternative)
        task('ch2_simple_return_handcheck',lambda: {'100_to_110':110/100-1,'expected':0.1,'matches':bool(np.isclose(110/100-1,0.1))})
    elif 'phase1_intro/03_' in source:
        def moving():
            d=yf_data('TSLA','2y');a,_=backtest(d);b,_=backtest(d,10,30)
            c=d.Close;plt.figure();plt.plot(c,label='TSLA');plt.plot(c.rolling(5).mean(),label='MA5');plt.plot(c.rolling(20).mean(),label='MA20');plt.legend();savefig('challenge_TSLA_ma')
            return {'symbol':'TSLA','ma5_20':a,'ma10_30':b}
        task('ch3_TSLA_and_MA10_30',moving)
    elif 'phase1_intro/04_' in source:
        for period in ['1y','2y','5y']:
            def bt(period=period):
                d=yf_data('NVDA',period);spy=yf_data('SPY',period);m,r=backtest(d)
                aligned=pd.DataFrame({'strategy':r,'benchmark':spy.Close.pct_change().fillna(0)}).dropna()
                nav=(1+aligned).cumprod();nav.plot();savefig('challenge_NVDA_'+period)
                return dict(m,period=period,benchmark_total_return=float(nav.benchmark.iloc[-1]-1),strategy_total_return_aligned=float(nav.strategy.iloc[-1]-1))
            task('ch4_NVDA_'+period,bt)
    elif 'phase2_intro/01_' in source:
        task('ch5_starter_completion',lambda:{'AMD_volatility':float(g['lv1_ann_vol']),'ranked_symbols':list(g['cand_vols'].index),'sample_count':len(require_nonempty(g['lv2_rets']))})
        def independent_vol():
            r=require_nonempty(g['lv2_rets']['MSFT']);a=float(r.std());b=float(np.sqrt(sum((float(x)-float(r.mean()))**2 for x in r)/(len(r)-1)))
            assert np.isclose(a,b,rtol=1e-12,atol=1e-12);return {'sample_std_numpy':a,'hand_formula':b}
        task('ch5_volatility_handcheck',independent_vol)
    elif 'phase2_intro/02_' in source:
        task('ch6_sharpe_top3',lambda:require_nonempty(g['metrics']).sort_values('Sharpe',ascending=False).head(3).to_dict(orient='index'))
        def beta():
            r=require_nonempty(g['returns']);values={}
            for t in ['JPM','NVDA']:
                a,b=g['beta_vs_market'](r[t],r['SPY']);assert np.isclose(a,b,atol=1e-12);values[t]={'numpy':float(a),'pandas':float(b),'distance_from_1':abs(float(a)-1)}
            return values
        task('ch6_JPM_NVDA_beta',beta)
    elif 'phase2_intro/03_' in source:
        def sizing():
            p=require_nonempty(g['prices'],31);r=p.pct_change();d=pd.DataFrame({'p':p,'r':r});s=(p.rolling(10).mean()>p.rolling(30).mean()).astype(float).shift(1)
            x=pd.DataFrame({'r':r*s,'short':p.rolling(10).mean(),'long':p.rolling(30).mean()}).dropna()
            return {'ma10_30_full_mdd':mdd(x.r),'ma10_30_30percent_mdd':mdd(x.r*.3),'rows':len(x)}
        task('ch7_MA10_30_position30',sizing)
        def recovery():
            e=require_nonempty(g['equity']);dd=e/e.cummax()-1;pos=int(np.argmin(dd.to_numpy()));peak=float(e.cummax().iloc[pos]);future=e.iloc[pos+1:pos+61];recovered=future[future>=peak]
            return {'trough_date':str(e.index[pos]),'observed_following_days':len(future),'recovered_within_60':bool(len(recovered)),'recovery_date':str(recovered.index[0]) if len(recovered) else None,'right_censored':len(future)<60 and len(recovered)==0}
        task('ch7_60day_recovery',recovery)
    elif 'phase2_intro/04_' in source:
        for years in [1,3,5]:
            def alternative(years=years):
                d=require_nonempty(g['fetch_us_prices'](['TSLA','NVDA','KO','SPY'],years));r=d.pct_change().dropna();require_nonempty(r)
                import seaborn as sns
                plt.figure(figsize=(6,5));sns.heatmap(r.corr(),annot=True,vmin=-1,vmax=1,cmap='RdBu_r');savefig('challenge_alternative_correlation_'+str(years)+'y')
                (r.std()*np.sqrt(252)).plot.bar();savefig('challenge_alternative_volatility_'+str(years)+'y')
                nav=pd.DataFrame({'portfolio':r[['TSLA','NVDA','KO']].mean(axis=1),'SPY':r.SPY}).add(1).cumprod();nav.plot();savefig('challenge_alternative_equity_'+str(years)+'y')
                return {'years':years,'rows':len(r),'correlation':r.corr().to_dict(),'annual_volatility':(r.std()*np.sqrt(252)).to_dict()}
            task('ch8_alternative_symbols_'+str(years)+'y',alternative)
        def combinations():
            r=require_nonempty(g['returns']);four=r[['AAPL','MSFT','JPM','XLE']].mean(axis=1);two=.5*r.AAPL+.5*r.MSFT;three=r[['AAPL','JPM','XLE']].mean(axis=1)
            v=lambda x:float(x.std()*np.sqrt(252))
            avg=float((r[['AAPL','MSFT','JPM','XLE']].std()*np.sqrt(252)).mean())
            covariance=r[['AAPL','MSFT']].cov().to_numpy();manual=float(np.sqrt(np.array([.5,.5])@covariance@np.array([.5,.5])*252))
            assert np.isclose(v(two),manual,rtol=1e-12,atol=1e-12)
            rolling=r.AAPL.rolling(60).corr(r.MSFT);rolling.plot();savefig('challenge_rolling60_correlation')
            return {'four_vol':v(four),'two_tech_vol':v(two),'three_vol':v(three),'single_vol_mean':avg,'two_covariance_formula':manual,'four_less_than_average':v(four)<=avg+1e-12,'cumulative_return_difference_vs_SPY':float((1+four).prod()-(1+r.SPY).prod()),'rolling_correlation_valid_rows':int(rolling.notna().sum()),'lowest_pair':list(g['low_pair'])}
        task('ch8_portfolio_and_covariance_checks',combinations)
    notes={'scope':'Source-prescribed quantitative challenges; thought questions have explanatory answers in course prose. No social-media posting performed.',
           'equity_definition':'Original equal-weight daily-return series implies daily rebalancing; it is not constant-share buy-and-hold.',
           'correlation_one':'With all pairwise correlations equal to one and long-only weights summing to one, portfolio volatility is the weighted average of component volatilities, not necessarily one named constituent.'}
    (out/'challenge_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'challenge_results':results},ensure_ascii=False,default=str))
