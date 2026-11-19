"""Evaluate the WhaleQuant research lesson inputs admitted by the lab checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
def main():
 a=argparse.ArgumentParser()
 a.add_argument('--inputs',type=Path,required=True);a.add_argument('--panel',type=Path,required=True)
 a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 info=json.loads((args.inputs/'admission.json').read_text(encoding='utf-8'))
 for item in info['snapshots']:
  assert hashlib.sha256((args.inputs/item['file']).read_bytes()).hexdigest()==item['sha256']
 rec=info['contracts'];results=[]
 def add(ch,lab,status,obs): results.append(dict(chapter=ch,experiment=lab,status=status,observation=obs))
 # The chapter supplies no fixed market or dates for this cross-sectional example.
 d=pd.read_csv(args.panel,usecols=['symbol','date','close','prev_close','total_market_cap'])
 assert not d.duplicated(['symbol','date']).any() and d.prev_close.gt(0).all()
 d['r']=d.close/d.prev_close-1; used=0;dates=0;error=0
 for _,g in d.dropna().groupby('date'):
  if len(g)<3 or g.total_market_cap.nunique()<2:continue
  x=g.total_market_cap;y=g.r
  mod=sm.OLS(y,sm.add_constant(x)).fit()
  a=y-mod.params['total_market_cap']*x
  # The upstream subtracts the size exposure only. The mean can remain nonzero.
  ref=y-(x-x.mean()).dot(y-y.mean())/((x-x.mean())**2).sum()*x
  error=max(error,float(abs(a-ref).max()))
  assert np.isfinite(a).all()
  used+=len(g);dates+=1
 assert dates>0 and error < 1e-7,(dates,error)
 add(4,'4.4 市值中性化','通过',f'冻结20股日线共有 {dates} 个可估截面。按原式减去市值斜率乘市值，结果与独立中心化公式最大绝对差 {error:.2g}；原式保留截距。')
 add(4,'4.5 PE/PB/ROE 多因子回归','终止','原示例未提供 stock_data.csv，也未给出 Returns 的持有期和配对时点。财务与行情列无法唯一恢复原回归目标。')
 port=rec['portfolio']
 if not port.get('adjusted_fields'):
  add(6,'6.3–6.4 上证50调仓','终止',f"原例使用运行时成分及其日期，并取后复权日线。所核扩展日线截至 {port.get('latest','未知')}，成分标记不能重建本次指定时点的官方成分日期与完整后复权窗口。")
 else:add(6,'6.3–6.4 上证50调仓','须核对','字段可能满足约束，需按原成分日验证。')
 if rec['wavelet']['eligible']:
  check=args.out/'wavelet_independent_quotes.csv'
  source=pd.read_csv(args.inputs/'wavelet.csv').set_index('date').sort_index()
  if not check.is_file():raise ValueError('Independent active-trading calendar/close check required')
  ref=pd.read_csv(check);ref=ref[ref.tradestatus.astype(str)=='1'].set_index('date').sort_index()
  if not source.index.equals(ref.index) or not np.allclose(source.close,ref.close,rtol=0,atol=.0051):
   raise ValueError('Daily closes or active trading sessions disagree; do not accept the SVR run')
  import pywt
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.svm import LinearSVR
  from sklearn.metrics import mean_squared_error,mean_absolute_error
  x=pd.read_csv(args.inputs/'wavelet.csv').sort_values('date').reset_index(drop=True);scaler=MinMaxScaler();data=scaler.fit_transform(x[['close']]).ravel()
  w=pywt.Wavelet('db8');cs=pywt.wavedec(data,w,level=pywt.dwt_max_level(len(data),w.dec_len))
  for i in range(1,len(cs)):cs[i]=pywt.threshold(cs[i],0.05*max(cs[i]))
  den=pywt.waverec(cs,'db8');lags=30;xx=np.array([den[i-lags:i] for i in range(lags,len(data))]);yy=data[lags:]
  split=int(x.index[x.date=='2021-12-31'][0])-lags
  mod=LinearSVR(random_state=2023,max_iter=1000).fit(xx[:split],yy[:split]); pred=mod.predict(xx[split:]);assert np.isfinite(pred).all() and len(pred)==len(yy[split:])
  raw_pred=scaler.inverse_transform(pred.reshape(-1,1)).ravel()
  raw_test=x.close.to_numpy()[lags+split:];assert len(raw_test)==len(raw_pred)
  pd.DataFrame({'date':x.date.iloc[lags+split:],'actual_close':raw_test,'predicted_close':raw_pred}).to_csv(args.out/'wavelet_evaluation.csv',index=False)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,4));days=pd.to_datetime(x.date.iloc[lags+split:]);ax.plot(days,raw_test,label='Observed');ax.plot(days,raw_pred,label='Predicted');ax.set_ylabel('Close (CNY)');ax.legend();fig.tight_layout();fig.savefig(args.out/'wavelet_evaluation.png');plt.close(fig)
  (args.out/'wavelet_metrics.json').write_text(json.dumps({'rows':len(pred),'mse':mean_squared_error(yy[split:],pred),'mae':mean_absolute_error(yy[split:],pred)}))
  add(8,'8.1 小波去噪与价格预测','通过','原标的、原窗口完成训练与评价。原例对全序列缩放并去噪，结果仅用于方法教学。')
 else:
  add(8,'8.1 小波去噪与价格预测','终止',f"原例从 2015 年至运行前一日。实验室可核日线截至 {rec['wavelet'].get('until','未知')}，未覆盖所需终点 {rec['wavelet'].get('expected_to','未知')}。")
 if rec['decomposition']['eligible']:
  d=pd.read_csv(args.inputs/'sp500.csv').sort_values('date').set_index('date');s=seasonal_decompose(d.close,model='additive',period=252);mask=s.trend.notna()&s.resid.notna()
  assert np.allclose((s.trend+s.seasonal+s.resid)[mask],d.close[mask])
  add(8,'8.3 标普500时序分解','通过','以原指数在2015–2022年的日收盘价完成周期252的加性分解，并逐行核对趋势、季节项和残差之和。')
 else:add(8,'8.3 标普500时序分解','终止','实验室指数目录无法提供经确认的标普500价格指数及指定期间的完整记录；其他指数与ETF对应不同资产。')
 m=rec['lightgbm']
 add(8,'8.2 LightGBM 多股票分类','终止' if m['missing_count'] else '须核对',f"按原公司表清理后的 {m['pool']} 个标的中，{m['missing_count']} 个无法在两个日线源按原标识定位。缩小股票池会改变原任务。" if m['missing_count'] else '全部标的文件可定位，继续对齐字段、价格、成交单位及划分。')
 add(8,'8.5–8.6 BigQuant SVC 与 LSTM','终止','两例依赖 BigQuant 的历史因子、算子、标签或交易引擎。共享日线无法在无影独立还原平台输入及撮合口径。')
 lgb_file=args.out.parent/'lgb'/'summary.json'
 if lgb_file.exists():
  saved=json.loads(lgb_file.read_text(encoding='utf-8'))
  receipt=json.loads((args.inputs.parent/'lgb_input_receipt.json').read_text(encoding='utf-8'))
  if (saved.get('input_sha256')!=receipt['sha256'] or saved.get('total_pool')!=receipt['universe'] or receipt['issues'] or receipt['matched']!=receipt['universe'] or m['missing_count']):
   raise ValueError('LightGBM summary does not match the admitted input receipt/universe')
  if saved['rounds']==2000:
   for row in results:
    if row['experiment'].startswith('8.2 '):
     row.update(status='通过',observation=f"2,000 轮完成；验证准确率 {saved['validation']['accuracy']:.6f}，测试准确率 {saved['test']['accuracy']:.6f}。")
 (args.out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps([(i['experiment'],i['status']) for i in results],ensure_ascii=True))
if __name__=='__main__':main()




