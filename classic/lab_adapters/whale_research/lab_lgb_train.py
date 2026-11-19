"""Run WhaleQuant 8.2's fixed universe, features, labels, splits and 2000-round model."""
import argparse,gc,hashlib,io,json,time,zipfile,os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix,accuracy_score
p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=8)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
receipt=json.loads(args.receipt.read_text(encoding='utf-8'))
if receipt['issues'] or receipt['universe']!=receipt['matched']:raise RuntimeError('Some original-universe inputs failed admission')
with args.archive.open('rb') as f:
 digest=hashlib.sha256()
 for b in iter(lambda:f.read(2**20),b''):digest.update(b)
assert digest.hexdigest()==receipt['sha256']
def progress(**d):
 d['unix_time']=time.time()
 temp=args.out/'status.tmp';temp.write_text(json.dumps(d));temp.replace(args.out/'status.json')
progress(stage='joining_inputs')
sets=[dict(x=[],y=[],closed_high=[],security=[],day=[]) for _ in range(3)]
with zipfile.ZipFile(args.archive) as z:
 meta=[row for row in json.loads(z.read('pool.json')) if row['rows']>=100]
 def read(item):return np.load(io.BytesIO(z.read(f"{item['id']:04}.npy")),allow_pickle=False)
 date_id={}
 for row in meta:
  for d in read(row)[:,0].astype(int):date_id.setdefault(d,len(date_id))
 for sid,row in enumerate(meta):
  v=read(row);day=v[:,0].astype(int);o,h,l,c,pc,vol,amt=v[:,1:].T;idx=np.array([date_id[d] for d in day]);n=len(date_id)
  # Match the original ts_date_id +/- k joins. Missing trading sessions remain missing.
  allclose=np.full(n+10,np.nan);allhi=allclose.copy();alllo=allclose.copy()
  allclose[idx+5]=c;allhi[idx+5]=h;alllo[idx+5]=l
  past=np.column_stack([allclose[idx+5-k] for k in range(1,6)])
  highs=np.column_stack([allhi[idx+5+k] for k in range(1,6)])
  lows=np.column_stack([alllo[idx+5+k] for k in range(1,6)])
  eligible=np.isfinite(past).all(axis=1)&np.isfinite(highs).all(axis=1)&np.isfinite(lows).all(axis=1)
  y=((np.max((highs-c[:,None])/c[:,None],axis=1)>.06)&(np.min((lows-c[:,None])/c[:,None],axis=1)>-.03)).astype('int8')
  # Tushare: raw change and pct_chg, volume in board lots, amount in thousands of CNY.
  change=np.round(c-pc,2);pct=np.round(100*(c-pc)/pc,4)
  X=np.column_stack((o,h,l,c,pc,change,pct,vol/100,amt/1000,np.full(len(c),sid),(c-pc)/pc,(o-pc)/pc,(h-pc)/pc,(l-pc)/pc,(c[:,None]-past)/past))
  if sid<2:
   # Independently reconstruct an included source row by date-id lookup.
   for line in np.flatnonzero(eligible)[::100]:
    lookup={int(k):j for j,k in enumerate(idx)};prev=[v[lookup[idx[line]-k],4] for k in range(1,6)];future=[v[lookup[idx[line]+k]] for k in range(1,6)]
    assert np.allclose(X[line,-5:],(c[line]-np.array(prev))/np.array(prev))
    assert y[line]==((max((a[2]-c[line])/c[line] for a in future)>.06) and (min((a[3]-c[line])/c[line] for a in future)>-.03))
  for s,(lo,hi) in zip(sets,[(20180701,20211231),(20220101,20220630),(20220701,20230630)]):
   mask=eligible&(day>=lo)&(day<=hi);s['x'].append(X[mask]);s['y'].append(y[mask]);s['closed_high'].append((c==h)[mask]);s['security'].append(np.full(mask.sum(),sid));s['day'].append(day[mask])
for s in sets:
 for k,v in list(s.items()):s[k]=np.concatenate(v)
assert all(len(s['y']) and s['x'].shape[1]==19 for s in sets)
tr,val,test=sets;N=[len(s['y']) for s in sets]
params={'learning_rate':.001,'boosting_type':'gbdt','objective':'binary','metric':'mse','num_leaves':128,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':5,'seed':1,'bagging_seed':1,'lambda_l1':.1,'feature_fraction_seed':7,'min_data_in_leaf':20,'nthread':args.threads,'verbose':-1}
dtrain=lgb.Dataset(tr.pop('x'),tr['y']);dval=lgb.Dataset(val['x'],val['y'],reference=dtrain);gc.collect()
progress(stage='training',round=0,rows=N)
def monitor(env):
 if (env.iteration+1)%50==0:progress(stage='training',round=env.iteration+1,rows=N)
model=lgb.train(params,dtrain,2000,valid_sets=[dtrain,dval],callbacks=[monitor]);assert model.current_iteration()==2000
model.save_model(str(args.out/'model.txt'))
out={'rounds':model.current_iteration(),'total_pool':receipt['universe'],'eligible_stocks':len(meta),'split_rows':N,'features':19,'input_sha256':receipt['sha256'],'threads':args.threads}
for k,s,threshold in [('validation',val,.5),('test',test,.6)]:
 prob=model.predict(s['x'],num_iteration=model.best_iteration)
 label=np.round(prob).astype(int) if k=='validation' else (prob>=threshold).astype(int)
 assert np.isfinite(prob).all()
 out[k]={'accuracy':float(accuracy_score(s['y'],label)),'confusion':confusion_matrix(s['y'],label).tolist(),'predicted_positive':int(label.sum()),'tp_over_predicted':float(s['y'][label==1].mean()) if np.any(label) else None}
 if k=='test':
  picked=(label==1)&~s['closed_high']; out['buy_filter']={'selected':int(picked.sum()),'positive_labels':int(s['y'][picked].sum())}
  np.savez_compressed(args.out/'test_predictions.npz',prob=prob,selected=picked,sid=s['security'],date=s['day'])
(args.out/'summary.json').write_text(json.dumps(out,indent=2));progress(stage='complete',round=2000,rows=N)

