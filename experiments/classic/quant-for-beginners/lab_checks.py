import json,os
from pathlib import Path
import numpy as np
assert len(df)==500 and df.index.is_unique and df.index.is_monotonic_increasing
assert df['Close'].notna().all() and (df['Close']>0).all()
assert np.allclose(df['MA5'].dropna(),df['Close'].rolling(5).mean().dropna())
assert np.allclose(df['MA20'].dropna(),df['Close'].rolling(20).mean().dropna())
assert df['signal'].equals((df['MA5']>df['MA20']).astype(int))
Path(os.environ['QFB_EVIDENCE_DIR']).parent.joinpath('lab_checks.json').write_text(json.dumps({'status':'pass','rows':len(df),'unique_dates':True,'MA5_and_MA20_independent_recalculation':True,'signal_formula':True,'golden_crosses':len(golden),'death_crosses':len(death)},indent=2),encoding='utf8')
