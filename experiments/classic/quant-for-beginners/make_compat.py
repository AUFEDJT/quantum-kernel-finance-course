"""Create attributed runtime copies; preserve original files and task/data semantics."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();src=Path(a.source_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);changes=[]
    replacements={
        'ak.stock_us_daily(':'ak.stock_us_hist_sina(',
        "df.last('6M')":"df.loc[df.index > (df.index[-1] - pd.offsets.MonthEnd(6))]",
        "df.last('12M')":"df.loc[df.index > (df.index[-1] - pd.offsets.MonthEnd(12))]",
        "plt.rcParams['font.sans-serif'] = ['SimHei']":"plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']",
    }
    for source in sorted(src.glob('notebooks/*/*.ipynb')):
        j=json.loads(source.read_text(encoding='utf8'));rel=source.relative_to(src);patches=[]
        for i,c in enumerate(j['cells']):
            if c['cell_type']!='code':continue
            text=''.join(c['source'])
            for old,new in replacements.items():
                if old in text:text=text.replace(old,new);patches.append({'cell':i+1,'before':old,'after':new})
            c['source']=text.splitlines(keepends=True);c['outputs']=[];c['execution_count']=None
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(j,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
        changes.append({'source_path':rel.as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runtime_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'changes':patches})
    (out/'compatibility_manifest.json').write_text(json.dumps({'upstream':'https://github.com/datawhalechina/quant-for-beginners','commit':'8fbf8942afc0e80d01538ee545e72174e099e81a','license':'MIT; preserve the upstream LICENSE','reason':'AkShare official interface rename, pandas removed last method, installed CJK font portability','references':['https://akshare.akfamily.xyz/changelog.html#changelog-rename'],'files':changes},ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
