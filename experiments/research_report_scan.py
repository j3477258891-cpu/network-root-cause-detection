"""Scan historical JSON reports for high offline F1 claims (research only)."""
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
hits=[]
def walk(x,path=''):
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(int,float)) and ('f1' in k.lower() or 'score' in k.lower()) and 0.93 <= float(v) <= 1.0:
                hits.append((float(v),path+'/'+str(k),v))
            walk(v,path+'/'+str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x): walk(v,path+f'[{i}]')
for p in ROOT.glob('experiments/**/*.json'):
    try: walk(json.loads(p.read_text(encoding='utf8')),str(p.relative_to(ROOT)))
    except Exception: pass
for v,p,x in sorted(hits,reverse=True)[:300]: print(f'{v:.9f}\t{p}')
