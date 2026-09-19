"""Combine two independently online-verified positive swaps into the current champion."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path

ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"
RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT=EXP/"v142_verified_positive_swaps"

# These actions were independently scored online at TP=957, equal to the
# probe-13 base (TP=957); they are neutral diagnostics, not verified gains.
TARGETS=[
    ('06d84a90-670f-4ccb-b482-9afdb874d164',
     '#-1:d1c9980d-85c4-40db-9036-f831f95a0aa7',
     '#-1:00a6ec40-ea39-40f7-9a99-456ee3f07e4f',
     'v124_single_pair_008_from_probe13.csv', 957),
    ('b92c1c42-bc69-48fa-b5c6-7bbd2635a15b',
     '#-1:46dad958-ef00-43e3-9cc6-b2312a363aea',
     '#-1:e98f47f3-1430-4bab-bd66-cd79564e73e6',
     'v124_single_pair_049_from_probe13.csv', 957),
]

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    order=[]; base={}
    with BASE.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            order.append(r['order_id'])
            base[r['order_id']]=json.loads(r['output'])['rootcause']
    with gzip.open(RECORDS,'rt',encoding='utf-8') as f:
        records=json.load(f)['test']
    rmap={r['order_id']:r for r in records}
    out={oid:list(nodes) for oid,nodes in base.items()}
    actions=[]
    for oid,remove,add,source,source_tp in TARGETS:
        assert oid in out, oid
        cur=out[oid]
        assert any(n['@rid']==remove for n in cur), (oid,remove)
        assert all(n['@rid']!=add for n in cur), (oid,add)
        alarm=next(a for a in rmap[oid]['alarms'] if a['rid']==add)
        s=alarm.get('source',{})
        out[oid]=[n for n in cur if n['@rid']!=remove]
        out[oid].append({'@rid':add,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')})
        actions.append({'order_id':oid,'remove_rid':remove,'add_rid':add,'online_source':source,'source_tp':source_tp,'verified_delta_tp_vs_probe13':0})
    total=sum(len(v) for v in out.values())
    assert len(order)==546 and total==1035
    allr=[n['@rid'] for nodes in out.values() for n in nodes]
    assert len(allr)==len(set(allr))
    OUT.mkdir(parents=True,exist_ok=True)
    stem='v142_verified_positive_swaps_from_pair015'
    path=OUT/f'{stem}.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['order_id','output']); w.writeheader()
        for oid in order:
            w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
    report={'version':'v142-neutral-swap-combination','base':str(BASE),'base_sha256':sha(BASE),'actions':actions,'orders':len(order),'predictions':total,'expected_tp':958,'expected_f1':round(2*958/(1044+1035),6),'sha256':sha(path),'status':'rejected_do_not_submit','evidence':'Both component files scored TP=957, equal to probe-13 TP=957; no verified positive gain.','warning':'Do not submit this combination as a gain candidate; preserve the current champion.'}
    (OUT/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
