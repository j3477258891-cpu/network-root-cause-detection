"""Build the two-swap set whose combined +1 TP was decoded online from V140-V139."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v145_verified_swap_pair"
TARGETS=[
 ('4c2b3b06-a6a2-4c1b-9734-def3d1f849d8','#-1:478860c0-cfb6-4e87-8e69-14f09e94b850','#-1:66facd91-e281-4331-bbae-04ab8bd9746d'),
 ('faa00ed5-c0b9-445e-98ac-671035a4a239','#-1:5173cc70-f178-481c-865b-a5a5f0d3c249','#-1:76e3a0a3-1943-4a4c-ad08-d31dc732cb73'),
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ids=[]; out={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f): ids.append(r['order_id']); out[r['order_id']]=json.loads(r['output'])['rootcause']
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f: rec=json.load(f)['test']
 rmap={r['order_id']:r for r in rec}; actions=[]
 for oid,rem,add in TARGETS:
  cur=out[oid]; assert any(x['@rid']==rem for x in cur); assert all(x['@rid']!=add for x in cur)
  a=next(a for a in rmap[oid]['alarms'] if a['rid']==add); s=a.get('source',{})
  out[oid]=[x for x in cur if x['@rid']!=rem]+[{'@rid':add,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')}]
  actions.append({'order_id':oid,'remove_rid':rem,'add_rid':add})
 roots=[x['@rid'] for v in out.values() for x in v]; assert len(ids)==546 and len(roots)==1035 and len(roots)==len(set(roots))
 OUT.mkdir(parents=True,exist_ok=True); stem='v145_verified_swap_pair_from_pair015'; p=OUT/f'{stem}.csv'
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']); w.writeheader()
  for oid in ids: w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 report={'version':'v145-online-decoded-two-swap-set','base':str(BASE),'base_sha256':sha(BASE),'actions':actions,'orders':546,'predictions':1035,'expected_tp':959,'expected_f1':round(2*959/(1044+1035),6),'sha256':sha(p),'status':'ready_for_online_confirmation','evidence':{'v139_score':0.919981,'v139_predictions':1043,'v139_tp':960,'v140_score':0.920939,'v140_predictions':1043,'v140_tp':961,'decoded_pair_delta_tp':1},'warning':'The pair-level +1 is public-score decoded; individual swap labels remain unresolved.'}
 (OUT/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
