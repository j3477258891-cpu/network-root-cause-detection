"""Build the highest-seed-margin untested V16 swap from the current champion."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v143_v16_top1"
OID='dd29409d-298e-451b-8231-e0c53c0e420a'; REMOVE='#-1:8b8bf8e9-9748-4726-8fd5-e9565379f047'; ADD='#-1:dd1a4fe7-36f6-40d6-976a-61e7f8d57656'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ids=[]; out={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f): ids.append(r['order_id']); out[r['order_id']]=json.loads(r['output'])['rootcause']
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f: rec=json.load(f)['test']
 rm={r['order_id']:r for r in rec}; cur=out[OID]
 assert any(n['@rid']==REMOVE for n in cur); assert all(n['@rid']!=ADD for n in cur)
 a=next(a for a in rm[OID]['alarms'] if a['rid']==ADD); s=a.get('source',{})
 out[OID]=[n for n in cur if n['@rid']!=REMOVE]+[{'@rid':ADD,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')}]
 total=sum(len(v) for v in out.values()); allr=[n['@rid'] for v in out.values() for n in v]; assert len(ids)==546 and total==1035 and len(allr)==len(set(allr))
 OUT.mkdir(parents=True,exist_ok=True); stem='v143_v16_top1_dd294_from_pair015'; p=OUT/f'{stem}.csv'
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']); w.writeheader()
  for oid in ids: w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 report={'version':'v143-v16-top1-single-swap','base':str(BASE),'base_sha256':sha(BASE),'order_id':OID,'remove_rid':REMOVE,'add_rid':ADD,'v16_seed_min_margin':0.860405194239452,'predictions':total,'orders':len(ids),'expected_if_positive':0.922559,'sha256':sha(p),'status':'ready_for_online_test','evidence':'Highest seed_min_margin among currently valid untested V16 actions; model-derived only, no online score yet.'}
 (OUT/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
