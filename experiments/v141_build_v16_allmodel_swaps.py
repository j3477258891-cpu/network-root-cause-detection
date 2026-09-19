"""Build the two current-test swaps supported by V16 and all V30 models."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong");EXP=ROOT/"experiments";BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv";DATA=EXP/"v25_semantic_router/cloud_dataset/v25_semantic_router.npz";REC=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz";OUT=EXP/"v141_v16_allmodel_swaps"
TARGETS=[('9ca5e12f-f73f-472c-99b3-8e74c01272fe','#-1:878ee284-26b7-41c1-b7dc-47e528d71504','#-1:247d7153-ba7f-4012-86c3-2d6b9b7e9730'),('6600d905-9faf-4ee3-8d44-223e9bc4dfdb','#-1:f38934ea-7530-401b-b354-bde0e9a2de71','#-1:960a3406-2124-4963-ab8c-91a9b9cd77c0'),('dd29409d-298e-451b-8231-e0c53c0e420a','#-1:8b8bf8e9-9748-4726-8fd5-e9565379f047','#-1:dd1a4fe7-36f6-40d6-976a-61e7f8d57656'),('29e6bb98-ef14-4333-920b-0cd42ff58a96','#-1:95ce5b4b-39bf-4501-82ca-9aca6b9ee799','#-1:c94052a7-1334-454b-bbd2-2dc1f0ee047b')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 order,base={},{}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):order[r['order_id']]=r['order_id'];base[r['order_id']]=json.loads(r['output'])['rootcause']
 with gzip.open(REC,'rt',encoding='utf-8') as f:rec=json.load(f)['test']
 rmap={o['order_id']:o for o in rec};out={oid:list(v) for oid,v in base.items()}
 meta=[]
 for oid,remove,add in TARGETS:
  cur=out[oid];assert any(n['@rid']==remove for n in cur);assert all(n['@rid']!=add for n in cur)
  out[oid]=[n for n in cur if n['@rid']!=remove];alarm=next(a for a in rmap[oid]['alarms'] if a['rid']==add);s=alarm.get('source',{});out[oid].append({'@rid':add,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')});meta.append({'order_id':oid,'remove_rid':remove,'add_rid':add})
 total=sum(len(v) for v in out.values());assert total==1035
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'v141_v16_allmodel_top4_from_pair015.csv'
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 report={'version':'v141-v16-highmargin-swaps','base':str(BASE),'base_sha256':sha(BASE),'actions':meta,'predictions':total,'orders':len(order),'sha256':sha(path),'oof_reference':'OOF: V16 seed margin >=0.5 with all three V30 margins >=0.2 had 2/2 positive; current four targets are V16-high-margin and unscored online.','status':'exploratory_not_scored','warning':'Model-derived evidence only; no online score yet.'}
 (OUT/'v141_v16_allmodel_top4_from_pair015.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
