"""Build the next cumulative probe on V145 while excluding the online-negative dd294 action."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"; BASE=EXP/"v145_verified_swap_pair/v145_verified_swap_pair_from_pair015.csv"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v147_skip_negative"
OID='6600d905-9faf-4ee3-8d44-223e9bc4dfdb'; REMOVE='#-1:f38934ea-7530-401b-b354-bde0e9a2de71'; ADD='#-1:960a3406-2124-4963-ab8c-91a9b9cd77c0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ids=[];out={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):ids.append(r['order_id']);out[r['order_id']]=json.loads(r['output'])['rootcause']
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f:rec=json.load(f)['test']
 rmap={r['order_id']:r for r in rec};cur=out[OID];assert any(x['@rid']==REMOVE for x in cur);assert all(x['@rid']!=ADD for x in cur)
 a=next(a for a in rmap[OID]['alarms'] if a['rid']==ADD);s=a.get('source',{});out[OID]=[x for x in cur if x['@rid']!=REMOVE]+[{'@rid':ADD,'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')}]
 roots=[x['@rid'] for v in out.values() for x in v];assert len(ids)==546 and len(roots)==1035 and len(roots)==len(set(roots))
 OUT.mkdir(parents=True,exist_ok=True);stem='v147_v145_plus_6600_skip_dd294';p=OUT/f'{stem}.csv'
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in ids:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 report={'version':'v147-v145-plus-6600-skip-negative','base':str(BASE),'base_sha256':sha(BASE),'excluded_online_negative_order':'dd29409d-298e-451b-8231-e0c53c0e420a','action':{'order_id':OID,'remove_rid':REMOVE,'add_rid':ADD,'v16_seed_min_margin':0.8343106766830353},'predictions':1035,'orders':546,'expected_if_positive':0.923521,'sha256':sha(p),'status':'ready_for_online_test','warning':'Only submit from the V145 champion; the earlier V146 step2 is invalid because it contains dd294.'}
 (OUT/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
