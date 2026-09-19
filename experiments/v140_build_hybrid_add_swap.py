"""Combine V139 high-confidence additions with disjoint V136 swaps."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
ROOT=Path(r"D:\zgyidong");EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"
ADD=EXP/"v139_unanimous_top20_additions/v139_unanimous_top20_single_root_add_top8_from_pair015.csv"
SWAP=EXP/"v136_consensus_swaps/v136_consensus_positive_swap_top8_t0p2_from_pair015.csv"
OUT=EXP/"v140_hybrid_add_swap"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 ids=[];d={}
 with p.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):ids.append(r['order_id']);d[r['order_id']]=json.loads(r['output'])['rootcause']
 return ids,d
def main():
 order,base=load(BASE);_,add=load(ADD);_,swap=load(SWAP)
 out={oid:list(nodes) for oid,nodes in base.items()}; add_orders=[]; swap_orders=[]
 for oid in order:
  b={x['@rid'] for x in base[oid]}; a={x['@rid'] for x in add[oid]}; s={x['@rid'] for x in swap[oid]}
  if a!=b:
   out[oid]=list(add[oid]);add_orders.append(oid)
  elif s!=b:
   out[oid]=list(swap[oid]);swap_orders.append(oid)
 total=sum(len(v) for v in out.values()); assert total==1043,(total,len(add_orders),len(swap_orders))
 stem='v140_hybrid_add8_swap2_from_pair015'; OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'{stem}.csv'
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 meta={'version':'v140-hybrid-v139-add-v136-swap','base':str(BASE),'base_sha256':sha(BASE),'addition_source':str(ADD),'swap_source':str(SWAP),'addition_orders':len(add_orders),'swap_orders':len(swap_orders),'predictions':total,'orders':len(order),'sha256':sha(path),'conditional_note':'8 additions plus 2 disjoint swaps; model/OOF only','status':'exploratory_not_scored','warning':'No online score yet; preserve the verified champion.'}
 (OUT/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
