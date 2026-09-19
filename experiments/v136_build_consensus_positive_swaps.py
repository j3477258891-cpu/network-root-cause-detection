"""Build one-for-one swaps whose margin is positive in all three V30 models."""
from __future__ import annotations
import csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"; DATA=EXP/"v25_semantic_router/cloud_dataset/v25_semantic_router.npz"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v136_consensus_swaps"
MODELS=['station_extra_trees','template_extra_trees','v30_consensus']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadbase():
 ids=[];d={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):ids.append(r['order_id']);d[r['order_id']]=json.loads(r['output'])['rootcause']
 return ids,d
def protected():
 inn=set(); out=set()
 p=EXP/'v37_online_equations/report.json'
 if p.exists():
  d=json.loads(p.read_text(encoding='utf-8')); inn|={(x['order_id'],x['rid']) for x in d.get('fixed_labels',[]) if x.get('label')==1}; out|={(x['order_id'],x['rid']) for x in d.get('fixed_labels',[]) if x.get('label')==0}
 inn.add(('4079fac3-5a5c-48e1-b171-d09393a78fc9','#-1:efbf6784-8142-4c84-beac-82ece429ad61'))
 return inn,out
def main():
 k=int(sys.argv[1]) if len(sys.argv)>1 else 5; threshold=float(sys.argv[2]) if len(sys.argv)>2 else .1; outdir=Path(sys.argv[3]) if len(sys.argv)>3 else OUT;outdir.mkdir(parents=True,exist_ok=True)
 order,base=loadbase();
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f: rec=json.load(f)['test']
 with np.load(DATA,allow_pickle=False) as z:ptr=z['test_alarm_ptr']
 arrays={n:np.load(EXP/'v30_meta_stack'/f'{n}_test.npy') for n in MODELS}; pin,pout=protected(); proposals=[]
 for i,o in enumerate(rec):
  oid=o['order_id']; cur={x['@rid'] for x in base[oid]}; alarms=o['alarms']; index={a['rid']:j for j,a in enumerate(alarms)}; sel=[a for a in alarms if a['rid'] in cur and (oid,a['rid']) not in pin]; uns=[a for a in alarms if a['rid'] not in cur and (oid,a['rid']) not in pout]
  best=None
  for add in uns:
   for rem in sel:
    margins=np.array([arrays[n][int(ptr[i])+index[add['rid']]]-arrays[n][int(ptr[i])+index[rem['rid']]] for n in MODELS])
    if margins.min() < threshold:continue
    key=(float(margins.min()),float(margins.mean()),add['rid'],rem['rid'])
    if best is None or key>best[0]:best=(key,add,rem,margins)
  if best:
   key,add,rem,margins=best; proposals.append({'order_id':oid,'remove_rid':rem['rid'],'add_rid':add['rid'],'min_margin':key[0],'mean_margin':key[1],'model_margins':dict(zip(MODELS,map(float,margins)))})
 proposals.sort(key=lambda x:(x['min_margin'],x['mean_margin'],x['order_id']),reverse=True); chosen=proposals[:k];out={oid:list(base[oid]) for oid in order};rmap={o['order_id']:o for o in rec}
 for x in chosen:
  oid=x['order_id'];out[oid]=[n for n in out[oid] if n['@rid']!=x['remove_rid']];a=next(a for a in rmap[oid]['alarms'] if a['rid']==x['add_rid']);s=a.get('source',{});out[oid].append({'@rid':a['rid'],'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')})
 total=sum(len(v) for v in out.values());assert total==1035
 stem=f'v136_consensus_positive_swap_top{k}_t{str(threshold).replace(".","p")}_from_pair015';path=outdir/f'{stem}.csv'
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 meta={'version':'v136-three-model-positive-swap','base':str(BASE),'base_sha256':sha(BASE),'models':MODELS,'minimum_margin':threshold,'available_orders':len(proposals),'slice_size':len(chosen),'selected':chosen,'predictions':total,'orders':len(order),'sha256':sha(path),'status':'exploratory_not_scored','warning':'All-model margins are model-derived; no online score yet.'}
 (outdir/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
