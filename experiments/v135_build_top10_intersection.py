"""Build additions appearing in the top-10 of all three V30 models."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"; DATA=EXP/"v25_semantic_router/cloud_dataset/v25_semantic_router.npz"; RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"; OUT=EXP/"v135_intersection_additions"
MODELS=['station_extra_trees','template_extra_trees','v30_consensus']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadbase():
 ids=[]; d={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f): ids.append(r['order_id']); d[r['order_id']]=json.loads(r['output'])['rootcause']
 return ids,d
def blocked():
 b={('4079fac3-5a5c-48e1-b171-d09393a78fc9','#-1:def8e14e-d628-48d0-bec0-a1cdc239c1b3'),('faeeeb88-468e-485d-b71b-d3f799baaaf0','#-1:9602714c-0fcb-43e5-b630-e6036059b9c5'),('70b84b0d-e9d5-4b40-b8dd-aef24ac338b3','#-1:c63b9fec-3656-43fc-adb8-e114e98247d4'),('c27a096b-576b-4fb7-9fb2-7f04feab066d','#-1:bdb85be4-661a-4a6a-9f56-fbff53833e80'),('808e05f4-3151-4437-9845-4994bc189ed0','#-1:4cf9e7ef-7cf0-4d22-b79a-77787a7ceb02')}
 return b
def main():
 k=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 7; outdir=Path(__import__('sys').argv[2]) if len(__import__('sys').argv)>2 else OUT; outdir.mkdir(parents=True,exist_ok=True)
 order,base=loadbase()
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f: rec=json.load(f)['test']
 with np.load(DATA,allow_pickle=False) as z: ptr=z['test_alarm_ptr']
 arrays={n:np.load(EXP/'v30_meta_stack'/f'{n}_test.npy') for n in MODELS}; ban=blocked(); by_oid={o['order_id']:o for o in rec}
 # Rank one eligible addition per order for each model, then intersect model top-10 sets.
 per={}
 for n in MODELS:
  rows=[]
  for i,o in enumerate(rec):
   oid=o['order_id']; cur={x['@rid'] for x in base[oid]}
   if len(cur)>=8: continue
   cand=[(float(arrays[n][int(ptr[i])+j]),a['rid']) for j,a in enumerate(o['alarms']) if a['rid'] not in cur and (oid,a['rid']) not in ban]
   if cand: rows.append(max(cand))
  rows.sort(reverse=True); per[n]=rows[:10]
 sets=[{rid for _,rid in per[n]} for n in MODELS]; common=set.intersection(*sets)
 # Average normalized model score gives a deterministic order within the intersection.
 def avg(rid):
  return float(np.mean([next(v for v,x in per[n] if x==rid) for n in MODELS]))
 common=sorted(common,key=lambda rid:(avg(rid),rid),reverse=True)
 proposals=[]
 for rid in common:
  owners=[o['order_id'] for o in rec if any(a['rid']==rid for a in o['alarms']) and (o['order_id'],rid) not in ban and rid not in {x['@rid'] for x in base[o['order_id']]}]
  if not owners: continue
  oid=owners[0]; proposals.append({'order_id':oid,'add_rid':rid,'mean_score':avg(rid),'model_scores':{n:next(v for v,x in per[n] if x==rid) for n in MODELS}})
 chosen=proposals[:k]; out={oid:list(base[oid]) for oid in order}
 for x in chosen:
  a=next(a for a in by_oid[x['order_id']]['alarms'] if a['rid']==x['add_rid']); s=a.get('source',{}); out[x['order_id']].append({'@rid':a['rid'],'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')})
 total=sum(len(v) for v in out.values()); assert total==1035+len(chosen)
 stem=f'v135_intersection_add_top{k}_from_pair015'; path=outdir/f'{stem}.csv'
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 meta={'version':'v135-top10-model-intersection','base':str(BASE),'base_sha256':sha(BASE),'models':MODELS,'topn_each':10,'blocked_known_false':len(ban),'available_intersection':len(proposals),'slice_size':k,'selected':chosen,'predictions':total,'orders':len(order),'sha256':sha(path),'status':'exploratory_not_scored','warning':'Intersection is model-derived; no online score yet.'}
 (outdir/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
