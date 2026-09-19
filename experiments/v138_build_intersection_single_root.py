"""Intersection of three model top-10 additions on single-root orders."""
from __future__ import annotations
import csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong");EXP=ROOT/"experiments";BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv";DATA=EXP/"v25_semantic_router/cloud_dataset/v25_semantic_router.npz";REC=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz";OUT=EXP/"v138_intersection_single_root"
MODELS=['station_extra_trees','template_extra_trees','v30_consensus']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadbase():
 ids=[];d={}
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):ids.append(r['order_id']);d[r['order_id']]=json.loads(r['output'])['rootcause']
 return ids,d
def ban():
 b={('4079fac3-5a5c-48e1-b171-d09393a78fc9','#-1:def8e14e-d628-48d0-bec0-a1cdc239c1b3'),('faeeeb88-468e-485d-b71b-d3f799baaaf0','#-1:9602714c-0fcb-43e5-b630-e6036059b9c5'),('70b84b0d-e9d5-4b40-b8dd-aef24ac338b3','#-1:c63b9fec-3656-43fc-adb8-e114e98247d4'),('c27a096b-576b-4fb7-9fb2-7f04feab066d','#-1:bdb85be4-661a-4a6a-9f56-fbff53833e80'),('808e05f4-3151-4437-9845-4994bc189ed0','#-1:4cf9e7ef-7cf0-4d22-b79a-77787a7ceb02')}
 p=EXP/'v37_online_equations/report.json'
 if p.exists():
  d=json.loads(p.read_text(encoding='utf-8'));b|={(x['order_id'],x['rid']) for x in d.get('fixed_labels',[]) if x.get('label')==0}
 return b
def main():
 k=int(sys.argv[1]) if len(sys.argv)>1 else 6;outdir=Path(sys.argv[2]) if len(sys.argv)>2 else OUT;outdir.mkdir(parents=True,exist_ok=True);order,base=loadbase()
 with gzip.open(REC,'rt',encoding='utf-8') as f:rec=json.load(f)['test']
 with np.load(DATA,allow_pickle=False) as z:ptr=z['test_alarm_ptr']
 arr={n:np.load(EXP/'v30_meta_stack'/f'{n}_test.npy') for n in MODELS};blocked=ban();per={}
 for n in MODELS:
  rows=[]
  for i,o in enumerate(rec):
   oid=o['order_id'];cur={x['@rid'] for x in base[oid]}
   if len(cur)!=1:continue
   c=[(float(arr[n][int(ptr[i])+j]),a['rid']) for j,a in enumerate(o['alarms']) if a['rid'] not in cur and (oid,a['rid']) not in blocked]
   if c:rows.append((max(c),oid))
  rows.sort(reverse=True);per[n]=rows[:10]
 common=set.intersection(*[{(oid,rid) for (sc,rid),oid in per[n]} for n in MODELS]);props=[]
 for oid,rid in common:
  vals={n:next(sc for (sc,x),o in per[n] if o==oid and x==rid) for n in MODELS};props.append({'order_id':oid,'add_rid':rid,'mean_score':float(np.mean(list(vals.values()))),'model_scores':vals})
 props.sort(key=lambda x:(x['mean_score'],x['order_id']),reverse=True);chosen=props[:k];out={oid:list(base[oid]) for oid in order};rmap={o['order_id']:o for o in rec}
 for x in chosen:
  a=next(a for a in rmap[x['order_id']]['alarms'] if a['rid']==x['add_rid']);s=a.get('source',{});out[x['order_id']].append({'@rid':a['rid'],'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')})
 total=sum(len(v) for v in out.values());assert total==1035+len(chosen)
 stem=f'v138_intersection_single_root_add_top{len(chosen)}_from_pair015';path=outdir/f'{stem}.csv'
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']);w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 meta={'version':'v138-intersection-single-root','base':str(BASE),'base_sha256':sha(BASE),'models':MODELS,'topn_each':10,'restriction':'base_root_count == 1','available_intersection':len(props),'selected':chosen,'predictions':total,'orders':len(order),'sha256':sha(path),'oof_reference':'three-model single-root top10 intersection: 4/4 true','status':'exploratory_not_scored','warning':'OOF/model evidence only; no online score yet.'}
 (outdir/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
