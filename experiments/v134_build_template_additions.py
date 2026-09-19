"""Build pure-addition probes ranked by template ExtraTrees."""
from __future__ import annotations
import csv, gzip, hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
BASE=EXP/"v124_campaign/v124_single_pair_015_from_probe13.csv"
DATA=EXP/"v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS=EXP/"v25_semantic_router/cloud_dataset/semantic_records.json.gz"
SCORES=EXP/"v30_meta_stack/template_extra_trees_test.npy"
V37=EXP/"v37_online_equations/report.json"; OUT=EXP/"v134_template_additions"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def base_load():
 ids={}; order=[]
 with BASE.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f): order.append(r['order_id']); ids[r['order_id']]=json.loads(r['output'])['rootcause']
 return order,ids
def blocked():
 b=set()
 if V37.exists():
  d=json.loads(V37.read_text(encoding='utf-8')); b|={(x['order_id'],x['rid']) for x in d.get('fixed_labels',[]) if x.get('label')==0}
 b|={
 ('4079fac3-5a5c-48e1-b171-d09393a78fc9','#-1:def8e14e-d628-48d0-bec0-a1cdc239c1b3'),
 ('faeeeb88-468e-485d-b71b-d3f799baaaf0','#-1:9602714c-0fcb-43e5-b630-e6036059b9c5'),
 ('70b84b0d-e9d5-4b40-b8dd-aef24ac338b3','#-1:c63b9fec-3656-43fc-adb8-e114e98247d4'),
 ('c27a096b-576b-4fb7-9fb2-7f04feab066d','#-1:bdb85be4-661a-4a6a-9f56-fbff53833e80'),
 ('808e05f4-3151-4437-9845-4994bc189ed0','#-1:4cf9e7ef-7cf0-4d22-b79a-77787a7ceb02')}
 return b
def main():
 k=int(sys.argv[1]) if len(sys.argv)>1 else 10; outdir=Path(sys.argv[2]) if len(sys.argv)>2 else OUT; outdir.mkdir(parents=True,exist_ok=True)
 order,base=base_load()
 with gzip.open(RECORDS,'rt',encoding='utf-8') as f: rec=json.load(f)['test']
 with np.load(DATA,allow_pickle=False) as z: ptr=z['test_alarm_ptr']
 scores=np.load(SCORES); ban=blocked(); props=[]
 for i,o in enumerate(rec):
  oid=o['order_id']; cur={x['@rid'] for x in base[oid]}
  if len(cur)>=8: continue
  c=[(float(scores[int(ptr[i])+j]),a) for j,a in enumerate(o['alarms']) if a['rid'] not in cur and (oid,a['rid']) not in ban]
  if c:
   sc,a=max(c,key=lambda x:(x[0],x[1]['rid'])); props.append({'order_id':oid,'add_rid':a['rid'],'template_score':sc,'base_root_count':len(cur)})
 props.sort(key=lambda x:(x['template_score'],x['order_id']),reverse=True); chosen=props[:k]; out={oid:list(base[oid]) for oid in order}
 rm={x['order_id']:x for x in rec}
 for x in chosen:
  a=next(a for a in rm[x['order_id']]['alarms'] if a['rid']==x['add_rid']); s=a.get('source',{}); out[x['order_id']].append({'@rid':a['rid'],'title':s.get('title',''),'location':s.get('location',''),'reason':s.get('reason','')})
 total=sum(len(v) for v in out.values()); assert total==1035+k,(total,k)
 stem=f'v134_template_add_top{k}_from_pair015'; path=outdir/f'{stem}.csv'
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['order_id','output']); w.writeheader()
  for oid in order:w.writerow({'order_id':oid,'output':json.dumps({'rootcause':out[oid]},ensure_ascii=False)})
 meta={'version':'v134-template-pure-addition','base':str(BASE),'base_sha256':sha(BASE),'source_scores':str(SCORES),'slice_size':k,'available_additions':len(props),'selected':chosen,'predictions':total,'orders':len(order),'sha256':sha(path),'status':'exploratory_not_scored','warning':'Template ranking is model-derived; no online score yet.'}
 (outdir/f'{stem}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
