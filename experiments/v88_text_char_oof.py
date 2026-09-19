from __future__ import annotations
import gzip,json,re,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; V30=ROOT/'experiments/v30_meta_stack'; DATA=ROOT/'experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz'; RECORDS=ROOT/'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'; OUT=ROOT/'experiments/v88_text_char'; OUT.mkdir(exist_ok=True); sys.path.insert(0,str(V30)); from v30_meta_stack import exact_count_mask
def norm(x):
 x=str(x or '').lower(); x=re.sub(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}','<uuid>',x); return re.sub(r'\d+','<num>',x)
def texts(records):
 out=[]
 for o in records:
  order=' order '+norm(o.get('target_summary',''))+' station '+' '.join(norm(x) for x in o.get('station_ids',[]))
  for a in o['alarms']:
   out.append(' title '+norm(a.get('title'))+' reason '+norm(a.get('reason'))+' location '+norm(a.get('location'))+' device '+norm(a.get('device'))+' vendor '+norm(a.get('vendor'))+' dtype '+norm(a.get('device_type'))+' board '+norm(a.get('board_type'))+' cause '+norm(a.get('cause'))+' radio '+norm(a.get('radio'))+' dep '+norm(a.get('deployment'))+' timeline '+norm(a.get('timeline'))+order)
 return out
def select(s,ptr,b):
 o=np.zeros(len(s),bool); opts=[]
 for a,z in zip(ptr[:-1],ptr[1:]):
  a,z=int(a),int(z); r=np.argsort(-s[a:z],kind='stable'); o[a+r[0]]=1; opts.extend((a+r[1:]).tolist())
 q=np.asarray(opts); q=q[np.argsort(-s[q],kind='stable')]; o[q[:b-int(o.sum())]]=1; return o
def main():
 with np.load(DATA) as z: arr={k:z[k] for k in z.files}
 d=json.load(gzip.open(RECORDS,'rt',encoding='utf-8')); tr,te=d['train'],d['test']; y=arr['train_labels'].astype(bool); order_folds=arr['train_folds']; ptr=arr['train_alarm_ptr']; folds=np.repeat(order_folds,np.diff(ptr)); base=np.load(V30/'v30_consensus_oof.npy'); testbase=np.load(V30/'v30_consensus_test.npy')
 alltxt=texts(tr+te); n=len(y); vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=160000,sublinear_tf=True,dtype=np.float32); X=vec.fit_transform(alltxt); Xtr,Xte=X[:n],X[n:]; report={'features':X.shape[1],'models':[]}
 for C in (.03,.1,.3,1.,3.):
  oof=np.zeros(n,np.float32); ts=[]
  for f in range(5):
   fit=folds!=f; val=~fit; m=LogisticRegression(C=C,class_weight='balanced',max_iter=120,n_jobs=-1,solver='liblinear'); m.fit(Xtr[fit],y[fit]); oof[val]=m.predict_proba(Xtr[val])[:,1]; ts.append(m.predict_proba(Xte)[:,1])
  tt=np.mean(ts,axis=0); np.save(OUT/f'oof_C{C}.npy',oof); np.save(OUT/f'test_C{C}.npy',tt); best=None
  for w in (0,.05,.1,.2,.3,.4,.5,.7,1.):
   sc=(1-w)*base+w*oof; mask=select(sc,ptr,3169); tp=int((mask&y).sum()); f1=2*tp/(3169+int(y.sum()));
   if best is None or f1>best['f1']: best={'w':w,'f1':f1,'tp':tp}
  report['models'].append({'C':C,'best':best})
 print(json.dumps(report,ensure_ascii=False,indent=2)); (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
