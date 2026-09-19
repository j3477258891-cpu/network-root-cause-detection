from __future__ import annotations
import gzip, json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V30 = ROOT / 'experiments/v30_meta_stack'
DATA = ROOT / 'experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz'
RECORDS = ROOT / 'experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz'
OUT = ROOT / 'experiments/v83_alarm_key_audit'
sys.path.insert(0, str(V30))
from v30_meta_stack import exact_count_mask

def norm(x):
    x = str(x or '').lower()
    x = re.sub(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', '<uuid>', x)
    return re.sub(r'\d+', '<num>', x)

def row_keys(a):
    title, reason, loc = norm(a.get('title')), norm(a.get('reason')), norm(a.get('location'))
    device, vendor = norm(a.get('device')), norm(a.get('vendor'))
    dtype, board, cause = norm(a.get('device_type')), norm(a.get('board_type')), norm(a.get('cause'))
    radio, dep, timeline = norm(a.get('radio')), norm(a.get('deployment')), norm(a.get('timeline'))
    return {
        'title': (title,), 'reason': (reason,), 'loc': (loc,), 'device': (device,),
        'title_loc': (title,loc), 'title_reason': (title,reason),
        'device_type_cause': (dtype,cause), 'title_timeline': (title,timeline),
        'board_cause': (board,cause), 'title_device': (title,device),
        'title_loc_cause': (title,loc,cause), 'title_reason_loc': (title,reason,loc),
        'target_title': (norm(a.get('target_summary')), title),
    }

def make_rows(records):
    out=[]; ptr=[0]
    for o in records:
        for a in o['alarms']: out.append(row_keys(a))
        ptr.append(len(out))
    return out, np.asarray(ptr,dtype=np.int64)

def encode(train_rows, train_y, query_rows, keys, alpha, prior):
    stats=defaultdict(lambda:[0,0])
    for i,k in enumerate(train_rows):
        key=tuple(k[n] for n in keys)
        stats[key][0]+=int(train_y[i]); stats[key][1]+=1
    out=np.empty(len(query_rows),dtype=np.float32)
    for i,k in enumerate(query_rows):
        pos,n=stats.get(tuple(k[n] for n in keys),(0,0))
        out[i]=(pos+alpha*prior)/(n+alpha)
    return out

def select(scores, ptr, budget):
    out=np.zeros(len(scores),dtype=bool); opts=[]
    for s,t in zip(ptr[:-1],ptr[1:]):
        s,t=int(s),int(t); order=np.argsort(-scores[s:t],kind='stable')
        out[s+order[0]]=1; opts.extend((s+order[1:]).tolist())
    opts=np.asarray(opts,dtype=np.int64); opts=opts[np.argsort(-scores[opts],kind='stable')]
    out[opts[:budget-int(out.sum())]]=1
    return out

def f1(mask,y):
    tp=int((mask&y).sum()); return 2*tp/(int(y.sum())+int(mask.sum())),tp,int(mask.sum())

def main():
    OUT.mkdir(exist_ok=True)
    with np.load(DATA) as z: arr={k:z[k] for k in z.files}
    d=json.load(gzip.open(RECORDS,'rt',encoding='utf-8')); tr,te=d['train'],d['test']
    trrows,ptr=make_rows(tr); terows,tptr=make_rows(te); y=arr['train_labels'].astype(bool)
    folds=arr['train_folds']; base_oof=np.load(V30/'v30_consensus_oof.npy'); base_test=np.load(V30/'v30_consensus_test.npy')
    prior=float(y.mean()); results=[]; test_scores={}
    keysets=[((n,),n) for n in ['title','reason','loc','device']]
    keysets += [((a,b),f'{a}_{b}') for a,b in [('title','loc'),('title','reason'),('device_type_cause','x')]]
    # Composite keys are already materialized under these names in row_keys.
    keysets = [((n,), n) for n in ['title','reason','loc','device','title_loc','title_reason','device_type_cause','title_timeline','board_cause','title_device','title_loc_cause','title_reason_loc','target_title']]
    for names,name in keysets:
        for alpha in (0.5,1.,2.,5.,10.,20.,50.):
            oof=np.zeros(len(y),np.float32)
            for fold in range(5):
                fit=np.flatnonzero(folds!=fold); val=np.flatnonzero(folds==fold)
                oof[val]=encode([trrows[i] for i in fit],y[fit],[trrows[i] for i in val],names,alpha,prior)
            test_scores[f'{name}_a{alpha:g}']=encode(trrows,y,terows,names,alpha,prior)
            for w in (0.,.02,.05,.1,.2,.35,.5,.7,1.):
                sc=(1-w)*base_oof+w*oof
                m=select(sc,ptr,3169); score,tp,p=f1(m,y)
                results.append({'name':name,'alpha':alpha,'w':w,'f1':score,'tp':tp,'p':p})
    results.sort(key=lambda z:z['f1'],reverse=True)
    best=results[:30]
    candidates=[]
    for r in best[:15]:
        sname=f"{r['name']}_a{r['alpha']:g}"; ts=test_scores[sname]
        for w in (r['w'],):
            candidates.append({'name':sname,'w':w,'score':r['f1'],'test':((1-w)*base_test+w*ts)})
    for c in candidates:
        for budget in (1035,1044,1060):
            m=select(c['test'],tptr,budget); c[f'pred_{budget}']=int(m.sum())
    report={'version':'v83-alarm-key-audit-1','base_oof':f1(select(base_oof,ptr,3169),y),'best':best,'test_candidates':[{k:v for k,v in c.items() if k!='test'} for c in candidates]}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'base':report['base_oof'],'best':best[:15],'test':report['test_candidates']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
