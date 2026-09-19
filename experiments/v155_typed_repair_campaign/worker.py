"""Portable V155 paired nested training. No CSV/leaderboard access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse, json, hashlib, time, signal, platform
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import importlib.metadata
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

TARGET=.937855
KINDS=('add','delete','swap')
FAMILIES=('control_catboost','control_v38','typed_catboost','typed_v38')
HERE=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def physical(p,meta,temp=1.):
    q=np.maximum(p,1e-9)**(1/temp)
    for i,a in enumerate(meta):
        if a['kind']=='add': q[i,0]=0
        if a['kind']=='delete': q[i,2]=0
    return q/q.sum(axis=1,keepdims=True) if len(q) else q
def subset(b,ids):
    return {**b,'x':b['x'][ids],'y':b['y'][ids], 'meta':[b['meta'][i] for i in ids]}
def weights(meta):
    counts=Counter((a['order_id'],a['kind']) for a in meta)
    return np.array([1/counts[a['order_id'],a['kind']] for a in meta])
def probabilities(m,x):
    out=np.zeros((len(x),3))
    predicted=m.predict_proba(x)
    for j,c in enumerate(m.classes_): out[:,int(c)+1]=predicted[:,j]
    return out
def fit_predict(family,param,fit,valid,seed,smoke=False):
    typed=family.startswith('typed_')
    if not len(valid['meta']): return np.empty((0,3)), []
    if len(np.unique(fit['y']))<2:
        # Smoothing only over physically allowed outcomes; no invented fitted model.
        p=np.zeros(3)
        for value in fit['y']: p[int(value)+1]+=1
        p+=1e-6
        return physical(np.tile(p,(len(valid['meta']),1)),valid['meta']), []
    if family.endswith('catboost'):
        from catboost import CatBoostClassifier
        loss='Logloss' if typed and fit['meta'][0]['kind']!='swap' else 'MultiClass'
        models=[CatBoostClassifier(iterations=5 if smoke else 240,depth=param,learning_rate=.05,
             l2_leaf_reg=5,loss_function=loss,thread_count=8,random_seed=seed,verbose=False,allow_writing_files=False)]
    else:
        models=[ExtraTreesClassifier(n_estimators=8 if smoke else 800,min_samples_leaf=4,max_features=.45,
                    class_weight=None if typed else 'balanced',n_jobs=8,random_state=seed),
                HistGradientBoostingClassifier(max_iter=5 if smoke else 260,learning_rate=.045,
                    max_leaf_nodes=15,min_samples_leaf=24,l2_regularization=1.5,early_stopping=False,random_state=seed)]
    ps=[]
    for model in models:
        model.fit(fit['x'],fit['y'],sample_weight=weights(fit['meta']) if typed else None)
        ps.append(probabilities(model,valid['x']))
    return physical(np.mean(ps,axis=0),valid['meta']),models
def choose(b,p,budget):
    gains=2*(p[:,2]-p[:,0])-TARGET*np.array([a['delta_p'] for a in b['meta']])
    ranked=sorted(range(len(gains)),key=lambda i:(-gains[i],b['meta'][i]['order_id'],b['meta'][i].get('remove_rid') or '',b['meta'][i].get('add_rid') or ''))
    result=[];used=set(); cap=max(1,round(budget*len(b['orders'])/546))
    for i in ranked:
        oid=b['meta'][i]['order_id']
        if oid in used or gains[i]<=0: continue
        used.add(oid);result.append(i)
        if len(result)==cap: break
    return result
def metrics(b,p,labels,ptr):
    rows=np.concatenate([np.arange(ptr[i],ptr[i+1]) for i in b['orders']])
    p0=int(b['mask'][rows].sum());tp0=int(labels[rows][b['mask'][rows]].sum());g=int(labels[rows].sum())
    result={}
    for budget in (16,32,48):
        ids=choose(b,p,budget);dt=int(b['y'][ids].sum());dp=sum(b['meta'][i]['delta_p'] for i in ids)
        kinds={}
        for kind in KINDS:
            yy=[int(b['y'][i]) for i in ids if b['meta'][i]['kind']==kind]
            kinds[kind]={'actions':len(yy),'delta_tp':sum(yy),'outcomes':dict(Counter(map(str,yy)))}
        result[str(budget)]=dict(orders=len(b['orders']),actions=len(ids),g=g,base_p=p0,base_tp=tp0,
             delta_p=dp,delta_tp=dt,u=2*dt-TARGET*dp,base_f1=2*tp0/(g+p0),f1=2*(tp0+dt)/(g+p0+dp),
             by_type=kinds,selected_action_indices=ids)
    return result
def temperature(p,y,meta):
    if not len(y): return 1.
    return min((float(-np.log(np.maximum(physical(p,meta,t)[np.arange(len(y)),y+1],1e-12)).mean()),t)
               for t in (.5,.75,1.,1.5,2.,3.))[1]
def pooled(folds,family,budget):
    items=[f['models'][family]['metrics'][budget] for f in folds]
    sums={k:sum(i[k] for i in items) for k in ('actions','orders','g','base_p','base_tp','delta_p','delta_tp','u')}
    sums.update(positive_folds=sum(i['u']>0 for i in items),worst_fold_u=min(i['u'] for i in items))
    sums['f1']=2*(sums['base_tp']+sums['delta_tp'])/(sums['g']+sums['base_p']+sums['delta_p'])
    sums['base_f1']=2*sums['base_tp']/(sums['g']+sums['base_p'])
    return sums
def gate(summary):
    controls=FAMILIES[:2]
    main=max(controls,key=lambda n:(summary[n]['32']['u'],summary[n]['32']['worst_fold_u'],n))
    allowed=[n for n in FAMILIES[2:] if summary[n]['32']['u']>max(0,*(summary[c]['32']['u'] for c in controls))
            and summary[n]['32']['positive_folds']>=3 and summary[n]['32']['worst_fold_u']>=summary[main]['32']['worst_fold_u']]
    return main,allowed
def run(bundle,out,smoke=False):
    manifest=read(bundle/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(bundle/name)!=digest: raise RuntimeError('Frozen input drift: '+name)
    deadline=datetime.fromisoformat(manifest['window']['deadline_at']).timestamp()
    def check():
        if time.time()>=deadline: raise TimeoutError('V155 original deadline expired')
    check()
    if hasattr(signal,'SIGALRM'):
        def expire(*_): raise TimeoutError('V155 original deadline expired')
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(deadline-time.time())))
    out.mkdir(exist_ok=True)
    meta=read(bundle/'shared_meta.json')
    with np.load(bundle/'shared_static.npz') as z: static=z['x']
    with np.load(bundle/'truth.npz') as z: labels,ptr=z['labels'],z['ptr']
    def load(name):
        b=read(bundle/(name+'.json'))
        with np.load(bundle/(name+'.npz')) as z: b.update({k:z[k] for k in z.files})
        ids=b.pop('row_ids');b['meta']=[meta[int(i)] for i in ids]
        b['x']=np.column_stack([static[ids],b.pop('context')]);return b
    versions={k:importlib.metadata.version(k) for k in ('numpy','scikit-learn','catboost')}
    signature=hashlib.sha256((sha(bundle/'manifest.json')+sha(Path(__file__))+json.dumps(versions,sort_keys=True)+str(smoke)).encode()).hexdigest()
    report=dict(status='running',complete=False,smoke_only=smoke,signature=signature,versions=versions,
                platform=platform.platform(),started_at=datetime.now(timezone.utc).isoformat(),folds=[],
                window=manifest['window'],target=TARGET,source_class='nested_group_proxy_not_public_score')
    write(out/'progress.json',report)
    for outer in range(1 if smoke else 5):
        fold={'fold':outer,'models':{}}
        for family in FAMILIES:
            check();cp=out/f'fold{outer}_{family}.json'
            if cp.exists():
                saved=read(cp)
                if saved['signature']!=signature: raise RuntimeError('Checkpoint signature drift')
                fold['models'][family]=saved['model'];continue
            print('TRAIN',outer+1,family,flush=True)
            typed=family.startswith('typed_');groups=KINDS if typed else ('all',)
            specs={};prediction=None
            fullfit,fullvalid=load(f'f{outer}_s2_fit'),load(f'f{outer}_s2_valid')
            prediction=np.zeros((len(fullvalid['meta']),3))
            for kind in groups:
                params=[4,6] if family.endswith('catboost') else [0]
                inner={p:[] for p in params}
                for s in range(2):
                    fb,vb=load(f'f{outer}_s{s}_fit'),load(f'f{outer}_s{s}_valid')
                    if typed:
                        fb=subset(fb,[i for i,a in enumerate(fb['meta']) if a['kind']==kind])
                        vb=subset(vb,[i for i,a in enumerate(vb['meta']) if a['kind']==kind])
                    for param in params:
                        check();p,_=fit_predict(family,param,fb,vb,manifest['seed'],smoke)
                        inner[param].append((p,vb,metrics(vb,p,labels,ptr)['32']['u']))
                param=max(params,key=lambda v:(np.mean([e[2] for e in inner[v]]),-v))
                es=inner[param]
                temp=temperature(np.concatenate([e[0] for e in es]),np.concatenate([e[1]['y'] for e in es]),[a for e in es for a in e[1]['meta']])
                ii=[i for i,a in enumerate(fullfit['meta']) if not typed or a['kind']==kind]
                jj=[i for i,a in enumerate(fullvalid['meta']) if not typed or a['kind']==kind]
                fb,vb=subset(fullfit,ii),subset(fullvalid,jj)
                check();p,_=fit_predict(family,param,fb,vb,manifest['seed'],smoke)
                prediction[jj]=physical(p,vb['meta'],temp); specs[kind]={'param':param,'temperature':temp}
            model={'specs':specs,'metrics':metrics(fullvalid,prediction,labels,ptr)}
            np.savez_compressed(out/f'fold{outer}_{family}.npz',probabilities=prediction)
            write(cp,{'signature':signature,'model':model});fold['models'][family]=model
            print('FOLD_RESULT',outer+1,family,json.dumps({b:m['u'] for b,m in model['metrics'].items()}),flush=True)
        report['folds'].append(fold);write(out/'progress.json',report)
    if smoke:
        report['status']='smoke_passed_not_eligible';write(out/'progress.json',report);return
    summary={f:{b:pooled(report['folds'],f,b) for b in ('16','32','48')} for f in FAMILIES}
    control,allowed=gate(summary)
    report.update(summary=summary,main_control=control,eligible_families=allowed)
    for family in allowed:
        check();fb,vb=load('full_fit'),load('full_test');p=np.zeros((len(vb['meta']),3));fitted={}
        for kind in KINDS:
            specs=[f['models'][family]['specs'][kind] for f in report['folds']]
            counts=Counter(s['param'] for s in specs);param=min(counts,key=lambda v:(-counts[v],v))
            temp=float(np.median([s['temperature'] for s in specs]))
            ii=[i for i,a in enumerate(fb['meta']) if a['kind']==kind];jj=[i for i,a in enumerate(vb['meta']) if a['kind']==kind]
            pp,models=fit_predict(family,param,subset(fb,ii),subset(vb,jj),manifest['seed'])
            p[jj]=physical(pp,subset(vb,jj)['meta'],temp);fitted[kind]=dict(models=models,param=param,temperature=temp)
        import joblib
        joblib.dump(fitted,out/(family+'.joblib'))
        write(out/(family+'_predictions.json'),{'meta':vb['meta'],'probabilities':p.tolist(),'signature':signature})
    check();report.update(status='completed',complete=True,finished_at=datetime.now(timezone.utc).isoformat())
    write(out/'progress.json',report);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=HERE/'bundle');parser.add_argument('--out',type=Path,default=HERE/'training');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    try:
        with threadpool_limits(limits=8): run(args.bundle,args.out,args.smoke)
    except Exception as exc:
        args.out.mkdir(exist_ok=True);write(args.out/'failure.json',{'error':str(exc),'complete':False,'at':datetime.now(timezone.utc).isoformat()});raise
