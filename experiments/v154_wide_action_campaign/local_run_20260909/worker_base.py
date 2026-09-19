"""Portable CPU training; original deadline; no leaderboard access or CSV output."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse
from pathlib import Path
import json
import hashlib
import time
import signal
import platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from protocol import probabilities, physical, score_actions, temperature, pooled

HERE=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(p)
def load(name):
    b=read(HERE/(name+'.json'))
    with np.load(HERE/(name+'.npz'),allow_pickle=False) as z: b.update({k:z[k] for k in z.files})
    return b
def fit_predict(family,param,fit,valid,seed,smoke=False):
    if family=='catboost':
        from catboost import CatBoostClassifier
        models=[CatBoostClassifier(iterations=5 if smoke else 240,depth=param,learning_rate=.05,
            l2_leaf_reg=5,loss_function='MultiClass',thread_count=8,random_seed=seed,
            verbose=False,allow_writing_files=False)]
    else:
        models=[ExtraTreesClassifier(n_estimators=8 if smoke else 800,min_samples_leaf=4,
            max_features=.45,class_weight='balanced',n_jobs=8,random_state=seed),
            HistGradientBoostingClassifier(max_iter=5 if smoke else 260,learning_rate=.045,
            max_leaf_nodes=15,min_samples_leaf=24,l2_regularization=1.5,early_stopping=False,random_state=seed)]
    ps=[]
    for m in models:
        m.fit(fit['x'],fit['y'])
        ps.append(probabilities(m,valid['x']))
    return physical(np.mean(ps,axis=0),valid['meta']),models
def run(args):
    manifest=read(HERE/'manifest.json')
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest, 'Input changed: '+name
    seconds=int(datetime.fromisoformat(manifest['deadline_at']).timestamp()-time.time())
    assert seconds>0,'Original deadline expired'
    def expired(*_): raise TimeoutError('Original training deadline reached; partial result cannot qualify')
    if hasattr(signal,'SIGALRM'):
        signal.signal(signal.SIGALRM,expired); signal.alarm(seconds)
    out=HERE/('smoke_output' if args.smoke else 'results')
    out.mkdir(exist_ok=True)
    with np.load(HERE/'truth.npz',allow_pickle=False) as z: labels,ptr=z['labels'],z['ptr']
    previous=read(HERE/'previous_comparison.json')
    report={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),
        'deadline_at':manifest['deadline_at'],'platform':platform.platform(),
        'seed':manifest['seed'],'source_class':'nested_group_oof_not_public_score',
        'smoke_only':args.smoke,'folds':[],
        'versions':{m:importlib.metadata.version(m) for m in ['numpy','scikit-learn','catboost']}}
    write(out/'progress.json',report)
    grid={'catboost':[4,6],'v38_control':[0]}
    for outer in range(1 if args.smoke else 5):
        fold={'fold':outer,'models':{}}
        for family,params in grid.items():
            print('TRAIN',outer,family,flush=True)
            inner={param:[] for param in params}
            for s in range(2):
                fb,vb=load(f'f{outer}_s{s}_fit'),load(f'f{outer}_s{s}_valid')
                for param in params:
                    p,_=fit_predict(family,param,fb,vb,manifest['seed'],args.smoke)
                    metric=score_actions(vb,p,labels,ptr)['64']
                    inner[param].append((p,vb,metric['f1_gain']))
            param=max(params,key=lambda v:(np.mean([e[2] for e in inner[v]]),-v))
            entries=inner[param]
            temp=temperature(np.concatenate([e[0] for e in entries]),np.concatenate([e[1]['y'] for e in entries]),[a for e in entries for a in e[1]['meta']])
            fb,vb=load(f'f{outer}_s2_fit'),load(f'f{outer}_s2_valid')
            p,_=fit_predict(family,param,fb,vb,manifest['seed'],args.smoke)
            p=physical(p,vb['meta'],temp)
            metrics=score_actions(vb,p,labels,ptr)
            fold['models'][family]={'param':param,'temperature':temp,'metrics':metrics}
            np.savez_compressed(out/f'fold{outer}_{family}.npz',probabilities=p)
            print('FOLD_RESULT',outer,family,json.dumps({b:m['f1_gain'] for b,m in metrics.items()}),flush=True)
        report['folds'].append(fold); write(out/'progress.json',report)
    if args.smoke:
        report['status']='smoke_passed_not_eligible'; write(out/'progress.json',report); return
    report['summary']={f:{b:pooled(report['folds'],f,b) for b in ('16','32','64')} for f in grid}
    eligible=[]
    for f in grid:
        m=report['summary'][f]['32']
        old=max(pooled(previous['folds'],other,'32')['f1'] for other in grid)
        if m['positive_folds']>=3 and m['f1']>m['base_f1'] and m['f1']>old: eligible.append(f)
    report.update(complete=True,eligible_families=eligible,gate='32-action pooled F1 above both original controls and >=3 positive folds',
        warning='Repeated seeds share validation data; not an unbiased global model-selection result or public improvement')
    write(out/'comparison.json',report)
    for f in eligible:
        fb,vb=load('full_fit'),load('full_test')
        ps=[]
        for param in sorted({fold['models'][f]['param'] for fold in report['folds']}):
            p,models=fit_predict(f,param,fb,vb,manifest['seed'])
            temp=float(np.median([fold['models'][f]['temperature'] for fold in report['folds'] if fold['models'][f]['param']==param]))
            ps.append(physical(p,vb['meta'],temp))
            import joblib
            joblib.dump(models,out/f'{f}_{param}.joblib')
        write(out/f'predictions_{f}.json',{'source_class':'model_estimated_not_equation_filtered',
            'meta':vb['meta'],'probabilities':np.mean(ps,axis=0).tolist()})
    report['status']='completed'; report['finished_at']=datetime.now(timezone.utc).isoformat()
    write(out/'progress.json',report)
    print('COMPLETE',json.dumps(report['summary']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    with threadpool_limits(limits=8): run(parser.parse_args())
