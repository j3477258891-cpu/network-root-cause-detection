"""Paired CPU action-generator experiment. No competition API or CSV creation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse,json,hashlib,time,platform,importlib.metadata,signal,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits
from worker_base import fit_predict
from protocol import physical,score_actions,temperature,pooled
HERE=Path(__file__).resolve().parent
_catalog = None
_static = None
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(p)
def load(name):
    global _catalog, _static
    b=read(HERE/(name+'.json'))
    with np.load(HERE/(name+'.npz'),allow_pickle=False) as z: b.update({k:z[k] for k in z.files})
    if 'row_ids' in b:
        if _catalog is None:
            _catalog=read(HERE/'shared_meta.json')
            with np.load(HERE/'shared_static.npz',allow_pickle=False) as z: _static=z['x']
        ids=b.pop('row_ids')
        b['x']=np.column_stack([_static[ids],b.pop('context')])
        b['meta']=[_catalog[int(i)] for i in ids]
    return b
def run(smoke=False):
    manifest=read(HERE/'manifest.json')
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,'Input drift: '+name
    seconds=int(datetime.fromisoformat(manifest['window']['deadline_at']).timestamp()-time.time())
    assert seconds>0,'Authorized window expired'
    if hasattr(signal,'SIGALRM'):
        def expired(*args): raise TimeoutError('Authorized deadline reached')
        signal.signal(signal.SIGALRM,expired); signal.alarm(seconds)
    out=HERE/('smoke_results' if smoke else 'results'); out.mkdir(exist_ok=True)
    versions={m:importlib.metadata.version(m) for m in ('numpy','scikit-learn','catboost')}
    report={'status':'running','complete':False,'smoke_only':smoke,'started_at':datetime.now(timezone.utc).isoformat(),
        'deadline_at':manifest['window']['deadline_at'],'versions':versions,'platform':platform.platform(),
        'source_class':'nested_group_oof_not_public_score','inner_selection_budget':32,'folds':[]}
    write(out/'progress.json',report)
    with np.load(HERE/'truth.npz',allow_pickle=False) as z: labels,ptr=z['labels'],z['ptr']
    specs=[(u,f) for u in ('narrow','wide') for f in ('catboost','v38_control')]
    for outer in range(1 if smoke else 5):
        fold={'fold':outer,'models':{}}
        for universe,family in specs:
            name=universe+'_'+family; checkpoint=out/f'fold{outer}_{name}.json'
            if checkpoint.exists():
                saved=read(checkpoint)
                assert saved['versions']==versions and saved['manifest_sha']==hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()
                fold['models'][name]=saved['model']; continue
            print('TRAIN',outer,name,flush=True)
            params=[4,6] if family=='catboost' else [0]
            inner={p:[] for p in params}
            for s in range(2):
                fb,vb=load(f'{universe}_f{outer}_s{s}_fit'),load(f'{universe}_f{outer}_s{s}_valid')
                for param in params:
                    p,_=fit_predict(family,param,fb,vb,manifest['seed'],smoke)
                    metric=score_actions(vb,p,labels,ptr)['32']
                    inner[param].append((p,vb,metric['f1_gain']))
            param=max(params,key=lambda v:(np.mean([e[2] for e in inner[v]]),-v))
            entries=inner[param]
            temp=temperature(np.concatenate([e[0] for e in entries]),np.concatenate([e[1]['y'] for e in entries]),[a for e in entries for a in e[1]['meta']])
            fb,vb=load(f'{universe}_f{outer}_s2_fit'),load(f'{universe}_f{outer}_s2_valid')
            p,_=fit_predict(family,param,fb,vb,manifest['seed'],smoke)
            p=physical(p,vb['meta'],temp); metrics=score_actions(vb,p,labels,ptr)
            model={'param':param,'temperature':temp,'metrics':metrics}
            fold['models'][name]=model
            write(checkpoint,{'model':model,'versions':versions,'manifest_sha':hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()})
            np.savez_compressed(out/f'fold{outer}_{name}.npz',probabilities=p)
            print('FOLD_RESULT',outer,name,json.dumps({b:m['f1_gain'] for b,m in metrics.items()}),flush=True)
        report['folds'].append(fold); write(out/'progress.json',report)
    if smoke:
        report.update(status='smoke_passed_not_eligible'); write(out/'progress.json',report); return
    names=[u+'_'+f for u,f in specs]
    report['summary']={n:{b:pooled(report['folds'],n,b) for b in ('16','32','64')} for n in names}
    best_narrow=max(report['summary'][n]['32']['f1'] for n in names if n.startswith('narrow_'))
    eligible=[n for n in names if n.startswith('wide_') and report['summary'][n]['32']['positive_folds']>=3
        and report['summary'][n]['32']['f1']>max(best_narrow,report['summary'][n]['32']['base_f1'])]
    report.update(complete=True,eligible_wide_sources=eligible,
        gate='Wide 32-action F1 above both same-runtime narrow controls; >=3 positive folds',
        warning='Shared validation set and proxy baseline; not an unbiased global optimum or a public gain')
    write(out/'comparison.json',report)
    for name in eligible:
        family=name.removeprefix('wide_'); fb,vb=load('wide_full_fit'),load('wide_full_test'); ps=[]
        for param in sorted({fold['models'][name]['param'] for fold in report['folds']}):
            p,models=fit_predict(family,param,fb,vb,manifest['seed'])
            temp=float(np.median([fold['models'][name]['temperature'] for fold in report['folds'] if fold['models'][name]['param']==param]))
            ps.append(physical(p,vb['meta'],temp))
            import joblib
            joblib.dump(models,out/f'{name}_{param}.joblib')
        write(out/f'predictions_{name}.json',{'meta':vb['meta'],'probabilities':np.mean(ps,axis=0).tolist(),
            'source_class':'model_estimated_requires_latest_equation_and_exclusion_screen'})
    report.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
    write(out/'progress.json',report); print('COMPLETE',json.dumps(report['summary']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    try:
        with threadpool_limits(limits=8): run(args.smoke)
    except Exception as exc:
        out=HERE/('smoke_results' if args.smoke else 'results'); out.mkdir(exist_ok=True)
        write(out/'failure.json',{'error':str(exc),'at':datetime.now(timezone.utc).isoformat(),'complete':False})
        traceback.print_exc(); raise
