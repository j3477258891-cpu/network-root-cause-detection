"""Portable paired narrow/wide actions using the original group-crossfit cache."""
from pathlib import Path
import sys, inspect, types, textwrap, json, zipfile, hashlib, time
from datetime import datetime
import numpy as np
HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'v153_complementary_repair_campaign'
sys.path.insert(0,str(FROZEN))
from training import Trainer
from bridge import c,t,protect_snapshot,verify_snapshot

def save(dest,b):
    np.savez_compressed(dest.with_suffix('.npz'),**{k:b[k] for k in ('x','y','prior','mask')})
    c.write(dest.with_suffix('.json'),{k:b[k] for k in ('meta','orders')})

def key(a): return a['order_id'],a['remove_rid'],a['add_rid']

def main():
    protected=protect_snapshot()
    ledger=c.sha(FROZEN/'online_scores.json')
    window=c.read(HERE/'training_window.json')
    hours=(datetime.fromisoformat(window['deadline_at']).timestamp()-time.time())/3600
    assert hours>0
    out=HERE/'bundle'; out.mkdir(exist_ok=False)
    trainer=Trainer(HERE/'export_cache',hours)
    source=textwrap.dedent(inspect.getsource(t.Trainer.actions))
    assert source.count('[:2]')==2, 'Action generator changed; review before mechanical expansion'
    wide_source=source.replace('[:2]','[:4]')
    namespace=dict(t.__dict__)
    exec(compile(wide_source,'verified_wide_actions','exec'),namespace)
    wide=types.MethodType(namespace['actions'],trainer)
    np.savez_compressed(out/'truth.npz',labels=trainer.y,ptr=trainer.arrays['train_alarm_ptr'],folds=trainer.folds)
    counts=[]
    def pair(prefix,split,orders,scores,selected=None):
        n=trainer.actions(split,orders,scores,selected)
        w=wide(split,orders,scores,selected)
        assert np.array_equal(n['mask'],w['mask']) and n['orders']==w['orders']
        lookup={key(a):i for i,a in enumerate(w['meta'])}
        assert len(lookup)==len(w['meta'])
        for i,a in enumerate(n['meta']):
            j=lookup[key(a)]
            assert np.array_equal(n['x'][i],w['x'][j]) and n['y'][i]==w['y'][j]
        save(out/('narrow_'+prefix),n); save(out/('wide_'+prefix),w)
        counts.append({'name':prefix,'narrow':len(n['meta']),'wide':len(w['meta'])})
    for outer in range(5):
        fit=np.flatnonzero(trainer.folds!=outer); valid=np.flatnonzero(trainer.folds==outer)
        for s,(f,v) in enumerate(list(t.group_splits(fit,trainer.folds))+[(fit,valid)]):
            assert not set(trainer.folds[f]) & set(trainer.folds[v])
            scores,_=trainer.crossfit_nodes(f,v)
            pair(f'f{outer}_s{s}_fit','train',f,scores)
            pair(f'f{outer}_s{s}_valid','train',v,scores)
        print('EXPORTED_OUTER',outer,flush=True)
    orders=np.arange(len(trainer.records['train']))
    scores,test_scores=trainer.crossfit_nodes(orders)
    pair('full_fit','train',orders,scores)
    _,_,_,nodes=c.baseline()
    selected=np.array([(o['order_id'],a['rid']) in nodes for o in trainer.records['test'] for a in o['alarms']])
    pair('full_test','test',np.arange(546),test_scores,selected)
    functions=('rows_for','probabilities','physical','selected_action_indices','score_actions','temperature','pooled')
    (out/'protocol.py').write_text('import numpy as np\nG=1044\nP0=1045\nTP0=969\n\n'+'\n\n'.join(inspect.getsource(getattr(t,f)) for f in functions),encoding='utf-8')
    (out/'wide_actions_source.py').write_text(wide_source,encoding='utf-8')
    (out/'worker_base.py').write_bytes((HERE.parent/'v153_cloud_supplement/worker.py').read_bytes())
    for name in ('worker.py','launch.py','training_window.json','PLAN.md'):
        (out/name).write_bytes((HERE/name).read_bytes())
    c.write(out/'generator_checks.json',{'counts':counts,'same_baseline_masks':True,'narrow_is_exact_subset':True,
        'shared_actions_features_and_labels_identical':True,'connected_group_isolation':True})
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    c.write(out/'manifest.json',{'files':files,'input_hashes':trainer.hashes,'window':window,'seed':20260906,
        'node_model_cache':'original V152/V153 cross-fitted predictions',
        'control':'narrow and wide action models retrained on same cloud runtime',
        'warning':'Proxy OOF is not public score. No test labels or secrets included.'})
    archive=HERE/'v154_wide_action_training.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p,'v154_wide_action/'+p.name)
    verify_snapshot(protected); assert c.sha(FROZEN/'online_scores.json')==ledger
    summary={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':c.sha(archive),
        'counts':counts[-2:],'protected_files_unchanged':len(protected)}
    c.write(HERE/'export_report.json',summary); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
