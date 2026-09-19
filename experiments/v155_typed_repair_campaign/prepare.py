"""Freeze p03 and export losslessly deduplicated, nested cross-fit actions."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys, json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.with_name('v153_complementary_repair_campaign')
sys.path.insert(0, str(PRIOR))
from bridge import c, t, protect_snapshot, verify_snapshot
from training import Trainer
import campaign as previous

def write(p, v):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(p)

def main():
    window_path = HERE/'training_window.json'
    if not window_path.exists():
        now = datetime.now(timezone.utc)
        write(window_path, dict(started_at=now.isoformat(), deadline_at=(now+timedelta(hours=24)).isoformat(),
                               authorization='User approved V155 24-hour plan, six remaining submissions, no uploads',
                               competition_submitted=False))
    window = c.read(window_path)
    hours = (datetime.fromisoformat(window['deadline_at'])-datetime.now(timezone.utc)).total_seconds()/3600
    c.require(hours > 0, 'Original V155 deadline expired')
    snapshot = protect_snapshot()
    for p in PRIOR.glob('*'):
        if p.is_file(): snapshot[str(p)] = c.sha(p)
    config, cat, manifest, ledger, byid, eq, leaves, champion = previous.context(PRIOR)
    c.require((champion['predictions'],champion['tp'],champion['score'])==(1049,971,'0.927855'), 'Champion changed; replan')
    c.require(config['budget']['total']-len(ledger['records'])==6, 'Budget changed; review')
    write(HERE/'protected_sources.json', snapshot)
    write(HERE/'campaign_config.json', dict(baseline=champion,target=.937855,G=1044,P0=1049,TP0=971,
           remaining_at_start=6,daily_limit=2,prior_ledger=str(PRIOR/'online_scores.json'),
           prior_ledger_sha256=c.sha(PRIOR/'online_scores.json'),max_probes=4,window=window))
    # Only an isolated imported module is rebound; frozen historical code is untouched.
    t.P0, t.TP0 = 1049, 971
    trainer = Trainer(HERE/'export_cache', hours)
    trainer.deadline = datetime.fromisoformat(window['deadline_at']).timestamp()
    out = HERE/'bundle'; out.mkdir(exist_ok=False)
    metas, static, lookup, checks = [], [], {}, []
    def save(name, split, orders, scores, selected=None):
        b = trainer.actions(split, orders, scores, selected)
        ids=[]
        for i,a in enumerate(b['meta']):
            key=(split,a['order_id'],a['remove_rid'],a['add_rid'])
            if key not in lookup:
                lookup[key]=len(metas); metas.append(a); static.append(b['x'][i,:-11].copy())
            j=lookup[key]
            c.require(metas[j]==a and np.array_equal(static[j], b['x'][i,:-11]), 'Lossless feature mismatch')
            ids.append(j)
        context=b['x'][:,-11:]
        c.require(np.array_equal(np.column_stack([np.asarray(static)[ids],context]),b['x']), 'Feature reconstruction failed')
        np.savez_compressed(out/(name+'.npz'),row_ids=np.asarray(ids,dtype=np.int32),context=context,
                            y=b['y'],prior=b['prior'],mask=b['mask'])
        write(out/(name+'.json'),{'orders':b['orders']})
        checks.append({'name':name,'actions':len(ids),'orders':len(orders)})
    for outer in range(5):
        fit=np.flatnonzero(trainer.folds!=outer); valid=np.flatnonzero(trainer.folds==outer)
        for s,(f,v) in enumerate(list(t.group_splits(fit,trainer.folds))+[(fit,valid)]):
            c.require(not set(trainer.folds[f]) & set(trainer.folds[v]), 'Group leakage')
            scores,_=trainer.crossfit_nodes(f,v)
            save(f'f{outer}_s{s}_fit','train',f,scores)
            save(f'f{outer}_s{s}_valid','train',v,scores)
        print('EXPORTED_FOLD',outer+1,flush=True)
    orders=np.arange(len(trainer.records['train']))
    scores,test_scores=trainer.crossfit_nodes(orders)
    save('full_fit','train',orders,scores)
    _,_,nodes=c.load_csv(Path(champion['file']))
    selected=np.array([(o['order_id'],a['rid']) in nodes for o in trainer.records['test'] for a in o['alarms']])
    c.require(selected.sum()==1049, 'Test baseline P mismatch')
    save('full_test','test',np.arange(546),test_scores,selected)
    np.savez_compressed(out/'shared_static.npz',x=np.asarray(static,dtype=np.float32))
    write(out/'shared_meta.json',metas)
    np.savez_compressed(out/'truth.npz',labels=trainer.y,ptr=trainer.arrays['train_alarm_ptr'],folds=trainer.folds)
    write(out/'manifest.json',dict(files={p.name:c.sha(p) for p in out.iterdir() if p.is_file()},
          window=window,seed=20260906,target=.937855,P0=1049,TP0=971,G=1044,source_hashes=trainer.hashes,
          baseline=champion,checks=checks,lossless=True,source_class='nested_group_proxy_not_public_score'))
    verify_snapshot(snapshot)
    print('PREPARE_COMPLETE',json.dumps(checks[-2:]),flush=True)

if __name__=='__main__': main()
