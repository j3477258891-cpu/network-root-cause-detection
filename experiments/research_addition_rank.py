"""Rank pure additions with descriptor transfer; never submit automatically."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import research_key_action_validation as R

ROOT = R.ROOT
OUT = ROOT / "experiments/research_addition_rank"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, a = R.load_data()
    trrows, trptr, troi = R.flat_orders(train)
    terows, teptr, teoi = R.flat_orders(test)
    y = a['train_labels'].astype(bool)
    prior = float(y.mean())
    folds = a['train_station_folds'].astype(int)
    base_train = R.exact_mask(a['train_v11'].astype(float), trptr, 3097)
    base_test = np.zeros(len(terows), dtype=bool)
    selected = R.load_submission(R.BASE)
    for i,(oi,alarm) in enumerate(zip(teoi,terows)):
        base_test[i] = alarm['rid'] in selected.get(test[int(oi)]['order_id'], set())
    kinds = ['full_no_location','title_reason_cause_timeline','title_cause_timeline','title_reason','title_label','reason_label']
    poofs={}; ptests={}; supports={}
    for kind in kinds:
        poof=np.zeros(len(trrows));
        for fold in range(5):
            fit=np.flatnonzero(folds!=fold); val_orders=np.flatnonzero(folds==fold)
            stats=R.stats_for(train,fit,kind,10.0,prior)
            val=np.flatnonzero(np.isin(troi,val_orders))
            poof[val]=R.predict(train,[trrows[int(i)] for i in val],troi[val],stats,kind,10.0,prior)
        stats=R.stats_for(train,np.arange(len(train)),kind,10.0,prior)
        ptest=R.predict(test,terows,teoi,stats,kind,10.0,prior)
        poofs[kind]=poof; ptests[kind]=ptest
        supports[kind]=np.asarray([stats.get(R.key(alarm,kind),(0,0))[1] for alarm in terows])
    # Evaluate fixed-count analogue additions with no test leakage. Include a
    # higher-quality V30 analogue to expose selection optimism.
    curves=[]
    for base_name, base_scores, target in [('v11',a['train_v11'].astype(float),3097),('v30',np.load(R.V30/'v30_consensus_oof.npy').astype(float),3169)]:
        bm=R.exact_mask(base_scores,trptr,target)
        for kind in kinds:
            for k in (5,10,20,30,40,50,60,80,100):
                m=R.add_ranked(bm,poofs[kind],troi,trptr,k)
                q=R.metrics(m,y); curves.append({'base_name':base_name,'kind':kind,'k':k,'base':R.metrics(bm,y),'net_gain':q['tp']-int((bm & y).sum()),'metrics':q})
    # Rank by mean posterior, while requiring at least two supporting keys.
    stack=np.column_stack([ptests[k] for k in kinds])
    support=(np.column_stack([supports[k] for k in kinds])>=20).sum(axis=1)
    # Robust score: mean of the two lowest available key posteriors. This
    # discourages a single memorized descriptor from dominating.
    sorted_stack=np.sort(stack,axis=1)
    robust=np.mean(sorted_stack[:,-2:],axis=1)
    mean=np.mean(stack,axis=1)
    rankings={}
    for name,score in [('mean',mean),('robust2',robust),('min',sorted_stack[:,0]),('reason_label',ptests['reason_label']),('title_reason',ptests['title_reason'])]:
        idx=np.flatnonzero(~base_test & (support>=2))
        idx=idx[np.argsort(-score[idx],kind='stable')]
        # Add at most 8 roots per order, and avoid fixed false labels.
        chosen=[]; counts=defaultdict(int)
        fixed={}
        if R.FIXED.exists():
            fixed={(x['order_id'],x['rid']):x.get('label') for x in json.loads(R.FIXED.read_text(encoding='utf8')).get('labels',[])}
        for i in idx:
            oid=test[int(teoi[i])]['order_id']
            if counts[oid] >= 8: continue
            if fixed.get((oid,terows[i]['rid'])) == 0: continue
            counts[oid]+=1
            chosen.append(int(i))
            if len(chosen)>=100: break
        rankings[name]=[{
            'rank':j+1,'order_id':test[int(teoi[i])]['order_id'],'rid':terows[i]['rid'],
            'score':float(score[i]),'mean':float(mean[i]),'robust2':float(robust[i]),
            'key_scores':{k:float(ptests[k][i]) for k in kinds},
            'support':{k:int(supports[k][i]) for k in kinds},
        } for j,i in enumerate(chosen)]
    report={'version':1,'base_train':R.metrics(base_train,y),'curves':sorted(curves,key=lambda x:(-x['net_gain'],x['k'])),'rankings':rankings,'test_base_p':int(base_test.sum()),'notes':['OOF curves are station-fold analogues, not leaderboard evidence.','Test rankings are unsubmitted candidates.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'base_train':report['base_train'],'top_curves':report['curves'][:20],'ranking_counts':{k:len(v) for k,v in rankings.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
