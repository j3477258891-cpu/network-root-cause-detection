"""Reuse honest cached predictions; fixed ranker, paired bootstrap, causal count replay.

No new fitted model or weight is selected using outer-fold outcomes. Existing
CatBoost/V38 inner-selected parameters remain unchanged. The new policy averages
their outcome distributions, ranks the complete action list by target utility,
and removes outcomes ruled out by earlier public equations before test ranking.
"""
from __future__ import annotations
import gzip, types
import numpy as np
from common import *
from evidence import Equations

FAMILIES=('catboost','v38_control')
SEED=20260912

def select(meta,p,orders,budget=48):
    p=np.asarray(p);utility=2*(p[:,2]-p[:,0])-TARGET*np.array([a['delta_p'] for a in meta])
    selected=[];used=set();cap=round(budget*len(orders)/546)
    for i in sorted(range(len(meta)),key=lambda i:(-utility[i],key(meta[i]))):
        if len(selected)>=cap:break
        if utility[i]<=0 or meta[i]['order_id'] in used:continue
        used.add(meta[i]['order_id']);selected.append(i)
    return selected

def statistics(bundle,p,labels,ptr):
    selected=select(bundle['meta'],p,bundle['orders'])
    chosen={bundle['meta'][i]['order_index']:i for i in selected};stats=[]
    for oi in bundle['orders']:
        rows=np.arange(ptr[oi],ptr[oi+1]);mask=bundle['mask'][rows];i=chosen.get(oi)
        stats.append([oi,int(labels[rows].sum()),int(mask.sum()),int(labels[rows][mask].sum()),
            bundle['meta'][i]['delta_p'] if i is not None else 0,int(bundle['y'][i]) if i is not None else 0])
    return np.array(stats,dtype=np.int64)

def gain(stats):
    g,p,tp,dp,dt=np.asarray(stats)[:,1:].sum(axis=0)
    return float(2*(tp+dt)/(g+p+dp)-2*tp/(g+p))

def station_clusters(records):
    parent=list(range(len(records)));owner={}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for i,r in enumerate(records):
        for station in r['station_ids']:
            if station in owner:parent[find(i)]=find(owner[station])
            else:owner[station]=i
    return [find(i) for i in range(len(records))]

def bootstrap(candidate,controls,clusters,iterations=2000):
    require(all(np.array_equal(candidate[:,0],b[:,0]) for b in controls.values()),'Unpaired bootstrap orders')
    groupids=sorted(set(clusters[int(i)] for i in candidate[:,0]));index={k:i for i,k in enumerate(groupids)}
    arrays={name:np.zeros((len(groupids),5),dtype=np.int64) for name in ('candidate',*controls)}
    for name,stats in [('candidate',candidate),*controls.items()]:
        for r in stats:arrays[name][index[clusters[int(r[0])]]]+=r[1:]
    rng=np.random.default_rng(SEED);values={k:[] for k in arrays}
    for _ in range(iterations):
        sample=rng.integers(len(groupids),size=len(groupids))
        for name,a in arrays.items():
            g,p,tp,dp,dt=a[sample].sum(axis=0)
            values[name].append(float(2*(tp+dt)/(g+p+dp)-2*tp/(g+p)))
    absolute=np.array(values['candidate'])
    return dict(iterations=iterations,station_clusters=len(groupids),absolute_lower95=float(np.quantile(absolute,.025)),
        relative_lower95={name:float(np.quantile(absolute-np.array(values[name]),.025)) for name in controls})

def outcome_filter(p,lo,hi):
    p=np.asarray(p,dtype=float).copy();p[(np.arange(3)-1<lo)|(np.arange(3)-1>hi)]=0
    require(p.sum()>0,'No model mass on feasible outcomes');return p/p.sum()

def convolve_counts(ps):
    d={0:1.}
    for p in ps:
        n={}
        for s,w in d.items():
            for v,q in zip((-1,0,1),p):n[s+v]=n.get(s+v,0)+w*float(q)
        d=n
    return d

def historical_replay(c):
    """Only first pool appearances; forecast with strictly preceding equations.

    This is a retrospective diagnostic with overlapping constraints, not a set
    of independent trials or a calibrated probability of hitting the target.
    """
    raw=read(EXP/'v153_complementary_repair_campaign/training_candidates.json')['families']
    maps={f:{key(a):a for a in raw[f]} for f in FAMILIES}
    base=EXP/'v149_online_calibrated_addition_campaign/v149_final_k08_from_v148.csv'
    _,base_roots,base_nodes=csv_nodes(base)
    names={'v150_probe_01_all80.csv','v153_probe_p01_group.csv','v153_probe_p02_group.csv','v153_probe_p03_group.csv'}
    report=[];prior=[]
    for e in c.system['equations']:
        c.check_deadline()
        if Path(e['file']).name in names:
            _,roots,nodes=csv_nodes(e['file']);actions=[];reason=None
            for oid in roots:
                removed={n['@rid'] for n in base_roots[oid]}-{n['@rid'] for n in roots[oid]}
                added={n['@rid'] for n in roots[oid]}-{n['@rid'] for n in base_roots[oid]}
                if not removed and not added:continue
                if max(len(removed),len(added))>1:reason='multi-node action not covered by frozen action models';break
                rem=next(iter(removed),None);add=next(iter(added),None)
                a={'order_id':oid,'remove_rid':rem,'add_rid':add,'delta_p':len(added)-len(removed)}
                if any(key(a) not in maps[f] for f in FAMILIES):reason='full pool not covered by frozen models; no partial-count scoring';break
                actions.append(a)
            row=dict(file=e['file'],actual_delta_tp=e['tp']-969,prior_equations=len(prior),independent_trial=False)
            if reason:row.update(evaluated=False,reason=reason)
            else:
                system={**c.system,'equations':list(prior)};eq=Equations(system)
                ps={f:[] for f in FAMILIES};ps['candidate']=[]
                for a in actions:
                    bound=eq.bound([a]);one=[]
                    for f in FAMILIES:
                        p=np.array([maps[f][key(a)]['probabilities'].get(str(k),0.) for k in (-1,0,1)])
                        ps[f].append(p);one.append(p)
                    ps['candidate'].append(outcome_filter(np.mean(one,axis=0),*bound))
                predictions={}
                for f,p in ps.items():
                    dist=convolve_counts(p);mean=sum(v*w for v,w in dist.items());actual=e['tp']-969
                    predictions[f]={'mean_delta_tp':mean,'absolute_count_error':abs(actual-mean),
                        'count_nll':float(-np.log(max(dist.get(actual,0.),1e-300)))}
                row.update(evaluated=True,actions=len(actions),predictions=predictions,
                    warning='Marginal outcome filtering retains dependencies only as hard constraints; count likelihood is a working independence diagnostic')
            report.append(row)
        prior.append(e)
    evaluated=[r for r in report if r['evaluated']]
    return dict(source_class='retrospective_first_pool_count_replay',rows=report,
        full_pool_evaluations=len(evaluated),fitted_new_parameters=False,
        future_equations_used=False,split_rows_counted_as_independent=False,
        mean_absolute_count_error={f:float(np.mean([r['predictions'][f]['absolute_count_error'] for r in evaluated])) for f in (*FAMILIES,'candidate')} if evaluated else {})

def run(c):
    out=c.out/'research';out.mkdir(exist_ok=True)
    result_path=out/'results.json'
    if result_path.exists():
        saved=read(result_path)
        if saved.get('complete'):
            require(saved['protocol_sha256']==digest(c.cfg['protocol']),'Completed protocol drift')
            require(sha(result_path)==read(out/'result_seal.json')['sha256'],'Completed research result drift')
            return saved
    report=dict(complete=False,passed=False,protocol_sha256=digest(c.cfg['protocol']),quota_cost=0,
        deadline_at=c.cfg['deadline_at'],source_class='fixed_ranker_nested_group_oof_plus_separate_count_replay',
        caveat='OOF proxy at V149 prediction density is not the corrected public champion; count replay is separate evidence',
        folds=[],new_fitted_parameters=False)
    try:
        c.check_deadline();t=training_module();run=EXP/'v152_error_repair_campaign/training'
        trainer=t.Trainer(out/'cache',48,False)
        trainer.deadline=datetime.fromisoformat(c.cfg['deadline_at']).timestamp()
        def cached_nodes(self,fit_orders):
            self.check_time();name=t.digest([self.signature,sorted(map(int,fit_orders))])+'.npz'
            p=run/'node_cache'/name
            require(p.exists(),'Missing frozen node predictions; training is not silently restarted')
            require(sha(p)==c.cfg['source_hashes'][str(p)],'Node cache changed')
            with np.load(p,allow_pickle=False) as z:return z['train'],z['test']
        trainer.node_predictions=types.MethodType(cached_nodes,trainer)
        oldreport=read(EXP/'v152_error_repair_campaign/model_comparison.json')
        require(oldreport['complete'] and oldreport['input_hashes']==trainer.hashes,'Incomplete or incompatible original five-fold lineage')
        allstats={f:[] for f in (*FAMILIES,'candidate')}
        for outer in range(5):
            c.check_deadline();saved=read(run/f'fold_{outer}.json')
            require(saved['signature']==trainer.signature,'Cached OOF signature drift')
            fit=np.array(saved['fit_orders']);valid=np.array(saved['valid_orders'])
            require(set(fit).isdisjoint(valid) and set(trainer.folds[fit]).isdisjoint(trainer.folds[valid]),'Outer-fold leakage')
            scores,_=trainer.crossfit_nodes(fit,valid)
            b=trainer.actions('train',valid,scores)
            require(b['meta']==read(run/f'fold_{outer}_actions.json'),'Reconstructed action metadata differs from cached predictions')
            probs={}
            for family in FAMILIES:
                with np.load(run/f'fold_{outer}_{family}.npz') as z:
                    probs[family]=z['probabilities'];require(np.array_equal(z['labels'],b['y']),'OOF label alignment drift')
                require(probs[family].shape==(len(b['meta']),3),'Prediction shape drift')
            probs['candidate']=np.mean([probs[f] for f in FAMILIES],axis=0)
            stats={f:statistics(b,probs[f],trainer.y,trainer.arrays['train_alarm_ptr']) for f in probs}
            for f,x in stats.items():allstats[f].append(x)
            fold={'fold':outer,'valid_orders':list(map(int,valid)),'gains':{f:gain(x) for f,x in stats.items()},
                  'stats':{f:x.tolist() for f,x in stats.items()}}
            report['folds'].append(fold);write(out/'progress.json',report)
            print('V158 cached OOF fold',outer+1,fold['gains'],flush=True)
        joined={f:np.concatenate(v) for f,v in allstats.items()}
        for f in joined:joined[f]=joined[f][np.argsort(joined[f][:,0])]
        require(np.array_equal(joined['candidate'][:,0],np.arange(len(trainer.records['train']))),'Five folds do not cover every order exactly once')
        clusters=station_clusters(trainer.records['train'])
        boot=bootstrap(joined['candidate'],{f:joined[f] for f in FAMILIES},clusters)
        pooled={f:gain(x) for f,x in joined.items()}
        criteria=dict(complete_five_folds=True,at_least_three_positive_folds=sum(f['gains']['candidate']>0 for f in report['folds'])>=3,
            pooled_beats_both_controls=pooled['candidate']>max(pooled[f] for f in FAMILIES),
            positive_absolute_lower95=boot['absolute_lower95']>0,
            positive_relative_lower95_both=all(x>0 for x in boot['relative_lower95'].values()))
        report.update(pooled_gains=pooled,bootstrap=boot,criteria=criteria)
        report['historical_replay']=historical_replay(c)
        report['preflight']=module('_v158_policy',HERE/'policy.py').preflight(c)
        # Replay is mandatory but is not promoted to independent validation.
        criteria['at_least_one_complete_historical_pool_replayed']=report['historical_replay']['full_pool_evaluations']>0
        criteria['provisional_integer_upper_at_least_target']=report['preflight']['oracle']['f1_upper']>=TARGET
        c.check_deadline();report.update(complete=True,passed=all(criteria.values()),completed_at=now(),
            decision='eligible_for_pool_integer_gate' if all(criteria.values()) else 'stop_research_gate_failed_keep_remaining_attempts')
        verify_sources(c.cfg['source_hashes'])
    except Exception as exc:
        report.update(passed=False,error=repr(exc),decision='stop_incomplete_or_deadline_no_new_csv',failed_at=now())
        write(result_path,report);raise
    write(result_path,report);write(out/'result_seal.json',{'sha256':sha(result_path),'completed_at':report['completed_at']})
    return report
