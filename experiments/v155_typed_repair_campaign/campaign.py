"""V155 fail-closed audit/build/recommend/record/emit-final. Never uploads."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import core as c
import worker
from threadpoolctl import threadpool_limits
from bridge import old152
from policy_adapter import Policy,Feasibility,WARNING

HERE=Path(__file__).resolve().parent
def key(a): return a['order_id'],a.get('remove_rid') or '',a.get('add_rid') or ''

def audit():
    config=c.read(HERE/'campaign_config.json')
    c.baseline()
    for p,h in c.read(HERE/'protected_sources.json').items():
        c.require(Path(p).is_file() and c.sha(p)==h,'Protected historical file changed: '+p)
    for name,h in c.read(HERE/'bundle/manifest.json')['files'].items():
        c.require(c.sha(HERE/'bundle'/name)==h,'Frozen training data changed: '+name)
    c.require(c.sha(config['prior_ledger'])==config['prior_ledger_sha256'],'Shared quota ledger changed')
    return {'integrity_passed':True,'actual_champion':config['baseline'],'remaining_at_start':6,
            'target':c.TARGET,'window':config['window'],'competition_submitted':False}

def train(directory):
    audit()
    c.require(not (directory/'progress.json').exists(),'Training output already exists: inspect active process and checkpoints; do not launch a duplicate. Completed training uses build.')
    with threadpool_limits(limits=8): worker.run(HERE/'bundle',directory)
    return {'training_report':str(directory/'progress.json'),'competition_submitted':False}

def comparison(directory):
    import numpy as np
    report=c.read(directory/'progress.json')
    c.require(report.get('complete') and not report.get('smoke_only') and report['status']=='completed','Full training incomplete')
    c.require(len(report['folds'])==5 and {f['fold'] for f in report['folds']}==set(range(5)),'Five folds required')
    signature=hashlib.sha256((c.sha(HERE/'bundle/manifest.json')+c.sha(HERE/'worker.py')+json.dumps(report['versions'],sort_keys=True)+'False').encode()).hexdigest()
    c.require(signature==report['signature'],'Training signature changed')
    window=c.read(HERE/'training_window.json')
    c.require(datetime.fromisoformat(report['finished_at'])<=datetime.fromisoformat(window['deadline_at']),'Training finished after original deadline')
    meta=c.read(HERE/'bundle/shared_meta.json')
    with np.load(HERE/'bundle/truth.npz',allow_pickle=False) as truth: labels,ptr=truth['labels'],truth['ptr']
    all_valid_orders=[]
    for f in report['folds']:
        name=f"f{f['fold']}_s2_valid"
        batch=c.read(HERE/'bundle'/(name+'.json'))
        all_valid_orders.extend(batch['orders'])
        with np.load(HERE/'bundle'/(name+'.npz'),allow_pickle=False) as data:
            batch.update(y=data['y'],mask=data['mask'],meta=[meta[int(i)] for i in data['row_ids']])
        for family in worker.FAMILIES:
            cp=c.read(directory/f"fold{f['fold']}_{family}.json")
            c.require(cp['signature']==signature and cp['model']==f['models'][family],'Fold checkpoint mismatch')
            with np.load(directory/f"fold{f['fold']}_{family}.npz",allow_pickle=False) as data: probabilities=data['probabilities']
            c.require(probabilities.shape==(len(batch['meta']),3) and np.all(np.isfinite(probabilities))
                      and np.min(probabilities)>=0 and np.allclose(probabilities.sum(1),1),'Invalid held-out probabilities')
            c.require(worker.metrics(batch,probabilities,labels,ptr)==cp['model']['metrics'],'Held-out prediction/metric mismatch')
    c.require(sorted(all_valid_orders)==list(range(len(ptr)-1)),'Validation orders are missing or repeated')
    summary={f:{b:worker.pooled(report['folds'],f,b) for b in ('16','32','48')} for f in worker.FAMILIES}
    main,eligible=worker.gate(summary)
    c.require(summary==report['summary'] and eligible==report['eligible_families'],'Recomputed gate mismatch')
    return report,summary,main,eligible

def select(legal):
    ranked=sorted(legal,key=lambda a:(-c.expected_u(a),key(a)))
    # One common deterministic order winner, before per-type quotas.
    unique=[];seen=set()
    for a in ranked:
        if a['order_id'] not in seen: unique.append(a);seen.add(a['order_id'])
    chosen=[]
    for kind in worker.KINDS: chosen.extend([a for a in unique if a['kind']==kind][:16])
    chosen_keys={key(a) for a in chosen}
    chosen.extend(a for a in unique if key(a) not in chosen_keys)
    chosen=sorted(chosen[:48],key=lambda a:(-c.expected_u(a),key(a)))
    for i,a in enumerate(chosen,1): a['candidate_id']=i
    groups=[]
    for kind in worker.KINDS:
        ids=[a['candidate_id'] for a in chosen if a['kind']==kind]
        for offset in range(0,len(ids),8): groups.append({'group_id':f'G{len(groups)+1:02d}','kind':kind,'ids':ids[offset:offset+8]})
    return chosen,groups

def build(directory):
    audit()
    c.require(not (HERE/'campaign.json').exists(),'Campaign already frozen; use recommend')
    if not (directory/'progress.json').exists() or not c.read(directory/'progress.json').get('complete'):
        result={'decision':'await_full_training_no_csv','allow_submission':False,'training_complete':False,
                'actual_champion':c.read(HERE/'campaign_config.json')['baseline'],'competition_submitted':False}
        c.write(HERE/'final_gate.json',result);return result
    report,summary,main,eligible=comparison(directory)
    gate={'training_complete':True,'eligible_families':eligible,'main_control':main,'summary32':{f:m['32'] for f,m in summary.items()},
          'allow_submission':False,'actual_champion':c.read(HERE/'campaign_config.json')['baseline'],'competition_submitted':False}
    gate['criteria_by_family']={f:{'positive_total_u':summary[f]['32']['u']>0,
            'beats_both_controls':summary[f]['32']['u']>max(summary[x]['32']['u'] for x in worker.FAMILIES[:2]),
            'at_least_three_positive_u_folds':summary[f]['32']['positive_folds']>=3,
            'worst_fold_not_worse_than_main_control':summary[f]['32']['worst_fold_u']>=summary[main]['32']['worst_fold_u']}
        for f in worker.FAMILIES[2:]}
    gate['evidence_class']='nested_group_validation_target_utility_not_online_gain'
    if not eligible:
        gate.update(decision='stop_training_gate_failed_no_csv');c.write(HERE/'final_gate.json',gate);return gate
    predictions={f:c.read(directory/(f+'_predictions.json')) for f in eligible}
    for f,p in predictions.items(): c.require(p['signature']==report['signature'],'Prediction signature mismatch')
    first=predictions[eligible[0]]['meta']
    c.require(all(p['meta']==first for p in predictions.values()),'Model action sets differ')
    _,seq,roots,nodes=c.baseline();system=c.collect_system(old152.records_data()['test'],seq)
    eq=c.Equations(system);c.require(eq.feasible(),'Historical equations conflict')
    excluded=old152.action_exclusions(nodes,system)
    allkeys={(a['order_id'],a[f]) for a in first for f in ('add_rid','remove_rid') if a.get(f)}
    bounds=c.legacy.EquationModel(system).classify(allkeys)
    feasible=Feasibility(eq,dict(enumerate(first)))
    # Protect all additions in the accepted p03 set, even if only a group count is known.
    _,_,v149=c.load_csv(c.EXP/'v149_online_calibrated_addition_campaign/v149_final_k08_from_v148.csv')
    protected=set(nodes)-set(v149)
    legal=[];rejected=[]
    for i,a0 in enumerate(first):
        a=dict(a0);c.validate_actions([a],roots,nodes);reason=None;oid=a['order_id']
        if oid==c.legacy.BLOCKED_ORDER: reason='permanent_order_exclusion'
        support={'add':(0,1),'delete':(-1,0),'swap':(-1,0,1)}[a['kind']]
        mp={f:{str(dt):predictions[f]['probabilities'][i][dt+1] for dt in support} for f in eligible}
        a['model_probabilities']=mp;a['probabilities']={str(dt):sum(p[str(dt)] for p in mp.values())/len(mp) for dt in support}
        c.distribution([a])
        for field in ('add_rid','remove_rid'):
            if not a.get(field): continue
            k=(oid,a[field]);reason=old152.exclusion_reason(field,excluded.get(k,[])) or reason
            if field=='remove_rid' and (k in protected or bounds[k]['min']==1): reason='protected_positive_or_p03'
            if field=='add_rid' and bounds[k]['max']==0: reason='equation_false_addition'
        if reason: rejected.append({'action':key(a),'reason':reason});continue
        outcomes=[dt for dt in support if feasible([([i],dt)])]
        c.require(outcomes,'No feasible action outcome')
        a['delta_tp_bounds']={'min':min(outcomes),'max':max(outcomes)}
        if c.f1(max(outcomes),a['delta_p'])<=c.f1():
            rejected.append({'action':key(a),'reason':'integer_proven_no_gain'});continue
        legal.append(a)
    candidates,groups=select(legal)
    c.validate_actions(candidates,roots,nodes)
    oracle=eq.oracle(candidates)
    catalog={'candidates':candidates,'groups':groups,'families':eligible,'rejected':rejected,'legal_actions':len(legal),'warning':WARNING}
    c.write(HERE/'candidate_catalog.json',catalog);c.write(HERE/'equation_system.json',system)
    gate.update(candidate_count=len(candidates),oracle=oracle)
    if oracle['f1_upper']<c.TARGET:
        gate.update(decision='stop_target_upper_gate_failed_no_csv');c.write(HERE/'final_gate.json',gate);return gate
    config=c.read(HERE/'campaign_config.json')
    config.update(version=155,baseline=config['baseline'],training_directory=str(directory),training_report_sha256=c.sha(directory/'progress.json'),
                  budget={'total':6,'daily':2,'max_probes':4},frozen_files={n:c.sha(HERE/n) for n in ('candidate_catalog.json','equation_system.json')},initial_oracle=oracle)
    c.write(HERE/'campaign.json',config);c.write(HERE/'submission_manifest.json',{'files':[]});c.write(HERE/'online_scores.json',{'records':[]})
    gate.update(decision='gates_passed_recommend_probe');c.write(HERE/'final_gate.json',gate)
    return gate

def context():
    audit();config,cat,system,manifest,ledger=c.frozen_load(HERE)
    c.require(c.sha(Path(config['training_directory'])/'progress.json')==config['training_report_sha256'],'Frozen model report changed')
    for e in system['equations']:
        c.require(Path(e['file']).is_file() and c.sha(e['file'])==e['sha256'],'Historical scored CSV changed')
    byid={a['candidate_id']:a for a in cat['candidates']};entries={e['probe_id']:e for e in manifest['files']}
    leaves=[];obs=[];champ=dict(config['baseline']);done=set()
    for r in ledger['records']:
        if r['status']!='accepted': continue
        e=entries[r['probe_id']];a=[byid[i] for i in e['ids']]
        checked=c.validate_csv(HERE/e['file'],a,e['sha256']);tp=c.infer_tp(r['score'],checked['predictions'])
        c.require(tp==r['tp'],'Ledger TP drift');_,_,nodes=c.load_csv(HERE/e['file'])
        index={(v['order_id'],v['rid']):i for i,v in enumerate(system['variables'])}
        obs.append({'indices':sorted(index[k] for k in nodes),'tp':tp})
        ids=tuple(sorted(e['ids']))
        if ids not in done and e['kind'] in ('group','split'): leaves=c.apply_observation(leaves,e,tp-c.TP0)
        done.add(ids)
        if 2*tp/(c.G+checked['predictions'])>2*champ['tp']/(c.G+champ['predictions']):
            champ={'file':str(HERE/e['file']),'tp':tp,'predictions':checked['predictions'],'score':r['score'],'sha256':e['sha256'],'source_class':'public_scored'}
    eq=c.Equations(system,obs);c.require(eq.feasible(),'Score equations conflict')
    return config,cat,manifest,ledger,byid,eq,leaves,champ

def recommend():
    if not (HERE/'campaign.json').exists():
        return c.read(HERE/'final_gate.json') if (HERE/'final_gate.json').exists() else {'decision':'await_training','allow_submission':False}
    config,cat,manifest,ledger,byid,eq,leaves,champ=context();used=len(ledger['records'])
    best=c.best_union(leaves,byid);actual=2*champ['tp']/(c.G+champ['predictions'])
    result={'actual_champion':champ,'best_known_union':best,'remaining_submissions':6-used,'allow_submission':False,'warning':WARNING,'platform_quota_live_checked':False}
    if any(r['status']=='anomaly' for r in ledger['records']): return dict(result,decision='pause_anomaly')
    if actual>=c.TARGET or used>=6: return dict(result,decision='stop_target_met_or_budget_exhausted')
    pending=[e for e in manifest['files'] if not any(r['probe_id']==e['probe_id'] for r in ledger['records'])]
    if pending: return dict(result,decision='await_feedback',pending=pending[0])
    oracle=eq.oracle(list(byid.values()));result['oracle']=oracle
    if best['f1']>=c.TARGET or used>=4:
        if best['f1']>actual:
            return dict(result,decision='emit_merge',next_action={'kind':'merge','ids':best['ids'],'expected_tp':c.TP0+best['delta_tp'],'expected_f1':best['f1']})
        return dict(result,decision='retain_champion_reserve_unused',risk_requires_explicit_review=True)
    if oracle['f1_upper']<=actual: return dict(result,decision='stop_no_gain')
    policy=Policy(byid,cat['groups'],cat['families'],Feasibility(eq,byid))
    entries={e['probe_id']:e for e in manifest['files']}
    attempted={tuple(sorted(entries[r['probe_id']]['ids'])) for r in ledger['records']}
    query=policy.recommend(leaves,attempted,actual,4-used)
    if query is None and best['f1']>actual: query={'kind':'merge','ids':best['ids'],'expected_tp':c.TP0+best['delta_tp'],'expected_f1':best['f1']}
    return dict(result,decision='prepare_'+query['kind'] if query else 'stop_no_value',next_action=query)

def emit_spec(final=False):
    result=recommend();query=result.get('next_action');c.require(query is not None,'No eligible new CSV')
    if final: c.require(query['kind']=='merge','No count-supported final merge is ready')
    config,cat,manifest,ledger,byid,eq,leaves,champ=context()
    pid=f"p{len(manifest['files'])+1:02d}";filename=f"v155_{'final' if query['kind']=='merge' else 'probe'}_{pid}_{query['kind']}.csv"
    spec={'base_file':str(c.BASE),'base_sha256':c.BASE_SHA,'output_file':str(HERE/filename),'preview_file':str(HERE/'previews'/f'{pid}.png'),
          'actions':[byid[i] for i in query['ids']],'query':query,'probe_id':pid}
    c.write(HERE/'authoring_specs'/f'{pid}.json',spec)
    return {'spec':str(HERE/'authoring_specs'/f'{pid}.json'),'csv_not_yet_authored':True,'recommendation':result}

def register(specpath):
    spec=c.read(specpath);config,cat,manifest,ledger,byid,eq,leaves,champ=context();q=spec['query']
    current=recommend().get('next_action');c.require(current and current['kind']==q['kind'] and current['ids']==q['ids'],'Stale authoring specification')
    q=current  # Never trust edited cached expected scores in an authoring spec.
    filename=Path(spec['output_file']);c.require(filename.parent.resolve()==HERE.resolve(),'Output outside campaign')
    checked=c.validate_csv(filename,[byid[i] for i in q['ids']])
    entry={**q,**checked,'probe_id':spec['probe_id'],'file':filename.name,'created_at':c.now()}
    manifest['files'].append(entry);c.write(HERE/'submission_manifest.json',manifest)
    return entry

def record(pid,score,stamp,evidence,failed=False):
    config,cat,manifest,ledger,byid,eq,leaves,champ=context();c.check_budget(ledger['records'],stamp)
    c.require(evidence.strip(),'Actual score/failure evidence required');e=next(x for x in manifest['files'] if x['probe_id']==pid)
    c.require(not any(r['probe_id']==pid and r['status']=='accepted' for r in ledger['records']),'Already recorded; do not charge quota twice')
    item={'probe_id':pid,'score':score,'submitted_at':stamp,'evidence':evidence,'status':'failed' if failed else 'accepted'}
    if not failed:
        try:
            checked=c.validate_csv(HERE/e['file'],[byid[i] for i in e['ids']],e['sha256']);tp=c.infer_tp(score,checked['predictions'])
            c.require(eq.feasible(eq.coeff([byid[i] for i in e['ids']]),tp-c.TP0),'Score conflicts with equations')
            c.require(tp==e.get('expected_tp',tp),'Exact merge score mismatch');item.update(tp=tp,delta_tp=tp-c.TP0)
        except ValueError as exc: item.update(status='anomaly',error=str(exc))
    ledger['records'].append(item);c.write(HERE/'online_scores.json',ledger)
    result=recommend();c.write(HERE/'recommendation.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','train','build','recommend','emit-spec','emit-final','register','record'])
    p.add_argument('--training',type=Path,default=HERE/'training_local');p.add_argument('--spec',type=Path);p.add_argument('--probe-id');p.add_argument('--score');p.add_argument('--submitted-at');p.add_argument('--evidence');p.add_argument('--failed',action='store_true');a=p.parse_args()
    if a.command=='audit': result=audit()
    elif a.command=='train': result=train(a.training)
    elif a.command=='build': result=build(a.training)
    elif a.command=='recommend': result=recommend()
    elif a.command in ('emit-spec','emit-final'): result=emit_spec(a.command=='emit-final')
    elif a.command=='register': result=register(a.spec)
    else: result=record(a.probe_id,a.score,a.submitted_at,a.evidence,a.failed)
    print(json.dumps(result,ensure_ascii=False,indent=2))
