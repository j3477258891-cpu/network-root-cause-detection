"""Eight new attempts, two-stage score-gated campaign. No platform upload code."""
from __future__ import annotations
import argparse, copy, itertools
from common import *
from evidence import Equations,collect

def initialize(out):
    out=Path(out)
    if (out/'campaign.json').exists():return Campaign(out).status()
    require(out.resolve().parent==EXP.resolve(),'Campaign must be a direct experiments child')
    old=v157().Campaign();s=old.status()
    require(s['feasible_worlds']==4 and old.champ['tp']==972 and old.champ['predictions']==1048,'Starting evidence changed; re-audit')
    system,prior=collect();require(Equations(system).solve() is not None,'Inconsistent starting equations')
    snapshot={e['file']:e['sha256'] for e in system['equations']}
    # Preserve historical code, metadata and ledgers, plus all research inputs.
    for directory in (EXP/'v157_joint_correction_campaign',EXP/'v156_joint_decode_campaign',
                      EXP/'v153_complementary_repair_campaign'):
        snapshot.update({str(p):sha(p) for p in directory.iterdir() if p.is_file()})
    for p in (EXP/'v152_error_repair_campaign/training').glob('fold*'):
        if p.is_file():snapshot[str(p)]=sha(p)
    t=training_module();snapshot.update(t.source_hashes())
    for p in (EXP/'v152_error_repair_campaign/training/node_cache').glob('*.npz'):snapshot[str(p)]=sha(p)
    for p in (EXP/'v152_error_repair_campaign/model_comparison.json',EXP/'v153_complementary_repair_campaign/training_candidates.json',
              EXP/'v155_typed_repair_campaign/history_evidence_cache.json'):
        snapshot[str(p)]=sha(p)
    start=datetime.now(TZ)
    cfg=dict(version=158,created_at=start.isoformat(),total_attempts=8,daily_limit=2,target=.94,
        quota_source='User explicitly authorized eight remaining attempts in this task; no historical reset',
        baseline=old.champ,initial_world_ids=list(old.s),source_hashes=snapshot,
        historical_attempts=prior,deadline_at=(start+timedelta(hours=48)).isoformat(),
        protocol={'candidate':'fixed equal-weight CatBoost/V38 distribution; historical impossible outcomes removed before ranking',
            'fit_new_parameters':False,'budget_per_546':48,'max_group_size':12,'max_groups':4,'max_split_probes':1,
            'folds':5,'bootstrap_iterations':2000,'bootstrap_seed':20260912,'target':.94,
            'requires_both_controls':True,'stop_if_gate_fails':True},
        implementation_hashes={p.name:sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'historical_equations.json',system)
    cfg['system_sha256']=sha(out/'historical_equations.json')
    write(out/'online_scores.json',{'records':[]});write(out/'submission_manifest.json',{'files':[]})
    initial=next(x for x in old.manifest['files'] if x['probe_id']=='probe_adaptive_p03')
    actions=[old.e.cat[i] for i in initial['all_ids']]
    base=old.e.freeze['base']['file'];file=EXP/'v157_joint_correction_campaign'/initial['file']
    check=validate_file(file,base,actions,initial['sha256'])
    entry=dict(probe_id='safe_probe',phase='safe',kind='probe',file=str(file),base_file=base,
               actions=actions,all_ids=initial['all_ids'],**check,prepared_at=now())
    write(out/'submission_manifest.json',{'files':[entry]})
    write(out/'campaign.json',cfg)
    # Complete, readable score history, with failed attempts explicitly separate.
    lines=['# V158 historical score audit','','Actual reported scores; TP inferred under G=1044. No simulation rows.','',
           '| File | P | TP | F1 |','|---|---:|---:|---:|']
    for e in system['equations']:lines.append(f"| {Path(e['file']).parent.name}/{Path(e['file']).name} | {e['predictions']} | {e['tp']} | {float(e['score']):.6f} |")
    lines+=['','Failed attempts:']+[f"- {r['attempt_id']}: {r['status']} (no equation)" for r in prior if r.get('status')!='accepted']
    (out/'HISTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return Campaign(out).status()

class Campaign:
    def __init__(self,out=HERE):
        self.out=Path(out);self.cfg=read(self.out/'campaign.json')
        verify_sources(self.cfg['source_hashes'])
        for name,h in self.cfg['implementation_hashes'].items():require(sha(HERE/name)==h,'Implementation drift; explicit audit required: '+name)
        require(sha(self.out/'historical_equations.json')==self.cfg['system_sha256'],'History cache drift')
        self.system=read(self.out/'historical_equations.json')
        self.old=v157().Campaign();self.s=tuple(self.cfg['initial_world_ids'])
        self.ledger=read(self.out/'online_scores.json');self.manifest=read(self.out/'submission_manifest.json')
        self.entries={e['probe_id']:e for e in self.manifest['files']}
        require(len(self.entries)==len(self.manifest['files']),'Duplicate artifact ID')
        self.champ=copy.deepcopy(self.cfg['baseline']);self.safe_baseline=None;self.blocked=[];seen=set();self.observations=[]
        self.eq=Equations(self.system)
        for r in self.ledger['records']:
            require(r['attempt_id'] not in seen,'Duplicate attempt ID');seen.add(r['attempt_id'])
            e=self.entries[r['probe_id']];check=self.validate(e)
            require(r['sha256']==e['sha256'],'Recorded file hash drift')
            if r['status']=='accepted':
                tp=infer(r['score'],e['predictions']);require(tp==r['tp'],'Cached TP mismatch')
                _,_,nodes=csv_nodes(e['file'])
                coeff=dict.fromkeys((self.eq.index[k] for k in nodes),1)
                self.observations.append((coeff,tp))
                if e['phase']=='safe':
                    self.s=tuple(i for i in self.s if self.old.e.sums(e['all_ids'])[i]==tp-969)
                    require(self.s,'New feedback contradicts frozen V157 worlds')
                    if e['kind']=='final':
                        require(tp==972 and e['predictions']==1045,'Safe final deviated from exact prediction')
                        self.safe_baseline=dict(file=e['file'],sha256=e['sha256'],tp=tp,predictions=e['predictions'],score=r['score'])
                candidate=dict(file=e['file'],sha256=e['sha256'],tp=tp,predictions=e['predictions'],score=r['score'],source_class='user_reported_online')
                if f1(tp,e['predictions'])>f1(self.champ['tp'],self.champ['predictions']):self.champ=candidate
            elif r['status'] in ('anomaly','failed'):self.blocked.append(r['attempt_id'])
        self.eq=Equations(self.system,self.observations)
        self.remaining=max(0,self.cfg['total_attempts']-len(seen))
        self.used=len(seen)
        self.accepted={r['probe_id'] for r in self.ledger['records'] if r['status']=='accepted'}

    def validate(self,e):return validate_file(e['file'],e['base_file'],e['actions'],e['sha256'])
    def check_deadline(self):require(datetime.now(TZ)<datetime.fromisoformat(self.cfg['deadline_at']),'Original 48-hour research deadline expired')
    def research_gate(self):
        p=self.out/'research/results.json'
        if not p.exists():return dict(passed=False,decision='research_not_run')
        result=read(p)
        require(result['protocol_sha256']==digest(self.cfg['protocol']),'Research protocol mismatch')
        if result.get('complete'):
            require(sha(p)==read(p.parent/'result_seal.json')['sha256'],'Completed research result changed')
            require(result['passed']==all(result['criteria'].values()),'Research gate flag disagrees with criteria')
        return result

    def status(self):
        today=datetime.now(TZ).date()
        prior=self.cfg['historical_attempts']+self.ledger['records']
        # Unknown/proxy timestamps remain explicitly a conservative daily estimate.
        daily=sum(datetime.fromisoformat(r['submitted_at']).astimezone(TZ).date()==today for r in prior if r.get('submitted_at') and not r.get('duplicate_of'))
        merge=self.old.e.merge(self.s);upper=self.old.e.upper(self.s)
        research=self.research_gate()
        result=dict(actual_champion=self.champ,remaining_submissions=self.remaining,new_attempts_used=self.used,
            total_attempts=8,daily_used_report_time_estimate=daily,daily_limit=2,platform_quota_live_checked=False,
            timing='wait_for_available_day' if daily>=2 else 'verify_platform_daily_availability',
            safe_feasible_worlds=len(self.s),safe_best_known_merge=merge,safe_pool_upper_not_forecast=upper,
            safe_baseline_publicly_confirmed=bool(self.safe_baseline),target_reached=f1(self.champ['tp'],self.champ['predictions'])>=TARGET,
            deadline_at=self.cfg['deadline_at'],research={k:research[k] for k in ('complete','passed','decision','criteria','pooled_gains','bootstrap','preflight','error') if k in research},blocked_attempts=self.blocked)
        if self.blocked:result['decision']='pause_failed_or_anomalous_attempt'
        elif result['target_reached']:result['decision']='stop_target_achieved'
        elif not self.remaining:result['decision']='stop_budget'
        elif not self.safe_baseline:
            if len(self.s)>1:
                result['decision']='submit_safe_probe' if self.remaining>=2 else 'retain_champion_insufficient_safe_route_budget'
                result['file']=self.entries['safe_probe']
                result['score_lookup']=[dict(score=f'{f1(972+dt,1044):.6f}',tp=972+dt,remaining_worlds=len(s),
                    final_merge=self.old.e.merge(s)) for dt,s in self.old.e.branches(self.s,(3,4,7,12)).items()]
            else:result['decision']='prepare_safe_final'
        elif not result['research'].get('passed'):result['decision']='stop_research_gate_failed' if result['research'].get('complete') else 'await_research'
        elif not (self.out/'candidate_catalog.json').exists():result['decision']='build_new_pool'
        else:
            result.update(module('_v158_policy',HERE/'policy.py').recommendation(self))
        return result

    def prepare(self):
        s=self.status();decision=s['decision']
        if decision=='submit_safe_probe':self.validate(self.entries['safe_probe']);return s
        if decision=='prepare_safe_final':
            m=self.old.e.merge(self.s);require(m['tp']==972 and m['predictions']==1045,'Unexpected safe optimum')
            ids=m['ids'];actions=[self.old.e.cat[i] for i in ids]
            e=self.emit('safe_final','safe','final',self.old.e.freeze['base']['file'],actions,all_ids=ids)
        elif decision in ('submit_new_group','submit_new_split','prepare_new_final'):
            cat=read(self.out/'candidate_catalog.json')['candidates'];ids=s['next_ids']
            actions=[a for a in cat if a['candidate_id'] in ids]
            kind='final' if decision=='prepare_new_final' else ('split' if decision=='submit_new_split' else 'group')
            pid=kind+'_'+digest(ids)[:12]
            e=self.emit(pid,'research',kind,self.safe_baseline['file'],actions,ids=ids)
        else:return s
        return {**s,'file':e}

    def emit(self,pid,phase,kind,base,actions,**fields):
        if pid in self.entries:self.validate(self.entries[pid]);return self.entries[pid]
        path=self.out/f'v158_{pid}_utf8.csv'
        check=validate_file(path,base,actions) if path.exists() else author(path,base,actions)
        e=dict(probe_id=pid,phase=phase,kind=kind,file=str(path),base_file=base,actions=actions,
               **fields,**check,prepared_at=now())
        self.manifest['files'].append(e);write(self.out/'submission_manifest.json',self.manifest)
        return e

    def record(self,pid,attempt_id,score,submitted_at,evidence,failed=False):
        require(pid in self.entries,'Unknown prepared artifact');require(attempt_id and evidence,'Attempt ID and evidence required')
        timestamp=datetime.fromisoformat(submitted_at);require(timestamp.tzinfo is not None,'Timestamp needs UTC offset; label proxy times in evidence')
        previous=next((r for r in self.ledger['records'] if r['attempt_id']==attempt_id),None)
        if previous:
            require((previous['probe_id'],previous['score'])==(pid,score),'Attempt ID reused with changed content')
            return {'idempotent':True,**self.status()}
        e=self.entries[pid];self.validate(e)
        row=dict(probe_id=pid,attempt_id=attempt_id,score=score,submitted_at=submitted_at,evidence=evidence,
                 recorded_at=now(),sha256=e['sha256'],predictions=e['predictions'],quota_cost=1,status='failed' if failed else 'accepted')
        if not failed:
            try:
                tp=infer(score,e['predictions']);row['tp']=tp
                _,_,nodes=csv_nodes(e['file']);coeff=dict.fromkeys((self.eq.index[k] for k in nodes),1)
                require(self.eq.solve(extra=[(coeff,tp)]) is not None,'Score conflicts with historical integer equations')
                if e['phase']=='safe':
                    require(any(self.old.e.sums(e['all_ids'])[i]==tp-969 for i in self.s),'Score conflicts with exact safe branches')
                    if e['kind']=='final':require(tp==972 and e['predictions']==1045,'Safe final score anomaly')
            except (ValueError,TypeError) as exc:row.update(status='anomaly',error=str(exc))
        else:require(score is None,'Failed attempt cannot supply a score')
        # Actual attempts are recorded even if the platform allowed more than expected.
        self.ledger['records'].append(row);write(self.out/'online_scores.json',self.ledger)
        return Campaign(self.out).status()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('init','audit','prepare','research','build-pool'):sub.add_parser(name)
    r=sub.add_parser('record');r.add_argument('--probe-id',required=True);r.add_argument('--attempt-id',required=True)
    r.add_argument('--score');r.add_argument('--submitted-at',required=True);r.add_argument('--evidence',required=True);r.add_argument('--failed',action='store_true')
    args=p.parse_args()
    if args.command=='audit':result=Campaign(args.out).status()
    else:
        with lock(args.out):
            if args.command=='init':result=initialize(args.out)
            elif args.command=='prepare':result=Campaign(args.out).prepare();write(args.out/'recommendation.json',result)
            elif args.command=='record':result=Campaign(args.out).record(args.probe_id,args.attempt_id,args.score,args.submitted_at,args.evidence,args.failed);write(args.out/'recommendation.json',result)
            elif args.command=='research':
                result=module('_v158_research',HERE/'research.py').run(Campaign(args.out))
            else:
                result=module('_v158_policy',HERE/'policy.py').build_pool(Campaign(args.out))
    if args.command=='research':result={k:v for k,v in result.items() if k!='folds'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
