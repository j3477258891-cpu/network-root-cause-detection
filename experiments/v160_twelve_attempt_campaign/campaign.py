"""Twelve user-authorized attempts. CSV creation is real; uploads are manual."""
import argparse
import re
from support import *
from engine import best_known

def initialize(out=HERE):
    out=Path(out)
    if (out/'campaign.json').exists():return Campaign(out).audit()
    require(not (out/'online_scores.json').exists(),'Refusing to reset an existing ledger')
    window=out/'execution_window.json'
    if not window.exists():
        start=datetime.now(TZ)
        write(window,dict(started_at=start.isoformat(),deadline_at=(start+timedelta(days=7)).isoformat(),
                          first_preparation_deadline=(start+timedelta(hours=24)).isoformat()))
    timing=read(window);parent=previous()
    require(parent.champ['score']=='0.930589' and parent.champ['tp']==972 and parent.champ['predictions']==1045,
            'Current champion changed; rebase before initialization')
    source=dict(read(V159/'campaign.json')['source_hashes'])
    for folder in (V158,V159):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='operation.lock':source[str(p)]=sha(p)
    for p in (EXP/'v150_adaptive_addition_campaign.py',EXP/'v150_adaptive_policy.py'):
        source[str(p)]=sha(p)
    catalog=build_catalog(parent)
    require(datetime.now(TZ)<datetime.fromisoformat(timing['first_preparation_deadline']),'First preparation exceeded 24 hours')
    cfg=dict(version=160,baseline=parent.champ,total_attempts=12,daily_limit=2,target='0.945000',
             quota_source='User latest explicit report: 12 remaining. Replaces old remaining quota; does not add six.',
             accepted_risk='Unvalidated probes and model-assisted final explicitly authorized',
             previous_statistical_gate='Retained as diagnostics, no longer a release condition',
             **timing,source_hashes=source,catalog_sha256=digest(catalog),
             scenario_count_per_family=256,rollouts_per_query=256,top_queries=24,seed=SEED)
    write(out/'candidate_catalog.json',catalog)
    write(out/'submission_manifest.json',dict(files=[]));write(out/'online_scores.json',dict(records=[]))
    write(out/'campaign.json',cfg)
    c=Campaign(out);maths=c.mathematics()
    require(abs(maths['f1_upper']-.9494274809160306)<1e-10,'Joint oracle changed')
    write(out/'initial_mathematics.json',dict(joint=maths,
        pool_a=parent.eq.oracle(c.pick(c.pools['A']),972,1045),
        pool_ab=parent.eq.oracle(c.pick(c.pools['A']+c.pools['B']),972,1045),
        first_lookup=c.lookup(c.pools['B']),second_lookup=c.lookup(c.pools['C']),
        equation='2 delta_TP - 0.945 delta_P >= 30.105'))
    old=read(V158/'research/results.json');capacity=read(V159/'validation_capacity.json')
    diag=dict(source_class='retrospective diagnostics; not new holdout validation',
              parent_gate_passed=old['passed'],parent_criteria=old['criteria'],
              parent_pooled_gains=old.get('pooled_gains'),parent_bootstrap=old.get('bootstrap'),
              cached_prediction_integrity_errors=capacity['integrity_errors'],
              labeled_train_orders=capacity['labeled_orders'],independent_campaign_trials=0,
              v150_actual_first_pool=dict(predictions=1125,tp=989,score='0.911941',true_additions=20,pool_size=80),
              v150_warning='Three-model nomination produced 20/80, so model support is not independent evidence.',
              historical_replay=old.get('historical_replay'),
              wide_typed_fn_retraining=False,gate_blocks_submission=False)
    # Keep the exact original historical replay key even when older report schemas differ.
    diag['available_parent_report_keys']=list(old)
    write(out/'diagnostics.json',diag)
    return c.audit()

class Campaign:
    def __init__(self,out=HERE):
        self.out=Path(out);self.cfg=read(self.out/'campaign.json')
        verify_sources(self.cfg['source_hashes']);self.parent=previous()
        self.catalog=read(self.out/'candidate_catalog.json')
        require(digest(self.catalog)==self.cfg['catalog_sha256'],'Catalog drift')
        self.actions=self.catalog['actions'];self.pools=self.catalog['pools']
        self.manifest=read(self.out/'submission_manifest.json')
        self.entries={e['probe_id']:e for e in self.manifest['files']}
        require(len(self.entries)==len(self.manifest['files']),'Duplicate artifact ID')
        self.ledger=read(self.out/'online_scores.json');self.records=self.ledger['records']
        self.champ=dict(self.cfg['baseline']);self.champion_ids=[];self.blocked=[];seen=set()
        observations=list(self.parent.observations);self.groups=[]
        self.add_group(self.pools['A'],17)
        self.accepted=set()
        for r in self.records:
            require(r['attempt_id'] not in seen,'Duplicate actual attempt ID');seen.add(r['attempt_id'])
            e=self.entries[r['probe_id']];self.validate(e)
            require(r['sha256']==e['sha256'] and r['quota_cost']==1,'Score artifact/quota drift')
            if r['status']!='accepted':self.blocked.append(r['attempt_id']);continue
            tp=infer(r['score'],e['predictions']);require(tp==r['tp'],'TP cache drift')
            _,_,nodes=csv_nodes(e['file'])
            observations.append((dict.fromkeys((self.parent.eq.index[k] for k in nodes),1),tp))
            self.add_group(e['ids'],tp-972)
            if e.get('query_ids'):
                dt=tp-972-e.get('anchor_delta_tp',0)
                self.add_group(e['query_ids'],dt)
                if e.get('parent_ids'):
                    p=next(g for g in self.groups if g['ids']==e['parent_ids'])
                    self.add_group(sorted(set(p['ids'])-set(e['query_ids'])),p['delta_tp']-dt)
            self.accepted.add(r['probe_id'])
            if f1(tp,e['predictions'])>f1(self.champ['tp'],self.champ['predictions']):
                self.champ=dict(file=e['file'],sha256=e['sha256'],tp=tp,predictions=e['predictions'],
                                score=r['score'],source_class='user_reported_online')
                self.champion_ids=e['ids']
        self.eq=Equations(self.parent.system,observations)
        self.remaining=max(0,self.cfg['total_attempts']-len(seen));self.champion_f1=f1(self.champ['tp'],self.champ['predictions'])
        code={p.name:sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test')}
        self.state_id=digest(dict(config=digest(self.cfg),records=self.records,code=code))[:20]
        self._math=None;self._known=None

    def pick(self,ids):return [self.actions[i] for i in ids]
    def add_group(self,ids,dt):
        ids=sorted(ids)
        if not ids:
            require(dt==0,'Empty group has nonzero count');return
        old=next((g for g in self.groups if g['ids']==ids),None)
        if old:require(old['delta_tp']==dt,'Known group count contradiction')
        else:self.groups.append(dict(ids=ids,delta_tp=int(dt)))
    def validate(self,e):
        require(e['ids']==sorted(set(e['ids'])),'Invalid action IDs')
        require(e['actions']==self.pick(e['ids']),'Manifest action/catalog mismatch')
        return validate_file(e['file'],self.cfg['baseline']['file'],e['actions'],e['sha256'])
    def expired(self):return datetime.now(TZ)>=datetime.fromisoformat(self.cfg['deadline_at'])
    def mathematics(self):
        if self._math is None:self._math=self.eq.oracle(self.actions,972,1045)
        return self._math
    @property
    def mode(self):return 'target' if self.mathematics()['f1_upper']>=float(TARGET) else 'maximize_retained_score'
    def known_merge(self):
        if self._known is None:
            self._known=best_known(self.groups,self.actions)
            m=self._known;require(self.eq.bound(self.pick(m['ids']))==(m['delta_tp'],m['delta_tp']),'Known union not proven')
        return self._known
    def lookup(self,ids):
        actions=self.pick(ids);dp=sum(x['delta_p'] for x in actions);p=1045+dp
        lo,hi=self.eq.bound(actions);rows=[]
        for dt in range(lo,hi+1):
            if self.eq.possible(actions,dt):
                s=f'{f1(972+dt,p):.6f}';require(infer(s,p)==972+dt,'Nonunique six-decimal TP mapping')
                rows.append(dict(score=s,tp=972+dt,delta_tp=dt,predictions=p))
        require(rows,'No feasible feedback');return rows
    def audit(self):
        require(self.eq.solve() is not None,'Ledger equations inconsistent')
        maths=self.mathematics();known=self.known_merge()
        allrows=self.parent.cfg['historical_attempts']+self.parent.ledger['records']
        allrows+=self.records;today=datetime.now(TZ).date()
        daily=sum(datetime.fromisoformat(r['submitted_at']).astimezone(TZ).date()==today for r in allrows if r.get('submitted_at') and not r.get('duplicate_of'))
        pending=[e['probe_id'] for e in self.manifest['files'] if e.get('offered') and e['probe_id'] not in self.accepted]
        decision='adaptive_search'
        if 'probe01_aux05' not in self.accepted:decision='prepare_aux05'
        elif 'probe02_tail12' not in self.accepted:decision='prepare_tail12'
        if self.remaining<=1:decision='prepare_final' if self.remaining else 'stop_budget'
        if reached(known['tp'],known['predictions']):decision='prepare_final'
        if maths['f1_upper']<=self.champion_f1+1e-12:decision='stop_no_remaining_gain'
        if pending:decision='await_prepared_feedback'
        if self.expired():decision='stop_deadline'
        if not self.remaining:decision='stop_budget'
        if reached(self.champ['tp'],self.champ['predictions']):decision='stop_target_achieved'
        if self.blocked:decision='pause_failed_or_anomalous_attempt'
        return dict(state_id=self.state_id,actual_champion=self.champ,target='0.945000',
                    target_reached=reached(self.champ['tp'],self.champ['predictions']),
                    total_attempts=12,new_attempts_used=len(self.records),remaining_submissions=self.remaining,
                    reserved_final=1,deadline_at=self.cfg['deadline_at'],daily_limit=2,
                    daily_used_report_time_estimate=daily,platform_quota_live_checked=False,
                    timing='wait_for_available_day' if daily>=2 else 'verify_platform_daily_availability',
                    mode=self.mode,optimistic_pool_upper=maths,known_merge=known,
                    exact_feasible_world_count=None,world_count_note='Not exhaustively enumerated; use integer bounds, not finite sample count',
                    measured_groups=len(self.groups),pending_probe_ids=pending,decision=decision,
                    blocked_attempts=self.blocked,protected_sources_verified=len(self.cfg['source_hashes']))

    def search(self,sample_count=None,rollouts=None,top_n=None,seconds=900):
        status=self.audit()
        if status['decision'] not in ('adaptive_search','prepare_final'):
            return status
        path=self.out/f'search_{self.state_id}.json'
        if path.exists():
            result=read(path)
            require(sha(path)==read(path.with_suffix('.seal.json'))['sha256'],'Search result drift')
            if result.get('scenario_file'):require(sha(result['scenario_file'])==result['scenario_sha256'],'Scenario drift')
            return result
        p=module('_v160_policy',HERE/'policy.py')
        result=p.search(self,sample_count=sample_count or self.cfg['scenario_count_per_family'],
                        rollouts=rollouts or self.cfg['rollouts_per_query'],top_n=top_n or self.cfg['top_queries'],seconds=seconds)
        write(path,result);write(path.with_suffix('.seal.json'),dict(sha256=sha(path)))
        return result

    def emit(self,pid,ids,kind,offered=True,**meta):
        require(self.remaining>0 and not self.expired() and not self.blocked,'Emission blocked by quota/deadline/anomaly')
        if kind!='final':require(self.remaining>=2,'Final slot is reserved')
        ids=sorted(ids);lookup=self.lookup(ids)
        if pid in self.entries:self.validate(self.entries[pid]);return self.entries[pid]
        file=self.out/f'v160_{pid}_utf8.csv'
        actions=self.pick(ids)
        check=validate_file(file,self.cfg['baseline']['file'],actions) if file.exists() else author(file,self.cfg['baseline']['file'],actions)
        e=dict(probe_id=pid,file=str(file),ids=ids,actions=actions,kind=kind,offered=offered,
               prepared_at=now(),prepared_state=self.state_id,**check,score_lookup=lookup,**meta)
        self.manifest['files'].append(e);write(self.out/'submission_manifest.json',self.manifest)
        self.entries[pid]=e
        return e

    def prepare_next(self):
        s=self.audit();d=s['decision']
        if d=='await_prepared_feedback':return dict(**s,file=self.entries[s['pending_probe_ids'][0]])
        if d=='prepare_aux05':e=self.emit('probe01_aux05',self.pools['B'],'total',query_ids=self.pools['B'])
        elif d=='prepare_tail12':e=self.emit('probe02_tail12',self.pools['C'],'total',query_ids=self.pools['C'])
        elif d=='prepare_final':return self.prepare_final()
        elif d=='adaptive_search':
            result=self.search()
            if result['decision']=='final':return self.prepare_final(result)
            if result['decision']!='probe':return dict(**s,search=result)
            q=result['query'];pid=f"probe{len(self.records)+1:02d}_split_{digest(q['ids'])[:8]}"
            e=self.emit(pid,q['ids'],'split',query_ids=q['query_ids'],parent_ids=q['parent_ids'],
                        anchor_ids=q['anchor_ids'],anchor_delta_tp=q['anchor_delta_tp'])
        else:return s
        result=dict(**s,file=e,submission_cost=1,upload_performed=False)
        write(self.out/'recommendation.json',result);return result

    def prepare_final(self,search_result=None):
        s=self.audit()
        if s['decision'].startswith(('stop','pause','await')):return s
        require(self.remaining>=1,'No final submission remaining')
        known=self.known_merge();files={}
        if known['ids'] and known['f1']>self.champion_f1+1e-12:
            files['known']=self.emit('final_known_'+self.state_id,known['ids'],'final',offered=False,
                                     evidence='exact known group counts',expected_tp=known['tp'])
        if reached(known['tp'],known['predictions']):chosen='known';comparison=dict(known_merge=known)
        else:
            comparison=search_result or self.search()
            model=comparison.get('model_assisted_final')
            if model and model['ids'] and model['conditional_expected_retained_f1']>self.champion_f1+1e-12:
                files['model']=self.emit('final_model_'+self.state_id,model['ids'],'final',offered=False,
                                         evidence='model-assisted unresolved labels',comparison=model)
            chosen='model' if 'model' in files else 'known'
        if chosen not in files:
            return dict(**s,final_decision='retain_actual_champion',comparison=comparison,
                        reason='No completed eligible final improvement; retain actual champion')
        files[chosen]['offered']=True;write(self.out/'submission_manifest.json',self.manifest)
        result=dict(**s,final_decision='submit_'+chosen,file=files[chosen],alternatives=files,
                    comparison=comparison,submission_cost=1,upload_performed=False)
        write(self.out/'recommendation.json',result);return result

    def record(self,pid,attempt_id,reported,submitted_at,evidence,failed=False):
        require(pid in self.entries,'Unknown prepared file')
        require(attempt_id and evidence,'Attempt ID and evidence required')
        require(datetime.fromisoformat(submitted_at).tzinfo is not None,'Timestamp timezone required; label proxy times')
        prior=next((r for r in self.records if r['attempt_id']==attempt_id),None)
        if prior:
            require((prior['probe_id'],prior['score'],prior['status']=='failed')==(pid,reported,failed),'Changed duplicate attempt')
            return dict(idempotent=True,**self.audit())
        require(not failed or reported is None,'Failed attempt cannot have a score')
        e=self.entries[pid];self.validate(e)
        row=dict(probe_id=pid,attempt_id=attempt_id,score=reported,submitted_at=submitted_at,
                 recorded_at=now(),evidence=evidence,sha256=e['sha256'],predictions=e['predictions'],
                 quota_cost=1,status='failed' if failed else 'accepted')
        if not failed:
            try:
                require(isinstance(reported,str) and re.fullmatch(r'0\.\d{6}|1\.000000',reported),'Six-decimal score required')
                tp=infer(reported,e['predictions']);row['tp']=tp
                require(self.eq.possible(self.pick(e['ids']),tp-972),'Score conflicts with full history')
            except (ValueError,TypeError) as exc:row.update(status='anomaly',error=str(exc))
        self.records.append(row);write(self.out/'online_scores.json',self.ledger)
        state=Campaign(self.out).audit();write(self.out/'recommendation.json',state)
        return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('init','audit','search','prepare-next','prepare-final'):sub.add_parser(name)
    r=sub.add_parser('record')
    for name in ('probe-id','attempt-id','submitted-at','evidence'):r.add_argument('--'+name,required=True)
    r.add_argument('--score');r.add_argument('--failed',action='store_true')
    a=p.parse_args()
    try:
        if a.command=='audit':result=Campaign(a.out).audit()
        else:
            a.out.mkdir(parents=True,exist_ok=True)
            with lock(a.out):
                if a.command=='init':result=initialize(a.out)
                else:
                    c=Campaign(a.out)
                    result=c.record(a.probe_id,a.attempt_id,a.score,a.submitted_at,a.evidence,a.failed) if a.command=='record' else getattr(c,a.command.replace('-','_'))()
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError) as exc:
        print(json.dumps(dict(error=str(exc),decision='pause_and_audit'),ensure_ascii=False));raise SystemExit(2)

if __name__=='__main__':main()
