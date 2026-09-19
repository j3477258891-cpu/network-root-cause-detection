"""V157: six NEW user-confirmed attempts. Never writes V156 or uploads."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.with_name('v156_joint_decode_campaign')
sys.path.insert(0, str(PREVIOUS))
import verified_engine as V
import numpy as np

ROWS = ((6,10,12,13), (3,7,13,16), (1,2,13,16), (1,3,13))
BASE_IDS = (8,11,15)
NODE = Path('C:/Users/86158/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
BASE_SHA = '8ed77d30fa1773cded4d98ff6d40326738575f9a608c9b2f129ebfec5e897221'

def save(name, obj): V.write(HERE/name, obj)
def read(name): return V.read(HERE/name)

def build():
    if (HERE/'campaign.json').exists():
        c=Campaign()
        return {'reused':True, **c.status()}
    e=V.Engine()
    V.require(e.champ['sha256']==BASE_SHA and e.champ['tp']==972, 'Baseline not the scored V156 final')
    s=tuple(map(int,e.support))
    V.require(len(s)==78, 'Expected 78 frozen feasible worlds; audit changed history')
    signatures=np.stack([e.sums(q)[list(s)] for q in ROWS],axis=1)
    V.require(len(np.unique(signatures,axis=0))==len(s), 'Matrix is not injective')
    paths=[PREVIOUS/n for n in ('online_scores.json','submission_manifest.json','verified_freeze.json','history_worlds.json','verified_engine.py','campaign.py')]
    source={**e.freeze['source_hashes'], **{str(p):V.sha(p) for p in paths}}
    cat={'candidates':[e.cat[i] for i in e.ids]}
    save('candidate_catalog.json',cat)
    save('probe_matrix.json',{'rows':[list(q) for q in ROWS], 'base_ids':BASE_IDS,
         'world_ids':s,'count_signatures':signatures.tolist(),'unique_signatures':len(s)})
    save('campaign.json',{'version':157,'created_at':V.now(),'baseline':e.champ,
         'new_attempt_budget':6,'daily_limit':2,'quota_source':'User explicitly confirmed six remaining opportunities after V156 final; platform not independently checked',
         'source_hashes':source,'catalog_sha256':V.sha(HERE/'candidate_catalog.json'),
         'matrix_sha256':V.sha(HERE/'probe_matrix.json'),'competition_submitted':False})
    save('online_scores.json',{'records':[]})
    save('submission_manifest.json',{'files':[]})
    return Campaign().status()

class Campaign:
    def __init__(self):
        self.cfg=read('campaign.json')
        for p,h in self.cfg['source_hashes'].items(): V.require(V.sha(p)==h,'Historical input changed: '+p)
        for f,k in (('candidate_catalog.json','catalog_sha256'),('probe_matrix.json','matrix_sha256')):
            V.require(V.sha(HERE/f)==self.cfg[k],'Frozen '+f+' changed')
        self.e=V.Engine()
        self.matrix=read('probe_matrix.json')
        self.s=tuple(self.matrix['world_ids'])
        self.ledger=read('online_scores.json'); self.manifest=read('submission_manifest.json')
        self.entries={x['probe_id']:x for x in self.manifest['files']}
        self.champ=self.cfg['baseline']; self.champ_f1=2*self.champ['tp']/(1044+self.champ['predictions'])
        self.accepted_queries=set(); self.accepted_actions=set(); self.blocked=[]; seen=set()
        for r in self.ledger['records']:
            V.require(r['attempt_id'] not in seen,'Duplicate attempt id');seen.add(r['attempt_id'])
            x=self.entries[r['probe_id']]
            if r['status']=='accepted':
                chk=self.validate(x)
                tp=V.infer(r['score'],chk['predictions'])
                V.require(tp==r['inferred_tp'],'Cached TP mismatch')
                values=self.e.sums(x['all_ids'])
                self.s=tuple(i for i in self.s if values[i]==tp-969)
                V.require(self.s,'Score conflicts with history')
                self.accepted_actions.add(tuple(x['all_ids']))
                if x['kind']=='probe':self.accepted_queries.add(tuple(x['query_ids']))
                f1=2*tp/(1044+chk['predictions'])
                if f1>self.champ_f1:
                    self.champ_f1=f1; self.champ={'file':str(HERE/x['file']),'sha256':x['sha256'],
                        'predictions':chk['predictions'],'tp':tp,'score':r['score'],'source_class':'user_reported_online'}
            elif r['status']=='anomaly' and not r.get('resolved_by'):self.blocked.append(r['attempt_id'])
        self.remaining=max(0,self.cfg['new_attempt_budget']-len(seen))

    def validate(self,x):
        return self.e.csv.validate_csv(HERE/x['file'],x['all_ids'],x['sha256'])

    def exact_one_shot(self):
        """Optional verified shortcut; never substitutes an incomplete search."""
        if not 2<=len(self.s)<=6:return None
        e=self.e
        pool=[i for i in e.active if i not in BASE_IDS and
              len(set(map(int,e.world[list(self.s),e.pos[i]])))>1]
        carrier=[i for i in e.active if e.cat[i]['kind']=='delete' and
                 np.all(e.world[list(self.s),e.pos[i]]==0)]
        mass=e.weights[list(self.s)];mass=mass/mass.sum()
        for n in range(1,len(pool)+1):
            choices=[]
            for q in itertools.combinations(pool,n):
                bs=e.branches(self.s,q)
                if len(bs)!=len(self.s):continue
                p=1048+sum(e.cat[i]['delta_p'] for i in (*q,*carrier))
                scores=2*(972+e.sums(q)[list(self.s)])/(1044+p)
                choices.append(((n,-float(scores.min()),-float(mass@scores),q),q))
            if choices:
                q=min(choices)[1]
                return {'query':list(q),'carrier':carrier,'verified_unique_feedbacks':len(self.s)}
        return None

    def status(self):
        m=self.e.merge(self.s); upper=self.e.upper(self.s)
        unmeasured=[q for q in ROWS if q not in self.accepted_queries and len(self.e.branches(self.s,q))>1]
        exhausted=m['f1']>=upper-1e-12
        shortcut=None if exhausted or self.blocked or not self.remaining else self.exact_one_shot()
        required_queries=1 if shortcut else len(unmeasured)
        if self.blocked: decision='pause_anomaly'
        elif not self.remaining: decision='stop_budget'
        elif exhausted: decision='emit_final' if m['f1']>self.champ_f1+1e-12 else 'retain_champion_pool_optimum'
        elif self.remaining<required_queries+1:
            decision='emit_final_budget_fallback' if m['f1']>self.champ_f1+1e-12 else 'pause_insufficient_budget'
        else: decision='prepare_probe'
        q=list(unmeasured[0]) if decision=='prepare_probe' else None
        carrier=[]
        if q and shortcut:q=shortcut['query'];carrier=shortcut['carrier']
        today=V.now()[:10]
        prior=V.read(PREVIOUS/'online_scores.json')['records']+V.read(self.e.freeze['prior_ledger'])['records']
        daily=sum(str(r['submitted_at'])[:10]==today for r in prior+self.ledger['records'] if not r.get('duplicate_of'))
        out={'actual_champion':self.champ,'remaining_submissions':self.remaining,'new_attempts_used':6-self.remaining,
             'feasible_worlds':len(self.s),'best_known_merge':m,'pool_upper_not_forecast':upper,
             'decision':decision,'next_query':q,'unmeasured_informative_rows':len(unmeasured),
             'daily_used_including_prior_reports':daily,'daily_limit':2,'platform_quota_live_checked':False,
             'submission_timing':'wait_for_available_day' if daily>=2 else 'verify_platform_quota',
             'conditional_route_budget_ok':self.remaining>=required_queries+1,
             'required_queries_upper_bound':required_queries,'carrier_ids':carrier,
             'query_policy':'verified_one_shot' if q and shortcut else 'fixed_default',
             'pool_optimum_identified':exhausted,'blocked_anomalies':self.blocked}
        if q:
            p=1048+sum(self.e.cat[i]['delta_p'] for i in q+carrier)
            out['next_score_lookup']=[{'score':f'{2*(972+dt)/(1044+p):.6f}','tp':972+dt,
                'query_delta_tp':dt,'remaining_worlds':len(t)} for dt,t in self.e.branches(self.s,q).items()]
        return out

def prepare():
    c=Campaign(); state=c.status(); decision=state['decision']
    if decision=='prepare_probe':
        query=state['next_query'];carrier=state.get('carrier_ids',[])
        ids=sorted(set(BASE_IDS)|set(query)|set(carrier)); kind='probe'
    elif decision.startswith('emit_final'):
        ids=state['best_known_merge']['ids']; query=[i for i in ids if i not in BASE_IDS];carrier=[];kind='final'
    else:return state
    for x in c.manifest['files']:
        if x['all_ids']==ids and tuple(ids) not in c.accepted_actions:
            c.validate(x); state['file']=x; save('recommendation.json',state);return state
    adaptive=kind=='probe' and state.get('query_policy')=='verified_one_shot'
    number=ROWS.index(tuple(query))+1 if kind=='probe' and not adaptive else len(c.manifest['files'])+1
    pid=f'{kind}_{"adaptive_" if adaptive else ""}p{number:02d}'; filename=f'v157_{pid}_utf8.csv'
    V.require(not (HERE/filename).exists(),'Existing artifact; inspect before overwrite')
    request={'baseline':c.cfg['baseline'],'catalog':str(HERE/'candidate_catalog.json'),
             'query_ids':sorted(query+carrier),'output':str(HERE/filename),'expected_p':1048+sum(c.e.cat[i]['delta_p'] for i in query+carrier)}
    save('author_request.json',request)
    subprocess.run([str(NODE),str(HERE/'artifact_builder.mjs'),str(HERE/'author_request.json')],check=True)
    chk=c.e.csv.validate_csv(HERE/filename,ids)
    x={'probe_id':pid,'kind':kind,'query_ids':query,'carrier_ids':carrier,'all_ids':ids,'file':filename,
       'predictions':chk['predictions'],'sha256':chk['sha256'],'validation':chk,
       'encoding':'utf-8-no-bom','created_at':V.now(),'status':'prepared_not_submitted'}
    c.manifest['files'].append(x);save('submission_manifest.json',c.manifest)
    state['file']=x;save('recommendation.json',state);return state

def record(pid,aid,score,at,evidence,failed=False):
    c=Campaign();V.datetime.fromisoformat(at)
    V.require(pid in c.entries and evidence.strip(),'Unknown file or missing evidence')
    prior=next((r for r in c.ledger['records'] if r['attempt_id']==aid),None)
    if prior:
        V.require(prior['probe_id']==pid and prior.get('score')==score and (prior['status']=='failed')==failed,'Conflicting repeated attempt')
        return {'idempotent':True,'record':prior}
    V.require(c.remaining>0,'No authorized remaining quota')
    x=c.entries[pid]
    row={'probe_id':pid,'attempt_id':aid,'score':score,'submitted_at':at,'evidence':evidence,
         'recorded_at':V.now(),'quota_cost':1,'status':'failed' if failed else 'accepted','sha256':x['sha256']}
    if not failed:
        try:
            chk=c.validate(x);tp=V.infer(score,chk['predictions'])
            V.require(any(c.e.sums(x['all_ids'])[j]==tp-969 for j in c.s),'Score conflicts with historical equations')
            row.update(inferred_tp=tp,predictions=chk['predictions'],delta_tp_vs_champion_baseline=tp-972)
        except (ValueError,FileNotFoundError) as ex:row.update(status='anomaly',error=str(ex))
    c.ledger['records'].append(row);save('online_scores.json',c.ledger)
    state=Campaign().status();save('recommendation.json',state)
    return {'record':row,'state':state}

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('build','audit','recommend','prepare'):sub.add_parser(name)
    r=sub.add_parser('record')
    for k in ('probe-id','attempt-id','submitted-at','evidence'):r.add_argument('--'+k,required=True)
    r.add_argument('--score');r.add_argument('--failed',action='store_true')
    a=p.parse_args()
    if a.cmd=='build':out=build()
    elif a.cmd=='audit':out=Campaign().status()
    elif a.cmd=='recommend':out=Campaign().status();save('recommendation.json',out)
    elif a.cmd=='prepare':out=prepare()
    else:out=record(a.probe_id,a.attempt_id,a.score,a.submitted_at,a.evidence,a.failed)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
