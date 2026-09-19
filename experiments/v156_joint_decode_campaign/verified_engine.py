"""Live V156 engine. Full joint worlds, signed counts, immutable attempts; no uploads."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('v153_complementary_repair_campaign')
NEW = (8,9,11,14,15,17)
FIXED = ((8,), (9,11), (9,14))
G, P0, TP0 = 1044,1045,969
WARN = 'Model-conditional expectation, not calibrated success probability or global optimum.'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(p, value):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)
def require(ok,msg):
    if not ok: raise ValueError(msg)
def f1(dt=0,dp=0): return 2*(TP0+dt)/(G+P0+dp)
def infer(score,p):
    s=str(score)
    require(len(s.split('.')[-1])==6,'Use the exact six-decimal public score')
    hits=[t for t in range(min(G,p)+1) if f'{2*t/(G+p):.6f}'==s]
    require(len(hits)==1,'Score has no unique integer TP')
    return hits[0]

def legacy():
    spec=importlib.util.spec_from_file_location('_v156_csv_legacy',HERE/'campaign.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cat=read(OLD/'candidate_catalog.json')['candidates']
    m.CAND={a['candidate_id']:{**a,'mp':a['model_probabilities']} for a in cat}
    return m

def history_context():
    """Use V153's validated historical equations without modifying its files."""
    saved={k:sys.modules.get(k) for k in ('bridge','policy','campaign')}
    for k in saved: sys.modules.pop(k,None)
    sys.path.insert(0,str(OLD))
    try:
        spec=importlib.util.spec_from_file_location('_v153_history_live',OLD/'campaign.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        ctx=m.context(OLD)
        for e in ctx[5].system['equations']:
            require(sha(e['file'])==e['sha256'],'Historical CSV hash drift: '+e['file'])
        return ctx,m.Feasibility
    finally:
        sys.path.remove(str(OLD))
        for k,v in saved.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

def source_hashes(ctx):
    paths=[OLD/n for n in ('campaign.json','candidate_catalog.json','equation_system.json','online_scores.json','submission_manifest.json')]
    paths += [Path(p) for p in ctx[0]['historical_ledger_hashes']]
    paths += [Path(e['file']) for e in ctx[5].system['equations']]
    paths += [Path(ctx[0]['baseline']['file']), Path(ctx[7]['file'])]
    paths += [OLD/e['file'] for e in ctx[2]['files'] if any(r['probe_id']==e['probe_id'] and r['status']=='accepted' for r in ctx[3]['records'])]
    return {str(p):sha(p) for p in sorted(set(paths))}

def build():
    ctx,FC=history_context(); hashes=source_hashes(ctx)
    fp=HERE/'verified_freeze.json'
    if fp.exists():
        old=read(fp);require(old['source_hashes']==hashes,'Frozen source changed; re-audit, do not silently refreeze')
        cache=read(HERE/'history_worlds.json')
        require(sha(HERE/'history_worlds.json')==old['worlds_sha256'],'World cache changed')
        return {'reused':True,'worlds':len(cache['worlds'])}
    m=legacy();ctx[5].feasible(); feas=FC(ctx[5],ctx[4])
    worlds=[]; raw=m.enumerate_combos()
    for j,c in enumerate(raw):
        if feas([([i],v) for i,v in c.items()]): worlds.append([c[i] for i in sorted(c)])
        if j%400==0: print(f'historical MILP audit {j}/{len(raw)}',flush=True)
    require(worlds,'No feasible joint world')
    write(HERE/'history_worlds.json',{'candidate_ids':sorted(ctx[4]),'worlds':worlds,'initial_enumeration':len(raw),'historical_equations':len(ctx[5].rows)})
    write(fp,{'source_hashes':hashes,'worlds_sha256':sha(HERE/'history_worlds.json'),
        'created_at':now(),'prior_attempts':len(ctx[3]['records']),'total_attempts':10,'daily_limit':2,
        'base':ctx[0]['baseline'],'champion':ctx[7],'prior_ledger':str(OLD/'online_scores.json')})
    return {'built':True,'worlds':len(worlds),'history_equations':len(ctx[5].rows)}

class Engine:
    def __init__(self):
        self.freeze=read(HERE/'verified_freeze.json')
        for p,h in self.freeze['source_hashes'].items():require(sha(p)==h,'Frozen input hash changed: '+p)
        require(sha(HERE/'history_worlds.json')==self.freeze['worlds_sha256'],'History cache drift')
        self.csv=legacy();self.cat=self.csv.CAND
        cache=read(HERE/'history_worlds.json');self.ids=tuple(cache['candidate_ids']);self.pos={i:j for j,i in enumerate(self.ids)}
        self.world=np.array(cache['worlds'],dtype=np.int8)
        self.ledger=read(HERE/'online_scores.json');self.manifest=read(HERE/'submission_manifest.json')
        self.entries={e['probe_id']:e for e in self.manifest['files']}
        self.champ=dict(self.freeze['champion']);self.champ_score=2*self.champ['tp']/(G+self.champ['predictions'])
        self.support=np.arange(len(self.world));self.accepted_sets=set(); self.blocked=[]
        attempts=set()
        for r in self.ledger['records']:
            if r.get('duplicate_of'):
                target=next((x for x in self.ledger['records'] if x['attempt_id']==r['duplicate_of']),None)
                require(target is not None and not target.get('duplicate_of') and
                        (target['probe_id'],target.get('score'),target.get('sha256'))==
                        (r['probe_id'],r.get('score'),r.get('sha256')),'Invalid explicit duplicate reconciliation')
                continue
            require(r['attempt_id'] not in attempts,'Duplicate attempt id');attempts.add(r['attempt_id'])
            e=self.entries[r['probe_id']];p=HERE/e['file']
            require(sha(p)==e['sha256'],'Attempted artifact hash changed')
            if r['status']=='accepted':
                chk=self.csv.validate_csv(p,e['all_ids'],e['sha256']);tp=infer(r['score'],chk['predictions'])
                require(tp==r['inferred_tp'],'Cached TP drift')
                dt=tp-TP0;ids=e['all_ids']; sums=self.sums(ids)
                self.support=self.support[sums[self.support]==dt]
                require(len(self.support)>0,'Accepted scores conflict')
                self.accepted_sets.add(tuple(sorted(ids)))
                value=2*tp/(G+chk['predictions'])
                if value>self.champ_score:
                    self.champ_score=value;self.champ={'file':str(p),'sha256':chk['sha256'],'tp':tp,'predictions':chk['predictions'],'score':r['score'],'source_class':'public_scored'}
            elif r['status']=='anomaly' and not r.get('resolved_by'):self.blocked.append(r['attempt_id'])
        self.used=self.freeze['prior_attempts']+len(attempts);self.remaining=max(0,10-self.used)
        today=now()[:10]
        prior=read(self.freeze['prior_ledger'])['records']
        self.daily_used=sum(str(r['submitted_at'])[:10]==today for r in prior+self.ledger['records'] if not r.get('duplicate_of'))
        # Mixture is applied ONCE to complete joint worlds. Conditioning couples model weights.
        probs=[]
        for model in ('catboost','v38_control'):
            logp=np.zeros(len(self.world))
            for i in self.ids:
                p=self.cat[i]['model_probabilities'][model]
                logp+=np.array([math.log(max(p.get(str(int(v)),0),1e-300)) for v in self.world[:,self.pos[i]]])
            probs.append(logp)
        shift=max(float(x.max()) for x in probs)
        self.model_parts=np.stack([np.exp(x-shift)*.5 for x in probs])
        self.weights=self.model_parts.sum(axis=0);self.weights/=self.weights.sum()
        self.active=tuple(i for i in self.ids if i!=5)
        self.bits=((np.arange(1<<len(self.active))[:,None]>>np.arange(len(self.active)))&1).astype(np.int16)
        self.dp=self.bits@np.array([self.cat[i]['delta_p'] for i in self.active])
        self.changes=self.bits.sum(axis=1);self.merge_cache={};self.decode_cache={}
    def sums(self,ids):return self.world[:,[self.pos[i] for i in ids]].sum(axis=1) if ids else np.zeros(len(self.world),dtype=int)
    def labels(self,s):return tuple(sorted(set(tuple(map(int,row)) for row in self.world[list(s)][:,[self.pos[i] for i in NEW]])))
    def depth(self,s):
        labs=self.labels(s)
        return decode_depth(labs)
    def branches(self,s,q):
        v=self.sums(q);return {int(k):tuple(i for i in s if v[i]==k) for k in np.unique(v[list(s)])}
    def merge(self,s):
        key=tuple(s)
        if key in self.merge_cache:return self.merge_cache[key]
        # Eliminate individually known outcomes before subset enumeration. This is
        # exact: a true add / false delete / +1 swap strictly improves every legal
        # merge; the converse can never improve it. Unknown group counts remain.
        w=self.world[list(s)]
        fixed=[];variable=[]
        for i in self.active:
            vals=np.unique(w[:,self.pos[i]])
            if len(vals)>1:variable.append(i)
            elif (self.cat[i]['kind']=='delete' and vals[0]==0) or (self.cat[i]['kind']!='delete' and vals[0]==1):fixed.append(i)
        fixed_dt=sum(int(w[0,self.pos[i]]) for i in fixed)
        fixed_dp=sum(self.cat[i]['delta_p'] for i in fixed)
        cols=[self.pos[i] for i in variable];a=w[:,cols].astype(np.int64)
        bits=((np.arange(1<<len(variable))[:,None]>>np.arange(len(variable)))&1).astype(np.int16)
        dp=bits@np.array([self.cat[i]['delta_p'] for i in variable],dtype=np.int64)+fixed_dp
        # Exact integer row-space basis of differences; no probability-based label removal.
        basis={}
        for row in np.unique(a-a[0],axis=0):
            v=[int(x) for x in row]
            for pivot,b in basis.items():
                if v[pivot]:
                    f=v[pivot];d=b[pivot];v=[x*d-y*f for x,y in zip(v,b)]
                    g=math.gcd(*v)
                    if g:v=[x//g for x in v]
            nz=next((i for i,x in enumerate(v) if x),None)
            if nz is not None:basis[nz]=v;basis=dict(sorted(basis.items()))
        valid=np.ones(len(bits),dtype=bool)
        for b in basis.values():valid &= bits@np.array(b,dtype=np.int64)==0
        dt=bits@a[0]+fixed_dt;vals=2*(TP0+dt)/(G+P0+dp);vals[~valid]=-1
        top=np.flatnonzero(vals==vals.max());n=top[np.argmin(bits[top].sum(axis=1))]
        ids=sorted(fixed+[i for j,i in enumerate(variable) if bits[n,j]])
        require(len(set(map(int,self.sums(ids)[list(s)])))==1,'Constant merge verification failed')
        out={'ids':ids,'delta_tp':int(dt[n]),'delta_p':int(dp[n]),'f1':float(vals[n]),'predictions':P0+int(dp[n]),'tp':TP0+int(dt[n])}
        self.merge_cache[key]=out;return out
    def upper(self,s):
        w=self.world[list(s)];best=0.
        for c in w:
            ids=[i for i in self.ids if (self.cat[i]['kind']=='delete' and c[self.pos[i]]==0) or (self.cat[i]['kind']!='delete' and c[self.pos[i]]==1)]
            best=max(best,f1(sum(int(c[self.pos[i]]) for i in ids),sum(self.cat[i]['delta_p'] for i in ids)))
        return best
    def status(self):
        s=tuple(self.support);weights=self.model_parts[:,self.support].sum(axis=1);weights/=weights.sum()
        return {'actual_champion':self.champ,'attempts_used':self.used,'remaining_submissions':self.remaining,
            'daily_used':self.daily_used,'daily_limit':2,'platform_quota_live_checked':False,'feasible_worlds':len(s),
            'six_node_label_states':len(self.labels(s)),'decode_probes_needed':self.depth(s),
            'best_known_merge':self.merge(s),'theoretical_upper_not_forecast':self.upper(s),
            'model_weights':dict(zip(('catboost','v38_control'),map(float,weights))),
            'blocked_anomalies':self.blocked,'protected_route_budget_ok':self.remaining>=self.depth(s)+1}

@lru_cache(None)
def decode_depth(labs):
    if len(labs)<=1:return 0
    best=6
    for mask in range(1,1<<len(NEW)):
        groups={}
        for row in labs:
            k=sum(v for j,v in enumerate(row) if mask>>j&1);groups.setdefault(k,[]).append(row)
        if len(groups)<2:continue
        d=1+max(decode_depth(tuple(g)) for g in groups.values())
        best=min(best,d)
    return best

class Search:
    """Complete policy values only; a verified fallback exists before timed optimization."""
    def __init__(self,e,weights,seconds):
        self.e=e;self.w=weights;self.end=time.monotonic()+seconds;self.memo={};self.fallback_cache={};self.truncated=False;self.nodes=0
    def queries(self,s,h):
        if self.e.depth(s):pool=NEW
        else:pool=tuple(i for i in self.e.active if i not in NEW)
        seen=set()
        for n in range(1,len(pool)+1):
            for q in itertools.combinations(pool,n):
                bs=self.e.branches(s,q)
                if len(bs)<2:continue
                signature=tuple(sorted(bs.values()))
                if signature in seen:continue
                seen.add(signature)
                if any(self.e.depth(t)>h-1 for t in bs.values()):continue
                yield q,bs
    def terminal(self,s,champ):
        m=self.e.merge(s);value=max(champ,m['f1'])
        return {'expected':value,'worst':value,'q':None,'branches':[],'complete':True}
    def evaluate(self,s,h,champ,q,bs,recursive):
        total=float(self.w[list(s)].sum());out=[];ev=0.;worst=1.
        require(total>0,'No modeled support; cannot rank')
        for dt,t in sorted(bs.items()):
            # Physical query carries known true additions only; exact contribution below.
            carrier=self.carrier(s,q);cdt=int(self.e.sums(carrier)[s[0]]) if carrier else 0
            dp=sum(self.e.cat[i]['delta_p'] for i in q+carrier)
            score=f1(dt+cdt,dp);nchamp=max(champ,score)
            child=recursive(t,h-1,nchamp)
            p=float(self.w[list(t)].sum()/total);ev+=p*child['expected'];worst=min(worst,child['worst'])
            out.append({'delta_tp':dt,'weight':p,'probe_f1':score,'worlds':len(t),'next':child})
        return {'expected':ev,'worst':worst,'q':list(q),'carrier':list(self.carrier(s,q)),'branches':out,'complete':True}
    def carrier(self,s,q):
        return tuple(i for i in NEW if i not in q and np.all(self.e.world[list(s),self.e.pos[i]]==1))
    def fallback(self,s,h,champ):
        key=(s,h,champ)
        if key in self.fallback_cache:return self.fallback_cache[key]
        base=self.terminal(s,champ)
        if h and self.e.depth(s):
            # Pick the safe split minimizing worst then average label ambiguity.
            choices=[]
            for q,bs in self.queries(s,h):
                choices.append(((max(len(self.e.labels(t)) for t in bs.values()),sum(len(t)**2 for t in bs.values()),len(q),q),q,bs))
            require(choices,'Budget cannot finish decoding')
            _,q,bs=min(choices,key=lambda x:x[0]);base=self.evaluate(s,h,champ,q,bs,self.fallback)
        self.fallback_cache[key]=base;return base
    def solve(self,s,h,champ):
        key=(s,h,champ)
        if key in self.memo:return self.memo[key]
        if time.monotonic()>self.end:raise TimeoutError()
        self.nodes+=1;best=self.fallback(s,h,champ)
        if h:
            for q,bs in self.queries(s,h):
                if time.monotonic()>self.end:raise TimeoutError()
                v=self.evaluate(s,h,champ,q,bs,self.solve)
                def rank(x):return (x['expected'],x['worst'],-len(x['q'] or []),tuple(-i for i in x['q'] or []))
                if rank(v)>rank(best):best=v
        self.memo[key]=best;return best
    def run(self,s,h,champ):
        best=self.fallback(s,h,champ)
        try:best=self.solve(s,h,champ)
        except TimeoutError:self.truncated=True
        return {**best,'search_truncated':self.truncated,'search_nodes':self.nodes,'global_optimum_claim':False}

def author(e,ids,query,carrier,kind='probe'):
    require(e.remaining>0 and not e.blocked,'No quota or unresolved anomaly')
    require(len(ids)==len(set(ids)),'Overlapping query/carrier')
    for old in e.manifest['files']:
        if old.get('status')=='prepared' and old.get('all_ids')==sorted(ids) and not any(r['probe_id']==old['probe_id'] for r in e.ledger['records']):
            e.csv.validate_csv(HERE/old['file'],ids,old['sha256']);return old
    serial=len(e.manifest['files'])+1;pid=f'{kind}_r{serial:02d}'
    name=f'v156_{pid}_utf8.csv';path=HERE/name;require(not path.exists(),'Artifact exists; refuse overwrite')
    e.csv.apply_actions(e.csv.BASE,ids,path);chk=e.csv.validate_csv(path,ids)
    entry={'probe_id':pid,'file':name,'kind':kind,'all_ids':sorted(ids),'query_ids':sorted(query),'carrier_ids':sorted(carrier),
        'predictions':chk['predictions'],'sha256':chk['sha256'],'local_validation':chk,'created_at':now(),'status':'prepared','encoding':'utf-8-no-bom'}
    e.manifest['files'].append(entry);write(HERE/'submission_manifest.json',e.manifest);return entry

def recommend(emit=False,seconds=1800):
    e=Engine();report=e.status();s=tuple(map(int,e.support));h=max(0,e.remaining-1)
    if e.blocked:report['decision']='pause_anomaly'
    elif not e.remaining:report['decision']='budget_exhausted'
    elif e.depth(s)>h:report['decision']='pause_insufficient_decode_budget'
    else:
        strategy=Search(e,e.weights,seconds);policy=strategy.run(s,h,e.champ_score)
        uniform=Search(e,np.ones(len(e.world)),min(seconds,60));u=uniform.run(s,h,e.champ_score)
        report.update(policy=policy,uniform_sensitivity=u,probability_warning=WARN)
        q=policy['q'];m=e.merge(s)
        if q:
            report['decision']='submit_probe'
            if emit:report['file']=author(e,sorted(q+policy.get('carrier',[])),q,policy.get('carrier',[]))
        elif m['f1']>e.champ_score+1e-12:
            report['decision']='submit_final'
            if emit:report['file']=author(e,m['ids'],m['ids'],[],'final')
        else:report['decision']='retain_champion'
        if e.daily_used>=2:report['submission_timing']='wait_next_day_and_verify_platform_quota'
        else:report['submission_timing']='verify_platform_remaining_quota_before_upload'
    write(HERE/'verified_recommendation.json',report);return report

def record(pid,aid,score,at,evidence,failed=False):
    # Keep failed attempt recording possible even before the expensive historical audit.
    ledger=read(HERE/'online_scores.json');man=read(HERE/'submission_manifest.json');entries={e['probe_id']:e for e in man['files']}
    require(pid in entries and evidence.strip(),'Unknown probe or missing evidence')
    datetime.fromisoformat(at)
    previous=next((r for r in ledger['records'] if r['attempt_id']==aid),None)
    if previous:
        require(previous['probe_id']==pid and previous.get('score')==score and (previous['status']=='failed')==failed,'Conflicting repeated attempt')
        return {'idempotent':True,'record':previous}
    e=entries[pid];path=HERE/e['file']
    row={'probe_id':pid,'attempt_id':aid,'score':score,'submitted_at':at,'evidence':evidence,'status':'failed' if failed else 'accepted',
        'file':str(path),'sha256':e['sha256'],'quota_cost':1,'recorded_at':now()}
    if not failed:
        try:
            engine=Engine();chk=engine.csv.validate_csv(path,e['all_ids'],e['sha256'])
            tp=infer(score,chk['predictions']);dt=tp-TP0
            require(np.any(engine.sums(e['all_ids'])[engine.support]==dt),'Score conflicts with historical equations')
            carrier=engine.sums(e.get('carrier_ids',[]))[engine.support]
            require(len(set(map(int,carrier)))==1,'Carrier TP not determined')
            row.update(inferred_tp=tp,predictions=chk['predictions'],delta_tp=dt,info_delta=dt-int(carrier[0]),source_class='public_scored')
        except (ValueError,FileNotFoundError) as ex:row.update(status='anomaly',error=str(ex))
    ledger['records'].append(row);write(HERE/'online_scores.json',ledger)
    return {'record':row,'next':'recommend; inspect before uploading'}

def emit_final():
    e=Engine();m=e.merge(tuple(e.support))
    require(e.remaining>0 and not e.blocked,'No quota or unresolved anomaly')
    if m['f1']<=e.champ_score+1e-12 or tuple(m['ids']) in e.accepted_sets:return {'decision':'retain_champion_no_duplicate'}
    out=author(e,m['ids'],m['ids'],[],'final');write(HERE/'verified_final_manifest.json',{**m,'file':out})
    return {'decision':'submit_final','merge':m,'file':out}

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('build','audit','emit-final'):sub.add_parser(name)
    r=sub.add_parser('recommend');r.add_argument('--emit',action='store_true');r.add_argument('--time-limit',type=float,default=1800)
    r=sub.add_parser('record')
    for name in ('probe-id','attempt-id','submitted-at','evidence'):r.add_argument('--'+name,required=True)
    r.add_argument('--score');r.add_argument('--failed',action='store_true')
    r=sub.add_parser('research');r.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.cmd=='build':out=build()
    elif a.cmd=='audit':out=Engine().status()
    elif a.cmd=='recommend':out=recommend(a.emit,a.time_limit)
    elif a.cmd=='record':out=record(a.probe_id,a.attempt_id,a.score,a.submitted_at,a.evidence,a.failed)
    elif a.cmd=='emit-final':out=emit_final()
    else:
        d=HERE.with_name('v156_fn_research')
        if a.run:subprocess.run([sys.executable,'-B',str(d/'verified_research.py')],check=True)
        out=read(d/'verified_status.json') if (d/'verified_status.json').exists() else {'status':'not_started_verified_research'}
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
