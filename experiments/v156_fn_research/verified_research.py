"""Nested, connected-group honest FN study. No public labels or uploads.

Old research.py/results.json are preserved, but are not valid gate evidence.
Uses training-label totals, four budgets selected only inside each outer fold,
matched-budget historical controls and paired station-cluster bootstrap.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
V155=HERE.with_name('v155_typed_repair_campaign')
sys.path.insert(0,str(V155))
from training import Trainer, t152

SEED=20260910
BUDGETS=(8,16,32,64)
FAMILIES=('catboost','v38_control','fn_base','fn_disagreement','fn_graph','fn_combined')
CHALLENGERS=FAMILIES[2:]
GRID={f:((0,) if f=='v38_control' else (4,6)) for f in FAMILIES}

def now():return datetime.now(timezone(timedelta(hours=8)))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def require(ok,msg):
    if not ok:raise ValueError(msg)

class ResearchTrainer(Trainer):
    def __init__(self,out,hours):
        super().__init__(out,hours,False)
        # No supervised saved logits/embeddings enter this experiment.
        self.extra={}
        for split in ('train','test'):
            self.extra[split]=np.nan_to_num(np.column_stack([self.arrays[split+'_alarm_x'],
                self.arrays[split+'_path_node_type'],self.arrays[split+'_path_edge_type'],
                self.arrays[split+'_path_length']])).astype('float32')
    def additions(self,split,orders,scores,selected=None):
        self.check_time();ptr=self.arrays[split+'_alarm_ptr'];records=self.records[split]
        selected=self.masks(split,orders,scores) if selected is None else selected
        compact=np.column_stack([self.x[split][:,:97],self.x[split][:,-192:]])
        feats=[];meta=[];ys=[];dis=[];graph=[];globalrows=[]
        for oi in orders:
            oi=int(oi);rows=np.arange(ptr[oi],ptr[oi+1]);count=int(selected[rows].sum())
            if count>=8:continue
            for r in rows[~selected[rows]]:
                a=records[oi]['alarms'][int(r-ptr[oi])];av=compact[r];pa=float(scores[r].mean())
                feats.append(np.r_[av,np.zeros_like(av),av,[0,1,count,len(rows),pa,0,pa,scores[rows].mean(),scores[rows].std(),float(a.get('label')=='TargetAlarm'),0]])
                meta.append({'order_index':oi,'order_id':records[oi]['order_id'],'kind':'add','delta_p':1,
                    'add_rid':a['rid'],'remove_rid':None,'node':{'@rid':a['rid'],**{k:a['source'].get(k,'') for k in ('title','location','reason')}}})
                ys.append(int(self.y[r]) if split=='train' else 0);globalrows.append(int(r))
                dis.append([scores[r,0],scores[r,1],abs(scores[r,0]-scores[r,1]),min(scores[r]),max(scores[r])])
                graph.append(self.extra[split][r])
        return {'x':np.asarray(feats,dtype='float32'),'meta':meta,'y':np.array(ys),'mask':selected,
            'orders':list(map(int,orders)),'dis':np.array(dis,dtype='float32'),'graph':np.array(graph,dtype='float32'),
            'rows':np.array(globalrows)}

def features(b,f):
    a=[b['x']]
    if f in ('fn_disagreement','fn_combined'):a.append(b['dis'])
    if f in ('fn_graph','fn_combined'):a.append(b['graph'])
    return np.column_stack(a)

def fit_scores(t,f,param,fit_add,fit_legacy,valid):
    t.check_time()
    if f in ('catboost','v38_control'):
        p,_=t.fit_predict(f,param,fit_legacy,valid);return p[:,2]
    y=fit_add['y']
    if len(np.unique(y))<2:return np.full(len(valid['y']),float(y[0]))
    model=CatBoostClassifier(iterations=240,depth=param,learning_rate=.05,l2_leaf_reg=5,
        loss_function='Logloss',thread_count=4,random_seed=SEED,verbose=False,allow_writing_files=False)
    counts={}
    for a in fit_add['meta']:counts[a['order_index']]=counts.get(a['order_index'],0)+1
    w=np.array([1/counts[a['order_index']] for a in fit_add['meta']])
    model.fit(features(fit_add,f),y,sample_weight=w)
    return model.predict_proba(features(valid,f))[:,1]

def calibrate(p,y):
    logits=np.log(np.clip(p,1e-8,1-1e-8)/np.clip(1-p,1e-8,1))
    def loss(temp):
        q=1/(1+np.exp(-np.clip(logits/temp,-40,40)))
        return float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))
    return min((.5,.75,1.,1.5,2.,3.),key=lambda x:(loss(x),x))
def temperature(p,t):
    z=np.log(np.clip(p,1e-8,1-1e-8)/np.clip(1-p,1e-8,1));return 1/(1+np.exp(-np.clip(z/t,-40,40)))

def metrics(t,b,p,budget):
    # One action/order. Selection never reads validation labels.
    n=max(0,round(budget*len(b['orders'])/546));used=set();chosen=[]
    for i in sorted(range(len(p)),key=lambda i:(-p[i],b['meta'][i]['order_id'],b['meta'][i]['add_rid'])):
        if len(chosen)>=n:break
        a=b['meta'][i]
        if a['order_index'] in used or p[i]<=971/2093:continue
        used.add(a['order_index']);chosen.append(i)
    stats=[];ptr=t.arrays['train_alarm_ptr'];chosen_by={b['meta'][i]['order_index']:i for i in chosen}
    for oi in b['orders']:
        rows=np.arange(ptr[oi],ptr[oi+1]);mask=b['mask'][rows]
        i=chosen_by.get(oi)
        stats.append([oi,int(t.y[rows].sum()),int(mask.sum()),int(t.y[rows][mask].sum()),int(i is not None),int(b['y'][i]) if i is not None else 0])
    return np.array(stats,dtype=np.int64)
def gain(stats):
    if not len(stats):return 0.
    g,p,tp,dp,dt=np.asarray(stats)[:,1:].sum(axis=0)
    return float(2*(tp+dt)/(g+p+dp)-2*tp/(g+p))

def clusters(records):
    parent=list(range(len(records)));owner={}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for oi,r in enumerate(records):
        for station in r['station_ids']:
            if station in owner:parent[find(oi)]=find(owner[station])
            else:owner[station]=oi
    return [find(i) for i in range(len(records))]

def bootstrap(candidate,control,station_clusters,iterations=2000):
    require(np.array_equal(candidate[:,0],control[:,0]),'Paired order alignment mismatch')
    c={};b={}
    for x,y in zip(candidate,control):
        key=station_clusters[int(x[0])];c[key]=c.get(key,np.zeros(5,dtype=int))+x[1:];b[key]=b.get(key,np.zeros(5,dtype=int))+y[1:]
    keys=sorted(c);a=np.array([c[k] for k in keys]);z=np.array([b[k] for k in keys]);rng=np.random.default_rng(SEED)
    absolute=[];relative=[]
    def calc(v):
        g,p,tp,dp,dt=v;return 2*(tp+dt)/(g+p+dp)-2*tp/(g+p)
    for _ in range(iterations):
        pick=rng.integers(len(keys),size=len(keys));av=calc(a[pick].sum(axis=0));bv=calc(z[pick].sum(axis=0));absolute.append(av);relative.append(av-bv)
    return {'iterations':iterations,'station_clusters':len(keys),'absolute_lower95':float(np.quantile(absolute,.025)),
        'relative_lower95':float(np.quantile(relative,.025)),
        'absolute_lower_familywise':float(np.quantile(absolute,.025/len(CHALLENGERS))),
        'relative_lower_familywise':float(np.quantile(relative,.025/len(CHALLENGERS)))}

def run():
    win=HERE/'verified_window.json'
    if win.exists():window=read(win)
    else:
        # Prior exploratory work began at directory creation; do not silently reset 24h.
        started=datetime.fromtimestamp((HERE/'research.py').stat().st_ctime,timezone(timedelta(hours=8)))
        started=min(started,now())
        window={'started_at':started.isoformat(),'deadline_at':(started+timedelta(hours=24)).isoformat(),
            'start_source':'earliest existing research script creation; conservative continuation'};write(win,window)
    deadline=datetime.fromisoformat(window['deadline_at']).timestamp();require(time.time()<deadline,'Original 24-hour research window expired')
    out=HERE/'verified_run';out.mkdir(exist_ok=True)
    t=ResearchTrainer(out,(deadline-time.time())/3600);t.deadline=deadline
    signature=t152.digest({'sources':t.hashes,'script':t152.sha(Path(__file__)),'protocol':'connected-5-inner2-all-unselected-v1'})
    report={'source_class':'nested_connected_group_oof','complete':False,'window':window,'signature':signature,
        'input_hashes':t.hashes,'folds':[],'gate_passed':False,'quota_cost':0,
        'excluded_features':['saved V11/V13/V19 logits: outer-split lineage not verified','V25 2-fold supervised embeddings: not eligible'],
        'legacy_result_invalid':['training F1 used test G=1044','missing complete bootstrap gate','no nested budget selection'],
        'baseline_note':'cross-fitted surrogate at p03 prediction density, not reproduction of online champion labels'}
    status={'state':'running','pid':__import__('os').getpid(),'started_at':window['started_at'],'deadline_at':window['deadline_at'],'folds_completed':0,'gate_passed':False,'quota_cost':0}
    write(HERE/'verified_status.json',status)
    for outer in range(5):
        cp=out/f'fold_{outer}.json'
        if cp.exists():
            fold=read(cp);require(fold['signature']==signature,'Research checkpoint drift');report['folds'].append(fold);continue
        fit_orders=np.flatnonzero(t.folds!=outer);valid_orders=np.flatnonzero(t.folds==outer)
        print(f'Honest FN outer fold {outer+1}/5',flush=True)
        inner={f:{p:[] for p in GRID[f]} for f in FAMILIES}
        for j,(fit,valid) in enumerate(t152.group_splits(fit_orders,t.folds)):
            scores,_=t.crossfit_nodes(fit,valid)
            fa=t.additions('train',fit,scores);fl=t.actions('train',fit,scores);va=t.additions('train',valid,scores)
            for fam in FAMILIES:
                for param in GRID[fam]:
                    p=fit_scores(t,fam,param,fa,fl,va)
                    inner[fam][param].append({'p':p,'y':va['y'],'b':va})
            print(f'  inner {j+1}/2 complete',flush=True)
        choices={};inner_gain={}
        for fam in FAMILIES:
            inner_gain[fam]={}
            for param,entries in inner[fam].items():
                temp=calibrate(np.concatenate([e['p'] for e in entries]),np.concatenate([e['y'] for e in entries]))
                for k in BUDGETS:
                    stats=np.concatenate([metrics(t,e['b'],temperature(e['p'],temp),k) for e in entries])
                    inner_gain[fam][param,k]=(gain(stats),temp)
            param,k=max(inner_gain[fam],key=lambda pk:(inner_gain[fam][pk][0],-pk[1],-pk[0]))
            choices[fam]={'param':param,'budget':k,'temperature':inner_gain[fam][param,k][1]}
        scores,_=t.crossfit_nodes(fit_orders,valid_orders)
        fa=t.additions('train',fit_orders,scores);fl=t.actions('train',fit_orders,scores);va=t.additions('train',valid_orders,scores)
        predictions={}; outerstats={}
        for fam in FAMILIES:
            # Train parameters selected inside, including matched-budget controls.
            needed={choices[fam]['param']}
            if fam in FAMILIES[:2]:
                needed.update(max(GRID[fam],key=lambda p:inner_gain[fam][p,k][0]) for k in BUDGETS)
            for param in needed:predictions[fam,param]=fit_scores(t,fam,param,fa,fl,va)
            outerstats[fam]={}
            for k in BUDGETS:
                param=max(GRID[fam],key=lambda p:(inner_gain[fam][p,k][0],-p)) if fam in FAMILIES[:2] else choices[fam]['param']
                temp=inner_gain[fam][param,k][1]
                outerstats[fam][str(k)]=metrics(t,va,temperature(predictions[fam,param],temp),k).tolist()
        matched={}
        for fam in CHALLENGERS:
            k=choices[fam]['budget']
            ctrl=max(FAMILIES[:2],key=lambda c:max(inner_gain[c][p,k][0] for p in GRID[c]))
            matched[fam]=ctrl
        fold={'fold':outer,'signature':signature,'choices':choices,'matched_controls':matched,'stats':outerstats,
            'fit_orders':list(map(int,fit_orders)),'valid_orders':list(map(int,valid_orders))}
        write(cp,fold);report['folds'].append(fold)
        status.update(folds_completed=len(report['folds']),updated_at=now().isoformat());write(HERE/'verified_status.json',status)
        write(out/'progress.json',report)
        print('  outer completed '+str(outer+1),flush=True)
    summaries={};station=clusters(t.records['train'])
    for fam in CHALLENGERS:
        aa=[];bb=[];fold_gains=[];ctrl_gains=[]
        for fold in report['folds']:
            k=str(fold['choices'][fam]['budget']);ctrl=fold['matched_controls'][fam]
            a=np.array(fold['stats'][fam][k]);b=np.array(fold['stats'][ctrl][k]);aa.append(a);bb.append(b);fold_gains.append(gain(a));ctrl_gains.append(gain(b))
        a=np.concatenate(aa);b=np.concatenate(bb);idx=np.argsort(a[:,0]);a=a[idx];b=b[idx]
        boot=bootstrap(a,b,station)
        passed=(sum(x>0 for x in fold_gains)>=3 and gain(a)>gain(b) and min(fold_gains)>=min(ctrl_gains)
            and boot['absolute_lower95']>0 and boot['relative_lower95']>0
            and boot['absolute_lower_familywise']>0 and boot['relative_lower_familywise']>0)
        summaries[fam]={'gain':gain(a),'control_gain':gain(b),'positive_folds':sum(x>0 for x in fold_gains),
            'worst_fold_gain':min(fold_gains),'control_worst_fold_gain':min(ctrl_gains),'bootstrap':boot,'passed':passed}
    eligible=[f for f in CHALLENGERS if summaries[f]['passed']]
    report.update(complete=True,summary=summaries,eligible=eligible,gate_passed=bool(eligible),completed_at=now().isoformat(),
        decision='eligible_for_full_fit_and_equation_screen_not_submission' if eligible else 'no_new_group_return_quota_to_decoding')
    # No speculative test CSV: full-fit/ranking/equation screen is a separate gate.
    write(HERE/'verified_results.json',report)
    status.update(state='completed',folds_completed=5,gate_passed=bool(eligible),decision=report['decision'],updated_at=now().isoformat())
    write(HERE/'verified_status.json',status)
    print(json.dumps({'summary':summaries,'decision':report['decision']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):run()
    except Exception as exc:
        p=HERE/'verified_status.json';s=read(p) if p.exists() else {}
        s.update(state='failed_or_deadline',error=str(exc),gate_passed=False,updated_at=now().isoformat());write(p,s)
        raise
