"""V160 read-only legacy adapters and campaign-specific arithmetic."""
from pathlib import Path
from fractions import Fraction
from datetime import timedelta
import sys
import time
import math
import json
import itertools
import collections
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
V158=EXP/'v158_eight_attempt_campaign'
V159=EXP/'v159_joint_threshold_campaign'
sys.path.append(str(V158))
from common import (read,write,sha,digest,module,now,lock,require,verify_sources,
                    csv_nodes,apply_actions,validate_file,author,infer,TZ,datetime,key)
from evidence import Equations

G=1044
TARGET=Fraction('0.945')
SEED=20260914

def previous():
    return module('_v160_parent',V158/'campaign.py').Campaign(V158)

def f1(tp,p):return 2*int(tp)/(G+int(p))
def reached(tp,p):return Fraction(2*int(tp),G+int(p))>=TARGET
def probability(a,family):
    if a['pool']=='A':
        # Weak rank odds, conditioned by the exact 17/75 historical equation.
        logit=math.log(17/58)-a['weak_rank']/75+.5
        p=1/(1+math.exp(-logit))
        return np.array([0.,1-p,p])
    names=('catboost','v38_control')
    vectors=[np.array([a['model_probabilities'][n].get(str(v),0.) for v in (-1,0,1)]) for n in names]
    p=vectors[names.index(family)] if family in names else np.sqrt(np.mean(vectors,axis=0))
    lo,hi=a['delta_tp_bounds']['min'],a['delta_tp_bounds']['max']
    p[(np.arange(3)-1<lo)|(np.arange(3)-1>hi)]=0
    require(np.isfinite(p).all() and (p>=0).all() and p.sum()>0,'Missing/invalid prior source')
    return p/p.sum()

def cutoff(seconds):return time.monotonic()+seconds
def within(deadline):require(time.monotonic()<deadline,'Search time limit exceeded; no optimality claim')

def build_catalog(parent):
    frozen=read(V159/'candidate_catalog.json')
    a=[]
    for rank,raw in enumerate(frozen['main'],1):
        row=dict(raw,pool='A',weak_rank=rank)
        lo,hi=parent.eq.bound([row]);row['delta_tp_bounds']=dict(min=lo,max=hi)
        a.append(row)
    b=[dict(raw,pool='B') for raw in frozen['auxiliary']]
    occupied={x['order_id'] for x in a+b}
    pre=read(V158/'research/preflight_catalog.json')
    raw=read(EXP/'v153_complementary_repair_campaign/training_candidates.json')['families']
    maps={name:{key(x):x for x in rows} for name,rows in raw.items()}
    rejected=[r for r in pre['rejected'] if r['reason']=='nonpositive_conditional_utility']
    options=[]
    for n,r in enumerate(rejected):
        k=tuple(r['key']);x=dict(maps['catboost'][k])
        if x['order_id'] in occupied:continue
        try:apply_actions(parent.champ['file'],[x])
        except ValueError:continue
        lo,hi=parent.eq.bound([x])
        if x['kind']=='add' and hi<=0 or x['kind']=='swap' and hi<=0 or x['kind']=='delete' and hi<0:continue
        mp={}
        for family in ('catboost','v38_control'):
            p=np.array([maps[family][k]['probabilities'].get(str(v),0.) if lo<=v<=hi else 0. for v in (-1,0,1)])
            require(p.sum()>0 and np.isfinite(p).all(),'Missing frozen prediction')
            p/=p.sum();mp[family]={str(v):float(p[v+1]) for v in (-1,0,1)}
        p=np.mean([[mp[f][str(v)] for v in (-1,0,1)] for f in mp],axis=0)
        u=float(2*(p[2]-p[0])-.945*x['delta_p'])
        x.update(model_probabilities=mp,delta_tp_bounds=dict(min=lo,max=hi),
                 pool='C',expected_target_utility=u,origin='V158 nonpositive_conditional_utility')
        options.append(x)
        if n%250==0:print('V160 pool C screening',n,len(options),flush=True)
    c=[];used=set(occupied)
    for row in sorted(options,key=lambda x:(-x['expected_target_utility'],key(x))):
        if row['order_id'] in used:continue
        used.add(row['order_id']);c.append(row)
        if len(c)==12:break
    actions=a+b+c
    for i,x in enumerate(actions):
        x['id']=i
        lo,hi=parent.eq.bound([x]);x['delta_tp_bounds']=dict(min=lo,max=hi)
        apply_actions(parent.champ['file'],[x])
        for f in ('catboost','v38_control','mixture_temperature2'):probability(x,f)
    require(len(a)==75 and len(b)==5 and len(c)==12,'Frozen pool sizes changed')
    require(len({x['order_id'] for x in actions})==92,'Overlapping pool orders')
    require(collections.Counter(x['kind'] for x in c)=={'delete':8,'swap':3,'add':1},'Pool C composition changed')
    require(parent.eq.bound(a)==(17,17),'V150 remaining true count changed')
    apply_actions(parent.champ['file'],actions)
    return dict(actions=actions,pools={p:[x['id'] for x in actions if x['pool']==p] for p in 'ABC'},
                main_true_count=17,tail_eligible_orders=len({x['order_id'] for x in options}),
                tail_ranking='descending 2*E(deltaTP)-0.945*deltaP; then order/remove/add',
                probability_class='uncalibrated model priors, not independent evidence')
