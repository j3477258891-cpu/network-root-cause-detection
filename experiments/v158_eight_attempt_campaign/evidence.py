"""Reconstruct actual score equations; exact integer bounds, never model labels."""
from __future__ import annotations
import copy, collections
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from common import *

def collect():
    system=copy.deepcopy(read(EXP/'v155_typed_repair_campaign/history_evidence_cache.json')['system'])
    index={(v['order_id'],v['rid']):i for i,v in enumerate(system['variables'])}
    prior_attempts=[]
    for folder in ('v153_complementary_repair_campaign','v156_joint_decode_campaign','v157_joint_correction_campaign'):
        directory=EXP/folder
        entries={e['probe_id']:e for e in read(directory/'submission_manifest.json')['files']}
        for r in read(directory/'online_scores.json')['records']:
            prior_attempts.append({**r,'source_ledger':str(directory/'online_scores.json')})
            if r.get('status')!='accepted': continue
            p=Path(r['file']) if r.get('file') else directory/entries[r['probe_id']]['file']
            _,_,nodes=csv_nodes(p,strict=False);tp=infer(str(r['score']),len(nodes))
            require(sha(p)==r['sha256'],'Score artifact changed')
            if any(e['sha256']==r['sha256'] for e in system['equations']): continue
            system['equations'].append(dict(file=str(p),sha256=r['sha256'],score=r['score'],tp=tp,
                predictions=len(nodes),indices=sorted(index[k] for k in nodes),source_class='user_reported_online',
                source_ledger=str(directory/'online_scores.json'),evidence=r.get('evidence'),submitted_at=r.get('submitted_at')))
    expected_orders=set(csv_nodes(EXP/'v156_joint_decode_campaign/v156_final_r04_utf8.csv')[0])
    for e in system['equations']:
        ids,_,nodes=csv_nodes(e['file'],strict=False)
        require(sha(e['file'])==e['sha256'],'Historical hash drift')
        require(set(ids)==expected_orders and len(nodes)==e['predictions'],'Historical P/order mismatch')
        require(infer(f"{float(e['score']):.6f}",len(nodes))==e['tp'],'Historical TP mismatch')
        require(sorted(index[k] for k in nodes)==e['indices'],'Historical indices mismatch')
    system['public_equation_count']=len(system['equations'])
    return system,prior_attempts

class Equations:
    def __init__(self,system,observations=(),time_limit=30):
        self.system=system;self.index={(v['order_id'],v['rid']):i for i,v in enumerate(system['variables'])}
        self.n=len(self.index);self.time_limit=time_limit;self.cache={}
        self.rows=[dict.fromkeys(e['indices'],1) for e in system['equations']]+[dict.fromkeys(range(self.n),1)]
        self.rhs=[e['tp'] for e in system['equations']]+[G]
        for coeff,value in observations:self.rows.append(coeff);self.rhs.append(value)

    def coeff(self,actions):
        result=collections.Counter()
        for a in actions:
            for field,sign in [('add_rid',1),('remove_rid',-1)]:
                if a.get(field):result[self.index[a['order_id'],a[field]]]+=sign
        return {i:v for i,v in result.items() if v}

    def solve(self,objective=None,extra=()):
        objective=objective or {};rows=self.rows+[c for c,v in extra];rhs=self.rhs+[v for c,v in extra]
        patterns=[[] for _ in range(self.n)]
        for r,row in enumerate(rows):
            for col,value in row.items():patterns[col].append((r,value))
        # Objective coefficient participates in grouping; no candidate information is lost.
        groups={};capacities=[];cost=[]
        for col,pattern in enumerate(patterns):
            k=(tuple(pattern),objective.get(col,0))
            if k not in groups:groups[k]=len(groups);capacities.append(0);cost.append(k[1])
            capacities[groups[k]]+=1
        rr=[];cc=[];vv=[]
        for (pattern,_),j in groups.items():
            for r,value in pattern:rr.append(r);cc.append(j);vv.append(value)
        mat=csc_matrix((vv,(rr,cc)),shape=(len(rows),len(groups)),dtype=float)
        result=milp(np.array(cost,dtype=float),integrality=np.ones(len(groups)),
            bounds=Bounds(np.zeros(len(groups)),capacities),constraints=LinearConstraint(mat,rhs,rhs),
            options={'time_limit':self.time_limit,'mip_rel_gap':0})
        if result.status==2:return None
        require(result.status==0,'Integer solver unresolved; timeout is not a proof')
        require(np.max(np.abs(mat@np.rint(result.x)-rhs))<1e-6,'Invalid integer witness')
        return result

    def bound(self,actions):
        coeff=self.coeff(actions);k=tuple(sorted(coeff.items()))
        if k not in self.cache:
            lo=self.solve(coeff);hi=self.solve({i:-v for i,v in coeff.items()})
            require(lo is not None and hi is not None,'Inconsistent history')
            self.cache[k]=(round(lo.fun),round(-hi.fun))
        return self.cache[k]

    def possible(self,actions,value):return self.solve(extra=[(self.coeff(actions),value)]) is not None

    def oracle(self,actions,base_tp,base_p):
        """Dinkelbach MILP over true labels and the selected action subset."""
        m=len(actions);n=self.n+3*m
        rows=[dict(r) for r in self.rows];lo=list(self.rhs);hi=list(self.rhs)
        def add(row,l,h):rows.append(row);lo.append(l);hi.append(h)
        # z=x*y, independently for added and removed node of every action.
        for k,a in enumerate(actions):
            x=self.n+k
            for offset,field in [(m,'add_rid'),(2*m,'remove_rid')]:
                z=self.n+offset+k
                if not a.get(field):add({z:1},0,0);continue
                y=self.index[a['order_id'],a[field]]
                add({z:1,x:-1},-np.inf,0);add({z:1,y:-1},-np.inf,0)
                add({z:1,x:-1,y:-1},-1,np.inf)
        rr=[];cc=[];vv=[]
        for r,row in enumerate(rows):
            for j,v in row.items():rr.append(r);cc.append(j);vv.append(v)
        mat=csc_matrix((vv,(rr,cc)),shape=(len(rows),n),dtype=float)
        ratio=f1(base_tp,base_p)
        for _ in range(30):
            cost=np.zeros(n);cost[self.n:self.n+m]=ratio*np.array([a['delta_p'] for a in actions])
            cost[self.n+m:self.n+2*m]=-2;cost[self.n+2*m:]=2
            r=milp(cost,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                constraints=LinearConstraint(mat,lo,hi),options={'time_limit':self.time_limit,'mip_rel_gap':0})
            require(r.status==0,'Oracle timeout/infeasibility does not pass target gate')
            z=np.rint(r.x).astype(int);require(np.all(mat@z>=np.array(lo)-1e-6) and np.all(mat@z<=np.array(hi)+1e-6),'Invalid oracle witness')
            p=int(base_p+sum(a['delta_p']*z[self.n+k] for k,a in enumerate(actions)))
            tp=base_tp+int(z[self.n+m:self.n+2*m].sum()-z[self.n+2*m:].sum())
            value=float(f1(tp,p))
            if abs(value-ratio)<1e-10:return dict(f1_upper=value,tp=tp,predictions=p,source_class='optimistic_integer_bound_not_forecast')
            ratio=value
        raise ValueError('Oracle did not converge')
