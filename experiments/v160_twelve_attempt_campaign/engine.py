"""Exact feasible witnesses and measured-group unions. Probabilities are approximate."""
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csc_matrix
from support import *

def sparse(rows,n):
    rr=[];cc=[];vv=[]
    for r,row in enumerate(rows):
        for c,v in row.items():rr.append(r);cc.append(c);vv.append(v)
    return csc_matrix((vv,(rr,cc)),shape=(len(rows),n),dtype=float)

class WitnessModel:
    """Compress unqueried identical columns; candidate node labels remain individual."""
    def __init__(self,eq,actions):
        self.eq=eq;self.actions=actions
        special={eq.index[x['order_id'],x[f]] for x in actions for f in ('add_rid','remove_rid') if x.get(f)}
        patterns=[[] for _ in range(eq.n)]
        for r,row in enumerate(eq.rows):
            for i,v in row.items():patterns[i].append((r,v))
        groups={};caps=[];self.mapping=[]
        for i,p in enumerate(patterns):
            signature=(tuple(p),i if i in special else None)
            if signature not in groups:groups[signature]=len(groups);caps.append(0)
            j=groups[signature];caps[j]+=1;self.mapping.append(j)
        rows=[{} for _ in eq.rows]
        for (pattern,_),j in groups.items():
            for r,v in pattern:rows[r][j]=v
        rhs=list(eq.rhs);self.nodes=len(caps);m=len(actions)
        upper=list(caps)+[1]*(3*m)
        for k,a in enumerate(actions):
            one={self.nodes+3*k+t:1 for t in range(3)}
            rows.append(one);rhs.append(1)
            row={self.nodes+3*k:1,self.nodes+3*k+2:-1}
            for f,sgn in [('add_rid',1),('remove_rid',-1)]:
                if a.get(f):
                    j=self.mapping[eq.index[a['order_id'],a[f]]];row[j]=row.get(j,0)+sgn
            rows.append(row);rhs.append(0)
            lo,hi=a['delta_tp_bounds']['min'],a['delta_tp_bounds']['max']
            for t in range(3):
                if not lo<=t-1<=hi:upper[self.nodes+3*k+t]=0
        self.mat=sparse(rows,len(upper));self.rhs=np.array(rhs)
        self.constraints=LinearConstraint(self.mat,rhs,rhs)
        self.bounds=Bounds(np.zeros(len(upper)),upper)
        self.n=len(upper)

    def draw(self,rng,family):
        cost=np.zeros(self.n)
        for k,a in enumerate(self.actions):
            p=probability(a,family)
            # Factor-wise perturb-and-MAP, deliberately not called posterior sampling.
            cost[self.nodes+3*k:self.nodes+3*k+3]=-np.log(np.maximum(p,1e-12))-rng.gumbel(size=3)
        result=milp(cost,integrality=np.ones(self.n),bounds=self.bounds,
                    constraints=self.constraints,options={'time_limit':30,'mip_rel_gap':0})
        require(result.status==0,'Feasible-world solve incomplete; not a probability proof')
        witness=np.rint(result.x).astype(int)
        require(np.max(abs(self.mat@witness-self.rhs))<1e-6,'Invalid history witness')
        delta=(witness[self.nodes:].reshape(-1,3)*np.array([-1,0,1])).sum(axis=1)
        return delta,witness

def scenarios(c,count=256,seconds=600):
    deadline=cutoff(seconds);model=WitnessModel(c.eq,c.actions);rng=np.random.default_rng(SEED)
    worlds=[];witnesses=[];families=[]
    for family in ('catboost','v38_control','mixture_temperature2'):
        for i in range(count):
            within(deadline);d,w=model.draw(rng,family)
            worlds.append(d);witnesses.append(w);families.append(family)
        print('V160 feasible scenarios',family,count,flush=True)
    return dict(worlds=np.array(worlds,dtype=np.int8),witnesses=np.array(witnesses,dtype=np.int32),
                families=families,method='factor perturb-and-MAP feasible finite ensemble; not calibrated posterior',
                unique_worlds=len({tuple(w) for w in worlds}))

def best_known(groups,actions,base_tp=972,base_p=1045):
    """Exact fractional set packing of disjoint, count-known whole groups."""
    groups=list({tuple(g['ids']):g for g in groups if g['ids']}.values())
    if not groups:return dict(ids=[],delta_tp=0,delta_p=0,tp=base_tp,predictions=base_p,f1=f1(base_tp,base_p))
    n=len(groups);rows=[]
    for a in actions:
        row={j:1 for j,g in enumerate(groups) if a['id'] in g['ids']}
        if row:rows.append(row)
    mat=sparse(rows,n);constraint=LinearConstraint(mat,np.zeros(len(rows)),np.ones(len(rows)))
    dp=np.array([sum(actions[i]['delta_p'] for i in g['ids']) for g in groups])
    dt=np.array([g['delta_tp'] for g in groups]);ratio=f1(base_tp,base_p)
    for _ in range(30):
        r=milp(ratio*dp-2*dt,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
               constraints=constraint,options={'time_limit':30,'mip_rel_gap':0})
        require(r.status==0,'Known-group union unresolved')
        selected=np.flatnonzero(np.rint(r.x));ids=sorted(i for j in selected for i in groups[j]['ids'])
        require(len(ids)==len(set(ids)),'Known groups overlap')
        d=int(dt[selected].sum());p=int(dp[selected].sum());value=f1(base_tp+d,base_p+p)
        if abs(value-ratio)<1e-12:
            return dict(ids=ids,delta_tp=d,delta_p=p,tp=base_tp+d,predictions=base_p+p,f1=value,
                        evidence='exact optimum over disjoint count-known groups')
        ratio=value
    raise ValueError('Known-group fractional optimization did not converge')

def mean_selection(mean,dp,base_tp=972,base_p=1045):
    ratio=f1(base_tp,base_p);mask=np.zeros(len(dp),dtype=bool)
    for _ in range(30):
        mask=2*mean-ratio*dp>1e-12
        value=float(2*(base_tp+mean[mask].sum())/(G+base_p+dp[mask].sum()))
        if abs(value-ratio)<1e-12:break
        ratio=value
    return tuple(np.flatnonzero(mask).tolist())

def final_candidates(worlds,dp,known_ids):
    masks={(),tuple(known_ids),mean_selection(worlds.mean(axis=0),dp)}
    for row in worlds:masks.add(mean_selection(row,dp))
    for subset in np.array_split(worlds,3):
        if len(subset):masks.add(mean_selection(subset.mean(axis=0),dp))
    ordered=sorted(masks,key=lambda x:(len(x),x));matrix=np.zeros((len(ordered),len(dp)),dtype=np.int8)
    for i,ids in enumerate(ordered):matrix[i,list(ids)]=1
    return ordered,matrix

def conditional_finals(c,worlds,known):
    dp=np.array([a['delta_p'] for a in c.actions],dtype=int)
    candidates,masks=final_candidates(worlds,dp,known['ids'])
    scores=2*(972+worlds.astype(np.int16)@masks.T)/(G+1045+masks@dp)[None,:]
    return candidates,masks,scores
