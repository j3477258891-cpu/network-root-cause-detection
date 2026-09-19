import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def solver(matrix, probs, trials=30000, seed=1):
    m=np.asarray(matrix,dtype=np.int8); n=m.shape[1]; cut=n//2; left=m[:,:cut]; right=m[:,cut:]
    # map right subset-sums to all masks; collision lists are retained
    d={}
    for mask in range(1<<right.shape[1]):
        bits=((mask>>np.arange(right.shape[1]))&1).astype(np.int8); key=tuple((right@bits).tolist()); d.setdefault(key,[]).append(mask)
    rng=np.random.default_rng(seed); success=0; exact_pos=0; avg_jacc=0; amb=0
    for _ in range(trials):
        y=(rng.random(n)<probs).astype(np.int8); target=m@y; candidates=[]
        for mask in range(1<<cut):
            bits=((mask>>np.arange(cut))&1).astype(np.int8); key=tuple((target-left@bits).tolist());
            for rm in d.get(key,[]):
                rbits=((rm>>np.arange(right.shape[1]))&1).astype(np.int8); z=np.r_[bits,rbits]; ll=float((z*np.log(np.maximum(probs,1e-6))+(1-z)*np.log(np.maximum(1-probs,1e-6))).sum()); candidates.append((ll,z))
        amb+=len(candidates)>1
        z=max(candidates,key=lambda x:x[0])[1]; success+=int(np.array_equal(z,y)); exact_pos+=int((z==y).sum()); avg_jacc+=int((z&y).sum())/max(1,int((z|y).sum()))
    return {'matrix':m.shape,'trials':trials,'map_exact':success/trials,'bit_accuracy':exact_pos/(trials*n),'jaccard':avg_jacc/trials,'ambiguous_rate':amb/trials}
def main():
    m=np.array(json.load(open(ROOT/'experiments/v73_compact_detecting/basis_15x25.json')))
    j=json.load(open(ROOT/'experiments/v75_dense_block_campaign/report.json',encoding='utf-8')); p=np.array(j['correctness_probabilities']); a=j['candidate_actions']; add=np.array([bool(x['add_rids']) for x in a]); idx=np.flatnonzero(add)[np.argsort(-p[add])]
    # evaluate 25 best additions and one-row deletions from proven 15x25 basis
    probs=p[idx[:25]]
    for r in range(15): print(r,solver(np.delete(m,r,0),probs,1000,100+r))
if __name__=='__main__':main()
