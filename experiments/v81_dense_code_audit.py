"""Compare dense 19-row MAP codes against the sparse V58 prefix."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong"); EXP=ROOT/"experiments"
sys.path.insert(0,str(EXP)); sys.path.insert(0,str(ROOT/".deps"))
from record_v58_result import map_assignment
BASE_TP,BASE_P,TRUE_ROOTS=956,1035,1044

def run(matrix, probs, actions, trials, seed, limit):
    rng=np.random.default_rng(seed); rows=[]
    add=np.asarray([bool(a["add_rids"]) for a in actions])
    for _ in range(trials):
        truth=(rng.random(len(probs))<probs).astype(np.int8)
        ass=map_assignment(matrix, matrix@truth, probs, time_limit=limit)
        if ass is None: continue
        tp=BASE_TP+int(np.sum(ass&truth&add))-int(np.sum(ass&(1-truth)&(~add)))
        pd=int(np.sum(ass&add))-int(np.sum(ass&(~add)))
        rows.append((2*tp/(TRUE_ROOTS+BASE_P+pd),float(np.mean(ass!=truth))))
    a=np.asarray(rows)
    return {"solved":len(a),"mean":float(a[:,0].mean()),"p05":float(np.quantile(a[:,0],.05)),"reach":float(np.mean(a[:,0]>=.945)),"label_error":float(a[:,1].mean())}

def main():
    report=json.loads((EXP/"v58_coded_campaign/report.json").read_text(encoding="utf8"))
    matrix=np.asarray(report["matrix"],dtype=np.float64); probs=np.asarray(report["correctness_probabilities"],dtype=np.float64); actions=report["candidate_actions"]
    rng=np.random.default_rng(20260820); n=len(actions)
    dense=np.zeros((19,n),dtype=np.float64)
    for i in range(19):
        dense[i]=rng.integers(0,2,size=n)
        # force balanced row weight
        while not (45<=dense[i].sum()<=85): dense[i]=rng.integers(0,2,size=n)
    candidates={"v58_rows_1_19":matrix[1:20],"v58_rows_1_20":matrix[1:21],"random_dense":dense}
    out={k:run(v,probs,actions,12,20260820,5) for k,v in candidates.items()}
    path=EXP/"v81_dense_code_audit";path.mkdir(exist_ok=True);(path/"report.json").write_text(json.dumps(out,indent=2),encoding="utf8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
