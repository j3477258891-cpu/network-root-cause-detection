"""Audit feasible/unique-order swap curves from saved OOF predictions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(r"D:\zgyidong")
sys.path.insert(0,str(ROOT))
from experiments.research_pairwise_error_model import (
    DATA, TRAIN_DIR, CURRENT, v10, read_submission, baseline_mask_for_orders,
    exact_mask, swap_actions, eval_actions, greedy_action_curve,
)
OUT=ROOT/"experiments/research_pairwise_error_model"
def main():
    a=dict(np.load(DATA,allow_pickle=False)); orders=v10.load_orders(TRAIN_DIR,True)
    ptr=a['train_alarm_ptr']; labels=a['train_labels'].astype(np.int8)
    base=exact_mask(a['train_v11'],ptr,round(1035/546*1634))
    base_tp=int((base & (labels==1)).sum())
    out={}
    for p in sorted(OUT.glob('*_oof.npy')):
        name=p.stem[:-4] if p.stem.endswith('_oof') else p.stem
        score=np.load(p)
        acts=eval_actions(swap_actions(orders,ptr,base,score),labels,ptr,base)
        for mode in ('feasible','unique'):
            seq=greedy_action_curve(acts,labels,ptr,one_per_order=(mode=='unique'),limit=200)
            curves={}
            for k in (5,10,20,30,40,60,100):
                ss=seq[:k]; curves[str(k)]={'delta':int(sum(x['true_delta'] for x in ss)), 'pos':int(sum(x['true_delta']>0 for x in ss)), 'neg':int(sum(x['true_delta']<0 for x in ss))}
            out[f'{name}_{mode}']=curves
    report={'base_tp':base_tp,'base_p':int(base.sum()),'curves':out}
    (OUT/'saved_curve_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
