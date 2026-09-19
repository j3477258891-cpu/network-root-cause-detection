import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    meta=json.load(open(ROOT/'experiments/v104_15day_positive_campaign/campaign.json',encoding='utf-8'))
    pa=np.asarray(meta['add_priors']); q=np.asarray(meta['delete_correct_priors']); mapping=json.load(open(ROOT/'experiments/v109_adaptive_portfolio.json',encoding='utf-8'))['mapping']
    rng=np.random.default_rng(20260821); n=500000
    add_true=(rng.random((n,len(pa)))<pa).sum(1); del_true=(rng.random((n,len(q)))<q).astype(np.int8); best=np.zeros(n)
    for add in np.unique(add_true):
        rows=np.flatnonzero(add_true==add); portfolio=mapping[str(int(np.clip(add,15,50)))]
        for item in portfolio:
            idx=np.asarray(item['indices'],dtype=int); k=len(idx); correct=del_true[rows[:,None],idx].sum(1); tp=956+add-(k-correct); pred=1035+add-k
            best[rows]=np.maximum(best[rows],2*tp/(1044+pred))
    report={'version':'v107-v104-independent-prior-sim-3','trials':n,'assumption':'independent correctness priors; not leaderboard evidence','portfolio':'adaptive to decoded add count via v109','mean_best_f1':float(best.mean()),'median_best_f1':float(np.median(best)),'p05':float(np.quantile(best,.05)),'p95':float(np.quantile(best,.95)),'probability_reach_0_945':float(np.mean(best>=.945))}
    (ROOT/'experiments/v104_15day_positive_campaign/offline_simulation.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
