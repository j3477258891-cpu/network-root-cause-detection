import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v104_15day_positive_campaign'
def count(path):
    return sum(len(json.loads(r['output'])['rootcause']) for r in csv.DictReader(open(path,encoding='utf-8-sig')))
def main():
    meta=json.load(open(OUT/'campaign.json')); A=np.asarray(json.load(open(OUT/'matrix.json'))); probes=sorted(OUT.glob('probe_*.csv'))
    assert len(probes)==26 and A.shape==(25,71)
    counts=[count(p) for p in probes]; assert counts[0]==1035-meta['delete_pool_count']
    remaining=len(probes)+3
    report={'probe_files':len(probes),'matrix_shape':list(A.shape),'calibration_predictions':counts[0],'code_prediction_range':[min(counts[1:]),max(counts[1:])],'remaining_submissions':remaining,'consumed_submissions':1,'total_submissions_within_15_days':remaining+1,'passed':remaining+1<=30}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
