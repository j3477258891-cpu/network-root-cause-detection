import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return {r['order_id']:{x['@rid'] for x in json.loads(r['output'])['rootcause']} for r in csv.DictReader(f)}
base=load(ROOT/'experiments/v60_combined_checkpoint/highest_verified_combined.csv')
bad=set()
for raw in ['experiments/v118_safe_positive_campaign/probe_00_calibration.csv','experiments/v119_joint_campaign/probe_01.csv','experiments/submissions/distance1_swap_equation_filtered.csv']:
    p=ROOT/raw
    if not p.exists(): continue
    cur=load(p)
    bad |= {(o,r) for o,rs in base.items() for r in rs-cur.get(o,set())}
rows=list(csv.DictReader((ROOT/'experiments/v16/v16_test_swap_catalog.csv').open(encoding='utf-8')))
rows.sort(key=lambda x:float(x['seed_min_margin']),reverse=True)
for i,x in enumerate(rows[:50],1):
    print(i,x['order_id'],'RISK' if (x['order_id'],x['removed_rid']) in bad else 'ok',x['seed_min_margin'],x['removed_rid'],x['added_rid'])
