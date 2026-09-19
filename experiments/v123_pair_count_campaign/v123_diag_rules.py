import sys
import numpy as np
from collections import Counter

sys.path.insert(0, r"D:/zgyidong/experiments")
import v123_pair_count_model as V

arrays = V.load_npz()
train_records, test_records = V.load_records()
count_oof, count_test = V.load_count_probs()

train_orders = [o["order_id"] for o in train_records]
train_ptr = arrays["train_alarm_ptr"]
train_labels = arrays["train_labels"].astype(np.int8)
folds = arrays["train_station_folds"].astype(np.int8)
node_scores = {"v11": arrays["train_v11"], "v13": arrays["train_v13"], "v19": arrays["train_v19"]}

# collect all candidates with features and labels (no limit, wider filters)
all_cands = []
ranges = V.build_order_index(train_ptr)
rid_off = {}
for oi, order in enumerate(train_records):
    s = ranges[oi][0]
    rid_off[order["order_id"]] = {a["rid"]: s + k for k, a in enumerate(order.get("alarms", []))}

for fold in range(5):
    fit_orders = {oid for oi, oid in enumerate(train_orders) if int(folds[oi]) != fold}
    ts = V.build_template_stats(train_records, train_labels, train_ptr, train_orders, fit_orders)
    cands = V.build_pair_candidates(node_scores, train_ptr, train_records, {}, None, count_oof, ts, train=True, order_fold=folds, fold_id=fold)
    for c in cands:
        off = rid_off.get(c["order_id"], {})
        ri, ai = off.get(c["remove_rid"]), off.get(c["add_rid"])
        rt = int(train_labels[ri]) if ri is not None else 0
        at = int(train_labels[ai]) if ai is not None else 0
        c["label"] = at - rt
        c["fold"] = fold
        all_cands.append(c)

print("total cands:", len(all_cands))

# Rule A: sort by count_residual ascending (order "over-reported" first)
def eval_rule(cands, key_fn, k=45):
    ordered = sorted(cands, key=key_fn)
    seen = set()
    top = []
    for c in ordered:
        if c["order_id"] in seen:
            continue
        seen.add(c["order_id"])
        top.append(c)
        if len(top) >= k:
            break
    if not top:
        return {"available": len(top), "precision": 0.0, "gain": 0}
    prec = float(np.mean([1 if c["label"] == 1 else 0 for c in top]))
    gain = int(np.sum([c["label"] for c in top]))
    return {"available": len(top), "precision": round(prec, 3), "gain": gain}

f_idx = V.FEATURE_NAMES.index("count_residual")
print("Rule A (residual asc):", eval_rule(all_cands, lambda c: c["features"][f_idx]))
print("Rule B (residual asc, only residual<=0.5):",
      eval_rule([c for c in all_cands if c["features"][f_idx] <= 0.5], lambda c: c["features"][f_idx]))
print("Rule C (residual asc, only residual<=0):",
      eval_rule([c for c in all_cands if c["features"][f_idx] <= 0.0], lambda c: c["features"][f_idx]))
# Rule D: add score high (semantic add stronger)
print("Rule D (d_v11 desc):", eval_rule(all_cands, lambda c: -c["features"][6]))
# Rule E: residual asc + d_v11 desc composite
print("Rule E (residual, then -d_v11):", eval_rule(all_cands, lambda c: (c["features"][f_idx], -c["features"][6])))
# Rule F: remove rank in baseline (later = worse) asc + residual
rank_idx = V.FEATURE_NAMES.index("rem_rank")
print("Rule F (rem_rank desc + residual asc):", eval_rule(all_cands, lambda c: (c["features"][rank_idx], c["features"][f_idx])))

# distribution of residual across labels
for lab in sorted(set(c["label"] for c in all_cands)):
    vals = [c["features"][f_idx] for c in all_cands if c["label"] == lab]
    print("label=%d: n=%d residual mean=%.2f" % (lab, len(vals), np.mean(vals)))
# how many orders have at least one +1 candidate?
pos_orders = {c["order_id"] for c in all_cands if c["label"] == 1}
print("orders with a +1 candidate:", len(pos_orders))
# orders with residual<=0.5 and a +1 candidate
sub = [c for c in all_cands if c["features"][f_idx] <= 0.5]
pos_sub = {c["order_id"] for c in sub if c["label"] == 1}
print("orders with residual<=0.5 and +1 cand:", len(pos_sub), "| sub cands:", len(sub))
