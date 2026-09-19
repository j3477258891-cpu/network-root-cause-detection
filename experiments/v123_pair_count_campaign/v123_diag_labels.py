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

all_labels = Counter()
rem_is_tp = Counter()
res_by_label = {}
d_by_label = {}
pairs_total = 0
for fold in range(5):
    fit_orders = {oid for oi, oid in enumerate(train_orders) if int(folds[oi]) != fold}
    ts = V.build_template_stats(train_records, train_labels, train_ptr, train_orders, fit_orders)
    cands = V.build_pair_candidates(node_scores, train_ptr, train_records, {}, None, count_oof, ts, train=True, order_fold=folds, fold_id=fold)
    ranges = V.build_order_index(train_ptr)
    rid_off = {}
    for oi, order in enumerate(train_records):
        s = ranges[oi][0]
        rid_off[order["order_id"]] = {a["rid"]: s + k for k, a in enumerate(order.get("alarms", []))}
    for c in cands:
        off = rid_off.get(c["order_id"], {})
        ri, ai = off.get(c["remove_rid"]), off.get(c["add_rid"])
        rt = int(train_labels[ri]) if ri is not None else 0
        at = int(train_labels[ai]) if ai is not None else 0
        lab = at - rt
        all_labels[lab] += 1
        rem_is_tp[rt] += 1
        res_by_label.setdefault(lab, []).append(c["features"][12])
        d_by_label.setdefault(lab, []).append(c["features"][6])
        pairs_total += 1

print("total pairs:", pairs_total)
print("label dist (+1 gain / 0 / -1):", dict(all_labels))
print("remove is TP:", rem_is_tp.get(1, 0), "| remove is FP:", rem_is_tp.get(0, 0))
for lab in sorted(res_by_label):
    ress = res_by_label[lab]
    print("  label=%d: n=%d count_residual mean=%.2f d_v11 mean=%.4f" % (lab, len(ress), np.mean(ress), np.mean(d_by_label[lab])))
