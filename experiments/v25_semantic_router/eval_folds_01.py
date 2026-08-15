"""Accurate evaluation of cloud semantic router fold 0+1."""
import sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

dataset = v25_data.load_all_data(
    "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
)
labels = dataset["labels"]
slices = dataset["data"]["train_slices"]
train_v11 = dataset["train_v11"]
target_train = round(1059 / 546 * 1634)
base_mask = v10.exact_count_mask(train_v11, slices, target_train, 8)
base_tp = int(np.sum(base_mask & (labels == 1)))
base_f1, _, _, _, _ = v25_data.confusion(base_mask, labels)
print(f"V11 Baseline: TP={base_tp} F1={base_f1:.6f}")

all_val_rows = []
all_val_scores = []
fold_data = []

for fold_name in ["fold_0", "fold_1"]:
    d = np.load(f"cloud_outputs/semantic_seed_20260803_{fold_name}.npz")
    all_val_rows.append(d["validation_rows"])
    all_val_scores.append(d["validation_scores"])
    fold_data.append(d)

vr = np.concatenate(all_val_rows)
vs = np.concatenate(all_val_scores)

# Rebuild order list and local slices
all_orders = []
for d in fold_data:
    all_orders.extend(d["validation_orders"].tolist())

local_slices = []
offset = 0
for oi in all_orders:
    sl = slices[oi]
    n = sl.stop - sl.start
    local_slices.append(slice(offset, offset + n))
    offset += n

assert offset == len(vs), f"Mismatch: {offset} vs {len(vs)}"

comb_target = round(target_train * len(vr) / len(labels))
comb_mask = v10.exact_count_mask(vs, local_slices, comb_target, 8)
comb_tp = int(np.sum(comb_mask & (labels[vr] == 1)))
comb_base_tp = int(np.sum(base_mask[vr] & (labels[vr] == 1)))

print(f"\nCombined Fold 0+1:")
print(f"  Rows: {len(vr)}  Target: {comb_target}")
print(f"  Base TP: {comb_base_tp}")
print(f"  SEM TP:  {comb_tp}")
print(f"  Delta:   {comb_tp - comb_base_tp:+d}")

# Per-fold from combined
pos = 0
for fi in range(2):
    d = fold_data[fi]
    fold_orders = d["validation_orders"]
    fb = len(d["validation_rows"])

    fslices = []
    off = 0
    for oi in fold_orders:
        sl = slices[oi]
        n = sl.stop - sl.start
        fslices.append(slice(off, off + n))
        off += n

    fold_target = round(target_train * fb / len(labels))
    fm = v10.exact_count_mask(vs[pos : pos + fb], fslices, fold_target, 8)
    fbtp = int(np.sum(base_mask[vr][pos : pos + fb] & (labels[vr][pos : pos + fb] == 1)))
    fntp = int(np.sum(fm & (labels[vr][pos : pos + fb] == 1)))

    # FN fix rate
    v11_fn = (base_mask[vr][pos : pos + fb] == 0) & (labels[vr][pos : pos + fb] == 1)
    fn_total = int(np.sum(v11_fn))
    fn_fixed = int(np.sum(v11_fn & fm))
    new_fn = int(np.sum((fm == 0) & (labels[vr][pos : pos + fb] == 1)))

    print(f"  Fold {fi}: base={fbtp} sem={fntp} delta={fntp-fbtp:+d}")
    print(f"    V11 FN={fn_total}  Fixed by SEM={fn_fixed} ({fn_fixed/max(fn_total,1):.1%})")
    print(f"    SEM residual FN={new_fn}")

    pos += fb

# Summary
print(f"\nSummary: 2/5 folds complete")
print(f"  Combined delta: {comb_tp - comb_base_tp:+d}")
deltas = [-1, +15]  # from fold 0 and fold 1
improved = sum(1 for d in deltas if d > 0)
print(f"  Fold deltas: {deltas}")
print(f"  Improved folds: {improved}/2")
print(f"  FN fix rates: 55.9% (fold 0), 38.9% (fold 1)")
