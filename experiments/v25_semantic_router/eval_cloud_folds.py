"""Quick evaluation of cloud fold-0 and fold-1 outputs."""
import sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

# Load V11 baseline
dataset = v25_data.load_all_data("D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11")
labels = dataset["labels"]
slices = dataset["data"]["train_slices"]
folds_arr = dataset["folds"]
train_v11 = dataset["train_v11"]
target_train = round(1059 / 546 * 1634)
base_mask = v10.exact_count_mask(train_v11, slices, target_train, 8)
base_tp = int(np.sum(base_mask & (labels == 1)))
base_f1, _, _, _, _ = v25_data.confusion(base_mask, labels)
print(f"V11 Baseline: TP={base_tp} F1={base_f1:.6f}")

for fold_idx, fold_name in [(0, "fold_0"), (1, "fold_1")]:
    d = np.load(f"cloud_outputs/semantic_seed_20260803_{fold_name}.npz")
    
    val_orders = d["validation_orders"]
    val_rows = d["validation_rows"]
    val_scores = d["validation_scores"]
    val_counts = d["validation_counts"]
    
    fold_labels = labels[val_rows]
    fold_base = base_mask[val_rows]
    fold_v11 = train_v11[val_rows]
    
    # Fixed-K evaluation with semantic scores
    fold_target = round(target_train * len(val_rows) / len(labels))
    
    # Build local slices for the validation orders
    local_slices = []
    offset = 0
    for oi in val_orders:
        n_alarms = len(dataset["train_orders"][oi]["alarms"])
        local_slices.append(slice(offset, offset + n_alarms))
        offset += n_alarms
    
    sem_mask = v10.exact_count_mask(val_scores, local_slices, fold_target, 8)
    sem_tp = int(np.sum(sem_mask & (fold_labels == 1)))
    sem_delta = sem_tp - int(np.sum(fold_base & (fold_labels == 1)))
    
    # Count head accuracy
    count_pred = np.argmax(val_counts, axis=1) + 1  # 1-8
    actual_counts = np.array([int(np.sum(fold_base[s])) for s in local_slices])
    count_acc = np.mean(count_pred == actual_counts)
    
    # V11 FN fix rate
    v11_fn = (fold_base == 0) & (fold_labels == 1)
    sem_fixed = int(np.sum(v11_fn & sem_mask))
    total_fn = int(np.sum(v11_fn))
    
    print(f"\n--- Fold {fold_idx} ---")
    print(f"  Orders: {len(val_orders)}  Rows: {len(val_rows)}  Target: {fold_target}")
    print(f"  SEM TP delta: {sem_delta:+d}")
    print(f"  V11 FN: {total_fn}  SEM fixed: {sem_fixed} ({sem_fixed/max(total_fn,1):.1%})")
    print(f"  Count accuracy: {count_acc:.2%}")
    print(f"  Score stats: mean={val_scores.mean():.4f} std={val_scores.std():.4f}")

# Also evaluate combined fold 0+1
print(f"\n{'='*50}")
print("Combined Fold 0+1:")
for fold_idx, fold_name in [(0, "fold_0"), (1, "fold_1")]:
    d = np.load(f"cloud_outputs/semantic_seed_20260803_{fold_name}.npz")
    if fold_idx == 0:
        all_rows = d["validation_rows"]
        all_scores = d["validation_scores"]
        all_labels = labels[d["validation_rows"]]
        all_base = base_mask[d["validation_rows"]]
        all_fold_rows = [(0, d["validation_rows"])]
        all_fold_orders = [d["validation_orders"]]
    else:
        vr = d["validation_rows"]
        all_rows = np.concatenate([all_rows, vr])
        all_scores = np.concatenate([all_scores, d["validation_scores"]])
        all_labels = np.concatenate([all_labels, labels[vr]])
        all_base = np.concatenate([all_base, base_mask[vr]])
        all_fold_rows.append((1, vr))
        all_fold_orders.append(d["validation_orders"])

# Build combined local slices
all_local_slices = []
offset = 0
for val_orders in all_fold_orders:
    for oi in val_orders:
        n_alarms = len(dataset["train_orders"][oi]["alarms"])
        all_local_slices.append(slice(offset, offset + n_alarms))
        offset += n_alarms

comb_target = round(target_train * len(all_rows) / len(labels))
comb_mask = v10.exact_count_mask(all_scores, all_local_slices, comb_target, 8)
comb_tp = int(np.sum(comb_mask & (all_labels == 1)))
comb_base_tp = int(np.sum(all_base & (all_labels == 1)))
comb_delta = comb_tp - comb_base_tp
print(f"  Rows: {len(all_rows)}  Target: {comb_target}")
print(f"  TP delta: {comb_delta:+d}")

# Fold-level breakdown
for fi, (fold_num, vr) in enumerate(all_fold_rows):
    fold_local = []
    off = 0
    for oi in all_fold_orders[fi]:
        n = len(dataset["train_orders"][oi]["alarms"])
        fold_local.append(slice(off, off+n))
        off += n
    fold_target_fi = round(target_train * len(vr) / len(labels))
    fm = v10.exact_count_mask(all_scores[off-len(vr):off], fold_local, fold_target_fi, 8)
    fbtp = int(np.sum(all_base[off-len(vr):off] & (all_labels[off-len(vr):off] == 1)))
    fntp = int(np.sum(fm & (all_labels[off-len(vr):off] == 1)))
    print(f"    Fold {fold_num}: base={fbtp} sem={fntp} delta={fntp-fbtp:+d}")
