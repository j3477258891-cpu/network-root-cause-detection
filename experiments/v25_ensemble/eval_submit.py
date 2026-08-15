"""V25 final evaluation and submission."""
import json, sys, numpy as np
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, str(Path(__file__).parent))
import v25_data

OUTPUT = Path("outputs")

# Load existing results
stack_oof = np.load(OUTPUT / "v25_stack_oof.npy")
stack_test = np.load(OUTPUT / "v25_stack_test.npy")
tpl_path = OUTPUT / "v25_tpl_scores.npy"
tpl_scores = np.load(tpl_path) if tpl_path.exists() else None

# Load data
dataset = v25_data.load_all_data(
    "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
)
labels = dataset["labels"]
slices = dataset["data"]["train_slices"]
test_slices = dataset["data"]["test_slices"]
folds_arr = dataset["folds"]
train_orders = dataset["train_orders"]
test_orders = dataset["test_orders"]
train_v11 = dataset["train_v11"]

target_train = round(1059 / 546 * 1634)
base_mask = v10.exact_count_mask(train_v11, slices, target_train, 8)
base_stats = v25_data.confusion(base_mask, labels)
base_tp = base_stats[1]
print(f"V11 Baseline: TP={base_tp}")

# Final OOF
final_oof = stack_oof
final_mask = v10.exact_count_mask(final_oof, slices, target_train, 8)
final_tp = int(np.sum(final_mask & (labels == 1)))
final_delta = final_tp - base_tp
print(f"Final OOF TP delta: {final_delta:+d}")

# Fold-level
all_orders = np.arange(len(train_orders))
fold_deltas = []
for fold in range(5):
    fold_oi = all_orders[folds_arr == fold]
    fold_rows = np.concatenate([
        np.arange(slices[i].start, slices[i].stop) for i in fold_oi
    ])
    fold_base = base_mask[fold_rows]
    fold_labels = labels[fold_rows]
    fold_scores = final_oof[fold_rows]
    fold_target = round(target_train * len(fold_rows) / len(labels))
    
    local_slices = []
    offset = 0
    for oi in fold_oi:
        sl = slices[oi]
        n = sl.stop - sl.start
        local_slices.append(slice(offset, offset + n))
        offset += n
    
    fold_new = v10.exact_count_mask(fold_scores, local_slices, fold_target, 8)
    fold_delta = int(np.sum(fold_new & (fold_labels == 1))) - int(
        np.sum(fold_base & (fold_labels == 1))
    )
    fold_deltas.append(fold_delta)

improved_folds = sum(d > 0 for d in fold_deltas)
per_order = v25_data.per_order_tp_delta(final_mask, base_mask, labels, slices)
boot_lower = v25_data.bootstrap_lower_bound(per_order)

print(f"Fold deltas: {fold_deltas}")
print(f"Improved: {improved_folds}/5")
print(f"Bootstrap 95% lower: {boot_lower:.1f}")

# Gate
gate_cfg = {
    "min_oof_tp_delta": 20,
    "min_improved_folds": 3,
    "min_fold_tp_delta": -3,
    "bootstrap_lower_bound": 5,
}
passed = (
    final_delta >= gate_cfg["min_oof_tp_delta"]
    and improved_folds >= gate_cfg["min_improved_folds"]
    and min(fold_deltas) >= gate_cfg["min_fold_tp_delta"]
    and boot_lower >= gate_cfg["bootstrap_lower_bound"]
)

print(f"\nGATE: {'PASSED' if passed else 'FAILED'}")
print(f"  TP delta: {final_delta:+d} (>= {gate_cfg['min_oof_tp_delta']})")
print(f"  Folds: {improved_folds}/5 (>= {gate_cfg['min_improved_folds']})")
print(f"  Min fold: {min(fold_deltas):+d} (>= {gate_cfg['min_fold_tp_delta']})")
print(f"  Boot 95%: {boot_lower:.1f} (>= {gate_cfg['bootstrap_lower_bound']})")

# Test
if passed:
    final_test = stack_test
    if tpl_scores is not None and len(tpl_scores) == len(stack_test):
        final_test = 0.95 * stack_test + 0.05 * tpl_scores
    
    test_mask = v10.exact_count_mask(final_test, test_slices, 1059, 8)
    test_count = int(np.sum(test_mask))
    print(f"\nTest predictions: {test_count}")
    
    import hashlib
    submission_dir = Path("submissions")
    submission_dir.mkdir(exist_ok=True)
    out_path = submission_dir / "result_record_v25_stacking_p1059.csv"
    v10.write_submission(out_path, test_orders, dataset["data"], test_mask)
    
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (submission_dir / "result_record_v25_stacking_p1059.sha256").write_text(sha)
    print(f"Submission: {out_path}")
    print(f"SHA256: {sha}")
else:
    print("\nNo submission generated.")
