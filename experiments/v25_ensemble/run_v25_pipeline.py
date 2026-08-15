"""
V25 主流程 —— 一站式训练、评估、提交
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add V10 path
V10_PATH = Path("D:/zgyidong/codexgz/work")
sys.path.insert(0, str(V10_PATH))
import v10_grouped_ensemble as v10

sys.path.insert(0, str(Path(__file__).parent))
import v25_data
from models import v25_dp
from transfer import template_match


def main():
    CONFIG_PATH = Path(__file__).parent / "config_v25.json"
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    
    paths = cfg["paths"]
    OUTPUT_DIR = Path(paths["output_dir"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════
    # Phase 1: Data Loading
    # ═══════════════════════════════════════
    print("=" * 60)
    print("V25 Full Pipeline")
    print("=" * 60)
    
    t0 = time.time()
    dataset = v25_data.load_all_data(
        paths["train_dir"], paths["test_dir"], paths["v11_dir"]
    )
    feature_sets = v25_data.build_feature_sets(dataset)
    
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    test_slices = dataset["data"]["test_slices"]
    folds_arr = dataset["folds"]
    train_orders = dataset["train_orders"]
    test_orders = dataset["test_orders"]
    train_v11 = dataset["train_v11"]
    test_v11 = dataset["test_v11"]
    
    print(f"Data loaded in {time.time() - t0:.1f}s", flush=True)
    
    # V11 baseline
    target_train = round(1059 / len(test_orders) * len(train_orders))
    base_mask = v10.exact_count_mask(train_v11, slices, target_train, 8)
    base_stats = v25_data.confusion(base_mask, labels)
    print(f"\nV11 Baseline: F1={base_stats[0]:.6f} TP={base_stats[1]} "
          f"N={base_stats[4]}", flush=True)
    
    # ═══════════════════════════════════════
    # Phase 2: Workflow A — Multi-Model Training
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 2: Multi-Model Training (Workflow A)")
    print("=" * 60)
    
    seeds = cfg["seeds"]
    model_configs = cfg["models"]
    
    all_oof = {}
    all_test = {}
    
    for model_key, mcfg in model_configs.items():
        print(f"\n--- {mcfg['name']} ({mcfg['algorithm']}) ---", flush=True)
        t1 = time.time()
        
        # Get features
        feat_key_map = {
            "M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4"
        }
        feat_key = feat_key_map.get(model_key, model_key)
        fs = feature_sets[feat_key]
        X_train_feats = fs["train"]
        X_test_feats = fs["test"]
        
        print(f"  Features: {X_train_feats.shape[1]} dim", flush=True)
        
        all_oof[model_key] = []
        all_test[model_key] = []
        
        for seed_idx, seed in enumerate(seeds):
            # 5-fold OOF
            oof = v25_data.train_model_cv(
                mcfg["name"], X_train_feats, labels, slices, folds_arr,
                mcfg["algorithm"], mcfg["params"], seed
            )
            all_oof[model_key].append(oof)
            
            # Full test
            test_pred = v25_data.train_full_model(
                mcfg["name"], X_train_feats, labels, X_test_feats,
                mcfg["algorithm"], mcfg["params"], seed
            )
            all_test[model_key].append(test_pred)
            
            print(f"    Seed {seed}: done", flush=True)
        
        # Average across seeds
        all_oof[model_key] = np.mean(all_oof[model_key], axis=0)
        all_test[model_key] = np.mean(all_test[model_key], axis=0)
        
        # Quick eval
        mask = v10.exact_count_mask(all_oof[model_key], slices, target_train, 8)
        tp = int(np.sum(mask & (labels == 1)))
        delta = tp - base_stats[1]
        print(f"  OOF TP delta: {delta:+d}", flush=True)
        print(f"  Time: {time.time() - t1:.1f}s", flush=True)
    
    # Save per-model results
    for model_key in model_configs:
        np.save(OUTPUT_DIR / f"v25_{model_key}_oof.npy", all_oof[model_key])
        np.save(OUTPUT_DIR / f"v25_{model_key}_test.npy", all_test[model_key])
    
    # ═══════════════════════════════════════
    # Phase 3: Stacking Ensemble
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 3: Stacking Ensemble")
    print("=" * 60)
    
    t1 = time.time()
    model_keys = list(model_configs.keys())
    stack_train_X = np.column_stack([all_oof[k] for k in model_keys])
    stack_test_X = np.column_stack([all_test[k] for k in model_keys])
    
    # Add V11 base
    v11_train = train_v11.reshape(-1, 1)
    v11_test = test_v11.reshape(-1, 1)
    stack_train_X = np.column_stack([stack_train_X, v11_train])
    stack_test_X = np.column_stack([stack_test_X, v11_test])
    
    # Train stacking meta-learner
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=seeds[0])
    stack_oof = np.zeros(len(labels), dtype=np.float32)
    
    import lightgbm as lgb
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(stack_train_X)):
        meta = lgb.LGBMRegressor(
            random_state=seeds[0] + fold_idx,
            verbose=-1,
            **cfg["stacking"]["params"]
        )
        meta.fit(stack_train_X[tr_idx], labels[tr_idx])
        stack_oof[val_idx] = np.clip(meta.predict(stack_train_X[val_idx]), 0, 1)
    
    # Full meta model for test
    meta_full = lgb.LGBMRegressor(
        random_state=seeds[0], verbose=-1,
        **cfg["stacking"]["params"]
    )
    meta_full.fit(stack_train_X, labels)
    stack_test = np.clip(meta_full.predict(stack_test_X), 0, 1)
    
    np.save(OUTPUT_DIR / "v25_stack_oof.npy", stack_oof)
    np.save(OUTPUT_DIR / "v25_stack_test.npy", stack_test)
    
    # Evaluate stacking
    stack_mask = v10.exact_count_mask(stack_oof, slices, target_train, 8)
    stack_tp = int(np.sum(stack_mask & (labels == 1)))
    stack_delta = stack_tp - base_stats[1]
    print(f"Stack OOF TP delta: {stack_delta:+d}", flush=True)
    
    # ═══════════════════════════════════════
    # Phase 4: Enhanced DP (Workflow B)
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 4: Enhanced DP (Workflow B)")
    print("=" * 60)
    
    # Use per-model and stack scores for DP
    dp_scores = {k: all_oof[k] for k in model_keys}
    dp_scores["stack"] = stack_oof
    dp_weights = [0.15, 0.15, 0.1, 0.05, 0.55]  # M1, M2, M3, M4, stack
    
    dp_result = v25_dp.enhanced_dp_pipeline(
        dp_scores, dp_weights, slices, labels, base_mask, train_v11,
        target_predictions=target_train
    )
    
    dp_delta = dp_result["tp_delta"]
    print(f"DP OOF TP delta: {dp_delta:+d} ({dp_result['changed_orders']} orders changed)", flush=True)
    
    # Apply DP to test
    dp_test_scores = {k: all_test[k] for k in model_keys}
    dp_test_scores["stack"] = stack_test
    
    # ═══════════════════════════════════════
    # Phase 5: Template Transfer (Workflow D)
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 5: Template Transfer (Workflow D)")
    print("=" * 60)
    
    tpl_result = template_match.build_template_features(
        train_orders, test_orders, labels, slices, cfg["template_transfer"]
    )
    tpl_scores = tpl_result["prior_scores"]
    
    # ═══════════════════════════════════════
    # Phase 6: Final Integration
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 6: Final Integration")
    print("=" * 60)
    
    # Combine stack + template scores
    tpl_weight = 0.05  # Conservative template weight
    final_oof = (1 - tpl_weight) * stack_oof
    
    # Evaluate final OOF (without template since template is test-only)
    final_mask = v10.exact_count_mask(final_oof, slices, target_train, 8)
    final_tp = int(np.sum(final_mask & (labels == 1)))
    final_delta = final_tp - base_stats[1]
    
    # Fold-level evaluation
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
        
        # Build local slices for this fold
        local_slices = []
        offset = 0
        for oi in fold_oi:
            sl = slices[oi]
            n = sl.stop - sl.start
            local_slices.append(slice(offset, offset + n))
            offset += n
        
        fold_new = v10.exact_count_mask(fold_scores, local_slices, fold_target, 8)
        fold_delta = int(np.sum(fold_new & (fold_labels == 1))) - int(np.sum(fold_base & (fold_labels == 1)))
        fold_deltas.append(fold_delta)
    
    improved_folds = sum(d > 0 for d in fold_deltas)
    
    # Bootstrap
    per_order = v25_data.per_order_tp_delta(final_mask, base_mask, labels, slices)
    boot_lower = v25_data.bootstrap_lower_bound(per_order)
    
    print(f"  Fold deltas: {fold_deltas}", flush=True)
    print(f"  Improved folds: {improved_folds}/5", flush=True)
    print(f"  Bootstrap 95% lower: {boot_lower:.1f}", flush=True)
    print(f"  Final OOF TP delta: {final_delta:+d}", flush=True)
    
    # ═══════════════════════════════════════
    # Gate Check
    # ═══════════════════════════════════════
    gate = cfg["gate"]
    passed = (
        final_delta >= gate["min_oof_tp_delta"]
        and improved_folds >= gate["min_improved_folds"]
        and min(fold_deltas) >= gate["min_fold_tp_delta"]
        and boot_lower >= gate["bootstrap_lower_bound"]
    )
    
    print(f"\n{'='*60}")
    print(f"GATE: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"  OOF TP delta: {final_delta:+d} (need >= {gate['min_oof_tp_delta']})")
    print(f"  Improved folds: {improved_folds}/5 (need >= {gate['min_improved_folds']})")
    print(f"  Min fold delta: {min(fold_deltas):+d} (need >= {gate['min_fold_tp_delta']})")
    print(f"  Bootstrap 95%: {boot_lower:.1f} (need >= {gate['bootstrap_lower_bound']})")
    print("=" * 60)
    
    # ═══════════════════════════════════════
    # Generate Test Predictions
    # ═══════════════════════════════════════
    # Apply template prior to test scores
    final_test = (1 - tpl_weight) * stack_test + tpl_weight * tpl_scores
    
    # Also apply DP to test
    dp_test_actions, _ = v25_dp.dp_decode(
        v25_dp.compute_action_utilities(
            [v for k, v in dp_test_scores.items()],
            dp_weights,
            None,  # no labels for test
            test_slices
        ),
        target_delta_k=0
    )
    
    # Stack test mask
    test_mask = v10.exact_count_mask(final_test, test_slices, 1059, 8)
    test_count = int(np.sum(test_mask))
    
    print(f"\nTest predictions: {test_count}", flush=True)
    
    # ═══════════════════════════════════════
    # Save Outputs
    # ═══════════════════════════════════════
    SUBMISSION_DIR = Path(paths["submission_dir"])
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(OUTPUT_DIR / "v25_final_test.npy", final_test)
    np.save(OUTPUT_DIR / "v25_tpl_scores.npy", tpl_scores)
    
    if passed:
        out_path = SUBMISSION_DIR / "result_record_v25_ensemble_p1059.csv"
        v10.write_submission(out_path, test_orders, dataset["data"], test_mask)
        print(f"\nSubmission written: {out_path}", flush=True)
        
        # SHA256
        import hashlib
        content = out_path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        (SUBMISSION_DIR / "result_record_v25_ensemble_p1059.sha256").write_text(sha)
        print(f"SHA256: {sha}", flush=True)
    else:
        print(f"\nGate not passed — no submission.", flush=True)
    
    # Report
    report = {
        "version": "v25-ensemble",
        "base": {"f1": base_stats[0], "tp": base_stats[1], "n": base_stats[4]},
        "stack": {"tp_delta": int(stack_delta)},
        "dp": {"tp_delta": int(dp_delta), "changed_orders": dp_result["changed_orders"]},
        "final": {"tp_delta": int(final_delta), "fold_deltas": fold_deltas,
                  "improved_folds": improved_folds, "bootstrap_95_lower": float(boot_lower)},
        "template": {"n_templates": tpl_result["n_templates"],
                     "coverage": tpl_result["coverage"],
                     "validation_accuracy": tpl_result["validation"]["accuracy"]},
        "gate_passed": passed,
        "test": {"predictions": test_count},
        "time_seconds": time.time() - t0,
    }
    
    with open(OUTPUT_DIR / "v25_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport: {OUTPUT_DIR / 'v25_report.json'}")
    print(f"Total time: {report['time_seconds']:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
