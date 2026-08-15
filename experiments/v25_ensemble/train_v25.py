"""
V25 多策略集成 — 主训练脚本
工作流 A: M1-M4 5-fold OOF 训练 + 3-fold Stacking
工作流 C: 文本特征自动包含在 M1 中
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import v25_data

# ── Optional imports ──
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed", flush=True)

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: catboost not installed", flush=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed", flush=True)


def get_config():
    with open(Path(__file__).parent / "config_v25.json", "r") as f:
        return json.load(f)


def train_model_cv(name, X_train, y_train, slices, folds_arr, model_type, params, seed):
    """Train a model with 5-fold CV, return OOF predictions."""
    n = len(y_train)
    oof_preds = np.zeros(n, dtype=np.float32)
    all_orders = np.arange(len(slices))
    
    for heldout in range(5):
        train_idx = v25_data.rows_for_orders(
            {"train_slices": slices}, all_orders[folds_arr != heldout]
        )
        val_idx = v25_data.rows_for_orders(
            {"train_slices": slices}, all_orders[folds_arr == heldout]
        )
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        if model_type == "lgb":
            if not HAS_LGB:
                raise RuntimeError("lightgbm not installed")
            model = lgb.LGBMClassifier(
                random_state=seed + heldout,
                verbose=-1,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_train[val_idx])],
                     eval_metric="auc", callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "catboost":
            if not HAS_CAT:
                raise RuntimeError("catboost not installed")
            model = CatBoostClassifier(
                random_seed=seed + heldout,
                verbose=0,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_train[val_idx]),
                     early_stopping_rounds=30)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "xgb":
            if not HAS_XGB:
                raise RuntimeError("xgboost not installed")
            model = xgb.XGBClassifier(
                random_state=seed + heldout,
                verbosity=0,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_train[val_idx])],
                     verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "lr":
            model = LogisticRegression(
                random_state=seed + heldout,
                **params
            )
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        t0 = time.time()
    
    return oof_preds


def train_full_model(name, X_train, y_train, X_test, model_type, params, seed):
    """Train a model on full training data, return test predictions."""
    if model_type == "lgb":
        if not HAS_LGB:
            raise RuntimeError("lightgbm not installed")
        model = lgb.LGBMClassifier(
            random_state=seed, verbose=-1, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "catboost":
        if not HAS_CAT:
            raise RuntimeError("catboost not installed")
        model = CatBoostClassifier(
            random_seed=seed, verbose=0, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        model = xgb.XGBClassifier(
            random_state=seed, verbosity=0, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "lr":
        model = LogisticRegression(random_state=seed, **params)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
    
    raise ValueError(f"Unknown model type: {model_type}")


def stacking_oof(train_models, labels, slices, folds_arr, stacking_params, seed):
    """3-fold stacking on OOF logits using LightGBM ranker."""
    n_models = len(train_models)
    n = len(labels)
    stack_oof = np.zeros(n, dtype=np.float32)
    
    # Use same folds for stacking
    all_orders = np.arange(len(slices))
    
    for stack_fold in range(3):
        # Simple: use folds [0,1,2] as inner CV
        val_folds = [stack_fold]
        tr_folds = [f for f in range(5) if f not in val_folds]
        
        val_idx = v25_data.rows_for_orders(
            {"train_slices": slices},
            all_orders[np.isin(folds_arr, val_folds)]
        )
        tr_idx = v25_data.rows_for_orders(
            {"train_slices": slices},
            all_orders[np.isin(folds_arr, tr_folds)]
        )
        
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        
        # Stack features: [M1_oof, M2_oof, M3_oof, M4_oof, V11_oof]
        X_tr_stack = np.column_stack([model["oof"][tr_idx] for model in train_models])
        X_val_stack = np.column_stack([model["oof"][val_idx] for model in train_models])
        y_tr = labels[tr_idx]
        
        if HAS_LGB:
            meta = lgb.LGBMRegressor(
                random_state=seed + stack_fold,
                verbose=-1,
                **stacking_params
            )
            meta.fit(X_tr_stack, y_tr)
            stack_oof[val_idx] = np.clip(meta.predict(X_val_stack), 0, 1)
        else:
            # Fallback: weighted average
            stack_oof[val_idx] = np.mean(X_val_stack, axis=1)
    
    return stack_oof


def evaluate_oof(scores, labels, slices, base_mask, target_count=None):
    """Evaluate OOF with fixed-K selection and compute TP delta."""
    if target_count is None:
        target_count = int(np.sum(base_mask))
    
    new_mask = v25_data.v10.exact_count_mask(scores, slices, target_count, 8)
    
    base_f1, base_tp, base_fp, base_fn, base_n = v25_data.confusion(base_mask, labels)
    new_f1, new_tp, new_fp, new_fn, new_n = v25_data.confusion(new_mask, labels)
    delta = new_tp - base_tp
    
    per_order = v25_data.per_order_tp_delta(new_mask, base_mask, labels, slices)
    boot_lower = v25_data.bootstrap_lower_bound(per_order)
    
    # Fold-level deltas
    folds_arr = np.zeros(len(slices), dtype=np.int8)  # This needs fold info; simplified here
    # We compute fold deltas by using the actual fold assignments
    
    return {
        "base_tp": base_tp, "new_tp": new_tp, "tp_delta": delta,
        "base_f1": base_f1, "new_f1": new_f1,
        "fp_delta": new_fp - base_fp, "fn_delta": new_fn - base_fn,
        "bootstrap_95_lower": boot_lower,
        "per_order_deltas": per_order,
    }


def main():
    cfg = get_config()
    paths = cfg["paths"]
    
    # Load data
    print("=" * 60)
    print("V25 Multi-Strategy Ensemble")
    print("=" * 60)
    
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
    
    # V11 baseline mask
    target_train = round(1059 / len(test_orders) * len(train_orders))
    base_mask = v25_data.v10.exact_count_mask(
        dataset["train_v11"], slices, target_train, 8
    )
    base_stats = v25_data.confusion(base_mask, labels)
    print(f"\nV11 Baseline OOF: F1={base_stats[0]:.6f} TP={base_stats[1]} "
          f"FP={base_stats[2]} FN={base_stats[3]} N={base_stats[4]}", flush=True)
    
    # ── Train M1-M4 ──
    model_configs = cfg["models"]
    seeds = cfg["seeds"]
    
    all_oof = {}  # model_name -> per-seed OOF list
    all_test = {}  # model_name -> per-seed test list
    
    for model_key, mcfg in model_configs.items():
        print(f"\n{'='*40}")
        print(f"Training {mcfg['name']} ({mcfg['algorithm']})...", flush=True)
        
        feat_key = mcfg["features"].upper()
        if feat_key not in feature_sets:
            feat_key = f"M{list(model_configs.keys()).index(model_key) + 1}"
        
        if isinstance(feature_sets.get(feat_key), dict):
            X_train_feats = feature_sets[feat_key]["train"]
            X_test_feats = feature_sets[feat_key]["test"]
        elif feat_key == "V11":
            X_train_feats = feature_sets["V11"]["train"].reshape(-1, 1)
            X_test_feats = feature_sets["V11"]["test"].reshape(-1, 1)
        else:
            print(f"  Feature key {feat_key} not found, skipping", flush=True)
            continue
        
        all_oof[model_key] = []
        all_test[model_key] = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})...", flush=True)
            
            # 5-fold OOF
            oof = train_model_cv(
                mcfg["name"], X_train_feats, labels, slices, folds_arr,
                mcfg["algorithm"], mcfg["params"], seed
            )
            all_oof[model_key].append(oof)
            
            # Full model for test
            test_pred = train_full_model(
                mcfg["name"], X_train_feats, labels, X_test_feats,
                mcfg["algorithm"], mcfg["params"], seed
            )
            all_test[model_key].append(test_pred)
    
    # Average across seeds
    for model_key in model_configs:
        all_oof[model_key] = np.mean(all_oof[model_key], axis=0)
        all_test[model_key] = np.mean(all_test[model_key], axis=0)
    
    # ── Evaluate individual models ──
    print(f"\n{'='*40}")
    print("Individual Model OOF Evaluation (fixed-K):", flush=True)
    
    model_deltas = {}
    for model_key, mcfg in model_configs.items():
        oof_scores = all_oof[model_key]
        result = evaluate_oof(oof_scores, labels, slices, base_mask, target_train)
        model_deltas[model_key] = result
        print(f"  {mcfg['name']}: TP_delta={result['tp_delta']:+d} "
              f"F1_delta={result['new_f1']-result['base_f1']:+.6f} "
              f"boot95_lower={result['bootstrap_95_lower']:.1f}", flush=True)
    
    # ── Stacking ──
    print(f"\n{'='*40}")
    print("Stacking Ensemble...", flush=True)
    
    # Build stacking features
    model_keys = list(model_configs.keys())
    stack_train_X = np.column_stack([all_oof[k] for k in model_keys])
    
    # Add V11 base logit as feature
    v11_base = feature_sets["V11"]["train"].reshape(-1, 1)
    stack_train_X = np.column_stack([stack_train_X, v11_base])
    
    stack_oof = stacking_oof(
        [{"oof": all_oof[k]} for k in model_keys] + [{"oof": v11_base.ravel()}],
        labels, slices, folds_arr, cfg["stacking"]["params"], seeds[0]
    )
    
    stack_result = evaluate_oof(stack_oof, labels, slices, base_mask, target_train)
    print(f"  Stack OOF: TP_delta={stack_result['tp_delta']:+d} "
          f"F1_delta={stack_result['new_f1']-stack_result['base_f1']:+.6f} "
          f"boot95_lower={stack_result['bootstrap_95_lower']:.1f}", flush=True)
    
    # ── Fold-level evaluation ──
    print(f"\n{'='*40}")
    print("Fold-level OOF Evaluation:", flush=True)
    
    all_orders = np.arange(len(train_orders))
    fold_deltas = []
    for fold in range(5):
        fold_oi = all_orders[folds_arr == fold]
        fold_rows = v25_data.rows_for_orders({"train_slices": slices}, fold_oi)
        
        fold_base = base_mask[fold_rows]
        fold_labels = labels[fold_rows]
        fold_scores = stack_oof[fold_rows]
        
        target_fold = round(target_train * len(fold_rows) / len(labels))
        fold_mask = v25_data.v10.exact_count_mask(
            fold_scores,
            [slice(0, len(fold_rows))],
            target_fold, 8
        )
        base_fold_tp = int(np.sum(fold_base & (fold_labels == 1)))
        new_fold_tp = int(np.sum(fold_mask[0] & (fold_labels == 1)))
        delta = new_fold_tp - base_fold_tp
        fold_deltas.append(delta)
        print(f"  Fold {fold}: TP_delta={delta:+d}", flush=True)
    
    improved = sum(d > 0 for d in fold_deltas)
    gate = cfg["gate"]
    
    # ── Gate check ──
    passed = (
        stack_result["tp_delta"] >= gate["min_oof_tp_delta"]
        and improved >= gate["min_improved_folds"]
        and min(fold_deltas) >= gate["min_fold_tp_delta"]
        and stack_result["bootstrap_95_lower"] >= gate["bootstrap_lower_bound"]
    )
    
    print(f"\n{'='*40}")
    print(f"GATE CHECK: {'PASSED' if passed else 'FAILED'}", flush=True)
    print(f"  OOF TP delta: {stack_result['tp_delta']:+d} (need >= {gate['min_oof_tp_delta']})")
    print(f"  Improved folds: {improved}/5 (need >= {gate['min_improved_folds']})")
    print(f"  Min fold delta: {min(fold_deltas):+d} (need >= {gate['min_fold_tp_delta']})")
    print(f"  Bootstrap 95% lower: {stack_result['bootstrap_95_lower']:.1f} (need >= {gate['bootstrap_lower_bound']})")
    
    # ── Test predictions ──
    print(f"\n{'='*40}")
    print("Generating test predictions...", flush=True)
    
    # Build test stacking features
    stack_test_X = np.column_stack([all_test[k] for k in model_keys])
    stack_test_X = np.column_stack([stack_test_X, feature_sets["V11"]["test"].reshape(-1, 1)])
    
    # Train full stacking model
    if HAS_LGB:
        meta_full = lgb.LGBMRegressor(
            random_state=seeds[0], verbose=-1,
            **cfg["stacking"]["params"]
        )
        meta_full.fit(stack_train_X, labels)
        test_stack_scores = np.clip(meta_full.predict(stack_test_X), 0, 1)
    else:
        test_stack_scores = np.mean(stack_test_X[:, :len(model_keys)], axis=1)
    
    # Fixed-K selection
    test_mask = v25_data.v10.exact_count_mask(test_stack_scores, test_slices, 1059, 8)
    
    # Save outputs
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "v25_stack_oof.npy", stack_oof)
    np.save(output_dir / "v25_stack_test.npy", test_stack_scores)
    
    for model_key in model_configs:
        np.save(output_dir / f"v25_{model_key}_oof.npy", all_oof[model_key])
        np.save(output_dir / f"v25_{model_key}_test.npy", all_test[model_key])
    
    # Write submission if gate passed
    submission_dir = Path(paths["submission_dir"])
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    if passed:
        out_path = submission_dir / "result_record_v25_stacking_p1059.csv"
        v25_data.v10.write_submission(out_path, test_orders, dataset["data"], test_mask)
        print(f"\nSubmission written: {out_path}", flush=True)
        print(f"Test predictions: {int(np.sum(test_mask))}", flush=True)
    else:
        print(f"\nGate not passed — no submission generated.", flush=True)
        # Still save test mask for analysis
        np.save(output_dir / "v25_test_mask.npy", test_mask)
    
    # Save report
    report = {
        "version": "v25-ensemble",
        "base_tp": base_stats[1],
        "base_f1": base_stats[0],
        "stack_tp_delta": stack_result["tp_delta"],
        "stack_f1_delta": stack_result["new_f1"] - stack_result["base_f1"],
        "model_deltas": {k: v["tp_delta"] for k, v in model_deltas.items()},
        "fold_deltas": fold_deltas,
        "improved_folds": improved,
        "bootstrap_95_lower": stack_result["bootstrap_95_lower"],
        "gate_passed": passed,
        "test_count": int(np.sum(test_mask)),
    }
    
    with open(output_dir / "v25_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved: {output_dir / 'v25_report.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
