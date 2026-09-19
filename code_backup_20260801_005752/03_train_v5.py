"""
03_train_v5.py - 训练脚本 V5（多种子集成 + TF-IDF特征）
=====================================================
V4 → V5 改进：
1. TF-IDF 文本特征（32维）
2. 多种子集成（5个 XGBoost 不同种子）
3. 三模型 × 多种子

运行：python 03_train_v5.py
"""

import pickle, json, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]


def train_seed_models(X, y, seed):
    """用指定种子训练三模型"""
    xgb_m = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, tree_method='hist',
        random_state=seed, verbosity=0
    )
    xgb_m.fit(X, y)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, random_state=seed, verbose=-1
    )
    lgb_m.fit(X, y)

    cb_m = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        subsample=0.8, rsm=0.8,
        auto_class_weights='Balanced',
        random_seed=seed, verbose=False
    )
    cb_m.fit(X, y)

    return xgb_m, lgb_m, cb_m


def predict_ensemble(models_list, Xs):
    """多种子平均"""
    all_probs = []
    for xgb_m, lgb_m, cb_m in models_list:
        p = (xgb_m.predict_proba(np.vstack(Xs))[:, 1] +
             lgb_m.predict_proba(np.vstack(Xs))[:, 1] +
             cb_m.predict_proba(np.vstack(Xs))[:, 1]) / 3.0
        all_probs.append(p)
    p_avg = np.mean(all_probs, axis=0)

    probs_list = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        probs_list.append(p_avg[offset:offset + n])
        offset += n
    return probs_list


def adaptive_topk_predict(probs_list, meta_list):
    all_preds = {}
    k_vals = []
    for probs, meta in zip(probs_list, meta_list):
        wo_id = meta["wo_id"]
        rids = meta["rids"]
        n = len(probs)
        if n == 0:
            all_preds[wo_id] = {"rootcause": []}
            k_vals.append(0)
            continue

        sorted_probs = np.sort(probs)[::-1]
        K = 1
        for i in range(1, min(n, 9)):
            if i < len(sorted_probs) and sorted_probs[i - 1] - sorted_probs[i] > 0.05:
                K = i
                break
            K = i + 1
        if K == 1 and probs.max() < 0.3:
            K = 1
        elif K > 8:
            K = 8

        ranked = np.argsort(-probs)
        top_k = ranked[:K]
        k_vals.append(len(top_k))
        rc_list = [{"@rid": rids[int(idx)]} for idx in top_k]
        all_preds[wo_id] = {"rootcause": rc_list}
    return all_preds, np.mean(k_vals)


def main():
    print("=" * 50)
    print("V5 训练流程（TF-IDF + 多种子）")
    print("=" * 50)

    with open(FEAT_DIR / "train_features_v5.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys, meta = data["features"], data["labels"], data["meta"]

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"训练节点: {len(X_all)}, 正样本率: {y_all.mean():.3f}, 特征: {X_all.shape[1]}")

    # 训练 5 个种子的模型
    models_list = []
    for seed in SEEDS:
        print(f"\n训练种子 {seed}...")
        models_list.append(train_seed_models(X_all, y_all, seed))

    # 保存所有模型
    all_models = {"seeds": SEEDS, "models": models_list}
    with open(MODEL_DIR / "v5_ensemble.pkl", "wb") as f:
        pickle.dump(all_models, f)
    print(f"\n✅ 5组模型已保存 (5种子 × 3模型 = 15个)")

    # 训练集验证
    all_probs = []
    for xgb_m, lgb_m, cb_m in models_list:
        p = (xgb_m.predict_proba(X_all)[:, 1] +
             lgb_m.predict_proba(X_all)[:, 1] +
             cb_m.predict_proba(X_all)[:, 1]) / 3.0
        all_probs.append(p)
    p_ens = np.mean(all_probs, axis=0)

    yp = (p_ens >= 0.5).astype(int)
    f1 = f1_score(y_all, yp, zero_division=0)
    print(f"\n训练集 Ensemble F1: {f1:.4f}")


if __name__ == "__main__":
    main()