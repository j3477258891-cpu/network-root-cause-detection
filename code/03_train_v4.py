"""
03_train_v4.py - 训练脚本 V4（三模型集成 + 全量训练）
======================================================
V3 → V4 改进：
1. 加入 CatBoost 模型，三模型概率平均
2. 用全部训练数据（不再 split）
3. 自适应阈值：每工单根据预测分数分布动态选 TopK

运行：python 03_train_v4.py
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
for d in [MODEL_DIR]: d.mkdir(exist_ok=True)


def train_three_models(X, y):
    """训练 XGBoost + LightGBM + CatBoost"""
    # XGBoost
    print("训练 XGBoost...")
    xgb_m = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, tree_method='hist',
        random_state=42, verbosity=0
    )
    xgb_m.fit(X, y)

    # LightGBM
    print("训练 LightGBM...")
    lgb_m = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, random_state=42, verbose=-1
    )
    lgb_m.fit(X, y)

    # CatBoost
    print("训练 CatBoost...")
    cb_m = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        subsample=0.8, rsm=0.8,
        auto_class_weights='Balanced',
        random_seed=42, verbose=False
    )
    cb_m.fit(X, y)

    return xgb_m, lgb_m, cb_m


def predict_ensemble(models, Xs):
    """三模型概率平均"""
    xgb_m, lgb_m, cb_m = models
    p_xgb = xgb_m.predict_proba(np.vstack(Xs))[:, 1]
    p_lgb = lgb_m.predict_proba(np.vstack(Xs))[:, 1]
    p_cb = cb_m.predict_proba(np.vstack(Xs))[:, 1]
    p_avg = (p_xgb + p_lgb + p_cb) / 3.0

    probs_list = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        probs_list.append(p_avg[offset:offset + n])
        offset += n
    return probs_list


def adaptive_topk_predict(probs_list, meta_list):
    """自适应 TopK：根据预测分数的分布动态选择"""
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

        # 自适应阈值：找概率分布中的明显跳跃
        sorted_probs = np.sort(probs)[::-1]

        # 方法：取前 K 个节点，其中 K 满足 prob[K-1] - prob[K] > 0.05（明显跳跃）
        K = 1
        for i in range(1, min(n, 9)):
            if i < len(sorted_probs) and sorted_probs[i - 1] - sorted_probs[i] > 0.05:
                K = i
                break
            K = i + 1

        # 兜底：如果最低分还 > 0.3，至少取 1 个
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
    print("V4 训练流程")
    print("=" * 50)

    # 加载数据
    with open(FEAT_DIR / "train_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys, meta = data["features"], data["labels"], data["meta"]

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"\n训练节点: {len(X_all)}, 正样本率: {y_all.mean():.3f}")
    print(f"特征维度: {X_all.shape[1]}")

    # 训练三模型（全量）
    models = train_three_models(X_all, y_all)

    # 训练集自身验证（不严格，只是看）
    p_xgb = models[0].predict_proba(X_all)[:, 1]
    p_lgb = models[1].predict_proba(X_all)[:, 1]
    p_cb = models[2].predict_proba(X_all)[:, 1]
    p_ens = (p_xgb + p_lgb + p_cb) / 3

    print(f"\n=== 训练集自身 F1（参考） ===")
    for name, p in [("XGBoost", p_xgb), ("LightGBM", p_lgb), ("CatBoost", p_cb), ("Ensemble", p_ens)]:
        yp = (p >= 0.5).astype(int)
        f1 = f1_score(y_all, yp, zero_division=0)
        p_v = precision_score(y_all, yp, zero_division=0)
        r = recall_score(y_all, yp, zero_division=0)
        print(f"  {name}: F1={f1:.4f}  P={p_v:.4f}  R={r:.4f}")

    # 保存模型
    with open(MODEL_DIR / "xgb_v4.pkl", "wb") as f:
        pickle.dump(models[0], f)
    with open(MODEL_DIR / "lgb_v4.pkl", "wb") as f:
        pickle.dump(models[1], f)
    with open(MODEL_DIR / "cb_v4.pkl", "wb") as f:
        pickle.dump(models[2], f)
    print("\n✅ 三个模型已保存")

    # 对训练集做自适应预测，检查 K 分布
    probs_list = predict_ensemble(models, Xs)
    preds, avg_k = adaptive_topk_predict(probs_list, meta)
    print(f"\n自适应 TopK 平均根因数: {avg_k:.2f}")
    print(f"真实平均根因数: {np.mean([sum(y) for y in ys]):.2f}")


if __name__ == "__main__":
    main()