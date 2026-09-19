"""
08_train_v8.py - 回归简单 + 概率校准
=====================================
改进：
1. 只用 V3 特征（45维，不要 TF-IDF 噪声）
2. 只用 XGBoost（去掉 LightGBM/CatBoost）
3. 5种子集成 + Platt Scaling 概率校准
4. 训练集上学习最优阈值

运行：python 08_train_v8.py
"""

import pickle, json, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)
SEEDS = [42, 123, 456, 789, 2024]


def main():
    print("V8: 回归简单 XGBoost + Platt 校准")
    print("=" * 50)

    # 用 V3 特征（45维，不要 TF-IDF）
    with open(FEAT_DIR / "train_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys, meta = data["features"], data["labels"], data["meta"]

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"训练节点: {len(X_all)}, 正样本率: {y_all.mean():.3f}, 特征: {X_all.shape[1]}")

    # 5种子训练 + 校准
    calibrated_models = []
    for seed in SEEDS:
        base = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2.5, tree_method='hist',
            random_state=seed, verbosity=0
        )
        # Platt Scaling 校准
        calib = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        calib.fit(X_all, y_all)
        calibrated_models.append(calib)
        print(f"  种子{seed}: 校准完成")

    # 保存
    with open(MODEL_DIR / "v8_models.pkl", "wb") as f:
        pickle.dump(calibrated_models, f)

    # 评估：寻找最优阈值
    print("\n=== 训练集 held-out 最优阈值搜索 ===")
    np.random.seed(42)
    n = len(Xs)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    tr_X = np.vstack([Xs[i] for i in idx[:split]])
    tr_y = np.concatenate([ys[i] for i in idx[:split]])
    val_X = np.vstack([Xs[i] for i in idx[split:]])
    val_y = np.concatenate([ys[i] for i in idx[split:]])

    # 训练校准模型
    calib = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
            scale_pos_weight=2.5, tree_method='hist', random_state=42, verbosity=0),
        method='sigmoid', cv=3
    )
    calib.fit(tr_X, tr_y)
    val_probs = calib.predict_proba(val_X)[:, 1]

    # 搜索最优阈值（按 F1）
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.2, 0.8, 0.02):
        yp = (val_probs >= th).astype(int)
        f1 = f1_score(val_y, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
            best_p = precision_score(val_y, yp, zero_division=0)
            best_r = recall_score(val_y, yp, zero_division=0)

    print(f"最优阈值={best_th:.2f}: F1={best_f1:.4f} P={best_p:.4f} R={best_r:.4f}")

    # 保存最优阈值
    with open(MODEL_DIR / "v8_threshold.pkl", "wb") as f:
        pickle.dump(best_th, f)
    print(f"\n✅ 5模型 + 最优阈值{best_th:.2f} 已保存")


if __name__ == "__main__":
    main()