"""
03_train_v9.py - V9 训练（XGBoost 5种子 + 工单级K预测器）
"""

import pickle, warnings
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)
SEEDS = [42, 123, 456, 789, 2024]


def main():
    print("V9: Node2Vec + N-gram + XGBoost 5-seed")
    print("=" * 50)

    with open(FEAT_DIR / "train_features_v9.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys = data["features"], data["labels"]

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"训练节点: {len(X_all)}, 正样本率: {y_all.mean():.3f}, 特征: {X_all.shape[1]}")

    # 训练 5 种子 XGBoost
    models = []
    for seed in SEEDS:
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2.5, tree_method='hist',
            random_state=seed, verbosity=0
        )
        m.fit(X_all, y_all)
        models.append(m)
        print(f"  种子{seed} 完成")

    with open(MODEL_DIR / "v9_xgb_5seed.pkl", "wb") as f:
        pickle.dump(models, f)

    # 训练工单级 K 预测器
    print("\n训练工单级 K 预测器...")
    # 5-fold 交叉预测来获取训练集概率
    from sklearn.model_selection import KFold
    all_probs = np.zeros(len(X_all))
    all_wo_ids = []
    for i, (Xs_i, ys_i) in enumerate(zip(Xs, ys)):
        all_wo_ids.extend([i] * len(ys_i))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X_all):
        # 为每个 fold 训练并且产生 out-of-fold 预测
        m_fold = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=2.5,
            tree_method='hist', random_state=42, verbosity=0
        )
        m_fold.fit(X_all[tr_idx], y_all[tr_idx])
        all_probs[val_idx] = m_fold.predict_proba(X_all[val_idx])[:, 1]

    # 构建 K 预测器特征
    K_X = []
    K_y = []
    offset = 0
    for i, (X_wo, y_wo) in enumerate(zip(Xs, ys)):
        n = len(X_wo)
        if n == 0: continue
        probs = all_probs[offset:offset + n]
        offset += n

        # 特征：概率的统计量
        K_X.append([
            float(np.mean(probs)),      # 均值
            float(np.std(probs)),        # 标准差
            float(np.max(probs)),        # 最大值
            float(np.min(probs)),        # 最小值
            float(np.max(probs) - np.min(probs)),  # 极差
            float(np.sum(probs >= 0.4)), # >=0.4 的数量
            float(np.sum(probs >= 0.5)), # >=0.5 的数量
            float(np.sum(probs >= 0.6)), # >=0.6 的数量
            float(n),                    # 总 Alarm 数
        ])
        K_y.append(int(y_wo.sum()))  # 真实根因数

    K_X = np.array(K_X, dtype=np.float32)
    K_y = np.array(K_y)

    k_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3,
        objective='reg:squarederror',
        random_state=42, verbosity=0
    )
    k_model.fit(K_X, K_y.astype(float))
    # 回归结果四舍五入
    k_pred = np.round(k_model.predict(K_X)).astype(int)
    k_pred = np.clip(k_pred, 1, 8)

    with open(MODEL_DIR / "v9_k_predictor.pkl", "wb") as f:
        pickle.dump(k_model, f)

    # 验证 K 预测器
    k_pred = np.round(k_model.predict(K_X)).astype(int)
    k_pred = np.clip(k_pred, 1, 8)
    k_acc = (k_pred == K_y).mean()
    k_mae = np.abs(k_pred - K_y).mean()
    print(f"K 预测器准确率: {k_acc:.3f}, MAE: {k_mae:.2f}")
    print(f"\n✅ V9 全部完成")


if __name__ == "__main__":
    main()