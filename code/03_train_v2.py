"""
03_train_v2.py - XGBoost V2 训练
================================
改进：
1. 只对 Alarm 节点训练
2. 自动计算 scale_pos_weight
3. 预测时校准阈值，控制输出数量

运行方式：python 03_train_v2.py
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("加载 V2 训练特征...")
    
    with open(DATA_DIR / "train_features_v2.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_all = np.vstack([x for x in data["features"] if len(x) > 0])
    y_all = np.hstack([y for y in data["labels"] if len(y) > 0])
    
    pos_ratio = y_all.sum() / len(y_all)
    scale_weight = int(1.0 / pos_ratio)
    
    print(f"节点数: {len(X_all)}, 正样本: {y_all.sum()} ({pos_ratio*100:.1f}%)")
    print(f"特征维: {X_all.shape[1]}, scale_pos_weight: {scale_weight}")
    
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    # 5-fold CV
    print("\n" + "=" * 60)
    print("5-Fold CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (tr, vl) in enumerate(skf.split(X_all, y_all)):
        m = XGBClassifier(**params)
        m.fit(X_all[tr], y_all[tr], eval_set=[(X_all[vl], y_all[vl])], verbose=False)
        
        yp = m.predict_proba(X_all[vl])[:, 1]
        yd = (yp >= 0.5).astype(int)
        
        auc = roc_auc_score(y_all[vl], yp)
        f1 = f1_score(y_all[vl], yd, zero_division=0)
        prec = precision_score(y_all[vl], yd, zero_division=0)
        rec = recall_score(y_all[vl], yd, zero_division=0)
        scores.append((auc, f1, prec, rec))
        print(f"  Fold {fold+1}: AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
    
    avg = np.mean(scores, axis=0)
    print(f"\n  平均: AUC={avg[0]:.4f}, F1={avg[1]:.4f}, Prec={avg[2]:.4f}, Rec={avg[3]:.4f}")
    
    # 全量训练
    print("\n" + "=" * 60)
    print("全量训练...")
    model = XGBClassifier(**params)
    model.fit(X_all, y_all, verbose=100)
    
    with open(MODEL_DIR / "xgboost_v2.pkl", "wb") as f:
        pickle.dump({"model": model, "params": params, "cv_avg": avg.tolist()}, f)
    print(f"✅ 模型已保存: {MODEL_DIR / 'xgboost_v2.pkl'}")
    
    # 特征重要性
    imp = model.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    print(f"\nTop 20 特征:")
    for r, idx in enumerate(top):
        print(f"  {r+1:2d}. f{idx:3d}  {imp[idx]:.4f}")


if __name__ == "__main__":
    main()
