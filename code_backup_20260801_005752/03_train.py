"""
03_train.py - XGBoost 模型训练
===========================
对每个节点做二分类：是否是根因节点。
使用 5-fold CV 评估，最终全量训练。

运行方式：python 03_train.py
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# ========== 配置 ==========
DATA_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)

PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 90,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def main():
    print("=" * 60)
    print("加载训练数据...")
    
    with open(DATA_DIR / "train_features.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_all = np.vstack([x for x in data["features"] if len(x) > 0])
    y_all = np.hstack([y for y in data["labels"] if len(y) > 0])
    
    print(f"总节点数: {len(X_all)}")
    print(f"正样本(根因): {y_all.sum()} ({y_all.sum()/len(y_all)*100:.2f}%)")
    print(f"特征维度: {X_all.shape[1]}")
    
    # 5-fold CV
    print("\n" + "=" * 60)
    print("5-Fold 交叉验证")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (tr, vl) in enumerate(skf.split(X_all, y_all)):
        Xt, Xv = X_all[tr], X_all[vl]
        yt, yv = y_all[tr], y_all[vl]
        
        m = XGBClassifier(**PARAMS)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        
        yp = m.predict_proba(Xv)[:, 1]
        yd = (yp >= 0.5).astype(int)
        
        auc = roc_auc_score(yv, yp)
        f1 = f1_score(yv, yd, zero_division=0)
        prec = precision_score(yv, yd, zero_division=0)
        rec = recall_score(yv, yd, zero_division=0)
        
        fold_metrics.append((auc, f1, prec, rec))
        print(f"  Fold {fold+1}: AUC={auc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
    
    avg = np.mean(fold_metrics, axis=0)
    print(f"\n  平均: AUC={avg[0]:.4f}, F1={avg[1]:.4f}, Prec={avg[2]:.4f}, Rec={avg[3]:.4f}")
    
    # 全量训练
    print("\n" + "=" * 60)
    print("全量训练中...")
    model = XGBClassifier(**PARAMS)
    model.fit(X_all, y_all, verbose=100)
    
    with open(MODEL_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump({"model": model, "PARAMS": PARAMS, "metrics": avg.tolist()}, f)
    print(f"\n✅ 模型已保存到 {MODEL_DIR / 'xgboost_model.pkl'}")
    
    # 特征重要性
    imp = model.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    print(f"\nTop 20 特征重要性:")
    for r, idx in enumerate(top):
        print(f"  {r+1:2d}. feature_{idx:3d}  {imp[idx]:.4f}")


if __name__ == "__main__":
    main()
