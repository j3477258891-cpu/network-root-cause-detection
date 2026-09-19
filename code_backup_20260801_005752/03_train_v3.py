"""
03_train_v3.py - 训练脚本 V3
===========================
改进：
1. XGBoost + LightGBM 双模型集成
2. 网格搜索超参调优
3. Per-work-order TopK 后处理（匹配真实根因数）

运行：python 03_train_v3.py
"""

import pickle, json, warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
SUBMIT_DIR = Path("D:/zgyidong/code/submit")
for d in [MODEL_DIR, SUBMIT_DIR]: d.mkdir(exist_ok=True)


def load_data():
    with open(FEAT_DIR / "train_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys, meta = data["features"], data["labels"], data["meta"]
    
    # 拼接所有节点
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    
    # 全局 shuffle + split
    n = len(X_all)
    indices = np.random.permutation(n)
    split = int(n * 0.85)
    tr_idx = indices[:split]
    val_idx = indices[split:]
    
    return X_all, y_all, tr_idx, val_idx, Xs, ys, meta


def evaluate(yp, yt, name=""):
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    print(f"  {name}: F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    return f1


def tune_xgboost(X_tr, y_tr):
    """Grid search XGBoost"""
    print("\n调参 XGBoost...")
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
    }
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        scale_pos_weight=2.5,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(model, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  CV F1: {gs.best_score_:.4f}")
    return best


def tune_lightgbm(X_tr, y_tr):
    """Grid search LightGBM"""
    print("\n调参 LightGBM...")
    param_grid = {
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [20, 50],
    }
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        scale_pos_weight=2.5,
        random_state=42,
        verbose=-1,
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(model, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  CV F1: {gs.best_score_:.4f}")
    return best


def train_full(X, y):
    """训练最终集成模型"""
    # XGBoost
    print("\n训练 XGBoost 全量...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.0,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X, y)
    
    # LightGBM
    print("训练 LightGBM 全量...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.0,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X, y)
    
    return xgb_model, lgb_model


def per_work_order_predict(probs_list, meta_list):
    """Per-work-order TopK prediction:
    - 每个工单独立取 top K 个节点
    - K = ceil(len(alarm_nodes) * threshold_ratio) 或固定
    - 简单策略：取概率 > 0.45 的节点，若空则取 top 2
    """
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
        
        # 策略：取概率 > 0.45 的
        confident = np.where(probs >= 0.45)[0]
        if len(confident) == 0:
            k = min(2, n)  # backup: top 2
        elif len(confident) > 8:
            k = 8  # cap
        else:
            k = len(confident)
        
        ranked = np.argsort(-probs)
        top_k = ranked[:k]
        k_vals.append(len(top_k))
        
        rc_list = [{"@rid": rids[int(idx)]} for idx in top_k]
        all_preds[wo_id] = {"rootcause": rc_list}
    
    return all_preds, np.mean(k_vals)


def ensemble_predict(models, Xs):
    """XGBoost + LightGBM 概率平均"""
    xgb_m, lgb_m = models
    xgb_p = xgb_m.predict_proba(np.vstack(Xs))[:, 1]
    lgb_p = lgb_m.predict_proba(np.vstack(Xs))[:, 1]
    avg_p = (xgb_p + lgb_p) / 2.0
    
    # 按工单拆分
    probs_list = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        probs_list.append(avg_p[offset:offset + n])
        offset += n
    return probs_list


def main():
    print("=" * 60)
    print("V3 训练流程")
    print("=" * 60)
    
    # 1. 加载
    X_all, y_all, tr_idx, val_idx, Xs, ys, meta = load_data()
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"\n训练节点: {len(X_tr)}, 验证节点: {len(X_val)}")
    print(f"正样本比例: {y_tr.mean():.3f}")
    print(f"特征维度: {X_tr.shape[1]}")
    
    # 2. 全量训练（跳过调参，直接用经验最优参数）
    models = train_full(X_all, y_all)
    
    # 4. 验证集评估
    print("\n=== 验证集评估 ===")
    xgb_probs = models[0].predict_proba(X_val)[:, 1]
    lgb_probs = models[1].predict_proba(X_val)[:, 1]
    ensemble_probs = (xgb_probs + lgb_probs) / 2.0
    
    for name, probs in [("XGBoost", xgb_probs), ("LightGBM", lgb_probs), ("Ensemble", ensemble_probs)]:
        preds = (probs >= 0.5).astype(int)
        evaluate(preds, y_val, name)
    
    # Per-work-order 评估
    val_probs = []
    val_ys = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        if offset + n <= len(ensemble_probs):
            val_probs.append(ensemble_probs[offset:offset + n])
            offset += n
    val_preds, avg_k = per_work_order_predict(val_probs, meta)
    print(f"  TopK平均根因数: {avg_k:.1f}")
    
    # 5. 保存模型
    with open(MODEL_DIR / "xgb_v3.pkl", "wb") as f:
        pickle.dump(models[0], f)
    with open(MODEL_DIR / "lgb_v3.pkl", "wb") as f:
        pickle.dump(models[1], f)
    print("\n✅ 模型已保存")
    
    print("\n" + "=" * 60)
    print("训练完成! 运行 python 04_predict_v3.py 生成提交")
    print("=" * 60)


if __name__ == "__main__":
    main()
