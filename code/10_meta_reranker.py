"""
10_meta_reranker.py - Meta 重排序器
=====================================
思路：
1. 在训练集 held-out 上，用 V9 找每工单的 top-K 候选
2. 训练一个小 XGBoost 判断候选里哪些是真根因
3. 测试集上：Probe 候选 ∪ V9 候选 → 重排序 → 选 TopK

运行：python 10_meta_reranker.py
"""

import pickle, json, csv, warnings
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score
import xgboost as xgb

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
SUBMIT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
SUBMIT_DIR.mkdir(exist_ok=True)


def build_reranker_training_data():
    """在训练集的 held-out 上构建重排序器训练数据"""
    print("构建重排序器训练数据...")

    with open(FEAT_DIR / "train_features_v9.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, ys, meta = data["features"], data["labels"], data["meta"]

    # 80/20 拆分
    np.random.seed(42)
    n = len(Xs)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    tr_X = np.vstack([Xs[i] for i in idx[:split]])
    tr_y = np.concatenate([ys[i] for i in idx[:split]])

    # 训练基模型
    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, scale_pos_weight=2.5,
        tree_method='hist', random_state=42, verbosity=0
    )
    base.fit(tr_X, tr_y)

    # 对 20% held-out 预测
    rerank_X = []
    rerank_y = []

    offset = 0
    for i in idx[split:]:
        X_wo = Xs[i]
        y_wo = ys[i]
        n_alarm = len(X_wo)

        probs = base.predict_proba(X_wo)[:, 1]
        ranked = np.argsort(-probs)

        # 取 top-8 个候选（或全部，如果少於 8）
        top_n = min(n_alarm, 8)
        candidates = ranked[:top_n]

        for pos, cand_idx in enumerate(candidates):
            feats = list(X_wo[cand_idx])  # 128 维 V9 特征
            feats.append(float(probs[cand_idx]))      # 基模型概率
            feats.append(float(pos))                    # 排名位置
            feats.append(float(probs[cand_idx] - probs[ranked[min(pos+1, n_alarm-1)]] if pos < n_alarm-1 else 0))  # 与下一个的差距

            rerank_X.append(feats)
            rerank_y.append(int(y_wo[cand_idx]))

    rerank_X = np.array(rerank_X, dtype=np.float32)
    rerank_y = np.array(rerank_y)

    pos_rate = rerank_y.mean()
    print(f"重排序训练数据: {len(rerank_X)} 样本, 正样本率: {pos_rate:.3f}")

    # 训练重排序器
    reranker = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=max(1.0, 1.0 / max(pos_rate, 0.01)),
        tree_method='hist', random_state=99, verbosity=0
    )
    reranker.fit(rerank_X, rerank_y)

    # 验证
    yp = reranker.predict(rerank_X)
    f1 = f1_score(rerank_y, yp, zero_division=0)
    print(f"重排序器 F1: {f1:.4f}")

    return reranker, base


def load_probe_predictions():
    """加载 Probe (0.905) 预测"""
    with open("D:/zgyidong/codexgz/result_record_probe_swap12_score_0.905373.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {r["order_id"]: json.loads(r["output"])["rootcause"] for r in reader}


def apply_reranker(reranker, base_model):
    """对测试集应用重排序"""
    print("\n应用重排序到测试集...")

    with open(FEAT_DIR / "test_features_v9.pkl", "rb") as f:
        test_data = pickle.load(f)
    Xs, meta = test_data["features"], test_data["meta"]

    probe_preds = load_probe_predictions()

    result = {}
    k_vals = []

    offset = 0
    for Xi, m in zip(Xs, meta):
        wo_id = m["wo_id"]
        n_alarm = len(Xi)
        probs = base_model.predict_proba(Xi)[:, 1]
        ranked = np.argsort(-probs)

        # 候选集合：Probe 的 @rid + V9 top-8
        probe_rids = set(r["@rid"] for r in probe_preds.get(wo_id, []))
        v9_topn = min(n_alarm, 8)
        v9_candidates = ranked[:v9_topn]

        # 构建重排序特征
        all_candidates = {}  # idx → features
        for pos, cand_idx in enumerate(v9_candidates):
            feats = list(Xi[cand_idx])
            feats.append(float(probs[cand_idx]))
            feats.append(float(pos))
            feats.append(float(probs[cand_idx] - probs[ranked[min(pos+1, n_alarm-1)]] if pos < n_alarm-1 else 0))
            all_candidates[cand_idx] = feats

        if not all_candidates:
            result[wo_id] = {"rootcause": []}
            k_vals.append(0)
            continue

        # 重排序打分
        cand_indices = list(all_candidates.keys())
        cand_feats = np.array([all_candidates[i] for i in cand_indices], dtype=np.float32)
        rerank_scores = reranker.predict_proba(cand_feats)[:, 1]

        # 按重排序分数排序
        reranked = sorted(zip(cand_indices, rerank_scores), key=lambda x: -x[1])

        # K = Probe 的 K
        K = len(probe_preds.get(wo_id, []))
        if K <= 0:
            K = 1

        selected_indices = [idx for idx, _ in reranked[:K]]
        k_vals.append(len(selected_indices))

        # 构建输出
        wo_dir = TEST_DIR / wo_id
        node_lookup = {}
        topo_path = wo_dir / f"{wo_id}.log.topo.json"
        if topo_path.exists():
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            node_lookup = {n["@rid"]: n for n in obj["nodes"]}

        rc_list = []
        for idx in selected_indices:
            rid = m["rids"][idx]
            nd = node_lookup.get(rid, {})
            rc_list.append({
                "@rid": rid, "title": nd.get("title", ""),
                "location": nd.get("location", ""), "reason": nd.get("reason", ""),
            })
        result[wo_id] = {"rootcause": rc_list}

    return result, k_vals


def main():
    print("=" * 50)
    print("Meta 重排序器")
    print("=" * 50)

    # 1. 训练重排序器
    reranker, base_model = build_reranker_training_data()

    # 2. 应用到测试集
    result, k_vals = apply_reranker(reranker, base_model)

    # 3. 写入 CSV
    out = SUBMIT_DIR / "result_record.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in result.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    dist = Counter(len(v["rootcause"]) for v in result.values())
    print(f"\n✅ Meta 重排序完成!")
    print(f"   工单: {len(result)}, 平均K: {np.mean(k_vals):.2f}")
    print(f"   分布: {dict(sorted(dist.items()))}")
    print(f"   文件: {out}")


if __name__ == "__main__":
    main()