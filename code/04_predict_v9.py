"""
04_predict_v9.py - V9 预测（5-seed 集成 + 工单级 K 预测）
"""

import pickle, json, csv, warnings
import numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
SUBMIT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
SUBMIT_DIR.mkdir(exist_ok=True)


def main():
    # 加载模型
    with open(MODEL_DIR / "v9_xgb_5seed.pkl", "rb") as f:
        models = pickle.load(f)
    with open(MODEL_DIR / "v9_k_predictor.pkl", "rb") as f:
        k_model = pickle.load(f)

    # 加载特征
    with open(FEAT_DIR / "test_features_v9.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, meta = data["features"], data["meta"]

    print(f"预测 {len(Xs)} 测试工单...")

    # 5-seed 概率平均
    all_p = [m.predict_proba(np.vstack(Xs))[:, 1] for m in models]
    p_avg = np.mean(all_p, axis=0)

    # 拆回工单 + 工单级 K 预测
    preds = {}
    k_vals = []
    offset = 0

    for Xi, m in zip(Xs, meta):
        n = len(Xi)
        probs = p_avg[offset:offset + n]
        offset += n

        # K 预测器特征
        k_feat = np.array([[
            float(np.mean(probs)),
            float(np.std(probs)) if n > 1 else 0.0,
            float(np.max(probs)),
            float(np.min(probs)),
            float(np.max(probs) - np.min(probs)),
            float(np.sum(probs >= 0.4)),
            float(np.sum(probs >= 0.5)),
            float(np.sum(probs >= 0.6)),
            float(n),
        ]], dtype=np.float32)

        K = int(np.round(k_model.predict(k_feat)[0]))
        K = max(1, min(K, 8))  # 限制范围

        ranked = np.argsort(-probs)
        top_k = ranked[:K]
        k_vals.append(len(top_k))

        # 节点 lookup
        wo_dir = TEST_DIR / m["wo_id"]
        node_lookup = {}
        topo_path = wo_dir / f'{m["wo_id"]}.log.topo.json'
        if topo_path.exists():
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            node_lookup = {n["@rid"]: n for n in obj["nodes"]}

        rc_list = []
        for idx in top_k:
            rid = m["rids"][int(idx)]
            nd = node_lookup.get(rid, {})
            rc_list.append({
                "@rid": rid, "title": nd.get("title", ""),
                "location": nd.get("location", ""), "reason": nd.get("reason", ""),
            })
        preds[m["wo_id"]] = {"rootcause": rc_list}

    # 写入 CSV
    out = SUBMIT_DIR / "result_record.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    dist = Counter(len(v["rootcause"]) for v in preds.values())
    print(f"\n✅ V9 预测完成!")
    print(f"   工单: {len(preds)}, 平均K: {np.mean(k_vals):.2f}")
    print(f"   分布: {dict(sorted(dist.items()))}")
    print(f"   文件: {out}")


if __name__ == "__main__":
    main()