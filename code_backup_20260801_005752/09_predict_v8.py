"""
09_predict_v8.py - V8 预测
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
    # 加载模型和阈值
    with open(MODEL_DIR / "v8_models.pkl", "rb") as f:
        models = pickle.load(f)
    with open(MODEL_DIR / "v8_threshold.pkl", "rb") as f:
        best_th = pickle.load(f)

    # V3 特征
    with open(FEAT_DIR / "test_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, meta = data["features"], data["meta"]

    print(f"预测 {len(Xs)} 工单, 阈值={best_th:.2f}...")

    # 5种子平均
    all_p = [m.predict_proba(np.vstack(Xs))[:, 1] for m in models]
    p_avg = np.mean(all_p, axis=0)

    preds = {}
    offset = 0
    for Xi, m in zip(Xs, meta):
        n = len(Xi)
        probs = p_avg[offset:offset + n]
        offset += n

        confident = np.where(probs >= best_th)[0]
        ranked = np.argsort(-probs)

        if len(confident) > 0:
            top_k = ranked[:min(len(confident), 5)]
        elif probs.max() >= 0.4:
            top_k = ranked[:1]
        else:
            top_k = ranked[:1]

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

    out = SUBMIT_DIR / "result_record.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    dist = Counter(len(v["rootcause"]) for v in preds.values())
    print(f"✅ V8: 阈值{best_th:.2f}, 平均K={np.mean(list(dist.elements()))/len(preds):.2f}")
    print(f"   文件: {out}")


if __name__ == "__main__":
    main()