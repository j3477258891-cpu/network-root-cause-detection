"""
04_predict_v4.py - 测试集预测 V4（三模型集成 + 自适应 TopK）
"""

import pickle, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
SUBMIT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
SUBMIT_DIR.mkdir(exist_ok=True)


def build_node_lookup(wo_dir):
    topo_path = wo_dir / f"{wo_dir.name}.log.topo.json"
    if not topo_path.exists():
        return {}
    obj = json.loads(topo_path.read_text(encoding="utf-8"))
    return {n["@rid"]: {
        "title": n.get("title", ""),
        "location": n.get("location", ""),
        "reason": n.get("reason", ""),
    } for n in obj.get("nodes", [])}


def adaptive_topk_predict(probs_list, meta_list, test_dir):
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

        # 自适应阈值
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

        wo_dir = test_dir / wo_id
        node_lookup = build_node_lookup(wo_dir)

        rc_list = []
        for idx in top_k:
            rid = rids[int(idx)]
            info = node_lookup.get(rid, {})
            rc_list.append({
                "@rid": rid,
                "title": info.get("title", ""),
                "location": info.get("location", ""),
                "reason": info.get("reason", ""),
            })
        all_preds[wo_id] = {"rootcause": rc_list}

    return all_preds, k_vals


def main():
    print("加载三模型...")
    with open(MODEL_DIR / "xgb_v4.pkl", "rb") as f:
        xgb_m = pickle.load(f)
    with open(MODEL_DIR / "lgb_v4.pkl", "rb") as f:
        lgb_m = pickle.load(f)
    with open(MODEL_DIR / "cb_v4.pkl", "rb") as f:
        cb_m = pickle.load(f)

    print("加载测试特征...")
    with open(FEAT_DIR / "test_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, meta = data["features"], data["meta"]

    print(f"预测 {len(Xs)} 个测试工单...")

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

    preds, k_vals = adaptive_topk_predict(probs_list, meta, TEST_DIR)

    out_path = SUBMIT_DIR / "result_record.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    from collections import Counter
    dist = Counter(len(v["rootcause"]) for v in preds.values())

    print(f"\n✅ V4 预测完成!")
    print(f"   工单: {len(preds)}")
    print(f"   平均根因数: {np.mean(k_vals):.2f}")
    print(f"   分布: {dict(sorted(dist.items()))}")
    print(f"   文件: {out_path}")


if __name__ == "__main__":
    main()