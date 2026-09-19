"""
04_predict_v5.py - 测试集预测 V5
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


def build_node_lookup(wo_dir):
    topo_path = wo_dir / f"{wo_dir.name}.log.topo.json"
    if not topo_path.exists(): return {}
    obj = json.loads(topo_path.read_text(encoding="utf-8"))
    return {n["@rid"]: {"title": n.get("title",""), "location": n.get("location",""),
                        "reason": n.get("reason","")} for n in obj.get("nodes", [])}


def main():
    print("加载 5 组模型...")
    with open(MODEL_DIR / "v5_ensemble.pkl", "rb") as f:
        all_models = pickle.load(f)
    models_list = all_models["models"]

    with open(FEAT_DIR / "test_features_v5.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, meta = data["features"], data["meta"]

    print(f"预测 {len(Xs)} 个测试工单...")

    # 多种子平均
    all_probs = []
    for xgb_m, lgb_m, cb_m in models_list:
        p = (xgb_m.predict_proba(np.vstack(Xs))[:, 1] +
             lgb_m.predict_proba(np.vstack(Xs))[:, 1] +
             cb_m.predict_proba(np.vstack(Xs))[:, 1]) / 3.0
        all_probs.append(p)
    p_avg = np.mean(all_probs, axis=0)

    probs_list = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        probs_list.append(p_avg[offset:offset + n])
        offset += n

    # 自适应 TopK
    preds = {}
    k_vals = []
    for probs, meta_item in zip(probs_list, meta):
        wo_id = meta_item["wo_id"]
        rids = meta_item["rids"]
        n = len(probs)
        if n == 0:
            preds[wo_id] = {"rootcause": []}
            k_vals.append(0)
            continue

        sorted_probs = np.sort(probs)[::-1]
        K = 1
        for i in range(1, min(n, 9)):
            if i < len(sorted_probs) and sorted_probs[i - 1] - sorted_probs[i] > 0.05:
                K = i; break
            K = i + 1
        if K == 1 and probs.max() < 0.3:
            K = 1
        elif K > 8:
            K = 8

        ranked = np.argsort(-probs)
        top_k = ranked[:K]
        k_vals.append(len(top_k))

        node_lookup = build_node_lookup(TEST_DIR / wo_id)
        rc_list = []
        for idx in top_k:
            rid = rids[int(idx)]
            info = node_lookup.get(rid, {})
            rc_list.append({
                "@rid": rid, "title": info.get("title",""),
                "location": info.get("location",""), "reason": info.get("reason",""),
            })
        preds[wo_id] = {"rootcause": rc_list}

    out_path = SUBMIT_DIR / "result_record.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    dist = Counter(len(v["rootcause"]) for v in preds.values())
    print(f"\n✅ V5 预测完成!")
    print(f"   工单: {len(preds)}, 平均根因数: {np.mean(k_vals):.2f}")
    print(f"   分布: {dict(sorted(dist.items()))}")
    print(f"   文件: {out_path}")


if __name__ == "__main__":
    main()