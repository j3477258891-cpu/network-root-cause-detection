"""
04_predict_v3.py - 测试集预测 V3 (修复版)
==========================================
修复：
1. 补全 title/location/reason（从测试集 topo.json 取）
2. 提高阈值（0.55），上限降到5，使根因数更接近真实分布

运行：python 04_predict_v3.py
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

THRESHOLD = 0.55
MAX_K = 5

def build_node_lookup(wo_dir):
    """读取 topo.json，构建 rid → {title, location, reason} 映射"""
    topo_path = wo_dir / f"{wo_dir.name}.log.topo.json"
    if not topo_path.exists():
        return {}
    obj = json.loads(topo_path.read_text(encoding="utf-8"))
    lookup = {}
    for nd in obj.get("nodes", []):
        lookup[nd["@rid"]] = {
            "title": nd.get("title", ""),
            "location": nd.get("location", ""),
            "reason": nd.get("reason", ""),
        }
    return lookup


def per_work_order_predict(probs_list, meta_list, test_dir):
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
        
        confident = np.where(probs >= THRESHOLD)[0]
        if len(confident) == 0:
            k = min(2, n)
        elif len(confident) > MAX_K:
            k = MAX_K
        else:
            k = len(confident)
        
        ranked = np.argsort(-probs)
        top_k = ranked[:k]
        k_vals.append(len(top_k))
        
        # 读取节点详细信息
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
    print("加载模型...")
    with open(MODEL_DIR / "xgb_v3.pkl", "rb") as f:
        xgb_m = pickle.load(f)
    with open(MODEL_DIR / "lgb_v3.pkl", "rb") as f:
        lgb_m = pickle.load(f)
    
    print("加载测试特征...")
    with open(FEAT_DIR / "test_features_v3.pkl", "rb") as f:
        data = pickle.load(f)
    Xs, meta = data["features"], data["meta"]
    
    print(f"预测 {len(Xs)} 个测试工单 (阈值={THRESHOLD}, 上限={MAX_K})...")
    
    xgb_p = xgb_m.predict_proba(np.vstack(Xs))[:, 1]
    lgb_p = lgb_m.predict_proba(np.vstack(Xs))[:, 1]
    avg_p = (xgb_p + lgb_p) / 2.0
    
    probs_list = []
    offset = 0
    for Xi in Xs:
        n = len(Xi)
        probs_list.append(avg_p[offset:offset + n])
        offset += n
    
    preds, k_vals = per_work_order_predict(probs_list, meta, TEST_DIR)
    
    # 校验
    empty = sum(1 for v in preds.values() if len(v["rootcause"]) == 0)
    missing_fields = 0
    for v in preds.values():
        for r in v["rootcause"]:
            if not r.get("title") and not r.get("location") and not r.get("reason"):
                missing_fields += 1
    
    # 保存：官方文件名 test_result.json
    out_path = SUBMIT_DIR / "test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    
    # 同时备份 predictions_v3.json
    backup_path = SUBMIT_DIR / "predictions_v3.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    
    # 同时生成 per-work-order 目录结构（备用格式）
    per_wo_dir = SUBMIT_DIR / "per_workorder"
    per_wo_dir.mkdir(exist_ok=True)
    for wo_id, v in preds.items():
        wo_path = per_wo_dir / f"{wo_id}.rootcause.json"
        with open(wo_path, "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False, indent=2)
    
    from collections import Counter
    dist = Counter(len(v["rootcause"]) for v in preds.values())
    
    print(f"\n✅ V3 预测完成!")
    print(f"   工单: {len(preds)}, 空预测: {empty}")
    print(f"   平均根因数: {np.mean(k_vals):.1f}")
    print(f"   缺字段: {missing_fields}")
    print(f"   分布: {dict(sorted(dist.items()))}")
    print(f"   官方文件: {out_path}")
    print(f"   备份文件: {backup_path}")
    print(f"   逐工单文件: {per_wo_dir}/")


if __name__ == "__main__":
    main()
