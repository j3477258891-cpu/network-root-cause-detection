"""
04_predict_v2.py - 测试集预测 V2
================================
改进：
1. 只对 Alarm 节点预测
2. 自适应阈值（根据概率分布自动确定 k）

运行方式：python 04_predict_v2.py
"""

import json, pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
OUTPUT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("加载 V2 模型和特征...")
    with open(MODEL_DIR / "xgboost_v2.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    with open(DATA_DIR / "test_features_v2.pkl", "rb") as f:
        data = pickle.load(f)
    
    Xs, meta = data["features"], data["meta"]
    print(f"测试工单: {len(Xs)}")
    
    all_preds = {}
    k_vals = []
    
    for i, (X, m) in enumerate(zip(Xs, meta)):
        if len(X) == 0:
            continue
        
        wo_id = m["wo_id"]
        rids = m["rids"]
        probs = model.predict_proba(X)[:, 1]
        
        # 自适应阈值：概率 > 0.5 硬阈值 或 top-5 保底
        # 动态 k = min(max(1, count(prob>0.5)), 5)
        confident = np.where(probs >= 0.5)[0]
        
        if len(confident) == 0:
            # 保底：取概率最高的
            k = 1
        elif len(confident) > 5:
            k = 5
        else:
            k = len(confident)
        
        ranked = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
        top_k = ranked[:min(k, len(ranked))]
        k_vals.append(len(top_k))
        
        # 获取节点详情
        rc_list = []
        topo_path = TEST_DIR / wo_id / f"{wo_id}.log.topo.json"
        topo_cache = None
        
        for idx in top_k:
            rid = rids[idx]
            title, location, reason = "", "", ""
            if topo_path.exists():
                if topo_cache is None:
                    topo_cache = json.loads(topo_path.read_text(encoding="utf-8"))
                for n in topo_cache.get("nodes", []):
                    if n["@rid"] == rid:
                        title = n.get("title", "")
                        location = n.get("location", "")
                        reason = n.get("reason", "")
                        break
            rc_list.append({"@rid": rid, "title": title, "location": location, "reason": reason})
        
        all_preds[wo_id] = {"rootcause": rc_list}
        
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(Xs)}] avg_k={np.mean(k_vals[-100:]):.1f}")
    
    out = OUTPUT_DIR / "predictions_v2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 预测完成 V2!")
    print(f"   工单: {len(all_preds)}, 平均根因数: {np.mean(k_vals):.1f}")
    print(f"   文件: {out}")
    
    # 预览
    print(f"\n样本预览:")
    for wo_id in list(all_preds.keys())[:3]:
        p = all_preds[wo_id]
        print(f"  {wo_id[:16]}... → {len(p['rootcause'])}个: {', '.join(rc['title'][:8] for rc in p['rootcause'])}")


if __name__ == "__main__":
    main()
