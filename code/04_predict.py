"""
04_predict.py - 测试集预测与提交
================================
对测试集每个工单的节点打分，输出根因预测结果。

运行方式：python 04_predict.py
"""

import json
import pickle
import numpy as np
from pathlib import Path

# ========== 配置 ==========
DATA_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
OUTPUT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # 1. 加载
    print("加载模型和特征...")
    with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    with open(DATA_DIR / "test_features.pkl", "rb") as f:
        data = pickle.load(f)
    
    Xs = data["features"]
    meta = data["meta"]
    print(f"测试工单数: {len(Xs)}")
    
    # 2. 预测
    all_preds = {}
    k_counts = []
    
    for i, (X, m) in enumerate(zip(Xs, meta)):
        if len(X) == 0:
            continue
        
        wo_id = m["wo_id"]
        rids = m["rids"]
        probs = model.predict_proba(X)[:, 1]
        
        # 选概率 > 0.25 的节点，至少选 1 个
        threshold = 0.25
        cand = np.where(probs >= threshold)[0]
        if len(cand) == 0:
            cand = np.argsort(probs)[::-1][:1]
        
        # 按概率排序
        ranked = sorted(cand, key=lambda x: probs[x], reverse=True)
        
        # 限制数量（最多12个）
        top_k = ranked[:min(len(ranked), 12)]
        k_counts.append(len(top_k))
        
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
                        title = n.get("title", n.get("zh_label", ""))
                        location = n.get("location", n.get("positon", ""))
                        reason = n.get("reason", "")
                        break
            
            rc_list.append({
                "@rid": rid,
                "title": title,
                "location": location,
                "reason": reason,
            })
        
        all_preds[wo_id] = {"rootcause": rc_list}
        
        # 进度
        if (i+1) % 100 == 0:
            avg_k = np.mean(k_counts[-100:])
            print(f"  [{i+1}/{len(Xs)}] 平均预测根因数: {avg_k:.1f}")
    
    # 3. 保存
    out_path = OUTPUT_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 预测完成!")
    print(f"   工单数: {len(all_preds)}")
    print(f"   平均预测根因数: {np.mean(k_counts):.1f}")
    print(f"   提交文件: {out_path}")
    
    # 预览
    print(f"\n前3个工单预览:")
    for wo_id in list(all_preds.keys())[:3]:
        pred = all_preds[wo_id]
        print(f"  {wo_id[:16]}... → {len(pred['rootcause'])} 个根因")
        for rc in pred["rootcause"][:2]:
            print(f"    {rc['@rid'][:50]}... | {rc.get('title','')[:20]}")


if __name__ == "__main__":
    main()
