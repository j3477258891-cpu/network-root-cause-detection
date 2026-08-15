"""
01_read_data.py - 数据加载和探索
===========================
加载训练集和测试集，统计基本信息，验证数据完整性。
运行方式：python 01_read_data.py
"""

import json
import os
from pathlib import Path
from collections import Counter

# ========== 配置 ==========
TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")

# ========== 加载数据 ==========
def load_work_order(wo_dir):
    """加载单个工单的拓扑和根因数据"""
    wo_name = wo_dir.name
    topo_path = wo_dir / f"{wo_name}.log.topo.json"
    rc_path = wo_dir / f"{wo_name}.rootcause.json"
    
    topo = json.loads(topo_path.read_text(encoding="utf-8")) if topo_path.exists() else None
    rc = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.exists() else None
    return topo, rc


def explore():
    """探索数据集，输出统计信息"""
    train_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
    
    print(f"=" * 60)
    print(f"训练集工单数: {len(train_dirs)}")
    print(f"测试集工单数: {len(test_dirs)}")
    print(f"=" * 60)
    
    # 统计数据
    node_counts, edge_counts = [], []
    rc_counts, target_alarm_counts = [], []
    node_classes = Counter()
    rc_reasons = Counter()
    
    # 检查：rootcause 节点类型分布
    rc_is_alarm = 0
    rc_total = 0
    
    for wo in train_dirs:
        topo, rc = load_work_order(wo)
        if not topo or not rc:
            continue
        
        nodes = topo.get("nodes", [])
        edges = topo.get("edges", [])
        rootcauses = rc.get("rootcause", [])
        
        node_counts.append(len(nodes))
        edge_counts.append(len(edges))
        rc_counts.append(len(rootcauses))
        
        # 统计 TargetAlarm
        target_cnt = sum(1 for n in nodes if n.get("label") == "TargetAlarm")
        target_alarm_counts.append(target_cnt)
        
        # 统计节点类型
        for n in nodes:
            node_classes[n.get("@class", "Unknown")] += 1
        
        # 统计 rootcause 原因类型
        for r in rootcauses:
            rc_reasons[r.get("title", "Unknown")[:20]] += 1
            
            # 检查 rootcause 是否在图中
            rc_rid = r["@rid"]
            rc_total += 1
            matching_nodes = [n for n in nodes if n["@rid"] == rc_rid]
            if matching_nodes:
                n = matching_nodes[0]
                if n.get("@class") == "Alarm":
                    rc_is_alarm += 1
    
    print(f"\n{'='*60}")
    print("图规模统计")
    print(f"{'='*60}")
    print(f"平均节点数: {sum(node_counts)/len(node_counts):.0f} (min={min(node_counts)}, max={max(node_counts)})")
    print(f"平均边数:   {sum(edge_counts)/len(edge_counts):.0f} (min={min(edge_counts)}, max={max(edge_counts)})")
    print(f"平均根因数: {sum(rc_counts)/len(rc_counts):.1f} (min={min(rc_counts)}, max={max(rc_counts)})")
    print(f"平均TargetAlarm数: {sum(target_alarm_counts)/len(target_alarm_counts):.1f} (min={min(target_alarm_counts)}, max={max(target_alarm_counts)})")
    
    print(f"\n根因节点中 Alarm 类型占比: {rc_is_alarm}/{rc_total} = {rc_is_alarm/rc_total*100:.1f}%")
    
    print(f"\n{'='*60}")
    print("节点类型分布 (Top 15)")
    print(f"{'='*60}")
    for cls, cnt in node_classes.most_common(15):
        print(f"  {cls:<30s} {cnt:>6d}")
    
    print(f"\n{'='*60}")
    print("根因 title 分布 (Top 10)")
    print(f"{'='*60}")
    for title, cnt in rc_reasons.most_common(10):
        print(f"  {title:<30s} {cnt:>4d}")
    
    # 检查数据完整性
    print(f"\n{'='*60}")
    print("数据完整性检查")
    print(f"{'='*60}")
    
    train_ok = 0
    for wo in train_dirs:
        topo, rc = load_work_order(wo)
        if topo and rc and "nodes" in topo and "edges" in topo and "rootcause" in rc:
            train_ok += 1
    print(f"训练集有效工单: {train_ok}/{len(train_dirs)}")
    
    test_ok = 0
    for wo in test_dirs:
        topo_path = wo / f"{wo.name}.log.topo.json"
        if topo_path.exists():
            try:
                topo = json.loads(topo_path.read_text(encoding="utf-8"))
                if "nodes" in topo and "edges" in topo:
                    test_ok += 1
            except:
                pass
    print(f"测试集有效工单: {test_ok}/{len(test_dirs)}")
    
    return {
        "train_count": len(train_dirs),
        "test_count": len(test_dirs),
        "avg_nodes": sum(node_counts) / len(node_counts),
        "avg_edges": sum(edge_counts) / len(edge_counts),
        "avg_rootcauses": sum(rc_counts) / len(rc_counts),
        "rc_alarm_ratio": rc_is_alarm / rc_total if rc_total else 0,
    }


if __name__ == "__main__":
    stats = explore()
    print(f"\n✅ 数据探索完成！")
