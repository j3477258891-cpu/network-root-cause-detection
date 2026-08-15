"""
02_features.py - 节点特征工程
===========================
从知识图谱中提取每个节点的特征，构建训练/预测数据集。
支持 train 模式（生成标签）和 predict 模式（无标签）。

运行方式：
  python 02_features.py          # 生成训练特征
  python 02_features.py --test   # 生成测试特征
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# ========== 配置 ==========
TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")
OUTPUT_DIR = Path("D:/zgyidong/code/features")
OUTPUT_DIR.mkdir(exist_ok=True)

# 确定性哈希函数
def stable_hash(s, mod=1000):
    """确定性字符串哈希（跨进程一致）"""
    if not isinstance(s, str) or not s:
        return 0
    h = 5381
    for c in s:
        h = ((h << 5) + h) + ord(c)
        h = h & 0xFFFFFFFF
    return h % mod


def build_vocab(train_dirs, field, topn=50):
    """从训练集构建类别词汇表（按频率取 top-n）"""
    cnt = Counter()
    for wo in train_dirs[:500]:
        p = wo / f"{wo.name}.log.topo.json"
        if not p.exists(): continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        for n in obj.get("nodes", []):
            v = n.get(field, "")
            if isinstance(v, str) and v:
                cnt[v] += 1
    return {v: i+1 for i, (v, _) in enumerate(cnt.most_common(topn))}


def extract_features(nodes, edges, fault_time, vocab):
    """
    从知识图谱中提取所有节点的特征。
    返回 (features_matrix, node_rids, node_classes)
    features_matrix: (N, D) float32
    """
    # 安全类型转换
    def safe_int(v, default=0):
        try: return int(v)
        except: return default
    
    fault_time = safe_int(fault_time)
    
    # 构建索引
    rid2idx = {n["@rid"]: i for i, n in enumerate(nodes)}
    
    # 构建邻接表
    adj = defaultdict(set)
    for e in edges:
        s = e.get("in", e.get("@from", ""))
        t = e.get("out", e.get("@to", ""))
        if s in rid2idx and t in rid2idx:
            si, ti = rid2idx[s], rid2idx[t]
            adj[si].add(ti)
            adj[ti].add(si)
    
    # 预计算 TargetAlarm 节点集合
    target_rids = set(n["@rid"] for n in nodes if n.get("label") == "TargetAlarm")
    target_idxs = set(rid2idx[r] for r in target_rids if r in rid2idx)
    
    features = []
    rids = []
    n_nodes = len(nodes)
    
    for i, node in enumerate(nodes):
        feats = []
        cls = node.get("@class", "Unknown")
        is_alarm = 1 if cls == "Alarm" else 0
        
        # ---- 基础属性 (7维) ----
        feats.append(is_alarm)
        feats.append(stable_hash(cls, 50))                          # class 哈希
        feats.append(1 if node.get("label") == "TargetAlarm" else 0)
        feats.append(stable_hash(node.get("vendor_name", node.get("vendor", "")) if not is_alarm else node.get("vendor", ""), 20))
        
        # ---- 图拓扑 (6维) ----
        neighbors = adj.get(i, set())
        deg = len(neighbors)
        feats.append(min(deg, 100))                                 # 度（截断）
        
        # 邻居类型统计
        n_alarm, n_target = 0, 0
        for nb in neighbors:
            if nb < n_nodes:
                nc = nodes[nb].get("@class", "")
                if nc == "Alarm":
                    n_alarm += 1
                if nodes[nb].get("label") == "TargetAlarm":
                    n_target += 1
        feats.append(n_alarm / max(deg, 1))                         # 邻居中 Alarm 比例
        feats.append(min(n_target, 20))                             # 邻居中 TargetAlarm 数
        feats.append(1 if i in target_idxs else 0)                  # 本身是 TargetAlarm
        
        # 与 TargetAlarm 的距离（简化：1跳内是否可达）
        reachable = 0
        for nb in neighbors:
            if nb in target_idxs:
                reachable = 1
                break
        feats.append(reachable)
        
        # 是否是孤立节点
        feats.append(1 if deg == 0 else 0)
        
        # ---- Alarm 节点特有特征 (10维) ----
        if is_alarm:
            # 告警标题
            title = node.get("title", "")
            feats.append(stable_hash(title, 100))
            
            # 相对时间（秒 → 分钟）
            t = safe_int(node.get("time", fault_time))
            td = max(0, (fault_time - t) / 60000)                  # 分钟
            feats.append(min(td, 1440))                             # 截断到 24 小时
            
            # timeLists 统计（30分钟滑窗，6个窗口）
            tl = node.get("timeLists", [])
            if isinstance(tl, list) and len(tl) > 0:
                tla = np.array(tl, dtype=np.float32)
                feats.append(float(np.sum(tla)))                    # 总告警频次
                feats.append(float(np.mean(tla)))                   # 平均
                feats.append(float(tla[-1]))                         # 最近窗口
                feats.append(float(tla[-2]) if len(tla) > 1 else 0) # 次近窗口
                feats.append(float(len([x for x in tl if x > 0])))  # 有告警的窗口数
            else:
                feats.extend([0, 0, 0, 0, 0])
            
            # 原因关键词
            reason = node.get("reason", "")
            feats.append(1 if "市电" in reason or "供电" in reason else 0)
            feats.append(1 if "故障" in reason else 0)
            feats.append(1 if "传输" in reason else 0)
            
        else:
            feats.extend([0] * 10)
        
        # ---- Device 节点特有特征 (6维) ----
        if not is_alarm:
            feats.append(stable_hash(node.get("device_type", ""), 50))
            feats.append(stable_hash(node.get("service_level", ""), 10))
            feats.append(stable_hash(node.get("zh_label", "")[:30], 100))
            
            # 依赖关系
            in_dep = node.get("in_dependon", node.get("in_dependOn", []))
            out_dep = node.get("out_dependon", node.get("out_dependOn", []))
            fed = len(in_dep) if isinstance(in_dep, list) else 0
            fed2 = len(out_dep) if isinstance(out_dep, list) else 0
            feats.append(min(fed, 50))
            feats.append(min(fed2, 50))
            feats.append(min(fed + fed2, 100))
        else:
            feats.extend([0] * 6)
        
        features.append(feats)
        rids.append(node["@rid"])
    
    return np.array(features, dtype=np.float32), rids


def build_dataset(dirs, vocab, mode="train"):
    """构建完整数据集"""
    all_X, all_y, all_meta = [], [], []
    
    for wo in dirs:
        try:
            topo_path = wo / f"{wo.name}.log.topo.json"
            rc_path = wo / f"{wo.name}.rootcause.json"
            
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            nodes = obj["nodes"]
            edges = obj.get("edges", [])
            fault_time = obj.get("time", 0)
            
            X, rids = extract_features(nodes, edges, fault_time, vocab)
            
            # 标签
            y = np.zeros(len(nodes), dtype=np.int32)
            if mode == "train" and rc_path.exists():
                rc = json.loads(rc_path.read_text(encoding="utf-8"))
                rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
                for i, rid in enumerate(rids):
                    if rid in rc_rids:
                        y[i] = 1
            
            all_X.append(X)
            all_y.append(y)
            all_meta.append({"wo_id": wo.name, "rids": rids, "n_nodes": len(rids)})
            
        except Exception as e:
            print(f"  ⚠ 跳过 {wo.name}: {e}")
    
    return all_X, all_y, all_meta


def _build_global_vocab():
    """构建全局词汇表”（目前用 stable_hash 不需要）"""
    return {}


if __name__ == "__main__":
    import sys
    vocab = _build_global_vocab()
    
    if "--test" in sys.argv:
        test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
        print(f"生成测试集特征... ({len(test_dirs)} 工单)")
        Xs, ys, meta = build_dataset(test_dirs, vocab, "test")
        
        with open(OUTPUT_DIR / "test_features.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta}, f)
        
        total = sum(m["n_nodes"] for m in meta)
        print(f"✅ 测试特征已保存 (工单:{len(Xs)}, 总节点:{total})")
    else:
        train_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"生成训练集特征... ({len(train_dirs)} 工单)")
        Xs, ys, meta = build_dataset(train_dirs, vocab, "train")
        
        total = sum(m["n_nodes"] for m in meta)
        pos = sum(y.sum() for y in ys)
        dim = Xs[0].shape[1] if Xs else 0
        print(f"总节点: {total}, 正样本: {pos} ({pos/max(total,1)*100:.2f}%), 特征维: {dim}")
        
        with open(OUTPUT_DIR / "train_features.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta, "feat_dim": dim}, f)
        print(f"✅ 训练特征已保存")
