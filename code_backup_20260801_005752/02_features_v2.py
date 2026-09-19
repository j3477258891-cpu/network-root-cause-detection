"""
02_features_v2.py - 节点特征工程 V2
===================================
改进：
1. 只对 Alarm 节点提取特征（根因 100% 是 Alarm）
2. 增强图拓扑特征（PageRank、2跳邻居统计）
3. 增强时序特征（timeLists 全6窗口）

运行方式：python 02_features_v2.py && python 02_features_v2.py --test
"""

import json, pickle, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")
OUTPUT_DIR = Path("D:/zgyidong/code/features")
OUTPUT_DIR.mkdir(exist_ok=True)


def stable_hash(s, mod=1000):
    if not isinstance(s, str) or not s: return 0
    h = 5381
    for c in s: h = ((h << 5) + h) + ord(c); h &= 0xFFFFFFFF
    return h % mod


def safe_int(v, default=0):
    try: return int(v)
    except: return default


def extract_alarm_features(nodes, edges, fault_time):
    """只提取 Alarm 节点的特征，含增强图拓扑"""
    n_nodes = len(nodes)
    rid2idx = {n["@rid"]: i for i, n in enumerate(nodes)}
    
    # 邻接表
    adj = defaultdict(set)
    for e in edges:
        s = e.get("in", e.get("@from", ""))
        t = e.get("out", e.get("@to", ""))
        if s in rid2idx and t in rid2idx:
            si, ti = rid2idx[s], rid2idx[t]
            adj[si].add(ti)
            adj[ti].add(si)
    
    ft = safe_int(fault_time)
    
    # 构建索引：哪些是 Alarm，哪些是 TargetAlarm
    alarm_indices = [i for i, n in enumerate(nodes) if n.get("@class") == "Alarm"]
    target_set = set(i for i in alarm_indices if nodes[i].get("label") == "TargetAlarm")
    
    # 预计算 PageRank（简化版）
    pr = _simple_pagerank(adj, n_nodes)
    
    features, rids = [], []
    
    for i in alarm_indices:
        node = nodes[i]
        f = []
        deg = len(adj.get(i, set()))
        
        # === 1. 节点自身属性 (8维) ===
        f.append(1 if i in target_set else 0)                          # TargetAlarm
        f.append(stable_hash(node.get("title", ""), 100))              # 标题
        f.append(stable_hash(node.get("vendor", ""), 15))              # 厂家
        f.append(stable_hash(node.get("device", ""), 100))             # 设备
        f.append(stable_hash(node.get("location", ""), 100))           # 位置
        f.append(stable_hash(node.get("fault1", node.get("fault2", "")), 30)) # 告警类型
        f.append(stable_hash(node.get("room", ""), 20))                # 机房
        f.append(len(node.get("addInfo", "")) // 100)                  # 补充信息长度
        
        # === 2. 时序特征 (8维) ===
        t = safe_int(node.get("time", ft))
        td = max(0, (ft - t) / 60000)
        f.append(min(td, 1440))                                        # 距故障时间(分)
        
        tl = node.get("timeLists", [])
        if isinstance(tl, list) and len(tl) >= 6:
            tla = np.array(tl[:6], dtype=np.float32)
            f.append(float(np.sum(tla)))                               # 总告警次数
            f.append(float(np.mean(tla)))                              # 平均密度
            f.append(float(np.std(tla)))                               # 波动
            f.append(float(tla[-1]))                                   # 最近窗口(5min前)
            f.append(float(tla[-2]))                                   # 次近窗口(10min前)
            f.append(float(tla[-3]))                                   # 15min前
            f.append(float(np.sum(tla[-3:])))                         # 近15分钟告警数
        else:
            f.extend([0]*7)
        
        # === 3. 原因关键词 (5维) ===
        reason = node.get("reason", "")
        f.append(1 if "市电" in reason or "供电" in reason else 0)
        f.append(1 if "故障" in reason else 0)
        f.append(1 if "传输" in reason else 0)
        f.append(1 if "光" in reason or "光纤" in reason else 0)
        f.append(len(reason) // 20)                                     # 原因长度
        
        # === 4. 图拓扑特征 (8维) ===
        f.append(min(deg, 100))                                        # 度
        f.append(float(pr[i]) if i < len(pr) else 0.0)                # PageRank
        
        # 邻居统计
        nbs = adj.get(i, set())
        n_alarm = sum(1 for nb in nbs if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        n_target = sum(1 for nb in nbs if nb in target_set)
        f.append(n_alarm / max(deg, 1))                                # 邻居中Alarm比例
        f.append(min(n_target, 20))                                    # 连接TargetAlarm数
        
        # 2跳邻居
        n2 = set()
        for nb in nbs:
            n2.update(adj.get(nb, set()))
        n2.discard(i)
        n2_alarm = sum(1 for nb in n2 if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        f.append(min(len(n2), 200))                                    # 2跳邻居数
        f.append(n2_alarm / max(len(n2), 1))                           # 2跳中Alarm比例
        
        # 是否是图中唯一的TargetAlarm
        f.append(1 if i in target_set and len(target_set) == 1 else 0)
        
        # Pagerank 排名百分位
        if len(pr) > 0 and i < len(pr):
            rank = sum(1 for p in pr if p > pr[i])
            f.append(rank / max(len(pr), 1))
        else:
            f.append(0.0)
        
        features.append(f)
        rids.append(node["@rid"])
    
    return np.array(features, dtype=np.float32), rids, alarm_indices


def _simple_pagerank(adj, n, iters=5, d=0.85):
    """简化 PageRank"""
    pr = np.ones(n) / n
    for _ in range(iters):
        new_pr = np.ones(n) * (1 - d) / n
        for u, vset in adj.items():
            if u < n and len(vset) > 0:
                share = d * pr[u] / len(vset)
                for v in vset:
                    if v < n: new_pr[v] += share
        pr = new_pr
    return pr


def build_dataset(dirs, mode="train"):
    all_X, all_y, all_meta = [], [], []
    for wo in dirs:
        try:
            p = wo / f"{wo.name}.log.topo.json"
            obj = json.loads(p.read_text(encoding="utf-8"))
            nodes = obj["nodes"]
            edges = obj.get("edges", [])
            ft = obj.get("time", 0)
            
            X, rids, alarm_idx = extract_alarm_features(nodes, edges, ft)
            
            y = np.zeros(len(alarm_idx), dtype=np.int32)
            if mode == "train":
                rc_path = wo / f"{wo.name}.rootcause.json"
                if rc_path.exists():
                    rc = json.loads(rc_path.read_text(encoding="utf-8"))
                    rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
                    for j, idx in enumerate(alarm_idx):
                        if nodes[idx]["@rid"] in rc_rids:
                            y[j] = 1
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_meta.append({"wo_id": wo.name, "rids": rids, "n_alarm": len(rids)})
        except Exception as e:
            print(f"  ⚠ {wo.name}: {e}")
    return all_X, all_y, all_meta


if __name__ == "__main__":
    if "--test" in sys.argv:
        dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
        print(f"生成测试特征 V2 ({len(dirs)} 工单)...")
        Xs, ys, meta = build_dataset(dirs, "test")
        with open(OUTPUT_DIR / "test_features_v2.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta}, f)
        total = sum(m["n_alarm"] for m in meta)
        print(f"✅ 测试V2: {len(Xs)}工单, {total} Alarm节点")
    else:
        dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"生成训练特征 V2 ({len(dirs)} 工单)...")
        Xs, ys, meta = build_dataset(dirs, "train")
        total = sum(m["n_alarm"] for m in meta)
        pos = sum(y.sum() for y in ys)
        dim = Xs[0].shape[1] if Xs else 0
        pos_pct = sum(y.sum() for y in ys) / max(sum(len(y) for y in ys), 1)
        print(f"Alarm节点: {total}, 正样本: {pos} ({pos_pct*100:.1f}%), 特征维: {dim}")
        with open(OUTPUT_DIR / "train_features_v2.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta, "feat_dim": dim}, f)
        print(f"✅ 训练V2已保存")
