"""
02_features_v3.py - 节点特征工程 V3
===================================
V2 → V3 改进：
1. Target encoding：用 title/device/vendor/location 在训练集中的根因比例替代纯哈希
2. timeLists 全 6 窗口原始值 + 趋势特征（斜率、峰值位置）
3. 邻居设备类型分布特征
4. 图结构特征：介数中心性近似

运行：python 02_features_v3.py          # 训练特征
      python 02_features_v3.py --test   # 测试特征
"""

import json, pickle, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")
OUTPUT_DIR = Path("D:/zgyidong/code/features")
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_int(v, default=0):
    try: return int(v)
    except: return default


def stable_hash(s, mod=1000):
    if not isinstance(s, str) or not s: return 0
    h = 5381
    for c in s: h = ((h << 5) + h) + ord(c); h &= 0xFFFFFFFF
    return h % mod


def _simple_pagerank(adj, n, iters=10, d=0.85):
    pr = np.ones(n) / n
    deg = np.array([len(adj.get(i, set())) for i in range(n)])
    deg[deg == 0] = 1  # avoid div by zero
    for _ in range(iters):
        new_pr = np.ones(n) * (1 - d) / n
        for u, vset in adj.items():
            if u < n and len(vset) > 0:
                share = d * pr[u] / len(vset)
                for v in vset:
                    if v < n: new_pr[v] += share
        pr = new_pr
    return pr


def _build_target_encodings(train_dirs):
    """从训练集中统计各字段的根因比例，用于 target encoding"""
    stats = {
        "title": defaultdict(lambda: [0, 0]),
        "device": defaultdict(lambda: [0, 0]),
        "vendor": defaultdict(lambda: [0, 0]),
        "location": defaultdict(lambda: [0, 0]),
        "room": defaultdict(lambda: [0, 0]),
        "fault_key": defaultdict(lambda: [0, 0]),
    }
    
    for wo in train_dirs:
        try:
            topo_path = wo / f"{wo.name}.log.topo.json"
            rc_path = wo / f"{wo.name}.rootcause.json"
            if not topo_path.exists() or not rc_path.exists():
                continue
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
            
            for nd in obj["nodes"]:
                if nd.get("@class") != "Alarm":
                    continue
                is_rc = 1 if nd["@rid"] in rc_rids else 0
                for k in ["title", "device", "vendor", "location", "room"]:
                    v = nd.get(k, "") or ""
                    stats[k][v][0] += is_rc
                    stats[k][v][1] += 1
                fk = nd.get("fault1", "") or nd.get("fault2", "")
                stats["fault_key"][fk][0] += is_rc
                stats["fault_key"][fk][1] += 1
        except:
            pass
    
    # 转为概率，未出现的给全局均值
    encodings = {}
    for k, d in stats.items():
        total_pos = sum(v[0] for v in d.values())
        total_all = sum(v[1] for v in d.values())
        global_rate = total_pos / max(total_all, 1)
        encodings[k] = {key: pos / max(cnt, 1) for key, (pos, cnt) in d.items()}
        encodings[k]["__global__"] = global_rate
    
    return encodings


def extract_features(nodes, edges, fault_time, target_encodings=None):
    """V3 特征提取：增强 + target encoding + 更多图特征"""
    n_nodes = len(nodes)
    rid2idx = {n["@rid"]: i for i, n in enumerate(nodes)}
    
    # 邻接表
    adj = defaultdict(set)
    for e in edges:
        s = e.get("in", "")
        t = e.get("out", "")
        if s in rid2idx and t in rid2idx:
            si, ti = rid2idx[s], rid2idx[t]
            adj[si].add(ti)
            adj[ti].add(si)
    
    ft = safe_int(fault_time)
    
    alarm_indices = [i for i, n in enumerate(nodes) if n.get("@class") == "Alarm"]
    target_set = set(i for i in alarm_indices if nodes[i].get("label") == "TargetAlarm")
    
    pr = _simple_pagerank(adj, n_nodes)
    
    # 介数中心性近似（基于最短路径采样）
    bc = _approx_betweenness(adj, n_nodes)
    
    features, rids = [], []
    
    for i in alarm_indices:
        node = nodes[i]
        f = []
        deg = len(adj.get(i, set()))
        
        # ====== 1. 基础标识 (3维) ======
        f.append(1 if i in target_set else 0)
        f.append(1 if i in target_set and len(target_set) == 1 else 0)
        
        # ====== 2. Target Encoding (6维) ======
        if target_encodings:
            for k in ["title", "device", "vendor", "location", "room"]:
                v = node.get(k, "") or ""
                rate = target_encodings[k].get(v, target_encodings[k]["__global__"])
                f.append(rate)
            fk = node.get("fault1", "") or node.get("fault2", "")
            rate = target_encodings["fault_key"].get(fk, target_encodings["fault_key"]["__global__"])
            f.append(rate)
        else:
            f.extend([0.0]*6)
        
        # ====== 3. 文本哈希特征 (5维) ======
        f.append(stable_hash(node.get("title", ""), 100))
        f.append(stable_hash(node.get("device", ""), 100))
        f.append(stable_hash(node.get("vendor", ""), 15))
        f.append(stable_hash(node.get("location", ""), 100))
        f.append(stable_hash(node.get("room", ""), 20))
        
        # ====== 4. 时序特征 (12维) ======
        t = safe_int(node.get("time", ft))
        td = max(0, (ft - t) / 60000)
        f.append(min(td, 1440))
        
        tl = node.get("timeLists", [])
        if isinstance(tl, list) and len(tl) >= 6:
            tla = np.array(tl[:6], dtype=np.float32)
            f.append(float(np.sum(tla)))
            f.append(float(np.mean(tla)))
            f.append(float(np.std(tla)))
            f.append(float(np.max(tla)))
            f.append(float(np.min(tla)))
            # 6 个窗口原始值
            for v in tla:
                f.append(float(v))
            # 趋势：最后窗口 vs 最大值比例
            f.append(float(tla[-1] / max(np.max(tla), 1e-6)))
            # 峰值窗口位置 (0-5)
            f.append(float(np.argmax(tla)) / 5)
        else:
            f.extend([0]*12)
        
        # ====== 5. 原因关键词 + fault 字段 (7维) ======
        reason = node.get("reason", "")
        f.append(1 if "市电" in reason or "供电" in reason or "电力" in reason else 0)
        f.append(1 if "故障" in reason else 0)
        f.append(1 if "传输" in reason else 0)
        f.append(1 if "光" in reason or "光纤" in reason else 0)
        f.append(len(reason) // 30)
        
        fk1 = node.get("fault1", "")
        fk2 = node.get("fault2", "")
        f.append(stable_hash(fk1 or fk2, 30))
        f.append(len(fk1) + len(fk2))
        
        # ====== 6. 图拓扑 (12维) ======
        f.append(min(deg, 100))
        f.append(float(pr[i]) if i < len(pr) else 0.0)
        f.append(float(bc[i]) if i < len(bc) else 0.0)
        
        nbs = adj.get(i, set())
        n_alarm = sum(1 for nb in nbs if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        n_target = sum(1 for nb in nbs if nb in target_set)
        n_device = len(nbs) - n_alarm
        
        f.append(n_alarm / max(deg, 1))
        f.append(min(n_target, 20))
        f.append(n_device / max(deg, 1))
        
        # 邻居平均 PageRank
        nb_pr = [pr[nb] for nb in nbs if nb < len(pr)]
        f.append(float(np.mean(nb_pr)) if nb_pr else 0.0)
        
        # 2跳统计
        n2 = set()
        for nb in nbs:
            n2.update(adj.get(nb, set()))
        n2.discard(i)
        n2_alarm = sum(1 for nb in n2 if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        f.append(min(len(n2), 200))
        f.append(n2_alarm / max(len(n2), 1))
        
        # PageRank 排名
        if len(pr) > 0 and i < len(pr):
            rank = sum(1 for p in pr if p > pr[i]) / max(len(pr), 1)
        else:
            rank = 0.0
        f.append(rank)
        
        # 局部聚类系数
        nb_list = list(nbs)
        nb_edges = 0
        for u in nb_list:
            for v in nb_list:
                if v in adj.get(u, set()):
                    nb_edges += 1
        nb_edges //= 2
        denom = deg * (deg - 1) / 2
        f.append(nb_edges / max(denom, 1))
        
        features.append(f)
        rids.append(node["@rid"])
    
    return np.array(features, dtype=np.float32), rids, alarm_indices


def _approx_betweenness(adj, n, sample=20):
    """近似介数中心性"""
    bc = np.zeros(n)
    nodes = list(range(n))
    if len(nodes) > sample:
        nodes = np.random.choice(nodes, sample, replace=False)
    
    for s in nodes:
        # BFS
        stack, paths = [], {s: [[s]]}
        dist = {s: 0}
        sigma = {s: 1}
        q = [s]
        while q:
            v = q.pop(0)
            stack.append(v)
            for w in adj.get(v, set()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist.get(w, 999) == dist[v] + 1:
                    sigma[w] = sigma.get(w, 0) + sigma[v]
                    if w not in paths: paths[w] = []
                    for p in paths[v]:
                        paths[w].append(p + [w])
        
        delta = {v: 0 for v in stack}
        while stack:
            w = stack.pop()
            for v in paths.get(w, [[], []]):
                if len(v) >= 2:
                    vv = v[-2]
                    delta[vv] += (sigma[vv] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    
    return bc / bc.max() if bc.max() > 0 else bc


def build_dataset(dirs, mode="train", target_encodings=None):
    all_X, all_y, all_meta = [], [], []
    for wo in dirs:
        try:
            p = wo / f"{wo.name}.log.topo.json"
            obj = json.loads(p.read_text(encoding="utf-8"))
            nodes = obj["nodes"]
            edges = obj.get("edges", [])
            ft = obj.get("time", 0)
            
            X, rids, alarm_idx = extract_features(nodes, edges, ft, target_encodings)
            
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
        # 加载训练时保存的 target encodings
        with open(OUTPUT_DIR / "target_encodings_v3.pkl", "rb") as f:
            target_encodings = pickle.load(f)
        
        dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
        print(f"生成测试特征 V3 ({len(dirs)} 工单)...")
        Xs, ys, meta = build_dataset(dirs, "test", target_encodings)
        with open(OUTPUT_DIR / "test_features_v3.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta}, f)
        total = sum(m["n_alarm"] for m in meta)
        print(f"✅ 测试V3: {len(Xs)}工单, {total} Alarm节点")
    else:
        dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"构建 target encoding ({len(dirs)} 工单)...")
        target_encodings = _build_target_encodings(dirs)
        with open(OUTPUT_DIR / "target_encodings_v3.pkl", "wb") as f:
            pickle.dump(target_encodings, f)
        
        print(f"生成训练特征 V3...")
        Xs, ys, meta = build_dataset(dirs, "train", target_encodings)
        total = sum(m["n_alarm"] for m in meta)
        pos = sum(y.sum() for y in ys)
        dim = Xs[0].shape[1] if Xs else 0
        pos_pct = pos / max(total, 1)
        print(f"✅ 训练V3: Alarm={total}, 正样本={pos} ({pos_pct*100:.1f}%), 特征={dim}维")
        with open(OUTPUT_DIR / "train_features_v3.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta, "feat_dim": dim}, f)
