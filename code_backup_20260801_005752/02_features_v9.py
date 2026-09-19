"""
02_features_v9.py - 特征工程 V9
================================
V8 → V9 改进：
1. Node2Vec 图结构嵌入（64维随机游走）
2. 字符 2-gram 文本特征（20维，抗分布偏移）
3. 保留 V3 全部 45 维特征

总维度：45 + 64 + 20 = 129 维

运行：python 02_features_v9.py && python 02_features_v9.py --test
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

# Node2Vec 参数
WALK_LEN = 20
NUM_WALKS = 10
EMBED_DIM = 64
WINDOW = 5

# 字符 n-gram 参数
NGRAM_N = 2
NGRAM_TOP = 20


def safe_int(v, default=0):
    try: return int(v)
    except: return default


def stable_hash(s, mod=1000):
    if not isinstance(s, str) or not s: return 0
    h = 5381
    for c in s: h = ((h << 5) + h) + ord(c); h &= 0xFFFFFFFF
    return h % mod


def char_ngrams(text, n=2):
    """提取字符 n-gram"""
    if not isinstance(text, str) or len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def _build_ngram_vocab(train_dirs):
    """从训练集收集高频字符 n-gram"""
    counter = Counter()
    for wo in train_dirs:
        try:
            topo = json.loads((wo / f"{wo.name}.log.topo.json").read_text(encoding="utf-8"))
            for n in topo["nodes"]:
                if n.get("@class") != "Alarm": continue
                for field in ["title", "device", "location", "reason"]:
                    txt = n.get(field, "") or ""
                    for gram in char_ngrams(txt, NGRAM_N):
                        counter[gram] += 1
        except: pass
    return set(g for g, _ in counter.most_common(NGRAM_TOP))


def _build_target_encodings(train_dirs):
    stats = {k: defaultdict(lambda: [0,0]) for k in ["title","device","vendor","location","room","fault_key"]}
    for wo in train_dirs:
        try:
            topo = json.loads((wo / f"{wo.name}.log.topo.json").read_text(encoding="utf-8"))
            rc = json.loads((wo / f"{wo.name}.rootcause.json").read_text(encoding="utf-8"))
            rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
            for nd in topo["nodes"]:
                if nd.get("@class") != "Alarm": continue
                is_rc = 1 if nd["@rid"] in rc_rids else 0
                for k in ["title","device","vendor","location","room"]:
                    v = nd.get(k, "") or ""
                    stats[k][v][0] += is_rc; stats[k][v][1] += 1
                fk = nd.get("fault1","") or nd.get("fault2","")
                stats["fault_key"][fk][0] += is_rc; stats["fault_key"][fk][1] += 1
        except: pass
    encodings = {}
    for k, d in stats.items():
        total_pos = sum(v[0] for v in d.values())
        total_all = sum(v[1] for v in d.values())
        encodings[k] = {key: pos/max(cnt,1) for key,(pos,cnt) in d.items()}
        encodings[k]["__global__"] = total_pos / max(total_all, 1)
    return encodings


def _simple_pagerank(adj, n, iters=10, d=0.85):
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


def node2vec_embed(adj, n_nodes, dim=64):
    """
    简化版 Node2Vec：随机游走 + SVD 分解跳过 Word2Vec 训练。
    构建共现矩阵，用 SVD 降维得到嵌入。
    """
    # 随机游走
    walks = []
    nodes = list(range(n_nodes))
    np.random.seed(42)
    for _ in range(NUM_WALKS):
        np.random.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(WALK_LEN - 1):
                curr = walk[-1]
                nbs = list(adj.get(curr, set()))
                if not nbs: break
                walk.append(nbs[np.random.randint(len(nbs))])
            if len(walk) >= 2:
                walks.append(walk)
    
    # 共现矩阵
    cooc = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for walk in walks:
        for i, u in enumerate(walk):
            for j in range(max(0, i-WINDOW), min(len(walk), i+WINDOW+1)):
                if i != j:
                    v = walk[j]
                    cooc[u, v] += 1.0 / abs(i - j)  # 距离加权
    
    # SVD 降维
    if n_nodes <= dim:
        return np.zeros((n_nodes, dim), dtype=np.float32)
    
    U, S, Vt = np.linalg.svd(cooc, full_matrices=False)
    embed = U[:, :min(dim, len(S))] * np.sqrt(S[:min(dim, len(S))])
    
    # 补齐维度
    if embed.shape[1] < dim:
        pad = np.zeros((n_nodes, dim - embed.shape[1]), dtype=np.float32)
        embed = np.hstack([embed, pad])
    
    # 归一化
    norms = np.linalg.norm(embed, axis=1, keepdims=True) + 1e-9
    return (embed / norms).astype(np.float32)


def extract_features_v9(nodes, edges, fault_time, target_encodings=None, ngram_vocab=None):
    n_nodes = len(nodes)
    rid2idx = {n["@rid"]: i for i, n in enumerate(nodes)}

    adj = defaultdict(set)
    for e in edges:
        s = e.get("in", ""); t = e.get("out", "")
        if s in rid2idx and t in rid2idx:
            si, ti = rid2idx[s], rid2idx[t]
            adj[si].add(ti); adj[ti].add(si)

    ft = safe_int(fault_time)
    alarm_indices = [i for i, n in enumerate(nodes) if n.get("@class") == "Alarm"]
    target_set = set(i for i in alarm_indices if nodes[i].get("label") == "TargetAlarm")
    pr = _simple_pagerank(adj, n_nodes)

    # Node2Vec 嵌入（全图所有节点）
    if n_nodes > 1:
        embed = node2vec_embed(adj, n_nodes, dim=EMBED_DIM)
    else:
        embed = np.zeros((1, EMBED_DIM), dtype=np.float32)

    features, rids = [], []
    for i in alarm_indices:
        node = nodes[i]
        f = []
        deg = len(adj.get(i, set()))

        # ====== V3 原始特征 (45维) ======
        # 1. 基础标识 (2)
        f.append(1 if i in target_set else 0)
        f.append(1 if i in target_set and len(target_set) == 1 else 0)

        # 2. Target Encoding (6)
        if target_encodings:
            for k in ["title","device","vendor","location","room"]:
                v = node.get(k, "") or ""
                f.append(target_encodings[k].get(v, target_encodings[k]["__global__"]))
            fk = node.get("fault1","") or node.get("fault2","")
            f.append(target_encodings["fault_key"].get(fk, target_encodings["fault_key"]["__global__"]))
        else:
            f.extend([0.0]*6)

        # 3. 文本哈希 (5)
        f.append(stable_hash(node.get("title",""), 100))
        f.append(stable_hash(node.get("device",""), 100))
        f.append(stable_hash(node.get("vendor",""), 15))
        f.append(stable_hash(node.get("location",""), 100))
        f.append(stable_hash(node.get("room",""), 20))

        # 4. 时序 (13)
        t = safe_int(node.get("time", ft))
        td = max(0, (ft - t) / 60000)
        f.append(min(td, 1440))
        tl = node.get("timeLists", [])
        if isinstance(tl, list) and len(tl) >= 6:
            tla = np.array(tl[:6], dtype=np.float32)
            f.extend([float(np.sum(tla)), float(np.mean(tla)), float(np.std(tla)),
                      float(np.max(tla)), float(np.min(tla))])
            f.extend([float(v) for v in tla])
            f.append(float(tla[-1] / max(np.max(tla), 1e-6)))
            f.append(float(np.argmax(tla)) / 5)
        else:
            f.extend([0]*13)

        # 5. 原因关键词 (7)
        reason = node.get("reason", "")
        f.append(1 if "市电" in reason or "供电" in reason or "电力" in reason else 0)
        f.append(1 if "故障" in reason else 0)
        f.append(1 if "传输" in reason else 0)
        f.append(1 if "光" in reason or "光纤" in reason else 0)
        f.append(len(reason) // 30)
        fk1 = node.get("fault1",""); fk2 = node.get("fault2","")
        f.append(stable_hash(fk1 or fk2, 30))
        f.append(len(fk1) + len(fk2))

        # 6. 图拓扑 (12)
        f.append(min(deg, 100))
        f.append(float(pr[i]) if i < len(pr) else 0.0)
        nbs = adj.get(i, set())
        n_alarm = sum(1 for nb in nbs if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        n_target = sum(1 for nb in nbs if nb in target_set)
        n_device = len(nbs) - n_alarm
        f.append(n_alarm / max(deg, 1))
        f.append(min(n_target, 20))
        f.append(n_device / max(deg, 1))
        nb_pr = [pr[nb] for nb in nbs if nb < len(pr)]
        f.append(float(np.mean(nb_pr)) if nb_pr else 0.0)
        n2 = set()
        for nb in nbs: n2.update(adj.get(nb, set()))
        n2.discard(i)
        n2_alarm = sum(1 for nb in n2 if nb < n_nodes and nodes[nb].get("@class") == "Alarm")
        f.append(min(len(n2), 200))
        f.append(n2_alarm / max(len(n2), 1))
        if len(pr) > 0 and i < len(pr):
            f.append(sum(1 for p in pr if p > pr[i]) / max(len(pr), 1))
        else:
            f.append(0.0)
        nb_list = list(nbs)
        nb_edges = sum(1 for u in nb_list for v in nb_list if v in adj.get(u, set())) // 2
        denom = deg * (deg - 1) / 2
        f.append(nb_edges / max(denom, 1))

        # ====== V9 新增1: Node2Vec 嵌入 (64维) ======
        f.extend(embed[i].tolist() if i < len(embed) else [0.0]*EMBED_DIM)

        # ====== V9 新增2: 字符 n-gram 特征 (20维) ======
        if ngram_vocab:
            grams = set()
            for field in ["title", "device", "location"]:
                grams.update(char_ngrams(node.get(field, ""), NGRAM_N))
            for g in sorted(ngram_vocab):
                f.append(1.0 if g in grams else 0.0)
        else:
            f.extend([0.0]*NGRAM_TOP)

        features.append(f)
        rids.append(node["@rid"])

    return np.array(features, dtype=np.float32), rids, alarm_indices


def build_dataset(dirs, mode="train", target_encodings=None, ngram_vocab=None):
    all_X, all_y, all_meta = [], [], []
    for wo in dirs:
        try:
            p = wo / f"{wo.name}.log.topo.json"
            obj = json.loads(p.read_text(encoding="utf-8"))
            nodes = obj["nodes"]
            X, rids, alarm_idx = extract_features_v9(
                nodes, obj.get("edges", []), obj.get("time", 0),
                target_encodings, ngram_vocab
            )
            y = np.zeros(len(alarm_idx), dtype=np.int32)
            if mode == "train":
                rc_path = wo / f"{wo.name}.rootcause.json"
                if rc_path.exists():
                    rc = json.loads(rc_path.read_text(encoding="utf-8"))
                    rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
                    for j, idx in enumerate(alarm_idx):
                        if nodes[idx]["@rid"] in rc_rids: y[j] = 1
            if len(X) > 0:
                all_X.append(X); all_y.append(y)
                all_meta.append({"wo_id": wo.name, "rids": rids, "n_alarm": len(rids)})
        except Exception as e:
            print(f"  ⚠ {wo.name}: {e}")
    return all_X, all_y, all_meta


if __name__ == "__main__":
    if "--test" in sys.argv:
        with open(OUTPUT_DIR / "target_encodings_v9.pkl", "rb") as f:
            target_encodings = pickle.load(f)
        with open(OUTPUT_DIR / "ngram_vocab_v9.pkl", "rb") as f:
            ngram_vocab = pickle.load(f)
        dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
        print(f"生成测试特征 V9 ({len(dirs)} 工单)...")
        Xs, ys, meta = build_dataset(dirs, "test", target_encodings, ngram_vocab)
        with open(OUTPUT_DIR / "test_features_v9.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta}, f)
        total = sum(m["n_alarm"] for m in meta)
        print(f"✅ 测试V9: {len(Xs)}工单, {total} Alarm节点, 特征={Xs[0].shape[1]}维")
    else:
        dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"构建 Target Encoding + N-gram 词表...")
        target_encodings = _build_target_encodings(dirs)
        ngram_vocab = _build_ngram_vocab(dirs)
        with open(OUTPUT_DIR / "target_encodings_v9.pkl", "wb") as f:
            pickle.dump(target_encodings, f)
        with open(OUTPUT_DIR / "ngram_vocab_v9.pkl", "wb") as f:
            pickle.dump(ngram_vocab, f)
        print(f"n-gram 词表大小: {len(ngram_vocab)}")

        print(f"生成训练特征 V9 (含 Node2Vec)...")
        Xs, ys, meta = build_dataset(dirs, "train", target_encodings, ngram_vocab)
        total = sum(m["n_alarm"] for m in meta)
        pos = sum(y.sum() for y in ys)
        dim = Xs[0].shape[1] if Xs else 0
        print(f"✅ 训练V9: Alarm={total}, 正样本={pos} ({pos/total*100:.1f}%), 特征={dim}维")
        with open(OUTPUT_DIR / "train_features_v9.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta, "feat_dim": dim}, f)