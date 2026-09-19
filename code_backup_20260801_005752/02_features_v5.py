"""
02_features_v5.py - 特征工程 V5
================================
V4 → V5 改进：
1. 加 TF-IDF 文本特征（title + device + location 的词袋）
2. 加入邻居节点的文本特征聚合
3. 全部训练数据用于训练（不再留验证）

运行：python 02_features_v5.py && python 02_features_v5.py --test
"""

import json, pickle, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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
    deg[deg == 0] = 1
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
    stats = {k: defaultdict(lambda: [0,0]) for k in ["title","device","vendor","location","room","fault_key"]}
    for wo in train_dirs:
        try:
            topo_path = wo / f"{wo.name}.log.topo.json"
            rc_path = wo / f"{wo.name}.rootcause.json"
            if not topo_path.exists() or not rc_path.exists(): continue
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
            for nd in obj["nodes"]:
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


def _collect_text_corpus(dirs):
    """收集所有 title/device/location 文本"""
    corpus = []
    for wo in dirs:
        try:
            topo = json.loads((wo / f"{wo.name}.log.topo.json").read_text(encoding="utf-8"))
            for n in topo["nodes"]:
                if n.get("@class") == "Alarm":
                    parts = [
                        n.get("title",""), n.get("device",""),
                        n.get("location",""), n.get("vendor",""),
                        n.get("reason",""), n.get("fault1",""),
                    ]
                    corpus.append(" ".join(p for p in parts if p))
        except: pass
    return corpus


def _tfidf_encode(texts, vectorizer=None, fit=False):
    if fit:
        vectorizer = TfidfVectorizer(max_features=32, token_pattern=r'\b\w+\b', lowercase=True)
        return vectorizer.fit_transform(texts).toarray(), vectorizer
    else:
        return vectorizer.transform(texts).toarray(), vectorizer


def extract_features_v5(nodes, edges, fault_time, target_encodings=None, tfidf_vec=None):
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

    # TF-IDF 编码
    alarm_texts = []
    for i in alarm_indices:
        nd = nodes[i]
        parts = [nd.get("title",""), nd.get("device",""), nd.get("location",""),
                 nd.get("vendor",""), nd.get("reason",""), nd.get("fault1","")]
        alarm_texts.append(" ".join(p for p in parts if p))

    if tfidf_vec:
        tfidf_feats = tfidf_vec.transform(alarm_texts).toarray()
    else:
        tfidf_feats = np.zeros((len(alarm_indices), 32))

    features, rids = [], []
    for idx, i in enumerate(alarm_indices):
        node = nodes[i]
        f = []
        deg = len(adj.get(i, set()))

        # 基础标识
        f.append(1 if i in target_set else 0)
        f.append(1 if i in target_set and len(target_set) == 1 else 0)

        # Target Encoding
        if target_encodings:
            for k in ["title","device","vendor","location","room"]:
                v = node.get(k, "") or ""
                f.append(target_encodings[k].get(v, target_encodings[k]["__global__"]))
            fk = node.get("fault1","") or node.get("fault2","")
            f.append(target_encodings["fault_key"].get(fk, target_encodings["fault_key"]["__global__"]))
        else:
            f.extend([0.0]*6)

        # 文本哈希
        f.append(stable_hash(node.get("title",""), 100))
        f.append(stable_hash(node.get("device",""), 100))
        f.append(stable_hash(node.get("vendor",""), 15))
        f.append(stable_hash(node.get("location",""), 100))
        f.append(stable_hash(node.get("room",""), 20))

        # 时序
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
            f.extend([0]*12)

        # 原因关键词
        reason = node.get("reason", "")
        f.append(1 if "市电" in reason or "供电" in reason or "电力" in reason else 0)
        f.append(1 if "故障" in reason else 0)
        f.append(1 if "传输" in reason else 0)
        f.append(1 if "光" in reason or "光纤" in reason else 0)
        f.append(len(reason) // 30)
        fk1 = node.get("fault1",""); fk2 = node.get("fault2","")
        f.append(stable_hash(fk1 or fk2, 30))
        f.append(len(fk1) + len(fk2))

        # 图拓扑
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

        # TF-IDF 特征（32维）
        f.extend(tfidf_feats[idx].tolist())

        features.append(f)
        rids.append(node["@rid"])

    return np.array(features, dtype=np.float32), rids, alarm_indices


def build_dataset(dirs, mode="train", target_encodings=None, tfidf_vec=None):
    all_X, all_y, all_meta = [], [], []
    for wo in dirs:
        try:
            p = wo / f"{wo.name}.log.topo.json"
            obj = json.loads(p.read_text(encoding="utf-8"))
            nodes = obj["nodes"]; edges = obj.get("edges", []); ft = obj.get("time", 0)
            X, rids, alarm_idx = extract_features_v5(nodes, edges, ft, target_encodings, tfidf_vec)
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
        with open(OUTPUT_DIR / "target_encodings_v5.pkl", "rb") as f:
            target_encodings = pickle.load(f)
        with open(OUTPUT_DIR / "tfidf_v5.pkl", "rb") as f:
            tfidf_vec = pickle.load(f)
        dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
        print(f"生成测试特征 V5 ({len(dirs)} 工单)...")
        Xs, ys, meta = build_dataset(dirs, "test", target_encodings, tfidf_vec)
        with open(OUTPUT_DIR / "test_features_v5.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta}, f)
        total = sum(m["n_alarm"] for m in meta)
        print(f"✅ 测试V5: {len(Xs)}工单, {total} Alarm节点, 特征维度: {Xs[0].shape[1]}")
    else:
        dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"构建 target encoding ({len(dirs)} 工单)...")
        target_encodings = _build_target_encodings(dirs)
        with open(OUTPUT_DIR / "target_encodings_v5.pkl", "wb") as f:
            pickle.dump(target_encodings, f)

        # 收集语料训练 TF-IDF
        print("训练 TF-IDF...")
        corpus = _collect_text_corpus(dirs)
        tfidf_vec = TfidfVectorizer(max_features=32, token_pattern=r'\b\w+\b', lowercase=True)
        tfidf_vec.fit(corpus)
        with open(OUTPUT_DIR / "tfidf_v5.pkl", "wb") as f:
            pickle.dump(tfidf_vec, f)

        print(f"生成训练特征 V5...")
        Xs, ys, meta = build_dataset(dirs, "train", target_encodings, tfidf_vec)
        total = sum(m["n_alarm"] for m in meta)
        pos = sum(y.sum() for y in ys)
        dim = Xs[0].shape[1] if Xs else 0
        pos_pct = pos / max(total, 1)
        print(f"✅ 训练V5: Alarm={total}, 正样本={pos} ({pos_pct*100:.1f}%), 特征={dim}维")
        with open(OUTPUT_DIR / "train_features_v5.pkl", "wb") as f:
            pickle.dump({"features": Xs, "labels": ys, "meta": meta, "feat_dim": dim}, f)