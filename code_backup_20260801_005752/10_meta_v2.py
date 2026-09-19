"""
10_meta_v2.py - OOF Meta 节点评分器（精简版）
==============================================
1. 5-fold OOF 基模型概率
2. 有向图结构特征
3. Meta 模型对全部告警节点评分
4. 每工单选 TopK
"""

import pickle, json, csv, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

FEAT_DIR = Path("D:/zgyidong/code/features")
MODEL_DIR = Path("D:/zgyidong/code/models")
SUBMIT_DIR = Path("D:/zgyidong/code/submit")
TEST_DIR = Path("D:/zgyidong/test")
TRAIN_DIR = Path("D:/zgyidong/train")
for d in [MODEL_DIR, SUBMIT_DIR]: d.mkdir(exist_ok=True)


def safe_int(v, default=0):
    try: return int(v)
    except: return default


def extract_graph_features(wo_dir):
    """有向图结构特征"""
    topo_path = wo_dir / f"{wo_dir.name}.log.topo.json"
    if not topo_path.exists():
        return {}
    obj = json.loads(topo_path.read_text(encoding="utf-8"))
    nodes = obj["nodes"]
    edges = obj.get("edges", [])
    n = len(nodes)
    rid2idx = {nd["@rid"]: i for i, nd in enumerate(nodes)}
    ft = safe_int(obj.get("time", 0))

    in_adj = defaultdict(set)
    out_adj = defaultdict(set)
    for e in edges:
        s = e.get("in", ""); t = e.get("out", "")
        if s in rid2idx and t in rid2idx:
            si, ti = rid2idx[s], rid2idx[t]
            in_adj[ti].add(si)
            out_adj[si].add(ti)

    target_nodes = [i for i, nd in enumerate(nodes) if nd.get("label") == "TargetAlarm"]

    def bfs_dist(start_set, adj):
        dist = {s: 0 for s in start_set}
        q = list(start_set)
        while q:
            u = q.pop(0)
            for v in adj.get(u, set()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    dist_to_target = bfs_dist(set(target_nodes), in_adj) if target_nodes else {}
    alarm_indices = [i for i, nd in enumerate(nodes) if nd.get("@class") == "Alarm"]
    features = {}

    for i in alarm_indices:
        nd = nodes[i]
        f = []
        f.append(float(len(in_adj.get(i, set()))))
        f.append(float(len(out_adj.get(i, set()))))
        f.append(float(dist_to_target.get(i, 20)))
        t = safe_int(nd.get("time", ft))
        td = max(0, (ft - t) / 60000)
        f.append(min(td, 1440))

        tl = nd.get("timeLists", [])
        if isinstance(tl, list) and len(tl) >= 6:
            tla = np.array(tl[:6], dtype=np.float32)
            f.append(float(np.sum(tla)))
            f.append(float(np.mean(tla)))
            f.append(float(np.std(tla)))
            if len(tla) >= 2:
                f.append(float(tla[-1] - tla[-2]))
                f.append(float(tla[-1] / max(np.max(tla), 1e-6)))
            else:
                f.extend([0.0, 0.0])
        else:
            f.extend([0.0] * 5)
        features[nd["@rid"]] = f

    return features


def main():
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    print("=" * 50)
    print("OOF Meta V2")
    print("=" * 50)

    # Phase 1: 加载 V3 特征
    with open(FEAT_DIR / "train_features_v3.pkl", "rb") as f:
        train = pickle.load(f)
    Xs, ys, meta = train["features"], train["labels"], train["meta"]
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)

    print(f"训练节点: {len(X_all)}, 正样本率: {y_all.mean():.3f}")

    # 5-fold OOF
    print("Phase 1: 5-fold OOF...")
    oof_xgb = np.zeros(len(X_all))
    oof_lgb = np.zeros(len(X_all))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_val = X_all[val_idx]

        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, scale_pos_weight=2.5, tree_method='hist',
            random_state=42+fold, verbosity=0)
        xgb_m.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_m.predict_proba(X_val)[:, 1]

        lgb_m = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.08,
            subsample=0.8, scale_pos_weight=2.5, random_state=42+fold, verbose=-1)
        lgb_m.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_m.predict_proba(X_val)[:, 1]

    oof_ens = (oof_xgb + oof_lgb) / 2.0

    # Phase 2: 构建 Meta 训练数据
    print("Phase 2: 构建 Meta 特征...")
    meta_X_list = []
    meta_y_list = []
    offset = 0

    for i, (Xi, yi, m) in enumerate(zip(Xs, ys, meta)):
        wo_dir = TRAIN_DIR / m["wo_id"]
        g_feats = extract_graph_features(wo_dir)
        n = len(Xi)
        probs = oof_ens[offset:offset + n]
        p_xgb = oof_xgb[offset:offset + n]
        p_lgb = oof_lgb[offset:offset + n]

        sorted_idx = np.argsort(-probs)
        ranks = np.zeros(n)
        gaps = np.zeros(n)
        for r, idx in enumerate(sorted_idx):
            ranks[idx] = r
            gaps[idx] = probs[idx] - probs[sorted_idx[r + 1]] if r < n - 1 else 0

        for j in range(n):
            rid = m["rids"][j]
            f = [float(probs[j]), float(p_xgb[j]), float(p_lgb[j]),
                 float(ranks[j]), float(gaps[j]),
                 float(abs(p_xgb[j] - p_lgb[j]))]

            gf = g_feats.get(rid, [0.0] * 9)
            for k in range(9):
                f.append(float(gf[k]) if k < len(gf) else 0.0)

            # V3 关键子集特征
            for si in [0, 1, 8, 14, 15, 16, 17, 18, 28, 32]:
                if si < len(Xi[j]):
                    f.append(float(Xi[j][si]))

            meta_X_list.append(f)
            meta_y_list.append(int(yi[j]))

        offset += n

    meta_X = np.array(meta_X_list, dtype=np.float32)
    meta_y = np.array(meta_y_list)
    print(f"  Meta样本: {len(meta_X)}, 正样本率: {meta_y.mean():.3f}, 特征: {meta_X.shape[1]}")

    # Phase 3: 训练 Meta 模型
    print("Phase 3: 训练 Meta...")
    meta_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=1.5, tree_method='hist',
        random_state=42, verbosity=0
    )
    meta_model.fit(meta_X, meta_y)

    # Phase 4: 训练全量基模型 + 预测测试集
    print("Phase 4: 训练全量基模型...")
    base_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, scale_pos_weight=2.5, tree_method='hist',
        random_state=42, verbosity=0)
    base_xgb.fit(X_all, y_all)
    base_lgb = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08,
        subsample=0.8, scale_pos_weight=2.5, random_state=42, verbose=-1)
    base_lgb.fit(X_all, y_all)

    print("Phase 5: 预测测试集...")
    with open(FEAT_DIR / "test_features_v3.pkl", "rb") as f:
        test_data = pickle.load(f)
    t_Xs, t_meta = test_data["features"], test_data["meta"]

    result = {}
    k_vals = []

    for Xi, m in zip(t_Xs, t_meta):
        n = len(Xi)
        p_xgb = base_xgb.predict_proba(Xi)[:, 1]
        p_lgb = base_lgb.predict_proba(Xi)[:, 1]
        p_ens = (p_xgb + p_lgb) / 2.0

        sorted_idx = np.argsort(-p_ens)
        ranks = np.zeros(n)
        gaps = np.zeros(n)
        for r, idx in enumerate(sorted_idx):
            ranks[idx] = r
            gaps[idx] = p_ens[idx] - p_ens[sorted_idx[r + 1]] if r < n - 1 else 0

        g_feats = extract_graph_features(TEST_DIR / m["wo_id"])

        scores = []
        for j in range(n):
            rid = m["rids"][j]
            f = [float(p_ens[j]), float(p_xgb[j]), float(p_lgb[j]),
                 float(ranks[j]), float(gaps[j]),
                 float(abs(p_xgb[j] - p_lgb[j]))]
            gf = g_feats.get(rid, [0.0] * 9)
            for k in range(9):
                f.append(float(gf[k]) if k < len(gf) else 0.0)
            for si in [0, 1, 8, 14, 15, 16, 17, 18, 28, 32]:
                if si < len(Xi[j]):
                    f.append(float(Xi[j][si]))
            s = meta_model.predict_proba(np.array([f], dtype=np.float32))[0, 1]
            scores.append((j, s))

        scores.sort(key=lambda x: -x[1])

        # K = Probe 的 K 分布：取概率>0.45的个数，但不少于1，不多于8
        K = max(1, int(np.sum(p_ens >= 0.45)))
        K = min(K, 8)

        selected = [idx for idx, _ in scores[:K]]
        k_vals.append(len(selected))

        # 读 title/location/reason
        wo_dir = TEST_DIR / m["wo_id"]
        node_lookup = {}
        topo_path = wo_dir / f'{m["wo_id"]}.log.topo.json'
        if topo_path.exists():
            obj = json.loads(topo_path.read_text(encoding="utf-8"))
            node_lookup = {n["@rid"]: n for n in obj["nodes"]}

        rc_list = []
        for idx in selected:
            rid = m["rids"][idx]
            nd = node_lookup.get(rid, {})
            rc_list.append({
                "@rid": rid,
                "title": nd.get("title", ""),
                "location": nd.get("location", ""),
                "reason": nd.get("reason", ""),
            })
        result[m["wo_id"]] = {"rootcause": rc_list}

    out = SUBMIT_DIR / "result_record.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in result.items():
            writer.writerow([wo_id, json.dumps(v, ensure_ascii=False)])

    dist = Counter(len(v["rootcause"]) for v in result.values())
    print(f"\n✅ Meta V2 完成!")
    print(f"   工单: {len(result)}, 平均K: {np.mean(k_vals):.2f}")
    print(f"   文件: {out}")


if __name__ == "__main__":
    main()