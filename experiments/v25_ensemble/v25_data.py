"""
V25 多策略集成 — 数据加载、特征工程、5折划分
基于 V10/V11 的 prepare() + grouped_folds() + directed_features()
新增: M1 context / M2 meta / M3 graph / M4 correct 四组特征
"""

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

# ── V10 imports ──
V10_PATH = Path("D:/zgyidong/codexgz/work")
if str(V10_PATH) not in sys.path:
    sys.path.insert(0, str(V10_PATH))
import v10_grouped_ensemble as v10

N_FOLDS = 5
MAX_ROOTCAUSES = 8


# ═══════════════════════════════════════════
# 1. Graph structure features (M3)
# ═══════════════════════════════════════════

def pagerank_scores(adjacency, damping=0.85, iterations=30):
    """Simple PageRank on undirected graph."""
    n = len(adjacency)
    pr = np.ones(n, dtype=np.float64) / n
    out_degree = np.array([max(len(neighbors), 1) for neighbors in adjacency], dtype=np.float64)
    for _ in range(iterations):
        new_pr = np.ones(n, dtype=np.float64) * (1 - damping) / n
        for i in range(n):
            for j in adjacency[i]:
                new_pr[j] += damping * pr[i] / out_degree[i]
        pr = new_pr
    return pr


def local_clustering(adjacency):
    """Local clustering coefficient per node."""
    coeffs = np.zeros(len(adjacency), dtype=np.float32)
    for i, neighbors in enumerate(adjacency):
        if len(neighbors) < 2:
            continue
        edges = 0
        for u in neighbors:
            for v in neighbors:
                if u < v and v in adjacency[u]:
                    edges += 1
        coeffs[i] = 2.0 * edges / (len(neighbors) * (len(neighbors) - 1))
    return coeffs


def shortest_distances(adjacency, starts):
    """Multi-source BFS distances."""
    dist = np.full(len(adjacency), len(adjacency) + 1, dtype=np.float32)
    queue = deque()
    for s in starts:
        dist[s] = 0
        queue.append(s)
    while queue:
        cur = queue.popleft()
        nd = dist[cur] + 1
        for nb in adjacency[cur]:
            if nd < dist[nb]:
                dist[nb] = nd
                queue.append(nb)
    return dist


def jaccard_similarity(adjacency, node_i, node_j):
    """Jaccard similarity of neighbor sets."""
    ni = adjacency[node_i]
    nj = adjacency[node_j]
    if not ni and not nj:
        return 0.0
    intersection = len(ni & nj)
    union = len(ni | nj)
    return intersection / max(union, 1)


def graph_features_for_order(order):
    """Extract graph-structure features (M3) for each alarm node."""
    nodes = order["topology"].get("nodes", [])
    alarms = order["alarms"]
    rid_to_idx = {node.get("@rid"): i for i, node in enumerate(nodes)}
    
    # Build undirected adjacency
    adjacency = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        src = rid_to_idx.get(edge.get("in"))
        dst = rid_to_idx.get(edge.get("out"))
        if src is not None and dst is not None:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    
    alarm_indices = [rid_to_idx.get(a.get("@rid"), -1) for a in alarms]
    alarm_set = set(alarm_indices)
    target_indices = [
        rid_to_idx.get(a.get("@rid")) for a in alarms 
        if a.get("label") == "TargetAlarm"
    ]
    
    # PageRank and clustering
    pr = pagerank_scores(adjacency)
    cc = local_clustering(adjacency)
    
    # Distances to targets
    dist_to_target = shortest_distances(adjacency, target_indices)
    unreachable = float(len(nodes) + 1)
    
    features = []
    for node_idx in alarm_indices:
        if node_idx < 0:
            features.append([0.0] * 20)
            continue
            
        neighbors = adjacency[node_idx]
        deg = len(neighbors)
        alarm_nbrs = sum(n in alarm_set for n in neighbors)
        target_nbrs = sum(n in target_indices for n in neighbors)
        
        # 2-hop neighborhood
        two_hop = set()
        for n in neighbors:
            two_hop.update(adjacency[n])
        two_hop.discard(node_idx)
        two_hop_alarm = sum(n in alarm_set for n in two_hop)
        
        # Distances
        d_to = min(float(dist_to_target[node_idx]), unreachable)
        
        # Jaccard with each target
        jaccards = [jaccard_similarity(adjacency, node_idx, t) for t in target_indices]
        max_jaccard = max(jaccards) if jaccards else 0.0
        mean_jaccard = np.mean(jaccards) if jaccards else 0.0
        
        # Eigenvector-like features from degree ratios
        deg_ratio = math.log1p(deg) / math.log1p(max(len(adjacency), 1))
        
        row = [
            math.log1p(deg),                                      # 0
            math.log1p(alarm_nbrs),                               # 1
            alarm_nbrs / max(deg, 1),                             # 2
            math.log1p(target_nbrs),                              # 3
            target_nbrs / max(deg, 1),                            # 4
            math.log1p(len(two_hop)),                             # 5
            math.log1p(two_hop_alarm),                            # 6
            two_hop_alarm / max(len(two_hop), 1),                 # 7
            math.log1p(d_to),                                     # 8
            float(d_to <= len(nodes)),                            # 9
            float(pr[node_idx]) * 100,                            # 10: scaled PageRank
            float(cc[node_idx]),                                  # 11: clustering coeff
            max_jaccard,                                          # 12
            mean_jaccard,                                         # 13
            deg_ratio,                                            # 14
            float(node_idx in target_indices),                    # 15: is_target
            math.log1p(abs(d_to - unreachable) + 1),              # 16: distance margin
            float(d_to == 1),                                     # 17: direct neighbor of target
            float(len(target_indices)),                           # 18: target count
            float(len(alarms)),                                   # 19: alarm count
        ]
        features.append(row)
    
    return np.asarray(features, dtype=np.float32)


def graph_features(orders):
    return np.vstack([graph_features_for_order(o) for o in orders])


# ═══════════════════════════════════════════
# 2. Context features (M1) — title/text
# ═══════════════════════════════════════════

KEYWORD_PATTERNS = [
    r"断链|链路断|link.?down|link.?fail",
    r"丢失|loss|丢包|packet.?loss",
    r"拥塞|congest",
    r"超时|timeout|time.?out",
    r"误码|error|bit.?error|BER",
    r"光模块|optical|SFP|光口|光功率",
    r"电压|voltage|电源|power.?fail",
    r"温度|temp|高温|过热",
    r"重启|reset|reboot|重启",
    r"时钟|clock|sync|同步",
    r"倒换|switchover|protect|保护",
    r"中断|interrupt|down",
    r"告警|alarm|fault|故障",
    r"端口|port|interface",
]


def context_features_for_order(order):
    """Extract context features (M1) from alarm titles and addInfo."""
    alarms = order["alarms"]
    nodes = order["topology"].get("nodes", [])
    
    # Collect all titles for TF-IDF-like features
    all_titles = [str(a.get("title", "")) for a in alarms]
    title_lengths = [len(t) for t in all_titles]
    
    # Title distribution stats
    title_counter = Counter(all_titles)
    
    features = []
    for alarm in alarms:
        title = str(alarm.get("title", ""))
        addinfo = str(alarm.get("addInfo", ""))
        reason = str(alarm.get("reason", ""))
        
        # Keyword matching (14 patterns)
        keywords = [float(bool(re.search(p, title + addinfo + reason, re.I))) 
                    for p in KEYWORD_PATTERNS]
        
        # Title statistics
        title_len = len(title)
        title_digits = sum(c.isdigit() for c in title)
        title_special = sum(not c.isalnum() for c in title)
        title_freq = title_counter[title]
        title_rank = sorted(title_counter.values(), reverse=True).index(title_freq) + 1
        
        # addInfo statistics
        addinfo_len = len(addinfo)
        addinfo_has_ip = float(bool(re.search(r"\d+\.\d+\.\d+\.\d+", addinfo)))
        addinfo_has_port = float(bool(re.search(r"port|端口|接口", addinfo, re.I)))
        addinfo_has_threshold = float(bool(re.search(r"阈值|threshold|门限", addinfo, re.I)))
        
        # Position in alarm list
        alarm_idx = alarms.index(alarm)
        position_ratio = alarm_idx / max(len(alarms) - 1, 1)
        
        row = [
            float(title_len),                                    # 0
            float(title_digits),                                 # 1
            title_digits / max(title_len, 1),                    # 2
            float(title_special),                                # 3
            float(title_freq),                                   # 4
            math.log1p(title_rank),                              # 5
            float(addinfo_len),                                  # 6
            addinfo_has_ip,                                      # 7
            addinfo_has_port,                                    # 8
            addinfo_has_threshold,                               # 9
            position_ratio,                                      # 10
            float(len(str(reason))),                             # 11
            float(bool(reason)),                                 # 12
            float(alarm.get("label") == "TargetAlarm"),          # 13
            *keywords,                                           # 14-27
        ]
        features.append(row)
    
    return np.asarray(features, dtype=np.float32)


def context_features(orders):
    return np.vstack([context_features_for_order(o) for o in orders])


# ═══════════════════════════════════════════
# 3. Meta features (M2) — time/statistics
# ═══════════════════════════════════════════

def meta_features_for_order(order):
    """Extract meta features (M2) — time series and aggregate statistics."""
    alarms = order["alarms"]
    topology = order["topology"]
    fault_time = float(topology.get("time", 0))
    
    # Time statistics across all alarms
    alarm_times = [float(a.get("time", fault_time)) for a in alarms]
    time_min = min(alarm_times)
    time_max = max(alarm_times)
    time_span = max(time_max - time_min, 1.0)
    
    # timeLists statistics
    all_time_lists = []
    for a in alarms:
        tl = a.get("timeLists", [])
        if isinstance(tl, list) and tl:
            all_time_lists.append([float(v) for v in tl[:6]])
    
    features = []
    for alarm in alarms:
        node_time = float(alarm.get("time", fault_time))
        time_delta = abs(fault_time - node_time)
        time_lists = alarm.get("timeLists", [])
        if not isinstance(time_lists, list):
            time_lists = []
        time_vals = np.array([float(v) for v in time_lists[:6]], dtype=np.float32)
        if len(time_vals) == 0:
            time_vals = np.array([0.0], dtype=np.float32)
        
        # Basic time stats
        log_time_delta = math.log1p(time_delta)
        normalized_time = (node_time - time_min) / time_span
        
        # timeLists statistics
        tl_len = len(time_lists)
        tl_sum = float(np.sum(time_vals))
        tl_mean = float(np.mean(time_vals))
        tl_std = float(np.std(time_vals))
        tl_max = float(np.max(time_vals))
        tl_min = float(np.min(time_vals))
        tl_range = tl_max - tl_min
        tl_trend = time_vals[-1] - time_vals[0] if len(time_vals) > 1 else 0.0
        tl_last_ratio = float(time_vals[-1] / max(tl_max, 1e-6))
        tl_peak_pos = float(np.argmax(time_vals)) / max(len(time_vals) - 1, 1)
        
        # Relative ordering
        time_rank = sum(1 for t in alarm_times if t <= node_time)
        time_rank_ratio = time_rank / len(alarms)
        
        # Alarm density features
        same_title_count = sum(1 for a in alarms if a.get("title") == alarm.get("title"))
        same_device_count = sum(1 for a in alarms if a.get("device") == alarm.get("device"))
        
        row = [
            log_time_delta,                                      # 0
            normalized_time,                                     # 1
            float(tl_len),                                       # 2
            math.log1p(tl_sum),                                  # 3
            tl_mean,                                             # 4
            tl_std,                                              # 5
            math.log1p(tl_range),                                # 6
            tl_trend,                                            # 7
            tl_last_ratio,                                       # 8
            tl_peak_pos,                                         # 9
            time_rank_ratio,                                     # 10
            float(same_title_count),                             # 11
            float(same_device_count),                            # 12
            math.log1p(len(alarms)),                             # 13
            float(len(alarms)),                                  # 14
            math.log1p(time_span),                               # 15
        ]
        features.append(row)
    
    return np.asarray(features, dtype=np.float32)


def meta_features(orders):
    return np.vstack([meta_features_for_order(o) for o in orders])


# ═══════════════════════════════════════════
# 4. Correction features (M4) — V11 error patterns
# ═══════════════════════════════════════════

def correct_features(v11_scores, slices, v11_base_mask, labels=None):
    """Extract V11 error-pattern features (M4)."""
    n = len(v11_scores)
    feats = np.zeros((n, 12), dtype=np.float32)
    
    # Handle both slice objects and integer pointer arrays
    if hasattr(slices[0], 'start'):
        # slices is a list of slice objects
        slice_list = slices
    else:
        # slices is an integer pointer array
        slice_list = [slice(int(slices[i]), int(slices[i + 1])) for i in range(len(slices) - 1)]
    
    for sl in slice_list:
        start, stop = sl.start, sl.stop
        if start >= stop:
            continue
        local_scores = v11_scores[start:stop]
        local_n = stop - start
        
        # Per-order statistics
        order_mean = float(np.mean(local_scores))
        order_std = float(np.std(local_scores))
        order_max = float(np.max(local_scores))
        order_median = float(np.median(local_scores))
        
        # V11 selected positions
        if v11_base_mask is not None:
            v11_selected = v11_base_mask[start:stop]
        else:
            # Select top-k
            ranked = np.argsort(-local_scores, kind="stable")
            k = min(MAX_ROOTCAUSES, local_n)
            v11_selected = np.zeros(local_n, dtype=bool)
            v11_selected[ranked[:k]] = True
        
        for j in range(local_n):
            score = local_scores[j]
            rank = (local_scores >= score).sum()
            
            feats[start + j] = [
                score,                                               # 0: V11 score
                score - order_mean,                                  # 1: deviation from mean
                (score - order_mean) / max(order_std, 1e-6),         # 2: z-score
                score - order_median,                                # 3: deviation from median
                score / max(order_max, 1e-6),                        # 4: ratio to max
                float(rank),                                         # 5: rank in order
                float(rank <= MAX_ROOTCAUSES),                       # 6: in top-K
                float(v11_selected[j]),                              # 7: V11 selected
                score - (order_max * 0.5),                           # 8: gap to half-max
                float(local_n),                                      # 9: alarm count
                order_std,                                           # 10: order score std
                order_max - order_mean,                              # 11: order score spread
            ]
    
    return feats


# ═══════════════════════════════════════════
# 5. Data loading + fold splitting
# ═══════════════════════════════════════════

def order_signature(order):
    """Template-based signature for fold grouping."""
    title_counts = Counter(v10.scalar(node.get("title")) for node in order["alarms"])
    target_titles = sorted(
        v10.scalar(node.get("title"))
        for node in order["alarms"]
        if node.get("label") == "TargetAlarm"
    )
    return (
        tuple(sorted(title_counts.items())),
        tuple(target_titles),
        len(order["alarms"]),
    )


def grouped_folds(orders):
    """Group orders by template signature, split into N_FOLDS."""
    groups = defaultdict(list)
    for idx, order in enumerate(orders):
        groups[order_signature(order)].append(idx)
    
    fold_sizes = [0] * N_FOLDS
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ranked:
        fold = min(range(N_FOLDS), key=lambda v: (fold_sizes[v], v))
        folds[indices] = fold
        fold_sizes[fold] += len(indices)
    
    repeated = sum(len(idx) for idx in groups.values() if len(idx) > 1)
    return folds, len(groups), repeated, fold_sizes


def load_all_data(train_dir, test_dir, v11_dir):
    """Load train/test orders, V11 logits, and build all feature matrices."""
    print("Loading orders...", flush=True)
    train_orders = v10.load_orders(Path(train_dir), True)
    test_orders = v10.load_orders(Path(test_dir), False)
    data = v10.prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    
    # V11 OOF logits
    v11_dir = Path(v11_dir)
    train_v11 = (
        0.25 * np.load(v11_dir / "v11_oof_context.npy")
        + 0.75 * np.load(v11_dir / "v11_oof_meta.npy")
    ).astype(np.float32)
    test_v11 = np.load(v11_dir / "v11_test_scores.npy").astype(np.float32)
    
    # Fold assignment
    folds, group_count, repeated, fold_sizes = grouped_folds(train_orders)
    
    # Train-TFIDF and directed features (from V11)
    train_tfidf, test_tfidf = v10.normalized_tfidf(
        data["train_alarm"], data["test_alarm"]
    )
    
    # V11 directed features
    sys.path.insert(0, str(Path(v11_dir).parent / "work"))
    
    print(f"train={len(train_orders)} test={len(test_orders)} "
          f"rows={len(labels)} pos={int(np.sum(labels))} groups={group_count} "
          f"repeated={repeated} folds={fold_sizes}", flush=True)
    
    return {
        "train_orders": train_orders,
        "test_orders": test_orders,
        "data": data,
        "labels": labels,
        "train_v11": train_v11,
        "test_v11": test_v11,
        "folds": folds,
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "train_tfidf": train_tfidf,
        "test_tfidf": test_tfidf,
    }


def build_feature_sets(dataset):
    """Build M1-M4 feature matrices."""
    data = dataset["data"]
    labels = dataset["labels"]
    train_v11 = dataset["train_v11"]
    test_v11 = dataset["test_v11"]
    train_orders = dataset["train_orders"]
    test_orders = dataset["test_orders"]
    
    print("Building features...", flush=True)
    
    # M1: Context features (title/text)
    print("  M1 context...", flush=True)
    m1_train = context_features(train_orders)
    m1_test = context_features(test_orders)
    
    # M2: Meta features (time/statistics)  
    print("  M2 meta...", flush=True)
    m2_train = meta_features(train_orders)
    m2_test = meta_features(test_orders)
    
    # M3: Graph structure features
    print("  M3 graph...", flush=True)
    m3_train = graph_features(train_orders)
    m3_test = graph_features(test_orders)
    
    # M4: V11 correction features
    print("  M4 correct...", flush=True)
    # Build V11 base mask for train
    target_train = round(1059 / len(test_orders) * len(train_orders))
    train_base_mask = v10.exact_count_mask(
        train_v11, data["train_slices"], target_train, MAX_ROOTCAUSES
    )
    m4_train = correct_features(train_v11, data["train_slices"], train_base_mask, labels)
    m4_test = correct_features(test_v11, data["test_slices"], None)
    
    # V11 OOF train (base logit)
    base_logit_train = np.log(np.clip(train_v11, 1e-5, 1 - 1e-5)) - \
                       np.log(1 - np.clip(train_v11, 1e-5, 1 - 1e-5))
    base_logit_test = np.log(np.clip(test_v11, 1e-5, 1 - 1e-5)) - \
                      np.log(1 - np.clip(test_v11, 1e-5, 1 - 1e-5))
    
    feature_sets = {
        "M1": {"train": m1_train.astype(np.float32), "test": m1_test.astype(np.float32),
               "name": "context", "dim": m1_train.shape[1]},
        "M2": {"train": m2_train.astype(np.float32), "test": m2_test.astype(np.float32),
               "name": "meta", "dim": m2_train.shape[1]},
        "M3": {"train": m3_train.astype(np.float32), "test": m3_test.astype(np.float32),
               "name": "graph", "dim": m3_train.shape[1]},
        "M4": {"train": m4_train.astype(np.float32), "test": m4_test.astype(np.float32),
               "name": "correct", "dim": m4_train.shape[1]},
        "V11": {"train": train_v11, "test": test_v11, 
                "train_logit": base_logit_train, "test_logit": base_logit_test,
                "name": "v11_base", "dim": 1},
    }
    
    for key, fs in feature_sets.items():
        print(f"  {key}: train={fs['train'].shape} test={fs['test'].shape} dim={fs['dim']}", flush=True)
    
    return feature_sets


def rows_for_orders(data, order_indices):
    """Get row indices for a set of orders."""
    row_indices = []
    for oi in order_indices:
        sl = data["train_slices"][int(oi)]
        row_indices.extend(range(sl.start, sl.stop))
    return np.asarray(row_indices, dtype=np.int64)


def confusion(mask, labels):
    tp = int(np.sum(mask & (labels == 1)))
    fp = int(np.sum(mask & (labels == 0)))
    fn = int(np.sum((~mask) & (labels == 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, tp, fp, fn, int(np.sum(mask))


def per_order_tp_delta(new_mask, base_mask, labels, slices):
    """Compute per-order TP delta."""
    deltas = []
    for sl in slices:
        start, stop = sl.start, sl.stop
        base_tp = int(np.sum(base_mask[start:stop] & (labels[start:stop] == 1)))
        new_tp = int(np.sum(new_mask[start:stop] & (labels[start:stop] == 1)))
        deltas.append(new_tp - base_tp)
    return np.array(deltas, dtype=np.int32)


def bootstrap_lower_bound(deltas, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% lower confidence bound for mean TP delta."""
    n = len(deltas)
    means = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        sample = deltas[rng.choice(n, n, replace=True)]
        means[i] = np.mean(sample)
    return float(np.percentile(means, alpha * 100))


# ═══════════════════════════════════════════
# 6. Model training helpers
# ═══════════════════════════════════════════

def train_model_cv(name, X_train, y_train, slices, folds_arr, model_type, params, seed):
    """Train a model with 5-fold CV, return OOF predictions."""
    n = len(y_train)
    oof_preds = np.zeros(n, dtype=np.float32)
    all_orders = np.arange(len(slices))
    
    for heldout in range(5):
        train_idx = rows_for_orders(
            {"train_slices": slices}, all_orders[folds_arr != heldout]
        )
        val_idx = rows_for_orders(
            {"train_slices": slices}, all_orders[folds_arr == heldout]
        )
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        if model_type == "lgb":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                random_state=seed + heldout,
                verbose=-1,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_train[val_idx])],
                     eval_metric="auc",
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                random_seed=seed + heldout,
                verbose=0,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_train[val_idx]),
                     early_stopping_rounds=30)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier(
                random_state=seed + heldout,
                verbosity=0,
                **params
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_train[val_idx])],
                     verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        elif model_type == "lr":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                random_state=seed + heldout,
                **params
            )
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return oof_preds


def train_full_model(name, X_train, y_train, X_test, model_type, params, seed):
    """Train a model on full training data, return test predictions."""
    if model_type == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            random_state=seed, verbose=-1, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            random_seed=seed, verbose=0, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            random_state=seed, verbosity=0, **params
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
        
    elif model_type == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, **params)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]
    
    raise ValueError(f"Unknown model type: {model_type}")
