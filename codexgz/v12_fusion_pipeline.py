"""
v12_fusion_pipeline.py - V12 融合管线
=======================================
整合 gen/ 核心技术：
  1. CV-TE + Bayesian shrinkage（解决 te_location 过拟合）
  2. KNN context scores（TF-IDF 工单相似度，7 配置加权）
  3. Logistic fusion（防止单特征统治 + 交互特征）
  
输出：
  - 多 target count 候选提交（1044, 1047, 1059, 1076, 1090）
  - 相对冠军的 atomic probe units
  - OOF 评估报告
"""

import csv, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# Config
# ============================================================
KNN_CONFIGS = [
    (5, 2.0, 0.06668020883603187),
    (1, 1.0, 0.04820534260274989),
    (8, 2.0, 0.23543407593447566),
    (2, 2.0, 0.2499238097838483),
    (12, 4.0, 0.03659548793379663),
    (3, 1.0, 0.16652631426606984),
    (3, 2.0, 0.19663476064302782),
]
N_FOLDS = 5
MAX_ROOTCAUSES = 8
BAYESIAN_SMOOTHING = 5
L2_REG = 10.0
CHAMPION_PATH = r"D:\zgyidong\codexgz\result_record_probe_swap12_score_0.905373.csv"

TRAIN_DIR = Path(r"D:\zgyidong\train")
TEST_DIR = Path(r"D:\zgyidong\test")
OUTPUT_DIR = Path(r"D:\zgyidong\codexgz\v12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data loading (adapted from gen/)
# ============================================================
def fold_of(order_id):
    return int(hashlib.md5(order_id.encode()).hexdigest(), 16) % N_FOLDS


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return value or ""


def location_shape(value):
    value = value or ""
    value = re.sub(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        "<UUID>",
        value,
    )
    return re.sub(r"\d+", "#", value)


def load_orders(base_dir, with_labels):
    orders = []
    for directory in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        order_id = directory.name
        topo = json.loads((directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8"))
        alarms = [n for n in topo["nodes"] if n.get("@class") == "Alarm"]
        roots = set()
        if with_labels:
            rc = json.loads((directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8"))
            roots = {n["@rid"] for n in rc["rootcause"]}
        orders.append((order_id, alarms, roots, topo))
    return orders


def prepare_data(train_orders, test_orders):
    titles = sorted({n.get("title", "") or "" for _, alarms, _, _ in train_orders for n in alarms})
    title_index = {t: i for i, t in enumerate(titles)}
    n_train = len(train_orders)
    n_titles = len(titles)

    train_alarm = np.zeros((n_train, n_titles), dtype=np.float32)
    train_root = np.zeros((n_train, n_titles), dtype=np.float32)
    folds = np.zeros(n_train, dtype=np.int8)
    train_rows = []
    train_slices = []

    for oi, (oid, alarms, roots, _) in enumerate(train_orders):
        folds[oi] = fold_of(oid)
        start = len(train_rows)
        for node in alarms:
            tid = title_index[node.get("title", "") or ""]
            label = int(node["@rid"] in roots)
            train_alarm[oi, tid] += 1
            train_root[oi, tid] += label
            train_rows.append((oi, tid, node, label))
        train_slices.append(slice(start, len(train_rows)))

    test_alarm = np.zeros((len(test_orders), n_titles), dtype=np.float32)
    test_rows = []
    test_slices = []
    for oi, (_, alarms, _, _) in enumerate(test_orders):
        start = len(test_rows)
        for node in alarms:
            tid = title_index.get(node.get("title", "") or "", -1)
            if tid >= 0:
                test_alarm[oi, tid] += 1
            test_rows.append((oi, tid, node))
        test_slices.append(slice(start, len(test_rows)))

    return {
        "titles": titles, "title_index": title_index,
        "train_alarm": train_alarm, "train_root": train_root, "folds": folds,
        "train_rows": train_rows, "train_slices": train_slices,
        "test_alarm": test_alarm, "test_rows": test_rows, "test_slices": test_slices,
    }


# ============================================================
# KNN Context Scores (from gen/optimize_online_feedback.py)
# ============================================================
def normalized_tfidf(train_alarm, test_alarm):
    df = np.sum(train_alarm > 0, axis=0)
    idf = np.log((1 + len(train_alarm)) / (1 + df)) + 1
    train = np.log1p(train_alarm) * idf
    test = np.log1p(test_alarm) * idf
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    return train, test


def context_scores(data):
    train_alarm = data["train_alarm"]
    train_root = data["train_root"]
    folds = data["folds"]
    rows = data["train_rows"]
    slices = data["train_slices"]
    train_tfidf, test_tfidf = normalized_tfidf(train_alarm, data["test_alarm"])
    oof = np.zeros(len(rows), dtype=np.float64)

    # 5-fold OOF KNN context
    for heldout in range(N_FOLDS):
        q_idx = np.flatnonzero(folds == heldout)
        r_idx = np.flatnonzero(folds != heldout)
        sims = train_tfidf[q_idx] @ train_tfidf[r_idx].T
        global_pos = np.sum(train_root[r_idx], axis=0)
        global_tot = np.sum(train_alarm[r_idx], axis=0)
        prior = (global_pos + 0.3) / (global_tot + 1.0)
        order_probs = np.zeros((len(q_idx), train_alarm.shape[1]))
        for k, power, w in KNN_CONFIGS:
            nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for ql, qi in enumerate(q_idx):
                nb = r_idx[nearest[ql]]
                weights = np.maximum(sims[ql, nearest[ql]], 1e-6) ** power
                pos = weights @ train_root[nb]
                tot = weights @ train_alarm[nb]
                order_probs[ql] += w * ((pos + 2.0 * prior) / (tot + 2.0))
        for ql, qi in enumerate(q_idx):
            sl = slices[qi]
            for ri in range(sl.start, sl.stop):
                oof[ri] = order_probs[ql, rows[ri][1]]

    # Test KNN context
    sims = test_tfidf @ train_tfidf.T
    global_pos = np.sum(train_root, axis=0)
    global_tot = np.sum(train_alarm, axis=0)
    prior = (global_pos + 0.3) / (global_tot + 1.0)
    test_order_probs = np.zeros((len(test_tfidf), train_alarm.shape[1]))
    for k, power, w in KNN_CONFIGS:
        nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for ti in range(len(test_tfidf)):
            nb = nearest[ti]
            weights = np.maximum(sims[ti, nb], 1e-6) ** power
            pos = weights @ train_root[nb]
            tot = weights @ train_alarm[nb]
            test_order_probs[ti] += w * ((pos + 2.0 * prior) / (tot + 2.0))
    test = np.asarray([
        test_order_probs[oi, tid] if tid >= 0 else 0.3
        for oi, tid, _ in data["test_rows"]
    ], dtype=np.float64)
    return oof, test


# ============================================================
# CV Target Encoding with Bayesian Shrinkage
# ============================================================
def target_encoding_scores(data, key_fn):
    train_rows = data["train_rows"]
    folds = data["folds"]
    oof = np.zeros(len(train_rows), dtype=np.float64)

    for heldout in range(N_FOLDS):
        title_counts = defaultdict(lambda: [0, 0])
        fine_counts = defaultdict(lambda: [0, 0])
        positives = total = 0
        for oi, _, node, label in train_rows:
            if folds[oi] == heldout:
                continue
            title = scalar(node.get("title"))
            title_counts[title][0] += label
            title_counts[title][1] += 1
            fine_counts[key_fn(node)][0] += label
            fine_counts[key_fn(node)][1] += 1
            positives += label
            total += 1
        global_prob = positives / total
        for ri, (oi, _, node, _) in enumerate(train_rows):
            if folds[oi] != heldout:
                continue
            title = scalar(node.get("title"))
            tp, tc = title_counts.get(title, (0, 0))
            title_prob = (tp + BAYESIAN_SMOOTHING * global_prob) / (tc + BAYESIAN_SMOOTHING)
            p, c = fine_counts.get(key_fn(node), (0, 0))
            oof[ri] = (p + BAYESIAN_SMOOTHING * title_prob) / (c + BAYESIAN_SMOOTHING)

    # Test encoding (full train)
    title_counts = defaultdict(lambda: [0, 0])
    fine_counts = defaultdict(lambda: [0, 0])
    positives = 0
    for _, _, node, label in train_rows:
        title = scalar(node.get("title"))
        title_counts[title][0] += label
        title_counts[title][1] += 1
        fine_counts[key_fn(node)][0] += label
        fine_counts[key_fn(node)][1] += 1
        positives += label
    global_prob = positives / len(train_rows)
    test = np.zeros(len(data["test_rows"]), dtype=np.float64)
    for ri, (_, _, node) in enumerate(data["test_rows"]):
        title = scalar(node.get("title"))
        tp, tc = title_counts.get(title, (0, 0))
        title_prob = (tp + BAYESIAN_SMOOTHING * global_prob) / (tc + BAYESIAN_SMOOTHING)
        p, c = fine_counts.get(key_fn(node), (0, 0))
        test[ri] = (p + BAYESIAN_SMOOTHING * title_prob) / (c + BAYESIAN_SMOOTHING)
    return oof, test


# ============================================================
# Logistic Fusion (from gen/optimize_online_feedback.py)
# ============================================================
def fit_logistic(features, labels, l2=10.0):
    coef = np.zeros(features.shape[1], dtype=np.float64)
    identity = np.eye(features.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0
    for _ in range(40):
        linear = np.clip(features @ coef, -25, 25)
        probs = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(probs * (1 - probs), 1e-6)
        grad = features.T @ (labels - probs) - l2 * identity @ coef
        hess = features.T @ (weights[:, None] * features) + l2 * identity
        try:
            update = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hess, grad, rcond=None)[0]
        coef += update
        if np.max(np.abs(update)) < 1e-8:
            break
    return coef


def logistic_features(context, location, reason):
    eps = 1e-5
    values = np.column_stack([
        np.ones(len(context)),
        np.log(np.clip(context, eps, 1 - eps) / np.clip(1 - context, eps, 1)),
        location - context,
        reason - context,
        (location - context) * (context - 0.6),
        (reason - context) * (context - 0.6),
    ])
    return values


# ============================================================
# Selection & Evaluation (from gen/)
# ============================================================
def selection_mask(scores, slices, threshold, cap=MAX_ROOTCAUSES):
    mask = np.zeros(len(scores), dtype=bool)
    for sl in slices:
        local = scores[sl]
        selected = np.flatnonzero(local >= threshold)
        if not len(selected):
            selected = np.asarray([int(np.argmax(local))])
        if len(selected) > cap:
            selected = selected[np.argsort(-local[selected], kind="stable")[:cap]]
        mask[sl.start + selected] = True
    return mask


def best_threshold(scores, y, slices, cap=MAX_ROOTCAUSES):
    mandatory = np.zeros(len(scores), dtype=bool)
    eligible = np.zeros(len(scores), dtype=bool)
    for sl in slices:
        local = scores[sl]
        ranked = np.argsort(-local, kind="stable")
        kept = ranked[:cap]
        eligible[sl.start + kept] = True
        mandatory[sl.start + ranked[0]] = True

    base_tp = int(np.sum(mandatory & (y == 1)))
    base_fp = int(np.sum(mandatory & (y == 0)))
    total_positive = int(np.sum(y))
    optional = np.flatnonzero(eligible & ~mandatory)
    order = optional[np.argsort(-scores[optional], kind="stable")]
    ordered_scores = scores[order]
    ordered_y = y[order]
    cum_tp = np.cumsum(ordered_y)
    cum_fp = np.cumsum(1 - ordered_y)
    boundaries = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    tp = base_tp + cum_tp[boundaries]
    fp = base_fp + cum_fp[boundaries]
    fn = total_positive - tp
    values = 2 * tp / (2 * tp + fp + fn)
    best_idx = int(np.argmax(values))
    boundary = int(boundaries[best_idx])
    return (
        float(values[best_idx]),
        int(tp[best_idx]),
        int(fp[best_idx]),
        int(fn[best_idx]),
        int(np.sum(mandatory) + boundary + 1),
        float(ordered_scores[boundary]),
    )


def evaluate(scores, y, slices, threshold, cap=MAX_ROOTCAUSES):
    pred = selection_mask(scores, slices, threshold, cap)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1, tp, fp, fn, int(np.sum(pred))


def select_for_target(scores, slices, target_count, cap=MAX_ROOTCAUSES):
    low, high = float(np.min(scores)), float(np.max(scores))
    best = None
    for _ in range(60):
        thresh = (low + high) / 2
        mask = selection_mask(scores, slices, thresh, cap)
        count = int(np.sum(mask))
        if best is None or abs(count - target_count) < abs(best[0] - target_count):
            best = (count, thresh, mask)
        if count > target_count:
            low = thresh
        else:
            high = thresh
    return best


# ============================================================
# Submission writer
# ============================================================
def write_submission(path, test_orders, data, mask):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for oi, (oid, alarms, _, _) in enumerate(test_orders):
            sl = data["test_slices"][oi]
            rc_list = []
            for li in np.flatnonzero(mask[sl]):
                node = alarms[int(li)]
                rc_list.append({
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                })
            writer.writerow([oid, json.dumps({"rootcause": rc_list}, ensure_ascii=False)])


def load_champion():
    champ = {}
    with open(CHAMPION_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            champ[row["order_id"]] = {n["@rid"] for n in json.loads(row["output"])["rootcause"]}
    return champ


def generate_probe_units(candidate_path, champion, test_orders, data):
    """提取相对冠军的原子替换"""
    candidate = {}
    with open(candidate_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            candidate[row["order_id"]] = {n["@rid"] for n in json.loads(row["output"])["rootcause"]}

    units = []
    unit_id = 0
    for oid in sorted(champion):
        champ_set = champion.get(oid, set())
        cand_set = candidate.get(oid, set())
        removed = sorted(champ_set - cand_set)
        added = sorted(cand_set - champ_set)
        if removed and added:
            # Paired swaps
            max_len = max(len(removed), len(added))
            for i in range(max_len):
                rm = removed[i] if i < len(removed) else removed[-1]
                ad = added[i] if i < len(added) else added[-1]
                unit_id += 1
                units.append({
                    "id": f"v12_unit_{unit_id:03d}",
                    "order_id": oid,
                    "remove_rid": rm,
                    "add_rid": ad,
                })
        elif removed and not added:
            # Pure removals
            for rm in removed:
                unit_id += 1
                units.append({
                    "id": f"v12_unit_{unit_id:03d}",
                    "order_id": oid,
                    "remove_rid": rm,
                    "add_rid": None,
                })
        elif added and not removed:
            # Pure additions
            for ad in added:
                unit_id += 1
                units.append({
                    "id": f"v12_unit_{unit_id:03d}",
                    "order_id": oid,
                    "remove_rid": None,
                    "add_rid": ad,
                })
    return units


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("=" * 60)
    print("V12 Fusion Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    train_orders = load_orders(TRAIN_DIR, True)
    test_orders = load_orders(TEST_DIR, False)
    print(f"  Train: {len(train_orders)} orders, Test: {len(test_orders)} orders")

    data = prepare_data(train_orders, test_orders)
    y = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    print(f"  Nodes: {len(y)}, Positives: {int(np.sum(y))} ({np.sum(y)/len(y)*100:.1f}%)")

    # KNN Context scores
    print("\n[2/6] Computing KNN context scores...")
    context_oof, context_test = context_scores(data)
    ctx_result = best_threshold(context_oof, y, data["train_slices"])
    print(f"  KNN context OOF: F1={ctx_result[0]:.4f} TP={ctx_result[1]} preds={ctx_result[4]}")

    # CV Target Encoding
    print("\n[3/6] Computing CV Target Encoding...")
    location_oof, location_test = target_encoding_scores(
        data,
        lambda node: (scalar(node.get("title")), location_shape(node.get("location"))),
    )
    reason_oof, reason_test = target_encoding_scores(
        data,
        lambda node: (scalar(node.get("title")), scalar(node.get("reason"))),
    )
    loc_result = best_threshold(location_oof, y, data["train_slices"])
    rea_result = best_threshold(reason_oof, y, data["train_slices"])
    print(f"  TE location: F1={loc_result[0]:.4f} TP={loc_result[1]}")
    print(f"  TE reason:   F1={rea_result[0]:.4f} TP={rea_result[1]}")

    # Linear blend search
    print("\n[4/6] Blending & Logistic fusion...")
    blends = []
    for lw in np.arange(0.0, 0.101, 0.01):
        for rw in np.arange(0.0, 0.101 - lw, 0.01):
            cw = 1.0 - lw - rw
            scores = cw * context_oof + lw * location_oof + rw * reason_oof
            result = best_threshold(scores, y, data["train_slices"])
            blends.append((result[0], lw, rw, result))
    blends.sort(reverse=True)

    # Logistic meta
    meta_train = logistic_features(context_oof, location_oof, reason_oof)
    meta_test = logistic_features(context_test, location_test, reason_test)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    row_folds = np.asarray([data["folds"][oi] for oi, _, _, _ in data["train_rows"]])
    for heldout in range(N_FOLDS):
        tr_mask = row_folds != heldout
        ho_mask = row_folds == heldout
        coef = fit_logistic(meta_train[tr_mask], y[tr_mask])
        linear = np.clip(meta_train[ho_mask] @ coef, -25, 25)
        meta_oof[ho_mask] = 1.0 / (1.0 + np.exp(-linear))
    meta_result = best_threshold(meta_oof, y, data["train_slices"])
    meta_coef = fit_logistic(meta_train, y)
    meta_linear = np.clip(meta_test @ meta_coef, -25, 25)
    meta_test_scores = 1.0 / (1.0 + np.exp(-meta_linear))

    best_blend = blends[0]
    print(f"  Best blend: ctx={1-best_blend[1]-best_blend[2]:.2f} loc={best_blend[1]:.2f} rea={best_blend[2]:.2f} F1={best_blend[0]:.4f}")
    print(f"  Meta logistic: F1={meta_result[0]:.4f} coef={meta_coef.tolist()}")

    # Select best model
    if meta_result[0] > best_blend[0]:
        test_scores = meta_test_scores
        print("  => SELECTED: Logistic Meta-fusion")
    else:
        cw = 1.0 - best_blend[1] - best_blend[2]
        test_scores = cw * context_test + best_blend[1] * location_test + best_blend[2] * reason_test
        print("  => SELECTED: Linear Blend")

    # Generate candidates at multiple target counts
    print("\n[5/6] Generating candidates...")
    champion = load_champion()
    candidates = {}
    for target in (1044, 1047, 1059, 1076, 1090):
        count, thresh, mask = select_for_target(test_scores, data["test_slices"], target)
        path = OUTPUT_DIR / f"v12_candidate_{count}.csv"
        write_submission(path, test_orders, data, mask)
        dist = Counter(int(np.sum(mask[sl])) for sl in data["test_slices"])
        print(f"  {path.name}: count={count} thresh={thresh:.6f} dist={dict(sorted(dist.items()))}")
        candidates[count] = {"path": str(path), "threshold": thresh}

    # Generate probe units for 1059 candidate
    print("\n[6/6] Extracting probe units (vs champion)...")
    cand_1059 = OUTPUT_DIR / "v12_candidate_1059.csv"
    units = generate_probe_units(cand_1059, champion, test_orders, data)
    print(f"  Found {len(units)} atomic probe units")

    # Save report
    report = {
        "pipeline": "V12",
        "description": "CV-TE + KNN context + Logistic fusion",
        "oof_blend": {"f1": best_blend[0], "weights": {"context": 1-best_blend[1]-best_blend[2], "location": best_blend[1], "reason": best_blend[2]}},
        "oof_meta": {"f1": meta_result[0], "coefficients": meta_coef.tolist()},
        "candidates": candidates,
        "probe_units": len(units),
        "probe_units_detail": units,
    }
    with open(OUTPUT_DIR / "v12_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE. Outputs in {OUTPUT_DIR}/")
    print(f"  - v12_candidate_*.csv (5 candidates)")
    print(f"  - v12_report.json")
    print(f"  - {len(units)} probe units extracted")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
