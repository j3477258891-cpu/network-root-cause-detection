"""
纯统计方案 — 零 ML 模型，仅用频率统计 + 模板匹配 + 图先验
哲学：不用任何训练模型，只用"训练集里发生过什么"来预测
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10

MAX_ROOT = 8
TARGET_COUNT = 1059


def load_all():
    train = v10.load_orders(Path("D:/zgyidong/train"), True)
    test = v10.load_orders(Path("D:/zgyidong/test"), False)
    data = v10.prepare(train, test)
    return train, test, data


def build_template_library(train_orders):
    """
    Build template → root_cause_title mapping.
    Template = frozenset of alarm titles in an order.
    Value = Counter of root cause titles.
    """
    lib = defaultdict(Counter)
    for order in train_orders:
        title_set = frozenset(
            str(a.get("title", "")) for a in order["alarms"]
        )
        for alarm in order["alarms"]:
            if alarm["@rid"] in order["roots"]:
                lib[title_set][str(alarm.get("title", ""))] += 1
    return dict(lib)


def build_title_prior(train_orders):
    """P(root | title) with Laplace smoothing."""
    pos = Counter()
    total = Counter()
    for order in train_orders:
        for alarm in order["alarms"]:
            title = str(alarm.get("title", ""))
            total[title] += 1
            if alarm["@rid"] in order["roots"]:
                pos[title] += 1

    global_mean = sum(pos.values()) / max(sum(total.values()), 1)
    prior = {}
    for title in total:
        prior[title] = (pos[title] + global_mean) / (total[title] + 1.0)
    return prior, global_mean


def build_title_cooccurrence(train_orders):
    """
    P(title_A is root | title_B appears in same order).
    cooc[(title_A, title_B)] = count of orders where both appear and title_A is root.
    """
    cooc = defaultdict(lambda: [0, 0])  # [root_count, total_count]
    for order in train_orders:
        titles = [str(a.get("title", "")) for a in order["alarms"]]
        root_titles = {
            str(a.get("title", ""))
            for a in order["alarms"]
            if a["@rid"] in order["roots"]
        }
        for a_title in titles:
            is_root = a_title in root_titles
            for b_title in titles:
                cooc[(a_title, b_title)][1] += 1
                if is_root:
                    cooc[(a_title, b_title)][0] += 1

    # Convert to probabilities
    result = {}
    global_mean = sum(v[0] for v in cooc.values()) / max(sum(v[1] for v in cooc.values()), 1)
    for (a, b), (r, t) in cooc.items():
        result[(a, b)] = (r + global_mean) / (t + 1.0)

    return result, global_mean


def graph_distance_prior(order):
    """Alarms closer to TargetAlarm get higher prior."""
    nodes = order["topology"].get("nodes", [])
    alarms = order["alarms"]
    rid_to_idx = {n.get("@rid"): i for i, n in enumerate(nodes)}

    # Undirected adjacency
    adj = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        s, t = rid_to_idx.get(edge.get("in")), rid_to_idx.get(edge.get("out"))
        if s is not None and t is not None:
            adj[s].add(t)
            adj[t].add(s)

    # Target indices
    targets = [
        rid_to_idx.get(a.get("@rid"))
        for a in alarms
        if a.get("label") == "TargetAlarm" and rid_to_idx.get(a.get("@rid")) is not None
    ]

    if not targets:
        return np.full(len(alarms), 0.5)

    # BFS distances
    dist = np.full(len(nodes), len(nodes) + 1, dtype=np.int32)
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)
    while q:
        cur = q.popleft()
        nd = dist[cur] + 1
        for nb in adj[cur]:
            if nd < dist[nb]:
                dist[nb] = nd
                q.append(nb)

    max_dist = max(dist.max(), 1)
    scores = np.array([
        1.0 / (1.0 + dist[rid_to_idx.get(a.get("@rid"), -1)])
        if rid_to_idx.get(a.get("@rid")) is not None
        else 0.2
        for a in alarms
    ])
    return scores


def score_order(order, template_lib, title_prior, global_prior, cooc_prior, cooc_global):
    """
    Score each alarm in an order using:
    - Template matching (if exact match)
    - Title prior
    - Co-occurrence with other alarms in the order
    - Graph distance prior
    """
    alarms = order["alarms"]
    n = len(alarms)
    titles = [str(a.get("title", "")) for a in alarms]
    title_set = frozenset(titles)

    # 1. Template prior
    tpl_prior = np.full(n, global_prior)
    if title_set in template_lib:
        tpl_counts = template_lib[title_set]
        tpl_total = sum(tpl_counts.values())
        if tpl_total > 0:
            for i, t in enumerate(titles):
                tpl_prior[i] = tpl_counts.get(t, 0.0) / tpl_total

    # 2. Title prior
    title_scores = np.array([title_prior.get(t, global_prior) for t in titles])

    # 3. Co-occurrence: average P(this_title is root | each_other_title)
    cooc_scores = np.zeros(n)
    for i, a_title in enumerate(titles):
        vals = []
        for j, b_title in enumerate(titles):
            if i != j:
                vals.append(cooc_prior.get((a_title, b_title), cooc_global))
        cooc_scores[i] = np.mean(vals) if vals else cooc_global

    # 4. Graph distance prior
    graph_scores = graph_distance_prior(order)

    # Blend: template=0.35, title=0.15, cooc=0.25, graph=0.25
    final = (
        0.35 * tpl_prior +
        0.15 * title_scores +
        0.25 * cooc_scores +
        0.25 * graph_scores
    )

    return final


def generate_submission(test_orders, scores_all, data, template_lib, out_path):
    """Generate 1059-prediction submission CSV."""
    test_slices = data["test_slices"]

    # Fixed-K selection
    selected = np.zeros(len(scores_all), dtype=bool)
    optional = []

    for oi, sl in enumerate(test_slices):
        start, stop = sl.start, sl.stop
        local_scores = scores_all[start:stop]
        n_local = stop - start
        ranked = np.argsort(-local_scores, kind="stable")

        # Each order gets at least 1
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:min(MAX_ROOT, n_local)]).tolist())

    remaining = TARGET_COUNT - int(selected.sum())
    optional = np.array(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores_all[optional], kind="stable")]
    selected[optional[:remaining]] = True

    # Write CSV
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])

        for oi, order in enumerate(test_orders):
            sl = test_slices[oi]
            rootcauses = []
            for local_idx in np.flatnonzero(selected[sl]):
                node = order["alarms"][int(local_idx)]
                rootcauses.append({
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                })
            writer.writerow([
                order["id"],
                json.dumps({"rootcause": rootcauses}, ensure_ascii=False),
            ])

    return selected


def main():
    print("=" * 60)
    print("PURE STATISTICAL APPROACH — Zero ML Models")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    train_orders, test_orders, data = load_all()
    print(f"  Train: {len(train_orders)} orders, Test: {len(test_orders)} orders")

    print("\n[2/4] Building template library...")
    template_lib = build_template_library(train_orders)
    exact_match_templates = {
        k: v for k, v in template_lib.items()
        if sum(v.values()) >= 3 and max(v.values()) / sum(v.values()) >= 0.90
    }
    print(f"  Total templates: {len(template_lib)}")
    print(f"  High-confidence (>=3 support, >=90% consistent): {len(exact_match_templates)}")

    # Template coverage on test
    test_matches = 0
    for order in test_orders:
        ts = frozenset(str(a.get("title", "")) for a in order["alarms"])
        if ts in exact_match_templates:
            test_matches += 1
    print(f"  Test orders with exact template match: {test_matches}/{len(test_orders)} ({test_matches/len(test_orders):.1%})")

    print("\n[3/4] Building statistical priors...")
    title_prior, global_prior = build_title_prior(train_orders)
    print(f"  Title prior: {len(title_prior)} unique titles, global mean={global_prior:.4f}")

    cooc_prior, cooc_global = build_title_cooccurrence(train_orders)
    print(f"  Co-occurrence pairs: {len(cooc_prior)}, global mean={cooc_global:.4f}")

    print("\n[4/4] Scoring test orders and generating submission...")
    scores_list = []
    for order in test_orders:
        scores = score_order(
            order, exact_match_templates, title_prior,
            global_prior, cooc_prior, cooc_global
        )
        scores_list.append(scores)

    scores_all = np.concatenate(scores_list).astype(np.float32)

    out_path = "D:/zgyidong/experiments/v25_semantic_router/submissions/result_record_pure_statistical_p1059.csv"
    selected = generate_submission(
        test_orders, scores_all, data, exact_match_templates, out_path
    )

    # Stats
    test_slices = data["test_slices"]
    distribution = Counter()
    for oi, sl in enumerate(test_slices):
        k = int(np.sum(selected[sl]))
        distribution[k] += 1

    print(f"\n{'=' * 60}")
    print("SUBMISSION GENERATED")
    print(f"{'=' * 60}")
    print(f"  Total predictions: {int(np.sum(selected))}")
    print(f"  K distribution: {dict(sorted(distribution.items()))}")
    print(f"  Template-matching orders: {test_matches}/{len(test_orders)}")
    print(f"  Output: {out_path}")

    # SHA256
    import hashlib
    sha = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    print(f"  SHA256: {sha}")

    # Compare with V11
    v11_mask = v10.exact_count_mask(
        np.load("D:/zgyidong/codexgz/v11/v11_test_scores.npy"),
        test_slices, TARGET_COUNT
    )
    same = int(np.sum(selected == v11_mask))
    total = len(selected)
    print(f"\n  Agreement with V11: {same}/{total} ({same/total:.2%})")
    print(f"  Diff count: {total - same}")

    # Honest assessment
    print(f"\n{'=' * 60}")
    print("HONEST ASSESSMENT")
    print(f"{'=' * 60}")
    print("  This is a pure frequency-statistics approach with ZERO ML models.")
    print("  It does NOT use ExtraTrees, GNNs, LLMs, or any trained classifier.")
    print("  Expected online F1: ~0.88-0.90 (below V11's 0.906324).")
    print("  Why: Template matching covers only ~27% of test orders;")
    print("       the co-occurrence prior is a weak signal compared to")
    print("       V11's trained ExtraTrees on 79-dimensional features.")
    print("  This CSV exists as a conceptual reference, not a contender.")

    return out_path


if __name__ == "__main__":
    main()
