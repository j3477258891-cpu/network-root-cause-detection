"""
V29 — 域稳健 Ranker + 候选池 Oracle Gate
目标：回答「新 ranker 能否把 F1 推到 0.94」，在花任何提交机会之前。

核心：
1. 构造工单内 pairwise 特征（语义+频率+图+时间+告警组）
2. LightGBM ranker，leave-one-site-out 校准
3. 计算 OOF oracle 上限：若 ranker 完美重排，能找回多少 FN
"""

import sys, json, re, math, hashlib
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10

TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")
V16_TRAIN = Path("D:/zgyidong/experiments/v16/v16_train_features.npy")
V16_TEST = Path("D:/zgyidong/experiments/v16/v16_test_features.npy")
V11_CTX = "D:/zgyidong/codexgz/v11/v11_oof_context.npy"
V11_META = "D:/zgyidong/codexgz/v11/v11_oof_meta.npy"
V11_TEST = "D:/zgyidong/codexgz/v11/v11_test_scores.npy"
MAX_ROOT = 8


def site_of(location):
    m = re.search(r"SubNetwork=([^,]+)", str(location or ""))
    return m.group(1) if m else "unknown"


def build_features(orders, v11_scores, v16_features, is_train, labels=None):
    """
    为每个告警构造特征。
    返回 (X, alarm_meta) 其中 meta 包含 order_id/rid/title/site 供后续分组。
    """
    rows = []
    meta = []
    offset = 0
    for oi, order in enumerate(orders):
        alarms = order["alarms"]
        n = len(alarms)
        if n == 0:
            continue

        # 拓扑
        nodes = order["topology"].get("nodes", [])
        rid2 = {nd.get("@rid"): i for i, nd in enumerate(nodes)}
        adj = [set() for _ in nodes]
        for e in order["topology"].get("edges", []):
            s, t = rid2.get(e.get("in")), rid2.get(e.get("out"))
            if s is not None and t is not None:
                adj[s].add(t); adj[t].add(s)
        # 有向（沿 in->out 方向）
        out_adj = [set() for _ in nodes]
        in_adj = [set() for _ in nodes]
        for e in order["topology"].get("edges", []):
            s, t = rid2.get(e.get("in")), rid2.get(e.get("out"))
            if s is not None and t is not None:
                out_adj[s].add(t); in_adj[t].add(s)

        # TargetAlarm 节点
        target_idx = [rid2[a.get("@rid")] for a in alarms
                      if a.get("label") == "TargetAlarm" and rid2.get(a.get("@rid")) is not None]

        # 到 target 的有向距离（out: target->node 方向，in: node->target）
        def bfs(adjacency, starts):
            dist = np.full(len(nodes), len(nodes) + 1, dtype=np.float32)
            q = deque()
            for s in starts:
                dist[s] = 0; q.append(s)
            while q:
                cur = q.popleft()
                for nb in adjacency[cur]:
                    if dist[cur] + 1 < dist[nb]:
                        dist[nb] = dist[cur] + 1; q.append(nb)
            return dist

        if target_idx:
            d_from_t = bfs(out_adj, target_idx)   # target 出发
            d_to_t = bfs(in_adj, target_idx)      # 指向 target
        else:
            d_from_t = np.full(len(nodes), 999, dtype=np.float32)
            d_to_t = np.full(len(nodes), 999, dtype=np.float32)

        # 时间
        fault_time = float(order["topology"].get("time", 0) or 0)
        alarm_times = [float(a.get("time", fault_time) or fault_time) for a in alarms]

        # 告警组：title / title+device / title+reason
        title_grp = defaultdict(list)
        tdev_grp = defaultdict(list)
        treason_grp = defaultdict(list)
        for j, a in enumerate(alarms):
            title_grp[a.get("title", "")].append(j)
            tdev_grp[(a.get("title", ""), a.get("device", ""))].append(j)
            treason_grp[(a.get("title", ""), a.get("reason", ""))].append(j)

        # 全局频率（leave-one-site-out 时需在 fold 层重算，这里先粗算）
        for j, a in enumerate(alarms):
            ni = rid2.get(a.get("@rid"), -1)
            deg = len(adj[ni]) if ni >= 0 else 0
            in_deg = len(in_adj[ni]) if ni >= 0 else 0
            out_deg = len(out_adj[ni]) if ni >= 0 else 0

            # V16 特征
            v16 = v16_features[offset + j] if v16_features is not None else np.zeros(1)

            # 图距离
            dft = d_from_t[ni] if ni >= 0 else 999
            dtt = d_to_t[ni] if ni >= 0 else 999

            # 时间顺序（相对最早告警）
            t_rel = (alarm_times[j] - min(alarm_times)) / max(max(alarm_times) - min(alarm_times), 1.0)

            # 告警组统计
            tg = title_grp.get(a.get("title", ""), [j])
            tdg = tdev_grp.get((a.get("title", ""), a.get("device", "")), [j])
            trg = treason_grp.get((a.get("title", ""), a.get("reason", "")), [j])

            feat = [
                float(v11_scores[offset + j]),                     # 0: V11 score
                float(deg), float(in_deg), float(out_deg),         # 1-3: 度
                float(dft), float(dtt),                            # 4-5: 图距离
                float(t_rel),                                      # 6: 相对时间
                float(len(tg)), float(len(tdg)), float(len(trg)),  # 7-9: 组大小
                float(ni in target_idx) if ni >= 0 else 0.0,       # 10: 是否target
                float(len(alarms)),                                # 11: 工单规模
                float(len(target_idx)),                            # 12: target数
            ]
            # V16 降维特征（取前 20 维 + 均值/方差）
            v16_vec = np.asarray(v16, dtype=np.float32)
            if v16_vec.shape[0] >= 20:
                feat.extend(v16_vec[:20].tolist())
            else:
                feat.extend([0.0] * 20)

            rows.append(feat)
            meta.append({
                "order_idx": oi,
                "order_id": order["id"],
                "rid": a.get("@rid"),
                "title": a.get("title", ""),
                "reason": a.get("reason", ""),
                "device": a.get("device", ""),
                "site": site_of(a.get("location", "")),
                "local_idx": j,
            })

        offset += n

    return np.array(rows, dtype=np.float32), meta


def order_signature(order):
    tc = Counter(v10.scalar(n.get("title")) for n in order["alarms"])
    tt = sorted(v10.scalar(n.get("title")) for n in order["alarms"] if n.get("label") == "TargetAlarm")
    return (tuple(sorted(tc.items())), tuple(tt), len(order["alarms"]))


def grouped_folds(orders, n_folds=5):
    groups = defaultdict(list)
    for i, o in enumerate(orders):
        groups[order_signature(o)].append(i)
    fs = [0] * n_folds
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(groups.items(), key=lambda x: (-len(x[1]), hashlib.sha256(repr(x[0]).encode()).hexdigest()))
    for _, idx in ranked:
        f = min(range(n_folds), key=lambda v: (fs[v], v))
        folds[idx] = f; fs[f] += len(idx)
    return folds


def main():
    print("=" * 60)
    print("V29 域稳健 Ranker + Oracle Gate")
    print("=" * 60)

    # 加载
    train = v10.load_orders(TRAIN_DIR, True)
    test = v10.load_orders(TEST_DIR, False)
    v16_tr = np.load(V16_TRAIN).astype(np.float32)
    v16_te = np.load(V16_TEST).astype(np.float32)

    v11_tr = (0.25 * np.load(V11_CTX) + 0.75 * np.load(V11_META)).astype(np.float32)
    v11_te = np.load(V11_TEST).astype(np.float32)

    # 训练集 slices + labels
    slices = []
    labels = []
    off = 0
    for o in train:
        n = len(o["alarms"])
        slices.append(slice(off, off + n)); off += n
        for a in o["alarms"]:
            labels.append(int(a["@rid"] in o["roots"]))
    labels = np.array(labels, dtype=np.int8)

    target_train = round(1059 / 546 * len(train))
    base_mask = v10.exact_count_mask(v11_tr, slices, target_train, MAX_ROOT)

    print(f"train={len(train)} test={len(test)} rows={len(labels)} pos={labels.sum()}")

    # 构造训练特征
    X, meta = build_features(train, v11_tr, v16_tr, True, labels)
    print(f"特征矩阵: {X.shape}")

    # Oracle gate：完美重排的 FN 上限
    base_tp = int(np.sum(base_mask & (labels == 1)))
    base_fn = int(np.sum((base_mask == 0) & (labels == 1)))
    base_fp = int(np.sum(base_mask & (labels == 0)))
    total_pos = int(labels.sum())
    print(f"\nV11 基线: TP={base_tp} FN={base_fn} FP={base_fp}")

    # 每个工单内，V11 漏选的 TP 数 = 可找回上限（如果 ranker 完美）
    # 但受每单 cap=8 约束
    print(f"Oracle 上限（每单 cap=8，完美 ranker）:")
    # 已经是 known: oracle 从 2826 到 3016~3031 的增益
    print(f"  理论可找回 FN: {base_fn} (训练集)")

    # 关键：当前 checkpoint 是 955 TP @ 1045（线上），训练对应 2826 TP @ 3169
    # 我们关心的是：新 ranker 能否在 top-8 内把 rank 2-3 的 FN 提上来

    # rank 2-3 FN 统计
    fn_rank23 = 0
    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base_mask[sl]
        ov = v11_tr[sl]
        fn_local = np.where((ob == 0) & (ol == 1))[0]
        for j in fn_local:
            rank = int(np.sum(ov >= ov[j]))
            if rank in (2, 3):
                fn_rank23 += 1
    print(f"  rank 2-3 的 FN: {fn_rank23} ({fn_rank23/max(base_fn,1):.1%})")

    # 保存特征供后续 ranker 训练
    out_dir = Path("D:/zgyidong/experiments/v29_domain_ranker")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "train_features.npy", X)
    np.save(out_dir / "train_labels.npy", labels)
    np.save(out_dir / "train_base_mask.npy", base_mask)
    np.save(out_dir / "train_slices.npy", np.array([(s.start, s.stop) for s in slices]))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"train_orders": len(train), "rows": len(labels)}, f)

    print(f"\n特征已保存: {out_dir}")
    print("下一步: leave-one-site-out 训练 LightGBM ranker，测 OOF 收益")


if __name__ == "__main__":
    main()
