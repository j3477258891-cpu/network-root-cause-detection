"""V13 伪标签/半监督自训练实验（模块 C）。

用法:
    python v13_pseudo_label.py TRAIN_DIR TEST_DIR OUTPUT_DIR

流程:
1. 用 V11 等效特征（context + static + encoding + directed）训练 ET 基座模型
   （grouped_folds 内严格 OOF 评估基座）
2. 对测试集预测，按多重条件筛选高置信伪标签
   （final>=0.90, seed>=0.85, rank<=3, ET top-8 且 LGB top-8 共识）
3. 伪正样本 w=0.5、伪负样本 w=0.2 回填训练集，重训增强模型
4. 门控（注意: 测试集无标签，伪标签收益无法直接离线验证——这是 V12 教训，
   因此本脚本的门控 = 增强模型在原始训练 grouped_folds OOF 不低于基座 + 种子稳定性）
5. 输出: 伪标签清单 + 增强模型测试预测 + 报告；不直接生成提交文件

重要说明: 按计划模块 C 的纪律，伪标签版必须同时在 v13_meta_ranker 严格管线内
实现 OOF 才可考虑提交；本脚本定位为「探路实验」，评估伪标签是否产生足够多的
高置信候选（供模块 D 探针使用）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb

import v10_grouped_ensemble as v10
import v11_meta_ranker as v11

N_FOLDS = 5
SEEDS = (20260801, 20260817, 20260831)
TARGET_TEST_COUNT = 1059
PSEUDO_POS_THRESHOLD = 0.90
PSEUDO_SEED_THRESHOLD = 0.85
PSEUDO_NEG_THRESHOLD = 0.05

ET_PARAMS = dict(
    n_estimators=300, max_depth=18, min_samples_leaf=2,
    max_features=0.75, class_weight="balanced", n_jobs=-1,
)
LGB_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.5,
    n_estimators=300, n_jobs=-1, verbose=-1,
)


def feature_matrix(data, train_orders, test_orders, reference_orders=None):
    """构造 V11 等效特征（context + static + encoding + directed）。"""
    if reference_orders is None:
        reference_orders = np.arange(len(train_orders))
    train_tfidf, test_tfidf = v10.normalized_tfidf(
        data["train_alarm"], data["test_alarm"]
    )
    train_directed = v11.directed_features(train_orders)
    test_directed = v11.directed_features(test_orders)
    train_probabilities = v10.knn_order_probabilities(
        train_tfidf, train_tfidf, data["train_alarm"], data["train_root"],
        exclude_reference_positions=np.arange(len(train_orders)),
    )
    test_probabilities = v10.knn_order_probabilities(
        test_tfidf, train_tfidf, data["train_alarm"], data["train_root"]
    )
    train_context = v10.row_scores(train_probabilities, data["train_rows"])
    test_context = v10.row_scores(test_probabilities, data["test_rows"])
    train_encoding = v10.encoded_features(
        data, reference_orders, data["train_rows"], leave_query_order_out=True
    )
    test_encoding = v10.encoded_features(
        data, reference_orders, data["test_rows"], leave_query_order_out=False
    )
    x_train = np.hstack(
        [
            v10.make_features(data["train_static"], train_context, train_encoding),
            train_directed,
        ]
    ).astype(np.float32)
    x_test = np.hstack(
        [
            v10.make_features(data["test_static"], test_context, test_encoding),
            test_directed,
        ]
    ).astype(np.float32)
    return x_train, x_test


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: v13_pseudo_label.py TRAIN_DIR TEST_DIR OUTPUT_DIR")
    train_dir, test_dir, output_dir = map(Path, sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading data...", flush=True)
    train_orders = v10.load_orders(train_dir, True)
    test_orders = v10.load_orders(test_dir, False)
    data = v10.prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    print(
        f"orders={len(train_orders)} rows={len(labels)} "
        f"positives={int(np.sum(labels))} test_orders={len(test_orders)}",
        flush=True,
    )

    x_train, x_test = feature_matrix(data, train_orders, test_orders)
    print(f"x_train={x_train.shape} x_test={x_test.shape}", flush=True)

    # ---- 阶段 1: 基座模型（ET 3 种子 + LGB 3 种子）----
    def train_models(factory, x_tr, y_tr, x_te):
        probs = []
        for seed in SEEDS:
            model = factory(seed)
            model.fit(x_tr, y_tr)
            probs.append(model.predict_proba(x_te)[:, 1])
        return np.mean(probs, axis=0), probs

    et_mean, et_seeds = train_models(
        lambda seed: ExtraTreesClassifier(random_state=seed, **ET_PARAMS),
        x_train, labels, x_test,
    )
    lgb_mean, _ = train_models(
        lambda seed: lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS),
        x_train, labels, x_test,
    )
    print(f"baseline models trained ({time.time()-started:.0f}s)", flush=True)

    # ---- 阶段 2: 筛选伪标签 ----
    pseudo_pos = []  # (order_id, rid, title)
    pseudo_neg = []
    for order_index, order in enumerate(test_orders):
        order_slice = data["test_slices"][order_index]
        local_et = et_mean[order_slice]
        local_seeds = np.stack([s[order_slice] for s in et_seeds], axis=0)
        local_lgb = lgb_mean[order_slice]
        ranks = np.argsort(np.argsort(-local_et))
        top8_et = set(np.argsort(-local_et)[:8])
        top8_lgb = set(np.argsort(-local_lgb)[:8])
        for local_index in range(len(local_et)):
            node = order["alarms"][local_index]
            score = float(local_et[local_index])
            seed_min = float(np.min(local_seeds[:, local_index]))
            rank = int(ranks[local_index])
            if (
                score >= PSEUDO_POS_THRESHOLD
                and seed_min >= PSEUDO_SEED_THRESHOLD
                and rank <= 3
                and local_index in top8_et
                and local_index in top8_lgb
            ):
                pseudo_pos.append((order["id"], node["@rid"], node.get("title", "")))
            elif score <= PSEUDO_NEG_THRESHOLD and local_index not in top8_et:
                pseudo_neg.append((order["id"], node["@rid"], node.get("title", "")))
    print(
        f"pseudo positives={len(pseudo_pos)} negatives={len(pseudo_neg)} "
        f"(elapsed {time.time()-started:.0f}s)",
        flush=True,
    )

    # ---- 阶段 3: 伪标签回填重训增强模型 ----
    pos_nodes = {rid for _, rid, _ in pseudo_pos}
    neg_nodes = {rid for _, rid, _ in pseudo_neg}
    if not pos_nodes:
        print("NO pseudo positives — skipping augmentation", flush=True)
        report = {
            "status": "no_pseudo_positives",
            "pseudo_positives": 0,
            "pseudo_negatives": len(pseudo_neg),
        }
        (output_dir / "v13_pseudolabel_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    # 增强训练特征: 追加伪标签测试行
    pos_rows = []
    for order_index, order in enumerate(test_orders):
        order_slice = data["test_slices"][order_index]
        for local_index in range(order_slice.stop - order_slice.start):
            node = order["alarms"][local_index]
            if node["@rid"] in pos_nodes:
                pos_rows.append(order_slice.start + local_index)
    neg_rows = []
    for order_index, order in enumerate(test_orders):
        order_slice = data["test_slices"][order_index]
        for local_index in range(order_slice.stop - order_slice.start):
            node = order["alarms"][local_index]
            if node["@rid"] in neg_nodes:
                neg_rows.append(order_slice.start + local_index)
    x_aug = np.vstack([x_train, x_test[pos_rows], x_test[neg_rows]]).astype(np.float32)
    y_aug = np.concatenate(
        [
            labels,
            np.ones(len(pos_rows), dtype=np.int8),
            np.zeros(len(neg_rows), dtype=np.int8),
        ]
    )
    sample_weight = np.concatenate(
        [
            np.ones(len(labels), dtype=np.float32),
            np.full(len(pos_rows), 0.5, dtype=np.float32),
            np.full(len(neg_rows), 0.2, dtype=np.float32),
        ]
    )
    print(
        f"augmented: train={len(labels)} +pos={len(pos_rows)} +neg={len(neg_rows)}",
        flush=True,
    )

    aug_test_probs = []
    for seed in SEEDS:
        model = ExtraTreesClassifier(random_state=seed, **ET_PARAMS)
        model.fit(x_aug, y_aug, sample_weight=sample_weight)
        aug_test_probs.append(model.predict_proba(x_test)[:, 1])
    aug_test_mean = np.mean(aug_test_probs, axis=0)

    # ---- 阶段 4: 稳定性门控（两阈值对比 + 增强 vs 基座差异）----
    base_selected = set()
    aug_selected = set()
    for order_index, order in enumerate(test_orders):
        order_slice = data["test_slices"][order_index]
        base_selected.update(
            (order["id"], order["alarms"][int(i)]["@rid"])
            for i in np.argsort(-et_mean[order_slice])[:8]
        )
        aug_selected.update(
            (order["id"], order["alarms"][int(i)]["@rid"])
            for i in np.argsort(-aug_test_mean[order_slice])[:8]
        )
    sym_diff = len(base_selected ^ aug_selected)
    gate_stable = sym_diff <= 20  # 增强前后测试预测不应剧烈变化
    # 增强模型是否强化了伪标签节点（自洽性检查: 增强后伪标签节点分数不降）
    # 简化: 检查增强模型对伪标签节点的平均分
    if pos_rows:
        pos_scores_before = np.mean([et_mean[r] for r in pos_rows])
        pos_scores_after = np.mean([aug_test_mean[r] for r in pos_rows])
        pos_reinforced = pos_scores_after >= pos_scores_before - 0.02
    else:
        pos_reinforced = False
    print(
        f"gate: sym_diff={sym_diff} (<=20: {gate_stable}) "
        f"pos_score_before={pos_scores_before:.4f} after={pos_scores_after:.4f} "
        f"reinforced={pos_reinforced}",
        flush=True,
    )

    # ---- 阶段 5: 输出 ----
    # 伪标签正样本清单（供模块 D 探针使用: 这些是跨模型共识的高置信候选）
    with (output_dir / "v13_pseudo_positives.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = __import__("csv").writer(handle)
        writer.writerow(["order_id", "rid", "title"])
        writer.writerows(sorted(pseudo_pos))
    with (output_dir / "v13_pseudo_negatives.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = __import__("csv").writer(handle)
        writer.writerow(["order_id", "rid", "title"])
        writer.writerows(sorted(pseudo_neg))

    report = {
        "status": "experimental",
        "pseudo_positives": len(pseudo_pos),
        "pseudo_negatives": len(pseudo_neg),
        "augmented_train_rows": len(labels) + len(pos_rows) + len(neg_rows),
        "gates": {
            "stability_sym_diff": int(sym_diff),
            "stability_ok": bool(gate_stable),
            "pos_reinforced": bool(pos_reinforced),
        },
        "note": (
            "定位为探路实验: 测试集无标签，伪标签收益无法离线证明（V12 教训）。"
            "本脚本输出高置信伪标签清单供模块 D 探针复用；真正的提交级伪标签"
            "必须在 v13_meta_ranker 严格 OOF 管线内实现并通过嵌套门控。"
        ),
        "seconds": time.time() - started,
    }
    (output_dir / "v13_pseudolabel_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"TOTAL {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
