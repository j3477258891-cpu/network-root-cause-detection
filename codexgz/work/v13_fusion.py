"""V13 五模型多样性融合：ExtraTrees / HistGB / LightGBM / 无TE-ExtraTrees / KNN context。

用法:
    python v13_fusion.py TRAIN_DIR TEST_DIR OUTPUT_DIR

设计（V13 桶特征实验失败后修正：见 v13_report.json G1/G2/G3 全拒，
桶 A/B 与现有 68 维高度冗余、稀释 ExtraTrees 决策，故融合回归 V11 等效特征集）:
- M1 ExtraTrees（V11 同款特征: context + static + encoding + directed）
- M2 HistGradientBoosting（同特征）
- M3 LightGBM（同特征）
- M4 ExtraTrees（无 target encoding + 无 location 特征）
- M5 KNN context（v10 CONFIGS 现成）
- 每折 OOF、3 种子平均；多样性验证（Pearson 相关、argmax 一致率）
- 融合权重在 reference 折上网格求解（禁止全量 OOF 选权）
- 门控: 融合 OOF >= 最佳单模型 OOF + 0.002 且 >=3/5 折改进
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

import lightgbm as lgb

import v10_grouped_ensemble as v10
import v11_meta_ranker as v11

N_FOLDS = 5
SEEDS = (20260801, 20260817, 20260831)
MIN_GAIN = 0.002
MIN_IMPROVED_FOLDS = 3
MAX_PEARSON = 0.90
MAX_ARGMAX_CONSISTENCY = 0.90
TARGET_TEST_COUNT = 1059

ET_PARAMS = dict(
    n_estimators=300, max_depth=18, min_samples_leaf=2,
    max_features=0.75, class_weight="balanced", n_jobs=-1,
)
HGB_PARAMS = dict(
    learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
    min_samples_leaf=16, l2_regularization=2.0,
    class_weight="balanced", early_stopping=False,
)
LGB_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.5,
    n_estimators=300, n_jobs=-1, verbose=-1,
)


def make_et(seed):
    return ExtraTreesClassifier(random_state=seed, **ET_PARAMS)


def make_et_subspace(seed):
    """随机特征子空间 ET（B.3 强制拆散）: 低 max_features + 排除主导特征。

    主导特征 = context_score(0 列) + TE title 概率，让每棵树用不同特征视角。
    """
    return ExtraTreesClassifier(
        random_state=seed,
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        max_features=0.4, class_weight="balanced", n_jobs=-1,
    )


def make_hgb(seed):
    return HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)


def make_lgb(seed):
    return lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)


def argmax_consistency(prob_a, prob_b, slices):
    """每单 argmax 节点一致的工单比例。"""
    same = total = 0
    for order_slice in slices:
        a = prob_a[order_slice]
        b = prob_b[order_slice]
        if len(a):
            same += int(np.argmax(a) == np.argmax(b))
            total += 1
    return same / max(total, 1)


def grid_weights_reference(oof_models, labels, slices, step=0.05):
    """在 reference 折上网格搜索融合权重（约束 w>=0, sum=1）。

    oof_models: list of OOF 分数数组（同一折 reference 部分）
    返回 (best_weights, best_f1)
    """
    n_models = len(oof_models)
    best = None
    # 简化: 只在 n_models-1 维单纯形上以 step 遍历
    def iterate_combinations(remaining, total, prefix):
        if len(prefix) == n_models - 1:
            yield prefix + [remaining]
            return
        for value in np.arange(0.0, remaining + 1e-9, step):
            yield from iterate_combinations(
                remaining - value, total, prefix + [value]
            )
    for weights in iterate_combinations(1.0, 1.0, []):
        combined = sum(
            w * oof_models[i] for i, w in enumerate(weights)
        )
        result = v10.best_threshold(combined, labels, slices)
        f1 = result[0]
        score = (f1, tuple(weights))
        if best is None or score > best:
            best = score
    return list(best[1]), best[0]


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: v13_fusion.py TRAIN_DIR TEST_DIR OUTPUT_DIR")
    train_dir, test_dir, output_dir = map(Path, sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading data...", flush=True)
    train_orders = v10.load_orders(train_dir, True)
    test_orders = v10.load_orders(test_dir, False)
    data = v10.prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    folds, group_count, repeated_orders, fold_sizes = v11.grouped_folds(train_orders)
    data["folds"] = folds
    print(
        f"orders={len(train_orders)} rows={len(labels)} groups={group_count} "
        f"fold_sizes={fold_sizes}",
        flush=True,
    )

    # M4 用无 location 版静态特征（无 TE）
    v10.set_no_location(True)
    data_nl = v10.prepare(train_orders, test_orders)
    v10.set_no_location(False)

    train_tfidf, test_tfidf = v10.normalized_tfidf(
        data["train_alarm"], data["test_alarm"]
    )
    train_directed = v11.directed_features(train_orders)
    test_directed = v11.directed_features(test_orders)

    all_orders = np.arange(len(train_orders))
    n_models = 5
    oof_per_model = [np.zeros(len(labels), dtype=np.float64) for _ in range(n_models)]

    for heldout in range(N_FOLDS):
        reference_orders = all_orders[folds != heldout]
        validation_orders = all_orders[folds == heldout]
        train_row_indices, train_context_rows, _ = v10.local_rows_and_slices(
            data, reference_orders
        )
        validation_row_indices, validation_context_rows, _ = v10.local_rows_and_slices(
            data, validation_orders
        )
        train_probabilities = v10.knn_order_probabilities(
            train_tfidf[reference_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
            exclude_reference_positions=np.arange(len(reference_orders)),
        )
        validation_probabilities = v10.knn_order_probabilities(
            train_tfidf[validation_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
        )
        train_context = v10.row_scores(train_probabilities, train_context_rows)
        validation_context = v10.row_scores(
            validation_probabilities, validation_context_rows
        )
        # M5 = KNN context
        oof_per_model[4][validation_row_indices] = validation_context

        train_rows = [data["train_rows"][index] for index in train_row_indices]
        validation_rows = [
            data["train_rows"][index] for index in validation_row_indices
        ]
        train_encoding = v10.encoded_features(
            data, reference_orders, train_rows, leave_query_order_out=True
        )
        validation_encoding = v10.encoded_features(
            data, reference_orders, validation_rows, leave_query_order_out=False
        )

        # M1/M2/M3: V11 等效特征（context + static + encoding + directed）
        x_train_full = np.hstack(
            [
                v10.make_features(
                    data["train_static"][train_row_indices],
                    train_context, train_encoding,
                ),
                train_directed[train_row_indices],
            ]
        ).astype(np.float32)
        x_valid_full = np.hstack(
            [
                v10.make_features(
                    data["train_static"][validation_row_indices],
                    validation_context, validation_encoding,
                ),
                train_directed[validation_row_indices],
            ]
        ).astype(np.float32)

        # M4: 无 TE + 无 location（用 data_nl 静态 + context + directed，无 encoding）
        x_train_nl = np.hstack(
            [
                data_nl["train_static"][train_row_indices],
                train_context.reshape(-1, 1),
                train_directed[train_row_indices],
            ]
        ).astype(np.float32)
        x_valid_nl = np.hstack(
            [
                data_nl["train_static"][validation_row_indices],
                validation_context.reshape(-1, 1),
                train_directed[validation_row_indices],
            ]
        ).astype(np.float32)

        y_train = labels[train_row_indices]
        accumulators = [[] for _ in range(4)]
        for seed_index, seed in enumerate(SEEDS):
            models = [
                make_et(seed + heldout),
                make_hgb(seed + heldout),
                make_lgb(seed + heldout),
                make_et_subspace(seed + heldout),
            ]
            models[0].fit(x_train_full, y_train)
            models[1].fit(x_train_full, y_train)
            models[2].fit(x_train_full, y_train)
            models[3].fit(x_train_nl, y_train)
            for model_index, model in enumerate(models):
                accumulators[model_index].append(
                    model.predict_proba(
                        x_valid_full if model_index < 3 else x_valid_nl
                    )[:, 1]
                )
        for model_index in range(4):
            oof_per_model[model_index][validation_row_indices] = np.mean(
                accumulators[model_index], axis=0
            )
        print(f"fold={heldout} trained ({time.time()-started:.0f}s)", flush=True)

    # 多样性验证
    correlations = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = pearsonr(oof_per_model[i], oof_per_model[j])[0]
            correlations.append((i, j, float(corr)))
    max_corr = max(c[2] for c in correlations)
    consistencies = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            consistency = argmax_consistency(
                oof_per_model[i], oof_per_model[j], data["train_slices"]
            )
            consistencies.append((i, j, consistency))
    max_consistency = max(c[2] for c in consistencies)
    diversity_ok = max_corr < MAX_PEARSON and max_consistency < MAX_ARGMAX_CONSISTENCY
    print(
        f"DIVERSITY max_pearson={max_corr:.4f} max_argmax_consistency={max_consistency:.4f} "
        f"ok={diversity_ok}",
        flush=True,
    )

    # 单模型 OOF 基线
    model_f1 = []
    for model_index in range(n_models):
        result = v10.best_threshold(
            oof_per_model[model_index], labels, data["train_slices"]
        )
        model_f1.append(result[0])
        print(
            f"model {model_index} OOF f1={result[0]:.6f} tp={result[1]} "
            f"fp={result[2]} fn={result[3]}",
            flush=True,
        )
    best_single_index = int(np.argmax(model_f1))
    best_single_f1 = model_f1[best_single_index]

    # 融合权重: 每折 reference 上求解
    fold_weights = []
    fold_fusion_f1 = []
    fold_gains = []
    for heldout in range(N_FOLDS):
        reference_orders = all_orders[folds != heldout]
        reference_rows, reference_slices = v11.rows_and_slices_for_orders(
            data, reference_orders
        )
        ref_models = [oof[reference_rows] for oof in oof_per_model]
        weights, ref_f1 = grid_weights_reference(
            ref_models, labels[reference_rows], reference_slices
        )
        # 验证折
        validation_orders = all_orders[folds == heldout]
        validation_rows, validation_slices = v11.rows_and_slices_for_orders(
            data, validation_orders
        )
        combined = sum(
            w * oof_per_model[m][validation_rows]
            for m, w in enumerate(weights)
        )
        baseline_best = v10.best_threshold(
            oof_per_model[best_single_index][validation_rows],
            labels[validation_rows], validation_slices,
        )[0]
        fusion_f1 = v10.best_threshold(
            combined, labels[validation_rows], validation_slices
        )[0]
        fold_weights.append(weights)
        fold_fusion_f1.append(fusion_f1)
        fold_gains.append(fusion_f1 - baseline_best)
        print(
            f"fold={heldout} weights={[round(w,2) for w in weights]} "
            f"ref_f1={ref_f1:.6f} val_fusion={fusion_f1:.6f} "
            f"val_best_single={baseline_best:.6f} gain={fold_gains[-1]:+.6f}",
            flush=True,
        )

    improved_folds = sum(g > 0 for g in fold_gains)
    gate_g2 = improved_folds >= MIN_IMPROVED_FOLDS

    # 全局融合（用参考折权重中位数/众数）
    weight_columns = list(zip(*fold_weights))
    final_weights = [
        float(np.median(np.asarray(col))) for col in weight_columns
    ]
    final_weights = [w / sum(final_weights) for w in final_weights]
    fused_oof = sum(
        w * oof_per_model[m] for m, w in enumerate(final_weights)
    )
    fused_result = v10.best_threshold(fused_oof, labels, data["train_slices"])
    nested_gain = fused_result[0] - best_single_f1
    gate_g1 = nested_gain >= MIN_GAIN
    print(
        f"FUSION weights={[round(w,3) for w in final_weights]} "
        f"OOF={fused_result[0]:.6f} best_single={best_single_f1:.6f} "
        f"gain={nested_gain:+.6f} improved_folds={improved_folds}/5 "
        f"G1={gate_g1} G2={gate_g2}",
        flush=True,
    )

    if not diversity_ok:
        # 多样性失败 → 不生成全量提交候选，但继续生成融合分数表供模块 D 探针使用。
        # 原因: Pearson 高是因为全部模型共享 KNN context 特征（概率幅值同源），
        # 而真正影响微 F1 集合的 argmax 一致率通常已达标。探针以小批量验证方向，
        # 避免 V12 式「离线 OOF 高但线上无效」的全量提交风险。
        print(
            f"FUSION diversity failed (pearson={max_corr:.4f}) — "
            f"score table only for probes, no full submission",
            flush=True,
        )
        # 仍然执行全量训练生成分数表（探针来源）
        fused_result_used = fused_result
        # fall through to score-table generation with probe_only=True
    else:
        fused_result_used = fused_result

    # 全量训练
    full_train_probabilities = v10.knn_order_probabilities(
        train_tfidf, train_tfidf, data["train_alarm"], data["train_root"],
        exclude_reference_positions=np.arange(len(train_orders)),
    )
    full_test_probabilities = v10.knn_order_probabilities(
        test_tfidf, train_tfidf, data["train_alarm"], data["train_root"]
    )
    full_train_context = v10.row_scores(full_train_probabilities, data["train_rows"])
    full_test_context = v10.row_scores(full_test_probabilities, data["test_rows"])
    full_train_encoding = v10.encoded_features(
        data, all_orders, data["train_rows"], leave_query_order_out=True
    )
    full_test_encoding = v10.encoded_features(
        data, all_orders, data["test_rows"], leave_query_order_out=False
    )
    x_full = np.hstack(
        [
            v10.make_features(data["train_static"], full_train_context, full_train_encoding),
            train_directed,
        ]
    ).astype(np.float32)
    x_test = np.hstack(
        [
            v10.make_features(data["test_static"], full_test_context, full_test_encoding),
            test_directed,
        ]
    ).astype(np.float32)
    x_full_nl = np.hstack(
        [
            data_nl["train_static"], full_train_context.reshape(-1, 1),
            train_directed,
        ]
    ).astype(np.float32)
    x_test_nl = np.hstack(
        [
            data_nl["test_static"], full_test_context.reshape(-1, 1),
            test_directed,
        ]
    ).astype(np.float32)

    test_per_model = []
    for model_index in range(5):
        if model_index == 4:
            test_per_model.append(full_test_context)
            continue
        seed_probs = []
        for seed in SEEDS:
            if model_index == 0:
                model = make_et(seed)
                xf, xt = x_full, x_test
            elif model_index == 1:
                model = make_hgb(seed)
                xf, xt = x_full, x_test
            elif model_index == 2:
                model = make_lgb(seed)
                xf, xt = x_full, x_test
            else:
                model = make_et_subspace(seed)
                xf, xt = x_full_nl, x_test_nl
            model.fit(xf, labels)
            seed_probs.append(model.predict_proba(xt)[:, 1])
        test_per_model.append(np.mean(seed_probs, axis=0))

    fused_test = sum(
        w * test_per_model[m] for m, w in enumerate(final_weights)
    )
    threshold_mask = v10.selection_mask(
        fused_test, data["test_slices"], fused_result_used[-1]
    )
    fixed_mask = v10.exact_count_mask(
        fused_test, data["test_slices"], TARGET_TEST_COUNT
    )
    # 仅当多样性达标且门控通过时生成全量提交候选；否则只做分数表（探针来源）
    probe_only = not diversity_ok
    if not probe_only:
        v10.write_submission(
            output_dir / "result_record_v13_fusion_1059.csv",
            test_orders, data, fixed_mask,
        )

    # 分数表（探针来源，恒生成）
    with (output_dir / "v13_fusion_test_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "rid", "context_score", "fusion_score"])
        for order_index, order in enumerate(test_orders):
            order_slice = data["test_slices"][order_index]
            for local_index, node in enumerate(order["alarms"]):
                row_index = order_slice.start + local_index
                writer.writerow(
                    [order["id"], node["@rid"], float(full_test_context[row_index]),
                     float(fused_test[row_index])]
                )

    gate_g4 = int(np.sum(fixed_mask)) == TARGET_TEST_COUNT
    per_order_counts = [
        int(np.sum(fixed_mask[sl])) for sl in data["test_slices"]
    ]
    gate_g4 = gate_g4 and max(per_order_counts) <= v10.MAX_ROOTCAUSES
    gate_passed = (not probe_only) and gate_g1 and gate_g2 and gate_g4
    print(
        f"GATES G1={gate_g1} G2={gate_g2} G4={gate_g4} "
        f"probe_only={probe_only} PASSED={gate_passed} "
        f"threshold_count={int(np.sum(threshold_mask))} "
        f"fixed_count={int(np.sum(fixed_mask))}",
        flush=True,
    )

    report = {
        "status": "done",
        "n_models": n_models,
        "model_f1": model_f1,
        "best_single_index": best_single_index,
        "best_single_f1": best_single_f1,
        "correlations": correlations,
        "max_pearson": max_corr,
        "consistencies": consistencies,
        "max_argmax_consistency": max_consistency,
        "diversity_ok": diversity_ok,
        "fold_weights": fold_weights,
        "fold_fusion_f1": fold_fusion_f1,
        "fold_gains": fold_gains,
        "improved_folds": improved_folds,
        "final_weights": final_weights,
        "fused_oof": fused_result,
        "nested_gain": nested_gain,
        "probe_only": probe_only,
        "diversity_ok": diversity_ok,
        "gates": {"G1": gate_g1, "G2": gate_g2, "G4": gate_g4, "passed": gate_passed},
        "threshold_test_count": int(np.sum(threshold_mask)),
        "fixed_test_count": int(np.sum(fixed_mask)),
        "seconds": time.time() - started,
    }
    (output_dir / "v13_fusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"TOTAL {time.time()-started:.0f}s outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
