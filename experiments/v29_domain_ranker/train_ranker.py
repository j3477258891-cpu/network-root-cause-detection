"""
V29 — Ranker 训练 + OOF 收益评估
用 LightGBM 在工单内重新排序，测量相对 V11 的 TP 增量。
两个验证口径：
1. 模板分组 5-fold（标准）
2. leave-one-site-out（域稳健）
"""

import sys, json, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v29_domain_ranker")
from build_features import build_features, site_of, order_signature, grouped_folds

import lightgbm as lgb

TRAIN_DIR = Path("D:/zgyidong/train")
V16_TRAIN = Path("D:/zgyidong/experiments/v16/v16_train_features.npy")
V11_CTX = "D:/zgyidong/codexgz/v11/v11_oof_context.npy"
V11_META = "D:/zgyidong/codexgz/v11/v11_oof_meta.npy"
MAX_ROOT = 8


def confusion(mask, labels):
    tp = int(np.sum(mask & (labels == 1)))
    fp = int(np.sum(mask & (labels == 0)))
    fn = int(np.sum((~mask) & (labels == 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, tp, fp, fn


def eval_reorder(ranker_scores, slices, labels, base_mask, target_total):
    """用 ranker 分数重排，比较 TP delta vs V11 base_mask。"""
    new_mask = v10.exact_count_mask(ranker_scores, slices, target_total, MAX_ROOT)
    base_tp = int(np.sum(base_mask & (labels == 1)))
    new_tp = int(np.sum(new_mask & (labels == 1)))
    return new_tp - base_tp, base_tp, new_tp


def train_and_eval_template_folds(X, labels, slices, base_mask, v11_tr, folds, meta):
    """5-fold 模板分组 OOF。"""
    all_oi = np.arange(len(slices))
    target = round(1059 / 546 * len(slices))
    oof_scores = np.zeros(len(labels), dtype=np.float32)

    # 关键：ranker 特征里 V11 score 是特征0，但我们要 ranker 学"除了V11还有什么信号"
    # 所以用 X 的全部特征（含 V11 score 作为基准 + 其余作为补充）
    for heldout in range(5):
        tr_oi = all_oi[folds != heldout]
        val_oi = all_oi[folds == heldout]

        tr_idx = []
        val_idx = []
        for oi in tr_oi:
            tr_idx.extend(range(slices[oi].start, slices[oi].stop))
        for oi in val_oi:
            val_idx.extend(range(slices[oi].start, slices[oi].stop))
        tr_idx = np.array(tr_idx); val_idx = np.array(val_idx)

        X_tr, y_tr = X[tr_idx], labels[tr_idx]
        X_val = X[val_idx]

        m = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, class_weight="balanced",
            random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        oof_scores[val_idx] = m.predict_proba(X_val)[:, 1]

    delta, base_tp, new_tp = eval_reorder(oof_scores, slices, labels, base_mask, target)
    return delta, base_tp, new_tp, oof_scores


def train_and_eval_site_loso(X, labels, slices, base_mask, meta, target):
    """leave-one-site-out：按 SubNetwork 站点留出。"""
    # meta 是 list of dict，带 site
    n = len(meta)
    sites = [m["site"] for m in meta]
    site_set = sorted(set(sites))

    oof_scores = np.zeros(len(labels), dtype=np.float32)

    for site in site_set:
        val_idx = np.array([i for i in range(n) if sites[i] == site])
        tr_idx = np.array([i for i in range(n) if sites[i] != site])

        if len(tr_idx) < 50 or len(val_idx) == 0:
            continue

        m = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, class_weight="balanced",
            random_state=42, verbose=-1,
        )
        m.fit(X[tr_idx], labels[tr_idx])
        oof_scores[val_idx] = m.predict_proba(X[val_idx])[:, 1]

    delta, base_tp, new_tp = eval_reorder(oof_scores, slices, labels, base_mask, target)
    return delta, base_tp, new_tp, oof_scores


def main():
    print("=" * 60)
    print("V29 Ranker OOF 收益评估")
    print("=" * 60)

    # 加载
    train = v10.load_orders(TRAIN_DIR, True)
    v16_tr = np.load(V16_TRAIN).astype(np.float32)
    v11_tr = (0.25 * np.load(V11_CTX) + 0.75 * np.load(V11_META)).astype(np.float32)

    slices = []; labels = []
    off = 0
    for o in train:
        n = len(o["alarms"])
        slices.append(slice(off, off + n)); off += n
        for a in o["alarms"]:
            labels.append(int(a["@rid"] in o["roots"]))
    labels = np.array(labels, dtype=np.int8)

    target = round(1059 / 546 * len(train))
    base_mask = v10.exact_count_mask(v11_tr, slices, target, MAX_ROOT)

    X, meta = build_features(train, v11_tr, v16_tr, True, labels)
    folds = grouped_folds(train)

    base_tp = int(np.sum(base_mask & (labels == 1)))
    print(f"V11 基线: TP={base_tp} (OOF)")

    # 口径1：模板分组
    print("\n[口径1] 模板分组 5-fold:")
    d1, bt1, nt1, oof1 = train_and_eval_template_folds(X, labels, slices, base_mask, v11_tr, folds, meta)
    print(f"  delta={d1:+d}  base={bt1}  new={nt1}")

    # 口径2：leave-one-site-out
    print("\n[口径2] leave-one-site-out:")
    d2, bt2, nt2, oof2 = train_and_eval_site_loso(X, labels, slices, base_mask, meta, target)
    print(f"  delta={d2:+d}  base={bt2}  new={nt2}")

    # 融合：两口径平均
    print("\n[融合] 两口径 OOF 平均:")
    oof_avg = (oof1 + oof2) / 2
    delta_avg, bt_avg, nt_avg = eval_reorder(oof_avg, slices, labels, base_mask, target)
    print(f"  delta={delta_avg:+d}  base={bt_avg}  new={nt_avg}")

    # 保存 OOF 分数供后续候选池构建
    out = Path("D:/zgyidong/experiments/v29_domain_ranker")
    np.save(out / "oof_template.npy", oof1)
    np.save(out / "oof_site_loso.npy", oof2)
    np.save(out / "oof_avg.npy", oof_avg)

    # 结论
    print("\n" + "=" * 60)
    print("闸门结论:")
    for name, d in [("模板5fold", d1), ("site-LOSO", d2), ("融合", delta_avg)]:
        verdict = "✅ 有增益" if d > 5 else ("⚠️ 微弱" if d > 0 else "❌ 无增益")
        print(f"  {name}: {d:+d} TP  {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
