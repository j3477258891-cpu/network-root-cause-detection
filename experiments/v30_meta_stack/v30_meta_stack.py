"""Cross-fitted multi-model meta ranker and conservative action audit."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
DEPS = ROOT / ".deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V16 = ROOT / "experiments/v16"
V25 = ROOT / "experiments/v25_ensemble/outputs"
OUTPUT = ROOT / "experiments/v30_meta_stack"
TARGET_TRAIN = 3169
SEED = 20260818


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:8]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional_array = np.asarray(optional, dtype=np.int64)
    optional_array = optional_array[np.argsort(-scores[optional_array], kind="stable")]
    remaining = target - int(selected.sum())
    if not 0 <= remaining <= len(optional_array):
        raise ValueError((target, int(selected.sum()), len(optional_array)))
    selected[optional_array[:remaining]] = True
    return selected


def score_matrix(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    suffix = "oof" if split == "train" else "test"
    columns = [
        arrays[f"{split}_v11"],
        arrays[f"{split}_v13"],
        arrays[f"{split}_v19"],
        np.load(V16 / f"v16_{suffix}_scores.npy"),
        np.load(V16 / f"v16_graph_time_{suffix}_scores.npy")
        if (V16 / f"v16_graph_time_{suffix}_scores.npy").exists()
        else np.load(V16 / f"v16_graph_time_{'oof' if split == 'train' else 'test_scores'}.npy"),
    ]
    for name in ("M1", "M2", "M3", "M4", "stack"):
        columns.append(np.load(V25 / f"v25_{name}_{suffix}.npy"))
    output = np.column_stack(columns).astype(np.float32)
    fallback = np.repeat(output[:, :1], output.shape[1], axis=1)
    return np.where(np.isfinite(output), output, fallback).astype(np.float32)


def relative_features(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    rows = np.zeros((len(scores), scores.shape[1] * 4 + 2), dtype=np.float32)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = scores[start:stop]
        count = stop - start
        ranks = np.empty_like(local, dtype=np.float32)
        for column in range(local.shape[1]):
            order = np.argsort(-local[:, column], kind="stable")
            ranks[order, column] = np.arange(count, dtype=np.float32)
        means = local.mean(axis=0)
        stds = local.std(axis=0) + 1e-6
        rows[start:stop, : scores.shape[1]] = ranks / max(count - 1, 1)
        rows[start:stop, scores.shape[1] : 2 * scores.shape[1]] = (
            local - means
        ) / stds
        rows[start:stop, 2 * scores.shape[1] : 3 * scores.shape[1]] = (
            local.max(axis=0) - local
        )
        rows[start:stop, 3 * scores.shape[1] : 4 * scores.shape[1]] = (
            local - local.min(axis=0)
        )
        rows[start:stop, -2] = count
        rows[start:stop, -1] = np.arange(count, dtype=np.float32) / max(count - 1, 1)
    return rows


def feature_matrix(arrays: dict[str, np.ndarray], split: str):
    scores = score_matrix(arrays, split)
    static = np.nan_to_num(
        arrays[f"{split}_alarm_x"], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    paths = np.column_stack(
        [
            arrays[f"{split}_path_node_type"],
            arrays[f"{split}_path_edge_type"],
            arrays[f"{split}_path_length"],
        ]
    ).astype(np.float32)
    relative = relative_features(scores, arrays[f"{split}_alarm_ptr"])
    disagreements = np.column_stack(
        [scores.mean(axis=1), scores.std(axis=1), scores.min(axis=1), scores.max(axis=1)]
    ).astype(np.float32)
    return np.column_stack([static, scores, relative, disagreements, paths]), scores


def model_for(kind: str, quick: bool, seed: int):
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=160 if quick else 600,
            max_depth=None,
            min_samples_leaf=3,
            max_features=0.65,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    return HistGradientBoostingClassifier(
        max_iter=80 if quick else 260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=18,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def crossfit(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_test: np.ndarray,
    order_rows: np.ndarray,
    folds: np.ndarray,
    kind: str,
    quick: bool,
):
    oof = np.zeros(len(labels), dtype=np.float32)
    test_predictions = []
    for fold in range(5):
        train_rows = folds[order_rows] != fold
        validation_rows = ~train_rows
        model = model_for(kind, quick, SEED + fold)
        model.fit(x_train[train_rows], labels[train_rows])
        oof[validation_rows] = model.predict_proba(x_train[validation_rows])[:, 1]
        test_predictions.append(model.predict_proba(x_test)[:, 1])
    return oof, np.mean(test_predictions, axis=0).astype(np.float32)


def action_curve(scores, ptr, base, labels):
    actions = []
    for order_index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        count = int(base[start:stop].sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[:count]
        proposed = np.zeros(stop - start, dtype=bool)
        proposed[ranked] = True
        add = proposed & ~base[start:stop]
        remove = base[start:stop] & ~proposed
        if not add.any() and not remove.any():
            continue
        utility = float(scores[start:stop][add].sum() - scores[start:stop][remove].sum())
        delta = int((proposed & labels[start:stop]).sum() - (base[start:stop] & labels[start:stop]).sum())
        actions.append((utility, order_index, int(add.sum()), int(remove.sum()), delta))
    actions.sort(reverse=True)
    curve = {}
    for n in (5, 8, 10, 15, 20, 30, 40, 50, 75, 100):
        subset = actions[:n]
        curve[str(n)] = {
            "delta_tp": int(sum(row[4] for row in subset)),
            "positive_orders": int(sum(row[4] > 0 for row in subset)),
            "negative_orders": int(sum(row[4] < 0 for row in subset)),
            "adds": int(sum(row[2] for row in subset)),
            "removes": int(sum(row[3] for row in subset)),
        }
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    x_train, train_scores = feature_matrix(arrays, "train")
    x_test, test_scores = feature_matrix(arrays, "test")
    labels = arrays["train_labels"].astype(np.int8)
    ptr = arrays["train_alarm_ptr"]
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    base = exact_count_mask(arrays["train_v11"], ptr, TARGET_TRAIN)
    base_tp = int((base & labels.astype(bool)).sum())
    results = {}
    oofs, tests = [], []
    for split_name, fold_key in (
        ("template", "train_folds"),
        ("station", "train_station_folds"),
    ):
        for kind in ("extra_trees", "hist_gradient"):
            key = f"{split_name}_{kind}"
            oof, test = crossfit(
                x_train,
                labels,
                x_test,
                order_rows,
                arrays[fold_key],
                kind,
                args.quick,
            )
            oofs.append(oof)
            tests.append(test)
            np.save(OUTPUT / f"{key}_oof.npy", oof)
            np.save(OUTPUT / f"{key}_test.npy", test)
            mask = exact_count_mask(oof, ptr, TARGET_TRAIN)
            results[key] = {
                "tp": int((mask & labels.astype(bool)).sum()),
                "delta_tp": int((mask & labels.astype(bool)).sum()) - base_tp,
                "action_curve": action_curve(oof, ptr, base, labels.astype(bool)),
            }
    consensus_oof = np.mean(oofs, axis=0)
    consensus_test = np.mean(tests, axis=0)
    np.save(OUTPUT / "v30_consensus_oof.npy", consensus_oof)
    np.save(OUTPUT / "v30_consensus_test.npy", consensus_test)
    mask = exact_count_mask(consensus_oof, ptr, TARGET_TRAIN)
    results["consensus"] = {
        "tp": int((mask & labels.astype(bool)).sum()),
        "delta_tp": int((mask & labels.astype(bool)).sum()) - base_tp,
        "action_curve": action_curve(consensus_oof, ptr, base, labels.astype(bool)),
    }
    report = {
        "version": "v30-meta-stack-1",
        "quick": args.quick,
        "feature_count": int(x_train.shape[1]),
        "score_count": int(train_scores.shape[1]),
        "base_tp": base_tp,
        "results": results,
    }
    (OUTPUT / "v30_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
