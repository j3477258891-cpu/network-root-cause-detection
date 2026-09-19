"""Research-only audit of semantic embedding nearest-neighbour scores.

This script does not create a submission.  It evaluates whether the saved V25
semantic embeddings contain a signal that can improve the current fixed-count
baseline under order-level cross-validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
OUT = ROOT / "experiments/v25_semantic_router/cloud_outputs/semantic_knn_research.json"
WORK = ROOT / "codexgz/work"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
import v10_grouped_ensemble as v10  # noqa: E402


def slices_from_ptr(ptr: np.ndarray) -> list[slice]:
    return [slice(int(a), int(b)) for a, b in zip(ptr[:-1], ptr[1:])]


def fixed_count_mask(scores: np.ndarray, slices: list[slice], counts: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    for oi, sl in enumerate(slices):
        n = int(min(max(counts[oi], 1), sl.stop - sl.start))
        order = np.argsort(-scores[sl], kind="stable")[:n]
        mask[sl.start + order] = True
    return mask


def exact_global_mask(scores: np.ndarray, slices: list[slice], target: int) -> np.ndarray:
    # Match the repository's global budget rule: at least one per order, then
    # distribute the remaining slots by within-order rank score.
    base = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for sl in slices:
        local = scores[sl]
        order = np.argsort(-local, kind="stable")
        base[sl.start + order[0]] = True
        optional.extend((sl.start + order[1: min(8, sl.stop - sl.start)]).tolist())
    remaining = target - int(base.sum())
    optional_arr = np.asarray(optional, dtype=np.int64)
    optional_arr = optional_arr[np.argsort(-scores[optional_arr], kind="stable")]
    if remaining > 0:
        base[optional_arr[:remaining]] = True
    elif remaining < 0:
        chosen = np.flatnonzero(base)
        drop = chosen[np.argsort(scores[chosen])[: -remaining]]
        base[drop] = False
    return base


def eval_mask(mask: np.ndarray, baseline: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "predictions": int(mask.sum()),
        "tp": int(labels[mask].sum()),
        "baseline_predictions": int(baseline.sum()),
        "baseline_tp": int(labels[baseline].sum()),
        "delta_tp": int(labels[mask].sum() - labels[baseline].sum()),
        "added": int(np.sum(mask & ~baseline)),
        "removed": int(np.sum(baseline & ~mask)),
    }


def main() -> None:
    z = np.load(DATASET, allow_pickle=True)
    ptr = z["train_alarm_ptr"]
    slices = slices_from_ptr(ptr)
    labels = z["train_labels"].astype(np.int8)
    folds = z["train_folds"].astype(np.int8)
    v11 = z["train_v11"].astype(np.float32)
    target = int(round(1059 / 546 * len(slices)))
    baseline = v10.exact_count_mask(v11, slices, target, 8)

    emb_paths = sorted((ROOT / "experiments/v25_semantic_router/cloud_outputs").glob("semantic_seed_*_fold_*.npz"))
    if not emb_paths:
        raise FileNotFoundError("no semantic fold embeddings")
    # The saved embedding bundles use the same encoder; use the first bundle
    # for this diagnostic and explicitly record all available bundles.
    emb = np.asarray(np.load(emb_paths[0], allow_pickle=True)["train_embeddings"], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-8)

    results: dict[str, object] = {
        "dataset": str(DATASET),
        "embedding_source": str(emb_paths[0]),
        "available_embedding_bundles": [str(p) for p in emb_paths],
        "train_rows": int(len(labels)),
        "target_train_predictions": target,
        "baseline_tp": int(labels[baseline].sum()),
        "settings": [],
        "folds": {},
    }

    for k in (5, 10, 20, 50, 100):
        results["settings"].append({"k": k, "weight": "cosine_rank"})
        fold_rows: dict[str, object] = {}
        all_mask = np.zeros(len(labels), dtype=bool)
        for fold in range(5):
            val_rows = np.flatnonzero(folds == fold)
            train_rows = np.flatnonzero(folds != fold)
            nn = NearestNeighbors(n_neighbors=min(k, len(train_rows)), metric="cosine", algorithm="brute", n_jobs=-1)
            nn.fit(emb[train_rows])
            dist, ind = nn.kneighbors(emb[val_rows], return_distance=True)
            sim = 1.0 - dist
            # Rank-weighted label estimate; nearest neighbours receive the
            # largest weight without treating raw cosine as calibrated.
            weights = 1.0 / np.arange(1, sim.shape[1] + 1, dtype=np.float32)
            scores = (z["train_labels"][train_rows[ind]] * weights).sum(axis=1) / weights.sum()
            all_scores = np.zeros(len(labels), dtype=np.float32)
            all_scores[val_rows] = scores
            local_slices = []
            offset = 0
            for oi in val_rows:
                n = slices[int(oi)].stop - slices[int(oi)].start
                local_slices.append(slice(offset, offset + n))
                offset += n
            # Reindex scores and labels to validation-only contiguous arrays.
            val_scores = np.concatenate([all_scores[s] for s in [slices[int(oi)] for oi in val_rows]])
            val_labels = np.concatenate([labels[s] for s in [slices[int(oi)] for oi in val_rows]])
            val_base = np.concatenate([baseline[s] for s in [slices[int(oi)] for oi in val_rows]])
            val_counts = np.asarray([int(baseline[s].sum()) for s in [slices[int(oi)] for oi in val_rows]])
            mask_local = fixed_count_mask(val_scores, local_slices, val_counts)
            exact_local = exact_global_mask(val_scores, local_slices, int(round(target * len(val_labels) / len(labels))))
            fold_rows[str(fold)] = {
                "fixed_count": eval_mask(mask_local, val_base, val_labels),
                "exact_global": eval_mask(exact_local, val_base, val_labels),
            }
            # put fixed-count mask back into global row positions for total
            off = 0
            for oi, s in zip(val_rows, [slices[int(oi)] for oi in val_rows]):
                n = s.stop - s.start
                all_mask[s.start:s.stop] = mask_local[off: off + n]
                off += n
        results["folds"][str(k)] = fold_rows
        results.setdefault("all_fixed_count", {})[str(k)] = eval_mask(all_mask, baseline, labels)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
