"""Fold-honest semantic/template retrieval and alarm label transfer."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from v25_common import Bundle, read_json, write_json


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def order_embeddings(alarm_embeddings, ptr):
    output = np.zeros((len(ptr) - 1, alarm_embeddings.shape[1]), dtype=np.float32)
    for index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        output[index] = np.asarray(alarm_embeddings[int(start) : int(stop)], dtype=np.float32).mean(axis=0)
    return normalize(output)


def title_jaccard(left, right):
    left = Counter(item["title"] for item in left["alarms"])
    right = Counter(item["title"] for item in right["alarms"])
    union = sum((left | right).values())
    return sum((left & right).values()) / max(union, 1)


def transfer_one(
    query_index,
    reference_indices,
    query_orders,
    reference_orders,
    query_embeddings,
    reference_embeddings,
    query_ptr,
    reference_ptr,
    labels,
    config,
):
    q_start, q_stop = map(int, query_ptr[query_index : query_index + 2])
    q_order = query_orders[query_index]
    q_vector = normalize(query_embeddings[q_start:q_stop])
    order_q = q_vector.mean(axis=0)
    similarities = []
    for reference_index in reference_indices:
        r_start, r_stop = map(int, reference_ptr[reference_index : reference_index + 2])
        order_r = normalize(reference_embeddings[r_start:r_stop]).mean(axis=0)
        semantic = float(np.dot(order_q, order_r) / max(np.linalg.norm(order_q) * np.linalg.norm(order_r), 1e-8))
        lexical = title_jaccard(q_order, reference_orders[int(reference_index)])
        similarities.append((0.75 * semantic + 0.25 * lexical, int(reference_index)))
    similarities.sort(reverse=True)
    neighbors = similarities[: config["neighbors"]]
    second = neighbors[1][0] if len(neighbors) > 1 else -1.0
    best_margin = neighbors[0][0] - second if neighbors else 0.0
    probabilities = np.full(q_stop - q_start, np.nan, dtype=np.float32)
    support = np.zeros(q_stop - q_start, dtype=np.float32)
    consistency = np.zeros(q_stop - q_start, dtype=np.float32)
    match_similarity = np.zeros(q_stop - q_start, dtype=np.float32)
    for local, query_alarm in enumerate(q_order["alarms"]):
        votes, weights, matched = [], [], []
        for order_similarity, reference_index in neighbors:
            r_start, r_stop = map(int, reference_ptr[reference_index : reference_index + 2])
            candidates = normalize(reference_embeddings[r_start:r_stop]) @ q_vector[local]
            reference = reference_orders[reference_index]
            for candidate_index, candidate in enumerate(reference["alarms"]):
                if candidate["title"] == query_alarm["title"]:
                    candidates[candidate_index] += 0.15
                if candidate["location"] == query_alarm["location"]:
                    candidates[candidate_index] += 0.05
            chosen = int(np.argmax(candidates))
            similarity = float(candidates[chosen])
            if similarity < 0.55:
                continue
            votes.append(int(labels[r_start + chosen]))
            weights.append(max(order_similarity, 0.0) * max(similarity, 0.0))
            matched.append(similarity)
        support[local] = len(votes)
        if not votes:
            continue
        positive = float(np.average(votes, weights=np.maximum(weights, 1e-6)))
        agreement = max(sum(votes), len(votes) - sum(votes)) / len(votes)
        consistency[local] = agreement
        match_similarity[local] = float(np.mean(matched))
        if (
            len(votes) >= config["minimum_valid_neighbors"]
            and agreement >= config["minimum_consistency"]
            and best_margin >= config["minimum_similarity_margin"]
        ):
            probabilities[local] = positive
    return probabilities, support, consistency, match_similarity, float(best_margin)


def run_seed(bundle, semantic_dir, seed, config):
    train_count = len(bundle.records["train"])
    test_count = len(bundle.records["test"])
    train_rows = int(bundle.ptr("train")[-1])
    test_rows = int(bundle.ptr("test")[-1])
    oof = np.full(train_rows, np.nan, dtype=np.float32)
    oof_support = np.zeros(train_rows, dtype=np.float32)
    oof_consistency = np.zeros(train_rows, dtype=np.float32)
    oof_similarity = np.zeros(train_rows, dtype=np.float32)
    oof_margin = np.zeros(train_count, dtype=np.float32)
    test_probabilities, test_support, test_consistency, test_similarity, test_margin = [], [], [], [], []
    folds = bundle.arrays["train_folds"]
    labels = bundle.arrays["train_labels"]
    for fold in range(5):
        archive = np.load(semantic_dir / f"semantic_seed_{seed}_fold_{fold}.npz")
        train_embeddings = np.asarray(archive["train_embeddings"], dtype=np.float32)
        test_embeddings = np.asarray(archive["test_embeddings"], dtype=np.float32)
        references = np.flatnonzero(folds != fold)
        for order_index in np.flatnonzero(folds == fold):
            values = transfer_one(
                int(order_index), references, bundle.records["train"], bundle.records["train"],
                train_embeddings, train_embeddings, bundle.ptr("train"), bundle.ptr("train"),
                labels, config,
            )
            start, stop = map(int, bundle.ptr("train")[order_index : order_index + 2])
            oof[start:stop], oof_support[start:stop], oof_consistency[start:stop], oof_similarity[start:stop], oof_margin[order_index] = values
        fold_test = [np.full(test_rows, np.nan, dtype=np.float32), np.zeros(test_rows), np.zeros(test_rows), np.zeros(test_rows)]
        fold_margin = np.zeros(test_count, dtype=np.float32)
        for order_index in range(test_count):
            values = transfer_one(
                order_index, references, bundle.records["test"], bundle.records["train"],
                test_embeddings, train_embeddings, bundle.ptr("test"), bundle.ptr("train"), labels, config,
            )
            start, stop = map(int, bundle.ptr("test")[order_index : order_index + 2])
            for target, value in zip(fold_test, values[:4]):
                target[start:stop] = value
            fold_margin[order_index] = values[4]
        test_probabilities.append(fold_test[0])
        test_support.append(fold_test[1])
        test_consistency.append(fold_test[2])
        test_similarity.append(fold_test[3])
        test_margin.append(fold_margin)
    return {
        "oof_probability": oof,
        "oof_support": oof_support,
        "oof_consistency": oof_consistency,
        "oof_similarity": oof_similarity,
        "oof_margin": oof_margin,
        "test_probability": np.nanmean(np.stack(test_probabilities), axis=0),
        "test_support": np.mean(test_support, axis=0),
        "test_consistency": np.mean(test_consistency, axis=0),
        "test_similarity": np.mean(test_similarity, axis=0),
        "test_margin": np.mean(test_margin, axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    bundle = Bundle.load(args.data_root)
    config = read_json(args.config)["retrieval"]
    result = run_seed(bundle, args.semantic_dir, args.seed, config)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"retrieval_seed_{args.seed}.npz", **result)
    write_json(
        args.output / f"retrieval_seed_{args.seed}.json",
        {
            "seed": args.seed,
            "oof_coverage": float(np.isfinite(result["oof_probability"]).mean()),
            "test_coverage": float(np.isfinite(result["test_probability"]).mean()),
            "requirements": config,
        },
    )


if __name__ == "__main__":
    main()
