"""Aggregate V17 folds, enforce gates, and emit protected submissions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v17_data import GraphStore


def exact_count_mask(scores, ptr, target_count, max_roots=8):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for index in range(len(ptr) - 1):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        ranked = np.argsort(-scores[start:stop], kind="stable")[:max_roots]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[: target_count - int(selected.sum())]] = True
    return selected


def aggregate_predictions(config, train, test, predictions_dir):
    seed_oof_scores, seed_oof_counts, seed_test_scores, seed_test_counts = [], [], [], []
    for seed in config["seeds"]:
        oof_scores = np.zeros(len(train.x), dtype=np.float32)
        oof_counts = np.zeros((len(train), 8), dtype=np.float32)
        fold_test_scores, fold_test_counts = [], []
        for fold in range(config["folds"]):
            path = predictions_dir / f"seed_{seed}_fold_{fold}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            data = np.load(path)
            oof_scores[data["validation_rows"]] = data["validation_scores"]
            oof_counts[data["validation_order_indices"]] = data["validation_count_probs"]
            fold_test_scores.append(data["test_scores"])
            fold_test_counts.append(data["test_count_probs"])
        seed_oof_scores.append(oof_scores)
        seed_oof_counts.append(oof_counts)
        seed_test_scores.append(np.mean(fold_test_scores, axis=0))
        seed_test_counts.append(np.mean(fold_test_counts, axis=0))
    return {
        "seed_oof_scores": seed_oof_scores,
        "seed_oof_counts": seed_oof_counts,
        "seed_test_scores": seed_test_scores,
        "seed_test_counts": seed_test_counts,
        "oof_scores": np.mean(seed_oof_scores, axis=0),
        "oof_counts": np.mean(seed_oof_counts, axis=0),
        "test_scores": np.mean(seed_test_scores, axis=0),
        "test_counts": np.mean(seed_test_counts, axis=0),
    }


def lock_maps(metadata):
    forced_in, forced_out = defaultdict(set), defaultdict(set)
    for item in metadata["locks"]["forced_in"]:
        forced_in[item["order_id"]].add(item["rid"])
    for item in metadata["locks"]["forced_out"]:
        forced_out[item["order_id"]].add(item["rid"])
    return forced_in, forced_out


def test_constraints(test, metadata):
    forced_in, forced_out = lock_maps(metadata)
    minimum, maximum = [], []
    local_forced_in, local_forced_out = [], []
    for order_index, order in enumerate(metadata["test"]):
        rid_to_local = {node["@rid"]: index for index, node in enumerate(order["nodes"])}
        include = {rid_to_local[rid] for rid in forced_in[order["order_id"]] if rid in rid_to_local}
        exclude = {rid_to_local[rid] for rid in forced_out[order["order_id"]] if rid in rid_to_local}
        local_forced_in.append(include)
        local_forced_out.append(exclude)
        minimum.append(max(1, len(include)))
        maximum.append(min(8, len(order["nodes"]) - len(exclude)))
    return np.asarray(minimum), np.asarray(maximum), local_forced_in, local_forced_out


def order_utility(scores, count_probs, start, stop, minimum, maximum, alpha, count_reference=None, penalty=0.0):
    ranked = np.argsort(-scores[start:stop], kind="stable")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores[start:stop][ranked], -20.0, 20.0)))
    cumulative = np.cumsum(probabilities)
    utilities = {}
    for count in range(int(minimum), int(maximum) + 1):
        value = math.log(max(float(count_probs[count - 1]), 1e-9)) + alpha * float(cumulative[count - 1])
        if count_reference is not None:
            value -= penalty * abs(count - int(count_reference))
        utilities[count] = value
    return utilities


def allocate_counts(scores, count_probs, ptr, total, alpha, minimum=None, maximum=None, reference=None, penalties=None):
    order_count = len(ptr) - 1
    minimum = np.ones(order_count, dtype=np.int16) if minimum is None else np.asarray(minimum)
    maximum = np.asarray(
        [min(8, int(ptr[i + 1] - ptr[i])) for i in range(order_count)], dtype=np.int16
    ) if maximum is None else np.asarray(maximum)
    penalties = np.zeros(order_count, dtype=np.float32) if penalties is None else np.asarray(penalties)
    dp = np.full(total + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices = np.full((order_count, total + 1), -1, dtype=np.int8)
    for order_index in range(order_count):
        start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
        utilities = order_utility(
            scores,
            count_probs[order_index],
            start,
            stop,
            minimum[order_index],
            maximum[order_index],
            alpha,
            None if reference is None else reference[order_index],
            penalties[order_index],
        )
        updated = np.full_like(dp, -np.inf)
        selected = np.full(total + 1, -1, dtype=np.int8)
        for count, utility in utilities.items():
            candidate = dp[: total + 1 - count] + utility
            updated_segment = updated[count:]
            selected_segment = selected[count:]
            better = candidate > updated_segment
            updated_segment[better] = candidate[better]
            selected_segment[better] = count
        dp = updated
        choices[order_index] = selected
    if not np.isfinite(dp[total]):
        raise ValueError(f"cannot allocate {total} predictions")
    counts = np.zeros(order_count, dtype=np.int8)
    cursor = total
    for order_index in range(order_count - 1, -1, -1):
        count = int(choices[order_index, cursor])
        if count < 1:
            raise AssertionError((order_index, cursor, count))
        counts[order_index] = count
        cursor -= count
    if cursor != 0:
        raise AssertionError(cursor)
    return counts


def mask_from_counts(scores, ptr, counts, forced_in=None, forced_out=None):
    mask = np.zeros(len(scores), dtype=bool)
    forced_in = forced_in or [set() for _ in counts]
    forced_out = forced_out or [set() for _ in counts]
    for order_index, count in enumerate(counts):
        start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
        include = set(forced_in[order_index])
        exclude = set(forced_out[order_index])
        if len(include) > int(count):
            raise AssertionError((order_index, count, include))
        ranked = np.argsort(-scores[start:stop], kind="stable")
        selected = list(sorted(include))
        if len(selected) < int(count):
            for local in ranked:
                local = int(local)
                if local in include or local in exclude:
                    continue
                selected.append(local)
                if len(selected) >= int(count):
                    break
        if len(selected) != int(count):
            raise AssertionError((order_index, count, selected))
        mask[start + np.asarray(selected, dtype=np.int64)] = True
    return mask


def fixed_k_mask(scores, ptr, reference_mask, forced_in=None, forced_out=None):
    counts = np.asarray(
        [int(reference_mask[int(ptr[i]) : int(ptr[i + 1])].sum()) for i in range(len(ptr) - 1)]
    )
    return mask_from_counts(scores, ptr, counts, forced_in, forced_out)


def tp(mask, labels):
    return int(np.sum(mask & (labels == 1)))


def tune_alpha(scores, counts, train, base_mask, target_count):
    best = None
    for alpha in (0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4):
        selected_counts = allocate_counts(scores, counts, train.order_ptr, target_count, alpha)
        mask = mask_from_counts(scores, train.order_ptr, selected_counts)
        record = {"alpha": alpha, "tp": tp(mask, train.labels), "mask": mask, "counts": selected_counts}
        if best is None or (record["tp"], -alpha) > (best["tp"], -best["alpha"]):
            best = record
    return best


def fold_deltas(scores, count_probs, train, base_mask, alpha):
    deltas = []
    for fold in range(5):
        orders = np.flatnonzero(train.folds == fold)
        local_ptr = [0]
        local_scores, local_probs, local_labels, local_base = [], [], [], []
        for order_index in orders:
            start, stop = int(train.order_ptr[order_index]), int(train.order_ptr[order_index + 1])
            local_scores.append(scores[start:stop])
            local_labels.append(train.labels[start:stop])
            local_base.append(base_mask[start:stop])
            local_probs.append(count_probs[order_index])
            local_ptr.append(local_ptr[-1] + stop - start)
        local_scores = np.concatenate(local_scores)
        local_labels = np.concatenate(local_labels)
        local_base = np.concatenate(local_base)
        local_ptr = np.asarray(local_ptr, dtype=np.int64)
        target = int(local_base.sum())
        selected_counts = allocate_counts(local_scores, np.asarray(local_probs), local_ptr, target, alpha)
        mask = mask_from_counts(local_scores, local_ptr, selected_counts)
        deltas.append(tp(mask, local_labels) - tp(local_base, local_labels))
    return deltas


def bootstrap_lower(mask, base_mask, train, iterations=10000):
    deltas = []
    for index in range(len(train)):
        start, stop = int(train.order_ptr[index]), int(train.order_ptr[index + 1])
        deltas.append(tp(mask[start:stop], train.labels[start:stop]) - tp(base_mask[start:stop], train.labels[start:stop]))
    deltas = np.asarray(deltas, dtype=np.int16)
    rng = np.random.default_rng(20260803)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def priority_adjusted_scores(scores, test, metadata, champion_mask):
    adjusted = scores.copy()
    excluded = set(metadata["exclusions"]["total"])
    penalties = np.zeros(len(test), dtype=np.float32)
    for index, order in enumerate(metadata["test"]):
        if order["order_id"] not in excluded:
            continue
        start, stop = int(test.order_ptr[index]), int(test.order_ptr[index + 1])
        block = adjusted[start:stop]
        local_champion = champion_mask[start:stop]
        block[local_champion] += 0.25
        block[~local_champion] -= 0.25
        penalties[index] = 0.5
    return adjusted, penalties


def change_summary(mask, champion_mask, test, metadata):
    changes = []
    templates = Counter()
    for index, order in enumerate(metadata["test"]):
        start, stop = int(test.order_ptr[index]), int(test.order_ptr[index + 1])
        before = set(np.flatnonzero(champion_mask[start:stop]).tolist())
        after = set(np.flatnonzero(mask[start:stop]).tolist())
        if before == after:
            continue
        templates[metadata["test_template_sha"][index]] += 1
        changes.append(
            {
                "order_id": order["order_id"],
                "removed": [order["nodes"][i]["@rid"] for i in sorted(before - after)],
                "added": [order["nodes"][i]["@rid"] for i in sorted(after - before)],
                "delta_p": len(after) - len(before),
            }
        )
    max_share = max(templates.values(), default=0) / max(len(changes), 1)
    return changes, max_share


def write_submission(path, mask, test, metadata, champion_sha, variant, report):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for index, order in enumerate(metadata["test"]):
            start, stop = int(test.order_ptr[index]), int(test.order_ptr[index + 1])
            roots = [order["nodes"][int(local)] for local in np.flatnonzero(mask[start:stop])]
            writer.writerow([order["order_id"], json.dumps({"rootcause": roots}, ensure_ascii=False)])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    changes, template_share = change_summary(mask, test.arrays["test_champion_mask"].astype(bool), test, metadata)
    diff = {
        "source_champion_sha256": champion_sha,
        "variant": variant,
        "orders": len(metadata["test"]),
        "predictions": int(mask.sum()),
        "changed_orders": len(changes),
        "max_template_share": template_share,
        "changes": changes,
        "gate_report": report,
        "sha256": digest,
    }
    path.with_suffix(".diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    metadata = train.metadata
    predictions = aggregate_predictions(config, train, test, args.predictions)
    target_train = round(config["target_test_predictions"] / len(test) * len(train))
    base_mask = exact_count_mask(train.v11, train.order_ptr, target_train, config["max_rootcauses"])
    base_tp = tp(base_mask, train.labels)
    tuned = tune_alpha(predictions["oof_scores"], predictions["oof_counts"], train, base_mask, target_train)
    oof_mask = tuned["mask"]
    fold_tp_deltas = fold_deltas(predictions["oof_scores"], predictions["oof_counts"], train, base_mask, tuned["alpha"])
    seed_deltas = []
    for scores, counts in zip(predictions["seed_oof_scores"], predictions["seed_oof_counts"]):
        seed_counts = allocate_counts(scores, counts, train.order_ptr, target_train, tuned["alpha"])
        seed_mask = mask_from_counts(scores, train.order_ptr, seed_counts)
        seed_deltas.append(tp(seed_mask, train.labels) - base_tp)
    bootstrap = bootstrap_lower(oof_mask, base_mask, train)

    champion_mask = test.arrays["test_champion_mask"].astype(bool)
    champion_counts = np.asarray(
        [int(champion_mask[int(test.order_ptr[i]) : int(test.order_ptr[i + 1])].sum()) for i in range(len(test))]
    )
    minimum, maximum, forced_in, forced_out = test_constraints(test, metadata)
    adjusted_scores, exclusion_penalties = priority_adjusted_scores(
        predictions["test_scores"], test, metadata, champion_mask
    )
    joint_counts = allocate_counts(
        adjusted_scores,
        predictions["test_counts"],
        test.order_ptr,
        config["target_test_predictions"],
        tuned["alpha"],
        minimum,
        maximum,
        champion_counts,
        exclusion_penalties,
    )
    joint_mask = mask_from_counts(adjusted_scores, test.order_ptr, joint_counts, forced_in, forced_out)
    rank_mask = fixed_k_mask(adjusted_scores, test.order_ptr, champion_mask, forced_in, forced_out)
    per_seed_rank = [
        fixed_k_mask(
            priority_adjusted_scores(scores, test, metadata, champion_mask)[0],
            test.order_ptr,
            champion_mask,
            forced_in,
            forced_out,
        )
        for scores in predictions["seed_test_scores"]
    ]
    safe_mask = champion_mask.copy()
    for index in range(len(test)):
        start, stop = int(test.order_ptr[index]), int(test.order_ptr[index + 1])
        if all(np.array_equal(mask[start:stop], rank_mask[start:stop]) for mask in per_seed_rank):
            safe_mask[start:stop] = rank_mask[start:stop]

    joint_changes, joint_template_share = change_summary(joint_mask, champion_mask, test, metadata)
    gates = config["oof_gate"]
    checks = {
        "joint_tp_delta": tp(oof_mask, train.labels) - base_tp,
        "seed_tp_deltas": seed_deltas,
        "fold_tp_deltas": fold_tp_deltas,
        "bootstrap_95_lower": bootstrap,
        "test_changed_orders": len(joint_changes),
        "test_max_template_share": joint_template_share,
    }
    passed = (
        checks["joint_tp_delta"] >= gates["joint_min_tp_delta"]
        and min(seed_deltas) >= gates["seed_min_tp_delta"]
        and min(fold_tp_deltas) >= gates["fold_min_tp_delta"]
        and bootstrap > gates["bootstrap_lower_bound"]
        and gates["test_min_changed_orders"] <= len(joint_changes) <= gates["test_max_changed_orders"]
        and joint_template_share <= gates["max_template_share"]
    )
    report = {
        "passed": passed,
        "alpha": tuned["alpha"],
        "baseline_tp": base_tp,
        "checks": checks,
        "required": gates,
        "champion_sha256": metadata["champion_sha256"],
    }
    (args.output / "gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "v17_aggregated_scores.npz",
        oof_scores=predictions["oof_scores"],
        oof_count_probs=predictions["oof_counts"],
        test_scores=predictions["test_scores"],
        test_count_probs=predictions["test_counts"],
    )
    if passed:
        outputs = {
            "joint_count": joint_mask,
            "rank_only": rank_mask,
            "safe_core": safe_mask,
        }
        for variant, mask in outputs.items():
            if int(mask.sum()) != config["target_test_predictions"]:
                raise AssertionError((variant, int(mask.sum())))
            path = args.output / f"v17_hgt_{variant}_p1059.csv"
            write_submission(path, mask, test, metadata, metadata["champion_sha256"], variant, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
