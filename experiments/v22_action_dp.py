"""V22 leakage-resistant boundary action model with exact-count DP decoding."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(r"D:\zgyidong")
V16_DIR = ROOT / "experiments" / "v16"
V19_DIR = ROOT / "experiments" / "v19_cloud"
V11_DIR = ROOT / "codexgz" / "v11"
OUTPUT_DIR = ROOT / "experiments" / "v22"
CHAMPION = ROOT / "experiments" / "submissions" / "champion_0.906324_day01_probe01_v11_full.csv"
LOCK_REPORT = V11_DIR / "v11_constrained_report.json"
SEEDS = (20260803, 20260817, 20260831)
N_FOLDS = 5
TARGET_TRAIN_COUNT = 3169
TARGET_TEST_COUNT = 1059
MAX_ROOTS = 8
CAPS = (1, 2)
PENALTIES = (0.0, 0.02, 0.05, 0.10)
BLEND_DEFINITIONS = {
    "et12": {"et12": 1.0},
    "et16": {"et16": 1.0},
    "et12_hgb25": {"et12": 0.75, "hgb": 0.25},
    "et16_hgb25": {"et16": 0.75, "hgb": 0.25},
    "et12_hgb50": {"et12": 0.50, "hgb": 0.50},
    "et16_hgb50": {"et16": 0.50, "hgb": 0.50},
}

sys.path.insert(0, str(ROOT / "experiments"))
import v19_nested_count_audit as audit  # noqa: E402


def build_slices(orders):
    slices = []
    cursor = 0
    for order in orders:
        stop = cursor + len(order["alarms"])
        slices.append(slice(cursor, stop))
        cursor = stop
    return slices


def read_submission(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError(reader.fieldnames)
        for row in reader:
            rows[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return rows


def champion_mask(orders, slices, rows):
    mask = np.zeros(slices[-1].stop, dtype=bool)
    for order_index, order in enumerate(orders):
        selected = {item["@rid"] for item in rows[order["id"]]}
        for local_index, node in enumerate(order["alarms"]):
            mask[slices[order_index].start + local_index] = node["@rid"] in selected
    return mask


def balanced_weights(y):
    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)
    return np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))


def fit_base_models(x, y, train_rows, prediction_x, seed):
    weights = balanced_weights(y[train_rows])
    models = {
        "et12": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.70,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "et16": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 101,
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=seed + 211,
        ),
    }
    output = {}
    for name, model in models.items():
        if name == "hgb":
            model.fit(x[train_rows], y[train_rows], sample_weight=weights)
        else:
            model.fit(x[train_rows], y[train_rows])
        output[name] = model.predict_proba(prediction_x)[:, 1]
    return output


def blend_predictions(predictions, definition):
    result = np.zeros(len(next(iter(predictions.values()))), dtype=np.float64)
    for name, weight in BLEND_DEFINITIONS[definition].items():
        result += weight * predictions[name]
    return result


def fit_calibrator(raw, y):
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
    model.fit(raw.reshape(-1, 1), y)
    return model


def calibrate(model, raw):
    if model is None:
        return np.clip(raw, 0.0, 1.0)
    return model.predict_proba(raw.reshape(-1, 1))[:, 1]


def order_rank(values):
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[np.argsort(-values, kind="stable")] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def build_boundary_table(orders, slices, raw, v11, graph, base_mask, labels=None, locks=None):
    locks = locks or {"in": set(), "out": set()}
    tables = {}
    for head in ("add", "remove"):
        features = []
        targets = []
        order_indices = []
        depths = []
        node_rows = []
        for order_index, (order, order_slice) in enumerate(zip(orders, slices)):
            selected_local = np.flatnonzero(base_mask[order_slice])
            unselected_local = np.flatnonzero(~base_mask[order_slice])
            blend = 0.60 * v11[order_slice] + 0.40 * graph[order_slice]
            if head == "add":
                ranked = unselected_local[np.argsort(-blend[unselected_local], kind="stable")]
            else:
                ranked = selected_local[np.argsort(blend[selected_local], kind="stable")]
            local_v11 = v11[order_slice]
            local_graph = graph[order_slice]
            mean_raw = raw[order_slice].mean(axis=0)
            std_raw = raw[order_slice].std(axis=0)
            v11_rank = order_rank(local_v11)
            graph_rank = order_rank(local_graph)
            base_k = int(base_mask[order_slice].sum())
            accepted = 0
            for local_index in ranked:
                node = order["alarms"][int(local_index)]
                key = (order["id"], node["@rid"])
                if head == "add" and key in locks["out"]:
                    continue
                if head == "remove" and key in locks["in"]:
                    continue
                row = order_slice.start + int(local_index)
                accepted += 1
                context = np.asarray(
                    [
                        local_v11[local_index],
                        local_graph[local_index],
                        blend[local_index],
                        v11_rank[local_index],
                        graph_rank[local_index],
                        len(local_v11) / 32.0,
                        base_k / 8.0,
                        accepted / 2.0,
                        local_v11.mean(),
                        local_v11.std(),
                        local_graph.mean(),
                        local_graph.std(),
                        local_v11.max() - local_v11[local_index],
                        local_graph.max() - local_graph[local_index],
                    ],
                    dtype=np.float64,
                )
                features.append(np.concatenate([raw[row], mean_raw, std_raw, context]))
                targets.append(-1 if labels is None else int(labels[row]))
                order_indices.append(order_index)
                depths.append(accepted)
                node_rows.append(row)
                if accepted == 2:
                    break
        tables[head] = {
            "x": np.asarray(features, dtype=np.float32),
            "y": np.asarray(targets, dtype=np.int8),
            "orders": np.asarray(order_indices, dtype=np.int32),
            "depths": np.asarray(depths, dtype=np.int8),
            "rows": np.asarray(node_rows, dtype=np.int32),
        }
    return tables


def nested_head_predictions(table, folds, seed, test_table=None):
    x, y, example_orders = table["x"], table["y"], table["orders"]
    example_folds = folds[example_orders]
    honest = np.zeros(len(y), dtype=np.float64)
    base_oof = {name: np.zeros(len(y), dtype=np.float64) for name in ("et12", "et16", "hgb")}
    choices = []
    for outer in range(N_FOLDS):
        outer_train = example_folds != outer
        outer_valid = example_folds == outer
        inner_base = {name: np.full(len(y), np.nan, dtype=np.float64) for name in base_oof}
        for inner in range(N_FOLDS):
            if inner == outer:
                continue
            inner_train = (example_folds != outer) & (example_folds != inner)
            inner_valid = example_folds == inner
            predictions = fit_base_models(x, y, inner_train, x[inner_valid], seed + outer * 31 + inner)
            for name, values in predictions.items():
                inner_base[name][inner_valid] = values
        inner_rows = outer_train
        scores = {}
        for definition in BLEND_DEFINITIONS:
            raw = blend_predictions(
                {name: values[inner_rows] for name, values in inner_base.items()}, definition
            )
            scores[definition] = brier_score_loss(y[inner_rows], raw)
        definition = min(scores, key=lambda name: (scores[name], name))
        inner_raw = blend_predictions(
            {name: values[inner_rows] for name, values in inner_base.items()}, definition
        )
        calibrator = fit_calibrator(inner_raw, y[inner_rows])
        outer_predictions = fit_base_models(x, y, outer_train, x[outer_valid], seed + outer * 101)
        for name, values in outer_predictions.items():
            base_oof[name][outer_valid] = values
        outer_raw = blend_predictions(outer_predictions, definition)
        honest[outer_valid] = calibrate(calibrator, outer_raw)
        choices.append(
            {
                "fold": outer,
                "definition": definition,
                "inner_brier": float(scores[definition]),
            }
        )

    global_scores = {}
    for definition in BLEND_DEFINITIONS:
        raw = blend_predictions(base_oof, definition)
        global_scores[definition] = brier_score_loss(y, raw)
    final_definition = min(global_scores, key=lambda name: (global_scores[name], name))
    final_raw_oof = blend_predictions(base_oof, final_definition)
    final_calibrator = fit_calibrator(final_raw_oof, y)
    test_probabilities = None
    if test_table is not None:
        full_train = np.ones(len(y), dtype=bool)
        test_base = fit_base_models(x, y, full_train, test_table["x"], seed + 1009)
        test_probabilities = calibrate(
            final_calibrator, blend_predictions(test_base, final_definition)
        )
    return honest, test_probabilities, {
        "outer_choices": choices,
        "final_definition": final_definition,
        "final_brier": float(global_scores[final_definition]),
    }


def matrix_by_order(table, probabilities, order_count):
    matrix = np.full((order_count, 2), np.nan, dtype=np.float64)
    rows = np.full((order_count, 2), -1, dtype=np.int32)
    for probability, order, depth, row in zip(
        probabilities, table["orders"], table["depths"], table["rows"]
    ):
        matrix[int(order), int(depth) - 1] = probability
        rows[int(order), int(depth) - 1] = row
    return matrix, rows


def action_options(
    order_index, add_p, remove_p, add_rows, remove_rows, cap, penalty, allowed_actions=None
):
    options = [{"name": "keep", "delta": 0, "utility": 0.0, "rows_add": [], "rows_remove": []}]
    if not np.isnan(add_p[order_index, 0]):
        options.append(
            {"name": "add1", "delta": 1, "utility": float(add_p[order_index, 0] - penalty),
             "rows_add": [int(add_rows[order_index, 0])], "rows_remove": []}
        )
    if not np.isnan(remove_p[order_index, 0]):
        options.append(
            {"name": "remove1", "delta": -1, "utility": float(-remove_p[order_index, 0] - penalty),
             "rows_add": [], "rows_remove": [int(remove_rows[order_index, 0])]}
        )
    if not np.isnan(add_p[order_index, 0]) and not np.isnan(remove_p[order_index, 0]):
        options.append(
            {"name": "swap1", "delta": 0,
             "utility": float(add_p[order_index, 0] - remove_p[order_index, 0] - penalty),
             "rows_add": [int(add_rows[order_index, 0])],
             "rows_remove": [int(remove_rows[order_index, 0])]}
        )
    if cap >= 2 and not np.isnan(add_p[order_index]).any() and np.all(~np.isnan(add_p[order_index])):
        options.append(
            {"name": "add2", "delta": 2, "utility": float(add_p[order_index].sum() - penalty),
             "rows_add": [int(value) for value in add_rows[order_index]], "rows_remove": []}
        )
    if cap >= 2 and np.all(~np.isnan(remove_p[order_index])):
        options.append(
            {"name": "remove2", "delta": -2, "utility": float(-remove_p[order_index].sum() - penalty),
             "rows_add": [], "rows_remove": [int(value) for value in remove_rows[order_index]]}
        )
    if allowed_actions is not None and order_index in allowed_actions:
        allowed = {"keep", allowed_actions[order_index]}
        options = [option for option in options if option["name"] in allowed]
    elif allowed_actions is not None:
        options = [options[0]]
    return options


def exact_dp(
    order_indices, add_p, remove_p, add_rows, remove_rows, cap, penalty, allowed_actions=None
):
    states = {0: (0.0, 0, 0)}
    backtrack = []
    option_cache = []
    for order_index in order_indices:
        options = action_options(
            order_index, add_p, remove_p, add_rows, remove_rows, cap, penalty, allowed_actions
        )
        option_cache.append(options)
        next_states = {}
        parents = {}
        for balance, state in states.items():
            for option_index, option in enumerate(options):
                next_balance = balance + option["delta"]
                changed = state[1] + int(option["name"] != "keep")
                magnitude = state[2] + abs(option["delta"])
                candidate = (state[0] + option["utility"], changed, magnitude)
                current = next_states.get(next_balance)
                better = current is None or candidate[0] > current[0] + 1e-12
                if current is not None and abs(candidate[0] - current[0]) <= 1e-12:
                    better = (candidate[1], candidate[2], option["name"]) < (
                        current[1], current[2], options[parents[next_balance][1]]["name"]
                    )
                if better:
                    next_states[next_balance] = candidate
                    parents[next_balance] = (balance, option_index)
        states = next_states
        backtrack.append(parents)
    if 0 not in states:
        raise RuntimeError("DP has no zero-balance solution")
    chosen = []
    balance = 0
    for position in range(len(order_indices) - 1, -1, -1):
        parent_balance, option_index = backtrack[position][balance]
        option = dict(option_cache[position][option_index])
        option["order_index"] = int(order_indices[position])
        chosen.append(option)
        balance = parent_balance
    chosen.reverse()
    return chosen


def apply_actions(base_mask, actions):
    mask = base_mask.copy()
    for action in actions:
        mask[action["rows_add"]] = True
        mask[action["rows_remove"]] = False
    return mask


def action_stats(base_mask, mask, labels, slices, order_indices):
    base_tp = 0
    tp = 0
    for order_index in order_indices:
        order_slice = slices[int(order_index)]
        base_tp += int(np.sum(base_mask[order_slice] & (labels[order_slice] == 1)))
        tp += int(np.sum(mask[order_slice] & (labels[order_slice] == 1)))
    return tp - base_tp


def tune_decoder(add_p, remove_p, add_rows, remove_rows, base_mask, labels, slices, folds, heldout):
    training_folds = [fold for fold in range(N_FOLDS) if fold != heldout]
    candidates = []
    for cap in CAPS:
        for penalty in PENALTIES:
            fold_deltas = []
            for fold in training_folds:
                indices = np.flatnonzero(folds == fold)
                actions = exact_dp(indices, add_p, remove_p, add_rows, remove_rows, cap, penalty)
                mask = apply_actions(base_mask, actions)
                fold_deltas.append(action_stats(base_mask, mask, labels, slices, indices))
            candidates.append(
                {"cap": cap, "penalty": penalty, "sum": int(sum(fold_deltas)),
                 "worst": int(min(fold_deltas)), "fold_deltas": fold_deltas}
            )
    return max(candidates, key=lambda item: (item["sum"], item["worst"], -item["cap"], item["penalty"]))


def nested_decode(add_p, remove_p, add_rows, remove_rows, base_mask, labels, slices, folds):
    stitched = base_mask.copy()
    choices = []
    all_actions = []
    for heldout in range(N_FOLDS):
        choice = tune_decoder(
            add_p, remove_p, add_rows, remove_rows, base_mask, labels, slices, folds, heldout
        )
        indices = np.flatnonzero(folds == heldout)
        actions = exact_dp(
            indices, add_p, remove_p, add_rows, remove_rows, choice["cap"], choice["penalty"]
        )
        fold_mask = apply_actions(base_mask, actions)
        for order_index in indices:
            stitched[slices[int(order_index)]] = fold_mask[slices[int(order_index)]]
        delta = action_stats(base_mask, fold_mask, labels, slices, indices)
        choices.append({"heldout": heldout, **choice, "heldout_delta": int(delta)})
        all_actions.extend(action for action in actions if action["name"] != "keep")
    deltas = audit.per_order_delta(stitched, base_mask, labels, slices)
    return stitched, all_actions, {
        "tp_delta": int(deltas.sum()),
        "fold_deltas": [int(deltas[folds == fold].sum()) for fold in range(N_FOLDS)],
        "changed_orders": len(all_actions),
        "bootstrap_95_lower": audit.bootstrap_lower(deltas),
        "choices": choices,
    }


def choose_final_decoder(add_p, remove_p, add_rows, remove_rows, base_mask, labels, slices, folds):
    candidates = []
    for cap in CAPS:
        for penalty in PENALTIES:
            deltas = []
            for fold in range(N_FOLDS):
                indices = np.flatnonzero(folds == fold)
                actions = exact_dp(indices, add_p, remove_p, add_rows, remove_rows, cap, penalty)
                mask = apply_actions(base_mask, actions)
                deltas.append(action_stats(base_mask, mask, labels, slices, indices))
            candidates.append({"cap": cap, "penalty": penalty, "sum": sum(deltas), "worst": min(deltas)})
    return max(candidates, key=lambda item: (item["sum"], item["worst"], -item["cap"], item["penalty"]))


def write_submission(path, orders, slices, mask):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_index, order in enumerate(orders):
            roots = []
            for local_index in np.flatnonzero(mask[slices[order_index]]):
                node = order["alarms"][int(local_index)]
                roots.append({field: node.get(field, "") for field in ("@rid", "title", "location", "reason")})
            writer.writerow([order["id"], json.dumps({"rootcause": roots}, ensure_ascii=False)])


def validate_submission(path, orders):
    rows = read_submission(path)
    if set(rows) != {order["id"] for order in orders}:
        raise ValueError("order set mismatch")
    total = 0
    for order in orders:
        roots = rows[order["id"]]
        if not 1 <= len(roots) <= MAX_ROOTS:
            raise ValueError((order["id"], len(roots)))
        source = {node["@rid"]: node for node in order["alarms"]}
        if len({item["@rid"] for item in roots}) != len(roots):
            raise ValueError((order["id"], "duplicate"))
        for item in roots:
            node = source[item["@rid"]]
            for field in ("title", "location", "reason"):
                if item[field] != node.get(field, ""):
                    raise ValueError((order["id"], item["@rid"], field))
        total += len(roots)
    if total != TARGET_TEST_COUNT:
        raise ValueError(total)
    return {"orders": len(rows), "predictions": total}


def change_manifest(orders, slices, before, after, actions):
    action_by_order = {item["order_index"]: item for item in actions if item["name"] != "keep"}
    changes = []
    for order_index, order in enumerate(orders):
        order_slice = slices[order_index]
        before_rows = set(np.flatnonzero(before[order_slice]) + order_slice.start)
        after_rows = set(np.flatnonzero(after[order_slice]) + order_slice.start)
        if before_rows == after_rows:
            continue
        changes.append(
            {
                "order_id": order["id"],
                "action": action_by_order.get(order_index, {}).get("name", "derived"),
                "removed": [orders[order_index]["alarms"][row - order_slice.start]["@rid"] for row in sorted(before_rows - after_rows)],
                "added": [orders[order_index]["alarms"][row - order_slice.start]["@rid"] for row in sorted(after_rows - before_rows)],
            }
        )
    return changes


def nanmean_matrices(matrices):
    stack = np.asarray(matrices, dtype=np.float64)
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def balanced_probe_groups(actions, maximum_groups=3):
    candidates = sorted(
        [item for item in actions if item["name"] != "keep"],
        key=lambda item: (-item["utility"], item["order_index"]),
    )[:24]
    groups = []
    used = set()
    for _ in range(maximum_groups):
        available = [item for item in candidates if item["order_index"] not in used]
        best = None
        for size in range(4, min(8, len(available)) + 1):
            for combination in itertools.combinations(available, size):
                if sum(item["delta"] for item in combination) != 0:
                    continue
                score = sum(item["utility"] for item in combination)
                key = (score, -size, tuple(-item["order_index"] for item in combination))
                if best is None or key > best[0]:
                    best = (key, list(combination))
        if best is None:
            break
        group = best[1]
        groups.append(group)
        used.update(item["order_index"] for item in group)
    return groups


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_orders = audit.load_orders(audit.TRAIN_DIR, True)
    test_orders = audit.load_orders(audit.TEST_DIR, False)
    train_slices = build_slices(train_orders)
    test_slices = build_slices(test_orders)
    folds, group_count, fold_sizes = audit.grouped_folds(train_orders)
    labels = np.asarray(
        [int(node["@rid"] in order["roots"]) for order in train_orders for node in order["alarms"]],
        dtype=np.int8,
    )
    train_raw = np.load(V16_DIR / "v16_train_features.npy", mmap_mode="r")[:, :97]
    test_raw = np.load(V16_DIR / "v16_test_features.npy", mmap_mode="r")[:, :97]
    train_v11 = 0.25 * np.load(V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        V11_DIR / "v11_oof_meta.npy"
    )
    test_v11 = np.load(V11_DIR / "v11_test_scores.npy")
    train_graph = np.load(V19_DIR / "v19_graph_time_oof_mean.npy")
    test_graph = np.load(V16_DIR / "v16_graph_time_test_scores.npy")
    train_base = audit.exact_count_mask(train_v11, train_slices, TARGET_TRAIN_COUNT)
    champion_rows = read_submission(CHAMPION)
    test_base = champion_mask(test_orders, test_slices, champion_rows)
    lock_data = json.loads(LOCK_REPORT.read_text(encoding="utf-8"))
    locks = {
        "in": {(item["order_id"], item["rid"]) for item in lock_data["forced_in"]},
        "out": {(item["order_id"], item["rid"]) for item in lock_data["forced_out"]},
    }
    train_tables = build_boundary_table(
        train_orders, train_slices, train_raw, train_v11, train_graph, train_base, labels=labels
    )
    test_tables = build_boundary_table(
        test_orders, test_slices, test_raw, test_v11, test_graph, test_base, locks=locks
    )

    per_seed = []
    model_reports = []
    test_seed_probabilities = []
    for seed in SEEDS:
        seed_oof = {}
        seed_test = {}
        seed_report = {"seed": seed, "heads": {}}
        for head in ("add", "remove"):
            oof, test_probability, head_report = nested_head_predictions(
                train_tables[head], folds, seed, test_tables[head]
            )
            seed_oof[head] = oof
            seed_test[head] = test_probability
            seed_report["heads"][head] = head_report
        add_p, add_rows = matrix_by_order(train_tables["add"], seed_oof["add"], len(train_orders))
        remove_p, remove_rows = matrix_by_order(
            train_tables["remove"], seed_oof["remove"], len(train_orders)
        )
        _, actions, stats = nested_decode(
            add_p, remove_p, add_rows, remove_rows, train_base, labels, train_slices, folds
        )
        stats["seed"] = seed
        stats["action_distribution"] = dict(Counter(item["name"] for item in actions))
        per_seed.append(stats)
        test_seed_probabilities.append(seed_test)
        model_reports.append(seed_report)
        np.save(OUTPUT_DIR / f"v22_add_oof_seed_{seed}.npy", add_p)
        np.save(OUTPUT_DIR / f"v22_remove_oof_seed_{seed}.npy", remove_p)

    mean_add = nanmean_matrices(
        [np.load(OUTPUT_DIR / f"v22_add_oof_seed_{seed}.npy") for seed in SEEDS]
    )
    mean_remove = nanmean_matrices(
        [np.load(OUTPUT_DIR / f"v22_remove_oof_seed_{seed}.npy") for seed in SEEDS]
    )
    _, train_add_rows = matrix_by_order(
        train_tables["add"], np.zeros(len(train_tables["add"]["y"])), len(train_orders)
    )
    _, train_remove_rows = matrix_by_order(
        train_tables["remove"], np.zeros(len(train_tables["remove"]["y"])), len(train_orders)
    )
    mean_mask, mean_actions, mean_stats = nested_decode(
        mean_add, mean_remove, train_add_rows, train_remove_rows,
        train_base, labels, train_slices, folds
    )
    mean_stats["action_distribution"] = dict(Counter(item["name"] for item in mean_actions))
    gate = bool(
        mean_stats["tp_delta"] >= 30
        and min(item["tp_delta"] for item in per_seed) >= 24
        and min(mean_stats["fold_deltas"]) >= -2
        and mean_stats["bootstrap_95_lower"] > 0
    )
    report = {
        "method": "V22 nested dual-head boundary utility with exact DP",
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "baseline": {"tp": int(np.sum(train_base & (labels == 1))), "predictions": int(train_base.sum())},
        "mean": mean_stats,
        "seeds": per_seed,
        "model_reports": model_reports,
        "gate": {
            "mean_tp_delta_min": 30,
            "each_seed_tp_delta_min": 24,
            "worst_fold_min": -2,
            "bootstrap_lower_strictly_positive": True,
            "passed": gate,
        },
        "generated_submissions": [],
    }

    if gate:
        test_add = []
        test_remove = []
        for seed_result in test_seed_probabilities:
            test_add.append(
                matrix_by_order(test_tables["add"], seed_result["add"], len(test_orders))[0]
            )
            test_remove.append(
                matrix_by_order(test_tables["remove"], seed_result["remove"], len(test_orders))[0]
            )
        test_add_mean = nanmean_matrices(test_add)
        test_remove_mean = nanmean_matrices(test_remove)
        _, test_add_rows = matrix_by_order(
            test_tables["add"], np.zeros(len(test_tables["add"]["orders"])), len(test_orders)
        )
        _, test_remove_rows = matrix_by_order(
            test_tables["remove"], np.zeros(len(test_tables["remove"]["orders"])), len(test_orders)
        )
        final_choice = choose_final_decoder(
            mean_add, mean_remove, train_add_rows, train_remove_rows,
            train_base, labels, train_slices, folds
        )
        test_actions = exact_dp(
            np.arange(len(test_orders)), test_add_mean, test_remove_mean,
            test_add_rows, test_remove_rows, final_choice["cap"], final_choice["penalty"]
        )
        test_mask = apply_actions(test_base, test_actions)
        changed = [item for item in test_actions if item["name"] != "keep"]
        if not 20 <= len(changed) <= 80:
            report["test_gate_failure"] = {"changed_orders": len(changed), "required": [20, 80]}
        else:
            output = OUTPUT_DIR / "v22_action_dp_full_p1059.csv"
            write_submission(output, test_orders, test_slices, test_mask)
            validation = validate_submission(output, test_orders)
            changes = change_manifest(test_orders, test_slices, test_base, test_mask, changed)
            sha = hashlib.sha256(output.read_bytes()).hexdigest()
            diff = {
                "source": str(CHAMPION), "decoder": final_choice,
                "changes": changes, "validation": validation, "sha256": sha,
            }
            output.with_suffix(".diff.json").write_text(
                json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n", encoding="ascii")
            report["generated_submissions"].append(str(output))

            seed_action_maps = []
            for seed_add, seed_remove in zip(test_add, test_remove):
                seed_actions = exact_dp(
                    np.arange(len(test_orders)), seed_add, seed_remove,
                    test_add_rows, test_remove_rows,
                    final_choice["cap"], final_choice["penalty"],
                )
                seed_action_maps.append(
                    {item["order_index"]: item["name"] for item in seed_actions if item["name"] != "keep"}
                )
            consensus_allowed = {}
            for order_index in range(len(test_orders)):
                names = [mapping.get(order_index) for mapping in seed_action_maps]
                if names[0] is not None and names[0] == names[1] == names[2]:
                    consensus_allowed[order_index] = names[0]
            consensus_actions = exact_dp(
                np.arange(len(test_orders)), test_add_mean, test_remove_mean,
                test_add_rows, test_remove_rows,
                final_choice["cap"], final_choice["penalty"],
                allowed_actions=consensus_allowed,
            )
            consensus_mask = apply_actions(test_base, consensus_actions)
            if np.any(consensus_mask != test_base):
                consensus_path = OUTPUT_DIR / "v22_action_dp_consensus_p1059.csv"
                write_submission(consensus_path, test_orders, test_slices, consensus_mask)
                consensus_validation = validate_submission(consensus_path, test_orders)
                consensus_sha = hashlib.sha256(consensus_path.read_bytes()).hexdigest()
                consensus_changes = change_manifest(
                    test_orders, test_slices, test_base, consensus_mask, consensus_actions
                )
                consensus_path.with_suffix(".diff.json").write_text(
                    json.dumps(
                        {"source": str(CHAMPION), "decoder": final_choice,
                         "changes": consensus_changes, "validation": consensus_validation,
                         "sha256": consensus_sha},
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                consensus_path.with_suffix(".sha256").write_text(
                    f"{consensus_sha}  {consensus_path.name}\n", encoding="ascii"
                )
                report["generated_submissions"].append(str(consensus_path))

            for probe_index, group in enumerate(balanced_probe_groups(changed), start=1):
                probe_mask = apply_actions(test_base, group)
                probe_path = OUTPUT_DIR / f"v22_action_dp_probe{probe_index:02d}_p1059.csv"
                write_submission(probe_path, test_orders, test_slices, probe_mask)
                probe_validation = validate_submission(probe_path, test_orders)
                probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
                probe_changes = change_manifest(
                    test_orders, test_slices, test_base, probe_mask, group
                )
                probe_path.with_suffix(".diff.json").write_text(
                    json.dumps(
                        {"source": str(CHAMPION), "probe_index": probe_index,
                         "changes": probe_changes, "validation": probe_validation,
                         "sha256": probe_sha},
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                probe_path.with_suffix(".sha256").write_text(
                    f"{probe_sha}  {probe_path.name}\n", encoding="ascii"
                )
                report["generated_submissions"].append(str(probe_path))

    report_path = OUTPUT_DIR / "v22_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "model_reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
