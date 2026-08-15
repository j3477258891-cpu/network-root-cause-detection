"""Candidate action generation, feature extraction, and exact DP decoding."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


ACTION_NAMES = ("keep", "add1", "add2", "remove1", "remove2", "swap1", "swap2")


@dataclass
class Action:
    order: int
    name: str
    add: tuple[int, ...]
    remove: tuple[int, ...]
    delta: int
    features: np.ndarray
    gain: int | None = None


def rank01(values):
    output = np.zeros(len(values), dtype=np.float32)
    output[np.argsort(-values, kind="stable")] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return output


def action_feature(
    name,
    add,
    remove,
    local_experts,
    base_local,
    count_probabilities,
    retrieval_fields,
    prepared=None,
):
    onehot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    onehot[ACTION_NAMES.index(name)] = 1.0
    if prepared is None:
        ranks = np.column_stack([rank01(local_experts[:, column]) for column in range(local_experts.shape[1])])
        disagreement = np.std(local_experts, axis=1)
        expected_count = float(np.dot(count_probabilities, np.arange(1, 9)))
        order = np.asarray(
            [
                len(base_local) / 32.0,
                float(base_local.sum()) / 8.0,
                expected_count / 8.0,
                float(disagreement.mean()),
                float(disagreement.max()),
                float(local_experts[:, 0].mean()),
                float(local_experts[:, 0].std()),
                float(len(add)),
                float(len(remove)),
            ],
            dtype=np.float32,
        )
    else:
        ranks = prepared["ranks"]
        order = prepared["order"].copy()
        order[-2:] = (len(add), len(remove))

    def summary(indices):
        if not indices:
            return np.zeros(local_experts.shape[1] * 3 + 4, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        values = local_experts[indices]
        return np.concatenate(
            [values.mean(axis=0), values.max(axis=0), ranks[indices].mean(axis=0),
             np.asarray([retrieval_fields[key][indices].mean() for key in ("support", "consistency", "similarity", "probability")], dtype=np.float32)]
        )

    return np.concatenate([onehot, summary(add), summary(remove), order])


def generate_actions(
    order_index,
    start,
    stop,
    base_mask,
    experts,
    count_probabilities,
    retrieval,
    depth=4,
    max_action_depth=2,
    labels=None,
    allowed_add=None,
    allowed_remove=None,
):
    local_base = base_mask[start:stop]
    local_experts = experts[start:stop]
    blend = np.mean(local_experts, axis=1)
    expert_orders_desc = [np.argsort(-local_experts[:, column], kind="stable") for column in range(local_experts.shape[1])]
    expert_orders_asc = [np.argsort(local_experts[:, column], kind="stable") for column in range(local_experts.shape[1])]
    expert_rank_desc = []
    expert_rank_asc = []
    for desc, asc in zip(expert_orders_desc, expert_orders_asc):
        desc_rank = np.empty(len(desc), dtype=np.int32)
        asc_rank = np.empty(len(asc), dtype=np.int32)
        desc_rank[desc] = np.arange(len(desc))
        asc_rank[asc] = np.arange(len(asc))
        expert_rank_desc.append(desc_rank)
        expert_rank_asc.append(asc_rank)
    selected = np.flatnonzero(local_base)
    unselected = np.flatnonzero(~local_base)
    add_pool, remove_pool = set(), set()
    for column in range(local_experts.shape[1]):
        add_pool.update(sorted(unselected, key=lambda value: expert_rank_desc[column][value])[:depth])
        remove_pool.update(sorted(selected, key=lambda value: expert_rank_asc[column][value])[:depth])
    add_pool.update(unselected[np.argsort(-blend[unselected], kind="stable")[:depth]])
    remove_pool.update(selected[np.argsort(blend[selected], kind="stable")[:depth]])
    pool_limit = depth + 2
    if len(add_pool) > pool_limit:
        add_pool = set(
            sorted(
                add_pool,
                key=lambda value: (
                    min(expert_rank_desc[column][value] for column in range(local_experts.shape[1])),
                    -blend[value],
                    value,
                ),
            )[:pool_limit]
        )
    if len(remove_pool) > pool_limit:
        remove_pool = set(
            sorted(
                remove_pool,
                key=lambda value: (
                    min(expert_rank_asc[column][value] for column in range(local_experts.shape[1])),
                    blend[value],
                    value,
                ),
            )[:pool_limit]
        )
    if allowed_add is not None:
        add_pool = {value for value in add_pool if start + value in allowed_add}
    if allowed_remove is not None:
        remove_pool = {value for value in remove_pool if start + value in allowed_remove}
    retrieval_local = {
        key: np.asarray(retrieval[key][start:stop], dtype=np.float32)
        for key in ("support", "consistency", "similarity", "probability")
    }
    disagreement = np.std(local_experts, axis=1)
    prepared = {
        "ranks": np.column_stack([rank01(local_experts[:, column]) for column in range(local_experts.shape[1])]),
        "order": np.asarray(
            [
                len(local_base) / 32.0,
                float(local_base.sum()) / 8.0,
                float(np.dot(count_probabilities[order_index], np.arange(1, 9))) / 8.0,
                float(disagreement.mean()),
                float(disagreement.max()),
                float(local_experts[:, 0].mean()),
                float(local_experts[:, 0].std()),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    }

    specs = [("keep", (), ())]
    for width in range(1, max_action_depth + 1):
        if int(local_base.sum()) + width <= min(8, stop - start):
            specs.extend((f"add{width}", values, ()) for values in itertools.combinations(sorted(add_pool), width))
        if int(local_base.sum()) - width >= 1:
            specs.extend((f"remove{width}", (), values) for values in itertools.combinations(sorted(remove_pool), width))
        specs.extend(
            (f"swap{width}", adds, removes)
            for adds in itertools.combinations(sorted(add_pool), width)
            for removes in itertools.combinations(sorted(remove_pool), width)
        )
    actions = []
    for name, add, remove in specs:
        gain = None
        if labels is not None:
            gain = int(labels[start + np.asarray(add, dtype=int)].sum()) - int(
                labels[start + np.asarray(remove, dtype=int)].sum()
            )
        actions.append(
            Action(
                order=order_index,
                name=name,
                add=tuple(start + value for value in add),
                remove=tuple(start + value for value in remove),
                delta=len(add) - len(remove),
                features=action_feature(
                    name, add, remove, local_experts, local_base,
                    count_probabilities[order_index], retrieval_local,
                    prepared,
                ),
                gain=gain,
            )
        )
    return actions


def exact_dp(action_groups, utilities, required_delta=0):
    states = {0: 0.0}
    history = []
    total_groups = len(action_groups)
    for group_index, (actions, scores) in enumerate(zip(action_groups, utilities)):
        best_by_delta = {}
        for action, score in zip(actions, scores):
            score = float(score)
            if action.delta not in best_by_delta or score > best_by_delta[action.delta][0]:
                best_by_delta[action.delta] = (score, action)
        updated = {}
        back = {}
        remaining = total_groups - group_index - 1
        for previous_delta, previous_utility in states.items():
            for action_delta, (score, action) in best_by_delta.items():
                delta = previous_delta + action_delta
                if abs(required_delta - delta) > 2 * remaining:
                    continue
                utility = previous_utility + score
                if delta not in updated or utility > updated[delta]:
                    updated[delta] = utility
                    back[delta] = (previous_delta, action)
        states = updated
        history.append(back)
    if required_delta not in states:
        raise ValueError(f"no DP solution for delta={required_delta}")
    chosen = []
    delta = required_delta
    for back in reversed(history):
        previous_delta, action = back[delta]
        chosen.append(action)
        delta = previous_delta
    chosen.reverse()
    return states[required_delta], chosen


def apply_actions(base_mask, actions):
    output = np.asarray(base_mask, dtype=bool).copy()
    for action in actions:
        output[list(action.remove)] = False
        output[list(action.add)] = True
    return output
