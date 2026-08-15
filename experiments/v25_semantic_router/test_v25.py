import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from actions import Action, apply_actions, exact_dp, generate_actions  # noqa: E402
from v25_common import connected_group_folds, exact_count_mask, normalized_location  # noqa: E402


def test_location_removes_ip_uuid_and_types_numbers():
    value = "ManagedElement=12345,Slot=7; IP:10.2.3.4; id=123e4567-e89b-12d3-a456-426614174000"
    normalized = normalized_location(value)
    assert "10.2.3.4" not in normalized
    assert "123e4567" not in normalized
    assert "STATION=<NUM>" in normalized
    assert "SLOT=<NUM>" in normalized


def test_connected_groups_never_cross_folds():
    orders = [
        {"signature": ("a",), "station_ids": ["S1"]},
        {"signature": ("b",), "station_ids": ["S1"]},
        {"signature": ("b",), "station_ids": ["S2"]},
        {"signature": ("c",), "station_ids": []},
    ]
    folds, report = connected_group_folds(orders, 2)
    assert folds[0] == folds[1] == folds[2]
    assert report["component_count"] == 2


def test_exact_count_and_dp_preserve_total():
    scores = np.asarray([0.9, 0.1, 0.8, 0.7, 0.2], dtype=np.float32)
    ptr = np.asarray([0, 2, 5])
    base = exact_count_mask(scores, ptr, 3)
    assert base.sum() == 3
    add = Action(0, "add1", (1,), (), 1, np.zeros(0), gain=1)
    keep0 = Action(0, "keep", (), (), 0, np.zeros(0), gain=0)
    remove = Action(1, "remove1", (), (3,), -1, np.zeros(0), gain=1)
    keep1 = Action(1, "keep", (), (), 0, np.zeros(0), gain=0)
    gain, chosen = exact_dp([[keep0, add], [keep1, remove]], [[0, 1], [0, 1]], 0)
    changed = apply_actions(base, chosen)
    assert gain == 2
    assert changed.sum() == base.sum()


def test_action_generation_respects_count_and_locks():
    base = np.asarray([True, False, False])
    experts = np.asarray([[0.9, 0.8], [0.7, 0.9], [0.2, 0.1]], dtype=np.float32)
    counts = np.tile(np.eye(8, dtype=np.float32)[0], (1, 1))
    retrieval = {
        "probability": np.asarray([0.8, 0.9, 0.1]),
        "support": np.asarray([3, 4, 0]),
        "consistency": np.asarray([1, 1, 0]),
        "similarity": np.asarray([0.9, 0.9, 0]),
    }
    actions = generate_actions(
        0, 0, 3, base, experts, counts, retrieval, depth=3, max_action_depth=2,
        allowed_add={2}, allowed_remove=set(),
    )
    assert all(1 <= int(base.sum()) + action.delta <= 3 for action in actions)
    assert all(1 not in action.add for action in actions)
    assert all(not action.remove for action in actions)

