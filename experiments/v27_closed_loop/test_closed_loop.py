import unittest

from closed_loop import (
    TARGET_SCORE,
    accepted_disjoint_actions,
    apply_actions,
    f1_score,
    infer_tp,
    normalize_action,
    required_tp,
    validate_actions,
)


class ScoreMathTests(unittest.TestCase):
    def test_historical_champion_score_decodes_to_953_tp(self):
        tp, reconstructed = infer_tp(0.906324, 1059)
        self.assertEqual(tp, 953)
        self.assertAlmostEqual(reconstructed, 1906 / 2103)

    def test_historical_previous_score_decodes_to_952_tp(self):
        tp, _ = infer_tp(0.905373, 1059)
        self.assertEqual(tp, 952)

    def test_target_requires_975_tp_at_fixed_count(self):
        self.assertEqual(required_tp(1059, TARGET_SCORE), 975)
        self.assertGreaterEqual(f1_score(975, 1059), TARGET_SCORE)
        self.assertLess(f1_score(974, 1059), TARGET_SCORE)

    def test_fifteen_swaps_and_false_removals_cross_target(self):
        self.assertGreaterEqual(f1_score(968, 1044), TARGET_SCORE)


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.base = {"a": ["r1", "r2"], "b": ["r3"]}
        self.topologies = {
            "a": {"r1": {}, "r2": {}, "r4": {}},
            "b": {"r3": {}, "r5": {}},
        }

    def test_legacy_swap_is_normalized(self):
        action = normalize_action(
            {
                "id": "legacy",
                "source": "legacy",
                "remove": {"order_id": "a", "rid": "r1"},
                "add": {"order_id": "a", "rid": "r4"},
            }
        )
        self.assertEqual(action["remove_rids"], ["r1"])
        self.assertEqual(action["add_rids"], ["r4"])

    def test_multi_node_action_applies(self):
        actions = [
            {
                "action_id": "swap",
                "order_id": "a",
                "remove_rids": ["r1", "r2"],
                "add_rids": ["r4"],
                "source": "test",
            }
        ]
        report = validate_actions(actions, self.base, self.topologies)
        self.assertEqual(report["prediction_delta"], -1)
        self.assertEqual(apply_actions(self.base, actions)["a"], ["r4"])

    def test_multiple_actions_on_one_order_are_rejected(self):
        actions = [
            {
                "action_id": "one",
                "order_id": "a",
                "remove_rids": ["r1"],
                "add_rids": [],
            },
            {
                "action_id": "two",
                "order_id": "a",
                "remove_rids": ["r2"],
                "add_rids": [],
            },
        ]
        with self.assertRaisesRegex(ValueError, "multiple actions"):
            validate_actions(actions, self.base, self.topologies)

    def test_accepted_child_replaces_accepted_parent_after_split(self):
        first = {
            "action_id": "first",
            "order_id": "a",
            "remove_rids": [],
            "add_rids": ["r4"],
        }
        second = {
            "action_id": "second",
            "order_id": "b",
            "remove_rids": [],
            "add_rids": ["r5"],
        }
        state = {
            "batches": [
                {
                    "batch_id": "parent",
                    "parent": None,
                    "decision": "accept",
                    "actions": [first, second],
                },
                {
                    "batch_id": "child_a",
                    "parent": "parent",
                    "decision": "accept",
                    "actions": [first],
                },
                {
                    "batch_id": "child_b",
                    "parent": "parent",
                    "decision": "reject",
                    "actions": [second],
                },
            ]
        }
        self.assertEqual(accepted_disjoint_actions(state), [first])


if __name__ == "__main__":
    unittest.main()
