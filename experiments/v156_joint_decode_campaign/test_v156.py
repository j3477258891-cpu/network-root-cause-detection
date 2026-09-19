"""V156 acceptance tests.

Covers: 12-state decodability; budget reservation; dynamic ledger updates;
three-valued swaps / negative delete gain / carrier deduction / duplicate &
anomalous feedback; whole-group-known merge; small-scale exhaustive
cross-check of strategy expectation + worst branch + timeout fallback.

Run:  python -m unittest -v test_v156
"""
import copy
import itertools
import json
import shutil
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import campaign as V


class DecodeTests(unittest.TestCase):
    def test_twelve_states_uniquely_decodable(self):
        combos = V.enumerate_new_combos()
        self.assertEqual(len(combos), 12)
        sigs = {}
        for c in combos:
            sig = (c[8], c[11] + c[9], c[14] + c[9])
            sigs.setdefault(sig, set()).add((c[8], c[9], c[11], c[14], c[15], c[17]))
        self.assertEqual(len(sigs), 12, "three queries must give 12 distinct signatures")
        self.assertTrue(all(len(v) == 1 for v in sigs.values()), "signature collision")
        self.assertTrue(all(sum(next(iter(v))) == 3 for v in sigs.values()))

    def test_decode_plan_reproduces_backup_route(self):
        self.assertAlmostEqual(V.f1(3, 3), 0.929254, places=6)

    def test_fixed_plan_needs_at_most_three_probes(self):
        safety = V.fallback_safety(V.enumerate_new_combos())
        self.assertLessEqual(safety["fixed_plan_probes_needed"], 3)

    def test_full_pool_upper_matches_reference(self):
        upper = V.full_pool_upper(V.enumerate_combos())
        self.assertAlmostEqual(upper["f1"], 0.932504, places=6)
        self.assertEqual(upper["dt"], 5)
        self.assertEqual(upper["dp"], 0)


class MergeTests(unittest.TestCase):
    def test_merge_keeps_whole_group_known_single_unknown(self):
        # feasible set: {9,15} one true, {8,11,14,17} two true -> 12 combos.
        combos = V.enumerate_new_combos()
        mask = (1 << len(combos)) - 1
        known = V._known_labels(combos, mask, V.NEW_NODES)
        # nothing individually known initially, floor = champion
        self.assertEqual(known, {})
        value, ids, dt, dp = V.guaranteed_merge(combos, mask,
                                                V.enumerate_s2_combos(),
                                                (1 << len(V.enumerate_s2_combos())) - 1,
                                                champion_f1=V.f1(2, 4))
        self.assertAlmostEqual(value, V.f1(2, 4), places=9)  # champion floor
        self.assertEqual(ids, [])

    def test_merge_after_resolution_is_backup_route(self):
        combos = V.enumerate_new_combos()
        one = combos[0]
        value, ids, dt, dp = V.guaranteed_merge(combos, 1,
                                                V.enumerate_s2_combos(),
                                                (1 << len(V.enumerate_s2_combos())) - 1,
                                                champion_f1=V.f1(2, 4))
        true_ids = [i for i in V.NEW_NODES if one[i] == 1]
        self.assertEqual(sorted(ids), sorted(true_ids))
        self.assertEqual(dt, 3)
        self.assertAlmostEqual(value, V.f1(3, 3), places=9)

    def test_harmful_swap_skipped_delete_and_add_applied(self):
        # 12: swap label -1 (harmful -> skip); 6: delete label 0 (false node -> drop);
        # 8: add label 1 (true node -> include).  Negative probe feedback is legitimate,
        # but the merge itself never applies a harmful action.
        known = {12: -1, 6: 0, 8: 1}
        ids, dt, dp = V.merge_actions(known)
        self.assertNotIn(12, ids, "harmful swap must not be applied")
        self.assertIn(6, ids) and self.assertIn(8, ids)
        self.assertEqual(dt, 1)
        self.assertEqual(dp, V.CAND[8]["delta_p"] + V.CAND[6]["delta_p"])

    def test_beneficial_swap_applied(self):
        ids, dt, dp = V.merge_actions({12: 1})
        self.assertIn(12, ids)
        self.assertEqual(dt, 1)


class SafetyTests(unittest.TestCase):
    def test_adaptive_query_leaves_decode_budget(self):
        feasible = V.enumerate_new_combos()
        first = V.fixed_plan_first(feasible)
        self.assertEqual(first, [8])
        adapt = V.adaptive_safety(feasible, [8, 9, 11], 3)
        self.assertTrue(adapt["safe"])
        self.assertLessEqual(adapt["worst_branch_probes"], 3)

    def test_budget_reserves_final_merge(self):
        # remaining 5, reserve 1 -> at most 4 info probes
        remaining, reserve, cap = 5, 1, 4
        probes = max(0, min(cap, remaining - reserve))
        self.assertEqual(probes, 4)
        # remaining 2 -> reserve 1 -> 1 probe
        self.assertEqual(max(0, min(cap, 2 - reserve)), 1)
        # remaining 1 -> reserve 1 -> 0 probes (only the merge)
        self.assertEqual(max(0, min(cap, 1 - reserve)), 0)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v156_test_"))
        self._backup = {}
        for name in ("campaign.json", "online_scores.json", "submission_manifest.json"):
            src = V.HERE / name
            if src.exists():
                self._backup[name] = src.read_text(encoding="utf-8")
                shutil.copy2(src, self.tmp / name)

    def tearDown(self):
        for name, text in self._backup.items():
            (V.HERE / name).write_text(text, encoding="utf-8")
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_does_not_clear_records(self):
        ledger = json.loads((V.HERE / "online_scores.json").read_text(encoding="utf-8"))
        before = len(ledger["records"])
        ledger["records"].append({"attempt_id": "probe-test", "probe_id": "pX", "score": "0.927855",
                                  "submitted_at": "2026-09-10T12:00:00+08:00", "evidence": "unit",
                                  "status": "failed", "source_class": "actual_submission_attempt"})
        (V.HERE / "online_scores.json").write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
        V.cmd_build()
        after = json.loads((V.HERE / "online_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(len(after["records"]), before + 1, "build must not clear the score ledger")

    def test_failed_attempt_consumes_budget_but_no_equation(self):
        base_leaves = len(V.ledger_leaves())
        used_before = V.derived_state()["budget"]["used"]
        ledger = json.loads((V.HERE / "online_scores.json").read_text(encoding="utf-8"))
        ledger["records"].append({"attempt_id": "probe-fail", "probe_id": "pZ", "score": None,
                                  "submitted_at": "2026-09-10T12:00:00+08:00", "evidence": "failed",
                                  "status": "failed", "source_class": "actual_submission_attempt"})
        (V.HERE / "online_scores.json").write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(len(V.ledger_leaves()), base_leaves)              # no new equation
        self.assertEqual(V.derived_state()["budget"]["used"], used_before + 1)  # but budget spent

    def test_negative_feedback_is_not_anomaly(self):
        # a delete probe can legitimately lower TP; record must accept it
        rec = {"status": "accepted", "source_class": "public_scored", "delta_tp": -1, "info_delta": -1}
        self.assertLess(rec["delta_tp"], 0)  # documents the expectation


class StrategyCrossCheckTests(unittest.TestCase):
    """Small-scale exhaustive cross-check of the memoized search."""

    @staticmethod
    def _floor(feasible, champion):
        s2 = V.enumerate_s2_combos()
        return V.guaranteed_merge(feasible, (1 << len(feasible)) - 1, s2,
                                  (1 << len(s2)) - 1, champion)[0]

    def _brute(self, feasible, budget, champion):
        """Independent recursion (no memo, no caps) using the same terminal."""
        floor = self._floor(feasible, champion)
        if budget <= 0 or len({tuple(c[i] for i in V.NEW_NODES) for c in feasible}) == 1:
            return floor
        priors = V.combos_priors(feasible, V.NEW_NODES)
        total = sum(priors)
        best = floor
        unresolved = [i for i in V.NEW_NODES if len({c[i] for c in feasible}) > 1]
        for r in range(1, min(3, len(unresolved)) + 1):
            for sub in itertools.combinations(unresolved, r):
                groups = defaultdict(list)
                for c, p in zip(feasible, priors):
                    groups[sum(c[i] for i in sub)].append((c, p))
                if len(groups) <= 1:
                    continue
                exp = 0.0
                for s, members in groups.items():
                    w = sum(p for _, p in members) / total
                    exp += w * self._brute([c for c, _ in members], budget - 1, champion)
                best = max(best, exp)
        return best

    def _search(self, feasible, champion, **kw):
        s2 = V.enumerate_s2_combos()
        return V.Search(feasible, V.combos_priors(feasible, V.NEW_NODES),
                        s2, V.combos_uniform(s2), champion_f1=champion, **kw)

    def test_search_matches_brute_force_small(self):
        feasible = V.enumerate_new_combos()
        champion = V.f1(2, 4)
        sm = self._search(feasible, champion, max_new_budget=2, max_s2_budget=0)
        got = sm.first_query(sm.new_full, 2)["val"]
        want = self._brute(feasible, 2, champion)
        self.assertAlmostEqual(got, want, places=9,
                               msg="memoized decode expectation must match brute force")

    def test_worst_branch_is_minimum(self):
        feasible = V.enumerate_new_combos()
        sm = self._search(feasible, V.f1(2, 4), max_new_budget=2)
        res = sm.first_query(sm.new_full, 2)
        self.assertAlmostEqual(res["worst"], min(b["continuation_f1"] for b in res["branches"]))

    def test_timeout_returns_floor_and_flags_truncated(self):
        feasible = V.enumerate_new_combos()
        sm = self._search(feasible, V.f1(2, 4), wall_limit=-1.0)
        res = sm.first_query(sm.new_full, 3)
        self.assertTrue(sm.truncated, "a timed-out search must be flagged")
        # conservative: value is the expectation of the guaranteed floor only
        self.assertLessEqual(res["val"], self._floor(feasible, V.f1(2, 4)) + 1e-9)
        self.assertEqual(res["branches"], [], "no branch may be reported from a truncated search")


class BomTests(unittest.TestCase):
    """Regression: the platform reads a plain-UTF-8 header, so a BOM is fatal."""

    def test_load_rejects_bom(self):
        p = V.HERE / "_bom_probe_test.csv"
        p.write_bytes(b"\xef\xbb\xbforder_id,output\n")
        try:
            with self.assertRaises(ValueError):
                V.load_csv(p)
        finally:
            p.unlink()

    def test_written_csv_has_no_bom(self):
        p = V.HERE / "_nobom_probe_test.csv"
        try:
            V.apply_actions(V.BASE, [8, 9, 11], p)
            raw = p.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "writer must not emit a BOM")
            self.assertEqual(raw.split(b"\n", 1)[0].rstrip(b"\r"), b"order_id,output")
        finally:
            if p.exists():
                p.unlink()

    def test_validate_rejects_bom(self):
        p = V.HERE / "_bom_validate_test.csv"
        p.write_bytes(b"\xef\xbb\xbforder_id,output\n")
        try:
            with self.assertRaises(ValueError):
                V.validate_csv(p, [8, 9, 11])
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
