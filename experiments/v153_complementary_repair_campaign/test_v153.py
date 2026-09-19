"""Synthetic outcomes never enter any real campaign ledger."""
import copy
import itertools
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bridge import c, action_key
from campaign import select_catalog
from policy import Policy, Feasibility, convolve


def action(i, kind='add', probs=None):
    probs = probs or {'add': {'0': .3, '1': .7}, 'delete': {'-1': .3, '0': .7},
                      'swap': {'-1': .2, '0': .3, '1': .5}}[kind]
    return {'candidate_id': i, 'order_id': f'o{i:03}', 'kind': kind,
            'add_rid': f'a{i}' if kind != 'delete' else None,
            'remove_rid': f'r{i}' if kind != 'add' else None,
            'delta_p': {'add': 1, 'delete': -1, 'swap': 0}[kind],
            'probabilities': probs, 'model_probabilities': {'a': probs, 'b': probs}}


def loose(observations): return True


def exhaustive_policy(byid, groups, remaining, observations=(), attempted=frozenset(), champion=None):
    """Independent reference: enumerate complete binary labels and all decision branches."""
    champion = c.f1() if champion is None else champion
    ids = sorted(byid)
    assignments = []
    for labels in itertools.product((0, 1), repeat=len(ids)):
        z = dict(zip(ids, labels))
        if any(sum(z[i] for i in leaf_ids) != dt for leaf_ids, dt in observations): continue
        weight = math.prod(byid[i]['probabilities'][str(z[i])] for i in ids)
        assignments.append((z, weight))
    total = sum(w for z, w in assignments)
    leaves = [{'ids': list(i), 'delta_tp': dt} for i, dt in observations]
    now = max(champion, c.best_union(leaves, byid)['f1'])
    if not remaining: return now
    measured = set(i for group, dt in observations for i in group)
    queries = [('group', tuple(g['ids']), None) for g in groups
               if not measured & set(g['ids']) and tuple(g['ids']) not in attempted]
    for parent, count in observations:
        for size in range(1, len(parent)//2+1):
            for sub in {parent[:size], parent[-size:]}:
                if tuple(sorted(sub)) not in attempted: queries.append(('split', sub, parent))
    result = now
    for kind, query, parent in queries:
        values = {}
        for z, w in assignments:
            dt = sum(z[i] for i in query); values[dt] = values.get(dt, 0)+w/total
        value = 0
        for dt, p in values.items():
            updated = list(observations)
            if parent:
                count = next(n for g, n in observations if g == parent)
                updated = [(g, n) for g, n in updated if g != parent]
                updated.append((tuple(i for i in parent if i not in query), count-dt))
            updated.append((query, dt))
            value += p*exhaustive_policy(byid, groups, remaining-1, tuple(updated),
                                         attempted | {tuple(sorted(query))}, champion)
        result = max(result, value)
    return result


class ScoreTests(unittest.TestCase):
    def test_public_baseline_and_v150(self):
        self.assertEqual(c.infer_tp('0.927717', 1045), 969)
        self.assertEqual(c.infer_tp('0.911941', 1125), 989)

    def test_invalid_score(self):
        for score in ('NaN', 'inf', '-0.01', '0.927718', '0.9277171'):
            with self.assertRaises(ValueError): c.infer_tp(score, 1045)

    def test_delete_zero_tp_is_positive(self):
        self.assertGreater(c.f1(0, -1), c.f1())
        self.assertLess(c.f1(0, 1), c.f1())

    def test_swap_three_values(self):
        a = action(1, 'swap')
        self.assertEqual(set(c.distribution([a])), {-1, 0, 1})

    def test_target_math(self):
        self.assertGreater(c.f1(13, 0), .94)
        self.assertLess(c.f1(12, 0), .94)


class RankingTests(unittest.TestCase):
    def catalog(self, count=30):
        aa = [action(i) for i in range(count)]
        bb = list(reversed(copy.deepcopy(aa)))
        for j, a in enumerate(aa): a['probabilities'] = {'0': .01+j*.02, '1': .99-j*.02}
        for j, a in enumerate(bb): a['probabilities'] = {'0': .01+j*.02, '1': .99-j*.02}
        return select_catalog({'a': aa, 'b': bb}, [], lambda a: (None, {'min': 0, 'max': 1}))

    def test_full_ranks_and_rrf(self):
        cat = self.catalog()
        for a in cat['candidates']:
            self.assertAlmostEqual(a['rrf'], sum(1/(60+r) for r in a['full_legal_ranks'].values()))

    def test_top16_union(self):
        self.assertEqual(self.catalog()['nominated_unique_actions'], 30)

    def test_nonpositive_mean_not_removed(self):
        a = action(1, probs={'0': .9, '1': .1})
        cat = select_catalog({'a': [a]}, [], lambda a: (None, {'min': 0, 'max': 1}))
        self.assertEqual(len(cat['candidates']), 1)
        self.assertLess(cat['candidates'][0]['expected_f1'], c.f1())

    def test_equation_exclusion_precedes_ranking(self):
        aa = [action(i) for i in range(20)]
        cat = select_catalog({'a': aa}, [], lambda a: ('false', None) if a['candidate_id'] < 5 else (None, {'min': 0, 'max': 1}))
        self.assertEqual(len(cat['candidates']), 15)
        self.assertTrue(all(int(a['order_id'][1:]) >= 5 for a in cat['candidates']))

    def test_order_conflict_and_group_bound(self):
        aa = [action(i) for i in range(20)]; aa[1]['order_id'] = aa[0]['order_id']
        cat = select_catalog({'a': aa}, aa, lambda a: (None, {'min': 0, 'max': 1}))
        self.assertEqual(len({a['order_id'] for a in cat['candidates']}), len(cat['candidates']))
        self.assertTrue(all(len(g['ids']) <= 8 for g in cat['groups']))

    def test_missing_model_coverage_rejected(self):
        with self.assertRaises(ValueError):
            select_catalog({'a': [action(1)], 'b': []}, [], lambda a: (None, {}))


class PolicyTests(unittest.TestCase):
    def setup_policy(self):
        aa = {i: action(i) for i in (1, 2, 3, 4)}
        groups = [{'group_id': 'G1', 'kind': 'add', 'ids': [1, 2]},
                  {'group_id': 'G2', 'kind': 'add', 'ids': [3, 4]}]
        return aa, groups, Policy(aa, groups, ['a', 'b'], loose)

    def test_equal_initial_weights(self):
        aa, groups, p = self.setup_policy()
        self.assertEqual(p.weights([]), {'a': .5, 'b': .5})

    def test_feedback_updates_model_weights(self):
        a = action(1); a['model_probabilities'] = {'a': {'0': .1, '1': .9}, 'b': {'0': .9, '1': .1}}
        p = Policy({1: a}, [], ['a', 'b'], loose)
        self.assertAlmostEqual(p.weights([{'ids': [1], 'delta_tp': 1}])['a'], .9)

    def test_two_step_matches_exhaustive_tree(self):
        aa, groups, p = self.setup_policy()
        exact = exhaustive_policy(aa, groups, 2)
        chosen = p.recommend([], set(), c.f1(), 2)
        self.assertAlmostEqual(chosen['expected_best_f1'], exact, places=12)

    def test_two_step_after_parent_count_matches_exhaustive(self):
        aa, groups, p = self.setup_policy()
        leaves = [{'ids': [1, 2], 'delta_tp': 1}]
        exact = exhaustive_policy(aa, groups, 2, (((1, 2), 1),), frozenset({(1, 2)}))
        chosen = p.recommend(leaves, {(1, 2)}, c.f1(), 2)
        self.assertAlmostEqual(chosen['expected_best_f1'], exact, places=12)

    def test_split_conservation_and_dependence(self):
        aa, groups, p = self.setup_policy()
        leaf = [{'ids': [1, 2], 'delta_tp': 1}]
        self.assertEqual(p.subset_dist([1, 2], leaf, 'a'), {1: 1.})
        self.assertAlmostEqual(p.subset_dist([1], leaf, 'a')[1], .5)
        q = {'kind': 'split', 'parent_ids': [1, 2], 'parent_delta_tp': 1, 'ids': [1]}
        for b in p.branches(q, leaf):
            self.assertEqual(sum(x['delta_tp'] for x in b['leaves']), 1)

    def test_joint_feasibility_removes_branch(self):
        aa, groups, p = self.setup_policy()
        p.feasible = lambda obs: not (obs[-1][1] == 2)
        self.assertEqual({b['delta_tp'] for b in p.branches(dict(groups[0], kind='group'), [])}, {0, 1})

    def test_incremental_pool_below_target_continues(self):
        aa = {1: action(1)}; groups = [{'group_id': 'G1', 'kind': 'add', 'ids': [1]}]
        p = Policy(aa, groups, ['a', 'b'], loose)
        self.assertLess(c.f1(1, 1), .94)
        self.assertIsNotNone(p.recommend([], set(), c.f1(), 2))

    def test_no_gain_no_query(self):
        a = action(1, probs={'0': 1., '1': 0.})
        p = Policy({1: a}, [{'group_id': 'G1', 'kind': 'add', 'ids': [1]}], ['a', 'b'], loose)
        self.assertIsNone(p.recommend([], set(), c.f1(), 2))
        self.assertIsNone(p.risk([], c.f1(), set()))

    def test_risk_uses_retained_champion_not_mean(self):
        a = action(1, probs={'0': .9, '1': .1})
        p = Policy({1: a}, [], ['a', 'b'], loose)
        risk = p.risk([], c.f1(), set())
        self.assertLess(risk['expected_candidate_f1'], c.f1())
        self.assertGreater(risk['expected_best_f1'], c.f1())

    def test_no_repeated_query(self):
        aa, groups, p = self.setup_policy()
        self.assertEqual(p.options([], {(1, 2), (3, 4)}), [])

    def test_union_includes_safe_delete_not_harmful_swap(self):
        aa = {1: action(1, 'delete'), 2: action(2, 'swap')}
        result = c.best_union([{'ids': [1], 'delta_tp': 0}, {'ids': [2], 'delta_tp': -1}], aa)
        self.assertEqual(result['ids'], [1])


class EquationTests(unittest.TestCase):
    def test_aggregation_matches_original_binary_system(self):
        system = {'ground_truth_positives': 2,
                  'variables': [{'order_id': f'o{i:03}', 'rid': f'a{i}'} for i in range(6)],
                  'equations': [{'indices': [0, 1, 2], 'tp': 1}]}
        aa = {i: action(i) for i in (0, 1)}
        eq = c.Equations(system); compressed = Feasibility(eq, aa)
        self.assertLess(compressed.n, eq.n)
        for x, y in itertools.product((0, 1), repeat=2):
            obs = [([0], x), ([1], y)]
            exact = eq.solve(extra=[(eq.coeff([aa[0]]), x), (eq.coeff([aa[1]]), y)]) is not None
            self.assertEqual(compressed(obs), exact)

    def test_conflicting_equations_fail(self):
        eq = c.Equations({'ground_truth_positives': 1, 'variables': [{'order_id': 'o001', 'rid': 'a1'}],
                          'equations': [{'indices': [0], 'tp': 0}]})
        self.assertFalse(eq.feasible())


class BudgetTests(unittest.TestCase):
    def test_failures_count_daily_and_total(self):
        stamp = datetime.now(c.TZ).isoformat()
        records = [{'submitted_at': stamp, 'status': 'failed'}]*2
        with self.assertRaises(ValueError): c.check_budget(records, stamp)
        with self.assertRaises(ValueError): c.check_budget(records*5, stamp)

    def test_next_day_allowed(self):
        today = datetime.now(c.TZ)
        previous = (today-timedelta(days=1)).isoformat()
        c.check_budget([{'submitted_at': previous}]*2, today.isoformat())

    def test_timestamp_timezone_required(self):
        with self.assertRaises(ValueError): c.check_budget([], '2026-09-07T09:00:00')


if __name__ == '__main__': unittest.main(verbosity=2)
