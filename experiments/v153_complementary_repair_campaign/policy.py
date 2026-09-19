"""Two-step conditional working policy; exact score arithmetic, not target confidence."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

from bridge import c

EPS = 1e-12
WARNING = ('Equal-prior model mixture with independent action outcomes within each model, '
           'conditioned on measured leaf counts and feasible score branches. '
           'Not an exact global label posterior, calibrated target probability, or global ten-step optimum.')


def normalize(weights):
    total = sum(weights.values())
    c.require(total > 0 and math.isfinite(total), 'No supported conditional model branch')
    return {k: v/total for k, v in weights.items()}


def convolve(left, right):
    out = defaultdict(float)
    for x, px in left.items():
        for y, py in right.items():
            out[x+y] += px*py
    return dict(out)


class Feasibility:
    """Aggregate indistinguishable non-action nodes into bounded integer counts.

    Every node that any candidate can change remains an individual binary variable.
    All other nodes with identical columns in EVERY current equation can be grouped
    exactly. This reduces solver overhead without assuming labels or dropping rows.
    """
    def __init__(self, eq, byid):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csc_matrix
        import numpy as np
        self.np, self.milp, self.LC, self.csc = np, milp, LinearConstraint, csc_matrix
        self.eq, self.byid, self.cache = eq, byid, {}
        protected = set().union(*(set(eq.coeff([a])) for a in byid.values())) if byid else set()
        patterns = [[] for _ in range(eq.n)]
        for r, row in enumerate(eq.rows):
            for col, value in row.items():
                if value: patterns[col].append((r, value))
        groups, mapping = {}, {}
        for col, pattern in enumerate(patterns):
            key = (tuple(pattern), col if col in protected else None)
            if key not in groups: groups[key] = len(groups)
            mapping[col] = groups[key]
        capacities = np.zeros(len(groups)); rows = [dict() for _ in eq.rows]
        for col, j in mapping.items(): capacities[j] += 1
        for (pattern, _), j in groups.items():
            for r, value in pattern: rows[r][j] = value
        self.rows, self.mapping, self.n = rows, mapping, len(groups)
        self.bounds = Bounds(np.zeros(self.n), capacities)
        self.base_rhs = eq.rhs

    def __call__(self, observations):
        key = tuple(sorted((tuple(sorted(ids)), int(dt)) for ids, dt in observations))
        if key in self.cache: return self.cache[key]
        rows, rhs = list(self.rows), list(self.base_rhs)
        for ids, dt in key:
            coeff = self.eq.coeff([self.byid[i] for i in ids])
            rows.append({self.mapping[col]: value for col, value in coeff.items()})
            rhs.append(dt)
        rr, cc, vv = [], [], []
        for r, terms in enumerate(rows):
            for col, value in terms.items(): rr.append(r); cc.append(col); vv.append(value)
        mat = self.csc((vv, (rr, cc)), shape=(len(rows), self.n), dtype=float)
        result = self.milp(self.np.zeros(self.n), integrality=self.np.ones(self.n), bounds=self.bounds,
                           constraints=self.LC(mat, rhs, rhs), options={'time_limit': 30, 'mip_rel_gap': 0})
        if result.status == 2:
            value = False
        else:
            c.require(result.status == 0, 'Unresolved integer feasibility is not infeasibility')
            c.require(self.np.max(self.np.abs(mat @ self.np.rint(result.x)-rhs)) < 1e-6, 'Bad integer witness')
            value = True
        self.cache[key] = value
        return value


class Policy:
    def __init__(self, byid, groups, families, feasible):
        self.byid, self.groups, self.families = byid, groups, tuple(sorted(families))
        self.feasible = feasible
        self.dist_cache, self.branch_cache, self.best_cache = {}, {}, {}

    @staticmethod
    def state(leaves):
        return tuple(sorted((tuple(x['ids']), x['delta_tp']) for x in leaves))

    def dist(self, ids, family):
        key = (tuple(sorted(ids)), family)
        if key not in self.dist_cache:
            actions = [{**self.byid[i], 'probabilities': self.byid[i]['model_probabilities'][family]} for i in ids]
            self.dist_cache[key] = c.distribution(actions)
        return self.dist_cache[key]

    def weights(self, leaves):
        logweights = {}
        for family in self.families:
            values = [self.dist(leaf['ids'], family).get(leaf['delta_tp'], 0) for leaf in leaves]
            logweights[family] = sum(math.log(v) for v in values) if all(v > 0 for v in values) else -math.inf
        peak = max(logweights.values())
        c.require(math.isfinite(peak), 'Observed counts unsupported by every model')
        return normalize({f: math.exp(w-peak) for f, w in logweights.items()})

    def best(self, leaves):
        key = self.state(leaves)
        if key not in self.best_cache: self.best_cache[key] = c.best_union(leaves, self.byid)
        return self.best_cache[key]

    def options(self, leaves, attempted):
        measured = set(i for leaf in leaves for i in leaf['ids'])
        options = [dict(g, kind='group', action_kind=g['kind']) for g in self.groups
                   if not set(g['ids']) & measured and tuple(sorted(g['ids'])) not in attempted]
        for leaf in leaves:
            ids = leaf['ids']
            supports = [set().union(*(set(map(int, self.byid[i]['model_probabilities'][f]))
                                      for f in self.families)) for i in ids]
            if leaf['delta_tp'] in (sum(min(s) for s in supports), sum(max(s) for s in supports)):
                continue  # Every action is determined at an extreme total; no information to buy.
            # A degenerate conditional total gives no information from further splitting.
            if all(sum(p > 0 for p in self.dist(ids, f).values()) == 1 for f in self.families): continue
            seen = set()
            for n in range(1, len(ids)//2+1):
                for sub in (ids[:n], ids[-n:]):
                    key = tuple(sorted(sub))
                    if key in seen or key in attempted: continue
                    seen.add(key)
                    options.append({'kind': 'split', 'ids': sub, 'parent_ids': ids,
                                    'parent_delta_tp': leaf['delta_tp']})
        return options if len(leaves) < 8 else []

    def branches(self, query, leaves):
        key = (self.state(leaves), tuple(sorted(query['ids'])), tuple(query.get('parent_ids', [])))
        if key in self.branch_cache: return self.branch_cache[key]
        ids = query['ids']; parent = query.get('parent_ids')
        rest = [i for i in parent if i not in ids] if parent else []
        model_dists = {}
        support = set()
        for family in self.families:
            dist = self.dist(ids, family)
            if parent:
                other = self.dist(rest, family)
                dist = {dt: p*other.get(query['parent_delta_tp']-dt, 0) for dt, p in dist.items()}
            model_dists[family] = dist
            support.update(dt for dt, p in dist.items() if p > 0)
        obs = [(leaf['ids'], leaf['delta_tp']) for leaf in leaves]
        allowed = {dt for dt in sorted(support) if self.feasible(obs + [(ids, dt)])}
        weights = self.weights(leaves)
        conditional = {}
        for family, dist in model_dists.items():
            filtered = {dt: p for dt, p in dist.items() if dt in allowed and p > 0}
            conditional[family] = normalize(filtered) if filtered else {}
        supported_weights = normalize({f: w for f, w in weights.items() if conditional[f]})
        result = []
        for dt in sorted(allowed):
            p = sum(w*conditional[f].get(dt, 0) for f, w in supported_weights.items())
            if not p: continue
            updated = c.apply_observation(copy.deepcopy(leaves), query, dt)
            result.append({'delta_tp': dt, 'conditional_weight': p, 'leaves': updated,
                           'best_f1': self.best(updated)['f1']})
        c.require(result and abs(sum(b['conditional_weight'] for b in result)-1) < 1e-8,
                  'No normalized feasible query branches')
        self.branch_cache[key] = result
        return result

    @staticmethod
    def ranking(q):
        return (q['expected_best_f1'], q['conditional_p_target'], -len(q['ids']),
                tuple(-i for i in sorted(q['ids'])))

    def evaluate(self, query, leaves, attempted, champion, depth):
        expected = ptarget = immediate = 0.
        summaries = []
        for branch in self.branches(query, leaves):
            value = max(champion, branch['best_f1'])
            terminal = {'expected_best_f1': value, 'conditional_p_target': float(value > c.TARGET)}
            continuation = None
            if depth > 1:
                used = attempted | {tuple(sorted(query['ids']))}
                choices = self.options(branch['leaves'], used)
                evaluated = [self.evaluate(q, branch['leaves'], used, champion, 1) for q in choices]
                if evaluated:
                    best_next = max(evaluated, key=self.ranking)
                    if best_next['expected_best_f1'] > value + EPS:
                        terminal = best_next
                        continuation = {k: best_next[k] for k in ('kind', 'ids', 'expected_best_f1')}
            weight = branch['conditional_weight']
            expected += weight*terminal['expected_best_f1']
            ptarget += weight*terminal['conditional_p_target']
            immediate += weight*max(champion, branch['best_f1'])
            summaries.append({'delta_tp': branch['delta_tp'], 'conditional_weight': weight,
                              'known_best_f1_after_feedback': value, 'next_if_this_feedback': continuation})
        return {**query, 'expected_best_f1': expected, 'one_step_expected_best_f1': immediate,
                'conditional_p_target': ptarget, 'lookahead_steps': depth,
                'branches': summaries, 'probability_warning': WARNING}

    def recommend(self, leaves, attempted, champion, remaining_probes):
        base = max(champion, self.best(leaves)['f1'])
        queries = self.options(leaves, attempted)
        scored = [self.evaluate(q, leaves, attempted, champion, min(2, remaining_probes)) for q in queries]
        if not scored: return None
        best = max(scored, key=self.ranking)
        return best if best['expected_best_f1'] > base + EPS else None

    def subset_dist(self, ids, leaves, family):
        """Preserve within-leaf count dependence for risk-set evaluation."""
        selected = set(ids); covered = set(); result = {0: 1.}
        for leaf in leaves:
            chosen = selected & set(leaf['ids'])
            if not chosen: continue
            covered |= chosen
            others = set(leaf['ids'])-chosen
            rest = self.dist(others, family)
            dist = {dt: p*rest.get(leaf['delta_tp']-dt, 0)
                    for dt, p in self.dist(chosen, family).items()}
            result = convolve(result, normalize({dt: p for dt, p in dist.items() if p > 0}))
        return convolve(result, self.dist(selected-covered, family))

    def risk(self, leaves, champion, attempted):
        weights = self.weights(leaves)
        marginals = {}
        obs = [(leaf['ids'], leaf['delta_tp']) for leaf in leaves]
        for i in self.byid:
            marginals[i] = sum(w*sum(dt*p for dt, p in self.subset_dist([i], leaves, f).items())
                               for f, w in weights.items())
        ranked = sorted(self.byid, key=lambda i: (-c.f1(marginals[i], self.byid[i]['delta_p']), i))
        best_known = set(self.best(leaves)['ids'])
        variants = {tuple(sorted(ranked[:n])) for n in range(1, len(ranked)+1)}
        variants |= {tuple(sorted(best_known | {i})) for i in ranked}
        variants |= {tuple(sorted(best_known - {i})) for i in best_known}
        result = None
        for ids in sorted(variants):
            if not ids or ids in attempted: continue
            dp = sum(self.byid[i]['delta_p'] for i in ids)
            mixed = defaultdict(float)
            for family, weight in weights.items():
                dist = self.subset_dist(ids, leaves, family)
                allowed = {dt: p for dt, p in dist.items() if p > 0 and self.feasible(obs+[(ids, dt)])}
                if allowed:
                    for dt, p in normalize(allowed).items(): mixed[dt] += weight*p
            if not mixed: continue
            dist = normalize(dict(mixed))
            value = sum(p*max(champion, c.f1(dt, dp)) for dt, p in dist.items())
            item = {'kind': 'risk', 'ids': list(ids), 'expected_best_f1': value,
                    'conditional_p_target': sum(p for dt, p in dist.items() if c.f1(dt, dp) > c.TARGET),
                    'expected_candidate_f1': sum(p*c.f1(dt, dp) for dt, p in dist.items()),
                    'warning': 'Unconfirmed within-group actions; score may decrease', 'probability_warning': WARNING}
            if value > champion+EPS and (result is None or self.ranking(item) > self.ranking(result)):
                result = item
        return result
