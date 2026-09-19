"""Exact finite-horizon count-query policy under conditional exchangeability.

No probabilities in this module are calibrated probabilities for the real pool.
Historical equations are a separate hard consistency check, not this prior.
"""
from functools import lru_cache
from itertools import product
from math import comb

BASE_TP = 969
DENOMINATOR = 2089
EPS = 1e-12


def f1(n=0, k=0):
    return 2 * (BASE_TP + k) / (DENOMINATOR + n)


def best_union(groups):
    """Return the exact best union of whole, disjoint known groups."""
    best = (0, 0, ())
    for bits in product((0, 1), repeat=len(groups)):
        indices = tuple(i for i, b in enumerate(bits) if b)
        n = sum(groups[i][0] for i in indices)
        k = sum(groups[i][1] for i in indices)
        bn, bk, bi = best
        left = (BASE_TP + k) * (DENOMINATOR + bn)
        right = (BASE_TP + bk) * (DENOMINATOR + n)
        if left > right or (left == right and (n, indices) < (bn, bi)):
            best = n, k, indices
    return best


def reached(groups, threshold=2566):
    # Existence of a union above .94 is an additive threshold test.
    return sum(max(0, 200*t - 94*n) for n, t in groups) > threshold


class Policy:
    def __init__(self, grid=1, threshold=2566):
        self.grid = grid
        self.threshold = threshold
        self.distribution = lru_cache(None)(self._distribution)
        self.best_one = lru_cache(None)(self._best_one)
        self.value = lru_cache(None)(self._value)
        self.secondary = lru_cache(None)(self._secondary)

    def sizes(self, n):
        return range(self.grid, n//2 + 1, self.grid)

    @staticmethod
    def _distribution(n, t, a):
        low, high = max(0, a-(n-t)), min(a, t)
        den = comb(n, a)
        pmf = tuple(comb(t, x)*comb(n-t, a-x)/den for x in range(low, high+1))
        cdf = [0.0]
        for p in pmf:
            cdf.append(cdf[-1] + p)
        return low, high, pmf, tuple(cdf)

    def _best_one(self, n, t, needed):
        best = 0.0
        for a in self.sizes(n):
            low, _, pmf, cdf = self.distribution(n, t, a)
            left = (200*t-94*(n-a)-needed-1)//200
            right = (needed+94*a)//200+1
            p = cdf[min(len(pmf), max(0, left-low+1))]
            p += 1-cdf[min(len(pmf), max(0, right-low))]
            best = max(best, min(1.0, max(0.0, p)))
        return best

    def branches(self, groups, index, a):
        n, t = groups[index]
        others = groups[:index] + groups[index+1:]
        low, high, pmf, _ = self.distribution(n, t, a)
        for x, p in zip(range(low, high+1), pmf):
            yield p, tuple(sorted(others + ((a, x), (n-a, t-x))))

    def actions(self, groups):
        seen = set()
        for i, (n, t) in enumerate(groups):
            if t in (0, n) or (n, t) in seen:
                continue
            seen.add((n, t))
            for a in self.sizes(n):
                yield i, a

    def action_value(self, groups, b, i, a):
        return sum(p*self.value(child, b-1) for p, child in self.branches(groups, i, a))

    def _value(self, groups, b):
        parts = tuple(max(0, 200*t-94*n) for n, t in groups)
        gain = sum(parts)
        if gain > self.threshold:
            return 1.0
        if not b:
            return 0.0
        if b == 1:
            return max((self.best_one(n, t, self.threshold-gain+parts[i])
                        for i, (n, t) in enumerate(groups) if 0 < t < n), default=0.0)
        return max((self.action_value(groups, b, i, a) for i, a in self.actions(groups)), default=0.0)

    def _secondary(self, groups, b):
        n, k, _ = best_union(groups)
        if not b or reached(groups, self.threshold):
            return f1(n, k)
        optimum = self.value(groups, b)
        best = f1(n, k)
        for i, a in self.actions(groups):
            if abs(self.action_value(groups, b, i, a)-optimum) <= EPS:
                expected = sum(p*self.secondary(child, b-1) for p, child in self.branches(groups, i, a))
                best = max(best, expected)
        return best

    def optimal_actions(self, groups, budget):
        groups = tuple(sorted(groups))
        primary = self.value(groups, budget)
        secondary = self.secondary(groups, budget)
        out = []
        for i, a in self.actions(groups):
            p = self.action_value(groups, budget, i, a)
            if abs(p-primary) > EPS:
                continue
            e = sum(prob*self.secondary(child, budget-1) for prob, child in self.branches(groups, i, a))
            if abs(e-secondary) <= EPS:
                out.append((groups[i], a))
        return primary, secondary, out


def choose_orientation(leaves, parent, query, policy):
    """Same measurement, best physical submission orientation and known anchors."""
    q = set(query)
    other = [g for g in leaves if g['ids'] != parent['ids']]
    n, t, a = len(parent['ids']), parent['true_count'], len(q)
    low, high, pmf, _ = policy.distribution(n, t, a)
    choices = []
    for complementary in (False, True):
        measured = sorted(set(parent['ids'])-q if complementary else q)
        for bits in product((0, 1), repeat=len(other)):
            anchors = [g for g, bit in zip(other, bits) if bit]
            selected = sorted(measured + [i for g in anchors for i in g['ids']])
            known_k = sum(g['true_count'] for g in anchors)
            probability = expected = 0.0
            for x, p in zip(range(low, high+1), pmf):
                k = known_k + (t-x if complementary else x)
                score = f1(len(selected), k)
                probability += p*(200*k-94*len(selected) > 2566)
                expected += p*score
            choices.append((probability, expected, selected, complementary, known_k,
                            [g['ids'] for g in anchors]))
    choices.sort(key=lambda c: (-round(c[0], 12), -round(c[1], 12), len(c[2]), c[2]))
    p, expected, selected, complement, known_k, anchors = choices[0]
    return {'selected_ids': selected, 'orientation': 'complement' if complement else 'query',
            'known_true_count': known_k, 'anchor_groups': anchors,
            'conditional_immediate_reach_probability': p, 'conditional_expected_probe_f1': expected}
