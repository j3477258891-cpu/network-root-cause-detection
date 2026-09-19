"""Integer evidence and adaptive policy. This module never uploads files."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments'
HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import v150_adaptive_addition_campaign as legacy

G, P0, TP0, TARGET = 1044, 1045, 969, .94
BASE = legacy.BASE
TZ = timezone(timedelta(hours=8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def f1(dt=0, dp=0):
    return 2 * (TP0 + dt) / (G + P0 + dp)


def infer_tp(score, predictions):
    try:
        value = Decimal(str(score))
        require(value.is_finite() and 0 <= value <= 1, 'Invalid score')
        require(value == value.quantize(Decimal('0.000001')), 'Supply the displayed six-decimal score')
    except InvalidOperation as exc:
        raise ValueError('Invalid score') from exc
    den = G + predictions
    center = int(value * den / 2)
    choices = [t for t in range(max(0, center-2), min(G, predictions, center+2)+1)
               if abs(Decimal(2*t)/den-value) <= Decimal('0.0000005')]
    require(len(choices) == 1, f'Score/P do not identify a unique integer TP: {choices}')
    return choices[0]


def load_csv(path, strict=True):
    return legacy.load_csv(path, strict=strict)


def baseline():
    return legacy.verified_base()


def collect_system(records, ids):
    system = legacy.collect_equations(records, ids)
    index = {(v['order_id'], v['rid']): i for i, v in enumerate(system['variables'])}
    ledger = EXP / 'v150_adaptive_addition_campaign/online_scores.json'
    for r in read(ledger)['records']:
        require(r.get('source_class') == 'public_scored' and r.get('status') == 'accepted', 'Nonpublic V150 record')
        file = Path(r['file'])
        seq, _, nodes = load_csv(file)
        require(seq == ids and sha(file) == r['sha256'], 'V150 file or order alignment changed')
        tp = infer_tp(r['score'], len(nodes))
        require(tp == r['inferred_tp'] and len(nodes) == r['predictions'], 'V150 P/TP mismatch')
        if not any(e['sha256'] == r['sha256'] for e in system['equations']):
            system['equations'].append({'file': str(file), 'sha256': r['sha256'], 'score': r['score'],
                'tp': tp, 'predictions': len(nodes), 'indices': sorted(index[k] for k in nodes),
                'source_class': 'public_scored', 'source_ledger': str(ledger)})
    system['public_equation_count'] = len(system['equations'])
    return system


class Equations:
    def __init__(self, system, observations=()):
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
        self.np, self.milp, self.LC, self.csr = np, milp, LinearConstraint, csr_matrix
        self.system = system
        self.index = {(v['order_id'], v['rid']): i for i, v in enumerate(system['variables'])}
        self.n = len(self.index)
        self.rows = [{i: 1 for i in e['indices']} for e in system['equations']]
        self.rhs = [e['tp'] for e in system['equations']]
        self.rows.append(dict.fromkeys(range(self.n), 1))
        self.rhs.append(system.get('ground_truth_positives', G))
        for o in observations:
            self.rows.append({i: 1 for i in o['indices']})
            self.rhs.append(o['tp'])
        self.bounds = Bounds(np.zeros(self.n), np.ones(self.n))
        self.cache = {}

    def coeff(self, actions):
        result = {}
        for a in actions:
            for field, sign in [('add_rid', 1), ('remove_rid', -1)]:
                if a.get(field):
                    i = self.index[(a['order_id'], a[field])]
                    result[i] = result.get(i, 0) + sign
        return {i: x for i, x in result.items() if x}

    def solve(self, objective=None, extra=()):
        np = self.np
        rows = self.rows + [x[0] for x in extra]
        rhs = self.rhs + [x[1] for x in extra]
        rr, cc, vv = [], [], []
        for r, terms in enumerate(rows):
            for i, v in terms.items():
                rr.append(r); cc.append(i); vv.append(v)
        mat = self.csr((vv, (rr, cc)), shape=(len(rows), self.n), dtype=float)
        obj = np.zeros(self.n)
        for i, v in (objective or {}).items():
            obj[i] = v
        result = self.milp(obj, integrality=np.ones(self.n), bounds=self.bounds,
            constraints=self.LC(mat, rhs, rhs), options={'time_limit': 60, 'mip_rel_gap': 0})
        if result.status == 2:
            return None
        require(result.status == 0, f'Integer solver unresolved (not proof of infeasibility): {result.message}')
        require(np.max(np.abs(mat @ np.rint(result.x) - rhs)) < 1e-6, 'Invalid integer witness')
        return result

    def feasible(self, coeff=None, value=None):
        key = (tuple(sorted((coeff or {}).items())), value)
        if key not in self.cache:
            self.cache[key] = self.solve(extra=[] if value is None else [(coeff, value)]) is not None
        return self.cache[key]

    def bounds_for(self, actions):
        coeff = self.coeff(actions)
        lo = self.solve(coeff)
        hi = self.solve({i: -v for i, v in coeff.items()})
        require(lo is not None and hi is not None, 'Conflicting online equations')
        return {'min': round(lo.fun), 'max': -round(hi.fun)}

    def oracle(self, actions):
        """Exact maximum F1 over action subsets AND feasible unknown labels; not a prediction."""
        from scipy.optimize import Bounds
        np = self.np
        n = self.n + len(actions)
        rows = list(self.rows)
        lower, upper = list(self.rhs), list(self.rhs)
        dt_coeff, dp_coeff = {}, {}
        for j,a in enumerate(actions):
            choice = self.n+j
            dp_coeff[choice] = a['delta_p']
            for i,sign in self.coeff([a]).items():
                product = n; n += 1
                dt_coeff[product] = sign
                rows += [{product:1,choice:-1},{product:1,i:-1},{product:1,i:-1,choice:-1}]
                lower += [-np.inf,-np.inf,-1]
                upper += [0,0,np.inf]
        rr,cc,vv=[],[],[]
        for r,terms in enumerate(rows):
            for c,v in terms.items(): rr.append(r);cc.append(c);vv.append(v)
        mat=self.csr((vv,(rr,cc)),shape=(len(rows),n),dtype=float)
        value=f1(); result=None
        for _ in range(30):
            c=np.zeros(n)
            for i,v in dt_coeff.items(): c[i]-=2*v
            for i,v in dp_coeff.items(): c[i]+=value*v
            result=self.milp(c,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                constraints=self.LC(mat,lower,upper),options={'time_limit':60,'mip_rel_gap':0})
            require(result.status==0,'Candidate oracle solver unresolved; no reachability claim permitted')
            x=np.rint(result.x)
            dt=round(sum(x[i]*v for i,v in dt_coeff.items()))
            dp=round(sum(x[i]*v for i,v in dp_coeff.items()))
            updated=f1(dt,dp)
            if abs(updated-value)<1e-10: break
            value=updated
        else:
            raise ValueError('Fractional oracle did not converge')
        return {'f1_upper':updated,'target_possible_under_equations':updated>TARGET,
                'selected_ids':[a['candidate_id'] for j,a in enumerate(actions) if x[self.n+j]>.5],
                'delta_tp':dt,'delta_p':dp,'source_class':'optimistic_integer_oracle_not_forecast'}


def validate_actions(actions, roots, nodes):
    orders = set()
    for a in actions:
        oid, add, remove = a['order_id'], a.get('add_rid'), a.get('remove_rid')
        require(oid in roots and oid not in orders, 'Unknown/repeated action order')
        orders.add(oid)
        require(bool(add) or bool(remove), 'Empty action')
        require(not remove or (oid, remove) in nodes, 'Removed node is absent')
        require(not add or (oid, add) not in nodes, 'Added node is already selected')
        require(add != remove, 'Duplicate/no-op RID')
        dp = int(bool(add)) - int(bool(remove))
        require(a['delta_p'] == dp and 1 <= len(roots[oid])+dp <= 8, 'Invalid action prediction/root count')
        expected_kind = 'swap' if add and remove else 'add' if add else 'delete'
        require(a['kind'] == expected_kind, 'Action kind mismatch')


def validate_csv(path, actions, expected_sha=None):
    _, seq, roots, nodes = baseline()
    validate_actions(actions, roots, nodes)
    outseq, outroots, outnodes = load_csv(path)
    adds = {(a['order_id'], a['add_rid']) for a in actions if a.get('add_rid')}
    removes = {(a['order_id'], a['remove_rid']) for a in actions if a.get('remove_rid')}
    require(outseq == seq, 'Order sequence changed')
    require(set(outnodes)-set(nodes) == adds and set(nodes)-set(outnodes) == removes, 'CSV difference mismatch')
    require(all(outnodes[k] == nodes[k] for k in set(nodes)-removes), 'Preserved metadata changed')
    for a in actions:
        if a.get('add_rid'):
            require(outnodes[a['order_id'], a['add_rid']] == a['node'], 'Added metadata mismatch')
    require(len(outnodes) == P0+sum(a['delta_p'] for a in actions), 'Prediction count mismatch')
    value = sha(path)
    require(not expected_sha or value == expected_sha, 'CSV hash changed')
    return {'orders': len(seq), 'predictions': len(outnodes), 'sha256': value,
            'exact_difference': True, 'root_counts_1_to_8': True, 'duplicates': 0}


def expected_dt(action):
    return sum(int(k)*v for k, v in action['probabilities'].items())


def distribution(actions):
    result = {0: 1.0}
    for a in actions:
        nxt = {}
        probs = a['probabilities']
        require(all(math.isfinite(v) and v >= 0 for v in probs.values()) and abs(sum(probs.values())-1)<1e-6,
                'Invalid action probability distribution')
        support = {'add': {0,1}, 'delete': {-1,0}, 'swap': {-1,0,1}}[a['kind']]
        require(set(map(int, probs)) <= support, 'Impossible signed action outcome')
        for x, px in result.items():
            for y, py in probs.items():
                nxt[x+int(y)] = nxt.get(x+int(y), 0) + px*py
        result = nxt
    return result


def best_union(leaves, byid):
    require(len(leaves) <= 8, 'More than eight measured leaves')
    flat = [i for leaf in leaves for i in leaf['ids']]
    require(len(flat) == len(set(flat)), 'Measured groups overlap')
    best = {'ids': [], 'delta_tp': 0, 'delta_p': 0, 'f1': f1()}
    for mask in range(1 << len(leaves)):
        chosen = [leaf for j, leaf in enumerate(leaves) if mask >> j & 1]
        ids = sorted(i for leaf in chosen for i in leaf['ids'])
        dt = sum(leaf['delta_tp'] for leaf in chosen)
        dp = sum(byid[i]['delta_p'] for i in ids)
        score = f1(dt, dp)
        if (score, -len(ids), tuple(-i for i in ids)) > (best['f1'], -len(best['ids']), tuple(-i for i in best['ids'])):
            best = {'ids': ids, 'delta_tp': dt, 'delta_p': dp, 'f1': score}
    return best


def next_queries(groups, leaves, tested_groups, byid, calibration_phase=False, banned_kinds=()):
    options = []
    for group in groups:
        if group['group_id'] not in tested_groups and group['kind'] not in banned_kinds:
            options.append({'kind': 'group', 'group_id': group['group_id'], 'ids': group['ids']})
    if not calibration_phase:
        for leaf in leaves:
            ids = leaf['ids']  # preserve frozen rank, not outcome-dependent reranking
            seen = set()
            for n in range(1, len(ids)//2+1):
                for subset in (ids[:n], ids[-n:]):
                    key = tuple(sorted(subset))
                    if key not in seen:
                        seen.add(key)
                        options.append({'kind': 'split', 'parent_ids': ids, 'parent_delta_tp': leaf['delta_tp'],
                                        'ids': subset})
    return options


def evaluate_query(query, leaves, byid, equations):
    ids = query['ids']
    subset = [byid[i] for i in ids]
    dist = distribution(subset)
    rest_ids = [i for i in query.get('parent_ids', []) if i not in set(ids)]
    rest_dist = distribution([byid[i] for i in rest_ids])
    branches = []
    for dt, weight in dist.items():
        if query['kind'] == 'split':
            weight *= rest_dist.get(query['parent_delta_tp']-dt, 0)
        if weight <= 0 or not equations.feasible(equations.coeff(subset), dt):
            continue
        changed = copy.deepcopy(leaves)
        if query['kind'] == 'split':
            changed = [x for x in changed if set(x['ids']) != set(query['parent_ids'])]
            changed.append({'ids': rest_ids, 'delta_tp': query['parent_delta_tp']-dt})
        changed.append({'ids': ids, 'delta_tp': dt})
        best = best_union(changed, byid)
        branches.append({'delta_tp': dt, 'weight': weight, 'best_f1': best['f1']})
    total = sum(b['weight'] for b in branches)
    require(total > 0, 'No feasible modeled score branch; pause instead of inventing a posterior')
    for b in branches:
        b['conditional_weight'] = b.pop('weight')/total
    return {**query, 'branches': branches,
        'conditional_p_target': sum(b['conditional_weight'] for b in branches if b['best_f1']>TARGET),
        'expected_best_f1': sum(b['conditional_weight']*b['best_f1'] for b in branches),
        'conditional_expected_probe_f1': sum(b['conditional_weight']*f1(b['delta_tp'],sum(a['delta_p'] for a in subset)) for b in branches),
        'probability_warning': 'working independent-order model, conditioned on parent count and feasible branches; NOT calibrated target probability or exact global posterior'}


def apply_observation(leaves, entry, dt):
    result = copy.deepcopy(leaves)
    if entry['kind'] == 'group':
        require(not any(set(x['ids']) & set(entry['ids']) for x in result), 'Group overlaps measured leaves')
        result.append({'ids': entry['ids'], 'delta_tp': dt})
    elif entry['kind'] == 'split':
        parent = next((x for x in result if set(x['ids']) == set(entry['parent_ids'])), None)
        require(parent is not None, 'Split refers to a stale/nonexistent parent')
        result.remove(parent)
        child = set(entry['ids'])
        require(child and child < set(parent['ids']), 'Not a proper split')
        result += [{'ids': entry['ids'], 'delta_tp': dt},
                   {'ids': [i for i in parent['ids'] if i not in child], 'delta_tp': parent['delta_tp']-dt}]
    return result


def check_budget(records, submitted_at, limit=10):
    stamp = datetime.fromisoformat(submitted_at)
    require(stamp.tzinfo is not None, 'Submission timestamp must include timezone')
    require(stamp <= datetime.now(TZ)+timedelta(minutes=10), 'Future submission timestamp')
    require(len(records) < limit, 'Ten submission attempts exhausted (failed attempts also count)')
    used = sum(datetime.fromisoformat(r['submitted_at']).astimezone(TZ).date() == stamp.astimezone(TZ).date() for r in records)
    require(used < 2, 'Daily limit of two attempts exceeded')


def frozen_load(out):
    config = read(out/'campaign.json')
    require(sha(BASE) == config['baseline']['sha256'], 'Frozen champion changed')
    for name, checksum in config['frozen_files'].items():
        require(sha(out/name) == checksum, f'Frozen input changed: {name}')
    catalog = read(out/'candidate_catalog.json')
    return config, catalog, read(out/'equation_system.json'), read(out/'submission_manifest.json'), read(out/'online_scores.json')
