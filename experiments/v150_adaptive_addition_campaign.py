"""V150: six submission all-integer adaptive addition campaign.

Real leaderboard records, conditional strategy calculations and hypothetical
tests are deliberately stored separately. No command uploads a submission.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from v150_adaptive_policy import Policy, best_union, choose_orientation, f1

ROOT = Path(r'D:\zgyidong')
EXP = ROOT/'experiments'
DEFAULT_OUT = EXP/'v150_adaptive_addition_campaign'
V149 = EXP/'v149_online_calibrated_addition_campaign'
BASE = V149/'v149_final_k08_from_v148.csv'
BASE_SHA = 'c8f304c44f6b3a7ed40599cae50d0f17c0c37f8456553e6171a2d7cfdc196d0a'
RUNTIME = Path(r'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies')
NODE = RUNTIME/'node/bin/node.exe'
TZ = timezone(timedelta(hours=8))
BLOCKED_ORDER = 'dd29409d-298e-451b-8231-e0c53c0e420a'
G, BASE_P, BASE_TP, LIMIT = 1044, 1045, 969, 6
RECORDS = EXP/'v25_semantic_router/cloud_dataset/semantic_records.json.gz'
NPZ = EXP/'v25_semantic_router/cloud_dataset/v25_semantic_router.npz'
CHANNELS = {
    'station': EXP/'v30_meta_stack/station_extra_trees_test.npy',
    'consensus': EXP/'v30_meta_stack/v30_consensus_test.npy',
    'template': EXP/'v30_meta_stack/template_extra_trees_test.npy',
}
LEDGERS = ['v124_campaign', 'v139_unanimous_top20_additions', 'v140_hybrid_add_swap',
           'v145_verified_swap_pair', 'v146_cumulative_chain', 'v148_two_true_decode',
           'v149_online_calibrated_addition_campaign']


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def infer_tp(score, predictions):
    try:
        s = Decimal(str(score))
    except InvalidOperation as e:
        raise ValueError('Score must be the actual displayed six-decimal number') from e
    require(s.is_finite() and 0 <= s <= 1, 'Invalid score range')
    require(s == s.quantize(Decimal('.000001')), 'Use at most six decimal places')
    den = G+predictions
    center = int(s*den/2)
    choices = [tp for tp in range(max(0, center-1), min(G, predictions, center+2)+1)
               if abs(Decimal(2*tp)/Decimal(den)-s) <= Decimal('.0000005')]
    require(len(choices) == 1, f'Score {s} at P={predictions} has TP candidates {choices}')
    return choices[0]


def load_csv(p, strict=True):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ['order_id', 'output'], f'Invalid CSV header: {p}')
        rows = list(reader)
    ids, nodes, roots = [], {}, {}
    for row in rows:
        oid = row['order_id']
        require(oid not in roots, f'Duplicate order: {p}: {oid}')
        obj = json.loads(row['output'])
        values = obj['rootcause']
        require(isinstance(values, list), 'rootcause must be a list')
        if strict:
            require(1 <= len(values) <= 8, f'Invalid root count: {oid}')
        ids.append(oid)
        roots[oid] = values
        for node in values:
            key = oid, node['@rid']
            require(key not in nodes, f'Duplicate order/RID: {p}: {key}')
            nodes[key] = node
    require(len(ids) == 546, f'Expected 546 orders: {p}')
    return ids, roots, nodes


def verified_base():
    record = read_json(V149/'online_scores.json')['final_result']
    require(record and record.get('source_class') == 'public_scored', 'V149 final is not publicly confirmed')
    require(record['matches_expected'] and record['score'] == .927717, 'Unexpected baseline result; replan required')
    ids, roots, nodes = load_csv(BASE)
    require(len(nodes) == BASE_P and infer_tp(record['score'], len(nodes)) == BASE_TP, 'Baseline TP/P mismatch')
    require(sha(BASE) == record['sha256'] == BASE_SHA, 'Baseline hash mismatch')
    return {'file': str(BASE), 'sha256': BASE_SHA, 'predictions': BASE_P, 'tp': BASE_TP,
            'score': record['score'], 'source_class': 'public_scored',
            'submitted_at': record.get('submitted_at')}, ids, roots, nodes


class EquationModel:
    def __init__(self, system):
        # SciPy is only used for integer constraints, never CSV authoring.
        try:
            import scipy.optimize
        except ImportError:
            sys.path.insert(0, str(ROOT/'.deps'))
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix, vstack
        self.np, self.milp, self.LC, self.stack = np, milp, LinearConstraint, vstack
        self.system = system
        self.index = {(x['order_id'], x['rid']): i for i, x in enumerate(system['variables'])}
        self.size = len(self.index)
        self.bounds = Bounds(np.zeros(self.size), np.ones(self.size))
        self.integrality = np.ones(self.size)
        self.csr = csr_matrix
        rr, cc, rhs = [], [], []
        for row, equation in enumerate(system['equations']):
            rr.extend([row]*len(equation['indices']))
            cc.extend(equation['indices'])
            rhs.append(equation['tp'])
        rr.extend([len(rhs)]*self.size)
        cc.extend(range(self.size))
        rhs.append(G)
        self.matrix = csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(rhs), self.size))
        self.rhs = np.asarray(rhs, dtype=float)

    def solve(self, objective=None, extra=()):
        np = self.np
        matrix, rhs = self.matrix, self.rhs
        if extra:
            rr, cc, tt = [], [], []
            for r, e in enumerate(extra):
                rr.extend([r]*len(e['indices'])); cc.extend(e['indices']); tt.append(e['tp'])
            matrix = self.stack([matrix, self.csr((np.ones(len(rr)), (rr, cc)), shape=(len(extra), self.size))], format='csr')
            rhs = np.concatenate([rhs, np.asarray(tt, dtype=float)])
        c = np.zeros(self.size) if objective is None else objective
        result = self.milp(c, integrality=self.integrality, bounds=self.bounds,
                           constraints=self.LC(matrix, rhs, rhs),
                           options={'time_limit': 60, 'mip_rel_gap': 0})
        require(result.status == 0, f'Integer equation solver failed or conflicted: status={result.status}, {result.message}')
        require(np.max(np.abs(matrix @ np.rint(result.x)-rhs)) < 1e-6, 'Noninteger/invalid equation witness')
        return result

    def classify(self, keys):
        """Witness coverage proves per-node 0/1 bounds without one solve/node."""
        np = self.np
        indices = {self.index[k] for k in keys}
        result = {i: {} for i in indices}
        for mode in ('max', 'min'):
            remaining = set(indices)
            while remaining:
                c = np.zeros(self.size)
                c[list(remaining)] = -1 if mode == 'max' else 1
                solved = self.solve(c)
                ones = {i for i in remaining if solved.x[i] > .5}
                if mode == 'max':
                    witnessed = ones
                    if not witnessed:
                        for i in remaining: result[i]['max'] = 0
                        break
                    for i in witnessed: result[i]['max'] = 1
                else:
                    witnessed = remaining-ones
                    if not witnessed:
                        for i in remaining: result[i]['min'] = 1
                        break
                    for i in witnessed: result[i]['min'] = 0
                remaining -= witnessed
        return {k: result[self.index[k]] for k in keys}

    def sum_bounds(self, keys):
        c = self.np.zeros(self.size)
        c[[self.index[k] for k in keys]] = 1
        return {'min': round(self.solve(c).fun), 'max': round(-self.solve(-c).fun)}


def collect_equations(records, order_ids):
    variables = [{'order_id': r['order_id'], 'rid': a['rid']} for r in records for a in r['alarms']]
    index = {(v['order_id'], v['rid']): i for i, v in enumerate(variables)}
    require(len(index) == len(variables), 'Duplicate semantic universe keys')
    queued = []
    old_path = EXP/'research_equation_audit/real_scored_records.json'
    for r in read_json(old_path)['records']:
        require(r.get('source_class') == 'public_scored' and r.get('status') == 'scored', 'Old ledger includes non-public entry')
        queued.append((Path(r['path']), r, str(old_path)))
    for directory in LEDGERS:
        ledger_path = EXP/directory/'online_scores.json'
        data = read_json(ledger_path)
        entries = list(data.get('records', []))
        if data.get('final_result'): entries.append(data['final_result'])
        if data.get('optional_6600_result'): entries.append(data['optional_6600_result'])
        if directory == V149.name and data.get('baseline'):
            b = data['baseline']
            queued.append((EXP/'v148_two_true_decode'/b['file'], b, str(ledger_path)))
        for r in entries:
            require(r.get('source_class', 'public_scored') == 'public_scored', f'Non-public record: {ledger_path}')
            require(r.get('status') not in ('inferred', 'simulated', 'failed'), f'Unusable status: {ledger_path}')
            queued.append((EXP/directory/r['file'], r, str(ledger_path)))
    dedup = {}
    for file, record, source in queued:
        actual_hash = sha(file)
        require(actual_hash == record['sha256'], f'Historical CSV hash changed: {file}')
        ids, _, nodes = load_csv(file, strict=False)
        require(set(ids) == set(order_ids), f'Historical order-set mismatch: {file}')
        tp = infer_tp(record['score'], len(nodes))
        require(len(nodes) == record['predictions'], f'Historical prediction mismatch: {file}')
        require(record.get('tp', record.get('inferred_tp')) == tp, f'Historical TP mismatch: {file}')
        require(set(nodes) <= index.keys(), f'Historical node not in semantic universe: {file}')
        row = {'file': str(file), 'sha256': actual_hash, 'score': record['score'], 'tp': tp,
               'predictions': len(nodes), 'indices': sorted(index[k] for k in nodes),
               'source_class': 'public_scored', 'source_ledger': source}
        if actual_hash in dedup:
            require(dedup[actual_hash]['tp'] == tp, f'Conflicting scores for same CSV: {file}')
        else:
            dedup[actual_hash] = row
    return {'version': 1, 'ground_truth_positives': G, 'variables': variables,
            'equations': list(dedup.values()), 'public_equation_count': len(dedup),
            'assumptions': ['1044 true nodes total', 'complete semantic test-node universe',
                            'deterministic micro F1', 'historical public ledgers trusted'],
            'excluded_evidence': ['OOF', 'outcomes', 'possible_delta', 'simulated', 'inferred labels as observations']}


def initial_exclusions(base_nodes):
    excluded = {}
    def add(key, reason):
        excluded.setdefault(key, []).append(reason)
    for x in read_json(EXP/'v37_online_equations/report.json').get('fixed_labels', []):
        if x['label'] == 0: add((x['order_id'], x['rid']), 'v37_fixed_false')
    # Read literal historical exclusions; never import/run the old generators.
    tree = ast.parse((EXP/'v132_build_station_additions.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.value, ast.Set):
            try:
                for k in ast.literal_eval(node.value): add(tuple(k), 'direct_online_negative_action')
            except (ValueError, TypeError):
                continue
    cat = read_json(V149/'candidate_catalog.json')
    decoded = set(read_json(V149/'decode_state.json')['decoded_positive_candidate_ids'])
    for c in cat['candidates']:
        if c['candidate_id'] not in decoded:
            add((c['order_id'], c['add_rid']), 'v149_decoded_false')
    for c in cat.get('excluded_candidates', []):
        rid = c.get('add_rid', c.get('rid'))
        if rid and c.get('reason') in ('v148_online_fixed_false', 'historical_fixed_false'):
            add((c['order_id'], rid), 'v149_exclusion_'+c['reason'])
    _, _, old = load_csv(EXP/'v120_swap_campaign/probe_00_baseline.csv')
    for directory, name in [('v118_safe_positive_campaign', 'probe_00_calibration.csv'),
                            ('v119_joint_campaign', 'probe_01.csv')]:
        _, _, altered = load_csv(EXP/directory/name, strict=False)
        for key in set(old)-set(altered): add(key, directory+'_deletion_pool')
    return excluded


def ranked_proposals(records, roots, blocked):
    import numpy as np
    with np.load(NPZ, allow_pickle=False) as data:
        ptr = data['test_alarm_ptr']
    require(len(ptr) == len(records)+1, 'Semantic array/order alignment mismatch')
    aggregate, source_ranks = {}, {}
    for channel, filename in CHANNELS.items():
        scores = np.load(filename, allow_pickle=False)
        require(len(scores) == int(ptr[-1]), 'Score-array length mismatch')
        proposals = []
        for i, r in enumerate(records):
            oid = r['order_id']; current = {x['@rid'] for x in roots[oid]}
            if oid == BLOCKED_ORDER or len(current) >= 8 or (channel == 'template' and len(current) != 1):
                continue
            candidates = [(float(scores[int(ptr[i])+j]), a) for j, a in enumerate(r['alarms'])
                          if a['rid'] not in current and (oid, a['rid']) not in blocked]
            if not candidates: continue
            score, alarm = max(candidates, key=lambda item: (item[0], item[1]['rid']))
            require(math.isfinite(score), 'Non-finite source score')
            proposals.append((score, oid, alarm))
        # Match the original source builders' tie rule before RRF aggregation.
        proposals.sort(key=lambda x: (x[0], x[1]), reverse=True)
        source_ranks[channel] = []
        for rank, (score, oid, alarm) in enumerate(proposals, 1):
            key = oid, alarm['rid']
            source = alarm.get('source', {})
            item = aggregate.setdefault(key, {'order_id': oid, 'add_rid': alarm['rid'],
                'ranks': {}, 'source_scores': {}, 'base_root_count': len(roots[oid]),
                'node': {'@rid': alarm['rid'], **{k: source.get(k, '') for k in ('title', 'location', 'reason')}}})
            item['ranks'][channel] = rank
            item['source_scores'][channel] = score
            source_ranks[channel].append({'order_id': oid, 'rid': alarm['rid'], 'rank': rank, 'score': score})
    for item in aggregate.values():
        item['rrf_score'] = sum(1/(60+r) for r in item['ranks'].values())
    return aggregate, source_ranks


def validate_file(file, baseline, candidates, selected, expected_sha=None):
    require(sha(baseline['file']) == baseline['sha256'], 'Frozen baseline changed')
    ids, _, base_nodes = load_csv(baseline['file'])
    out_ids, _, nodes = load_csv(file)
    require(out_ids == ids, 'Submission order sequence changed')
    wanted = {(candidates[i-1]['order_id'], candidates[i-1]['add_rid']) for i in selected}
    require(len(wanted) == len(selected) == len(set(selected)), 'Duplicate selected candidate')
    require(set(nodes)-set(base_nodes) == wanted and set(base_nodes) <= set(nodes), 'Actual CSV difference mismatch')
    require(all(nodes[k] == v for k, v in base_nodes.items()), 'Existing root metadata was altered')
    for i in selected:
        c = candidates[i-1]
        require(nodes[c['order_id'], c['add_rid']] == c['node'], 'Added root metadata mismatch')
    require(len(nodes) == BASE_P+len(selected), 'Prediction count mismatch')
    file_hash = sha(file)
    if expected_sha: require(file_hash == expected_sha, f'Frozen submission changed: {file}')
    require(not Path(file).read_bytes().startswith(b'\xef\xbb\xbf'), 'Submission must not have UTF-8 BOM')
    return {'orders': len(ids), 'predictions': len(nodes), 'sha256': file_hash,
            'order_sequence_unchanged': True, 'exact_difference': True, 'root_counts_1_to_8': True,
            'duplicates': 0, 'existing_root_metadata_unchanged': True}


def freeze_hashes(out):
    return {name: sha(out/name) for name in ('candidate_catalog.json', 'equation_system.json')}


def load_campaign(out):
    config = read_json(out/'campaign.json')
    for name, expected in config['frozen_files'].items():
        require(sha(out/name) == expected, f'Frozen campaign input changed: {name}')
    require(sha(config['baseline']['file']) == config['baseline']['sha256'], 'Baseline changed')
    return config, read_json(out/'candidate_catalog.json'), read_json(out/'equation_system.json'), read_json(out/'submission_manifest.json'), read_json(out/'online_scores.json')


def author(out, config, catalog, file_id, kind, selected, details):
    manifest = read_json(out/'submission_manifest.json')
    require(not any(x['probe_id'] == file_id for x in manifest['files']), 'Probe id already emitted')
    require(selected and len(selected) == len(set(selected)), 'Cannot emit empty/duplicate selection')
    for old in manifest['files']:
        require(old['selected_ids'] != sorted(selected), f'Same prediction set already exists: {old["file"]}')
    name = f'v150_{file_id}.csv'
    payload = {'base_file': config['baseline']['file'], 'output_file': str(out/name),
               'preview_file': str(out/'previews'/f'{file_id}.png'),
               'additions': [catalog['candidates'][i-1] for i in sorted(selected)]}
    spec_file = out/'authoring_specs'/f'{file_id}.json'
    write_json(spec_file, payload)
    subprocess.run([str(NODE), str(DEFAULT_OUT/'v150_artifact_builder.mjs'), str(spec_file)], check=True)
    checked = validate_file(out/name, config['baseline'], catalog['candidates'], selected)
    entry = {'probe_id': file_id, 'kind': kind, 'file': name, 'selected_ids': sorted(selected),
             'predictions': BASE_P+len(selected), 'sha256': checked['sha256'],
             'created_at': now(), 'local_validation': checked, **details}
    manifest['files'].append(entry)
    write_json(out/'submission_manifest.json', manifest)
    return entry


def build(out):
    require(not (out/'campaign.json').exists(), 'Refusing to rebuild a frozen campaign')
    baseline, ids, roots, nodes = verified_base()
    with gzip.open(RECORDS, 'rt', encoding='utf-8') as f:
        records = json.load(f)['test']
    require({r['order_id'] for r in records} == set(ids), 'Dataset order mismatch')
    system = collect_equations(records, ids)
    model = EquationModel(system)
    model.solve()
    print(f'Validated {system["public_equation_count"]} public scored equations.', flush=True)
    blocked = initial_exclusions(nodes)
    bounds, iterations = {}, 0
    while True:
        aggregate, source_ranks = ranked_proposals(records, roots, blocked)
        unknown = set(aggregate)-set(bounds)
        if unknown: bounds.update(model.classify(unknown))
        fixed_false = {key for key in aggregate if bounds[key]['max'] == 0}
        iterations += 1
        print(f'Candidate pass {iterations}: {len(aggregate)} proposals, {len(fixed_false)} equation-false.', flush=True)
        if not fixed_false: break
        for key in fixed_false: blocked.setdefault(key, []).append('fresh_public_equations_fixed_false')
    ordered = sorted(aggregate.values(), key=lambda c: (-c['rrf_score'], c['order_id'], c['add_rid']))
    eligible, seen = [], set()
    for c in ordered:
        if c['order_id'] not in seen:
            seen.add(c['order_id']); eligible.append(c)
    require(len(eligible) >= 80, 'Fewer than 80 valid distinct-order candidates; no probe emitted')
    chosen = copy.deepcopy(eligible[:80])
    for i, c in enumerate(chosen, 1):
        c['candidate_id'] = i
        c['equation_label_bounds'] = bounds[c['order_id'], c['add_rid']]
    require(not any(c['equation_label_bounds']['min'] == 1 for c in chosen),
            'A selected addition is already equation-proven true; merge it before a new blind pool campaign')
    pool_bounds = model.sum_bounds([(c['order_id'], c['add_rid']) for c in chosen])
    catalog = {'version': 1, 'baseline': baseline, 'ranking': 'sum(1/(60+source_rank)) descending; order_id,rid ascending',
               'source_ties': 'Original source score and order_id descending; per-order score and rid descending',
               'model_probability_calibrated': False, 'eligible_distinct_orders': len(eligible),
               'eligible_unique_proposals': len(aggregate), 'candidates': chosen,
               'pool_true_count_equation_bounds': pool_bounds,
               'exclusions': [{'order_id': k[0], 'rid': k[1], 'reasons': v} for k, v in sorted(blocked.items())],
               'excluded_orders': [BLOCKED_ORDER], 'source_rankings': source_ranks,
               'source_hashes': {str(p): sha(p) for p in [RECORDS, NPZ, *CHANNELS.values()]}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'equation_system.json', system)
    write_json(out/'candidate_catalog.json', catalog)
    config = {'version': 1, 'baseline': baseline, 'target': .94, 'comparison': 'strictly_greater',
              'candidate_count': 80, 'budget': {'total': 6, 'max_daily': 2, 'final_reserved': 1,
              'basis': 'six opportunities AFTER V149 final confirmation', 'platform_quota_verified': False},
              'strategy': 'total_then_four_all_integer_adaptive_splits_then_best_union',
              'probability_model': 'conditional uniform labels given counts; uncalibrated for the real pool',
              'frozen_files': freeze_hashes(out), 'created_at': now()}
    write_json(out/'campaign.json', config)
    write_json(out/'submission_manifest.json', {'version': 1, 'files': []})
    write_json(out/'online_scores.json', {'version': 1, 'baseline': baseline, 'records': [],
               'note': 'Only user-provided actual leaderboard submissions. No simulated results.'})
    first = author(out, config, catalog, 'probe_01_all80', 'total', list(range(1,81)), {})
    write_json(out/'first_probe_score_lookup.json', {
        'predictions': 1125, 'possible_counts_from_historical_equations': pool_bounds,
        'rows': [{'true_count': t, 'tp': BASE_TP+t, 'score': f'{f1(80,t):.6f}',
                  'action': 'target_achieved' if t>=51 else 'stop_target_pool' if t<=24 else 'adaptive_split'}
                 for t in range(pool_bounds['min'], pool_bounds['max']+1)]})
    audit_result = audit(out)
    write_json(out/'validation.json', audit_result)
    save_state(out)
    print(json.dumps({'first_submission': first, 'public_equations': system['public_equation_count'],
                      'eligible_orders': len(eligible), 'pool_true_count_bounds': pool_bounds}, ensure_ascii=False, indent=2))


def checked_partition(leaves):
    ids = [i for g in leaves for i in g['ids']]
    require(sorted(ids) == list(range(1,81)), 'Leaves must partition the frozen 80 candidates')
    for g in leaves:
        require(g['ids'] == sorted(g['ids']) and 0 <= g['true_count'] <= len(g['ids']), 'Invalid group count')
    require(sum(g['true_count'] for g in leaves) <= G-BASE_TP, 'More than 75 missing true nodes')


def apply_measurement(leaves, entry, tp):
    total_k = tp-BASE_TP
    require(0 <= total_k <= min(len(entry['selected_ids']), G-BASE_TP), 'Impossible added TP count')
    if entry['kind'] == 'total':
        require(not leaves, 'Total already known')
        result = [{'ids': list(range(1,81)), 'true_count': total_k}]
    elif entry['kind'] == 'split':
        parents = [g for g in leaves if g['ids'] == entry['parent_ids']]
        require(len(parents) == 1, 'Split parent is not a current leaf')
        parent = parents[0]; q = set(entry['query_ids']); p = set(parent['ids'])
        require(q < p and 1 <= len(q) <= len(p)//2, 'Invalid integer split')
        others = [g for g in leaves if g is not parent]
        anchors = entry['anchor_groups']
        require(len(anchors) == len({tuple(x) for x in anchors}), 'Duplicate anchor')
        for a in anchors: require(any(a == g['ids'] for g in others), 'Anchor is not a known other group')
        known = sum(g['true_count'] for g in others if g['ids'] in anchors)
        require(known == entry['known_true_count'], 'Known anchor TP mismatch')
        require(entry['orientation'] in ('query', 'complement'), 'Invalid orientation')
        measured = p-q if entry['orientation'] == 'complement' else q
        expected_ids = sorted(measured | {i for a in anchors for i in a})
        require(expected_ids == entry['selected_ids'], 'Physical probe does not match encoded measurement')
        observed = total_k-known
        qt = parent['true_count']-observed if entry['orientation'] == 'complement' else observed
        require(max(0,len(q)-len(p)+parent['true_count']) <= qt <= min(len(q),parent['true_count']), 'Parent/child count conflict')
        result = others + [{'ids': sorted(q), 'true_count': qt},
                           {'ids': sorted(p-q), 'true_count': parent['true_count']-qt}]
    elif entry['kind'] == 'final':
        selected = set(entry['selected_ids']); known = 0; covered = set()
        for g in leaves:
            intersection = selected.intersection(g['ids'])
            require(not intersection or intersection == set(g['ids']), 'Final includes partial unknown group')
            if intersection: covered |= intersection; known += g['true_count']
        require(covered == selected and known == total_k, 'Final score mismatches exact group sum')
        result = leaves
    else:
        raise ValueError('Unknown measurement kind')
    checked_partition(result)
    return sorted(result, key=lambda g: g['ids'])


def replay(config, manifest, scores):
    by_id = {x['probe_id']: x for x in manifest['files']}
    leaves, paused, accepted, measured = [], [], [], {}
    champion = dict(config['baseline'])
    for event in scores['records']:
        if event.get('status') != 'accepted':
            paused.append(event.get('error', 'Unresolved scoring anomaly'))
            continue
        require(not paused, 'Accepted record follows an unresolved anomaly')
        entry = by_id[event['probe_id']]
        if event['probe_id'] in measured:
            require(measured[event['probe_id']] == event['inferred_tp'], 'Repeat measurement changed deterministic TP')
        else:
            leaves = apply_measurement(leaves, entry, event['inferred_tp'])
            measured[event['probe_id']] = event['inferred_tp']
        accepted.append(event)
        score = 2*event['inferred_tp']/(G+entry['predictions'])
        old = 2*champion['tp']/(G+champion['predictions'])
        if score > old:
            champion = {'file': event['file'], 'sha256': event['sha256'], 'score': event['score'],
                        'tp': event['inferred_tp'], 'predictions': entry['predictions'], 'source_class': 'public_scored'}
    counts = tuple((len(g['ids']), g['true_count']) for g in leaves)
    n, k, indices = best_union(counts)
    selected = sorted(i for j in indices for i in leaves[j]['ids'])
    used = len(scores['records'])
    status = 'paused' if paused else 'target_achieved' if 2*champion['tp'] > .94*(G+champion['predictions']) else 'awaiting_total' if not leaves else 'stop_target_pool' if sum(g['true_count'] for g in leaves)<=24 else 'ready_to_merge' if f1(n,k)>.94 or used>=5 else 'ready_to_split'
    return {'status': status, 'leaves': leaves, 'used_submissions': used, 'remaining_submissions': max(0,LIMIT-used),
            'remaining_split_queries': min(4,max(0,5-used)), 'champion': champion,
            'best_known_union': {'selected_ids': selected, 'added_predictions': n, 'added_tp': k,
                                 'predictions': BASE_P+n, 'tp': BASE_TP+k, 'expected_f1': f1(n,k)},
            'total_true_count': sum(g['true_count'] for g in leaves) if leaves else None,
            'oracle_f1_if_all_true_nodes_identified': f1(sum(g['true_count'] for g in leaves),sum(g['true_count'] for g in leaves)) if leaves else None,
            'anomalies': paused, 'conditional_probability_is_not_calibrated': True}


def save_state(out):
    config, _, _, manifest, scores = load_campaign(out)
    state = replay(config, manifest, scores)
    write_json(out/'state.json', state)
    return state


def audit(out):
    config, cat, system, manifest, scores = load_campaign(out)
    require(len(cat['candidates']) == 80 and len({c['order_id'] for c in cat['candidates']}) == 80, 'Candidate order count')
    require([c['candidate_id'] for c in cat['candidates']] == list(range(1,81)), 'Candidate ids not canonical')
    excluded = {(r['order_id'],r['rid']) for r in cat['exclusions']}
    for c in cat['candidates']:
        require(c['order_id'] != BLOCKED_ORDER and (c['order_id'],c['add_rid']) not in excluded, 'Excluded candidate')
        require(c['equation_label_bounds'] == {'min':0,'max':1}, 'Candidate is not unresolved after screening')
        require(c['rrf_score'] == sum(1/(60+r) for r in c['ranks'].values()), 'RRF mismatch')
    checks = [validate_file(out/e['file'], config['baseline'], cat['candidates'], e['selected_ids'], e['sha256']) for e in manifest['files']]
    for e in system['equations']:
        require(sha(e['file']) == e['sha256'], f'Historical CSV changed: {e["file"]}')
    model = EquationModel(system)
    extra = []
    for r in scores['records']:
        if r.get('status') == 'accepted':
            _, _, nodes = load_csv(r['file'])
            require(sha(r['file']) == r['sha256'], 'Recorded CSV changed')
            require(infer_tp(r['score'], len(nodes)) == r['inferred_tp'], 'Stored online TP mismatch')
            extra.append({'indices':[model.index[k] for k in nodes], 'tp':r['inferred_tp']})
    model.solve(extra=extra)
    state = replay(config, manifest, scores)
    return {'all_checks_passed': True, 'csv_checks': checks, 'public_equations': system['public_equation_count'],
            'new_accepted_equations': len(extra), 'integer_system_feasible': True, 'state': state,
            'evidence_class': 'local_structure_and_equation_validation_not_a_new_public_score'}


def not_before(scores, config):
    dates = [r['submitted_at'][:10] for r in scores['records']]
    baseline_date = (config['baseline'].get('submitted_at') or '')[:10]
    if baseline_date: dates.append(baseline_date)
    day = datetime.now(TZ).date()
    while dates.count(day.isoformat()) >= 2: day += timedelta(days=1)
    return day.isoformat()


def record(out, args):
    if args.score is not None:
        try:
            parsed = Decimal(args.score)
        except InvalidOperation as error:
            raise ValueError('Invalid score input; no submission recorded') from error
        require(parsed.is_finite() and 0 <= parsed <= 1, 'Invalid score input; no submission recorded')
        require(parsed == parsed.quantize(Decimal('.000001')), 'Use the actual six-decimal displayed score')
    config, cat, system, manifest, scores = load_campaign(out)
    entry = next((e for e in manifest['files'] if e['probe_id'] == args.probe_id), None)
    require(entry is not None, 'Unknown or ungenerated probe id')
    attempt = args.attempt_id or args.probe_id
    for previous in scores['records']:
        if previous['attempt_id'] == attempt:
            require(previous['probe_id'] == args.probe_id and str(previous.get('score')) == str(float(args.score) if args.score is not None else None), 'Attempt already recorded differently')
            print(json.dumps({'idempotent': True, 'record': previous}, ensure_ascii=False)); return
    event = {'attempt_id':attempt, 'probe_id':args.probe_id, 'kind':entry['kind'], 'file':str(out/entry['file']),
             'sha256':entry['sha256'], 'predictions':entry['predictions'], 'score':float(args.score) if args.score is not None else None,
             'submitted_at':args.submitted_at or now(), 'source_class':'public_scored' if not args.failed else 'public_submission_failed',
             'status':'pending_validation', 'evidence':args.evidence}
    try:
        require(not args.failed, args.reason or 'Platform rejected submission')
        require(args.score is not None, 'Missing actual score')
        require(len(scores['records']) < LIMIT, 'Submission budget exceeded')
        stamp = datetime.fromisoformat(event['submitted_at'])
        require(stamp.tzinfo is not None, 'Submission timestamp needs UTC offset')
        event['submitted_at'] = stamp.astimezone(TZ).isoformat(timespec='seconds')
        state = replay(config, manifest, scores)
        require(state['status'] != 'paused', 'Resolve previous anomaly before further measurements')
        validate_file(out/entry['file'], config['baseline'], cat['candidates'], entry['selected_ids'], entry['sha256'])
        tp = infer_tp(args.score, entry['predictions'])
        previous_measurements = [r for r in scores['records'] if r['probe_id']==args.probe_id and r['status']=='accepted']
        if previous_measurements:
            require(all(r['inferred_tp']==tp for r in previous_measurements), 'Repeated file received a conflicting TP')
            event['repeat_measurement'] = True
        else:
            apply_measurement(state['leaves'], entry, tp)
        model = EquationModel(system)
        extra = []
        for r in scores['records']:
            if r['status'] == 'accepted':
                _,_,nodes=load_csv(r['file']); extra.append({'indices':[model.index[k] for k in nodes], 'tp':r['inferred_tp']})
        _,_,nodes=load_csv(out/entry['file'])
        extra.append({'indices':[model.index[k] for k in nodes], 'tp':tp})
        model.solve(extra=extra)
        event.update(status='accepted', inferred_tp=tp, true_count_in_submission=tp-BASE_TP,
                     reconstructed_score=2*tp/(G+entry['predictions']))
        if entry['kind'] == 'final': event['matches_expected'] = True
    except (ValueError, KeyError, OSError) as error:
        event.update(status='anomaly', error=str(error))
    scores['records'].append(event)
    write_json(out/'online_scores.json', scores)
    state = save_state(out)
    print(json.dumps({'recorded':event, 'state':state, 'next_step':'recommend (read-only) or resolve anomaly'},ensure_ascii=False,indent=2))


def recommendation(out):
    checked = audit(out)
    config, cat, _, manifest, scores = load_campaign(out)
    state = checked['state']
    common = {'state': state, 'not_before_date': not_before(scores, config),
              'quota_note':'Includes known V149 confirmation and this campaign only; check other submissions on platform.',
              'probability_note':'Conditional exchangeable-label strategy calculation; not a calibrated real-pool success probability.'}
    if state['status'] in ('paused','target_achieved','stop_target_pool') or state['remaining_submissions']==0:
        return {**common, 'action':'stop', 'reason':state['status']}
    recorded_ids = {r['probe_id'] for r in scores['records']}
    pending = [e for e in manifest['files'] if e['probe_id'] not in recorded_ids]
    if pending:
        require(len(pending)==1,'Multiple unscored files; audit before choosing')
        return {**common, 'action':'submit_existing', 'submission':pending[0]}
    best = state['best_known_union']
    champion = state['champion']
    champion_exact = 2*champion['tp']/(G+champion['predictions'])
    if state['status']=='ready_to_merge':
        return {**common, 'action':'emit_best' if best['expected_f1']>champion_exact+1e-12 else 'stop', 'reason':'known_best_union'}
    require(state['status']=='ready_to_split','Unexpected recommendation state')
    leaves = state['leaves']
    groups = tuple(sorted((len(g['ids']),g['true_count']) for g in leaves))
    policy = Policy()
    p, expected, optimal = policy.optimal_actions(groups,state['remaining_split_queries'])
    options=[]
    for nt,a in optimal:
        for g in leaves:
            if (len(g['ids']),g['true_count'])==nt: options.append((a,g['ids'][-a:],g))
    require(options,'No legal adaptive action')
    a,query,parent=min(options,key=lambda x:(x[0],x[1],x[2]['ids']))
    orientation = choose_orientation(leaves,parent,query,policy)
    return {**common,'action':'emit_split','conditional_reach_probability':p,'conditional_expected_final_f1':expected,
            'split':{'parent_ids':parent['ids'],'parent_true_count':parent['true_count'],'query_ids':query,**orientation},
            'search_space':'all integer binary splits, four-or-fewer remaining queries, full depth; known group unions',
            'dp_states':policy.value.cache_info().currsize}


def emit_next(out):
    rec = recommendation(out)
    if rec['action']=='emit_best': return emit_best(out)
    if rec['action']!='emit_split':
        print(json.dumps(rec,ensure_ascii=False,indent=2)); return
    config, cat, _, manifest, _ = load_campaign(out)
    number = 1+sum(e['kind']=='split' for e in manifest['files'])
    split=rec['split']
    entry=author(out,config,cat,f'probe_{number+1:02d}_split','split',split['selected_ids'],
                 {k:v for k,v in split.items() if k!='selected_ids'})
    write_json(out/'last_recommendation.json',rec)
    save_state(out)
    print(json.dumps({'submission':entry,'not_before_date':rec['not_before_date'],
                      'conditional_reach_probability':rec['conditional_reach_probability'],
                      'warning':rec['probability_note']},ensure_ascii=False,indent=2))


def emit_best(out):
    checked=audit(out); state=checked['state']
    config,cat,_,manifest,scores=load_campaign(out)
    require(state['status'] not in ('paused','target_achieved','awaiting_total','stop_target_pool'),'Final gate not met')
    require(state['remaining_submissions']>=1,'No final submission budget')
    best=state['best_known_union']; champ=state['champion']
    require(best['expected_f1']>2*champ['tp']/(G+champ['predictions'])+1e-12,'No improvement over actual champion')
    require(state['remaining_split_queries']==0 or best['expected_f1']>.94,'Still exploring; do not spend reserved final prematurely')
    for e in manifest['files']:
        if e['selected_ids']==best['selected_ids']:
            print(json.dumps({'use_existing':e,'expected_f1':best['expected_f1']}));return
    entry=author(out,config,cat,'final_best_union','final',best['selected_ids'],
                 {'expected_tp':best['tp'],'expected_f1':best['expected_f1'],
                  'target_expected':best['expected_f1']>.94})
    write_json(out/'final_manifest.json',{'status':'awaiting_online_confirmation','submission':entry,
                                        'rollback_champion':champ})
    print(json.dumps(entry,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir',type=Path,default=DEFAULT_OUT)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('build','audit','recommend','emit-next','emit-best'): sub.add_parser(name)
    p=sub.add_parser('record');p.add_argument('--probe-id',required=True);p.add_argument('--score')
    p.add_argument('--submitted-at');p.add_argument('--evidence',default='user_provided_actual_leaderboard_result')
    p.add_argument('--attempt-id');p.add_argument('--failed',action='store_true');p.add_argument('--reason')
    args=parser.parse_args();out=args.campaign_dir.resolve()
    if args.command=='build':build(out)
    elif args.command=='record':record(out,args)
    elif args.command=='emit-next':emit_next(out)
    elif args.command=='emit-best':emit_best(out)
    elif args.command=='audit':print(json.dumps(audit(out),ensure_ascii=False,indent=2))
    else:print(json.dumps(recommendation(out),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
