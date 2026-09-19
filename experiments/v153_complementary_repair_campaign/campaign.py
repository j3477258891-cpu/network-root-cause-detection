"""V153 audit/train/build/record/recommend/emit-final. Never uploads or spends quota."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from bridge import HERE, PRIOR, c, t, old, action_key, verify_snapshot
from policy import EPS, WARNING, Feasibility, Policy


def select_catalog(families, anchors, hard_filter):
    """Filter BEFORE full ranking, top-16 nomination and reciprocal-rank fusion."""
    maps = {f: {action_key(a): a for a in actions} for f, actions in families.items()}
    c.require(maps and all(len(maps[f]) == len(actions) for f, actions in families.items()),
              'Empty models or duplicate action keys')
    for actions in families.values():
        for a in actions: c.distribution([a])  # validate finite normalized, physically possible support
    keys = set(next(iter(maps.values())))
    c.require(all(set(m) == keys for m in maps.values()), 'Incomplete per-model action coverage')
    rejected, legal, bounds = [], {}, {}
    first = next(iter(maps))
    for key in sorted(keys):
        a = maps[first][key]
        reason, bound = hard_filter(a)
        if reason:
            rejected.append({'order_id': key[0], 'remove_rid': key[1], 'add_rid': key[2], 'reason': reason})
        else:
            legal[key] = a; bounds[key] = bound
    ranks = {}
    nominated = set()
    for family, rows in maps.items():
        ranked = sorted(legal, key=lambda k: (-c.f1(c.expected_dt(rows[k]), rows[k]['delta_p']), k))
        ranks[family] = {key: i+1 for i, key in enumerate(ranked)}
        nominated.update(ranked[:16])
    anchor_keys = {action_key(a) for a in anchors}
    nominated |= anchor_keys & set(legal)
    rrf = {key: sum(1/(60+r[key]) for r in ranks.values()) for key in nominated}
    selected, used = [], set()
    for key in sorted(nominated, key=lambda k: (-rrf[k], k)):
        if key[0] in used:
            rejected.append({'order_id': key[0], 'remove_rid': key[1], 'add_rid': key[2],
                             'reason': 'independent_order_lower_fused_rank'})
            continue
        if len(selected) == 48: break
        used.add(key[0])
        model_probs = {f: m[key]['probabilities'] for f, m in maps.items()}
        main = min(ranks, key=lambda f: (ranks[f][key], f))
        support = {'add': (0, 1), 'delete': (-1, 0), 'swap': (-1, 0, 1)}[legal[key]['kind']]
        mixed = {str(k): sum(p.get(str(k), 0) for p in model_probs.values())/len(model_probs) for k in support}
        item = {**legal[key], 'candidate_id': len(selected)+1, 'model_probabilities': model_probs,
                'probabilities': mixed, 'primary_source': main,
                'full_legal_ranks': {f: r[key] for f, r in ranks.items()}, 'rrf': rrf[key],
                'v152_anchor': key in anchor_keys, 'delta_tp_bounds': bounds[key]}
        item['expected_f1'] = c.f1(c.expected_dt(item), item['delta_p'])
        selected.append(item)
    groups = []
    for kind in ('add', 'delete', 'swap'):
        for family in sorted(families):
            ids = [a['candidate_id'] for a in selected if a['kind'] == kind and a['primary_source'] == family]
            for start in range(0, len(ids), 8):
                groups.append({'group_id': f'G{len(groups)+1:02d}', 'kind': kind,
                               'primary_source': family, 'ids': ids[start:start+8]})
    return {'candidates': selected, 'groups': groups, 'rejected': rejected,
            'legal_actions': len(legal), 'nominated_unique_actions': len(nominated),
            'families': sorted(families), 'source_class': 'model_estimated',
            'warning': WARNING, 'ranking': 'full legal rank reciprocal fusion; no consensus confidence claim'}


def build(out):
    c.require(not (out/'campaign.json').exists(), 'Campaign frozen; do not rebuild')
    report = c.read(out/'model_comparison.json')
    c.require(report['complete'], 'Model comparison incomplete')
    c.require(report['input_hashes'] == t.source_hashes(), 'Training inputs changed')
    raw = c.read(out/'training_candidates.json')
    c.require(raw['report_sha256'] == c.sha(out/'model_comparison.json'), 'Prediction/report mismatch')
    verify_snapshot(c.read(out/'protected_sources.json'))
    b, seq, roots, nodes = c.baseline()
    system = c.collect_system(old.records_data()['test'], seq)
    eq = c.Equations(system)
    c.require(eq.feasible(), 'Historical score equations conflict')
    blocked = old.action_exclusions(nodes, system)
    first = next(iter(raw['families'].values()))
    allkeys = {(a['order_id'], a[field]) for a in first for field in ('add_rid', 'remove_rid') if a.get(field)}
    print('classifying historical node bounds', flush=True)
    bounds = c.legacy.EquationModel(system).classify(allkeys)
    feasible = Feasibility(eq, {i: a for i, a in enumerate(first)})
    ids_by_key = {action_key(a): i for i, a in enumerate(first)}
    def hard_filter(a):
        oid = a['order_id']
        if oid == c.legacy.BLOCKED_ORDER: return 'permanently_excluded_order', None
        c.validate_actions([a], roots, nodes)
        for field in ('add_rid', 'remove_rid'):
            if not a.get(field): continue
            key = (oid, a[field])
            reason = old.exclusion_reason(field, blocked.get(key, []))
            if reason: return reason, None
            if field == 'remove_rid' and bounds[key]['min'] == 1: return 'protected_equation_true', None
            if field == 'add_rid' and bounds[key]['max'] == 0: return 'equation_false_addition', None
        # For adds/deletes the node bound is exact. Only swaps need a joint query.
        if a['kind'] == 'add': bound = bounds[oid, a['add_rid']]
        elif a['kind'] == 'delete':
            node_bound = bounds[oid, a['remove_rid']]
            bound = {'min': -node_bound['max'], 'max': -node_bound['min']}
        else:
            i = ids_by_key[action_key(a)]
            support = [dt for dt in (-1, 0, 1) if feasible([([i], dt)])]
            c.require(support, 'No feasible swap outcome')
            bound = {'min': min(support), 'max': max(support)}
        if c.f1(bound['max'], a['delta_p']) <= c.f1()+EPS:
            return 'equation_proven_no_positive_f1_gain', bound
        return None, bound
    anchors = c.read(PRIOR/'candidate_catalog.json')['candidates']
    catalog = select_catalog(raw['families'], anchors, hard_filter)
    c.validate_actions(catalog['candidates'], roots, nodes)
    oracle = eq.oracle(catalog['candidates'])
    c.write(out/'candidate_catalog.json', catalog)
    c.write(out/'equation_system.json', system)
    config = {'version': 153, 'created_at': c.now(), 'baseline': b, 'target': c.TARGET,
              'objective': 'maximize_realized_best_f1',
              'budget': {'total': 10, 'daily': 2, 'max_probes': 8, 'merge_reserve': 1, 'anomaly_or_risk': 1},
              'frozen_files': {name: c.sha(out/name) for name in ('candidate_catalog.json', 'equation_system.json',
                              'model_comparison.json', 'training_candidates.json')},
              'historical_ledger_hashes': {str(p): c.sha(p) for p in (PRIOR/'online_scores.json',
                  HERE.with_name('v149_online_calibrated_addition_campaign')/'online_scores.json',
                  HERE.with_name('v150_adaptive_addition_campaign')/'online_scores.json')},
              'initial_oracle': oracle, 'note': 'An optimistic cap below .94 does not prohibit incremental gain'}
    c.write(out/'campaign.json', config)
    c.write(out/'submission_manifest.json', {'files': []})
    c.write(out/'online_scores.json', {'baseline': b, 'records': [],
            'note': 'Only actual user-supplied attempts. Failed and anomalous attempts count.'})
    return {'built': True, 'candidate_count': len(catalog['candidates']),
            'groups': catalog['groups'], 'oracle': oracle, 'next': 'recommend --emit'}


def context(out):
    config, cat, system, manifest, ledger = c.frozen_load(out)
    for p, h in config['historical_ledger_hashes'].items():
        c.require(c.sha(p) == h, 'Historical ledger updated outside V153; re-audit before continuing')
    byid = {a['candidate_id']: a for a in cat['candidates']}
    entries = {e['probe_id']: e for e in manifest['files']}
    # Derive indices/P/TP from immutable files, not writable ledger's cached columns.
    observations, leaves, accepted_sets = [], [], set()
    champ = copy.deepcopy(config['baseline'])
    for r in ledger['records']:
        c.require(r['probe_id'] in entries, 'Unknown probe in score ledger')
        if r['status'] != 'accepted': continue
        e = entries[r['probe_id']]
        checked = c.validate_csv(out/e['file'], [byid[i] for i in e['ids']], e['sha256'])
        tp = c.infer_tp(r['score'], checked['predictions'])
        c.require((tp, checked['predictions'], checked['sha256']) ==
                  (r['inferred_tp'], r['predictions'], r['sha256']), 'Score ledger P/TP/hash drift')
        c.require(r['delta_tp'] == tp-c.TP0, 'Ledger delta mismatch')
        observations.append({'indices': r['indices'], 'tp': tp})
        key = tuple(sorted(e['ids']))
        if key not in accepted_sets and e['kind'] in ('group', 'split'):
            leaves = c.apply_observation(leaves, e, tp-c.TP0)
        accepted_sets.add(key)
        if c.f1(tp-c.TP0, checked['predictions']-c.P0) > 2*champ['tp']/(c.G+champ['predictions']):
            champ = {'file': str(out/e['file']), 'sha256': e['sha256'], 'predictions': r['predictions'],
                     'tp': tp, 'score': r['score'], 'source_class': 'public_scored'}
    eq = c.Equations(system, observations)
    # Reconstruct and compare every record's selected indices.
    for r in ledger['records']:
        if r['status'] != 'accepted': continue
        e = entries[r['probe_id']]
        _, _, nodes = c.load_csv(out/e['file'])
        c.require(r['indices'] == sorted(eq.index[k] for k in nodes), 'Ledger indices drift')
    c.require(eq.feasible(), 'Current public score equations conflict')
    return config, cat, manifest, ledger, byid, eq, leaves, champ


def recommendation(out):
    if not (out/'campaign.json').exists(): return {'decision': 'train_then_build', 'allow_submission': False}
    config, cat, manifest, ledger, byid, eq, leaves, champ = context(out)
    attempts = len(ledger['records']); champ_f1 = 2*champ['tp']/(c.G+champ['predictions'])
    best = c.best_union(leaves, byid)
    entries = {e['probe_id']: e for e in manifest['files']}
    report = {'actual_champion': champ, 'attempts_used': attempts, 'remaining_submissions': 10-attempts,
              'measured_leaves': leaves, 'best_known_union': best, 'allow_submission': False,
              'probability_warning': WARNING, 'platform_quota_live_checked': False}
    for e in manifest['files']:
        c.require((out/e['file']).is_file() and c.sha(out/e['file']) == e['sha256'], 'Generated submission hash changed')
    for e in eq.system['equations']:
        c.require(Path(e['file']).is_file() and c.sha(e['file']) == e['sha256'], 'Historical CSV hash changed')
    if any(r['status'] == 'anomaly' and not r.get('resolved_by') for r in ledger['records']):
        return dict(report, decision='pause_anomaly_recheck_same_file')
    if any(r['status'] == 'accepted' and entries[r['probe_id']]['kind'] == 'risk' for r in ledger['records']):
        return dict(report, decision='final_risk_completed_retain_best')
    if attempts >= 10: return dict(report, decision='budget_exhausted_retain_champion')
    pending = [e for e in manifest['files'] if not any(r['probe_id'] == e['probe_id'] for r in ledger['records'])]
    if pending: return dict(report, decision='await_existing_file_feedback', pending=pending[0])
    oracle = eq.oracle(list(byid.values())); report['oracle'] = oracle
    report['target_possible_under_equations'] = oracle['f1_upper'] > c.TARGET
    if oracle['f1_upper'] <= champ_f1+EPS:
        return dict(report, decision='stop_no_remaining_gain_retain_champion')
    feasible = Feasibility(eq, byid)
    policy = Policy(byid, cat['groups'], cat['families'], feasible)
    report['model_weights'] = policy.weights(leaves)
    attempted = {tuple(sorted(entries[r['probe_id']]['ids'])) for r in ledger['records']}
    probe_attempts = sum(entries[r['probe_id']]['kind'] in ('group', 'split') for r in ledger['records'])
    remaining_probes = min(8-probe_attempts, 8-attempts)
    merge = {'kind': 'merge', 'ids': best['ids'], 'expected_tp': c.TP0+best['delta_tp'],
             'expected_f1': best['f1'], 'evidence': 'exact integer counts of disjoint measured groups'}
    selected = None
    if remaining_probes > 0:
        selected = policy.recommend(leaves, attempted, champ_f1, remaining_probes)
    if selected is None and best['f1'] > champ_f1+EPS:
        selected = merge
    risk_used = any(entries[r['probe_id']]['kind'] == 'risk' for r in ledger['records'])
    if selected is None and not risk_used:
        selected = policy.risk(leaves, champ_f1, attempted)
    if selected is None: return dict(report, decision='stop_no_valuable_query_retain_champion')
    if tuple(sorted(selected['ids'])) in attempted:
        return dict(report, decision='stop_duplicate_prediction_set')
    today = datetime.now(c.TZ).date()
    daily = sum(datetime.fromisoformat(r['submitted_at']).astimezone(c.TZ).date() == today for r in ledger['records'])
    return dict(report, decision='submit_'+selected['kind'], allow_submission=daily < 2,
                daily_attempts_in_campaign=daily, next_action=selected,
                earliest_submission='next local day' if daily >= 2 else 'after checking platform quota',
                integer_query_cache_entries=len(feasible.cache))


def author(out, query):
    config, cat, manifest, ledger, byid, eq, leaves, champ = context(out)
    same = next((e for e in manifest['files'] if set(e['ids']) == set(query['ids'])), None)
    if same: return same
    probe_id = f'p{len(manifest["files"])+1:02d}'
    kind = query['kind']
    filename = f'v153_{"final" if kind in ("merge", "risk") else "probe"}_{probe_id}_{kind}.csv'
    actions = [byid[i] for i in query['ids']]
    spec = {'base_file': str(c.BASE), 'base_sha256': config['baseline']['sha256'],
            'output_file': str(out/filename), 'preview_file': str(out/'previews'/f'{probe_id}.png'), 'actions': actions}
    spec_path = out/'authoring_specs'/f'{probe_id}.json'
    c.write(spec_path, spec)
    subprocess.run([str(old.NODE), str(HERE/'artifact_builder.mjs'), str(spec_path)], check=True)
    validated = c.validate_csv(out/filename, actions)
    entry = {**query, 'probe_id': probe_id, 'file': filename, 'sha256': validated['sha256'],
             'predictions': validated['predictions'], 'delta_p': sum(a['delta_p'] for a in actions),
             'local_validation': validated, 'created_at': c.now()}
    manifest['files'].append(entry)
    c.write(out/'submission_manifest.json', manifest)
    return entry


def recommend(out, emit=False):
    try:
        result = recommendation(out)
    except (ValueError, FileNotFoundError) as exc:
        result = {'decision': 'pause_integrity_or_equation_error', 'allow_submission': False, 'error': str(exc)}
    if emit:
        if result.get('next_action') and result['allow_submission']:
            result['file'] = author(out, result['next_action'])
        c.write(out/'recommendation.json', result)
        c.write(out/'final_gate.json', result)
    return result


def record(out, probe_id, attempt_id, score, submitted_at, evidence, failed=False, resolves=None):
    config, cat, manifest, ledger, byid, eq, leaves, champ = context(out)
    c.require(evidence.strip(), 'Actual score/failure evidence required')
    previous = next((r for r in ledger['records'] if r['attempt_id'] == attempt_id), None)
    if previous:
        c.require(previous['probe_id'] == probe_id and previous['score'] == score and
                  previous['submitted_at'] == submitted_at, 'Attempt-id content conflict')
        return {'idempotent': True, 'record': previous}
    c.check_budget(ledger['records'], submitted_at)
    e = next((e for e in manifest['files'] if e['probe_id'] == probe_id), None)
    c.require(e is not None, 'Unknown emitted probe')
    anomaly = None
    if resolves:
        anomaly = next((r for r in ledger['records'] if r['attempt_id'] == resolves), None)
        c.require(anomaly and anomaly['status'] == 'anomaly' and anomaly['probe_id'] == probe_id,
                  'Resolution needs a new real repeat of the SAME file')
    attempt = {'attempt_id': attempt_id, 'probe_id': probe_id, 'score': score,
               'submitted_at': submitted_at, 'evidence': evidence,
               'status': 'failed' if failed else 'accepted', 'source_class': 'actual_submission_attempt'}
    if not failed:
        try:
            actions = [byid[i] for i in e['ids']]
            verified = c.validate_csv(out/e['file'], actions, e['sha256'])
            tp = c.infer_tp(score, verified['predictions'])
            c.require(eq.feasible(eq.coeff(actions), tp-c.TP0), 'Actual score conflicts with historical equations')
            c.require(tp == e.get('expected_tp', tp), 'Exact merge differs from actual score')
            _, _, nodes = c.load_csv(out/e['file'])
            attempt.update(status='accepted', source_class='public_scored', inferred_tp=tp, delta_tp=tp-c.TP0,
                           predictions=verified['predictions'], sha256=verified['sha256'],
                           indices=sorted(eq.index[k] for k in nodes))
        except (ValueError, KeyError, TypeError) as exc:
            attempt.update(status='anomaly', error=str(exc))
    if anomaly is not None and attempt['status'] == 'accepted': anomaly['resolved_by'] = attempt_id
    ledger['records'].append(attempt)
    c.write(out/'online_scores.json', ledger)
    # Review only: recording a result NEVER silently generates the next file.
    result = recommend(out)
    c.write(out/'recommendation.json', result); c.write(out/'final_gate.json', result)
    return {'record': attempt, 'assessment': result}


def prepare(out):
    """Prepare only a reviewed probe; never relax daily submission or final gates."""
    result = recommend(out)
    if result.get('decision') == 'await_existing_file_feedback':
        return result
    c.require(result.get('decision') in ('submit_group', 'submit_split') and
              result.get('next_action'), 'No valid next probe to prepare')
    before = c.sha(out/'online_scores.json')
    result['file'] = author(out, result['next_action'])
    c.require(c.sha(out/'online_scores.json') == before, 'Preparation altered actual scores')
    result['prepared_only'] = True
    result['competition_upload_performed'] = False
    c.write(out/'recommendation.json', result)
    c.write(out/'final_gate.json', result)
    return result


def audit(out):
    config, cat, manifest, ledger, byid, eq, leaves, champ = context(out)
    for e in manifest['files']:
        c.validate_csv(out/e['file'], [byid[i] for i in e['ids']], e['sha256'])
    protected = verify_snapshot(c.read(out/'protected_sources.json'))
    return {'integer_feasible': True, 'public_equations': len(eq.system['equations']),
            'actual_champion': champ, 'candidate_count': len(byid), 'group_count': len(cat['groups']),
            'files_checked': len(manifest['files']), 'protected_files_unchanged': protected,
            'attempts': len(ledger['records']), 'remaining_submissions': 10-len(ledger['records']),
            'platform_quota_live_checked': False, 'best_known_union': c.best_union(leaves, byid)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('audit'); sub.add_parser('train'); sub.add_parser('build'); sub.add_parser('emit-final')
    sub.add_parser('prepare')
    r = sub.add_parser('recommend'); r.add_argument('--emit', action='store_true')
    r = sub.add_parser('record')
    for name in ('probe-id', 'attempt-id', 'submitted-at', 'evidence'): r.add_argument('--'+name, required=True)
    r.add_argument('--score'); r.add_argument('--failed', action='store_true'); r.add_argument('--resolves-attempt')
    args = p.parse_args(); out = args.out.resolve()
    if args.command == 'train':
        from training import train
        from threadpoolctl import threadpool_limits
        with threadpool_limits(4): result = train(out)
    elif args.command == 'build': result = build(out)
    elif args.command == 'audit': result = audit(out)
    elif args.command == 'prepare': result = prepare(out)
    elif args.command == 'record':
        result = record(out, args.probe_id, args.attempt_id, args.score, args.submitted_at,
                        args.evidence, args.failed, args.resolves_attempt)
    elif args.command == 'emit-final':
        suggested = recommend(out)
        c.require(suggested['decision'] in ('submit_merge', 'submit_risk') and suggested['allow_submission'],
                  'No permitted final currently recommended')
        result = recommend(out, True)
    else: result = recommend(out, args.emit)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'decision': 'do_not_submit'}, ensure_ascii=False), file=sys.stderr)
        raise
