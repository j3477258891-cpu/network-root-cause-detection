"""Gate complete local comparison, then screen with ALL verified V153 observations.

Writes an assessment only. Does not emit CSV, reset budgets, or edit old ledgers.
"""
from pathlib import Path
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
V153 = HERE.with_name('v153_complementary_repair_campaign')
sys.path.insert(0, str(V153))
import campaign as v153
from bridge import c, old, action_key
from policy import Feasibility, EPS
from training import probabilities_to_actions


def gate(report):
    if not report.get('complete') or report.get('status') != 'completed' or len(report.get('folds', [])) != 5:
        return []
    summary = report['summary']
    control = max(summary[n]['32']['f1'] for n in ('narrow_catboost','narrow_v38_control'))
    return [n for n in ('wide_catboost','wide_v38_control')
            if summary[n]['32']['positive_folds'] >= 3
            and summary[n]['32']['f1'] > max(control, summary[n]['32']['base_f1'])]


def main():
    run = HERE / 'local_run_20260909'
    report = c.read(run / 'results/progress.json')
    c.require(report.get('status') == 'completed' and report.get('complete'), 'Five-fold training has not completed')
    allowed = gate(report)
    c.require(allowed == report['eligible_wide_sources'], 'Training/assessment gate disagreement')
    out = HERE / 'local_assessment'
    out.mkdir(exist_ok=False)
    config, previous, manifest, ledger, byid, eq, leaves, champion = v153.context(V153)
    old_hash = c.sha(V153 / 'online_scores.json')
    decision = {'training_complete': True, 'eligible_wide_sources': allowed,
                'actual_champion': champion, 'remaining_submissions': config['budget']['total'] - len(ledger['records']),
                'source_ledger_sha256': old_hash, 'training_report_sha256': c.sha(run / 'results/progress.json'),
                'generate_csv': False, 'competition_submitted': False, 'target': .94,
                'warning': 'OOF comparison and optimistic bounds are not public performance or success probability'}
    if not allowed:
        decision.update(decision='stop_wide_expansion', reason='Neither wide source outperformed both paired narrow controls with >=3 positive folds at budget32')
        c.write(out / 'final_gate.json', decision)
        print(json.dumps(decision, ensure_ascii=False)); return
    _, seq, roots, nodes = c.baseline()
    system = copy.deepcopy(eq.system)
    for record in ledger['records']:
        if record['status'] != 'accepted': continue
        entry = next(e for e in manifest['files'] if e['probe_id'] == record['probe_id'])
        if any(e['sha256'] == record['sha256'] for e in system['equations']): continue
        system['equations'].append({'file': str(V153 / entry['file']), 'sha256': record['sha256'],
            'score': record['score'], 'tp': record['inferred_tp'], 'predictions': record['predictions'],
            'indices': record['indices'], 'source_class': 'public_scored', 'source_ledger': str(V153 / 'online_scores.json')})
    system['public_equation_count'] = len(system['equations'])
    current_eq = c.Equations(system)
    c.require(current_eq.feasible(), 'Current score equations conflict')
    families = {}
    for family in allowed:
        raw = c.read(run / 'results' / ('predictions_' + family + '.json'))
        families[family] = probabilities_to_actions(raw['meta'], raw['probabilities'], family)
    first = next(iter(families.values()))
    allkeys = {(a['order_id'], a[f]) for a in first for f in ('remove_rid','add_rid') if a.get(f)}
    print('Classifying current equation bounds', len(allkeys), flush=True)
    bounds = c.legacy.EquationModel(system).classify(allkeys)
    blocked = old.action_exclusions(nodes, system)
    feasibility = Feasibility(current_eq, dict(enumerate(first)))
    index = {action_key(a): i for i, a in enumerate(first)}
    def hard_filter(action):
        oid = action['order_id']
        if oid == c.legacy.BLOCKED_ORDER: return 'permanently_excluded_order', None
        c.validate_actions([action], roots, nodes)
        for field in ('remove_rid', 'add_rid'):
            if not action.get(field): continue
            key = oid, action[field]
            reason = old.exclusion_reason(field, blocked.get(key, []))
            if reason: return reason, None
            if field == 'remove_rid' and bounds[key]['min'] == 1: return 'protected_equation_true', None
            if field == 'add_rid' and bounds[key]['max'] == 0: return 'equation_false_addition', None
        if action['kind'] == 'add': bound = bounds[oid, action['add_rid']]
        elif action['kind'] == 'delete':
            b = bounds[oid, action['remove_rid']]
            bound = {'min': -b['max'], 'max': -b['min']}
        else:
            i = index[action_key(action)]
            support = [dt for dt in (-1,0,1) if feasibility([([i], dt)])]
            c.require(support, 'No feasible replacement outcome')
            bound = {'min': min(support), 'max': max(support)}
        if c.f1(bound['max'], action['delta_p']) <= c.f1() + EPS:
            return 'equation_proven_no_positive_f1_gain', bound
        return None, bound
    catalog = v153.select_catalog(families, previous['candidates'], hard_filter)
    c.validate_actions(catalog['candidates'], roots, nodes)
    oracle = current_eq.oracle(catalog['candidates'])
    c.write(out / 'candidate_catalog.json', catalog)
    c.write(out / 'equation_system.json', system)
    decision.update(decision='review_new_pool_against_pending_v153_probe', candidate_count=len(catalog['candidates']),
                    oracle=oracle, reason='Training gate passed; submission selection still requires shared-budget adaptive review')
    c.require(c.sha(V153 / 'online_scores.json') == old_hash, 'Ledger changed during assessment; rerun with updated evidence')
    c.write(out / 'final_gate.json', decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == '__main__': main()
