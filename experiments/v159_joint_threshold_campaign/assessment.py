"""Exact arithmetic and evidence sufficiency; no synthetic online observations."""
from fractions import Fraction
import gzip
import json
import math
from collections import Counter
import numpy as np
from scipy.stats import beta
from bridge import *


def score(tp, p, g=1044):
    return Fraction(2 * int(tp), int(g) + int(p))


def above(tp, p, target='0.938154', g=1044):
    return score(tp, p, g) > Fraction(target)


def exact_interval(successes, trials):
    require(0 <= successes <= trials, 'Invalid binomial counts')
    if not trials:
        return None
    return [0.0 if successes == 0 else float(beta.ppf(.025, successes, trials-successes+1)),
            1.0 if successes == trials else float(beta.ppf(.975, successes+1, trials-successes))]


def minimum_perfect_trials(lower=.8):
    n = 1
    while exact_interval(n, n)[0] < lower:
        n += 1
    return n


def input_paths():
    data = EXP / 'v25_semantic_router/cloud_dataset'
    training = EXP / 'v152_error_repair_campaign/training'
    paths = [data/'semantic_records.json.gz', data/'v25_semantic_router.npz',
             EXP/'v152_error_repair_campaign/training.py',
             EXP/'v150_adaptive_addition_campaign/candidate_catalog.json',
             EXP/'v150_adaptive_addition_campaign/online_scores.json',
             EXP/'v150_adaptive_policy.py']
    for fold in range(5):
        paths += [training/f'fold_{fold}.json', training/f'fold_{fold}_actions.json']
        paths += [training/f'fold_{fold}_{family}.npz'
                  for family in ('catboost', 'v38_control', 'v30_control')]
    for family in ('station_extra_trees', 'v30_consensus', 'template_extra_trees'):
        paths += [EXP/'v30_meta_stack'/f'{family}_{split}.npy' for split in ('oof', 'test')]
    return paths


def capacity_audit():
    data = EXP/'v25_semantic_router/cloud_dataset'
    with gzip.open(data/'semantic_records.json.gz', 'rt', encoding='utf-8') as f:
        records = json.load(f)
    with np.load(data/'v25_semantic_router.npz', allow_pickle=False) as z:
        folds = z['train_connected_folds'].copy()
        ptr = z['train_alarm_ptr'].copy()
        labels = z['train_labels'].copy()
    train, test = records['train'], records['test']
    errors = []
    if len(folds) != len(train) or len(set(r['order_id'] for r in train)) != len(train):
        errors.append('order alignment or uniqueness mismatch')
    require(len(folds) == len(train), 'Fold/order lengths differ')
    station_owner, signature_owner = {}, {}
    station_leaks, template_leaks = set(), set()
    for r, f in zip(train, folds):
        for station in r['station_ids']:
            if station_owner.setdefault(station, int(f)) != f:
                station_leaks.add(station)
        sig = json.dumps(r['signature'], sort_keys=True)
        if signature_owner.setdefault(sig, int(f)) != f:
            template_leaks.add(sig)
    if station_leaks or template_leaks:
        errors.append('station/template crosses connected folds')
    actual_labels = np.array([a['is_root'] for r in train for a in r['alarms']])
    if not np.array_equal(actual_labels, labels) or len(labels) != int(ptr[-1]):
        errors.append('label/node alignment mismatch')
    if any(a.get('is_root') is not None for r in test for a in r['alarms']):
        errors.append('test set unexpectedly contains labels')
    reports = []
    directory = EXP/'v152_error_repair_campaign/training'
    for f in range(5):
        metadata = read(directory/f'fold_{f}.json')
        fit, valid = set(metadata['fit_orders']), set(metadata['valid_orders'])
        expected = set(np.flatnonzero(folds == f).tolist())
        if valid != expected or fit != set(range(len(train)))-expected:
            errors.append(f'fold {f}: fit/validation alignment mismatch')
        actions = read(directory/f'fold_{f}_actions.json')
        if any(a['order_index'] not in valid or
               a['order_id'] != train[a['order_index']]['order_id'] for a in actions):
            errors.append(f'fold {f}: action/order alignment mismatch')
        family_rows = {}
        for family in ('catboost', 'v38_control', 'v30_control'):
            path = directory/f'fold_{f}_{family}.npz'
            with np.load(path, allow_pickle=False) as z:
                p, y = z['probabilities'], z['labels']
                ok = len(p) == len(y) == len(actions) and p.shape == (len(actions), 3)
                ok = ok and bool(np.isfinite(p).all()) and bool((p >= 0).all())
                ok = ok and bool(np.allclose(p.sum(axis=1), 1))
                family_rows[family] = dict(rows=len(y),aligned=ok)
                if not ok:
                    errors.append(f'fold {f}: {family} cached prediction mismatch')
        reports.append(dict(fold=f,valid_orders=len(valid),fit_orders=len(fit),
                            actions=len(actions),families=family_rows,
                            signature=metadata['signature'],
                            source_class='cached_outer_fold_not_full_campaign_trial'))
    signatures = {r['signature'] for r in reports}
    if len(signatures) != 1:
        errors.append('cached folds have different training signatures')
    # This bound grants perfect station separation. Station constraints can only reduce it.
    capacity = len(train)//len(test)
    required = minimum_perfect_trials()
    return dict(labeled_orders=len(train),target_task_orders=len(test),
                order_only_disjoint_task_upper=capacity,
                task_upper_ignores_station_constraints=True,
                complete_comparable_observed_campaign_trials=0,
                observed_success_rate=None,observed_success_interval=None,
                minimum_independent_all_success_trials=required,
                best_case_interval_at_capacity=exact_interval(capacity,capacity),
                five_perfect_trials_counterfactual=exact_interval(5,5),
                capacity_sufficient=capacity >= required,
                station_cross_fold_count=len(station_leaks),
                template_cross_fold_count=len(template_leaks),folds=reports,
                integrity_errors=errors,
                scope='Specified V150/V152/V158 data and cached predictions. Not an assertion about unavailable external datasets.',
                warning='Counterfactual intervals are sample-size bounds, not measured strategy success rates. Cached folds and resampling are not independent campaign trials.')


def build_catalog(c):
    _, _, base = csv_nodes(c.champ['file'])
    _, _, old = csv_nodes(EXP/'v149_online_calibrated_addition_campaign/v149_final_k08_from_v148.csv')
    _, _, probe = csv_nodes(EXP/'v150_adaptive_addition_campaign/v150_probe_01_all80.csv')
    added = set(probe)-set(old)
    outstanding = added-set(base)
    raw = read(EXP/'v150_adaptive_addition_campaign/candidate_catalog.json')['candidates']
    main, excluded = [], []
    for original in raw:
        if (original['order_id'],original['add_rid']) not in outstanding:
            continue
        a = dict(original,kind='add',delta_p=1,
                 action_id=f"v150:{original['candidate_id']:03d}")
        bounds = c.eq.bound([a])
        a['current_delta_tp_bounds'] = list(bounds)
        if bounds == (0,0):
            excluded.append(a)
        else:
            apply_actions(c.champ['file'],[a])
            main.append(a)
    aux = []
    for original in read(PREVIOUS/'research/preflight_catalog.json')['candidates']:
        a = dict(original,action_id=f"aux:{original['candidate_id']:03d}")
        apply_actions(c.champ['file'],[a])
        a['current_delta_tp_bounds'] = list(c.eq.bound([a]))
        aux.append(a)
    require(len(added) == 80 and len(added & set(base)) == 3, 'V150 lineage changed')
    require({a['candidate_id'] for a in excluded} == {1,27}, 'Excluded V150 actions changed')
    require(len(main) == 75 and len(aux) == 5, 'Pool sizes changed')
    require(len({a['order_id'] for a in main+aux}) == 80, 'Conflicting orders')
    require(Counter(a['kind'] for a in aux) == {'delete':3,'swap':2}, 'Auxiliary kinds changed')
    require(c.eq.bound(main) == (17,17), 'Remaining V150 count changed')
    return dict(main=main,auxiliary=aux,excluded=excluded,
                main_total_true=17,source_class='research_only_not_submission_eligible',
                legacy_auxiliary_gate_passed=False,
                rankings=['rrf','station','consensus','template'])


def mathematical_audit(c, catalog):
    main, aux = catalog['main'], catalog['auxiliary']
    base = c.champ
    branches = []
    lo, hi = c.eq.bound(aux)
    for dt in range(lo,hi+1):
        if not c.eq.possible(aux,dt):
            continue
        p, tp = base['predictions']-3, base['tp']+dt
        needed = next((n for n in range(18) if above(tp+n,p+n)),None)
        branches.append(dict(delta_tp=dt,predictions=p,tp=tp,
                             score=f'{float(score(tp,p)):.6f}',
                             extra_perfect_additions_needed=needed,
                             individual_auxiliary_labels_inferred=False))
    return dict(main_true_bounds=list(c.eq.bound(main)),auxiliary_delta_tp_bounds=[lo,hi],
                route_a_oracle=c.eq.oracle(main,base['tp'],base['predictions']),
                auxiliary_oracle=c.eq.oracle(aux,base['tp'],base['predictions']),
                joint_oracle=c.eq.oracle(main+aux,base['tp'],base['predictions']),
                auxiliary_feedback_branches=branches,
                safe_replay=[dict(score=r['score'],tp=r['tp'],probe_id=r['probe_id'])
                             for r in c.ledger['records']],
                safe_feasible_worlds=len(c.s),
                threshold_integer_rule='2000000 * delta_TP - 938154 * delta_P > 15803706',
                source_class='exact_integer_bounds_not_strategy_or_online_score')


def evaluate_snapshot(capacity, mathematics, expired=False):
    reasons = []
    if capacity['integrity_errors']:
        reasons.append('cached_prediction_integrity_failed')
    if not capacity['capacity_sufficient']:
        reasons.append('insufficient_independent_comparable_validation_tasks')
    if expired:
        reasons.append('original_deadline_expired')
    # No complete comparable independent campaign outcomes were found in these inputs.
    # Never substitute a cached action OOF gain or a conditional policy simulation.
    if not capacity['complete_comparable_observed_campaign_trials']:
        reasons.append('no_complete_comparable_holdout_campaign_results')
    return dict(assessment_complete=True,passed=False,decision='stop_keep_six_attempts',
                reasons=reasons,capacity=capacity,mathematics=mathematics,
                route_results={name:dict(status='not_run_evidence_gate_failed',
                    planned_probes=5,main_probes=5 if name == 'A' else 4,
                    auxiliary_probes=0 if name == 'A' else 1,final_reserved=1,
                    success_rate=None,success_interval95=None,
                    full_horizon_search_completed=False,selected=False)
                    for name in ('A','B')},
                bootstrap=dict(requested_iterations=2000,executed_iterations=0,
                               absolute_lower95=None,relative_lower95=None,
                               status='not_run_no_valid_complete_policy_outcomes'),
                release_authorized=False,
                outcome='Evidence-first stop required by the accepted plan. No CSV emission or full policy search.')
