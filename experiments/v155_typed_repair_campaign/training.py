"""V155 typed error-repair training: separate add/delete/swap models, U-based gate.

Reuses the frozen V152 node cross-fitting cache and dataset helpers (same data,
same node models, same seed) but trains per-action-type CatBoost and ET/HGB models
with equal per-order-per-type sample weights and no auto class balancing.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from bridge import PRIOR, t152
from core import G, P0, TP0, TARGET, U_TARGET, f1, baseline, read, write, require, now

SEED = t152.SEED
KINDS = ('add', 'delete', 'swap')
GRID = {'v30_control': [0], 'v38_control': [0], 'catboost': [4, 6], 'v38_typed': [0]}
CONTROLS = ('v30_control', 'v38_control')
TYPED = ('catboost', 'v38_typed')


class Trainer(t152.Trainer):
    def __init__(self, out, deadline_hours=24, smoke=False):
        # Reuse the frozen V152 dataset/arrays/signature verbatim.
        super().__init__(out, deadline_hours, smoke)
        # Override the density target to the p03 champion (P0=1049).
        self.p0 = P0

    def node_predictions(self, fit_orders):
        self.check_time()
        fit_orders = sorted(map(int, fit_orders))
        key = t152.digest([self.signature, fit_orders])
        previous = PRIOR / 'training' / 'node_cache' / (key + '.npz')
        if previous.exists():
            with np.load(previous, allow_pickle=False) as z:
                return z['train'], z['test']
        return super().node_predictions(fit_orders)

    def masks(self, split, orders, scores):
        ptr = self.arrays[f'{split}_alarm_ptr']
        mask = np.zeros(len(scores), dtype=bool)
        optional = []
        for oi in orders:
            rows = np.arange(ptr[oi], ptr[oi + 1])
            ranked = rows[np.argsort(-scores[rows].mean(1), kind='stable')][:8]
            mask[ranked[0]] = True
            optional.extend(ranked[1:])
        # p03 champion prediction density (P0=1049), not V149 (1045).
        target = min(len(optional) + len(orders), max(len(orders), round(self.p0 * len(orders) / 546)))
        optional.sort(key=lambda i: (-float(scores[i].mean()), int(i)))
        mask[optional[:target - len(orders)]] = True
        return mask

    def typed_models(self, family, kind, param, seed):
        if family == 'catboost':
            from catboost import CatBoostClassifier
            return [CatBoostClassifier(iterations=30 if self.smoke else 240, depth=param,
                learning_rate=.05, l2_leaf_reg=5, loss_function='MultiClass', thread_count=4,
                random_seed=seed, verbose=False, allow_writing_files=False)]
        # v38_typed: ET/HGB ensemble WITHOUT auto class balancing.
        return [ExtraTreesClassifier(n_estimators=32 if self.smoke else 800, min_samples_leaf=4,
                max_features=.45, n_jobs=4, random_state=seed),
                HistGradientBoostingClassifier(max_iter=20 if self.smoke else 260, learning_rate=.045,
                max_leaf_nodes=15, min_samples_leaf=24, l2_regularization=1.5,
                early_stopping=False, random_state=seed)]

    def fit_typed(self, family, param, fit, valid, seed=SEED):
        self.check_time()
        require(len(fit['meta']) and len(valid['meta']), 'No boundary actions in fold')
        out = np.zeros((len(valid['meta']), 3))
        models = {}
        for kind in KINDS:
            fi = [i for i, a in enumerate(fit['meta']) if a['kind'] == kind]
            vi = [i for i, a in enumerate(valid['meta']) if a['kind'] == kind]
            if not fi or not vi:
                continue
            xf = fit['x'][fi]; yf = fit['y'][fi]; xv = valid['x'][vi]
            if len(np.unique(yf)) < 2:
                p = np.full((len(vi), 3), 1e-6)
                p[:, int(yf[0]) + 1] = 1
                out[vi] = p
                continue
            # Equal total sample weight per order within this action type.
            counts = Counter(fit['meta'][i]['order_id'] for i in fi)
            w = np.array([1.0 / counts[fit['meta'][i]['order_id']] for i in fi], dtype=float)
            models_kind = self.typed_models(family, kind, param, seed)
            results = []
            for model in models_kind:
                model.fit(xf, yf, sample_weight=w)
                results.append(t152.probabilities(model, xv))
            out[vi] = np.mean(results, axis=0)
            models[kind] = models_kind
        return t152.physical(out, valid['meta']), models

    def fit_predict(self, family, param, fit, valid, seed=SEED):
        self.check_time()
        require(len(fit['meta']) and len(valid['meta']), 'No boundary actions in fold')
        if family == 'v30_control':
            return valid['prior'].copy(), None
        if family in TYPED:
            return self.fit_typed(family, param, fit, valid, seed)
        # v38_control: mixed untyped ET/HGB (the V152 control, class-balanced).
        if len(np.unique(fit['y'])) < 2:
            p = np.full((len(valid['meta']), 3), 1e-6)
            p[:, int(fit['y'][0]) + 1] = 1
            return t152.physical(p, valid['meta']), None
        models = [ExtraTreesClassifier(n_estimators=32 if self.smoke else 800, min_samples_leaf=4,
                   max_features=.45, class_weight='balanced', n_jobs=4, random_state=seed),
                  HistGradientBoostingClassifier(max_iter=20 if self.smoke else 260, learning_rate=.045,
                   max_leaf_nodes=15, min_samples_leaf=24, l2_regularization=1.5,
                   early_stopping=False, random_state=seed)]
        results = []
        for model in models:
            model.fit(fit['x'], fit['y'])
            results.append(t152.probabilities(model, valid['x']))
        return t152.physical(np.mean(results, axis=0), valid['meta']), models


def selected_action_indices_u(bundle, probabilities_):
    """Freeze decisions without reading validation labels or validation F1."""
    expected = probabilities_[:, 2] - probabilities_[:, 0]
    gains = [2 * expected[i] - TARGET * bundle['meta'][i]['delta_p'] for i in range(len(expected))]
    ranked = sorted(range(len(expected)), key=lambda i: (-gains[i],
        bundle['meta'][i]['order_id'], bundle['meta'][i].get('add_rid') or '',
        bundle['meta'][i].get('remove_rid') or ''))
    unique, used = [], set()
    for i in ranked:
        oid = bundle['meta'][i]['order_id']
        if oid in used or gains[i] <= 0:
            continue
        used.add(oid)
        unique.append(i)
    return unique


def score_actions(bundle, probabilities_, labels, ptr, budgets=(16, 32, 48)):
    unique = selected_action_indices_u(bundle, probabilities_)
    orders = bundle['orders']
    rows = t152.rows_for(orders, ptr)
    p0 = int(bundle['mask'][rows].sum())
    tp0 = int(labels[rows][bundle['mask'][rows]].sum())
    g = int(labels[rows].sum())
    output = {}
    for budget in budgets:
        selected = unique[:max(1, round(budget * len(orders) / 546))]
        dt = int(bundle['y'][selected].sum())
        dp = sum(bundle['meta'][i]['delta_p'] for i in selected)
        output[str(budget)] = {'orders': len(orders), 'actions': len(selected), 'g': g,
            'base_p': p0, 'base_tp': tp0, 'delta_p': dp, 'delta_tp': dt,
            'U': 2 * dt - TARGET * dp, 'f1': f1(dt, dp),
            'add_tp': int(sum(bundle['y'][i] for i in selected if bundle['meta'][i]['kind'] == 'add')),
            'delete_tp': int(sum(bundle['y'][i] for i in selected if bundle['meta'][i]['kind'] == 'delete')),
            'swap_tp': int(sum(bundle['y'][i] for i in selected if bundle['meta'][i]['kind'] == 'swap')),
            'selected_action_indices': selected}
    return output


def pooled(folds, family, budget='32'):
    items = [f['models'][family]['metrics'][budget] for f in folds]
    dt = sum(x['delta_tp'] for x in items)
    dp = sum(x['delta_p'] for x in items)
    return {'U': 2 * dt - TARGET * dp, 'delta_tp': dt, 'delta_p': dp,
            'f1': f1(dt, dp), 'positive_folds': sum(x['U'] > 0 for x in items),
            'worst_fold_U': min(x['U'] for x in items), 'actions': sum(x['actions'] for x in items)}


def train(out, max_hours=24, smoke=False):
    out = Path(out)
    require(not (out / 'campaign.json').exists(), 'Frozen campaign cannot be retrained')
    run = out / ('smoke_training' if smoke else 'training')
    run.mkdir(parents=True, exist_ok=True)
    require(0 < max_hours <= 24, 'Training window must be positive and at most 24 hours')
    window_path = out / ('smoke_training_window.json' if smoke else 'training_window.json')
    if window_path.exists():
        window = read(window_path)
    else:
        start = datetime.fromisoformat(now())
        window = {'started_at': start.isoformat(), 'deadline_at': (start + timedelta(hours=max_hours)).isoformat()}
        write(window_path, window)
    deadline = datetime.fromisoformat(window['deadline_at']).timestamp()
    remaining = min(max_hours, (deadline - time.time()) / 3600)
    require(remaining > 0, 'Original training deadline expired; do not silently reset the clock')
    t = Trainer(run, remaining, smoke)
    t.deadline = min(t.deadline, deadline)
    families = list(CONTROLS) + list(TYPED)
    report = {'version': 155, 'source_class': 'nested_group_oof', 'started_at': window['started_at'],
        'resumed_at': now(), 'deadline_at': window['deadline_at'], 'complete': False,
        'protocol': {'outer_folds': 5, 'inner_folds': 2, 'node_crossfit_folds': 2,
            'primary_budget_per_546_orders': 32, 'aux_budgets': [16, 48],
            'baseline': 'p03 champion P=1049,TP=971,F1=0.927855',
            'target': TARGET, 'U_metric': '2*dTP - 0.937855*dP', 'U_target': U_TARGET,
            'typed': 'separate add/delete/swap models; equal order weight per type; no auto class balance',
            'controls': 'v30 node prior + v38 mixed untyped ET/HGB',
            'features': 'raw V16 graph/time + fixed text hashing; no saved supervised scores'},
        'input_hashes': t.hashes, 'smoke_only': smoke, 'folds': [],
        'warnings': ['OOF model selection is not a calibrated probability of reaching 0.937855']}
    write(run / 'progress.json', report)
    for outer in range(1 if smoke else 5):
        checkpoint = run / f'fold_{outer}.json'
        if checkpoint.exists():
            saved = read(checkpoint)
            require(saved['signature'] == t.signature, 'Checkpoint source/protocol drift; use a new output directory')
            report['folds'].append(saved)
            write(run / 'progress.json', report)
            continue
        fit_orders, valid_orders = np.flatnonzero(t.folds != outer), np.flatnonzero(t.folds == outer)
        print(f'outer fold {outer + 1}/5: fit {len(fit_orders)}, validate {len(valid_orders)} orders', flush=True)
        inner_results = {family: {param: [] for param in GRID[family]} for family in families}
        for inner, (fit, valid) in enumerate(t152.group_splits(fit_orders, t.folds)):
            scores, _ = t.crossfit_nodes(fit, valid)
            train_bundle = t.actions('train', fit, scores)
            valid_bundle = t.actions('train', valid, scores)
            for family in list(families):
                for param in GRID[family]:
                    p, _ = t.fit_predict(family, param, train_bundle, valid_bundle)
                    metrics = score_actions(valid_bundle, p, t.y, t.arrays['train_alarm_ptr'])
                    inner_results[family][param].append((p, valid_bundle, metrics['32']['U']))
            print(f'  inner {inner + 1}/2 complete', flush=True)
        scores, _ = t.crossfit_nodes(fit_orders, valid_orders)
        train_bundle = t.actions('train', fit_orders, scores)
        valid_bundle = t.actions('train', valid_orders, scores)
        fold = {'fold': outer, 'signature': t.signature, 'models': {},
                'fit_orders': list(map(int, fit_orders)), 'valid_orders': list(map(int, valid_orders))}
        for family in families:
            param = max(GRID[family], key=lambda p: (np.mean([x[2] for x in inner_results[family][p]]), -p))
            entries = inner_results[family][param]
            inner_p = np.concatenate([x[0] for x in entries])
            inner_y = np.concatenate([x[1]['y'] for x in entries])
            inner_meta = [m for x in entries for m in x[1]['meta']]
            temp = t152.temperature(inner_p, inner_y, inner_meta)
            p, _ = t.fit_predict(family, param, train_bundle, valid_bundle)
            p = t152.physical(p, valid_bundle['meta'], temp)
            metrics = score_actions(valid_bundle, p, t.y, t.arrays['train_alarm_ptr'])
            fold['models'][family] = {'param': param, 'temperature': temp, 'metrics': metrics}
            print(f'  {family}: U32={metrics["32"]["U"]:.4f}, dTP={metrics["32"]["delta_tp"]}', flush=True)
        write(checkpoint, fold)
        report['folds'].append(fold)
        write(run / 'progress.json', report)
    common = set.intersection(*(set(f['models']) for f in report['folds']))
    summary = {family: {b: pooled(report['folds'], family, b) for b in ('16', '32', '48')}
               for family in sorted(common)}
    control_summary = {f: summary[f]['32'] for f in CONTROLS}
    main_control = max(CONTROLS, key=lambda f: (control_summary[f]['U'], control_summary[f]['worst_fold_U'], f))
    eligible = []
    for f in TYPED:
        if f not in summary:
            continue
        s = summary[f]['32']
        ok = (s['U'] > 0 and s['positive_folds'] >= 3
              and s['U'] > max(control_summary[c]['U'] for c in CONTROLS)
              and s['worst_fold_U'] >= control_summary[main_control]['worst_fold_U'])
        if ok:
            eligible.append(f)
    winner = max(eligible, key=lambda f: (summary[f]['32']['U'], summary[f]['32']['worst_fold_U'], f)) if eligible and not smoke else None
    report.update({'complete': not smoke and len(report['folds']) == 5, 'completed_at': now(),
        'summary': summary, 'controls': control_summary, 'main_control': main_control,
        'eligible_typed': eligible, 'winner': winner, 'gate_passed': bool(winner),
        'decision': 'build_candidates' if winner else 'stop_training_gate',
        'reason': 'typed family needs pooled U>0, > both controls, >=3 positive folds, worst-fold U >= main control'})
    write(run / 'model_comparison.json', report)
    if not smoke:
        write(out / 'model_comparison.json', report)
    if eligible and not smoke:
        all_orders = np.arange(len(t.records['train']))
        scores, test_scores = t.crossfit_nodes(all_orders)
        fit = t.actions('train', all_orders, scores)
        _, _, _, base_nodes = baseline()
        selected = np.array([(o['order_id'], a['rid']) in base_nodes for o in t.records['test'] for a in o['alarms']])
        test = t.actions('test', np.arange(546), test_scores, selected)
        family_preds = {}
        for fam in eligible:
            probs = []
            for param in sorted({f['models'][fam]['param'] for f in report['folds']}):
                p, models = t.fit_predict(fam, param, fit, test)
                temp = float(np.median([f['models'][fam]['temperature'] for f in report['folds']
                                        if f['models'][fam]['param'] == param]))
                probs.append(t152.physical(p, test['meta'], temp))
            family_preds[fam] = np.mean(probs, axis=0)
        candidates = []
        for j, a in enumerate(test['meta']):
            support = {'add': [0, 1], 'delete': [-1, 0], 'swap': [-1, 0, 1]}[a['kind']]
            model_probs = {fam: {str(k): float(family_preds[fam][j][k + 1]) for k in support} for fam in eligible}
            combined = {str(k): float(np.mean([family_preds[fam][j][k + 1] for fam in eligible])) for k in support}
            candidates.append({**a, 'probabilities': combined, 'model_probabilities': model_probs,
                               'source_model': '+'.join(eligible)})
        write(out / 'training_candidates.json', {'source_class': 'model_estimated',
            'report_sha256': t152.sha(out / 'model_comparison.json'), 'input_hashes': t.hashes,
            'families': eligible, 'candidates': candidates})
    print(json.dumps({'gate_passed': bool(winner), 'winner': winner, 'summary': summary,
                      'controls': control_summary}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    import sys
    with threadpool_limits(4):
        print(json.dumps(train(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.').resolve()),
                             ensure_ascii=False))
