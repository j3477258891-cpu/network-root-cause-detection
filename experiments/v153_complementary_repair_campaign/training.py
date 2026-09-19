"""Reuse validated CatBoost; full-fit V38; optional locally preflighted TabPFNv2."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from bridge import HERE, PRIOR, c, t, action_key, protect_snapshot, verify_snapshot


class Trainer(t.Trainer):
    def node_predictions(self, orders):
        self.check_time()
        key = c.digest([self.signature, sorted(map(int, orders))])
        previous = PRIOR / 'training/node_cache' / (key + '.npz')
        if previous.exists():
            with np.load(previous, allow_pickle=False) as z:
                return z['train'], z['test']
        return super().node_predictions(orders)  # writes only the new campaign cache


def preflight(out):
    report = {'model_version': 'v2_explicit_not_package_default',
              'permission': 'user confirmed public pretrained models allowed; not independent rule verification',
              'status': 'not_run', 'local_only': True, 'checked_at': c.now()}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    report['gpu'] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    missing = [m for m in ('torch', 'tabpfn') if importlib.util.find_spec(m) is None]
    if missing:
        report.update(status='dependency_unavailable', missing=missing,
                      reason='Optional challenger unavailable in this runtime; do not alter V152 environment')
    else:
        import torch
        report['torch_version'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
        report['status'] = 'resource_check_passed' if torch.cuda.is_available() else 'cuda_unavailable'
    c.write(out / 'tabpfn_preflight.json', report)
    return report


def probabilities_to_actions(meta, p, family):
    result = []
    for a, row in zip(meta, p):
        support = {'add': (0, 1), 'delete': (-1, 0), 'swap': (-1, 0, 1)}[a['kind']]
        result.append({**a, 'probabilities': {str(k): float(row[k+1]) for k in support},
                       'source_model': family})
    return result


def fit_final(trainer, family, report, out):
    checkpoint = out / ('predictions_' + family + '.json')
    if checkpoint.exists():
        saved = c.read(checkpoint)
        c.require(saved['input_hashes'] == trainer.hashes, 'Final-model checkpoint source drift')
        return saved['candidates']
    all_orders = np.arange(len(trainer.records['train']))
    scores, test_scores = trainer.crossfit_nodes(all_orders)
    fit = trainer.actions('train', all_orders, scores)
    _, _, _, nodes = c.baseline()
    selected = np.array([(o['order_id'], a['rid']) in nodes
                         for o in trainer.records['test'] for a in o['alarms']])
    test = trainer.actions('test', np.arange(546), test_scores, selected)
    predictions = []
    for param in sorted({fold['models'][family]['param'] for fold in report['folds']}):
        print('full fit ' + family + ' param=' + str(param), flush=True)
        p, models = trainer.fit_predict(family, param, fit, test)
        temp = float(np.median([f['models'][family]['temperature'] for f in report['folds']
                                if f['models'][family]['param'] == param]))
        predictions.append(t.physical(p, test['meta'], temp))
        import joblib
        joblib.dump(models, out / 'training' / f'final_{family}_{param}.joblib')
    actions = probabilities_to_actions(test['meta'], np.mean(predictions, axis=0), family)
    c.write(checkpoint, {'input_hashes': trainer.hashes, 'candidates': actions,
                         'source_class': 'model_estimated_not_public_score'})
    return actions


def tab_worker(out):
    """Separate process, bounded by the controller and original deadline."""
    os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
    os.environ['TABPFN_MODEL_CACHE_DIR'] = str(out / 'tabpfn_weights')
    window = c.read(PRIOR / 'training_window.json')
    remaining = (datetime.fromisoformat(window['deadline_at']).timestamp()-time.time())/3600
    c.require(remaining > 0, 'Original training window expired')
    trainer = Trainer(out / 'tabpfn_training', remaining)
    report = {'folds': [], 'input_hashes': trainer.hashes, 'complete': False,
              'permission': 'user_confirmed', 'deadline_at': window['deadline_at']}
    # Preflight actual model download/fit/predict on training-only data, never test labels.
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cuda', n_estimators=1)
    model.fit(trainer.x['train'][:128, :32], trainer.y[:128])
    model.predict_proba(trainer.x['train'][128:132, :32])
    c.write(out / 'tabpfn_runtime_check.json', {'status': 'fit_predict_passed', 'model_version': 'v2'})
    for outer in range(5):
        dest = out / 'tabpfn_training' / f'fold_{outer}.json'
        if dest.exists():
            fold = c.read(dest)
            c.require(fold['input_hashes'] == trainer.hashes, 'TabPFN checkpoint changed')
            report['folds'].append(fold)
            continue
        fit_orders = np.flatnonzero(trainer.folds != outer)
        valid_orders = np.flatnonzero(trainer.folds == outer)
        inner = []
        for fit, valid in t.group_splits(fit_orders, trainer.folds):
            scores, _ = trainer.crossfit_nodes(fit, valid)
            fb, vb = trainer.actions('train', fit, scores), trainer.actions('train', valid, scores)
            p, _ = trainer.fit_predict('tabpfn_v2', 0, fb, vb)
            inner.append((p, vb))
        temp = t.temperature(np.concatenate([p for p, b in inner]),
                             np.concatenate([b['y'] for p, b in inner]),
                             [a for p, b in inner for a in b['meta']])
        scores, _ = trainer.crossfit_nodes(fit_orders, valid_orders)
        fb = trainer.actions('train', fit_orders, scores)
        vb = trainer.actions('train', valid_orders, scores)
        p, _ = trainer.fit_predict('tabpfn_v2', 0, fb, vb)
        p = t.physical(p, vb['meta'], temp)
        metrics = t.score_actions(vb, p, trainer.y, trainer.arrays['train_alarm_ptr'])
        fold = {'fold': outer, 'input_hashes': trainer.hashes,
                'models': {'tabpfn_v2': {'param': 0, 'temperature': temp, 'metrics': metrics}}}
        c.write(dest, fold)
        report['folds'].append(fold)
        c.write(out / 'tabpfn_training/progress.json', report)
    metric = t.pooled(report['folds'], 'tabpfn_v2', '32')
    report.update(complete=True, metric=metric,
                  eligible=metric['positive_folds'] >= 3 and metric['f1'] > metric['base_f1'])
    if report['eligible']:
        fit_final(trainer, 'tabpfn_v2', report, out)
    c.write(out / 'tabpfn_comparison.json', report)


def train(out):
    out.mkdir(parents=True, exist_ok=True)
    c.require(not (out / 'campaign.json').exists(), 'Frozen V153 cannot be retrained')
    if not (out / 'protected_sources.json').exists():
        c.write(out / 'protected_sources.json', protect_snapshot())
    snapshot = c.read(out / 'protected_sources.json')
    verify_snapshot(snapshot)
    original = c.read(PRIOR / 'model_comparison.json')
    c.require(original['complete'] and len(original['folds']) == 5, 'V152 five folds incomplete')
    c.require(original['input_hashes'] == t.source_hashes(), 'V152 sources changed')
    window = c.read(PRIOR / 'training_window.json')
    c.write(out / 'training_window.json', window)
    remaining = (datetime.fromisoformat(window['deadline_at']).timestamp()-time.time())/3600
    c.require(remaining > 0, 'Original 48-hour deadline expired; cannot reset it')
    trainer = Trainer(out / 'training', remaining)
    raw_cat = c.read(PRIOR / 'training_candidates.json')
    c.require(raw_cat['report_sha256'] == c.sha(PRIOR / 'model_comparison.json'), 'CatBoost provenance mismatch')
    cat = raw_cat['candidates']
    v38 = fit_final(trainer, 'v38_control', original, out)
    c.require([action_key(a) for a in v38] == [action_key(a) for a in cat], 'Model action universes differ')
    families = {'v38_control': v38, 'catboost': cat}
    pf = preflight(out)
    if pf['status'] == 'resource_check_passed':
        timeout = min(3600, max(1, int(datetime.fromisoformat(window['deadline_at']).timestamp()-time.time())))
        try:
            with (out / 'tabpfn_worker.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-B', str(Path(__file__)), '--tab-worker', str(out)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            challenger = c.read(out / 'tabpfn_comparison.json')
            pf.update(status='validated' if challenger['eligible'] else 'validation_gate_failed')
            if challenger['eligible']:
                families['tabpfn_v2'] = c.read(out / 'predictions_tabpfn_v2.json')['candidates']
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            pf.update(status='runtime_failed_or_timed_out', reason=str(exc))
    c.write(out / 'tabpfn_preflight.json', pf)
    metrics = {f: original['summary'][f] for f in ('v38_control', 'catboost')}
    for f in metrics:
        metric = metrics[f]['32']
        c.require(metric['positive_folds'] >= 3 and metric['f1'] > metric['base_f1'], 'Model gate failed: ' + f)
    result = {'complete': True, 'families': list(families), 'source_class': 'nested_group_oof_not_public',
              'metrics': metrics, 'input_hashes': trainer.hashes, 'tabpfn': pf,
              'source_report': str(PRIOR / 'model_comparison.json'),
              'source_report_sha256': c.sha(PRIOR / 'model_comparison.json'),
              'deadline_at': window['deadline_at'], 'completed_at': c.now(),
              'warning': 'OOF baseline .857374 is not an offline reconstruction of public .927717'}
    c.write(out / 'model_comparison.json', result)
    c.write(out / 'training_candidates.json', {'families': families,
            'report_sha256': c.sha(out / 'model_comparison.json'), 'source_class': 'model_estimated'})
    verify_snapshot(snapshot)
    return {**result, 'action_counts': {f: len(a) for f, a in families.items()}}


if __name__ == '__main__':
    with threadpool_limits(4):
        if sys.argv[1:2] == ['--tab-worker']:
            tab_worker(Path(sys.argv[2]))
        else:
            print(json.dumps(train(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE), ensure_ascii=False))
