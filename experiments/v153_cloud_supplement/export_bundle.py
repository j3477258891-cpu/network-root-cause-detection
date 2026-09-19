"""Export cached cross-fitted actions, never credentials or leaderboard labels."""
from pathlib import Path
import inspect
import json
import sys
import zipfile
import hashlib
import time
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'v153_complementary_repair_campaign'
sys.path.insert(0, str(FROZEN))
from training import Trainer
from bridge import c, t, protect_snapshot, verify_snapshot

def save_bundle(dest, bundle):
    np.savez_compressed(dest.with_suffix('.npz'), **{k: bundle[k] for k in ('x','y','prior','mask')})
    c.write(dest.with_suffix('.json'), {k:bundle[k] for k in ('meta','orders')})

def main():
    snapshot = protect_snapshot()
    deadline = c.read(FROZEN / 'training_window.json')['deadline_at']
    remaining = datetime.fromisoformat(deadline).timestamp() - time.time()
    assert remaining > 0, 'Original training window expired'
    out = HERE / 'bundle'
    out.mkdir(exist_ok=False)
    trainer = Trainer(HERE / 'export_cache', remaining / 3600)
    np.savez_compressed(out/'truth.npz', labels=trainer.y, ptr=trainer.arrays['train_alarm_ptr'], folds=trainer.folds)
    for outer in range(5):
        fit_orders = np.flatnonzero(trainer.folds != outer)
        valid_orders = np.flatnonzero(trainer.folds == outer)
        splits = list(t.group_splits(fit_orders, trainer.folds)) + [(fit_orders, valid_orders)]
        for level, (fit, valid) in enumerate(splits):
            assert not set(trainer.folds[fit]) & set(trainer.folds[valid])
            scores, _ = trainer.crossfit_nodes(fit, valid)
            for name, orders in [('fit',fit),('valid',valid)]:
                save_bundle(out/f'f{outer}_s{level}_{name}', trainer.actions('train',orders,scores))
        print('exported outer fold', outer, flush=True)
    orders = np.arange(len(trainer.records['train']))
    scores, test_scores = trainer.crossfit_nodes(orders)
    save_bundle(out/'full_fit', trainer.actions('train', orders, scores))
    _, _, _, nodes = c.baseline()
    selected = np.array([(o['order_id'],a['rid']) in nodes for o in trainer.records['test'] for a in o['alarms']])
    save_bundle(out/'full_test', trainer.actions('test', np.arange(546), test_scores, selected))
    # Mechanical source extraction preserves the exact existing scoring protocol.
    funcs = ('rows_for','probabilities','physical','selected_action_indices','score_actions','temperature','pooled')
    protocol = 'import numpy as np\nG=1044\nP0=1045\nTP0=969\n\n' + '\n\n'.join(inspect.getsource(getattr(t,f)) for f in funcs)
    (out/'protocol.py').write_text(protocol, encoding='utf-8')
    (out/'worker.py').write_bytes((HERE/'worker.py').read_bytes())
    c.write(out/'previous_comparison.json', c.read(FROZEN/'model_comparison.json'))
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    c.write(out/'manifest.json', {'deadline_at':deadline,'input_hashes':trainer.hashes,'files':hashes,
        'seed':20260908,'purpose':'seed-complement replication; not proven improvement',
        'training_labels':'training set only; no public test labels',
        'baseline_proxy_warning':'Cross-fitted proxy, not an honest offline reconstruction of public V149',
        'frozen_v153':'unchanged; no automatic candidate replacement or competition submission'})
    archive = HERE/'v153_cloud_supplement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p,'v153_cloud_supplement/'+p.name)
    verify_snapshot(snapshot)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)

if __name__ == '__main__': main()
