"""Nested group validation of action models; no leaderboard labels in training."""
from __future__ import annotations

import gzip
import importlib.metadata
import importlib.util
import json
import os
import time
import itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
from threadpoolctl import threadpool_limits

from core import ROOT, EXP, G, P0, TP0, baseline, digest, read, write, sha, now, require

SEED = 20260906
DATA = EXP/'v25_semantic_router/cloud_dataset/v25_semantic_router.npz'
RECORDS = EXP/'v25_semantic_router/cloud_dataset/semantic_records.json.gz'
RAW_TRAIN = EXP/'v16/v16_train_features.npy'
RAW_TEST = EXP/'v16/v16_test_features.npy'
KINDS = ('add', 'delete', 'swap')
GRID = {'catboost': [4, 6], 'v38_control': [0], 'v30_control': [0], 'tabpfn_v2': [0]}


def source_hashes():
    return {str(p): sha(p) for p in [DATA, RECORDS, RAW_TRAIN, RAW_TEST, Path(__file__), Path(__file__).with_name('core.py')]}


def dataset():
    with gzip.open(RECORDS, 'rt', encoding='utf-8') as handle:
        records = json.load(handle)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    vectors = {}
    hasher = HashingVectorizer(analyzer='char', ngram_range=(2,4), n_features=64, alternate_sign=False)
    categorical_hasher = FeatureHasher(n_features=128,input_type='dict',alternate_sign=False)
    # These source fields are available at inference; never include is_root.
    allowed = ('title','reason','device_type','board_type','cause','radio','deployment','timeline','target_summary')
    for split, file in [('train', RAW_TRAIN), ('test', RAW_TEST)]:
        text = [' '.join(str(a.get(k,'')) for k in allowed) for o in records[split] for a in o['alarms']]
        raw = np.load(file, allow_pickle=False)
        require(len(text) == len(raw) == arrays[f'{split}_alarm_ptr'][-1], 'Feature/node alignment mismatch')
        # Raw V16 graph/time/categorical inputs; exclude every saved supervised score.
        categorical = [{f'{k}={a.get(k,"")}':1 for k in ('title','device_type','board_type','cause','radio','deployment','label')}
                       for o in records[split] for a in o['alarms']]
        vectors[split] = np.column_stack([np.nan_to_num(raw),categorical_hasher.transform(categorical).toarray(),
                                         hasher.transform(text).toarray()]).astype('float32')
    labels = np.array([a['is_root'] for o in records['train'] for a in o['alarms']], dtype='int8')
    require(np.array_equal(labels, arrays['train_labels']), 'Semantic labels differ from node array alignment')
    require(all(a.get('is_root') is None for o in records['test'] for a in o['alarms']), 'Unexpected test labels')
    require(len(records['train']) == 1634 and len(records['test']) == 546, 'Dataset order-count drift')
    folds = arrays['train_connected_folds'].astype(int)
    require(set(folds) == set(range(5)), 'Five connected folds required')
    station_owner = {}
    signature_owner = {}
    for o, f in zip(records['train'], folds):
        for station in o['station_ids']:
            require(station_owner.setdefault(station, int(f)) == f, 'Shared station crosses folds')
        sig = json.dumps(o['signature'], sort_keys=True)
        require(signature_owner.setdefault(sig, int(f)) == f, 'Exact template crosses folds')
    return records, arrays, vectors, labels, folds


def rows_for(orders, ptr):
    return np.concatenate([np.arange(ptr[i], ptr[i+1]) for i in orders]) if len(orders) else np.array([], dtype=int)


def group_splits(orders, folds, n=2):
    orders = np.asarray(orders, dtype=int)
    unique = np.unique(folds[orders])
    require(len(unique) >= 2, 'Insufficient independent groups for cross-fitting')
    require(n == 2, 'This protocol specifies two inner folds')
    # A size-only GroupKFold can put the giant component against three bins,
    # leaving a one-bin inner training set that cannot itself be cross-fitted.
    # Balance the number of independent bins first, then their order counts.
    counts = {int(g): int(sum(folds[orders] == g)) for g in unique}
    choices = itertools.combinations(map(int, unique), len(unique)//2)
    left = min(choices, key=lambda gs: (abs(2*sum(counts[g] for g in gs)-len(orders)), gs))
    is_left = np.isin(folds[orders], left)
    for mask in (is_left, ~is_left):
        fit, valid = orders[~mask], orders[mask]
        require(not set(folds[fit]) & set(folds[valid]), 'Nested group leakage')
        yield fit, valid


def probabilities(model, x):
    result = np.zeros((len(x),3), dtype=float)
    if len(x):
        predicted = model.predict_proba(x)
        for j, label in enumerate(model.classes_):
            result[:, int(label)+1] = predicted[:,j]
    return result


def physical(p, meta, temperature=1.0):
    p = np.maximum(p, 1e-9)**(1/temperature)
    for j, a in enumerate(meta):
        if a['kind'] == 'add': p[j,0] = 0
        if a['kind'] == 'delete': p[j,2] = 0
    return p / p.sum(axis=1, keepdims=True) if len(p) else p


class Trainer:
    def __init__(self, out, deadline_hours=48, smoke=False):
        self.out = Path(out)
        self.cache = self.out/'node_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.records, self.arrays, self.x, self.y, self.folds = dataset()
        self.hashes = source_hashes()
        self.signature = digest({'sources': self.hashes, 'smoke': smoke, 'seed': SEED, 'node_version': 1})
        self.deadline = time.time()+deadline_hours*3600
        self.smoke = smoke
        self.trace = []

    def check_time(self):
        require(time.time() < self.deadline, 'Training deadline reached; partial OOF cannot pass the gate')

    def node_predictions(self, fit_orders):
        """Cache scores, not in-memory forests; use held-out rows only downstream."""
        self.check_time()
        fit_orders = sorted(map(int, fit_orders))
        key = digest([self.signature, fit_orders])
        dest = self.cache/(key+'.npz')
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                return z['train'], z['test']
        rows = rows_for(fit_orders, self.arrays['train_alarm_ptr'])
        et = ExtraTreesClassifier(n_estimators=24 if self.smoke else 160, min_samples_leaf=3,
            max_features=.65, class_weight='balanced', random_state=SEED, n_jobs=4)
        hgb = HistGradientBoostingClassifier(max_iter=20 if self.smoke else 100,
            learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=18, l2_regularization=1,
            early_stopping=False, random_state=SEED)
        outputs = {'train': [], 'test': []}
        for model in (et, hgb):
            model.fit(self.x['train'][rows], self.y[rows])
            for split in outputs:
                outputs[split].append(model.predict_proba(self.x[split])[:,list(model.classes_).index(1)])
        tr, te = np.column_stack(outputs['train']), np.column_stack(outputs['test'])
        np.savez_compressed(dest, train=tr, test=te)
        self.trace.append({'node_fit_orders': fit_orders, 'cache': key})
        return tr, te

    def crossfit_nodes(self, fit_orders, eval_orders=None):
        tr = np.zeros((len(self.y),2))
        for fit, valid in group_splits(fit_orders, self.folds):
            scores, _ = self.node_predictions(fit)
            rows = rows_for(valid, self.arrays['train_alarm_ptr'])
            tr[rows] = scores[rows]
        full, te = self.node_predictions(fit_orders)
        if eval_orders is not None:
            rows = rows_for(eval_orders, self.arrays['train_alarm_ptr'])
            tr[rows] = full[rows]
        return tr, te

    def masks(self, split, orders, scores):
        ptr = self.arrays[f'{split}_alarm_ptr']
        mask = np.zeros(len(scores), dtype=bool)
        optional = []
        for oi in orders:
            rows = np.arange(ptr[oi], ptr[oi+1])
            ranked = rows[np.argsort(-scores[rows].mean(1), kind='stable')][:8]
            mask[ranked[0]] = True
            optional.extend(ranked[1:])
        # Fixed public prediction density, not the validation labels/true order counts.
        target = min(len(optional)+len(orders), max(len(orders), round(P0*len(orders)/546)))
        optional.sort(key=lambda i: (-float(scores[i].mean()), int(i)))
        mask[optional[:target-len(orders)]] = True
        return mask

    def actions(self, split, orders, scores, selected=None):
        ptr, records, x = self.arrays[f'{split}_alarm_ptr'], self.records[split], self.x[split]
        selected = self.masks(split, orders, scores) if selected is None else selected
        compact = np.column_stack([x[:,:97], x[:,-192:]])
        features, meta, labels, priors = [], [], [], []
        for oi in orders:
            oi = int(oi)
            rows = np.arange(ptr[oi], ptr[oi+1])
            inside = sorted(rows[selected[rows]], key=lambda r:(float(scores[r].mean()),int(r)))[:2]
            outside = sorted(rows[~selected[rows]], key=lambda r:(-float(scores[r].mean()),int(r)))[:2]
            count = int(selected[rows].sum())
            pairs = []
            if count < 8: pairs += [(None, r) for r in outside]
            if count > 1: pairs += [(r, None) for r in inside]
            pairs += [(r,a) for r in inside for a in outside]
            for remove, add in pairs:
                dp = int(add is not None)-int(remove is not None)
                kind = 'swap' if add is not None and remove is not None else 'add' if add is not None else 'delete'
                ar = records[oi]['alarms'][int(add-ptr[oi])] if add is not None else None
                rr = records[oi]['alarms'][int(remove-ptr[oi])] if remove is not None else None
                av = compact[add] if add is not None else np.zeros(compact.shape[1])
                rv = compact[remove] if remove is not None else np.zeros(compact.shape[1])
                pa = float(scores[add].mean()) if add is not None else 0.
                pr = float(scores[remove].mean()) if remove is not None else 0.
                features.append(np.r_[av,rv,av-rv,[KINDS.index(kind),dp,count,len(rows),pa,pr,pa-pr,
                    scores[rows].mean(),scores[rows].std(),float(bool(ar) and ar.get('label')=='TargetAlarm'),
                    float(bool(rr) and rr.get('label')=='TargetAlarm')]])
                m = {'order_index':oi,'order_id':records[oi]['order_id'],'kind':kind,'delta_p':dp,
                     'add_rid':ar['rid'] if ar else None,'remove_rid':rr['rid'] if rr else None}
                if ar:
                    source = ar['source']
                    m['node'] = {'@rid':ar['rid'], **{k:source.get(k,'') for k in ('title','location','reason')}}
                meta.append(m)
                labels.append((int(self.y[add]) if add is not None else 0)-(int(self.y[remove]) if remove is not None else 0) if split=='train' else 0)
                # V30-style direct node-score control; pair independence is an explicit model assumption.
                priors.append([pr*(1-pa), pa*pr+(1-pa)*(1-pr), pa*(1-pr)])
        features = np.nan_to_num(np.asarray(features,dtype='float32'))
        return {'x':features,'meta':meta,'y':np.asarray(labels,dtype=int),
                'prior':physical(np.asarray(priors),meta),'mask':selected,'orders':list(map(int,orders))}

    def fit_predict(self, family, param, fit, valid, seed=SEED):
        self.check_time()
        require(len(fit['meta']) and len(valid['meta']), 'No boundary actions in fold')
        if family == 'v30_control':
            return valid['prior'].copy(), None
        if len(np.unique(fit['y'])) < 2:
            p = np.full((len(valid['meta']),3), 1e-6)
            p[:,int(fit['y'][0])+1] = 1
            return physical(p,valid['meta']), None
        if family == 'catboost':
            from catboost import CatBoostClassifier
            models = [CatBoostClassifier(iterations=30 if self.smoke else 240, depth=param,
                learning_rate=.05, l2_leaf_reg=5, loss_function='MultiClass', thread_count=4,
                random_seed=seed, verbose=False, allow_writing_files=False)]
        elif family == 'v38_control':
            models = [ExtraTreesClassifier(n_estimators=32 if self.smoke else 800, min_samples_leaf=4,
                max_features=.45,class_weight='balanced',n_jobs=4,random_state=seed),
                HistGradientBoostingClassifier(max_iter=20 if self.smoke else 260,learning_rate=.045,
                max_leaf_nodes=15,min_samples_leaf=24,l2_regularization=1.5,early_stopping=False,random_state=seed)]
        else:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            import torch
            require(torch.cuda.is_available(), 'TabPFN requires a working local CUDA runtime for this campaign')
            os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
            selector = SelectKBest(f_classif,k=min(128,fit['x'].shape[1])).fit(fit['x'],fit['y'])
            rng = np.random.default_rng(seed)
            selected = rng.permutation(len(fit['x']))[:4096]
            model = TabPFNClassifier.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=1)
            model.fit(selector.transform(fit['x'][selected]),fit['y'][selected])
            return physical(probabilities(model,selector.transform(valid['x'])),valid['meta']), (selector,model)
        results = []
        for model in models:
            model.fit(fit['x'],fit['y'])
            results.append(probabilities(model,valid['x']))
        return physical(np.mean(results,axis=0),valid['meta']), models


def selected_action_indices(bundle, probabilities_):
    """Freeze decisions without reading ANY validation labels or validation F1."""
    expected = probabilities_[:,2]-probabilities_[:,0]
    reference = 2*TP0/(G+P0)
    gains = [2*(TP0+expected[i])/(G+P0+a['delta_p'])-reference
             for i,a in enumerate(bundle['meta'])]
    ranked = sorted(range(len(expected)), key=lambda i:(-gains[i],
        bundle['meta'][i]['order_id'],bundle['meta'][i].get('add_rid') or '',bundle['meta'][i].get('remove_rid') or ''))
    unique, used = [], set()
    for i in ranked:
        oid = bundle['meta'][i]['order_id']
        if oid in used or gains[i] <= 0: continue
        used.add(oid); unique.append(i)
    return unique


def score_actions(bundle, probabilities_, labels, ptr, budgets=(16,32,64)):
    unique = selected_action_indices(bundle, probabilities_)
    orders = bundle['orders']
    rows = rows_for(orders,ptr)
    p0 = int(bundle['mask'][rows].sum())
    tp0 = int(labels[rows][bundle['mask'][rows]].sum())
    g = int(labels[rows].sum())
    before = 2*tp0/(g+p0)
    output = {}
    for budget in budgets:
        selected = unique[:max(1,round(budget*len(orders)/546))]
        dt = int(bundle['y'][selected].sum())
        dp = sum(bundle['meta'][i]['delta_p'] for i in selected)
        output[str(budget)] = {'orders':len(orders),'actions':len(selected),'g':g,'base_p':p0,'base_tp':tp0,
            'delta_p':dp,'delta_tp':dt,'base_f1':before,'f1':2*(tp0+dt)/(g+p0+dp),
            'f1_gain':2*(tp0+dt)/(g+p0+dp)-before,
            'positive_tp_actions':int(sum(bundle['y'][i]>0 for i in selected)),
            'negative_tp_actions':int(sum(bundle['y'][i]<0 for i in selected)),
            'selected_action_indices':selected}
    return output


def temperature(p, y, meta):
    best = (float('inf'),1.)
    for value in (.5,.75,1.,1.5,2.,3.):
        q = physical(p,meta,value)
        loss = float(-np.log(np.maximum(q[np.arange(len(y)),y+1],1e-12)).mean())
        best = min(best,(loss,value))
    return best[1]


def pooled(folds, family, budget='64'):
    items = [f['models'][family]['metrics'][budget] for f in folds]
    p0,tp0,g = (sum(x[k] for x in items) for k in ('base_p','base_tp','g'))
    dp,dt = sum(x['delta_p'] for x in items),sum(x['delta_tp'] for x in items)
    return {'f1':2*(tp0+dt)/(g+p0+dp),'base_f1':2*tp0/(g+p0),
        'delta_tp':dt,'delta_p':dp,'positive_folds':sum(x['f1_gain']>0 for x in items),
        'worst_fold_gain':min(x['f1_gain'] for x in items),'actions':sum(x['actions'] for x in items)}


def train(out, max_hours=48, smoke=False, rules_evidence=None):
    out = Path(out)
    require(not (out/'campaign.json').exists(), 'Frozen campaign cannot be retrained')
    run = out/('smoke_training' if smoke else 'training')
    run.mkdir(parents=True,exist_ok=True)
    require(0 < max_hours <= 48, 'Training window must be positive and at most 48 hours')
    window_path = out/('smoke_training_window.json' if smoke else 'training_window.json')
    if window_path.exists():
        window = read(window_path)
    else:
        start = datetime.fromisoformat(now())
        window = {'started_at':start.isoformat(), 'deadline_at':(start+timedelta(hours=max_hours)).isoformat()}
        write(window_path, window)
    deadline = datetime.fromisoformat(window['deadline_at']).timestamp()
    remaining = min(max_hours, (deadline-time.time())/3600)
    require(remaining > 0, 'Original training deadline expired; do not silently reset the clock')
    t = Trainer(run,remaining,smoke)
    t.deadline = min(t.deadline, deadline)
    families = ['v30_control','v38_control','catboost']
    tab_status = {'status':'not_run','reason':'competition permission for external pretrained weights not found; supply --tabpfn-rules-evidence'}
    if rules_evidence:
        require(Path(rules_evidence).is_file(), 'Rules evidence must be an existing reviewed document')
        if importlib.util.find_spec('tabpfn') and importlib.util.find_spec('torch'):
            families.append('tabpfn_v2')
            tab_status = {'status':'requested','rules_evidence':str(rules_evidence),'sha256':sha(rules_evidence)}
        else:
            tab_status = {'status':'not_run','reason':'TabPFN/PyTorch CUDA dependencies unavailable','rules_evidence':str(rules_evidence)}
    report = {'version':2,'source_class':'nested_group_oof','started_at':window['started_at'],
        'resumed_at':now(),'deadline_at':window['deadline_at'],'complete':False,
        'protocol':{'outer_folds':5,'inner_folds':2,'node_crossfit_folds':2,'primary_budget_per_546_orders':64,
            'features':'raw V16 graph/time + fixed text hashing; no saved supervised scores',
            'controls':'V30-style node ET/HGB and V38-style retrained action ET/HGB, not unchanged historical artifacts',
            'base_proxy':'cross-fitted node ensemble at V149 prediction density; NOT an offline reproduction of its corrected labels',
            'grouping':'existing station/template connected components; shared-station and exact-template checks passed',
            'inner_partition':'equal independent-bin counts first, then order-count balance; never split a connected bin',
            'action_selection':'fixed public V149 reference (G=1044,P=1045,TP=969), no validation labels in ranking or eligibility',
            'node_model':{'et_trees':160,'hgb_iterations':100},'catboost':{'depth_grid':[4,6],'iterations':240}},
        'input_hashes':t.hashes,'tabpfn':tab_status,'smoke_only':smoke,'folds':[],
        'warnings':['OOF model selection is not a calibrated probability of reaching 0.94',
                    'Legacy best public champion is not reconstructible as an honest train-set oracle']}
    write(run/'progress.json',report)
    for outer in range(1 if smoke else 5):
        checkpoint = run/f'fold_{outer}.json'
        if checkpoint.exists():
            saved = read(checkpoint)
            require(saved['signature']==t.signature,'Checkpoint source/protocol drift; use a new output directory')
            report['folds'].append(saved)
            write(run/'progress.json',report)
            continue
        fit_orders,valid_orders = np.flatnonzero(t.folds!=outer),np.flatnonzero(t.folds==outer)
        print(f'outer fold {outer+1}/5: fit {len(fit_orders)}, validate {len(valid_orders)} orders',flush=True)
        inner_results = {family:{param:[] for param in GRID[family]} for family in families}
        for inner,(fit,valid) in enumerate(group_splits(fit_orders,t.folds)):
            scores,_ = t.crossfit_nodes(fit,valid)
            train_bundle=t.actions('train',fit,scores)
            valid_bundle=t.actions('train',valid,scores)
            for family in list(families):
                for param in GRID[family]:
                    try:
                        p,_=t.fit_predict(family,param,train_bundle,valid_bundle)
                    except Exception as exc:
                        if family!='tabpfn_v2': raise
                        report['tabpfn']={'status':'failed_smoke','reason':repr(exc)}
                        families.remove(family)
                        break
                    metrics=score_actions(valid_bundle,p,t.y,t.arrays['train_alarm_ptr'])
                    inner_results[family][param].append((p,valid_bundle,metrics['64']['f1_gain']))
            print(f'  inner {inner+1}/2 complete',flush=True)
        scores,_=t.crossfit_nodes(fit_orders,valid_orders)
        train_bundle=t.actions('train',fit_orders,scores)
        valid_bundle=t.actions('train',valid_orders,scores)
        fold={'fold':outer,'signature':t.signature,'models':{},'fit_orders':list(map(int,fit_orders)),
              'valid_orders':list(map(int,valid_orders))}
        for family in families:
            param=max(GRID[family],key=lambda p:(np.mean([x[2] for x in inner_results[family][p]]),-p))
            entries=inner_results[family][param]
            inner_p=np.concatenate([x[0] for x in entries])
            inner_y=np.concatenate([x[1]['y'] for x in entries])
            inner_meta=[m for x in entries for m in x[1]['meta']]
            temp=temperature(inner_p,inner_y,inner_meta)
            p,_=t.fit_predict(family,param,train_bundle,valid_bundle)
            p=physical(p,valid_bundle['meta'],temp)
            metrics=score_actions(valid_bundle,p,t.y,t.arrays['train_alarm_ptr'])
            fold['models'][family]={'param':param,'temperature':temp,'metrics':metrics}
            np.savez_compressed(run/f'fold_{outer}_{family}.npz',probabilities=p,labels=valid_bundle['y'])
            print(f'  {family}: deltaTP={metrics["64"]["delta_tp"]}, F1gain={metrics["64"]["f1_gain"]:.6f}',flush=True)
        write(run/f'fold_{outer}_actions.json',valid_bundle['meta'])
        write(checkpoint,fold)
        report['folds'].append(fold)
        write(run/'progress.json',report)
    common=set.intersection(*(set(f['models']) for f in report['folds']))
    summary={family:{b:pooled(report['folds'],family,b) for b in ('16','32','64')} for family in sorted(common)}
    controls=max(summary[f]['64']['f1'] for f in ('v30_control','v38_control'))
    eligible=[f for f in common-set(('v30_control','v38_control')) if summary[f]['64']['f1']>controls
              and summary[f]['64']['f1']>summary[f]['64']['base_f1'] and summary[f]['64']['positive_folds']>=3]
    winner=max(eligible,key=lambda f:(summary[f]['64']['f1'],summary[f]['64']['worst_fold_gain'],f)) if eligible and not smoke else None
    report.update({'complete':not smoke and len(report['folds'])==5,'completed_at':now(),'summary':summary,
        'winner':winner,'gate_passed':bool(winner),'decision':'build_candidates' if winner else 'stop_training_gate',
        'reason':'requires positive pooled gain, >=3 positive folds, and higher primary-budget F1 than BOTH controls'})
    write(run/'model_comparison.json',report)
    write(run/'node_fit_trace.json',t.trace)
    if not smoke:
        write(out/'model_comparison.json',report)
    if winner:
        all_orders=np.arange(len(t.records['train']))
        scores,test_scores=t.crossfit_nodes(all_orders)
        fit=t.actions('train',all_orders,scores)
        _,_,_,base_nodes=baseline()
        selected=np.array([(o['order_id'],a['rid']) in base_nodes for o in t.records['test'] for a in o['alarms']])
        test=t.actions('test',np.arange(546),test_scores,selected)
        probs=[]
        for param in sorted({f['models'][winner]['param'] for f in report['folds']}):
            p,models=t.fit_predict(winner,param,fit,test)
            temp=float(np.median([f['models'][winner]['temperature'] for f in report['folds'] if f['models'][winner]['param']==param]))
            probs.append(physical(p,test['meta'],temp))
            import joblib
            joblib.dump(models,run/f'final_{winner}_{param}.joblib')
        predicted=np.mean(probs,axis=0)
        candidates=[]
        for a,p in zip(test['meta'],predicted):
            support={'add':[0,1],'delete':[-1,0],'swap':[-1,0,1]}[a['kind']]
            a={**a,'probabilities':{str(k):float(p[k+1]) for k in support},'source_model':winner}
            candidates.append(a)
        write(out/'training_candidates.json',{'source_class':'model_estimated','report_sha256':sha(out/'model_comparison.json'),
            'input_hashes':t.hashes,'candidates':candidates})
    print(json.dumps({'gate_passed':bool(winner),'winner':winner,'summary':summary,'tabpfn':report['tabpfn']},ensure_ascii=False),flush=True)
    return report
