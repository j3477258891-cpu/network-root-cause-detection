import numpy as np
G=1044
P0=1045
TP0=969

def rows_for(orders, ptr):
    return np.concatenate([np.arange(ptr[i], ptr[i+1]) for i in orders]) if len(orders) else np.array([], dtype=int)


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
