def actions(self, split, orders, scores, selected=None):
    ptr, records, x = self.arrays[f'{split}_alarm_ptr'], self.records[split], self.x[split]
    selected = self.masks(split, orders, scores) if selected is None else selected
    compact = np.column_stack([x[:,:97], x[:,-192:]])
    features, meta, labels, priors = [], [], [], []
    for oi in orders:
        oi = int(oi)
        rows = np.arange(ptr[oi], ptr[oi+1])
        inside = sorted(rows[selected[rows]], key=lambda r:(float(scores[r].mean()),int(r)))[:4]
        outside = sorted(rows[~selected[rows]], key=lambda r:(-float(scores[r].mean()),int(r)))[:4]
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
