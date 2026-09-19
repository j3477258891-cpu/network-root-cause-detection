"""Full-list screening and small disjoint count groups; only measured merges are final."""
from __future__ import annotations
import itertools, math
import numpy as np
from common import *
R=module('_v158_research',HERE/'research.py')
outcome_filter,convolve_counts,FAMILIES=R.outcome_filter,R.convolve_counts,R.FAMILIES

def screen(c,base,progress=None):
    source=read(EXP/'v153_complementary_repair_campaign/training_candidates.json')
    require(source['source_class']=='model_estimated','Unexpected prior class')
    families=source['families'];maps={f:{key(a):a for a in families[f]} for f in FAMILIES}
    require(set(maps[FAMILIES[0]])==set(maps[FAMILIES[1]]),'Incomplete source coverage')
    _,roots,nodes=csv_nodes(base['file'])
    _,oldroots,oldnodes=csv_nodes(EXP/'v149_online_calibrated_addition_campaign/v149_final_k08_from_v148.csv')
    blocked=legacy.initial_exclusions(oldnodes)
    old_orders={a['order_id'] for a in c.old.e.cat.values()}
    legal=[];rejected=[]
    for n,k in enumerate(sorted(maps[FAMILIES[0]])):
        c.check_deadline();a=dict(maps[FAMILIES[0]][k]);oid=a['order_id']
        reason=None
        if oid==legacy.BLOCKED_ORDER:reason='historical_dd294_exclusion'
        elif oid in old_orders:reason='old_decoded_pool_order_no_duplicate_gain'
        elif roots[oid]!=oldroots[oid]:reason='cached_model_baseline_context_changed'
        elif any((oid,a[f]) in blocked for f in ('add_rid','remove_rid') if a.get(f)):reason='historical_excluded_node_or_risk_pool'
        else:
            try:apply_actions(base['file'],[a])
            except ValueError:reason='not_legal_against_current_baseline'
        if reason:rejected.append({'key':k,'reason':reason});continue
        lo,hi=c.eq.bound([a])
        # Skip outcomes with no possible helpful F1 change. In particular,
        # fixed-zero additions and nonpositive swaps cannot become positive by a prior.
        if (a['kind']=='add' and hi==0) or (a['kind']=='swap' and hi<=0) or (a['kind']=='delete' and hi<0):
            rejected.append({'key':k,'reason':'no_feasible_beneficial_outcome','bounds':[lo,hi]});continue
        modelp={f:outcome_filter([maps[f][k]['probabilities'].get(str(v),0) for v in (-1,0,1)],lo,hi) for f in FAMILIES}
        p=np.mean(list(modelp.values()),axis=0);u=float(2*(p[2]-p[0])-TARGET*a['delta_p'])
        a.update(delta_tp_bounds={'min':lo,'max':hi},expected_u=u,
            probabilities={str(v):float(p[v+1]) for v in (-1,0,1)},
            model_probabilities={f:{str(v):float(q[v+1]) for v in (-1,0,1)} for f,q in modelp.items()})
        if u<=0:rejected.append({'key':k,'reason':'nonpositive_conditional_utility','expected_u':u});continue
        legal.append(a)
        if progress and n%200==0:progress(n,len(legal))
    selected=[];used=set()
    for a in sorted(legal,key=lambda a:(-a['expected_u'],key(a))):
        if a['order_id'] in used:continue
        used.add(a['order_id']);a['candidate_id']=len(selected)+1;selected.append(a)
        if len(selected)==48:break
    groups=[{'group_id':f'G{i//12+1:02d}','ids':[a['candidate_id'] for a in selected[i:i+12]]} for i in range(0,len(selected),12)]
    return dict(candidates=selected,groups=groups,rejected=rejected,raw_actions=len(maps[FAMILIES[0]]),
        positive_legal_actions=len(legal),selected_orders=len(used),
        ranking='historical outcome bounds then fixed equal distribution mixture target U, order/remove/add tie',
        probabilities_are_uncalibrated=True,baseline=base)

def preflight(c):
    """Research-only screening before safe feedback; never registers a CSV."""
    c.check_deadline()
    cat=screen(c,c.cfg['baseline'],lambda n,k:print('V158 equation screening',n,'legal',k,flush=True))
    oracle=c.eq.oracle(cat['candidates'],972,1045)
    cat['provisional_after_safe_merge_oracle']=oracle
    cat['source_class']='provisional_research_not_submission_pool'
    write(c.out/'research/preflight_catalog.json',cat)
    return dict(raw_actions=cat['raw_actions'],positive_legal_actions=cat['positive_legal_actions'],
                selected_orders=cat['selected_orders'],oracle=oracle,not_submission_eligible=True)

def build_pool(c):
    require(c.safe_baseline is not None,'Safe final must first receive its actual expected online score')
    require(c.research_gate().get('complete') and c.research_gate().get('passed'),'Research gate did not pass')
    p=c.out/'candidate_catalog.json'
    if p.exists():return read(c.out/'pool_gate.json')
    c.check_deadline();cat=screen(c,c.safe_baseline)
    require(0<len(cat['candidates'])<=48 and len(cat['groups'])<=4,'Empty/oversized candidate pool')
    oracle=c.eq.oracle(cat['candidates'],c.safe_baseline['tp'],c.safe_baseline['predictions'])
    gate=dict(oracle=oracle,passed=oracle['f1_upper']>=TARGET,baseline=c.safe_baseline,
              research_sha256=sha(c.out/'research/results.json'),source_class='pool_bound_not_score')
    if gate['passed']:
        cat.update(research_sha256=gate['research_sha256'],system_observations_sha256=digest(c.observations),frozen_at=now())
        write(p,cat);gate['catalog_sha256']=sha(p)
    gate['decision']='eligible_for_first_group' if gate['passed'] else 'stop_target_pool_keep_remaining_attempts'
    write(c.out/'pool_gate.json',gate);return gate

def measured_leaves(c,cat):
    leaves=[];split_used=0;measured=[]
    for r in c.ledger['records']:
        if r['status']!='accepted':continue
        e=c.entries[r['probe_id']]
        if e['phase']!='research' or e['kind']=='final':continue
        ids=tuple(sorted(e['ids']));value=r['tp']-c.safe_baseline['tp']
        if ids in measured:continue
        measured.append(ids)
        if e['kind']=='group':
            require(not any(set(ids)&set(x['ids']) for x in leaves),'Overlapping independent groups')
            leaves.append({'ids':ids,'dt':value})
        else:
            split_used+=1
            parent=next((x for x in leaves if set(ids)<set(x['ids'])),None)
            require(parent is not None,'Split has no measured parent')
            leaves.remove(parent);leaves += [{'ids':ids,'dt':value},
                {'ids':tuple(sorted(set(parent['ids'])-set(ids))),'dt':parent['dt']-value}]
    return leaves,split_used,measured

def best_merge(leaves,byid,base):
    best=dict(ids=[],tp=base['tp'],predictions=base['predictions'],f1=f1(base['tp'],base['predictions']))
    for bits in itertools.product((0,1),repeat=len(leaves)):
        ids=sorted(i for take,leaf in zip(bits,leaves) if take for i in leaf['ids'])
        dt=sum(leaf['dt'] for take,leaf in zip(bits,leaves) if take)
        tp=base['tp']+dt;p=base['predictions']+sum(byid[i]['delta_p'] for i in ids);value=f1(tp,p)
        if (value,-len(ids))>(best['f1'],-len(best['ids'])):best=dict(ids=ids,tp=tp,predictions=p,f1=value)
    return best

def recommendation(c):
    gate=read(c.out/'pool_gate.json');require(gate['passed'],'Pool gate failed')
    require(sha(c.out/'candidate_catalog.json')==gate['catalog_sha256'],'Frozen pool changed')
    require(sha(c.out/'research/results.json')==gate['research_sha256'],'Research result changed')
    cat=read(c.out/'candidate_catalog.json');byid={a['candidate_id']:a for a in cat['candidates']}
    leaves,splits,measured=measured_leaves(c,cat);best=best_merge(leaves,byid,c.safe_baseline)
    bound=c.eq.oracle(cat['candidates'],c.safe_baseline['tp'],c.safe_baseline['predictions'])
    result=dict(pool_upper_not_forecast=bound['f1_upper'],best_count_determined_merge=best,
        final_slot_reserved=True,conditional_probabilities_are_not_target_confidence=True)
    current=f1(c.champ['tp'],c.champ['predictions'])
    def finish():
        return {**result,'decision':'prepare_new_final' if best['f1']>current+1e-12 else 'retain_champion', 'next_ids':best['ids']}
    if c.remaining<=1 or bound['f1_upper']<TARGET or best['f1']>=TARGET:return finish()
    queries=[]
    for g in cat['groups']:
        ids=tuple(g['ids'])
        if not any(set(ids)&set(m) for m in measured):queries.append(('group',ids,None))
    if splits<1:
        for leaf in leaves:
            ids=sorted(leaf['ids'],key=lambda i:(-byid[i]['expected_u'],i))
            if len(ids)>1:queries.append(('split',tuple(sorted(ids[:len(ids)//2])),leaf))
    # Working model weights condition on nonoverlapping measured leaves once.
    distributions={}
    def dist(model,ids):
        k=(model,tuple(sorted(ids)))
        if k not in distributions:
            distributions[k]=convolve_counts([[byid[i]['model_probabilities'][model].get(str(v),0.) for v in (-1,0,1)] for i in ids])
        return distributions[k]
    logw={f:sum(math.log(max(dist(f,x['ids']).get(x['dt'],0.),1e-300)) for x in leaves) for f in FAMILIES}
    shift=max(logw.values());weights={f:math.exp(v-shift) for f,v in logw.items()}
    z=sum(weights.values());weights={f:v/z for f,v in weights.items()}
    ranked=[]
    for kind,ids,parent in queries:
        actions=[byid[i] for i in ids];lo,hi=c.eq.bound(actions);branches=[]
        for value in range(lo,hi+1):
            if not c.eq.possible(actions,value):continue
            mass=0.
            for f,w in weights.items():
                prob=dist(f,ids).get(value,0.)
                if parent:
                    rest=tuple(sorted(set(parent['ids'])-set(ids)))
                    prob*=dist(f,rest).get(parent['dt']-value,0.)/max(dist(f,parent['ids']).get(parent['dt'],0.),1e-300)
                mass+=w*prob
            updated=[x for x in leaves if x is not parent]+[{'ids':ids,'dt':value}]
            if parent:updated.append({'ids':tuple(sorted(set(parent['ids'])-set(ids))),'dt':parent['dt']-value})
            final=best_merge(updated,byid,c.safe_baseline)
            p=c.safe_baseline['predictions']+sum(a['delta_p'] for a in actions)
            score=f1(c.safe_baseline['tp']+value,p)
            branches.append(dict(delta_tp=value,score=f'{score:.6f}',mass=mass,retained_f1=max(current,score,final['f1'])))
        if len(branches)<2:continue
        total=sum(b['mass'] for b in branches)
        if total<=1e-290:
            for b in branches:b['mass']=1/len(branches)
        else:
            for b in branches:b['mass']/=total
        expected=sum(b['mass']*b['retained_f1'] for b in branches)
        entropy=-sum(b['mass']*math.log(max(b['mass'],1e-300)) for b in branches)
        ranked.append(((expected,entropy,-len(ids),tuple(-i for i in ids)),kind,ids,branches))
    if not ranked:return finish()
    _,kind,ids,branches=max(ranked)
    return {**result,'decision':'submit_new_split' if kind=='split' else 'submit_new_group','next_ids':list(ids),
        'score_lookup':branches,'policy':'one-step retained F1 then entropy; model-conditional approximation, not global optimum'}
