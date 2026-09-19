"""Finite-ensemble full-budget rollouts; no calibrated success probability claims."""
from support import *
from engine import scenarios,best_known,conditional_finals

def entropy(values):
    _,counts=np.unique(values,return_counts=True)
    p=counts/len(values)
    return float(-(p*np.log(p)).sum())

def query_candidates(c):
    known={tuple(g['ids']) for g in c.groups}
    seen={};actions=c.actions
    for group in c.groups:
        ids=tuple(group['ids']);n=len(ids)
        if n<2:continue
        subsets=set()
        if all(actions[i]['pool'] in ('B','C') for i in ids) and n<=12:
            subsets={tuple(ids[j] for j in range(n) if bits>>j&1) for bits in range(1,2**n-1)}
        else:
            rankings=[sorted(ids,key=lambda i:(actions[i].get('candidate_id',i),i))]
            for source in ('station','consensus','template'):
                rankings.append(sorted(ids,key=lambda i:(actions[i].get('ranks',{}).get(source,10**9),i)))
            for order in rankings:
                for k in range(1,n):
                    subsets.add(tuple(sorted(order[:k])));subsets.add(tuple(sorted(order[k:])))
        other=[g for g in c.groups if not set(g['ids'])&set(ids)]
        carrier=best_known(other,actions)
        for q in sorted(subsets):
            if q in known or q in seen:continue
            comp=tuple(i for i in ids if i not in set(q))
            if comp in known:continue # parent and complement already determine q
            qdp=sum(actions[i]['delta_p'] for i in q)
            lo=sum(actions[i]['delta_tp_bounds']['min'] for i in q)
            hi=sum(actions[i]['delta_tp_bounds']['max'] for i in q)
            good=bool(carrier['ids']) and all(
                f1(972+t+carrier['delta_tp'],1045+qdp+carrier['delta_p'])>f1(972+t,1045+qdp)
                for t in (lo,hi))
            anchors=carrier['ids'] if good else []
            seen[q]=dict(query_ids=list(q),parent_ids=list(ids),
                         anchor_ids=anchors,anchor_delta_tp=carrier['delta_tp'] if good else 0,
                         ids=sorted(list(q)+anchors))
    return sorted(seen.values(),key=lambda q:(len(q['ids']),q['ids'],q['parent_ids']))

def assess_final(c,worlds,known):
    candidates,masks,scores=conditional_finals(c,worlds,known)
    retained=np.maximum(scores,c.champion_f1)
    reach=(scores>=float(TARGET)).mean(axis=0) if c.mode=='target' else np.zeros(len(candidates))
    expected=retained.mean(axis=0)
    chosen=max(range(len(candidates)),key=lambda j:(reach[j],expected[j],-len(candidates[j]),tuple(-i for i in candidates[j])))
    if reached(known['tp'],known['predictions']):
        chosen=candidates.index(tuple(known['ids']))
    ids=list(candidates[chosen]);lo,hi=c.eq.bound([c.actions[i] for i in ids])
    p=1045+sum(c.actions[i]['delta_p'] for i in ids)
    report=dict(ids=ids,predictions=p,tp_range=[972+lo,972+hi],
                worst_feasible_f1=f1(972+lo,p),best_feasible_f1=f1(972+hi,p),
                conditional_expected_raw_f1=float(scores[:,chosen].mean()),
                conditional_expected_retained_f1=float(expected[chosen]),
                conditional_target_frequency=float((scores[:,chosen]>=float(TARGET)).mean()),
                evidence='finite feasible perturb-and-MAP ensemble; not a calibrated online forecast',
                guaranteed=(lo==hi),known_merge=known)
    return report,candidates,scores,chosen

def exact_fallback(c,queries,reason):
    # Small single-group query; only exact integer information can earn a probe.
    for q in sorted(queries,key=lambda x:(len(x['query_ids']),x['query_ids'])):
        try:
            lookup=c.lookup(q['ids'])
            if len(lookup)>1:
                return dict(decision='probe',query=q,score_lookup=lookup,
                            method='exact single-group fallback',fallback_reason=reason,
                            calibrated_online_probability=None)
        except ValueError:continue
    return dict(decision='needs_solver_audit',fallback_reason=reason,
                reason='No completed exact informative-query check; does not prove all actions resolved')

def search(c,sample_count=256,rollouts=256,top_n=24,seconds=900):
    start=cutoff(seconds);queries=query_candidates(c)
    known=c.known_merge()
    try:
        bundle=scenarios(c,count=sample_count,seconds=min(600,max(1,start-time.monotonic())))
        worlds=bundle['worlds'];within(start)
        final,candidates,final_scores,final_index=assess_final(c,worlds,known)
        scenario_path=c.out/f'scenarios_{c.state_id}.npz'
        np.savez_compressed(scenario_path,worlds=worlds,witnesses=bundle['witnesses'],
                            families=np.array(bundle['families']))
        base=dict(state_id=c.state_id,scenario_file=str(scenario_path),scenario_sha256=sha(scenario_path),
                  scenario_method=bundle['method'],unique_scenarios=bundle['unique_worlds'],
                  sample_count_per_family=sample_count,known_merge=known,model_assisted_final=final,
                  mode=c.mode,calibrated_online_probability=None)
        if c.remaining<=1 or reached(known['tp'],known['predictions']) or not queries:
            return dict(**base,decision='final',reason='last_slot_or_known_target_or_no_queries')
        masks=np.zeros((len(queries),len(c.actions)),dtype=np.int8)
        for j,q in enumerate(queries):masks[j,q['ids']]=1
        outcomes=worlds.astype(np.int16)@masks.T
        entropies=np.array([entropy(outcomes[:,j]) for j in range(len(queries))])
        ranking=sorted(range(len(queries)),key=lambda j:(-entropies[j],len(queries[j]['ids']),queries[j]['ids']))
        top=ranking[:top_n];future=ranking[:64]
        dp=masks@np.array([a['delta_p'] for a in c.actions])
        probe_scores=2*(972+outcomes)/(G+1045+dp)[None,:]
        rng=np.random.default_rng(SEED+len(c.records))
        truths=rng.integers(len(worlds),size=rollouts)
        terminal_cache={};next_cache={}
        def terminal(indices,champ):
            cachekey=(tuple(indices),round(champ,12))
            if cachekey not in terminal_cache:
                block=final_scores[indices]
                hit=(block>=float(TARGET)).mean(axis=0) if c.mode=='target' else np.zeros(block.shape[1])
                retain=np.maximum(block,champ).mean(axis=0)
                j=max(range(block.shape[1]),key=lambda j:(hit[j],retain[j],-len(candidates[j]),tuple(-i for i in candidates[j])))
                terminal_cache[cachekey]=j
            return terminal_cache[cachekey]
        estimates=[];steps=c.remaining-1
        for count,j in enumerate(top):
            within(start);hits=[];retained=[]
            for truth in truths:
                indices=np.arange(len(worlds));champ=c.champion_f1;query=j
                for depth in range(steps):
                    response=outcomes[truth,query]
                    champ=max(champ,float(probe_scores[truth,query]))
                    indices=indices[outcomes[indices,query]==response]
                    if champ>=float(TARGET):break
                    signature=tuple(indices)
                    if signature not in next_cache:
                        scores=[entropy(outcomes[indices,k]) for k in future]
                        k=int(np.argmax(scores));next_cache[signature]=(future[k],scores[k])
                    query,info=next_cache[signature]
                    if info<=1e-12:break
                candidate=terminal(indices,champ)
                result=max(champ,float(final_scores[truth,candidate]))
                hits.append(result>=float(TARGET));retained.append(result)
            estimates.append(dict(query_index=j,conditional_target_frequency=float(np.mean(hits)),
                                  conditional_expected_retained_f1=float(np.mean(retained)),entropy=float(entropies[j])))
            if count%6==0:print('V160 full-budget rollouts',count+1,len(top),flush=True)
        estimates.sort(key=lambda x:(-(x['conditional_target_frequency'] if c.mode=='target' else 0),
                                      -x['conditional_expected_retained_f1'],len(queries[x['query_index']]['ids']),
                                      queries[x['query_index']]['ids']))
        immediate=(final['conditional_target_frequency'] if c.mode=='target' else 0,
                   final['conditional_expected_retained_f1'])
        for item in estimates:
            within(start);q=queries[item['query_index']];lookup=c.lookup(q['ids'])
            if len(lookup)<2:continue
            future_value=(item['conditional_target_frequency'] if c.mode=='target' else 0,
                          item['conditional_expected_retained_f1'])
            # A complete conditional comparison can prefer finalizing. It is never a proof of impossibility.
            finish=future_value<=immediate and (known['f1']>c.champion_f1 or final['conditional_expected_retained_f1']>c.champion_f1)
            return dict(**base,decision='final' if finish else 'probe',query=q,score_lookup=lookup,
                        method='finite fixed-library receding-horizon rollout, not global optimality',
                        rollout_count_per_query=rollouts,initial_queries_evaluated=len(estimates),
                        maximum_probe_depth=steps,query_library_size=len(queries),rollout_future_library_size=len(future),
                        estimates=estimates,immediate_final_comparison=list(immediate),
                        comparison_prefers_final=finish,
                        approximation='Future rollouts use the current top-64 legal queries and current final candidate library; the real next turn regenerates both.')
        return {**base,**exact_fallback(c,queries,'Finite scenarios contain no informative supported query')}
    except ValueError as exc:
        return dict(state_id=c.state_id,known_merge=known,**exact_fallback(c,queries,str(exc)))
