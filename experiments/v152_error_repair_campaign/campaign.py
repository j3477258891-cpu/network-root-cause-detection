"""V152 command line: audit/train/build/record/recommend/emit-final.

No command communicates with the competition platform. Real results must be
entered explicitly, with an attempt id and evidence. Generated files are not
automatically submissions and do not imply a leaderboard gain.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from core import *

DEFAULT = Path(__file__).resolve().parent
RUNTIME = Path(r'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies')
NODE = RUNTIME/'node/bin/node.exe'
MODULES = RUNTIME/'node/node_modules'


def records_data():
    with gzip.open(legacy.RECORDS,'rt',encoding='utf-8') as f:
        return json.load(f)


def project_index():
    excluded={'.git','.deps','.venv','node_modules','__pycache__','.codex_tmp'}
    projects={}
    for parent,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=[x for x in dirs if x not in excluded and not Path(parent,x).is_symlink()]
        rel=Path(parent).relative_to(ROOT)
        if rel.parts[:2]==('experiments','v152_error_repair_campaign'): continue
        parts=rel.parts
        project='/'.join(parts[:2] if parts and parts[0] in ('experiments','codexgz') else parts[:1]) or 'root'
        item=projects.setdefault(project,{'project':project,'file_count':0,'code':[],'reports':[]})
        for name in files:
            p=Path(parent,name);item['file_count']+=1
            if p.suffix in ('.py','.mjs','.js','.sh','.ps1','.ipynb'):
                record={'file':str(p.relative_to(ROOT)),'sha256':sha(p),'inspection':'content hash / structural index'}
                if p.suffix=='.py':
                    try:
                        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
                        record['definitions']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
                    except (SyntaxError,UnicodeError) as e: record['parse_error']=str(e)
                item['code'].append(record)
            if p.suffix=='.json' and any(s in name.lower() for s in ('report','summary','gate')):
                r={'file':str(p.relative_to(ROOT)),'sha256':sha(p),'evidence':'not automatically public-scored'}
                try:
                    data=read(p)
                    if isinstance(data,dict):
                        r['headline']={k:v for k,v in data.items() if k in ('version','status','decision','rank_tp_delta','tp_delta','gate_passed','submission_generated')}
                except (ValueError,UnicodeError) as e: r['parse_error']=str(e)
                item['reports'].append(r)
    return {'created_at':now(),'coverage':'directory inventory, deduplicated source structure and report index; not line-by-line review of every file',
        'projects':list(projects.values()),'project_count':len(projects),
        'file_count':sum(x['file_count'] for x in projects.values()),
        'unique_code_hashes':len({c['sha256'] for p in projects.values() for c in p['code']}),
        'reuse_decisions':{'V11/V16/V30/V33':'features and retrained controls; reject unaudited supervised scores',
            'V18/V24/V25':'negative or limited measured benefit; no automatic large-model rerun',
            'V38':'retrained ET/HGB action control under the same honest split',
            'V148/V149':'public baseline and online equation/merge mechanism',
            'V150':'20 of 80 count equation retained; not a standalone .94 route',
            'archives/backups':'indexed as assets, not additional independent model votes'}}


def audit(out, refresh=False):
    b,seq,_,_=baseline()
    if (out/'campaign.json').exists():
        config,cat,system,manifest,ledger=frozen_load(out)
        observations=[{'indices':r['indices'],'tp':r['inferred_tp']} for r in ledger['records'] if r['status']=='accepted']
        eq=Equations(system,observations)
        require(eq.feasible(),'Online equations conflict')
        for e in manifest['files']:
            validate_csv(out/e['file'],[cat['candidates'][i-1] for i in e['ids']],e['sha256'])
        result={'baseline':b,'integer_feasible':True,'attempts':len(ledger['records']),
                'remaining_submissions':10-len(ledger['records']),'files_checked':len(manifest['files'])}
    else:
        system=collect_system(records_data()['test'],seq)
        require(Equations(system).feasible(),'Historical equations conflict')
        result={'baseline':b,'public_equations':len(system['equations']),'variables':len(system['variables']),
                'integer_feasible':True,'campaign_built':False,'remaining_submissions':10}
    report_path=out/'model_comparison.json'
    progress_path=out/'training/progress.json'
    if report_path.exists() or progress_path.exists():
        tr=read(report_path if report_path.exists() else progress_path)
        hashes=tr.get('input_hashes',{})
        result['training']={'completed_folds':len(tr.get('folds',[])), 'complete':tr.get('complete',False),
            'gate_passed':tr.get('gate_passed',False),'winner':tr.get('winner'),
            'source_hashes_match':bool(hashes) and all(Path(p).is_file() and sha(p)==h for p,h in hashes.items()),
            'tabpfn':tr.get('tabpfn'), 'report':str(report_path if report_path.exists() else progress_path)}
    if refresh:
        write(out/'project_audit.json',project_index())
        write(out/'audit.json',result)
    return result


def action_exclusions(nodes, system):
    blocked = legacy.initial_exclusions(nodes)
    # V150's addition-only helper did not carry the V117 removal-risk pool.
    prior = read(EXP/'v117_distance1_online_result.json')
    file = Path(prior['submitted_file'])
    row = next((e for e in system['equations'] if e['sha256']==sha(file)),None)
    require(row is not None and row['tp']==prior['inferred_tp'], 'V117 risk-pool source is not in verified equations')
    _,_,before = load_csv(EXP/'v120_swap_campaign/probe_00_baseline.csv')
    # Historical scored files may predate today's root-count constraints;
    # their counts remain evidence, but no new output inherits that format.
    _,_,after = load_csv(file, strict=False)
    removed = set(before)-set(after)
    require(len(removed)==prior['changed_deletions'], 'V117 removal-pool drift')
    for key in removed: blocked.setdefault(key,[]).append('v117_deletion_pool')
    return blocked


def exclusion_reason(field, reasons):
    # A false-addition exclusion is not a reason to forbid deleting a known FP.
    # Risk/failed-action policies still apply to removals in either case.
    relevant = reasons if field=='add_rid' else [r for r in reasons if 'deletion_pool' in r or 'negative_action' in r]
    return 'historical_exclusion:'+','.join(relevant) if relevant else None


def build(out):
    require(not (out/'campaign.json').exists(),'Campaign already frozen; cannot overwrite or rebuild')
    report_path=out/'model_comparison.json'
    require(report_path.exists(),'Run train first')
    report=read(report_path)
    if not (report.get('complete') and report.get('gate_passed') and not report.get('smoke_only')):
        gate={'allow_submission':False,'reason':'training_gate_failed_or_incomplete','baseline_f1':f1(),
              'remaining_submissions':10,'decision':'retain_champion','model_report':str(report_path)}
        write(out/'final_gate.json',gate)
        return gate
    require(len(report['folds'])==5 and {x['fold'] for x in report['folds']}==set(range(5)),'Incomplete outer validation')
    from training import pooled, source_hashes
    require(report['input_hashes']==source_hashes(),'Training inputs/code changed; frozen estimates invalid')
    winner=report['winner']
    performance=pooled(report['folds'],winner)
    require(performance['positive_folds']>=3 and performance['f1']>performance['base_f1'] and
            performance['f1']>max(pooled(report['folds'],c)['f1'] for c in ('v30_control','v38_control')),'Recomputed training gate failed')
    trained=read(out/'training_candidates.json')
    require(trained['report_sha256']==sha(report_path),'Model report/candidate mismatch')
    b,seq,roots,nodes=baseline()
    data=records_data()
    system=collect_system(data['test'],seq)
    eq=Equations(system)
    require(eq.feasible(),'Historical equations conflict')
    blocked=action_exclusions(nodes,system)
    all_keys={ (a['order_id'],a[field]) for a in trained['candidates'] for field in ('add_rid','remove_rid') if a.get(field)}
    bounds=legacy.EquationModel(system).classify(all_keys)
    proposals=sorted(trained['candidates'],key=lambda a:(-f1(expected_dt(a),a['delta_p']),a['order_id'],a.get('add_rid') or '',a.get('remove_rid') or ''))
    candidates=[]; rejected=[]; seen=set()
    for a in proposals:
        reason=None;oid=a['order_id']
        if oid==legacy.BLOCKED_ORDER: reason='permanently_excluded_order'
        for field in ('add_rid','remove_rid'):
            rid=a.get(field)
            if not rid: continue
            key=(oid,rid)
            if key in blocked: reason=exclusion_reason(field,blocked[key]) or reason
            if field=='remove_rid' and bounds[key]['min']==1: reason='protected_equation_true'
            if field=='add_rid' and bounds[key]['max']==0: reason='equation_false_addition'
        if oid in seen: reason='independent_order_already_selected'
        if f1(expected_dt(a),a['delta_p'])<=f1(): reason='nonpositive_model_expected_gain'
        if reason:
            rejected.append({**a,'reason':reason});continue
        validate_actions([a],roots,nodes)
        delta_bounds=eq.bounds_for([a])
        if f1(delta_bounds['max'],a['delta_p'])<=f1():
            rejected.append({**a,'reason':'equation_proven_no_f1_gain','delta_tp_bounds':delta_bounds});continue
        seen.add(oid)
        candidates.append({**a,'candidate_id':len(candidates)+1,'delta_tp_bounds':delta_bounds,
            'expected_delta_tp':expected_dt(a),'expected_f1':f1(expected_dt(a),a['delta_p'])})
        if len(candidates)==64: break
    if not candidates:
        gate={'allow_submission':False,'reason':'no_eligible_independent_actions','decision':'retain_champion'}
        write(out/'candidate_catalog.json',{'candidates':[],'rejected':rejected})
        write(out/'final_gate.json',gate);return gate
    groups=[]
    for kind in ('add','delete','swap'):
        ids=[a['candidate_id'] for a in candidates if a['kind']==kind]
        for start in range(0,len(ids),16):
            groups.append({'group_id':f'G{len(groups)+1:02d}','kind':kind,'ids':ids[start:start+16]})
    catalog={'candidates':candidates,'rejected':rejected,'groups':groups,'source_class':'model_estimated',
             'probabilities_are_conditional_not_target_confidence':True,'model_report_sha256':sha(report_path)}
    oracle=eq.oracle(candidates)
    write(out/'equation_system.json',system)
    write(out/'candidate_catalog.json',catalog)
    config={'version':1,'created_at':now(),'baseline':b,'target':TARGET,'comparison':'strictly_greater',
        'budget':{'total':10,'daily':2,'max_probes':8,'source':'latest_user_confirmed_10_remaining'},
        'frozen_files':{name:sha(out/name) for name in ('equation_system.json','candidate_catalog.json','model_comparison.json')},
        'initial_oracle':oracle}
    write(out/'campaign.json',config)
    write(out/'submission_manifest.json',{'files':[]})
    write(out/'online_scores.json',{'baseline':b,'records':[],'note':'only real user-supplied attempts; failed/anomalous attempts count'})
    if not oracle['target_possible_under_equations']:
        gate={'allow_submission':False,'reason':'candidate_pool_oracle_cannot_reach_target','oracle':oracle,'decision':'retain_champion'}
        write(out/'final_gate.json',gate)
        return gate
    return recommend(out,emit=True)


def context(out):
    config,cat,system,manifest,ledger=frozen_load(out)
    byid={a['candidate_id']:a for a in cat['candidates']}
    observations=[{'indices':r['indices'],'tp':r['inferred_tp']} for r in ledger['records'] if r['status']=='accepted']
    eq=Equations(system,observations)
    require(eq.feasible(),'Current public equations conflict')
    leaves=[];tested=[];root_results={};champ=copy.deepcopy(config['baseline'])
    entries={e['probe_id']:e for e in manifest['files']}
    for r in ledger['records']:
        if r['status']!='accepted': continue
        e=entries[r['probe_id']]
        if not r.get('duplicate_equation'):
            leaves=apply_observation(leaves,e,r['delta_tp'])
        if e['kind']=='group' and not r.get('duplicate_equation'):
            tested.append(e['group_id'])
            kind=byid[e['ids'][0]]['kind']
            root_results.setdefault(kind,[]).append(f1(r['delta_tp'],e['delta_p'])>f1())
        if 2*r['inferred_tp']/(G+r['predictions']) > 2*champ['tp']/(G+champ['predictions']):
            champ={'file':str(out/e['file']),'sha256':e['sha256'],'predictions':r['predictions'],
                   'tp':r['inferred_tp'],'score':r['score'],'source_class':'public_scored'}
    return config,cat,manifest,ledger,byid,eq,leaves,tested,root_results,champ


def risk_choice(leaves,byid,eq,champ):
    adjusted={i:copy.deepcopy(a) for i,a in byid.items()}
    for leaf in leaves:
        for i in leaf['ids']:
            a=adjusted[i];others=distribution([byid[j] for j in leaf['ids'] if j!=i])
            probabilities={k:p*others.get(leaf['delta_tp']-int(k),0) for k,p in a['probabilities'].items()}
            total=sum(probabilities.values())
            require(total>0,'Unsupported conditional leaf count')
            a['probabilities']={k:v/total for k,v in probabilities.items()}
    ranked=[]
    for i,a in adjusted.items():
        probs={k:v for k,v in a['probabilities'].items() if v>0 and eq.feasible(eq.coeff([a]),int(k))}
        require(sum(probs.values())>0,'No feasible action marginal')
        a['probabilities']={k:v/sum(probs.values()) for k,v in probs.items()}
        ranked.append(i)
    ranked.sort(key=lambda i:(-f1(expected_dt(adjusted[i]),adjusted[i]['delta_p']),i))
    result=None;dt=dp=0
    for n,i in enumerate(ranked,1):
        dt+=expected_dt(adjusted[i]);dp+=adjusted[i]['delta_p']
        value=f1(dt,dp)
        if value>2*champ['tp']/(G+champ['predictions']) and (result is None or value>result['expected_f1']):
            result={'kind':'risk','ids':sorted(ranked[:n]),'expected_f1':value,
                    'warning':'unverified within-group selection; conditional model expectation, possible loss'}
    return result


def recommend(out,emit=False):
    if not (out/'campaign.json').exists():
        return read(out/'final_gate.json') if (out/'final_gate.json').exists() else {'decision':'train_then_build','allow_submission':False}
    config,cat,manifest,ledger,byid,eq,leaves,tested,root_results,champ=context(out)
    attempts=len(ledger['records']);best=best_union(leaves,byid)
    report={'actual_champion':champ,'attempts_used':attempts,'remaining_submissions':10-attempts,
            'measured_leaves':leaves,'best_known_union':best,'allow_submission':False}
    def finish(decision,**extra):
        report.update(decision=decision,**extra)
        if emit:
            write(out/'recommendation.json',report);write(out/'final_gate.json',report)
        return report
    for e in manifest['files']:
        if e.get('file') and (not (out/e['file']).is_file() or sha(out/e['file'])!=e['sha256']):
            return finish('pause_changed_submission_file',changed_file=e['file'])
    for e in getattr(eq,'system',{}).get('equations',[]):
        if e.get('file') and (not Path(e['file']).is_file() or sha(e['file'])!=e['sha256']):
            return finish('pause_changed_historical_source',changed_file=e['file'])
    if any(r['status']=='anomaly' and not r.get('resolved_by') for r in ledger['records']): return finish('pause_conflicting_or_invalid_observation')
    if champ['tp']*2/(G+champ['predictions'])>TARGET: return finish('target_achieved_online')
    if attempts>=10: return finish('budget_exhausted')
    pending=[e for e in manifest['files'] if not any(r['probe_id']==e['probe_id'] for r in ledger['records'])]
    if pending: return finish('await_existing_file_feedback',pending=pending[0])
    selected=None
    # Only positive improvement over an already scored champion merits a known merge.
    if best['f1']>2*champ['tp']/(G+champ['predictions']) and (best['f1']>TARGET or attempts>=8):
        selected={'kind':'merge','ids':best['ids'],'expected_tp':TP0+best['delta_tp'],'expected_f1':best['f1'],
                  'evidence':'exact independent-group integer counts'}
    oracle=eq.oracle(list(byid.values()));report['oracle']=oracle
    if selected is None and not oracle['target_possible_under_equations']:
        if best['f1']>2*champ['tp']/(G+champ['predictions']):
            selected={'kind':'merge','ids':best['ids'],'expected_tp':TP0+best['delta_tp'],'expected_f1':best['f1'],'evidence':'incremental gain only; target unreachable in this pool'}
        else: return finish('stop_target_pool_retain_champion')
    probe_attempts=sum(next(e for e in manifest['files'] if e['probe_id']==r['probe_id'])['kind'] in ('group','split') for r in ledger['records'])
    if selected is None and attempts<8 and probe_attempts<8:
        banned=[kind for kind,values in root_results.items() if len(values)>=2 and not any(values[-2:])]
        options=next_queries(cat['groups'],leaves,tested,byid,calibration_phase=probe_attempts<2,banned_kinds=banned)
        old_sets={tuple(sorted(e['ids'])) for e in manifest['files']}
        options=[q for q in options if tuple(sorted(q['ids'])) not in old_sets]
        evaluated=[evaluate_query(q,leaves,byid,eq) for q in options]
        if evaluated:
            selected=max(evaluated,key=lambda q:(q['conditional_expected_probe_f1'] if probe_attempts<2 else q['conditional_p_target'],
                q['expected_best_f1'],-len(q['ids']),tuple(-i for i in sorted(q['ids']))))
        elif best['f1']>2*champ['tp']/(G+champ['predictions']):
            selected={'kind':'merge','ids':best['ids'],'expected_tp':TP0+best['delta_tp'],'expected_f1':best['f1']}
    # A redundant ninth merge must not be uploaded just to unlock refinement.
    # If eight probes already scored the best exact union, skip that merge and
    # allow one risky refinement while leaving the unused slot for anomalies.
    risk_already_used=any(next(e for e in manifest['files'] if e['probe_id']==r['probe_id'])['kind']=='risk'
                          for r in ledger['records'])
    if selected is None and attempts>=8 and not risk_already_used:
        selected=risk_choice(leaves,byid,eq,champ)
        if selected is not None: report['redundant_merge_skipped']=attempts==8
    if selected is None: return finish('no_valuable_new_submission_retain_champion')
    if any(set(e['ids'])==set(selected['ids']) for e in manifest['files']):
        return finish('selected_set_already_exists_no_duplicate_submission')
    today=datetime.now(TZ).date()
    daily=sum(datetime.fromisoformat(r['submitted_at']).astimezone(TZ).date()==today for r in ledger['records'])
    report['daily_attempts_in_campaign']=daily
    report['earliest_submission']='next local day' if daily>=2 else 'after confirming remaining platform daily quota'
    report.update({'decision':'submit_'+selected['kind'],'allow_submission':daily<2,'next_action':selected,
                   'probability_warning':'model-conditional policy scores, not probability of real target success'})
    if emit:
        entry=author(out,selected,byid,config,manifest)
        report['file']=entry
        write(out/'recommendation.json',report);write(out/'final_gate.json',report)
    return report


def author(out,query,byid,config,manifest):
    require(not any(set(e['ids'])==set(query['ids']) for e in manifest['files']),'Duplicate prediction set')
    probe_id=f'p{len(manifest["files"])+1:02d}'
    kind=query['kind'];filename=f'v152_{"final" if kind in ("merge","risk") else "probe"}_{probe_id}_{kind}.csv'
    actions=[byid[i] for i in query['ids']]
    _,_,roots,nodes=baseline();validate_actions(actions,roots,nodes)
    spec={'base_file':str(BASE),'base_sha256':config['baseline']['sha256'],'output_file':str(out/filename),
          'preview_file':str(out/'previews'/f'{probe_id}.png'),'actions':actions}
    spec_path=out/'authoring_specs'/f'{probe_id}.json';write(spec_path,spec)
    subprocess.run([str(NODE),str(DEFAULT/'artifact_builder.mjs'),str(spec_path)],check=True)
    verified=validate_csv(out/filename,actions)
    entry={**query,'probe_id':probe_id,'file':filename,'ids':query['ids'],
           'delta_p':sum(a['delta_p'] for a in actions),'predictions':verified['predictions'],
           'sha256':verified['sha256'],'created_at':now(),'local_validation':verified}
    manifest['files'].append(entry);write(out/'submission_manifest.json',manifest)
    return entry


def record(out,probe_id,attempt_id,score,submitted_at,evidence,failed=False,resolves=None):
    config,cat,manifest,ledger,byid,eq,leaves,tested,root_results,champ=context(out)
    require(evidence.strip(),'An actual score or failure needs explicit evidence')
    old=next((r for r in ledger['records'] if r['attempt_id']==attempt_id),None)
    if old:
        require(old['probe_id']==probe_id and old.get('score')==score and old['submitted_at']==submitted_at,'Conflicting reuse of attempt id')
        return {'idempotent':True,'record':old}
    check_budget(ledger['records'],submitted_at)
    if resolves:
        anomaly=next((r for r in ledger['records'] if r['attempt_id']==resolves),None)
        require(anomaly is not None and anomaly['status']=='anomaly' and anomaly['probe_id']==probe_id,
                'Only a new actual repeat of the same probe can resolve that anomaly')
    entry=next((e for e in manifest['files'] if e['probe_id']==probe_id),None)
    require(entry is not None,'Unknown emitted probe id')
    attempt={'attempt_id':attempt_id,'probe_id':probe_id,'submitted_at':submitted_at,'evidence':evidence,
             'score':score,'status':'failed' if failed else 'accepted','source_class':'actual_submission_attempt'}
    if not failed:
        try:
            require(score is not None,'Successful submission requires a real score')
            checked=validate_csv(out/entry['file'],[byid[i] for i in entry['ids']],entry['sha256'])
            tp=infer_tp(score,checked['predictions'])
            _,_,nodes=load_csv(out/entry['file'])
            require(eq.feasible(eq.coeff([byid[i] for i in entry['ids']]),tp-TP0),'Returned score conflicts with history or signed action counts')
            require(entry.get('expected_tp',tp)==tp,'Final score differs from exact merge calculation')
            attempt.update({'predictions':checked['predictions'],'sha256':checked['sha256'],'inferred_tp':tp,
                'delta_tp':tp-TP0,'indices':sorted(eq.index[k] for k in nodes),'source_class':'public_scored'})
            # Repeated independent platform attempts count, but their group need not be applied twice.
            if any(r['probe_id']==probe_id and r['status']=='accepted' for r in ledger['records']):
                require(any(r.get('inferred_tp')==tp for r in ledger['records'] if r['probe_id']==probe_id),'Repeated score disagreement')
                attempt['duplicate_equation']=True
        except (ValueError,KeyError) as exc:
            attempt.update(status='anomaly',error=str(exc))
    if resolves and attempt['status']=='accepted':
        anomaly['resolved_by']=attempt_id
    ledger['records'].append(attempt);write(out/'online_scores.json',ledger)
    decision=recommend(out,emit=False)
    write(out/'recommendation.json',decision);write(out/'final_gate.json',decision)
    return {'record':attempt,'assessment':decision}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--refresh',action='store_true')
    t=sub.add_parser('train');t.add_argument('--max-hours',type=float,default=48);t.add_argument('--smoke',action='store_true');t.add_argument('--tabpfn-rules-evidence',type=Path)
    sub.add_parser('build')
    r=sub.add_parser('recommend');r.add_argument('--emit',action='store_true')
    sub.add_parser('emit-final')
    r=sub.add_parser('record');r.add_argument('--probe-id',required=True);r.add_argument('--attempt-id',required=True)
    r.add_argument('--score');r.add_argument('--submitted-at',required=True);r.add_argument('--evidence',required=True);r.add_argument('--failed',action='store_true');r.add_argument('--resolves-attempt')
    args=parser.parse_args();out=args.out.resolve()
    if args.command not in ('audit','recommend') or getattr(args,'refresh',False) or getattr(args,'emit',False):
        out.mkdir(parents=True,exist_ok=True)
    if args.command=='audit': result=audit(out,args.refresh)
    elif args.command=='train':
        from training import train
        from threadpoolctl import threadpool_limits
        with threadpool_limits(4): result=train(out,args.max_hours,args.smoke,args.tabpfn_rules_evidence)
        write(out/('smoke_training' if args.smoke else 'training')/'progress.json',result)
        result={k:result[k] for k in ('complete','gate_passed','winner','summary','tabpfn','decision')}
        result['report']=str(out/('smoke_training/model_comparison.json' if args.smoke else 'model_comparison.json'))
    elif args.command=='build': result=build(out)
    elif args.command=='recommend': result=recommend(out,args.emit)
    elif args.command=='emit-final':
        suggestion=recommend(out)
        require(suggestion['decision'] in ('submit_merge','submit_risk'),'No gated final is currently recommended')
        result=recommend(out,True)
    else: result=record(out,args.probe_id,args.attempt_id,args.score,args.submitted_at,args.evidence,args.failed,args.resolves_attempt)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error':str(exc),'type':type(exc).__name__,'decision':'do_not_submit'},ensure_ascii=False),file=sys.stderr)
        raise
