"""No test fabricates a public score in the real campaign directory."""
import copy
import gc
import json
import unittest
import contextlib
import io
from argparse import Namespace
from unittest.mock import patch, MagicMock
from itertools import combinations, product
from math import comb
from time import perf_counter

import v150_adaptive_addition_campaign as campaign
from v150_adaptive_policy import Policy, best_union, choose_orientation, f1, reached


class CountTests(unittest.TestCase):
    def test_unique_score_inversion(self):
        for n in range(81):
            for k in range(min(n,75)+1):
                self.assertEqual(campaign.infer_tp(f'{f1(n,k):.6f}',1045+n),969+k)
        with self.assertRaises(ValueError):campaign.infer_tp('0.123456',1125)
        with self.assertRaises(ValueError):campaign.infer_tp('nan',1125)
        with self.assertRaises(ValueError):campaign.infer_tp('0.1234567',1125)

    def test_target_and_oracle_boundaries(self):
        self.assertLess(f1(80,50),.94)
        self.assertGreater(f1(80,51),.94)
        self.assertLess(f1(24,24),.94)
        self.assertGreater(f1(25,25),.94)
        self.assertGreater(f1(79,50),.94)

    def test_exact_best_union_and_threshold(self):
        for ts in product(range(5),repeat=3):
            groups=tuple((4,t) for t in ts)
            n,k,_=best_union(groups)
            maximum=max(f1(sum(4*b for b in bits),sum(t*b for t,b in zip(ts,bits))) for bits in product((0,1),repeat=3))
            self.assertAlmostEqual(f1(n,k),maximum)
            self.assertEqual(reached(groups),maximum>.94)
        self.assertEqual(best_union(((8,8),(8,0))),(8,8,(0,)))

    def test_count_conservation_both_orientations(self):
        leaves=[{'ids':list(range(1,9)),'true_count':5},{'ids':list(range(9,81)),'true_count':45}]
        common={'kind':'split','parent_ids':list(range(1,9)),'query_ids':[7,8],
                'known_true_count':45,'anchor_groups':[list(range(9,81))]}
        q={**common,'orientation':'query','selected_ids':[7,8]+list(range(9,81))}
        c={**common,'orientation':'complement','selected_ids':list(range(1,7))+list(range(9,81))}
        a=campaign.apply_measurement(leaves,q,969+45+1)
        b=campaign.apply_measurement(leaves,c,969+45+4)
        self.assertEqual(a,b)
        self.assertEqual(sum(g['true_count'] for g in a),50)
        with self.assertRaises(ValueError):campaign.apply_measurement(leaves,q,969+45+3)
        with self.assertRaises(ValueError):campaign.apply_measurement(leaves,{**q,'known_true_count':44},969+46)
        with self.assertRaises(ValueError):campaign.apply_measurement(leaves,{**q,'selected_ids':[7,8]},969+46)

    def test_orientation_information_equivalence(self):
        leaves=[{'ids':list(range(1,81)),'true_count':50}]
        choice=choose_orientation(leaves,leaves[0],[80],Policy())
        self.assertEqual(choice['orientation'],'complement')
        self.assertEqual(choice['selected_ids'],list(range(1,80)))
        for true in (0,1):
            entry={'kind':'split','parent_ids':list(range(1,81)),'query_ids':[80],**choice}
            result=campaign.apply_measurement(leaves,entry,969+50-true)
            self.assertEqual(next(g['true_count'] for g in result if g['ids']==[80]),true)

    def test_final_exactness(self):
        leaves=[{'ids':list(range(1,41)),'true_count':30},{'ids':list(range(41,81)),'true_count':20}]
        e={'kind':'final','selected_ids':list(range(1,41))}
        self.assertEqual(campaign.apply_measurement(leaves,e,999),leaves)
        with self.assertRaises(ValueError):campaign.apply_measurement(leaves,e,998)
        with self.assertRaises(ValueError):campaign.apply_measurement(leaves,{**e,'selected_ids':[1]},970)

    def test_replay_terminal_states_and_reserve(self):
        config={'baseline':{'tp':969,'predictions':1045,'score':.927717,'file':'base','sha256':'base'}}
        entry={'probe_id':'total','kind':'total','selected_ids':list(range(1,81)),'predictions':1125}
        for t,status in [(0,'stop_target_pool'),(24,'stop_target_pool'),(25,'ready_to_split'),(50,'ready_to_split'),(51,'target_achieved')]:
            e={'probe_id':'total','status':'accepted','inferred_tp':969+t,'file':'total','sha256':'h','score':round(f1(80,t),6)}
            state=campaign.replay(config,{'files':[entry]},{'records':[e]})
            self.assertEqual(state['status'],status)
            self.assertEqual(state['remaining_split_queries'],4)
            self.assertEqual(state['remaining_submissions'],5)
        state=campaign.replay(config,{'files':[entry]},{'records':[{'status':'anomaly','error':'bad hash'}]})
        self.assertEqual(state['status'],'paused')
        self.assertEqual(state['champion'],config['baseline'])

    def test_last_step_closed_form(self):
        p=Policy()
        for n in range(2,13):
            for t in range(1,n):
                for needed in (max(0,200*t-94*n)+d for d in (0,1,86,94,200)):
                    direct=0
                    for a in range(1,n//2+1):
                        value=0
                        for x in range(max(0,a-n+t),min(a,t)+1):
                            weight=comb(t,x)*comb(n-t,a-x)/comb(n,a)
                            value+=weight*(max(0,200*x-94*a)+max(0,200*(t-x)-94*(n-a))>needed)
                        direct=max(direct,value)
                    self.assertAlmostEqual(p.best_one(n,t,needed),direct)

    def test_record_idempotence_anomaly_and_no_simulation_files(self):
        config={'baseline':{'tp':969,'predictions':1045,'score':.927717,'file':'base','sha256':'base'}}
        entry={'probe_id':'total','kind':'total','file':'probe.csv','selected_ids':list(range(1,81)),
               'predictions':1125,'sha256':'fake_hash_only_in_memory'}
        manifest={'files':[entry]}; scores={'records':[]}; cat={'candidates':[]}
        fake_model=MagicMock(); fake_model.index={}
        from pathlib import Path
        args=Namespace(probe_id='total',score=f'{f1(80,30):.6f}',attempt_id=None,submitted_at='2026-09-06T11:00:00+08:00',
                       failed=False,reason=None,evidence='UNIT_TEST_ONLY_NEVER_PERSISTED')
        writes=[]
        with patch.object(campaign,'load_campaign',return_value=(config,cat,{},manifest,scores)), \
             patch.object(campaign,'validate_file'), patch.object(campaign,'load_csv',return_value=([],{},{})), \
             patch.object(campaign,'EquationModel',return_value=fake_model), \
             patch.object(campaign,'write_json',side_effect=lambda p,x:writes.append(str(p))), \
             patch.object(campaign,'save_state',side_effect=lambda out:campaign.replay(config,manifest,scores)), \
             contextlib.redirect_stdout(io.StringIO()):
            campaign.record(Path('IN_MEMORY_NOT_A_DIRECTORY'),args)
            self.assertEqual(len(scores['records']),1)
            self.assertEqual(scores['records'][0]['status'],'accepted')
            campaign.record(Path('IN_MEMORY_NOT_A_DIRECTORY'),args)
            self.assertEqual(len(scores['records']),1)
            args.attempt_id='actual_second_attempt'
            campaign.record(Path('IN_MEMORY_NOT_A_DIRECTORY'),args)
            self.assertEqual(len(scores['records']),2)
            self.assertEqual(scores['records'][1]['status'],'accepted')
            self.assertTrue(scores['records'][1]['repeat_measurement'])
            self.assertEqual(campaign.replay(config,manifest,scores)['remaining_submissions'],4)
            args.attempt_id='actual_third_attempt';args.score='0.123456'
            campaign.record(Path('IN_MEMORY_NOT_A_DIRECTORY'),args)
            self.assertEqual(len(scores['records']),3)
            self.assertEqual(scores['records'][2]['status'],'anomaly')
            self.assertEqual(campaign.replay(config,manifest,scores)['status'],'paused')
            args.score='NaN';args.attempt_id='invalid_input'
            with self.assertRaises(ValueError):campaign.record(Path('IN_MEMORY_NOT_A_DIRECTORY'),args)
            self.assertEqual(len(scores['records']),3)
        self.assertEqual(len(writes),3)

    def test_explicit_label_tree_matches_count_dp(self):
        # Independent enumerator branches on actual binary masks, not HG counts.
        from functools import lru_cache
        n,t,threshold=8,5,422
        worlds=tuple(sum(1<<i for i in c) for c in combinations(range(n),t))
        @lru_cache(None)
        def brute(groups,possible,b):
            counts=[(possible[0]&g).bit_count() for g in groups]
            if sum(max(0,200*tt-94*g.bit_count()) for g,tt in zip(groups,counts))>threshold:return 1.0
            if not b:return 0.0
            best=0
            for i,(g,tt) in enumerate(zip(groups,counts)):
                bits=[j for j in range(n) if (g>>j)&1]
                if tt in (0,len(bits)):continue
                for a in range(1,len(bits)//2+1):
                    q=sum(1<<j for j in bits[-a:]); buckets={}
                    for w in possible:buckets.setdefault((w&q).bit_count(),[]).append(w)
                    child=tuple(sorted(groups[:i]+groups[i+1:]+(q,g^q)))
                    value=sum(len(ws)/len(possible)*brute(child,tuple(ws),b-1) for ws in buckets.values())
                    best=max(best,value)
            return best
        self.assertAlmostEqual(Policy(threshold=threshold).value(((8,5),),2),brute((255,),worlds,2))


def benchmark():
    results=[]
    for grid,expected in [(8,.6867273708107092),(4,.7697652778157258),(1,.999660808660013)]:
        p=Policy(grid=grid);started=perf_counter();v=p.value(((80,50),),4)
        assert abs(v-expected)<1e-12,(grid,v)
        results.append({'grid':grid,'conditional_reach_probability':v,'seconds':perf_counter()-started,
                        'states':p.value.cache_info().currsize})
        p.value.cache_clear();p.best_one.cache_clear();p.distribution.cache_clear();p.secondary.cache_clear()
        del p;gc.collect()
    assert abs(results[-1]['conditional_reach_probability']-(1-comb(50,15)/comb(80,15)))<1e-12
    return {'evidence_class':'hypothetical_conditional_calculation_not_public_performance',
            'scenario':{'n':80,'true':50,'uniform_labels':True,'split_queries_after_total':4},'results':results}


if __name__=='__main__':
    import sys
    if '--benchmark' in sys.argv:
        print(json.dumps(benchmark(),indent=2))
    else:
        unittest.main()
