"""Small exact problems exercise gates without fabricated public observations."""
import copy
import itertools
import json
import math
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import *
import campaign


def action(i, kind='add', probability=.7):
    return {'candidate_id':i,'order_id':f'o{i}','kind':kind,
            'add_rid':f'a{i}' if kind!='delete' else None,
            'remove_rid':f'r{i}' if kind!='add' else None,
            'delta_p':{'add':1,'delete':-1,'swap':0}[kind],
            'probabilities':{'add':{'0':1-probability,'1':probability},
                'delete':{'-1':1-probability,'0':probability},
                'swap':{'-1':.1,'0':1-probability-.1,'1':probability}}[kind]}


def small_system():
    return {'ground_truth_positives':2,'variables':[{'order_id':f'o{i}','rid':rid}
            for i in (1,2) for rid in (f'r{i}',f'a{i}')],
            'equations':[{'indices':[0,2],'tp':1}]}


class ScoreTests(unittest.TestCase):
    def test_public_score_mapping(self):
        self.assertEqual(infer_tp('0.927717',1045),969)
        self.assertEqual(infer_tp('0.911941',1125),989)

    def test_score_validation(self):
        for score in ('NaN','Infinity','-0.1','1.000001','0.9277171','0.927718'):
            with self.assertRaises(ValueError): infer_tp(score,1045)

    def test_target_math(self):
        self.assertGreater(f1(13,0),.94)
        self.assertLess(f1(12,0),.94)
        self.assertGreater(f1(25,25),.94)
        self.assertLess(f1(20,20),.94)
        self.assertGreater(f1(0,-1),f1())  # zero TP deletion is beneficial, not a no-op

    def test_signed_distribution_against_enumeration(self):
        actions=[action(1,'add'),action(2,'delete'),action(3,'swap')]
        exact={}
        for outcomes in itertools.product(*[list(a['probabilities'].items()) for a in actions]):
            dt=sum(int(x[0]) for x in outcomes)
            exact[dt]=exact.get(dt,0)+math.prod(x[1] for x in outcomes)
        actual=distribution(actions)
        for dt in exact: self.assertAlmostEqual(actual[dt],exact[dt])
        self.assertAlmostEqual(sum(actual.values()),1)

    def test_impossible_support(self):
        a=action(1);a['probabilities']={'-1':1.}
        with self.assertRaises(ValueError): distribution([a])


class EquationTests(unittest.TestCase):
    def test_feasible_bounds_and_conflict(self):
        eq=Equations(small_system())
        self.assertTrue(eq.feasible())
        self.assertEqual(eq.bounds_for([action(1,'swap')]),{'min':-1,'max':1})
        self.assertFalse(eq.feasible(eq.coeff([action(1,'swap')]),2))
        system=small_system();system['equations'].append({'indices':[0,2],'tp':0})
        self.assertFalse(Equations(system).feasible())

    def test_oracle_matches_exhaustive_labels_and_actions(self):
        acts=[action(1,'swap'),action(2,'delete')]
        eq=Equations(small_system())
        best=f1()
        for labels in itertools.product((0,1),repeat=4):
            if sum(labels)!=2 or labels[0]+labels[2]!=1: continue
            for selected in itertools.product((0,1),repeat=2):
                dt=selected[0]*(labels[1]-labels[0])-selected[1]*labels[2]
                dp=-selected[1]
                best=max(best,f1(dt,dp))
        self.assertAlmostEqual(eq.oracle(acts)['f1_upper'],best)

    def test_existing_observation_is_hard(self):
        eq=Equations(small_system(),[{'indices':[1,2],'tp':2}])
        self.assertEqual(eq.bounds_for([action(1,'swap')]),{'min':1,'max':1})


class PolicyTests(unittest.TestCase):
    def test_split_conservation(self):
        leaves=[{'ids':[1,2,3,4],'delta_tp':-1}]
        entry={'kind':'split','parent_ids':[1,2,3,4],'ids':[1,2]}
        result=apply_observation(leaves,entry,1)
        self.assertEqual(sum(x['delta_tp'] for x in result),-1)
        self.assertEqual(result[1]['delta_tp'],-2)
        self.assertEqual(leaves,[{'ids':[1,2,3,4],'delta_tp':-1}])

    def test_no_leaf_overlap(self):
        byid={1:action(1)}
        with self.assertRaises(ValueError): best_union([{'ids':[1],'delta_tp':1}]*2,byid)

    def test_best_union_includes_safe_delete(self):
        byid={1:action(1),2:action(2,'delete'),3:action(3,'swap')}
        leaves=[{'ids':[1],'delta_tp':1},{'ids':[2],'delta_tp':0},{'ids':[3],'delta_tp':-1}]
        self.assertEqual(best_union(leaves,byid)['ids'],[1,2])

    def test_prefix_suffix_options(self):
        ids=[1,2,3,4]
        result=next_queries([], [{'ids':ids,'delta_tp':2}],[],{i:action(i) for i in ids})
        self.assertEqual({tuple(q['ids']) for q in result},{(1,),(4,),(1,2),(3,4)})

    def test_conditional_branch_normalization(self):
        class Fake:
            def coeff(self,actions): return actions
            def feasible(self,coeff,value): return value!=0
        byid={i:action(i) for i in range(1,5)}
        q={'kind':'split','parent_ids':[1,2,3,4],'parent_delta_tp':2,'ids':[1,2]}
        result=evaluate_query(q,[{'ids':[1,2,3,4],'delta_tp':2}],byid,Fake())
        self.assertAlmostEqual(sum(b['conditional_weight'] for b in result['branches']),1)
        self.assertTrue(all(b['delta_tp']!=0 for b in result['branches']))


class BudgetAndGateTests(unittest.TestCase):
    def test_audit_cli_does_not_create_missing_output_directory(self):
        with tempfile.TemporaryDirectory(prefix='v152_readonly_test_') as tmp:
            out=Path(tmp)/'not-created'
            with patch.object(sys,'argv',['campaign.py','--out',str(out),'audit']), \
                    patch.object(campaign,'audit',return_value={'read_only':True}), redirect_stdout(io.StringIO()):
                campaign.main()
            self.assertFalse(out.exists())

    def test_false_addition_is_not_a_ban_on_safe_fp_deletion(self):
        self.assertIsNone(campaign.exclusion_reason('remove_rid',['v149_decoded_false']))
        self.assertIsNotNone(campaign.exclusion_reason('add_rid',['v149_decoded_false']))
        self.assertIsNotNone(campaign.exclusion_reason('remove_rid',['v117_deletion_pool']))

    def test_skip_redundant_merge_without_unlocking_multiple_risky_attempts(self):
        class Fake:
            def oracle(self,actions): return {'target_possible_under_equations':True,'f1_upper':.95}
        byid={1:action(1),2:action(2)}
        champ={'tp':TP0,'predictions':P0,'score':f1()}
        stamp=(datetime.now(TZ)-timedelta(days=2)).isoformat()
        manifest={'files':[{'probe_id':'p01','kind':'group','ids':[1]}]}
        ledger={'records':[{'probe_id':'p01','status':'accepted','submitted_at':stamp} for _ in range(8)]}
        state=({}, {'groups':[]}, manifest, ledger, byid, Fake(), [{'ids':[1],'delta_tp':0}],['G01'],{},champ)
        with tempfile.TemporaryDirectory(prefix='v152_policy_test_') as tmp:
            out=Path(tmp);write(out/'campaign.json',{'synthetic_test':True})
            with patch.object(campaign,'context',return_value=state), patch.object(campaign,'risk_choice',
                    return_value={'kind':'risk','ids':[2],'expected_f1':f1(1,1)}):
                result=campaign.recommend(out)
            self.assertEqual(result['decision'],'submit_risk')
            self.assertTrue(result['redundant_merge_skipped'])
            manifest['files'][0]['kind']='risk'
            with patch.object(campaign,'context',return_value=state), patch.object(campaign,'risk_choice') as choose:
                result=campaign.recommend(out)
                choose.assert_not_called()
            self.assertFalse(result['allow_submission'])

    def test_failures_count_toward_daily_and_total(self):
        stamp=datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
        with self.assertRaises(ValueError): check_budget([{'submitted_at':stamp,'status':'failed'}]*2,stamp)
        old=(datetime.now(TZ)-timedelta(days=2)).isoformat()
        with self.assertRaises(ValueError): check_budget([{'submitted_at':old,'status':'failed'}]*10,stamp)
        with self.assertRaises(ValueError): check_budget([],'2026-09-01T10:00:00')

    def test_future_timestamp(self):
        with self.assertRaises(ValueError): check_budget([],(datetime.now(TZ)+timedelta(days=1)).isoformat())

    def test_failed_or_partial_training_never_builds(self):
        for report in ({'complete':False,'gate_passed':True},{'complete':True,'gate_passed':False},
                       {'complete':True,'gate_passed':True,'smoke_only':True}):
            with tempfile.TemporaryDirectory(prefix='v152_gate_test_') as tmp:
                out=Path(tmp);write(out/'model_comparison.json',report)
                result=campaign.build(out)
                self.assertFalse(result['allow_submission'])
                self.assertFalse(list(out.glob('*.csv')))

    def test_source_hash_tampering(self):
        with tempfile.TemporaryDirectory(prefix='v152_hash_test_') as tmp:
            out=Path(tmp);write(out/'candidate_catalog.json',{'x':1})
            write(out/'campaign.json',{'baseline':{'sha256':sha(BASE)},'frozen_files':{'candidate_catalog.json':sha(out/'candidate_catalog.json')}})
            write(out/'candidate_catalog.json',{'x':2})
            with self.assertRaises(ValueError): frozen_load(out)

    def test_actual_baseline_is_read_only_verified(self):
        b,seq,roots,nodes=baseline()
        self.assertEqual((b['tp'],len(seq),len(nodes)),(969,546,1045))
        self.assertEqual(sha(BASE),b['sha256'])

    def test_action_validation(self):
        roots={'o1':[{'@rid':'r1'}]};nodes={('o1','r1'):{'@rid':'r1'}}
        validate_actions([action(1,'swap')],roots,nodes)
        with self.assertRaises(ValueError): validate_actions([action(1,'delete')],roots,nodes)
        with self.assertRaises(ValueError): validate_actions([action(1),action(1)],roots,nodes)


if __name__=='__main__': unittest.main()
