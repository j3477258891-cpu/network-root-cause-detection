"""Regression tests: synthetic feedback is confined to temporary directories."""
import copy, json, shutil, tempfile, unittest
from unittest.mock import patch
import numpy as np
from common import *
from evidence import Equations
C=module('_v158_campaign_tests',HERE/'campaign.py')
R=module('_v158_research',HERE/'research.py')
P=module('_v158_policy',HERE/'policy.py')

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v158-test-');self.out=Path(self.temp.name)
        for name in ('campaign.json','historical_equations.json','online_scores.json','submission_manifest.json'):
            shutil.copy2(HERE/name,self.out/name)
        # Always begin from zero hypothetical attempts, regardless of later live use.
        write(self.out/'online_scores.json',{'records':[]})
        manifest=read(self.out/'submission_manifest.json')
        manifest['files']=[e for e in manifest['files'] if e['probe_id']=='safe_probe']
        write(self.out/'submission_manifest.json',manifest)
    def tearDown(self):self.temp.cleanup()
    def c(self):return C.Campaign(self.out)
    def record(self,pid,score,attempt='synthetic-1',failed=False):
        return self.c().record(pid,attempt,score,'2026-09-13T12:00:00+08:00','TEMPORARY SYNTHETIC UNIT TEST',failed)

    def test_all_four_feedback_branches_then_actual_final(self):
        for score,expected_ids in [('0.927203',[1,2,6,8,11,15]),('0.928161',[1,4,8,10,11,15]),
                                   ('0.929119',[1,2,3,8,11,15]),('0.930077',[1,4,7,8,11,15])]:
            with self.subTest(score=score):
                write(self.out/'online_scores.json',{'records':[]})
                original=read(HERE/'submission_manifest.json')['files']
                write(self.out/'submission_manifest.json',{'files':[x for x in original if x['probe_id']=='safe_probe']})
                (self.out/'v158_safe_final_utf8.csv').unlink(missing_ok=True)
                s=self.record('safe_probe',score);self.assertEqual(s['remaining_submissions'],7)
                self.assertEqual(s['safe_feasible_worlds'],1)
                self.assertFalse(s['safe_baseline_publicly_confirmed'])
                prepared=self.c().prepare()['file'];self.assertEqual(prepared['all_ids'],expected_ids)
                self.assertEqual(prepared['predictions'],1045)
                self.c().validate(prepared)
                self.assertEqual(self.c().status()['remaining_submissions'],7) # authoring costs zero
                s=self.record('safe_final','0.930589','synthetic-2')
                self.assertTrue(s['safe_baseline_publicly_confirmed']);self.assertEqual(s['remaining_submissions'],6)
                self.assertEqual(s['actual_champion']['score'],'0.930589')

    def test_idempotency_and_real_duplicate_attempt_cost(self):
        self.record('safe_probe','0.927203')
        self.assertEqual(self.record('safe_probe','0.927203')['remaining_submissions'],7)
        self.assertEqual(self.record('safe_probe','0.927203','another-real-attempt')['remaining_submissions'],6)
        with self.assertRaises(ValueError):self.record('safe_probe','0.928161')

    def test_failed_attempt_costs_one_and_pauses(self):
        s=self.record('safe_probe',None,failed=True)
        self.assertEqual(s['remaining_submissions'],7);self.assertIn('pause',s['decision'])
        self.assertEqual(len(read(self.out/'online_scores.json')['records']),1)

    def test_anomaly_preserved_without_forcing_tp(self):
        s=self.record('safe_probe','0.940000')
        self.assertEqual(s['remaining_submissions'],7);self.assertIn('pause',s['decision'])
        self.assertEqual(self.c().champ['score'],'0.929254')
        self.assertEqual(read(self.out/'online_scores.json')['records'][0]['status'],'anomaly')

    def test_timestamp_and_failed_score_validation(self):
        with self.assertRaises(ValueError):self.c().record('safe_probe','x','0.927203','2026-09-13T12:00:00','test')
        with self.assertRaises(ValueError):self.record('safe_probe','0.927203',failed=True)
        self.assertEqual(self.c().remaining,8)

    def test_no_research_pool_before_safe_final_or_failed_gate(self):
        with self.assertRaises(ValueError):P.build_pool(self.c())
        self.record('safe_probe','0.927203');self.c().prepare();self.record('safe_final','0.930589','second')
        with self.assertRaises(ValueError):P.build_pool(self.c())
        self.assertFalse((self.out/'candidate_catalog.json').exists())

    def test_expired_window_does_not_reset(self):
        cfg=read(self.out/'campaign.json');cfg['deadline_at']='2020-01-01T00:00:00+08:00';write(self.out/'campaign.json',cfg)
        with self.assertRaises(ValueError):self.c().check_deadline()
        self.assertEqual(read(self.out/'campaign.json')['deadline_at'],cfg['deadline_at'])

    def test_changed_registered_csv_rejected(self):
        self.record('safe_probe','0.927203');e=self.c().prepare()['file']
        path=Path(e['file']);path.write_bytes(b'\xef\xbb\xbf'+path.read_bytes())
        with self.assertRaises(ValueError):self.c().validate(e)

    def test_conflicting_action_order_rejected(self):
        e=self.c().entries['safe_probe']
        with self.assertRaises(ValueError):apply_actions(e['base_file'],[e['actions'][0],e['actions'][0]])

    def test_gate_result_tamper_rejected(self):
        c=self.c();p=self.out/'research/results.json'
        r=dict(complete=True,passed=False,criteria={'positive':False},protocol_sha256=digest(c.cfg['protocol']))
        write(p,r);write(p.parent/'result_seal.json',{'sha256':sha(p)})
        r['passed']=True;write(p,r)
        with self.assertRaises(ValueError):self.c().status()

    def test_prepared_safe_file_is_idempotent(self):
        self.record('safe_probe','0.927203');a=self.c().prepare()['file'];b=self.c().prepare()['file']
        self.assertEqual(a,b);self.assertEqual(self.c().remaining,7)

class ArithmeticTests(unittest.TestCase):
    def test_exact_target_and_addition_limit(self):
        self.assertAlmostEqual(.94*(1044+1045)-1944,19.66)
        self.assertGreaterEqual(f1(982,1045),.94)
        self.assertLess(f1(989,1062),.94)
        self.assertGreaterEqual(f1(991,1064),.94)

    def test_final_merge_uses_counts_not_individual_labels(self):
        cat={1:{'delta_p':1},2:{'delta_p':1},3:{'delta_p':-1}}
        best=P.best_merge([{'ids':(1,2),'dt':1},{'ids':(3,),'dt':0}],cat,{'tp':972,'predictions':1045})
        self.assertEqual(best['ids'],[1,2,3])
        self.assertEqual(best['tp'],973)
        self.assertNotEqual(best['ids'],[1,3])

    def test_target_ranking_does_not_read_labels(self):
        meta=[dict(order_id=str(i),delta_p=1,kind='add',add_rid='x',remove_rid=None) for i in range(30)]
        p=np.array([[0,.1,.9]]*30)
        selected=R.select(meta,p,list(range(546)))
        self.assertEqual(len(selected),30)
        self.assertEqual(len({meta[i]['order_id'] for i in selected}),30)

    def test_station_bootstrap_pairing_and_positive_signal(self):
        candidate=np.array([[i,2,2,1,0,1] for i in range(10)])
        control=candidate.copy();control[:,-1]=0
        result=R.bootstrap(candidate,{'control':control},list(range(10)),2000)
        self.assertEqual(result['iterations'],2000);self.assertGreater(result['absolute_lower95'],0)
        self.assertGreater(result['relative_lower95']['control'],0)
        with self.assertRaises(ValueError):R.bootstrap(candidate,{'control':control[::-1]},list(range(10)))

    def test_impossible_outcome_overrides_model(self):
        self.assertTrue(np.array_equal(R.outcome_filter([0,.001,.999],0,0),[0,1,0]))

    def test_signed_convolution(self):
        p=R.convolve_counts([[.5,.5,0],[0,.5,.5]])
        self.assertEqual(p[-1],.25);self.assertEqual(p[0],.5);self.assertEqual(p[1],.25)

class IntegerTests(unittest.TestCase):
    def system(self):
        variables=[{'order_id':str(i),'rid':'node'} for i in range(1045)]
        return dict(variables=variables,equations=[{'indices':list(range(1043)),'tp':1043}])
    def test_bounds_couple_labels_and_oracle(self):
        eq=Equations(self.system())
        aa=[dict(order_id=str(i),add_rid='node',remove_rid=None,delta_p=1) for i in (1043,1044)]
        self.assertEqual(eq.bound(aa),(1,1));self.assertEqual(eq.bound(aa[:1]),(0,1))
        self.assertFalse(eq.possible(aa,2));self.assertTrue(eq.possible(aa,1))
        oracle=eq.oracle(aa,1043,1043)
        self.assertAlmostEqual(oracle['f1_upper'],1.)
        self.assertIsInstance(oracle['predictions'],int)
        json.dumps(oracle,allow_nan=False) # Persisting results must support native integer counts.
    def test_timeout_is_not_feasibility_or_success(self):
        import evidence
        class Timeout:status=1
        with patch.object(evidence,'milp',return_value=Timeout()):
            with self.assertRaisesRegex(ValueError,'unresolved'):Equations(self.system()).solve()

class GroupPolicyTests(unittest.TestCase):
    """Exercise the gated group's passing path without touching any real ledger."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v158-policy-');self.out=Path(self.temp.name)
        self.actions=[dict(candidate_id=i,order_id=str(i),delta_p=1,expected_u=.66,
            model_probabilities={f:{'-1':0.,'0':.2,'1':.8} for f in R.FAMILIES}) for i in range(1,9)]
        self.cat={'candidates':self.actions,'groups':[{'group_id':str(i),'ids':[i,i+1]} for i in (1,3,5,7)]}
        write(self.out/'candidate_catalog.json',self.cat);write(self.out/'research/results.json',{'fixture':True})
        write(self.out/'pool_gate.json',dict(passed=True,catalog_sha256=sha(self.out/'candidate_catalog.json'),
              research_sha256=sha(self.out/'research/results.json')))
        class ToyEquations:
            def oracle(self,*args):return {'f1_upper':.95}
            def bound(self,actions):return (0,len(actions))
            def possible(self,actions,value):return 0<=value<=len(actions)
        self.c=type('Fixture',(),{})()
        self.c.out=self.out;self.c.eq=ToyEquations();self.c.ledger={'records':[]};self.c.entries={}
        self.c.safe_baseline={'tp':972,'predictions':1045};self.c.champ=dict(self.c.safe_baseline);self.c.remaining=6
    def tearDown(self):self.temp.cleanup()
    def observed(self,pid,ids,dt,kind='group'):
        self.c.entries[pid]={'phase':'research','kind':kind,'ids':ids}
        self.c.ledger['records'].append({'status':'accepted','probe_id':pid,'tp':972+dt})
    def test_first_group_has_integer_feedback_and_final_reserve(self):
        s=P.recommendation(self.c)
        self.assertEqual(s['decision'],'submit_new_group');self.assertEqual(len(s['next_ids']),2)
        self.assertEqual([x['delta_tp'] for x in s['score_lookup']],[0,1,2]);self.assertTrue(s['final_slot_reserved'])
    def test_last_slot_only_measured_merge(self):
        self.observed('G1',[1,2],1);self.c.remaining=1
        s=P.recommendation(self.c);self.assertEqual(s['decision'],'prepare_new_final')
        self.assertEqual(s['next_ids'],[1,2]);self.assertEqual(s['best_count_determined_merge']['tp'],973)
    def test_split_counts_conserved_and_duplicate_score_not_double_counted(self):
        self.observed('G1',[1,2],1);self.observed('S1',[1],0,'split')
        self.c.ledger['records'].append(dict(self.c.ledger['records'][-1]))
        leaves,splits,_=P.measured_leaves(self.c,self.cat)
        self.assertEqual(splits,1);self.assertEqual(sum(x['dt'] for x in leaves),1)
        self.assertEqual(P.best_merge(leaves,{a['candidate_id']:a for a in self.actions},self.c.safe_baseline)['ids'],[2])
    def test_target_upper_failure_stops_new_queries(self):
        self.observed('G1',[1,2],1)
        self.c.eq.oracle=lambda *args:{'f1_upper':.935}
        self.assertEqual(P.recommendation(self.c)['decision'],'prepare_new_final')
    def test_group_overlap_is_never_double_counted(self):
        self.observed('G1',[1,2],1);self.observed('G2',[2,3],1)
        with self.assertRaises(ValueError):P.measured_leaves(self.c,self.cat)

if __name__=='__main__':unittest.main(verbosity=2)
