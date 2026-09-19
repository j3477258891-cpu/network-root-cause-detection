"""Synthetic outcomes are confined to temporary directories, never the live ledger."""
import unittest
import tempfile
import shutil
from unittest.mock import patch
from support import *
from engine import WitnessModel,scenarios,best_known
C=module('_v160_test_campaign',HERE/'campaign.py')
P=module('_v160_test_policy',HERE/'policy.py')

class MathTests(unittest.TestCase):
    def test_exact_target(self):
        self.assertFalse(reached(972,1045))
        self.assertTrue(reached(945,956)) # 1890/2000 = .945 exactly
        self.assertFalse(reached(944,956))
        self.assertAlmostEqual(float(TARGET)*(1044+1045)-2*972,30.105)

    def test_known_union_packing_does_not_double_count(self):
        actions=[dict(id=i,delta_p=1) for i in range(4)]
        groups=[dict(ids=[0,1],delta_tp=2),dict(ids=[1,2],delta_tp=2),dict(ids=[3],delta_tp=1)]
        m=best_known(groups,actions)
        self.assertEqual(len(m['ids']),3);self.assertEqual(m['delta_tp'],3)
        self.assertEqual(len(set(m['ids'])),3)

    def test_false_deletion_can_still_be_known_group_gain(self):
        actions=[dict(id=i,delta_p=-1) for i in range(3)]
        m=best_known([dict(ids=[0,1,2],delta_tp=-1)],actions)
        self.assertGreater(m['f1'],f1(972,1045))
        self.assertEqual(m['delta_tp'],-1)

class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v160-test-');self.out=Path(self.tmp.name)
        for name in ('campaign.json','candidate_catalog.json','execution_window.json'):
            shutil.copy2(HERE/name,self.out/name)
        write(self.out/'online_scores.json',dict(records=[]));write(self.out/'submission_manifest.json',dict(files=[]))
    def tearDown(self):self.tmp.cleanup()
    def c(self):return C.Campaign(self.out)
    def record(self,pid,score,attempt='synthetic-1',failed=False):
        return self.c().record(pid,attempt,score,'2026-09-14T08:00:00+08:00','SYNTHETIC TEMPORARY TEST',failed)
    def first(self):return self.c().prepare_next()['file']
    def two(self,b='0.931927',cdt=0):
        self.first();self.record('probe01_aux05',b)
        second=self.c().prepare_next()['file']
        self.record(second['probe_id'],f'{f1(972+cdt,1038):.6f}','synthetic-2')
        return self.c()

    def test_baseline_pools_and_joint_oracle(self):
        c=self.c();s=c.audit()
        self.assertEqual(s['remaining_submissions'],12)
        self.assertEqual(s['actual_champion']['score'],'0.930589')
        self.assertEqual(len(c.actions),92)
        self.assertEqual(c.eq.bound(c.pick(c.pools['A'])),(17,17))
        self.assertAlmostEqual(s['optimistic_pool_upper']['f1_upper'],.9494274809160306)
        self.assertEqual(len({a['order_id'] for a in c.actions}),92)

    def test_first_and_second_complete_lookup_and_no_early_second(self):
        first=self.first()
        self.assertEqual(first['predictions'],1042)
        self.assertEqual(len(first['score_lookup']),8)
        self.assertEqual([r['delta_tp'] for r in first['score_lookup']],list(range(-5,3)))
        self.assertEqual(self.c().prepare_next()['file']['probe_id'],'probe01_aux05')
        self.assertNotIn('probe02_tail12',self.c().entries)
        self.record('probe01_aux05','0.931927')
        second=self.c().prepare_next()['file'];self.assertEqual(second['predictions'],1038)
        self.assertEqual([r['delta_tp'] for r in second['score_lookup']],list(range(-11,5)))
        for row in first['score_lookup']+second['score_lookup']:
            self.assertEqual(infer(row['score'],row['predictions']),row['tp'])

    def test_all_aux_feedbacks_valid_without_false_labels(self):
        first=self.first();c=self.c()
        for row in first['score_lookup']:
            eq=Equations(c.parent.system,c.parent.observations+[(c.eq.coeff(c.pick(c.pools['B'])),row['delta_tp'])])
            self.assertIsNotNone(eq.solve())
            self.assertEqual(eq.bound(c.pick(c.pools['B'])),(row['delta_tp'],row['delta_tp']))
        eq=Equations(c.parent.system,c.parent.observations+[(c.eq.coeff(c.pick(c.pools['B'])),0)])
        self.assertTrue(any(eq.bound([c.actions[i]])[0]!=eq.bound([c.actions[i]])[1] for i in c.pools['B']))

    def test_ledger_idempotency_and_real_duplicate_cost(self):
        self.first();self.record('probe01_aux05','0.931927')
        self.assertTrue(self.record('probe01_aux05','0.931927')['idempotent'])
        self.assertEqual(self.record('probe01_aux05','0.931927','synthetic-repeat')['remaining_submissions'],10)
        with self.assertRaises(ValueError):self.record('probe01_aux05','0.933845')

    def test_failed_or_conflicting_score_is_preserved_and_pauses(self):
        self.first();s=self.record('probe01_aux05','0.945000')
        self.assertEqual(s['decision'],'pause_failed_or_anomalous_attempt')
        self.assertEqual(s['remaining_submissions'],11)
        self.assertEqual(s['actual_champion']['score'],'0.930589')
        s=self.record('probe01_aux05',None,'synthetic-failure',True)
        self.assertEqual(s['remaining_submissions'],10)
        self.assertEqual(self.c().prepare_next()['decision'],'pause_failed_or_anomalous_attempt')

    def test_csv_validation_bom_and_metadata_guard(self):
        e=self.first();p=Path(e['file']);p.write_bytes(b'\xef\xbb\xbf'+p.read_bytes())
        with self.assertRaises(ValueError):self.record('probe01_aux05','0.931927')
        self.assertEqual(read(self.out/'online_scores.json')['records'],[])

    def test_budget_reserves_final_and_unknown_feedback_rejected(self):
        self.first();self.record('probe01_aux05','0.931927')
        ledger=read(self.out/'online_scores.json');template=ledger['records'][0]
        ledger['records']=[dict(template,attempt_id=f'synthetic-{i}') for i in range(11)]
        write(self.out/'online_scores.json',ledger)
        c=self.c();self.assertEqual(c.remaining,1)
        self.assertEqual(c.audit()['decision'],'prepare_final')
        with self.assertRaisesRegex(ValueError,'Final slot'):c.emit('forbidden',c.pools['C'],'total')
        with self.assertRaisesRegex(ValueError,'Unknown'):c.record('unknown','x','0.930589',now(),'test')

    def test_dates_daily_estimate_and_deadline_not_reset(self):
        c=self.two();s=c.audit()
        # Source timestamp is deliberately fixed; avoid assuming test execution date.
        self.assertEqual(s['daily_limit'],2)
        before=read(self.out/'campaign.json')['deadline_at']
        self.assertEqual(C.initialize(self.out)['deadline_at'],before)
        with patch.object(C.Campaign,'expired',return_value=True):
            self.assertEqual(self.c().prepare_next()['decision'],'stop_deadline')

    def test_full_history_worlds_and_nonoptimal_feedback(self):
        c=self.two();model=WitnessModel(c.eq,c.actions);rng=np.random.default_rng(SEED)
        for family in ('catboost','v38_control','mixture_temperature2'):
            d,w=model.draw(rng,family)
            self.assertEqual(int(d[c.pools['A']].sum()),17)
            self.assertEqual(int(d[c.pools['B']].sum()),0)
            self.assertEqual(int(d[c.pools['C']].sum()),0)
            self.assertLess(float(abs(model.mat@w-model.rhs).max()),1e-6)

    def test_split_and_complement_conserve_counts(self):
        c=self.two();qs=P.query_candidates(c)
        q=next(q for q in qs if len(q['query_ids'])==1 and set(q['parent_ids'])==set(c.pools['B']))
        e=c.emit('temporary_split',q['ids'],'split',query_ids=q['query_ids'],parent_ids=q['parent_ids'],
                 anchor_ids=q['anchor_ids'],anchor_delta_tp=q['anchor_delta_tp'])
        row=e['score_lookup'][0];self.record('temporary_split',row['score'],'synthetic-3')
        updated=self.c();find=lambda ids:next(g['delta_tp'] for g in updated.groups if g['ids']==sorted(ids))
        self.assertEqual(find(q['query_ids'])+find(set(q['parent_ids'])-set(q['query_ids'])),0)

    def test_fallback_never_infers_infeasibility_from_timeout(self):
        c=self.two();queries=P.query_candidates(c)
        with patch.object(P,'scenarios',side_effect=ValueError('synthetic timeout')):
            result=P.search(c,sample_count=2,rollouts=2,top_n=2)
        self.assertEqual(result['decision'],'probe')
        self.assertEqual(result['method'],'exact single-group fallback')
        self.assertGreater(len(result['score_lookup']),1)

    def test_small_full_horizon_and_real_final_file_paths(self):
        c=self.two()
        result=P.search(c,sample_count=4,rollouts=4,top_n=2,seconds=120)
        self.assertIn(result['decision'],('probe','final'))
        self.assertIn('model_assisted_final',result)
        self.assertEqual(result.get('maximum_probe_depth',9),9)
        model=result['model_assisted_final']
        self.assertLessEqual(model['tp_range'][0],model['tp_range'][1])
        final=c.prepare_final(result)
        if 'file' in final:
            self.assertEqual(final['file']['kind'],'final')
            c.validate(final['file'])
            self.assertIn('comparison',final)

    def test_target_reached_and_upper_below_target_modes(self):
        c=self.c()
        with patch.object(c,'mathematics',return_value=dict(f1_upper=.942463)):
            self.assertEqual(c.mode,'maximize_retained_score')
        c.champ.update(tp=995,predictions=1052,score='0.949427')
        c.champion_f1=f1(995,1052)
        self.assertEqual(c.audit()['decision'],'stop_target_achieved')

    def test_source_and_catalog_drift_stop(self):
        p=self.out/'protected';p.write_text('original',encoding='utf-8')
        cfg=read(self.out/'campaign.json');cfg['source_hashes'][str(p)]=sha(p)
        write(self.out/'campaign.json',cfg);p.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):self.c()

if __name__=='__main__':unittest.main(verbosity=2)
