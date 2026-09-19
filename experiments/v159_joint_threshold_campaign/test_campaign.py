"""Regression tests use temporary ledgers only; no synthetic feedback enters V159."""
import copy
import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch
from bridge import *
from assessment import above, score, exact_interval, minimum_perfect_trials, evaluate_snapshot
C = module('_v159_tested_campaign', HERE/'campaign.py')
Campaign, initialize = C.Campaign, C.initialize


class ArithmeticTests(unittest.TestCase):
    def test_exact_threshold_equality_does_not_pass(self):
        self.assertEqual(score(469077,998956),Fraction('0.938154'))
        self.assertFalse(above(469077,998956))
        self.assertTrue(above(469078,998956))

    def test_addition_and_auxiliary_thresholds(self):
        self.assertFalse(above(986,1059)) # 14 perfect additions
        self.assertTrue(above(987,1060)) # 15 perfect additions
        self.assertFalse(above(982,1050)) # best auxiliary + 8 additions
        self.assertTrue(above(983,1051)) # best auxiliary + 9 additions
        self.assertEqual(f'{float(score(983,1051)):.6f}','0.938425')

    def test_exact_interval_and_minimum_evidence(self):
        self.assertEqual(minimum_perfect_trials(),17)
        self.assertLess(exact_interval(16,16)[0],.8)
        self.assertGreater(exact_interval(17,17)[0],.8)
        self.assertAlmostEqual(exact_interval(5,5)[0],.4781762498950185)
        self.assertEqual(exact_interval(0,3)[0],0)
        self.assertIsNone(exact_interval(0,0))
        with self.assertRaises(ValueError): exact_interval(4,3)

    def test_missing_evidence_never_invents_success_or_bootstrap(self):
        cap=dict(integrity_errors=[],capacity_sufficient=False,
                 complete_comparable_observed_campaign_trials=0)
        r=evaluate_snapshot(cap,{},True)
        self.assertFalse(r['passed'])
        self.assertIn('original_deadline_expired',r['reasons'])
        self.assertEqual(r['bootstrap']['executed_iterations'],0)
        self.assertIsNone(r['route_results']['A']['success_rate'])
        cap['integrity_errors']=['prediction source missing']
        self.assertIn('cached_prediction_integrity_failed',evaluate_snapshot(cap,{})['reasons'])


class LiveEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=Campaign()
        cls.result=cls.c.evaluation()

    def test_current_feedback_replay_and_budget(self):
        s=self.c.audit()
        self.assertEqual(s['remaining_submissions'],6)
        self.assertEqual(s['new_attempts_used'],0)
        self.assertEqual(s['actual_champion']['score'],'0.930589')
        self.assertEqual(s['safe_feasible_worlds'],1)
        rows=self.result['mathematics']['safe_replay']
        self.assertEqual([(r['score'],r['tp']) for r in rows],[('0.929119',970),('0.930589',972)])

    def test_pool_counts_exclusions_and_oracles(self):
        catalog=self.c.catalog;math=self.result['mathematics']
        self.assertEqual({a['candidate_id'] for a in catalog['excluded']},{1,27})
        self.assertEqual(self.c.eq.bound(catalog['main']),(17,17))
        self.assertEqual(len({a['order_id'] for a in catalog['main']+catalog['auxiliary']}),80)
        self.assertAlmostEqual(math['route_a_oracle']['f1_upper'],.9392212725546059)
        self.assertAlmostEqual(math['joint_oracle']['f1_upper'],.9424631478839752)

    def test_all_auxiliary_feedbacks_have_unique_integer_tp(self):
        branches=self.result['mathematics']['auxiliary_feedback_branches']
        self.assertEqual([r['delta_tp'] for r in branches],list(range(-5,3)))
        for row in branches:
            self.assertEqual(infer(row['score'],row['predictions']),row['tp'])
            self.assertFalse(row['individual_auxiliary_labels_inferred'])

    def test_group_total_does_not_label_members(self):
        aux=self.c.catalog['auxiliary']
        eq=Equations(self.c.previous.system,
                     self.c.previous.observations+[(self.c.eq.coeff(aux),0)])
        bounds=[eq.bound([a]) for a in aux]
        self.assertEqual(eq.bound(aux),(0,0))
        self.assertTrue(any(lo != hi for lo,hi in bounds))
        # The complement count is exact only as a group, never a guessed label.
        main=self.c.catalog['main'];first=main[:10];rest=main[10:]
        low,high=self.c.eq.bound(first)
        for k in range(low,high+1):
            if not self.c.eq.possible(first,k): continue
            e=Equations(self.c.previous.system,
                        self.c.previous.observations+[(self.c.eq.coeff(first),k)])
            self.assertEqual(e.bound(rest),(17-k,17-k))

    def test_capacity_and_holdout_integrity(self):
        cap=self.c.capacity
        self.assertEqual(cap['labeled_orders'],1634)
        self.assertEqual(cap['order_only_disjoint_task_upper'],2)
        self.assertEqual(cap['integrity_errors'],[])
        self.assertEqual([r['valid_orders'] for r in cap['folds']],[782,213,213,213,213])
        self.assertEqual(cap['complete_comparable_observed_campaign_trials'],0)

    def test_solver_timeout_is_not_an_upper_bound(self):
        import evidence
        class Timeout:
            status=1
        with patch.object(evidence,'milp',return_value=Timeout()):
            with self.assertRaisesRegex(ValueError,'timeout'):
                self.c.eq.oracle(self.c.catalog['auxiliary'],972,1045)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v159-test-')
        self.out=Path(self.tmp.name)
        for name in ('campaign.json','candidate_catalog.json','validation_capacity.json',
                     'online_scores.json','submission_manifest.json','evaluation.json','evaluation_seal.json'):
            shutil.copy2(HERE/name,self.out/name)

    def tearDown(self):
        self.tmp.cleanup()

    def c(self): return Campaign(self.out)

    def fake_registered_baseline(self):
        # Test-only manifest. Production prepare never emits this artifact.
        c=self.c();base=c.cfg['baseline']
        file=self.out/'synthetic_registered_baseline.csv'
        shutil.copy2(base['file'],file)
        entry=dict(probe_id='temporary_test_only',file=str(file),sha256=sha(file),
                   predictions=1045,actions=[],kind='final')
        write(self.out/'submission_manifest.json',dict(files=[entry]))
        return entry

    def record(self,score='0.930589',attempt='synthetic-1',failed=False):
        return self.c().record('temporary_test_only',attempt,score,
                              '2026-09-13T12:00:00+08:00','SYNTHETIC TEST IN TEMP DIRECTORY',failed)

    def test_prepare_is_non_emitting_and_read_only(self):
        before={p.name:sha(p) for p in self.out.iterdir() if p.is_file()}
        s=self.c().prepare()
        self.assertFalse(s['emitted'])
        self.assertEqual(s['remaining_submissions'],6)
        self.assertEqual(before,{p.name:sha(p) for p in self.out.iterdir() if p.is_file()})
        self.assertEqual(list(self.out.glob('*.csv')),[])

    def test_unknown_feedback_cannot_create_attempt(self):
        with self.assertRaisesRegex(ValueError,'No registered'):
            self.record()
        self.assertEqual(read(self.out/'online_scores.json')['records'],[])

    def test_idempotency_and_true_duplicate_attempt_cost(self):
        self.fake_registered_baseline()
        self.assertEqual(self.record()['remaining_submissions'],5)
        self.assertTrue(self.record()['idempotent'])
        self.assertEqual(self.record(attempt='synthetic-2')['remaining_submissions'],4)
        with self.assertRaisesRegex(ValueError,'Changed duplicate'):
            self.record(score='0.930588')

    def test_failed_and_conflicting_feedback_pause_and_preserve_champion(self):
        self.fake_registered_baseline()
        s=self.record('0.931546') # TP=973, contradicts this exact baseline's score
        self.assertEqual(s['decision'],'pause_failed_or_anomalous_attempt')
        self.assertEqual(s['remaining_submissions'],5)
        self.assertEqual(s['actual_champion']['score'],'0.930589')
        self.assertEqual(self.record(None,'synthetic-failed',True)['remaining_submissions'],4)

    def test_final_reserved_even_with_one_slot_remaining(self):
        self.fake_registered_baseline()
        for i in range(5): self.record(attempt=f'synthetic-{i}')
        s=self.c().prepare()
        self.assertEqual(s['remaining_submissions'],1)
        self.assertEqual(s['reserved_final'],1)
        self.assertFalse(s['emitted'])

    def test_tampered_evaluation_cannot_enable_prepare(self):
        r=read(self.out/'evaluation.json');r['passed']=True
        write(self.out/'evaluation.json',r)
        with self.assertRaisesRegex(ValueError,'Evaluation result changed'): self.c().prepare()
        write(self.out/'evaluation_seal.json',dict(sha256=sha(self.out/'evaluation.json')))
        with self.assertRaisesRegex(ValueError,'no valid release certificate'): self.c().prepare()

    def test_protected_source_and_catalog_drift_rejected(self):
        p=self.out/'source';p.write_text('initial',encoding='utf-8')
        cfg=read(self.out/'campaign.json');cfg['source_hashes'][str(p)]=sha(p)
        write(self.out/'campaign.json',cfg);p.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'): self.c()

    def test_catalog_drift_rejected(self):
        cat=read(self.out/'candidate_catalog.json');cat['main_total_true']=18
        write(self.out/'candidate_catalog.json',cat)
        with self.assertRaisesRegex(ValueError,'Frozen catalog changed'): self.c()

    def test_partial_initialization_never_resets_ledger(self):
        (self.out/'campaign.json').unlink()
        before=sha(self.out/'online_scores.json')
        with self.assertRaisesRegex(ValueError,'refusing to reset'): initialize(self.out)
        self.assertEqual(sha(self.out/'online_scores.json'),before)

    def test_csv_byte_or_metadata_changes_rejected(self):
        e=self.fake_registered_baseline()
        p=Path(e['file']);p.write_bytes(b'\xef\xbb\xbf'+p.read_bytes())
        with self.assertRaises(ValueError): self.record()
        self.assertEqual(read(self.out/'online_scores.json')['records'],[])

    def test_deadline_cannot_release_prepared_file(self):
        with patch.object(Campaign,'expired',return_value=True):
            self.assertFalse(self.c().prepare()['emitted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
