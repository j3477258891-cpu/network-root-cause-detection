"""Use a temporary copy; hypothetical scores never touch public ledgers."""
import copy
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import campaign
from bridge import HERE, c


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v153_synthetic_test_')
        self.out = Path(self.tmp.name)
        config = c.read(HERE/'campaign.json')
        for name in set(config['frozen_files']) | {'campaign.json', 'online_scores.json', 'submission_manifest.json'}:
            shutil.copy2(HERE/name, self.out/name)
        manifest = c.read(self.out/'submission_manifest.json')
        self.assertTrue(manifest['files'], 'Generate and locally validate the first real file before integration tests')
        # Live scores are evidence, never the starting state for synthetic tests.
        # Reset only this temporary fixture so tests remain valid after real submissions.
        manifest['files'] = [manifest['files'][0]]
        c.write(self.out/'submission_manifest.json', manifest)
        c.write(self.out/'online_scores.json', {'baseline': config['baseline'], 'records': [],
                                              'note': 'SYNTHETIC TEMPORARY TEST ONLY'})
        for e in manifest['files']: shutil.copy2(HERE/e['file'], self.out/e['file'])
        self.entry = manifest['files'][0]
        self.stamp = datetime.now(c.TZ).isoformat(timespec='seconds')
        self.mock = patch.object(campaign, 'recommend', return_value={'decision': 'synthetic_review_only', 'allow_submission': False})
        self.mock.start()

    def tearDown(self):
        self.mock.stop(); self.tmp.cleanup()

    def feasible_score(self):
        config, cat, manifest, ledger, byid, eq, leaves, champ = campaign.context(self.out)
        witness = eq.solve()
        delta = round(sum(witness.x[k]*v for k, v in eq.coeff([byid[i] for i in self.entry['ids']]).items()))
        return f'{c.f1(delta, self.entry["delta_p"]):.6f}', delta

    def record(self, score, attempt='test-1', **kwargs):
        return campaign.record(self.out, self.entry['probe_id'], attempt, score, self.stamp,
                               'SYNTHETIC TEMPORARY TEST ONLY', **kwargs)

    def test_csv_structure_and_difference(self):
        byid = {a['candidate_id']: a for a in c.read(self.out/'candidate_catalog.json')['candidates']}
        result = c.validate_csv(self.out/self.entry['file'], [byid[i] for i in self.entry['ids']], self.entry['sha256'])
        self.assertEqual(result['orders'], 546)
        self.assertEqual(result['predictions'], 1045+self.entry['delta_p'])

    def test_record_reconstructs_group_count(self):
        score, dt = self.feasible_score()
        result = self.record(score)
        self.assertEqual(result['record']['status'], 'accepted')
        self.assertEqual(result['record']['delta_tp'], dt)
        *_, leaves, champ = campaign.context(self.out)
        self.assertEqual(sum(x['delta_tp'] for x in leaves), dt)

    def test_idempotent_record_does_not_consume_twice(self):
        score, _ = self.feasible_score()
        self.record(score)
        self.assertTrue(self.record(score)['idempotent'])
        self.assertEqual(len(c.read(self.out/'online_scores.json')['records']), 1)

    def test_conflicting_attempt_id_rejected(self):
        score, _ = self.feasible_score(); self.record(score)
        with self.assertRaises(ValueError): self.record('0.000001')

    def test_duplicate_actual_submission_counts_once_as_equation(self):
        score, _ = self.feasible_score()
        self.record(score); self.record(score, 'test-2')
        context = campaign.context(self.out)
        self.assertEqual(len(context[3]['records']), 2)
        self.assertEqual(len(context[-2]), 1)

    def test_invalid_score_preserved_as_anomaly(self):
        result = self.record('0.000001')
        self.assertEqual(result['record']['status'], 'anomaly')
        self.assertEqual(len(c.read(self.out/'online_scores.json')['records']), 1)

    def test_actual_repeat_resolves_anomaly_without_overwriting(self):
        score, _ = self.feasible_score()
        self.record('0.000001')
        result = self.record(score, 'test-2', resolves='test-1')
        records = c.read(self.out/'online_scores.json')['records']
        self.assertEqual(records[0]['status'], 'anomaly')
        self.assertEqual(records[0]['resolved_by'], 'test-2')
        self.assertEqual(result['record']['status'], 'accepted')

    def test_failed_attempt_uses_quota_without_equation(self):
        result = self.record(None, failed=True)
        self.assertEqual(result['record']['status'], 'failed')
        context = campaign.context(self.out)
        self.assertEqual(len(context[3]['records']), 1)
        self.assertEqual(context[-2], [])

    def test_tampered_file_pauses_record(self):
        with (self.out/self.entry['file']).open('a', encoding='utf-8') as f: f.write('\n')
        result = self.record('0.927717')
        self.assertEqual(result['record']['status'], 'anomaly')

    def test_tampered_ledger_delta_rejected(self):
        score, _ = self.feasible_score(); self.record(score)
        ledger = c.read(self.out/'online_scores.json'); ledger['records'][0]['delta_tp'] += 1
        c.write(self.out/'online_scores.json', ledger)
        with self.assertRaises(ValueError): campaign.context(self.out)

    def test_final_requires_expected_exact_tp(self):
        score, dt = self.feasible_score()
        manifest = c.read(self.out/'submission_manifest.json')
        manifest['files'][0].update(kind='merge', expected_tp=c.TP0+dt+1)
        c.write(self.out/'submission_manifest.json', manifest)
        result = self.record(score)
        self.assertEqual(result['record']['status'], 'anomaly')

    def test_record_only_reviews_never_authors_next(self):
        score, _ = self.feasible_score()
        with patch.object(campaign, 'author', side_effect=AssertionError('Must not author')):
            self.record(score)
        self.assertEqual(len(c.read(self.out/'submission_manifest.json')['files']), 1)


if __name__ == '__main__': unittest.main(verbosity=2)
