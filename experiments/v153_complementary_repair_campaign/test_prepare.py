"""File preparation does not grant quota or bypass conflict/final gates."""
import unittest
from unittest.mock import patch
from pathlib import Path
import campaign

class PreparationTests(unittest.TestCase):
    def test_daily_limit_does_not_prevent_preparation_or_grant_submission(self):
        recommendation={'decision':'submit_split','allow_submission':False,
                        'next_action':{'kind':'split','ids':[5]},'earliest_submission':'next local day'}
        with patch.object(campaign,'recommend',return_value=recommendation), \
             patch.object(campaign,'author',return_value={'file':'test.csv'}) as author, \
             patch.object(campaign.c,'sha',return_value='unchanged'), \
             patch.object(campaign.c,'write'):
            result=campaign.prepare(Path('unused'))
            author.assert_called_once()
            self.assertFalse(result['allow_submission'])
            self.assertTrue(result['prepared_only'])
            self.assertFalse(result['competition_upload_performed'])
    def test_conflicts_and_finals_cannot_be_prepared(self):
        for decision in ('pause_anomaly_recheck_same_file','submit_merge','submit_risk','stop_no_remaining_gain_retain_champion'):
            with patch.object(campaign,'recommend',return_value={'decision':decision,'next_action':{'ids':[5]}}), \
                 patch.object(campaign,'author') as author:
                with self.assertRaises(ValueError): campaign.prepare(Path('unused'))
                author.assert_not_called()
    def test_pending_file_is_not_replaced(self):
        pending={'decision':'await_existing_file_feedback','pending':{'probe_id':'p05'}}
        with patch.object(campaign,'recommend',return_value=pending),patch.object(campaign,'author') as author:
            self.assertEqual(campaign.prepare(Path('unused')),pending)
            author.assert_not_called()

if __name__=='__main__': unittest.main()
