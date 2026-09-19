"""Local-only, temporary synthetic outcomes; never update the real score ledger."""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import *
import campaign


class CampaignIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='v152_synthetic_test_')
        cls.out=Path(cls.directory.name)
        cls.b,cls.ids,cls.roots,cls.nodes=baseline()
        data=campaign.records_data()['test']
        cls.system=collect_system(data,cls.ids)
        cls.eq=Equations(cls.system)
        witness=cls.eq.solve().x
        # Select an absent node from a legal order. Its test outcome is one feasible
        # synthetic witness, never a real inferred label or a submit recommendation.
        found=None
        for order in data:
            if len(cls.roots[order['order_id']])>=8: continue
            for alarm in order['alarms']:
                if (order['order_id'],alarm['rid']) not in cls.nodes:
                    found=(order,alarm);break
            if found:break
        order,alarm=found
        cls.a={'candidate_id':1,'order_id':order['order_id'],'kind':'add','delta_p':1,
            'add_rid':alarm['rid'],'remove_rid':None,'probabilities':{'0':.5,'1':.5},
            'node':{'@rid':alarm['rid'],**{k:alarm['source'].get(k,'') for k in ('title','location','reason')}}}
        cls.dt=round(witness[cls.eq.index[order['order_id'],alarm['rid']]])
        write(cls.out/'candidate_catalog.json',{'candidates':[cls.a],'groups':[{'group_id':'G01','kind':'add','ids':[1]}]})
        write(cls.out/'equation_system.json',cls.system)
        cls.config={'baseline':cls.b,'frozen_files':{n:sha(cls.out/n) for n in ('candidate_catalog.json','equation_system.json')}}
        write(cls.out/'campaign.json',cls.config)
        cls.entry=campaign.author(cls.out,{'kind':'group','group_id':'G01','ids':[1]}, {1:cls.a}, cls.config, {'files':[]})
        # Optional visual QA retains ONLY a clearly named test preview, never
        # the synthetic CSV or its fake score ledger in the real campaign.
        if os.environ.get('V152_TEST_PREVIEW_DIR'):
            dest=Path(os.environ['V152_TEST_PREVIEW_DIR']);dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cls.out/'previews/p01.png',dest/'synthetic_structure_test_only.png')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        write(self.out/'online_scores.json',{'baseline':self.b,'records':[]})
        self.stamp=datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
        self.score=f'{f1(self.dt,1):.6f}'

    def test_artifact_roundtrip_and_structure(self):
        result=validate_csv(self.out/self.entry['file'],[self.a],self.entry['sha256'])
        self.assertEqual(result['predictions'],1046)
        self.assertTrue((self.out/'previews/p01.png').is_file())

    def test_record_and_idempotence(self):
        with patch.object(campaign,'recommend',return_value={'decision':'synthetic_test_only'}):
            first=campaign.record(self.out,'p01','test-1',self.score,self.stamp,'SYNTHETIC TEST ONLY')
            second=campaign.record(self.out,'p01','test-1',self.score,self.stamp,'SYNTHETIC TEST ONLY')
        self.assertEqual(first['record']['status'],'accepted')
        self.assertTrue(second['idempotent'])
        self.assertEqual(len(read(self.out/'online_scores.json')['records']),1)
        context=campaign.context(self.out)
        self.assertEqual(context[6],[{'ids':[1],'delta_tp':self.dt}])

    def test_duplicate_actual_attempt_counts_without_double_leaf(self):
        with patch.object(campaign,'recommend',return_value={'decision':'synthetic_test_only'}):
            for attempt in ('first','second'):
                campaign.record(self.out,'p01',attempt,self.score,self.stamp,'SYNTHETIC TEST ONLY')
        state=campaign.context(self.out)
        self.assertEqual(len(state[3]['records']),2)
        self.assertEqual(len(state[6]),1)

    def test_anomalous_score_consumes_attempt_and_pauses(self):
        with patch.object(campaign,'recommend',return_value={'decision':'synthetic_test_only'}):
            bad=campaign.record(self.out,'p01','bad','0.100000',self.stamp,'SYNTHETIC BAD SCORE')
        self.assertEqual(bad['record']['status'],'anomaly')
        self.assertEqual(campaign.recommend(self.out)['decision'],'pause_conflicting_or_invalid_observation')

    def test_confirmed_repeat_resolves_anomaly_without_erasing_attempt(self):
        with patch.object(campaign,'recommend',return_value={'decision':'synthetic_test_only'}):
            campaign.record(self.out,'p01','bad','0.100000',self.stamp,'SYNTHETIC BAD SCORE')
            campaign.record(self.out,'p01','repeat',self.score,self.stamp,'SYNTHETIC REPEAT',resolves='bad')
        records=read(self.out/'online_scores.json')['records']
        self.assertEqual(records[0]['resolved_by'],'repeat')
        self.assertEqual(len(records),2)

    def test_csv_tampering_is_detected(self):
        with self.assertRaises(ValueError): validate_csv(self.out/self.entry['file'],[self.a],'0'*64)

    def test_legacy_v117_removals_remain_policy_excluded(self):
        blocked=campaign.action_exclusions(self.nodes,self.system)
        self.assertEqual(sum('v117_deletion_pool' in reasons for reasons in blocked.values()),40)


if __name__=='__main__':unittest.main()
