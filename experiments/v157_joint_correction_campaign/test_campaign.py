import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import campaign as C
import numpy as np

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=C.Campaign()
        # Initial protocol fixture must not change when real feedback arrives.
        cls.c.s=tuple(cls.c.matrix['world_ids'])
        cls.c.ledger={'records':[]};cls.c.accepted_queries=set();cls.c.accepted_actions=set()
        cls.c.blocked=[];cls.c.remaining=6
        cls.c.champ=cls.c.cfg['baseline']
        cls.c.champ_f1=2*972/(1044+1048)

    def test_first_csv(self):
        c=self.c;x=c.entries['probe_p01'];check=c.validate(x)
        self.assertEqual(check['orders'],546);self.assertEqual(check['predictions'],1045)
        self.assertTrue((C.HERE/x['file']).read_bytes().startswith(b'order_id,output\r\n'))
        base_ids,_,base=c.e.csv.load_csv(c.cfg['baseline']['file'])
        ids,_,actual=c.e.csv.load_csv(C.HERE/x['file'])
        self.assertEqual(ids,base_ids)
        self.assertEqual(len(set(base)-set(actual)),4)
        self.assertEqual(len(set(actual)-set(base)),1)
        self.assertEqual({oid for oid,rid in set(base)^set(actual)},
                         {c.e.cat[i]['order_id'] for i in (6,10,12,13)})

    def test_all_78_worlds_and_four_query_budget(self):
        c=self.c;s=c.matrix['world_ids'];e=c.e
        sig=np.stack([e.sums(q)[s] for q in C.ROWS],axis=1)
        self.assertEqual(len(np.unique(sig,axis=0)),78)
        for target in s:
            remaining=tuple(s)
            for step,q in enumerate(C.ROWS,1):
                remaining=e.branches(remaining,q)[int(e.sums(q)[target])]
                self.assertLessEqual(4-step+1,6-step)  # remaining queries plus final reserve
                if e.merge(remaining)['f1']>=e.upper(remaining)-1e-12: break
            m=e.merge(remaining)
            self.assertGreaterEqual(m['f1'],1944/2089-1e-12)
            self.assertAlmostEqual(m['f1'],e.upper((target,)))
            self.assertEqual(sum(e.cat[i]['kind']=='delete' for i in m['ids']),3)

    def test_score_mapping_signed(self):
        rows=self.c.status()['next_score_lookup']
        self.assertEqual([x['query_delta_tp'] for x in rows],list(range(-4,2)))
        for x in rows:self.assertEqual(C.V.infer(x['score'],1045),x['tp'])
        self.assertEqual(sum(x['remaining_worlds'] for x in rows),78)

    def test_audit_and_build_preserve_ledger(self):
        names=('campaign.json','online_scores.json','submission_manifest.json')
        before=[C.V.sha(C.HERE/n) for n in names]
        self.c.status();C.build()
        self.assertEqual(before,[C.V.sha(C.HERE/n) for n in names])
        for p,h in self.c.cfg['source_hashes'].items():self.assertEqual(C.V.sha(p),h)

    def test_feedback_branches_and_idempotence_in_temporary_copy(self):
        real=C.HERE
        for outcome in self.c.status()['next_score_lookup']:
            with tempfile.TemporaryDirectory() as td:
                dst=Path(td)
                for name in ('campaign.json','candidate_catalog.json','probe_matrix.json','online_scores.json','submission_manifest.json','v157_probe_p01_utf8.csv'):
                    shutil.copy2(real/name,dst/name)
                C.V.write(dst/'online_scores.json',{'records':[]})
                with patch.object(C,'HERE',dst):
                    args=('probe_p01','synthetic-test',outcome['score'],'2026-09-12T10:00:00+08:00','Synthetic unit test, not online evidence')
                    r=C.record(*args)
                    self.assertEqual(r['record']['status'],'accepted')
                    self.assertEqual(r['state']['remaining_submissions'],5)
                    self.assertEqual(r['state']['feasible_worlds'],outcome['remaining_worlds'])
                    self.assertTrue(C.record(*args)['idempotent'])
                    self.assertEqual(len(C.read('online_scores.json')['records']),1)
                    with self.assertRaises(ValueError):
                        C.record('probe_p01','synthetic-test','0.999999',args[3],args[4])
                    r=C.record('probe_p01','synthetic-bad','0.999999',args[3],args[4])
                    self.assertEqual(r['record']['status'],'anomaly')
                    self.assertEqual(r['state']['decision'],'pause_anomaly')
                    self.assertEqual(r['state']['remaining_submissions'],4)

    def test_failed_attempt_counts_without_equation(self):
        real=C.HERE
        with tempfile.TemporaryDirectory() as td:
            dst=Path(td)
            for name in ('campaign.json','candidate_catalog.json','probe_matrix.json','online_scores.json','submission_manifest.json','v157_probe_p01_utf8.csv'):
                shutil.copy2(real/name,dst/name)
            C.V.write(dst/'online_scores.json',{'records':[]})
            with patch.object(C,'HERE',dst):
                result=C.record('probe_p01','synthetic-failed',None,'2026-09-12T10:00:00+08:00','Synthetic failure',True)
                self.assertEqual(result['state']['remaining_submissions'],5)
                self.assertEqual(result['state']['feasible_worlds'],78)
                self.assertTrue(result['state']['conditional_route_budget_ok'])

    def test_adaptive_one_query_decodes_all_four_states(self):
        real=C.HERE;x=C.Campaign()
        # This fixture pins the two reported p01/p02 equations explicitly.
        x.s=tuple(j for j in x.matrix['world_ids'] if x.e.sums(C.ROWS[0])[j]==-3 and x.e.sums(C.ROWS[1])[j]==-3)
        self.assertEqual(len(x.s),4)
        shortcut=x.exact_one_shot()
        self.assertEqual(shortcut['query'],[3,4,7,12]);self.assertEqual(shortcut['carrier'],[1])
        branches=x.e.branches(x.s,shortcut['query'])
        self.assertEqual(len(branches),4)
        for dt,s in branches.items():
            self.assertEqual(len(s),1)
            self.assertAlmostEqual(x.e.merge(s)['f1'],1944/2089)
        paths=('campaign.json','candidate_catalog.json','probe_matrix.json','online_scores.json','submission_manifest.json')
        for score in ('0.927203','0.928161','0.929119','0.930077'):
            with tempfile.TemporaryDirectory() as td:
                dst=Path(td)
                for name in paths:shutil.copy2(real/name,dst/name)
                for entry in C.read('submission_manifest.json')['files']:
                    shutil.copy2(real/entry['file'],dst/entry['file'])
                with patch.object(C,'HERE',dst):
                    ledger=C.read('online_scores.json')
                    ledger['records']=[r for r in ledger['records'] if r['probe_id'] in ('probe_p01','probe_p02')]
                    C.save('online_scores.json',ledger)
                    r=C.record('probe_adaptive_p03','synthetic-adaptive',score,'2026-09-13T10:00:00+08:00','Synthetic test only')
                    self.assertEqual(r['record']['status'],'accepted')
                    self.assertEqual(r['state']['feasible_worlds'],1)
                    self.assertEqual(r['state']['remaining_submissions'],3)
                    self.assertEqual(r['state']['decision'],'emit_final')
                    self.assertAlmostEqual(r['state']['best_known_merge']['f1'],1944/2089)

if __name__=='__main__':unittest.main()
