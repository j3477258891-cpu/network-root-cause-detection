"""Regression tests for the live entrypoint, not the retained legacy policy."""
import copy
import hashlib
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import verified_engine as V

class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.e=V.Engine()
    def test_encoding_regression(self):
        e=self.e
        with self.assertRaisesRegex(ValueError,'BOM'):
            e.csv.validate_csv(V.HERE/'v156_probe_p01.csv',[8,9,11])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'clean.csv';e.csv.apply_actions(e.csv.BASE,[8,9,11],p)
            result=e.csv.validate_csv(p,[8,9,11]);self.assertEqual(result['predictions'],1048)
            self.assertTrue(p.read_bytes().startswith(b'order_id,output\r\n'))
            self.assertEqual(p.read_bytes(),(V.HERE/'v156_probe_p01.csv').read_bytes()[3:])
    def test_history_and_budget(self):
        self.assertEqual(len(self.e.world),3240)
        attempts={r['attempt_id'] for r in self.e.ledger['records'] if not r.get('duplicate_of')}
        self.assertEqual(self.e.used,self.e.freeze['prior_attempts']+len(attempts))
        self.assertEqual(self.e.remaining,max(0,10-self.e.used))
        self.assertEqual(self.e.depth(tuple(range(len(self.e.world)))),3)
    def test_all_twelve_decode(self):
        initial=tuple(range(len(self.e.world)))
        labs=self.e.labels(initial);self.assertEqual(len(labs),12)
        signatures={tuple(sum(row[V.NEW.index(i)] for i in q) for q in V.FIXED) for row in labs}
        self.assertEqual(len(signatures),12)
        self.assertEqual(V.decode_depth(labs),3)
        for _,s in self.e.branches(initial,(8,9,11)).items():self.assertLessEqual(self.e.depth(s),2)
    def test_known_group_merge(self):
        # No individual true addition is known yet; six-node group sum is known.
        m=self.e.merge(tuple(range(len(self.e.world))))
        self.assertEqual(m['ids'],[8,9,11,14,15,17])
        self.assertEqual(m['delta_tp'],3);self.assertEqual(m['delta_p'],6)
        self.assertAlmostEqual(m['f1'],1944/2095)
    def test_signed_scores(self):
        for dt in (-1,0,1):
            self.assertEqual(V.infer(f'{V.f1(dt):.6f}',1045),969+dt)
        self.assertEqual(V.infer(f'{V.f1(-1,-1):.6f}',1044),968)
        with self.assertRaises(ValueError):V.infer('0.999999',1048)
    def test_decoded_merge_and_upper(self):
        initial=tuple(range(len(self.e.world)))
        for labs in self.e.labels(initial):
            s=tuple(j for j in initial if tuple(self.e.world[j,self.e.pos[i]] for i in V.NEW)==labs)
            m=self.e.merge(s);self.assertAlmostEqual(m['f1'],1944/2092)
        self.assertAlmostEqual(self.e.upper(tuple(self.e.support)),.932503590234562)
    def test_timeout_complete_fallback(self):
        e=self.e;s=tuple(range(len(e.world)));search=V.Search(e,e.weights,0)
        tree=search.run(s,3,e.champ_score)
        self.assertTrue(tree['search_truncated']);self.assertTrue(tree['complete'])
        self.assertGreaterEqual(tree['worst'],1944/2092-1e-12)
        def visit(node,depth):
            self.assertLessEqual(depth,3)
            for b in node['branches']:visit(b['next'],depth+1)
            if node['branches']:
                self.assertAlmostEqual(node['expected'],sum(b['weight']*b['next']['expected'] for b in node['branches']))
                self.assertAlmostEqual(node['worst'],min(b['next']['worst'] for b in node['branches']))
        visit(tree,0)
    def test_audit_is_readonly(self):
        paths=[V.HERE/p for p in ('campaign.json','online_scores.json','submission_manifest.json','verified_freeze.json')]
        before=[V.sha(p) for p in paths];self.e.status();after=[V.sha(p) for p in paths]
        self.assertEqual(before,after)
    def test_record_updates_and_idempotent_anomaly(self):
        # Point only the mutable engine artifacts at a temporary independent campaign.
        real=V.HERE
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            for name in ('campaign.py','verified_freeze.json','history_worlds.json','online_scores.json','submission_manifest.json','v156_probe_p01.csv'):
                shutil.copy2(real/name,dest/name)
            ledger=V.read(dest/'online_scores.json')
            ledger['records']=[r for r in ledger['records'] if r['probe_id']=='p01' and r['status']=='failed']
            self.assertEqual(len(ledger['records']),1)
            V.write(dest/'online_scores.json',ledger)
            old=V.legacy();p=dest/'test_clean.csv';old.apply_actions(old.BASE,[8,9,11],p)
            man=V.read(dest/'submission_manifest.json');man['files'].append({'probe_id':'test','file':p.name,'sha256':V.sha(p),'all_ids':[8,9,11],'query_ids':[8,9,11],'carrier_ids':[]})
            V.write(dest/'submission_manifest.json',man)
            with patch.object(V,'HERE',dest),patch.object(V,'legacy',return_value=old):
                args=('test','test-attempt','0.928298','2026-09-10T18:00:00+08:00','test fixture')
                V.record(*args);self.assertTrue(V.record(*args)['idempotent'])
                e=V.Engine();self.assertEqual(e.remaining,3);self.assertEqual(len(e.labels(tuple(e.support))),5)
                self.assertEqual(e.champ['tp'],971);self.assertEqual(e.champ['predictions'],1048)
                V.record('test','anomaly-attempt','0.999999','2026-09-11T10:00:00+08:00','test anomaly')
                e=V.Engine();self.assertEqual(e.remaining,2);self.assertEqual(len(e.blocked),1)

    def test_current_correction_branches_against_full_enumeration(self):
        # Independently enumerate all 65536 action subsets, without the live
        # optimizer's fixed-outcome pruning or integer row-space shortcut.
        e=self.e
        s=tuple(j for j,row in enumerate(e.world)
                if all(int(row[e.pos[i]])==int(i in (8,11,15)) for i in V.NEW))
        branches=e.branches(s,(1,12,16))
        self.assertEqual(set(branches),set(range(-3,3)))
        for dt,t in branches.items():
            raw=e.world[list(t)][:,[e.pos[i] for i in e.active]].astype(np.int16) @ e.bits.T
            constant=raw.min(axis=0)==raw.max(axis=0)
            scores=2*(V.TP0+raw[0])/(V.G+V.P0+e.dp)
            expected=float(scores[constant].max())
            actual=e.merge(t)
            self.assertAlmostEqual(actual['f1'],expected,places=14)
            self.assertGreaterEqual(actual['f1'],1944/2092)
            probe_score=V.f1(dt+3,2)
            self.assertEqual(V.infer(f'{probe_score:.6f}',1047),972+dt)

if __name__=='__main__':unittest.main()
