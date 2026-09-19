import copy
import unittest
from assess_local_results import gate


class GateTests(unittest.TestCase):
    def report(self):
        return {'status':'completed','complete':True,'folds':[{}]*5,
                'summary':{name:{'32':{'f1':value,'base_f1':.8,'positive_folds':4}}
                           for name,value in [('narrow_catboost',.84),('narrow_v38_control',.85),
                                              ('wide_catboost',.86),('wide_v38_control',.83)]}}
    def test_selects_only_paired_winner(self):
        self.assertEqual(gate(self.report()), ['wide_catboost'])
    def test_partial_not_eligible(self):
        r=self.report(); r['folds']=r['folds'][:4]
        self.assertEqual(gate(r), [])
    def test_two_positive_folds_not_eligible(self):
        r=self.report(); r['summary']['wide_catboost']['32']['positive_folds']=2
        self.assertEqual(gate(r), [])
    def test_tie_does_not_pass(self):
        r=self.report(); r['summary']['wide_catboost']['32']['f1']=.85
        self.assertEqual(gate(r), [])


if __name__ == '__main__': unittest.main()
