"""Protocol regressions: validation labels must not choose their own actions."""
import copy
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training import group_splits, selected_action_indices, score_actions, train
from core import write


class ProtocolTests(unittest.TestCase):
    def test_every_outer_partition_remains_nested_crossfittable(self):
        folds = np.repeat(np.arange(5), [782,213,213,213,213])
        for outer in range(5):
            orders = np.flatnonzero(folds != outer)
            seen = []
            for fit, valid in group_splits(orders, folds):
                self.assertGreaterEqual(len(set(folds[fit])), 2)
                self.assertFalse(set(folds[fit]) & set(folds[valid]))
                seen.extend(valid)
                for node_fit, node_valid in group_splits(fit, folds):
                    self.assertGreater(len(node_fit), 0)
                    self.assertFalse(set(folds[node_fit]) & set(folds[node_valid]))
            self.assertEqual(sorted(seen), list(orders))

    def test_decisions_do_not_change_when_validation_labels_change(self):
        # Add .44 is below the fixed .927717/2 threshold even when a weak
        # validation baseline would have made it look eligible.
        meta = [{'order_id':'o1','kind':'add','add_rid':'a','remove_rid':None,'delta_p':1},
                {'order_id':'o2','kind':'delete','add_rid':None,'remove_rid':'r','delta_p':-1}]
        bundle = {'orders':[0,1], 'meta':meta, 'mask':np.array([True,False,True,False]),
                  'y':np.array([1,0])}
        p = np.array([[0,.56,.44],[.4,.6,0]])
        ptr = np.array([0,2,4])
        a = score_actions(bundle,p,np.array([1,0,1,0]),ptr)
        other = copy.deepcopy(bundle); other['y'] = np.array([0,-1])
        b = score_actions(other,p,np.array([0,1,1,1]),ptr)
        self.assertEqual(selected_action_indices(bundle,p), [1])
        for budget in ('16','32','64'):
            self.assertEqual(a[budget]['selected_action_indices'],b[budget]['selected_action_indices'])
        self.assertNotEqual(a['64']['f1'],b['64']['f1'])

    def test_one_independent_action_per_order(self):
        meta = [{'order_id':'o','delta_p':0,'add_rid':str(i),'remove_rid':'r'} for i in range(2)]
        self.assertEqual(selected_action_indices({'meta':meta},np.array([[.1,.1,.8],[.1,.2,.7]])),[0])

    def test_expired_window_is_not_reset(self):
        with tempfile.TemporaryDirectory(prefix='v152_expired_test_') as folder:
            out=Path(folder)
            write(out/'training_window.json',{'started_at':'2020-01-01T00:00:00+08:00',
                                             'deadline_at':'2020-01-03T00:00:00+08:00'})
            with self.assertRaisesRegex(ValueError,'Original training deadline expired'):
                train(out)


if __name__ == '__main__': unittest.main()
