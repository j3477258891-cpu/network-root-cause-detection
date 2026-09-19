import unittest
import numpy as np
from worker import choose,weights,physical,gate,TARGET,fit_predict

class TypedTests(unittest.TestCase):
    def test_target_arithmetic(self):
        self.assertAlmostEqual(TARGET*2093-1942,20.930515)
        self.assertGreaterEqual(2*982/2093,TARGET)
        self.assertLess(2*981/2093,TARGET)
    def test_weights_per_order_type(self):
        a=[dict(order_id='a',kind='swap')]*4+[dict(order_id='b',kind='swap')]
        self.assertTrue(np.allclose(weights(a),[.25]*4+[1]))
    def test_support(self):
        p=physical(np.ones((3,3)),[{'kind':k} for k in ('add','delete','swap')])
        self.assertEqual(p[0,0],0);self.assertEqual(p[1,2],0)
        self.assertTrue(np.allclose(p.sum(1),1))
    def test_unique_and_no_validation_labels_in_ranking(self):
        meta=[dict(order_id=o,kind='swap',delta_p=0) for o in ('a','a','b')]
        b=dict(meta=meta,orders=list(range(546)),y=np.array([-1,1,-1]))
        p=np.array([[0,0,1],[.1,.1,.8],[1,0,0]])
        self.assertEqual(choose(b,p,32),[0]);b['y']=-b['y']
        self.assertEqual(choose(b,p,32),[0])
    def test_false_positive_delete_is_positive_utility(self):
        b=dict(meta=[dict(order_id='a',kind='delete',delta_p=-1)],orders=list(range(546)))
        self.assertEqual(choose(b,np.array([[0,1,0]]),32),[0])
    def test_gate_all_conditions(self):
        summary={n:{'32':dict(u=u,positive_folds=4,worst_fold_u=w)} for n,u,w in
            [('control_catboost',3,-2),('control_v38',4,-1),('typed_catboost',5,-1),('typed_v38',6,-3)]}
        self.assertEqual(gate(summary),('control_v38',['typed_catboost']))
        summary['typed_catboost']['32']['positive_folds']=2
        self.assertEqual(gate(summary)[1],[])
    def test_binary_delete_mapping(self):
        meta=[dict(order_id=str(i),kind='delete') for i in range(18)]
        b=dict(meta=meta,x=np.arange(54).reshape(18,3),y=np.arange(18)%2-1)
        p,_=fit_predict('typed_catboost',4,b,b,1,True)
        self.assertTrue(np.allclose(p[:,2],0));self.assertTrue(np.allclose(p.sum(1),1))

if __name__=='__main__': unittest.main()
