import itertools
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import core as c
import campaign
from policy_adapter import Policy,Feasibility

def action(i,kind='add',p=.7):
    support={'add':{'0':1-p,'1':p},'delete':{'-1':1-p,'0':p},'swap':{'-1':0.,'0':1-p,'1':p}}[kind]
    return {'candidate_id':i,'order_id':str(i),'kind':kind,'add_rid':str(i) if kind!='delete' else None,
            'remove_rid':str(i)+'old' if kind!='add' else None,'delta_p':{'add':1,'delete':-1,'swap':0}[kind],
            'probabilities':support,'model_probabilities':{'model':support}}

class CampaignTests(unittest.TestCase):
    def test_failed_gate_emits_no_csv(self):
        with TemporaryDirectory() as temp,patch.object(campaign,'audit',return_value={}),patch.object(campaign,'HERE',Path(temp)):
            root=Path(temp);c.write(root/'progress.json',{'complete':True});c.write(root/'campaign_config.json',{'baseline':{}})
            summary={f:{'32':{'u':-1,'positive_folds':0,'worst_fold_u':-1}} for f in campaign.worker.FAMILIES}
            with patch.object(campaign,'comparison',return_value=({},summary,'control_v38',[])):
                result=campaign.build(root)
            self.assertFalse(result['allow_submission']);self.assertFalse(list(root.glob('*.csv')))
            self.assertFalse((root/'campaign.json').exists())

    def test_incomplete_gate_emits_no_csv(self):
        with TemporaryDirectory() as temp,patch.object(campaign,'audit',return_value={}),patch.object(campaign,'HERE',Path(temp)):
            root=Path(temp);c.write(root/'campaign_config.json',{'baseline':{}})
            result=campaign.build(root)
            self.assertEqual(result['decision'],'await_full_training_no_csv');self.assertFalse(list(root.glob('*.csv')))

    def test_baseline_score(self):
        self.assertEqual(c.infer_tp('0.927855',1049),971)
        self.assertAlmostEqual(c.U_TARGET,20.930515)
        with self.assertRaises(ValueError): c.infer_tp('0.999999',1049)

    def test_independent_selection_type_cap(self):
        rows=[action(i,k,.9-i/10000) for i,k in enumerate(['add']*30+['delete']*20+['swap']*20,1)]
        rows.append({**action(99,'add',.999),'order_id':'1'})
        chosen,groups=campaign.select(rows)
        self.assertEqual(len(chosen),48)
        self.assertEqual(len({a['order_id'] for a in chosen}),48)
        self.assertTrue(all(len(g['ids'])<=8 for g in groups))
        self.assertEqual(sorted(i for g in groups for i in g['ids']),list(range(1,49)))
        self.assertEqual({k:sum(a['kind']==k for a in chosen) for k in ('add','delete','swap')},{'add':16,'delete':16,'swap':16})

    def test_split_conservation_and_overlap(self):
        leaves=[{'ids':[1,2,3],'delta_tp':1}]
        split={'kind':'split','ids':[1],'parent_ids':[1,2,3]}
        after=c.apply_observation(leaves,split,-1)
        self.assertEqual(sum(x['delta_tp'] for x in after),1)
        self.assertEqual(after[1]['delta_tp'],2)
        with self.assertRaises(ValueError): c.apply_observation(after,{'kind':'group','ids':[1]},0)

    def test_signed_union(self):
        byid={1:action(1,'delete'),2:action(2,'add'),3:action(3,'swap')}
        leaves=[{'ids':[1],'delta_tp':0},{'ids':[2],'delta_tp':0},{'ids':[3],'delta_tp':1}]
        best=c.best_union(leaves,byid)
        self.assertEqual(best['ids'],[1,3]);self.assertAlmostEqual(best['f1'],2*972/2092)

    def test_integer_oracle_against_enumeration(self):
        variables=[{'order_id':str(i),'rid':str(i)} for i in range(3)]
        system={'variables':variables,'equations':[{'indices':[0,1],'tp':1}],'ground_truth_positives':1}
        eq=c.Equations(system)
        aa=[action(i) for i in range(3)]
        oracle=eq.oracle(aa)
        brute=max(c.f1(sum(y[i] for i in range(3) if mask>>i&1),mask.bit_count())
                  for y in itertools.product((0,1),repeat=3) if sum(y)==1 and y[0]+y[1]==1 for mask in range(8))
        self.assertAlmostEqual(oracle['f1_upper'],brute)
        f=Feasibility(eq,{i:a for i,a in enumerate(aa)})
        self.assertFalse(f([([2],1)]));self.assertTrue(f([([0],1)]))

    def test_two_step_against_label_tree(self):
        byid={i:action(i,p=p) for i,p in enumerate((.2,.85,.55,.7),1)}
        groups=[{'group_id':'A','kind':'add','ids':[1,2]},{'group_id':'B','kind':'add','ids':[3,4]}]
        states=[]
        for y in itertools.product((0,1),repeat=4):
            w=1.
            for i,v in enumerate(y,1): w*=byid[i]['probabilities'][str(v)]
            states.append((y,w))
        def compatible(obs): return any(all(sum(y[i-1] for i in ids)==dt for ids,dt in obs) for y,w in states)
        policy=Policy(byid,groups,['model'],compatible)
        def terminal(leaves):
            best=c.best_union(leaves,byid)['f1'];return float(best>=c.TARGET),best
        def options(leaves,attempted):
            measured={i for l in leaves for i in l['ids']}
            result=[{**g,'kind':'group'} for g in groups if not measured.intersection(g['ids']) and tuple(g['ids']) not in attempted]
            for l in leaves:
                if l['delta_tp'] in (0,len(l['ids'])): continue
                for ids in ([l['ids'][0]],[l['ids'][-1]]):
                    if tuple(ids) not in attempted: result.append({'kind':'split','ids':ids,'parent_ids':l['ids']})
            return result
        def value(q,leaves,possible,attempted,depth):
            branches={}
            for y,w in possible: branches.setdefault(sum(y[i-1] for i in q['ids']),[]).append((y,w))
            total=sum(w for _,w in possible);out=[0.,0.]
            for dt,ss in branches.items():
                ll=c.apply_observation(leaves,q,dt);best=terminal(ll);used=attempted|{tuple(q['ids'])}
                if depth>1 and not best[0]:
                    for qq in options(ll,used): best=max(best,value(qq,ll,ss,used,1))
                weight=sum(w for _,w in ss)/total
                out=[out[i]+weight*best[i] for i in (0,1)]
            return tuple(out)
        with patch.object(c,'TARGET',.9288):
            for q in options([],set()):
                expected=value(q,[],states,set(),2);got=policy.evaluate(q,[],set(),c.f1(),2)
                self.assertAlmostEqual(got['conditional_p_target'],expected[0]);self.assertAlmostEqual(got['expected_best_f1'],expected[1])

    def test_target_priority_not_expected_only(self):
        a={'ids':[1],'conditional_p_target':.1,'expected_best_f1':.935}
        b={'ids':[2],'conditional_p_target':.2,'expected_best_f1':.930}
        self.assertGreater(Policy.ranking(b),Policy.ranking(a))

if __name__=='__main__': unittest.main()
