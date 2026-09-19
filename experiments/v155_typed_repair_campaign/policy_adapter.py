"""Reuse tested V153 mechanics in an isolated namespace; V155 target-first policy."""
import importlib.util
import sys
import types
from pathlib import Path
import core as c

path=Path(__file__).resolve().parent.with_name('v153_complementary_repair_campaign')/'policy.py'
saved=sys.modules.get('bridge')
try:
    sys.modules['bridge']=types.SimpleNamespace(c=c)
    spec=importlib.util.spec_from_file_location('_v155_policy_base',path)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
finally:
    if saved is None: sys.modules.pop('bridge',None)
    else: sys.modules['bridge']=saved

Feasibility=base.Feasibility
WARNING='Conditional independent-action model calculation, not actual target confidence or a global optimum.'

class Policy(base.Policy):
    @staticmethod
    def ranking(q):
        return q['conditional_p_target'],q['expected_best_f1'],-len(q['ids']),tuple(-i for i in sorted(q['ids']))

    def evaluate(self,query,leaves,attempted,champion,depth):
        expected=ptarget=immediate=0.;summaries=[]
        for b in self.branches(query,leaves):
            value=max(champion,b['best_f1'])
            terminal={'expected_best_f1':value,'conditional_p_target':float(value>=c.TARGET),'ids':[]}
            continuation=None
            if depth>1 and value<c.TARGET:
                used=attempted|{tuple(sorted(query['ids']))}
                choices=[self.evaluate(q,b['leaves'],used,champion,1) for q in self.options(b['leaves'],used)]
                if choices:
                    nxt=max(choices,key=self.ranking)
                    if self.ranking(nxt)>self.ranking(terminal):
                        terminal=nxt;continuation={k:nxt[k] for k in ('kind','ids','expected_best_f1','conditional_p_target')}
            w=b['conditional_weight'];expected+=w*terminal['expected_best_f1'];ptarget+=w*terminal['conditional_p_target'];immediate+=w*value
            summaries.append({'delta_tp':b['delta_tp'],'conditional_weight':w,'known_best_f1_after_feedback':value,'next_if_this_feedback':continuation})
        return {**query,'expected_best_f1':expected,'conditional_p_target':ptarget,'one_step_expected_best_f1':immediate,
                'lookahead_steps':depth,'branches':summaries,'probability_warning':WARNING}
