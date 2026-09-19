"""Read-only history verification; evidence cache belongs only to V155."""
from pathlib import Path
import json
import core as c
from bridge import old152
from campaign import audit

def main():
    audit();_,seq,roots,nodes=c.baseline()
    system=c.collect_system(old152.records_data()['test'],seq)
    eq=c.Equations(system);c.require(eq.feasible(),'Historical conflict')
    blocked=old152.action_exclusions(nodes,system)
    source=Path(__file__).resolve().parent
    import numpy as np
    with np.load(source/'bundle/full_test.npz') as data: ids=data['row_ids']
    allmeta=c.read(source/'bundle/shared_meta.json');actions=[allmeta[int(i)] for i in ids]
    keys={(a['order_id'],a[f]) for a in actions for f in ('add_rid','remove_rid') if a.get(f)}
    bounds=c.legacy.EquationModel(system).classify(keys)
    report={'historical_equations':len(system['equations']),'variables':len(system['variables']),
            'equations_feasible':True,'test_actions':len(actions),'action_nodes':len(keys),
            'fixed_true_nodes':sum(b['min']==1 for b in bounds.values()),
            'fixed_false_nodes':sum(b['max']==0 for b in bounds.values()),
            'unresolved_nodes':sum(b['min']!=b['max'] for b in bounds.values()),
            'exclusion_keys':len(blocked),'baseline_sha256':c.BASE_SHA,'competition_submitted':False}
    c.write(source/'history_audit.json',report)
    c.write(source/'history_evidence_cache.json',{'system':system,'bounds':[{'order_id':k[0],'rid':k[1],**b} for k,b in sorted(bounds.items())]})
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
