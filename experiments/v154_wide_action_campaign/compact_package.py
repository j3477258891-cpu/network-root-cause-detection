"""Lossless deduplication of action features across nested CV folds."""
from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
HERE=Path(__file__).resolve().parent
source=HERE/'bundle'; dest=HERE/'compact_bundle'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
def main():
    dest.mkdir(exist_ok=False)
    meta=[]; static=[]; lookup={}; count=0
    for p in sorted(source.glob('*.npz')):
        if p.name=='truth.npz': shutil.copy2(p,dest/p.name); continue
        info=read(p.with_suffix('.json')); split='test' if p.stem.endswith('full_test') else 'train'
        with np.load(p,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        x=arrays.pop('x'); ids=[]
        for i,m in enumerate(info['meta']):
            key=(split,m['order_id'],m['remove_rid'],m['add_rid'])
            if key not in lookup:
                lookup[key]=len(meta); meta.append(m); static.append(x[i,:-11].copy())
            j=lookup[key]
            assert meta[j]==m and np.array_equal(static[j],x[i,:-11]), 'Dedup mismatch'
            ids.append(j)
        arrays.update(row_ids=np.array(ids,dtype=np.int32),context=x[:,-11:])
        np.savez_compressed(dest/p.name,**arrays)
        write(dest/p.with_suffix('.json').name,{'orders':info['orders']})
        count+=len(ids)
    matrix=np.array(static,dtype=np.float32)
    np.savez_compressed(dest/'shared_static.npz',x=matrix); write(dest/'shared_meta.json',meta)
    for name in ('protocol.py','worker_base.py','wide_actions_source.py','generator_checks.json','training_window.json','PLAN.md'):
        shutil.copy2(source/name,dest/name)
    for name in ('worker.py','launch.py'): shutil.copy2(HERE/name,dest/name)
    # Reconstruct every original array and metadata; no quantization or dropped rows.
    for p in sorted(source.glob('*.npz')):
        if p.name=='truth.npz': continue
        with np.load(p,allow_pickle=False) as a, np.load(dest/p.name,allow_pickle=False) as b:
            rebuilt=np.column_stack([matrix[b['row_ids']],b['context']])
            assert np.array_equal(rebuilt,a['x'])
            for k in ('y','prior','mask'): assert np.array_equal(a[k],b[k])
            assert [meta[int(i)] for i in b['row_ids']]==read(p.with_suffix('.json'))['meta']
    manifest=read(source/'manifest.json')
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
    manifest['lossless_storage']={'action_rows':count,'unique_static_rows':len(meta),'all_arrays_and_metadata_verified_equal':True}
    write(dest/'manifest.json',manifest)
    archive=HERE/'v154_compact_training.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()): z.write(p,'v154_wide_action/'+p.name)
    report={'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),**manifest['lossless_storage']}
    write(HERE/'compact_report.json',report); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
