"""V158 shared primitives. Only this campaign's directory is writable."""
from __future__ import annotations
import contextlib, csv, hashlib, importlib.util, json, os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
TZ = timezone(timedelta(hours=8))
G, TARGET = 1044, .94
PYTHON = EXP/'v152_error_repair_campaign/.venv/Scripts/python.exe'

def require(ok, message):
    if not ok: raise ValueError(message)

def now(): return datetime.now(TZ).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

@contextlib.contextmanager
def lock(out):
    p = Path(out)/'operation.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(p, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise ValueError('Another operation owns operation.lock; inspect its PID before recovery')
    try:
        os.write(fd, json.dumps({'pid':os.getpid(),'at':now()}).encode()); os.close(fd)
        yield
    finally: p.unlink(missing_ok=True)

def module(alias, path):
    if alias in sys.modules: return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m

sys.path.insert(0, str(EXP))
import v150_adaptive_addition_campaign as legacy

def v157():
    return module('_v158_previous', EXP/'v157_joint_correction_campaign/campaign.py')

def training_module():
    # Dependencies are rebound only during import, never on disk.
    old = EXP/'v152_error_repair_campaign'
    c = module('_v158_training_core', old/'core.py')
    saved = sys.modules.get('core'); sys.modules['core'] = c
    try: return module('_v158_training', old/'training.py')
    finally:
        if saved is None: sys.modules.pop('core', None)
        else: sys.modules['core'] = saved

def infer(score, p):
    require(isinstance(score,str) and len(score.split('.')[-1])==6, 'Supply an exact six-decimal score')
    return legacy.infer_tp(score,p)

def key(a): return (a['order_id'],a.get('remove_rid') or '',a.get('add_rid') or '')
def f1(tp,p): return 2*tp/(G+p)

def verify_sources(sources):
    for p,h in sources.items(): require(sha(p)==h, 'Frozen source changed: '+p)

def csv_nodes(path, strict=True): return legacy.load_csv(path,strict=strict)

def apply_actions(base, actions):
    ids, roots, nodes = csv_nodes(base)
    result = {oid:[dict(n) for n in roots[oid]] for oid in ids}
    used = set()
    for a in actions:
        oid = a['order_id']; require(oid in roots and oid not in used,'Conflicting/unknown order')
        used.add(oid); row = result[oid]
        rem,add = a.get('remove_rid'),a.get('add_rid')
        if rem:
            require(sum(n['@rid']==rem for n in row)==1,'Removal is absent')
            row = [n for n in row if n['@rid']!=rem]
        if add:
            require(all(n['@rid']!=add for n in row),'Addition already present')
            require(a['node']['@rid']==add,'Addition metadata mismatch')
            row.append(dict(a['node']))
        require(1<=len(row)<=8,'Invalid root count after action')
        result[oid] = row
    return ids,result

def validate_file(path, base, actions, expected_sha=None):
    require(Path(path).read_bytes()[:3]!=b'\xef\xbb\xbf','BOM is forbidden')
    if expected_sha: require(sha(path)==expected_sha,'CSV hash mismatch')
    ids,roots,nodes = csv_nodes(path)
    wanted,expected = apply_actions(base,actions)
    require(ids==wanted and roots==expected,'CSV exact order/metadata/action difference mismatch')
    with Path(base).open(encoding='utf-8-sig',newline='') as f: before=list(csv.DictReader(f))
    with Path(path).open(encoding='utf-8-sig',newline='') as f: after=list(csv.DictReader(f))
    for x,y in zip(before,after):
        a=json.loads(x['output']);b=json.loads(y['output']);a['rootcause']=expected[x['order_id']]
        require(a==b,'Top-level output metadata changed')
    return dict(orders=len(ids),predictions=len(nodes),sha256=sha(path),exact_difference=True,
                root_counts_1_to_8=True,duplicates=0,encoding='utf-8-no-bom')

def author(path, base, actions):
    ids,roots = apply_actions(base,actions)
    # Preserve top-level JSON fields as well as all unchanged node metadata.
    with Path(base).open(encoding='utf-8',newline='') as f: original=list(csv.DictReader(f))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\r\n');writer.writerow(['order_id','output'])
        for oid,row in zip(ids,original):
            obj=json.loads(row['output']);obj['rootcause']=roots[oid]
            writer.writerow([oid,json.dumps(obj,ensure_ascii=False,separators=(',',':'))])
    return validate_file(path,base,actions)
