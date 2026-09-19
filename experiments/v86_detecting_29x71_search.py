import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'experiments'; sys.path.insert(0,str(EXP)); sys.path.insert(0,str(EXP/'v30_meta_stack'))
from v58_coded_campaign import sidon_matrix
from v57_detecting_matrix_search import collision
OUT=EXP/'v86_detecting_29x71'; OUT.mkdir(exist_ok=True)
def main():
 attempts=[]; found=None
 for seed in range(20260821,20260841):
  try: m=sidon_matrix(29,71,36,seed,attempts=50000)
  except Exception as e: attempts.append({'seed':seed,'status':'gen_fail'}); continue
  c=collision(m,0.5); status='injective' if c is None else ('timeout' if isinstance(c,str) else f'collision_{int(np.count_nonzero(c))}')
  attempts.append({'seed':seed,'status':status}); print(attempts[-1],flush=True)
  if c is None: found=m; break
 report={'rows':29,'columns':71,'attempts':attempts,'injective_found':found is not None}
 if found is not None:
  report['path']=str(OUT/'basis_29x71.json'); (OUT/'basis_29x71.json').write_text(json.dumps(found.astype(int).tolist()),encoding='utf-8')
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
if __name__=='__main__':main()
