"""Own-process-only deadline controller; isolated venv and official PyPI wheels."""
from pathlib import Path
import sys,subprocess,json,time,os,signal,zipfile
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
window=json.loads((HERE/'training_window.json').read_text())
deadline=datetime.fromisoformat(window['deadline_at']).timestamp()
def run(cmd,limit=None):
    remaining=deadline-time.time()
    if remaining<=0: raise TimeoutError('Authorized training window expired')
    proc=subprocess.Popen(cmd,cwd=HERE,start_new_session=True)
    try: code=proc.wait(timeout=min(remaining,limit) if limit else remaining)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        raise
    if code: raise RuntimeError('Command failed with exit '+str(code))
def main():
    if (HERE/'launch_state.json').exists(): raise RuntimeError('Previous launch exists; inspect before any repeat')
    state={'status':'preparing','started_at':datetime.now(timezone.utc).isoformat(),'deadline_at':window['deadline_at'],
           'launcher_pid':os.getpid(),'competition_upload':False}
    def save(): (HERE/'launch_state.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        env=HERE/'.venv'
        run([sys.executable,'-m','venv',str(env)],120)
        python=env/'bin/python'
        run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:',
             'numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.2','catboost==1.2.8',
             'threadpoolctl==3.6.0','joblib==1.5.2'],900)
        with (HERE/'requirements-resolved.txt').open('w') as f:
            subprocess.run([str(python),'-m','pip','freeze'],stdout=f,check=True,timeout=30)
        state['status']='cloud_smoke_running'; save()
        run([str(python),'-u','worker.py','--smoke'],300)
        state['status']='cloud_full_training_running'; save()
        run([str(python),'-u','worker.py'])
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed_or_timed_out',error=str(exc)); raise
    finally:
        state['updated_at']=datetime.now(timezone.utc).isoformat(); save()
        with zipfile.ZipFile(HERE/'cloud_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for dirname in ('results','smoke_results'):
                for p in (HERE/dirname).glob('*'):
                    if p.is_file() and p.suffix!='.joblib': z.write(p,str(p.relative_to(HERE)))
            z.write(HERE/'launch_state.json','launch_state.json')
if __name__=='__main__': main()
