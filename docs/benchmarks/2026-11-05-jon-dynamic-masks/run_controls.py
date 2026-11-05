"""Run after run.py, keeping all benchmark processes sequential."""
from pathlib import Path
import json, os, subprocess
ROOT=Path(__file__).resolve().parent
PYTHON='/tmp/dioptrin-bslz4-env-20261105/bin/python'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def run(name,args):
    print('RUN',name,flush=True)
    with (ROOT/(name+'.jsonl')).open('w') as raw,(ROOT/(name+'.stderr')).open('w') as err:
        p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,env=ENV,text=True)
        for line in p.stdout:
            raw.write(line);raw.flush();e=json.loads(line)
            if e['event']=='measurement':print(name,e.get('threshold'),e.get('errors'),e['repeat'],round(e['fps'],2),'fps',flush=True)
            elif e['event']=='numerical':print(name,'numerical checks passed',flush=True)
        code=p.wait()
    if code:raise RuntimeError((name,code,(ROOT/(name+'.stderr')).read_text()))

run('jon-w8-scipy-poisson',[PYTHON,str(ROOT/'bench.py'),'--workers','8','--errors','--variance-via-scipy','--thresholds','230','--output',str(ROOT/'jon-w8-scipy-poisson.json')])
run('dioptrin-api-w8',[PYTHON,str(ROOT/'dioptrin_api_control.py'),'--workers','8','--output',str(ROOT/'dioptrin-api-w8.json')])
