"""Sequential runner; no benchmark processes overlap."""
from pathlib import Path
import json, os, subprocess
ROOT=Path(__file__).resolve().parent
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
PYTHON='/tmp/dioptrin-bslz4-env-20261105/bin/python'

def run(name,args):
    print('RUN',name,flush=True)
    with (ROOT/(name+'.jsonl')).open('w') as raw,(ROOT/(name+'.stderr')).open('w') as err:
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,text=True,env=ENV)
        for line in proc.stdout:
            raw.write(line);raw.flush();e=json.loads(line)
            if e['event']=='measurement':
                print(name,e.get('threshold'),'errors='+str(e.get('errors')),e.get('variant','Jon rebuild'),e['repeat'],round(e['fps'],2),'fps',flush=True)
            elif e['event'] in ('numerical','baseline_validation','strict_validation'):
                print(name,e['event'],e.get('threshold'),e.get('errors'),flush=True)
        code=proc.wait()
    if code:raise RuntimeError((name,code,(ROOT/(name+'.stderr')).read_text()))

for workers in [8,1]:
    for errors in [False,True]:
        name=f'jon-w{workers}-'+('poisson' if errors else 'intensity')
        args=[PYTHON,str(ROOT/'bench.py'),'--workers',str(workers),'--output',str(ROOT/(name+'.json'))]
        if errors:args.append('--errors')
        if workers==1:args+=['--thresholds','230','--repeats','3']
        run(name,args)
run('dioptrin-native-w8-repeat',[
    '/Users/clemens/Programming/Dioptrin/target/release/dioptrin-dynamic-mask-f64-probe','8',
    '/tmp/dioptrin-bridge-investigation-20261105','/tmp/dioptrin-bslz4-lut-20261105/pixel_split','230','5','1'])
