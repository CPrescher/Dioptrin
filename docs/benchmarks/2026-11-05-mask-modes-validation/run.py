from pathlib import Path
import subprocess,os
ROOT=Path(__file__).resolve().parent
OLD='/tmp/dioptrin-bslz4-env-20261105/bin/python'
NEW='/tmp/dioptrin-mask-modes-env-20261105/bin/python'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name,python,script in [('published-fixed',OLD,'fixed_batch.py'),('new-fixed',NEW,'fixed_batch.py'),('published-dynamic-api',OLD,'dynamic_api.py'),('new-dynamic-api',NEW,'dynamic_api.py')]:
    print('RUN',name,flush=True)
    with (ROOT/(name+'.jsonl')).open('w') as out,(ROOT/(name+'.stderr')).open('w') as err:
        args=[python,str(ROOT/script),'--output',str(ROOT/(name+'.json'))]
        if name=='published-dynamic-api':args+=['--mask-mode','legacy']
        p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,text=True,env=ENV)
        for line in p.stdout:out.write(line);out.flush();print(name,line.rstrip(),flush=True)
        code=p.wait()
    if code:raise RuntimeError((name,code,(ROOT/(name+'.stderr')).read_text()))
