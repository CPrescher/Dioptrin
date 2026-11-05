from pathlib import Path
import json,os,subprocess
ROOT=Path(__file__).resolve().parent
PYTHON='/tmp/dioptrin-mask-modes-env-20261105/bin/python'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for errors in [False,True]:
    name='w8-paired-'+('poisson' if errors else 'intensity')
    args=[PYTHON,str(ROOT/'bench_full_file.py'),'--hdf5','/Users/clemens/Programming/Dioptrin/test_data/eiger_0000.h5','--poni','/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni','--mask','/Users/clemens/Programming/Dioptrin/test_data/eiger_0000.mask','--lut','/tmp/dioptrin-bslz4-lut-20261105/pixel_split','--output',str(ROOT/(name+'.json')),'--workers','8','--samples','16','--repeats','5','--target-seconds','1','--full-file','--reuse-control-only','--pipeline-json','{"dot":"csc"}','--input-sha256','2b749693d7a942b35d89806d99b9a1025ee119850be0ade27428f1dc11d0a1f3']
    if errors:args.append('--errors')
    print('RUN',name,flush=True)
    with (ROOT/(name+'.log')).open('w') as raw,(ROOT/(name+'.stderr')).open('w') as err:
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=err,text=True,env=ENV)
        for line in proc.stdout:
            raw.write(line);raw.flush()
            if line.startswith('{'):
                e=json.loads(line)
                if 'name' in e:print(name,e['name'],e['repeat'],round(e['fps'],1),'fps','other busy cores',round(e['other_busy_cores'],1),flush=True)
                elif 'summary' in e:print(name,'SUMMARY',json.dumps(e['summary']),flush=True)
                elif 'numerical' in e:print(name,'sampled numerical checks passed',flush=True)
            else:print(name,line.rstrip(),flush=True)
        code=proc.wait()
    if code:raise RuntimeError((name,code,(ROOT/(name+'.stderr')).read_text()))
