"""Run warmed native kernel comparisons; preserve all samples and host load."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import psutil

folder=Path(__file__).resolve().parent
binary=folder/'target/release/fixed-csr-csc'
inputs=Path('/tmp/dioptrin-bridge-investigation-20261105')
lut=Path('/tmp/dioptrin-bslz4-lut-20261105/pixel_split')
for workers in (1,4,8):
    own=resource.getrusage(resource.RUSAGE_CHILDREN);host=psutil.cpu_times();start=time.perf_counter()
    raw=folder/f'w{workers}.jsonl'
    with raw.open('w') as f:
        subprocess.run([str(binary),str(workers),str(inputs),str(lut)],stdout=f,check=True,
                       env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    elapsed=time.perf_counter()-start
    own_after=resource.getrusage(resource.RUSAGE_CHILDREN);after=psutil.cpu_times()
    cpu=own_after.ru_utime+own_after.ru_stime-own.ru_utime-own.ru_stime
    other=max(0,(after.user+after.system-host.user-host.system-cpu)/elapsed)
    events=[json.loads(line) for line in raw.read_text().splitlines()]
    data={'workers':workers,'elapsed_seconds':elapsed,'process_cpu_seconds':cpu,
          'mean_other_busy_cores_over_entire_probe':other,'events':events,
          'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [binary,folder/'main.rs',folder/'lut/fixed_csc.rs',folder/'../../../src/lut/prepared.rs',folder/'../../../src/lut/csc.rs']},
          'inputs':str(inputs),'lut':str(lut)}
    (folder/f'w{workers}.json').write_text(json.dumps(data,indent=2)+'\n')
    from statistics import median
    print('workers',workers,'other cores',round(other,2),flush=True)
    for frozen in (False,True):
        for errors in (False,True):
            values={kernel:median(e['fps'] for e in events if e['event']=='timing' and
                    e['frozen_threshold']==frozen and e['errors']==errors and e['kernel']==kernel)
                    for kernel in ('production_fixed_csr','experimental_fixed_csc_cached_norm')}
            print(frozen,errors,values,flush=True)
