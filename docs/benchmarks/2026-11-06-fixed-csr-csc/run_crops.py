"""Matched crop comparisons; includes all native timing samples and host load."""
from pathlib import Path
import hashlib
import json
import os
import resource
from statistics import median
import subprocess
import time
import psutil
folder=Path(__file__).resolve().parent
binary=folder/'target/release/fixed-csr-csc'
for size in (1024,256):
 root=Path(f'/tmp/dioptrin-fixed-crop-{size}')
 for workers in (1,4,8):
  before=resource.getrusage(resource.RUSAGE_CHILDREN);host=psutil.cpu_times();t=time.perf_counter()
  raw=folder/f'crop{size}-w{workers}.jsonl'
  with raw.open('w') as stream:
   subprocess.run([str(binary),str(workers),str(root),str(root/'lut'),'0.35'],stdout=stream,check=True,
                  env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  elapsed=time.perf_counter()-t;after=resource.getrusage(resource.RUSAGE_CHILDREN);h=psutil.cpu_times()
  cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
  events=[json.loads(s) for s in raw.read_text().splitlines()]
  record={'size':size,'workers':workers,'inputs':json.loads((root/'inputs.json').read_text()),
          'events':events,'mean_other_busy_cores_over_entire_probe':max(0,(h.user+h.system-host.user-host.system-cpu)/elapsed),
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [binary,folder/'main.rs',folder/'lut/fixed_csc.rs',root/'calibration.poni',root/'mask.u8',root/'lut/weights.f64']}}
  (folder/f'crop{size}-w{workers}.json').write_text(json.dumps(record,indent=2)+'\n')
  print('crop',size,'workers',workers,flush=True)
  for errors in (False,True):
   print('errors',errors,{k:round(median(e['fps'] for e in events if e['event']=='timing' and not e['frozen_threshold'] and e['errors']==errors and e['kernel']==k),2)
                for k in ('production_fixed_csr','experimental_fixed_csc_cached_norm')},flush=True)
