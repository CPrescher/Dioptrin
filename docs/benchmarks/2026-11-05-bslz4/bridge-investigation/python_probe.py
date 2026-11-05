"""Measure native-call return and Python result materialization separately."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import dioptrin
import numpy as np

root=Path('/tmp/dioptrin-bridge-investigation-20261105')
threads=int(sys.argv[1])
dioptrin.set_rayon_threads(threads)
meta=json.loads((root/'inputs.json').read_text())
images=np.memmap(root/'frames.f64',dtype='<f8',mode='r',shape=tuple(meta['shape']))
frames=[images[i] for i in range(len(images))]
mask=np.fromfile(root/'mask.u8',dtype=np.uint8).reshape(meta['shape'][1:])
ai=dioptrin.Integrator('/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni',method='pixel_split',unit='2th_deg',polarization_factor=.965,fast_math=True)
ai.set_mask(mask)
def call():
    return ai.batch1d(frames,2000,num_workers=threads,num_threads=1)
def consume(results):
    return np.asarray([r.result.intensity for r in results])
def call_consumed():
    return consume(call())
for _ in range(3):call_consumed()
existing=call()
reference=consume(existing)
np.save(root/f'python-reference-w{threads}.npy',reference)
native=root/f'native-reference-w{threads}.f64'
if native.exists():
    expected=np.fromfile(native,dtype='<f8').reshape(reference.shape)
    np.testing.assert_allclose(reference,expected,rtol=1e-12,atol=1e-10)
    print(json.dumps(dict(numerical_relative_l2=float(np.linalg.norm(reference-expected)/np.linalg.norm(expected)))),flush=True)
jobs=[('python_return',call),('python_consumed',call_consumed),('materialize_existing',lambda:consume(existing))]
rows=[]
for rep in range(5):
    for name,fn in jobs if rep%2==0 else jobs[::-1]:
        gc.collect()
        start=time.perf_counter();n=0
        while True:
            result=fn();del result;n+=1
            elapsed=time.perf_counter()-start
            if elapsed>=1.:break
        row=dict(workflow=name,threads=threads,repeat=rep,loops=n,seconds=elapsed,fps=len(frames)*n/elapsed,seconds_per_frame=elapsed/len(frames)/n)
        rows.append(row);print(json.dumps(row),flush=True)
print(json.dumps(dict(summary={name:statistics.median(x['fps'] for x in rows if x['workflow']==name) for name,_ in jobs})),flush=True)
