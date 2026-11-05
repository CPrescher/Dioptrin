"""Unmodified Dioptrin 0.7 API control with the same NumPy mask creation.

Independent processes allow concurrent per-frame masks while retaining each
worker's geometry/LUT. Each process uses one Rayon thread. This avoids the
single-image Python method's GIL serialization without changing the package.
"""
from pathlib import Path
import argparse, json, os, time, statistics, multiprocessing, concurrent.futures
import numpy as np
DATA=Path('/tmp/dioptrin-bridge-investigation-20261105')
PONI='/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni'

def init_worker(barrier, mode):
    import dioptrin
    global FRAMES, BASE, AI
    dioptrin.set_rayon_threads(1)
    meta=json.loads((DATA/'inputs.json').read_text());shape=tuple(meta['shape'])
    FRAMES=np.memmap(DATA/'frames.f64',dtype='<f8',mode='r',shape=shape)
    BASE=np.fromfile(DATA/'mask.u8',dtype=np.uint8).reshape(shape[1:]).astype(bool)
    kwargs = {} if mode == 'legacy' else {'mask_mode': mode}
    AI=dioptrin.Integrator(PONI,method='pixel_split',unit='2th_deg',polarization_factor=.965,fast_math=True,**kwargs)
    AI.set_mask(BASE.astype(np.uint8)); AI.integrate1d(FRAMES[0],2000)
    barrier.wait(timeout=120)

def job(item):
    i,threshold,errors=item; frame=FRAMES[i];start=time.perf_counter()
    plugin=np.zeros(frame.shape,dtype=bool);plugin |= frame>threshold
    mask=np.logical_or(BASE,plugin).astype(np.uint8)
    AI.set_mask(mask)
    after_mask=time.perf_counter()
    result=AI.integrate1d(frame,2000,errors=errors)
    out=np.concatenate([result.intensity,result.errors]) if errors else np.array(result.intensity)
    end=time.perf_counter()
    return out, {'mask_and_set_mask_seconds':after_mask-start,'integration_seconds':end-after_mask}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--output',type=Path,required=True);p.add_argument('--mask-mode', choices=['legacy','dynamic'],default='dynamic');args=p.parse_args()
    import importlib.metadata
    record={'mask_mode':args.mask_mode,'dioptrin_version':importlib.metadata.version('dioptrin'),'workers':args.workers,'measurements':[],'numerical':[]}
    meta=json.loads((DATA/'inputs.json').read_text());n=meta['shape'][0]
    context=multiprocessing.get_context('spawn'); barrier=context.Barrier(args.workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=init_worker,initargs=(barrier,args.mask_mode)) as pool:
        for errors in [False,True]:
            items=[(i,230.,errors) for i in range(n)]
            # All process startup and LUT preparation finish before measurement.
            list(pool.map(job,items))
            reference=None
            for rep in range(5):
                start=time.perf_counter();loops=0
                while True:
                    results=list(pool.map(job,items));loops+=1
                    elapsed=time.perf_counter()-start
                    if elapsed>=1.:break
                arrays=np.stack([x[0] for x in results])
                if reference is None:reference=arrays.copy()
                np.testing.assert_array_equal(arrays,reference)
                row={'event':'measurement','workers':args.workers,'threshold':230.,'errors':errors,'repeat':rep,'frames':n,'seconds':elapsed,'loops':loops,'fps':n*loops/elapsed,
                     'stage_mean_seconds':{k:statistics.mean(x[1][k] for x in results) for k in results[0][1]}}
                print(json.dumps(row),flush=True);record['measurements'].append(row)
            outpath=args.output.with_suffix('.poisson.npz' if errors else '.intensity.npz')
            np.savez(outpath,output=reference)
            record['numerical'].append({'errors':errors,'reference_file':str(outpath),'repeats_byte_identical':True})
            args.output.write_text(json.dumps(record,indent=2)+'\n')
    args.output.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
