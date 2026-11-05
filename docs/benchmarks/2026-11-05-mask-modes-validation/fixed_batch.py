"""Published versus local fixed-mask batch API, same real Eiger sample."""
import argparse, hashlib, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import dioptrin
ROOT=Path('/tmp/dioptrin-bridge-investigation-20261105')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
meta=json.loads((ROOT/'inputs.json').read_text());images=np.memmap(ROOT/'frames.f64',dtype='<f8',mode='r',shape=tuple(meta['shape']));frames=list(images)
mask=np.fromfile(ROOT/'mask.u8',dtype=np.uint8).reshape(meta['shape'][1:])
dioptrin.set_rayon_threads(8)
ai=dioptrin.Integrator('/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni',method='pixel_split',unit='2th_deg',polarization_factor=.965)
ai.set_mask(mask)
record={'version':importlib.metadata.version('dioptrin'),'numpy':np.__version__,'mask_mode':getattr(ai,'mask_mode','legacy'),'extension_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(dioptrin.__file__).parent.glob('*.so')},'measurements':[]}
for errors in [False,True]:
    def run():
        rows=ai.batch1d(frames,2000,errors=errors,num_workers=8,num_threads=1)
        assert all(x.is_ok() for x in rows)
        return np.stack([np.concatenate([x.result.intensity,x.result.errors]) if errors else x.result.intensity for x in rows])
    reference=run();run()
    np.savez(a.output.with_suffix('.poisson.npz' if errors else '.intensity.npz'),output=reference)
    for rep in range(5):
        start=time.perf_counter();loops=0
        while True:
            result=run();del result;loops+=1
            elapsed=time.perf_counter()-start
            if elapsed>=1.:break
        row={'errors':errors,'repeat':rep,'loops':loops,'seconds':elapsed,'fps':len(frames)*loops/elapsed}
        record['measurements'].append(row);print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(record,indent=2)+'\n')
