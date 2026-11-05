"""Unmodified Jon Wright public API: rebuild mask-folded matrix every frame.

No precomputed per-frame masks, denominators, matrices, plans or contexts are
reused. Only original geometry-dependent coefficient matrices are cached.
"""
import argparse, collections, gc, hashlib, importlib.metadata, json, os
from pathlib import Path
import statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
import bslz4_to_sparse as b
import numpy as np
import psutil
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = Path('/tmp/dioptrin-bridge-investigation-20261105')
LUT = Path('/tmp/dioptrin-bslz4-lut-20261105/pixel_split')


def other_cpu():
    out = {}
    for p in psutil.process_iter(['pid', 'create_time', 'cpu_times']):
        try:
            if p.pid == os.getpid() or p.info['cpu_times'] is None: continue
            t = p.info['cpu_times']; out[(p.pid, p.info['create_time'])] = t.user+t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return out


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[323.,230.,207.])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    meta = json.loads((DATA/'inputs.json').read_text()); nframes,h,w = meta['shape']; npix=h*w
    frames = np.memmap(DATA/'frames.f64', dtype='<f8', mode='r', shape=tuple(meta['shape']))
    base = np.fromfile(DATA/'mask.u8', dtype=np.uint8).reshape(h,w).astype(bool)
    ptr = np.fromfile(LUT/'indptr.u64', dtype='<u8').astype(np.int64)
    idx = np.fromfile(LUT/'indices.u32', dtype='<u4').astype(np.int32)
    weights = np.fromfile(LUT/'weights.f64', dtype='<f8')
    csr = sparse.csr_matrix((weights,idx,ptr),shape=(len(ptr)-1,npix))
    csr.sort_indices(); assert csr.has_sorted_indices
    pixel_norm = np.fromfile(LUT/'normalization.f64', dtype='<f8')
    radial = np.fromfile(LUT/'radial.f64', dtype='<f8'); bins=len(radial)
    # Fixed geometry conversion is excluded, just as LUT construction is
    # excluded for Dioptrin. All per-frame mask preparation remains timed.
    geometry = csr
    if args.errors:
        variance = csr.copy(); variance.data **= 2
        geometry = sparse.vstack([csr,variance],format='csr'); del variance
    csc = geometry.tocsc().astype(np.float32)
    if args.errors: del geometry
    gc.collect()
    # Matrix arrays stay constant and read-only; the public constructor copies,
    # validates, folds the new mask, analyses and builds the selected layout.
    for a in [csc.data,csc.indices,csc.indptr]: a.flags.writeable=False
    record = {'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in ['bslz4_to_sparse','dioptrin','numpy','scipy']},
              'build_info':b.build_info(),'input_metadata':meta,'weights':'Jon float32, denominator float64 original weights',
              'pipeline':{'dot':'csc','route':'dense'},'measurements':[],'numerical':[],'mask_stats':[]}
    emit({'event':'setup','nnz':int(csr.nnz),'jon_nnz_before_mask':int(csc.nnz),'workers':args.workers,'errors':args.errors,'build_info':b.build_info()})

    def job(item):
        i,threshold = item; frame=frames[i]
        start=time.perf_counter()
        # Same upper-threshold rule as Dioptas ThresholdMaskPlugin, followed
        # by the static-mask OR. Invert to Jon's convention: 1 means valid.
        plugin_mask=np.zeros(frame.shape,dtype=bool)
        plugin_mask |= frame > threshold
        invalid=np.logical_or(base,plugin_mask)
        valid=np.logical_not(invalid).astype(np.uint8)
        after_mask=time.perf_counter()
        # Recalculate the SAME float64 normalization denominator for this
        # mask. Masked pixels contribute zero; no denominator is cached.
        masked_norm=np.where(invalid.ravel(),0.,pixel_norm)
        denominator=np.asarray(csr@masked_norm).ravel()
        del masked_norm
        after_norm=time.perf_counter()
        decoder=b.chunk2sparseCSC(valid,csc,dtype=np.float64,codec='lz4',pipeline={'dot':'csc','route':'dense'})
        after_plan=time.perf_counter()
        powder=decoder.from_frames(frame[None],float(1<<32))[2][0]
        out=np.zeros(bins*(2 if args.errors else 1),dtype=np.float64)
        np.divide(powder[:bins],denominator,out=out[:bins],where=denominator!=0)
        if args.errors:
            vv=np.zeros(bins,dtype=np.float64)
            np.divide(powder[bins:],denominator*denominator,out=vv,where=denominator!=0)
            out[bins:]=np.sqrt(np.maximum(vv,0.))
        end=time.perf_counter()
        assert np.isfinite(out).all()
        stats={'mask_seconds':after_mask-start,'normalization_seconds':after_norm-after_mask,
               'matrix_plan_seconds':after_plan-after_norm,'integration_output_seconds':end-after_plan,
               'job_seconds_before_cleanup':end-start}
        # Destruction of plan, matrices, workspace and masks happens as this
        # function returns and is included by the enclosing wall-clock timer.
        return out, stats

    count=1 if args.smoke else nframes
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for threshold in args.thresholds:
            items=[(i,threshold) for i in range(count)]
            for rep in range(-1,args.repeats if not args.smoke else 0):
                gc.collect()
                before=other_cpu(); p0=time.process_time(); start=time.perf_counter()
                results=list(pool.map(job,items))
                elapsed=time.perf_counter()-start; process=time.process_time()-p0
                after=other_cpu()
                row={'event':'measurement','workers':args.workers,'errors':args.errors,'threshold':threshold,
                     'repeat':rep,'frames':count,'seconds':elapsed,'fps':count/elapsed,'process_cores':process/elapsed,
                     'observed_other_busy_cores':sum(max(0.,v-before.get(k,v)) for k,v in after.items())/elapsed,
                     'stage_mean_seconds':{k:statistics.mean(s[k] for _,s in results) for k in results[0][1] if k.endswith('seconds') or k=='job_seconds_before_cleanup'},
                     'rss_gib_after_batch':psutil.Process().memory_info().rss/2**30}
                emit(row)
                if rep>=0: record['measurements'].append(row)
                if rep==-1:
                    # Check every frame against original float64 coefficients
                    # and against the rounded-weight matrix actually used.
                    sums={key:[] for key in ['intensity','sigma']}; rounded=[]; extras=[]
                    for i,((got,stats),_) in enumerate(zip(results,items)):
                        frame=frames[i].ravel(); invalid=np.logical_or(base.ravel(),frame>threshold)
                        den=np.asarray(csr@np.where(invalid,0.,pixel_norm)).ravel()
                        sum64=np.asarray(csr@np.where(invalid,0.,frame)).ravel()
                        expected=np.zeros_like(got); np.divide(sum64,den,out=expected[:bins],where=den!=0)
                        raw32=np.asarray(csc@np.where(invalid,0.,frame)).ravel()
                        expected32=np.zeros_like(got);np.divide(raw32[:bins],den,out=expected32[:bins],where=den!=0)
                        if args.errors:
                            # Avoid a second full sparse variance table just
                            # for reference: canonical squares are evaluated
                            # in bounded bin rows.
                            v=np.array([np.dot(csr.data[csr.indptr[j]:csr.indptr[j+1]]**2,
                                np.where(invalid[csr.indices[csr.indptr[j]:csr.indptr[j+1]]],0.,
                                    frame[csr.indices[csr.indptr[j]:csr.indptr[j+1]]])) for j in range(bins)])
                            z=np.zeros(bins);np.divide(v,den*den,out=z,where=den!=0);expected[bins:]=np.sqrt(np.maximum(z,0.))
                            z=np.zeros(bins);np.divide(raw32[bins:],den*den,out=z,where=den!=0);expected32[bins:]=np.sqrt(np.maximum(z,0.))
                        np.testing.assert_allclose(got,expected32,rtol=2e-12,atol=1e-10)
                        np.testing.assert_allclose(got,expected,rtol=2e-6,atol=1e-8)
                        for name,sl in [('intensity',slice(0,bins)),('sigma',slice(bins,None))]:
                            a=got[sl];e=expected[sl];sums[name].append((float(np.dot(a-e,a-e)),float(np.dot(e,e)),float(np.max(np.abs(a-e))) if a.size else 0.))
                        rounded.append(float(np.max(np.abs(got-expected32))))
                        extras.append(int(np.count_nonzero(invalid&~base.ravel())))
                    numeric={'event':'numerical','threshold':threshold,'errors':args.errors,'frames':count,
                        'vs_float64':{key:{'relative_l2':(sum(x[0] for x in val)/sum(x[1] for x in val))**.5 if sum(x[1] for x in val) else 0.,'max_absolute':max(x[2] for x in val)} for key,val in sums.items()},
                        'max_absolute_vs_rounded_weight_reference':max(rounded),'extra_masked_pixels':extras}
                    emit(numeric);record['numerical'].append(numeric)
                del results
                args.output.write_text(json.dumps(record,indent=2)+'\n')
    args.output.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
