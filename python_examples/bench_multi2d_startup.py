"""Matched first-frame/cached-frame timing and numerical comparison.

Runs native Rust and Python APIs separately. Image decoding and input preparation
are excluded from integration timing. Each method has a fresh native process and
fresh Python Integrators. Three preloaded frames cycle during warm calls.
Python timing ends at IntegrationResult return; materializing its intensity/errors
properties into Python lists or NumPy arrays is excluded, as is UI serialization.

Example:
  RAYON_NUM_THREADS=8 python python_examples/bench_multi2d_startup.py \
    --eiger-dir /path/to/test_data --report result.json
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import dioptrin
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["polygon_fractional", "polygon_pyfai", "subpixels21", "subpixels41"]


def digest(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def timing(construct, first, samples):
    return dict(construct_ms=construct, first_integration_ms=first,
                first_frame_ms=construct+first, cached_median_ms=float(np.median(samples)),
                cached_p95_ms=float(np.percentile(samples,95)), samples_ms=samples)


def write_poni(path, p, shape):
    path.write_text("poni_version: 2.1\nDetector: Detector\nDetector_config: "+
                    json.dumps(dict(pixel1=p["pixel1"],pixel2=p["pixel2"],max_shape=shape,orientation=3))+"\n"+
                    "\n".join(f"{key}: {p[name]}" for key,name in
                              [("Distance","distance"),("Poni1","poni1"),("Poni2","poni2"),
                               ("Rot1","rot1"),("Rot2","rot2"),("Rot3","rot3"),("Wavelength","wavelength")])+"\n")


def cases(eiger_dir):
    p=dict(pixel1=0.000172,pixel2=0.000172,distance=0.2,poni1=0.09,poni2=0.08,
           rot1=0.01,rot2=-0.01,rot3=0.02,wavelength=3.738e-11)
    shapes=[(1043,981),(768,640)]
    geometries=[p,dict(p,pixel1=0.00015,pixel2=0.0002,distance=0.25,poni1=0.05,poni2=0.055,rot1=-0.04)]
    masks=[np.ascontiguousarray((np.indices(shape).sum(axis=0)%97)==0,dtype=np.uint8) for shape in shapes]
    frames=[]
    for f in range(3):
        frames.append([np.ascontiguousarray(100.0+30*np.sin(np.indices(shape)[0]/17.0)**2+
                                            (np.indices(shape)[1]+f)%23,dtype=np.float64) for shape in shapes])
    yield "unequal_pair_1p5M",geometries,masks,frames,dict(input="synthetic",shapes=shapes)
    del frames
    if eiger_dir:
        import fabio
        import h5py
        import hdf5plugin  # registers the bitshuffle filter
        import pyFAI
        h5=eiger_dir/"eiger_0000.h5";poni=eiger_dir/"eiger_0000.poni"
        geometry=pyFAI.load(str(poni))
        if int(geometry.detector.orientation)!=2:
            raise ValueError("This Eiger fixture adapter expects orientation 2")
        mask=np.ascontiguousarray((fabio.open(str(eiger_dir/"eiger_0000.mask")).data!=0),dtype=np.uint8)
        decode=[];frames=[]
        with h5py.File(h5,"r") as handle:
            dataset=handle["entry_0000/instrument/eiger/data"]
            for f in range(3):
                t=time.perf_counter();raw=dataset[f];decode.append((time.perf_counter()-t)*1000)
                image=np.ascontiguousarray(raw[::-1],dtype=np.float64)
                if np.any((raw[::-1]>=np.iinfo(np.uint32).max-1)&(mask==0)):
                    raise ValueError("Eiger sentinel values are not covered by the supplied mask")
                frames.append([image])
        p=dict(pixel1=geometry.pixel1,pixel2=geometry.pixel2,distance=geometry.dist,
               poni1=geometry.poni1,poni2=geometry.poni2,rot1=geometry.rot1,rot2=geometry.rot2,
               rot3=geometry.rot3,wavelength=geometry.wavelength)
        metadata=dict(input=str(h5),dataset="entry_0000/instrument/eiger/data",frames=[0,1,2],
                      shapes=[frames[0][0].shape],mask_pixels=int(mask.sum()),
                      array_sha256=[digest(f[0]) for f in frames],mask_sha256=digest(mask),
                      hdf5_read_decode_ms=decode,orientation="original image 2; image rows flipped; supplied TIFF mask already bottom-up",
                      geometry=p)
        yield "eiger_9M",[p],[mask],frames,metadata


def compare(actual, reference, count, ref_count):
    covered=count>1e-6;ref_covered=ref_count>1e-6
    common=covered&ref_covered
    positive=common&(reference>0)
    delta=actual[common]-reference[common]
    relative=np.abs(actual[positive]-reference[positive])/np.maximum(1,np.abs(reference[positive]))
    return dict(common_covered_bins=int(common.sum()),positive_reference_bins=int(positive.sum()),
                method_only_covered_bins=int(np.count_nonzero(covered&~ref_covered)),
                reference_only_covered_bins=int(np.count_nonzero(ref_covered&~covered)),
                normalized_rms=float(np.sqrt(np.mean(delta**2))/max(1,np.sqrt(np.mean(reference[common]**2)))),
                relative_median=float(np.median(relative)),relative_p95=float(np.percentile(relative,95)),
                relative_p99=float(np.percentile(relative,99)),relative_max=float(relative.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eiger-dir",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--case",choices=["all","unequal_pair_1p5M","eiger_9M"],default="all")
    parser.add_argument("--repeats",type=int,default=30)
    args=parser.parse_args()
    if args.threads<1 or args.repeats<2:parser.error("positive threads and at least two repeats required")
    if args.case == "eiger_9M" and args.eiger_dir is None:parser.error("--case eiger_9M requires --eiger-dir")
    dioptrin.set_rayon_threads(args.threads)
    subprocess.run(["cargo","build","--release","--example","bench_multi2d_case"],cwd=ROOT,check=True)
    report=dict(platform=platform.platform(),cpu=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
                python=platform.python_version(),rayon_threads=args.threads,repeats=args.repeats,
                timing_scope="predecoded input to returned result; excludes decoding, process/threadpool startup, Python list getters and UI serialization",
                n_radial=2000,n_azimuthal=360,unit="q_A^-1",radial_range=[0,20],errors=True,
                polarization_factor=0.6,normalization="absolute",cases=[])
    for name,geometries,masks,frames,metadata in cases(args.eiger_dir):
        if args.case != "all" and name != args.case:
            continue
        row=dict(name=name,input=metadata,results=[])
        print(json.dumps(dict(case=name,shapes=[f.shape for f in frames[0]])),flush=True)
        with tempfile.TemporaryDirectory(prefix="dioptrin-startup-") as tmp:
            directory=Path(tmp);nd=len(geometries)
            text=f"{nd} {len(frames)} 2000 360 0 20\n"
            for d,(p,mask) in enumerate(zip(geometries,masks)):
                write_poni(directory/f"det{d}.poni",p,mask.shape);mask.tofile(directory/f"det{d}.mask")
                text+=f"{mask.shape[0]} {mask.shape[1]} {d+1}\n"
            (directory/"config.txt").write_text(text)
            for f,images in enumerate(frames):
                for d,image in enumerate(images):image.astype("<f8",copy=False).tofile(directory/f"frame{f}-det{d}.f64")
            arrays={}
            for label in LABELS:
                t=time.perf_counter()
                process=subprocess.run([str(ROOT/"target/release/examples/bench_multi2d_case"),str(directory),label,
                                        str(args.repeats),str(args.threads)],text=True,capture_output=True,check=True)
                native=json.loads(process.stdout)
                native=timing(native["construct_ms"],native["first_integration_ms"],native["samples_ms"])
                arrays[label]={k:np.fromfile(directory/f"{label}-{k}.f64",dtype="<f8") for k in ["intensity","errors","count"]}
                print(json.dumps(dict(case=name,method=label,native_first_ms=native["first_frame_ms"],
                                     native_cached_ms=native["cached_median_ms"],wall_seconds=time.perf_counter()-t)),flush=True)
                gc.collect()
                method="supersampled" if label.startswith("subpixels") else "pixel_split"
                n_ss=int(label.removeprefix("subpixels")) if method=="supersampled" else 3
                t=time.perf_counter()
                integrators=[dioptrin.Integrator.from_poni_dict(p,method=method,n_ss=n_ss,unit="q_A^-1",
                                                              radial_range=(0,20),polarization_factor=0.6) for p in geometries]
                for ai,mask in zip(integrators,masks):ai.set_mask(mask)
                multi=dioptrin.MultiIntegrator(integrators,pixel_split_mode="pyfai_histogram" if label=="polygon_pyfai" else "fractional")
                construct=(time.perf_counter()-t)*1000
                def run(f):return multi.integrate2d(frames[f],2000,360,errors=True,normalization="absolute",normalization_factors=list(range(1,nd+1)))
                t=time.perf_counter();first=run(0);first_ms=(time.perf_counter()-t)*1000
                actual=np.array(first.intensity);error=np.array(first.errors)
                np.testing.assert_allclose(actual,arrays[label]["intensity"],rtol=2e-12,atol=1e-8)
                np.testing.assert_allclose(error,arrays[label]["errors"],rtol=2e-12,atol=1e-8)
                del first,actual,error
                samples=[]
                for i in range(args.repeats):
                    t=time.perf_counter();result=run((i+1)%len(frames));samples.append((time.perf_counter()-t)*1000);del result
                entry=dict(method=label,native=native,python_api=timing(construct,first_ms,samples),native_python_agree=True)
                row["results"].append(entry)
                print(json.dumps(dict(case=name,method=label,python_first_ms=entry["python_api"]["first_frame_ms"],
                                     python_cached_ms=entry["python_api"]["cached_median_ms"])),flush=True)
                del multi,integrators,ai;gc.collect()
            ref=arrays["subpixels41"]
            for entry in row["results"]:
                current=arrays[entry["method"]]
                entry["intensity_vs_subpixels41"]=compare(current["intensity"],ref["intensity"],current["count"],ref["count"])
                entry["std_vs_subpixels41"]=compare(current["errors"],ref["errors"],current["count"],ref["count"])
        report["cases"].append(row)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    print("Report:",args.report,flush=True)


if __name__=="__main__":main()
