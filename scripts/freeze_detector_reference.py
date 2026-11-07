"""Small independent pyFAI fixtures for the native physical detector model."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyFAI
from pyFAI.detectors import ALL_DETECTORS
from pyFAI.geometry import Geometry
from pyFAI.spline import Spline
root=Path(__file__).resolve().parents[1]
rows=[]
for cls in sorted(set(ALL_DETECTORS.values()),key=lambda c:c.__name__):
    if cls.__name__=="Detector":continue
    d=cls(); d._orientation=3
    if d.CORNERS!=4:continue
    h,w=d.shape
    r=np.array([0.5,1.125,h/3+0.37,h-1.25,h-0.5])
    c=np.array([0.5,w-0.5,w/2+0.13,w/4+0.77,w-0.5])
    y,x,z=d.calc_cartesian_positions(r-0.5,c-0.5)
    m=d.calc_mask()
    rows.append(dict(name=(d.aliases or [cls.__name__])[0],shape=[h,w],rows=r.tolist(),cols=c.tolist(),
                     positions=np.array([np.zeros_like(y) if z is None else z,y,x]).T.tolist(),
                     masked=0 if m is None else int(np.count_nonzero(m)),mask_sha256=hashlib.sha256(bytes(h*w) if m is None else np.asarray(m,dtype=bool).astype(np.uint8).tobytes()).hexdigest()))
    print(rows[-1]["name"],flush=True)
    del d,m;gc.collect()
spl=Spline(sys.argv[1]); samples=[]
for x,y in [(-1.,-5.),(0.,0.),(.5,.5),(100.25,255.75),(1024.5,1024.5),(2048.,2048.),(2500.,2500.)]:
    xx=np.array([x]);yy=np.array([y])
    samples.append([x,y,float(spl.splineFuncX(xx,yy,True)[0]),float(spl.splineFuncY(xx,yy,True)[0])])
fit=[]
for r1,r2,r3 in [(0.,0.,0.),(.12,-.07,.03),(-.21,.17,-.13),(.002,.003,.1),(.6,-.2,.8)]:
    g=Geometry(dist=.234,poni1=.043,poni2=.055,rot1=r1,rot2=r2,rot3=r3,pixel1=79e-6,pixel2=123e-6,wavelength=.4e-10)
    f=g.getFit2D()
    fit.append(dict(rotations=[r1,r2,r3],values={k:float(f[k]) for k in ["directDist","centerX","centerY","tilt","tiltPlanRotation","pixelX","pixelY"]}))
(root/"test_data/detector_reference.json").write_text(json.dumps(dict(pyfai=pyFAI.version,detectors=rows,spline=samples,fit2d=fit),indent=2)+"\n")
(root/"test_data/detector_reference.spline").write_bytes(Path(sys.argv[1]).read_bytes())
