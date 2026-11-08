"""Independent pyFAI oracle for method='pyfai_full'; run against the built wheel."""
import json
import tempfile
from pathlib import Path
import numpy as np
import tifffile
import dioptrin
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.ext.splitPixel import fullSplit1D_engine, fullSplit2D_engine

shape=(24,32)
y,x=np.indices(shape)
data=np.ascontiguousarray(30.+y*3+x*2+120*np.exp(-((y-13)**2+(x-17)**2)/30))
p=dict(dist=.08,poni1=-.0013,poni2=-.0021,rot1=.007,rot2=-.011,rot3=.02,pixel1=.0001,pixel2=.00012,wavelength=.5e-10)
ai=AzimuthalIntegrator(detector=Detector(pixel1=p['pixel1'],pixel2=p['pixel2']),**{k:v for k,v in p.items() if k not in ('pixel1','pixel2')})
receipts=[]
with tempfile.TemporaryDirectory(prefix='dioptrin-pyfai-full-') as directory:
    directory=Path(directory)
    poni=directory/'geometry.poni';ai.save(str(poni))
    image=directory/'image.tif';tifffile.imwrite(image,data[::-1].astype(np.float64))
    for unit in ('2th_rad','2th_deg','q_A^-1','q_nm^-1'):
      for pol in (0.,.6):
        integrator=dioptrin.Integrator(str(poni),method='pyfai_full',unit=unit,polarization_factor=pol)
        result=integrator.integrate1d(data,64,errors=True)
        centers=np.asarray(result.radial);step=centers[1]-centers[0]
        bounds=(centers[0]-.5*step,(centers[-1]+.5*step)/(1+np.finfo(np.float32).eps))
        corners=ai.array_from_unit(shape,typ='corner',unit=unit)
        ref=fullSplit1D_engine(corners,data,bins=64,pos0_range=bounds,variance=data,solidangle=ai.solidAngleArray(shape),polarization=ai.polarization(shape,factor=pol))
        np.testing.assert_allclose(result.intensity,ref.intensity,rtol=3e-5,atol=2e-4)
        np.testing.assert_allclose(result.errors,ref.sigma,rtol=3e-5,atol=2e-5)
        cake=integrator.integrate2d(data,64,24,errors=True)
        ref2=fullSplit2D_engine(corners,data,bins=(64,24),pos0_range=bounds,pos1_range=(-np.pi,np.pi/(1+np.finfo(np.float32).eps)),variance=data,solidangle=ai.solidAngleArray(shape),polarization=ai.polarization(shape,factor=pol))
        np.testing.assert_allclose(np.asarray(cake.intensity).reshape(24,64),ref2.intensity,rtol=1e-4,atol=8e-4)
        np.testing.assert_allclose(np.asarray(cake.errors).reshape(24,64),ref2.sigma,rtol=1e-4,atol=8e-5)
        # The same public method supports cached, typed-file and method-switch paths.
        typed=integrator.integrate1d(str(image),64,errors=True)
        np.testing.assert_allclose(typed.intensity,result.intensity,rtol=1e-12,atol=1e-10)
        integrator.set_method('pixel_split')
        polygon=integrator.integrate1d(data,64)
        integrator.set_method('pyfai_full')
        np.testing.assert_array_equal(integrator.integrate1d(data,64).intensity,result.intensity)
        assert not np.allclose(polygon.intensity,result.intensity,rtol=1e-9,atol=1e-9)
        receipts.append(dict(unit=unit,polarization=pol,maxDifference=float(np.max(np.abs(np.asarray(result.intensity)-ref.intensity)))))
print(json.dumps(dict(pyFAI=pyFAI.version,cases=receipts),indent=2))
