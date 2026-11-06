"""Prepare actual Eiger crops, adjusted PONI, production weights and API references."""
from pathlib import Path
import json
import re
import subprocess
import dioptrin
import numpy as np

folder=Path(__file__).resolve().parent
root=Path('/tmp/dioptrin-bridge-investigation-20261105')
source=Path('/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni')
meta=json.loads((root/'inputs.json').read_text())
frames=np.memmap(root/'frames.f64',dtype='<f8',mode='r',shape=tuple(meta['shape']))
mask=np.fromfile(root/'mask.u8',dtype=np.uint8).reshape(meta['shape'][1:])
text=source.read_text()
value=lambda name:float(re.search(r'^'+name+r':\s*(.*)$',text,re.M)[1])
for size,bins in [(1024,2000),(256,512)]:
 y=int(value('Poni1')/value('PixelSize1'))-size//2
 x=int(value('Poni2')/value('PixelSize2'))-size//2
 out=Path(f'/tmp/dioptrin-fixed-crop-{size}')
 out.mkdir(exist_ok=True)
 local=np.ascontiguousarray(frames[:,y:y+size,x:x+size],dtype='<f8')
 static=np.ascontiguousarray(mask[y:y+size,x:x+size],dtype=np.uint8)
 local.tofile(out/'frames.f64');static.tofile(out/'mask.u8')
 record={**meta,'shape':list(local.shape),'bins':bins,'crop_origin':[y,x],
         'source_calibration':str(source),'source_frames':str(root/'frames.f64')}
 (out/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
 poni=text
 for name,offset,pixel in [('Poni1',y,'PixelSize1'),('Poni2',x,'PixelSize2')]:
  poni=re.sub(r'^'+name+r':.*$',f'{name}: {value(name)-offset*value(pixel):.17g}',poni,flags=re.M)
 (out/'calibration.poni').write_text(poni)
 subprocess.run([str(folder/'target/release/export-crop-lut'),str(out/'calibration.poni'),str(size),'0','0',str(bins),str(out/'lut')],check=True)
 native=dioptrin.Integrator(str(out/'calibration.poni'),unit='2th_deg',polarization_factor=.965,mask_mode='fixed')
 for frozen in (False,True):
  effective=np.ascontiguousarray(static | (local[0]>230) if frozen else static,dtype=np.uint8)
  native.set_mask(effective)
  results=[native.integrate1d(frame,bins,errors=True) for frame in local]
  name='frozen' if frozen else 'static'
  np.asarray([r.intensity for r in results],dtype='<f8').tofile(out/f'current-native-{name}.intensity.f64')
  np.asarray([r.errors for r in results],dtype='<f8').tofile(out/f'current-native-{name}.errors.f64')
 print('Prepared crop',size,'bins',bins,'origin',(y,x),flush=True)
