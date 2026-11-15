from pathlib import Path
import hashlib,json
import h5py,hdf5plugin,numpy as np,tifffile
hdf5plugin.register(force=True)
B=Path(__file__).resolve().parent
old=Path('/Users/clemens/Documents/Codex/2026-10-23/can-x20/work/real-world-io')
datasets=[]
for source in json.loads((old/'datasets.json').read_text()):
 name=source['name'];paths=[x['path'] for x in source['files']]
 datasets.append(dict(name=name+'-TIFF',detector=name,format='TIFF',files=[paths[i%len(paths)] for i in range(32)],poni=source['poni'],mask=source['mask'],frames=32))
 unique=paths[:1] if name=='Pilatus1M' else paths[:16]
 for fmt,kw in [('bitshuffle',dict(hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))),('gzip',dict(compression='gzip',compression_opts=1,shuffle=True))]:
  dest=B/'data'/f'{name}-{fmt}.h5';dest.parent.mkdir(exist_ok=True)
  if dest.exists():
   with h5py.File(dest,'r') as f:complete=f.attrs.get('complete',False)
   if not complete:dest.rename(dest.with_suffix('.incomplete-'+str(__import__('time').time_ns())+'.h5'))
  if not dest.exists():
   with h5py.File(dest,'w') as f:
    ds=f.create_dataset('entry/data/data',shape=(16,*source['shape']),dtype='int32',chunks=(1,*source['shape']),**kw)
    for i in range(16):ds[i]=tifffile.imread(unique[i%len(unique)])
    f.attrs['complete']=True
  datasets.append(dict(name=name+'-'+fmt,detector=name,format=fmt,files=[str(dest)]*2,poni=source['poni'],mask=source['mask'],frames=32,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(B/'datasets.json').write_text(json.dumps(datasets,indent=2)+'\n')
print('Prepared',len(datasets),'TIFF and compressed HDF5 datasets',flush=True)
