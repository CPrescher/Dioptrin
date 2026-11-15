"""Actual Dioptas models and readers, separate fresh-process measurements."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[k]='1'
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import ctypes,gc,hashlib,json,resource,statistics,sys,time
import numpy as np,psutil
B=Path(__file__).resolve().parent
def busy():return sum(x.user+x.system+x.nice for x in psutil.cpu_times(percpu=True))
class VM(ctypes.Structure):
 _fields_=[(k,ctypes.c_uint64) for k in ['resident','resident_peak','physical_footprint','reusable','internal','external','compressed','malloc_in_use','malloc_reserved','malloc_peak']]+[('malloc_blocks',ctypes.c_uint32),('task_info_result',ctypes.c_int)]
helper=ctypes.CDLL('/Users/clemens/Documents/Codex/2026-10-23/can-x20/work/code-audit/memory_snapshot.dylib')
helper.audit_memory_snapshot.argtypes=[ctypes.POINTER(VM)];helper.audit_memory_snapshot.restype=None
def snapshot(stage):
 v=VM();helper.audit_memory_snapshot(ctypes.byref(v));assert v.task_info_result==0
 r=dict(stage=stage,rss=psutil.Process().memory_info().rss,vm={k:getattr(v,k) for k,_ in v._fields_})
 print(json.dumps(r),flush=True);return r
def worker(version,dataset,workers,trial,mode='batch',errors=False,backend='dioptrin'):
 import hdf5plugin
 hdf5plugin.register(force=True)
 from qtpy.QtWidgets import QApplication
 app=QApplication.instance() or QApplication([])
 from dioptas.model.Configuration import Configuration
 from dioptas.model.integration_task import PersistentIntegrationEngine
 from dioptas.model.loader.hdf5Loader import Hdf5Image
 import dioptrin,dioptas
 from importlib.metadata import version as package_version
 dioptrin.validate_license()
 # Same explicit reader budget in baseline/candidate, independent of machine defaults.
 Hdf5Image.gen_frames.__defaults__=(None,min(2,workers))
 d=next(x for x in json.loads((B/'datasets.json').read_text()) if x['name']==dataset)
 out=dict(version=version,dataset=dataset,workers=workers,reader_workers=min(2,workers),trial=trial,mode=mode,errors=errors if mode!='batch' else False,backend=backend,records=[],validation=[],python=sys.version,dioptrin=package_version('dioptrin'),dioptas_source=str(Path(dioptas.__file__).parent))
 out['dependencies']={k:package_version(k) for k in ['numpy','h5py','hdf5plugin','bitshuffle','fabio','Pillow','pyFAI','qtpy']}
 import qtpy
 out['qt_binding']=qtpy.API_NAME;out['qt_version']=qtpy.QT_VERSION
 native=Path(dioptrin.__file__)
 if native.suffix!='.so':native=next(native.parent.glob('*.so'))
 out['native_sha256']=hashlib.sha256(native.read_bytes()).hexdigest()
 out['memory']=[snapshot('setup')]
 cfg=Configuration();cfg.auto_integrate_pattern=False;cfg.auto_integrate_cake=False
 t=time.perf_counter();cfg.calibration_model.load(d['poni']);cfg.img_model.load(d['files'][0])
 cal=cfg.calibration_model;cal.use_dioptrin=backend=='dioptrin'
 if backend=='dioptrin':cal._create_dioptrin_integrator();assert cal.use_dioptrin
 cal.dioptrin_num_workers=workers;cal.polarization_factor=.965
 cfg.integration_rad_points=2000;cfg.trim_trailing_zeros=False;cfg.use_mask=True
 cfg.mask_model.set_mask_data(np.load(d['mask']))
 cfg.batch_model.set_image_files(d['files']);out['configuration_s']=time.perf_counter()-t
 out['batch_native_eligible']=cal.can_use_dioptrin_batch(cfg.integration_unit,cfg.oned_azimuth_range)
 if mode=='batch':assert out['batch_native_eligible']
 out['memory'].append(snapshot('configured'))
 def batch():
  cfg.batch_model.integrate_raw_data(0,32,1,use_all=True)
  return dict(axis=cfg.batch_model.binning,intensity=cfg.batch_model.data,positions=cfg.batch_model.pos_map)
 engine=PersistentIntegrationEngine();cfg.calculate_poisson_errors=errors
 def interactive():
  current=None;values=[];sigmas=[];axis=None
  for file_index,pos in cfg.batch_model.pos_map_all:
   if file_index!=current:
    cfg.img_model.load(d['files'][file_index]);current=file_index
   cfg.img_model.load_series_img(int(pos)+1)
   task=cfg.create_integration_task(calculate_pattern=True,calculate_cake=False)
   result=engine.compute(task).pattern;axis=result.radial
   values.append(result.intensity)
   if errors:sigmas.append(result.sigma)
  r=dict(axis=axis,intensity=np.array(values))
  if errors:r['sigma']=np.array(sigmas)
  return r
 run=batch if mode=='batch' else interactive
 reference_mode='batch' if mode=='batch' else f'interactive-{backend}-{errors}'
 def validate(result,mode):
  ref=B/'references'/f'{dataset}-{mode}.npz';ref.parent.mkdir(exist_ok=True)
  if version in ['baseline','published'] and trial==0:np.savez_compressed(ref,**result)
  with np.load(ref) as expected:
   for k,a in result.items():assert a.dtype==expected[k].dtype and a.shape==expected[k].shape and a.tobytes()==expected[k].tobytes(),(dataset,mode,k)
  out['validation'].append(dict(mode=mode,frames=32,fields=list(result),bitwise_equal=True,reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
 out['memory'].append(snapshot('first_batch'))
 t=time.perf_counter();r=run();out['first_batch_s']=time.perf_counter()-t;validate(r,reference_mode);del r
 if mode!='batch' and backend=='dioptrin':assert engine.dioptrin_integrator is not None
 out['memory'].append(snapshot('warm'))
 out['rss_ready']=psutil.Process().memory_info().rss
 blocks=[];gc.collect();t=time.perf_counter();r=run();pilot=time.perf_counter()-t;del r
 loops=max(1,min(50,int(.5/max(pilot,1e-6))))
 for _ in range(3):
  t=time.perf_counter();cpu=time.process_time();a=busy()
  for __ in range(loops):r=run()
  wall=time.perf_counter()-t;used=time.process_time()-cpu
  blocks.append(dict(wall_s=wall,loops=loops,fps=32*loops/wall,cpu_s=used,other_busy_cores=max(0,(busy()-a-used)/wall)));validate(r,reference_mode);del r
 out['records'].append(dict(mode='actual_BatchModel' if mode=='batch' else 'load_create_task_PersistentIntegrationEngine',errors=out['errors'],fps=statistics.median(x['fps'] for x in blocks),blocks=blocks))
 out['rss_after']=psutil.Process().memory_info().rss;out['peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 out['memory'].append(snapshot('after_warm'))
 out['qualified']=statistics.median(x['other_busy_cores'] for x in blocks)<=3.6
 suffix='' if mode=='batch' else f'-interactive-{backend}-errors{errors}'
 (B/f'point-{version}-{dataset}-c{workers}-t{trial}{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(version=version,dataset=dataset,workers=workers,fps=out['records'][0]['fps'],rss=out['peak_rss'],qualified=out['qualified'])),flush=True)
if __name__=='__main__':worker(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),*(sys.argv[5:6]),**({'errors':sys.argv[6]=='True','backend':sys.argv[7]} if len(sys.argv)>6 else {}))
