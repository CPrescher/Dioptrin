"""Interleaved fresh-process application benchmarks; retain rejected load trials."""
import hashlib,json,os,subprocess,threading,time
from pathlib import Path
import psutil
B=Path(__file__).resolve().parent
PY='/Users/clemens/Programming/Dioptas/.venv/bin/python'
def busy():return sum(x.user+x.system+x.nice for x in psutil.cpu_times(percpu=True))
def idle():
 a=busy();t=time.monotonic();time.sleep(1)
 return (busy()-a)/(time.monotonic()-t)
def arguments(v,ds,c,t,mode,errors,backend):
 return [PY,str(B/'bench.py'),v,ds,str(c),str(t),mode,str(errors),backend]
def env(v):
 e=os.environ.copy();parts=[B/('baseline-Dioptas' if v in ['baseline','published'] else 'Dioptas')]
 if v=='candidate':parts.append(B/'candidate-native')
 elif v in ['baseline','app_only']:parts.append(B/'baseline-native')
 parts.append(B/'runtime');e['PYTHONPATH']=':'.join(map(str,parts));return e
def filename(v,ds,c,t,mode,errors,backend):
 return f'point-{v}-{ds}-c{c}-t{t}'+('' if mode=='batch' else f'-interactive-{backend}-errors{errors}')+'.json'
def run(v,ds,c,t,mode='batch',errors=False,backend='dioptrin',attempt=0):
 tag=filename(v,ds,c,t,mode,errors,backend);samples=[];stage=['startup'];stop=threading.Event();events=[]
 with (B/'logs'/(tag+f'-a{attempt}.log')).open('w') as log:
  child=subprocess.Popen(arguments(v,ds,c,t,mode,errors,backend),env=env(v),stdout=subprocess.PIPE,stderr=log,text=True);proc=psutil.Process(child.pid)
  def sample():
   while not stop.is_set():
    try:samples.append(dict(stage=stage[0],rss=proc.memory_info().rss))
    except psutil.NoSuchProcess:pass
    stop.wait(.01)
  thread=threading.Thread(target=sample,daemon=True);thread.start()
  for line in child.stdout:
   log.write(line);log.flush()
   try:event=json.loads(line)
   except json.JSONDecodeError:continue
   if 'stage' in event:stage[0]=event['stage'];events.append(event)
  rc=child.wait();stop.set();thread.join()
 if rc:raise RuntimeError(tag+' '+(B/'logs'/(tag+f'-a{attempt}.log')).read_text()[-3000:])
 path=B/tag;out=json.loads(path.read_text());out.update(attempt=attempt,rss_samples=samples)
 out['warm_peak_rss']=max(x['rss'] for x in samples if x['stage']=='warm')
 out['sampled_peak_rss']=max(x['rss'] for x in samples)
 destination=B/'results'/tag if out['qualified'] else B/'load-rejected'/(tag[:-5]+f'-a{attempt}-{time.time_ns()}.json')
 if t==0:destination=B/'reference-processes'/tag
 destination.write_text(json.dumps(out,indent=2)+'\n');path.unlink()
 print(json.dumps(dict(version=v,dataset=ds,mode=mode,errors=errors,backend=backend,workers=c,trial=t,fps=round(out['records'][0]['fps'],1),warm_rss_GiB=round(out['warm_peak_rss']/2**30,3),qualified=out['qualified'])),flush=True)
 return out
def main():
 for name in ['results','logs','load-rejected','reference-processes']:(B/name).mkdir(exist_ok=True)
 datasets=[x['name'] for x in json.loads((B/'datasets.json').read_text())]
 interactive=[x for x in datasets if x.endswith(('-TIFF','-bitshuffle'))]
 # Frozen results from the unchanged application/published 0.6.1 extension.
 for ds in datasets:
  if not (B/'references'/(ds+'-batch.npz')).exists():run('baseline',ds,1,0)
 for ds in interactive:
  for errors in [False,True]:
   for backend in ['dioptrin','pyfai']:
    if not (B/'references'/f'{ds}-interactive-{backend}-{errors}.npz').exists():run('published',ds,1,0,'interactive',errors,backend)
 cases=[]
 for t in [1,2]:
  for c in ([1,8,2,4] if t==1 else [4,2,8,1]):
   for ds in datasets:
    for v in (['baseline','candidate'] if (t+c+datasets.index(ds))%2 else ['candidate','baseline']):cases.append((v,ds,c,t,'batch',False,'dioptrin'))
  for ds in interactive:
   for errors in [False,True]:
    for backend in ['dioptrin','pyfai']:
     for v in (['baseline','candidate'] if t==1 else ['candidate','baseline']):cases.append((v,ds,1,t,'interactive',errors,backend))
  # Application-only control: retain the installed published extension.
  for ds in datasets:cases.append(('app_only',ds,8,t,'batch',False,'dioptrin'))
  for ds in datasets:cases.append(('published',ds,8,t,'batch',False,'dioptrin'))
 done=0
 for args in cases:
  v,ds,c,t,mode,errors,backend=args;tag=filename(*args)
  path=B/'results'/tag
  if path.exists() and json.loads(path.read_text())['qualified']:done+=1;continue
  attempt=0
  while True:
   waited=0
   while idle()>3.6:
    if waited%6==0:print('WAIT lower background load',done,'/',len(cases),flush=True)
    time.sleep(5);waited+=1
   attempt+=1;out=run(*args,attempt=attempt)
   if out['qualified']:break
   if attempt>=6:raise RuntimeError('Repeated background-load rejections; preserve evidence before retrying')
  done+=1;print('PROGRESS',done,'/',len(cases),flush=True)
 (B/'finished.json').write_text(json.dumps(dict(complete=True,points=done,total=len(cases)),indent=2)+'\n')
if __name__=='__main__':main()
