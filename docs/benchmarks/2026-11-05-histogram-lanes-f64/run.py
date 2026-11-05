from pathlib import Path
import json, subprocess, time
import psutil
ROOT=Path(__file__).resolve().parent
BINARY='/Users/clemens/Programming/Dioptrin/target/release/dioptrin-histogram-lanes-f64-probe'
def counter(skip):
    result={}
    for p in psutil.process_iter(['pid','create_time','cpu_times']):
        try:
            c=p.info['cpu_times']
            if p.pid in skip or c is None:continue
            result[(p.pid,p.info['create_time'])]=c.user+c.system
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result
for threads in [8,1]:
    events=[];start=None
    with (ROOT/f'w{threads}.jsonl').open('w') as raw,(ROOT/f'w{threads}.stderr').open('w') as err:
        p=subprocess.Popen([BINARY,str(threads)],stdout=subprocess.PIPE,stderr=err,text=True)
        for line in p.stdout:
            raw.write(line);raw.flush();e=json.loads(line)
            if e['event']=='begin':start=(time.monotonic(),counter({p.pid,psutil.Process().pid}))
            if e['event']=='measurement':
                now=time.monotonic();end=counter({p.pid,psutil.Process().pid})
                e['other_busy_cores']=sum(max(0.,v-start[1].get(k,v)) for k,v in end.items())/(now-start[0])
                print(threads,'threads','errors',e['errors'],'lanes',e['lanes'],'repeat',e['repeat'],round(e['fps'],1),'fps',flush=True)
            events.append(e)
        code=p.wait()
    (ROOT/f'w{threads}.json').write_text(json.dumps({'threads':threads,'exit_code':code,'events':events},indent=2)+'\n')
    if code:raise RuntimeError((code,(ROOT/f'w{threads}.stderr').read_text()))
