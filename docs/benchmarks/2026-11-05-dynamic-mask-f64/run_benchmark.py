"""Run native probes sequentially; retain every sample and approximate competing CPU load."""
import json, pathlib, subprocess, time
import psutil
ROOT = pathlib.Path(__file__).resolve().parent
BINARY = '/Users/clemens/Programming/Dioptrin/target/release/dioptrin-dynamic-mask-f64-probe'
DATA = '/tmp/dioptrin-bridge-investigation-20261105'
LUT = '/tmp/dioptrin-bslz4-lut-20261105/pixel_split'

def counters(exclude):
    out = {}
    for p in psutil.process_iter(['pid', 'create_time', 'cpu_times']):
        try:
            if p.pid in exclude or p.info['cpu_times'] is None: continue
            c = p.info['cpu_times']
            out[(p.pid, p.info['create_time'])] = c.user + c.system
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return out

def run(name, threads, threshold=None):
    args = [BINARY, str(threads), DATA, LUT]
    if threshold is not None: args.append(str(threshold))
    print('RUN', name, flush=True)
    events = []
    with (ROOT / (name + '.stderr')).open('w') as err, (ROOT / (name + '.jsonl')).open('w') as raw:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err, text=True)
        begin = None
        for line in p.stdout:
            raw.write(line); raw.flush()
            event = json.loads(line)
            if event['event'] == 'begin':
                begin = (time.monotonic(), counters({p.pid, psutil.Process().pid}))
            elif event['event'] == 'measurement':
                if begin:
                    now = time.monotonic(); end = counters({p.pid, psutil.Process().pid})
                    event['observed_other_busy_cores'] = sum(max(0., cpu - begin[1].get(pid, cpu)) for pid, cpu in end.items()) / (now - begin[0])
                print(name, event['threshold'], 'errors='+str(event['errors']), event['variant'], event['repeat'], round(event['fps'],1), 'fps; other busy cores', round(event.get('observed_other_busy_cores',0.),1), flush=True)
            elif event['event'] in ('mask_stats','strict_validation','baseline_validation'):
                print(name, event['event'], event.get('threshold'), event.get('errors',''), flush=True)
            events.append(event)
        code = p.wait()
    (ROOT / (name + '.json')).write_text(json.dumps({'command':args, 'exit_code':code, 'events':events}, indent=2))
    if code: raise RuntimeError(f'{name} exited {code}: {(ROOT/(name+".stderr")).read_text()}')

if __name__ == '__main__':
    run('w8', 8)
    run('w1', 1, 230)
    run('w4', 4, 230)
