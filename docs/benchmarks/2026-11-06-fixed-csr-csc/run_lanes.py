"""Check 1/2/4 CSC histogram lanes for fixed-mask Poisson integration."""
from pathlib import Path
import json
import subprocess
from statistics import median
folder=Path(__file__).resolve().parent
for name,root,table in [
 ('full',Path('/tmp/dioptrin-bridge-investigation-20261105'),Path('/tmp/dioptrin-bslz4-lut-20261105/pixel_split')),
 ('crop1024',Path('/tmp/dioptrin-fixed-crop-1024'),Path('/tmp/dioptrin-fixed-crop-1024/lut')),
 ('crop256',Path('/tmp/dioptrin-fixed-crop-256'),Path('/tmp/dioptrin-fixed-crop-256/lut'))]:
 for workers in (4,8):
  raw=folder/f'lanes-{name}-w{workers}.jsonl'
  with raw.open('w') as out:
   subprocess.run([str(folder/'target/release/fixed-csr-csc'),str(workers),str(root),str(table),'0.35','lanes'],stdout=out,check=True)
  events=[json.loads(line) for line in raw.read_text().splitlines()]
  print(name,workers,{key:round(median(e['fps'] for e in events if e['event']=='timing' and e['kernel']==key),2) for key in sorted({e['kernel'] for e in events if e['event']=='timing'})},flush=True)
