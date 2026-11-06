"""Summarize native fixed-mask layout probes without discarding samples."""
from pathlib import Path
import json
from statistics import median
root=Path(__file__).resolve().parent
records=[]
for name in ['w1','w4','w8']+[f'crop{s}-w{w}' for s in (1024,256) for w in (1,4,8)]:
 d=json.loads((root/(name+'.json')).read_text())
 size='3262×3108' if name.startswith('w') else str(d['size'])+'×'+str(d['size'])
 for frozen in (False,True):
  for errors in (False,True):
   values={k:median(e['fps'] for e in d['events'] if e['event']=='timing' and
       e['frozen_threshold']==frozen and e['errors']==errors and e['kernel']==k)
       for k in ('production_fixed_csr','experimental_fixed_csc_cached_norm')}
   records.append({'image':size,'bins':512 if size.startswith('256') else 2000,
                   'workers':d['workers'],'frozen_threshold':frozen,'errors':errors,**values})
lanes=[]
for name in ('full','crop1024','crop256'):
 for w in (4,8):
  events=[json.loads(line) for line in (root/f'lanes-{name}-w{w}.jsonl').read_text().splitlines()]
  values={k:median(e['fps'] for e in events if e['event']=='timing' and e['kernel']==k)
          for k in {e['kernel'] for e in events if e['event']=='timing'}}
  lanes.append({'image':name,'workers':w,**values})
(root/'summary.json').write_text(json.dumps({'primary':records,'poisson_lane_sensitivity':lanes},indent=2)+'\n')
rows=[]
for r in records:
 if not r['frozen_threshold'] and r['workers'] in (4,8):
  a,b=r['production_fixed_csr'],r['experimental_fixed_csc_cached_norm']
  rows.append(f"| {r['image']} | {r['bins']} | {r['workers']} | {'yes' if r['errors'] else 'no'} | {a:.0f} | {b:.0f} |")
text='''# Fixed-mask CSR versus CSC: Eiger and smaller crops

These experiments test whether CSR has a demonstrated advantage for fixed masks, rather than assuming it does. CSC is not mathematically incapable of fixed-mask integration. An experimental fixed-mask CSC kernel caches normalization, compacts masked pixels, and uses the same original float64 coefficients as the current production CSR implementation. No float32 coefficient conversion is used.

Primary measurements, median fps over five alternating blocks, original static mask:

| Image | Bins | Frame workers | Poisson errors | Production CSR fps | Fixed CSC candidate fps |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

For large-image intensity-only integration the CSC candidate wins: 265 versus 235 fps at four workers, 448 versus 420 at eight. At 1024 square intensity-only results are close at four workers. Smaller-image and Poisson results show actual CSR advantages, especially at 256 square. These very high small-image rates have little practical impact on the application's end-to-end experience, so optimizing large-image throughput remains the stated product priority.

A follow-up Poisson probe checks one, two and four independent CSC histograms. At 1024 square, the best tested CSC variant (two histograms) reaches 1260/2226 fps at four/eight workers versus CSR's 1328/2467. At 256 square, the best CSC candidate reaches 23518/38119 fps versus CSR's 55082/82044. Thus four histograms alone do not explain away all smaller-image CSR wins. These follow-up runs also illustrate variability: for the full image at eight workers, the four-histogram CSC candidate reached 410 versus CSR's 351 fps, whereas the primary pass showed CSR ahead for Poisson calculations. The results do not establish a universal layout winner or prove that further CSC optimization cannot close a gap.

The practical conclusion is to favor CSC development for large-image fast integration, including cached normalization for fixed masks, rather than retain CSR solely to maximize tiny-image fps. The benchmark does not replace the production fixed-mask kernel: the current library still uses CSR there. Strict accumulation and azimuthal-spread routines also currently use CSR; that is an implementation/compatibility fact, not a proof of mathematical necessity.

Protocol: native-only Rust, no Python bridge, reading, mask computation or GUI work inside the timed blocks. Sixteen real Eiger frames sampled across indices 0–624 are reused from the earlier verified float64 input export. Three representative frames (0, middle, last) were rechecked directly against the original HDF5 dataset. The original static mask is 8.7548%; a second case freezes threshold >230 from the first frame and combines it with that mask. Raw records include both mask cases. All variants are compared on identical weights and masks. The fixed CSC candidate omits masked pixels and uses EXACTLY production CSR's cached bin denominators. Transposition is checked coefficient-by-coefficient for bit identity. Both kernels generate radial coordinates and return identical requested fields; Poisson arithmetic uses the same variance/normalization/square-root order. Different histogram summation order causes only expected float64 rounding differences, checked at rtol 5e-12 and atol 1e-9. All output frames also match independent current-library API references generated outside the timers.

The cropped inputs are actual 1024×1024 and 256×256 regions around the original PONI position, using origins (1004, 961) and (1388, 1345). PONI1/PONI2 are translated by the crop origin times the pixel size, preserving physical geometry and rotations. Pixel-split tables and correction arrays are generated by the current Rust geometry implementation and checked against fresh Python API integrations. The 1024 crop retains 2000 bins; the 256 crop uses 512 bins, with identical settings for both layouts within each comparison. These are crops of this detector/calibration, not an exhaustive survey of small detectors or image/bin combinations.

Frame parallelism uses one Rayon pool of 1, 4 or 8 workers, with the same 16 frames per batch. Five alternating blocks per mask/error/layout combination run for at least 0.8 seconds on full-size data and 0.35 seconds on crops. One/two/four histogram sensitivity uses the same five alternating blocks and static-mask Poisson workload. Native CSR and CSC source modules are imported directly from `src/lut`; only the private Pixel trait's f64 conversion is shimmed identically. The candidate CSC implementation lives in `lut/fixed_csc.rs`. Snapshot files preserve the primary four-histogram probe source before adding the sensitivity modes. Background CPU activity is recorded for the primary full/crop runs, so these are observations on the current machine rather than isolated hardware maxima.

Build: `CARGO_BUILD_JOBS=2 cargo build --release --manifest-path docs/benchmarks/2026-11-06-fixed-csr-csc/Cargo.toml`. Full-size runs use `run.py`; `prepare_crops.py` adjusts calibration and generates crop tables/references; `run_crops.py` measures them; `run_lanes.py` performs histogram sensitivity. `summarize.py` creates this summary from all raw records. The original 1.3 GB frame export, masks, calibration and table files are external inputs under the paths recorded in the raw JSON and preparation scripts; large detector artifacts and build outputs are excluded from Git. All preparation/validation is outside timing. Use the companion Qt Dioptas environment for the Python preparation scripts (NumPy, psutil, local Dioptrin API and detector-file dependencies).

Raw full-size files: `w1.json`, `w4.json`, `w8.json` and JSONL samples. Crops: `crop1024-w*.json`, `crop256-w*.json`. Histogram sensitivity: `lanes-*.jsonl`. All medians: [summary.json](summary.json). Source/binary/calibration hashes are included in the raw records. No production algorithm selection was changed based on this probe.
'''
(root/'README.md').write_text(text)
