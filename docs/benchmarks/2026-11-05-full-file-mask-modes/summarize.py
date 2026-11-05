from pathlib import Path
import csv,hashlib,json,statistics,subprocess
ROOT=Path(__file__).resolve().parent
runs={name:json.loads((ROOT/(name+'.json')).read_text()) for name in ['w8-intensity','w8-poisson','w8-paired-intensity','w8-paired-poisson']}
rows=[]
for name,run in runs.items():
    assert run['input']['shape'][0]==625
    for workflow in run['summary']:
        items=[e for e in run['measurements'] if e['name']==workflow]
        rows.append(dict(run=name,workflow=workflow,errors=run['arguments']['errors'],samples=len(items),frames=625,
          median_fps=statistics.median(e['fps'] for e in items),min_fps=min(e['fps'] for e in items),max_fps=max(e['fps'] for e in items),
          median_seconds_per_file=statistics.median(e['seconds']/e['loops'] for e in items),
          median_process_cores=statistics.median(e['process_cores'] for e in items),median_other_busy_cores=statistics.median(e['other_busy_cores'] for e in items)))
with (ROOT/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
labels={'dioptrin_h5py_stream':'h5py stream → Dioptrin','dioptrin_parallel_decode':'Parallel decode → Dioptrin','dioptrin_reusable_decode':'Reusable-buffer decode → Dioptrin','bslz4_mmap_stream':'Jon: fused decode + CSC'}
primary=[]
for workflow,label in labels.items():
    prefix='w8-paired-' if workflow in ['dioptrin_reusable_decode','bslz4_mmap_stream'] else 'w8-'
    a=next(r for r in rows if r['run']==prefix+'intensity' and r['workflow']==workflow)
    b=next(r for r in rows if r['run']==prefix+'poisson' and r['workflow']==workflow)
    primary.append(dict(label=label,workflow=workflow,intensity_fps=a['median_fps'],poisson_fps=b['median_fps'],intensity_seconds_per_file=a['median_seconds_per_file'],poisson_seconds_per_file=b['median_seconds_per_file']))
conclusions={'frames':625,'file_bytes':runs['w8-intensity']['input']['bytes'],'mask':'same fixed detector mask','primary':primary,
 'read_decode_float64_only_fps':next(r['median_fps'] for r in rows if r['workflow']=='read_decode_convert_only'),
 'cache':'Repeated warm OS-cache passes, with all frame reads and decompression inside timings; cold physical-storage performance not measured.',
 'weights':'Dioptrin float64; unmodified Jon float32 weights, float64 accumulation/output; identical float64 normalization denominators.',
 'checks':{name:r['numerical'] for name,r in runs.items()}}
(ROOT/'conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')
protocol={'implementation_commit':'10435d6','thread_budget':'8 decoding/integration workers, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1',
 'settings':'2000 uniform two-theta degree bins, full pixel splitting, solid angle, polarization0.965, same detector mask and geometry weights; all625frames in file order.',
 'sampling':'Three alternating whole-file passes for h5py/parallel/reusable/fused comparison. Five additional alternating direct paired passes of reusable Dioptrin and fused Jon supply primary optimized numbers. All results and host CPU load retained.',
 'full_validation':'Parallel-decoded Dioptrin and Jon outputs checked against all625h5py-derived Dioptrin outputs. Direct paired625-frame Jon/reusable-Dioptrin results checked independently in each error mode. Sampled uncompressed versus compressed decoding also checked.',
 'setup_excluded':'HDF5 chunk-index discovery, calibration/LUT/plan initialization, mmap opening and initial buffer allocation. h5py stream opens its dataset inside each timed pass. Fixed calibration/mask reused for every frame.',
 'cache':'No cache purge. Every workflow is warmed with a full625-frame pass before timing. Includes file-backed accesses, decoding, orientation conversion where needed, integration, normalization and output allocation/materialization. Not cold-device I/O or first-file initialization.',
 'precision':'Jon coefficient storage is float32; Dioptrin is float64. Poisson Jon adapter stacks signal and squared-weight rows; Dioptrin accumulates variance in its integration traversal. Both use identical exported float64 normalization.',
 'scope':'Dioptas remains unchanged. These are measured input-pipeline prototypes, not a new native HDF5 API shipped by Dioptrin.'}
(ROOT/'provenance.json').write_text(json.dumps({'protocol':protocol,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'builds':{name:{'versions':r['versions'],'bslz4_build':r['bslz4_build'],'binaries':r['binaries'],'file_reading_protocol':r['file_reading_protocol']} for name,r in runs.items()},'input':runs['w8-intensity']['input']},indent=2)+'\n')
trs=''.join(f'<tr><td>{r["label"]}</td><td>{r["intensity_fps"]:.1f}</td><td>{r["intensity_seconds_per_file"]:.2f}s</td><td>{r["poisson_fps"]:.1f}</td><td>{r["poisson_seconds_per_file"]:.2f}s</td></tr>' for r in primary)
(ROOT/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-file Eiger integration comparison</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}</style><h1>Full625-frame Eiger HDF5 comparison</h1><p>Measured November5,2026, Apple M4 Pro. Each pass reads/decompresses the complete5.69GB file, then integrates with the same fixed mask and settings. Repeated passes use the OS file cache; cold physical-device bandwidth is not measured.</p><table><tr><th>Path</th><th>Intensity frames/s</th><th>Whole file</th><th>With Poisson frames/s</th><th>Whole file</th></tr>{trs}</table><p>The current h5py producer limits the stream to about23frames/s, while the parallel paths use much more of the machine. The integration-only gain does not translate directly into full-file throughput: read/decode/float64 conversion alone reaches {conclusions['read_decode_float64_only_fps']:.1f}frames/s, and those stages precede integration in the staged Dioptrin path. Jon's fused path avoids materializing a whole float64 image. Dioptrin remains float64-weighted; Jon uses float32 coefficients. With errors, this Jon adapter reads a doubled signal/variance matrix.</p><p>Primary optimized numbers come from five alternating direct paired passes. Standard h5py and parallel allocation numbers come from three whole-file passes. The CSV retains every measured range and observed process/other-process CPU load; different runs experience ordinary host-load variability.</p><h2>Protocol</h2><ul>{''.join(f'<li><b>{k}:</b> {v}</li>' for k,v in protocol.items())}</ul><p><a href="summary.csv">All results/ranges</a> · <a href="conclusions.json">Numerical checks</a> · <a href="provenance.json">Input/build provenance</a> · <a href="bench_full_file.py">Benchmark source</a></p><p>Reproduce run.py followed by run_paired.py, then summarize.py. Large HDF5 inputs, mask and exported LUT stay outside Git at their recorded paths. No Dioptas code was modified.</p></html>''')
print(json.dumps({'primary':primary,'decode_only_fps':conclusions['read_decode_float64_only_fps']},indent=2))
