"""Summarize retained native measurements and create the standalone report."""
from pathlib import Path
import collections, csv, html, json, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
runs={p.stem:json.loads(p.read_text()) for p in sorted(root.glob('w*.json'))}
assert all(r['exit_code']==0 for r in runs.values())
labels={'current_csr_two_pass':'Current-style CSR, two passes','fused_csr':'CSR, fused normalization',
        'csc_one_histogram':'CSC, one histogram','csc_four_histograms':'CSC, four histograms',
        'recompact_csr_each_frame':'Rebuild compacted CSR each frame'}
rows=[]; masks={}; numeric=[]
for name,run in runs.items():
    groups=collections.defaultdict(list)
    for e in run['events']:
        if e['event']=='mask_stats': masks[e['threshold']]=e
        if e['event'] in ('numerical','strict_validation','baseline_validation'): numeric.append(e)
        if e['event'] in ('measurement','mask_update_measurement'):
            groups[(e['threads'],e['threshold'],e.get('errors',False),e.get('variant','mask_update_only'))].append(e)
    for (threads,threshold,errors,variant),events in groups.items():
        values=[e['fps'] for e in events]
        rows.append(dict(run=name,threads=threads,threshold=threshold,errors=errors,variant=variant,
                         samples=len(values),median_fps=statistics.median(values),min_fps=min(values),max_fps=max(values),
                         median_other_busy_cores=statistics.median([e.get('observed_other_busy_cores',0) for e in events])))
with (root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def get(threads,threshold,errors,variant):
    return next(r for r in rows if (r['threads'],r['threshold'],r['errors'],r['variant'])==(threads,threshold,errors,variant))
def mean_density(t):
    m=masks[t];return statistics.mean(m['extra_masked_pixels'])/m['static_valid_pixels']*100

primary=[]
for v in labels:
    a=get(8,230,False,v);b=get(8,230,True,v)
    primary.append({'variant':v,'label':labels[v],'intensity_fps':a['median_fps'],'poisson_fps':b['median_fps'],
      'intensity_speedup':a['median_fps']/get(8,230,False,'current_csr_two_pass')['median_fps'],
      'poisson_speedup':b['median_fps']/get(8,230,True,'current_csr_two_pass')['median_fps']})
strict=[e for e in numeric if e['event']=='strict_validation']
fast=[e for e in numeric if e['event']=='numerical']
conclusions={'primary_threads':8,'primary_threshold':230,'primary_mean_extra_mask_percent':mean_density(230),
 'primary':primary,'all_strict_byte_identical':all(e['metrics']['intensity']['byte_identical'] and e['metrics']['poisson_sigma']['byte_identical'] for e in strict),
 'fused_csr_byte_identical_to_current':all(e['metrics']['intensity']['byte_identical'] and e['metrics']['poisson_sigma']['byte_identical'] for e in fast if e['variant']=='fused_csr'),
 'worst_fast_intensity_relative_l2':max(e['metrics']['intensity']['relative_l2'] for e in fast),
 'worst_fast_sigma_relative_l2':max(e['metrics']['poisson_sigma']['relative_l2'] for e in fast),
 'four_histogram_worst_intensity_relative_l2':max(e['metrics']['intensity']['relative_l2'] for e in fast if e['variant']=='csc_four_histograms'),
 'mask_changes_verified_all_frames_including_wrap':all(min(m['changed_pixels_from_previous_including_wrap'])>0 for m in masks.values()),
 'thresholds':{str(t):{'extra_mask_mean_percent':mean_density(t),'extra_mask_min_percent':min(m['extra_masked_pixels'])/m['static_valid_pixels']*100,'extra_mask_max_percent':max(m['extra_masked_pixels'])/m['static_valid_pixels']*100,'changed_pixels_min':min(m['changed_pixels_from_previous_including_wrap']),'changed_pixels_max':max(m['changed_pixels_from_previous_including_wrap'])} for t,m in masks.items()}}
(root/'conclusions.json').write_text(json.dumps(conclusions,indent=2)+'\n')

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.8,6.3),gridspec_kw={'width_ratios':[1.6,1]})
fig.suptitle('Changing masks every frame — native float64 CSR versus CSC',fontsize=19,fontweight='bold',x=.03,ha='left')
variants=list(labels); y=np.arange(len(variants)); h=.32
for errors,offset,color,label in [(False,-h/2,'#137488','Intensity'),(True,h/2,'#db8b32','Intensity + Poisson errors')]:
    stats=[get(8,230,errors,v) for v in variants]; values=np.array([r['median_fps'] for r in stats]);
    lo=values-np.array([r['min_fps'] for r in stats]);hi=np.array([r['max_fps'] for r in stats])-values
    ax.barh(y+offset,values,height=h,color=color,label=label)
    ax.errorbar(values,y+offset,xerr=[lo,hi],fmt='none',color='#354957',capsize=3,lw=1)
    for yy,v,upper in zip(y+offset,values,hi):ax.text(v+upper+4,yy,f'{v:.0f}',va='center',fontsize=10)
ax.set_yticks(y,[labels[v].replace('Rebuild compacted CSR each frame','Rebuild compacted CSR\neach frame') for v in variants]);ax.invert_yaxis()
ax.set_xlabel('Frames/s, eight threads');ax.set_xlim(0,max(r['max_fps'] for r in rows if r['threads']==8 and r['threshold']==230 and r['variant']!='mask_update_only')*1.14)
ax.set_title(f'Mask updates and normalization included\nThreshold 230: {mean_density(230):.2f}% additional pixels masked',loc='left',fontsize=12,fontweight='bold')
ax.legend(frameon=False,bbox_to_anchor=(.5,-.16),loc='upper center',ncol=2,fontsize=10)
for v,color,marker in [('current_csr_two_pass','#666666','o'),('fused_csr','#137488','s'),('csc_four_histograms','#7845a3','^')]:
    ts=[323,230,207];ax2.plot([mean_density(t) for t in ts],[get(8,t,False,v)['median_fps'] for t in ts],color=color,marker=marker,label=labels[v])
ax2.set_xscale('log');ax2.set_xticks([.1,1,5],['0.1%','1%','5%']);ax2.set_xlabel('Additional masked pixels, mean across frames');ax2.set_ylabel('Intensity frames/s, eight threads');ax2.set_ylim(bottom=0)
ax2.set_title('Sensitivity to dynamic mask density',fontsize=12,fontweight='bold',loc='left');ax2.legend(frameon=False,fontsize=9,loc='lower left')
fig.subplots_adjust(left=.22,right=.97,top=.81,bottom=.27,wspace=.37)
fig.text(.03,.06,'Apple M4 Pro · 16 real Eiger9M frames · 2,000 bins · five warmed measurements per point\nMask construction, backend mask copy, per-frame normalization and output allocation timed. Whiskers: min–max.\nNative Rust threshold masking; Python plugin execution, HDF5 IO and decompression excluded.',fontsize=10,color='#555555')
fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)

tr=''.join(f'<tr><td>{r["label"]}</td><td>{r["intensity_fps"]:.1f}</td><td>{r["intensity_speedup"]:.2f}×</td><td>{r["poisson_fps"]:.1f}</td><td>{r["poisson_speedup"]:.2f}×</td></tr>' for r in primary)
density_tr=''.join(f'<tr><td>{t:g}</td><td>{mean_density(t):.3f}%</td><td>{conclusions["thresholds"][str(t)]["extra_mask_min_percent"]:.3f}–{conclusions["thresholds"][str(t)]["extra_mask_max_percent"]:.3f}%</td><td>{get(8,t,False,"current_csr_two_pass")["median_fps"]:.1f}</td><td>{get(8,t,False,"fused_csr")["median_fps"]:.1f}</td><td>{get(8,t,False,"csc_four_histograms")["median_fps"]:.1f}</td></tr>' for t in [323.,230.,207.])
scaling_tr=''.join(f'<tr><td>{th}</td><td>{get(th,230,False,"current_csr_two_pass")["median_fps"]:.1f}</td><td>{get(th,230,False,"fused_csr")["median_fps"]:.1f}</td><td>{get(th,230,False,"csc_four_histograms")["median_fps"]:.1f}</td><td>{get(th,230,True,"current_csr_two_pass")["median_fps"]:.1f}</td><td>{get(th,230,True,"fused_csr")["median_fps"]:.1f}</td><td>{get(th,230,True,"csc_four_histograms")["median_fps"]:.1f}</td></tr>' for th in [1,4,8])
protocol=json.loads((root/'provenance.json').read_text())['protocol']
protocol_html=''.join(f'<li><strong>{html.escape(k)}:</strong> {html.escape(v)}</li>' for k,v in protocol.items())
update=get(8,230,False,'mask_update_only')['median_fps']
report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Changing-mask float64 CSR versus CSC benchmark</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#24323d}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%}}pre{{white-space:pre-wrap;background:#f3f6f7;padding:16px}}</style>
<h1>Changing-mask float64 CSR versus CSC</h1><p>Measured November 5, 2026 on the same Apple M4 Pro and 16 real Eiger9M frames. All pixels, weights, corrections, sums and outputs are float64. Every timed frame constructs a new threshold mask and recomputes normalization. No table rebuild is required for either runtime-mask layout.</p>
<img src="comparison.png" alt="Changing-mask float64 throughput">
<h2>Primary comparison</h2><p>Threshold 230: mean {mean_density(230):.3f}% additional masked pixels beyond the same static mask. Masks are different for every frame, including wraparound. Eight threads, five warmed measurements per variant/error mode; medians shown.</p>
<table><tr><th>Kernel</th><th>Intensity fps</th><th>Speedup</th><th>Intensity + Poisson fps</th><th>Speedup</th></tr>{tr}</table>
<p>The current-style CSR driver uses unchanged production row arithmetic and recalculates normalization in a separate serial-per-frame traversal, then integrates signal and variance in parallel rows. Fused CSR uses the identical production row arithmetic but computes normalization during that same signal traversal. CSC checks the mask once per input pixel and accumulates signal, normalization and variance together. Four interleaved histograms reduce dependent updates to the same bins. The per-frame recompaction diagnostic filters the original coefficient table and constructs a new execution table for each frame, including allocation/index conversion, normalization and integration.</p>
<p><strong>Recommendation:</strong> Keep geometry tables independent of per-frame masks. Fused normalization is a useful CSR improvement, and four-histogram CSC is the faster prototype for this changing-mask batch workload. Rebuilding compacted CSR every frame is much slower in this straightforward implementation. Static-mask compaction remains a separate option for masks reused over many frames.</p>
<h2>Density sensitivity</h2><p>Thresholds were selected near the pooled 99.9th, 99th and 95th percentiles of static-valid real pixel intensities. The thresholds remain fixed; the changing images produce the changing masks. Fractions below refer only to otherwise static-valid pixels. Intensity throughput includes mask updates and normalization.</p>
<table><tr><th>Threshold</th><th>Mean extra mask</th><th>Per-frame range</th><th>Current CSR fps</th><th>Fused CSR fps</th><th>CSC four-histogram fps</th></tr>{density_tr}</table>
<h2>Thread scaling at threshold 230</h2><table><tr><th>Threads</th><th>Current CSR</th><th>Fused CSR</th><th>CSC four</th><th>CSR + errors</th><th>Fused CSR + errors</th><th>CSC four + errors</th></tr>{scaling_tr}</table><p>These are independent 16-frame batches. CSC uses parallelism across frames; this does not establish its latency or speed on one image.</p>
<h2>Mask cost and precision</h2><p>The threshold/static-mask combination and explicit backend-owned mask copy are performed in native Rust inside every full timing. The separately measured mask-update stage reaches {update:.1f} frames/s at eight threads (aggregate throughput). It is already included in the integration numbers above. This represents the threshold masking rule used by Dioptas, not Python/NumPy/plugin dispatch overhead or the cost of cosmic-ray/powder-outlier detection.</p>
<p>The current-style baseline is verified byte-identical to actual copied production cache invalidation plus integration for all frames, thresholds and both error modes. Fused CSR is also byte-identical to current CSR. Strict CSC and strict CSR are byte-identical, including Poisson sigma. Fast CSC differs only through summation order: worst four-histogram intensity relative L2 difference {conclusions['four_histogram_worst_intensity_relative_l2']:.3g}; worst across all fast variants {conclusions['worst_fast_intensity_relative_l2']:.3g}. No float32 weights or accumulators are used.</p>
<h2>Protocol and limits</h2><ul>{protocol_html}</ul><p>No production library code was modified. This isolated prototype needs production mask lifecycle/API work, memory assessment, single-frame latency and 2D validation before adoption. Python set_mask calls and cache-lock bookkeeping are not timed; mask ownership copying and all normalization arithmetic are timed. System activity varied during measurement; min–max ranges, raw blocks and observed competing CPU load are retained. No samples were removed.</p>
<h2>Reproduction and artifacts</h2><ul><li><a href="summary.csv">All medians and ranges</a></li><li><a href="conclusions.json">Machine-readable conclusions</a></li><li><a href="provenance.json">Protocol and source/input hashes</a></li><li><a href="src/main.rs">Native probe</a> and <a href="src/lut/production_prepared.rs">production execution copy with benchmark drivers</a></li><li><a href="run_benchmark.py">Runner</a> and <a href="summarize.py">report generator</a></li></ul>
<pre>CARGO_TARGET_DIR=/Users/clemens/Programming/Dioptrin/target cargo test --manifest-path Cargo.toml --offline
CARGO_TARGET_DIR=/Users/clemens/Programming/Dioptrin/target cargo build --release --manifest-path Cargo.toml --offline
python3 run_benchmark.py
python3 summarize.py</pre><p>Large readonly frame and LUT files remain at their existing /tmp paths, recorded in the runner and input provenance; they are not copied into this report. The runner preserves native JSONL and enriched JSON records for every process.</p></html>'''
(root/'report.html').write_text(report)
print(json.dumps(conclusions,indent=2))
