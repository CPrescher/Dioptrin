//! Precision-controlled, native-only CSR/CSC experiment. No production library edits.
use memmap2::MmapOptions;
use rayon::prelude::*;
use serde_json::json;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// These shims let the unchanged production execution file compile in the probe.
mod frame {
    pub(crate) trait Pixel: Copy + Send + Sync {
        fn to_f64(self) -> f64;
    }
    impl Pixel for f64 {
        #[inline(always)]
        fn to_f64(self) -> f64 {
            self
        }
    }
}
mod math {
    pub use dioptrin::math::histogram;
}
#[allow(dead_code)]
mod lut {
    pub use dioptrin::lut::{IntermediateResult1d, IntermediateResult2d, Lut1d, Lut2d};
    pub mod production_prepared;
}
use lut::production_prepared::Prepared1d;

struct Csr {
    ptr: Vec<usize>,
    pixel: Vec<u32>,
    weights: Vec<f64>,
}
struct Csc {
    ptr: Vec<u32>,
    bin: Vec<u32>,
    weights: Vec<f64>,
}
type Output = (Vec<f64>, Vec<f64>, Vec<f64>);

fn read_f64(path: &Path) -> Vec<f64> {
    fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn read_u32(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn fold_mask(csr: &Csr, mask: &[u8]) -> Csr {
    let mut ptr = Vec::with_capacity(csr.ptr.len());
    let mut pixel = Vec::new();
    let mut weights = Vec::new();
    ptr.push(0);
    for b in 0..csr.ptr.len() - 1 {
        for k in csr.ptr[b]..csr.ptr[b + 1] {
            if mask[csr.pixel[k] as usize] == 0 {
                pixel.push(csr.pixel[k]);
                weights.push(csr.weights[k]);
            }
        }
        ptr.push(pixel.len());
    }
    Csr {
        ptr,
        pixel,
        weights,
    }
}
fn transpose(csr: &Csr, npix: usize) -> Csc {
    assert_eq!(csr.pixel.len(), csr.weights.len());
    assert_eq!(*csr.ptr.last().unwrap(), csr.weights.len());
    let mut ptr = vec![0u32; npix + 1];
    for &p in &csr.pixel {
        ptr[p as usize + 1] += 1;
    }
    for p in 0..npix {
        ptr[p + 1] += ptr[p];
    }
    let mut next = ptr[..npix].to_vec();
    let mut bin = vec![0u32; csr.pixel.len()];
    let mut weights = vec![0.; csr.weights.len()];
    for b in 0..csr.ptr.len() - 1 {
        for k in csr.ptr[b]..csr.ptr[b + 1] {
            let p = csr.pixel[k] as usize;
            let at = next[p] as usize;
            next[p] += 1;
            bin[at] = b.try_into().unwrap();
            weights[at] = csr.weights[k];
        }
    }
    // Verify that conversion preserved every masked CSR coefficient bit-for-bit.
    let mut back = csr.ptr[..csr.ptr.len() - 1].to_vec();
    for p in 0..npix {
        for k in ptr[p] as usize..ptr[p + 1] as usize {
            let b = bin[k] as usize;
            let at = back[b];
            back[b] += 1;
            assert_eq!(csr.pixel[at] as usize, p);
            assert_eq!(csr.weights[at].to_bits(), weights[k].to_bits());
        }
    }
    Csc { ptr, bin, weights }
}
fn prepared(csr: Csr, min: f64, step: f64) -> Prepared1d {
    Prepared1d::new(lut::Lut1d {
        row_ptr: csr.ptr,
        col_idx: csr.pixel.into_iter().map(|v| v as usize).collect(),
        values: Some(csr.weights),
        hist_min: min,
        hist_step: step,
    })
}

fn check(got: &[Output], expected: &[Output]) -> serde_json::Value {
    let field = |index: usize| {
        let mut diff = 0.;
        let mut base = 0.;
        let mut max_abs: f64 = 0.;
        let mut max_relative: f64 = 0.;
        let mut exact = true;
        for (g, e) in got.iter().zip(expected) {
            let (g, e) = if index == 0 {
                (&g.1, &e.1)
            } else {
                (&g.2, &e.2)
            };
            assert_eq!(g.len(), e.len());
            for (&x, &y) in g.iter().zip(e) {
                assert!(x.is_finite() && y.is_finite());
                let d = (x - y).abs();
                diff += d * d;
                base += y * y;
                max_abs = max_abs.max(d);
                if y != 0. {
                    max_relative = max_relative.max(d / y.abs());
                }
                exact &= x.to_bits() == y.to_bits();
                assert!(
                    d <= 1e-10 + 2e-12 * y.abs(),
                    "precision check failed: {x} vs {y}"
                );
            }
        }
        json!({"relative_l2":if base==0. {0.} else {(diff/base).sqrt()},"max_absolute":max_abs,"max_relative":max_relative,"byte_identical":exact})
    };
    json!({"intensity":field(0),"poisson_sigma":field(1)})
}

// Native equivalent of upper-threshold plugin + OR static mask, followed by
// an explicit backend-owned copy. Both allocations and the copy are timed.
fn update_mask(signal: &[f64], base: &[u8], threshold: f64) -> Vec<u8> {
    let combined: Vec<u8> = signal
        .iter()
        .zip(base)
        .map(|(&x, &m)| u8::from(m != 0 || x > threshold))
        .collect();
    let backend_mask = combined.clone();
    black_box(&combined);
    backend_mask
}

fn csc_dynamic<const FAST: bool, const ERR: bool, const LANES: usize>(
    signal: &[f64],
    mask: &[u8],
    table: &Csc,
    pixel_norm: &[f64],
    bins: usize,
    min: f64,
    step: f64,
) -> Output {
    assert!(LANES == 1 || LANES == 4);
    let mut out = vec![0.; bins * LANES];
    let mut norm = vec![0.; bins * LANES];
    let mut var = vec![0.; if ERR { bins * LANES } else { 0 }];
    // All indices/dimensions validated by transpose and setup; private histograms.
    unsafe {
        let dest = out.as_mut_ptr();
        let denom = norm.as_mut_ptr();
        let variance = var.as_mut_ptr();
        for p in 0..signal.len() {
            if *mask.get_unchecked(p) != 0 {
                continue;
            }
            let x = *signal.get_unchecked(p);
            let n = *pixel_norm.get_unchecked(p);
            let lane = (p % LANES) * bins;
            let start = *table.ptr.get_unchecked(p) as usize;
            let end = *table.ptr.get_unchecked(p + 1) as usize;
            for k in start..end {
                let b = *table.bin.get_unchecked(k) as usize + lane;
                let w = *table.weights.get_unchecked(k);
                *dest.add(b) = if FAST {
                    w.mul_add(x, *dest.add(b))
                } else {
                    *dest.add(b) + w * x
                };
                *denom.add(b) = if FAST {
                    w.mul_add(n, *denom.add(b))
                } else {
                    *denom.add(b) + w * n
                };
                if ERR {
                    *variance.add(b) = if FAST {
                        (w * w).mul_add(x.max(0.), *variance.add(b))
                    } else {
                        *variance.add(b) + w * w * x.max(0.)
                    };
                }
            }
        }
    }
    if LANES == 4 {
        for b in 0..bins {
            out[b] = (out[b] + out[bins + b]) + (out[2 * bins + b] + out[3 * bins + b]);
            norm[b] = (norm[b] + norm[bins + b]) + (norm[2 * bins + b] + norm[3 * bins + b]);
            if ERR {
                var[b] = (var[b] + var[bins + b]) + (var[2 * bins + b] + var[3 * bins + b]);
            }
        }
        out.truncate(bins);
        norm.truncate(bins);
        if ERR {
            var.truncate(bins);
        }
    }
    for b in 0..bins {
        out[b] = if norm[b] == 0. { 0. } else { out[b] / norm[b] };
        if ERR {
            var[b] = (if norm[b] == 0. {
                0.
            } else {
                var[b] / (norm[b] * norm[b])
            })
            .max(0.)
            .sqrt();
        }
    }
    (math::histogram::bins(bins, min, step), out, var)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args[1].parse().unwrap();
    let root = Path::new(&args[2]);
    let lutroot = Path::new(&args[3]);
    let only_threshold: Option<f64> = args.get(4).map(|x| x.parse().unwrap());
    let reps: usize = args.get(5).map(|x| x.parse().unwrap()).unwrap_or(5);
    let block_seconds: f64 = args.get(6).map(|x| x.parse().unwrap()).unwrap_or(1.);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let npix =
        meta["shape"][1].as_u64().unwrap() as usize * meta["shape"][2].as_u64().unwrap() as usize;
    let nframes = meta["shape"][0].as_u64().unwrap() as usize;
    let file = File::open(root.join("frames.f64")).unwrap();
    let map = unsafe { MmapOptions::new().map(&file).unwrap() };
    assert!(cfg!(target_endian = "little"));
    assert_eq!(map.as_ptr().align_offset(std::mem::align_of::<f64>()), 0);
    assert_eq!(map.len(), npix * nframes * 8);
    let data = unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<f64>(), map.len() / 8) };
    let frames: Vec<&[f64]> = data.chunks_exact(npix).collect();
    let base = fs::read(root.join("mask.u8")).unwrap();
    assert_eq!(base.len(), npix);
    let original = Csr {
        ptr: fs::read(lutroot.join("indptr.u64"))
            .unwrap()
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect(),
        pixel: read_u32(&lutroot.join("indices.u32")),
        weights: read_f64(&lutroot.join("weights.f64")),
    };
    let started = Instant::now();
    let table = transpose(&original, npix);
    let transpose_seconds = started.elapsed().as_secs_f64();
    let pixels_norm = read_f64(&lutroot.join("normalization.f64"));
    assert_eq!(pixels_norm.len(), npix);
    let radial = read_f64(&lutroot.join("radial.f64"));
    let bins = radial.len();
    let step = radial[1] - radial[0];
    let min = radial[0] - step / 2.;
    let mut production = prepared(
        Csr {
            ptr: original.ptr.clone(),
            pixel: original.pixel.clone(),
            weights: original.weights.clone(),
        },
        min,
        step,
    );
    println!(
        "{}",
        json!({"event":"setup","threads":threads,"frames":nframes,"npix":npix,"bins":bins,
        "nnz":original.weights.len(),"transpose_seconds_including_verification":transpose_seconds,
        "precision":"f64 throughout","mask_update":"native threshold + static OR, allocation + explicit backend copy",
        "baseline":"unchanged production row, serial uncached denominator then parallel signal; cache lifecycle represented by recalculating each frame",
        "transpose":"coefficient-by-coefficient bit-preserving roundtrip checked"})
    );
    let names = [
        "current_csr_two_pass",
        "fused_csr",
        "csc_one_histogram",
        "csc_four_histograms",
        "recompact_csr_each_frame",
    ];
    for threshold in [323., 230., 207.] {
        if only_threshold.is_some_and(|x| x != threshold) {
            continue;
        }
        let masks: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| update_mask(f, &base, threshold))
            .collect();
        let base_count = base.iter().filter(|&&m| m == 0).count();
        let extra: Vec<usize> = masks
            .iter()
            .map(|m| {
                m.iter()
                    .zip(&base)
                    .filter(|(m, b)| **m != 0 && **b == 0)
                    .count()
            })
            .collect();
        let changes: Vec<usize> = (0..nframes)
            .map(|i| {
                masks[i]
                    .iter()
                    .zip(&masks[(i + nframes - 1) % nframes])
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .collect();
        assert!(changes.iter().all(|&x| x > 0));
        println!(
            "{}",
            json!({"event":"mask_stats","threshold":threshold,"static_valid_pixels":base_count,
            "extra_masked_pixels":extra,"changed_pixels_from_previous_including_wrap":changes})
        );
        // Correctness: helper baseline must match actual cache invalidation and
        // public copied execution bit for bit on every real frame/error mode.
        for errors in [false, true] {
            let mut actual = Vec::new();
            let mut helper = Vec::new();
            let mut strict = Vec::new();
            for (signal, mask) in frames.iter().zip(&masks) {
                production.invalidate_normalization();
                actual.push(if errors {
                    production.integrate::<true, true>(signal, Some(mask), &pixels_norm)
                } else {
                    production.integrate::<true, false>(signal, Some(mask), &pixels_norm)
                });
                helper.push(if errors {
                    production.benchmark_dynamic::<true, true, false>(
                        signal,
                        Some(mask),
                        &pixels_norm,
                    )
                } else {
                    production.benchmark_dynamic::<true, false, false>(
                        signal,
                        Some(mask),
                        &pixels_norm,
                    )
                });
                strict.push(if errors {
                    production.benchmark_dynamic::<false, true, true>(
                        signal,
                        Some(mask),
                        &pixels_norm,
                    )
                } else {
                    production.benchmark_dynamic::<false, false, true>(
                        signal,
                        Some(mask),
                        &pixels_norm,
                    )
                });
            }
            let metrics = check(&helper, &actual);
            assert!(metrics["intensity"]["byte_identical"].as_bool().unwrap());
            assert!(
                metrics["poisson_sigma"]["byte_identical"]
                    .as_bool()
                    .unwrap()
            );
            println!(
                "{}",
                json!({"event":"baseline_validation","threshold":threshold,"errors":errors,"metrics":metrics})
            );
            let strict_csc: Vec<Output> = frames
                .iter()
                .zip(&masks)
                .map(|(s, m)| {
                    if errors {
                        csc_dynamic::<false, true, 1>(s, m, &table, &pixels_norm, bins, min, step)
                    } else {
                        csc_dynamic::<false, false, 1>(s, m, &table, &pixels_norm, bins, min, step)
                    }
                })
                .collect();
            let metrics = check(&strict_csc, &strict);
            assert!(metrics["intensity"]["byte_identical"].as_bool().unwrap());
            assert!(
                metrics["poisson_sigma"]["byte_identical"]
                    .as_bool()
                    .unwrap()
            );
            println!(
                "{}",
                json!({"event":"strict_validation","threshold":threshold,"errors":errors,"metrics":metrics})
            );
            let run = |which: usize| -> Vec<Output> {
                frames
                    .par_iter()
                    .map(|signal| {
                        let mask = update_mask(signal, &base, threshold);
                        match (which, errors) {
                            (0, false) => production.benchmark_dynamic::<true, false, false>(
                                signal,
                                Some(&mask),
                                &pixels_norm,
                            ),
                            (0, true) => production.benchmark_dynamic::<true, true, false>(
                                signal,
                                Some(&mask),
                                &pixels_norm,
                            ),
                            (1, false) => production.benchmark_dynamic::<true, false, true>(
                                signal,
                                Some(&mask),
                                &pixels_norm,
                            ),
                            (1, true) => production.benchmark_dynamic::<true, true, true>(
                                signal,
                                Some(&mask),
                                &pixels_norm,
                            ),
                            (2, false) => csc_dynamic::<true, false, 1>(
                                signal,
                                &mask,
                                &table,
                                &pixels_norm,
                                bins,
                                min,
                                step,
                            ),
                            (2, true) => csc_dynamic::<true, true, 1>(
                                signal,
                                &mask,
                                &table,
                                &pixels_norm,
                                bins,
                                min,
                                step,
                            ),
                            (3, false) => csc_dynamic::<true, false, 4>(
                                signal,
                                &mask,
                                &table,
                                &pixels_norm,
                                bins,
                                min,
                                step,
                            ),
                            (3, true) => csc_dynamic::<true, true, 4>(
                                signal,
                                &mask,
                                &table,
                                &pixels_norm,
                                bins,
                                min,
                                step,
                            ),
                            (4, false) => prepared(fold_mask(&original, &mask), min, step)
                                .benchmark_dynamic::<true, false, false>(
                                    signal,
                                    None,
                                    &pixels_norm,
                                ),
                            (4, true) => prepared(fold_mask(&original, &mask), min, step)
                                .benchmark_dynamic::<true, true, false>(signal, None, &pixels_norm),
                            _ => unreachable!(),
                        }
                    })
                    .collect()
            };
            for (which, name) in names.iter().enumerate() {
                let result = run(which);
                println!(
                    "{}",
                    json!({"event":"numerical","threads":threads,"threshold":threshold,"errors":errors,
                    "variant":name,"metrics":check(&result,&actual)})
                );
            }
            // The four runtime-mask paths run at each thread count. Recompaction
            // diagnostic runs only at eight threads, to limit allocation churn.
            let nvariants = if threads == 8 { 5 } else { 4 };
            for rep in 0..reps {
                let mut order: Vec<usize> = (0..nvariants).collect();
                order.rotate_left(rep % nvariants);
                if rep % 2 == 1 {
                    order.reverse();
                }
                for which in order {
                    black_box(run(which));
                    println!(
                        "{}",
                        json!({"event":"begin","threads":threads,"threshold":threshold,"errors":errors,"variant":names[which],"repeat":rep})
                    );
                    std::io::stdout().flush().unwrap();
                    let started = Instant::now();
                    let mut loops = 0;
                    loop {
                        black_box(run(which));
                        loops += 1;
                        if started.elapsed().as_secs_f64() >= block_seconds {
                            break;
                        }
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    println!(
                        "{}",
                        json!({"event":"measurement","threads":threads,"threshold":threshold,"errors":errors,
                        "variant":names[which],"repeat":rep,"loops":loops,"frames":nframes,"seconds":elapsed,"fps":nframes as f64*loops as f64/elapsed})
                    );
                    std::io::stdout().flush().unwrap();
                }
            }
        }
        // Mask-generation stage alone; this is also included in every full run.
        for rep in 0..reps {
            let started = Instant::now();
            let mut loops = 0;
            loop {
                let results: Vec<Vec<u8>> = frames
                    .par_iter()
                    .map(|s| update_mask(s, &base, threshold))
                    .collect();
                black_box(results);
                loops += 1;
                if started.elapsed().as_secs_f64() >= block_seconds {
                    break;
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({"event":"mask_update_measurement","threads":threads,"threshold":threshold,
                "repeat":rep,"loops":loops,"seconds":elapsed,"fps":nframes as f64*loops as f64/elapsed})
            );
            std::io::stdout().flush().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_masks_recompute_denominators_and_allow_unmasking() {
        let original = Csr {
            ptr: vec![0, 3, 4, 4],
            pixel: vec![0, 1, 2, 2],
            weights: vec![0.25, 0.5, 0.75, 1.],
        };
        let table = transpose(&original, 3);
        let csr = prepared(original, 0., 1.);
        let norm = [1., 2., 3.];
        for (signal, mask) in [
            ([4., f64::NAN, 9.], [0, 1, 0]),
            ([4., 8., 9.], [0, 0, 0]),
            ([4., 8., 9.], [1, 1, 1]),
        ] {
            let expected = csr.benchmark_dynamic::<false, true, false>(&signal, Some(&mask), &norm);
            let fused = csr.benchmark_dynamic::<false, true, true>(&signal, Some(&mask), &norm);
            let csc = csc_dynamic::<false, true, 1>(&signal, &mask, &table, &norm, 3, 0., 1.);
            assert_eq!(expected, fused);
            assert_eq!(expected, csc);
            if mask == [0, 1, 0] {
                assert_eq!(expected.1, vec![7.75 / 2.5, 3., 0.]);
                assert_eq!(expected.2, vec![(5.3125f64 / 6.25).sqrt(), 1., 0.]);
            }
            let fast = csr.benchmark_dynamic::<true, true, true>(&signal, Some(&mask), &norm);
            let four = csc_dynamic::<true, true, 4>(&signal, &mask, &table, &norm, 3, 0., 1.);
            check(&[four], &[fast]);
        }
        assert_eq!(update_mask(&[4., 8., 9.], &[0, 1, 0], 8.), vec![0, 1, 1]);
    }
}
