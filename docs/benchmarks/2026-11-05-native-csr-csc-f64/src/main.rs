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

// One private histogram per frame. Pixel values, coefficients, normalization,
// accumulators and outputs are all f64. Only sparse indices are u32.
fn csc<const FAST: bool, const ERR: bool, const LANES: usize>(
    signal: &[f64],
    table: &Csc,
    norm: &[f64],
    min: f64,
    step: f64,
) -> Output {
    assert!(LANES == 1 || LANES == 4);
    let mut out = vec![0.; norm.len() * LANES];
    let mut var = vec![0.; if ERR { norm.len() * LANES } else { 0 }];
    // SAFETY: transpose validates every index; frame dimensions and denominator
    // length are validated at setup. Each frame owns its output arrays.
    unsafe {
        let bins = table.bin.as_ptr();
        let weights = table.weights.as_ptr();
        let ptr = table.ptr.as_ptr();
        let values = signal.as_ptr();
        let dst = out.as_mut_ptr();
        let variance = var.as_mut_ptr();
        for p in 0..signal.len() {
            let x = *values.add(p);
            let start = *ptr.add(p) as usize;
            let end = *ptr.add(p + 1) as usize;
            for k in start..end {
                let b = *bins.add(k) as usize + (p % LANES) * norm.len();
                let w = *weights.add(k);
                let old = *dst.add(b);
                *dst.add(b) = if FAST { w.mul_add(x, old) } else { old + w * x };
                if ERR {
                    let old = *variance.add(b);
                    let v = x.max(0.);
                    *variance.add(b) = if FAST {
                        (w * w).mul_add(v, old)
                    } else {
                        old + w * w * v
                    };
                }
            }
        }
    }
    if LANES == 4 {
        for b in 0..norm.len() {
            out[b] = (out[b] + out[norm.len() + b])
                + (out[2 * norm.len() + b] + out[3 * norm.len() + b]);
            if ERR {
                var[b] = (var[b] + var[norm.len() + b])
                    + (var[2 * norm.len() + b] + var[3 * norm.len() + b]);
            }
        }
        out.truncate(norm.len());
        if ERR {
            var.truncate(norm.len());
        }
    }
    for b in 0..norm.len() {
        out[b] = if norm[b] == 0. { 0. } else { out[b] / norm[b] };
        if ERR {
            var[b] = if norm[b] == 0. {
                0.
            } else {
                (var[b] / (norm[b] * norm[b])).max(0.).sqrt()
            };
        }
    }
    (math::histogram::bins(norm.len(), min, step), out, var)
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args[1].parse().unwrap();
    let data_root = Path::new(&args[2]);
    let lut_root = Path::new(&args[3]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(data_root.join("inputs.json")).unwrap()).unwrap();
    let npix =
        meta["shape"][1].as_u64().unwrap() as usize * meta["shape"][2].as_u64().unwrap() as usize;
    let nframes = meta["shape"][0].as_u64().unwrap() as usize;
    let file = File::open(data_root.join("frames.f64")).unwrap();
    // This file is immutable for the entire experiment and page-aligned. Its
    // readonly mapping outlives all native slices; every f64 bit pattern is valid.
    let map = unsafe { MmapOptions::new().map(&file).unwrap() };
    assert!(cfg!(target_endian = "little"));
    assert_eq!(map.as_ptr().align_offset(std::mem::align_of::<f64>()), 0);
    assert_eq!(map.len(), npix * nframes * 8);
    let data = unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<f64>(), map.len() / 8) };
    let frames: Vec<&[f64]> = data.chunks_exact(npix).collect();
    let mask = fs::read(data_root.join("mask.u8")).unwrap();
    assert_eq!(mask.len(), npix);
    let ptr = fs::read(lut_root.join("indptr.u64"))
        .unwrap()
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()) as usize)
        .collect();
    let original = Csr {
        ptr,
        pixel: read_u32(&lut_root.join("indices.u32")),
        weights: read_f64(&lut_root.join("weights.f64")),
    };
    let folded = fold_mask(&original, &mask);
    let table = transpose(&folded, npix);
    let norm_pixels = read_f64(&lut_root.join("normalization.f64"));
    assert_eq!(norm_pixels.len(), npix);
    let radial = read_f64(&lut_root.join("radial.f64"));
    let step = radial[1] - radial[0];
    let min = radial[0] - step / 2.;
    println!(
        "{}",
        json!({"event":"setup","threads":threads,"frames":nframes,"npix":npix,"bins":radial.len(),"weights":"f64 throughout","inputs":"same readonly f64 mapping","original_nnz":original.weights.len(),"folded_nnz":folded.weights.len(),"csr_bytes":original.weights.len()*12+original.ptr.len()*8,"folded_csr_bytes":folded.weights.len()*12+folded.ptr.len()*8,"csc_bytes":table.weights.len()*12+table.ptr.len()*4})
    );
    let production = prepared(original, min, step);
    let strict_norm = production
        .raw1d::<false, false>(frames[0], Some(&mask), &norm_pixels)
        .sum_norm;
    let fast_norm = production
        .raw1d::<true, false>(frames[0], Some(&mask), &norm_pixels)
        .sum_norm;
    let compact = prepared(folded, min, step);
    compact.benchmark_seed_normalization(strict_norm.clone(), fast_norm.clone());
    let names = [
        "production_csr_fast",
        "folded_csr_fast",
        "csc_f64_fma",
        "csc_f64_four_histograms",
        "folded_csr_strict",
        "csc_f64_strict",
    ];
    for errors in [false, true] {
        let run = |which: usize| -> Vec<Output> {
            frames
                .par_iter()
                .map(|signal| match (which, errors) {
                    (0, false) => {
                        production.integrate::<true, false>(signal, Some(&mask), &norm_pixels)
                    }
                    (0, true) => {
                        production.integrate::<true, true>(signal, Some(&mask), &norm_pixels)
                    }
                    (1, false) => compact.integrate::<true, false>(signal, None, &norm_pixels),
                    (1, true) => compact.integrate::<true, true>(signal, None, &norm_pixels),
                    (2, false) => csc::<true, false, 1>(signal, &table, &fast_norm, min, step),
                    (2, true) => csc::<true, true, 1>(signal, &table, &fast_norm, min, step),
                    (3, false) => csc::<true, false, 4>(signal, &table, &fast_norm, min, step),
                    (3, true) => csc::<true, true, 4>(signal, &table, &fast_norm, min, step),
                    (4, false) => compact.integrate::<false, false>(signal, None, &norm_pixels),
                    (4, true) => compact.integrate::<false, true>(signal, None, &norm_pixels),
                    (5, false) => csc::<false, false, 1>(signal, &table, &strict_norm, min, step),
                    (5, true) => csc::<false, true, 1>(signal, &table, &strict_norm, min, step),
                    _ => unreachable!(),
                })
                .collect()
        };
        let fast_reference = run(0);
        let strict_reference = run(4);
        for (which, name) in names.iter().enumerate() {
            let result = run(which);
            let reference = if which >= 4 {
                &strict_reference
            } else {
                &fast_reference
            };
            println!(
                "{}",
                json!({"event":"numerical","threads":threads,"errors":errors,"variant":name,"comparison":if which>=4 {"folded_csr_strict"} else {"production_csr_fast"},"metrics":check(&result,reference)})
            );
        }
        for rep in 0..5 {
            let mut order: Vec<usize> = (0..names.len()).collect();
            order.rotate_left(rep % names.len());
            if rep % 2 == 1 {
                order.reverse();
            }
            for which in order {
                black_box(run(which));
                println!(
                    "{}",
                    json!({"event":"begin","threads":threads,"errors":errors,"variant":names[which],"repeat":rep})
                );
                std::io::stdout().flush().unwrap();
                let start = Instant::now();
                let mut loops = 0;
                loop {
                    black_box(run(which));
                    loops += 1;
                    if start.elapsed().as_secs_f64() >= 1. {
                        break;
                    }
                }
                let elapsed = start.elapsed().as_secs_f64();
                println!(
                    "{}",
                    json!({"event":"measurement","threads":threads,"errors":errors,"variant":names[which],"repeat":rep,"frames":nframes,"loops":loops,"seconds":elapsed,"fps":nframes as f64*loops as f64/elapsed})
                );
                std::io::stdout().flush().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transpose_and_float64_kernel_preserve_coefficients_and_masked_nan() {
        let original = Csr {
            ptr: vec![0, 3, 4, 4],
            pixel: vec![0, 1, 2, 2],
            weights: vec![0.25, 0.5, 0.75, 1.],
        };
        let folded = fold_mask(&original, &[0, 1, 0]);
        let table = transpose(&folded, 3);
        let norm = [2.5, 3., 0.];
        let signal = [4., f64::NAN, 9.];
        let strict = csc::<false, true, 1>(&signal, &table, &norm, 0., 1.);
        let fast = csc::<true, true, 1>(&signal, &table, &norm, 0., 1.);
        assert_eq!(strict.1, vec![7.75 / 2.5, 3., 0.]);
        assert_eq!(strict.2, vec![(5.3125f64 / 6.25).sqrt(), 1., 0.]);
        let four = csc::<true, true, 4>(&signal, &table, &norm, 0., 1.);
        assert_eq!(strict.1, four.1);
        assert_eq!(strict.2, four.2);
        assert_eq!(strict.1, fast.1);
        assert_eq!(strict.2, fast.2);
        let csr = prepared(folded, 0., 1.);
        csr.benchmark_seed_normalization(norm.to_vec(), norm.to_vec());
        let reference = csr.integrate::<true, true>(&signal, None, &[1., 2., 3.]);
        assert_eq!(reference.1, fast.1);
        assert_eq!(reference.2, fast.2);
    }
}
