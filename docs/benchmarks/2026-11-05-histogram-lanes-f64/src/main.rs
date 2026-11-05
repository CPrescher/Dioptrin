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
    assert!(LANES.is_power_of_two() && LANES <= 16);
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
    let mut stride = 1;
    while stride < LANES {
        for lane in (0..LANES).step_by(stride * 2) {
            for b in 0..bins {
                let dst = lane * bins + b;
                let src = (lane + stride) * bins + b;
                out[dst] += out[src];
                norm[dst] += norm[src];
                if ERR {
                    var[dst] += var[src];
                }
            }
        }
        stride *= 2;
    }
    out.truncate(bins);
    norm.truncate(bins);
    if ERR {
        var.truncate(bins);
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
    let fixed = args.get(2).is_some_and(|x| x == "fixed");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let root = Path::new("/tmp/dioptrin-bridge-investigation-20261105");
    let lutroot = Path::new("/tmp/dioptrin-bslz4-lut-20261105/pixel_split");
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let npix =
        meta["shape"][1].as_u64().unwrap() as usize * meta["shape"][2].as_u64().unwrap() as usize;
    let nframes = meta["shape"][0].as_u64().unwrap() as usize;
    let file = File::open(root.join("frames.f64")).unwrap();
    let map = unsafe { MmapOptions::new().map(&file).unwrap() };
    assert_eq!(map.len(), nframes * npix * 8);
    assert_eq!(map.as_ptr().align_offset(std::mem::align_of::<f64>()), 0);
    let data = unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<f64>(), map.len() / 8) };
    let frames: Vec<&[f64]> = data.chunks_exact(npix).collect();
    let base = fs::read(root.join("mask.u8")).unwrap();
    let original = Csr {
        ptr: fs::read(lutroot.join("indptr.u64"))
            .unwrap()
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect(),
        pixel: read_u32(&lutroot.join("indices.u32")),
        weights: read_f64(&lutroot.join("weights.f64")),
    };
    let table = transpose(&original, npix);
    let norm = read_f64(&lutroot.join("normalization.f64"));
    let radial = read_f64(&lutroot.join("radial.f64"));
    let bins = radial.len();
    let step = radial[1] - radial[0];
    let min = radial[0] - step / 2.;
    let reference = prepared(original, min, step);
    println!(
        "{}",
        json!({"event":"setup","threads":threads,"fixed":fixed,"frames":nframes,"bins":bins,"npix":npix,"lanes":[1,2,4,8,16],"precision":"f64 throughout","threshold":230,"normalization":"recomputed per frame in every CSC kernel","mask":"native threshold OR static mask and backend copy inside timing, or clone fixed mask"})
    );
    let lanes = [1, 2, 4, 8, 16];
    for errors in [false, true] {
        let expected: Vec<Output> = frames
            .iter()
            .map(|s| {
                let mask = if fixed {
                    base.clone()
                } else {
                    update_mask(s, &base, 230.)
                };
                if errors {
                    reference.benchmark_dynamic::<false, true, true>(s, Some(&mask), &norm)
                } else {
                    reference.benchmark_dynamic::<false, false, true>(s, Some(&mask), &norm)
                }
            })
            .collect();
        let run = |which: usize| -> Vec<Output> {
            frames
                .par_iter()
                .map(|s| {
                    let mask = if fixed {
                        base.clone()
                    } else {
                        update_mask(s, &base, 230.)
                    };
                    macro_rules! call {
                        ($n:literal) => {
                            if errors {
                                csc_dynamic::<true, true, $n>(
                                    s, &mask, &table, &norm, bins, min, step,
                                )
                            } else {
                                csc_dynamic::<true, false, $n>(
                                    s, &mask, &table, &norm, bins, min, step,
                                )
                            }
                        };
                    }
                    match which {
                        0 => call!(1),
                        1 => call!(2),
                        2 => call!(4),
                        3 => call!(8),
                        4 => call!(16),
                        _ => unreachable!(),
                    }
                })
                .collect()
        };
        for (which, n) in lanes.iter().enumerate() {
            println!(
                "{}",
                json!({"event":"numerical","threads":threads,"fixed":fixed,"errors":errors,"lanes":n,"metrics":check(&run(which),&expected)})
            );
        }
        for rep in 0..5 {
            let mut order: Vec<usize> = (0..5).collect();
            order.rotate_left(rep);
            if rep % 2 == 1 {
                order.reverse();
            }
            for which in order {
                black_box(run(which));
                println!(
                    "{}",
                    json!({"event":"begin","threads":threads,"fixed":fixed,"errors":errors,"lanes":lanes[which],"repeat":rep})
                );
                std::io::stdout().flush().unwrap();
                let started = Instant::now();
                let mut loops = 0;
                loop {
                    black_box(run(which));
                    loops += 1;
                    if started.elapsed().as_secs_f64() >= 1. {
                        break;
                    }
                }
                let elapsed = started.elapsed().as_secs_f64();
                println!(
                    "{}",
                    json!({"event":"measurement","threads":threads,"fixed":fixed,"errors":errors,"lanes":lanes[which],"repeat":rep,"loops":loops,"seconds":elapsed,"fps":nframes as f64*loops as f64/elapsed})
                );
                std::io::stdout().flush().unwrap();
            }
        }
    }
}
