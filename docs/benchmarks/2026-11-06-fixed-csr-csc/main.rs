use memmap2::MmapOptions;
use rayon::prelude::*;
use serde_json::json;
use std::{fs, fs::File, hint::black_box, path::Path, time::Instant};
mod frame {
    pub trait Pixel: Copy + Send + Sync {
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
mod integration {
    pub use dioptrin::integration::MaskMode;
}
#[allow(dead_code)]
mod lut {
    pub use dioptrin::lut::{IntermediateResult1d, IntermediateResult2d, Lut1d, Lut2d};
    #[path = "../snapshots/production_csc.rs"]
    mod csc;
    pub mod fixed_csc;
    #[path = "../snapshots/production_prepared.rs"]
    pub mod prepared;
}
fn f64s(p: &Path) -> Vec<f64> {
    fs::read(p)
        .unwrap()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let workers: usize = args[1].parse().unwrap();
    let root = Path::new(&args[2]);
    let table = Path::new(&args[3]);
    let block_seconds: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()
        .unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let npix =
        meta["shape"][1].as_u64().unwrap() as usize * meta["shape"][2].as_u64().unwrap() as usize;
    let file = File::open(root.join("frames.f64")).unwrap();
    let map = unsafe { MmapOptions::new().map(&file).unwrap() };
    assert_eq!(map.as_ptr().align_offset(8), 0);
    assert!(cfg!(target_endian = "little"));
    let data = unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<f64>(), map.len() / 8) };
    let frames: Vec<_> = data.chunks_exact(npix).collect();
    let original_rows: Vec<usize> = fs::read(table.join("indptr.u64"))
        .unwrap()
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect();
    let original_cols: Vec<usize> = fs::read(table.join("indices.u32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect();
    let original_weights = f64s(&table.join("weights.f64"));
    let norm = f64s(&table.join("normalization.f64"));
    let radial = f64s(&table.join("radial.f64"));
    let bins = radial.len();
    let step = radial[1] - radial[0];
    let min = radial[0] - step / 2.;
    let quick = args.get(5).is_some_and(|s| s == "lanes");
    for dense in [false, true] {
        if quick && dense {
            continue;
        }
        let mut mask = fs::read(root.join("mask.u8")).unwrap();
        if dense {
            for (p, m) in mask.iter_mut().enumerate() {
                *m |= u8::from(frames[0][p] > 230.);
            }
        }
        let mut rows = vec![0usize];
        let mut cols = Vec::new();
        let mut weights = Vec::new();
        for b in 0..bins {
            for k in original_rows[b]..original_rows[b + 1] {
                if mask[original_cols[k]] == 0 {
                    cols.push(original_cols[k]);
                    weights.push(original_weights[k]);
                }
            }
            rows.push(cols.len());
        }
        let started = Instant::now();
        let csr = lut::prepared::Prepared1d::new(lut::Lut1d {
            row_ptr: rows.clone(),
            col_idx: cols.clone(),
            values: Some(weights.clone()),
            hist_min: min,
            hist_step: step,
        });
        csr.prepare_normalization(None, &norm, true);
        let denoms = csr.raw1d::<true, false>(frames[0], None, &norm).sum_norm;
        let csr_setup = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let csc = lut::fixed_csc::FixedCsc::new(&rows, &cols, &weights, npix, denoms);
        let csc_setup = started.elapsed().as_secs_f64();
        println!(
            "{}",
            json!({"event":"setup","workers":workers,"frozen_threshold":dense,"masked_fraction":mask.iter().filter(|&&m|m!=0).count() as f64/npix as f64,"csr_prepare_seconds":csr_setup,"csc_prepare_including_transpose_check_seconds":csc_setup,"nnz":cols.len()})
        );
        for errors in [false, true] {
            if quick && !errors {
                continue;
            }
            let execute = |which: usize| -> Vec<(Vec<f64>, Vec<f64>)> {
                frames
                    .par_iter()
                    .map(|image| {
                        if which == 0 {
                            let (_, y, e) = if errors {
                                csr.integrate::<true, true>(image, None, &norm)
                            } else {
                                csr.integrate::<true, false>(image, None, &norm)
                            };
                            (y, e)
                        } else {
                            let _radial = black_box(csr.bins());
                            if errors {
                                match which {
                                    2 => csc.integrate::<true, 2>(image),
                                    3 => csc.integrate::<true, 1>(image),
                                    _ => csc.integrate::<true, 4>(image),
                                }
                            } else {
                                match which {
                                    2 => csc.integrate::<false, 2>(image),
                                    3 => csc.integrate::<false, 1>(image),
                                    _ => csc.integrate::<false, 4>(image),
                                }
                            }
                        }
                    })
                    .collect()
            };
            let reference = execute(0);
            let candidate = execute(1);
            for which in [2, 3] {
                let output = execute(which);
                for (a, b) in reference.iter().zip(output) {
                    for (x, y) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
                        assert!((x - y).abs() <= 1e-9 + 5e-12 * x.abs());
                    }
                }
            }
            let mut diff = 0.;
            let mut sum = 0.;
            let name = if dense { "frozen" } else { "static" };
            let external = f64s(&root.join(format!("current-native-{name}.intensity.f64")));
            let external_errors = f64s(&root.join(format!("current-native-{name}.errors.f64")));
            for (i, (y, e)) in reference.iter().enumerate() {
                for (b, &v) in y.iter().enumerate() {
                    let expected = external[i * bins + b];
                    assert!(
                        (v - expected).abs() <= 1e-9 + 5e-12 * expected.abs(),
                        "exported table differs from current native API"
                    );
                }
                if errors {
                    for (b, &v) in e.iter().enumerate() {
                        let expected = external_errors[i * bins + b];
                        assert!((v - expected).abs() <= 1e-9 + 5e-12 * expected.abs());
                    }
                }
            }

            for (a, b) in reference.iter().zip(&candidate) {
                for (x, y) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
                    assert!((x - y).abs() <= 1e-9 + 5e-12 * x.abs(), "{x} vs {y}");
                    diff += (x - y) * (x - y);
                    sum += x * x;
                }
            }
            for repeat in 0..5 {
                let order = if quick {
                    if repeat % 2 == 0 {
                        vec![0, 1, 2, 3]
                    } else {
                        vec![3, 2, 1, 0]
                    }
                } else if repeat % 2 == 0 {
                    vec![0, 1]
                } else {
                    vec![1, 0]
                };
                for which in order {
                    let t = Instant::now();
                    let mut loops = 0;
                    while t.elapsed().as_secs_f64() < block_seconds {
                        black_box(execute(which));
                        loops += 1;
                    }
                    let elapsed = t.elapsed().as_secs_f64();
                    println!(
                        "{}",
                        json!({"event":"timing","workers":workers,"frozen_threshold":dense,"errors":errors,"kernel":match which {0=>"production_fixed_csr",2=>"fixed_csc_two_lanes",3=>"fixed_csc_one_lane",_=>"experimental_fixed_csc_cached_norm"},"repeat":repeat,"seconds":elapsed,"frames":loops*frames.len(),"fps":loops as f64*frames.len() as f64/elapsed,"relative_l2":(diff/sum).sqrt()})
                    );
                }
            }
        }
    }
}
