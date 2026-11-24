//! Comprehensive before/after CPU vs Metal benchmark suite (1D, 2D, native ints, zero-copy, BSLZ4).
//!
//! Run with:
//! `cargo test --release --features metal --lib integration::metal::bench::run_metal_benchmarks -- --ignored --nocapture`

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use super::*;
use crate::frame::{Frame, Pixel};
use crate::geometry::Geometry;
use crate::integration::{Bslz4Dtype, IntegrationMethod, Integrator, MaskMode};
use crate::mask::Mask;
use crate::poni::Poni;

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn time_ms(warmups: usize, repeats: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..warmups {
        f();
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1_000.0);
    }
    let med = median_ms(&mut samples);
    let min = samples[0];
    let max = samples[samples.len() - 1];
    (med, min, max)
}

type LoadedFixture = (Vec<f64>, (usize, usize), Poni, Option<Mask>);

fn load_fixture(name: &str) -> Option<LoadedFixture> {
    match name {
        "pilatus" => {
            let frame = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".to_string()).ok()?;
            let poni = Poni::load("test_data/v1.poni").ok()?;
            let mask_bytes: Vec<u8> = frame
                .data
                .iter()
                .map(|&v| u8::from(!v.is_finite() || v < 0.0))
                .collect();
            let mask = Mask::from_vec(mask_bytes, frame.dim1, frame.dim2).ok()?;
            Some((frame.data, (frame.dim1, frame.dim2), poni, Some(mask)))
        }
        "eiger" => {
            let tiff_candidates = [
                "test_data/eiger_0000.tiff",
                "../../../test_data/eiger_0000.tiff",
            ];
            let tiff_path = tiff_candidates.iter().find(|p| Path::new(p).is_file())?;
            let frame = Frame::from_tiff((*tiff_path).to_string()).ok()?;
            let poni = Poni::load("docs/benchmarks/2026-11-05-bslz4/calibration.poni")
                .or_else(|_| Poni::load("test_data/eiger_0000.poni"))
                .ok()?;
            let mask_frame = Frame::from_tiff("test_data/eiger_0000.mask".to_string()).ok()?;
            assert_eq!((mask_frame.dim1, mask_frame.dim2), (frame.dim1, frame.dim2));
            let mask_bytes: Vec<u8> = frame
                .data
                .iter()
                .zip(mask_frame.data.iter())
                .map(|(&v, &m)| u8::from(m != 0.0 || !v.is_finite() || v < 0.0))
                .collect();
            let mask = Mask::from_vec(mask_bytes, frame.dim1, frame.dim2).ok()?;
            Some((frame.data, (frame.dim1, frame.dim2), poni, Some(mask)))
        }
        _ => None,
    }
}

fn make_integrator(
    poni: &Poni,
    shape: (usize, usize),
    mask: &Option<Mask>,
    method: IntegrationMethod,
) -> Integrator {
    let mut ai = Integrator::new(Geometry::new(shape.0, shape.1, poni.clone()))
        .with_method(method)
        .with_mask_mode(MaskMode::Fixed);
    ai.set_polarization_factor(Some(0.965));
    if let Some(m) = mask {
        ai.set_mask(Mask::from_vec(m.data.clone(), m.dim1, m.dim2).unwrap());
    }
    ai
}

/// Runs the current `main` 4-frame SIMD-tiled CPU 1D batch path across Rayon workers,
/// or a prebuilt `Fixed1dPlan` for single-frame calls.
fn cpu_batch1d<P: Pixel + Sync>(
    ai: &Integrator,
    single_plan: &crate::integration::Fixed1dPlan,
    frames: &[&[P]],
    shape: (usize, usize),
    bins: usize,
    errors: bool,
) {
    if frames.len() >= 4 && frames.len().is_multiple_of(4) {
        let groups: Vec<[&[P]; 4]> = frames.as_chunks::<4>().0.to_vec();
        groups.into_par_iter().for_each(|g| {
            let _ = ai.try_integrate1d_batch4(g, shape.0, shape.1, bins, errors);
        });
    } else {
        frames.par_iter().for_each(|fr| {
            let f64_fr: Vec<f64> = fr.iter().map(|&v| v.to_f64()).collect();
            let _ = single_plan.integrate_frame(&f64_fr, errors);
        });
    }
}

/// Runs the current `main` 4-frame SIMD-tiled CPU 2D batch path across Rayon workers.
fn cpu_batch2d<P: Pixel + Sync>(
    ai: &Integrator,
    frames: &[&[P]],
    shape: (usize, usize),
    n_rad: usize,
    n_azim: usize,
    errors: bool,
) {
    if frames.len() >= 4 && frames.len().is_multiple_of(4) {
        let groups: Vec<[&[P]; 4]> = frames.as_chunks::<4>().0.to_vec();
        groups.into_par_iter().for_each(|g| {
            let _ = ai.try_integrate2d_batch4(g, shape.0, shape.1, n_rad, n_azim, None, errors);
        });
    } else {
        frames.par_iter().for_each(|fr| {
            let f = Frame {
                name: String::new(),
                data: fr.iter().map(|&v| v.to_f64()).collect(),
                dim1: shape.0,
                dim2: shape.1,
            };
            if errors {
                let _ = ai.integrate2d_with_errors(&f, n_rad, n_azim);
            } else {
                let _ = ai.integrate2d(&f, n_rad, n_azim);
            }
        });
    }
}

/// Fast Bitshuffle + LZ4 encoder (with RLE matches for realistic compression on diffraction frames).
fn encode_bslz4_realistic(raw: &[u8], elem_bytes: usize, block_bytes: usize) -> Vec<u8> {
    let n_elems = raw.len() / elem_bytes;
    let block_elems = block_bytes / elem_bytes;
    let n_full = n_elems / block_elems;
    let rem_elems = n_elems % block_elems;
    let rem_comp_elems = (rem_elems / 8) * 8;
    let tail_raw_elems = rem_elems % 8;

    let mut out = Vec::new();
    out.extend_from_slice(&(raw.len() as u64).to_be_bytes());
    out.extend_from_slice(&(block_bytes as u32).to_be_bytes());

    let compress_block = |src: &[u8], dst: &mut Vec<u8>| {
        let count = src.len() / elem_bytes;
        let plane_bytes = count / 8;
        let bit_count = elem_bytes * 8;
        let mut shuf = vec![0u8; src.len()];
        for e in 0..count {
            let mut val = 0u32;
            for b in 0..elem_bytes {
                val |= (src[e * elem_bytes + b] as u32) << (b * 8);
            }
            for bit in 0..bit_count {
                if ((val >> bit) & 1) != 0 {
                    shuf[bit * plane_bytes + (e >> 3)] |= 1u8 << (e & 7);
                }
            }
        }
        let mut lz4 = Vec::with_capacity(shuf.len() + 32);
        let mut i = 0usize;
        let mut anchor = 0usize;
        while i < shuf.len() {
            let b = shuf[i];
            let mut run = 1usize;
            while i + run < shuf.len() && shuf[i + run] == b {
                run += 1;
            }
            if run >= 6 && i + run < shuf.len() {
                let lit_end = i + 1;
                let lit_len = lit_end - anchor;
                let match_len = run - 1;
                let tok_lit = lit_len.min(15) as u8;
                let tok_mat = (match_len - 4).min(15) as u8;
                lz4.push((tok_lit << 4) | tok_mat);
                if lit_len >= 15 {
                    let mut r = lit_len - 15;
                    while r >= 255 {
                        lz4.push(255);
                        r -= 255;
                    }
                    lz4.push(r as u8);
                }
                lz4.extend_from_slice(&shuf[anchor..lit_end]);
                lz4.extend_from_slice(&1u16.to_le_bytes());
                if match_len - 4 >= 15 {
                    let mut r = (match_len - 4) - 15;
                    while r >= 255 {
                        lz4.push(255);
                        r -= 255;
                    }
                    lz4.push(r as u8);
                }
                i += run;
                anchor = i;
            } else {
                i += run;
            }
        }
        let lit_len = shuf.len() - anchor;
        let tok_lit = lit_len.min(15) as u8;
        lz4.push(tok_lit << 4);
        if lit_len >= 15 {
            let mut r = lit_len - 15;
            while r >= 255 {
                lz4.push(255);
                r -= 255;
            }
            lz4.push(r as u8);
        }
        lz4.extend_from_slice(&shuf[anchor..]);
        dst.extend_from_slice(&(lz4.len() as u32).to_be_bytes());
        dst.extend_from_slice(&lz4);
    };

    for b in 0..n_full {
        let start = b * block_bytes;
        compress_block(&raw[start..start + block_bytes], &mut out);
    }
    if rem_comp_elems > 0 {
        let start = n_full * block_bytes;
        compress_block(&raw[start..start + rem_comp_elems * elem_bytes], &mut out);
    }
    if tail_raw_elems > 0 {
        let start = (n_full * block_elems + rem_comp_elems) * elem_bytes;
        out.extend_from_slice(&raw[start..]);
    }
    out
}

#[test]
#[ignore]
fn run_metal_benchmarks() {
    let info = metal_info(0);
    println!("=== Metal Benchmark Suite (Before vs After on Apple Silicon) ===");
    println!(
        "Device: {} | Unified: {:?} | Recommended Working Set: {:.1} GiB | Rayon threads: {}",
        info.device_name.as_deref().unwrap_or("?"),
        info.has_unified_memory,
        info.recommended_max_working_set_size.unwrap_or(0) as f64 / (1u64 << 30) as f64,
        rayon::current_num_threads()
    );

    // 1. 1D Benchmarks (2000 bins, with Poisson std)
    const BINS_1D: usize = 2000;
    println!("\n--- 1D Integration ({BINS_1D} bins, Poisson std, compensated f32, ms/frame) ---");
    println!(
        "{:<8} {:<11} {:>5} {:<7} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} | {:>7}",
        "Fixture",
        "Method",
        "Batch",
        "Layout",
        "CPU f64",
        "CPU f32",
        "CPU u16",
        "Mtl f64",
        "Mtl f32",
        "Mtl zc32",
        "Mtl zc16",
        "Best x"
    );

    for fixture_name in ["pilatus", "eiger"] {
        let Some((frame_f64, shape, poni, mask)) = load_fixture(fixture_name) else {
            println!("Skipping {fixture_name} (fixture not found)");
            continue;
        };
        let pixels = shape.0 * shape.1;
        let frame_f32: Vec<f32> = frame_f64.iter().map(|&v| v as f32).collect();
        let frame_u16: Vec<u16> = frame_f64
            .iter()
            .map(|&v| v.clamp(0.0, 65535.0) as u16)
            .collect();

        for method in [
            IntegrationMethod::Counting,
            IntegrationMethod::PixelSplit,
            IntegrationMethod::PyfaiFull,
        ] {
            let ai = make_integrator(&poni, shape, &mask, method);
            let single_cpu_plan = ai.prepare_fixed1d(shape, BINS_1D).unwrap();
            let plan = ai
                .prepare_metal1d(shape, BINS_1D, &Metal1dOptions::new())
                .expect("prepare_metal1d");

            for &batch in &[1usize, 16, 64] {
                let stack_f64: Vec<f64> = std::iter::repeat_n(frame_f64.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();
                let stack_f32: Vec<f32> = std::iter::repeat_n(frame_f32.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();
                let stack_u16: Vec<u16> = std::iter::repeat_n(frame_u16.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();

                let mut zc_f32 = plan
                    .alloc_shared_bytes(batch * pixels * size_of::<f32>())
                    .unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_f32.as_ptr().cast::<u8>(),
                        zc_f32.as_mut_ptr(),
                        batch * pixels * size_of::<f32>(),
                    );
                }
                let zc_f32_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        zc_f32.as_slice().as_ptr().cast::<f32>(),
                        batch * pixels,
                    )
                };

                let mut zc_u16 = plan
                    .alloc_shared_bytes(batch * pixels * size_of::<u16>())
                    .unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_u16.as_ptr().cast::<u8>(),
                        zc_u16.as_mut_ptr(),
                        batch * pixels * size_of::<u16>(),
                    );
                }
                let zc_u16_slice: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        zc_u16.as_slice().as_ptr().cast::<u16>(),
                        batch * pixels,
                    )
                };

                let repeats = if fixture_name == "eiger" && batch == 64 {
                    5
                } else {
                    11
                };
                let refs_f64: Vec<&[f64]> = (0..batch)
                    .map(|i| &stack_f64[i * pixels..(i + 1) * pixels])
                    .collect();
                let refs_f32: Vec<&[f32]> = (0..batch)
                    .map(|i| &stack_f32[i * pixels..(i + 1) * pixels])
                    .collect();
                let refs_u16: Vec<&[u16]> = (0..batch)
                    .map(|i| &stack_u16[i * pixels..(i + 1) * pixels])
                    .collect();

                let (cpu_f64_ms, _, _) = time_ms(2, repeats, || {
                    cpu_batch1d(&ai, &single_cpu_plan, &refs_f64, shape, BINS_1D, true);
                });
                let (cpu_f32_ms, _, _) = time_ms(2, repeats, || {
                    cpu_batch1d(&ai, &single_cpu_plan, &refs_f32, shape, BINS_1D, true);
                });
                let (cpu_u16_ms, _, _) = time_ms(2, repeats, || {
                    cpu_batch1d(&ai, &single_cpu_plan, &refs_u16, shape, BINS_1D, true);
                });

                let mut out_int = vec![0.0_f64; batch * BINS_1D];
                let mut out_err = vec![0.0_f64; batch * BINS_1D];
                let (mtl_f64_ms, _, _) = time_ms(2, repeats, || {
                    plan.integrate_stack_into(&stack_f64, &mut out_int, Some(&mut out_err))
                        .unwrap();
                });
                let (mtl_f32_ms, _, _) = time_ms(2, repeats, || {
                    plan.integrate_stack_into(&stack_f32, &mut out_int, Some(&mut out_err))
                        .unwrap();
                });
                let (mtl_zc32_ms, _, _) = time_ms(2, repeats, || {
                    plan.integrate_stack_into(zc_f32_slice, &mut out_int, Some(&mut out_err))
                        .unwrap();
                });
                let (mtl_zc16_ms, _, _) = time_ms(2, repeats, || {
                    plan.integrate_stack_into(zc_u16_slice, &mut out_int, Some(&mut out_err))
                        .unwrap();
                });

                let n = batch as f64;
                let speedup_u16 = (cpu_u16_ms / n) / (mtl_zc16_ms / n);
                println!(
                    "{:<8} {:<11} {:>5} {:<7} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:>6.2}x",
                    fixture_name,
                    format!("{method:?}"),
                    batch,
                    plan.engine.layout().name(),
                    cpu_f64_ms / n,
                    cpu_f32_ms / n,
                    cpu_u16_ms / n,
                    mtl_f64_ms / n,
                    mtl_f32_ms / n,
                    mtl_zc32_ms / n,
                    mtl_zc16_ms / n,
                    speedup_u16,
                );
            }
        }
    }

    // 2. 2D Cake Benchmarks (1000 x 360 bins, intensity only)
    println!(
        "\n--- 2D Cake Integration (1000 x 360 bins, intensity only, compensated f32, ms/frame) ---"
    );
    println!(
        "{:<8} {:<11} {:>5} {:<7} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>7}",
        "Fixture",
        "Method",
        "Batch",
        "Layout",
        "CPU f64",
        "CPU f32",
        "CPU u16",
        "Mtl f32",
        "Mtl zc32",
        "Mtl zc16",
        "Best x"
    );

    let (n_rad, n_azim) = (1000usize, 360usize);
    for fixture_name in ["pilatus", "eiger"] {
        let Some((frame_f64, shape, poni, mask)) = load_fixture(fixture_name) else {
            continue;
        };
        let pixels = shape.0 * shape.1;
        let frame_f32: Vec<f32> = frame_f64.iter().map(|&v| v as f32).collect();
        let frame_u16: Vec<u16> = frame_f64
            .iter()
            .map(|&v| v.clamp(0.0, 65535.0) as u16)
            .collect();

        for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
            let ai = make_integrator(&poni, shape, &mask, method);
            let plan2d = ai
                .prepare_metal2d(shape, n_rad, n_azim, None, &Metal2dOptions::new())
                .expect("prepare_metal2d");

            for &batch in &[1usize, 16, 64] {
                let stack_f64: Vec<f64> = std::iter::repeat_n(frame_f64.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();
                let stack_f32: Vec<f32> = std::iter::repeat_n(frame_f32.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();
                let stack_u16: Vec<u16> = std::iter::repeat_n(frame_u16.as_slice(), batch)
                    .flatten()
                    .copied()
                    .collect();

                let mut zc_f32 = plan2d
                    .alloc_shared_bytes(batch * pixels * size_of::<f32>())
                    .unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_f32.as_ptr().cast::<u8>(),
                        zc_f32.as_mut_ptr(),
                        batch * pixels * size_of::<f32>(),
                    );
                }
                let zc_f32_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        zc_f32.as_slice().as_ptr().cast::<f32>(),
                        batch * pixels,
                    )
                };

                let mut zc_u16 = plan2d
                    .alloc_shared_bytes(batch * pixels * size_of::<u16>())
                    .unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stack_u16.as_ptr().cast::<u8>(),
                        zc_u16.as_mut_ptr(),
                        batch * pixels * size_of::<u16>(),
                    );
                }
                let zc_u16_slice: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        zc_u16.as_slice().as_ptr().cast::<u16>(),
                        batch * pixels,
                    )
                };

                let repeats = if fixture_name == "eiger" && batch == 64 {
                    3
                } else {
                    7
                };
                let refs_f64: Vec<&[f64]> = (0..batch)
                    .map(|i| &stack_f64[i * pixels..(i + 1) * pixels])
                    .collect();
                let refs_f32: Vec<&[f32]> = (0..batch)
                    .map(|i| &stack_f32[i * pixels..(i + 1) * pixels])
                    .collect();
                let refs_u16: Vec<&[u16]> = (0..batch)
                    .map(|i| &stack_u16[i * pixels..(i + 1) * pixels])
                    .collect();

                let (cpu_f64_ms, _, _) = time_ms(1, repeats, || {
                    cpu_batch2d(&ai, &refs_f64, shape, n_rad, n_azim, false);
                });
                let (cpu_f32_ms, _, _) = time_ms(1, repeats, || {
                    cpu_batch2d(&ai, &refs_f32, shape, n_rad, n_azim, false);
                });
                let (cpu_u16_ms, _, _) = time_ms(1, repeats, || {
                    cpu_batch2d(&ai, &refs_u16, shape, n_rad, n_azim, false);
                });

                let mut out_int = vec![0.0_f64; batch * n_rad * n_azim];
                let (mtl_f32_ms, _, _) = time_ms(1, repeats, || {
                    plan2d
                        .integrate_stack_into(&stack_f32, &mut out_int, None)
                        .unwrap();
                });
                let (mtl_zc32_ms, _, _) = time_ms(1, repeats, || {
                    plan2d
                        .integrate_stack_into(zc_f32_slice, &mut out_int, None)
                        .unwrap();
                });
                let (mtl_zc16_ms, _, _) = time_ms(1, repeats, || {
                    plan2d
                        .integrate_stack_into(zc_u16_slice, &mut out_int, None)
                        .unwrap();
                });

                let n = batch as f64;
                let speedup_u16 = (cpu_u16_ms / n) / (mtl_zc16_ms / n);
                println!(
                    "{:<8} {:<11} {:>5} {:<7} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3} | {:>6.2}x",
                    fixture_name,
                    format!("{method:?}"),
                    batch,
                    plan2d.engine.layout().name(),
                    cpu_f64_ms / n,
                    cpu_f32_ms / n,
                    cpu_u16_ms / n,
                    mtl_f32_ms / n,
                    mtl_zc32_ms / n,
                    mtl_zc16_ms / n,
                    speedup_u16,
                );
            }
        }
    }

    // 3. GPU BSLZ4 decode + 1D/2D integrate benchmark
    println!("\n--- GPU Bitshuffle+LZ4 Decode + Integrate (u16, 16-frame batch) ---");
    for fixture_name in ["pilatus", "eiger"] {
        let Some((frame_f64, shape, poni, mask)) = load_fixture(fixture_name) else {
            continue;
        };
        let frame_u16: Vec<u16> = frame_f64
            .iter()
            .map(|&v| v.clamp(0.0, 65535.0) as u16)
            .collect();
        let raw_u16: Vec<u8> = frame_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let chunk = encode_bslz4_realistic(&raw_u16, 2, 8192);
        let ratio = raw_u16.len() as f64 / chunk.len() as f64;

        let ai = make_integrator(&poni, shape, &mask, IntegrationMethod::PixelSplit);
        let plan1d = ai
            .prepare_metal1d(shape, BINS_1D, &Metal1dOptions::new())
            .unwrap();
        let plan2d = ai
            .prepare_metal2d(shape, 1000, 360, None, &Metal2dOptions::new())
            .unwrap();

        let batch = 16usize;
        let chunks: Vec<&[u8]> = vec![chunk.as_slice(); batch];
        let mut out1d = vec![0.0_f64; batch * BINS_1D];
        let mut err1d = vec![0.0_f64; batch * BINS_1D];
        let mut out2d = vec![0.0_f64; batch * 1000 * 360];

        let (ms_1d, _, _) = time_ms(2, 7, || {
            plan1d
                .integrate_bslz4_into(&chunks, Bslz4Dtype::U16, &mut out1d, Some(&mut err1d))
                .unwrap();
        });
        let (ms_2d, _, _) = time_ms(1, 5, || {
            plan2d
                .integrate_bslz4_into(&chunks, Bslz4Dtype::U16, &mut out2d, None)
                .unwrap();
        });
        println!(
            "{:<8} u16 BSLZ4 ({:.1}x comp, {:.2} MB/frame): 1D+std = {:.3} ms/frame | 2D = {:.3} ms/frame",
            fixture_name,
            ratio,
            chunk.len() as f64 / 1e6,
            ms_1d / batch as f64,
            ms_2d / batch as f64
        );
    }
}
