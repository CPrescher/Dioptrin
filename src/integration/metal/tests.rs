use super::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::integration::{IntegrationMethod, Integrator, MaskMode};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::mask::Mask;
use crate::poni::Poni;

fn test_integrator(method: IntegrationMethod, shape: (usize, usize)) -> Integrator {
    let poni = Poni::load("test_data/v1.poni").expect("v1.poni");
    let geom = Geometry::new(shape.0, shape.1, poni);
    let mut ai = Integrator::new(geom)
        .with_method(method)
        .with_mask_mode(MaskMode::Fixed);
    ai.set_polarization_factor(Some(0.965));
    ai
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn make_frame(data: Vec<f64>, shape: (usize, usize)) -> Frame {
    Frame {
        name: String::new(),
        data,
        dim1: shape.0,
        dim2: shape.1,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn synthetic_frame_f64(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| {
            let x = ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(374761393)) % 10_000) as f64;
            if i % 97 == 0 { -5.0 } else { x * 0.25 }
        })
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn assert_close(actual: &[f64], expected: &[f64], rtol: f64, atol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = if e.abs() > 0.0 { diff / e.abs() } else { diff };
        if diff > atol + rtol * e.abs() {
            panic!("{label}: bin {i} mismatch: actual={a}, expected={e}, diff={diff}, rel={rel}");
        }
    }
}

#[test]
fn metal_info_does_not_panic() {
    let info = metal_info(0);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        assert!(info.compiled);
        assert!(info.available, "expected Metal GPU: {:?}", info.reason);
        assert!(info.device_name.is_some());
    }
    #[cfg(not(feature = "metal"))]
    {
        assert!(!info.compiled);
        assert!(!info.available);
    }
}

#[test]
fn configuration_validation_runs_before_device() {
    let ai = test_integrator(IntegrationMethod::Histogram, (64, 64));
    let err = ai
        .prepare_metal1d((64, 64), 100, &Metal1dOptions::new())
        .unwrap_err();
    assert!(matches!(err, MetalError::Unsupported(_)));

    let ai =
        test_integrator(IntegrationMethod::PixelSplit, (64, 64)).with_mask_mode(MaskMode::Dynamic);
    let err = ai
        .prepare_metal1d((64, 64), 100, &Metal1dOptions::new())
        .unwrap_err();
    assert!(matches!(err, MetalError::Unsupported(_)));

    let ai = test_integrator(IntegrationMethod::PixelSplit, (64, 64));
    let err = ai
        .prepare_metal1d((64, 64), 0, &Metal1dOptions::new())
        .unwrap_err();
    assert!(matches!(err, MetalError::InvalidInput(_)));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_1d_matches_cpu_across_methods_precisions_and_dtypes() {
    let shape = (128, 128);
    let pixels = shape.0 * shape.1;
    let bins = 200;

    let mut mask_data = vec![0u8; pixels];
    for i in (0..pixels).step_by(19) {
        mask_data[i] = 1;
    }

    let methods = [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::PyfaiFull,
        IntegrationMethod::Supersampled { n_ss: 4 },
    ];

    let frames_f64: Vec<Vec<f64>> = (0..9)
        .map(|f| synthetic_frame_f64(shape.0, shape.1, f + 1))
        .collect();
    let frames_f32: Vec<Vec<f32>> = frames_f64
        .iter()
        .map(|fr| fr.iter().map(|&v| v as f32).collect())
        .collect();
    let frames_u16: Vec<Vec<u16>> = (0..9)
        .map(|f| {
            (0..pixels)
                .map(|i| ((i.wrapping_mul(31) + f * 17) % 4096) as u16)
                .collect()
        })
        .collect();
    let frames_u32: Vec<Vec<u32>> = frames_u16
        .iter()
        .map(|fr| fr.iter().map(|&v| v as u32 * 3).collect())
        .collect();
    let frames_i32: Vec<Vec<i32>> = frames_u16
        .iter()
        .map(|fr| {
            fr.iter()
                .enumerate()
                .map(|(i, &v)| if i % 23 == 0 { -(v as i32) } else { v as i32 })
                .collect()
        })
        .collect();

    for method in methods {
        let mut ai = test_integrator(method, shape);
        ai.set_mask(Mask::from_vec(mask_data.clone(), shape.0, shape.1).unwrap());

        for precision in [MetalPrecision::F32, MetalPrecision::F32Compensated] {
            let plan = ai
                .prepare_metal1d(
                    shape,
                    bins,
                    &Metal1dOptions::new().with_precision(precision),
                )
                .expect("prepare_metal1d");

            // f32 stack (9 frames -> tests 8-frame tiled + 1 remainder frame)
            let stack_f32: Vec<f32> = frames_f32.iter().flatten().copied().collect();
            let batch_res = plan.integrate_stack(&stack_f32, true).expect("batch f32");
            assert_eq!(batch_res.len(), 9);

            // Single-frame must be bit-identical to batch for the same frame.
            let single_res = plan
                .integrate_frame(&frames_f32[0], true)
                .expect("single f32");
            assert_eq!(single_res.intensity, batch_res[0].intensity);
            assert_eq!(single_res.errors, batch_res[0].errors);

            // Check against CPU strict f64 reference
            for (f, res) in batch_res.iter().enumerate() {
                let frame = make_frame(frames_f64[f].clone(), shape);
                let (_rad, cpu_int, cpu_err) = ai.integrate1d_with_errors(&frame, bins);
                let rtol = if precision == MetalPrecision::F32Compensated {
                    2e-6
                } else {
                    2e-5
                };
                assert_close(
                    &res.intensity,
                    &cpu_int,
                    rtol,
                    1e-5,
                    &format!("{method:?}/{precision}/f32/frame{f}/int"),
                );
                assert_close(
                    res.errors.as_ref().unwrap(),
                    &cpu_err,
                    rtol,
                    1e-5,
                    &format!("{method:?}/{precision}/f32/frame{f}/err"),
                );
            }

            // Native u16, u32, i32
            let stack_u16: Vec<u16> = frames_u16.iter().flatten().copied().collect();
            let res_u16 = plan.integrate_stack(&stack_u16, true).expect("u16");
            let cpu_u16_0 = make_frame(frames_u16[0].iter().map(|&v| v as f64).collect(), shape);
            let (_, ref_u16_int, _) = ai.integrate1d_with_errors(&cpu_u16_0, bins);
            assert_close(
                &res_u16[0].intensity,
                &ref_u16_int,
                2e-5,
                1e-4,
                &format!("{method:?}/{precision}/u16/int"),
            );

            let stack_u32: Vec<u32> = frames_u32.iter().flatten().copied().collect();
            let res_u32 = plan.integrate_stack(&stack_u32, false).expect("u32");
            let cpu_u32_0 = make_frame(frames_u32[0].iter().map(|&v| v as f64).collect(), shape);
            let (_, ref_u32_int) = ai.integrate1d(&cpu_u32_0, bins);
            assert_close(
                &res_u32[0].intensity,
                &ref_u32_int,
                2e-5,
                1e-4,
                &format!("{method:?}/{precision}/u32/int"),
            );

            let stack_i32: Vec<i32> = frames_i32.iter().flatten().copied().collect();
            let res_i32 = plan.integrate_stack(&stack_i32, true).expect("i32");
            let cpu_i32_0 = make_frame(frames_i32[0].iter().map(|&v| v as f64).collect(), shape);
            let (_, ref_i32_int, ref_i32_err) = ai.integrate1d_with_errors(&cpu_i32_0, bins);
            assert_close(
                &res_i32[0].intensity,
                &ref_i32_int,
                2e-5,
                1e-4,
                &format!("{method:?}/{precision}/i32/int"),
            );
            assert_close(
                res_i32[0].errors.as_ref().unwrap(),
                &ref_i32_err,
                2e-5,
                1e-4,
                &format!("{method:?}/{precision}/i32/err"),
            );
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_2d_matches_cpu_including_subwarp_and_shared_zero_copy() {
    let shape = (128, 128);
    let pixels = shape.0 * shape.1;
    let n_rad = 64;
    let n_azim = 36;

    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::PyfaiFull,
        IntegrationMethod::Supersampled { n_ss: 4 },
    ] {
        let ai = test_integrator(method, shape);
        let plan = ai
            .prepare_metal2d(shape, n_rad, n_azim, None, &Metal2dOptions::new())
            .expect("prepare_metal2d");
        assert_eq!(plan.engine.layout(), device::Layout::SubWarp);

        // Allocate shared buffer for 4 frames (tests zero-copy binding!)
        let mut shared = plan
            .alloc_shared_bytes(4 * pixels * size_of::<f32>())
            .expect("alloc_shared_bytes");
        let frames_f64: Vec<Vec<f64>> = (0..4)
            .map(|f| synthetic_frame_f64(shape.0, shape.1, 100 + f))
            .collect();
        {
            let slice_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(shared.as_mut_ptr().cast::<f32>(), 4 * pixels)
            };
            for (f, fr) in frames_f64.iter().enumerate() {
                for (i, &v) in fr.iter().enumerate() {
                    slice_f32[f * pixels + i] = v as f32;
                }
            }
        }
        let stack_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(shared.as_slice().as_ptr().cast::<f32>(), 4 * pixels)
        };

        let mut out_int = vec![0.0_f64; 4 * n_rad * n_azim];
        let mut out_err = vec![0.0_f64; 4 * n_rad * n_azim];
        plan.integrate_stack_into(stack_f32, &mut out_int, Some(&mut out_err))
            .expect("2d zero-copy integrate_stack_into");

        for (f, fr) in frames_f64.iter().enumerate() {
            let frame = make_frame(fr.clone(), shape);
            let (cpu_int, cpu_err) = ai.integrate2d_with_errors(&frame, n_rad, n_azim);
            let s = f * n_rad * n_azim;
            assert_close(
                &out_int[s..s + n_rad * n_azim],
                &cpu_int,
                2e-6,
                1e-5,
                &format!("2d/{method:?}/frame{f}/int"),
            );
            assert_close(
                &out_err[s..s + n_rad * n_azim],
                &cpu_err,
                2e-6,
                1e-5,
                &format!("2d/{method:?}/frame{f}/err"),
            );
        }
    }
}

/// Minimal Bitshuffle + uncompressed-literal LZ4 encoder for unit testing GPU decode.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn encode_bslz4_literals(raw: &[u8], elem_bytes: usize, block_bytes: usize) -> Vec<u8> {
    let n_elems = raw.len() / elem_bytes;
    let block_elems = block_bytes / elem_bytes;
    let n_full = n_elems / block_elems;
    let rem_elems = n_elems % block_elems;
    let rem_comp_elems = (rem_elems / 8) * 8;
    let tail_raw_elems = rem_elems % 8;

    let mut out = Vec::new();
    out.extend_from_slice(&(raw.len() as u64).to_be_bytes());
    out.extend_from_slice(&(block_bytes as u32).to_be_bytes());

    let bitshuffle_and_lz4 = |src: &[u8], dst: &mut Vec<u8>| {
        let count = src.len() / elem_bytes;
        let plane_bytes = count / 8;
        let bit_count = elem_bytes * 8;
        let mut shuffled = vec![0u8; src.len()];
        for e in 0..count {
            let mut val = 0u32;
            for b in 0..elem_bytes {
                val |= (src[e * elem_bytes + b] as u32) << (b * 8);
            }
            for bit in 0..bit_count {
                if ((val >> bit) & 1) != 0 {
                    shuffled[bit * plane_bytes + (e >> 3)] |= 1u8 << (e & 7);
                }
            }
        }
        let mut lz4 = Vec::with_capacity(shuffled.len() + 32);
        let lit_len = shuffled.len();
        if lit_len < 15 {
            lz4.push((lit_len as u8) << 4);
        } else {
            lz4.push(15u8 << 4);
            let mut rem = lit_len - 15;
            while rem >= 255 {
                lz4.push(255);
                rem -= 255;
            }
            lz4.push(rem as u8);
        }
        lz4.extend_from_slice(&shuffled);
        dst.extend_from_slice(&(lz4.len() as u32).to_be_bytes());
        dst.extend_from_slice(&lz4);
    };

    for b in 0..n_full {
        let start = b * block_bytes;
        bitshuffle_and_lz4(&raw[start..start + block_bytes], &mut out);
    }
    if rem_comp_elems > 0 {
        let start = n_full * block_bytes;
        let len = rem_comp_elems * elem_bytes;
        bitshuffle_and_lz4(&raw[start..start + len], &mut out);
    }
    if tail_raw_elems > 0 {
        let start = (n_full * block_elems + rem_comp_elems) * elem_bytes;
        out.extend_from_slice(&raw[start..]);
    }
    out
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_bslz4_decode_and_integrate_u16_and_u32() {
    let shape = (65, 67); // non-multiple of 8 to exercise full, partial, and raw-tail paths!
    let pixels = shape.0 * shape.1;
    let bins = 120;

    let ai = test_integrator(IntegrationMethod::PixelSplit, shape);
    let plan1d = ai
        .prepare_metal1d(shape, bins, &Metal1dOptions::new())
        .expect("prepare_metal1d");
    let plan2d = ai
        .prepare_metal2d(shape, 32, 18, None, &Metal2dOptions::new())
        .expect("prepare_metal2d");

    // u16 test
    let frame_u16: Vec<u16> = (0..pixels)
        .map(|i| ((i * 73 + 11) % 65000) as u16)
        .collect();
    let raw_u16: Vec<u8> = frame_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let chunk_u16 = encode_bslz4_literals(&raw_u16, 2, 1024);

    let direct_1d = plan1d.integrate_frame(&frame_u16, true).unwrap();
    let bslz4_1d = plan1d
        .integrate_bslz4(&[chunk_u16.as_slice()], Bslz4Dtype::U16, true)
        .unwrap();
    assert_eq!(bslz4_1d[0].intensity, direct_1d.intensity);
    assert_eq!(bslz4_1d[0].errors, direct_1d.errors);

    // u32 test (2D)
    let frame_u32: Vec<u32> = (0..pixels)
        .map(|i| ((i * 197 + 5) % 500_000) as u32)
        .collect();
    let raw_u32: Vec<u8> = frame_u32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let chunk_u32 = encode_bslz4_literals(&raw_u32, 4, 2048);

    let direct_2d = plan2d.integrate_frame(&frame_u32, true).unwrap();
    let bslz4_2d = plan2d
        .integrate_bslz4(&[chunk_u32.as_slice()], Bslz4Dtype::U32, true)
        .unwrap();
    assert_eq!(bslz4_2d[0].intensity, direct_2d.intensity);
    assert_eq!(bslz4_2d[0].errors, direct_2d.errors);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_readonly_static_memory_integrates_cleanly() {
    // Static `.rodata` array in read-only memory pages exercises
    // `newBufferWithBytesNoCopy` (and fallback) on read-only input pages.
    static READONLY_FRAME: [u16; 64 * 64] = [42u16; 64 * 64];
    let shape = (64, 64);
    let bins = 64;
    let ai = test_integrator(IntegrationMethod::PixelSplit, shape);
    let plan = ai
        .prepare_metal1d(shape, bins, &Metal1dOptions::new())
        .expect("prepare_metal1d");
    let res = plan
        .integrate_frame(&READONLY_FRAME[..], true)
        .expect("readonly frame integration");
    let cpu_frame = make_frame(READONLY_FRAME.iter().map(|&v| v as f64).collect(), shape);
    let (_, cpu_int, cpu_err) = ai.integrate1d_with_errors(&cpu_frame, bins);
    assert_close(&res.intensity, &cpu_int, 2e-6, 1e-5, "readonly/int");
    assert_close(
        res.errors.as_ref().unwrap(),
        &cpu_err,
        2e-6,
        1e-5,
        "readonly/err",
    );
}
