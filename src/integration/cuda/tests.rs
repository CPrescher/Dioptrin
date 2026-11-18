use super::*;
use crate::{geometry::Geometry, mask::Mask, poni::Poni};

#[test]
fn fixed_mask_batches_match_strict_cpu() -> Result<()> {
    let Some(cuda) = Cuda::open()? else {
        assert!(
            std::env::var_os("DIOPTRIN_REQUIRE_CUDA").is_none(),
            "CUDA driver, GPU or NVRTC unavailable"
        );
        eprintln!("Skipped CUDA test because the driver, GPU or NVRTC is unavailable");
        return Ok(());
    };
    let shape = (61, 113);
    let pixels = shape.0 * shape.1;
    let bins = 137;
    let mut poni = Poni::new();
    poni.pixel1 = 1.0e-4;
    poni.pixel2 = 1.2e-4;
    poni.distance = 0.08;
    poni.poni1 = 0.002;
    poni.poni2 = 0.004;
    poni.rot1 = 0.017;
    let input: Vec<f32> = (0..pixels * 2)
        .map(|i| ((i * 271 % 293) as f32 - 14.0) / 7.0)
        .collect();
    let reference_input: Vec<f64> = input.iter().copied().map(f64::from).collect();
    for method in [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::PyfaiFull,
    ] {
        for all_masked in [false, true] {
            let mask = Mask::from_vec(
                (0..pixels)
                    .map(|i| u8::from(all_masked || i % 19 == 0))
                    .collect(),
                shape.0,
                shape.1,
            )?;
            let integrator = Integrator::new(Geometry::new(shape.0, shape.1, poni.clone()))
                .with_method(method)
                .with_mask(mask)
                .with_polarization_factor(0.72)
                .with_fast_math(false);
            let plan = cuda.prepare(&integrator, shape, bins)?;
            assert_eq!(plan.entries == 0, all_masked);
            for errors in [false, true] {
                let stride = 1 + usize::from(errors);
                let expected: Vec<_> = reference_input
                    .chunks_exact(pixels)
                    .map(|frame| {
                        if errors {
                            let (_, signal, sigma) = integrator.integrate1d_with_errors_from_slice(
                                frame,
                                shape.0,
                                shape.1,
                                bins as usize,
                            );
                            (signal, sigma)
                        } else {
                            let (_, signal) = integrator.integrate1d_from_slice(
                                frame,
                                shape.0,
                                shape.1,
                                bins as usize,
                            );
                            (signal, Vec::new())
                        }
                    })
                    .collect();
                let mut batch = plan.batch(2, errors)?;
                batch.upload(&input)?;
                let mut output = vec![0.0; 2 * bins as usize * stride];
                for layout in Layout::ALL {
                    for mode in Accumulation::ALL {
                        batch.run(&plan.kernel(mode, layout, errors)?)?;
                        batch.download(&mut output)?;
                        cuda.stream.synchronize()?;
                        for (frame, (signal, sigma)) in
                            output.chunks_exact(bins as usize * stride).zip(&expected)
                        {
                            for (bin, row) in frame.chunks_exact(stride).enumerate() {
                                let check = |value: f64, reference: f64| {
                                    if reference == 0.0 {
                                        assert_eq!(value, 0.0);
                                    } else {
                                        assert!(
                                            (value - reference).abs() <= 2e-6 * reference.abs(),
                                            "{method:?} {layout:?} {mode:?} bin {bin}: {value} vs {reference}"
                                        );
                                    }
                                };
                                check(row[0], signal[bin]);
                                if errors {
                                    check(row[1], sigma[bin]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let shape = (2, 2);
    let integrator = Integrator::new(Geometry::new(shape.0, shape.1, poni))
        .with_method(IntegrationMethod::Counting)
        .with_fast_math(false);
    let plan = cuda.prepare(&integrator, shape, 1)?;
    let mut batch = plan.batch(257, true)?;
    let mut input = vec![1.0; 257 * 4];
    input[256 * 4..].fill(7.0);
    batch.upload(&input)?;
    batch.run(&plan.kernel(Accumulation::Compensated, Layout::Warp, true)?)?;
    let mut output = vec![0.0; 257 * 2];
    batch.download(&mut output)?;
    cuda.stream.synchronize()?;
    let (_, intensity, sigma) = integrator.integrate1d_with_errors_from_slice(&[7.0; 4], 2, 2, 1);
    assert!((output[512] - intensity[0]).abs() < 1e-12);
    assert!((output[513] - sigma[0]).abs() < 1e-12);
    assert!(output[512] > output[510]);
    Ok(())
}
