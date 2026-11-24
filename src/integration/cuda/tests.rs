use super::*;
use crate::{geometry::Geometry, integration::MaskMode, mask::Mask, poni::Poni};

/// Device tests skip (pass after printing "Skipped CUDA test: <reason>")
/// when device 0 is unavailable. Setting `DIOPTRIN_REQUIRE_CUDA` to any value
/// other than the empty string or `0` (for example `1`) makes them fail
/// instead; unset, empty or `0` allows the skip. Use it on GPU nodes, so a
/// missing driver or NVRTC cannot pass silently.
#[cfg(feature = "cuda")]
pub(super) fn cuda_required() -> bool {
    std::env::var_os("DIOPTRIN_REQUIRE_CUDA").is_some_and(|value| !value.is_empty() && value != "0")
}

fn poni() -> Poni {
    let mut poni = Poni::new();
    poni.pixel1 = 1.0e-4;
    poni.pixel2 = 1.2e-4;
    poni.distance = 0.08;
    poni.poni1 = 0.002;
    poni.poni2 = 0.004;
    poni.rot1 = 0.017;
    poni.wavelength = 1.0e-10;
    poni
}

fn integrator(shape: (usize, usize), method: IntegrationMethod) -> Integrator {
    Integrator::new(Geometry::new(shape.0, shape.1, poni())).with_method(method)
}

fn invalid(result: Result<Cuda1dPlan, CudaError>) -> bool {
    matches!(result, Err(CudaError::InvalidInput(_)))
}

fn unsupported(result: Result<Cuda1dPlan, CudaError>) -> bool {
    matches!(result, Err(CudaError::Unsupported(_)))
}

#[test]
fn configuration_errors_precede_the_driver() {
    let shape = (6, 7);
    let options = Cuda1dOptions::new();
    let counting = integrator(shape, IntegrationMethod::Counting);
    assert!(invalid(counting.prepare_cuda1d(shape, 0, &options)));
    assert!(invalid(counting.prepare_cuda1d(
        shape,
        i32::MAX as usize + 1,
        &options
    )));
    assert!(invalid(counting.prepare_cuda1d((0, 7), 10, &options)));
    assert!(invalid(counting.prepare_cuda1d((6, 0), 10, &options)));
    assert!(invalid(counting.prepare_cuda1d(
        (6, 7),
        10,
        &Cuda1dOptions::new().with_max_batch_bytes(0)
    )));

    let dynamic =
        integrator(shape, IntegrationMethod::PixelSplit).with_mask_mode(MaskMode::Dynamic);
    assert!(unsupported(dynamic.prepare_cuda1d(shape, 10, &options)));
    // Step 1 (sizes) is checked before step 2 (mask mode).
    assert!(invalid(dynamic.prepare_cuda1d(shape, 0, &options)));
    assert!(unsupported(
        integrator(shape, IntegrationMethod::Histogram).prepare_cuda1d(shape, 10, &options)
    ));

    let mask = Mask::from_vec(vec![0; 6 * 7], 6, 7).unwrap();
    let masked = integrator(shape, IntegrationMethod::PyfaiFull).with_mask(mask);
    let mismatch = masked.prepare_cuda1d((7, 6), 10, &options);
    assert!(
        invalid(mismatch),
        "mask/shape mismatch must be InvalidInput"
    );
}

#[test]
fn valid_configuration_without_a_device() {
    let shape = (6, 7);
    let result = integrator(shape, IntegrationMethod::PyfaiFull).prepare_cuda1d(
        shape,
        10,
        &Cuda1dOptions::new(),
    );
    if cfg!(feature = "cuda") {
        if cuda_info(0).available {
            assert!(result.is_ok(), "{result:?}");
        } else {
            assert!(
                matches!(result, Err(CudaError::Unavailable(_))),
                "{result:?}"
            );
        }
    } else {
        assert!(matches!(result, Err(CudaError::NotCompiled)), "{result:?}");
    }
}

#[test]
fn info_reports_compilation_and_reason() {
    let info = cuda_info(0);
    assert_eq!(info.compiled, cfg!(feature = "cuda"));
    assert_eq!(info.device, 0);
    assert_eq!(info.available, info.reason.is_none());
    if !cfg!(feature = "cuda") {
        assert!(!info.available);
        assert!(info.reason.as_deref().unwrap().contains("cuda"));
        assert_eq!(info.bindings_version, None);
    } else {
        assert_eq!(info.bindings_version, Some((13, 4)));
    }
    let missing = cuda_info(1_000_000);
    assert!(!missing.available);
    assert_eq!(missing.device, 1_000_000);
    assert!(missing.reason.is_some());
}

#[test]
fn public_types_are_thread_safe() {
    fn check<T: Send + Sync + 'static>() {}
    check::<CudaError>();
    check::<Cuda1dPlan>();
    check::<CudaInfo>();
    check::<Cuda1dOptions>();
    let error: Box<dyn std::error::Error + Send + Sync> = Box::new(CudaError::NotCompiled);
    assert!(error.to_string().contains("cuda"));
}

#[test]
fn options_and_precision_names() {
    let options = Cuda1dOptions::new();
    assert_eq!(options.precision(), CudaPrecision::F32Compensated);
    assert_eq!(options.device(), 0);
    assert_eq!(options.max_batch_bytes(), None);
    let options = options
        .with_precision(CudaPrecision::F64)
        .with_device(2)
        .with_max_batch_bytes(1 << 20);
    assert_eq!(
        (
            options.precision(),
            options.device(),
            options.max_batch_bytes()
        ),
        (CudaPrecision::F64, 2, Some(1 << 20))
    );
    for precision in [
        CudaPrecision::F32,
        CudaPrecision::F32Compensated,
        CudaPrecision::F64,
    ] {
        assert_eq!(precision.as_str().parse::<CudaPrecision>(), Ok(precision));
        assert_eq!(precision.to_string(), precision.as_str());
    }
    assert!(matches!(
        "f16".parse::<CudaPrecision>(),
        Err(CudaError::InvalidInput(_))
    ));
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::integration::RadialUnit;
    use std::{error::Error, sync::Arc};

    type TestResult = Result<(), Box<dyn Error>>;

    const PRECISIONS: [CudaPrecision; 3] = [
        CudaPrecision::F32,
        CudaPrecision::F32Compensated,
        CudaPrecision::F64,
    ];
    const METHODS: [IntegrationMethod; 4] = [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::PyfaiFull,
        IntegrationMethod::Supersampled { n_ss: 3 },
    ];

    /// Whether device 0 is usable; otherwise logs a skip, or fails when
    /// `DIOPTRIN_REQUIRE_CUDA` requires CUDA (see [`cuda_required`]).
    fn device_available() -> bool {
        let info = cuda_info(0);
        if !info.available {
            let reason = info.reason.unwrap_or_default();
            assert!(!cuda_required(), "CUDA required but unavailable: {reason}");
            eprintln!("Skipped CUDA test: {reason}");
        }
        info.available
    }

    fn options(precision: CudaPrecision) -> Cuda1dOptions {
        Cuda1dOptions::new().with_precision(precision)
    }

    /// CPU reference `(radial, intensity, sigma)` of one f64 frame.
    fn reference(
        integrator: &Integrator,
        frame: &[f64],
        shape: (usize, usize),
        bins: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        integrator.integrate1d_with_errors_from_slice(frame, shape.0, shape.1, bins)
    }

    /// CPU integration of `|frame|`: per bin `Σ w·|s| / den`, the scale of
    /// the rounding error of any summation order of `Σ w·s / den` (and an
    /// upper bound of `|Σ w·s / den|`). Zero exactly where the CPU result is
    /// zero for lack of entries or denominator.
    fn magnitude(
        integrator: &Integrator,
        frame: &[f64],
        shape: (usize, usize),
        bins: usize,
    ) -> Vec<f64> {
        let absolute: Vec<f64> = frame.iter().map(|x| x.abs()).collect();
        let (_, scale) = integrator.integrate1d_from_slice(&absolute, shape.0, shape.1, bins);
        scale.iter().map(|x| x.abs()).collect()
    }

    /// Relative tolerance against the CPU, applied to [`magnitude`]. `f64`
    /// differs only by summation order (`≲ n·2⁻⁵³` for `n` entries per bin).
    /// `f32` storage adds the rounding of each weight (`≤ 2⁻²⁴`), and plain
    /// `f32` accumulation `≲ (n/128 + 7)·2⁻²⁴` (128 threads per bin, then a
    /// tree); 2e-6 leaves ample room for the row lengths used here.
    fn tolerance(precision: CudaPrecision) -> f64 {
        match precision {
            CudaPrecision::F64 => 1e-12,
            _ => 2e-6,
        }
    }

    /// `|actual - expected| <= tolerance(precision) × magnitude` per bin; a
    /// bin of magnitude 0 must match exactly.
    fn assert_close(
        actual: &[f64],
        expected: &[f64],
        magnitude: &[f64],
        precision: CudaPrecision,
        what: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        assert_eq!(magnitude.len(), expected.len(), "{what}");
        let relative = tolerance(precision);
        for (bin, ((&value, &reference), &scale)) in
            actual.iter().zip(expected).zip(magnitude).enumerate()
        {
            assert!(
                (value - reference).abs() <= relative * scale,
                "{what} bin {bin}: {value} vs {reference} (magnitude {scale})"
            );
        }
    }

    /// The input the device integrates for `precision`: `f64` input is
    /// rounded to `f32` unless the precision is `F64`.
    fn device_input(input: &[f64], precision: CudaPrecision) -> Vec<f64> {
        match precision {
            CudaPrecision::F64 => input.to_vec(),
            _ => input.iter().map(|&x| f64::from(x as f32)).collect(),
        }
    }

    /// Checks one device result against the CPU on `input`, the values the
    /// device integrated (see [`device_input`]).
    #[allow(clippy::too_many_arguments)]
    fn check_result(
        result: &Fixed1dResult,
        cpu: &Integrator,
        input: &[f64],
        shape: (usize, usize),
        bins: usize,
        precision: CudaPrecision,
        errors: bool,
        what: &str,
    ) {
        let (radial, intensity, sigma) = reference(cpu, input, shape, bins);
        assert_eq!(result.radial, radial, "{what}: radial");
        let scale = magnitude(cpu, input, shape, bins);
        assert_close(&result.intensity, &intensity, &scale, precision, what);
        match &result.errors {
            // The variance terms are not negative: sigma is its own scale.
            Some(actual) => {
                assert!(errors, "{what}: unrequested errors");
                assert_close(actual, &sigma, &sigma, precision, what);
            }
            None => assert!(!errors, "{what}: missing errors"),
        }
    }

    /// Asserts that every result has the same bits as `expected`, and as many.
    fn assert_same(actual: &[Fixed1dResult], expected: &[Fixed1dResult], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: result count");
        for (frame, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(same_bits(a, b), "{what}: frame {frame} differs");
        }
    }

    fn same_bits(a: &Fixed1dResult, b: &Fixed1dResult) -> bool {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        bits(&a.radial) == bits(&b.radial)
            && bits(&a.intensity) == bits(&b.intensity)
            && a.errors.as_deref().map(bits) == b.errors.as_deref().map(bits)
    }

    /// Bin values of a host table gathered the way the kernels do:
    /// `(Σ w·s / den, sqrt(max(Σ w²·max(s, 0), 0)) / |den|)`, 0 for a zero
    /// denominator. `weight(k)` and `pixel(p)` give the values the device
    /// reads, widened to `f64`; for `f32` storage their products are exact in
    /// `f64`, which models the compensated sums up to `f64` rounding.
    fn host_gather(
        table: &HostTable<'_>,
        columns: &[u32],
        weight: impl Fn(usize) -> f64,
        pixel: impl Fn(usize) -> f64,
    ) -> (Vec<f64>, Vec<f64>) {
        (0..table.bins as usize)
            .map(|bin| {
                let (mut signal, mut variance) = (0.0, 0.0);
                let entries = table.rows[bin] as usize..table.rows[bin + 1] as usize;
                for (k, &column) in entries.clone().zip(&columns[entries]) {
                    let (w, s) = (weight(k), pixel(column as usize));
                    signal += w * s;
                    variance += w * w * s.max(0.0);
                }
                let den = table.denominator[bin];
                if den == 0.0 {
                    (0.0, 0.0)
                } else {
                    (signal / den, variance.max(0.0).sqrt() / den.abs())
                }
            })
            .unzip()
    }

    /// Checks what `Cuda::upload` copies for `table` (dummy entries,
    /// `weighted` flag), gathers it in `f64` and with the `f32` storage
    /// encoding, and compares both with the CPU results on `frame` within
    /// the device tolerances: `f64` as is, `f32` storage on the `f32`-rounded
    /// frame.
    fn check_host_table(
        table: &HostTable<'_>,
        cpu: &Integrator,
        frame: &[f64],
        shape: (usize, usize),
        what: &str,
    ) {
        let (pixels, bins) = (shape.0 * shape.1, table.bins as usize);
        assert_eq!(table.pixels as usize, pixels, "{what}");
        assert_eq!(table.rows.len(), bins + 1, "{what}");
        assert_eq!(table.rows[0], 0, "{what}");
        assert!(table.rows.windows(2).all(|r| r[0] <= r[1]), "{what}");
        assert_eq!(*table.rows.last().unwrap(), table.entries, "{what}");
        assert_eq!(table.columns.len(), table.entries as usize, "{what}");
        assert!(
            table.columns.iter().all(|&p| (p as usize) < pixels),
            "{what}"
        );
        assert_eq!(table.denominator.len(), bins, "{what}");
        if let Some(weights) = table.weights {
            assert_eq!(weights.len(), table.entries as usize, "{what}");
        }
        // The arrays `Cuda::upload` copies, for both storage types.
        let encoded = table.weights.map(encode_f32);
        let (columns, wide, weighted) = device_arrays(table, table.weights, &[1.0_f64]);
        let (narrow_columns, narrow, narrow_weighted) =
            device_arrays(table, encoded.as_deref(), &[1.0_f32]);
        assert_eq!(columns, narrow_columns, "{what}");
        assert_eq!(weighted, u32::from(table.weights.is_some()), "{what}");
        assert_eq!(narrow_weighted, weighted, "{what}");
        if table.entries == 0 {
            assert_eq!(columns, &[0], "{what}: dummy column");
        } else {
            assert_eq!(columns, table.columns.as_ref(), "{what}");
        }
        match table.weights {
            Some(weights) if !weights.is_empty() => {
                assert_eq!(wide, weights, "{what}");
                let rounded: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
                assert_eq!(narrow, rounded.as_slice(), "{what}");
            }
            _ => {
                assert_eq!(wide, &[1.0], "{what}: dummy weight");
                assert_eq!(narrow, &[1.0], "{what}: dummy weight");
            }
        }
        // The kernels read weights[k] only when `weighted` is set.
        let weighted = weighted != 0;
        let (intensity, sigma) = host_gather(
            table,
            columns,
            |k| if weighted { wide[k] } else { 1.0 },
            |p| frame[p],
        );
        let (radial, expected, expected_sigma) = reference(cpu, frame, shape, bins);
        assert_eq!(table.radial, radial, "{what}");
        let scale = magnitude(cpu, frame, shape, bins);
        let f64_what = format!("{what} f64");
        let precision = CudaPrecision::F64;
        assert_close(&intensity, &expected, &scale, precision, &f64_what);
        assert_close(
            &sigma,
            &expected_sigma,
            &expected_sigma,
            precision,
            &f64_what,
        );

        let rounded = device_input(frame, CudaPrecision::F32Compensated);
        let (intensity, sigma) = host_gather(
            table,
            narrow_columns,
            |k| f64::from(if weighted { narrow[k] } else { 1.0 }),
            |p| rounded[p],
        );
        let (_, expected, expected_sigma) = reference(cpu, &rounded, shape, bins);
        let scale = magnitude(cpu, &rounded, shape, bins);
        let f32_what = format!("{what} f32 storage");
        let precision = CudaPrecision::F32Compensated;
        assert_close(&intensity, &expected, &scale, precision, &f32_what);
        assert_close(
            &sigma,
            &expected_sigma,
            &expected_sigma,
            precision,
            &f32_what,
        );
    }

    /// Runs without a device: the host table and the arrays that are
    /// uploaded, gathered in `f64` the way the kernels do, reproduce CPU
    /// strict integration; with the `f32` weights uploaded for `f32` storage
    /// they agree within the `f32` tolerance. Covers no mask, an all-zero
    /// mask (the unmasked table), a partial mask and a full mask (the empty
    /// table with dummy entries).
    #[test]
    fn host_table_gathered_in_f64_matches_cpu() -> TestResult {
        let shape = (37, 41);
        let pixels = shape.0 * shape.1;
        let bins = 53;
        let frame: Vec<f64> = (0..pixels)
            .map(|i| ((i * 271 % 293) as f64 - 14.0) / 7.0)
            .collect();
        type MaskValue = fn(usize) -> u8;
        let masks: [(&str, Option<MaskValue>); 4] = [
            ("no mask", None),
            ("zero mask", Some(|_| 0)),
            ("partial mask", Some(|i| u8::from(i % 7 == 0))),
            ("full mask", Some(|_| 1)),
        ];
        for unit in [RadialUnit::TthDeg, RadialUnit::QAngstrom] {
            for method in METHODS {
                for range in [None, Some(0.2)] {
                    for (name, mask) in masks {
                        let mut cpu = integrator(shape, method)
                            .with_radial_unit(unit)
                            .with_polarization_factor(0.6)
                            .with_fast_math(false);
                        if let Some(fraction) = range {
                            let (radial, _, _) = reference(&cpu, &frame, shape, bins);
                            let span = radial[bins - 1] - radial[0];
                            cpu.set_radial_range(Some((
                                radial[0] + fraction * span,
                                radial[bins - 1] - fraction * span,
                            )));
                        }
                        if let Some(mask) = mask {
                            cpu.set_mask(Mask::from_vec(
                                (0..pixels).map(mask).collect(),
                                shape.0,
                                shape.1,
                            )?);
                        }
                        check_configuration(&cpu, shape, bins, None)?;
                        let what = format!("{unit:?} {method:?} {range:?} {name}");
                        with_table(
                            &cpu,
                            shape,
                            bins,
                            crate::integration::InputOrder::Detector,
                            |table| {
                                if matches!(method, IntegrationMethod::Counting) {
                                    assert!(table.weights.is_none(), "{what}: counting weights");
                                }
                                match name {
                                    "full mask" => assert_eq!(table.entries, 0, "{what}"),
                                    _ => assert!(table.entries > 0, "{what}"),
                                }
                                if name == "partial mask" {
                                    assert!(
                                        table.columns.iter().all(|&p| p % 7 != 0),
                                        "{what}: masked pixel in the table"
                                    );
                                }
                                check_host_table(table, &cpu, &frame, shape, &what);
                                Ok(())
                            },
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Runs without a device: the uploaded table against an integrator
    /// configured as the comparison harness configures it (default
    /// `fast_math`, unit 2th_deg, polarisation 0.965, explicit radial range,
    /// no mask). The CPU sums in reassociated order with fast math while the
    /// table's denominators are strict; on non-negative data both are within
    /// `≲ n·2⁻⁵³` (n ≤ a few hundred entries per bin) of the exact sums, so
    /// 1e-12 relative to the magnitude is a generous bound.
    #[test]
    fn host_table_matches_default_integrator() -> TestResult {
        let shape = (43, 59);
        let pixels = shape.0 * shape.1;
        let bins = 97;
        let frame: Vec<f64> = (0..pixels)
            .map(|i| (i * 271 % 293) as f64 * 0.75 + 3.0)
            .collect();
        for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
            let full = integrator(shape, method)
                .with_radial_unit(RadialUnit::TthDeg)
                .with_polarization_factor(0.965);
            assert!(full.fast_math(), "fast_math must be the default");
            let (radial, _, _) = reference(&full, &frame, shape, bins);
            let span = radial[bins - 1] - radial[0];
            let cpu =
                full.with_radial_range(radial[0] + 0.05 * span, radial[bins - 1] - 0.1 * span);
            check_configuration(&cpu, shape, bins, None)?;
            let what = format!("default {method:?}");
            let (det_int, det_err) = with_table(
                &cpu,
                shape,
                bins,
                crate::integration::InputOrder::Detector,
                |table| {
                    assert!(table.weights.is_some(), "{what}: splitting has weights");
                    check_host_table(table, &cpu, &frame, shape, &what);
                    let w = table.weights.unwrap();
                    Ok(host_gather(
                        table,
                        table.columns.as_ref(),
                        |k| w[k],
                        |p| frame[p],
                    ))
                },
            )?;
            let flipped: Vec<f64> = frame
                .chunks_exact(shape.1)
                .rev()
                .flatten()
                .copied()
                .collect();
            with_table(
                &cpu,
                shape,
                bins,
                crate::integration::InputOrder::FlipVertical,
                |table| {
                    let w = table.weights.unwrap();
                    let (flip_int, flip_err) =
                        host_gather(table, table.columns.as_ref(), |k| w[k], |p| flipped[p]);
                    assert_eq!(flip_int, det_int, "{what}: FlipVertical intensity mismatch");
                    assert_eq!(flip_err, det_err, "{what}: FlipVertical sigma mismatch");
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    #[test]
    fn methods_precisions_and_inputs_match_cpu() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (61, 113);
        let pixels = shape.0 * shape.1;
        let bins = 137;
        let narrow: Vec<f32> = (0..pixels * 2)
            .map(|i| ((i * 271 % 293) as f32 - 14.0) / 7.0)
            .collect();
        let wide: Vec<f64> = (0..pixels * 2)
            .map(|i| ((i * 271 % 293) as f64 - 14.0) / 7.0)
            .collect();
        let narrow_as_f64: Vec<f64> = narrow.iter().map(|&x| f64::from(x)).collect();
        let wide_rounded: Vec<f64> = wide.iter().map(|&x| f64::from(x as f32)).collect();
        assert_ne!(narrow_as_f64, wide);
        let mask = || {
            Mask::from_vec(
                (0..pixels).map(|i| u8::from(i % 19 == 0)).collect(),
                shape.0,
                shape.1,
            )
        };
        for unit in [RadialUnit::TthDeg, RadialUnit::QAngstrom] {
            for method in METHODS {
                let make = |range: Option<(f64, f64)>| -> Result<Integrator, String> {
                    let mut integrator = integrator(shape, method)
                        .with_radial_unit(unit)
                        .with_mask(mask()?)
                        .with_polarization_factor(0.72)
                        .with_fast_math(false);
                    integrator.set_radial_range(range);
                    Ok(integrator)
                };
                let (radial, _, _) = reference(&make(None)?, &wide, shape, bins);
                let span = radial[bins - 1] - radial[0];
                let range = (radial[0] + 0.1 * span, radial[bins - 1] - 0.1 * span);
                let cpu = make(Some(range))?;
                for precision in PRECISIONS {
                    let plan = cpu.prepare_cuda1d(shape, bins, &options(precision))?;
                    assert_eq!(plan.precision(), precision);
                    assert_eq!((plan.shape(), plan.n_points()), (shape, bins));
                    for errors in [false, true] {
                        for f32_input in [true, false] {
                            let (results, expected_input) = if f32_input {
                                (plan.integrate_stack(&narrow, errors)?, &narrow_as_f64)
                            } else if precision == CudaPrecision::F64 {
                                (plan.integrate_stack(&wide, errors)?, &wide)
                            } else {
                                (plan.integrate_stack(&wide, errors)?, &wide_rounded)
                            };
                            assert_eq!(results.len(), 2);
                            for (frame, result) in results.iter().enumerate() {
                                let what = format!(
                                    "{unit:?} {method:?} {precision:?} errors={errors} f32_input={f32_input} frame {frame}"
                                );
                                let input = &expected_input[frame * pixels..(frame + 1) * pixels];
                                assert_eq!(plan.radial(), result.radial.as_slice(), "{what}");
                                check_result(
                                    result, &cpu, input, shape, bins, precision, errors, &what,
                                );
                                let single = if f32_input {
                                    plan.integrate_frame(
                                        &narrow[frame * pixels..(frame + 1) * pixels],
                                        errors,
                                    )?
                                } else {
                                    plan.integrate_frame(
                                        &wide[frame * pixels..(frame + 1) * pixels],
                                        errors,
                                    )?
                                };
                                assert!(same_bits(&single, result), "{what}: batch differs");
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Bytes per frame of the chunk budget (see `with_max_batch_bytes`).
    fn frame_bytes(precision: CudaPrecision, pixels: usize, bins: usize) -> usize {
        let storage = match precision {
            CudaPrecision::F64 => 8,
            _ => 4 + 8,
        };
        pixels * storage + bins * 16
    }

    #[test]
    fn chunks_and_launch_groups_give_identical_results() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (2, 2);
        let frames = 300;
        let cpu = integrator(shape, IntegrationMethod::Counting)
            .with_polarization_factor(0.9)
            .with_fast_math(false);
        let stack: Vec<f64> = (0..frames * 4)
            .map(|i| 1.0 + (i / 4) as f64 * 0.5 + (i % 4) as f64 * 0.125)
            .collect();
        let narrow: Vec<f32> = stack.iter().map(|&x| x as f32).collect();
        for precision in PRECISIONS {
            let whole = cpu.prepare_cuda1d(shape, 1, &options(precision))?;
            assert!(whole.max_batch_frames() >= frames);
            let budget = frame_bytes(precision, 4, 1) * 7 + 3;
            let chunked =
                cpu.prepare_cuda1d(shape, 1, &options(precision).with_max_batch_bytes(budget))?;
            assert_eq!(chunked.max_batch_frames(), 7);
            for errors in [false, true] {
                let what = format!("{precision:?} errors={errors}");
                let expected = whole.integrate_stack(&stack, errors)?;
                assert_eq!(expected.len(), frames);
                assert_same(&chunked.integrate_stack(&stack, errors)?, &expected, &what);
                let narrow_whole = whole.integrate_stack(&narrow, errors)?;
                assert_eq!(narrow_whole.len(), frames);
                let narrow_chunked = chunked.integrate_stack(&narrow, errors)?;
                assert_same(&narrow_chunked, &narrow_whole, &what);
                for (frame, result) in expected.iter().enumerate() {
                    let input = device_input(&stack[frame * 4..(frame + 1) * 4], precision);
                    let what = format!("{what} frame {frame}");
                    check_result(result, &cpu, &input, shape, 1, precision, errors, &what);
                }
            }
        }
        Ok(())
    }

    /// More than one launch of 256 frames in one chunk, several bins and
    /// different data in every frame: a frame or bin mix-up across launches
    /// shows as a mismatch with the CPU.
    #[test]
    fn many_distinct_frames_match_cpu() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (16, 20);
        let pixels = shape.0 * shape.1;
        let (bins, frames) = (7, 300);
        let cpu = integrator(shape, IntegrationMethod::PyfaiFull)
            .with_polarization_factor(0.8)
            .with_fast_math(false);
        let stack: Vec<f64> = (0..frames * pixels)
            .map(|i| ((i * 271 + (i / pixels) * 17) % 293) as f64 / 4.0 + 1.0)
            .collect();
        let narrow: Vec<f32> = stack.iter().map(|&x| x as f32).collect();
        for precision in PRECISIONS {
            let plan = cpu.prepare_cuda1d(shape, bins, &options(precision))?;
            assert!(plan.max_batch_frames() >= frames);
            let results = plan.integrate_stack(&stack, true)?;
            assert_eq!(results.len(), frames);
            let list: Vec<&[f64]> = stack.chunks_exact(pixels).collect();
            assert_same(&plan.integrate_frames(&list, true)?, &results, "frame list");
            for (frame, result) in results.iter().enumerate() {
                let input = device_input(&stack[frame * pixels..(frame + 1) * pixels], precision);
                let what = format!("{precision:?} frame {frame}");
                check_result(result, &cpu, &input, shape, bins, precision, true, &what);
            }
            let narrow_results = plan.integrate_stack(&narrow, false)?;
            assert_eq!(narrow_results.len(), frames);
            for (frame, result) in narrow_results.iter().enumerate().step_by(13) {
                let input: Vec<f64> = narrow[frame * pixels..(frame + 1) * pixels]
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect();
                let what = format!("{precision:?} f32 frame {frame}");
                check_result(result, &cpu, &input, shape, bins, precision, false, &what);
            }
        }
        Ok(())
    }

    /// Frame counts that are exact multiples of the chunk size (7 and 256)
    /// and of the launch size (256, 512): every count and result is checked.
    #[test]
    fn exact_multiples_of_chunks_and_launches() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (3, 5);
        let pixels = shape.0 * shape.1;
        let (bins, frames) = (2, 512);
        let cpu = integrator(shape, IntegrationMethod::PixelSplit)
            .with_polarization_factor(0.8)
            .with_fast_math(false);
        let stack: Vec<f64> = (0..frames * pixels)
            .map(|i| 1.0 + (i / pixels) as f64 * 0.25 + (i % pixels) as f64 * 0.5)
            .collect();
        for precision in PRECISIONS {
            let whole = cpu.prepare_cuda1d(shape, bins, &options(precision))?;
            assert!(whole.max_batch_frames() >= frames);
            // One chunk, two launches of 256 frames.
            let expected = whole.integrate_stack(&stack, true)?;
            assert_eq!(expected.len(), frames);
            for (frame, result) in expected.iter().enumerate() {
                let input = device_input(&stack[frame * pixels..(frame + 1) * pixels], precision);
                let what = format!("{precision:?} frame {frame}");
                check_result(result, &cpu, &input, shape, bins, precision, true, &what);
            }
            for chunk in [7, 256] {
                let budget = frame_bytes(precision, pixels, bins) * chunk + 3;
                let chunked = cpu.prepare_cuda1d(
                    shape,
                    bins,
                    &options(precision).with_max_batch_bytes(budget),
                )?;
                assert_eq!(chunked.max_batch_frames(), chunk);
                for count in [chunk, 2 * chunk, frames] {
                    let what = format!("{precision:?} chunk {chunk} count {count}");
                    let part = &stack[..count * pixels];
                    let actual = chunked.integrate_stack(part, true)?;
                    assert_same(&actual, &expected[..count], &what);
                    let list: Vec<&[f64]> = part.chunks_exact(pixels).collect();
                    let actual = chunked.integrate_frames(&list, true)?;
                    assert_same(&actual, &expected[..count], &what);
                }
            }
        }
        Ok(())
    }

    /// No mask and the default `fast_math`, as the comparison harness
    /// configures the integrator; the CPU's reassociated sums stay far
    /// inside the tolerances (see `host_table_matches_default_integrator`).
    #[test]
    fn default_integrator_without_mask_matches_cpu() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (61, 113);
        let pixels = shape.0 * shape.1;
        let bins = 137;
        let stack: Vec<f64> = (0..2 * pixels)
            .map(|i| (i * 271 % 293) as f64 * 0.75 + 3.0)
            .collect();
        let narrow: Vec<f32> = stack.iter().map(|&x| x as f32).collect();
        for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
            let full = integrator(shape, method)
                .with_radial_unit(RadialUnit::TthDeg)
                .with_polarization_factor(0.965);
            assert!(full.fast_math());
            let (radial, _, _) = reference(&full, &stack[..pixels], shape, bins);
            let span = radial[bins - 1] - radial[0];
            let cpu =
                full.with_radial_range(radial[0] + 0.05 * span, radial[bins - 1] - 0.1 * span);
            for precision in PRECISIONS {
                let plan = cpu.prepare_cuda1d(shape, bins, &options(precision))?;
                for errors in [false, true] {
                    let wide = plan.integrate_stack(&stack, errors)?;
                    let single = plan.integrate_stack(&narrow, errors)?;
                    assert_eq!((wide.len(), single.len()), (2, 2));
                    for frame in 0..2 {
                        let values = frame * pixels..(frame + 1) * pixels;
                        let what =
                            format!("{method:?} {precision:?} errors={errors} frame {frame}");
                        let input = device_input(&stack[values.clone()], precision);
                        let result = &wide[frame];
                        check_result(result, &cpu, &input, shape, bins, precision, errors, &what);
                        let input: Vec<f64> =
                            narrow[values].iter().map(|&x| f64::from(x)).collect();
                        let what = format!("{what} f32 input");
                        let result = &single[frame];
                        check_result(result, &cpu, &input, shape, bins, precision, errors, &what);
                    }
                }
            }
        }
        Ok(())
    }

    /// A chunk larger than any device (about 1 TB of `f32` frames) fails as
    /// `CudaError::Device` when its scratch is allocated, and the plan stays
    /// usable. The frames are one 1 MB frame referenced a million times, so
    /// the host needs only about 32 MB; an explicit budget is not checked
    /// against device memory, so `prepare_cuda1d` accepts it.
    #[test]
    fn oversized_chunk_is_a_device_error_and_the_plan_recovers() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (512, 512);
        let pixels = shape.0 * shape.1;
        let bins = 2;
        let cpu = integrator(shape, IntegrationMethod::Counting).with_fast_math(false);
        let plan = cpu.prepare_cuda1d(
            shape,
            bins,
            &Cuda1dOptions::new().with_max_batch_bytes(usize::MAX),
        )?;
        let frame: Vec<f32> = (0..pixels).map(|i| (i % 23) as f32).collect();
        let before = plan.integrate_frame(&frame, true)?;
        let frames = vec![frame.as_slice(); 1_000_000];
        assert!(plan.max_batch_frames() >= frames.len());
        let result = plan.integrate_frames(&frames, false);
        match result {
            Err(CudaError::Device(_)) => {}
            Err(other) => panic!("expected a device error, got {other:?}"),
            Ok(results) => panic!("1 TB chunk succeeded with {} results", results.len()),
        }
        drop(frames);
        let after = plan.integrate_frame(&frame, true)?;
        assert!(same_bits(&before, &after), "result changed after the error");
        let input: Vec<f64> = frame.iter().map(|&x| f64::from(x)).collect();
        let precision = plan.precision();
        check_result(
            &after,
            &cpu,
            &input,
            shape,
            bins,
            precision,
            true,
            "after error",
        );
        let stack: Vec<f32> = frame.iter().chain(&frame).chain(&frame).copied().collect();
        let again = plan.integrate_stack(&stack, true)?;
        assert_eq!(again.len(), 3);
        for result in &again {
            assert!(same_bits(result, &before), "regrown scratch");
        }
        Ok(())
    }

    #[test]
    fn two_plans_run_concurrently() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let (shape_a, shape_b) = ((40, 50), (33, 47));
        let a = integrator(shape_a, IntegrationMethod::PixelSplit);
        let mask = Mask::from_vec(
            (0..33 * 47).map(|i| u8::from(i % 11 == 0)).collect(),
            shape_b.0,
            shape_b.1,
        )?;
        let b = integrator(shape_b, IntegrationMethod::PyfaiFull).with_mask(mask);
        let plan_a = a.prepare_cuda1d(shape_a, 64, &Cuda1dOptions::new())?;
        let plan_b = b.prepare_cuda1d(shape_b, 80, &options(CudaPrecision::F64))?;
        let stack_a: Vec<f32> = (0..40 * 50 * 5).map(|i| (i % 101) as f32).collect();
        let stack_b: Vec<f64> = (0..33 * 47 * 3).map(|i| (i % 89) as f64 * 0.5).collect();
        let expected_a = plan_a.integrate_stack(&stack_a, true)?;
        let expected_b = plan_b.integrate_stack(&stack_b, true)?;
        assert_eq!((expected_a.len(), expected_b.len()), (5, 3));
        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                for _ in 0..8 {
                    let actual = plan_a.integrate_stack(&stack_a, true).unwrap();
                    assert_same(&actual, &expected_a, "plan a");
                }
            });
            let second = scope.spawn(|| {
                for _ in 0..8 {
                    let actual = plan_b.integrate_stack(&stack_b, true).unwrap();
                    assert_same(&actual, &expected_b, "plan b");
                }
            });
            first.join().unwrap();
            second.join().unwrap();
        });
        Ok(())
    }

    /// Scratch shrinks and grows on one plan across input types: 300 `f64`
    /// frames, one `f32` frame, 300 other `f64` frames, each compared with a
    /// fresh plan.
    #[test]
    fn scratch_reuse_across_sizes_and_input_types() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (8, 9);
        let pixels = shape.0 * shape.1;
        let bins = 5;
        let cpu = integrator(shape, IntegrationMethod::PixelSplit).with_fast_math(false);
        let first: Vec<f64> = (0..300 * pixels).map(|i| (i % 97) as f64 * 0.5).collect();
        let narrow: Vec<f32> = (0..pixels).map(|i| (i % 13) as f32 + 0.25).collect();
        let last: Vec<f64> = (0..300 * pixels)
            .map(|i| (i % 83) as f64 * 1.5 + 2.0)
            .collect();
        for precision in PRECISIONS {
            let fresh = || cpu.prepare_cuda1d(shape, bins, &options(precision));
            let plan = fresh()?;
            for errors in [true, false] {
                let what = format!("{precision:?} errors={errors}");
                let a = plan.integrate_stack(&first, errors)?;
                let b = plan.integrate_frame(&narrow, errors)?;
                let c = plan.integrate_stack(&last, errors)?;
                assert_same(&a, &fresh()?.integrate_stack(&first, errors)?, &what);
                assert_same(&[b], &[fresh()?.integrate_frame(&narrow, errors)?], &what);
                assert_same(&c, &fresh()?.integrate_stack(&last, errors)?, &what);
                let input = device_input(&last[299 * pixels..], precision);
                check_result(&c[299], &cpu, &input, shape, bins, precision, errors, &what);
            }
        }
        Ok(())
    }

    #[test]
    fn fully_masked_table_gives_zeros() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (5, 9);
        for method in METHODS {
            let mask = Mask::from_vec(vec![1; 45], shape.0, shape.1)?;
            let cpu = integrator(shape, method).with_mask(mask);
            let frame: Vec<f64> = (0..45).map(|i| i as f64).collect();
            let (_, intensity, sigma) = reference(&cpu, &frame, shape, 11);
            assert!(intensity.iter().chain(&sigma).all(|&x| x == 0.0));
            for precision in PRECISIONS {
                let plan = cpu.prepare_cuda1d(shape, 11, &options(precision))?;
                assert_eq!(plan.table_entries(), 0);
                let result = plan.integrate_frame(&frame, true)?;
                assert_eq!(result.intensity, intensity);
                assert_eq!(result.errors.as_deref(), Some(sigma.as_slice()));
            }
        }
        Ok(())
    }

    #[test]
    fn plan_is_a_snapshot_of_the_mask() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (31, 29);
        let pixels = shape.0 * shape.1;
        let mut cpu = integrator(shape, IntegrationMethod::PixelSplit).with_mask(Mask::from_vec(
            vec![0; pixels],
            shape.0,
            shape.1,
        )?);
        let frame: Vec<f64> = (0..pixels).map(|i| (i % 17) as f64).collect();
        let plan = cpu.prepare_cuda1d(shape, 40, &Cuda1dOptions::new())?;
        let entries = plan.table_entries();
        let before = plan.integrate_frame(&frame, true)?;
        cpu.set_mask(Mask::from_vec(
            (0..pixels).map(|i| u8::from(i % 3 == 0)).collect(),
            shape.0,
            shape.1,
        )?);
        cpu.set_polarization_factor(Some(0.5));
        let after = plan.integrate_frame(&frame, true)?;
        assert_eq!(plan.table_entries(), entries);
        assert!(same_bits(&before, &after));
        let replanned = cpu.prepare_cuda1d(shape, 40, &Cuda1dOptions::new())?;
        assert!(replanned.table_entries() < entries);
        Ok(())
    }

    #[test]
    fn two_threads_share_one_plan() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (40, 50);
        let pixels = shape.0 * shape.1;
        let cpu = integrator(shape, IntegrationMethod::PyfaiFull);
        let plan = cpu.prepare_cuda1d(shape, 64, &Cuda1dOptions::new())?;
        let stack: Vec<f32> = (0..pixels * 5).map(|i| (i % 101) as f32).collect();
        let expected = plan.integrate_stack(&stack, true)?;
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        for _ in 0..8 {
                            let actual = plan.integrate_stack(&stack, true).unwrap();
                            assert!(expected.iter().zip(&actual).all(|(a, b)| same_bits(a, b)));
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Ok(())
    }

    /// Compares bin by bin: a NaN reference needs NaN, an infinite one the
    /// same infinity, a finite one closeness within the tolerance.
    fn assert_close_or_special(
        actual: &[f64],
        expected: &[f64],
        magnitude: &[f64],
        precision: CudaPrecision,
        what: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        for (bin, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
            let what = format!("{what} bin {bin}: {value} vs {reference}");
            if reference.is_nan() {
                assert!(value.is_nan(), "{what}");
            } else if reference.is_infinite() {
                assert_eq!(value, reference, "{what}");
            } else {
                let scale = magnitude[bin];
                assert!(scale.is_finite(), "{what}");
                assert!(
                    (value - reference).abs() <= tolerance(precision) * scale,
                    "{what}"
                );
            }
        }
    }

    /// `±inf` pixels give the CPU's `±inf` or NaN in intensity and errors
    /// (the variance ignores `-inf` as it does every negative pixel), with
    /// `f64` and `f32` input. Counting weights are 1, so `0 · inf` cannot
    /// occur on either side.
    #[test]
    fn infinite_pixels_stay_infinite() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (12, 15);
        let bins = 9;
        let cpu = integrator(shape, IntegrationMethod::Counting).with_fast_math(false);
        let mut frame: Vec<f64> = (0..180).map(|i| (i % 7) as f64).collect();
        frame[100] = f64::INFINITY;
        frame[37] = f64::NEG_INFINITY;
        let narrow: Vec<f32> = frame.iter().map(|&x| x as f32).collect();
        let (_, intensity, sigma) = reference(&cpu, &frame, shape, bins);
        assert!(intensity.iter().any(|x| x.is_infinite() || x.is_nan()));
        assert!(sigma.iter().any(|x| x.is_infinite()));
        let scale = magnitude(&cpu, &frame, shape, bins);
        for precision in PRECISIONS {
            let plan = cpu.prepare_cuda1d(shape, bins, &options(precision))?;
            for errors in [false, true] {
                let wide = plan.integrate_frame(&frame, errors)?;
                let single = plan.integrate_frame(&narrow, errors)?;
                for (result, input) in [(wide, "f64"), (single, "f32")] {
                    let what = format!("{precision:?} errors={errors} {input} input");
                    assert_close_or_special(
                        &result.intensity,
                        &intensity,
                        &scale,
                        precision,
                        &what,
                    );
                    match (&result.errors, errors) {
                        (Some(actual), true) => {
                            assert_close_or_special(actual, &sigma, &sigma, precision, &what)
                        }
                        (None, false) => {}
                        _ => panic!("{what}: errors {:?}", result.errors.is_some()),
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_frames_and_devices() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (4, 6);
        let cpu = integrator(shape, IntegrationMethod::Counting);
        let plan = cpu.prepare_cuda1d(shape, 5, &Cuda1dOptions::new())?;
        assert!(!plan.device_name().is_empty());
        assert_eq!(plan.device(), 0);
        assert!(matches!(
            plan.integrate_frame(&[1.0_f64; 23], false),
            Err(CudaError::InvalidInput(_))
        ));
        assert!(matches!(
            plan.integrate_frames(&[&[1.0_f32; 24][..], &[1.0_f32; 25][..]], false),
            Err(CudaError::InvalidInput(_))
        ));
        assert!(matches!(
            plan.integrate_stack(&[1.0_f64; 25], false),
            Err(CudaError::InvalidInput(_))
        ));
        assert!(plan.integrate_frames::<f64>(&[], true)?.is_empty());
        assert!(plan.integrate_stack::<f32>(&[], true)?.is_empty());
        assert!(matches!(
            cpu.prepare_cuda1d(shape, 5, &Cuda1dOptions::new().with_max_batch_bytes(10)),
            Err(CudaError::InvalidInput(_))
        ));
        assert!(matches!(
            cpu.prepare_cuda1d(shape, 5, &Cuda1dOptions::new().with_device(1_000_000)),
            Err(CudaError::Unavailable(_))
        ));
        Ok(())
    }

    // Prototype tests of the internal kernels, including the layouts and
    // accumulations the public plans do not use.

    #[test]
    fn native_integer_upload_matches_f64_exactly() -> TestResult {
        let Some(cuda) = Cuda::probe() else {
            return Ok(());
        };
        let shape = (3, 4);
        let mut poni = Poni::new();
        poni.pixel1 = 7.5e-5;
        poni.pixel2 = 7.5e-5;
        poni.distance = 0.1;
        let cpu = Integrator::new(Geometry::new(shape.0, shape.1, poni))
            .with_method(IntegrationMethod::PyfaiFull);
        let plan = cuda.prepare_f64(&cpu, shape, 23)?;
        let u32_data = [0, 1, 16_777_217, u32::MAX, 19, 7, 0, 0, 4, 5, 6, 30];
        let i32_data = [-1, 1, -16_777_217, i32::MAX, i32::MIN, 7, 0, 0, 4, 5, 6, 30];
        for layout in Layout::ALL {
            let mut wide = plan.batch(1, true)?;
            let mut unsigned = plan.batch_with_input(1, true, Arc::clone(&cuda.stream))?;
            let mut signed = plan.batch_with_input(1, true, Arc::clone(&cuda.stream))?;
            unsigned.upload(0, u32_data.as_slice())?;
            signed.upload(0, i32_data.as_slice())?;
            unsigned.run(&plan.kernel_u32(layout, true)?)?;
            signed.run(&plan.kernel_i32(layout, true)?)?;
            let mut actual = vec![0.; 46];
            let mut expected = vec![0.; 46];
            let kernel = plan.kernel(layout, true)?;
            for (pixels, integer_batch) in [
                (u32_data.map(f64::from), &unsigned.output),
                (i32_data.map(f64::from), &signed.output),
            ] {
                wide.upload(0, pixels.as_slice())?;
                wide.run(&kernel)?;
                wide.download(&mut expected)?;
                cuda.stream.memcpy_dtoh(integer_batch, &mut actual)?;
                cuda.stream.synchronize()?;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn fixed_mask_batches_match_strict_cpu() -> TestResult {
        let Some(cuda) = Cuda::probe() else {
            return Ok(());
        };
        let shape = (61, 113);
        let pixels = shape.0 * shape.1;
        let bins = 137;
        let poni = poni();
        let input: Vec<f32> = (0..pixels * 2)
            .map(|i| ((i * 271 % 293) as f32 - 14.0) / 7.0)
            .collect();
        let reference_input: Vec<f64> = input.iter().copied().map(f64::from).collect();
        let wide_input: Vec<f64> = (0..pixels * 2)
            .map(|i| ((i * 271 % 293) as f64 - 14.0) / 7.0)
            .collect();
        assert_ne!(reference_input, wide_input);
        for method in METHODS {
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
                let wide_plan = cuda.prepare_f64(&integrator, shape, bins)?;
                assert_eq!(plan.entries == 0, all_masked);
                for errors in [false, true] {
                    let stride = 1 + usize::from(errors);
                    let references = |input: &[f64]| {
                        input
                            .chunks_exact(pixels)
                            .map(|frame| {
                                let (_, signal, sigma) =
                                    reference(&integrator, frame, shape, bins as usize);
                                let scale = magnitude(&integrator, frame, shape, bins as usize);
                                (signal, sigma, scale)
                            })
                            .collect::<Vec<_>>()
                    };
                    let expected = references(&reference_input);
                    let wide_expected = references(&wide_input);
                    let mut batch = plan.batch(2, errors)?;
                    batch.upload(0, &input)?;
                    let mut wide_batch = wide_plan.batch(2, errors)?;
                    wide_batch.upload(0, &wide_input)?;
                    let mut output = vec![0.0; 2 * bins as usize * stride];
                    for layout in Layout::ALL {
                        for mode in Accumulation::ALL {
                            batch.run(&plan.kernel(mode, layout, errors)?)?;
                            batch.download(&mut output)?;
                            cuda.stream.synchronize()?;
                            for (frame, (signal, sigma, scale)) in
                                output.chunks_exact(bins as usize * stride).zip(&expected)
                            {
                                for (bin, row) in frame.chunks_exact(stride).enumerate() {
                                    // Tolerance scaled to the bin's magnitude
                                    // (see `tolerance`); sigma is its own.
                                    let check = |value: f64, reference: f64, scale: f64| {
                                        assert!(
                                            (value - reference).abs() <= 2e-6 * scale,
                                            "{method:?} {layout:?} {mode:?} bin {bin}: {value} vs {reference}"
                                        );
                                    };
                                    check(row[0], signal[bin], scale[bin]);
                                    if errors {
                                        check(row[1], sigma[bin], sigma[bin]);
                                    }
                                }
                            }
                        }
                        wide_batch.run(&wide_plan.kernel(layout, errors)?)?;
                        wide_batch.download(&mut output)?;
                        cuda.stream.synchronize()?;
                        for (frame, (signal, sigma, _)) in output
                            .chunks_exact(bins as usize * stride)
                            .zip(&wide_expected)
                        {
                            for (bin, row) in frame.chunks_exact(stride).enumerate() {
                                let check = |value: f64, reference: f64| {
                                    assert!(
                                        (value - reference).abs()
                                            <= 1e-12 * reference.abs().max(1.0),
                                        "{method:?} {layout:?} full f64 bin {bin}: {value} vs {reference}"
                                    );
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
        let shape = (2, 2);
        let integrator = Integrator::new(Geometry::new(shape.0, shape.1, poni))
            .with_method(IntegrationMethod::Counting)
            .with_fast_math(false);
        let plan = cuda.prepare(&integrator, shape, 1)?;
        let mut batch = plan.batch(257, true)?;
        let mut input = vec![1.0; 257 * 4];
        input[256 * 4..].fill(7.0);
        batch.upload(0, &input[..256 * 4])?;
        batch.upload(256 * 4, &input[256 * 4..])?;
        batch.run(&plan.kernel(Accumulation::Compensated, Layout::Warp, true)?)?;
        let mut output = vec![0.0; 257 * 2];
        batch.download(&mut output)?;
        cuda.stream.synchronize()?;
        let (_, intensity, sigma) =
            integrator.integrate1d_with_errors_from_slice(&[7.0; 4], 2, 2, 1);
        assert!((output[512] - intensity[0]).abs() < 1e-12);
        assert!((output[513] - sigma[0]).abs() < 1e-12);
        assert!(output[512] > output[510]);
        Ok(())
    }

    #[test]
    fn cuda2d_methods_precisions_and_azimuthal_range_match_cpu() -> TestResult {
        if !device_available() {
            return Ok(());
        }
        let shape = (32, 48);
        let pixels = shape.0 * shape.1;
        let (n_rad, n_az) = (20, 16);
        let frames: Vec<f32> = (0..pixels * 5)
            .map(|i| ((i * 271 % 293) as f32) + 5.0)
            .collect();
        let frames_f64: Vec<f64> = frames.iter().map(|&x| f64::from(x)).collect();
        let mask = || {
            Mask::from_vec(
                (0..pixels).map(|i| u8::from(i % 17 == 0)).collect(),
                shape.0,
                shape.1,
            )
        };
        for method in METHODS {
            for az_range in [None, Some((-1.2, 1.8))] {
                let cpu = integrator(shape, method)
                    .with_radial_unit(RadialUnit::TthDeg)
                    .with_mask(mask()?)
                    .with_polarization_factor(0.75)
                    .with_fast_math(false);
                for precision in PRECISIONS {
                    let plan =
                        cpu.prepare_cuda2d(shape, n_rad, n_az, az_range, &options(precision))?;
                    assert_eq!((plan.n_radial(), plan.n_azimuthal()), (n_rad, n_az));
                    let stack = plan.integrate_stack(&frames, true)?;
                    assert_eq!(stack.len(), 5);
                    for (f, res) in stack.iter().enumerate() {
                        let single =
                            plan.integrate_frame(&frames[f * pixels..(f + 1) * pixels], true)?;
                        assert_eq!(res.intensity, single.intensity);
                        assert_eq!(res.errors, single.errors);
                        let slice = &frames_f64[f * pixels..(f + 1) * pixels];
                        let (want_i, want_e) = match az_range {
                            None => {
                                let (i, e) = cpu.integrate2d_with_errors_from_slice(
                                    slice, shape.0, shape.1, n_rad, n_az,
                                );
                                (i, e)
                            }
                            Some(r) => {
                                let out = cpu.integrate2d_on_azimuthal_range_from_slice(
                                    slice, shape.0, shape.1, n_rad, n_az, r, true,
                                )?;
                                assert_eq!(res.radial.as_deref(), Some(out.radial.as_slice()));
                                assert_eq!(
                                    res.azimuthal.as_deref(),
                                    Some(out.azimuthal.as_slice())
                                );
                                (out.intensity, out.errors.unwrap())
                            }
                        };
                        let scale: Vec<f64> = want_i.iter().map(|x| x.abs().max(1.0)).collect();
                        let what = format!("2d {method:?} {precision:?} az={az_range:?} frame={f}");
                        assert_close(&res.intensity, &want_i, &scale, precision, &what);
                        assert_close(
                            res.errors.as_deref().unwrap(),
                            &want_e,
                            &want_e,
                            precision,
                            &what,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
