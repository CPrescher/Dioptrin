//! Benchmark-only C bridge: prepare Dioptrin and Metal data in one process.
use dioptrin::{
    frame::Frame,
    geometry::Geometry,
    integration::{IntegrationMethod, Integrator, RadialUnit, SparseLayout},
    lut::{self, Lut1d},
    mask::Mask,
    poni::Poni,
};
use std::{
    ffi::{CStr, c_char, c_void},
    panic::AssertUnwindSafe,
};

pub struct Context {
    frame: Frame,
    mask: Mask,
    bins: usize,
    cpu: Option<Integrator>,
    table: Option<Lut1d>,
    normalization: Vec<f64>,
    rows: Vec<u32>,
    pixels: Vec<u32>,
    weights: Vec<f32>,
    bin_norm: Vec<f64>,
    signal32: Vec<f32>,
}

unsafe fn string<'a>(p: *const c_char) -> &'a str {
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .expect("UTF-8 argument")
}

/// backend: 0 default CSC, 1 fast CSR, 2 strict CSR, 3 GPU export.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_new(
    image: *const c_char,
    poni: *const c_char,
    mask: *const c_char,
    weighted: bool,
    bins: usize,
    threads: usize,
    backend: u32,
) -> *mut Context {
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<Context, Box<dyn std::error::Error>> {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()?;
            let frame = Frame::from_tiff(unsafe { string(image) }.into())?;
            let mut mask = if unsafe { string(mask) } == "-" {
                Mask {
                    data: vec![0; frame.data.len()],
                    dim1: frame.dim1,
                    dim2: frame.dim2,
                }
            } else {
                Mask::from_tiff(unsafe { string(mask) }.into())?
            };
            mask.assert_compatible(frame.dim1, frame.dim2);
            for (&x, m) in frame.data.iter().zip(&mut mask.data) {
                if !x.is_finite() || x < 0. {
                    *m = 1;
                }
            }
            let poni = Poni::load(unsafe { string(poni) })?;
            assert!(poni.pixel1 > 0. && poni.pixel2 > 0. && bins > 0);
            let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
            let method = if weighted {
                IntegrationMethod::PixelSplit
            } else {
                IntegrationMethod::Counting
            };
            let mut context = Context {
                frame,
                mask,
                bins,
                cpu: None,
                table: None,
                normalization: vec![],
                rows: vec![],
                pixels: vec![],
                weights: vec![],
                bin_norm: vec![],
                signal32: vec![],
            };
            if backend != 3 {
                let mut cpu = Integrator::new(geometry)
                    .with_method(method)
                    .with_radial_unit(RadialUnit::TthDeg)
                    .with_polarization_factor(0.965)
                    .with_sparse_layout(if backend == 0 {
                        SparseLayout::Csc
                    } else {
                        SparseLayout::Csr
                    })
                    .with_fast_math(backend != 2);
                cpu.set_mask(Mask {
                    data: context.mask.data.clone(),
                    dim1: context.mask.dim1,
                    dim2: context.mask.dim2,
                });
                context.cpu = Some(cpu);
            } else {
                let tth = geometry.calc_tth_all();
                let radial = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
                let table = if weighted {
                    lut::build_lut1d_ps_csr(&geometry, &radial, bins, RadialUnit::TthDeg)
                } else {
                    lut::build_lut1d_csr(&radial, bins)
                };
                let sa = geometry.calc_solid_angle();
                let pol = geometry.calc_polarization(0.965, &tth, &geometry.calc_azimuth_all());
                context.normalization = pol.iter().zip(sa).map(|(p, s)| p * s).collect();
                context.rows.push(0);
                context.pixels.reserve(table.col_idx.len());
                context.weights.reserve(table.col_idx.len());
                context.bin_norm = vec![0.; bins];
                for b in 0..bins {
                    for k in table.row_ptr[b]..table.row_ptr[b + 1] {
                        let p = table.col_idx[k];
                        if context.mask.data[p] != 0 {
                            continue;
                        }
                        let w = table.values.as_ref().map_or(1., |v| v[k]);
                        context.pixels.push(u32::try_from(p)?);
                        context.weights.push(w as f32);
                        context.bin_norm[b] += w * context.normalization[p];
                    }
                    context.rows.push(u32::try_from(context.pixels.len())?);
                }
                context.signal32 = context.frame.data.iter().map(|&v| v as f32).collect();
                // Retained solely for reference validation after timing. No CPU
                // reference integration is done in the measured startup interval.
                context.table = Some(table);
            }
            Ok(context)
        },
    ));
    match result {
        Ok(Ok(context)) => Box::into_raw(Box::new(context)),
        Ok(Err(error)) => {
            eprintln!("setup failed: {error}");
            std::ptr::null_mut()
        }
        Err(_) => {
            eprintln!("setup panicked");
            std::ptr::null_mut()
        }
    }
}

/// 0 signal64, 1 rows, 2 pixels, 3 weights, 4 bin normalization, 5 signal32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_ptr(context: *const Context, kind: u32) -> *const c_void {
    let c = unsafe { &*context };
    match kind {
        0 => c.frame.data.as_ptr().cast(),
        1 => c.rows.as_ptr().cast(),
        2 => c.pixels.as_ptr().cast(),
        3 => c.weights.as_ptr().cast(),
        4 => c.bin_norm.as_ptr().cast(),
        5 => c.signal32.as_ptr().cast(),
        _ => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_len(context: *const Context, kind: u32) -> usize {
    let c = unsafe { &*context };
    match kind {
        0 | 5 => c.frame.data.len(),
        1 => c.rows.len(),
        2 | 3 => c.pixels.len(),
        4 => c.bins,
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_integrate(
    context: *const Context,
    intensity: *mut f64,
    std: *mut f64,
) {
    let c = unsafe { &*context };
    let (_, y, e) = c
        .cpu
        .as_ref()
        .expect("CPU context")
        .integrate1d_with_errors(&c.frame, c.bins);
    unsafe {
        std::ptr::copy_nonoverlapping(y.as_ptr(), intensity, c.bins);
        std::ptr::copy_nonoverlapping(e.as_ptr(), std, c.bins);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_reference(
    context: *const Context,
    intensity: *mut f64,
    std: *mut f64,
) {
    let c = unsafe { &*context };
    let variance: Vec<_> = c.frame.data.iter().map(|v| v.max(0.)).collect();
    let (_, y, e) = lut::integrate1d_with_errors(
        &c.frame.data,
        &variance,
        c.table.as_ref().expect("GPU reference"),
        Some(&c.mask.data),
        &c.normalization,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(y.as_ptr(), intensity, c.bins);
        std::ptr::copy_nonoverlapping(e.as_ptr(), std, c.bins);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup_free(context: *mut Context) {
    drop(unsafe { Box::from_raw(context) });
}
