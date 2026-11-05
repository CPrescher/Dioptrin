use super::*;
use crate::{
    geometry::Geometry,
    integration::{IntegrationMethod, Integrator, MaskMode},
    mask::Mask,
    poni::Poni,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tiff::encoder::{TiffEncoder, colortype};

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn pointer(pixels: &tiff::decoder::DecodingResult) -> usize {
    macro_rules! ptr { ($($variant:ident),*) => {match pixels {$(tiff::decoder::DecodingResult::$variant(v)=>v.as_ptr() as usize,)*}}; }
    ptr!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64)
}

#[test]
fn typed_tiff_reuse_preserves_all_sample_types_and_csr_arithmetic() {
    let path = std::env::temp_dir().join(format!(
        "dioptrin-typed-{}-{}.tif",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut typed = TiffFrame::default();
    macro_rules! check {
        ($color:ty, $values:expr) => {{
            let values = $values;
            for flip in [false, true] {
                let mut pixels = values.clone();
                if flip {
                    pixels.reverse();
                }
                {
                    let mut enc = TiffEncoder::new(File::create(&path).unwrap()).unwrap();
                    let mut image = enc.new_image::<$color>(3, 3).unwrap();
                    image.rows_per_strip(2).unwrap();
                    image.write_data(&pixels).unwrap();
                }
                typed.reload(path.to_str().unwrap()).unwrap();
                let reference = Frame::from_tiff(path.to_str().unwrap().to_owned()).unwrap();
                same(&typed.to_frame().data, &reference.data);
                let before = pointer(&typed.pixels);
                typed.reload(path.to_str().unwrap()).unwrap();
                assert_eq!(before, pointer(&typed.pixels));
                same(&typed.to_frame().data, &reference.data);
                // Exercise actual same-type decoding reuse with changed pixels.
                for method in [
                    IntegrationMethod::Counting,
                    IntegrationMethod::PixelSplit,
                    IntegrationMethod::Supersampled { n_ss: 3 },
                    IntegrationMethod::Histogram,
                ] {
                    for (fast, mode) in [false, true].into_iter().flat_map(|fast| {
                        [MaskMode::Fixed, MaskMode::Dynamic].map(|mode| (fast, mode))
                    }) {
                        let mut p = Poni::new();
                        p.pixel1 = 0.1;
                        p.pixel2 = 0.1;
                        p.distance = 1.0;
                        p.poni1 = 0.12;
                        p.poni2 = 0.09;
                        let mut ai = Integrator::new(Geometry::new(3, 3, p))
                            .with_method(method)
                            .with_fast_math(fast)
                            .with_mask_mode(mode)
                            .with_polarization_factor(0.965);
                        ai.set_mask(Mask {
                            data: vec![0, 1, 0, 0, 0, 0, 0, 0, 1],
                            dim1: 3,
                            dim2: 3,
                        });
                        let expected = ai.integrate1d_with_errors(&reference, 7);
                        let actual = ai.integrate_tiff1d::<true>(&typed, 7);
                        same(&expected.0, &actual.0);
                        same(&expected.1, &actual.1);
                        same(&expected.2, &actual.2);
                        let expected = ai.integrate1d(&reference, 7);
                        let actual = ai.integrate_tiff1d::<false>(&typed, 7);
                        same(&expected.0, &actual.0);
                        same(&expected.1, &actual.1);
                        for az in [None, Some((-0.3, 0.7))] {
                            let expected = if let Some((a, b)) = az {
                                ai.integrate2d_with_azimuthal_range_and_errors(
                                    &reference, 5, 4, a, b,
                                )
                            } else {
                                ai.integrate2d_with_errors(&reference, 5, 4)
                            };
                            let actual = ai.integrate_tiff2d::<true>(&typed, 5, 4, az);
                            same(&expected.0, &actual.0);
                            same(&expected.1, &actual.1);
                        }
                    }
                }
            }
        }};
    }
    check!(colortype::Gray8, vec![0u8, 1, 2, 3, 255, 4, 5, 6, 7]);
    check!(colortype::Gray16, vec![0u16, 1, 65535, 3, 4, 5, 6, 7, 8]);
    check!(colortype::Gray32, vec![0u32, 1, u32::MAX, 3, 4, 5, 6, 7, 8]);
    check!(
        colortype::Gray64,
        vec![0u64, 1, u64::MAX, 9_007_199_254_740_993, 4, 5, 6, 7, 8]
    );
    check!(colortype::GrayI8, vec![-128i8, -1, 127, 3, 4, 5, 6, 7, 8]);
    check!(
        colortype::GrayI16,
        vec![i16::MIN, -1, i16::MAX, 3, 4, 5, 6, 7, 8]
    );
    check!(
        colortype::GrayI32,
        vec![i32::MIN, -1, i32::MAX, 3, 4, 5, 6, 7, 8]
    );
    check!(
        colortype::GrayI64,
        vec![i64::MIN, -1, i64::MAX, 9_007_199_254_740_993, 4, 5, 6, 7, 8]
    );
    check!(
        colortype::Gray32Float,
        vec![
            -0.0f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            4.,
            -5.,
            6.,
            7.,
            8.
        ]
    );
    check!(
        colortype::Gray64Float,
        vec![
            -0.0f64,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            4.,
            -5.,
            6.,
            7.,
            8.
        ]
    );
    assert!(typed.reload("/no/such/dioptrin-image.tif").is_err());
    typed.reload(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(path).unwrap();
}
