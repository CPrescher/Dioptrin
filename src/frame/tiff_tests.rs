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
    let integrators: Vec<(IntegrationMethod, Integrator)> = [
        IntegrationMethod::Counting,
        IntegrationMethod::PixelSplit,
        IntegrationMethod::Supersampled { n_ss: 3 },
        IntegrationMethod::Histogram,
    ]
    .into_iter()
    .flat_map(|method| {
        [false, true].into_iter().flat_map(move |fast| {
            [MaskMode::Fixed, MaskMode::Dynamic]
                .into_iter()
                .map(move |mode| {
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
                    (method, ai)
                })
        })
    })
    .collect();
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
                for (method, ai) in &integrators {
                    let expected = ai.integrate1d_with_errors(&reference, 7);
                    let actual = ai.integrate_tiff1d::<true>(&typed, 7);
                    same(&expected.0, &actual.0);
                    same(&expected.1, &actual.1);
                    same(&expected.2, &actual.2);
                    let expected = ai.integrate1d(&reference, 7);
                    let actual = ai.integrate_tiff1d::<false>(&typed, 7);
                    same(&expected.0, &actual.0);
                    same(&expected.1, &actual.1);
                    // Histogram has no 2D binning; its rejection is tested below.
                    if matches!(method, IntegrationMethod::Histogram) {
                        continue;
                    }
                    for az in [None, Some((-0.3, 0.7))] {
                        let expected = if let Some((a, b)) = az {
                            ai.integrate2d_with_azimuthal_range_and_errors(&reference, 5, 4, a, b)
                        } else {
                            ai.integrate2d_with_errors(&reference, 5, 4)
                        };
                        let actual = ai.integrate_tiff2d::<true>(&typed, 5, 4, az);
                        same(&expected.0, &actual.0);
                        same(&expected.1, &actual.1);
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

#[test]
#[should_panic(expected = "is not supported for Histogram method")]
fn typed_tiff_2d_rejects_histogram() {
    let path = std::env::temp_dir().join(format!(
        "dioptrin-typed-hist2d-{}-{}.tif",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let mut enc = TiffEncoder::new(File::create(&path).unwrap()).unwrap();
        enc.write_image::<colortype::Gray16>(3, 3, &[1u16, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
    }
    let mut typed = TiffFrame::default();
    typed.reload(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut p = Poni::new();
    p.pixel1 = 0.1;
    p.pixel2 = 0.1;
    p.distance = 1.0;
    let ai = Integrator::new(Geometry::new(3, 3, p)).with_method(IntegrationMethod::Histogram);
    let _ = ai.integrate_tiff2d::<false>(&typed, 5, 4, None);
}

fn temp_tiff(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "dioptrin-{tag}-{}-{}.tif",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

/// Writes a 4 x 2 (width x height) image of `C` and returns its path.
fn write_colour<C>(tag: &str, samples: usize) -> String
where
    C: tiff::encoder::colortype::ColorType,
    C::Inner: Copy + TryFrom<u8>,
    [C::Inner]: tiff::encoder::TiffValue,
{
    let path = temp_tiff(tag);
    let data: Vec<C::Inner> = (0..4 * 2 * samples)
        .map(|v| C::Inner::try_from(v as u8).ok().unwrap())
        .collect();
    let mut enc = TiffEncoder::new(File::create(&path).unwrap()).unwrap();
    enc.write_image::<C>(4, 2, &data).unwrap();
    path.to_str().unwrap().to_owned()
}

/// Issue 58: a colour TIFF used to load as the first rows x cols samples of
/// its interleaved buffer and integrate to meaningless values. Every loader
/// now rejects it with an error naming the photometric interpretation, the
/// samples per pixel and the shape; nothing panics.
#[test]
fn colour_tiffs_are_rejected_by_every_loader() {
    let gray = temp_tiff("gray-ok");
    {
        let mut enc = TiffEncoder::new(File::create(&gray).unwrap()).unwrap();
        enc.write_image::<colortype::Gray16>(4, 2, &[1u16, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
    }
    let gray = gray.to_str().unwrap().to_owned();
    let cases = [
        (
            write_colour::<colortype::RGB8>("rgb8", 3),
            "RGB, 3 samples per pixel (bits per sample [8, 8, 8])",
        ),
        (
            write_colour::<colortype::RGB16>("rgb16", 3),
            "RGB, 3 samples per pixel (bits per sample [16, 16, 16])",
        ),
        (
            write_colour::<colortype::RGBA8>("rgba8", 4),
            "RGB, 4 samples per pixel (bits per sample [8, 8, 8, 8])",
        ),
        (
            write_colour::<colortype::RGBA16>("rgba16", 4),
            "RGB, 4 samples per pixel (bits per sample [16, 16, 16, 16])",
        ),
        (
            write_colour::<colortype::CMYK8>("cmyk8", 4),
            "CMYK, 4 samples per pixel (bits per sample [8, 8, 8, 8])",
        ),
        (
            write_colour::<colortype::RGB32Float>("rgbf32", 3),
            "RGB, 3 samples per pixel (bits per sample [32, 32, 32])",
        ),
    ];
    let mut p = Poni::new();
    p.pixel1 = 0.1;
    p.pixel2 = 0.1;
    p.distance = 1.0;
    let integrator = std::sync::Arc::new(Integrator::new(Geometry::new(2, 4, p)));
    for (path, kind) in &cases {
        let expected = format!(
            "unsupported TIFF: photometric interpretation {kind}, image shape (2, 4); \
             only single-channel (grayscale) TIFFs are supported"
        );
        let err = |r: Result<(), Box<dyn std::error::Error>>| r.unwrap_err().to_string();
        assert_eq!(err(Frame::from_tiff(path.clone()).map(|_| ())), expected);
        assert_eq!(err(Mask::from_tiff(path.clone()).map(|_| ())), expected);

        // A typed frame that held a good image exposes nothing after the error.
        let mut typed = TiffFrame::default();
        typed.reload(&gray).unwrap();
        assert_eq!(err(typed.reload(path)), expected);
        assert_eq!((typed.dim1, typed.dim2), (0, 0));
        assert!(typed.legacy().is_none());
        let mut frame = Frame::from_tiff(gray.clone()).unwrap();
        assert_eq!(err(frame.reload_from_tiff(path.clone())), expected);

        // Batches: a per-item error, the grayscale file still integrates.
        let files = vec![path.clone(), gray.clone()];
        let task = crate::batch::Integration1D { n_bins: 4 };
        let check = |helper: &str, mut results: Vec<(String, Result<_, String>)>| {
            results.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, result) in results {
                if name == *path {
                    assert_eq!(result.err().as_deref(), Some(expected.as_str()), "{helper}");
                } else {
                    assert!(result.is_ok(), "{helper}: {:?}", result.err());
                }
            }
        };
        check(
            "process_files",
            crate::batch::process_files(&files, integrator.clone(), task, 2, 1),
        );
        check(
            "process_files_sequential",
            crate::batch::process_files_sequential(&files, &integrator, task),
        );
        let options = crate::batch::StreamOptions::new(2, 1);
        let streamed =
            crate::batch::process_streaming(files.clone(), integrator.clone(), task, options);
        check(
            "process_streaming",
            streamed.unwrap().map(|(_, name, r)| (name, r)).collect(),
        );
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_file(gray).unwrap();
}

/// The invariant behind the colour check: a decoded buffer that does not
/// hold exactly rows x cols samples is an error, whatever produced it.
#[test]
fn decoded_buffer_must_hold_one_sample_per_pixel() {
    use tiff::decoder::DecodingResult;
    assert!(crate::frame::check_decoded_len(&DecodingResult::U8(vec![0; 6]), 2, 3).is_ok());
    let err =
        crate::frame::check_decoded_len(&DecodingResult::F32(vec![0.0; 18]), 2, 3).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported TIFF: decoded 18 samples for an image of shape (2, 3); \
         only single-channel (grayscale) TIFFs are supported"
    );
    assert!(crate::frame::check_decoded_len(&DecodingResult::U16(vec![]), usize::MAX, 2).is_err());
}

/// Grayscale loading is unchanged by the colour check: every sample format
/// loads as its values converted to f64, rows flipped up-down, bit for bit,
/// through `Frame`, the typed `TiffFrame` and `Mask`.
#[test]
fn grayscale_tiffs_load_bit_identically() {
    macro_rules! check {
        ($color:ty, $values:expr) => {{
            let values = $values;
            let path = temp_tiff("gray-identity");
            {
                let mut enc = TiffEncoder::new(File::create(&path).unwrap()).unwrap();
                enc.write_image::<$color>(3, 2, &values).unwrap();
            }
            let path = path.to_str().unwrap().to_owned();
            // Row 1 then row 0: the loaders flip up-down.
            let expected: Vec<f64> = values[3..]
                .iter()
                .chain(&values[..3])
                .map(|&v| v as f64)
                .collect();
            let frame = Frame::from_tiff(path.clone()).unwrap();
            assert_eq!((frame.dim1, frame.dim2), (2, 3));
            same(&frame.data, &expected);
            let mut typed = TiffFrame::default();
            typed.reload(&path).unwrap();
            same(&typed.to_frame().data, &expected);
            let mask = Mask::from_tiff(path.clone()).unwrap();
            let zero: Vec<u8> = expected.iter().map(|&v| u8::from(v != 0.0)).collect();
            assert_eq!(mask.data, zero);
            std::fs::remove_file(path).unwrap();
        }};
    }
    check!(colortype::Gray8, [0u8, 1, 2, 3, 255, 4]);
    check!(colortype::Gray16, [0u16, 1, 65535, 3, 4, 5]);
    check!(colortype::Gray32, [0u32, 1, u32::MAX, 3, 4, 5]);
    check!(colortype::Gray64, [0u64, 1, u64::MAX, 3, 4, (1 << 53) + 1]);
    check!(colortype::GrayI8, [-128i8, -1, 127, 0, 4, 5]);
    check!(colortype::GrayI16, [i16::MIN, -1, i16::MAX, 0, 4, 5]);
    check!(colortype::GrayI32, [i32::MIN, -1, i32::MAX, 0, 4, 5]);
    check!(colortype::GrayI64, [i64::MIN, -1, i64::MAX, 0, 4, 5]);
    check!(
        colortype::Gray32Float,
        [f32::MIN, -0.0, f32::MAX, 0.5, 1e-30, f32::INFINITY]
    );
    check!(
        colortype::Gray64Float,
        [f64::MIN, -0.0, f64::MAX, 0.5, 1e-300, f64::NAN]
    );
}

/// Writes a little-endian, uncompressed, single-strip TIFF with the given
/// 16-bit tags (besides the size and strip tags).
fn write_raw_tiff(
    path: &std::path::Path,
    width: u16,
    height: u16,
    tags: &[(u16, u16)],
    data: &[u8],
) {
    let mut entries: Vec<(u16, u16, u32)> = vec![
        (256, 3, width as u32),
        (257, 3, height as u32),
        (259, 3, 1),
        (278, 3, height as u32),
        (279, 4, data.len() as u32),
    ];
    entries.extend(tags.iter().map(|&(tag, value)| (tag, 3, value as u32)));
    let n = entries.len() + 1; // + StripOffsets
    let data_offset = (8 + 2 + 12 * n + 4) as u32;
    entries.push((273, 4, data_offset));
    entries.sort();
    let mut out = b"II".to_vec();
    out.extend(42u16.to_le_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend((n as u16).to_le_bytes());
    for (tag, kind, value) in entries {
        out.extend(tag.to_le_bytes());
        out.extend(kind.to_le_bytes());
        out.extend(1u32.to_le_bytes());
        if kind == 3 {
            out.extend((value as u16).to_le_bytes());
            out.extend([0, 0]);
        } else {
            out.extend(value.to_le_bytes());
        }
    }
    out.extend(0u32.to_le_bytes());
    assert_eq!(out.len() as u32, data_offset);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

/// Review of #58: SamplesPerPixel 3 or 4 with a single BitsPerSample value
/// (accepted by libtiff and tifffile) is reported as grayscale by the tiff
/// crate, which then decodes the first rows x cols interleaved samples. It
/// is rejected like any other colour file; the same file with
/// SamplesPerPixel 1 (or no such tag) still loads.
#[test]
fn multi_sample_tiff_with_one_bits_per_sample_entry_is_rejected() {
    let data: Vec<u8> = (0..4 * 2 * 4).map(|v| v as u8).collect();
    for (photometric, name) in [(1u16, "BlackIsZero"), (0, "WhiteIsZero")] {
        for samples in [3u16, 4] {
            let path = temp_tiff("spp-bps1");
            let tags = [(258, 8), (262, photometric), (277, samples)];
            write_raw_tiff(&path, 4, 2, &tags, &data[..8 * samples as usize]);
            let expected = format!(
                "unsupported TIFF: photometric interpretation {name}, {samples} samples \
                 per pixel (bits per sample [8]), image shape (2, 4); \
                 only single-channel (grayscale) TIFFs are supported"
            );
            let file = path.to_str().unwrap().to_owned();
            let err = |r: Result<(), Box<dyn std::error::Error>>| r.unwrap_err().to_string();
            assert_eq!(err(Frame::from_tiff(file.clone()).map(|_| ())), expected);
            assert_eq!(err(Mask::from_tiff(file.clone()).map(|_| ())), expected);
            assert_eq!(err(TiffFrame::default().reload(&file)), expected);
            std::fs::remove_file(&path).unwrap();
        }
        for tags in [
            vec![(258, 8), (262, photometric), (277, 1)],
            vec![(258, 8), (262, photometric)],
        ] {
            let path = temp_tiff("spp1-bps1");
            write_raw_tiff(&path, 4, 2, &tags, &data[..8]);
            let frame = Frame::from_tiff(path.to_str().unwrap().to_owned()).unwrap();
            assert_eq!((frame.dim1, frame.dim2, frame.data.len()), (2, 4, 8));
            std::fs::remove_file(&path).unwrap();
        }
    }
}
