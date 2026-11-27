use super::*;

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} versus {b}");
    }
}

#[test]
fn prepared_matches_original_strict_exactly_and_fast_with_reassociation() {
    for size in 0..14 {
        for weighted in [false, true] {
            for masked in [false, true] {
                let mut signal: Vec<_> =
                    (0..size).map(|i| (i as f64 * 0.91).sin() * 30.0).collect();
                if size > 3 {
                    signal[2] = -0.0;
                }
                let norm: Vec<_> = (0..size).map(|i| 0.2 + i as f64 * 0.07).collect();
                let mask: Vec<_> = (0..size).map(|i| u8::from(i % 3 == 0)).collect();
                let m = masked.then_some(mask.as_slice());
                let variance: Vec<_> = signal.iter().map(|&x| x.max(0.0)).collect();
                let make = || Lut1d {
                    row_ptr: vec![0, 0, size, size],
                    col_idx: (0..size).collect(),
                    values: weighted.then(|| (0..size).map(|i| 0.17 + i as f64 * 0.01).collect()),
                    hist_min: 0.1,
                    hist_step: 0.2,
                };
                let l = make();
                let p = Prepared1d::new(make());
                assert!(matches!(p.col_idx, Columns::Compact(_)));
                for fast in [false, true] {
                    let compare = |a: &[f64], b: &[f64]| {
                        if !fast || !masked {
                            same(a, b);
                        } else {
                            // Mask compaction changes four-lane assignment,
                            // while retaining every remaining float64 weight.
                            for (&x, &y) in a.iter().zip(b) {
                                assert!((x - y).abs() <= 2e-13 * x.abs().max(1.));
                            }
                        }
                    };
                    let expected = if fast {
                        super::super::fast::integrate1d_with_errors(
                            &signal, &variance, &l, m, &norm,
                        )
                    } else {
                        super::super::integrate1d_with_errors(&signal, &variance, &l, m, &norm)
                    };
                    let actual = if fast {
                        p.integrate::<true, true>(&signal, m, &norm)
                    } else {
                        p.integrate::<false, true>(&signal, m, &norm)
                    };
                    compare(&expected.0, &actual.0);
                    compare(&expected.1, &actual.1);
                    compare(&expected.2, &actual.2);
                    let expected = if fast {
                        super::super::fast::integrate1d(&signal, &l, m, &norm)
                    } else {
                        super::super::integrate1d(&signal, &l, m, &norm)
                    };
                    let actual = if fast {
                        p.integrate::<true, false>(&signal, m, &norm)
                    } else {
                        p.integrate::<false, false>(&signal, m, &norm)
                    };
                    compare(&expected.0, &actual.0);
                    compare(&expected.1, &actual.1);
                    let expected = if fast {
                        super::super::fast::integrate1d_raw_with_variance(
                            &signal, &variance, &l, m, &norm,
                        )
                    } else {
                        super::super::integrate1d_raw_with_variance(
                            &signal, &variance, &l, m, &norm,
                        )
                    };
                    let actual = if fast {
                        p.raw1d::<true, true>(&signal, m, &norm)
                    } else {
                        p.raw1d::<false, true>(&signal, m, &norm)
                    };
                    compare(&expected.sum_signal, &actual.sum_signal);
                    compare(&expected.sum_norm, &actual.sum_norm);
                    compare(&expected.count, &actual.count);
                    compare(
                        expected.sum_variance.as_ref().unwrap(),
                        actual.sum_variance.as_ref().unwrap(),
                    );
                    let l2 = Lut2d {
                        row_ptr: l.row_ptr.clone(),
                        col_idx: l.col_idx.clone(),
                        values: l.values.clone(),
                        n_tth: 3,
                        n_azimuthal: 1,
                    };
                    let p2 = Prepared2d::new(Lut2d {
                        row_ptr: l.row_ptr.clone(),
                        col_idx: l.col_idx.clone(),
                        values: l.values.clone(),
                        n_tth: 3,
                        n_azimuthal: 1,
                    });
                    let expected = if fast {
                        super::super::fast::integrate2d_with_errors(
                            &signal, &variance, &l2, m, &norm,
                        )
                    } else {
                        super::super::integrate2d_with_errors(&signal, &variance, &l2, m, &norm)
                    };
                    let actual = if fast {
                        p2.integrate::<true, true>(&signal, m, &norm)
                    } else {
                        p2.integrate::<false, true>(&signal, m, &norm)
                    };
                    compare(&expected.0, &actual.0);
                    compare(&expected.1, &actual.1);
                    let (direct, errors) = if fast {
                        p2.integrate::<true, false>(&signal, m, &norm)
                    } else {
                        p2.integrate::<false, false>(&signal, m, &norm)
                    };
                    same(&actual.0, &direct);
                    assert!(errors.is_empty());
                }
            }
        }
    }
}

#[test]
fn normalization_policies_are_cached_independently_and_resettable() {
    let mut p = Prepared1d::new(Lut1d {
        row_ptr: vec![0, 5],
        col_idx: (0..5).collect(),
        values: Some(vec![0.3; 5]),
        hist_min: 0.0,
        hist_step: 1.0,
    });
    let n = vec![1e16, 1.0, 1.0, -1e16, 2.0];
    p.prepare_normalization(None, &n, false);
    assert!(p.strict_norm.get().is_some());
    assert!(p.fast_norm.get().is_none());
    let ptr = p.strict_norm.get().unwrap().as_ptr();
    p.prepare_normalization(None, &n, false);
    assert_eq!(p.strict_norm.get().unwrap().as_ptr(), ptr);
    p.prepare_normalization(None, &n, true);
    assert!(p.fast_norm.get().is_some());
    p.invalidate_normalization();
    assert!(p.strict_norm.get().is_none());
    assert!(p.fast_norm.get().is_none());
}

#[test]
fn compact_indices_are_checked_and_large_indices_have_a_wide_fallback() {
    let c = Columns::new(vec![0, 13, u32::MAX as usize]);
    assert!(matches!(c, Columns::Compact(_)));
    assert_eq!(c.get(2), u32::MAX as usize);
    #[cfg(target_pointer_width = "64")]
    {
        let large = u32::MAX as usize + 1;
        let c = Columns::new(vec![0, large]);
        assert!(matches!(c, Columns::Wide(_)));
        assert_eq!(c.get(1), large);
    }
}

#[test]
fn nonfinite_negative_and_zero_normalization_semantics_match_original() {
    let cases = [
        vec![f64::NAN, 1.0, -3.0, 0.0],
        vec![f64::INFINITY, 1.0, 2.0, 3.0],
        vec![f64::NEG_INFINITY, 2.0, -0.0, 1.0],
    ];
    for signal in cases {
        for norm in [vec![1.0; 4], vec![0.0; 4]] {
            let variance: Vec<_> = signal.iter().map(|&x| x.max(0.0)).collect();
            let make = || Lut1d {
                row_ptr: vec![0, 4],
                col_idx: (0..4).collect(),
                values: Some(vec![0.3; 4]),
                hist_min: 0.0,
                hist_step: 1.0,
            };
            for fast in [false, true] {
                let l = make();
                let p = Prepared1d::new(make());
                let expected = if fast {
                    super::super::fast::integrate1d_with_errors(&signal, &variance, &l, None, &norm)
                } else {
                    super::super::integrate1d_with_errors(&signal, &variance, &l, None, &norm)
                };
                let actual = if fast {
                    p.integrate::<true, true>(&signal, None, &norm)
                } else {
                    p.integrate::<false, true>(&signal, None, &norm)
                };
                same(&expected.1, &actual.1);
                same(&expected.2, &actual.2);
            }
        }
    }
}

#[test]
fn mask_policies_refresh_all_raw_sums_and_restore_previously_masked_pixels() {
    let make = || {
        Prepared1d::new(Lut1d {
            row_ptr: vec![0, 3, 4, 4],
            col_idx: vec![0, 1, 2, 2],
            values: Some(vec![0.25, 0.5, 0.75, 1.]),
            hist_min: 0.,
            hist_step: 1.,
        })
    };
    let norm = [1., 2., 3.];
    for mode in [MaskMode::Fixed, MaskMode::Dynamic] {
        for fast in [false, true] {
            let mut p = make();
            p.set_mask_mode(mode);
            let original = match &p.col_idx {
                Columns::Compact(c) => c.as_ptr() as usize,
                _ => unreachable!(),
            };
            let masks = [vec![0, 1, 0], vec![0, 0, 0], vec![1, 1, 1], vec![0, 1, 0]];
            let mut csc_address = None;
            for (i, mask) in masks.iter().enumerate() {
                p.invalidate_mask();
                let signal = if i == 0 {
                    [4., f64::NAN, 9.]
                } else {
                    [4., 8., 9.]
                };
                let got = if fast {
                    p.raw1d::<true, true>(&signal, Some(mask), &norm)
                } else {
                    p.raw1d::<false, true>(&signal, Some(mask), &norm)
                };
                let (s, n, v, c) = match i {
                    1 => (
                        vec![11.75, 9., 0.],
                        vec![3.5, 3., 0.],
                        vec![7.3125, 9., 0.],
                        vec![1.5, 1., 0.],
                    ),
                    2 => (vec![0.; 3], vec![0.; 3], vec![0.; 3], vec![0.; 3]),
                    _ => (
                        vec![7.75, 9., 0.],
                        vec![2.5, 3., 0.],
                        vec![5.3125, 9., 0.],
                        vec![1., 1., 0.],
                    ),
                };
                same(&got.sum_signal, &s);
                same(&got.sum_norm, &n);
                same(got.sum_variance.as_ref().unwrap(), &v);
                same(&got.count, &c);
                if fast && mode == MaskMode::Dynamic {
                    let address = p.csc.get().unwrap().as_ref().unwrap() as *const Csc as usize;
                    assert_eq!(*csc_address.get_or_insert(address), address);
                    assert!(p.fixed.get().is_none());
                }
                assert_eq!(
                    match &p.col_idx {
                        Columns::Compact(c) => c.as_ptr() as usize,
                        _ => unreachable!(),
                    },
                    original
                );
            }
            p.invalidate_mask();
            let got = p.raw1d::<false, true>(&[4., 8., 9.], None, &norm);
            same(&got.sum_signal, &[11.75, 9., 0.]);
            same(&got.sum_norm, &[3.5, 3., 0.]);
        }
    }
}

#[test]
fn switching_policy_retains_valid_layouts_and_corrections_preserve_fixed_table() {
    let mut p = Prepared1d::new(Lut1d {
        row_ptr: vec![0, 2],
        col_idx: vec![0, 1],
        values: Some(vec![0.3, 0.7]),
        hist_min: 0.,
        hist_step: 1.,
    });
    p.prepare_normalization(Some(&[0, 1]), &[1., 1.], false);
    let fixed = p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize;
    p.invalidate_normalization();
    p.prepare_normalization(Some(&[0, 1]), &[2., 2.], false);
    assert_eq!(
        p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize,
        fixed
    );
    let out = p.integrate::<true, true>(&[4., 8.], Some(&[0, 1]), &[2., 2.]);
    p.set_mask_mode(MaskMode::Dynamic);
    assert_eq!(
        p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize,
        fixed
    );
    p.prepare_normalization(Some(&[0, 1]), &[2., 2.], true);
    assert!(p.csc.get().unwrap().is_some());
    let dynamic = p.integrate::<true, true>(&[4., 8.], Some(&[0, 1]), &[2., 2.]);
    for (a, b) in out
        .1
        .iter()
        .chain(&out.2)
        .zip(dynamic.1.iter().chain(&dynamic.2))
    {
        assert!((a - b).abs() < 1e-14);
    }
    p.set_mask_mode(MaskMode::Fixed);
    assert!(p.csc.get().unwrap().is_some());
    p.prepare_normalization(Some(&[0, 1]), &[2., 2.], true);
    assert_eq!(
        p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize,
        fixed
    );
    // A mask edit still invalidates mask-specific state, while CSC weights stay.
    p.invalidate_mask();
    assert!(p.fixed.get().is_none());
    assert!(p.csc.get().unwrap().is_some());
    let changed = p.integrate::<true, true>(&[4., 8.], Some(&[1, 0]), &[2., 2.]);
    assert!((changed.1[0] - 4.).abs() < 1e-14);
    p.set_mask_mode(MaskMode::Dynamic);
    let dynamic_changed = p.integrate::<true, true>(&[4., 8.], Some(&[1, 0]), &[2., 2.]);
    assert!((dynamic_changed.1[0] - changed.1[0]).abs() < 1e-14);
}

#[test]
fn dynamic_first_use_is_safe_when_all_pool_workers_integrate_concurrently() {
    let mut p = Prepared1d::new(Lut1d {
        row_ptr: vec![0, 3, 4, 4],
        col_idx: vec![0, 1, 2, 2],
        values: Some(vec![0.25, 0.5, 0.75, 1.]),
        hist_min: 0.,
        hist_step: 1.,
    });
    p.set_mask_mode(MaskMode::Dynamic);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let outputs: Vec<_> = pool.install(|| {
        (0..8)
            .into_par_iter()
            .map(|_| {
                p.integrate::<true, true>(&[4., f64::NAN, 9.], Some(&[0, 1, 0]), &[1., 2., 3.])
            })
            .collect()
    });
    for x in &outputs[1..] {
        assert_eq!(x, &outputs[0]);
    }
}

#[test]
fn fixed_fast_1d_reuses_csc_and_cached_selection_across_policies() {
    let make = || {
        Prepared1d::new(Lut1d {
            row_ptr: vec![0, 3, 4],
            col_idx: vec![0, 1, 2, 2],
            values: Some(vec![0.25, 0.5, 0.75, 1.]),
            hist_min: 0.,
            hist_step: 1.,
        })
    };
    let norm = [1., 2., 3.];
    let signal = [4., 8., 9.];
    let mut p = make();
    p.prepare_normalization(Some(&[0, 1, 0]), &norm, true);
    assert!(p.csc.get().unwrap().is_some());
    assert!(p.fixed.get().is_none()); // fast 1D did not compile a masked CSR copy
    let pixels = p.fixed_pixels.get().unwrap().as_ptr();
    let cached_norm = p.fast_norm.get().unwrap().as_ptr();
    let fixed = p.integrate::<true, true>(&signal, Some(&[0, 1, 0]), &norm);
    p.set_mask_mode(MaskMode::Dynamic);
    let dynamic = p.integrate::<true, true>(&signal, Some(&[0, 1, 0]), &norm);
    for (a, b) in fixed
        .1
        .iter()
        .chain(&fixed.2)
        .zip(dynamic.1.iter().chain(&dynamic.2))
    {
        assert!((a - b).abs() < 1e-14);
    }
    p.set_mask_mode(MaskMode::Fixed);
    p.prepare_normalization(Some(&[0, 1, 0]), &norm, true);
    assert_eq!(pixels, p.fixed_pixels.get().unwrap().as_ptr());
    assert_eq!(cached_norm, p.fast_norm.get().unwrap().as_ptr());
    p.invalidate_normalization();
    assert!(p.fast_norm.get().is_none());
    assert_eq!(pixels, p.fixed_pixels.get().unwrap().as_ptr());
    p.prepare_normalization(Some(&[0, 1, 0]), &[2., 4., 6.], true);
    let corrected = p.integrate::<true, true>(&signal, Some(&[0, 1, 0]), &[2., 4., 6.]);
    for (a, b) in fixed.1.iter().zip(&corrected.1) {
        assert!((a / 2. - b).abs() < 1e-14);
    }
    p.invalidate_mask();
    assert!(p.fixed_pixels.get().is_none());
    assert!(p.fast_norm.get().is_none());
    assert!(p.csc.get().unwrap().is_some());
    let updated = p.integrate::<true, true>(&signal, Some(&[0, 0, 0]), &norm);
    let expected = make().integrate::<true, true>(&signal, Some(&[0, 0, 0]), &norm);
    same(&updated.1, &expected.1);
    same(&updated.2, &expected.2);
    let strict = p.integrate::<false, true>(&signal, Some(&[0, 0, 0]), &norm);
    // An all-zero mask initializes the CSR policy without making a copy.
    assert!(p.fixed.get().is_some());
    for (a, b) in strict
        .1
        .iter()
        .chain(&strict.2)
        .zip(updated.1.iter().chain(&updated.2))
    {
        assert!((a - b).abs() < 1e-14);
    }
}

/// A 6 x 4 cake over 40 pixels whose rows hold 0 to 13 entries, so fast rows
/// cover full four-lane blocks and every remainder length, with repeated
/// pixels, and with empty rows (zero normalization).
fn cake_table(weighted: bool) -> Lut2d {
    let npix = 40;
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for r in 0..24 {
        for e in 0..(r * 5) % 14 {
            col_idx.push((r * 7 + e * 3) % npix);
            values.push(0.15 + ((r + e) % 9) as f64 * 0.11);
        }
        row_ptr.push(col_idx.len());
    }
    Lut2d {
        row_ptr,
        col_idx,
        values: weighted.then_some(values),
        n_tth: 6,
        n_azimuthal: 4,
    }
}

/// Integer-valued pixels with negatives and zeros, exactly representable in
/// every `Pixel` type used below.
fn cake_frame(f: usize) -> Vec<i32> {
    (0..40)
        .map(|i| ((i * (17 + f * 7) + f) % 23) as i32 - 6)
        .collect()
}

fn cake_norm() -> Vec<f64> {
    (0..40)
        .map(|i| {
            if i % 9 == 4 {
                0.0
            } else {
                0.3 + i as f64 * 0.05
            }
        })
        .collect()
}

fn cake_mask() -> Vec<u8> {
    (0..40).map(|i| u8::from(i % 7 == 2)).collect()
}

/// The pre-change normalisation of raw sums: `signal / norm` and
/// `sqrt(max(variance / norm², 0))`, zero where `norm == 0`.
fn normalized_from_raw(raw: &IntermediateResult2d) -> (Vec<f64>, Vec<f64>) {
    let variance = raw.sum_variance.as_ref().expect("raw variance");
    raw.sum_signal
        .iter()
        .zip(&raw.sum_norm)
        .zip(variance)
        .map(|((&s, &n), &v)| {
            let y = if n == 0.0 { 0.0 } else { s / n };
            let e = if n == 0.0 { 0.0 } else { v / (n * n) };
            (y, e.max(0.0).sqrt())
        })
        .unzip()
}

#[test]
fn direct_fixed_2d_path_matches_normalized_raw_sums_bit_for_bit() {
    let norm = cake_norm();
    let mask = cake_mask();
    let mut signal: Vec<f64> = cake_frame(1).iter().map(|&x| x as f64 * 0.37).collect();
    signal[3] = -0.0;
    signal[5] = 0.0;
    for weighted in [false, true] {
        for m in [None, Some(mask.as_slice())] {
            macro_rules! check {
                ($fast:literal) => {{
                    let p = Prepared2d::new(cake_table(weighted));
                    let raw = p.raw2d::<$fast, true>(&signal, m, &norm);
                    let (want_y, want_e) = normalized_from_raw(&raw);
                    let (y, e) = p.integrate::<$fast, true>(&signal, m, &norm);
                    same(&y, &want_y);
                    same(&e, &want_e);
                    let (y, e) = p.integrate::<$fast, false>(&signal, m, &norm);
                    same(&y, &want_y);
                    assert!(e.is_empty());
                    let (y, e) = p.integrate_pixels::<$fast, true, f32>(
                        &signal.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                        m,
                        &norm,
                    );
                    let narrowed: Vec<f64> = signal.iter().map(|&x| x as f32 as f64).collect();
                    let raw = p.raw2d::<$fast, true>(&narrowed, m, &norm);
                    let (want_y, want_e) = normalized_from_raw(&raw);
                    same(&y, &want_y);
                    same(&e, &want_e);
                }};
            }
            check!(true);
            check!(false);
        }
    }
}

#[test]
fn dynamic_2d_mask_keeps_its_own_path_and_matches_raw_sums() {
    let norm = cake_norm();
    let mask = cake_mask();
    let signal: Vec<f64> = cake_frame(2).iter().map(|&x| x as f64 * 0.61).collect();
    let frames: [&[f64]; 4] = [&signal, &signal, &signal, &signal];
    for weighted in [false, true] {
        for m in [None, Some(mask.as_slice())] {
            macro_rules! check {
                ($fast:literal) => {{
                    let mut p = Prepared2d::new(cake_table(weighted));
                    p.set_mask_mode(MaskMode::Dynamic);
                    assert!(!p.can_batch4($fast, true));
                    assert!(!p.can_batch4($fast, false));
                    assert!(
                        p.integrate_batch4::<$fast, true, f64>(frames, m, &norm)
                            .is_none()
                    );
                    let raw = p.raw2d::<$fast, true>(&signal, m, &norm);
                    let (want_y, want_e) = normalized_from_raw(&raw);
                    let (y, e) = p.integrate::<$fast, true>(&signal, m, &norm);
                    same(&y, &want_y);
                    same(&e, &want_e);
                    // A dynamic mask is read on every call, never compacted.
                    assert!(p.fixed.get().is_none());
                }};
            }
            check!(true);
            check!(false);
        }
    }
}

/// Every pixel type the tiled kernel is instantiated for: each frame of a
/// 4-frame tile must equal its own per-frame integration bit for bit, for
/// fast and strict arithmetic, with and without errors, weights and mask.
#[test]
fn tiled_2d_matches_per_frame_bit_for_bit_for_every_pixel_type() {
    fn check<P: Pixel>(convert: impl Fn(i32) -> P) {
        let norm = cake_norm();
        let mask = cake_mask();
        let frames: Vec<Vec<P>> = (0..4)
            .map(|f| cake_frame(f).into_iter().map(&convert).collect())
            .collect();
        let tile: [&[P]; 4] = std::array::from_fn(|f| frames[f].as_slice());
        for weighted in [false, true] {
            for m in [None, Some(mask.as_slice())] {
                macro_rules! case {
                    ($fast:literal, $var:literal) => {{
                        let p = Prepared2d::new(cake_table(weighted));
                        assert!(p.can_batch4($fast, $var));
                        let batch = p
                            .integrate_batch4::<$fast, $var, P>(tile, m, &norm)
                            .expect("fixed-mask tile");
                        for (f, (y, e)) in batch.iter().enumerate() {
                            let (want_y, want_e) =
                                p.integrate_pixels::<$fast, $var, P>(tile[f], m, &norm);
                            same(y, &want_y);
                            if $var {
                                same(e, &want_e);
                            } else {
                                assert!(e.is_empty() && want_e.is_empty());
                            }
                        }
                    }};
                }
                case!(true, true);
                case!(true, false);
                case!(false, true);
                case!(false, false);
            }
        }
    }
    check(|x| x as f64);
    check(|x| x as f32);
    check(|x| x);
    check(|x| x as i16);
    check(|x| x as i64);
    check(|x| (x + 6) as u16);
    check(|x| (x + 6) as u32);
    check(|x| (x + 6) as u8);
}

/// The 1D CSR tile shares `cached_results_batch4_into` (strict arithmetic,
/// and fast intensity with the CSR layout), so it must also match its own
/// per-frame path bit for bit.
#[test]
fn tiled_1d_csr_matches_per_frame_bit_for_bit() {
    let norm = cake_norm();
    let mask = cake_mask();
    let frames: Vec<Vec<f64>> = (0..4)
        .map(|f| cake_frame(f).iter().map(|&x| x as f64 * 0.43).collect())
        .collect();
    let tile: [&[f64]; 4] = std::array::from_fn(|f| frames[f].as_slice());
    for weighted in [false, true] {
        for m in [None, Some(mask.as_slice())] {
            let make = || {
                let l = cake_table(weighted);
                let mut p = Prepared1d::new(Lut1d {
                    row_ptr: l.row_ptr,
                    col_idx: l.col_idx,
                    values: l.values,
                    hist_min: 0.0,
                    hist_step: 1.0,
                });
                p.set_sparse_layout(SparseLayout::Csr);
                p
            };
            macro_rules! case {
                ($fast:literal, $var:literal) => {{
                    let p = make();
                    let batch = p
                        .integrate_batch4::<$fast, $var, f64>(tile, m, &norm)
                        .expect("tiled CSR");
                    for (f, (_, y, e)) in batch.iter().enumerate() {
                        let (_, want_y, want_e) =
                            p.integrate_pixels::<$fast, $var, f64>(tile[f], m, &norm);
                        same(y, &want_y);
                        same(e, &want_e);
                    }
                }};
            }
            case!(false, true);
            case!(false, false);
            case!(true, false);
        }
    }
}

/// Bit equality, except that a NaN only has to be matched by a NaN. Rust
/// and LLVM do not preserve NaN sign or payload (operands of `+` and FMA may
/// be commuted), so a bin mixing +NaN and -NaN inputs can yield either sign.
/// On aarch64 release builds the unchanged strict tile already differs from
/// its per-frame path in exactly this way; every non-NaN result, including
/// signed zeros, infinities and subnormals, must match bit for bit.
fn same_up_to_nan_sign(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "element {i}: {a} ({:#x}) versus {b} ({:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// Tile `frames` through `integrate_batch4` and compare every frame with its
/// own per-frame integration (`same_up_to_nan_sign`), for fast and strict
/// arithmetic, with and without errors.
fn assert_tiles_match(
    make: impl Fn() -> Lut2d,
    frames: &[Vec<f64>; 4],
    m: Option<&[u8]>,
    norm: &[f64],
) {
    let tile: [&[f64]; 4] = std::array::from_fn(|f| frames[f].as_slice());
    macro_rules! case {
        ($fast:literal, $var:literal) => {{
            let p = Prepared2d::new(make());
            let batch = p
                .integrate_batch4::<$fast, $var, f64>(tile, m, norm)
                .expect("fixed-mask tile");
            for (f, (y, e)) in batch.iter().enumerate() {
                let (want_y, want_e) = p.integrate_pixels::<$fast, $var, f64>(tile[f], m, norm);
                same_up_to_nan_sign(y, &want_y);
                same_up_to_nan_sign(e, &want_e);
            }
        }};
    }
    case!(true, true);
    case!(true, false);
    case!(false, true);
    case!(false, false);
}

#[test]
fn tiled_2d_matches_per_frame_for_nonfinite_subnormal_and_extreme_pixels() {
    let special = [
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(1),                      // smallest subnormal
        -f64::from_bits(0x000f_ffff_ffff_ffff), // largest negative subnormal
        f64::MIN_POSITIVE,
        -0.0,
        0.0,
        1e300,
        -1e300,
        f64::MAX,
    ];
    // Each frame places the special values at different pixels, so bins mix
    // them with ordinary values in every lane position.
    let frames: [Vec<f64>; 4] = std::array::from_fn(|f| {
        let mut x: Vec<f64> = cake_frame(f).iter().map(|&v| v as f64 * 1.3).collect();
        for (i, &v) in special.iter().enumerate() {
            x[(i * 3 + f * 5) % 40] = v;
        }
        x
    });
    let norm = cake_norm();
    let mask = cake_mask();
    for weighted in [false, true] {
        for m in [None, Some(mask.as_slice())] {
            assert_tiles_match(|| cake_table(weighted), &frames, m, &norm);
        }
    }
}

#[test]
fn tiled_2d_matches_per_frame_for_negative_zero_and_extreme_weights() {
    // Negative and signed-zero weights, and weights whose square underflows
    // (1e-170² = 0) or overflows (1e170² = inf).
    let extreme = [
        -0.7, 0.0, -0.0, 1e-170, -1e-170, 1e170, -2.5, 3.0e-160, 0.25,
    ];
    let make = || {
        let mut l = cake_table(true);
        let w = l.values.as_mut().unwrap();
        for (k, v) in w.iter_mut().enumerate() {
            *v = extreme[k % extreme.len()];
        }
        l
    };
    let frames: [Vec<f64>; 4] =
        std::array::from_fn(|f| cake_frame(f).iter().map(|&v| v as f64 * 0.9).collect());
    let norm = cake_norm();
    let mask = cake_mask();
    for m in [None, Some(mask.as_slice())] {
        assert_tiles_match(make, &frames, m, &norm);
    }
}

#[test]
fn tiled_2d_with_every_pixel_masked_gives_zeros_like_per_frame() {
    let frames: [Vec<f64>; 4] =
        std::array::from_fn(|f| cake_frame(f).iter().map(|&v| v as f64).collect());
    let norm = cake_norm();
    let mask = vec![1u8; 40];
    for weighted in [false, true] {
        assert_tiles_match(|| cake_table(weighted), &frames, Some(&mask), &norm);
        // The compacted table keeps every bin but no entries.
        let p = Prepared2d::new(cake_table(weighted));
        let (y, e) = p.integrate::<true, true>(&frames[0], Some(&mask), &norm);
        assert_eq!(y.len(), 24);
        assert!(y.iter().chain(&e).all(|v| v.to_bits() == 0));
    }
}

#[test]
fn tiled_2d_matches_per_frame_for_a_row_with_thousands_of_entries() {
    // Row 0 holds 4321 entries (not a multiple of 4), the others a few.
    let npix = 64;
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for k in 0..4321 {
        col_idx.push((k * 37) % npix);
        values.push(0.05 + (k % 17) as f64 * 0.013);
    }
    row_ptr.push(col_idx.len());
    for r in 1..4 {
        for e in 0..r * 3 {
            col_idx.push((r * 11 + e) % npix);
            values.push(0.3);
        }
        row_ptr.push(col_idx.len());
    }
    let make = |weighted: bool| Lut2d {
        row_ptr: row_ptr.clone(),
        col_idx: col_idx.clone(),
        values: weighted.then(|| values.clone()),
        n_tth: 2,
        n_azimuthal: 2,
    };
    let frames: [Vec<f64>; 4] = std::array::from_fn(|f| {
        (0..npix)
            .map(|i| ((i * (13 + f * 5)) % 29) as f64 - 7.5)
            .collect()
    });
    let norm: Vec<f64> = (0..npix).map(|i| 0.5 + i as f64 * 0.01).collect();
    let mask: Vec<u8> = (0..npix).map(|i| u8::from(i % 9 == 1)).collect();
    for weighted in [false, true] {
        for m in [None, Some(mask.as_slice())] {
            assert_tiles_match(|| make(weighted), &frames, m, &norm);
        }
    }
}
