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
fn switching_policy_releases_inactive_layout_and_corrections_preserve_fixed_table() {
    let mut p = Prepared1d::new(Lut1d {
        row_ptr: vec![0, 2],
        col_idx: vec![0, 1],
        values: Some(vec![0.3, 0.7]),
        hist_min: 0.,
        hist_step: 1.,
    });
    p.prepare_normalization(Some(&[0, 1]), &[1., 1.], true);
    let fixed = p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize;
    p.invalidate_normalization();
    p.prepare_normalization(Some(&[0, 1]), &[2., 2.], true);
    assert_eq!(
        p.fixed.get().unwrap().as_ref().unwrap().as_ref() as *const PreparedCsr as usize,
        fixed
    );
    let out = p.integrate::<true, true>(&[4., 8.], Some(&[0, 1]), &[2., 2.]);
    p.set_mask_mode(MaskMode::Dynamic);
    assert!(p.fixed.get().is_none());
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
    assert!(p.csc.get().is_none());
    p.prepare_normalization(Some(&[0, 1]), &[2., 2.], true);
    assert!(p.fixed.get().unwrap().is_some());
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
