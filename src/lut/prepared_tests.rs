use super::*;

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} versus {b}");
    }
}

#[test]
fn prepared_matches_original_strict_and_fast_kernels_bit_for_bit() {
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
                    same(&expected.0, &actual.0);
                    same(&expected.1, &actual.1);
                    same(&expected.2, &actual.2);
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
                    same(&expected.0, &actual.0);
                    same(&expected.1, &actual.1);
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
                    same(&expected.sum_signal, &actual.sum_signal);
                    same(&expected.sum_norm, &actual.sum_norm);
                    same(&expected.count, &actual.count);
                    same(
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
                    same(&expected.0, &actual.0);
                    same(&expected.1, &actual.1);
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
