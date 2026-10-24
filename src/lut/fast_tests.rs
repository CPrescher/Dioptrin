use super::*;
use crate::lut as strict;

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan());
        } else if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b);
        } else {
            assert!(
                (a - b).abs() <= 2e-12 * a.abs().max(b.abs()).max(1.0),
                "{a} != {b}"
            );
        }
    }
}

#[test]
fn weighted_unweighted_masked_tails_empty_rows_and_raw_sums() {
    let s: Vec<_> = (0..73)
        .map(|i| (i as f64 * 0.37).sin() * 15.0 + 20.0)
        .collect();
    let n: Vec<_> = (0..73).map(|i| 0.1 + i as f64 / 73.0).collect();
    let v: Vec<_> = s.iter().map(|x| x.abs() + 0.5).collect();
    let mask: Vec<_> = (0..73).map(|i| u8::from(i % 3 == 0)).collect();
    // Exercise every 4-lane tail, an empty row, and non-contiguous pixel indices.
    let mut rows = vec![0];
    let mut cols = vec![];
    for len in 0..10 {
        for j in 0..len {
            cols.push((j * 17 + len * 5) % 73);
        }
        rows.push(cols.len());
    }
    for weighted in [false, true] {
        let l = Lut1d {
            row_ptr: rows.clone(),
            col_idx: cols.clone(),
            values: weighted.then(|| {
                (0..cols.len())
                    .map(|i| 0.02 + (i % 13) as f64 / 17.0)
                    .collect()
            }),
            hist_min: 0.2,
            hist_step: 0.1,
        };
        let l2 = Lut2d {
            row_ptr: rows.clone(),
            col_idx: cols.clone(),
            values: l.values.clone(),
            n_tth: 5,
            n_azimuthal: 2,
        };
        for masked in [false, true] {
            let m = masked.then_some(mask.as_slice());
            let (x, y) = integrate1d(&s, &l, m, &n);
            let (rx, ry) = strict::integrate1d(&s, &l, m, &n);
            assert_eq!(x, rx);
            close(&y, &ry);
            let (_, a, b) = integrate1d_with_errors(&s, &v, &l, m, &n);
            let (_, ra, rb) = strict::integrate1d_with_errors(&s, &v, &l, m, &n);
            close(&a, &ra);
            close(&b, &rb);
            close(
                &integrate2d(&s, &l2, m, &n),
                &strict::integrate2d(&s, &l2, m, &n),
            );
            let (a, b) = integrate2d_with_errors(&s, &v, &l2, m, &n);
            let (ra, rb) = strict::integrate2d_with_errors(&s, &v, &l2, m, &n);
            close(&a, &ra);
            close(&b, &rb);
            let a = integrate1d_raw(&s, &l, m, &n);
            let b = strict::integrate1d_raw(&s, &l, m, &n);
            close(&a.sum_signal, &b.sum_signal);
            close(&a.sum_norm, &b.sum_norm);
            close(&a.count, &b.count);
            assert!(a.sum_variance.is_none());
            let a = integrate1d_raw_with_variance(&s, &v, &l, m, &n);
            let b = strict::integrate1d_raw_with_variance(&s, &v, &l, m, &n);
            close(&a.sum_signal, &b.sum_signal);
            close(&a.sum_norm, &b.sum_norm);
            close(
                a.sum_variance.as_ref().unwrap(),
                b.sum_variance.as_ref().unwrap(),
            );
            close(&a.count, &b.count);
            let a = integrate2d_raw(&s, &l2, m, &n);
            let b = strict::integrate2d_raw(&s, &l2, m, &n);
            close(&a.sum_signal, &b.sum_signal);
            close(&a.sum_norm, &b.sum_norm);
            close(&a.count, &b.count);
            assert_eq!(a.n_radial, 5);
            assert_eq!(a.n_azimuthal, 2);
            let a = integrate2d_raw_with_variance(&s, &v, &l2, m, &n);
            let b = strict::integrate2d_raw_with_variance(&s, &v, &l2, m, &n);
            close(&a.sum_signal, &b.sum_signal);
            close(&a.sum_norm, &b.sum_norm);
            close(
                a.sum_variance.as_ref().unwrap(),
                b.sum_variance.as_ref().unwrap(),
            );
            close(&a.count, &b.count);
        }
    }
}

#[test]
fn zero_normalization_and_all_masked_are_zero() {
    let l = Lut1d {
        row_ptr: vec![0, 5],
        col_idx: (0..5).collect(),
        values: Some(vec![0.2; 5]),
        hist_min: 0.0,
        hist_step: 1.0,
    };
    for (m, n) in [(None, vec![0.0; 5]), (Some(vec![1u8; 5]), vec![1.0; 5])] {
        let (_, y, e) = integrate1d_with_errors(&[2.; 5], &[2.; 5], &l, m.as_deref(), &n);
        assert_eq!(y, vec![0.]);
        assert_eq!(e, vec![0.]);
    }
}

#[test]
fn exceptional_values_are_not_assumed_finite_and_masked_nan_is_ignored() {
    let l = Lut1d {
        row_ptr: vec![0, 5],
        col_idx: (0..5).collect(),
        values: Some(vec![0.2; 5]),
        hist_min: 0.0,
        hist_step: 1.0,
    };
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let s = [1., 2., x, 4., 5.];
        let (_, a) = integrate1d(&s, &l, None, &[1.; 5]);
        let (_, b) = strict::integrate1d(&s, &l, None, &[1.; 5]);
        close(&a, &b);
        let (_, a) = integrate1d(&s, &l, Some(&[0, 0, 1, 0, 0]), &[1.; 5]);
        assert!(a[0].is_finite());
    }
}
