//! Opt-in reassociated accumulation. Calibration and LUT weights are untouched.
//!
//! Four independent sums expose instruction/vector parallelism. This does not
//! assume finite inputs or enable global LLVM fast-math flags. The normal kernels
//! remain the default; this module is only used by an explicitly configured
//! Integrator, whose cached LUT and input dimensions have already been checked.
#![allow(clippy::too_many_arguments)] // Preserve the previous kernel signatures as test oracles.
use super::{IntermediateResult1d, IntermediateResult2d, Lut1d, Lut2d};
use rayon::prelude::*;

#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    // Avoid an emulated/library FMA in portable builds without hardware support.
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

#[inline(always)]
fn reduce(a: [f64; 4]) -> f64 {
    (a[0] + a[1]) + (a[2] + a[3])
}

#[inline(always)]
fn row<const VAR: bool, const COUNT: bool, const WEIGHT: bool, const MASK: bool>(
    start: usize,
    end: usize,
    cols: &[usize],
    weights: &[f64],
    mask: &[u8],
    signal: &[f64],
    norm: &[f64],
    variance: &[f64],
) -> [f64; 4] {
    let mut s = [0.0; 4];
    let mut n = [0.0; 4];
    let mut v = [0.0; 4];
    let mut c = [0.0; 4];
    let mut k = start;
    // SAFETY: only called for Integrator-built CSR tables, with matching image,
    // correction, variance and mask dimensions. k and k+lane stay inside this
    // row; builder column indices are valid detector indices, as in the strict
    // kernels. The const branches prevent reads from absent optional arrays.
    unsafe {
        while k + 4 <= end {
            for lane in 0..4 {
                let j = *cols.get_unchecked(k + lane);
                if !MASK || *mask.get_unchecked(j) == 0 {
                    let w = if WEIGHT {
                        *weights.get_unchecked(k + lane)
                    } else {
                        1.0
                    };
                    if WEIGHT {
                        s[lane] = madd(w, *signal.get_unchecked(j), s[lane]);
                        n[lane] = madd(w, *norm.get_unchecked(j), n[lane]);
                        if VAR {
                            v[lane] = madd(w * w, *variance.get_unchecked(j), v[lane]);
                        }
                    } else {
                        s[lane] += *signal.get_unchecked(j);
                        n[lane] += *norm.get_unchecked(j);
                        if VAR {
                            v[lane] += *variance.get_unchecked(j);
                        }
                    }
                    if COUNT {
                        c[lane] += w;
                    }
                }
            }
            k += 4;
        }
        for lane in 0..end - k {
            let j = *cols.get_unchecked(k + lane);
            if !MASK || *mask.get_unchecked(j) == 0 {
                let w = if WEIGHT {
                    *weights.get_unchecked(k + lane)
                } else {
                    1.0
                };
                if WEIGHT {
                    s[lane] = madd(w, *signal.get_unchecked(j), s[lane]);
                    n[lane] = madd(w, *norm.get_unchecked(j), n[lane]);
                    if VAR {
                        v[lane] = madd(w * w, *variance.get_unchecked(j), v[lane]);
                    }
                } else {
                    s[lane] += *signal.get_unchecked(j);
                    n[lane] += *norm.get_unchecked(j);
                    if VAR {
                        v[lane] += *variance.get_unchecked(j);
                    }
                }
                if COUNT {
                    c[lane] += w;
                }
            }
        }
    }
    [reduce(s), reduce(n), reduce(v), reduce(c)]
}

#[inline(always)]
fn sum<const VAR: bool, const COUNT: bool>(
    i: usize,
    rows: &[usize],
    cols: &[usize],
    weights: Option<&[f64]>,
    mask: Option<&[u8]>,
    signal: &[f64],
    norm: &[f64],
    variance: &[f64],
) -> [f64; 4] {
    let a = rows[i];
    let b = rows[i + 1];
    match (weights, mask) {
        (Some(w), Some(m)) => {
            row::<VAR, COUNT, true, true>(a, b, cols, w, m, signal, norm, variance)
        }
        (Some(w), None) => {
            row::<VAR, COUNT, true, false>(a, b, cols, w, &[], signal, norm, variance)
        }
        (None, Some(m)) => {
            row::<VAR, COUNT, false, true>(a, b, cols, &[], m, signal, norm, variance)
        }
        (None, None) => {
            row::<VAR, COUNT, false, false>(a, b, cols, &[], &[], signal, norm, variance)
        }
    }
}

fn normalized(
    signal: &[f64],
    rows: &[usize],
    cols: &[usize],
    w: Option<&[f64]>,
    mask: Option<&[u8]>,
    norm: &[f64],
) -> Vec<f64> {
    (0..rows.len() - 1)
        .into_par_iter()
        .map(|i| {
            let [s, n, _, _] = sum::<false, false>(i, rows, cols, w, mask, signal, norm, &[]);
            if n == 0.0 { 0.0 } else { s / n }
        })
        .collect()
}

fn normalized_errors(
    signal: &[f64],
    variance: &[f64],
    rows: &[usize],
    cols: &[usize],
    w: Option<&[f64]>,
    mask: Option<&[u8]>,
    norm: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let results: Vec<_> = (0..rows.len() - 1)
        .into_par_iter()
        .map(|i| {
            let [s, n, v, _] = sum::<true, false>(i, rows, cols, w, mask, signal, norm, variance);
            if n == 0.0 {
                (0.0, 0.0)
            } else {
                (s / n, v / (n * n))
            }
        })
        .collect();
    results
        .into_iter()
        .map(|(s, v)| (s, v.max(0.0).sqrt()))
        .unzip()
}

pub(crate) fn integrate1d(
    s: &[f64],
    l: &Lut1d,
    m: Option<&[u8]>,
    n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    (
        l.bins(),
        normalized(s, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n),
    )
}
pub(crate) fn integrate1d_with_errors(
    s: &[f64],
    v: &[f64],
    l: &Lut1d,
    m: Option<&[u8]>,
    n: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (y, e) = normalized_errors(s, v, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n);
    (l.bins(), y, e)
}
pub(crate) fn integrate2d(s: &[f64], l: &Lut2d, m: Option<&[u8]>, n: &[f64]) -> Vec<f64> {
    normalized(s, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n)
}
pub(crate) fn integrate2d_with_errors(
    s: &[f64],
    v: &[f64],
    l: &Lut2d,
    m: Option<&[u8]>,
    n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    normalized_errors(s, v, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n)
}
fn raw<const VAR: bool>(
    s: &[f64],
    v: &[f64],
    rows: &[usize],
    cols: &[usize],
    w: Option<&[f64]>,
    m: Option<&[u8]>,
    n: &[f64],
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<f64>) {
    let result: Vec<_> = (0..rows.len() - 1)
        .into_par_iter()
        .map(|i| sum::<VAR, true>(i, rows, cols, w, m, s, n, v))
        .collect();
    let mut signal = Vec::with_capacity(result.len());
    let mut norm = Vec::with_capacity(result.len());
    let mut variance = VAR.then(|| Vec::with_capacity(result.len()));
    let mut count = Vec::with_capacity(result.len());
    for [s, n, v, c] in result {
        signal.push(s);
        norm.push(n);
        if let Some(ref mut vv) = variance {
            vv.push(v);
        }
        count.push(c);
    }
    (signal, norm, variance, count)
}
pub(crate) fn integrate1d_raw(
    s: &[f64],
    l: &Lut1d,
    m: Option<&[u8]>,
    n: &[f64],
) -> IntermediateResult1d {
    let (sum_signal, sum_norm, sum_variance, count) =
        raw::<false>(s, &[], &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n);
    IntermediateResult1d {
        bins: l.bins(),
        sum_signal,
        sum_norm,
        sum_variance,
        count,
    }
}
pub(crate) fn integrate1d_raw_with_variance(
    s: &[f64],
    v: &[f64],
    l: &Lut1d,
    m: Option<&[u8]>,
    n: &[f64],
) -> IntermediateResult1d {
    let (sum_signal, sum_norm, sum_variance, count) =
        raw::<true>(s, v, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n);
    IntermediateResult1d {
        bins: l.bins(),
        sum_signal,
        sum_norm,
        sum_variance,
        count,
    }
}
pub(crate) fn integrate2d_raw(
    s: &[f64],
    l: &Lut2d,
    m: Option<&[u8]>,
    n: &[f64],
) -> IntermediateResult2d {
    let (sum_signal, sum_norm, sum_variance, count) =
        raw::<false>(s, &[], &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n);
    IntermediateResult2d {
        sum_signal,
        sum_norm,
        sum_variance,
        count,
        n_radial: l.n_tth,
        n_azimuthal: l.n_azimuthal,
    }
}
pub(crate) fn integrate2d_raw_with_variance(
    s: &[f64],
    v: &[f64],
    l: &Lut2d,
    m: Option<&[u8]>,
    n: &[f64],
) -> IntermediateResult2d {
    let (sum_signal, sum_norm, sum_variance, count) =
        raw::<true>(s, v, &l.row_ptr, &l.col_idx, l.values.as_deref(), m, n);
    IntermediateResult2d {
        sum_signal,
        sum_norm,
        sum_variance,
        count,
        n_radial: l.n_tth,
        n_azimuthal: l.n_azimuthal,
    }
}

#[cfg(test)]
#[path = "fast_tests.rs"]
mod tests;
