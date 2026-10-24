//! Optional population spread of corrected pixels within each radial bin.
use rayon::prelude::*;

use super::Lut1d;

/// A 1D pattern with azimuthal spread, optionally accompanied by Poisson errors.
#[derive(Clone, Debug)]
pub struct AzimuthalIntegrationResult {
    pub radial: Vec<f64>,
    pub intensity: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub azimuthal_std: Vec<f64>,
}

/// Calculate the weighted population standard deviation of corrected intensities.
///
/// For signal `s`, normalization `n`, and pixel fraction `c`, use `x=s/n` and
/// weight `w=c*n`. Thus the mean remains `sum(c*s)/sum(c*n)` and the variance is
/// `sum(w*(x-mean)^2)/sum(w)`. This is neither RMS nor uncertainty in the mean.
/// The optional input variance is propagated independently with squared pixel
/// fractions, exactly as in `integrate1d_with_errors`.
///
/// Masked/nonfinite pixels, nonpositive normalizations and nonpositive fractions
/// are excluded. Empty bins have zero intensity/errors and NaN spread; one valid
/// contribution has zero spread. A shifted two-pass algorithm avoids subtracting
/// large squared moments. Split pixels are contributions, not extra observations.
pub fn integrate1d_with_azimuthal_std(
    signal: &[f64],
    lut: &Lut1d,
    mask: Option<&[u8]>,
    normalization: &[f64],
    variance: Option<&[f64]>,
) -> AzimuthalIntegrationResult {
    integrate_with_columns(
        signal,
        &lut.row_ptr,
        |k| lut.col_idx[k],
        lut.values.as_deref(),
        lut.bins(),
        mask,
        normalization,
        variance,
    )
}

// Pass the existing CSR components directly so both index widths share one kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_with_columns(
    signal: &[f64],
    row_ptr: &[usize],
    column: impl Fn(usize) -> usize + Sync,
    values: Option<&[f64]>,
    bins: Vec<f64>,
    mask: Option<&[u8]>,
    normalization: &[f64],
    variance: Option<&[f64]>,
) -> AzimuthalIntegrationResult {
    let rows: Vec<(f64, f64, f64)> = (0..row_ptr.len() - 1)
        .into_par_iter()
        .map(|i| {
            let contributions = || {
                (row_ptr[i]..row_ptr[i + 1]).filter_map(|k| {
                    let j = column(k);
                    if mask.is_some_and(|m| m[j] != 0) {
                        return None;
                    }
                    let n = normalization[j];
                    let c = values.map_or(1.0, |v| v[k]);
                    let s = signal[j];
                    let x = s / n;
                    let w = c * n;
                    (s.is_finite()
                        && n.is_finite()
                        && n > 0.0
                        && c.is_finite()
                        && c > 0.0
                        && x.is_finite()
                        && w.is_finite()
                        && w > 0.0)
                        .then_some((j, c, s, x, w))
                })
            };
            let Some((_, _, _, origin, _)) = contributions().next() else {
                return (0.0, 0.0, f64::NAN);
            };
            let mut sum_signal = 0.0;
            let mut sum_weight = 0.0;
            let mut sum_shifted = 0.0;
            let mut sum_variance = 0.0;
            for (j, c, s, x, w) in contributions() {
                sum_signal += c * s;
                sum_weight += w;
                sum_shifted += w * (x - origin);
                if let Some(v) = variance {
                    sum_variance += c * c * v[j];
                }
            }
            let mean_shifted = sum_shifted / sum_weight;
            let m2: f64 = contributions()
                .map(|(_, _, _, x, w)| {
                    let delta = (x - origin) - mean_shifted;
                    w * delta * delta
                })
                .sum();
            (
                sum_signal / sum_weight,
                sum_variance.max(0.0).sqrt() / sum_weight,
                (m2 / sum_weight).max(0.0).sqrt(),
            )
        })
        .collect();
    AzimuthalIntegrationResult {
        radial: bins,
        intensity: rows.iter().map(|r| r.0).collect(),
        errors: variance.map(|_| rows.iter().map(|r| r.1).collect()),
        azimuthal_std: rows.iter().map(|r| r.2).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_bin(values: Option<Vec<f64>>) -> Lut1d {
        Lut1d {
            row_ptr: vec![0, 3],
            col_idx: vec![0, 1, 2],
            values,
            hist_min: 0.0,
            hist_step: 1.0,
        }
    }

    #[test]
    fn corrected_weighted_population_and_independent_errors() {
        // Corrected intensities [2, 4, 8], weights [1, 1, 2].
        let lut = one_bin(Some(vec![0.5, 0.25, 1.0]));
        let r = integrate1d_with_azimuthal_std(
            &[4.0, 16.0, 16.0],
            &lut,
            None,
            &[2.0, 4.0, 2.0],
            Some(&[4.0, 16.0, 16.0]),
        );
        assert_eq!(r.intensity, vec![5.5]);
        assert!((r.azimuthal_std[0] - 6.75_f64.sqrt()).abs() < 1e-12);
        assert!((r.errors.unwrap()[0] - 18.0_f64.sqrt() / 4.0).abs() < 1e-12);
    }

    #[test]
    fn stable_at_large_offset_and_invariant_to_intensity_scaling() {
        let lut = one_bin(None);
        let r = integrate1d_with_azimuthal_std(
            &[1e12 - 1.0, 1e12, 1e12 + 1.0],
            &lut,
            None,
            &[1.0; 3],
            None,
        );
        assert!((r.azimuthal_std[0] - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        let scaled =
            integrate1d_with_azimuthal_std(&[6.0, 12.0, 24.0], &lut, None, &[1.0; 3], None);
        assert!((scaled.azimuthal_std[0] - 56.0_f64.sqrt()).abs() < 1e-12);
        assert!(scaled.errors.is_none());
    }

    #[test]
    fn masks_invalid_pixels_and_empty_bins() {
        let lut = one_bin(None);
        let r = integrate1d_with_azimuthal_std(
            &[10.0, 1e6, f64::NAN],
            &lut,
            Some(&[0, 1, 0]),
            &[2.0; 3],
            None,
        );
        assert_eq!(r.intensity, vec![5.0]);
        assert_eq!(r.azimuthal_std, vec![0.0]);
        let r = integrate1d_with_azimuthal_std(
            &[10.0; 3],
            &lut,
            Some(&[1; 3]),
            &[1.0; 3],
            Some(&[10.0; 3]),
        );
        assert_eq!(r.intensity, vec![0.0]);
        assert!(r.azimuthal_std[0].is_nan());
        assert_eq!(r.errors.unwrap(), vec![0.0]);
        let r = integrate1d_with_azimuthal_std(
            &[10.0; 3],
            &lut,
            None,
            &[0.0, -1.0, f64::INFINITY],
            None,
        );
        assert!(r.azimuthal_std[0].is_nan());
    }

    #[test]
    fn empty_rows_and_zero_fraction_do_not_report_smooth_signal() {
        let lut = Lut1d {
            row_ptr: vec![0, 0, 1],
            col_idx: vec![0],
            values: Some(vec![0.0]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let r = integrate1d_with_azimuthal_std(&[42.0], &lut, None, &[1.0], None);
        assert_eq!(r.intensity, vec![0.0, 0.0]);
        assert!(r.azimuthal_std.iter().all(|s| s.is_nan()));
    }

    #[test]
    fn weighted_spread_is_invariant_to_pixel_order() {
        let lut = one_bin(Some(vec![0.1, 0.5, 1.0]));
        let a = integrate1d_with_azimuthal_std(
            &[-10.0, 60.0, 80.0],
            &lut,
            None,
            &[1.0, 2.0, 4.0],
            None,
        );
        let reversed = Lut1d {
            row_ptr: vec![0, 3],
            col_idx: vec![2, 1, 0],
            values: Some(vec![1.0, 0.5, 0.1]),
            hist_min: 0.0,
            hist_step: 1.0,
        };
        let b = integrate1d_with_azimuthal_std(
            &[-10.0, 60.0, 80.0],
            &reversed,
            None,
            &[1.0, 2.0, 4.0],
            None,
        );
        assert!((a.azimuthal_std[0] - b.azimuthal_std[0]).abs() < 1e-12);
    }

    #[test]
    fn constant_corrected_signal_has_zero_spread() {
        let r = integrate1d_with_azimuthal_std(
            &[20.0, 40.0, 80.0],
            &one_bin(Some(vec![0.2, 0.7, 1.0])),
            None,
            &[1.0, 2.0, 4.0],
            None,
        );
        assert_eq!(r.intensity, vec![20.0]);
        assert_eq!(r.azimuthal_std, vec![0.0]);
    }
}
