#![allow(dead_code)]

use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::{lut, math};

/// Calculates a normalized histogram of weights at data values, optionally excluding masked pixels.
///
/// # Panics
///
/// If `weights` and `data` differ in length, or `mask` is shorter than them.
pub fn histogram(
    data: &[f64],
    weights: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
    check_lengths(data, weights);
    let (min, _max, step) = min_max_step(data, n_points);

    // calculate bin index for each data point
    let index: Vec<usize> = data
        .par_iter()
        .map(|x| get_bin_clamped(*x, min, step, n_points))
        .collect();

    // Calculate the histogram, with (sum, count) of a bin next to each other
    let mut acc = vec![[0.0f64; 2]; n_points];
    for (i, (&idx, y)) in index.iter().zip(weights).enumerate() {
        if mask.is_some_and(|m| m[i] != 0) {
            continue;
        }
        let bin = &mut acc[idx];
        bin[0] += y;
        bin[1] += 1.0;
    }

    // normalize to amount of points
    let normed_sum = acc
        .iter()
        .map(|&[sum, count]| {
            if mask.is_some() && count == 0.0 {
                0.0
            } else {
                sum / count
            }
        })
        .collect();

    let bins = bins(n_points, min, step);
    (bins, normed_sum)
}

/// Calculates a normalized histogram of weights with per-bin errors using provided variances.
///
/// # Panics
///
/// If `weights` and `data`, or `variances` and `weights`, differ in length,
/// or `mask` is shorter than them.
pub fn histogram_with_errors(
    data: &[f64],
    weights: &[f64],
    variances: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    check_lengths(data, weights);
    assert_eq!(
        variances.len(),
        weights.len(),
        "histogram: variances must have one value per weight"
    );
    let (min, _max, step) = min_max_step(data, n_points);

    let index: Vec<usize> = data
        .par_iter()
        .map(|x| get_bin_clamped(*x, min, step, n_points))
        .collect();

    // (sum, variance sum, count) of a bin next to each other; counts stay
    // exact in f64
    let mut acc = vec![[0.0f64; 3]; n_points];
    for (i, ((&idx, y), v)) in index.iter().zip(weights).zip(variances).enumerate() {
        if mask.is_some_and(|m| m[i] != 0) {
            continue;
        }
        let bin = &mut acc[idx];
        bin[0] += y;
        bin[1] += v;
        bin[2] += 1.0;
    }

    let mut normed_sum = vec![0.0; n_points];
    let mut sigma = vec![0.0; n_points];
    for (i, &[sum, sum_var, c_f]) in acc.iter().enumerate() {
        if c_f == 0.0 {
            continue;
        }
        normed_sum[i] = sum / c_f;
        let var = sum_var / (c_f * c_f);
        sigma[i] = var.max(0.0).sqrt();
    }

    let bins = bins(n_points, min, step);
    (bins, normed_sum, sigma)
}

fn check_lengths(data: &[f64], weights: &[f64]) {
    assert_eq!(
        weights.len(),
        data.len(),
        "histogram: weights must have one value per data value"
    );
}

/// Calculates the bin index for a value in a histogram
/// min is the min value of the data, step is the bin width
/// Note: This can return an index >= n_bins if value >= max, and even for the
/// maximum of the data a grid was derived from, because `(max - min) / step`
/// can round to `n_bins`. Use [`get_bin_clamped`] to index a histogram.
pub fn get_bin(value: f64, min: f64, step: f64) -> usize {
    ((value - min) / step) as usize
}

/// Calculates the bin index for a value in a histogram, clamped to valid range
/// min is the min value of the data, step is the bin width, n_bins is the total number of bins
pub fn get_bin_clamped(value: f64, min: f64, step: f64, n_bins: usize) -> usize {
    let bin = ((value - min) / step) as usize;
    bin.min(n_bins - 1)
}

/// Calculates the bin index for a value on an explicit grid of `n_bins` bins
/// covering `[min, max]`. Values outside that closed interval (and NaN) belong
/// to no bin and return `None`; they are not moved into the first or last bin.
/// A value at `max` belongs to the last bin, whatever `(max - min) / step`
/// rounds to.
pub fn get_bin_in_range(value: f64, min: f64, max: f64, step: f64, n_bins: usize) -> Option<usize> {
    if !(value >= min && value <= max) {
        return None;
    }
    Some(get_bin_clamped(value, min, step, n_bins))
}

/// Calculates a wrapped bin index for a value in a periodic histogram.
///
/// The returned index is in `0..n_bins` using Euclidean wrapping.
pub fn get_bin_wrapped(value: f64, min: f64, step: f64, n_bins: usize) -> usize {
    let raw = ((value - min) / step).floor() as isize;
    raw.rem_euclid(n_bins as isize) as usize
}

pub fn ss_tth_min_max_step(
    geometry: &Geometry,
    radial: &[f64],
    n_points: usize,
    n_ss: usize,
    unit: crate::integration::RadialUnit,
) -> (f64, f64, f64) {
    let (hull_d1, hull_d2) = lut::hull_grid(geometry.dim1, geometry.dim2, n_ss);
    let hull_tth = geometry.calc_tth(&hull_d1, &hull_d2);
    let hull_radial = unit.convert_from_tth(&hull_tth, geometry.poni.wavelength);

    let min = math::min(radial).min(math::min(&hull_radial));
    let max = math::max(radial).max(math::max(&hull_radial)) + 3.0 * f64::EPSILON;
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

pub fn supersampled_azimuthal_min_max_step(
    geometry: &Geometry,
    azimuthal: &[f64],
    n_points: usize,
    n_subpixels: usize,
) -> (f64, f64, f64) {
    let (hull_d1, hull_d2) = lut::hull_grid(geometry.dim1, geometry.dim2, n_subpixels);
    let hull_azimuthal = geometry.calc_azimuth(&hull_d1, &hull_d2);

    let min = math::min_slice(azimuthal).min(math::min(&hull_azimuthal));
    let max = math::max_slice(azimuthal).max(math::max(&hull_azimuthal)) + 3.0 * f64::EPSILON;
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

/// Computes min, step for a histogram with an explicit range.
pub fn min_max_step_from_range(min: f64, max: f64, n_points: usize) -> (f64, f64, f64) {
    let max = max + 10.0 * f64::EPSILON;
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

/// Estimates min, max and step size for a histogram
pub fn min_max_step(data: &[f64], n_points: usize) -> (f64, f64, f64) {
    let min = math::min(data);
    let max = math::max(data) + 10.0 * f64::EPSILON; // the +10.0 * epsilon is a hack to avoid rounding errors
    let step = (max - min) / n_points as f64;
    (min, max, step)
}

/// Calculates the bin centers for a histogram
pub fn bins(n_points: usize, min: f64, step: f64) -> Vec<f64> {
    (0..n_points)
        .map(|i| min + ((i as f64) + 0.5) * step)
        .collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "weights must have one value per data value")]
    fn histogram_rejects_weights_longer_than_data() {
        histogram(&[0.0; 10], &[1.0; 12], 4, None);
    }

    #[test]
    #[should_panic(expected = "weights must have one value per data value")]
    fn histogram_with_errors_rejects_weights_longer_than_data() {
        histogram_with_errors(&[0.0; 10], &[1.0; 12], &[1.0; 12], 4, None);
    }

    #[test]
    #[should_panic(expected = "variances must have one value per weight")]
    fn histogram_with_errors_rejects_short_variances() {
        histogram_with_errors(&[0.0; 10], &[1.0; 10], &[1.0; 8], 4, None);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn histogram_rejects_short_mask() {
        histogram(&[0.0; 10], &[1.0; 10], 4, Some(&[0; 8]));
    }

    #[test]
    fn histogram_bins_every_element() {
        // The last 4 of 20 values carry a large weight and must all be binned.
        let data = ramp(2.0, 20);
        let mut weights = vec![1.0; 20];
        weights[16..].fill(100.0);
        let (_, y) = histogram(&data, &weights, 2, None);
        assert_eq!(y, vec![1.0, 40.6]);
    }

    #[test]
    fn histogram_ignores_mask_values_past_the_data() {
        // A longer mask is accepted and its extra values are not used.
        let data = [0.0, 1.0, 2.0, 3.0];
        let weights = [1.0, 2.0, 3.0, 4.0];
        let (_, y) = histogram(&data, &weights, 2, Some(&[0, 0, 0, 0, 1, 1]));
        assert_eq!(y, [1.5, 3.5]);
    }

    #[test]
    fn histogram_with_errors_skips_masked_values() {
        // Two bins: 0 and 1 fall in the first, 2 and 3 in the second.
        let data = [0.0, 1.0, 2.0, 3.0];
        let weights = [1.0, 2.0, 3.0, 4.0];
        let variances = [1.0, 4.0, 9.0, 16.0];
        let (_, y, sigma) =
            histogram_with_errors(&data, &weights, &variances, 2, Some(&[0, 1, 0, 0]));
        assert_eq!(y, [1.0, 3.5]);
        assert_eq!(sigma, [1.0, 2.5]);
        // A bin whose values are all masked reports zero, not NaN.
        let (_, y, sigma) =
            histogram_with_errors(&data, &weights, &variances, 2, Some(&[1, 1, 0, 0]));
        assert_eq!(y, [0.0, 3.5]);
        assert_eq!(sigma, [0.0, 2.5]);
    }

    /// Evenly spaced values from 1 to `hi`; the last one is the maximum, which
    /// can round to bin index `n_points` once `hi` is too large for the
    /// epsilon padding of the range to have any effect.
    fn ramp(hi: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 1.0 + (hi - 1.0) * i as f64 / (len - 1) as f64)
            .collect()
    }

    #[test]
    fn maximum_value_lands_in_the_last_bin() {
        let len = 4800;
        for hi in [3.0, 30.8, 60.0, 1234.5] {
            let data = ramp(hi, len);
            // Mark the maximum so that the bin holding it is recognizable.
            let mut weights = vec![1.0; len];
            weights[len - 1] = 1e6;
            let variances = vec![1.0; len];
            let mask = vec![0u8; len];
            for n in [7, 100, 1000, 1500, 3000] {
                for mask in [None, Some(mask.as_slice())] {
                    let (_, y) = histogram(&data, &weights, n, mask);
                    let (_, y_err, sigma) =
                        histogram_with_errors(&data, &weights, &variances, n, mask);
                    for y in [&y, &y_err] {
                        assert_eq!(y.len(), n);
                        assert!(y[n - 1] > 1e3, "hi {hi}, n {n}: {}", y[n - 1]);
                        assert!(y[..n - 1].iter().all(|&v| v <= 1.0), "hi {hi}, n {n}");
                    }
                    assert!(sigma[n - 1] > 0.0, "hi {hi}, n {n}");
                }
            }
        }
    }
}
