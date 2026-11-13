#![allow(dead_code)]

use std::sync::mpsc;
use std::thread;

use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::{lut, math};

/// Calculates a normalized histogram of weights at data values, optionally excluding masked pixels.
pub fn histogram(
    data: &[f64],
    weights: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    // calculate bin index for each data point
    let index: Vec<usize> = data
        .par_iter()
        .map(|x| get_bin_clamped(*x, min, step, n_points))
        .collect();

    // Calulate the histogram
    match mask {
        Some(mask) => {
            for (i, y) in weights.iter().enumerate() {
                if mask[i] != 0 {
                    continue;
                }
                sum[index[i]] += y;
                count[index[i]] += 1.0;
            }
        }
        None => {
            for (i, y) in weights.iter().enumerate() {
                sum[index[i]] += y;
                count[index[i]] += 1.0;
            }
        }
    }

    // normalize to amount of points
    let normed_sum = match mask {
        Some(_) => sum
            .iter()
            .zip(count.iter())
            .map(|(sum, count)| if *count == 0.0 { 0.0 } else { sum / count })
            .collect::<Vec<f64>>(),
        None => sum
            .iter()
            .zip(count.iter())
            .map(|(sum, count)| sum / count)
            .collect::<Vec<f64>>(),
    };

    let bins = bins(n_points, min, step);
    (bins, normed_sum)
}

/// Calculates a normalized histogram of weights with per-bin errors using provided variances.
pub fn histogram_with_errors(
    data: &[f64],
    weights: &[f64],
    variances: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);

    let mut sum = vec![0.0; n_points];
    let mut sum_var = vec![0.0; n_points];
    let mut count = vec![0usize; n_points];

    let index: Vec<usize> = data
        .par_iter()
        .map(|x| get_bin_clamped(*x, min, step, n_points))
        .collect();

    match mask {
        Some(mask) => {
            for (i, y) in weights.iter().enumerate() {
                if mask[i] != 0 {
                    continue;
                }
                let idx = index[i];
                sum[idx] += y;
                sum_var[idx] += variances[i];
                count[idx] += 1;
            }
        }
        None => {
            for (i, y) in weights.iter().enumerate() {
                let idx = index[i];
                sum[idx] += y;
                sum_var[idx] += variances[i];
                count[idx] += 1;
            }
        }
    }

    let mut normed_sum = vec![0.0; n_points];
    let mut sigma = vec![0.0; n_points];
    for i in 0..n_points {
        let c = count[i];
        if c == 0 {
            continue;
        }
        let c_f = c as f64;
        normed_sum[i] = sum[i] / c_f;
        let var = sum_var[i] / (c_f * c_f);
        sigma[i] = var.max(0.0).sqrt();
    }

    let bins = bins(n_points, min, step);
    (bins, normed_sum, sigma)
}

/// Calculates a normalized histogram of weights at data values using a parallel implementation
pub fn histogram_parallel(
    data: &[f64],
    weights: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];

    let (tx, rx) = mpsc::channel();

    let n_threads = 16;
    let chunk_size = data.len() / n_threads;

    for i in 0..n_threads {
        let tx_thread = mpsc::Sender::clone(&tx);
        let data_thread = data[i * chunk_size..(i + 1) * chunk_size].to_vec();
        let weights_thread = weights[i * chunk_size..(i + 1) * chunk_size].to_vec();
        let mask_thread = mask.map(|mask| mask[i * chunk_size..(i + 1) * chunk_size].to_vec());

        thread::spawn(move || {
            let mut sum_thread = vec![0.0; n_points];
            let mut count_thread = vec![0.0; n_points];

            match mask_thread.as_ref() {
                Some(mask_thread) => {
                    for (i, y) in data_thread.iter().enumerate() {
                        if mask_thread[i] != 0 {
                            continue;
                        }
                        let index = get_bin_clamped(*y, min, step, n_points);
                        sum_thread[index] += weights_thread[i];
                        count_thread[index] += 1.0;
                    }
                }
                None => {
                    for (i, y) in data_thread.iter().enumerate() {
                        let index = get_bin_clamped(*y, min, step, n_points);
                        sum_thread[index] += weights_thread[i];
                        count_thread[index] += 1.0;
                    }
                }
            }

            tx_thread.send((sum_thread, count_thread)).unwrap();
        });
    }
    drop(tx);

    for (sum_thread, count_thread) in rx {
        for i in 0..n_points {
            sum[i] += sum_thread[i];
            count[i] += count_thread[i];
        }
    }

    // normalize to amount of points
    let normed_sum = match mask {
        Some(_) => sum
            .iter()
            .zip(count.iter())
            .map(|(sum, count)| if *count == 0.0 { 0.0 } else { sum / count })
            .collect::<Vec<f64>>(),
        None => sum
            .iter()
            .zip(count.iter())
            .map(|(sum, count)| sum / count)
            .collect::<Vec<f64>>(),
    };

    // create tth values for bin centers
    let bin = bins(n_points, min, step);
    (bin, normed_sum)
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
        let len = 4800; // a multiple of histogram_parallel's 16 threads
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
                    let (_, y_par) = histogram_parallel(&data, &weights, n, mask);
                    for y in [&y, &y_err, &y_par] {
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
