#![allow(dead_code)]

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

/// Calculates a normalized histogram of weights at data values using a parallel implementation.
///
/// Gives the same bins and counts as [`histogram`]; the per-bin sums hold the
/// same terms but are added in a different order, so they can differ from it
/// by floating-point rounding (relatively much more when weights within a bin
/// nearly cancel). The input is split into chunks whose size depends only on
/// the input length and `n_points`, and their partial sums are combined in
/// chunk order, so the result is bit-identical from run to run and for any
/// thread count.
pub fn histogram_parallel(
    data: &[f64],
    weights: &[f64],
    n_points: usize,
    mask: Option<&[u8]>,
) -> (Vec<f64>, Vec<f64>) {
    let (min, _max, step) = min_max_step(data, n_points);
    let chunk = parallel_chunk_len(data.len(), n_points);

    // One (sum, count) pair per bin, so that each value touches one cache line.
    let partials: Vec<Vec<[f64; 2]>> = data
        .par_chunks(chunk)
        .zip(weights.par_chunks(chunk))
        .enumerate()
        .map(|(c, (data, weights))| {
            let mask = mask.map(|m| &m[c * chunk..c * chunk + data.len()]);
            let mut acc = vec![[0.0; 2]; n_points];
            for (i, (x, y)) in data.iter().zip(weights).enumerate() {
                if mask.is_some_and(|m| m[i] != 0) {
                    continue;
                }
                let bin = &mut acc[get_bin_clamped(*x, min, step, n_points)];
                bin[0] += y;
                bin[1] += 1.0;
            }
            acc
        })
        .collect();

    let mut sum = vec![0.0; n_points];
    let mut count = vec![0.0; n_points];
    for partial in &partials {
        for (i, [s, c]) in partial.iter().enumerate() {
            sum[i] += s;
            count[i] += c;
        }
    }

    // normalize to amount of points
    let normed_sum = sum
        .iter()
        .zip(&count)
        .map(|(sum, count)| {
            if mask.is_some() && *count == 0.0 {
                0.0
            } else {
                sum / count
            }
        })
        .collect();

    // create tth values for bin centers
    (bins(n_points, min, step), normed_sum)
}

/// Chunk length for [`histogram_parallel`]. It depends only on the input
/// length and the bin count, never on the thread count, which keeps the
/// result deterministic. There are at most `MAX_PARTIALS` chunks, so memory
/// stays bounded, and each chunk holds at least `VALUES_PER_BIN` values per
/// bin, so allocating and merging its partial histogram stays small next to
/// binning the chunk.
fn parallel_chunk_len(len: usize, n_points: usize) -> usize {
    const MIN_CHUNK: usize = 1 << 14;
    const MAX_PARTIALS: usize = 32;
    const VALUES_PER_BIN: usize = 8;
    MIN_CHUNK
        .max(len.div_ceil(MAX_PARTIALS))
        .max(n_points.saturating_mul(VALUES_PER_BIN))
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

    /// NaN-aware exact comparison.
    fn assert_same(a: &[f64], b: &[f64], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x == y || (x.is_nan() && y.is_nan()),
                "{ctx}, bin {i}: {x} != {y}"
            );
        }
    }

    /// Irregularly spread values, varying weights and a mask of every fifth value.
    /// The weights are multiples of 0.25, so every bin sum is exact whatever
    /// the summation order and the histograms can be compared exactly.
    fn sample(len: usize) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
        let data = (0..len)
            .map(|i| 5.0 + 3.0 * ((i as f64) * 0.37).sin())
            .collect();
        let weights = (0..len).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
        let mask = (0..len).map(|i| u8::from(i % 5 == 3)).collect();
        (data, weights, mask)
    }

    #[test]
    fn parallel_histogram_bins_every_element() {
        // The reported case: the last 4 of 20 values carry a large weight.
        let data = ramp(2.0, 20);
        let mut weights = vec![1.0; 20];
        weights[16..].fill(100.0);
        let (_, y) = histogram(&data, &weights, 2, None);
        let (_, y_par) = histogram_parallel(&data, &weights, 2, None);
        assert_eq!(y, vec![1.0, 40.6]);
        assert_same(&y_par, &y, "len 20");

        // Lengths that are not multiples of 16, including lengths below 16;
        // lengths around the minimum chunk of 16384 values; lengths around
        // 32 * 16384, above which the chunk grows with the input; and, with
        // 4096 bins, lengths around the chunk of 8 * 4096 values.
        let lengths = [
            1, 2, 5, 15, 17, 20, 33, 1001, 16_383, 16_384, 16_385, 32_768, 32_769, 40_007, 65_537,
            100_003, 524_288, 524_289,
        ];
        for len in lengths {
            let (data, weights, mask) = sample(len);
            for n in [1, 2, 7, 100, 4096] {
                for mask in [None, Some(mask.as_slice())] {
                    let (x, y) = histogram(&data, &weights, n, mask);
                    let (x_par, y_par) = histogram_parallel(&data, &weights, n, mask);
                    let ctx = format!("len {len}, n {n}, mask {}", mask.is_some());
                    assert_eq!(x_par, x, "{ctx}");
                    assert_same(&y_par, &y, &ctx);
                }
            }
        }
        assert_eq!(parallel_chunk_len(524_288, 7), 16_384);
        assert_eq!(parallel_chunk_len(524_289, 7), 16_385);
        assert_eq!(parallel_chunk_len(32_769, 4096), 32_768);
    }

    #[test]
    fn parallel_histogram_is_identical_for_any_thread_count() {
        let (data, _, mask) = sample(1_000_003);
        // Weights whose sums round, so that a different summation order shows.
        let weights: Vec<f64> = (0..data.len())
            .map(|i| 1.0 + 0.3 * ((i as f64) * 0.11).cos())
            .collect();
        for n in [7, 4096, 200_000] {
            for mask in [None, Some(mask.as_slice())] {
                let run = |threads: usize| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let (x, y) = pool.install(|| histogram_parallel(&data, &weights, n, mask));
                    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
                    (bits(x), bits(y))
                };
                let reference = run(1);
                for threads in [2, 3, 8] {
                    assert!(
                        run(threads) == reference,
                        "n {n}, mask {}, {threads} threads",
                        mask.is_some()
                    );
                }
            }
        }
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
