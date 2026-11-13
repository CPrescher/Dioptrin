//! Massif peak picking — port of pyFAI's `pyFAI.massif.Massif.find_peaks`.
//!
//! Algorithm:
//!
//! 1. Gaussian-blur the image to estimate the local background.
//! 2. Binarize as `image > blurred` — each connected blob of `true` pixels is
//!    one "massif" (i.e. one ring fragment, since powder-ring intensity sits
//!    above the local background).
//! 3. Label the binary mask into connected components (4-connectivity).
//! 4. The "region" associated with a seed pixel is the component the seed
//!    lives in.
//! 5. Walk uphill from the seed to its integer-pixel local maximum; refine to
//!    sub-pixel via a quadratic fit on the 3×3 neighbourhood. That is the
//!    seed's parent peak.
//! 6. For every pixel inside the region whose intensity is above the region
//!    mean, walk uphill and dedupe — every distinct sub-peak is returned.
//!
//! Background estimation retains full resolution. The separable convolution
//! processes contiguous row slices in parallel while preserving the summation
//! order for each output pixel. `PreparedMassif` reuses that background and
//! connected-component labeling across seeds on an unchanged image.

#![allow(clippy::needless_range_loop)]

#[derive(Debug, Clone, Copy)]
pub struct MassifOptions {
    /// Standard deviation of the Gaussian background filter, in pixels.
    /// `None` selects pyFAI's adaptive value: `max(5, max(dim1, dim2) / 50)`.
    /// For a 2048×2048 image this gives ≈ 41 px.
    pub valley_size: Option<f64>,
    /// Maximum number of sub-peaks to return per call.
    pub max_peaks: usize,
    /// Sub-peaks within this Euclidean pixel distance of an existing peak are
    /// merged into the existing one.
    pub dedup_dist: f64,
}

impl Default for MassifOptions {
    fn default() -> Self {
        Self {
            valley_size: None,
            max_peaks: 200,
            dedup_dist: 1.0,
        }
    }
}

/// Resolve the effective Gaussian σ — matches pyFAI's `init_valley_size`.
fn effective_valley_size(opts: &MassifOptions, dim1: usize, dim2: usize) -> f64 {
    match opts.valley_size {
        Some(v) => v,
        None => 5.0_f64.max(dim1.max(dim2) as f64 / 50.0),
    }
}

// ---------- Gaussian blur (separable 1D, reflect-padded) ----------

fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    // pyFAI's gaussian_filter is scipy's default: kernel half-width = ceil(4σ).
    let half = (4.0 * sigma).ceil() as usize;
    let n = 2 * half + 1;
    let two_s2 = 2.0 * sigma * sigma;
    let mut k = Vec::with_capacity(n);
    let mut sum = 0.0;
    for i in 0..n {
        let x = i as f64 - half as f64;
        let v = (-x * x / two_s2).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

use rayon::prelude::*;

// Retain the original reflection/clamping rule, including kernels wider than
// the image. The common interior is handled separately without index checks.
fn reflected(index: isize, size: usize) -> usize {
    let mut index = index;
    if index < 0 {
        index = -index;
    }
    if index >= size as isize {
        index = 2 * size as isize - 2 - index;
    }
    index.clamp(0, size as isize - 1) as usize
}

fn convolve_horizontal(input: &[f64], dim1: usize, dim2: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut out = vec![0.0; dim1 * dim2];
    if dim2 == 0 {
        return out;
    }
    out.par_chunks_mut(dim2).enumerate().for_each(|(y, row)| {
        let source = &input[y * dim2..(y + 1) * dim2];
        for (k, &weight) in kernel.iter().enumerate() {
            let offset = k as isize - half as isize;
            let start = (-offset).clamp(0, dim2 as isize) as usize;
            let end = (dim2 as isize - offset).clamp(start as isize, dim2 as isize) as usize;
            for (x, value) in row[..start].iter_mut().enumerate() {
                *value += weight * source[reflected(x as isize + offset, dim2)];
            }
            let source_start = (start as isize + offset) as usize;
            if end > start {
                // Each pixel still accumulates kernel taps in the original order.
                // Contiguous independent outputs allow SIMD without reassociation.
                for (value, &sample) in row[start..end]
                    .iter_mut()
                    .zip(&source[source_start..source_start + end - start])
                {
                    *value += weight * sample;
                }
            }
            for (x, value) in row[end..].iter_mut().enumerate() {
                *value += weight * source[reflected((x + end) as isize + offset, dim2)];
            }
        }
    });
    out
}

fn convolve_vertical(input: &[f64], dim1: usize, dim2: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut out = vec![0.0; dim1 * dim2];
    if dim2 == 0 {
        return out;
    }
    out.par_chunks_mut(dim2).enumerate().for_each(|(y, row)| {
        for (k, &weight) in kernel.iter().enumerate() {
            let source_y = reflected(y as isize + k as isize - half as isize, dim1);
            let source = &input[source_y * dim2..(source_y + 1) * dim2];
            for (value, &sample) in row.iter_mut().zip(source) {
                *value += weight * sample;
            }
        }
    });
    out
}

pub(super) fn gaussian_blur(input: &[f64], dim1: usize, dim2: usize, sigma: f64) -> Vec<f64> {
    let kernel = gaussian_kernel_1d(sigma);
    let intermediate = convolve_horizontal(input, dim1, dim2, &kernel);
    convolve_vertical(&intermediate, dim1, dim2, &kernel)
}

// ---------- Connected-component labeling (4-connectivity, iterative BFS) ----------

/// Returns a label per pixel; label 0 is background, labels ≥ 1 are
/// foreground connected components.
pub(super) fn label_components(binary: &[bool], dim1: usize, dim2: usize) -> Vec<u32> {
    let mut labels = vec![0u32; dim1 * dim2];
    let mut next_label: u32 = 1;
    let mut stack: Vec<usize> = Vec::new();
    for seed in 0..(dim1 * dim2) {
        if !binary[seed] || labels[seed] != 0 {
            continue;
        }
        let label = next_label;
        next_label += 1;
        stack.push(seed);
        labels[seed] = label;
        while let Some(p) = stack.pop() {
            let i = p / dim2;
            let j = p % dim2;
            // 4-connectivity.
            if i > 0 {
                let q = p - dim2;
                if binary[q] && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if i + 1 < dim1 {
                let q = p + dim2;
                if binary[q] && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if j > 0 {
                let q = p - 1;
                if binary[q] && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
            if j + 1 < dim2 {
                let q = p + 1;
                if binary[q] && labels[q] == 0 {
                    labels[q] = label;
                    stack.push(q);
                }
            }
        }
    }
    labels
}

// ---------- Uphill walk + sub-pixel refinement ----------

#[inline]
fn at(image: &[f64], dim2: usize, i: usize, j: usize) -> f64 {
    image[i * dim2 + j]
}

/// Walk uphill through 4-neighbours from the integer-pixel seed until reaching
/// a local maximum, then refine to sub-pixel via parabolic fit on the 3×3
/// neighbourhood around that local max. Returns (d1, d2) — sub-pixel center.
///
/// The walk is bounded by `max_steps` so a pathological flat region doesn't
/// loop forever. Stops on image boundary contact.
pub(super) fn uphill_subpixel(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    seed: (usize, usize),
    max_steps: usize,
) -> Option<(f64, f64)> {
    let mut i = seed.0.min(dim1.saturating_sub(1));
    let mut j = seed.1.min(dim2.saturating_sub(1));
    for _ in 0..max_steps {
        let v = at(image, dim2, i, j);
        let mut best = v;
        let mut step = None;
        if i > 0 && at(image, dim2, i - 1, j) > best {
            best = at(image, dim2, i - 1, j);
            step = Some((i - 1, j));
        }
        if i + 1 < dim1 && at(image, dim2, i + 1, j) > best {
            best = at(image, dim2, i + 1, j);
            step = Some((i + 1, j));
        }
        if j > 0 && at(image, dim2, i, j - 1) > best {
            best = at(image, dim2, i, j - 1);
            step = Some((i, j - 1));
        }
        if j + 1 < dim2 && at(image, dim2, i, j + 1) > best {
            step = Some((i, j + 1));
        }
        match step {
            Some(s) => {
                i = s.0;
                j = s.1;
            }
            None => break,
        }
    }
    // Sub-pixel refinement via 1-D parabolic fit along each axis.
    if i == 0 || i + 1 >= dim1 || j == 0 || j + 1 >= dim2 {
        return Some((i as f64, j as f64));
    }
    let center = at(image, dim2, i, j);
    let up = at(image, dim2, i - 1, j);
    let down = at(image, dim2, i + 1, j);
    let left = at(image, dim2, i, j - 1);
    let right = at(image, dim2, i, j + 1);
    let denom_i = up + down - 2.0 * center;
    let denom_j = left + right - 2.0 * center;
    let off_i = if denom_i.abs() > 1.0e-30 {
        (0.5 * (up - down) / denom_i).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let off_j = if denom_j.abs() > 1.0e-30 {
        (0.5 * (left - right) / denom_j).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some((i as f64 + off_i, j as f64 + off_j))
}

// ---------- top-level find_peaks ----------

/// Full-resolution Massif preprocessing for repeated seed searches.
///
/// Owns an immutable image and its connected-component labels. Construct a new
/// instance whenever samples, image shape, masking, or valley size change.
/// Peak order, subpixel fitting and deduplication match [`find_peaks`].
pub struct PreparedMassif {
    image: Vec<f64>,
    labels: Vec<u32>,
    dim1: usize,
    dim2: usize,
    options: MassifOptions,
}

impl PreparedMassif {
    pub fn new(image: Vec<f64>, dim1: usize, dim2: usize, options: MassifOptions) -> Self {
        assert_eq!(image.len(), dim1 * dim2, "image size mismatch");
        let sigma = effective_valley_size(&options, dim1, dim2);
        let bg = gaussian_blur(&image, dim1, dim2, sigma);
        let binary: Vec<bool> = image.iter().zip(&bg).map(|(v, b)| v > b).collect();
        let labels = label_components(&binary, dim1, dim2);
        Self {
            image,
            labels,
            dim1,
            dim2,
            options,
        }
    }

    /// Immutable samples used for both preprocessing and subpixel localization.
    pub fn image(&self) -> &[f64] {
        &self.image
    }

    pub fn find_peaks(&self, seed: (usize, usize)) -> Vec<(f64, f64)> {
        let Self {
            image,
            labels,
            dim1,
            dim2,
            options,
        } = self;
        let (dim1, dim2) = (*dim1, *dim2);
        if seed.0 >= dim1 || seed.1 >= dim2 {
            return vec![];
        }
        let seed_label = labels[seed.0 * dim2 + seed.1];
        if seed_label == 0 {
            return vec![];
        }

        // 4. Region mean (over image values inside the seed's component).
        let mut sum = 0.0;
        let mut count = 0usize;
        for k in 0..image.len() {
            if labels[k] == seed_label {
                sum += image[k];
                count += 1;
            }
        }
        if count == 0 {
            return vec![];
        }
        let mean = sum / count as f64;

        // 5. Central peak via uphill walk from the seed.
        let max_steps = (dim1 + dim2) * 2;
        let mut peaks: Vec<(f64, f64)> = Vec::new();
        if let Some(p) = uphill_subpixel(image, dim1, dim2, seed, max_steps) {
            peaks.push(p);
        }

        // 6. Sub-peaks: every pixel in the region above mean, uphill-walk + dedupe.
        let dedup2 = options.dedup_dist * options.dedup_dist;
        'outer: for i in 0..dim1 {
            for j in 0..dim2 {
                let idx = i * dim2 + j;
                if labels[idx] != seed_label || image[idx] <= mean {
                    continue;
                }
                let p = match uphill_subpixel(image, dim1, dim2, (i, j), max_steps) {
                    Some(p) => p,
                    None => continue,
                };
                // Confirm peak is inside the same massif.
                let pi = p.0.round() as isize;
                let pj = p.1.round() as isize;
                if pi < 0 || pj < 0 || pi as usize >= dim1 || pj as usize >= dim2 {
                    continue;
                }
                if labels[pi as usize * dim2 + pj as usize] != seed_label {
                    continue;
                }
                // Dedupe.
                let dup = peaks.iter().any(|q| {
                    let di = q.0 - p.0;
                    let dj = q.1 - p.1;
                    di * di + dj * dj < dedup2
                });
                if !dup {
                    peaks.push(p);
                    if peaks.len() >= options.max_peaks {
                        break 'outer;
                    }
                }
            }
        }

        peaks
    }
}

/// Find peaks for a single seed. For repeated picks, reuse [`PreparedMassif`].
pub fn find_peaks(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    seed: (usize, usize),
    options: &MassifOptions,
) -> Vec<(f64, f64)> {
    assert_eq!(image.len(), dim1 * dim2, "image size mismatch");
    if seed.0 >= dim1 || seed.1 >= dim2 {
        return vec![];
    }
    PreparedMassif::new(image.to_vec(), dim1, dim2, *options).find_peaks(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen pre-optimization convolution and seeded search. This comparison
    // detects changes to background thresholds, traversal order and coordinates.
    mod reference {
        use super::super::*;
        /// Convolve each row of `input` (shape dim1 × dim2, row-major) with `kernel`,
        /// using reflect padding at the boundaries.
        fn convolve_horizontal(
            input: &[f64],
            dim1: usize,
            dim2: usize,
            kernel: &[f64],
        ) -> Vec<f64> {
            let half = kernel.len() / 2;
            let mut out = vec![0.0; dim1 * dim2];
            for i in 0..dim1 {
                let row_off = i * dim2;
                for j in 0..dim2 {
                    let mut s = 0.0;
                    for k in 0..kernel.len() {
                        // Reflect index around the boundary.
                        let mut jj = j as isize + k as isize - half as isize;
                        if jj < 0 {
                            jj = -jj;
                        }
                        let dim2_i = dim2 as isize;
                        if jj >= dim2_i {
                            jj = 2 * dim2_i - 2 - jj;
                        }
                        let jj = jj.clamp(0, dim2_i - 1) as usize;
                        s += kernel[k] * input[row_off + jj];
                    }
                    out[row_off + j] = s;
                }
            }
            out
        }

        /// Convolve each column of `input` (shape dim1 × dim2, row-major) with
        /// `kernel`, with reflect padding.
        fn convolve_vertical(input: &[f64], dim1: usize, dim2: usize, kernel: &[f64]) -> Vec<f64> {
            let half = kernel.len() / 2;
            let mut out = vec![0.0; dim1 * dim2];
            for j in 0..dim2 {
                for i in 0..dim1 {
                    let mut s = 0.0;
                    for k in 0..kernel.len() {
                        let mut ii = i as isize + k as isize - half as isize;
                        if ii < 0 {
                            ii = -ii;
                        }
                        let dim1_i = dim1 as isize;
                        if ii >= dim1_i {
                            ii = 2 * dim1_i - 2 - ii;
                        }
                        let ii = ii.clamp(0, dim1_i - 1) as usize;
                        s += kernel[k] * input[ii * dim2 + j];
                    }
                    out[i * dim2 + j] = s;
                }
            }
            out
        }

        pub(super) fn gaussian_blur(
            input: &[f64],
            dim1: usize,
            dim2: usize,
            sigma: f64,
        ) -> Vec<f64> {
            let kernel = gaussian_kernel_1d(sigma);
            let intermediate = convolve_horizontal(input, dim1, dim2, &kernel);
            convolve_vertical(&intermediate, dim1, dim2, &kernel)
        }

        pub fn find_peaks(
            image: &[f64],
            dim1: usize,
            dim2: usize,
            seed: (usize, usize),
            options: &MassifOptions,
        ) -> Vec<(f64, f64)> {
            assert_eq!(image.len(), dim1 * dim2, "image size mismatch");
            if seed.0 >= dim1 || seed.1 >= dim2 {
                return vec![];
            }

            // 1. Local-background estimate.
            let sigma = effective_valley_size(options, dim1, dim2);
            let bg = gaussian_blur(image, dim1, dim2, sigma);

            // 2. Binarize: pixels above local background.
            let mut binary: Vec<bool> = Vec::with_capacity(image.len());
            for k in 0..image.len() {
                binary.push(image[k] > bg[k]);
            }

            // 3. Connected components.
            let labels = label_components(&binary, dim1, dim2);
            let seed_label = labels[seed.0 * dim2 + seed.1];
            if seed_label == 0 {
                return vec![];
            }

            // 4. Region mean (over image values inside the seed's component).
            let mut sum = 0.0;
            let mut count = 0usize;
            for k in 0..image.len() {
                if labels[k] == seed_label {
                    sum += image[k];
                    count += 1;
                }
            }
            if count == 0 {
                return vec![];
            }
            let mean = sum / count as f64;

            // 5. Central peak via uphill walk from the seed.
            let max_steps = (dim1 + dim2) * 2;
            let mut peaks: Vec<(f64, f64)> = Vec::new();
            if let Some(p) = uphill_subpixel(image, dim1, dim2, seed, max_steps) {
                peaks.push(p);
            }

            // 6. Sub-peaks: every pixel in the region above mean, uphill-walk + dedupe.
            let dedup2 = options.dedup_dist * options.dedup_dist;
            'outer: for i in 0..dim1 {
                for j in 0..dim2 {
                    let idx = i * dim2 + j;
                    if labels[idx] != seed_label || image[idx] <= mean {
                        continue;
                    }
                    let p = match uphill_subpixel(image, dim1, dim2, (i, j), max_steps) {
                        Some(p) => p,
                        None => continue,
                    };
                    // Confirm peak is inside the same massif.
                    let pi = p.0.round() as isize;
                    let pj = p.1.round() as isize;
                    if pi < 0 || pj < 0 || pi as usize >= dim1 || pj as usize >= dim2 {
                        continue;
                    }
                    if labels[pi as usize * dim2 + pj as usize] != seed_label {
                        continue;
                    }
                    // Dedupe.
                    let dup = peaks.iter().any(|q| {
                        let di = q.0 - p.0;
                        let dj = q.1 - p.1;
                        di * di + dj * dj < dedup2
                    });
                    if !dup {
                        peaks.push(p);
                        if peaks.len() >= options.max_peaks {
                            break 'outer;
                        }
                    }
                }
            }

            peaks
        }
    }

    #[test]
    fn convolution_is_bitwise_equal_to_original_for_edges_and_wide_kernels() {
        for (h, w) in [(0, 0), (1, 1), (1, 7), (7, 1), (3, 5), (33, 29), (128, 131)] {
            let image: Vec<_> = (0..h * w)
                .map(|i| {
                    let hash = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (hash >> 32) as f64 / 100003.0 - 9000.0
                })
                .collect();
            for sigma in [0.5, 2.0, 5.0, 20.0] {
                let actual = gaussian_blur(&image, h, w, sigma);
                let expected = reference::gaussian_blur(&image, h, w, sigma);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "shape {h}x{w}, sigma {sigma}"
                );
            }
        }
    }

    #[test]
    fn prepared_and_single_seed_searches_match_original_peak_order_and_positions() {
        let (h, w) = (71, 83);
        let mut image: Vec<_> = (0..h * w)
            .map(|i| ((i * 997) % 101) as f64 * 0.001)
            .collect();
        for (y, x) in [(3.2, 4.7), (30.4, 27.7), (32.6, 35.2), (68.7, 80.1)] {
            add_gaussian(&mut image, h, w, y, x, 90.0, 2.0);
        }
        for valley_size in [None, Some(2.0), Some(20.0)] {
            let options = MassifOptions {
                valley_size,
                max_peaks: 17,
                dedup_dist: 1.5,
            };
            let prepared = PreparedMassif::new(image.clone(), h, w, options);
            for seed in [
                (0, 0),
                (3, 5),
                (30, 28),
                (33, 35),
                (69, 80),
                (70, 82),
                (h, w),
            ] {
                let expected = reference::find_peaks(&image, h, w, seed, &options);
                assert_eq!(
                    prepared.find_peaks(seed),
                    expected,
                    "prepared seed {seed:?}"
                );
                assert_eq!(
                    find_peaks(&image, h, w, seed, &options),
                    expected,
                    "single seed {seed:?}"
                );
            }
        }
    }

    /// Place a 2-D Gaussian peak at (i0, j0) with amplitude A and width σ.
    fn add_gaussian(
        image: &mut [f64],
        dim1: usize,
        dim2: usize,
        i0: f64,
        j0: f64,
        amp: f64,
        sigma: f64,
    ) {
        let two_s2 = 2.0 * sigma * sigma;
        for i in 0..dim1 {
            let di = i as f64 - i0;
            for j in 0..dim2 {
                let dj = j as f64 - j0;
                image[i * dim2 + j] += amp * (-(di * di + dj * dj) / two_s2).exp();
            }
        }
    }

    #[test]
    fn single_peak_recovered_to_subpixel() {
        let dim = 64;
        let mut img = vec![1.0_f64; dim * dim]; // uniform background
        // Peak deliberately at a non-integer location.
        let truth = (32.4, 27.7);
        add_gaussian(&mut img, dim, dim, truth.0, truth.1, 100.0, 1.8);
        let opts = MassifOptions {
            valley_size: Some(4.0),
            ..MassifOptions::default()
        };
        let peaks = find_peaks(&img, dim, dim, (32, 28), &opts);
        assert!(!peaks.is_empty(), "expected at least one peak");
        let (di, dj) = (peaks[0].0 - truth.0, peaks[0].1 - truth.1);
        let err = (di * di + dj * dj).sqrt();
        assert!(err < 0.1, "sub-pixel recovery err = {} px", err);
    }

    #[test]
    fn arc_of_peaks_all_found() {
        // Drop several Gaussians along an arc; verify they all come back.
        let dim = 128;
        let mut img = vec![0.0_f64; dim * dim];
        let truths: Vec<(f64, f64)> = (0..6)
            .map(|k| {
                let t = std::f64::consts::PI * 0.3 + 0.1 * k as f64;
                (64.0 + 40.0 * t.cos(), 64.0 + 40.0 * t.sin())
            })
            .collect();
        for &(i, j) in &truths {
            add_gaussian(&mut img, dim, dim, i, j, 50.0, 1.5);
        }
        // Use the first peak as the seed; all peaks should be in the same
        // massif because they overlap above the local background.
        let seed = (truths[0].0.round() as usize, truths[0].1.round() as usize);
        let opts = MassifOptions {
            valley_size: Some(6.0),
            max_peaks: 50,
            dedup_dist: 1.5,
        };
        let peaks = find_peaks(&img, dim, dim, seed, &opts);
        // Each truth should be near at least one detected peak.
        for t in &truths {
            let nearest = peaks
                .iter()
                .map(|p| {
                    let di = p.0 - t.0;
                    let dj = p.1 - t.1;
                    (di * di + dj * dj).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 0.5, "truth {:?} not within 0.5 px of any peak", t);
        }
    }

    #[test]
    fn seed_on_background_returns_empty() {
        let dim = 32;
        let mut img = vec![1.0_f64; dim * dim];
        // Peak far away from the seed.
        add_gaussian(&mut img, dim, dim, 5.0, 5.0, 100.0, 1.5);
        let opts = MassifOptions::default();
        let peaks = find_peaks(&img, dim, dim, (28, 28), &opts);
        assert!(peaks.is_empty(), "background seed should return no peaks");
    }

    #[test]
    fn image_wide_peak_finder_recovers_multiple_peaks() {
        let dim = 128;
        let mut img = vec![1.0_f64; dim * dim];
        let truths: Vec<(f64, f64)> = vec![(30.4, 40.7), (70.2, 50.3), (90.6, 110.1)];
        for &(i, j) in &truths {
            add_gaussian(&mut img, dim, dim, i, j, 80.0, 1.5);
        }
        let opts = ImagePeakOptions {
            valley_size: Some(6.0),
            threshold_factor: 3.0,
            max_peaks: 200,
        };
        let peaks = find_peaks_image(&img, dim, dim, &opts);
        // Every truth should be near some detected peak.
        for t in &truths {
            let nearest = peaks
                .iter()
                .map(|p| ((p.0 - t.0).powi(2) + (p.1 - t.1).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 0.5, "truth {:?} not within 0.5 px", t);
        }
    }
}

// ---------- image-wide peak finder (no seed) ----------

#[derive(Debug, Clone, Copy)]
pub struct ImagePeakOptions {
    /// Gaussian σ for the background filter, in pixels.
    /// `None` → adaptive pyFAI-style `max(5, max(dim) / 50)`.
    pub valley_size: Option<f64>,
    /// Pixels are flagged as foreground when
    /// `image > background + threshold_factor * std(image - background)`.
    /// A value of 2-4 works well in practice; lower picks up more spotty
    /// peaks at the cost of more outliers.
    pub threshold_factor: f64,
    /// Cap on returned peaks.
    pub max_peaks: usize,
}

impl Default for ImagePeakOptions {
    fn default() -> Self {
        Self {
            valley_size: None,
            threshold_factor: 3.0,
            max_peaks: 5000,
        }
    }
}

/// Image-wide peak extraction without a seed.
///
/// Subtract a Gaussian-blurred background, threshold the residual to flag
/// foreground pixels, then return every 3×3 local maximum within the
/// foreground mask, sub-pixel refined. Unlike a one-centroid-per-blob
/// approach, this yields many peaks along the length of each Debye-Scherrer
/// ring (which is what RANSAC ellipse clustering needs).
pub fn find_peaks_image(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    options: &ImagePeakOptions,
) -> Vec<(f64, f64)> {
    assert_eq!(image.len(), dim1 * dim2, "image size mismatch");
    let sigma = match options.valley_size {
        Some(v) => v,
        None => 5.0_f64.max(dim1.max(dim2) as f64 / 50.0),
    };
    let bg = gaussian_blur(image, dim1, dim2, sigma);
    // Residual image and its std (over the whole frame).
    let residual: Vec<f64> = image.iter().zip(&bg).map(|(i, b)| i - b).collect();
    let n = image.len() as f64;
    let mean_r = residual.iter().sum::<f64>() / n;
    let var_r = residual
        .iter()
        .map(|r| (r - mean_r) * (r - mean_r))
        .sum::<f64>()
        / n;
    let std_r = var_r.sqrt();
    let threshold = mean_r + options.threshold_factor * std_r;

    // A pixel is a peak if (a) it's above threshold in the residual image
    // and (b) it's a strict local max over a 3×3 neighbourhood of the *raw*
    // image. Iterate over the interior of the frame; the border is rarely
    // useful for diffraction calibration anyway.
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    let max_steps = 4; // tiny — the seed is already on the peak ridge
    for i in 1..(dim1.saturating_sub(1)) {
        for j in 1..(dim2.saturating_sub(1)) {
            let idx = i * dim2 + j;
            if residual[idx] <= threshold {
                continue;
            }
            let v = image[idx];
            // Strict local max over 3×3.
            let mut is_max = true;
            for di in -1isize..=1 {
                for dj in -1isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = (i as isize + di) as usize;
                    let nj = (j as isize + dj) as usize;
                    if image[ni * dim2 + nj] > v {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            if !is_max {
                continue;
            }
            if let Some(p) = uphill_subpixel(image, dim1, dim2, (i, j), max_steps) {
                peaks.push(p);
                if peaks.len() >= options.max_peaks {
                    return peaks;
                }
            }
        }
    }
    peaks
}
