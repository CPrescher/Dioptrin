//! Radial-binning ring extraction from peaks given a beam center.
//!
//! Once the beam center is known (e.g. via [`super::center::estimate_beam_center`])
//! the per-peak distance from it forms a clean radial histogram with one
//! sharp peak per Debye-Scherrer ring. We:
//!
//! 1. Compute the radial distance of each peak from the given center.
//! 2. Build a smoothed histogram and pick local maxima above a count
//!    threshold — these are the ring radii.
//! 3. For each ring, gather peaks within `±radial_tol_px` and fit an
//!    ellipse via [`super::ellipse::fit_least_squares`].
//! 4. Drop any ring whose ellipse fit has fewer than `min_inliers_per_ring`
//!    surviving inliers (Sampson distance test, like RANSAC's check) or
//!    whose ellipse exceeds the `max_axis_ratio` constraint.
//!
//! Why this is better than 5-point RANSAC for diffraction data: the radial
//! structure of concentric rings is exactly the structure we exploit, so
//! every peak from each ring goes into that ring's clean least-squares fit.
//! No 5-of-5-same-ring Monte Carlo coincidence is required.

#![allow(clippy::needless_range_loop)]

use super::ellipse::{Ellipse, fit_least_squares};

#[derive(Debug, Clone, Copy)]
pub struct RadialExtractOptions {
    /// Width of one radial-histogram bin (pixels).
    pub bin_width_px: f64,
    /// Gaussian smoothing σ applied to the histogram before peak finding
    /// (in bins).
    pub smooth_sigma_bins: f64,
    /// A histogram bin is a ring candidate if its smoothed count is a
    /// strict local maximum AND ≥ `min_bin_count`.
    pub min_bin_count: usize,
    /// Half-width (pixels) around each ring radius from which inliers are
    /// gathered for the per-ring ellipse fit.
    pub radial_tol_px: f64,
    /// Minimum number of points required to attempt an ellipse fit for a ring.
    pub min_inliers_per_ring: usize,
    /// Optional axis-ratio cap on accepted ellipses (mirrors
    /// [`super::ellipse::RansacOptions::max_axis_ratio`]).
    pub max_axis_ratio: Option<f64>,
    /// Maximum radius considered.
    pub max_radius_px: f64,
}

impl Default for RadialExtractOptions {
    fn default() -> Self {
        Self {
            bin_width_px: 2.0,
            smooth_sigma_bins: 1.5,
            min_bin_count: 8,
            radial_tol_px: 6.0,
            min_inliers_per_ring: 8,
            max_axis_ratio: Some(1.2),
            max_radius_px: 5000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RadialExtractResult {
    /// One ellipse per detected ring (sorted by semi-major ascending).
    pub ellipses: Vec<Ellipse>,
    /// Per detected ellipse: ring radius (px from center) it was fit from.
    pub ring_radii: Vec<f64>,
    /// Per-input-point assignment: `Some(k)` = inlier of `ellipses[k]`,
    /// `None` = unassigned (point didn't fall in any ring's radial band, or
    /// fell outside the fitted ellipse's Sampson tolerance).
    pub assignments: Vec<Option<usize>>,
}

/// 1-D Gaussian smoothing kernel (reflect padding).
fn smooth_1d(counts: &[usize], sigma: f64) -> Vec<f64> {
    let n = counts.len();
    let half = (3.0 * sigma).ceil() as usize;
    let two_s2 = 2.0 * sigma * sigma;
    let kernel: Vec<f64> = (0..=(2 * half))
        .map(|i| {
            let x = i as f64 - half as f64;
            (-x * x / two_s2).exp()
        })
        .collect();
    let ksum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|v| v / ksum).collect();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for (k, &w) in kernel.iter().enumerate() {
            let mut idx = i as isize + k as isize - half as isize;
            if idx < 0 {
                idx = -idx;
            }
            let ni = n as isize;
            if idx >= ni {
                idx = 2 * ni - 2 - idx;
            }
            let idx = idx.clamp(0, ni - 1) as usize;
            s += w * counts[idx] as f64;
        }
        out[i] = s;
    }
    out
}

/// Strict local maxima in a 1-D array, above `min_val`.
fn local_maxima_above(values: &[f64], min_val: f64) -> Vec<usize> {
    let n = values.len();
    let mut peaks = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if values[i] >= min_val && values[i] > values[i - 1] && values[i] > values[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Extract one ellipse per Debye-Scherrer ring from a set of peaks and a
/// known beam center.
pub fn extract_rings_radial(
    peaks: &[(f64, f64)],
    center: (f64, f64),
    options: &RadialExtractOptions,
) -> RadialExtractResult {
    let n = peaks.len();
    let radii: Vec<f64> = peaks
        .iter()
        .map(|p| {
            let dx = p.0 - center.0;
            let dy = p.1 - center.1;
            (dx * dx + dy * dy).sqrt()
        })
        .collect();

    let bin_w = options.bin_width_px.max(0.1);
    let n_bins = (options.max_radius_px / bin_w).ceil() as usize;
    let mut counts = vec![0usize; n_bins];
    for &r in &radii {
        if r >= options.max_radius_px {
            continue;
        }
        let bin = (r / bin_w) as usize;
        if bin < n_bins {
            counts[bin] += 1;
        }
    }
    let smoothed = smooth_1d(&counts, options.smooth_sigma_bins);
    let peak_bins = local_maxima_above(&smoothed, options.min_bin_count as f64);

    let mut ellipses: Vec<Ellipse> = Vec::new();
    let mut ring_radii: Vec<f64> = Vec::new();
    let mut assignments: Vec<Option<usize>> = vec![None; n];
    let tol = options.radial_tol_px;

    for &bin in &peak_bins {
        let r_ring = (bin as f64 + 0.5) * bin_w;
        let mut inlier_idx: Vec<usize> = Vec::new();
        for (i, &r) in radii.iter().enumerate() {
            if (r - r_ring).abs() <= tol {
                inlier_idx.push(i);
            }
        }
        if inlier_idx.len() < options.min_inliers_per_ring {
            continue;
        }
        let ring_pts: Vec<(f64, f64)> = inlier_idx.iter().map(|&i| peaks[i]).collect();
        let ellipse = match fit_least_squares(&ring_pts) {
            Some(e) => e,
            None => continue,
        };
        if let Some(limit) = options.max_axis_ratio {
            let ratio = ellipse.semi_major / ellipse.semi_minor.max(1e-12);
            if ratio > limit {
                continue;
            }
        }
        // Re-test inliers against the actual ellipse (some radial-band
        // outliers may have crept in, especially for tilted detectors).
        let surviving: Vec<usize> = inlier_idx
            .into_iter()
            .filter(|&i| ellipse.sampson_distance(peaks[i]) <= tol)
            .collect();
        if surviving.len() < options.min_inliers_per_ring {
            continue;
        }
        let k = ellipses.len();
        for i in &surviving {
            assignments[*i] = Some(k);
        }
        ellipses.push(ellipse);
        ring_radii.push(r_ring);
    }

    // Sort by semi-major axis (smallest first).
    let mut order: Vec<usize> = (0..ellipses.len()).collect();
    order.sort_by(|&a, &b| {
        ellipses[a]
            .semi_major
            .partial_cmp(&ellipses[b].semi_major)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_ellipses: Vec<Ellipse> = order.iter().map(|&i| ellipses[i]).collect();
    let sorted_radii: Vec<f64> = order.iter().map(|&i| ring_radii[i]).collect();
    let remap: Vec<usize> = {
        let mut r = vec![0usize; ellipses.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            r[old_idx] = new_idx;
        }
        r
    };
    let assignments = assignments
        .into_iter()
        .map(|a| a.map(|k| remap[k]))
        .collect();

    RadialExtractResult {
        ellipses: sorted_ellipses,
        ring_radii: sorted_radii,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_concentric_rings(cx: f64, cy: f64, radii: &[f64], per_ring: usize) -> Vec<(f64, f64)> {
        let mut peaks = Vec::new();
        for &r in radii {
            for i in 0..per_ring {
                let t = 2.0 * PI * (i as f64) / (per_ring as f64);
                peaks.push((cx + r * t.cos(), cy + r * t.sin()));
            }
        }
        peaks
    }

    #[test]
    fn extract_six_rings_with_known_center() {
        let peaks =
            make_concentric_rings(500.0, 600.0, &[80.0, 130.0, 175.0, 220.0, 265.0, 310.0], 50);
        let opts = RadialExtractOptions::default();
        let result = extract_rings_radial(&peaks, (500.0, 600.0), &opts);
        assert_eq!(
            result.ellipses.len(),
            6,
            "expected 6 rings, got {}",
            result.ellipses.len()
        );
        // Each ellipse should be essentially a circle centered at (500, 600).
        for (i, e) in result.ellipses.iter().enumerate() {
            assert!((e.center.0 - 500.0).abs() < 0.1, "ring {} cx", i);
            assert!((e.center.1 - 600.0).abs() < 0.1, "ring {} cy", i);
        }
        // Smallest ellipse should be radius 80.
        assert!((result.ellipses[0].semi_major - 80.0).abs() < 0.1);
        assert!((result.ellipses[5].semi_major - 310.0).abs() < 0.1);
        // Every input point should be assigned (none unassigned).
        assert!(result.assignments.iter().all(|a| a.is_some()));
    }

    #[test]
    fn extract_drops_undersampled_rings() {
        // Three rings, one of them has only 3 peaks → should be dropped.
        let mut peaks = make_concentric_rings(200.0, 200.0, &[60.0, 120.0], 40);
        // Sparse third ring.
        for i in 0..3 {
            let t = 2.0 * PI * (i as f64) / 3.0;
            peaks.push((200.0 + 180.0 * t.cos(), 200.0 + 180.0 * t.sin()));
        }
        let opts = RadialExtractOptions::default();
        let result = extract_rings_radial(&peaks, (200.0, 200.0), &opts);
        assert_eq!(
            result.ellipses.len(),
            2,
            "expected 2 rings (sparse one dropped)"
        );
    }
}
