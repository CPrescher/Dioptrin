//! Robust beam-center estimation from peak coordinates.
//!
//! Approach: **3-point circle RANSAC**. Three points on the same Debye-Scherrer
//! ring uniquely determine the circle they lie on (closed form), and that
//! circle's center is the beam center. Sampling triples of peaks and scoring
//! each candidate center by radial-band consistency (how many peaks fall in
//! tight radial annuli) finds the right center even when:
//!
//! - the beam is off the detector (only partial arcs visible),
//! - only one ring's peaks are visible (every triple is then "same ring"),
//! - many peaks are spread across many rings (a few % of triples will be
//!   same-ring → still enough for RANSAC consensus).
//!
//! This is more general than centroid-based estimators, which fail whenever
//! peaks are not azimuthally uniform around the beam.
//!
//! # API
//!
//! - [`fit_circle_3point`]: closed-form circle through three points.
//! - [`estimate_beam_center`]: RANSAC over 3-point samples + radial-band
//!   scoring. Returns `(cx, cy)` or `None` if too few peaks / no consensus.

#![allow(clippy::needless_range_loop)]

/// Closed-form circle through 3 points. Returns `((cx, cy), r)` or `None`
/// for (near-)colinear input.
pub fn fit_circle_3point(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
) -> Option<((f64, f64), f64)> {
    // Subtract pairwise gives a 2×2 linear system for (cx, cy):
    //   2(x1-x2) cx + 2(y1-y2) cy = (x1²-x2²) + (y1²-y2²)
    //   2(x2-x3) cx + 2(y2-y3) cy = (x2²-x3²) + (y2²-y3²)
    let a11 = 2.0 * (p1.0 - p2.0);
    let a12 = 2.0 * (p1.1 - p2.1);
    let a21 = 2.0 * (p2.0 - p3.0);
    let a22 = 2.0 * (p2.1 - p3.1);
    let b1 = p1.0 * p1.0 - p2.0 * p2.0 + p1.1 * p1.1 - p2.1 * p2.1;
    let b2 = p2.0 * p2.0 - p3.0 * p3.0 + p2.1 * p2.1 - p3.1 * p3.1;
    let det = a11 * a22 - a12 * a21;
    if det.abs() < 1e-12 {
        return None;
    }
    let cx = (b1 * a22 - b2 * a12) / det;
    let cy = (a11 * b2 - a21 * b1) / det;
    let dx = p1.0 - cx;
    let dy = p1.1 - cy;
    let r = (dx * dx + dy * dy).sqrt();
    if !r.is_finite() {
        return None;
    }
    Some(((cx, cy), r))
}

#[derive(Debug, Clone, Copy)]
pub struct BeamCenterOptions {
    /// Number of 3-point RANSAC trials.
    pub max_trials: usize,
    /// Width of each radial bin (pixels) used for scoring.
    pub radial_bin_width_px: f64,
    /// Sum the counts of this many top bins to score a candidate center.
    /// Powder patterns have many distinct ring radii — counting the top
    /// few captures all of them while remaining insensitive to noise.
    pub top_k_bins: usize,
    /// Maximum radius considered. Bins above this are ignored. Set to a
    /// value larger than the longest diagonal that could plausibly be a
    /// ring radius (e.g. 2× the detector's largest dimension).
    pub max_radius_px: f64,
    pub rng_seed: u64,
}

impl Default for BeamCenterOptions {
    fn default() -> Self {
        Self {
            max_trials: 500,
            radial_bin_width_px: 4.0,
            top_k_bins: 10,
            max_radius_px: 5000.0,
            rng_seed: 0xBEAC_0CEF_F00D_F00D,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BeamCenter {
    pub d1: f64,
    pub d2: f64,
    /// Score of the chosen candidate (sum of top-K radial-bin counts).
    /// Useful for diagnosing how well-supported the estimate is.
    pub score: usize,
    /// Number of RANSAC trials that yielded valid (non-colinear) fits.
    pub n_valid_trials: usize,
}

/// Splitmix64 PRNG (also used elsewhere in the calibration module).
#[inline]
fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sample3_distinct(state: &mut u64, n: usize) -> Option<(usize, usize, usize)> {
    if n < 3 {
        return None;
    }
    let i = (next_rand(state) as usize) % n;
    let mut j;
    loop {
        j = (next_rand(state) as usize) % n;
        if j != i {
            break;
        }
    }
    let mut k;
    loop {
        k = (next_rand(state) as usize) % n;
        if k != i && k != j {
            break;
        }
    }
    Some((i, j, k))
}

/// For a candidate center, count how many peaks fall in each radial bin and
/// return the sum of the top-K bin counts. Powder rings produce sharp radial
/// peaks; the right center maximises this score, wrong centers smear it out.
fn radial_band_score(
    peaks: &[(f64, f64)],
    center: (f64, f64),
    bin_width: f64,
    top_k: usize,
    max_radius: f64,
) -> usize {
    let n_bins = (max_radius / bin_width).ceil() as usize;
    if n_bins == 0 {
        return 0;
    }
    let mut hist = vec![0usize; n_bins];
    for &(d1, d2) in peaks {
        let dx = d1 - center.0;
        let dy = d2 - center.1;
        let r = (dx * dx + dy * dy).sqrt();
        if !r.is_finite() || r >= max_radius {
            continue;
        }
        let bin = (r / bin_width) as usize;
        if bin < n_bins {
            hist[bin] += 1;
        }
    }
    if top_k >= hist.len() {
        return hist.iter().sum();
    }
    // Partial-sort: select the top_k values without sorting everything.
    hist.sort_unstable_by(|a, b| b.cmp(a));
    hist.iter().take(top_k).sum()
}

/// Estimate the beam center via 3-point circle RANSAC + radial-band scoring.
///
/// Returns `None` if fewer than 3 peaks are provided, or if no trial
/// produced a valid (non-colinear) circle fit.
pub fn estimate_beam_center(
    peaks: &[(f64, f64)],
    options: &BeamCenterOptions,
) -> Option<BeamCenter> {
    if peaks.len() < 3 {
        return None;
    }
    let mut rng = options.rng_seed;
    let mut best: Option<((f64, f64), usize)> = None;
    let mut n_valid = 0usize;
    for _ in 0..options.max_trials {
        let (i, j, k) = sample3_distinct(&mut rng, peaks.len())?;
        let (center, _r) = match fit_circle_3point(peaks[i], peaks[j], peaks[k]) {
            Some(c) => c,
            None => continue,
        };
        // Sanity: reject absurd centers (e.g., several detector diameters
        // away). Speeds up scoring by avoiding obviously-wrong candidates.
        if center.0.abs() > options.max_radius_px || center.1.abs() > options.max_radius_px {
            continue;
        }
        n_valid += 1;
        let score = radial_band_score(
            peaks,
            center,
            options.radial_bin_width_px,
            options.top_k_bins,
            options.max_radius_px,
        );
        if best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((center, score));
        }
    }
    best.map(|(center, score)| BeamCenter {
        d1: center.0,
        d2: center.1,
        score,
        n_valid_trials: n_valid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ring_points(
        cx: f64,
        cy: f64,
        r: f64,
        n: usize,
        az_start: f64,
        az_end: f64,
    ) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let t = az_start + (az_end - az_start) * (i as f64) / ((n - 1) as f64);
                (cx + r * t.cos(), cy + r * t.sin())
            })
            .collect()
    }

    #[test]
    fn fit_three_points_on_a_circle() {
        let truth = ((30.0, 50.0), 25.0);
        let pts = ring_points(truth.0.0, truth.0.1, truth.1, 3, 0.0, 2.0 * PI / 3.0);
        let (c, r) = fit_circle_3point(pts[0], pts[1], pts[2]).expect("fit ok");
        assert!((c.0 - truth.0.0).abs() < 1e-9);
        assert!((c.1 - truth.0.1).abs() < 1e-9);
        assert!((r - truth.1).abs() < 1e-9);
    }

    #[test]
    fn fit_rejects_colinear_points() {
        let pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        assert!(fit_circle_3point(pts[0], pts[1], pts[2]).is_none());
    }

    #[test]
    fn center_from_full_multi_ring_pattern() {
        // Three concentric circles, beam at (200, 300), full 2π coverage.
        let mut peaks = Vec::new();
        for r in [80.0, 140.0, 200.0] {
            peaks.extend(ring_points(200.0, 300.0, r, 60, 0.0, 2.0 * PI));
        }
        let opts = BeamCenterOptions {
            max_trials: 300,
            radial_bin_width_px: 3.0,
            top_k_bins: 5,
            max_radius_px: 800.0,
            rng_seed: 42,
        };
        let c = estimate_beam_center(&peaks, &opts).expect("center ok");
        assert!((c.d1 - 200.0).abs() < 0.5, "cx got {}", c.d1);
        assert!((c.d2 - 300.0).abs() < 0.5, "cy got {}", c.d2);
    }

    #[test]
    fn center_from_partial_arc_only() {
        // Single ring, only a quarter arc visible (azimuth ∈ [0, π/2]).
        // This is the tough case the user flagged: a centroid-based estimator
        // would put the center somewhere in the arc, not at the true center.
        let truth = (150.0, 250.0);
        let r = 180.0;
        let peaks = ring_points(truth.0, truth.1, r, 40, 0.0, PI / 2.0);
        let centroid = (
            peaks.iter().map(|p| p.0).sum::<f64>() / peaks.len() as f64,
            peaks.iter().map(|p| p.1).sum::<f64>() / peaks.len() as f64,
        );
        // Sanity: the centroid IS off the true center by ~70 px.
        let centroid_err = ((centroid.0 - truth.0).powi(2) + (centroid.1 - truth.1).powi(2)).sqrt();
        assert!(
            centroid_err > 50.0,
            "centroid err {} - test setup broken",
            centroid_err
        );

        let opts = BeamCenterOptions {
            max_trials: 200,
            radial_bin_width_px: 2.0,
            top_k_bins: 3,
            max_radius_px: 500.0,
            rng_seed: 7,
        };
        let c = estimate_beam_center(&peaks, &opts).expect("center ok");
        assert!(
            (c.d1 - truth.0).abs() < 1.0 && (c.d2 - truth.1).abs() < 1.0,
            "got ({}, {}) vs truth ({}, {})  centroid was ({}, {})",
            c.d1,
            c.d2,
            truth.0,
            truth.1,
            centroid.0,
            centroid.1,
        );
    }

    #[test]
    fn center_handles_off_detector_beam() {
        // Beam at (-150, 200) — well off the "detector" coordinate system.
        // Visible peaks: arcs of three concentric circles, only the part in
        // the +x half-plane (i.e. on the actual detector).
        let truth = (-150.0_f64, 200.0_f64);
        let mut peaks = Vec::new();
        for r in [400.0, 500.0, 600.0] {
            peaks.extend(ring_points(truth.0, truth.1, r, 30, -PI / 3.0, PI / 3.0));
        }
        let opts = BeamCenterOptions {
            max_trials: 300,
            radial_bin_width_px: 3.0,
            top_k_bins: 5,
            max_radius_px: 1000.0,
            rng_seed: 13,
        };
        let c = estimate_beam_center(&peaks, &opts).expect("center ok");
        assert!((c.d1 - truth.0).abs() < 1.0, "cx {}", c.d1);
        assert!((c.d2 - truth.1).abs() < 1.0, "cy {}", c.d2);
    }
}
