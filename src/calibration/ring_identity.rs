//! Map a list of detected ellipses (sorted ascending by semi-major axis) to
//! calibrant ring indices using the size-ratio invariant.
//!
//! ## Why ratios
//!
//! For a flat detector at distance `D` (no tilt), the projected radius of
//! the Debye-Scherrer cone for ring `k` is
//!
//!     r_k = D · tan(2θ_k)
//!
//! Therefore `r_k / r_0 = tan(2θ_k) / tan(2θ_0)`. The distance `D` cancels
//! and the *ratio sequence* of ring sizes is determined entirely by the
//! calibrant's d-spacings and the wavelength — both known a priori.
//!
//! For tilted detectors the same invariant holds approximately if we use the
//! ellipse's semi-major axis as `r_k`; the deviation is second-order in the
//! tilt angles. In the small-tilt regime typical of real calibrations
//! (a few mrad), the approximation is well within the match tolerance.
//!
//! ## Algorithm
//!
//! 1. Compute the calibrant's `R_k = tan(2θ_k)` for every ring whose Bragg
//!    condition is physical.
//! 2. Observed values: `s_i = semi_major_i` for each detected ellipse,
//!    sorted ascending.
//! 3. For each `(anchor_ellipse, anchor_ring)` pair, walk through other
//!    ellipses and find the calibrant ring whose ratio matches the observed
//!    ratio (`R_k / R_{k_0}  ≈  s_i / s_anchor`) within relative tolerance.
//!    Score by number of matched ellipses; among ties, pick the hypothesis
//!    with the tightest distance-consistency (`D = s_i / R_k` should agree
//!    across all matched pairs).
//! 4. Score = number of matched ellipses. Pick the highest-scoring anchor.
//!
//! Naturally handles:
//! - Missing calibrant rings: just skipped during the search
//! - Spurious ellipses (not on any real ring): marked unmatched
//! - Detected ellipses beyond the calibrant's largest ring: unmatched
//! - "Ellipse 0 isn't ring 0": the anchor enumeration finds the right `k₀`.

use super::calibrant::Calibrant;
use super::ellipse::Ellipse;

#[derive(Debug, Clone, Copy)]
pub struct RingIdOptions {
    /// Relative tolerance on `s_i/s_0` vs `R_k/R_{k_0}` for a match.
    /// 0.02 (2%) is a generous default that survives moderate tilt-induced
    /// deviation from the pure flat-detector ratio.
    pub ratio_tolerance: f64,
    /// Cap on anchor hypotheses to test (anchors = which calibrant ring is
    /// the smallest detected ellipse). For most calibrants the right anchor
    /// is `k₀ ∈ {0, 1, 2}`; testing further is cheap insurance.
    pub max_anchor_ring: usize,
    /// Minimum number of matched ellipses needed to call the result
    /// "confident". Returned in `RingIdResult` so the caller can decide
    /// whether to trust the assignment.
    pub min_matches: usize,
}

impl Default for RingIdOptions {
    fn default() -> Self {
        Self {
            ratio_tolerance: 0.02,
            max_anchor_ring: 8,
            min_matches: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RingIdResult {
    /// Per detected ellipse: `Some(k)` = calibrant ring index, `None` = no match.
    pub ring_index: Vec<Option<usize>>,
    /// Calibrant ring assigned to the smallest detected ellipse (i.e.
    /// `ring_index[0]`). `None` if that ellipse is a spurious outlier the
    /// algorithm couldn't identify, or if no anchor hypothesis reached
    /// `min_matches`.
    pub anchor_ring: Option<usize>,
    /// Number of ellipses that received a ring assignment.
    pub n_matched: usize,
    /// `true` iff `n_matched >= options.min_matches`.
    pub confident: bool,
}

/// Identify which calibrant ring each detected ellipse corresponds to,
/// using the size-ratio invariant.
pub fn identify_rings(
    ellipses: &[Ellipse],
    calibrant: &Calibrant,
    wavelength_m: f64,
    options: &RingIdOptions,
) -> RingIdResult {
    let n_obs = ellipses.len();
    if n_obs == 0 {
        return RingIdResult {
            ring_index: Vec::new(),
            anchor_ring: None,
            n_matched: 0,
            confident: false,
        };
    }

    // Calibrant reference values R_k = tan(2θ_k) for every physical ring.
    let two_th = calibrant.two_theta_rad(wavelength_m);
    let mut ring_ref: Vec<(usize, f64)> = Vec::new();
    for (k, opt) in two_th.iter().enumerate() {
        if let Some(t) = opt {
            let r = t.tan();
            if r.is_finite() && r > 0.0 {
                ring_ref.push((k, r));
            }
        }
    }
    if ring_ref.is_empty() {
        return RingIdResult {
            ring_index: vec![None; n_obs],
            anchor_ring: None,
            n_matched: 0,
            confident: false,
        };
    }

    let mut best_score = 0usize;
    let mut best_spread = f64::INFINITY;
    let mut best_assignment: Vec<Option<usize>> = vec![None; n_obs];

    let r_lookup = |k: usize| ring_ref.iter().find(|(idx, _)| *idx == k).map(|t| t.1);

    let max_anchor_k = options.max_anchor_ring.min(ring_ref.len() - 1);

    // Iterate over every (anchor_ellipse, anchor_ring) pair. This makes the
    // algorithm robust to spurious ellipses smaller than ring 0 — if we
    // anchored only on the smallest ellipse, a spurious outlier in front of
    // the real rings would corrupt every ratio. By trying each ellipse as
    // the anchor, the *real* ring-0 ellipse can win on the match count.
    for anchor_obs in 0..n_obs {
        let s_anchor = ellipses[anchor_obs].semi_major;
        if s_anchor.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        for anchor_idx in 0..=max_anchor_k {
            let (k0, r0) = ring_ref[anchor_idx];
            let mut assignment = vec![None; n_obs];
            assignment[anchor_obs] = Some(k0);
            let mut matches = 1usize;
            let mut distances: Vec<f64> = vec![s_anchor / r0];

            // Forward: ellipses larger than the anchor → rings with k > k0.
            let mut last_k = k0;
            for i in (anchor_obs + 1)..n_obs {
                let s_i = ellipses[i].semi_major;
                let target = (s_i / s_anchor) * r0;
                let mut best: Option<(usize, f64)> = None;
                for &(k, r) in &ring_ref {
                    if k <= last_k {
                        continue;
                    }
                    let rel_err = (r - target).abs() / target.max(1e-30);
                    if rel_err <= options.ratio_tolerance
                        && best.map(|(_, e)| rel_err < e).unwrap_or(true)
                    {
                        best = Some((k, rel_err));
                    }
                }
                if let Some((k, _)) = best {
                    assignment[i] = Some(k);
                    last_k = k;
                    matches += 1;
                    if let Some(r_k) = r_lookup(k) {
                        distances.push(s_i / r_k);
                    }
                }
            }

            // Backward: ellipses smaller than the anchor → rings with k < k0.
            let mut last_k = k0;
            for i in (0..anchor_obs).rev() {
                let s_i = ellipses[i].semi_major;
                let target = (s_i / s_anchor) * r0;
                let mut best: Option<(usize, f64)> = None;
                for &(k, r) in &ring_ref {
                    if k >= last_k {
                        continue;
                    }
                    let rel_err = (r - target).abs() / target.max(1e-30);
                    if rel_err <= options.ratio_tolerance
                        && best.map(|(_, e)| rel_err < e).unwrap_or(true)
                    {
                        best = Some((k, rel_err));
                    }
                }
                if let Some((k, _)) = best {
                    assignment[i] = Some(k);
                    last_k = k;
                    matches += 1;
                    if let Some(r_k) = r_lookup(k) {
                        distances.push(s_i / r_k);
                    }
                }
            }

            // D-consistency: the implied detector distance D = s_i / R_k
            // should agree across all matched pairs. The right anchor has
            // tiny spread; wrong anchors with coincidentally-consistent
            // ratios produce noticeably worse agreement.
            let spread = if distances.len() >= 2 {
                let mean = distances.iter().sum::<f64>() / distances.len() as f64;
                distances
                    .iter()
                    .map(|d| (d - mean).abs() / mean.max(1e-30))
                    .fold(0.0_f64, f64::max)
            } else {
                0.0
            };
            if spread > options.ratio_tolerance {
                continue;
            }
            if matches < options.min_matches {
                continue;
            }

            // Among hypotheses passing min_matches and the spread cap,
            // pick the one with the *tightest* spread — that's the right
            // anchor. (Density of LaB6-like calibrants creates multiple
            // many-match hypotheses at scaled distances, but only one has
            // ~0 spread.)
            if spread < best_spread {
                best_score = matches;
                best_spread = spread;
                best_assignment = assignment;
            }
        }
    }
    let confident = best_score >= options.min_matches;
    if !confident {
        best_assignment = vec![None; n_obs];
    }
    let best_anchor = best_assignment.first().copied().flatten();

    RingIdResult {
        ring_index: best_assignment,
        anchor_ring: best_anchor,
        n_matched: if confident { best_score } else { 0 },
        confident,
    }
}

#[cfg(test)]
mod tests {
    use super::super::ellipse::Ellipse;
    use super::*;

    fn fake_ellipse(semi_major: f64) -> Ellipse {
        Ellipse {
            conic: [0.0; 6],
            center: (0.0, 0.0),
            semi_major,
            semi_minor: semi_major,
            theta: 0.0,
        }
    }

    fn lab6() -> Calibrant {
        Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").unwrap()
    }

    /// Build an ellipse set whose semi-major axes equal `D · tan(2θ_k)` for
    /// the chosen calibrant ring indices, simulating a perfect flat-detector
    /// observation.
    fn synth_ellipses(
        cal: &Calibrant,
        wavelength: f64,
        d_distance: f64,
        rings: &[usize],
    ) -> Vec<Ellipse> {
        let tth = cal.two_theta_rad(wavelength);
        rings
            .iter()
            .map(|&k| fake_ellipse(d_distance * tth[k].unwrap().tan()))
            .collect()
    }

    #[test]
    fn identify_full_lab6_sequence() {
        let cal = lab6();
        let wl = 1.5406e-10; // Cu Kα
        let dist = 1000.0; // arbitrary; ratios cancel
        let ellipses = synth_ellipses(&cal, wl, dist, &[0, 1, 2, 3, 4, 5]);
        let result = identify_rings(&ellipses, &cal, wl, &RingIdOptions::default());
        assert!(result.confident);
        assert_eq!(result.anchor_ring, Some(0));
        assert_eq!(
            result.ring_index,
            vec![Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)]
        );
        assert_eq!(result.n_matched, 6);
    }

    #[test]
    fn identify_with_missing_rings() {
        // Simulate that we only detected rings 0, 2, 5, 7 (missed 1, 3, 4, 6).
        let cal = lab6();
        let wl = 1.5406e-10;
        let dist = 800.0;
        let ellipses = synth_ellipses(&cal, wl, dist, &[0, 2, 5, 7]);
        let result = identify_rings(&ellipses, &cal, wl, &RingIdOptions::default());
        assert!(result.confident);
        assert_eq!(result.anchor_ring, Some(0));
        assert_eq!(result.ring_index, vec![Some(0), Some(2), Some(5), Some(7)]);
        assert_eq!(result.n_matched, 4);
    }

    #[test]
    fn identify_with_anchor_offset() {
        // The smallest detected ellipse is actually calibrant ring 2.
        let cal = lab6();
        let wl = 1.5406e-10;
        let dist = 500.0;
        let ellipses = synth_ellipses(&cal, wl, dist, &[2, 3, 4, 5, 6]);
        let result = identify_rings(&ellipses, &cal, wl, &RingIdOptions::default());
        assert!(result.confident);
        assert_eq!(result.anchor_ring, Some(2));
        assert_eq!(
            result.ring_index,
            vec![Some(2), Some(3), Some(4), Some(5), Some(6)]
        );
    }

    #[test]
    fn identify_with_spurious_extra_ellipse() {
        // Real LaB6 rings + one spurious ellipse with a size that lies below
        // the smallest real ring. With a dense calibrant like LaB6 (91 rings)
        // the algorithm must distinguish "spurious ellipse + 4 real rings"
        // from "match the spurious to some ring k₀ ≠ 0 and see whether a few
        // accidental ratios fall within tolerance".
        let cal = lab6();
        let wl = 1.5406e-10;
        let dist = 600.0;
        let mut ellipses = synth_ellipses(&cal, wl, dist, &[0, 1, 2, 3]);
        // Insert a fake ellipse whose size is unrelated to any LaB6 ring.
        ellipses.push(fake_ellipse(dist * 0.18));
        // Sort by semi_major to mimic ransac_ellipses output.
        ellipses.sort_by(|a, b| a.semi_major.partial_cmp(&b.semi_major).unwrap());
        let result = identify_rings(&ellipses, &cal, wl, &RingIdOptions::default());
        assert!(result.confident);
        // Each real ring should be identified; the spurious one should be None.
        // After sorting, the order is: [spurious(108), ring0, ring1, ring2, ring3].
        assert_eq!(
            result.ring_index,
            vec![None, Some(0), Some(1), Some(2), Some(3)],
            "expected the spurious smallest ellipse to be unmatched"
        );
        assert_eq!(result.n_matched, 4);
    }

    #[test]
    fn identify_rejects_random_sizes() {
        // 4 ellipses with sizes unrelated to LaB6 → no confident match.
        let cal = lab6();
        let wl = 1.5406e-10;
        let ellipses = vec![
            fake_ellipse(50.0),
            fake_ellipse(57.0),
            fake_ellipse(63.0),
            fake_ellipse(71.0),
        ];
        let result = identify_rings(&ellipses, &cal, wl, &RingIdOptions::default());
        // Heuristically the algorithm may match nothing OR find 1 spurious
        // pair by accident — but it should not be confident.
        if result.confident {
            // At most 2 random sizes can survive a 2% tolerance by chance.
            assert!(
                result.n_matched <= 2,
                "matched {} random-size ellipses, suspicious",
                result.n_matched
            );
        }
    }
}
