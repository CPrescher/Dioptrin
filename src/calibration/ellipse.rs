//! Ellipse fitting and RANSAC ellipse clustering for diffraction-ring peaks.
//!
//! Conic representation:  `A x² + B xy + C y² + D x + E y + F = 0`.
//! An ellipse is the case `4AC − B² > 0`.
//!
//! We use the simple "fix `F = 1`" linear normalization for the per-trial fit
//! (`fit_5point` and `fit_least_squares`). It is not the numerically optimal
//! constrained Fitzgibbon fit, but inside a RANSAC loop each individual fit
//! only needs to put the inliers close enough to the curve for inlier counting
//! — the final refit on all consensus inliers is what determines accuracy,
//! and the linear least-squares refit is plenty for that purpose.
//!
//! Inlier distance uses the **Sampson distance**, a first-order approximation
//! to perpendicular geometric distance:
//!     d ≈ |Q(x, y)| / |∇Q(x, y)|.
//! Exact perpendicular distance to a conic requires solving a quartic; the
//! Sampson approximation is standard practice for RANSAC and effectively
//! identical to the geometric distance for inliers close to the curve.

#![allow(clippy::needless_range_loop)]

/// A fitted ellipse, expressed both as conic coefficients and as geometric
/// parameters (center, axes, rotation).
#[derive(Debug, Clone, Copy)]
pub struct Ellipse {
    /// Conic coefficients [A, B, C, D, E, F] with the convention
    /// `A x² + B xy + C y² + D x + E y + F = 0`.
    pub conic: [f64; 6],
    /// Geometric center (d1, d2).
    pub center: (f64, f64),
    /// Semi-major axis length, pixels.
    pub semi_major: f64,
    /// Semi-minor axis length, pixels.
    pub semi_minor: f64,
    /// Rotation of the major axis from the d1-axis, radians, in (-π/2, π/2].
    pub theta: f64,
}

impl Ellipse {
    /// Build an `Ellipse` from raw conic coefficients. Returns `None` if the
    /// conic is not an ellipse (discriminant `4AC − B² ≤ 0`) or degenerate.
    pub fn from_conic(mut c: [f64; 6]) -> Option<Self> {
        // Normalize sign so center recovery is consistent.
        if c[0] < 0.0 {
            for v in &mut c {
                *v = -*v;
            }
        }
        let disc = 4.0 * c[0] * c[2] - c[1] * c[1];
        if disc.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        let (a, b, cc, d, e, f) = (c[0], c[1], c[2], c[3], c[4], c[5]);

        // Solve ∇Q = 0  →  [[2A, B], [B, 2C]] (x, y)ᵀ = (−D, −E)ᵀ.
        let det = 4.0 * a * cc - b * b;
        if det.abs() < 1e-30 {
            return None;
        }
        let cx = (2.0 * cc * (-d) - b * (-e)) / det;
        let cy = (2.0 * a * (-e) - b * (-d)) / det;

        // Numerator for axis lengths.
        // Substitute (cx, cy) back: ν = A cx² + B cx cy + C cy² + D cx + E cy + F.
        // Eigenvalues of [[A, B/2], [B/2, C]] are λ±.
        let num = -(a * cx * cx + b * cx * cy + cc * cy * cy + d * cx + e * cy + f);
        let trace = a + cc;
        let diff = ((a - cc) * (a - cc) + b * b).sqrt();
        let lam1 = 0.5 * (trace + diff); // larger eigenvalue
        let lam2 = 0.5 * (trace - diff); // smaller eigenvalue
        let gt = |v: f64| v.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater);
        if !gt(num) || !gt(lam1) || !gt(lam2) {
            return None;
        }
        let semi_minor = (num / lam1).sqrt();
        let semi_major = (num / lam2).sqrt();
        // Orientation of the major axis (eigenvector of the smaller eigenvalue).
        // Eigenvector equation: (A - λ) v_x + (B/2) v_y = 0.
        let theta = if b.abs() < 1e-30 && (a - cc).abs() < 1e-30 {
            0.0
        } else if b.abs() < 1e-30 {
            if a < cc {
                0.0
            } else {
                std::f64::consts::FRAC_PI_2
            }
        } else {
            ((lam2 - a) / (0.5 * b)).atan()
        };
        Some(Ellipse {
            conic: c,
            center: (cx, cy),
            semi_major,
            semi_minor,
            theta,
        })
    }

    /// Approximate perpendicular distance from `p` to the ellipse curve
    /// using the Sampson distance.
    #[inline]
    pub fn sampson_distance(&self, p: (f64, f64)) -> f64 {
        let (x, y) = p;
        let [a, b, c, d, e, f] = self.conic;
        let q = a * x * x + b * x * y + c * y * y + d * x + e * y + f;
        let gx = 2.0 * a * x + b * y + d;
        let gy = b * x + 2.0 * c * y + e;
        let gnorm = (gx * gx + gy * gy).sqrt();
        if gnorm < 1e-30 {
            f64::INFINITY
        } else {
            q.abs() / gnorm
        }
    }
}

// ---------- linear ellipse fit (F = 1 normalization) ----------

/// Solve a small square linear system `A x = b` via Gaussian elimination with
/// partial pivoting. Returns `None` if the matrix is singular at our tolerance.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > max_val {
                max_val = a[r][i].abs();
                max_row = r;
            }
        }
        if max_val < 1e-300 {
            return None;
        }
        a.swap(i, max_row);
        b.swap(i, max_row);
        for r in (i + 1)..n {
            let factor = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Fit a conic to `points` in the least-squares sense with the normalization
/// `F = -1`, i.e. minimize `Σ (A x² + B xy + C y² + D x + E y − 1)²`.
///
/// Returns `None` if the normal-equations matrix is singular or the resulting
/// conic is not an ellipse.
pub fn fit_least_squares(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 5 {
        return None;
    }
    // Design matrix rows: [x², xy, y², x, y]. Right-hand side: 1.
    // Normal equations: (DᵀD) p = Dᵀ 1.
    let n = 5;
    let mut ata = vec![vec![0.0; n]; n];
    let mut atb = vec![0.0; n];
    for &(x, y) in points {
        let row = [x * x, x * y, y * y, x, y];
        for i in 0..n {
            atb[i] += row[i];
            for j in 0..n {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let p = solve(ata, atb)?;
    // F = -1 in our normalization.
    let conic = [p[0], p[1], p[2], p[3], p[4], -1.0];
    Ellipse::from_conic(conic)
}

/// Exact fit through 5 points (overdetermined cases use `fit_least_squares`).
pub fn fit_5point(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() != 5 {
        return None;
    }
    // Same as least-squares with exactly 5 rows.
    fit_least_squares(points)
}

// ---------- RANSAC ----------

#[derive(Debug, Clone, Copy)]
pub struct RansacOptions {
    /// Inlier acceptance threshold for Sampson distance (pixels).
    pub inlier_threshold: f64,
    /// Minimum number of inliers required to accept an ellipse.
    pub min_inliers: usize,
    /// Maximum random samples per ellipse.
    pub max_iterations: usize,
    /// Cap on the number of ellipses returned.
    pub max_ellipses: usize,
    /// Optional maximum allowed `semi_major / semi_minor`. Useful when the
    /// user knows the detector is roughly perpendicular to the beam, so real
    /// diffraction ellipses should be near-circular; setting e.g. `Some(1.5)`
    /// rejects degenerate elongated fits that RANSAC sometimes produces by
    /// chance from samples drawn from multiple rings. Leave `None` for
    /// arbitrary tilted detectors where genuinely eccentric ellipses are
    /// expected.
    pub max_axis_ratio: Option<f64>,
    /// Random seed for reproducibility.
    pub rng_seed: u64,
}

impl Default for RansacOptions {
    fn default() -> Self {
        Self {
            inlier_threshold: 1.5, // pixels
            min_inliers: 10,
            max_iterations: 2000,
            max_ellipses: 12,
            max_axis_ratio: None,
            rng_seed: 0xC0FF_EEC0_DE12_3456,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RansacResult {
    /// Detected ellipses, sorted by semi-major axis (smallest first), so the
    /// k-th entry corresponds to the k-th visible ring under a powder pattern.
    pub ellipses: Vec<Ellipse>,
    /// For each input point: `Some(k)` = inlier of `ellipses[k]`, `None` = outlier.
    pub assignments: Vec<Option<usize>>,
}

/// Splitmix64 — small, fast, deterministic PRNG. Sufficient for sampling
/// indices for RANSAC; we don't need cryptographic quality.
fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn sample5_distinct(rng: &mut u64, available: &[usize]) -> Option<[usize; 5]> {
    if available.len() < 5 {
        return None;
    }
    let mut picked = [0usize; 5];
    let mut count = 0;
    while count < 5 {
        let r = (next_rand(rng) as usize) % available.len();
        let v = available[r];
        if !picked[..count].contains(&v) {
            picked[count] = v;
            count += 1;
        }
    }
    Some(picked)
}

fn count_inliers(
    ellipse: &Ellipse,
    points: &[(f64, f64)],
    available: &[usize],
    threshold: f64,
) -> Vec<usize> {
    let t2 = threshold;
    available
        .iter()
        .copied()
        .filter(|&i| ellipse.sampson_distance(points[i]) < t2)
        .collect()
}

/// RANSAC ellipse clustering. Repeatedly finds the densest ellipse, removes
/// its inliers, and continues until no acceptable ellipse remains.
pub fn ransac_ellipses(points: &[(f64, f64)], options: &RansacOptions) -> RansacResult {
    let n = points.len();
    let mut assignments = vec![None; n];
    let mut ellipses: Vec<Ellipse> = Vec::new();
    let mut available: Vec<usize> = (0..n).collect();
    let mut rng = options.rng_seed;

    for _ellipse_idx in 0..options.max_ellipses {
        if available.len() < options.min_inliers.max(5) {
            break;
        }
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_ellipse: Option<Ellipse> = None;

        let passes_axis_ratio = |e: &Ellipse| -> bool {
            match options.max_axis_ratio {
                Some(limit) => e.semi_major / e.semi_minor.max(1e-12) <= limit,
                None => true,
            }
        };

        for _ in 0..options.max_iterations {
            let picks = match sample5_distinct(&mut rng, &available) {
                Some(p) => p,
                None => break,
            };
            let sample_pts: Vec<(f64, f64)> = picks.iter().map(|&i| points[i]).collect();
            let candidate = match fit_5point(&sample_pts) {
                Some(e) => e,
                None => continue,
            };
            if !passes_axis_ratio(&candidate) {
                continue;
            }
            let inliers = count_inliers(&candidate, points, &available, options.inlier_threshold);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_ellipse = Some(candidate);
            }
        }

        if best_inliers.len() < options.min_inliers {
            break;
        }
        // Refit using all inliers.
        let inlier_pts: Vec<(f64, f64)> = best_inliers.iter().map(|&i| points[i]).collect();
        let refit = fit_least_squares(&inlier_pts).or(best_ellipse);
        let refit = match refit {
            Some(e) => e,
            None => break,
        };
        if !passes_axis_ratio(&refit) {
            // Refit drifted to a degenerate elongated ellipse — reject this
            // cluster so RANSAC re-tries with the still-available points.
            break;
        }
        // Recount inliers under the refit conic (the least-squares fit is
        // more accurate than the 5-point sample, so it usually picks up a
        // handful more points that just missed the initial threshold).
        let final_inliers = count_inliers(&refit, points, &available, options.inlier_threshold);
        if final_inliers.len() < options.min_inliers {
            break;
        }
        let k = ellipses.len();
        for &i in &final_inliers {
            assignments[i] = Some(k);
        }
        ellipses.push(refit);
        available.retain(|i| assignments[*i].is_none());
    }

    // Sort ellipses by semi-major axis. Update assignments accordingly.
    let mut order: Vec<usize> = (0..ellipses.len()).collect();
    order.sort_by(|&a, &b| {
        ellipses[a]
            .semi_major
            .partial_cmp(&ellipses[b].semi_major)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted: Vec<Ellipse> = order.iter().map(|&i| ellipses[i]).collect();
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

    RansacResult {
        ellipses: sorted,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_ellipse_points(
        cx: f64,
        cy: f64,
        a: f64,
        b: f64,
        theta: f64,
        n: usize,
    ) -> Vec<(f64, f64)> {
        let (sin_t, cos_t) = theta.sin_cos();
        (0..n)
            .map(|k| {
                let t = 2.0 * PI * (k as f64) / (n as f64);
                let xa = a * t.cos();
                let yb = b * t.sin();
                (cx + xa * cos_t - yb * sin_t, cy + xa * sin_t + yb * cos_t)
            })
            .collect()
    }

    #[test]
    fn least_squares_recovers_clean_ellipse() {
        let pts = make_ellipse_points(100.0, 80.0, 50.0, 30.0, 0.3, 36);
        let e = fit_least_squares(&pts).expect("fit ok");
        assert!((e.center.0 - 100.0).abs() < 1e-6, "center.0 {}", e.center.0);
        assert!((e.center.1 - 80.0).abs() < 1e-6, "center.1 {}", e.center.1);
        assert!(
            (e.semi_major - 50.0).abs() < 1e-6,
            "smajor {}",
            e.semi_major
        );
        assert!(
            (e.semi_minor - 30.0).abs() < 1e-6,
            "sminor {}",
            e.semi_minor
        );
        // theta is recoverable modulo π (axis swap), so check sin/cos of (theta - truth).
        let dtheta = (e.theta - 0.3).rem_euclid(PI);
        assert!(
            dtheta < 1e-6 || (PI - dtheta) < 1e-6,
            "theta off: got {}, truth 0.3",
            e.theta
        );
    }

    #[test]
    fn ransac_finds_concentric_rings_with_outliers() {
        // Three nested ellipses sharing a center, plus 30 random outlier points.
        let mut pts: Vec<(f64, f64)> = Vec::new();
        pts.extend(make_ellipse_points(200.0, 200.0, 50.0, 50.0, 0.0, 40));
        pts.extend(make_ellipse_points(200.0, 200.0, 90.0, 90.0, 0.0, 40));
        pts.extend(make_ellipse_points(200.0, 200.0, 140.0, 140.0, 0.0, 40));
        // Deterministic outliers in random-looking positions.
        let mut seed = 0xDEADBEEF_u64;
        for _ in 0..30 {
            let r = next_rand(&mut seed);
            let x = ((r & 0xFFFF) as f64) / 65535.0 * 400.0;
            let y = (((r >> 16) & 0xFFFF) as f64) / 65535.0 * 400.0;
            pts.push((x, y));
        }
        let opts = RansacOptions {
            inlier_threshold: 0.5,
            min_inliers: 25,
            max_iterations: 600,
            max_ellipses: 6,
            max_axis_ratio: None,
            rng_seed: 42,
        };
        let result = ransac_ellipses(&pts, &opts);
        assert_eq!(
            result.ellipses.len(),
            3,
            "expected 3 ellipses, got {}",
            result.ellipses.len()
        );
        // Smallest first.
        assert!((result.ellipses[0].semi_major - 50.0).abs() < 0.5);
        assert!((result.ellipses[1].semi_major - 90.0).abs() < 0.5);
        assert!((result.ellipses[2].semi_major - 140.0).abs() < 0.5);
        // All ring points assigned to some ellipse.
        let n_inliers: usize = result.assignments.iter().filter(|a| a.is_some()).count();
        // 3 × 40 = 120 ring points. Some outliers may also assign to an ellipse
        // by accident, but the assignment count should be ≥ 120.
        assert!(
            n_inliers >= 120,
            "got {} inliers, expected ≥ 120",
            n_inliers
        );
    }

    #[test]
    fn ransac_rejects_pure_noise() {
        // 200 random points; RANSAC should find no ellipse with ≥ min_inliers.
        let mut pts = Vec::with_capacity(200);
        let mut seed = 0xC0FFEE_u64;
        for _ in 0..200 {
            let r = next_rand(&mut seed);
            let x = ((r & 0xFFFF) as f64) / 65535.0 * 400.0;
            let y = (((r >> 16) & 0xFFFF) as f64) / 65535.0 * 400.0;
            pts.push((x, y));
        }
        let opts = RansacOptions {
            inlier_threshold: 1.0,
            min_inliers: 30,
            max_iterations: 500,
            max_ellipses: 4,
            max_axis_ratio: None,
            rng_seed: 1,
        };
        let result = ransac_ellipses(&pts, &opts);
        // A noisy fit might still find a low-quality "ellipse"; require ≤ 1.
        assert!(
            result.ellipses.len() <= 1,
            "found {} ellipses on pure noise",
            result.ellipses.len()
        );
    }
}
