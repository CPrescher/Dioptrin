#![allow(clippy::too_many_arguments, clippy::type_complexity)]
//! Coarse grid search bootstrap for `auto_calibrate`.
//!
//! Bypasses ellipse fitting entirely. Given a set of peak coordinates and a
//! calibrant, we sample a coarse grid in `(distance, rot1, rot2, poni1, poni2)`
//! and score each candidate by summing Gaussian weights of peak residuals
//! within a 2θ tolerance of the nearest calibrant ring. The grid winner is then handed to L-M
//! refinement which can recover the exact geometry from a reasonable basin.
//!
//! Why this works where ellipse-fit-based bootstraps fail:
//!
//! - On a strip detector or any geometry that shows only short arcs of each
//!   ring, 5-point ellipse RANSAC is numerically degenerate — a 30-px arc fits
//!   many ellipses with very different eccentricities to within tolerance.
//! - The 2θ forward model is well-behaved everywhere on the detector, so a
//!   direct "is this peak near a predicted ring?" score is robust.
//! - Scoring is parallel across grid cells; cost scales with the number of
//!   cells, sampled peaks, and calibrant rings.

use super::calibrant::Calibrant;
use crate::poni::Poni;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct GridSearchOptions {
    /// Distance search range in metres, sampled logarithmically.
    pub distance_min: f64,
    pub distance_max: f64,
    pub distance_steps: usize,
    /// Half-range of the rotation grid in radians: rot1, rot2 ∈ [-rot_max, +rot_max].
    pub rot_max: f64,
    pub rot_steps: usize,
    /// PONI hint search radius in pixels (around `(poni_d1_hint_px, poni_d2_hint_px)`).
    pub poni_search_radius_px: f64,
    /// Number of PONI grid points per axis (1 = use hint only).
    pub poni_grid_steps: usize,
    /// 2θ tolerance (radians) — a peak is an "inlier" for a candidate
    /// geometry when its predicted 2θ is within this much of any
    /// calibrant ring.
    pub tolerance_rad: f64,
    /// Cap peaks used for grid scoring (uniform stride subsample).
    /// L-M refinement afterwards still uses all peaks.
    pub max_peaks_for_grid: usize,
    /// Maximum calibrant ring index considered when scoring.
    pub max_ring_index: usize,
    /// Return the top-N grid cells (by Gaussian-weighted residual score) for
    /// downstream L-M refine-and-pick disambiguation.
    pub top_k: usize,
}

impl Default for GridSearchOptions {
    fn default() -> Self {
        Self {
            distance_min: 0.05,
            distance_max: 0.6,
            distance_steps: 14,
            rot_max: 60.0_f64.to_radians(),
            rot_steps: 13,
            poni_search_radius_px: 200.0,
            poni_grid_steps: 5,
            tolerance_rad: 2.0e-3,
            max_peaks_for_grid: 400,
            max_ring_index: 12,
            top_k: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub poni: Poni,
    /// Inlier count at the winning cell.
    pub n_inliers: usize,
    /// Total grid cells evaluated.
    pub n_cells: usize,
}

#[derive(Debug, Clone)]
pub struct GridSearchCandidate {
    pub poni: Poni,
    pub n_inliers: usize,
    pub n_rings_hit: usize,
}

#[derive(Copy, Clone)]
struct TrigConsts {
    sr1: f64,
    cr1: f64,
    sr2: f64,
    cr2: f64,
}

impl TrigConsts {
    fn new(rot1: f64, rot2: f64) -> Self {
        Self {
            sr1: rot1.sin(),
            cr1: rot1.cos(),
            sr2: rot2.sin(),
            cr2: rot2.cos(),
        }
    }
}

/// 2θ at a detector pixel — inlined, rot3 = 0, pyFAI pixel-center convention.
#[inline]
fn tth_at(
    d1: f64,
    d2: f64,
    distance: f64,
    poni1: f64,
    poni2: f64,
    pixel1: f64,
    pixel2: f64,
    tc: &TrigConsts,
) -> f64 {
    let p1 = (d1 + 0.5) * pixel1 - poni1;
    let p2 = (d2 + 0.5) * pixel2 - poni2;
    let p3 = distance;
    // Forward model from src/calibration/refinement.rs::tth_at, specialised for rot3=0.
    let x = p1 * tc.cr2 + p2 * tc.sr1 * tc.sr2 - p3 * tc.cr1 * tc.sr2;
    let y = p2 * tc.cr1 + p3 * tc.sr1;
    let z = p1 * tc.sr2 - p2 * tc.cr2 * tc.sr1 + p3 * tc.cr1 * tc.cr2;
    (x * x + y * y).sqrt().atan2(z)
}

/// Linear range helper: equally spaced from `lo` to `hi` with `n` points
/// (or single value `lo` if `n == 1`).
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// Geometric range helper: log-spaced from `lo` to `hi` with `n` points.
fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let ratio = (hi / lo).powf(1.0 / (n - 1) as f64);
    (0..n).map(|i| lo * ratio.powi(i as i32)).collect()
}

/// Grid search for a starting geometry.
///
/// `poni_d1_hint_px`, `poni_d2_hint_px` — best guess for the PONI in pixel
/// coordinates (typically the centroid of the peaks, or an explicit user hint).
/// The grid samples a square of radius `poni_search_radius_px` around this point.
pub fn grid_search(
    peaks: &[(f64, f64)],
    poni_d1_hint_px: f64,
    poni_d2_hint_px: f64,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &GridSearchOptions,
) -> Option<GridSearchResult> {
    if peaks.is_empty() {
        return None;
    }

    let tth_targets: Vec<f64> = calibrant
        .two_theta_rad(wavelength_m)
        .into_iter()
        .take(options.max_ring_index)
        .flatten()
        .collect();
    if tth_targets.is_empty() {
        return None;
    }

    // Deterministic uniform-stride subsample of peaks.
    let working: Vec<(f64, f64)> = if peaks.len() > options.max_peaks_for_grid {
        let step = peaks.len().div_ceil(options.max_peaks_for_grid);
        peaks.iter().step_by(step).copied().collect()
    } else {
        peaks.to_vec()
    };

    let distances = logspace(
        options.distance_min,
        options.distance_max,
        options.distance_steps,
    );
    let rots = linspace(-options.rot_max, options.rot_max, options.rot_steps);
    let poni_offsets = linspace(
        -options.poni_search_radius_px,
        options.poni_search_radius_px,
        options.poni_grid_steps,
    );

    let mut cells: Vec<(f64, f64, f64, f64, f64)> =
        Vec::with_capacity(distances.len() * rots.len().pow(2) * poni_offsets.len().pow(2));
    for &d in &distances {
        for &r1 in &rots {
            for &r2 in &rots {
                for &off1 in &poni_offsets {
                    for &off2 in &poni_offsets {
                        let p1 = (poni_d1_hint_px + off1 + 0.5) * pixel1;
                        let p2 = (poni_d2_hint_px + off2 + 0.5) * pixel2;
                        cells.push((d, p1, p2, r1, r2));
                    }
                }
            }
        }
    }
    if cells.is_empty() {
        return None;
    }

    let tol = options.tolerance_rad;
    // Gaussian-weighted score: each peak with residual `r` to its nearest
    // ring contributes `exp(-r² / 2σ²)` with `σ = 0.4 · tol`. Peaks at
    // residual ~0 score ≈ 1; peaks at residual tol score ≈ 0.044. This
    // rewards tight concentration around predicted rings (the true basin)
    // over loose "many peaks within tolerance" (wrong-anchor basins).
    let sigma = (0.4 * tol).max(1.0e-6);
    let two_sigma2 = 2.0 * sigma * sigma;
    let scored: Vec<(f64, usize, (f64, f64, f64, f64, f64))> = cells
        .par_iter()
        .map(|&(d, p1, p2, r1, r2)| {
            let tc = TrigConsts::new(r1, r2);
            let mut weight_sum = 0.0_f64;
            let mut count = 0usize;
            for &(d1, d2) in &working {
                let tth = tth_at(d1, d2, d, p1, p2, pixel1, pixel2, &tc);
                let mut best_r = tol;
                for &t in &tth_targets {
                    let r = (tth - t).abs();
                    if r < best_r {
                        best_r = r;
                    }
                }
                if best_r < tol {
                    weight_sum += (-(best_r * best_r) / two_sigma2).exp();
                    count += 1;
                }
            }
            (weight_sum, count, (d, p1, p2, r1, r2))
        })
        .collect();

    let &(_, best_count, (d, p1, p2, r1, r2)) = scored
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .filter(|s| s.1 > 0)?;

    let mut poni = Poni::new();
    poni.pixel1 = pixel1;
    poni.pixel2 = pixel2;
    poni.distance = d;
    poni.poni1 = p1;
    poni.poni2 = p2;
    poni.rot1 = r1;
    poni.rot2 = r2;
    poni.rot3 = 0.0;
    poni.wavelength = wavelength_m;

    Some(GridSearchResult {
        poni,
        n_inliers: best_count,
        n_cells: cells.len(),
    })
}

/// Precomputed O(1) lookup table mapping a 2θ value to its nearest calibrant
/// ring, with a spacing-aware tolerance so dense high-order rings cannot absorb
/// arbitrary peaks at an artificially tiny detector distance while never
/// shrinking below the coarse grid discretization error.
struct RingLut {
    inv_step: f64,
    /// `(tth_near, ring_idx, local_tol, inv_two_sigma2)` per bin.
    entries: Vec<(f64, u16, f64, f64)>,
}

impl RingLut {
    const STEP: f64 = 2.0e-4;

    fn build(tth_targets: &[f64], tol: f64) -> Option<Self> {
        if tth_targets.is_empty() {
            return None;
        }
        let max_tth = tth_targets.iter().copied().fold(0.0_f64, f64::max) + tol + Self::STEP;
        let n_bins = ((max_tth / Self::STEP).ceil() as usize).clamp(1, 20_000);

        let n_rings = tth_targets.len();
        let min_local_tol = (0.15 * tol).min(1.0e-3);
        let mut ring_params: Vec<(f64, f64)> = Vec::with_capacity(n_rings);
        for k in 0..n_rings {
            let gap_lo = if k > 0 {
                (tth_targets[k] - tth_targets[k - 1]).abs()
            } else {
                f64::INFINITY
            };
            let gap_hi = if k + 1 < n_rings {
                (tth_targets[k + 1] - tth_targets[k]).abs()
            } else {
                f64::INFINITY
            };
            let spacing = gap_lo.min(gap_hi);
            let local_tol = if spacing.is_finite() {
                tol.min(0.45 * spacing).max(min_local_tol)
            } else {
                tol
            };
            let sigma = (0.4 * local_tol).max(1.0e-6);
            ring_params.push((local_tol, 0.5 / (sigma * sigma)));
        }

        let mut entries = Vec::with_capacity(n_bins);
        let mut cursor = 0usize;
        for b in 0..n_bins {
            let tth = (b as f64 + 0.5) * Self::STEP;
            while cursor + 1 < n_rings
                && (tth_targets[cursor + 1] - tth).abs() <= (tth_targets[cursor] - tth).abs()
            {
                cursor += 1;
            }
            let (local_tol, inv_two_sigma2) = ring_params[cursor];
            entries.push((
                tth_targets[cursor],
                cursor as u16,
                local_tol,
                inv_two_sigma2,
            ));
        }

        Some(Self {
            inv_step: 1.0 / Self::STEP,
            entries,
        })
    }

    #[inline]
    fn lookup(&self, tth: f64) -> Option<(f64, u16, f64, f64)> {
        if !tth.is_finite() || tth < 0.0 {
            return None;
        }
        let idx = (tth * self.inv_step) as usize;
        self.entries.get(idx).copied()
    }
}

/// Shared parallel scorer for a list of `(d, p1, p2, r1, r2)` grid cells.
///
/// Ranks cells by Gaussian-weighted residual concentration and deduplicates
/// near-identical cells so `top_k` covers distinct basins.
fn score_cells_top_k(
    cells: &[(f64, f64, f64, f64, f64)],
    working: &[(f64, f64)],
    lut: &RingLut,
    pixel1: f64,
    pixel2: f64,
    wavelength_m: f64,
    top_k: usize,
) -> Vec<GridSearchCandidate> {
    if cells.is_empty() || working.is_empty() {
        return Vec::new();
    }

    let scored: Vec<(f64, usize, usize, (f64, f64, f64, f64, f64))> = cells
        .par_iter()
        .map(|&(d, p1, p2, r1, r2)| {
            let tc = TrigConsts::new(r1, r2);
            let mut ring_mask_lo = 0u64;
            let mut ring_mask_hi = 0u64;
            let mut weight_sum = 0.0_f64;
            let mut count = 0usize;
            for &(d1, d2) in working {
                let tth = tth_at(d1, d2, d, p1, p2, pixel1, pixel2, &tc);
                if let Some((t_near, k_near, local_tol, inv_two_sigma2)) = lut.lookup(tth) {
                    let r = (tth - t_near).abs();
                    if r < local_tol {
                        if k_near < 64 {
                            ring_mask_lo |= 1u64 << k_near;
                        } else if k_near < 128 {
                            ring_mask_hi |= 1u64 << (k_near - 64);
                        }
                        weight_sum += (-(r * r) * inv_two_sigma2).exp();
                        count += 1;
                    }
                }
            }
            let n_rings = (ring_mask_lo.count_ones() + ring_mask_hi.count_ones()) as usize;
            (weight_sum, n_rings, count, (d, p1, p2, r1, r2))
        })
        .collect();

    let mut sorted = scored;
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy basin deduplication so `top_k` explores distinct local optima.
    let k_target = top_k.max(1);
    let mut chosen: Vec<(f64, usize, usize, (f64, f64, f64, f64, f64))> =
        Vec::with_capacity(k_target);
    let p_tol1 = 35.0 * pixel1;
    let p_tol2 = 35.0 * pixel2;
    let rot_tol = 4.0_f64.to_radians();

    for entry in sorted {
        if entry.2 == 0 {
            continue;
        }
        let (d, p1, p2, r1, r2) = entry.3;
        let too_close = chosen.iter().any(|(_, _, _, (cd, cp1, cp2, cr1, cr2))| {
            (d / cd - 1.0).abs() < 0.08
                && (r1 - cr1).abs() < rot_tol
                && (r2 - cr2).abs() < rot_tol
                && (p1 - cp1).abs() < p_tol1
                && (p2 - cp2).abs() < p_tol2
        });
        if !too_close {
            chosen.push(entry);
            if chosen.len() >= k_target {
                break;
            }
        }
    }

    chosen
        .into_iter()
        .map(|(_weight, n_rings, n_inliers, (d, p1, p2, r1, r2))| {
            let mut poni = Poni::new();
            poni.pixel1 = pixel1;
            poni.pixel2 = pixel2;
            poni.distance = d;
            poni.poni1 = p1;
            poni.poni2 = p2;
            poni.rot1 = r1;
            poni.rot2 = r2;
            poni.rot3 = 0.0;
            poni.wavelength = wavelength_m;
            GridSearchCandidate {
                poni,
                n_inliers,
                n_rings_hit: n_rings,
            }
        })
        .collect()
}

/// Like `grid_search` but returns the top-K cells (sorted best-first by
/// Gaussian-weighted residual score). Each candidate is independent of the others;
/// the caller is expected to L-M refine each and pick the lowest-residual
/// result to disambiguate between basins with similar grid-score.
#[allow(clippy::too_many_arguments)]
pub fn grid_search_top_k(
    peaks: &[(f64, f64)],
    poni_d1_hint_px: f64,
    poni_d2_hint_px: f64,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &GridSearchOptions,
) -> Vec<GridSearchCandidate> {
    if peaks.is_empty() {
        return Vec::new();
    }
    let mut tth_targets: Vec<f64> = calibrant
        .two_theta_rad(wavelength_m)
        .into_iter()
        .take(options.max_ring_index)
        .flatten()
        .collect();
    tth_targets.sort_by(|a, b| a.total_cmp(b));
    let lut = match RingLut::build(&tth_targets, options.tolerance_rad) {
        Some(l) => l,
        None => return Vec::new(),
    };

    let working: Vec<(f64, f64)> = if peaks.len() > options.max_peaks_for_grid {
        let step = peaks.len().div_ceil(options.max_peaks_for_grid);
        peaks.iter().step_by(step).copied().collect()
    } else {
        peaks.to_vec()
    };

    let distances = logspace(
        options.distance_min,
        options.distance_max,
        options.distance_steps,
    );
    let rots = linspace(-options.rot_max, options.rot_max, options.rot_steps);
    let poni_offsets = linspace(
        -options.poni_search_radius_px,
        options.poni_search_radius_px,
        options.poni_grid_steps,
    );

    let mut cells: Vec<(f64, f64, f64, f64, f64)> =
        Vec::with_capacity(distances.len() * rots.len().pow(2) * poni_offsets.len().pow(2));
    for &d in &distances {
        for &r1 in &rots {
            for &r2 in &rots {
                for &off1 in &poni_offsets {
                    for &off2 in &poni_offsets {
                        let p1 = (poni_d1_hint_px + off1 + 0.5) * pixel1;
                        let p2 = (poni_d2_hint_px + off2 + 0.5) * pixel2;
                        cells.push((d, p1, p2, r1, r2));
                    }
                }
            }
        }
    }

    score_cells_top_k(
        &cells,
        &working,
        &lut,
        pixel1,
        pixel2,
        wavelength_m,
        options.top_k,
    )
}

/// Variant of [`grid_search_top_k`] parameterized around the **direct-beam
/// intersection** `(beam_d1_hint_px, beam_d2_hint_px)` rather than the PONI.
///
/// For a detector tilted by `(rot1, rot2)` while the beam stays near the
/// detector, the PONI shifts away from the beam by
/// `(-D · tan(rot2) / cos(rot1), +D · tan(rot1))`, which can easily exceed a
/// local PONI search box around the detector center. Parameterizing by the
/// beam position covers this regime directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grid_search_top_k_beam_centered(
    peaks: &[(f64, f64)],
    beam_d1_hint_px: f64,
    beam_d2_hint_px: f64,
    calibrant: &Calibrant,
    wavelength_m: f64,
    pixel1: f64,
    pixel2: f64,
    options: &GridSearchOptions,
) -> Vec<GridSearchCandidate> {
    if peaks.is_empty() {
        return Vec::new();
    }
    let mut tth_targets: Vec<f64> = calibrant
        .two_theta_rad(wavelength_m)
        .into_iter()
        .take(options.max_ring_index)
        .flatten()
        .collect();
    tth_targets.sort_by(|a, b| a.total_cmp(b));
    let lut = match RingLut::build(&tth_targets, options.tolerance_rad) {
        Some(l) => l,
        None => return Vec::new(),
    };

    let working: Vec<(f64, f64)> = if peaks.len() > options.max_peaks_for_grid {
        let step = peaks.len().div_ceil(options.max_peaks_for_grid);
        peaks.iter().step_by(step).copied().collect()
    } else {
        peaks.to_vec()
    };

    let distances = logspace(
        options.distance_min,
        options.distance_max,
        options.distance_steps,
    );
    let rots = linspace(-options.rot_max, options.rot_max, options.rot_steps);
    let beam_offsets = linspace(
        -options.poni_search_radius_px,
        options.poni_search_radius_px,
        options.poni_grid_steps,
    );

    let mut cells: Vec<(f64, f64, f64, f64, f64)> =
        Vec::with_capacity(distances.len() * rots.len().pow(2) * beam_offsets.len().pow(2));
    for &d in &distances {
        for &r1 in &rots {
            let (sr1, cr1) = r1.sin_cos();
            if cr1.abs() < 1e-4 {
                continue;
            }
            let shift2 = d * (sr1 / cr1);
            for &r2 in &rots {
                let shift1 = -d * r2.tan() / cr1;
                for &off1 in &beam_offsets {
                    let p1 = (beam_d1_hint_px + off1 + 0.5) * pixel1 + shift1;
                    for &off2 in &beam_offsets {
                        let p2 = (beam_d2_hint_px + off2 + 0.5) * pixel2 + shift2;
                        cells.push((d, p1, p2, r1, r2));
                    }
                }
            }
        }
    }

    score_cells_top_k(
        &cells,
        &working,
        &lut,
        pixel1,
        pixel2,
        wavelength_m,
        options.top_k,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::Calibrant;
    use crate::geometry::Geometry;

    fn synthetic_calibrant() -> Calibrant {
        // FCC, a = 5.6 Å (≈ CeO2)
        let a_ang = 5.6;
        let ds: Vec<f64> = [
            (1, 1, 1),
            (2, 0, 0),
            (2, 2, 0),
            (3, 1, 1),
            (2, 2, 2),
            (4, 0, 0),
        ]
        .iter()
        .map(|&(h, k, l)| {
            let h2 = (h * h + k * k + l * l) as f64;
            a_ang / h2.sqrt()
        })
        .collect();
        Calibrant::from_d_spacings("fake_fcc", ds)
    }

    /// Sample the detector with the real Geometry forward model and return
    /// pixel coordinates (pyFAI INDEX form: integer = pixel corner, center at
    /// index + 0.5) where 2θ ≈ ring_k. Works for arbitrary tilts.
    fn make_peaks_for(
        geometry: &Geometry,
        n_per_ring: usize,
        ring_count: usize,
    ) -> Vec<(f64, f64)> {
        let wavelength = geometry.poni.wavelength;
        let cal = synthetic_calibrant();
        let tths: Vec<f64> = cal
            .two_theta_rad(wavelength)
            .into_iter()
            .take(ring_count)
            .flatten()
            .collect();
        let dim1 = geometry.dim1() as f64;
        let dim2 = geometry.dim2() as f64;
        // Probe on a dense grid; INDEX coordinates (integer = pixel corner).
        let probe_n = 200usize;
        let mut idx_d1 = Vec::with_capacity(probe_n * probe_n);
        let mut idx_d2 = Vec::with_capacity(probe_n * probe_n);
        // Geometry::calc_tth applies `d * pixel` directly (no +0.5), so to
        // evaluate at pixel CENTER we pass `index + 0.5`.
        let mut center_d1 = Vec::with_capacity(probe_n * probe_n);
        let mut center_d2 = Vec::with_capacity(probe_n * probe_n);
        for i in 0..probe_n {
            for j in 0..probe_n {
                let i_idx = i as f64 * dim1 / probe_n as f64;
                let j_idx = j as f64 * dim2 / probe_n as f64;
                idx_d1.push(i_idx);
                idx_d2.push(j_idx);
                center_d1.push(i_idx + 0.5);
                center_d2.push(j_idx + 0.5);
            }
        }
        let probe_tth = geometry.calc_tth(&center_d1, &center_d2);

        let mut peaks = Vec::new();
        for &tth_target in &tths {
            let mut hits: Vec<(usize, f64)> = probe_tth
                .iter()
                .enumerate()
                .map(|(i, &t)| (i, (t - tth_target).abs()))
                .filter(|&(_, r)| r < 1.5e-3)
                .collect();
            hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let stride = (hits.len() / n_per_ring.max(1)).max(1);
            for k in (0..hits.len()).step_by(stride).take(n_per_ring) {
                let p = hits[k].0;
                peaks.push((idx_d1[p], idx_d2[p]));
            }
        }
        peaks
    }

    // No standalone "perpendicular" grid-only test: grid scoring alone is
    // intentionally ambiguous on small synthetic patterns (a wrong-D + tilt
    // combination can coincidentally distribute peaks across more rings than
    // truth). The disambiguation is done in `auto_calibrate_grid`, which
    // L-M refines top-K candidates and picks the lowest-RMS result. The
    // end-to-end behavior is covered by `auto_calibrate_recovers_synthetic_geometry`.

    #[test]
    fn grid_search_recovers_tilted_geometry() {
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.20;
        poni.poni1 = 512.5 * 1e-4;
        poni.poni2 = 512.5 * 1e-4;
        poni.rot1 = 25.0_f64.to_radians();
        poni.rot2 = 0.0;
        poni.rot3 = 0.0;
        poni.wavelength = 1.0e-10;
        let geom = Geometry::new(1024, 1024, poni.clone());
        let cal = synthetic_calibrant();
        let peaks = make_peaks_for(&geom, 32, 6);
        // Tilted geometry — many peaks fall off the detector
        assert!(peaks.len() > 30, "need enough peaks, got {}", peaks.len());

        let opts = GridSearchOptions {
            distance_min: 0.05,
            distance_max: 0.6,
            distance_steps: 20,
            rot_max: 45.0_f64.to_radians(),
            rot_steps: 11,
            poni_search_radius_px: 100.0,
            poni_grid_steps: 3,
            tolerance_rad: 8.0e-3,
            max_peaks_for_grid: 400,
            max_ring_index: 10,
            top_k: 8,
        };
        let result = grid_search(
            &peaks,
            512.0,
            512.0,
            &cal,
            poni.wavelength,
            poni.pixel1,
            poni.pixel2,
            &opts,
        )
        .expect("grid search should succeed on tilted geometry");

        // With 9° rotation step, expect rot1 within ±10° of truth
        let dr1 = (result.poni.rot1 - poni.rot1).abs();
        assert!(
            dr1 < 12.0_f64.to_radians(),
            "rot1 off by {:.2}°",
            dr1.to_degrees()
        );
        // Distance within ~25%
        let d_rel = (result.poni.distance - poni.distance).abs() / poni.distance;
        assert!(d_rel < 0.3, "distance off by {:.2}%", d_rel * 100.0);
    }
}
