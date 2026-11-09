// Most of the inner loops here are small dense linear-algebra / Jacobian
// kernels — indexing is the natural way to express them.
#![allow(clippy::needless_range_loop)]

//! Levenberg–Marquardt geometry refinement with optional Huber IRLS.
//!
//! Parameter vector (7 entries, any subset can be fixed):
//! `[distance, poni1, poni2, rot1, rot2, rot3, wavelength]`.
//!
//! Residual per control point i:
//!     r_i = tth_pred(d1_i, d2_i; distance, poni1, poni2, rot1..rot3, pixel*) − tth_target(ring_i; wavelength)
//!
//! Forward model is inlined here (rather than calling Geometry::calc_tth) to
//! avoid the per-call Vec allocations and to keep parameter dependence
//! explicit for finite differencing.

use crate::geometry::Geometry;
use crate::poni::Poni;

use super::calibrant::Calibrant;

// ---------- public types ----------

#[derive(Debug, Clone)]
pub struct CalibrationPoints {
    /// Row coordinate (slow axis), in pixel units, sub-pixel allowed.
    pub d1: Vec<f64>,
    /// Column coordinate (fast axis).
    pub d2: Vec<f64>,
    /// `Some(k)` = point belongs to the k-th ring of the calibrant.
    /// `None` = unassigned; valid only if `CalibrationOptions::auto_assign_rings`.
    pub ring: Vec<Option<usize>>,
    /// Per-point weight ≥ 0. Treated as 1.0 if `weight.is_empty()`.
    pub weight: Vec<f64>,
}

impl CalibrationPoints {
    pub fn new() -> Self {
        Self {
            d1: Vec::new(),
            d2: Vec::new(),
            ring: Vec::new(),
            weight: Vec::new(),
        }
    }

    pub fn push(&mut self, d1: f64, d2: f64, ring: Option<usize>) {
        self.d1.push(d1);
        self.d2.push(d2);
        self.ring.push(ring);
    }

    pub fn len(&self) -> usize {
        self.d1.len()
    }

    pub fn is_empty(&self) -> bool {
        self.d1.is_empty()
    }

    #[allow(dead_code)] // kept as a helper for future per-iteration weight code
    fn point_weight(&self, i: usize) -> f64 {
        if self.weight.is_empty() {
            1.0
        } else {
            self.weight[i]
        }
    }
}

impl Default for CalibrationPoints {
    fn default() -> Self {
        Self::new()
    }
}

// Default for ParamMask is `fix_wavelength` — that's a meaningful default, not
// just `all_false`, so we can't `#[derive(Default)]`.

/// Which PONI parameters are free during refinement.
#[derive(Debug, Clone, Copy)]
pub struct ParamMask {
    pub distance: bool,
    pub poni1: bool,
    pub poni2: bool,
    pub rot1: bool,
    pub rot2: bool,
    pub rot3: bool,
    pub wavelength: bool,
}

impl ParamMask {
    /// All seven parameters free.
    pub fn all_free() -> Self {
        Self {
            distance: true,
            poni1: true,
            poni2: true,
            rot1: true,
            rot2: true,
            rot3: true,
            wavelength: true,
        }
    }

    /// Common default: wavelength is known and fixed, everything else free.
    pub fn fix_wavelength() -> Self {
        let mut m = Self::all_free();
        m.wavelength = false;
        m
    }

    #[allow(dead_code)] // useful for diagnostics / future ParamMask API
    fn count_free(&self) -> usize {
        [
            self.distance,
            self.poni1,
            self.poni2,
            self.rot1,
            self.rot2,
            self.rot3,
            self.wavelength,
        ]
        .iter()
        .filter(|b| **b)
        .count()
    }
}

impl Default for ParamMask {
    fn default() -> Self {
        Self::fix_wavelength()
    }
}

/// Loss function applied to per-point 2θ residuals.
#[derive(Debug, Clone, Copy, Default)]
pub enum Loss {
    /// Plain least squares.
    #[default]
    L2,
    /// Huber loss with threshold `c` (in radians of 2θ). Residuals with
    /// `|r| ≤ c` keep weight 1; larger residuals are down-weighted as `c/|r|`.
    /// Typical c: 3× the expected per-point 2θ error (e.g. 1–3 mrad).
    Huber { c: f64 },
}

#[derive(Debug, Clone)]
pub struct CalibrationOptions {
    pub mask: ParamMask,
    pub loss: Loss,
    /// If true, ring assignment for points with `ring == None` is recomputed
    /// at each L-M iteration as the index of the nearest predicted-2θ ring.
    pub auto_assign_rings: bool,
    /// Maximum 2θ difference (rad) for auto-assignment to accept a ring.
    /// Points farther than this from every ring become inliers=false.
    pub assign_tolerance_rad: f64,
    pub max_iter: usize,
    /// Convergence: stop when relative change in cost is below this.
    pub tol: f64,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        Self {
            mask: ParamMask::default(),
            loss: Loss::default(),
            auto_assign_rings: false,
            assign_tolerance_rad: 0.05, // ~3°, generous
            max_iter: 100,
            tol: 1e-9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub poni: Poni,
    pub n_iter: usize,
    /// Final un-weighted Σ r²  (rad²).
    pub chi2: f64,
    /// Final RMS residual in 2θ (rad).
    pub rms_rad: f64,
    /// Ring assigned to each control point in the final iteration.
    pub assignments: Vec<Option<usize>>,
    /// Whether each control point counted as an inlier (always true for L2,
    /// reflects Huber weight > 0 for robust losses).
    pub inliers: Vec<bool>,
}

#[derive(Debug)]
pub enum CalibrationError {
    NoPoints,
    /// Fewer non-zero-weight points than free parameters → fit underdetermined.
    Underdetermined {
        n_points: usize,
        n_free: usize,
    },
    /// Normal equations were singular (degenerate parameter setup).
    Singular,
    /// L-M did not reduce cost after `max_iter` iterations.
    NoConvergence,
    /// A point had `ring = None` but `auto_assign_rings = false`.
    UnassignedRing {
        index: usize,
    },
    /// A point's ring index is out of bounds for the calibrant.
    RingOutOfRange {
        index: usize,
        ring: usize,
        n_rings: usize,
    },
    /// Wavelength is fixed but missing/non-positive in the starting PONI.
    BadWavelength,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::NoPoints => write!(f, "no control points provided"),
            CalibrationError::Underdetermined { n_points, n_free } => write!(
                f,
                "underdetermined: {} effective points < {} free parameters",
                n_points, n_free
            ),
            CalibrationError::Singular => write!(f, "normal-equations matrix is singular"),
            CalibrationError::NoConvergence => {
                write!(f, "L-M did not converge within the iteration budget")
            }
            CalibrationError::UnassignedRing { index } => write!(
                f,
                "point {} has no ring assigned but auto_assign_rings is off",
                index
            ),
            CalibrationError::RingOutOfRange {
                index,
                ring,
                n_rings,
            } => write!(
                f,
                "point {} references ring {} but calibrant has only {} rings",
                index, ring, n_rings
            ),
            CalibrationError::BadWavelength => {
                write!(f, "wavelength must be positive and finite")
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

// ---------- forward model ----------

/// Precomputed sin/cos for the three PONI rotations.
#[derive(Clone, Copy)]
struct TrigConsts {
    sr1: f64,
    cr1: f64,
    sr2: f64,
    cr2: f64,
    sr3: f64,
    cr3: f64,
}

impl TrigConsts {
    fn new(rot1: f64, rot2: f64, rot3: f64) -> Self {
        let (sr1, cr1) = rot1.sin_cos();
        let (sr2, cr2) = rot2.sin_cos();
        let (sr3, cr3) = rot3.sin_cos();
        Self {
            sr1,
            cr1,
            sr2,
            cr2,
            sr3,
            cr3,
        }
    }
}

/// 2θ (rad) at a single detector pixel given full PONI parameters.
/// Uses pyFAI/Dioptas pixel-center convention: integer index i refers to
/// the lower-left corner of pixel i, so the pixel center is at i + 0.5.
#[inline]
#[allow(clippy::too_many_arguments)]
fn tth_at(
    d1: f64,
    d2: f64,
    distance: f64,
    poni1: f64,
    poni2: f64,
    geometry: &Geometry,
    tc: &TrigConsts,
) -> f64 {
    let p = geometry.detector_position(d1 + 0.5, d2 + 0.5);
    let p1 = p[1] - poni1;
    let p2 = p[2] - poni2;
    let p3 = distance + p[0];

    let x = p1 * tc.cr2 * tc.cr3 + p2 * (tc.cr3 * tc.sr1 * tc.sr2 - tc.cr1 * tc.sr3)
        - p3 * (tc.cr1 * tc.cr3 * tc.sr2 + tc.sr1 * tc.sr3);
    let y = p1 * tc.cr2 * tc.sr3 + p2 * (tc.cr1 * tc.cr3 + tc.sr1 * tc.sr2 * tc.sr3)
        - p3 * (-(tc.cr3 * tc.sr1) + tc.cr1 * tc.sr2 * tc.sr3);
    let z = p1 * tc.sr2 - p2 * tc.cr2 * tc.sr1 + p3 * tc.cr1 * tc.cr2;

    (x * x + y * y).sqrt().atan2(z)
}

// ---------- parameter packing ----------

/// Internal compact parameter vector (only the free entries).
struct ParamPack {
    /// Indices into the seven-vector for each free slot.
    free_indices: Vec<usize>,
    /// The full 7-vector of parameter values (free + fixed).
    full: [f64; 7],
}

const P_DIST: usize = 0;
const P_PONI1: usize = 1;
const P_PONI2: usize = 2;
const P_ROT1: usize = 3;
const P_ROT2: usize = 4;
const P_ROT3: usize = 5;
const P_WAVELENGTH: usize = 6;

impl ParamPack {
    fn new(poni: &Poni, mask: ParamMask) -> Self {
        let full = [
            poni.distance,
            poni.poni1,
            poni.poni2,
            poni.rot1,
            poni.rot2,
            poni.rot3,
            poni.wavelength,
        ];
        let flags = [
            mask.distance,
            mask.poni1,
            mask.poni2,
            mask.rot1,
            mask.rot2,
            mask.rot3,
            mask.wavelength,
        ];
        let free_indices: Vec<usize> = (0..7).filter(|i| flags[*i]).collect();
        Self { free_indices, full }
    }

    fn n_free(&self) -> usize {
        self.free_indices.len()
    }

    fn read_free(&self) -> Vec<f64> {
        self.free_indices.iter().map(|&i| self.full[i]).collect()
    }

    fn write_free(&mut self, x: &[f64]) {
        for (k, &i) in self.free_indices.iter().enumerate() {
            self.full[i] = x[k];
        }
    }

    fn as_poni(&self, base: &Poni) -> Poni {
        let mut p = base.clone();
        p.distance = self.full[P_DIST];
        p.poni1 = self.full[P_PONI1];
        p.poni2 = self.full[P_PONI2];
        p.rot1 = self.full[P_ROT1];
        p.rot2 = self.full[P_ROT2];
        p.rot3 = self.full[P_ROT3];
        p.wavelength = self.full[P_WAVELENGTH];
        p
    }
}

// ---------- residuals + Jacobian ----------

/// Compute the 2θ predicted by the model at every control point.
fn predicted_2theta(points: &CalibrationPoints, pack: &ParamPack, geometry: &Geometry) -> Vec<f64> {
    let tc = TrigConsts::new(pack.full[P_ROT1], pack.full[P_ROT2], pack.full[P_ROT3]);
    let distance = pack.full[P_DIST];
    let poni1 = pack.full[P_PONI1];
    let poni2 = pack.full[P_PONI2];
    points
        .d1
        .iter()
        .zip(&points.d2)
        .map(|(&d1, &d2)| tth_at(d1, d2, distance, poni1, poni2, geometry, &tc))
        .collect()
}

/// Target 2θ for each control point, given its assigned ring and current
/// wavelength. Returns `None` for points without an assignment or with an
/// out-of-range ring (the caller decides whether to skip or error).
fn target_2theta(
    assignments: &[Option<usize>],
    calibrant: &Calibrant,
    wavelength_m: f64,
) -> Vec<Option<f64>> {
    let ring_tth = calibrant.two_theta_rad(wavelength_m);
    assignments
        .iter()
        .map(|opt| match opt {
            Some(k) => ring_tth.get(*k).copied().flatten(),
            None => None,
        })
        .collect()
}

/// Assign each unassigned point to the calibrant ring whose predicted 2θ is
/// closest to the observed (predicted) one, subject to `tolerance`.
fn auto_assign(
    points: &CalibrationPoints,
    pack: &ParamPack,
    calibrant: &Calibrant,
    geometry: &Geometry,
    tolerance: f64,
) -> Vec<Option<usize>> {
    let pred = predicted_2theta(points, pack, geometry);
    let rings = calibrant.two_theta_rad(pack.full[P_WAVELENGTH]);
    let mut out = points.ring.clone();
    for (i, slot) in out.iter_mut().enumerate() {
        if slot.is_some() {
            continue;
        }
        let p = pred[i];
        // Nearest finite ring.
        let mut best: Option<(usize, f64)> = None;
        for (k, r) in rings.iter().enumerate() {
            if let Some(rv) = r {
                let d = (rv - p).abs();
                if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                    best = Some((k, d));
                }
            }
        }
        if let Some((k, d)) = best
            && d <= tolerance
        {
            *slot = Some(k);
        }
    }
    out
}

/// Build residual vector r_i = pred_i − target_i (signed, in radians of 2θ).
/// Points with no target (unassigned ring) get r_i = 0 and `mask_out[i] = false`.
fn residuals(
    points: &CalibrationPoints,
    pack: &ParamPack,
    assignments: &[Option<usize>],
    calibrant: &Calibrant,
    geometry: &Geometry,
) -> (Vec<f64>, Vec<bool>) {
    let pred = predicted_2theta(points, pack, geometry);
    let tgt = target_2theta(assignments, calibrant, pack.full[P_WAVELENGTH]);
    let mut r = vec![0.0; pred.len()];
    let mut active = vec![false; pred.len()];
    for i in 0..pred.len() {
        if let Some(t) = tgt[i] {
            r[i] = pred[i] - t;
            active[i] = true;
        }
    }
    (r, active)
}

/// Finite-difference Jacobian: J[i, k] = ∂r_i / ∂x_k, where x_k is the k-th
/// free parameter.
///
/// Each parameter is perturbed in turn by a relative or absolute step (whichever
/// is larger), and the whole residual vector is re-evaluated. For up to ~8
/// free parameters and a few hundred control points this is essentially free.
fn finite_diff_jacobian(
    points: &CalibrationPoints,
    pack: &ParamPack,
    assignments: &[Option<usize>],
    calibrant: &Calibrant,
    geometry: &Geometry,
    base_residuals: &[f64],
) -> Vec<Vec<f64>> {
    let n_pts = points.len();
    let n_free = pack.n_free();
    let mut jac = vec![vec![0.0; n_free]; n_pts];
    let mut probe = ParamPack {
        free_indices: pack.free_indices.clone(),
        full: pack.full,
    };
    // Per-parameter typical scale for step-size selection. The step is
    // `eps * scale`; choosing scale ~ O(1) of the parameter's natural range
    // keeps FD on the right side of cancellation noise.
    let scales = [
        pack.full[P_DIST].abs().max(1e-2),  // distance ~ 0.1-1 m
        pack.full[P_PONI1].abs().max(1e-2), // poni ~ 0.05-0.2 m
        pack.full[P_PONI2].abs().max(1e-2),
        1.0, // rotations: ~1 rad typical range
        1.0,
        1.0,
        pack.full[P_WAVELENGTH].abs().max(1e-11), // wavelength ~ 0.1-1 Å
    ];
    let eps = (f64::EPSILON).sqrt(); // ~1.5e-8
    for (k, &p_idx) in pack.free_indices.iter().enumerate() {
        let h = eps * scales[p_idx];
        let saved = probe.full[p_idx];
        probe.full[p_idx] = saved + h;
        let (r_plus, _) = residuals(points, &probe, assignments, calibrant, geometry);
        probe.full[p_idx] = saved;
        for i in 0..n_pts {
            jac[i][k] = (r_plus[i] - base_residuals[i]) / h;
        }
    }
    jac
}

// ---------- loss / weighting ----------

fn loss_weights(residuals_in: &[f64], active: &[bool], loss: Loss) -> Vec<f64> {
    let mut w = vec![0.0; residuals_in.len()];
    for i in 0..residuals_in.len() {
        if !active[i] {
            continue;
        }
        w[i] = match loss {
            Loss::L2 => 1.0,
            Loss::Huber { c } => {
                let ar = residuals_in[i].abs();
                if ar <= c {
                    1.0
                } else if ar > 0.0 {
                    c / ar
                } else {
                    1.0
                }
            }
        };
    }
    w
}

// ---------- linear algebra: Gauss elimination with partial pivoting ----------

/// Solve A x = b for small dense systems (≤ 8 unknowns in our case).
/// Returns `None` if the matrix is singular at our tolerance.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        // Partial pivot.
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for r in (i + 1)..n {
            let v = a[r][i].abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-300 {
            return None;
        }
        a.swap(i, max_row);
        b.swap(i, max_row);
        // Eliminate below.
        for r in (i + 1)..n {
            let factor = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    // Back-substitute.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

// ---------- L-M outer loop ----------

/// Compute J^T W J and J^T W r for a weighted residual problem.
fn normal_equations(
    jac: &[Vec<f64>],
    residuals_in: &[f64],
    weights: &[f64],
    n_free: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut a = vec![vec![0.0; n_free]; n_free];
    let mut g = vec![0.0; n_free];
    for i in 0..jac.len() {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let row = &jac[i];
        for k in 0..n_free {
            g[k] += w * row[k] * residuals_in[i];
            for l in k..n_free {
                a[k][l] += w * row[k] * row[l];
            }
        }
    }
    // Symmetrize.
    for k in 0..n_free {
        for l in (k + 1)..n_free {
            a[l][k] = a[k][l];
        }
    }
    (a, g)
}

fn weighted_cost(residuals_in: &[f64], weights: &[f64]) -> f64 {
    residuals_in
        .iter()
        .zip(weights)
        .map(|(r, w)| w * r * r)
        .sum()
}

fn unweighted_chi2(residuals_in: &[f64], active: &[bool]) -> f64 {
    residuals_in
        .iter()
        .zip(active)
        .filter_map(|(r, a)| if *a { Some(r * r) } else { None })
        .sum()
}

pub fn refine(
    geometry: &Geometry,
    points: &CalibrationPoints,
    calibrant: &Calibrant,
    options: &CalibrationOptions,
) -> Result<CalibrationResult, CalibrationError> {
    if points.is_empty() {
        return Err(CalibrationError::NoPoints);
    }
    let wavelength = geometry.poni.wavelength;
    if !(wavelength.is_finite() && wavelength > 0.0) {
        return Err(CalibrationError::BadWavelength);
    }
    // Validate ring indices that were supplied explicitly.
    let n_rings = calibrant.n_rings();
    for (i, r) in points.ring.iter().enumerate() {
        if let Some(k) = r {
            if *k >= n_rings {
                return Err(CalibrationError::RingOutOfRange {
                    index: i,
                    ring: *k,
                    n_rings,
                });
            }
        } else if !options.auto_assign_rings {
            return Err(CalibrationError::UnassignedRing { index: i });
        }
    }

    let mask = options.mask;
    let mut pack = ParamPack::new(&geometry.poni, mask);
    let n_free = pack.n_free();

    // Initial assignments + residuals.
    let mut assignments = if options.auto_assign_rings {
        auto_assign(
            points,
            &pack,
            calibrant,
            geometry,
            options.assign_tolerance_rad,
        )
    } else {
        points.ring.clone()
    };

    let (mut r, mut active) = residuals(points, &pack, &assignments, calibrant, geometry);
    let mut w = loss_weights(&r, &active, options.loss);
    // Multiply in per-point user weights (if any).
    if !points.weight.is_empty() {
        for i in 0..w.len() {
            w[i] *= points.weight[i];
        }
    }

    // Underdetermined check: number of points with positive effective weight.
    let n_eff = w.iter().filter(|x| **x > 0.0).count();
    if n_eff < n_free {
        return Err(CalibrationError::Underdetermined {
            n_points: n_eff,
            n_free,
        });
    }

    let mut cost = weighted_cost(&r, &w);

    // Marquardt damping.
    let mut lambda = 1.0e-3;
    let lambda_up = 10.0;
    let lambda_down = 0.5;
    let lambda_max = 1.0e12;
    let lambda_min = 1.0e-12;

    let mut last_iter = 0;
    for iter in 0..options.max_iter {
        last_iter = iter;
        let jac = finite_diff_jacobian(points, &pack, &assignments, calibrant, geometry, &r);
        let (mut a, g) = normal_equations(&jac, &r, &w, n_free);
        // Apply Marquardt damping to the diagonal.
        for k in 0..n_free {
            a[k][k] *= 1.0 + lambda;
        }
        // Solve (J^T W J + λ diag) δp = − J^T W r
        let neg_g: Vec<f64> = g.iter().map(|x| -x).collect();
        let step = match solve_linear(a, neg_g) {
            Some(s) => s,
            None => return Err(CalibrationError::Singular),
        };
        let old = pack.read_free();
        let trial: Vec<f64> = old.iter().zip(&step).map(|(o, s)| o + s).collect();

        // Evaluate trial step.
        let saved_full = pack.full;
        pack.write_free(&trial);
        // Re-assign if requested, then recompute residuals/weights/cost.
        let new_assignments = if options.auto_assign_rings {
            auto_assign(
                points,
                &pack,
                calibrant,
                geometry,
                options.assign_tolerance_rad,
            )
        } else {
            points.ring.clone()
        };
        let (new_r, new_active) = residuals(points, &pack, &new_assignments, calibrant, geometry);
        let mut new_w = loss_weights(&new_r, &new_active, options.loss);
        if !points.weight.is_empty() {
            for i in 0..new_w.len() {
                new_w[i] *= points.weight[i];
            }
        }
        let new_cost = weighted_cost(&new_r, &new_w);

        if new_cost < cost {
            // Accept. Wrap rotations into (-π, π] so they can't drift
            // arbitrarily far along the periodic axis (matches pyFAI bounds).
            let pi = std::f64::consts::PI;
            for idx in [P_ROT1, P_ROT2, P_ROT3] {
                let mut v = pack.full[idx];
                while v > pi {
                    v -= 2.0 * pi;
                }
                while v <= -pi {
                    v += 2.0 * pi;
                }
                pack.full[idx] = v;
            }
            let rel = (cost - new_cost) / cost.max(1.0e-300);
            cost = new_cost;
            r = new_r;
            active = new_active;
            w = new_w;
            assignments = new_assignments;
            lambda = (lambda * lambda_down).max(lambda_min);
            if rel < options.tol {
                break;
            }
        } else {
            // Reject — restore params and increase damping.
            pack.full = saved_full;
            lambda *= lambda_up;
            if lambda > lambda_max {
                // No further progress possible.
                break;
            }
        }
    }

    let poni = pack.as_poni(&geometry.poni);
    let chi2 = unweighted_chi2(&r, &active);
    let n_active = active.iter().filter(|b| **b).count().max(1);
    let rms_rad = (chi2 / n_active as f64).sqrt();
    let inliers = w.iter().map(|x| *x > 0.0).collect();

    Ok(CalibrationResult {
        poni,
        n_iter: last_iter + 1,
        chi2,
        rms_rad,
        assignments,
        inliers,
    })
}
