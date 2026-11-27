//! Post-calibration `rot3` determination from X-ray linear polarization.
//!
//! Debye–Scherrer ring positions (`2θ`) are completely invariant to in-plane
//! detector rotation `rot3` around the incident beam, so geometry refinement
//! from ring control points cannot constrain `rot3`.
//!
//! For a linearly polarized X-ray beam with polarization factor `P ∈ [-1, 1]`
//! (matching pyFAI's convention: `P > 0` for horizontal polarization along the
//! laboratory `x2` axis, e.g. `P ≈ +0.99` at synchrotrons; `P < 0` for vertical
//! polarization along `x1`), the solid-angle-corrected intensity around each
//! powder ring varies as:
//!
//! ```text
//! I_corr(2θ, χ₀) = I(2θ, χ₀) / Ω(χ₀)
//!                ∝ ½ [1 + cos²(2θ) − P · sin²(2θ) · cos(2(χ₀ − rot3))]
//! ```
//!
//! where `χ₀ = atan2(x₁⁽⁰⁾, x₂⁽⁰⁾)` is the sample-frame azimuth evaluated at
//! `rot3 = 0` (matching [`crate::geometry::math::f_polarization`]).
//!
//! Writing `c = β cos(2·rot3)` and `s = β sin(2·rot3)` with free amplitude
//! ratio `β` (expected `≈ 1.0` when the beam matches `polarization_factor`),
//! each ring's normalized azimuthal profile satisfies the linear model:
//!
//! ```text
//! I_corr(k, b) / A_k ≈ 1 − α_k · (c · ⟨cos 2χ₀⟩_{k,b} + s · ⟨sin 2χ₀⟩_{k,b})
//! ```
//!
//! Fitting `(c, s)` freely via Huber IRLS eliminates angle bias when the true
//! polarization factor differs from the nominal value, recovers `rot3 = ½ atan2(s, c)`
//! modulo `π` in `(-π/2, π/2]`, and yields closed-form 2×2 covariance estimates
//! for `uncertainty_rad`, `modulation_snr`, and `fitted_polarization_ratio = hypot(c, s)`.
//!
//! This function works after **any** calibration workflow — manual peak
//! picking + [`super::refine`], [`super::auto_calibrate`],
//! [`super::auto_calibrate_image`], or an existing `.poni` file.

use std::f64::consts::PI;
use std::fmt;

use rayon::prelude::*;

use crate::geometry::math::{
    TrigonometricConstants, f_azimuth, f_solid_angle, f_tth, f_x1, f_x2, f_x3,
};
use crate::poni::Poni;

use super::calibrant::Calibrant;

/// Configuration for [`calibrate_rot3_polarization`].
#[derive(Debug, Clone)]
pub struct PolarizationRot3Options {
    /// Number of azimuthal bins around `[-π, π]` per ring (default `72`, 5°/bin).
    pub n_az_bins: usize,
    /// Minimum scattering angle `2θ` in radians for rings included in the fit
    /// (default `5°` — lower angles have negligible `sin²(2θ)` modulation).
    pub min_tth_rad: f64,
    /// Maximum number of calibrant rings (above `min_tth_rad`) to inspect
    /// (default `32`).
    pub max_rings: usize,
    /// Optional fixed half-width in `2θ` (radians) for integrating each ring's
    /// peak band. `None` (default) auto-selects from the local pixel angular
    /// size (`max(3.5 · Δ2θ_px, 0.002 rad)`), clamped to avoid neighboring rings.
    pub ring_half_width_rad: Option<f64>,
    /// Minimum populated unmasked azimuthal bins required for a ring to
    /// contribute (default `12`).
    pub min_samples_per_ring: usize,
    /// Minimum total `(ring, azimuth)` samples across all rings required to
    /// accept a fit (default `24`).
    pub min_total_samples: usize,
}

impl Default for PolarizationRot3Options {
    fn default() -> Self {
        Self {
            n_az_bins: 72,
            min_tth_rad: 5.0_f64.to_radians(),
            max_rings: 32,
            ring_half_width_rad: None,
            min_samples_per_ring: 12,
            min_total_samples: 24,
        }
    }
}

/// Output of [`calibrate_rot3_polarization`].
#[derive(Debug, Clone)]
pub struct PolarizationRot3Result {
    /// Calibrated in-plane detector rotation `rot3` in radians, wrapped into
    /// `(-π/2, π/2]` (linear polarization is π-periodic).
    pub rot3_rad: f64,
    /// Copy of the input [`Poni`] with `rot3` updated to `rot3_rad`.
    pub poni: Poni,
    /// Estimated 1σ uncertainty on `rot3_rad` (radians), derived from the
    /// closed-form 2×2 covariance of `(c, s)` via the delta method.
    pub uncertainty_rad: f64,
    /// Number of calibrant rings that contributed to the fit.
    pub n_rings_used: usize,
    /// Total number of `(ring, azimuth)` intensity samples used in the fit.
    pub n_samples: usize,
    /// Wald detection significance (`√([c, s] · Σ⁻¹ · [c, s])`) of the
    /// `cos(2χ)` polarization modulation relative to an unpolarized model.
    pub modulation_snr: f64,
    /// Ratio `hypot(c, s)` of the observed `cos(2χ)` modulation amplitude to
    /// the theoretical amplitude predicted by `polarization_factor` (near `1.0`
    /// when the image matches the chosen polarization factor).
    pub fitted_polarization_ratio: f64,
    /// Whether the fit passes quality gates (`modulation_snr >= 5.0`,
    /// `uncertainty_rad <= 3.5°`, and `fitted_polarization_ratio ∈ [0.5, 1.5]`)
    /// and is safe to apply automatically to a calibrated `Poni`.
    pub reliable: bool,
}

/// Failure modes for [`calibrate_rot3_polarization`].
#[derive(Debug, Clone)]
pub enum PolarizationRot3Error {
    InvalidInput(String),
    InsufficientSignal(String),
}

impl fmt::Display for PolarizationRot3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input for rot3 calibration: {msg}"),
            Self::InsufficientSignal(msg) => {
                write!(
                    f,
                    "insufficient polarization signal for rot3 calibration: {msg}"
                )
            }
        }
    }
}

impl std::error::Error for PolarizationRot3Error {}

#[derive(Debug, Clone, Copy)]
struct ActiveRing {
    tth: f64,
    alpha: f64,
    half_w: f64,
    bg_lo: f64,
    bg_hi: f64,
}

#[derive(Debug, Clone)]
struct RingAzSample {
    cos2: f64, // exact pixel-averaged ⟨cos(2χ₀)⟩ inside this bin
    sin2: f64, // exact pixel-averaged ⟨sin(2χ₀)⟩ inside this bin
    y: f64,    // median-normalized solid-angle-corrected excess intensity
}

#[derive(Debug, Clone)]
struct RingAzData {
    alpha: f64,
    sigma_rel: f64,
    samples: Vec<RingAzSample>,
}

#[derive(Default, Clone, Copy)]
struct BinAcc {
    ring_sum: f64,
    sa_sum: f64,
    cos2_sum: f64,
    sin2_sum: f64,
    ring_cnt: u32,
    masked_ring_cnt: u32,
    bg_sum: f64,
    bg_cnt: u32,
}

impl BinAcc {
    #[inline]
    fn merge(&mut self, other: &Self) {
        self.ring_sum += other.ring_sum;
        self.sa_sum += other.sa_sum;
        self.cos2_sum += other.cos2_sum;
        self.sin2_sum += other.sin2_sum;
        self.ring_cnt += other.ring_cnt;
        self.masked_ring_cnt += other.masked_ring_cnt;
        self.bg_sum += other.bg_sum;
        self.bg_cnt += other.bg_cnt;
    }
}

/// Wrap an angle in radians into `(-π/2, π/2]` (matching the π-periodicity of
/// linear polarization `cos(2(χ - rot3))`).
fn wrap_half_pi(angle: f64) -> f64 {
    let mut a = (angle + 0.5 * PI).rem_euclid(PI) - 0.5 * PI;
    if a <= -0.5 * PI + 1e-12 {
        a = 0.5 * PI;
    }
    a
}

fn median_of(vals: &mut [f64]) -> f64 {
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    vals[mid]
}

/// Calibrate `rot3` from the azimuthal intensity modulation of powder rings
/// under a known linear X-ray polarization factor `polarization_factor`.
///
/// Works on any calibrated [`Poni`] (from manual peak picking + `refine`,
/// `auto_calibrate`, `auto_calibrate_image`, or a loaded `.poni` file).
#[allow(clippy::too_many_arguments)]
pub fn calibrate_rot3_polarization(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    calibrant: &Calibrant,
    polarization_factor: f64,
    options: &PolarizationRot3Options,
) -> Result<PolarizationRot3Result, PolarizationRot3Error> {
    let expected_len = dim1
        .checked_mul(dim2)
        .ok_or_else(|| PolarizationRot3Error::InvalidInput("image dimensions overflow".into()))?;
    if image.len() != expected_len || dim1 == 0 || dim2 == 0 {
        return Err(PolarizationRot3Error::InvalidInput(format!(
            "image length {} does not match {}x{}",
            image.len(),
            dim1,
            dim2
        )));
    }
    if let Some(m) = mask
        && m.len() != expected_len
    {
        return Err(PolarizationRot3Error::InvalidInput(format!(
            "mask length {} does not match image length {}",
            m.len(),
            expected_len
        )));
    }
    if !polarization_factor.is_finite()
        || polarization_factor.abs() < 0.05
        || polarization_factor.abs() > 1.0
    {
        return Err(PolarizationRot3Error::InvalidInput(format!(
            "polarization_factor must be in [-1, -0.05] or [0.05, 1], got {polarization_factor}"
        )));
    }
    if !poni.distance.is_finite()
        || poni.distance <= 0.0
        || !poni.wavelength.is_finite()
        || poni.wavelength <= 0.0
        || !poni.pixel1.is_finite()
        || poni.pixel1 <= 0.0
        || !poni.pixel2.is_finite()
        || poni.pixel2 <= 0.0
    {
        return Err(PolarizationRot3Error::InvalidInput(
            "poni has non-finite or non-positive distance, wavelength, or pixel size".into(),
        ));
    }

    let n_az = options.n_az_bins.clamp(16, 360);

    // 2D-strided sample for positive-baseline detection (skip negative sentinel
    // gap pixels like -1/-2 on photon-counting detectors without truncating
    // dark-subtracted images whose noise fluctuates around a small pedestal).
    let row_step = (dim1 / 128).max(1);
    let col_step = (dim2 / 128).max(1);
    let mut bg_sample = Vec::with_capacity(16_384);
    let mut r = 0;
    while r < dim1 {
        let row_off = r * dim2;
        let mut c = 0;
        while c < dim2 {
            let idx = row_off + c;
            let v = image[idx];
            let masked = mask.is_some_and(|m| m[idx]);
            if !masked && v.is_finite() {
                bg_sample.push(v);
            }
            c += col_step;
        }
        r += row_step;
    }
    if bg_sample.is_empty() {
        return Err(PolarizationRot3Error::InsufficientSignal(
            "no valid finite pixels in image".into(),
        ));
    }
    let img_med = median_of(&mut bg_sample);
    let mut img_devs: Vec<f64> = bg_sample.iter().map(|&v| (v - img_med).abs()).collect();
    let img_sigma = 1.4826 * median_of(&mut img_devs);
    let mask_negative = img_med > 5.0 && (img_med - 4.0 * img_sigma) > 0.0;

    // Collect calibrant ring 2θ values and compute non-overlapping integration
    // and flanking background bands for each ring above `min_tth_rad`.
    let all_tths: Vec<f64> = calibrant
        .two_theta_rad(poni.wavelength)
        .into_iter()
        .flatten()
        .filter(|t| t.is_finite() && *t > 1e-4 && *t < PI * 0.49)
        .collect();
    if all_tths.is_empty() {
        return Err(PolarizationRot3Error::InsufficientSignal(
            "calibrant has no valid rings at this wavelength".into(),
        ));
    }

    let px_max = poni.pixel1.max(poni.pixel2);
    let mut active_rings: Vec<ActiveRing> = Vec::new();
    for &tth in all_tths
        .iter()
        .filter(|&&t| t >= options.min_tth_rad)
        .take(options.max_rings)
    {
        let spacing = all_tths
            .iter()
            .filter_map(|&other| {
                let d = (other - tth).abs();
                (d > 1e-6).then_some(d)
            })
            .fold(f64::INFINITY, f64::min);

        let cos_t = tth.cos();
        let sin_t = tth.sin();
        let cos2 = cos_t * cos_t;
        let sin2 = sin_t * sin_t;
        let d_tth_px = (px_max / poni.distance) * cos2;

        let base_hw = options
            .ring_half_width_rad
            .unwrap_or_else(|| (3.5 * d_tth_px).max(0.002));
        let half_w = base_hw.min(0.32 * spacing);
        let bg_lo = 1.3 * half_w;
        let bg_hi = (2.5 * half_w).min(0.48 * spacing);
        if half_w < 0.5 * d_tth_px || bg_hi <= bg_lo + 0.5 * d_tth_px {
            continue;
        }
        let alpha = polarization_factor * sin2 / (1.0 + cos2);
        if alpha.abs() < 1e-4 {
            continue;
        }
        active_rings.push(ActiveRing {
            tth,
            alpha,
            half_w,
            bg_lo,
            bg_hi,
        });
    }
    active_rings.sort_by(|a, b| a.tth.total_cmp(&b.tth));
    if active_rings.is_empty() {
        return Err(PolarizationRot3Error::InsufficientSignal(
            "no isolated calibrant rings above min_tth_rad".into(),
        ));
    }

    let n_rings = active_rings.len();
    let total_bins = n_rings * n_az;
    let tth_min = active_rings.first().unwrap().tth - active_rings.first().unwrap().bg_hi;
    let tth_max = active_rings.last().unwrap().tth + active_rings.last().unwrap().bg_hi;

    // Evaluate geometry with rot3 = 0 so azimuth is χ₀ = χ + rot3.
    let mut poni0 = poni.clone();
    poni0.rot3 = 0.0;
    let tc = TrigonometricConstants::new(&poni0);
    let p3 = poni0.distance;
    let inv_bin_w = (n_az as f64) / (2.0 * PI);

    let chunk_rows = (dim1 / rayon::current_num_threads().max(1)).clamp(8, 128);
    let accs: Vec<BinAcc> = (0..dim1)
        .into_par_iter()
        .chunks(chunk_rows)
        .map(|rows| {
            let mut local = vec![BinAcc::default(); total_bins];
            for r in rows {
                let p1 = (r as f64 + 0.5) * poni0.pixel1 - poni0.poni1;
                let row_off = r * dim2;
                for c in 0..dim2 {
                    let idx = row_off + c;
                    let v = image[idx];
                    let is_bad = mask.is_some_and(|m| m[idx])
                        || !v.is_finite()
                        || (mask_negative && v < 0.0);

                    let p2 = (c as f64 + 0.5) * poni0.pixel2 - poni0.poni2;
                    let x1 = f_x1(&p1, &p2, &p3, &tc);
                    let x2 = f_x2(&p1, &p2, &p3, &tc);
                    let x3 = f_x3(&p1, &p2, &p3, &tc);
                    let tth = f_tth(&x1, &x2, &x3);
                    if tth < tth_min || tth > tth_max {
                        continue;
                    }
                    let pos = active_rings.partition_point(|ring| ring.tth < tth);
                    let lo = pos.saturating_sub(1);
                    let hi = (pos + 1).min(n_rings);
                    let mut matched: Option<(usize, bool)> = None;
                    for (rk, ring) in active_rings.iter().enumerate().take(hi).skip(lo) {
                        let dt = (tth - ring.tth).abs();
                        if dt <= ring.half_w {
                            matched = Some((rk, true));
                            break;
                        } else if dt >= ring.bg_lo && dt <= ring.bg_hi {
                            matched = Some((rk, false));
                        }
                    }
                    if let Some((rk, is_ring)) = matched {
                        let az0 = f_azimuth(&x1, &x2);
                        let b = (((az0 + PI) * inv_bin_w).floor() as isize)
                            .rem_euclid(n_az as isize) as usize;
                        let bin = &mut local[rk * n_az + b];
                        if is_ring {
                            if is_bad {
                                // Track masked/saturated pixels inside the ring
                                // band so partially-masked peak cores are
                                // dropped rather than biasing the bin mean down.
                                bin.masked_ring_cnt += 1;
                            } else {
                                let sa = f_solid_angle(&p1, &p2, &p3);
                                let two_az = 2.0 * az0;
                                bin.ring_sum += v;
                                bin.sa_sum += sa;
                                bin.cos2_sum += two_az.cos();
                                bin.sin2_sum += two_az.sin();
                                bin.ring_cnt += 1;
                            }
                        } else if !is_bad {
                            bin.bg_sum += v;
                            bin.bg_cnt += 1;
                        }
                    }
                }
            }
            local
        })
        .reduce(
            || vec![BinAcc::default(); total_bins],
            |mut a, b| {
                for (dst, src) in a.iter_mut().zip(b.iter()) {
                    dst.merge(src);
                }
                a
            },
        );

    // Build per-ring solid-angle-corrected azimuthal intensity profiles, dropping
    // any (ring, bin) touched by masked/saturated pixels in the ring band.
    let mut ring_data: Vec<RingAzData> = Vec::new();
    for (rk, ring) in active_rings.iter().enumerate() {
        let mut samples: Vec<RingAzSample> = Vec::new();
        for b in 0..n_az {
            let acc = &accs[rk * n_az + b];
            if acc.ring_cnt < 3 || acc.bg_cnt < 3 || acc.masked_ring_cnt > 0 || acc.sa_sum <= 0.0 {
                continue;
            }
            let inv_rc = 1.0 / acc.ring_cnt as f64;
            let mean_ring = acc.ring_sum * inv_rc;
            let mean_bg = acc.bg_sum / acc.bg_cnt as f64;
            let mean_sa = acc.sa_sum * inv_rc;
            let excess = mean_ring - mean_bg;
            samples.push(RingAzSample {
                cos2: acc.cos2_sum * inv_rc,
                sin2: acc.sin2_sum * inv_rc,
                y: excess / mean_sa,
            });
        }
        if samples.len() < options.min_samples_per_ring {
            continue;
        }
        let mut vals: Vec<f64> = samples.iter().map(|s| s.y).collect();
        let med = median_of(&mut vals);
        if !med.is_finite() || med <= 0.0 {
            continue;
        }
        for s in &mut samples {
            s.y /= med;
        }
        let mut devs: Vec<f64> = samples.iter().map(|s| (s.y - 1.0).abs()).collect();
        let mad = 1.4826 * median_of(&mut devs);
        let est_noise = (mad - 0.5 * ring.alpha.abs()).clamp(0.015, 0.50);
        ring_data.push(RingAzData {
            alpha: ring.alpha,
            sigma_rel: est_noise,
            samples,
        });
    }

    let n_rings_used = ring_data.len();
    let n_samples: usize = ring_data.iter().map(|r| r.samples.len()).sum();
    if n_rings_used == 0 || n_samples < options.min_total_samples {
        return Err(PolarizationRot3Error::InsufficientSignal(format!(
            "only {n_samples} valid unmasked ring-azimuth samples across {n_rings_used} rings (need >= {})",
            options.min_total_samples
        )));
    }

    // Joint Huber IRLS for free (c, s) and per-ring scales A_k with exact
    // Schur-complement elimination of per-ring intercepts (critical for partial
    // arcs where ⟨cos 2χ⟩ ≠ 0 and ⟨sin 2χ⟩ ≠ 0):
    //   y_{k,b} / A_k − 1 ≈ δa_k + (−α_k cos2_{k,b}) c + (−α_k sin2_{k,b}) s
    // where c = β cos(2·rot3), s = β sin(2·rot3), β = fitted_polarization_ratio.
    const DELTA_HUBER: f64 = 1.5;
    let mut a_scales = vec![1.0_f64; n_rings_used];
    let mut sigmas: Vec<f64> = ring_data.iter().map(|r| r.sigma_rel).collect();
    let mut c_hat = 0.0_f64;
    let mut s_hat = 0.0_f64;
    let mut resids_buf = Vec::with_capacity(n_az);

    for iter in 0..15 {
        // Solve Schur-complement 2×2 normal equations for (c, s) with δa_k
        // eliminated analytically per ring.
        let mut h_cc = 0.0;
        let mut h_cs = 0.0;
        let mut h_ss = 0.0;
        let mut g_c = 0.0;
        let mut g_s = 0.0;

        for (k, rd) in ring_data.iter().enumerate() {
            let a_k = a_scales[k];
            let sig = sigmas[k];
            let inv_s2 = 1.0 / (sig * sig);

            // Pass 1: weighted ring means for Schur complement.
            let mut w_sum = 0.0;
            let mut bc_sum = 0.0;
            let mut bs_sum = 0.0;
            let mut t_sum = 0.0;
            for sm in &rd.samples {
                let bc = -rd.alpha * sm.cos2;
                let bs = -rd.alpha * sm.sin2;
                let target = sm.y / a_k - 1.0;
                let w = if iter == 0 {
                    inv_s2
                } else {
                    let pred = c_hat * bc + s_hat * bs;
                    let u = ((target - pred) / sig).abs();
                    if u <= DELTA_HUBER {
                        inv_s2
                    } else {
                        inv_s2 * (DELTA_HUBER / u)
                    }
                };
                w_sum += w;
                bc_sum += w * bc;
                bs_sum += w * bs;
                t_sum += w * target;
            }
            if w_sum <= 1e-18 {
                continue;
            }
            let bc_bar = bc_sum / w_sum;
            let bs_bar = bs_sum / w_sum;
            let t_bar = t_sum / w_sum;

            // Pass 2: accumulate centered 2×2 system.
            for sm in &rd.samples {
                let bc = -rd.alpha * sm.cos2;
                let bs = -rd.alpha * sm.sin2;
                let target = sm.y / a_k - 1.0;
                let w = if iter == 0 {
                    inv_s2
                } else {
                    let pred = c_hat * bc + s_hat * bs;
                    let u = ((target - pred) / sig).abs();
                    if u <= DELTA_HUBER {
                        inv_s2
                    } else {
                        inv_s2 * (DELTA_HUBER / u)
                    }
                };
                let dbc = bc - bc_bar;
                let dbs = bs - bs_bar;
                let dt = target - t_bar;
                h_cc += w * dbc * dbc;
                h_cs += w * dbc * dbs;
                h_ss += w * dbs * dbs;
                g_c += w * dbc * dt;
                g_s += w * dbs * dt;
            }
        }

        let det = h_cc * h_ss - h_cs * h_cs;
        if det <= 1e-18 {
            return Err(PolarizationRot3Error::InsufficientSignal(
                "azimuthal coverage is degenerate for cos(2χ)/sin(2χ) separation".into(),
            ));
        }
        let new_c = (h_ss * g_c - h_cs * g_s) / det;
        let new_s = (-h_cs * g_c + h_cc * g_s) / det;
        let step = (new_c - c_hat).hypot(new_s - s_hat);
        c_hat = new_c;
        s_hat = new_s;

        // Update per-ring scales A_k and residual-based per-ring noise σ_k.
        for (k, rd) in ring_data.iter().enumerate() {
            let sig = sigmas[k];
            let mut num = 0.0;
            let mut den = 0.0;
            for sm in &rd.samples {
                let m = (1.0 - rd.alpha * (c_hat * sm.cos2 + s_hat * sm.sin2)).max(0.05);
                let w = if iter == 0 {
                    1.0
                } else {
                    let u = ((sm.y / a_scales[k] - m) / sig).abs();
                    if u <= DELTA_HUBER {
                        1.0
                    } else {
                        DELTA_HUBER / u
                    }
                };
                num += w * sm.y * m;
                den += w * m * m;
            }
            if den > 1e-12 {
                a_scales[k] = (num / den).max(1e-6);
            }
            resids_buf.clear();
            for sm in &rd.samples {
                let m = 1.0 - rd.alpha * (c_hat * sm.cos2 + s_hat * sm.sin2);
                resids_buf.push((sm.y / a_scales[k] - m).abs());
            }
            sigmas[k] = (1.4826 * median_of(&mut resids_buf)).clamp(0.01, 0.50);
        }

        if iter >= 3 && step < 1e-7 {
            break;
        }
    }

    // Compute final Schur-complement Fisher matrix H and Huber-weighted RSS
    // so Σ = s² · H⁻¹ accurately accounts for per-ring scale uncertainty.
    let mut weighted_rss = 0.0;
    let mut h_cc = 0.0;
    let mut h_cs = 0.0;
    let mut h_ss = 0.0;
    for (k, rd) in ring_data.iter().enumerate() {
        let a_k = a_scales[k];
        let sig = sigmas[k];
        let inv_s2 = 1.0 / (sig * sig);
        let mut w_sum = 0.0;
        let mut bc_sum = 0.0;
        let mut bs_sum = 0.0;
        for sm in &rd.samples {
            let bc = -rd.alpha * sm.cos2;
            let bs = -rd.alpha * sm.sin2;
            let resid = (sm.y / a_k - 1.0) - (c_hat * bc + s_hat * bs);
            let u = (resid / sig).abs();
            let w = if u <= DELTA_HUBER {
                inv_s2
            } else {
                inv_s2 * (DELTA_HUBER / u)
            };
            weighted_rss += w * resid * resid;
            w_sum += w;
            bc_sum += w * bc;
            bs_sum += w * bs;
        }
        if w_sum <= 1e-18 {
            continue;
        }
        let bc_bar = bc_sum / w_sum;
        let bs_bar = bs_sum / w_sum;
        for sm in &rd.samples {
            let bc = -rd.alpha * sm.cos2;
            let bs = -rd.alpha * sm.sin2;
            let resid = (sm.y / a_k - 1.0) - (c_hat * bc + s_hat * bs);
            let u = (resid / sig).abs();
            let w = if u <= DELTA_HUBER {
                inv_s2
            } else {
                inv_s2 * (DELTA_HUBER / u)
            };
            let dbc = bc - bc_bar;
            let dbs = bs - bs_bar;
            h_cc += w * dbc * dbc;
            h_cs += w * dbc * dbs;
            h_ss += w * dbs * dbs;
        }
    }

    let det = h_cc * h_ss - h_cs * h_cs;
    if det <= 1e-18 {
        return Err(PolarizationRot3Error::InsufficientSignal(
            "degenerate Fisher information matrix for (c, s)".into(),
        ));
    }

    let dof = n_samples.saturating_sub(n_rings_used + 2).max(1) as f64;
    let s2 = (weighted_rss / dof).max(1e-12);
    // Covariance Σ = s² · H⁻¹
    let cov_cc = s2 * h_ss / det;
    let cov_cs = -s2 * h_cs / det;
    let cov_ss = s2 * h_cc / det;

    let beta = c_hat.hypot(s_hat);
    let rot3_rad = wrap_half_pi(0.5 * s_hat.atan2(c_hat));

    // Delta-method 1σ uncertainty on rot3 = ½ atan2(s, c):
    //   ∇rot3 = (−s, c) / (2 β²)
    let uncertainty_rad = if beta > 1e-6 {
        let var_num =
            s_hat * s_hat * cov_cc - 2.0 * c_hat * s_hat * cov_cs + c_hat * c_hat * cov_ss;
        (var_num.max(0.0).sqrt() / (2.0 * beta * beta)).min(0.5 * PI)
    } else {
        0.5 * PI
    };

    // Wald significance √([c, s] · Σ⁻¹ · [c, s]) = √((c² H_cc + 2cs H_cs + s² H_ss) / s²).
    let quad = (c_hat * c_hat * h_cc + 2.0 * c_hat * s_hat * h_cs + s_hat * s_hat * h_ss).max(0.0);
    let modulation_snr = (quad / s2).sqrt();

    let reliable = modulation_snr >= 5.0
        && uncertainty_rad <= 3.5_f64.to_radians()
        && (0.5..=1.5).contains(&beta);

    let mut out_poni = poni.clone();
    out_poni.rot3 = rot3_rad;

    Ok(PolarizationRot3Result {
        rot3_rad,
        poni: out_poni,
        uncertainty_rad,
        n_rings_used,
        n_samples,
        modulation_snr,
        fitted_polarization_ratio: beta,
        reliable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::simulation::{RingIntensity, SimulationOptions, simulate_diffraction};

    fn make_poni(rot3_deg: f64, rot1_deg: f64, rot2_deg: f64, poni2_px: f64) -> Poni {
        Poni {
            pixel1: 172e-6,
            pixel2: 172e-6,
            distance: 0.12,
            poni1: 256.0 * 172e-6,
            poni2: poni2_px * 172e-6,
            rot1: rot1_deg.to_radians(),
            rot2: rot2_deg.to_radians(),
            rot3: rot3_deg.to_radians(),
            wavelength: 0.4066e-10,
            ..Poni::new()
        }
    }

    #[test]
    fn recovers_known_rot3_across_angles_and_polarizations() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");

        for &(true_rot3_deg, pol_factor, rot1_deg, rot2_deg, poni2_px) in &[
            (0.0, 0.99, 0.0, 0.0, 256.0),
            (22.5, 0.99, 0.0, 0.0, 256.0),
            (-37.0, 0.99, 12.0, -8.0, 256.0),
            (68.0, 0.95, 5.0, 10.0, 140.0),   // off-center + tilted
            (-18.0, -0.95, -6.0, 4.0, 256.0), // vertical polarization
            (89.5, 0.99, 0.0, 0.0, 256.0),    // wrap_half_pi boundary
        ] {
            let truth = make_poni(true_rot3_deg, rot1_deg, rot2_deg, poni2_px);
            let geom = Geometry::new(512, 512, truth.clone());
            let opts_sim = SimulationOptions {
                fwhm_rad: 2.5e-3,
                ring_intensity: RingIntensity::FromCalibrant { fallback: 20_000.0 },
                background: 50.0,
                apply_solid_angle: true,
                polarization_factor: Some(pol_factor),
                poisson_noise: true,
                readout_noise: 2.0,
                rng_seed: 1234,
                ..Default::default()
            };
            let frame = simulate_diffraction(&geom, &cal, &opts_sim).expect("simulate");

            let mut input_poni = truth.clone();
            input_poni.rot3 = 0.0;

            let res = calibrate_rot3_polarization(
                &frame.data,
                frame.dim1,
                frame.dim2,
                None,
                &input_poni,
                &cal,
                pol_factor,
                &PolarizationRot3Options::default(),
            )
            .expect("calibrate_rot3_polarization should succeed");

            let err_deg = wrap_half_pi(res.rot3_rad - truth.rot3).abs().to_degrees();
            let tol_deg = (3.5 * res.uncertainty_rad.to_degrees()).max(1.0);
            assert!(
                err_deg < tol_deg,
                "case (rot3={true_rot3_deg}°, P={pol_factor}, tilt=({rot1_deg},{rot2_deg}), p2={poni2_px}): got {:.2}° (err={err_deg:.2}°, unc={:.2}°, snr={:.1}, ratio={:.2})",
                res.rot3_rad.to_degrees(),
                res.uncertainty_rad.to_degrees(),
                res.modulation_snr,
                res.fitted_polarization_ratio,
            );
            assert!(res.reliable, "expected reliable fit, got {res:?}");
            assert!(
                (res.fitted_polarization_ratio - 1.0).abs() < 0.15,
                "expected polarization ratio ~1.0, got {:.2}",
                res.fitted_polarization_ratio
            );
        }
    }

    #[test]
    fn unbiased_under_polarization_mismatch_and_saturation_masking() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let truth = make_poni(27.0, 5.0, -4.0, 140.0);
        let geom = Geometry::new(512, 512, truth.clone());

        // 1. True P = 0.70, fitted with nominal P = 0.99: angle must stay unbiased
        //    and fitted_polarization_ratio must recover ~0.70 / 0.99 = 0.707.
        let frame = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                fwhm_rad: 2.5e-3,
                ring_intensity: RingIntensity::FromCalibrant { fallback: 25_000.0 },
                background: 50.0,
                apply_solid_angle: true,
                polarization_factor: Some(0.70),
                poisson_noise: true,
                rng_seed: 77,
                ..Default::default()
            },
        )
        .expect("simulate");
        let mut input = truth.clone();
        input.rot3 = 0.0;
        let res = calibrate_rot3_polarization(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &input,
            &cal,
            0.99,
            &PolarizationRot3Options::default(),
        )
        .expect("mismatch fit");
        let err_deg = wrap_half_pi(res.rot3_rad - truth.rot3).abs().to_degrees();
        assert!(err_deg < 1.5, "P-mismatch angle biased: err={err_deg:.2}°");
        assert!(
            (res.fitted_polarization_ratio - (0.70 / 0.99)).abs() < 0.08,
            "expected ratio ~0.707, got {:.3}",
            res.fitted_polarization_ratio
        );

        // 2. Masking saturated peak-core pixels must not bias rot3.
        let frame_sat = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                fwhm_rad: 2.5e-3,
                ring_intensity: RingIntensity::FromCalibrant { fallback: 30_000.0 },
                background: 50.0,
                apply_solid_angle: true,
                polarization_factor: Some(0.99),
                poisson_noise: true,
                rng_seed: 88,
                ..Default::default()
            },
        )
        .expect("simulate sat");
        let sat_mask: Vec<bool> = frame_sat.data.iter().map(|&v| v > 12_000.0).collect();
        let res_sat = calibrate_rot3_polarization(
            &frame_sat.data,
            frame_sat.dim1,
            frame_sat.dim2,
            Some(&sat_mask),
            &input,
            &cal,
            0.99,
            &PolarizationRot3Options::default(),
        )
        .expect("saturated-masked fit");
        let err_sat = wrap_half_pi(res_sat.rot3_rad - truth.rot3)
            .abs()
            .to_degrees();
        let tol_sat = (3.0 * res_sat.uncertainty_rad.to_degrees()).max(2.0);
        assert!(
            err_sat < tol_sat,
            "masked saturation biased rot3: err={err_sat:.2}° (unc={:.2}°)",
            res_sat.uncertainty_rad.to_degrees()
        );
    }

    #[test]
    fn unpolarized_data_is_flagged_unreliable() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let truth = make_poni(0.0, 0.0, 0.0, 256.0);
        let geom = Geometry::new(512, 512, truth.clone());
        let frame = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                fwhm_rad: 2.5e-3,
                ring_intensity: RingIntensity::FromCalibrant { fallback: 10_000.0 },
                background: 50.0,
                apply_solid_angle: true,
                polarization_factor: None, // unpolarized!
                poisson_noise: true,
                rng_seed: 42,
                ..Default::default()
            },
        )
        .expect("simulate");
        let res = calibrate_rot3_polarization(
            &frame.data,
            frame.dim1,
            frame.dim2,
            None,
            &truth,
            &cal,
            0.99,
            &PolarizationRot3Options::default(),
        )
        .expect("unpolarized returns result");
        assert!(
            !res.reliable,
            "unpolarized data must not be marked reliable: {res:?}"
        );
        assert!(
            res.fitted_polarization_ratio < 0.25,
            "unpolarized ratio should be near 0, got {:.3}",
            res.fitted_polarization_ratio
        );
    }

    #[test]
    fn rejects_invalid_polarization_factor() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let poni = make_poni(0.0, 0.0, 0.0, 256.0);
        let img = vec![10.0; 64 * 64];
        assert!(matches!(
            calibrate_rot3_polarization(
                &img,
                64,
                64,
                None,
                &poni,
                &cal,
                0.0,
                &PolarizationRot3Options::default(),
            ),
            Err(PolarizationRot3Error::InvalidInput(_))
        ));
    }
}
