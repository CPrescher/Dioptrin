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

/// Which image signal to fit for `rot3` polarization calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolarizationRot3Source {
    /// Automatically select the best source (default): tries calibrant powder
    /// rings first (fast and immune to diffuse dark-pedestal bias when rings
    /// are smooth), and falls back to between-rings diffuse background when
    /// rings are spotty, weak, or unreliable.
    #[default]
    Auto,
    /// Use only the azimuthal intensity variation along calibrant powder rings.
    Rings,
    /// Use only the diffuse background in radial `2θ` bands between calibrant
    /// rings (immune to crystallite spottiness on area detectors).
    Diffuse,
}

/// Configuration for [`calibrate_rot3_polarization`].
#[derive(Debug, Clone)]
pub struct PolarizationRot3Options {
    /// Which image signal to fit (`Auto` default, `Rings`, or `Diffuse`).
    pub source: PolarizationRot3Source,
    /// Number of azimuthal bins around `[-π, π]` per ring or radial band
    /// (default `72`, 5°/bin).
    pub n_az_bins: usize,
    /// Minimum scattering angle `2θ` in radians for rings/bands included in the
    /// fit (default `5°` — lower angles have negligible `sin²(2θ)` modulation).
    pub min_tth_rad: f64,
    /// Maximum number of calibrant rings (above `min_tth_rad`) to inspect
    /// (default `32`).
    pub max_rings: usize,
    /// Optional fixed half-width in `2θ` (radians) for integrating each ring's
    /// peak band. `None` (default) auto-selects from the local pixel angular
    /// size (`max(3.5 · Δ2θ_px, 0.002 rad)`), clamped to avoid neighboring rings.
    pub ring_half_width_rad: Option<f64>,
    /// Minimum populated unmasked azimuthal bins required for a ring or radial
    /// band to contribute (default `12`, i.e. ≥ 60° azimuthal span at 72 bins).
    pub min_samples_per_ring: usize,
    /// Minimum total `(ring/band, azimuth)` samples across all rings/bands
    /// required to accept a fit (default `24`).
    pub min_total_samples: usize,
}

impl Default for PolarizationRot3Options {
    fn default() -> Self {
        Self {
            source: PolarizationRot3Source::Auto,
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
    /// Number of calibrant rings (for [`PolarizationRot3Source::Rings`]) or
    /// radial diffuse bands (for [`PolarizationRot3Source::Diffuse`]) that
    /// contributed to the fit.
    pub n_rings_used: usize,
    /// Total number of `(ring/band, azimuth)` intensity samples used in the fit.
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
    /// Which signal source produced this fit ([`PolarizationRot3Source::Rings`]
    /// or [`PolarizationRot3Source::Diffuse`]).
    pub source_used: PolarizationRot3Source,
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

fn median_of_f32(vals: &mut [f32]) -> f64 {
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    vals[mid] as f64
}

#[derive(Default, Clone)]
struct DiffuseCellAcc {
    vals: Vec<f32>,
    cos2_sum: f64,
    sin2_sum: f64,
}

impl DiffuseCellAcc {
    fn merge(&mut self, other: &mut Self) {
        self.vals.append(&mut other.vals);
        self.cos2_sum += other.cos2_sum;
        self.sin2_sum += other.sin2_sum;
    }
}

/// Calibrate `rot3` from the azimuthal intensity modulation of powder rings
/// and/or between-rings diffuse scattering under a known linear X-ray
/// polarization factor `polarization_factor`.
///
/// By default ([`PolarizationRot3Source::Auto`]), tries calibrant powder rings
/// first (fast and free of dark-pedestal bias when rings are smooth), and
/// automatically falls back to between-rings diffuse background when rings are
/// spotty or unreliable.
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

    let all_tths: Vec<f64> = calibrant
        .two_theta_rad(poni.wavelength)
        .into_iter()
        .flatten()
        .filter(|t| t.is_finite() && *t > 1e-4 && *t < PI * 0.49)
        .collect();

    match options.source {
        PolarizationRot3Source::Rings => fit_rings(
            image,
            dim1,
            dim2,
            mask,
            poni,
            &all_tths,
            polarization_factor,
            options,
            n_az,
            mask_negative,
        ),
        PolarizationRot3Source::Diffuse => fit_diffuse(
            image,
            dim1,
            dim2,
            mask,
            poni,
            &all_tths,
            polarization_factor,
            options,
            n_az,
            mask_negative,
        ),
        PolarizationRot3Source::Auto => {
            let rings_res = fit_rings(
                image,
                dim1,
                dim2,
                mask,
                poni,
                &all_tths,
                polarization_factor,
                options,
                n_az,
                mask_negative,
            );
            // When powder rings yield a reliable fit, return it directly:
            // ring-background subtraction cancels any constant dark pedestal,
            // so reliable ring fits are fast and free of diffuse pedestal bias.
            if let Ok(ref res) = rings_res
                && res.reliable
            {
                return Ok(res.clone());
            }

            let diffuse_res = fit_diffuse(
                image,
                dim1,
                dim2,
                mask,
                poni,
                &all_tths,
                polarization_factor,
                options,
                n_az,
                mask_negative,
            );
            match (rings_res, diffuse_res) {
                (Ok(r), Ok(d)) => {
                    if d.reliable || d.modulation_snr > r.modulation_snr {
                        Ok(d)
                    } else {
                        Ok(r)
                    }
                }
                (Ok(r), Err(_)) => Ok(r),
                (Err(_), Ok(d)) => Ok(d),
                (Err(e), Err(_)) => Err(e),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_rings(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    all_tths: &[f64],
    polarization_factor: f64,
    options: &PolarizationRot3Options,
    n_az: usize,
    mask_negative: bool,
) -> Result<PolarizationRot3Result, PolarizationRot3Error> {
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
                        || (mask_negative && v < 0.0)
                        || v >= 4_294_967_000.0;

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

    solve_polarization_irls(
        &ring_data,
        poni,
        n_az,
        options.min_total_samples,
        PolarizationRot3Source::Rings,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_diffuse(
    image: &[f64],
    dim1: usize,
    dim2: usize,
    mask: Option<&[bool]>,
    poni: &Poni,
    all_tths: &[f64],
    polarization_factor: f64,
    options: &PolarizationRot3Options,
    n_az: usize,
    mask_negative: bool,
) -> Result<PolarizationRot3Result, PolarizationRot3Error> {
    let mut poni0 = poni.clone();
    poni0.rot3 = 0.0;
    let tc = TrigonometricConstants::new(&poni0);
    let p3 = poni0.distance;

    // Estimate visible maximum 2θ from a fast 64×64 grid + corners.
    let mut tth_max = 0.0_f64;
    let r_step = (dim1 / 64).max(1);
    let c_step = (dim2 / 64).max(1);
    for r in (0..dim1).step_by(r_step).chain(std::iter::once(dim1 - 1)) {
        let p1 = (r as f64 + 0.5) * poni0.pixel1 - poni0.poni1;
        for c in (0..dim2).step_by(c_step).chain(std::iter::once(dim2 - 1)) {
            let p2 = (c as f64 + 0.5) * poni0.pixel2 - poni0.poni2;
            let x1 = f_x1(&p1, &p2, &p3, &tc);
            let x2 = f_x2(&p1, &p2, &p3, &tc);
            let x3 = f_x3(&p1, &p2, &p3, &tc);
            tth_max = tth_max.max(f_tth(&x1, &x2, &x3));
        }
    }

    // Prefer higher 2θ where sin²(2θ)/(1+cos²(2θ)) is stronger and low-angle
    // sample-holder/capillary absorption bias is weaker, while still working
    // on low-angle setups.
    let adaptive_lo = (0.35 * tth_max).min(12.0_f64.to_radians());
    let mut tth_min = options.min_tth_rad.max(adaptive_lo);
    if tth_max <= tth_min + 0.02 {
        tth_min = options.min_tth_rad;
    }
    if tth_max <= tth_min + 0.02 {
        return Err(PolarizationRot3Error::InsufficientSignal(
            "detector 2θ range above min_tth_rad is too narrow for diffuse rot3 fit".into(),
        ));
    }

    // Exclude bands around every calibrant ring so spotty Bragg peaks and
    // their wings do not contaminate the diffuse background.
    let px_max = poni.pixel1.max(poni.pixel2);
    let mut ring_exclusions: Vec<(f64, f64)> = all_tths
        .iter()
        .copied()
        .filter(|&t| t < tth_max + 0.05)
        .map(|tth| {
            let cos2 = tth.cos().powi(2);
            let d_tth_px = (px_max / poni.distance) * cos2;
            let hw = options
                .ring_half_width_rad
                .map(|w| 1.5 * w)
                .unwrap_or_else(|| (5.0 * d_tth_px).max(0.004));
            (tth, hw)
        })
        .collect();
    ring_exclusions.sort_by(|a, b| a.0.total_cmp(&b.0));

    const N_BANDS: usize = 48;
    let band_w = (tth_max - tth_min) / N_BANDS as f64;
    let inv_az = (n_az as f64) / (2.0 * PI);
    let total_cells = N_BANDS * n_az;

    let chunk_rows = (dim1 / rayon::current_num_threads().max(1)).clamp(8, 128);
    let (mut cells, tth_sums, tth_cnts) = (0..dim1)
        .into_par_iter()
        .chunks(chunk_rows)
        .map(|rows| {
            let mut local_cells = vec![DiffuseCellAcc::default(); total_cells];
            let mut local_tth_sum = [0.0_f64; N_BANDS];
            let mut local_tth_cnt = [0_u64; N_BANDS];
            for r in rows {
                let p1 = (r as f64 + 0.5) * poni0.pixel1 - poni0.poni1;
                let row_off = r * dim2;
                for c in 0..dim2 {
                    let idx = row_off + c;
                    let v = image[idx];
                    // Skip masked, non-finite, negative sentinels, Eiger
                    // u32::MAX gap sentinels, and zero-padded dead corners on
                    // positive-pedestal detectors.
                    if mask.is_some_and(|m| m[idx])
                        || !v.is_finite()
                        || (mask_negative && v <= 0.0)
                        || v >= 4_294_967_000.0
                    {
                        continue;
                    }
                    let p2 = (c as f64 + 0.5) * poni0.pixel2 - poni0.poni2;
                    let x1 = f_x1(&p1, &p2, &p3, &tc);
                    let x2 = f_x2(&p1, &p2, &p3, &tc);
                    let x3 = f_x3(&p1, &p2, &p3, &tc);
                    let tth = f_tth(&x1, &x2, &x3);
                    if tth < tth_min || tth >= tth_max {
                        continue;
                    }
                    if !ring_exclusions.is_empty() {
                        let pos = ring_exclusions.partition_point(|&(rt, _)| rt < tth);
                        let lo = pos.saturating_sub(1);
                        let hi = (pos + 1).min(ring_exclusions.len());
                        let mut in_ring = false;
                        for &(rt, hw) in &ring_exclusions[lo..hi] {
                            if (tth - rt).abs() <= hw {
                                in_ring = true;
                                break;
                            }
                        }
                        if in_ring {
                            continue;
                        }
                    }
                    let sa = f_solid_angle(&p1, &p2, &p3);
                    if sa <= 0.0 {
                        continue;
                    }
                    let b_rad = (((tth - tth_min) / band_w).floor() as usize).min(N_BANDS - 1);
                    let az0 = f_azimuth(&x1, &x2);
                    let b_az =
                        (((az0 + PI) * inv_az).floor() as isize).rem_euclid(n_az as isize) as usize;
                    let cell = &mut local_cells[b_rad * n_az + b_az];
                    cell.vals.push((v / sa) as f32);
                    let two_az = 2.0 * az0;
                    cell.cos2_sum += two_az.cos();
                    cell.sin2_sum += two_az.sin();
                    local_tth_sum[b_rad] += tth;
                    local_tth_cnt[b_rad] += 1;
                }
            }
            (local_cells, local_tth_sum, local_tth_cnt)
        })
        .reduce(
            || {
                (
                    vec![DiffuseCellAcc::default(); total_cells],
                    [0.0_f64; N_BANDS],
                    [0_u64; N_BANDS],
                )
            },
            |(mut a_cells, mut a_ts, mut a_tc), (mut b_cells, b_ts, b_tc)| {
                for (dst, src) in a_cells.iter_mut().zip(b_cells.iter_mut()) {
                    dst.merge(src);
                }
                for i in 0..N_BANDS {
                    a_ts[i] += b_ts[i];
                    a_tc[i] += b_tc[i];
                }
                (a_cells, a_ts, a_tc)
            },
        );

    let mut band_data: Vec<RingAzData> = Vec::new();
    for b_rad in 0..N_BANDS {
        if tth_cnts[b_rad] == 0 {
            continue;
        }
        let mean_tth = tth_sums[b_rad] / tth_cnts[b_rad] as f64;
        let cos2 = mean_tth.cos().powi(2);
        let sin2 = mean_tth.sin().powi(2);
        let alpha = polarization_factor * sin2 / (1.0 + cos2);
        if alpha.abs() < 1e-4 {
            continue;
        }
        let mut samples = Vec::new();
        for b_az in 0..n_az {
            let cell = &mut cells[b_rad * n_az + b_az];
            let cnt = cell.vals.len();
            if cnt < 10 {
                continue;
            }
            let y = median_of_f32(&mut cell.vals);
            if !y.is_finite() {
                continue;
            }
            let inv_cnt = 1.0 / cnt as f64;
            samples.push(RingAzSample {
                cos2: cell.cos2_sum * inv_cnt,
                sin2: cell.sin2_sum * inv_cnt,
                y,
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
        let est_noise = (mad - 0.5 * alpha.abs()).clamp(0.008, 0.50);
        band_data.push(RingAzData {
            alpha,
            sigma_rel: est_noise,
            samples,
        });
    }

    solve_polarization_irls(
        &band_data,
        poni,
        n_az,
        options.min_total_samples,
        PolarizationRot3Source::Diffuse,
    )
}

fn solve_polarization_irls(
    ring_data: &[RingAzData],
    poni: &Poni,
    n_az: usize,
    min_total_samples: usize,
    source_used: PolarizationRot3Source,
) -> Result<PolarizationRot3Result, PolarizationRot3Error> {
    let n_rings_used = ring_data.len();
    let n_samples: usize = ring_data.iter().map(|r| r.samples.len()).sum();
    if n_rings_used == 0 || n_samples < min_total_samples {
        return Err(PolarizationRot3Error::InsufficientSignal(format!(
            "only {n_samples} valid unmasked azimuth samples across {n_rings_used} rings/bands (need >= {min_total_samples})"
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
            // Winsorize standardized residuals so a single corrupted/sentinel
            // bin (e.g. unmasked u32::MAX Eiger gap) cannot explode s² and
            // collapse SNR to 0 while Huber IRLS already downweighted it.
            let u_clip = u.min(2.0 * DELTA_HUBER);
            weighted_rss += u_clip * u_clip;
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
        source_used,
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

    #[test]
    fn diffuse_mode_recovers_rot3_when_rings_are_spotty_or_absent() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let true_rot3_deg = -24.5;
        let pol_factor = 0.99;
        let truth = make_poni(true_rot3_deg, 6.0, -4.0, 220.0);
        let dim1 = 512usize;
        let dim2 = 512usize;
        let tc = TrigonometricConstants::new(&truth);
        let p3 = truth.distance;

        // Synthesize pure polarized diffuse scattering (no smooth powder rings)
        // plus a few huge random Bragg spikes on ring positions.
        let mut img = vec![0.0_f64; dim1 * dim2];
        for r in 0..dim1 {
            let p1 = (r as f64 + 0.5) * truth.pixel1 - truth.poni1;
            for c in 0..dim2 {
                let p2 = (c as f64 + 0.5) * truth.pixel2 - truth.poni2;
                let x1 = f_x1(&p1, &p2, &p3, &tc);
                let x2 = f_x2(&p1, &p2, &p3, &tc);
                let x3 = f_x3(&p1, &p2, &p3, &tc);
                let tth = f_tth(&x1, &x2, &x3);
                let az = f_azimuth(&x1, &x2);
                let sa = f_solid_angle(&p1, &p2, &p3);
                let cos2 = tth.cos().powi(2);
                let sin2 = tth.sin().powi(2);
                let pol = 0.5 * (1.0 + cos2 - pol_factor * sin2 * (2.0 * az).cos());
                // Smooth radially decaying diffuse intensity (e.g. air + Compton)
                let diffuse = 200.0 * (-3.0 * tth).exp() + 40.0;
                img[r * dim2 + c] = diffuse * sa * pol;
            }
        }

        let mut input = truth.clone();
        input.rot3 = 0.0;

        // Auto mode should fall back to Diffuse and recover rot3 accurately.
        let res = calibrate_rot3_polarization(
            &img,
            dim1,
            dim2,
            None,
            &input,
            &cal,
            pol_factor,
            &PolarizationRot3Options::default(),
        )
        .expect("Auto should fall back to Diffuse");
        assert_eq!(res.source_used, PolarizationRot3Source::Diffuse);
        assert!(res.reliable, "expected reliable diffuse fit: {res:?}");
        let err_deg = wrap_half_pi(res.rot3_rad - truth.rot3).abs().to_degrees();
        assert!(
            err_deg < 0.5,
            "diffuse rot3 error too large: got {:.2}°, expected {true_rot3_deg:.2}° (err={err_deg:.2}°)",
            res.rot3_rad.to_degrees()
        );
        assert!(
            (res.fitted_polarization_ratio - 1.0).abs() < 0.05,
            "expected β ~ 1.0, got {:.3}",
            res.fitted_polarization_ratio
        );
    }

    #[test]
    fn real_spotty_datasets_auto_fallback_to_diffuse_and_strip_rejects() {
        use crate::frame::Frame;

        // 1. CeO2_Pilatus1M (spotty rings -> Auto falls back to Diffuse)
        let f_ceo2 = Frame::from_tiff("test_data/CeO2_Pilatus1M.tif".to_string()).expect("load");
        let p_ceo2 = Poni::load("test_data/v1.poni").expect("load v1.poni");
        let c_ceo2 = Calibrant::from_d_file("test_data/calibrants/CeO2.D").expect("load CeO2");
        let res_ceo2 = calibrate_rot3_polarization(
            &f_ceo2.data,
            f_ceo2.dim1,
            f_ceo2.dim2,
            None,
            &p_ceo2,
            &c_ceo2,
            0.99,
            &PolarizationRot3Options::default(),
        )
        .expect("CeO2_Pilatus1M rot3");
        assert_eq!(res_ceo2.source_used, PolarizationRot3Source::Diffuse);
        assert!(
            res_ceo2.reliable,
            "CeO2_Pilatus1M should be reliable: {res_ceo2:?}"
        );
        assert!(
            res_ceo2.rot3_rad.to_degrees().abs() < 3.0,
            "CeO2_Pilatus1M rot3 expected near 0°, got {:.2}°",
            res_ceo2.rot3_rad.to_degrees()
        );

        // 2. LaB6_40keV_MarCCD (spotty rings -> Auto falls back to Diffuse)
        let f_mar = Frame::from_tiff("test_data/LaB6_40keV_MarCCD.tif".to_string()).expect("load");
        let p_mar = Poni::load("test_data/LaB6_40keV_MarCCD.poni").expect("load mar poni");
        let c_lab6 = Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").expect("load");
        let res_mar = calibrate_rot3_polarization(
            &f_mar.data,
            f_mar.dim1,
            f_mar.dim2,
            None,
            &p_mar,
            &c_lab6,
            0.99,
            &PolarizationRot3Options::default(),
        )
        .expect("LaB6_40keV_MarCCD rot3");
        assert_eq!(res_mar.source_used, PolarizationRot3Source::Diffuse);
        assert!(
            res_mar.reliable,
            "LaB6_40keV_MarCCD should be reliable: {res_mar:?}"
        );
        assert!(
            res_mar.rot3_rad.to_degrees().abs() < 4.0,
            "LaB6_40keV_MarCCD rot3 expected near 0°, got {:.2}°",
            res_mar.rot3_rad.to_degrees()
        );

        // 3. ceo2_side (narrow strip detector -> must remain unreliable)
        let f_side = Frame::from_tiff("test_data/ceo2_side.tif".to_string()).expect("load");
        let p_side = Poni::load("test_data/ceo2_side.poni").expect("load ceo2_side poni");
        let res_side = calibrate_rot3_polarization(
            &f_side.data,
            f_side.dim1,
            f_side.dim2,
            None,
            &p_side,
            &c_ceo2,
            0.99,
            &PolarizationRot3Options::default(),
        );
        if let Ok(r) = res_side {
            assert!(
                !r.reliable,
                "narrow strip ceo2_side must not be marked reliable: {r:?}"
            );
        }
    }
}
