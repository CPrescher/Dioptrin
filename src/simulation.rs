//! Powder diffraction image simulation for a given [`Geometry`] and [`Calibrant`].
//!
//! Computes a 2D synthetic diffraction [`Frame`] by evaluating Debye–Scherrer
//! rings at the Bragg angles `2θ = 2 arcsin(λ / 2d)` of a [`Calibrant`] (loaded
//! from a pyFAI-style `.D` file or constructed from explicit d-spacings) using
//! the exact pixel geometry of [`Geometry`] (including detector tilt, off-center
//! beam, spline distortions, module gaps, and non-planar detectors).
//!
//! # Example
//!
//! ```no_run
//! use dioptrin::calibration::Calibrant;
//! use dioptrin::geometry::Geometry;
//! use dioptrin::poni::Poni;
//! use dioptrin::simulation::{SimulationOptions, simulate_diffraction};
//!
//! let poni = Poni::load("test_data/LaB6_40keV_MarCCD.poni").unwrap();
//! let geometry = Geometry::new(2048, 2048, poni);
//! let calibrant = Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").unwrap();
//! let frame = simulate_diffraction(&geometry, &calibrant, &SimulationOptions::default()).unwrap();
//! assert_eq!((frame.dim1, frame.dim2), (2048, 2048));
//! ```

use std::fmt;
use std::path::Path;

use rayon::prelude::*;

use crate::calibration::{Calibrant, CalibrantError};
use crate::frame::Frame;
use crate::geometry::Geometry;

/// Radial peak profile shape in `2θ` (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeakProfile {
    /// Gaussian profile: `exp(-4 ln(2) * (Δ2θ / fwhm)^2)`.
    Gaussian,
    /// Lorentzian (Cauchy) profile: `1 / (1 + (2 Δ2θ / fwhm)^2)`.
    Lorentzian,
    /// Pseudo-Voigt profile: `(1 - eta) * Gaussian + eta * Lorentzian`,
    /// where `eta` is the Lorentzian fraction in `[0.0, 1.0]`.
    PseudoVoigt { eta: f64 },
}

/// How per-ring relative amplitudes are determined before scaling by `i_max`.
#[derive(Debug, Clone, PartialEq)]
pub enum RingIntensity {
    /// Every Bragg ring has equal relative amplitude `1.0`.
    Uniform,
    /// Use relative intensities parsed from the calibrant `.D` file, normalized
    /// so the strongest listed ring has relative amplitude `1.0`. Rings without
    /// a listed intensity use `fallback` (or `1.0` if no ring in the calibrant
    /// has a listed intensity).
    FromCalibrant { fallback: f64 },
    /// Explicit per-ring relative amplitudes aligned with `calibrant.d_spacings`.
    Custom(Vec<f64>),
}

/// Configuration for [`simulate_diffraction`].
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationOptions {
    /// Peak profile shape in `2θ`. Default: [`PeakProfile::Gaussian`].
    pub profile: PeakProfile,
    /// Full width at half maximum (FWHM) of the diffraction rings in `2θ`
    /// (radians). Default: `2.0e-3` rad (~0.115°).
    pub fwhm_rad: f64,
    /// Peak intensity scale factor (before solid-angle / polarization
    /// modulation). Default: `1000.0`.
    pub i_max: f64,
    /// Constant background offset added to every pixel. Default: `0.0`.
    pub background: f64,
    /// Per-ring relative intensity model. Default: [`RingIntensity::Uniform`].
    pub ring_intensity: RingIntensity,
    /// Subpixel oversampling factor per axis. `N` evaluates an `N × N` uniform
    /// grid inside each pixel and averages the result; `1` evaluates at pixel
    /// centers only. Default: `1`.
    pub oversampling: usize,
    /// Multiply the simulated ring intensity by the pixel solid angle
    /// (`Geometry::calc_solid_angle`). Default: `true`, matching Dioptrin's
    /// [`crate::integration::Integrator`] (which always divides by solid angle)
    /// and pyFAI's `fake_calibration_image`.
    pub apply_solid_angle: bool,
    /// Optional linear polarization factor in `[-1.0, 1.0]`. When `Some(p)`,
    /// multiplies the simulated ring intensity by the same polarization factor
    /// that integration corrects for (`Geometry::calc_polarization`).
    /// Default: `None`.
    pub polarization_factor: Option<f64>,
    /// When `Geometry` has an attached [`crate::detector::Detector`] with a
    /// built-in mask (e.g. module gaps), masked pixels are set to `Some(val)`.
    /// Set to `None` to leave masked pixels unmasked. Default: `Some(0.0)`.
    pub mask_value: Option<f64>,
    /// Apply Poisson (shot) noise to the simulated pixel intensities.
    /// Default: `false`.
    pub poisson_noise: bool,
    /// Standard deviation of additive zero-mean Gaussian readout noise.
    /// `0.0` disables readout noise. Note that additive Gaussian noise may
    /// produce negative, non-integer pixel values. Default: `0.0`.
    pub readout_noise: f64,
    /// Deterministic RNG seed used when `poisson_noise` is enabled or
    /// `readout_noise > 0.0`. Output is independent of thread count.
    pub rng_seed: u64,
}

impl Default for SimulationOptions {
    fn default() -> Self {
        Self {
            profile: PeakProfile::Gaussian,
            fwhm_rad: 2.0e-3,
            i_max: 1000.0,
            background: 0.0,
            ring_intensity: RingIntensity::Uniform,
            oversampling: 1,
            apply_solid_angle: true,
            polarization_factor: None,
            mask_value: Some(0.0),
            poisson_noise: false,
            readout_noise: 0.0,
            rng_seed: 0x5EED_D109_7A50_0001,
        }
    }
}

/// Errors returned by [`simulate_diffraction`] and
/// [`simulate_diffraction_from_file`].
#[derive(Debug)]
pub enum SimulationError {
    /// The geometry has zero rows or zero columns.
    EmptyGeometry { dim1: usize, dim2: usize },
    /// `geometry.poni.wavelength` must be finite and strictly positive (meters).
    InvalidWavelength(f64),
    /// One of the [`SimulationOptions`] values is out of range or non-finite.
    InvalidOption(String),
    /// Failed to load or parse a `.D` calibrant file.
    Calibrant(CalibrantError),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationError::EmptyGeometry { dim1, dim2 } => write!(
                f,
                "cannot simulate diffraction image for empty geometry ({dim1} x {dim2})"
            ),
            SimulationError::InvalidWavelength(wl) => write!(
                f,
                "invalid wavelength {wl}: geometry.poni.wavelength must be finite and > 0 (meters)"
            ),
            SimulationError::InvalidOption(msg) => write!(f, "invalid simulation option: {msg}"),
            SimulationError::Calibrant(err) => write!(f, "calibrant error: {err}"),
        }
    }
}

impl std::error::Error for SimulationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SimulationError::Calibrant(err) => Some(err),
            _ => None,
        }
    }
}

impl From<CalibrantError> for SimulationError {
    fn from(err: CalibrantError) -> Self {
        SimulationError::Calibrant(err)
    }
}

/// Simulate a 2D powder diffraction image for the given [`Geometry`] and
/// [`Calibrant`].
///
/// The returned [`Frame`] has shape `(geometry.dim1(), geometry.dim2())` in
/// Dioptrin's standard row-major `(d1, d2)` pixel convention and can be passed
/// directly to [`crate::integration::Integrator`], peak pickers, or calibration
/// routines.
pub fn simulate_diffraction(
    geometry: &Geometry,
    calibrant: &Calibrant,
    options: &SimulationOptions,
) -> Result<Frame, SimulationError> {
    let (dim1, dim2) = geometry.shape();
    if dim1 == 0 || dim2 == 0 {
        return Err(SimulationError::EmptyGeometry { dim1, dim2 });
    }

    let wavelength = geometry.poni.wavelength;
    if !wavelength.is_finite() || wavelength <= 0.0 {
        return Err(SimulationError::InvalidWavelength(wavelength));
    }

    validate_options(options, calibrant)?;

    let rings = build_active_rings(calibrant, wavelength, options);
    let evaluator = ProfileEvaluator::new(options.profile, options.fwhm_rad);

    let solid_angle = if options.apply_solid_angle {
        Some(geometry.calc_solid_angle())
    } else {
        None
    };

    let detector_mask = match (options.mask_value, geometry.detector()) {
        (Some(_), Some(det)) => Some(
            det.builtin_mask([dim1, dim2])
                .map_err(SimulationError::InvalidOption)?,
        ),
        _ => None,
    };

    let transform = geometry.subpixel_transform();
    let affine = transform.affine_plane();
    let n_sub = options.oversampling;
    let inv_n_sub = 1.0 / (n_sub as f64);
    let inv_n_sub2 = inv_n_sub * inv_n_sub;
    let pol_factor = options.polarization_factor;
    let background = options.background;
    let apply_noise = options.poisson_noise || options.readout_noise > 0.0;

    let mut data = vec![0.0_f64; dim1 * dim2];

    data.par_chunks_mut(dim2)
        .enumerate()
        .for_each(|(r, row_out)| {
            let mut rng = if apply_noise {
                Some(RowRng::new(options.rng_seed, r as u64))
            } else {
                None
            };
            let row_offset = r * dim2;
            let r_f = r as f64;

            for (c, out_px) in row_out.iter_mut().enumerate() {
                let idx = row_offset + c;

                if let (Some(mv), Some(mask)) = (options.mask_value, detector_mask.as_deref())
                    && mask[idx] != 0
                {
                    *out_px = mv;
                    continue;
                }

                let c_f = c as f64;

                let (mut signal, center_tth_az) = if n_sub == 1 {
                    let (tth, az) = eval_tth_az(
                        &transform,
                        affine.as_ref(),
                        r_f + 0.5,
                        c_f + 0.5,
                        pol_factor.is_some(),
                    );
                    (evaluator.eval(&rings, tth), (tth, az))
                } else {
                    let mut sum = 0.0;
                    for sr in 0..n_sub {
                        let d1 = r_f + (sr as f64 + 0.5) * inv_n_sub;
                        for sc in 0..n_sub {
                            let d2 = c_f + (sc as f64 + 0.5) * inv_n_sub;
                            let tth = eval_tth(&transform, affine.as_ref(), d1, d2);
                            sum += evaluator.eval(&rings, tth);
                        }
                    }
                    let center = if pol_factor.is_some() {
                        eval_tth_az(&transform, affine.as_ref(), r_f + 0.5, c_f + 0.5, true)
                    } else {
                        (0.0, 0.0)
                    };
                    (sum * inv_n_sub2, center)
                };

                if let Some(sa) = solid_angle.as_deref() {
                    signal *= sa[idx];
                }

                if let Some(pf) = pol_factor {
                    let (tth, az) = center_tth_az;
                    let cos2_tth = tth.cos().powi(2);
                    let pol = crate::geometry::math::f_polarization(&pf, &cos2_tth, &az);
                    signal *= pol;
                }

                let mut val = signal + background;

                if let Some(rng) = rng.as_mut() {
                    if options.poisson_noise {
                        val = rng.poisson(val);
                    }
                    if options.readout_noise > 0.0 {
                        val += options.readout_noise * rng.normal();
                    }
                }

                *out_px = val;
            }
        });

    Ok(Frame {
        name: format!("simulated:{}", calibrant.name),
        data,
        dim1,
        dim2,
    })
}

/// Convenience wrapper that loads a pyFAI-style `.D` calibrant file from
/// `calibrant_path` and simulates the diffraction image for `geometry`.
pub fn simulate_diffraction_from_file<P: AsRef<Path>>(
    geometry: &Geometry,
    calibrant_path: P,
    options: &SimulationOptions,
) -> Result<Frame, SimulationError> {
    let calibrant = Calibrant::from_d_file(calibrant_path)?;
    simulate_diffraction(geometry, &calibrant, options)
}

// ---------- Validation & ring preparation ----------

fn validate_options(
    options: &SimulationOptions,
    calibrant: &Calibrant,
) -> Result<(), SimulationError> {
    if !options.fwhm_rad.is_finite() || options.fwhm_rad <= 0.0 {
        return Err(SimulationError::InvalidOption(format!(
            "fwhm_rad must be finite and > 0, got {}",
            options.fwhm_rad
        )));
    }
    if !options.i_max.is_finite() || options.i_max < 0.0 {
        return Err(SimulationError::InvalidOption(format!(
            "i_max must be finite and >= 0, got {}",
            options.i_max
        )));
    }
    if !options.background.is_finite() {
        return Err(SimulationError::InvalidOption(format!(
            "background must be finite, got {}",
            options.background
        )));
    }
    if options.oversampling == 0 {
        return Err(SimulationError::InvalidOption(
            "oversampling must be >= 1".to_string(),
        ));
    }
    if let PeakProfile::PseudoVoigt { eta } = options.profile
        && (!eta.is_finite() || !(0.0..=1.0).contains(&eta))
    {
        return Err(SimulationError::InvalidOption(format!(
            "PseudoVoigt eta must be in [0.0, 1.0], got {eta}"
        )));
    }
    if let Some(pf) = options.polarization_factor
        && (!pf.is_finite() || !(-1.0..=1.0).contains(&pf))
    {
        return Err(SimulationError::InvalidOption(format!(
            "polarization_factor must be in [-1.0, 1.0], got {pf}"
        )));
    }
    if !options.readout_noise.is_finite() || options.readout_noise < 0.0 {
        return Err(SimulationError::InvalidOption(format!(
            "readout_noise must be finite and >= 0, got {}",
            options.readout_noise
        )));
    }
    match &options.ring_intensity {
        RingIntensity::Uniform => {}
        RingIntensity::FromCalibrant { fallback } => {
            if !fallback.is_finite() || *fallback < 0.0 {
                return Err(SimulationError::InvalidOption(format!(
                    "FromCalibrant fallback must be finite and >= 0, got {fallback}"
                )));
            }
        }
        RingIntensity::Custom(weights) => {
            if weights.len() != calibrant.n_rings() {
                return Err(SimulationError::InvalidOption(format!(
                    "Custom ring_intensity length ({}) must match calibrant.n_rings() ({})",
                    weights.len(),
                    calibrant.n_rings()
                )));
            }
            if let Some(&bad) = weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
                return Err(SimulationError::InvalidOption(format!(
                    "Custom ring_intensity values must be finite and >= 0, got {bad}"
                )));
            }
        }
    }
    Ok(())
}

/// Active Bragg ring `(two_theta_rad, amplitude)`.
fn build_active_rings(
    calibrant: &Calibrant,
    wavelength_m: f64,
    options: &SimulationOptions,
) -> Vec<(f64, f64)> {
    let tths = calibrant.two_theta_rad(wavelength_m);

    let max_cal_intensity = match &options.ring_intensity {
        RingIntensity::FromCalibrant { .. } => calibrant
            .intensities
            .iter()
            .filter_map(|&v| v)
            .fold(0.0_f64, f64::max),
        _ => 0.0,
    };

    let mut rings: Vec<(f64, f64)> = tths
        .into_iter()
        .enumerate()
        .filter_map(|(idx, tth_opt)| {
            let tth = tth_opt?;
            let rel = match &options.ring_intensity {
                RingIntensity::Uniform => 1.0,
                RingIntensity::FromCalibrant { fallback } => {
                    if max_cal_intensity > 0.0 {
                        calibrant
                            .intensities
                            .get(idx)
                            .copied()
                            .flatten()
                            .map(|v| v / max_cal_intensity)
                            .unwrap_or(*fallback)
                    } else {
                        1.0
                    }
                }
                RingIntensity::Custom(weights) => weights[idx],
            };
            let amp = options.i_max * rel;
            if amp == 0.0 { None } else { Some((tth, amp)) }
        })
        .collect();

    rings.sort_by(|a, b| a.0.total_cmp(&b.0));
    rings
}

// ---------- Profile evaluation ----------

struct ProfileEvaluator {
    profile: PeakProfile,
    /// `4 * ln(2) / fwhm^2` = `1 / (2 * sigma^2)`.
    inv_2sigma2: f64,
    /// `4 / fwhm^2` = `1 / gamma^2`.
    inv_gamma2: f64,
    /// Cutoff radius in radians for Gaussian binary-search windowing (`6 * sigma`).
    gaussian_cutoff: f64,
}

impl ProfileEvaluator {
    fn new(profile: PeakProfile, fwhm_rad: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let inv_2sigma2 = 4.0 * ln2 / (fwhm_rad * fwhm_rad);
        let inv_gamma2 = 4.0 / (fwhm_rad * fwhm_rad);
        let sigma = fwhm_rad / (2.0 * (2.0 * ln2).sqrt());
        Self {
            profile,
            inv_2sigma2,
            inv_gamma2,
            gaussian_cutoff: 6.0 * sigma,
        }
    }

    #[inline]
    fn eval(&self, rings: &[(f64, f64)], tth: f64) -> f64 {
        match self.profile {
            PeakProfile::Gaussian => {
                let lo = tth - self.gaussian_cutoff;
                let hi = tth + self.gaussian_cutoff;
                let start = rings.partition_point(|&(rt, _)| rt < lo);
                let mut sum = 0.0;
                for &(rt, amp) in &rings[start..] {
                    if rt > hi {
                        break;
                    }
                    let d = tth - rt;
                    sum += amp * (-d * d * self.inv_2sigma2).exp();
                }
                sum
            }
            PeakProfile::Lorentzian => {
                let mut sum = 0.0;
                for &(rt, amp) in rings {
                    let d = tth - rt;
                    sum += amp / (1.0 + d * d * self.inv_gamma2);
                }
                sum
            }
            PeakProfile::PseudoVoigt { eta } => {
                let one_minus_eta = 1.0 - eta;
                let mut sum = 0.0;
                for &(rt, amp) in rings {
                    let d = tth - rt;
                    let d2 = d * d;
                    let g = if d.abs() <= self.gaussian_cutoff {
                        (-d2 * self.inv_2sigma2).exp()
                    } else {
                        0.0
                    };
                    let l = 1.0 / (1.0 + d2 * self.inv_gamma2);
                    sum += amp * (one_minus_eta * g + eta * l);
                }
                sum
            }
        }
    }
}

// ---------- Fast geometry helpers ----------

#[inline]
fn eval_tth(
    transform: &crate::geometry::SubpixelTransform,
    affine: Option<&([f64; 3], [f64; 3], [f64; 3])>,
    d1: f64,
    d2: f64,
) -> f64 {
    if let Some((p0, s1, s2)) = affine {
        let x = p0[0] + d1 * s1[0] + d2 * s2[0];
        let y = p0[1] + d1 * s1[1] + d2 * s2[1];
        let z = p0[2] + d1 * s1[2] + d2 * s2[2];
        (y * y + x * x).sqrt().atan2(z)
    } else {
        transform.tth(d1, d2)
    }
}

#[inline]
fn eval_tth_az(
    transform: &crate::geometry::SubpixelTransform,
    affine: Option<&([f64; 3], [f64; 3], [f64; 3])>,
    d1: f64,
    d2: f64,
    need_az: bool,
) -> (f64, f64) {
    if let Some((p0, s1, s2)) = affine {
        let x = p0[0] + d1 * s1[0] + d2 * s2[0];
        let y = p0[1] + d1 * s1[1] + d2 * s2[1];
        let z = p0[2] + d1 * s1[2] + d2 * s2[2];
        let tth = (y * y + x * x).sqrt().atan2(z);
        let az = if need_az { x.atan2(y) } else { 0.0 };
        (tth, az)
    } else if need_az {
        transform.tth_and_azimuth(d1, d2)
    } else {
        (transform.tth(d1, d2), 0.0)
    }
}

// ---------- Deterministic per-row PRNG (no external dependencies) ----------

/// SplitMix64-seeded per-row PRNG for reproducible noise independent of thread
/// scheduling.
struct RowRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl RowRng {
    fn new(seed: u64, row: u64) -> Self {
        let mut s = seed;
        let mut mixed = splitmix64(&mut s) ^ row;
        let state = splitmix64(&mut mixed);
        Self {
            state,
            spare_normal: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }

    /// Uniform float strictly inside `(0, 1)`.
    #[inline]
    fn uniform_open(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }

    /// Standard normal `N(0, 1)` via Box–Muller.
    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.uniform_open();
        let u2 = self.uniform_open();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Poisson sample with mean `lambda` (returns `0.0` for `lambda <= 0.0`).
    fn poisson(&mut self, lambda: f64) -> f64 {
        if !lambda.is_finite() || lambda <= 0.0 {
            return 0.0;
        }
        if lambda < 30.0 {
            // Knuth's exact algorithm for small lambda.
            let limit = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.uniform_open();
                if p <= limit {
                    return (k - 1) as f64;
                }
            }
        } else {
            // Transformed rejection with squeeze (Hörmann 1993, PTRS) for lambda >= 30.
            let b = 0.931 + 2.53 * lambda.sqrt();
            let a = -0.059 + 0.02483 * b;
            let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
            let v_r = 0.9277 - 3.6224 / (b - 2.0);
            let ln_lambda = lambda.ln();

            loop {
                let u = self.uniform_open() - 0.5;
                let v = self.uniform_open();
                let us = 0.5 - u.abs();
                let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
                if k < 0.0 {
                    continue;
                }
                if us >= 0.07 && v <= v_r {
                    return k;
                }
                if us < 0.013 && v > us {
                    continue;
                }
                let lhs = (v * inv_alpha / (a / (us * us) + b)).ln();
                let rhs = -lambda + k * ln_lambda - ln_gamma_int(k + 1.0);
                if lhs <= rhs {
                    return k;
                }
            }
        }
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Lanczos approximation to `ln(Gamma(x))` for positive `x` (accurate to ~1e-15).
fn ln_gamma_int(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::calibration::{ImagePeakOptions, find_peaks_image};
    use crate::detector;
    use crate::integration::{Integrator, RadialUnit};
    use crate::poni::Poni;

    fn test_poni() -> Poni {
        Poni {
            pixel1: 1.72e-4,
            pixel2: 1.72e-4,
            distance: 0.2,
            poni1: 256.0 * 1.72e-4,
            poni2: 256.0 * 1.72e-4,
            rot1: 0.05,
            rot2: -0.04,
            rot3: 0.0,
            spline_file: String::new(),
            wavelength: 0.3738e-10,
        }
    }

    #[test]
    fn simulates_rings_at_expected_bragg_angles() {
        let poni = test_poni();
        let geom = Geometry::new(512, 512, poni.clone());
        let cal = Calibrant::from_d_file("test_data/calibrants/LaB6_SRM660b.D").unwrap();

        let opts = SimulationOptions {
            fwhm_rad: 2.0e-3,
            i_max: 1000.0,
            background: 50.0,
            ..Default::default()
        };

        let frame = simulate_diffraction(&geom, &cal, &opts).unwrap();
        assert_eq!((frame.dim1, frame.dim2), (512, 512));
        assert_eq!(frame.data.len(), 512 * 512);
        assert!(frame.data.iter().all(|v| v.is_finite() && *v >= 50.0));

        // Integrate 1D and verify peaks appear at the exact calibrant 2θ positions.
        let mut integrator = Integrator::new(geom);
        integrator.set_radial_unit(RadialUnit::TthRad);
        let (radial, intensity) = integrator.integrate1d(&frame, 2000);

        let r_max = *radial.last().unwrap();
        let ring_tths: Vec<f64> = cal
            .two_theta_rad(poni.wavelength)
            .into_iter()
            .flatten()
            .filter(|&t| t > 0.01 && t < r_max - 0.01)
            .collect();
        assert!(
            ring_tths.len() >= 5,
            "expected at least 5 LaB6 rings on detector, got {}",
            ring_tths.len()
        );

        let dr = radial[1] - radial[0];
        for expected_tth in &ring_tths[..5] {
            // Find the local maximum within +/- 0.005 rad of expected_tth.
            let mut best_i = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (i, (&r, &val)) in radial.iter().zip(&intensity).enumerate() {
                if (r - expected_tth).abs() < 0.005 && val > best_val {
                    best_val = val;
                    best_i = i;
                }
            }
            assert!(
                best_val > 800.0,
                "ring at 2θ={expected_tth:.5} rad had weak peak {best_val:.1}"
            );
            assert!(
                (radial[best_i] - expected_tth).abs() <= dr * 1.5,
                "ring peak at {:.6} rad differs from expected {:.6} rad (bin width {:.6})",
                radial[best_i],
                expected_tth,
                dr
            );
        }
    }

    #[test]
    fn simulate_from_file_matches_simulate_with_calibrant() {
        let geom = Geometry::new(128, 128, test_poni());
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
        let opts = SimulationOptions::default();

        let f1 = simulate_diffraction(&geom, &cal, &opts).unwrap();
        let f2 =
            simulate_diffraction_from_file(&geom, "test_data/calibrants/CeO2.D", &opts).unwrap();
        assert_eq!(f1.data, f2.data);
    }

    #[test]
    fn calibrant_intensities_scale_rings_proportionally() {
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
        // Verify CeO2.D inline intensities were parsed.
        assert_eq!(cal.intensities[0], Some(100.0));
        assert_eq!(cal.intensities[1], Some(27.73));
        assert_eq!(cal.intensities[9], None);

        let mut poni = test_poni();
        poni.rot1 = 0.0;
        poni.rot2 = 0.0;
        let geom = Geometry::new(512, 512, poni.clone());
        let opts = SimulationOptions {
            fwhm_rad: 1.5e-3,
            i_max: 1000.0,
            ring_intensity: RingIntensity::FromCalibrant { fallback: 0.05 },
            ..Default::default()
        };
        let frame = simulate_diffraction(&geom, &cal, &opts).unwrap();

        let mut integrator = Integrator::new(geom);
        integrator.set_radial_unit(RadialUnit::TthRad);
        let (radial, intensity) = integrator.integrate1d(&frame, 2000);

        let tths = cal.two_theta_rad(poni.wavelength);
        let peak_at = |target: f64| -> f64 {
            radial
                .iter()
                .zip(&intensity)
                .filter(|&(&r, _)| (r - target).abs() < 0.003)
                .map(|(_, &v)| v)
                .fold(0.0_f64, f64::max)
        };
        let p0 = peak_at(tths[0].unwrap());
        let p1 = peak_at(tths[1].unwrap());
        let ratio = p1 / p0;
        assert!(
            (ratio - 0.2773).abs() < 0.02,
            "expected ring1/ring0 ratio ~0.2773, got {ratio:.4} (p0={p0:.1}, p1={p1:.1})"
        );
    }

    #[test]
    fn solid_angle_and_polarization_cancel_on_integration() {
        let mut poni = test_poni();
        poni.distance = 0.12;
        poni.rot1 = 0.15;
        poni.rot2 = -0.10;
        let geom = Geometry::new(256, 256, poni.clone());
        let cal = Calibrant::from_d_spacings("single_ring", vec![2.5]);

        let opts_raw = SimulationOptions {
            fwhm_rad: 3.0e-3,
            i_max: 1000.0,
            apply_solid_angle: false,
            polarization_factor: None,
            ..Default::default()
        };
        let opts_phys = SimulationOptions {
            apply_solid_angle: true,
            polarization_factor: Some(0.95),
            ..opts_raw.clone()
        };

        let f_raw = simulate_diffraction(&geom, &cal, &opts_raw).unwrap();
        let f_phys = simulate_diffraction(&geom, &cal, &opts_phys).unwrap();

        // Physical image (with solid-angle & polarization) integrated with
        // polarization correction should recover the exact same 2D cake as
        // an unmodulated image (without polarization) integrated with solid-angle
        // correction divided back out — specifically, pixel-wise
        // `f_phys[i] / (solid_angle[i] * polarization[i]) == f_raw[i]`.
        let sa = geom.calc_solid_angle();
        let tth = geom.calc_tth_all();
        let az = geom.calc_azimuth_all();
        let pol = geom.calc_polarization(0.95, &tth, &az);
        for i in 0..f_raw.data.len() {
            let recovered = f_phys.data[i] / (sa[i] * pol[i]);
            assert!(
                (recovered - f_raw.data[i]).abs() < 1e-10,
                "pixel {i}: recovered={recovered} raw={}",
                f_raw.data[i]
            );
        }

        // Also verify 1D integration of `f_phys` with `polarization_factor = Some(0.95)`
        // matches `f_sa_only` (solid angle only, no polarization) integrated with
        // `polarization_factor = None`, proving polarization cancels identically.
        let opts_sa_only = SimulationOptions {
            apply_solid_angle: true,
            polarization_factor: None,
            ..opts_raw
        };
        let f_sa_only = simulate_diffraction(&geom, &cal, &opts_sa_only).unwrap();

        let int_sa = Integrator::new(geom.clone());
        let (_, i_sa) = int_sa.integrate1d(&f_sa_only, 400);

        let mut int_phys = Integrator::new(geom);
        int_phys.set_polarization_factor(Some(0.95));
        let (_, i_phys) = int_phys.integrate1d(&f_phys, 400);

        let max_diff = i_sa
            .iter()
            .zip(&i_phys)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // Within a finite radial bin where the Gaussian ring intensity varies
        // across the bin width, the polarization weighting shifts the bin
        // weighted average by a tiny fraction (< 0.01% of i_max = 1000).
        assert!(
            max_diff < 0.1,
            "expected polarization to cancel on 1D integration, max_diff={max_diff:e}"
        );
    }

    #[test]
    fn profiles_lorentzian_and_pseudo_voigt_behave_as_expected() {
        let mut poni = test_poni();
        poni.rot1 = 0.0;
        poni.rot2 = 0.0;
        let geom = Geometry::new(128, 128, poni);
        let cal = Calibrant::from_d_spacings("one", vec![3.0]);

        let f_gauss = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                profile: PeakProfile::Gaussian,
                ..Default::default()
            },
        )
        .unwrap();
        let f_lor = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                profile: PeakProfile::Lorentzian,
                ..Default::default()
            },
        )
        .unwrap();
        let f_pv = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                profile: PeakProfile::PseudoVoigt { eta: 0.5 },
                ..Default::default()
            },
        )
        .unwrap();

        // Pseudo-Voigt with eta=0.5 must be the exact midpoint of Gaussian and Lorentzian.
        for i in 0..f_pv.data.len() {
            let expected = 0.5 * (f_gauss.data[i] + f_lor.data[i]);
            assert!(
                (f_pv.data[i] - expected).abs() < 1e-10,
                "pixel {i}: pv={} expected={expected}",
                f_pv.data[i]
            );
        }

        // Lorentzian has heavier tails than Gaussian, so total integrated intensity is larger.
        let sum_g: f64 = f_gauss.data.iter().sum();
        let sum_l: f64 = f_lor.data.iter().sum();
        assert!(
            sum_l > sum_g,
            "expected Lorentzian sum {sum_l} > Gaussian sum {sum_g}"
        );
    }

    #[test]
    fn oversampling_smooths_narrow_rings() {
        let geom = Geometry::new(128, 128, test_poni());
        let cal = Calibrant::from_d_spacings("one", vec![3.0]);
        let base = SimulationOptions {
            fwhm_rad: 5.0e-4,
            oversampling: 1,
            ..Default::default()
        };
        let over = SimulationOptions {
            oversampling: 3,
            ..base.clone()
        };
        let f1 = simulate_diffraction(&geom, &cal, &base).unwrap();
        let f3 = simulate_diffraction(&geom, &cal, &over).unwrap();
        assert_eq!(f1.data.len(), f3.data.len());
        // Oversampling should change pixel values on a sub-pixel narrow ring
        // while preserving approximate total intensity.
        let sum1: f64 = f1.data.iter().sum();
        let sum3: f64 = f3.data.iter().sum();
        assert!((sum1 - sum3).abs() / sum3 < 0.05);
        assert_ne!(f1.data, f3.data);
    }

    #[test]
    fn noise_is_deterministic_and_has_expected_statistics() {
        let geom = Geometry::new(200, 200, test_poni());
        // Empty / far ring so every pixel has mean = background = 100.0
        let cal = Calibrant::from_d_spacings("far", vec![0.2]);
        let opts = SimulationOptions {
            background: 100.0,
            poisson_noise: true,
            readout_noise: 5.0,
            rng_seed: 42,
            ..Default::default()
        };

        let f1 = simulate_diffraction(&geom, &cal, &opts).unwrap();
        let f2 = simulate_diffraction(&geom, &cal, &opts).unwrap();
        assert_eq!(
            f1.data, f2.data,
            "noise must be deterministic for same seed"
        );

        let n = f1.data.len() as f64;
        let mean = f1.data.iter().sum::<f64>() / n;
        let var = f1.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        // Expected mean = 100, expected variance = 100 (Poisson) + 25 (readout) = 125.
        assert!((mean - 100.0).abs() < 1.0, "mean={mean}");
        assert!((var - 125.0).abs() < 10.0, "var={var}");
    }

    #[test]
    fn detector_builtin_mask_is_applied() {
        let det = Arc::new(detector::named("Pilatus1M").unwrap());
        let [d1, d2] = det.max_shape.unwrap();
        let poni = Poni {
            pixel1: det.pixel1,
            pixel2: det.pixel2,
            distance: 0.2,
            poni1: (d1 as f64 / 2.0) * det.pixel1,
            poni2: (d2 as f64 / 2.0) * det.pixel2,
            rot1: 0.0,
            rot2: 0.0,
            rot3: 0.0,
            spline_file: String::new(),
            wavelength: 0.3738e-10,
        };
        let geom = Geometry::new(d1, d2, poni)
            .with_detector(det.clone())
            .unwrap();
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
        let opts = SimulationOptions {
            background: 10.0,
            mask_value: Some(-1.0),
            ..Default::default()
        };
        let frame = simulate_diffraction(&geom, &cal, &opts).unwrap();
        let mask = det.builtin_mask([d1, d2]).unwrap();
        let masked_count = mask.iter().filter(|&&m| m != 0).count();
        assert!(masked_count > 0);
        for (px, &m) in frame.data.iter().zip(&mask) {
            if m != 0 {
                assert_eq!(*px, -1.0);
            } else {
                assert!(*px >= 10.0);
            }
        }
    }

    #[test]
    fn round_trip_calibration_recovers_simulated_geometry() {
        use crate::calibration::{CalibrationOptions, CalibrationPoints, Loss, ParamMask, refine};

        let poni = Poni {
            pixel1: 1.72e-4,
            pixel2: 1.72e-4,
            distance: 0.10,
            poni1: 260.0 * 1.72e-4,
            poni2: 245.0 * 1.72e-4,
            rot1: 0.08,
            rot2: -0.06,
            rot3: 0.0,
            spline_file: String::new(),
            wavelength: 0.3738e-10,
        };
        let geom = Geometry::new(512, 512, poni.clone());
        let cal = Calibrant::from_d_file("test_data/calibrants/CeO2.D").unwrap();
        let frame = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                fwhm_rad: 3.0e-3,
                i_max: 1000.0,
                background: 20.0,
                oversampling: 2,
                ..Default::default()
            },
        )
        .unwrap();

        let peaks = find_peaks_image(
            &frame.data,
            frame.dim1,
            frame.dim2,
            &ImagePeakOptions::default(),
        );
        assert!(
            peaks.len() > 100,
            "expected >100 peaks, got {}",
            peaks.len()
        );

        // Start from a perturbed guess and refine with Huber loss + auto ring assignment.
        let mut start_poni = poni.clone();
        start_poni.distance += 0.002;
        start_poni.poni1 -= 0.0005;
        start_poni.poni2 += 0.0005;
        start_poni.rot1 -= 0.01;
        start_poni.rot2 += 0.01;
        let start_geom = Geometry::new(512, 512, start_poni);

        let mut points = CalibrationPoints::new();
        for &(d1, d2) in &peaks {
            points.push(d1, d2, None);
        }
        let opts = CalibrationOptions {
            mask: ParamMask::fix_wavelength(),
            loss: Loss::Huber { c: 1.0e-3 },
            auto_assign_rings: true,
            ..Default::default()
        };
        let cal_result = refine(&start_geom, &points, &cal, &opts).unwrap();
        let fitted = &cal_result.poni;

        assert!(
            (fitted.distance - poni.distance).abs() < 5e-4,
            "distance mismatch: fitted={} true={}",
            fitted.distance,
            poni.distance
        );
        assert!(
            (fitted.poni1 - poni.poni1).abs() < 5e-4,
            "poni1 mismatch: fitted={} true={}",
            fitted.poni1,
            poni.poni1
        );
        assert!(
            (fitted.poni2 - poni.poni2).abs() < 5e-4,
            "poni2 mismatch: fitted={} true={}",
            fitted.poni2,
            poni.poni2
        );
        assert!(
            (fitted.rot1 - poni.rot1).abs() < 5e-3,
            "rot1 mismatch: fitted={} true={}",
            fitted.rot1,
            poni.rot1
        );
        assert!(
            (fitted.rot2 - poni.rot2).abs() < 5e-3,
            "rot2 mismatch: fitted={} true={}",
            fitted.rot2,
            poni.rot2
        );
    }

    #[test]
    fn validates_invalid_inputs() {
        let cal = Calibrant::from_d_spacings("test", vec![3.0]);
        let mut poni = test_poni();

        // Empty geometry
        assert!(matches!(
            simulate_diffraction(
                &Geometry::new(0, 10, poni.clone()),
                &cal,
                &SimulationOptions::default()
            ),
            Err(SimulationError::EmptyGeometry { dim1: 0, dim2: 10 })
        ));

        // Zero wavelength
        poni.wavelength = 0.0;
        assert!(matches!(
            simulate_diffraction(
                &Geometry::new(10, 10, poni),
                &cal,
                &SimulationOptions::default()
            ),
            Err(SimulationError::InvalidWavelength(_))
        ));

        // Bad FWHM
        let geom = Geometry::new(10, 10, test_poni());
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    fwhm_rad: 0.0,
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));

        // Bad eta
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    profile: PeakProfile::PseudoVoigt { eta: 1.5 },
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));

        // Bad oversampling
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    oversampling: 0,
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));

        // Bad custom weights length
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    ring_intensity: RingIntensity::Custom(vec![1.0, 2.0]),
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));

        // Negative i_max and negative Custom weight
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    i_max: -10.0,
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));
        assert!(matches!(
            simulate_diffraction(
                &geom,
                &cal,
                &SimulationOptions {
                    ring_intensity: RingIntensity::Custom(vec![-1.0]),
                    ..Default::default()
                }
            ),
            Err(SimulationError::InvalidOption(_))
        ));
    }

    #[test]
    fn pixel_exact_matches_calc_tth_all() {
        let geom = Geometry::new(64, 64, test_poni());
        let cal = Calibrant::from_d_spacings("one", vec![2.5]);
        let ring_tth = cal.two_theta_rad(geom.poni.wavelength)[0].unwrap();
        let fwhm = 3.0e-3;
        let i_max = 800.0;
        let background = 12.5;

        let frame = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                profile: PeakProfile::Gaussian,
                fwhm_rad: fwhm,
                i_max,
                background,
                apply_solid_angle: false,
                ..Default::default()
            },
        )
        .unwrap();

        let tth_all = geom.calc_tth_all();
        let inv_2sigma2 = 4.0 * std::f64::consts::LN_2 / (fwhm * fwhm);
        for (i, (&got, &tth)) in frame.data.iter().zip(&tth_all).enumerate() {
            let d = tth - ring_tth;
            let expected = background + i_max * (-d * d * inv_2sigma2).exp();
            // Cutoff at 6 sigma introduces < 2e-5 absolute error at i_max=800.
            assert!(
                (got - expected).abs() < 1e-4,
                "pixel {i}: got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn physical_detector_transform_path_matches_calc_tth_all() {
        // An oriented / binned Pilatus1M exercises the full Detector::position
        // SubpixelTransform path (affine_plane() == None) as well as solid angle
        // and built-in mask.
        let mut det = detector::named("Pilatus1M").unwrap();
        det.orientation = 1;
        let det = Arc::new(det);
        let [d1, d2] = det.max_shape.unwrap();
        let poni = Poni {
            pixel1: det.pixel1,
            pixel2: det.pixel2,
            distance: 0.2,
            poni1: (d1 as f64 / 2.0) * det.pixel1,
            poni2: (d2 as f64 / 2.0) * det.pixel2,
            rot1: 0.04,
            rot2: -0.03,
            rot3: 0.0,
            spline_file: String::new(),
            wavelength: 0.3738e-10,
        };
        let geom = Geometry::new(d1, d2, poni).with_detector(det).unwrap();
        let cal = Calibrant::from_d_spacings("one", vec![2.5]);
        let ring_tth = cal.two_theta_rad(geom.poni.wavelength)[0].unwrap();
        let fwhm = 3.0e-3;
        let i_max = 500.0;

        let frame = simulate_diffraction(
            &geom,
            &cal,
            &SimulationOptions {
                profile: PeakProfile::Lorentzian,
                fwhm_rad: fwhm,
                i_max,
                apply_solid_angle: true,
                mask_value: None,
                ..Default::default()
            },
        )
        .unwrap();

        let tth_all = geom.calc_tth_all();
        let sa = geom.calc_solid_angle();
        let inv_gamma2 = 4.0 / (fwhm * fwhm);
        for (i, (&got, (&tth, &sa_i))) in frame.data.iter().zip(tth_all.iter().zip(&sa)).enumerate()
        {
            let d = tth - ring_tth;
            let expected = sa_i * i_max / (1.0 + d * d * inv_gamma2);
            assert!(
                (got - expected).abs() < 1e-10,
                "pixel {i}: got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn poisson_knuth_branch_low_lambda() {
        let geom = Geometry::new(200, 200, test_poni());
        let cal = Calibrant::from_d_spacings("far", vec![0.2]);
        let opts = SimulationOptions {
            background: 5.0,
            poisson_noise: true,
            readout_noise: 0.0,
            rng_seed: 12345,
            ..Default::default()
        };
        let frame = simulate_diffraction(&geom, &cal, &opts).unwrap();
        let n = frame.data.len() as f64;
        let mean = frame.data.iter().sum::<f64>() / n;
        let var = frame.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!((mean - 5.0).abs() < 0.1, "mean={mean}");
        assert!((var - 5.0).abs() < 0.25, "var={var}");
    }
}
