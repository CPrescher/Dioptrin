//! Python bindings for powder diffraction image simulation.

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::calibration::Calibrant;
use crate::detector;
use crate::geometry::Geometry;
use crate::poni::Poni;
use crate::simulation::{
    PeakProfile, RingIntensity, SimulationError, SimulationOptions, SpottyOptions,
    simulate_diffraction,
};

use super::calibration::PyCalibrant;
use super::integrator::PyIntegrator;
use super::utils::poni_from_dict;

fn simulation_err_to_py(err: SimulationError) -> PyErr {
    match err {
        SimulationError::Calibrant(e) => PyIOError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn resolve_geometry(
    geometry: &Bound<'_, PyAny>,
    shape: Option<(usize, usize)>,
    detector_name: Option<&str>,
) -> PyResult<Geometry> {
    let explicit_det = match detector_name {
        Some(name) => Some(Arc::new(
            detector::named(name).map_err(PyValueError::new_err)?,
        )),
        None => None,
    };

    if let Ok(integrator) = geometry.extract::<PyRef<PyIntegrator>>() {
        let mut geom = integrator.inner().geometry().clone();
        // When an explicit detector is requested and `shape` is omitted, prefer
        // that detector's native `max_shape` before falling back to the
        // integrator's existing shape.
        let target_shape = shape.or_else(|| {
            if let Some(det) = explicit_det.as_ref()
                && let Some([d1, d2]) = det.max_shape
            {
                return Some((d1, d2));
            }
            let s = geom.shape();
            if s.0 > 0 && s.1 > 0 {
                Some(s)
            } else {
                geom.detector()
                    .and_then(|d| d.max_shape)
                    .map(|[d1, d2]| (d1, d2))
            }
        });
        let Some((d1, d2)) = target_shape else {
            return Err(PyValueError::new_err(
                "shape=(dim1, dim2) is required when the integrator has no detector shape yet",
            ));
        };
        if let Some(det) = explicit_det {
            geom = Geometry::new(d1, d2, geom.poni.clone())
                .with_detector(det)
                .map_err(PyValueError::new_err)?;
        } else if geom.shape() != (d1, d2) {
            geom.set_shape(d1, d2).map_err(PyValueError::new_err)?;
        }
        return Ok(geom);
    }

    let poni: Poni = if let Ok(dict) = geometry.cast::<PyDict>() {
        poni_from_dict(dict)?
    } else if let Ok(path) = geometry.extract::<String>() {
        Poni::load(&path).map_err(|e| PyIOError::new_err(e.to_string()))?
    } else {
        return Err(PyValueError::new_err(
            "geometry must be a dioptrin.Integrator, a PONI dict, or a path to a .poni file",
        ));
    };

    let target_shape = shape.or_else(|| {
        explicit_det
            .as_ref()
            .and_then(|d| d.max_shape)
            .map(|[d1, d2]| (d1, d2))
    });
    let Some((d1, d2)) = target_shape else {
        return Err(PyValueError::new_err(
            "shape=(dim1, dim2) is required when geometry is a PONI dict or file without a fixed-shape detector",
        ));
    };

    let mut geom = Geometry::new(d1, d2, poni);
    if let Some(det) = explicit_det {
        geom = geom.with_detector(det).map_err(PyValueError::new_err)?;
    }
    Ok(geom)
}

fn resolve_calibrant(calibrant: &Bound<'_, PyAny>) -> PyResult<Calibrant> {
    if let Ok(cal) = calibrant.extract::<PyRef<PyCalibrant>>() {
        Ok(cal.inner.clone())
    } else if let Ok(path) = calibrant.extract::<String>() {
        Calibrant::from_d_file(&path).map_err(|e| PyIOError::new_err(e.to_string()))
    } else {
        Err(PyValueError::new_err(
            "calibrant must be a dioptrin.Calibrant or a path to a .D calibrant file",
        ))
    }
}

fn parse_profile(profile: &str, eta: f64) -> PyResult<PeakProfile> {
    match profile.to_lowercase().as_str() {
        "gaussian" => Ok(PeakProfile::Gaussian),
        "lorentzian" => Ok(PeakProfile::Lorentzian),
        "pseudo_voigt" | "pseudo-voigt" | "pseudovoigt" => Ok(PeakProfile::PseudoVoigt { eta }),
        other => Err(PyValueError::new_err(format!(
            "unknown profile '{other}'; use 'gaussian', 'lorentzian', or 'pseudo_voigt'"
        ))),
    }
}

fn parse_ring_intensity(obj: &Bound<'_, PyAny>, fallback: f64) -> PyResult<RingIntensity> {
    if let Ok(s) = obj.extract::<String>() {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(RingIntensity::Uniform),
            "calibrant" | "from_calibrant" => Ok(RingIntensity::FromCalibrant { fallback }),
            other => Err(PyValueError::new_err(format!(
                "unknown ring_intensity '{other}'; use 'uniform', 'calibrant', or a sequence of floats"
            ))),
        }
    } else if let Ok(weights) = obj.extract::<Vec<f64>>() {
        Ok(RingIntensity::Custom(weights))
    } else {
        Err(PyValueError::new_err(
            "ring_intensity must be 'uniform', 'calibrant', or a sequence of floats",
        ))
    }
}

/// Simulate a 2D powder diffraction image for a given geometry and calibrant.
///
/// # Arguments
///
/// * `geometry`            — a `dioptrin.Integrator`, a PONI dict, or a path to a `.poni` file.
///                           Note: `.poni` paths and dicts carry scalar PONI fields only; pass
///                           `detector="..."` or an `Integrator` with a physical detector attached
///                           to include splines, non-planar maps, or built-in module-gap masks.
///                           Passing an `Integrator` uses its `Geometry` and does not inherit its
///                           `polarization_factor` unless explicitly passed.
/// * `calibrant`           — a `dioptrin.Calibrant` or a path to a pyFAI-style `.D` file.
/// * `shape`               — `(dim1, dim2)` image shape; optional if `geometry` or `detector`
///                           already defines a shape.
/// * `detector`            — optional built-in detector name (e.g. `"Pilatus1M"`).
/// * `profile`             — `"gaussian"` (default), `"lorentzian"`, or `"pseudo_voigt"`.
/// * `fwhm_rad`            — ring FWHM in 2θ radians (default `2e-3`).
/// * `eta`                 — Lorentzian mixing fraction for `"pseudo_voigt"` in `[0, 1]` (default `0.5`).
/// * `i_max`               — peak intensity scale (default `1000.0`).
/// * `background`          — constant background offset (default `0.0`).
/// * `ring_intensity`      — `"uniform"` (default), `"calibrant"`, or a list of per-ring weights
///                           aligned with `calibrant.d_spacings` (sorted descending in d / ascending 2θ).
/// * `fallback_intensity`  — relative intensity for rings missing an intensity in `.D` when
///                           `ring_intensity="calibrant"` (default `0.1`).
/// * `oversampling`        — subpixel oversampling per axis (default `1`).
/// * `apply_solid_angle`   — multiply signal by pixel solid angle (default `True`, matching
///                           `Integrator` and pyFAI).
/// * `polarization_factor` — optional linear polarization factor in `[-1, 1]` (default `None`).
/// * `mask_value`          — value written to built-in detector mask gaps, or `None` (default `0.0`).
/// * `poisson_noise`              — apply Poisson shot noise (default `False`).
/// * `readout_noise`              — Gaussian readout noise std dev (default `0.0`; may produce negative,
///                                  non-integer pixel values).
/// * `rng_seed`                   — deterministic RNG seed for noise/spots (default `0x5EED_D109_7A50_0001`).
/// * `spotty_spots_per_ring`      — optional number of discrete crystallite spots per ring; when set,
///                                  enables coarse-grain ("spotty") azimuthal modulation (default `None`).
/// * `spotty_azimuthal_fwhm_rad`  — azimuthal FWHM of each spot in radians when `spotty_spots_per_ring`
///                                  is set (default `0.03`).
/// * `spotty_continuous_fraction` — continuous ring amplitude in `[0, 1]` superposed under discrete
///                                  spots when `spotty_spots_per_ring` is set (default `0.0`).
///
/// Returns a 2D `numpy.ndarray` (`float64`) of shape `(dim1, dim2)`.
#[pyfunction]
#[pyo3(name = "simulate_diffraction")]
#[pyo3(signature = (
    geometry,
    calibrant,
    shape=None,
    detector=None,
    profile="gaussian",
    fwhm_rad=2.0e-3,
    eta=0.5,
    i_max=1000.0,
    background=0.0,
    ring_intensity=None,
    fallback_intensity=0.1,
    oversampling=1,
    apply_solid_angle=true,
    polarization_factor=None,
    mask_value=Some(0.0),
    poisson_noise=false,
    readout_noise=0.0,
    rng_seed=0x5EED_D109_7A50_0001,
    spotty_spots_per_ring=None,
    spotty_azimuthal_fwhm_rad=0.03,
    spotty_continuous_fraction=0.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_diffraction_py<'py>(
    py: Python<'py>,
    geometry: &Bound<'py, PyAny>,
    calibrant: &Bound<'py, PyAny>,
    shape: Option<(usize, usize)>,
    detector: Option<&str>,
    profile: &str,
    fwhm_rad: f64,
    eta: f64,
    i_max: f64,
    background: f64,
    ring_intensity: Option<&Bound<'py, PyAny>>,
    fallback_intensity: f64,
    oversampling: usize,
    apply_solid_angle: bool,
    polarization_factor: Option<f64>,
    mask_value: Option<f64>,
    poisson_noise: bool,
    readout_noise: f64,
    rng_seed: u64,
    spotty_spots_per_ring: Option<usize>,
    spotty_azimuthal_fwhm_rad: f64,
    spotty_continuous_fraction: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let geom = resolve_geometry(geometry, shape, detector)?;
    let cal = resolve_calibrant(calibrant)?;
    let parsed_profile = parse_profile(profile, eta)?;
    let parsed_intensity = match ring_intensity {
        Some(obj) => parse_ring_intensity(obj, fallback_intensity)?,
        None => RingIntensity::Uniform,
    };
    let spotty = spotty_spots_per_ring.map(|spots_per_ring| SpottyOptions {
        spots_per_ring,
        azimuthal_fwhm_rad: spotty_azimuthal_fwhm_rad,
        continuous_fraction: spotty_continuous_fraction,
    });

    let options = SimulationOptions {
        profile: parsed_profile,
        fwhm_rad,
        i_max,
        background,
        ring_intensity: parsed_intensity,
        spotty,
        oversampling,
        apply_solid_angle,
        polarization_factor,
        mask_value,
        poisson_noise,
        readout_noise,
        rng_seed,
    };

    let frame = py
        .detach(|| simulate_diffraction(&geom, &cal, &options))
        .map_err(simulation_err_to_py)?;

    let arr = PyArray1::from_vec(py, frame.data);
    arr.reshape([frame.dim1, frame.dim2])
}
