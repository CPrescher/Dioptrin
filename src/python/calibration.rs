//! Python bindings for the calibration module.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::calibration::{
    Calibrant, CalibrationError, CalibrationOptions, CalibrationPoints, CalibrationResult, Loss,
    ParamMask, refine,
};
use crate::geometry::Geometry;
use crate::poni::Poni;

use super::integrator::PyIntegrator;
use super::utils::poni_from_dict;

fn calibration_err_to_py(err: CalibrationError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

// ---------- Calibrant ----------

#[pyclass(name = "Calibrant")]
pub struct PyCalibrant {
    pub inner: Calibrant,
}

#[pymethods]
impl PyCalibrant {
    /// Load a calibrant from a pyFAI-style `.D` file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let cal = Calibrant::from_d_file(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: cal })
    }

    /// Build from an explicit list of d-spacings (Å).
    #[staticmethod]
    fn from_d_spacings(name: &str, d_spacings: Vec<f64>) -> Self {
        Self {
            inner: Calibrant::from_d_spacings(name, d_spacings),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn d_spacings(&self) -> Vec<f64> {
        self.inner.d_spacings.clone()
    }

    fn n_rings(&self) -> usize {
        self.inner.n_rings()
    }

    /// 2θ (radians) for each ring at the given wavelength (meters). Rings
    /// outside the Bragg-physical range yield `None`.
    fn two_theta_rad(&self, wavelength_m: f64) -> Vec<Option<f64>> {
        self.inner.two_theta_rad(wavelength_m)
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibrant(name={:?}, n_rings={})",
            self.inner.name,
            self.inner.n_rings()
        )
    }
}

// ---------- CalibrationPoints ----------

#[pyclass(name = "CalibrationPoints")]
pub struct PyCalibrationPoints {
    pub inner: CalibrationPoints,
}

#[pymethods]
impl PyCalibrationPoints {
    #[new]
    fn new() -> Self {
        Self {
            inner: CalibrationPoints::new(),
        }
    }

    /// Append one control point.
    ///
    /// * `d1`, `d2` — pixel coordinates (row, column); sub-pixel allowed
    /// * `ring`    — calibrant ring index (`None` for auto-assignment)
    #[pyo3(signature = (d1, d2, ring=None))]
    fn add(&mut self, d1: f64, d2: f64, ring: Option<usize>) {
        self.inner.push(d1, d2, ring);
    }

    /// Append from parallel arrays.
    fn extend(&mut self, d1: Vec<f64>, d2: Vec<f64>, ring: Vec<Option<usize>>) -> PyResult<()> {
        if d1.len() != d2.len() || d1.len() != ring.len() {
            return Err(PyValueError::new_err(
                "d1, d2, and ring lists must have the same length",
            ));
        }
        for ((a, b), r) in d1.iter().zip(d2.iter()).zip(ring.iter()) {
            self.inner.push(*a, *b, *r);
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn clear(&mut self) {
        self.inner = CalibrationPoints::new();
    }
}

// ---------- options helpers ----------

fn parse_loss(loss: Option<&Bound<'_, PyAny>>) -> PyResult<Loss> {
    let Some(obj) = loss else {
        return Ok(Loss::L2);
    };
    if let Ok(s) = obj.extract::<String>() {
        match s.to_lowercase().as_str() {
            "l2" | "lsq" | "least_squares" => Ok(Loss::L2),
            "huber" => Ok(Loss::Huber { c: 1.0e-3 }), // default 1 mrad
            other => Err(PyValueError::new_err(format!(
                "unknown loss '{}'; use 'l2' or 'huber' or a dict like {{type:'huber', c:1e-3}}",
                other
            ))),
        }
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let kind: String = dict
            .get_item("type")?
            .ok_or_else(|| PyValueError::new_err("loss dict needs a 'type' key"))?
            .extract()?;
        match kind.to_lowercase().as_str() {
            "l2" => Ok(Loss::L2),
            "huber" => {
                let c: f64 = dict
                    .get_item("c")?
                    .ok_or_else(|| PyValueError::new_err("huber loss needs 'c' (rad)"))?
                    .extract()?;
                Ok(Loss::Huber { c })
            }
            other => Err(PyValueError::new_err(format!(
                "unknown loss type '{}'",
                other
            ))),
        }
    } else {
        Err(PyValueError::new_err(
            "loss must be a string or a dict like {type:'huber', c:1e-3}",
        ))
    }
}

fn parse_mask(fix: Option<&Bound<'_, PyList>>) -> PyResult<ParamMask> {
    let mut mask = ParamMask::all_free();
    let Some(list) = fix else {
        // Default: wavelength fixed (matches pyFAI/Dioptas).
        return Ok(ParamMask::fix_wavelength());
    };
    for item in list.iter() {
        let name: String = item.extract()?;
        match name.as_str() {
            "distance" | "dist" => mask.distance = false,
            "poni1" => mask.poni1 = false,
            "poni2" => mask.poni2 = false,
            "rot1" => mask.rot1 = false,
            "rot2" => mask.rot2 = false,
            "rot3" => mask.rot3 = false,
            "wavelength" | "wl" => mask.wavelength = false,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown parameter '{}' in fix list",
                    other
                )));
            }
        }
    }
    Ok(mask)
}

// ---------- CalibrationResult ----------

#[pyclass(name = "CalibrationResult")]
pub struct PyCalibrationResult {
    pub result: CalibrationResult,
}

#[pymethods]
impl PyCalibrationResult {
    #[getter]
    fn n_iter(&self) -> usize {
        self.result.n_iter
    }
    #[getter]
    fn chi2(&self) -> f64 {
        self.result.chi2
    }
    #[getter]
    fn rms_rad(&self) -> f64 {
        self.result.rms_rad
    }
    #[getter]
    fn assignments(&self) -> Vec<Option<usize>> {
        self.result.assignments.clone()
    }
    #[getter]
    fn inliers(&self) -> Vec<bool> {
        self.result.inliers.clone()
    }
    #[getter]
    fn poni<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let p = &self.result.poni;
        d.set_item("pixel1", p.pixel1)?;
        d.set_item("pixel2", p.pixel2)?;
        d.set_item("distance", p.distance)?;
        d.set_item("poni1", p.poni1)?;
        d.set_item("poni2", p.poni2)?;
        d.set_item("rot1", p.rot1)?;
        d.set_item("rot2", p.rot2)?;
        d.set_item("rot3", p.rot3)?;
        d.set_item("wavelength", p.wavelength)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationResult(n_iter={}, chi2={:e}, rms_rad={:e})",
            self.result.n_iter, self.result.chi2, self.result.rms_rad
        )
    }
}

// ---------- refine() ----------

/// Refine detector geometry from picked control points and a known calibrant.
///
/// # Arguments
///
/// * `start`         — starting guess: either a `dioptrin.Integrator` (its PONI is used) or
///                     a PONI dict.
/// * `points`        — a `CalibrationPoints`.
/// * `calibrant`     — a `Calibrant`.
/// * `fix`           — list of parameter names to lock (default: `['wavelength']`).
///                     Valid names: `distance`, `poni1`, `poni2`, `rot1`, `rot2`, `rot3`, `wavelength`.
/// * `loss`          — `'l2'` (default), `'huber'`, or `{type:'huber', c:<rad>}`.
/// * `auto_assign`   — assign unassigned points to nearest predicted ring each iteration.
/// * `assign_tol`    — auto-assignment tolerance in rad of 2θ (default 0.05).
/// * `max_iter`      — L-M iterations (default 100).
/// * `tol`           — relative cost tolerance for convergence (default 1e-9).
///
/// Returns a `CalibrationResult`.
#[pyfunction]
#[pyo3(name = "refine")]
#[pyo3(signature = (start, points, calibrant, fix=None, loss=None, auto_assign=false, assign_tol=0.05, max_iter=100, tol=1e-9))]
#[allow(clippy::too_many_arguments)]
pub fn refine_py(
    start: &Bound<'_, PyAny>,
    points: &PyCalibrationPoints,
    calibrant: &PyCalibrant,
    fix: Option<&Bound<'_, PyList>>,
    loss: Option<&Bound<'_, PyAny>>,
    auto_assign: bool,
    assign_tol: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyCalibrationResult> {
    // Build a starting Poni from either an Integrator (.inner().geometry.poni)
    // or a dict.
    let start_poni: Poni = if let Ok(integrator) = start.extract::<PyRef<PyIntegrator>>() {
        integrator.inner().geometry().poni.clone()
    } else if let Ok(dict) = start.cast::<PyDict>() {
        poni_from_dict(dict)?
    } else {
        return Err(PyValueError::new_err(
            "start must be a dioptrin.Integrator or a PONI dict",
        ));
    };

    // Detector dimensions are not strictly needed by `refine`, but we keep
    // them for diagnostics if available from the integrator. Use 0,0 if not.
    let geometry = Geometry::new(0, 0, start_poni);

    let options = CalibrationOptions {
        mask: parse_mask(fix)?,
        loss: parse_loss(loss)?,
        auto_assign_rings: auto_assign,
        assign_tolerance_rad: assign_tol,
        max_iter,
        tol,
    };

    let result = refine(&geometry, &points.inner, &calibrant.inner, &options)
        .map_err(calibration_err_to_py)?;
    Ok(PyCalibrationResult { result })
}
