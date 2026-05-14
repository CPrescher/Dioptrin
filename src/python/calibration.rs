//! Python bindings for the calibration module.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::calibration::{
    AutoCalibrateOptions, AutoMethod, Calibrant, CalibrationError, CalibrationOptions,
    CalibrationPoints, CalibrationResult, ImagePeakOptions, Loss, MassifOptions, ParamMask,
    RansacOptions, RansacResult, RingIdOptions, RingIdResult, auto_calibrate, find_peaks,
    find_peaks_image, identify_rings, ransac_ellipses, refine,
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

// ---------- find_peaks_massif (peak picking) ----------

/// Massif-based peak picking from a seed.
///
/// Given a 2D image (any float dtype convertible to f64) and an integer
/// (d1, d2) seed, return a list of sub-pixel (d1, d2) peaks that belong to
/// the same connected "above-local-background" region as the seed. This is
/// the same algorithm Dioptas's `find_peaks_automatic` uses via pyFAI's
/// Massif.
///
/// # Arguments
///
/// * `image`        — 2D `float64` numpy array
/// * `seed`         — `(d1, d2)` pixel indices near a peak
/// * `valley_size`  — Gaussian σ for the background filter, in pixels.
///                    Pass `None` (default) to use pyFAI's adaptive value
///                    `max(5, max(dim1, dim2) / 50)`.
/// * `max_peaks`    — cap on returned sub-peaks (default 200)
/// * `dedup_dist`   — merge peaks closer than this in pixel space (default 1.0)
#[pyfunction]
#[pyo3(name = "find_peaks_massif")]
#[pyo3(signature = (image, seed, valley_size=None, max_peaks=200, dedup_dist=1.0))]
pub fn find_peaks_massif_py(
    image: PyReadonlyArray2<'_, f64>,
    seed: (usize, usize),
    valley_size: Option<f64>,
    max_peaks: usize,
    dedup_dist: f64,
) -> PyResult<Vec<(f64, f64)>> {
    let view = image.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "image must be C-contiguous (row-major) float64",
        ));
    }
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];
    let slice = image
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("image must be contiguous: {}", e)))?;
    let opts = MassifOptions {
        valley_size,
        max_peaks,
        dedup_dist,
    };
    Ok(find_peaks(slice, dim1, dim2, seed, &opts))
}

// ---------- RANSAC ellipse clustering ----------

#[pyclass(name = "EllipseFit")]
pub struct PyEllipseFit {
    inner: crate::calibration::Ellipse,
}

#[pymethods]
impl PyEllipseFit {
    #[getter]
    fn center(&self) -> (f64, f64) {
        self.inner.center
    }
    #[getter]
    fn semi_major(&self) -> f64 {
        self.inner.semi_major
    }
    #[getter]
    fn semi_minor(&self) -> f64 {
        self.inner.semi_minor
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    /// Conic coefficients `[A, B, C, D, E, F]` with the convention
    /// `A x² + B xy + C y² + D x + E y + F = 0`.
    #[getter]
    fn conic(&self) -> [f64; 6] {
        self.inner.conic
    }
    fn distance(&self, point: (f64, f64)) -> f64 {
        self.inner.sampson_distance(point)
    }
    fn __repr__(&self) -> String {
        format!(
            "EllipseFit(center=({:.2}, {:.2}), semi=({:.2}, {:.2}), theta={:.4})",
            self.inner.center.0,
            self.inner.center.1,
            self.inner.semi_major,
            self.inner.semi_minor,
            self.inner.theta,
        )
    }
}

#[pyclass(name = "RansacEllipseResult")]
pub struct PyRansacEllipseResult {
    inner: RansacResult,
}

#[pymethods]
impl PyRansacEllipseResult {
    /// List of `EllipseFit`, sorted by semi-major axis (smallest first).
    #[getter]
    fn ellipses(&self) -> Vec<PyEllipseFit> {
        self.inner
            .ellipses
            .iter()
            .map(|&e| PyEllipseFit { inner: e })
            .collect()
    }
    /// Per-input-point assignment: `Some(k)` = inlier of `ellipses[k]`,
    /// `None` = outlier rejected from every ellipse.
    #[getter]
    fn assignments(&self) -> Vec<Option<usize>> {
        self.inner.assignments.clone()
    }
    /// `True` for inliers (any ellipse), `False` for outliers.
    fn inlier_mask(&self) -> Vec<bool> {
        self.inner.assignments.iter().map(|a| a.is_some()).collect()
    }
    fn n_ellipses(&self) -> usize {
        self.inner.ellipses.len()
    }
}

/// RANSAC ellipse clustering of a list of 2-D peak coordinates.
///
/// Returns a `RansacEllipseResult` listing the detected ellipses (sorted by
/// semi-major axis — for a powder diffraction pattern this matches ring
/// index k = 0, 1, 2 …) and a per-point assignment vector.
///
/// # Arguments
///
/// * `points`            — list of `(d1, d2)` peak coordinates
/// * `inlier_threshold`  — Sampson distance (pixels) for inlier acceptance
/// * `min_inliers`       — minimum points required to accept an ellipse
/// * `max_iterations`    — random samples per ellipse search
/// * `max_ellipses`      — cap on detected ellipses
/// * `rng_seed`          — for reproducibility
#[pyfunction]
#[pyo3(name = "ransac_ellipses")]
#[pyo3(signature = (points, inlier_threshold=1.5, min_inliers=10, max_iterations=2000, max_ellipses=12, max_axis_ratio=None, dedupe_center_dist=5.0, dedupe_size_rel=0.02, rng_seed=0xC0FF_EEC0_DE12_3456_u64))]
#[allow(clippy::too_many_arguments)]
pub fn ransac_ellipses_py(
    points: Vec<(f64, f64)>,
    inlier_threshold: f64,
    min_inliers: usize,
    max_iterations: usize,
    max_ellipses: usize,
    max_axis_ratio: Option<f64>,
    dedupe_center_dist: f64,
    dedupe_size_rel: f64,
    rng_seed: u64,
) -> PyRansacEllipseResult {
    let opts = RansacOptions {
        inlier_threshold,
        min_inliers,
        max_iterations,
        max_ellipses,
        max_axis_ratio,
        dedupe_center_dist,
        dedupe_size_rel,
        rng_seed,
    };
    let result = ransac_ellipses(&points, &opts);
    PyRansacEllipseResult { inner: result }
}

// ---------- Ring identity ----------

#[pyclass(name = "RingIdResult")]
pub struct PyRingIdResult {
    inner: RingIdResult,
}

#[pymethods]
impl PyRingIdResult {
    /// Per-ellipse calibrant ring assignment. `None` = ellipse couldn't be
    /// matched to any calibrant ring within tolerance.
    #[getter]
    fn ring_index(&self) -> Vec<Option<usize>> {
        self.inner.ring_index.clone()
    }
    /// Calibrant ring of the smallest ellipse, or `None` if that ellipse is
    /// unmatched.
    #[getter]
    fn anchor_ring(&self) -> Option<usize> {
        self.inner.anchor_ring
    }
    /// Number of ellipses confidently matched to calibrant rings.
    #[getter]
    fn n_matched(&self) -> usize {
        self.inner.n_matched
    }
    /// True iff `n_matched >= min_matches`.
    #[getter]
    fn confident(&self) -> bool {
        self.inner.confident
    }
    fn __repr__(&self) -> String {
        format!(
            "RingIdResult(n_matched={}, confident={}, anchor_ring={:?})",
            self.inner.n_matched, self.inner.confident, self.inner.anchor_ring
        )
    }
}

/// Identify which calibrant ring each detected ellipse corresponds to,
/// using the size-ratio invariant `r_k / r_0 = tan(2θ_k) / tan(2θ_0)`.
///
/// Iterates over every (anchor_ellipse, anchor_ring) hypothesis and picks
/// the assignment that maximises matched ellipses subject to a distance-
/// consistency check. Naturally handles missing calibrant rings, spurious
/// detected ellipses, and the case where the smallest detected ellipse is
/// not the calibrant's ring 0.
///
/// # Arguments
///
/// * `ellipses`         — list of `EllipseFit` from `dioptrin.ransac_ellipses`
///                        (or constructed manually). Must be sorted by
///                        semi-major axis (ascending).
/// * `calibrant`        — a `dioptrin.Calibrant`
/// * `wavelength`       — beam wavelength in metres
/// * `ratio_tolerance`  — relative tolerance on size-ratio matching (default 0.02 = 2 %)
/// * `max_anchor_ring`  — highest calibrant ring index to try as the anchor (default 8)
/// * `min_matches`      — minimum matched ellipses for `confident=True` (default 3)
#[pyfunction]
#[pyo3(name = "identify_rings")]
#[pyo3(signature = (ellipses, calibrant, wavelength, ratio_tolerance=0.02, max_anchor_ring=8, min_matches=3))]
pub fn identify_rings_py(
    ellipses: Vec<PyRef<PyEllipseFit>>,
    calibrant: &PyCalibrant,
    wavelength: f64,
    ratio_tolerance: f64,
    max_anchor_ring: usize,
    min_matches: usize,
) -> PyRingIdResult {
    let ell_list: Vec<crate::calibration::Ellipse> = ellipses.iter().map(|e| e.inner).collect();
    let opts = RingIdOptions {
        ratio_tolerance,
        max_anchor_ring,
        min_matches,
    };
    let result = identify_rings(&ell_list, &calibrant.inner, wavelength, &opts);
    PyRingIdResult { inner: result }
}

// ---------- image-wide peak finder ----------

/// Image-wide peak extraction without a seed. Returns sub-pixel
/// `(d1, d2)` peak centers for every above-background blob.
///
/// # Arguments
///
/// * `image`            — 2D `float64` numpy array
/// * `valley_size`      — Gaussian σ for the background filter (pyFAI-adaptive default if `None`)
/// * `threshold_factor` — pixels flagged as foreground when value exceeds
///                        `mean + threshold_factor · std` of the residual (default 3.0)
/// * `max_peaks`        — cap on returned peaks (default 5000)
#[pyfunction]
#[pyo3(name = "find_peaks_image")]
#[pyo3(signature = (image, valley_size=None, threshold_factor=3.0, max_peaks=5000))]
pub fn find_peaks_image_py(
    image: PyReadonlyArray2<'_, f64>,
    valley_size: Option<f64>,
    threshold_factor: f64,
    max_peaks: usize,
) -> PyResult<Vec<(f64, f64)>> {
    let view = image.as_array();
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "image must be C-contiguous (row-major) float64",
        ));
    }
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];
    let slice = image
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("image must be contiguous: {}", e)))?;
    let opts = ImagePeakOptions {
        valley_size,
        threshold_factor,
        max_peaks,
    };
    Ok(find_peaks_image(slice, dim1, dim2, &opts))
}

// ---------- end-to-end auto calibration ----------

#[pyclass(name = "AutoCalibrateResult")]
pub struct PyAutoCalibrateResult {
    refined: CalibrationResult,
    n_detected_ellipses: usize,
    n_identified_ellipses: usize,
    bootstrap_distance: f64,
    bootstrap_poni1: f64,
    bootstrap_poni2: f64,
}

#[pymethods]
impl PyAutoCalibrateResult {
    #[getter]
    fn n_iter(&self) -> usize {
        self.refined.n_iter
    }
    #[getter]
    fn chi2(&self) -> f64 {
        self.refined.chi2
    }
    #[getter]
    fn rms_rad(&self) -> f64 {
        self.refined.rms_rad
    }
    #[getter]
    fn n_detected_ellipses(&self) -> usize {
        self.n_detected_ellipses
    }
    #[getter]
    fn n_identified_ellipses(&self) -> usize {
        self.n_identified_ellipses
    }
    #[getter]
    fn bootstrap_distance(&self) -> f64 {
        self.bootstrap_distance
    }
    #[getter]
    fn bootstrap_poni1(&self) -> f64 {
        self.bootstrap_poni1
    }
    #[getter]
    fn bootstrap_poni2(&self) -> f64 {
        self.bootstrap_poni2
    }
    #[getter]
    fn poni<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let p = &self.refined.poni;
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
            "AutoCalibrateResult(n_iter={}, rms_rad={:.3e}, detected={}, identified={})",
            self.refined.n_iter,
            self.refined.rms_rad,
            self.n_detected_ellipses,
            self.n_identified_ellipses
        )
    }
}

/// End-to-end automatic calibration: peaks → ellipses → ring identity →
/// bootstrap PONI → L-M refine. No manual seeds, no geometry hint.
///
/// # Arguments
///
/// * `peaks`            — list of `(d1, d2)` peak coordinates (e.g. from
///                        `find_peaks_image`)
/// * `calibrant`        — reference powder
/// * `wavelength`       — beam wavelength in metres
/// * `pixel1`, `pixel2` — detector pixel pitch in metres
///
/// RANSAC / ring-id / refine sub-options can be tuned via the optional
/// keyword arguments; defaults are sensible for typical lab/synchrotron
/// data (near-perpendicular detector, Huber loss for the bootstrap).
#[pyfunction]
#[pyo3(name = "auto_calibrate")]
#[pyo3(signature = (
    peaks, calibrant, wavelength, pixel1, pixel2,
    method="radial",
    ransac_inlier_threshold=1.5, ransac_min_inliers=10, ransac_max_iterations=2000,
    ransac_max_ellipses=12, ransac_max_axis_ratio=1.2,
    ransac_dedupe_center_dist=5.0, ransac_dedupe_size_rel=0.02,
    ratio_tolerance=0.03, min_matches=3,
    center_consistency_px=30.0,
    fix=None, huber_c=2.0e-3, max_iter=200, tol=1e-9,
    rng_seed=0xC0FF_EEC0_DE12_3456_u64,
))]
#[allow(clippy::too_many_arguments)]
pub fn auto_calibrate_py(
    peaks: Vec<(f64, f64)>,
    calibrant: &PyCalibrant,
    wavelength: f64,
    pixel1: f64,
    pixel2: f64,
    method: &str,
    ransac_inlier_threshold: f64,
    ransac_min_inliers: usize,
    ransac_max_iterations: usize,
    ransac_max_ellipses: usize,
    ransac_max_axis_ratio: f64,
    ransac_dedupe_center_dist: f64,
    ransac_dedupe_size_rel: f64,
    ratio_tolerance: f64,
    min_matches: usize,
    center_consistency_px: f64,
    fix: Option<&Bound<'_, PyList>>,
    huber_c: f64,
    max_iter: usize,
    tol: f64,
    rng_seed: u64,
) -> PyResult<PyAutoCalibrateResult> {
    let auto_method = match method.to_lowercase().as_str() {
        "radial" => AutoMethod::Radial,
        "ransac" => AutoMethod::Ransac,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method '{}'; valid: 'radial' (default) or 'ransac'",
                other
            )));
        }
    };
    let opts = AutoCalibrateOptions {
        method: auto_method,
        center: Default::default(),
        radial: Default::default(),
        ransac: RansacOptions {
            inlier_threshold: ransac_inlier_threshold,
            min_inliers: ransac_min_inliers,
            max_iterations: ransac_max_iterations,
            max_ellipses: ransac_max_ellipses,
            max_axis_ratio: Some(ransac_max_axis_ratio),
            dedupe_center_dist: ransac_dedupe_center_dist,
            dedupe_size_rel: ransac_dedupe_size_rel,
            rng_seed,
        },
        ring_id: RingIdOptions {
            ratio_tolerance,
            max_anchor_ring: 8,
            min_matches,
        },
        refine_mask: parse_mask(fix)?,
        refine_loss: Loss::Huber { c: huber_c },
        max_iter,
        tol,
        center_consistency_px,
    };

    let (refined, diag) =
        auto_calibrate(&peaks, &calibrant.inner, wavelength, pixel1, pixel2, &opts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyAutoCalibrateResult {
        refined,
        n_detected_ellipses: diag.n_detected_ellipses,
        n_identified_ellipses: diag.n_identified_ellipses,
        bootstrap_distance: diag.bootstrap_distance,
        bootstrap_poni1: diag.bootstrap_poni1,
        bootstrap_poni2: diag.bootstrap_poni2,
    })
}
