//! Python bindings for the calibration module.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::calibration::{
    AutoCalibrateImageOptions, AutoCalibrateOptions, AutoMethod, Calibrant, CalibrationError,
    CalibrationOptions, CalibrationPoints, CalibrationResult, DistortionImageObservation,
    DistortionImageOptions, DistortionImageResult, DistortionModel, DistortionObservation,
    DistortionOptions, DistortionResult, GridSearchOptions, ImagePeakOptions, Loss, MassifOptions,
    ParamMask, PolarizationRot3Options, PolarizationRot3Result, PolarizationRot3Source,
    RansacOptions, RansacResult, RefineImageOptions, RefineImageResult, RingCenterOptions,
    RingIdOptions, RingIdResult, auto_calibrate, auto_calibrate_image_with_mask,
    calibrate_rot3_polarization, estimate_distortion, estimate_distortion_image,
    estimate_distortion_image_multi, estimate_distortion_multi, estimate_ring_center,
    extract_ring_peaks, find_peaks, find_peaks_image, identify_rings, ransac_ellipses, refine,
    refine_image,
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
    /// Load a built-in calibrant by name (e.g. `"CeO2"`, `"LaB6"`, `"Si"`).
    #[staticmethod]
    fn builtin(name: &str) -> PyResult<Self> {
        let cal = Calibrant::builtin(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: cal })
    }

    /// Return the names of all built-in calibrants bundled with Dioptrin.
    #[staticmethod]
    fn builtin_names() -> Vec<&'static str> {
        Calibrant::builtin_names().to_vec()
    }

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

    /// Build from d-spacings (Å) and optional per-ring relative intensities.
    #[staticmethod]
    fn from_d_spacings_and_intensities(
        name: &str,
        d_spacings: Vec<f64>,
        intensities: Vec<Option<f64>>,
    ) -> Self {
        Self {
            inner: Calibrant::from_d_spacings_and_intensities(name, d_spacings, intensities),
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

    #[getter]
    fn intensities(&self) -> Vec<Option<f64>> {
        self.inner.intensities.clone()
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
    bootstrap_ring_center_d1: f64,
    bootstrap_ring_center_d2: f64,
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
    /// Mean ring-center (d1, meters) used to initialise the bootstrap PONI.
    /// **Caveat: equals PONI only for near-perpendicular detectors.** For
    /// tilted detectors PONI and ring-center are distinct (see
    /// `dioptrin.estimate_ring_center` docstring).
    #[getter]
    fn bootstrap_ring_center_d1(&self) -> f64 {
        self.bootstrap_ring_center_d1
    }
    #[getter]
    fn bootstrap_ring_center_d2(&self) -> f64 {
        self.bootstrap_ring_center_d2
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
/// * `method`           — `radial` (default), `ransac`, `grid`, or experimental `auto`
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
    grid_distance_min=0.05, grid_distance_max=0.6, grid_distance_steps=18,
    grid_rot_max_deg=45.0, grid_rot_steps=15,
    grid_poni_search_radius_px=400.0, grid_poni_steps=13,
    grid_tolerance_rad=8.0e-3, grid_max_peaks=400, grid_max_ring_index=12,
    grid_top_k=50,
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
    grid_distance_min: f64,
    grid_distance_max: f64,
    grid_distance_steps: usize,
    grid_rot_max_deg: f64,
    grid_rot_steps: usize,
    grid_poni_search_radius_px: f64,
    grid_poni_steps: usize,
    grid_tolerance_rad: f64,
    grid_max_peaks: usize,
    grid_max_ring_index: usize,
    grid_top_k: usize,
    fix: Option<&Bound<'_, PyList>>,
    huber_c: f64,
    max_iter: usize,
    tol: f64,
    rng_seed: u64,
) -> PyResult<PyAutoCalibrateResult> {
    let auto_method = match method.to_lowercase().as_str() {
        "auto" => AutoMethod::Auto,
        "radial" => AutoMethod::Radial,
        "ransac" => AutoMethod::Ransac,
        "grid" => AutoMethod::Grid,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method '{}'; valid: 'radial' (default), 'ransac', 'grid', or 'auto'",
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
        grid: GridSearchOptions {
            distance_min: grid_distance_min,
            distance_max: grid_distance_max,
            distance_steps: grid_distance_steps,
            rot_max: grid_rot_max_deg.to_radians(),
            rot_steps: grid_rot_steps,
            poni_search_radius_px: grid_poni_search_radius_px,
            poni_grid_steps: grid_poni_steps,
            tolerance_rad: grid_tolerance_rad,
            max_peaks_for_grid: grid_max_peaks,
            max_ring_index: grid_max_ring_index,
            top_k: grid_top_k,
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
        bootstrap_ring_center_d1: diag.bootstrap_ring_center_d1,
        bootstrap_ring_center_d2: diag.bootstrap_ring_center_d2,
    })
}

/// End-to-end automatic calibration directly from a 2D diffraction image with
/// zero manual clicks.
///
/// # Arguments
///
/// * `image`             — 2D `float64` numpy array (same orientation as
///                         `Frame::from_tiff`, i.e. row 0 at bottom / `data[::-1]`
///                         from fabio)
/// * `calibrant`         — reference powder (`Calibrant`)
/// * `wavelength`        — X-ray wavelength in metres
/// * `pixel1`            — pixel pitch along axis 0 (rows) in metres
/// * `pixel2`            — pixel pitch along axis 1 (cols) in metres (defaults to `pixel1`)
/// * `starting_distance` — optional rough detector distance in metres (speeds up
///                         and focuses search; `None` performs a blind search)
/// * `distance_tolerance_factor` — search margin `[d0 / factor, d0 * factor]` when
///                         `starting_distance` is given (default `2.0`)
/// * `distance_min`      — lower blind-search distance bound in metres (default `0.04`)
/// * `distance_max`      — upper blind-search distance bound in metres (default `1.5`)
/// * `fix`               — optional list of parameter names to hold fixed (defaults to
///                         `["wavelength", "rot3"]`)
/// * `enable_tilt_fallback` — fall back to 5D grid search for heavily tilted or strip
///                         detectors (default `True`)
/// * `max_iter`          — maximum L-M iterations per refinement pass (default `200`)
/// * `tol`               — L-M convergence tolerance (default `1e-9`)
/// * `intensity_limit`   — optional upper intensity cutoff; when `None` (default), auto-detects
///                         flat-topped saturation plateaus (`>= 65_000`)
/// * `peak_threshold_sigma` — optional MAD peak threshold multiplier; when `None` (default),
///                         starts at `3.5σ` and relaxes automatically on faint images
/// * `search_half_width_px` — optional 2D ring-profile search half-width (pixels); when `None`
///                         (default), adapts to the 1D ring width and local inter-ring spacing
#[pyfunction]
#[pyo3(name = "auto_calibrate_image")]
#[pyo3(signature = (
    image, calibrant, wavelength, pixel1, pixel2=None,
    starting_distance=None,
    mask=None,
    distance_tolerance_factor=2.0,
    distance_min=0.04,
    distance_max=1.5,
    fix=None,
    enable_tilt_fallback=true,
    max_ring_index=100,
    max_iter=200,
    tol=1e-9,
    intensity_limit=None,
    peak_threshold_sigma=None,
    search_half_width_px=None,
    polarization_factor=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn auto_calibrate_image_py(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    calibrant: &PyCalibrant,
    wavelength: f64,
    pixel1: f64,
    pixel2: Option<f64>,
    starting_distance: Option<f64>,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    distance_tolerance_factor: f64,
    distance_min: f64,
    distance_max: f64,
    fix: Option<&Bound<'_, PyList>>,
    enable_tilt_fallback: bool,
    max_ring_index: usize,
    max_iter: usize,
    tol: f64,
    intensity_limit: Option<f64>,
    peak_threshold_sigma: Option<f64>,
    search_half_width_px: Option<usize>,
    polarization_factor: Option<f64>,
) -> PyResult<PyAutoCalibrateResult> {
    let view = image.as_array();
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];
    let pixel2 = pixel2.unwrap_or(pixel1);

    let refine_mask = match fix {
        Some(list) => parse_mask(Some(list))?,
        None => AutoCalibrateImageOptions::default().refine_mask,
    };

    let opts = AutoCalibrateImageOptions {
        starting_distance,
        distance_tolerance_factor,
        distance_min,
        distance_max,
        refine_mask,
        enable_tilt_fallback,
        max_ring_index,
        intensity_limit,
        peak_threshold_sigma,
        search_half_width_px,
        polarization_factor,
        detector: None,
        max_iter,
        tol,
    };

    // Always materialize in row-major (C) order before detaching from the GIL.
    // Note: rust-numpy's `as_slice()` succeeds for Fortran-contiguous arrays too,
    // so checking `is_standard_layout()` avoids reading F-ordered arrays transposed.
    let img_buf: Vec<f64> = if view.is_standard_layout() {
        image
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|_| view.iter().copied().collect())
    } else {
        view.iter().copied().collect()
    };

    let mask_buf: Option<Vec<bool>> = match mask {
        Some(m) => {
            let m_view = m.as_array();
            let m_shape = m_view.shape();
            if m_shape != shape {
                return Err(PyValueError::new_err(format!(
                    "mask shape ({}, {}) does not match image shape ({}, {})",
                    m_shape[0], m_shape[1], dim1, dim2
                )));
            }
            Some(if m_view.is_standard_layout() {
                m.as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|_| m_view.iter().copied().collect())
            } else {
                m_view.iter().copied().collect()
            })
        }
        None => None,
    };

    let cal = calibrant.inner.clone();
    let (refined, diag) = py
        .detach(|| {
            auto_calibrate_image_with_mask(
                &img_buf,
                dim1,
                dim2,
                mask_buf.as_deref(),
                &cal,
                wavelength,
                pixel1,
                pixel2,
                &opts,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyAutoCalibrateResult {
        refined,
        n_detected_ellipses: diag.n_detected_ellipses,
        n_identified_ellipses: diag.n_identified_ellipses,
        bootstrap_distance: diag.bootstrap_distance,
        bootstrap_ring_center_d1: diag.bootstrap_ring_center_d1,
        bootstrap_ring_center_d2: diag.bootstrap_ring_center_d2,
    })
}

// ---------- ring-center estimation ----------

/// Estimate the **ring center** from a list of peak coordinates via 3-point
/// circle RANSAC + radial-band consistency scoring.
///
/// IMPORTANT: "ring center" is *not* the same as **PONI** (point of normal
/// incidence) or **beam center** on a tilted detector. The three coincide
/// only for a perpendicular detector. For tilted detectors each ring's
/// projected ellipse has its own geometric center (depending on the ring's
/// 2θ); this function returns a consensus from circle fits that's a useful
/// starting guess only when rings are approximately circular (rot1, rot2
/// at most a few degrees).
///
/// Robust to partial arcs and off-detector beams as long as the
/// near-circular assumption holds. For heavily tilted detectors a
/// tilt-aware ellipse-based center estimator is required (planned).
///
/// Returns `(d1, d2)` or `None` if too few peaks / no consensus.
///
/// # Arguments
///
/// * `peaks`           — list of `(d1, d2)` peak coordinates
/// * `max_trials`      — number of 3-point RANSAC samples (default 500)
/// * `radial_bin_width`— width (px) of radial histogram bins for scoring
/// * `top_k_bins`      — score = sum of the top-K bin counts
/// * `max_radius_px`   — radii beyond this are ignored
/// * `rng_seed`        — for reproducibility
#[pyfunction]
#[pyo3(name = "estimate_ring_center")]
#[pyo3(signature = (peaks, max_trials=500, radial_bin_width=4.0, top_k_bins=10, max_radius_px=5000.0, rng_seed=0xBEAC_0CEF_F00D_F00D_u64))]
pub fn estimate_ring_center_py(
    peaks: Vec<(f64, f64)>,
    max_trials: usize,
    radial_bin_width: f64,
    top_k_bins: usize,
    max_radius_px: f64,
    rng_seed: u64,
) -> Option<(f64, f64)> {
    let opts = RingCenterOptions {
        max_trials,
        radial_bin_width_px: radial_bin_width,
        top_k_bins,
        max_radius_px,
        rng_seed,
    };
    estimate_ring_center(&peaks, &opts).map(|c| (c.d1, c.d2))
}

// ---------- polarization-based rot3 calibration ----------

#[pyclass(name = "PolarizationRot3Result")]
pub struct PyPolarizationRot3Result {
    inner: PolarizationRot3Result,
}

#[pymethods]
impl PyPolarizationRot3Result {
    #[getter]
    fn rot3(&self) -> f64 {
        self.inner.rot3_rad
    }
    #[getter]
    fn uncertainty_rad(&self) -> f64 {
        self.inner.uncertainty_rad
    }
    #[getter]
    fn n_rings_used(&self) -> usize {
        self.inner.n_rings_used
    }
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }
    #[getter]
    fn modulation_snr(&self) -> f64 {
        self.inner.modulation_snr
    }
    #[getter]
    fn fitted_polarization_ratio(&self) -> f64 {
        self.inner.fitted_polarization_ratio
    }
    #[getter]
    fn reliable(&self) -> bool {
        self.inner.reliable
    }
    #[getter]
    fn source_used(&self) -> &'static str {
        match self.inner.source_used {
            PolarizationRot3Source::Rings => "rings",
            PolarizationRot3Source::Diffuse => "diffuse",
            PolarizationRot3Source::Auto => unreachable!("source_used is always resolved"),
        }
    }
    #[getter]
    fn poni<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        poni_to_py_dict(py, &self.inner.poni)
    }
    fn __repr__(&self) -> String {
        format!(
            "PolarizationRot3Result(rot3={:.4} rad [{:.2}°], uncertainty={:.2}°, snr={:.1}, rings={}, source={:?})",
            self.inner.rot3_rad,
            self.inner.rot3_rad.to_degrees(),
            self.inner.uncertainty_rad.to_degrees(),
            self.inner.modulation_snr,
            self.inner.n_rings_used,
            self.source_used()
        )
    }
}

/// Calibrate `rot3` from the azimuthal intensity modulation of powder rings
/// and/or between-rings diffuse scattering under a known linear X-ray
/// polarization factor `polarization_factor`.
///
/// Works after **any** calibration workflow — manual peak picking + `refine`,
/// `auto_calibrate`, `auto_calibrate_image`, or a loaded `.poni` file.
#[pyfunction]
#[pyo3(name = "calibrate_rot3_polarization")]
#[pyo3(signature = (
    image,
    poni,
    calibrant,
    polarization_factor=0.99,
    mask=None,
    n_az_bins=72,
    min_tth_deg=5.0,
    max_rings=32,
    ring_half_width_rad=None,
    min_samples_per_ring=12,
    min_total_samples=24,
    source="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn calibrate_rot3_polarization_py(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    poni: &Bound<'_, PyAny>,
    calibrant: &PyCalibrant,
    polarization_factor: f64,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    n_az_bins: usize,
    min_tth_deg: f64,
    max_rings: usize,
    ring_half_width_rad: Option<f64>,
    min_samples_per_ring: usize,
    min_total_samples: usize,
    source: &str,
) -> PyResult<PyPolarizationRot3Result> {
    let view = image.as_array();
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];

    let (poni_val, detector) = if let Ok(integ) = poni.extract::<PyRef<PyIntegrator>>() {
        let geom = integ.inner().geometry();
        if geom.dim1() != dim1 || geom.dim2() != dim2 {
            return Err(PyValueError::new_err(format!(
                "image shape ({dim1}, {dim2}) does not match Integrator shape ({}, {})",
                geom.dim1(),
                geom.dim2()
            )));
        }
        (geom.poni.clone(), geom.detector().cloned())
    } else if let Ok(dict) = poni.cast::<PyDict>() {
        (poni_from_dict(dict)?, None)
    } else {
        return Err(PyValueError::new_err(
            "poni must be a dioptrin.Integrator or a PONI dict",
        ));
    };
    let img_buf: Vec<f64> = if view.is_standard_layout() {
        image
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|_| view.iter().copied().collect())
    } else {
        view.iter().copied().collect()
    };

    let mask_buf: Option<Vec<bool>> = match mask {
        Some(m) => {
            let m_view = m.as_array();
            let m_shape = m_view.shape();
            if m_shape != shape {
                return Err(PyValueError::new_err(format!(
                    "mask shape ({}, {}) does not match image shape ({}, {})",
                    m_shape[0], m_shape[1], dim1, dim2
                )));
            }
            Some(if m_view.is_standard_layout() {
                m.as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|_| m_view.iter().copied().collect())
            } else {
                m_view.iter().copied().collect()
            })
        }
        None => None,
    };

    let source_enum = match source.to_ascii_lowercase().as_str() {
        "auto" => PolarizationRot3Source::Auto,
        "rings" => PolarizationRot3Source::Rings,
        "diffuse" => PolarizationRot3Source::Diffuse,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid source {source:?}; expected 'auto', 'rings', or 'diffuse'"
            )));
        }
    };
    let opts = PolarizationRot3Options {
        source: source_enum,
        n_az_bins,
        min_tth_rad: min_tth_deg.to_radians(),
        max_rings,
        ring_half_width_rad,
        min_samples_per_ring,
        min_total_samples,
        detector,
    };
    let cal = calibrant.inner.clone();
    let res = py
        .detach(|| {
            calibrate_rot3_polarization(
                &img_buf,
                dim1,
                dim2,
                mask_buf.as_deref(),
                &poni_val,
                &cal,
                polarization_factor,
                &opts,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyPolarizationRot3Result { inner: res })
}

// ---------- image-based ring peak extraction & refinement ----------

#[pyclass(name = "RefineImageResult")]
pub struct PyRefineImageResult {
    inner: RefineImageResult,
}

#[pymethods]
impl PyRefineImageResult {
    #[getter]
    fn n_iter(&self) -> usize {
        self.inner.calibration.n_iter
    }
    #[getter]
    fn chi2(&self) -> f64 {
        self.inner.calibration.chi2
    }
    #[getter]
    fn rms_rad(&self) -> f64 {
        self.inner.calibration.rms_rad
    }
    #[getter]
    fn n_rings(&self) -> usize {
        self.inner.n_rings
    }
    #[getter]
    fn start_rms_rad(&self) -> Option<f64> {
        self.inner.start_rms_rad
    }
    #[getter]
    fn assignments(&self) -> Vec<Option<usize>> {
        self.inner.calibration.assignments.clone()
    }
    #[getter]
    fn inliers(&self) -> Vec<bool> {
        self.inner.calibration.inliers.clone()
    }
    #[getter]
    fn points(&self) -> PyCalibrationPoints {
        PyCalibrationPoints {
            inner: self.inner.points.clone(),
        }
    }
    #[getter]
    fn rot3_calibration(&self) -> Option<PyPolarizationRot3Result> {
        self.inner
            .rot3_calibration
            .clone()
            .map(|r| PyPolarizationRot3Result { inner: r })
    }
    #[getter]
    fn poni<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let p = &self.inner.calibration.poni;
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
            "RefineImageResult(n_rings={}, n_points={}, rms_rad={:e})",
            self.inner.n_rings,
            self.inner.points.len(),
            self.inner.calibration.rms_rad
        )
    }
}

/// Extract sub-pixel ring control points uniformly around all visible calibrant
/// rings predicted by `poni`, using the mask-aware DoG residual and `8×8`
/// local-MAD noise model (zero `intensity_min_factor` tuning).
#[pyfunction]
#[pyo3(name = "extract_ring_peaks")]
#[pyo3(signature = (
    image,
    poni,
    calibrant,
    mask=None,
    max_ring_index=100,
    max_rings=None,
    n_azimuth=72,
    intensity_limit=None,
    search_half_width_px=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn extract_ring_peaks_py(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    poni: &Bound<'_, PyAny>,
    calibrant: &PyCalibrant,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    max_ring_index: usize,
    max_rings: Option<usize>,
    n_azimuth: usize,
    intensity_limit: Option<f64>,
    search_half_width_px: Option<usize>,
) -> PyResult<PyCalibrationPoints> {
    let view = image.as_array();
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];

    let (poni_val, detector) = if let Ok(integ) = poni.extract::<PyRef<PyIntegrator>>() {
        let geom = integ.inner().geometry();
        if geom.dim1() != dim1 || geom.dim2() != dim2 {
            return Err(PyValueError::new_err(format!(
                "image shape ({dim1}, {dim2}) does not match Integrator shape ({}, {})",
                geom.dim1(),
                geom.dim2()
            )));
        }
        (geom.poni.clone(), geom.detector().cloned())
    } else if let Ok(dict) = poni.cast::<PyDict>() {
        (poni_from_dict(dict)?, None)
    } else {
        return Err(PyValueError::new_err(
            "poni must be a dioptrin.Integrator or a PONI dict",
        ));
    };

    let img_buf: Vec<f64> = if view.is_standard_layout() {
        image
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|_| view.iter().copied().collect())
    } else {
        view.iter().copied().collect()
    };

    let mask_buf: Option<Vec<bool>> = match mask {
        Some(m) => {
            let m_view = m.as_array();
            let m_shape = m_view.shape();
            if m_shape != shape {
                return Err(PyValueError::new_err(format!(
                    "mask shape ({}, {}) does not match image shape ({}, {})",
                    m_shape[0], m_shape[1], dim1, dim2
                )));
            }
            Some(if m_view.is_standard_layout() {
                m.as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|_| m_view.iter().copied().collect())
            } else {
                m_view.iter().copied().collect()
            })
        }
        None => None,
    };

    let opts = RefineImageOptions {
        max_ring_index,
        max_rings,
        n_azimuth,
        intensity_limit,
        search_half_width_px,
        detector,
        ..RefineImageOptions::default()
    };
    let cal = calibrant.inner.clone();
    let pts = py
        .detach(|| {
            extract_ring_peaks(
                &img_buf,
                dim1,
                dim2,
                mask_buf.as_deref(),
                &poni_val,
                &cal,
                &opts,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyCalibrationPoints { inner: pts })
}

/// Refine an existing `start` PONI directly against a 2D diffraction image
/// using iterative local-MAD 2D ring-profile extraction + Huber L-M + optional
/// polarization-based `rot3` calibration.
#[pyfunction]
#[pyo3(name = "refine_image")]
#[pyo3(signature = (
    image,
    start,
    calibrant,
    mask=None,
    fix=None,
    max_ring_index=100,
    max_rings=None,
    n_azimuth=72,
    n_passes=3,
    intensity_limit=None,
    search_half_width_px=None,
    polarization_factor=None,
    max_iter=200,
    tol=1e-9,
))]
#[allow(clippy::too_many_arguments)]
pub fn refine_image_py(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    start: &Bound<'_, PyAny>,
    calibrant: &PyCalibrant,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    fix: Option<&Bound<'_, PyList>>,
    max_ring_index: usize,
    max_rings: Option<usize>,
    n_azimuth: usize,
    n_passes: usize,
    intensity_limit: Option<f64>,
    search_half_width_px: Option<usize>,
    polarization_factor: Option<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyRefineImageResult> {
    let view = image.as_array();
    let shape = view.shape();
    let dim1 = shape[0];
    let dim2 = shape[1];

    let (poni_val, detector) = if let Ok(integ) = start.extract::<PyRef<PyIntegrator>>() {
        let geom = integ.inner().geometry();
        if geom.dim1() != dim1 || geom.dim2() != dim2 {
            return Err(PyValueError::new_err(format!(
                "image shape ({dim1}, {dim2}) does not match Integrator shape ({}, {})",
                geom.dim1(),
                geom.dim2()
            )));
        }
        (geom.poni.clone(), geom.detector().cloned())
    } else if let Ok(dict) = start.cast::<PyDict>() {
        (poni_from_dict(dict)?, None)
    } else {
        return Err(PyValueError::new_err(
            "start must be a dioptrin.Integrator or a PONI dict",
        ));
    };

    let refine_mask = match fix {
        Some(list) => parse_mask(Some(list))?,
        None => RefineImageOptions::default().refine_mask,
    };

    let img_buf: Vec<f64> = if view.is_standard_layout() {
        image
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|_| view.iter().copied().collect())
    } else {
        view.iter().copied().collect()
    };

    let mask_buf: Option<Vec<bool>> = match mask {
        Some(m) => {
            let m_view = m.as_array();
            let m_shape = m_view.shape();
            if m_shape != shape {
                return Err(PyValueError::new_err(format!(
                    "mask shape ({}, {}) does not match image shape ({}, {})",
                    m_shape[0], m_shape[1], dim1, dim2
                )));
            }
            Some(if m_view.is_standard_layout() {
                m.as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|_| m_view.iter().copied().collect())
            } else {
                m_view.iter().copied().collect()
            })
        }
        None => None,
    };

    let opts = RefineImageOptions {
        refine_mask,
        max_ring_index,
        max_rings,
        n_azimuth,
        intensity_limit,
        search_half_width_px,
        n_passes,
        polarization_factor,
        detector,
        max_iter,
        tol,
    };
    let cal = calibrant.inner.clone();
    let res = py
        .detach(|| {
            refine_image(
                &img_buf,
                dim1,
                dim2,
                mask_buf.as_deref(),
                &poni_val,
                &cal,
                &opts,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyRefineImageResult { inner: res })
}

// ---------- Spatial distortion estimation ----------

#[pyclass(name = "DistortionResult")]
pub struct PyDistortionResult {
    inner: DistortionResult,
    dim1: usize,
    dim2: usize,
}

#[pymethods]
impl PyDistortionResult {
    #[getter]
    fn poni<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        poni_to_py_dict(py, &self.inner.poni)
    }
    #[getter]
    fn ponis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in &self.inner.ponis {
            list.append(poni_to_py_dict(py, p)?)?;
        }
        Ok(list)
    }
    #[getter]
    fn rms_before_rad(&self) -> f64 {
        self.inner.rms_before_rad
    }
    #[getter]
    fn rms_after_rad(&self) -> f64 {
        self.inner.rms_after_rad
    }
    #[getter]
    fn rms_before_px(&self) -> f64 {
        self.inner.rms_before_px
    }
    #[getter]
    fn rms_after_px(&self) -> f64 {
        self.inner.rms_after_px
    }
    #[getter]
    fn cv_rms_px(&self) -> f64 {
        self.inner.cv_rms_px
    }
    #[getter]
    fn significant(&self) -> bool {
        self.inner.significant
    }
    #[getter]
    fn max_displacement_px(&self) -> f64 {
        self.inner.max_displacement_px
    }
    #[getter]
    fn n_points_used(&self) -> usize {
        self.inner.n_points_used
    }
    #[getter]
    fn point_residuals_before_px(&self) -> Vec<f64> {
        self.inner.point_residuals_before_px.clone()
    }
    #[getter]
    fn point_residuals_after_px(&self) -> Vec<f64> {
        self.inner.point_residuals_after_px.clone()
    }
    #[getter]
    fn module_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for m in &self.inner.module_offsets {
            let d = PyDict::new(py);
            d.set_item("row_index", m.row_index)?;
            d.set_item("col_index", m.col_index)?;
            d.set_item("dd1_px", m.dd1_px)?;
            d.set_item("dd2_px", m.dd2_px)?;
            d.set_item("dz_m", m.dz_m)?;
            d.set_item("n_points", m.n_points)?;
            list.append(d)?;
        }
        Ok(list)
    }
    /// Evaluate in-plane pixel displacement `(dd1_px, dd2_px)` at `(d1, d2)`.
    fn displacement_px(&self, d1: f64, d2: f64) -> (f64, f64) {
        self.inner.field.displacement_px(d1, d2)
    }
    /// Evaluate out-of-plane displacement `dz` in metres at `(d1, d2)`.
    fn dz_m(&self, d1: f64, d2: f64) -> f64 {
        self.inner.field.dz_m(d1, d2)
    }
    /// Export the fitted smooth in-plane distortion as a Fit2D / pyFAI `.spline` string.
    fn to_fit2d_spline(&self) -> PyResult<String> {
        let sp = self
            .inner
            .field
            .to_spline(
                self.dim1,
                self.dim2,
                self.inner.poni.pixel1,
                self.inner.poni.pixel2,
            )
            .map_err(PyValueError::new_err)?;
        Ok(sp.content)
    }
    /// Evaluate `(dd1_px, dd2_px)` on the full `(dim1, dim2)` pixel grid.
    fn displacement_grid<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
    ) {
        let mut d1_arr = numpy::ndarray::Array2::<f64>::zeros((self.dim1, self.dim2));
        let mut d2_arr = numpy::ndarray::Array2::<f64>::zeros((self.dim1, self.dim2));
        for r in 0..self.dim1 {
            for c in 0..self.dim2 {
                let (a, b) = self.inner.field.displacement_px(r as f64, c as f64);
                d1_arr[[r, c]] = a;
                d2_arr[[r, c]] = b;
            }
        }
        (
            numpy::PyArray2::from_owned_array(py, d1_arr),
            numpy::PyArray2::from_owned_array(py, d2_arr),
        )
    }
    /// Evaluate out-of-plane `dz` (metres) on the full `(dim1, dim2)` pixel grid.
    fn dz_grid<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        let mut dz_arr = numpy::ndarray::Array2::<f64>::zeros((self.dim1, self.dim2));
        for r in 0..self.dim1 {
            for c in 0..self.dim2 {
                dz_arr[[r, c]] = self.inner.field.dz_m(r as f64, c as f64);
            }
        }
        numpy::PyArray2::from_owned_array(py, dz_arr)
    }
    fn __repr__(&self) -> String {
        format!(
            "DistortionResult(rms_px={:.3}->{:.3}, cv_rms_px={:.3}, max_disp={:.2}px, significant={}, n_pts={})",
            self.inner.rms_before_px,
            self.inner.rms_after_px,
            self.inner.cv_rms_px,
            self.inner.max_displacement_px,
            self.inner.significant,
            self.inner.n_points_used
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_distortion_model(
    model: &str,
    max_order: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: Option<f64>,
    max_degree: usize,
    module_height: Option<usize>,
    module_width: Option<usize>,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
) -> PyResult<DistortionModel> {
    match model.to_ascii_lowercase().as_str() {
        "radial_optical" => Ok(DistortionModel::RadialOptical { max_order }),
        "smooth_spline" => Ok(DistortionModel::SmoothSpline {
            grid_nx,
            grid_ny,
            smoothness: smoothness.unwrap_or(1.0),
        }),
        "bow_z" => Ok(DistortionModel::BowZ {
            max_degree,
            smoothness: smoothness.unwrap_or(1e-3),
        }),
        "per_module" => {
            let mh = module_height
                .ok_or_else(|| PyValueError::new_err("per_module model requires module_height"))?;
            let mw = module_width
                .ok_or_else(|| PyValueError::new_err("per_module model requires module_width"))?;
            Ok(DistortionModel::PerModule {
                module_height: mh,
                module_width: mw,
                gap_height,
                gap_width,
                fit_z,
                regularization,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "invalid model {model:?}; expected 'smooth_spline', 'radial_optical', 'bow_z', or 'per_module'"
        ))),
    }
}

fn parse_distortion_mask(fix: Option<&Bound<'_, PyList>>) -> PyResult<ParamMask> {
    let mut mask = DistortionOptions::default().refine_mask;
    if let Some(list) = fix {
        let user = parse_mask(Some(list))?;
        mask.distance &= user.distance;
        mask.poni1 &= user.poni1;
        mask.poni2 &= user.poni2;
        mask.rot1 &= user.rot1;
        mask.rot2 &= user.rot2;
        mask.rot3 = false;
        mask.wavelength &= user.wavelength;
    }
    Ok(mask)
}

/// Estimate spatial detector distortion from calibration control points and a
/// converged PONI.
#[pyfunction]
#[pyo3(name = "estimate_distortion")]
#[pyo3(signature = (
    points,
    shape,
    poni,
    calibrant,
    model="smooth_spline",
    max_order=5,
    grid_nx=8,
    grid_ny=8,
    smoothness=None,
    max_degree=3,
    module_height=None,
    module_width=None,
    gap_height=0,
    gap_width=0,
    fit_z=false,
    regularization=1e-2,
    huber_c_px=1.5,
    refine_poni=true,
    fix=None,
    cv_folds=5,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_distortion_py(
    py: Python<'_>,
    points: &PyCalibrationPoints,
    shape: (usize, usize),
    poni: &Bound<'_, PyAny>,
    calibrant: &PyCalibrant,
    model: &str,
    max_order: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: Option<f64>,
    max_degree: usize,
    module_height: Option<usize>,
    module_width: Option<usize>,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
    huber_c_px: f64,
    refine_poni: bool,
    fix: Option<&Bound<'_, PyList>>,
    cv_folds: usize,
) -> PyResult<PyDistortionResult> {
    let (dim1, dim2) = shape;
    let poni_val: Poni = if let Ok(integ) = poni.extract::<PyRef<PyIntegrator>>() {
        integ.inner().geometry().poni.clone()
    } else if let Ok(dict) = poni.cast::<PyDict>() {
        poni_from_dict(dict)?
    } else {
        return Err(PyValueError::new_err(
            "poni must be a dioptrin.Integrator or a PONI dict",
        ));
    };

    let model_enum = parse_distortion_model(
        model,
        max_order,
        grid_nx,
        grid_ny,
        smoothness,
        max_degree,
        module_height,
        module_width,
        gap_height,
        gap_width,
        fit_z,
        regularization,
    )?;
    let refine_mask = parse_distortion_mask(fix)?;

    let opts = DistortionOptions {
        model: model_enum,
        huber_c_px,
        refine_poni,
        refine_mask,
        cv_folds,
    };
    let pts = points.inner.clone();
    let cal = calibrant.inner.clone();
    let res = py
        .detach(|| estimate_distortion(&pts, dim1, dim2, &poni_val, &cal, &opts))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyDistortionResult {
        inner: res,
        dim1,
        dim2,
    })
}

fn poni_to_py_dict<'py>(py: Python<'py>, p: &Poni) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
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

/// Jointly estimate a shared spatial detector distortion field from multiple
/// calibration exposures `(points, poni, calibrant)` measured at different
/// detector positions, distances, tilts, or wavelengths.
#[pyfunction]
#[pyo3(name = "estimate_distortion_multi")]
#[pyo3(signature = (
    observations,
    shape,
    model="smooth_spline",
    max_order=5,
    grid_nx=8,
    grid_ny=8,
    smoothness=None,
    max_degree=3,
    module_height=None,
    module_width=None,
    gap_height=0,
    gap_width=0,
    fit_z=false,
    regularization=1e-2,
    huber_c_px=1.5,
    refine_poni=true,
    fix=None,
    cv_folds=5,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_distortion_multi_py(
    py: Python<'_>,
    observations: &Bound<'_, PyList>,
    shape: (usize, usize),
    model: &str,
    max_order: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: Option<f64>,
    max_degree: usize,
    module_height: Option<usize>,
    module_width: Option<usize>,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
    huber_c_px: f64,
    refine_poni: bool,
    fix: Option<&Bound<'_, PyList>>,
    cv_folds: usize,
) -> PyResult<PyDistortionResult> {
    let (dim1, dim2) = shape;
    let mut owned: Vec<(CalibrationPoints, Poni, Calibrant)> =
        Vec::with_capacity(observations.len());
    for item in observations.iter() {
        let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("each observation must be a (points, poni, calibrant) 3-tuple")
        })?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "each observation must be a (points, poni, calibrant) 3-tuple",
            ));
        }
        let pts_ref: PyRef<PyCalibrationPoints> = tuple.get_item(0)?.extract()?;
        let poni_any = tuple.get_item(1)?;
        let poni_val: Poni = if let Ok(integ) = poni_any.extract::<PyRef<PyIntegrator>>() {
            integ.inner().geometry().poni.clone()
        } else if let Ok(dict) = poni_any.cast::<PyDict>() {
            poni_from_dict(dict)?
        } else {
            return Err(PyValueError::new_err(
                "poni must be a dioptrin.Integrator or a PONI dict",
            ));
        };
        let cal_ref: PyRef<PyCalibrant> = tuple.get_item(2)?.extract()?;
        owned.push((pts_ref.inner.clone(), poni_val, cal_ref.inner.clone()));
    }

    let model_enum = parse_distortion_model(
        model,
        max_order,
        grid_nx,
        grid_ny,
        smoothness,
        max_degree,
        module_height,
        module_width,
        gap_height,
        gap_width,
        fit_z,
        regularization,
    )?;
    let refine_mask = parse_distortion_mask(fix)?;
    let opts = DistortionOptions {
        model: model_enum,
        huber_c_px,
        refine_poni,
        refine_mask,
        cv_folds,
    };
    let res = py
        .detach(|| {
            let refs: Vec<DistortionObservation<'_>> = owned
                .iter()
                .map(|(pts, p, cal)| DistortionObservation {
                    points: pts,
                    poni: p,
                    calibrant: cal,
                })
                .collect();
            estimate_distortion_multi(&refs, dim1, dim2, &opts)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyDistortionResult {
        inner: res,
        dim1,
        dim2,
    })
}

#[pyclass(name = "DistortionImageResult")]
pub struct PyDistortionImageResult {
    inner: DistortionImageResult,
    dim1: usize,
    dim2: usize,
}

#[pymethods]
impl PyDistortionImageResult {
    #[getter]
    fn distortion(&self) -> PyDistortionResult {
        PyDistortionResult {
            inner: self.inner.distortion.clone(),
            dim1: self.dim1,
            dim2: self.dim2,
        }
    }
    #[getter]
    fn points(&self) -> Vec<PyCalibrationPoints> {
        self.inner
            .points
            .iter()
            .cloned()
            .map(|p| PyCalibrationPoints { inner: p })
            .collect()
    }
    #[getter]
    fn n_rings(&self) -> Vec<usize> {
        self.inner.n_rings.clone()
    }
    #[getter]
    fn initial_rms_px(&self) -> f64 {
        self.inner.initial_rms_px
    }
    #[getter]
    fn n_passes_completed(&self) -> usize {
        self.inner.n_passes_completed
    }
    fn __repr__(&self) -> String {
        format!(
            "DistortionImageResult(passes={}, rings={:?}, rms_px={:.3}->{:.3}, cv_rms_px={:.3}, significant={})",
            self.inner.n_passes_completed,
            self.inner.n_rings,
            self.inner.initial_rms_px,
            self.inner.distortion.rms_after_px,
            self.inner.distortion.cv_rms_px,
            self.inner.distortion.significant,
        )
    }
}

/// Iteratively extract ring peaks from a 2D image along distorted trajectories
/// and estimate spatial detector distortion.
#[pyfunction]
#[pyo3(name = "estimate_distortion_image")]
#[pyo3(signature = (
    image,
    poni,
    calibrant,
    mask=None,
    model="smooth_spline",
    max_order=5,
    grid_nx=8,
    grid_ny=8,
    smoothness=None,
    max_degree=3,
    module_height=None,
    module_width=None,
    gap_height=0,
    gap_width=0,
    fit_z=false,
    regularization=1e-2,
    huber_c_px=1.5,
    refine_poni=true,
    fix=None,
    cv_folds=5,
    n_passes=3,
    max_ring_index=100,
    max_rings=None,
    n_azimuth=72,
    intensity_limit=None,
    search_half_width_px=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_distortion_image_py(
    py: Python<'_>,
    image: PyReadonlyArray2<'_, f64>,
    poni: &Bound<'_, PyAny>,
    calibrant: &PyCalibrant,
    mask: Option<PyReadonlyArray2<'_, bool>>,
    model: &str,
    max_order: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: Option<f64>,
    max_degree: usize,
    module_height: Option<usize>,
    module_width: Option<usize>,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
    huber_c_px: f64,
    refine_poni: bool,
    fix: Option<&Bound<'_, PyList>>,
    cv_folds: usize,
    n_passes: usize,
    max_ring_index: usize,
    max_rings: Option<usize>,
    n_azimuth: usize,
    intensity_limit: Option<f64>,
    search_half_width_px: Option<usize>,
) -> PyResult<PyDistortionImageResult> {
    let view = image.as_array();
    let shape = view.shape();
    let (dim1, dim2) = (shape[0], shape[1]);

    let poni_val: Poni = if let Ok(integ) = poni.extract::<PyRef<PyIntegrator>>() {
        integ.inner().geometry().poni.clone()
    } else if let Ok(dict) = poni.cast::<PyDict>() {
        poni_from_dict(dict)?
    } else {
        return Err(PyValueError::new_err(
            "poni must be a dioptrin.Integrator or a PONI dict",
        ));
    };

    let img_buf: Vec<f64> = if view.is_standard_layout() {
        image
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|_| view.iter().copied().collect())
    } else {
        view.iter().copied().collect()
    };
    let mask_buf: Option<Vec<bool>> = match mask {
        Some(m) => {
            let mv = m.as_array();
            if mv.shape() != shape {
                return Err(PyValueError::new_err(format!(
                    "mask shape ({}, {}) does not match image shape ({dim1}, {dim2})",
                    mv.shape()[0],
                    mv.shape()[1]
                )));
            }
            Some(if mv.is_standard_layout() {
                m.as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|_| mv.iter().copied().collect())
            } else {
                mv.iter().copied().collect()
            })
        }
        None => None,
    };

    let model_enum = parse_distortion_model(
        model,
        max_order,
        grid_nx,
        grid_ny,
        smoothness,
        max_degree,
        module_height,
        module_width,
        gap_height,
        gap_width,
        fit_z,
        regularization,
    )?;
    let refine_mask = parse_distortion_mask(fix)?;
    let opts = DistortionImageOptions {
        distortion: DistortionOptions {
            model: model_enum,
            huber_c_px,
            refine_poni,
            refine_mask,
            cv_folds,
        },
        n_passes,
        max_ring_index,
        max_rings,
        n_azimuth,
        intensity_limit,
        search_half_width_px,
    };
    let cal = calibrant.inner.clone();
    let res = py
        .detach(|| {
            estimate_distortion_image(
                &img_buf,
                dim1,
                dim2,
                mask_buf.as_deref(),
                &poni_val,
                &cal,
                &opts,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyDistortionImageResult {
        inner: res,
        dim1,
        dim2,
    })
}

/// Multi-exposure iterative image-based distortion estimation. Each entry in
/// `observations` is a `(image, poni, calibrant)` or `(image, poni, calibrant, mask)` tuple.
#[pyfunction]
#[pyo3(name = "estimate_distortion_image_multi")]
#[pyo3(signature = (
    observations,
    model="smooth_spline",
    max_order=5,
    grid_nx=8,
    grid_ny=8,
    smoothness=None,
    max_degree=3,
    module_height=None,
    module_width=None,
    gap_height=0,
    gap_width=0,
    fit_z=false,
    regularization=1e-2,
    huber_c_px=1.5,
    refine_poni=true,
    fix=None,
    cv_folds=5,
    n_passes=3,
    max_ring_index=100,
    max_rings=None,
    n_azimuth=72,
    intensity_limit=None,
    search_half_width_px=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_distortion_image_multi_py(
    py: Python<'_>,
    observations: &Bound<'_, PyList>,
    model: &str,
    max_order: usize,
    grid_nx: usize,
    grid_ny: usize,
    smoothness: Option<f64>,
    max_degree: usize,
    module_height: Option<usize>,
    module_width: Option<usize>,
    gap_height: usize,
    gap_width: usize,
    fit_z: bool,
    regularization: f64,
    huber_c_px: f64,
    refine_poni: bool,
    fix: Option<&Bound<'_, PyList>>,
    cv_folds: usize,
    n_passes: usize,
    max_ring_index: usize,
    max_rings: Option<usize>,
    n_azimuth: usize,
    intensity_limit: Option<f64>,
    search_half_width_px: Option<usize>,
) -> PyResult<PyDistortionImageResult> {
    if observations.is_empty() {
        return Err(PyValueError::new_err("no image observations provided"));
    }
    struct OwnedImageObs {
        image: Vec<f64>,
        dim1: usize,
        dim2: usize,
        mask: Option<Vec<bool>>,
        poni: Poni,
        calibrant: Calibrant,
    }
    let mut owned: Vec<OwnedImageObs> = Vec::with_capacity(observations.len());
    for item in observations.iter() {
        let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err(
                "each observation must be (image, poni, calibrant) or (image, poni, calibrant, mask)",
            )
        })?;
        if tuple.len() != 3 && tuple.len() != 4 {
            return Err(PyValueError::new_err(
                "each observation must be (image, poni, calibrant) or (image, poni, calibrant, mask)",
            ));
        }
        let img_ro: PyReadonlyArray2<'_, f64> = tuple.get_item(0)?.extract()?;
        let view = img_ro.as_array();
        let shape = view.shape();
        let (d1, d2) = (shape[0], shape[1]);
        let img_buf: Vec<f64> = if view.is_standard_layout() {
            img_ro
                .as_slice()
                .map(|s| s.to_vec())
                .unwrap_or_else(|_| view.iter().copied().collect())
        } else {
            view.iter().copied().collect()
        };
        let poni_any = tuple.get_item(1)?;
        let poni_val: Poni = if let Ok(integ) = poni_any.extract::<PyRef<PyIntegrator>>() {
            integ.inner().geometry().poni.clone()
        } else if let Ok(dict) = poni_any.cast::<PyDict>() {
            poni_from_dict(dict)?
        } else {
            return Err(PyValueError::new_err(
                "poni must be a dioptrin.Integrator or a PONI dict",
            ));
        };
        let cal_ref: PyRef<PyCalibrant> = tuple.get_item(2)?.extract()?;
        let mask_buf: Option<Vec<bool>> = if tuple.len() == 4 {
            let m_item = tuple.get_item(3)?;
            if m_item.is_none() {
                None
            } else {
                let m_ro: PyReadonlyArray2<'_, bool> = m_item.extract()?;
                let mv = m_ro.as_array();
                if mv.shape() != shape {
                    return Err(PyValueError::new_err(format!(
                        "mask shape ({}, {}) does not match image shape ({d1}, {d2})",
                        mv.shape()[0],
                        mv.shape()[1]
                    )));
                }
                Some(if mv.is_standard_layout() {
                    m_ro.as_slice()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|_| mv.iter().copied().collect())
                } else {
                    mv.iter().copied().collect()
                })
            }
        } else {
            None
        };
        owned.push(OwnedImageObs {
            image: img_buf,
            dim1: d1,
            dim2: d2,
            mask: mask_buf,
            poni: poni_val,
            calibrant: cal_ref.inner.clone(),
        });
    }
    let (dim1, dim2) = (owned[0].dim1, owned[0].dim2);
    let model_enum = parse_distortion_model(
        model,
        max_order,
        grid_nx,
        grid_ny,
        smoothness,
        max_degree,
        module_height,
        module_width,
        gap_height,
        gap_width,
        fit_z,
        regularization,
    )?;
    let refine_mask = parse_distortion_mask(fix)?;
    let opts = DistortionImageOptions {
        distortion: DistortionOptions {
            model: model_enum,
            huber_c_px,
            refine_poni,
            refine_mask,
            cv_folds,
        },
        n_passes,
        max_ring_index,
        max_rings,
        n_azimuth,
        intensity_limit,
        search_half_width_px,
    };
    let res = py
        .detach(|| {
            let refs: Vec<DistortionImageObservation<'_>> = owned
                .iter()
                .map(|o| DistortionImageObservation {
                    image: &o.image,
                    dim1: o.dim1,
                    dim2: o.dim2,
                    mask: o.mask.as_deref(),
                    poni: &o.poni,
                    calibrant: &o.calibrant,
                })
                .collect();
            estimate_distortion_image_multi(&refs, &opts)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyDistortionImageResult {
        inner: res,
        dim1,
        dim2,
    })
}
