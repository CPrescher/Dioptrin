use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::thread;

use crate::batch::{
    process_files, Integration1D, Integration1DWithErrors, Integration2D, Integration2DWithErrors,
    Integration2DWithAzimuthalRange, Integration2DWithAzimuthalRangeAndErrors,
};
use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::integration::{IntegrationMethod, Integrator, RadialUnit};
use crate::poni::Poni;

fn parse_method(method: Option<&str>, n_ss: Option<usize>) -> Result<IntegrationMethod, PyErr> {
    match method.unwrap_or("pixel_split").to_lowercase().as_str() {
        "pixel_split" | "pixelsplit" | "ps" | "pixelsplitting" => Ok(IntegrationMethod::PixelSplit),
        "counting" | "count" => Ok(IntegrationMethod::Counting),
        "histogram" | "hist" => Ok(IntegrationMethod::Histogram),
        "supersampled" | "supersample" | "ss" => {
            let n_ss = n_ss.unwrap_or(5);
            Ok(IntegrationMethod::Supersampled { n_ss })
        }
        other => Err(PyValueError::new_err(format!(
            "Unknown integration method: {}",
            other
        ))),
    }
}

fn parse_unit(unit: Option<&str>) -> Result<RadialUnit, PyErr> {
    match unit.unwrap_or("2th_rad").to_lowercase().as_str() {
        "2th_rad" | "tth_rad" | "2theta_rad" => Ok(RadialUnit::TthRad),
        "2th_deg" | "tth_deg" | "2theta_deg" => Ok(RadialUnit::TthDeg),
        "q_a^-1" | "q_angstrom" | "q_a" => Ok(RadialUnit::QAngstrom),
        "q_nm^-1" | "q_nm" => Ok(RadialUnit::QNm),
        other => Err(PyValueError::new_err(format!(
            "Unknown radial unit: {}. Valid options: 2th_rad, 2th_deg, q_A^-1, q_nm^-1",
            other
        ))),
    }
}

fn resolve_workers_threads(
    num_workers: Option<usize>,
    num_threads: Option<usize>,
) -> PyResult<(usize, usize)> {
    let num_threads = num_threads.unwrap_or(2);
    if num_threads == 0 {
        return Err(PyValueError::new_err("num_threads must be > 0"));
    }

    let num_workers = match num_workers {
        Some(workers) => {
            if workers == 0 {
                return Err(PyValueError::new_err("num_workers must be > 0"));
            }
            workers
        }
        None => {
            let cores = thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let max_workers = cores / num_threads;
            let computed = max_workers.saturating_sub(1);
            if computed < 1 {
                1
            } else {
                computed
            }
        }
    };

    Ok((num_workers, num_threads))
}

fn poni_from_dict(poni_dict: &Bound<'_, PyDict>) -> PyResult<Poni> {
    let mut poni = Poni::new();

    if let Some(val) = poni_dict.get_item("pixel1")? {
        poni.pixel1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("pixel2")? {
        poni.pixel2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("distance")? {
        poni.distance = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("poni1")? {
        poni.poni1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("poni2")? {
        poni.poni2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot1")? {
        poni.rot1 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot2")? {
        poni.rot2 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("rot3")? {
        poni.rot3 = val.extract()?;
    }
    if let Some(val) = poni_dict.get_item("wavelength")? {
        poni.wavelength = val.extract()?;
    }

    Ok(poni)
}

/// Perform 1D radial integration of a diffraction pattern.
///
/// This function loads a PONI calibration file and a TIFF image, then performs
/// 1D integration along the radial (scattering angle) axis.
///
/// # Arguments
///
/// * `poni_path` - Path to the PONI calibration file (v1 or v2 format)
/// * `tiff_path` - Path to the TIFF image file containing the diffraction pattern
/// * `n_points` - Number of bins for the 1D integration output
/// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
/// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
/// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
/// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
///
/// # Returns
///
/// Tuple of (radial_values, intensities) as lists of floats
///
/// # Example
///
/// ```python
/// import dioptrin
/// radial, intensity = dioptrin.integrate1d(
///     "calibration.poni",
///     "pattern.tif",
///     n_points=1000,
///     method="pixel_split",
///     unit="2th_deg"
/// )
/// ```
#[pyfunction]
#[pyo3(
    text_signature = "(poni_path, tiff_path, n_points, method=None, polarization_factor=None, n_ss=None, unit=None)"
)]
fn integrate1d(
    poni_path: &str,
    tiff_path: &str,
    n_points: usize,
    method: Option<&str>,
    polarization_factor: Option<f64>,
    n_ss: Option<usize>,
    unit: Option<&str>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let poni = Poni::load(poni_path)
        .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
    let frame = Frame::from_tiff(tiff_path.to_string())
        .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(parse_method(method, n_ss)?)
        .with_radial_unit(parse_unit(unit)?);
    if let Some(polarization_factor) = polarization_factor {
        integrator = integrator.with_polarization_factor(polarization_factor);
    }

    Ok(integrator.integrate1d(&frame, n_points))
}

/// Perform 1D radial integration with error propagation.
///
/// Similar to `integrate1d`, but also returns error estimates (standard deviations)
/// for each bin based on Poisson statistics.
///
/// # Arguments
///
/// * `poni_path` - Path to the PONI calibration file (v1 or v2 format)
/// * `tiff_path` - Path to the TIFF image file containing the diffraction pattern
/// * `n_points` - Number of bins for the 1D integration output
/// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
/// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
/// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
/// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
///
/// # Returns
///
/// Tuple of (radial_values, intensities, errors) as lists of floats
///
/// # Example
///
/// ```python
/// import dioptrin
/// radial, intensity, errors = dioptrin.integrate1d_with_errors(
///     "calibration.poni",
///     "pattern.tif",
///     n_points=1000
/// )
/// ```
#[pyfunction]
#[pyo3(
    text_signature = "(poni_path, tiff_path, n_points, method=None, polarization_factor=None, n_ss=None, unit=None)"
)]
fn integrate1d_with_errors(
    poni_path: &str,
    tiff_path: &str,
    n_points: usize,
    method: Option<&str>,
    polarization_factor: Option<f64>,
    n_ss: Option<usize>,
    unit: Option<&str>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let poni = Poni::load(poni_path)
        .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
    let frame = Frame::from_tiff(tiff_path.to_string())
        .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry)
        .with_method(parse_method(method, n_ss)?)
        .with_radial_unit(parse_unit(unit)?);
    if let Some(polarization_factor) = polarization_factor {
        integrator = integrator.with_polarization_factor(polarization_factor);
    }

    Ok(integrator.integrate1d_with_errors(&frame, n_points))
}

/// Get the current number of threads used by Rayon for parallel processing.
///
/// # Returns
///
/// Number of threads currently configured for parallel operations
///
/// # Example
///
/// ```python
/// import dioptrin
/// threads = dioptrin.rayon_num_threads()
/// print(f"Using {threads} threads")
/// ```
#[pyfunction]
fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Set the number of threads for Rayon parallel processing.
///
/// This affects all parallel operations in the library, including geometry
/// calculations and integration.
///
/// # Arguments
///
/// * `threads` - Number of threads to use (must be > 0)
///
/// # Raises
///
/// ValueError if thread count is invalid or cannot be set
///
/// # Example
///
/// ```python
/// import dioptrin
/// dioptrin.set_rayon_threads(8)  # Use 8 threads
/// ```
#[pyfunction]
#[pyo3(text_signature = "(threads)")]
fn set_rayon_threads(threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| PyValueError::new_err(format!("Failed to set Rayon threads: {}", err)))
}

#[pyclass(name = "Integrator", unsendable)]
struct PyIntegrator {
    integrator: Arc<Integrator>,
}

#[pymethods]
impl PyIntegrator {
    /// Create a new Integrator instance.
    ///
    /// The integrator is configured with detector geometry from a PONI file.
    /// The dimensions will be determined automatically when the first image
    /// is integrated.
    ///
    /// # Arguments
    ///
    /// * `poni_path` - Path to the PONI calibration file (v1 or v2 format)
    /// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
    /// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Returns
    ///
    /// New Integrator instance
    ///
    /// # Example
    ///
    /// ```python
    /// import dioptrin
    /// integrator = dioptrin.Integrator(
    ///     "calibration.poni",
    ///     method="pixel_split",
    ///     unit="2th_deg"
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (poni_path, method=None, polarization_factor=None, n_ss=None, unit=None))]
    fn new(
        poni_path: &str,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
    ) -> PyResult<Self> {
        let poni = Poni::load(poni_path)
            .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(parse_method(method, n_ss)?)
            .with_radial_unit(parse_unit(unit)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator: Arc::new(integrator),
        })
    }

    /// Create an Integrator from a PONI dictionary.
    ///
    /// The dictionary should contain PONI calibration parameters as keys:
    /// "pixel1", "pixel2", "distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength".
    ///
    /// # Arguments
    ///
    /// * `poni_dict` - Dictionary containing PONI calibration parameters
    /// * `method` - Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
    /// * `polarization_factor` - Optional polarization correction factor (0.0 to 1.0)
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Returns
    ///
    /// New Integrator instance
    ///
    /// # Example
    ///
    /// ```python
    /// import dioptrin
    /// poni_dict = {
    ///     "pixel1": 0.000172,
    ///     "pixel2": 0.000172,
    ///     "distance": 0.2,
    ///     "poni1": 0.1,
    ///     "poni2": 0.1,
    ///     "wavelength": 1e-10
    /// }
    /// integrator = dioptrin.Integrator.from_poni_dict(poni_dict)
    /// ```
    #[classmethod]
    #[pyo3(signature = (poni_dict, method=None, polarization_factor=None, n_ss=None, unit=None))]
    fn from_poni_dict(
        _cls: &Bound<'_, PyType>,
        poni_dict: &Bound<'_, PyDict>,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
    ) -> PyResult<Self> {
        let poni = poni_from_dict(poni_dict)?;
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(parse_method(method, n_ss)?)
            .with_radial_unit(parse_unit(unit)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator: Arc::new(integrator),
        })
    }

    /// Perform 1D radial integration from a TIFF file.
    ///
    /// Loads a TIFF image and integrates it along the radial axis using the
    /// configured integration method and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_path` - Path to the TIFF image file
    /// * `n_points` - Number of bins for the 1D integration output
    ///
    /// # Returns
    ///
    /// Tuple of (radial_values, intensities) as lists of floats
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// radial, intensity = integrator.integrate1d("pattern.tif", n_points=1000)
    /// ```
    #[pyo3(signature = (tiff_path, n_points))]
    fn integrate1d(&self, tiff_path: &str, n_points: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        Ok(self.integrator.integrate1d(&frame, n_points))
    }

    /// Perform 1D radial integration from a NumPy array.
    ///
    /// Integrates a 2D NumPy array (detector image) along the radial axis.
    /// The array must be C-contiguous (row-major) and of type float64.
    ///
    /// # Arguments
    ///
    /// * `array` - 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
    /// * `n_points` - Number of bins for the 1D integration output
    ///
    /// # Returns
    ///
    /// Tuple of (radial_values, intensities) as lists of floats
    ///
    /// # Raises
    ///
    /// ValueError if the array is not C-contiguous or not float64
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// data = np.array(image_data, dtype=np.float64)
    /// radial, intensity = integrator.integrate1d_array(data, n_points=1000)
    /// ```
    #[pyo3(signature = (array, n_points))]
    fn integrate1d_array(
        &self,
        array: PyReadonlyArray2<f64>,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let view = array.as_array();
        if !view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "Array must be C-contiguous (row-major) and float64",
            ));
        }
        let shape = view.shape();
        let dim1 = shape[0];
        let dim2 = shape[1];

        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        Ok(self
            .integrator
            .integrate1d_from_slice(slice, dim1, dim2, n_points))
    }

    /// Perform 1D radial integration from a NumPy array with error propagation.
    ///
    /// Similar to `integrate1d_array`, but also returns error estimates
    /// (standard deviations) for each bin based on Poisson statistics.
    ///
    /// # Arguments
    ///
    /// * `array` - 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
    /// * `n_points` - Number of bins for the 1D integration output
    ///
    /// # Returns
    ///
    /// Tuple of (radial_values, intensities, errors) as lists of floats
    ///
    /// # Raises
    ///
    /// ValueError if the array is not C-contiguous or not float64
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// data = np.array(image_data, dtype=np.float64)
    /// radial, intensity, errors = integrator.integrate1d_array_with_errors(data, n_points=1000)
    /// ```
    #[pyo3(signature = (array, n_points))]
    fn integrate1d_array_with_errors(
        &self,
        array: PyReadonlyArray2<f64>,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let view = array.as_array();
        if !view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "Array must be C-contiguous (row-major) and float64",
            ));
        }
        let shape = view.shape();
        let dim1 = shape[0];
        let dim2 = shape[1];

        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        Ok(self
            .integrator
            .integrate1d_with_errors_from_slice(slice, dim1, dim2, n_points))
    }

    /// Perform 1D radial integration from a TIFF file with error propagation.
    ///
    /// Similar to `integrate1d`, but also returns error estimates
    /// (standard deviations) for each bin based on Poisson statistics.
    ///
    /// # Arguments
    ///
    /// * `tiff_path` - Path to the TIFF image file
    /// * `n_points` - Number of bins for the 1D integration output
    ///
    /// # Returns
    ///
    /// Tuple of (radial_values, intensities, errors) as lists of floats
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// radial, intensity, errors = integrator.integrate1d_with_errors("pattern.tif", n_points=1000)
    /// ```
    #[pyo3(signature = (tiff_path, n_points))]
    fn integrate1d_with_errors(
        &self,
        tiff_path: &str,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        Ok(self.integrator.integrate1d_with_errors(&frame, n_points))
    }

    /// Set the integration method.
    ///
    /// Changes the integration method used for subsequent integrations.
    /// Available methods: "pixel_split", "counting", "histogram", or "supersampled".
    ///
    /// # Arguments
    ///
    /// * `method` - Integration method name
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (required if method is "supersampled")
    ///
    /// # Raises
    ///
    /// ValueError if the method name is invalid
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_method("supersampled", n_ss=7)
    /// ```
    #[pyo3(signature = (method, n_ss=None))]
    fn set_method(&mut self, method: &str, n_ss: Option<usize>) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator).ok_or_else(|| {
            PyValueError::new_err("Integrator is in use by a batch operation")
        })?;
        integrator.set_method(parse_method(Some(method), n_ss)?);
        Ok(())
    }

    /// Set the polarization correction factor.
    ///
    /// Applies polarization correction to the integration. The factor should
    /// be between 0.0 and 1.0. Set to None to disable polarization correction.
    ///
    /// # Arguments
    ///
    /// * `polarization_factor` - Polarization factor (0.0 to 1.0) or None to disable
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_polarization_factor(0.99)  # Enable correction
    /// integrator.set_polarization_factor(None)  # Disable correction
    /// ```
    fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator).ok_or_else(|| {
            PyValueError::new_err("Integrator is in use by a batch operation")
        })?;
        integrator.set_polarization_factor(polarization_factor);
        Ok(())
    }

    /// Set the radial unit for integration output.
    ///
    /// Changes the unit used for the radial (scattering angle) axis in integration results.
    ///
    /// # Arguments
    ///
    /// * `unit` - Radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Raises
    ///
    /// ValueError if the unit name is invalid
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_unit("q_A^-1")  # Switch to q-space in inverse Angstroms
    /// ```
    fn set_unit(&mut self, unit: &str) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator).ok_or_else(|| {
            PyValueError::new_err("Integrator is in use by a batch operation")
        })?;
        integrator.set_radial_unit(parse_unit(Some(unit))?);
        Ok(())
    }

    /// Get the current radial unit as a string.
    ///
    /// Returns the unit currently configured for the radial axis.
    ///
    /// # Returns
    ///
    /// String representation of the current radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"
    ///
    /// # Example
    ///
    /// ```python
    /// unit = integrator.get_unit()
    /// print(f"Current unit: {unit}")
    /// ```
    fn get_unit(&self) -> &'static str {
        match self.integrator.radial_unit() {
            RadialUnit::TthRad => "2th_rad",
            RadialUnit::TthDeg => "2th_deg",
            RadialUnit::QAngstrom => "q_A^-1",
            RadialUnit::QNm => "q_nm^-1",
        }
    }

    /// Perform batch 1D radial integration over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_points, num_workers=None, num_threads=None))]
    fn batch_integrate1d(
        &self,
        tiff_paths: Vec<String>,
        n_points: usize,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<(Vec<f64>, Vec<f64>)>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration1D { n_bins: n_points };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }

    /// Perform batch 1D radial integration with errors over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_points, num_workers=None, num_threads=None))]
    fn batch_integrate1d_with_errors(
        &self,
        tiff_paths: Vec<String>,
        n_points: usize,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<(Vec<f64>, Vec<f64>, Vec<f64>)>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration1DWithErrors { n_bins: n_points };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }

    /// Perform batch 2D radial integration over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, num_workers=None, num_threads=None))]
    fn batch_integrate2d(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<Vec<f64>>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration2D {
            n_radial,
            n_azimuthal,
        };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }

    /// Perform batch 2D radial integration with errors over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, num_workers=None, num_threads=None))]
    fn batch_integrate2d_with_errors(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<(Vec<f64>, Vec<f64>)>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration2DWithErrors {
            n_radial,
            n_azimuthal,
        };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }

    /// Perform batch 2D integration with an azimuthal range over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `azimuthal_min` - Minimum azimuthal angle (radians)
    /// * `azimuthal_max` - Maximum azimuthal angle (radians)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, azimuthal_min, azimuthal_max, num_workers=None, num_threads=None))]
    fn batch_integrate2d_with_azimuthal_range(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_min: f64,
        azimuthal_max: f64,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<Vec<f64>>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration2DWithAzimuthalRange {
            n_radial,
            n_azimuthal,
            azimuthal_min,
            azimuthal_max,
        };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }

    /// Perform batch 2D integration with azimuthal range and errors over multiple TIFF files.
    ///
    /// This reuses the integrator's geometry, method, unit, and corrections.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `azimuthal_min` - Minimum azimuthal angle (radians)
    /// * `azimuthal_max` - Maximum azimuthal angle (radians)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Worker/thread defaults
    ///
    /// If not provided, the defaults are `num_threads = 2` and
    /// `num_workers = max(1, floor(available_cores / num_threads) - 1)`.
    ///
    /// # Returns
    ///
    /// List of (filename, result, error) tuples in the same order as input.
    /// For each entry, result is None when error is set.
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, azimuthal_min, azimuthal_max, num_workers=None, num_threads=None))]
    fn batch_integrate2d_with_azimuthal_range_and_errors(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_min: f64,
        azimuthal_max: f64,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<(String, Option<(Vec<f64>, Vec<f64>)>, Option<String>)>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        let task = Integration2DWithAzimuthalRangeAndErrors {
            n_radial,
            n_azimuthal,
            azimuthal_min,
            azimuthal_max,
        };
        let results = process_files(
            &tiff_paths,
            Arc::clone(&self.integrator),
            task,
            num_workers,
            num_threads,
        );

        Ok(results
            .into_iter()
            .map(|(filename, result)| match result {
                Ok(output) => (filename, Some(output), None),
                Err(err) => (filename, None, Some(err)),
            })
            .collect())
    }
}

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrate1d, m)?)?;
    m.add_function(wrap_pyfunction!(integrate1d_with_errors, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_class::<PyIntegrator>()?;
    Ok(())
}
