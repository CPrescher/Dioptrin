use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::batch::{
    Integration1D, Integration1DWithErrors, Integration2D, Integration2DWithAzimuthalRange,
    Integration2DWithAzimuthalRangeAndErrors, Integration2DWithErrors, WorkItem, process_files,
    process_files_streaming, process_queue_streaming,
};
use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::integration::{Integrator, RadialUnit};
use crate::mask::Mask;
use crate::poni::Poni;

use super::results::{BatchReceiver, PyBatchIterator, PyBatchResult, PyIntegrationResult};
use super::utils::{
    BatchImageInput, ClassifiedInput, FrameOrPath, ImageInput, classify_batch_input,
    extract_array_dims, extract_frame_from_pyany, parse_method, parse_unit, poni_from_dict,
    resolve_workers_threads,
};

#[derive(FromPyObject)]
pub enum MaskInput<'p> {
    Path(String),
    ArrayU8(PyReadonlyArray2<'p, u8>),
    ArrayBool(PyReadonlyArray2<'p, bool>),
}

#[pyclass(name = "Integrator", unsendable)]
pub struct PyIntegrator {
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
        #[cfg(feature = "licensed")]
        {
            crate::license::verify_license()
                .map_err(|e| crate::python::LicenseError::new_err(e.to_string()))?;
        }
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
        #[cfg(feature = "licensed")]
        {
            crate::license::verify_license()
                .map_err(|e| crate::python::LicenseError::new_err(e.to_string()))?;
        }
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

    /// Perform 1D radial integration.
    ///
    /// # Arguments
    ///
    /// * `image` - Path to the TIFF image file or 2D NumPy array
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    ///
    /// # Returns
    ///
    /// IntegrationResult with radial values, intensities, and optional errors
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// result = integrator.integrate1d("pattern.tif", n_points=1000)
    /// result = integrator.integrate1d("pattern.tif", n_points=1000, errors=True)
    /// ```
    #[pyo3(signature = (image, n_points, errors=false))]
    fn integrate1d(
        &self,
        image: ImageInput,
        n_points: usize,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        match image {
            ImageInput::Path(tiff_path) => {
                let frame = Frame::from_tiff(tiff_path).map_err(|err| {
                    PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                })?;
                if errors {
                    let (radial, intensity, errs) =
                        self.integrator.integrate1d_with_errors(&frame, n_points);
                    Ok(PyIntegrationResult::from_1d_with_errors(
                        radial, intensity, errs,
                    ))
                } else {
                    let (radial, intensity) = self.integrator.integrate1d(&frame, n_points);
                    Ok(PyIntegrationResult::from_1d(radial, intensity))
                }
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                if errors {
                    let (radial, intensity, errs) = self
                        .integrator
                        .integrate1d_with_errors_from_slice(slice, dim1, dim2, n_points);
                    Ok(PyIntegrationResult::from_1d_with_errors(
                        radial, intensity, errs,
                    ))
                } else {
                    let (radial, intensity) = self
                        .integrator
                        .integrate1d_from_slice(slice, dim1, dim2, n_points);
                    Ok(PyIntegrationResult::from_1d(radial, intensity))
                }
            }
        }
    }

    /// Perform 2D integration.
    ///
    /// Set `errors=True` to include per-bin error estimates. Use `azimuthal_range`
    /// to restrict the azimuthal range.
    ///
    /// # Arguments
    ///
    /// * `image` - Path to the TIFF image file or 2D NumPy array
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    ///
    /// # Returns
    ///
    /// IntegrationResult with flat intensity (and optional errors) sized
    /// `n_radial * n_azimuthal`
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// result = integrator.integrate2d("pattern.tif", n_radial=100, n_azimuthal=360)
    /// result = integrator.integrate2d(
    ///     "pattern.tif", n_radial=100, n_azimuthal=90, azimuthal_range=(0, 1.57), errors=True
    /// )
    /// ```
    #[pyo3(signature = (image, n_radial, n_azimuthal, errors=false, azimuthal_range=None))]
    fn integrate2d(
        &self,
        image: ImageInput,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
    ) -> PyResult<PyIntegrationResult> {
        match image {
            ImageInput::Path(tiff_path) => {
                let frame = Frame::from_tiff(tiff_path).map_err(|err| {
                    PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                })?;

                let result = match (errors, azimuthal_range) {
                    (false, None) => {
                        let intensity = self.integrator.integrate2d(&frame, n_radial, n_azimuthal);
                        PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
                    }
                    (true, None) => {
                        let (intensity, errs) =
                            self.integrator
                                .integrate2d_with_errors(&frame, n_radial, n_azimuthal);
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errs,
                            n_radial,
                            n_azimuthal,
                        )
                    }
                    (false, Some((azimuthal_min, azimuthal_max))) => {
                        let intensity = self.integrator.integrate2d_with_azimuthal_range(
                            &frame,
                            n_radial,
                            n_azimuthal,
                            azimuthal_min,
                            azimuthal_max,
                        );
                        PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
                    }
                    (true, Some((azimuthal_min, azimuthal_max))) => {
                        let (intensity, errs) =
                            self.integrator.integrate2d_with_azimuthal_range_and_errors(
                                &frame,
                                n_radial,
                                n_azimuthal,
                                azimuthal_min,
                                azimuthal_max,
                            );
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errs,
                            n_radial,
                            n_azimuthal,
                        )
                    }
                };
                Ok(result)
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                let result = match (errors, azimuthal_range) {
                    (false, None) => {
                        let intensity = self.integrator.integrate2d_from_slice(
                            slice,
                            dim1,
                            dim2,
                            n_radial,
                            n_azimuthal,
                        );
                        PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
                    }
                    (true, None) => {
                        let (intensity, errs) = self.integrator.integrate2d_with_errors_from_slice(
                            slice,
                            dim1,
                            dim2,
                            n_radial,
                            n_azimuthal,
                        );
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errs,
                            n_radial,
                            n_azimuthal,
                        )
                    }
                    (false, Some((azimuthal_min, azimuthal_max))) => {
                        let intensity =
                            self.integrator.integrate2d_with_azimuthal_range_from_slice(
                                slice,
                                dim1,
                                dim2,
                                n_radial,
                                n_azimuthal,
                                azimuthal_min,
                                azimuthal_max,
                            );
                        PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
                    }
                    (true, Some((azimuthal_min, azimuthal_max))) => {
                        let (intensity, errs) = self
                            .integrator
                            .integrate2d_with_azimuthal_range_and_errors_from_slice(
                                slice,
                                dim1,
                                dim2,
                                n_radial,
                                n_azimuthal,
                                azimuthal_min,
                                azimuthal_max,
                            );
                        PyIntegrationResult::from_2d_with_errors(
                            intensity,
                            errs,
                            n_radial,
                            n_azimuthal,
                        )
                    }
                };
                Ok(result)
            }
        }
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
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
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
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
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
    ///     integrator.set_unit("q_A^-1")  # Switch to q-space in inverse Angstroms
    /// ```
    fn set_unit(&mut self, unit: &str) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_radial_unit(parse_unit(Some(unit))?);
        Ok(())
    }

    /// Set the mask for the integrator.
    ///
    /// The mask can be provided as a path to a TIFF file or as a NumPy array.
    /// In the mask, non-zero values indicate pixels to be masked (ignored).
    ///
    /// # Arguments
    ///
    /// * `mask` - Path to a TIFF file (str), a NumPy array (uint8 or bool), or None to clear the mask.
    ///
    /// # Raises
    ///
    /// ValueError if the mask file cannot be loaded or if the array has an incorrect shape or type.
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// # From file
    /// integrator.set_mask("mask.tif")
    ///
    /// # From array
    /// mask_array = np.zeros((1043, 981), dtype=np.uint8)
    /// mask_array[500:600, :] = 1
    /// integrator.set_mask(mask_array)
    ///
    /// # Clear mask
    /// integrator.set_mask(None)
    /// ```
    #[pyo3(signature = (mask))]
    fn set_mask(&mut self, mask: Option<MaskInput>) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;

        match mask {
            Some(MaskInput::Path(path)) => {
                let mask_obj = Mask::from_tiff(path).map_err(|err| {
                    PyValueError::new_err(format!("Failed to load mask TIFF: {}", err))
                })?;
                integrator.set_mask(mask_obj);
            }
            Some(MaskInput::ArrayU8(array)) => {
                let (dim1, dim2) = extract_array_dims(&array, "uint8")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Mask must be contiguous: {}", err))
                })?;
                let mask_obj =
                    Mask::from_vec(slice.to_vec(), dim1, dim2).map_err(PyValueError::new_err)?;
                integrator.set_mask(mask_obj);
            }
            Some(MaskInput::ArrayBool(array)) => {
                let (dim1, dim2) = extract_array_dims(&array, "bool")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Mask must be contiguous: {}", err))
                })?;
                let data_u8: Vec<u8> = slice.iter().map(|&b| b as u8).collect();
                let mask_obj =
                    Mask::from_vec(data_u8, dim1, dim2).map_err(PyValueError::new_err)?;
                integrator.set_mask(mask_obj);
            }
            None => {
                integrator.clear_mask();
            }
        }
        Ok(())
    }
    /// Clear the mask from the integrator.
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.clear_mask()
    /// ```
    fn clear_mask(&mut self) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.clear_mask();
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

    /// Perform batch 1D integration.
    ///
    /// This is the recommended batch API. Returns a list of BatchResult objects
    /// with IntegrationResult containing optional error estimates.
    ///
    /// # Arguments
    ///
    /// * `images` - List of TIFF file paths (strings) or NumPy arrays (2D float64)
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename/index and IntegrationResult.
    ///
    /// # Example
    ///
    /// ```python
    /// # From TIFF files
    /// results = integrator.batch1d(file_paths, n_points=1000, errors=True)
    ///
    /// # From HDF5 via NumPy arrays
    /// import h5py
    /// with h5py.File('data.h5', 'r') as f:
    ///     frames = [f['/entry/data/data'][i] for i in range(100)]
    /// results = integrator.batch1d(frames, n_points=1000)
    ///
    /// for r in results:
    ///     if r.is_ok():
    ///         print(r.result.intensity)
    /// ```
    #[pyo3(signature = (images, n_points, errors=false, num_workers=None, num_threads=None))]
    fn batch1d(
        &self,
        images: BatchImageInput,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<PyBatchResult>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match images {
            BatchImageInput::Paths(tiff_paths) => {
                // Original file-based processing
                if errors {
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
                            Ok((radial, intensity, errs)) => PyBatchResult {
                                filename,
                                result: Some(PyIntegrationResult::from_1d_with_errors(
                                    radial, intensity, errs,
                                )),
                                error: None,
                            },
                            Err(err) => PyBatchResult {
                                filename,
                                result: None,
                                error: Some(err),
                            },
                        })
                        .collect())
                } else {
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
                            Ok((radial, intensity)) => PyBatchResult {
                                filename,
                                result: Some(PyIntegrationResult::from_1d(radial, intensity)),
                                error: None,
                            },
                            Err(err) => PyBatchResult {
                                filename,
                                result: None,
                                error: Some(err),
                            },
                        })
                        .collect())
                }
            }
            BatchImageInput::Arrays(arrays) => {
                // Array-based processing: convert arrays to frames then integrate
                let num_arrays = arrays.len();

                // Convert all arrays to owned Frame objects
                let frames: Result<Vec<Frame>, PyErr> = arrays
                    .iter()
                    .enumerate()
                    .map(|(i, array)| {
                        let (dim1, dim2) = extract_array_dims(array, "float64")?;
                        let slice = array.as_slice().map_err(|err| {
                            PyValueError::new_err(format!(
                                "Array {} must be contiguous: {}",
                                i, err
                            ))
                        })?;
                        Ok(Frame {
                            name: format!("array[{}]", i),
                            data: slice.to_vec(),
                            dim1,
                            dim2,
                        })
                    })
                    .collect();

                let frames = frames?;

                // Process frames in parallel using rayon
                use rayon::prelude::*;
                let results: Vec<PyBatchResult> = (0..num_arrays)
                    .into_par_iter()
                    .map(|i| {
                        let frame = &frames[i];
                        let integration_result = if errors {
                            let (radial, intensity, errs) =
                                self.integrator.integrate1d_with_errors(frame, n_points);
                            PyIntegrationResult::from_1d_with_errors(radial, intensity, errs)
                        } else {
                            let (radial, intensity) = self.integrator.integrate1d(frame, n_points);
                            PyIntegrationResult::from_1d(radial, intensity)
                        };

                        PyBatchResult {
                            filename: format!("array[{}]", i),
                            result: Some(integration_result),
                            error: None,
                        }
                    })
                    .collect();

                Ok(results)
            }
        }
    }

    /// Perform streaming batch 1D integration.
    ///
    /// Returns an iterator that yields results as each frame completes.
    /// More memory-efficient for large batches. Accepts lists or generators/iterators.
    ///
    /// # Arguments
    ///
    /// * `images` - List of TIFF file paths, list of NumPy arrays, or a generator/iterator
    ///   yielding file paths (str) or NumPy arrays (2D float64)
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// BatchIterator that yields BatchResult objects as they complete.
    /// The iterator has `total`, `completed`, and `progress` properties.
    /// When using a generator, `total` is 0 and `progress` returns None.
    ///
    /// # Example
    ///
    /// ```python
    /// # With files
    /// it = integrator.batch1d_iter(file_paths, n_points=1000)
    ///
    /// # With a generator (constant memory)
    /// def frame_gen():
    ///     with h5py.File('data.h5') as f:
    ///         for i in range(1000):
    ///             yield f['/data'][i].astype(np.float64)
    ///
    /// for result in integrator.batch1d_iter(frame_gen(), n_points=1000):
    ///     process(result.result)
    /// ```
    #[pyo3(signature = (images, n_points, errors=false, num_workers=None, num_threads=None))]
    fn batch1d_iter(
        &self,
        images: &Bound<'_, PyAny>,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<PyBatchIterator> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match classify_batch_input(images)? {
            ClassifiedInput::Paths(tiff_paths) => {
                // File-based streaming
                let total = tiff_paths.len();

                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let batch = process_files_streaming(
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1DWithErrors(Mutex::new(
                            batch.into_receiver(),
                        )),
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let batch = process_files_streaming(
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
            }
            ClassifiedInput::Arrays(arrays) => {
                let num_arrays = arrays.len();

                // Convert all arrays to owned Frame objects
                let frames: Result<Vec<Frame>, PyErr> = arrays
                    .iter()
                    .enumerate()
                    .map(|(i, array)| {
                        let (dim1, dim2) = extract_array_dims(array, "float64")?;
                        let slice = array.as_slice().map_err(|err| {
                            PyValueError::new_err(format!(
                                "Array {} must be contiguous: {}",
                                i, err
                            ))
                        })?;
                        Ok(Frame {
                            name: format!("array[{}]", i),
                            data: slice.to_vec(),
                            dim1,
                            dim2,
                        })
                    })
                    .collect();

                let frames = frames?;

                use std::sync::mpsc;
                let integrator = Arc::clone(&self.integrator);

                if errors {
                    let (tx, rx) = mpsc::channel();

                    std::thread::spawn(move || {
                        use rayon::prelude::*;
                        (0..num_arrays)
                            .into_par_iter()
                            .map(|i| {
                                let frame = &frames[i];
                                let result = integrator.integrate1d_with_errors(frame, n_points);
                                (i, format!("array[{}]", i), Ok(result))
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .for_each(|result| {
                                let _ = tx.send(result);
                            });
                    });

                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1DWithErrors(Mutex::new(rx)),
                        total: num_arrays,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                } else {
                    let (tx, rx) = mpsc::channel();

                    std::thread::spawn(move || {
                        use rayon::prelude::*;
                        (0..num_arrays)
                            .into_par_iter()
                            .map(|i| {
                                let frame = &frames[i];
                                let result = integrator.integrate1d(frame, n_points);
                                (i, format!("array[{}]", i), Ok(result))
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .for_each(|result| {
                                let _ = tx.send(result);
                            });
                    });

                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1D(Mutex::new(rx)),
                        total: num_arrays,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
            }
            ClassifiedInput::Iterator(iter) => {
                // Generator/iterator path: spawn producer thread with GIL
                let iter = iter.unbind();
                let (work_tx, work_rx) = std::sync::mpsc::sync_channel(num_workers * 2);

                // Producer thread: pull items from the Python iterator
                std::thread::spawn(move || {
                    let mut index = 0usize;
                    Python::attach(|py| {
                        let iter = iter.bind(py);
                        loop {
                            match iter.call_method0("__next__") {
                                Ok(item) => {
                                    let work_item = match extract_frame_from_pyany(&item, index) {
                                        Ok(FrameOrPath::Path(path)) => {
                                            WorkItem::Path { index, path }
                                        }
                                        Ok(FrameOrPath::Frame(frame)) => WorkItem::Frame {
                                            index,
                                            name: frame.name.clone(),
                                            frame,
                                        },
                                        Err(_) => break, // Extraction error, stop
                                    };
                                    if work_tx.send(work_item).is_err() {
                                        break; // Workers gone
                                    }
                                    index += 1;
                                }
                                Err(err) => {
                                    // StopIteration means end of iterator; any other error, stop
                                    if !err.is_instance_of::<pyo3::exceptions::PyStopIteration>(py)
                                    {
                                        // Log non-StopIteration errors as the first error result
                                        // by sending a poison path that will fail in the worker
                                        let _ = work_tx.send(WorkItem::Frame {
                                            index,
                                            name: format!("iter[{}]", index),
                                            frame: Frame {
                                                name: format!("error: {}", err),
                                                data: vec![],
                                                dim1: 0,
                                                dim2: 0,
                                            },
                                        });
                                    }
                                    break;
                                }
                            }
                        }
                    });
                    // work_tx dropped here → channel closes → workers drain and stop
                });

                let integrator = Arc::clone(&self.integrator);
                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let batch = process_queue_streaming(
                        work_rx,
                        integrator,
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1DWithErrors(Mutex::new(
                            batch.into_receiver(),
                        )),
                        total: 0,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let batch = process_queue_streaming(
                        work_rx,
                        integrator,
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                        total: 0,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
            }
        }
    }

    /// Perform batch 2D integration with unified result type.
    ///
    /// This is the recommended batch API for 2D integration. Returns a list of
    /// BatchResult objects with IntegrationResult containing optional error estimates.
    ///
    /// # Arguments
    ///
    /// * `tiff_paths` - List of TIFF file paths to process
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename and IntegrationResult.
    ///
    /// # Example
    ///
    /// ```python
    /// # Full 2D integration
    /// results = integrator.batch2d(files, n_radial=100, n_azimuthal=360)
    ///
    /// # With azimuthal range and errors
    /// results = integrator.batch2d(
    ///     files, n_radial=100, n_azimuthal=90,
    ///     azimuthal_range=(0, 1.57),  # 0 to pi/2
    ///     errors=True
    /// )
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal, errors=false, azimuthal_range=None, num_workers=None, num_threads=None))]
    fn batch2d(
        &self,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<PyBatchResult>> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match (errors, azimuthal_range) {
            (false, None) => {
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
                        Ok(intensity) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d(
                                intensity,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (true, None) => {
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
                        Ok((intensity, errs)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                errs,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
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
                        Ok(intensity) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d(
                                intensity,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
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
                        Ok((intensity, errs)) => PyBatchResult {
                            filename,
                            result: Some(PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                errs,
                                n_radial,
                                n_azimuthal,
                            )),
                            error: None,
                        },
                        Err(err) => PyBatchResult {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
        }
    }

    /// Perform streaming batch 2D integration.
    ///
    /// Returns an iterator that yields results as each file completes.
    /// More memory-efficient for large batches. Accepts lists or generators/iterators.
    ///
    /// # Arguments
    ///
    /// * `images` - List of TIFF file paths, list of NumPy arrays, or a generator/iterator
    ///   yielding file paths (str) or NumPy arrays (2D float64)
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_range` - Optional (min, max) tuple for azimuthal range in radians
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// BatchIterator that yields BatchResult objects as they complete.
    /// The iterator has `total`, `completed`, and `progress` properties.
    /// When using a generator, `total` is 0 and `progress` returns None.
    ///
    /// # Example
    ///
    /// ```python
    /// # With a list
    /// for result in integrator.batch2d_iter(files, n_radial=100, n_azimuthal=360):
    ///     if result.is_ok():
    ///         cake = result.result.intensity
    ///
    /// # With a generator
    /// for result in integrator.batch2d_iter(frame_gen(), n_radial=100, n_azimuthal=360):
    ///     process(result.result)
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (images, n_radial, n_azimuthal, errors=false, azimuthal_range=None, num_workers=None, num_threads=None))]
    fn batch2d_iter(
        &self,
        images: &Bound<'_, PyAny>,
        n_radial: usize,
        n_azimuthal: usize,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<PyBatchIterator> {
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        let classified = classify_batch_input(images)?;

        // For paths (backward compat), use process_files_streaming directly
        if let ClassifiedInput::Paths(ref tiff_paths) = classified {
            let total = tiff_paths.len();

            return match (errors, azimuthal_range) {
                (false, None) => {
                    let task = Integration2D {
                        n_radial,
                        n_azimuthal,
                    };
                    let batch = process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration2D {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                        },
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
                (true, None) => {
                    let task = Integration2DWithErrors {
                        n_radial,
                        n_azimuthal,
                    };
                    let batch = process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration2DWithErrors {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                        },
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
                (false, Some((azimuthal_min, azimuthal_max))) => {
                    let task = Integration2DWithAzimuthalRange {
                        n_radial,
                        n_azimuthal,
                        azimuthal_min,
                        azimuthal_max,
                    };
                    let batch = process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration2D {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                        },
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
                (true, Some((azimuthal_min, azimuthal_max))) => {
                    let task = Integration2DWithAzimuthalRangeAndErrors {
                        n_radial,
                        n_azimuthal,
                        azimuthal_min,
                        azimuthal_max,
                    };
                    let batch = process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(PyBatchIterator {
                        receiver: BatchReceiver::Integration2DWithErrors {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                        },
                        total,
                        completed: 0,
                        buffer: BTreeMap::new(),
                        next_index: 0,
                    })
                }
            };
        }

        // For arrays or iterators, use the queue-based streaming approach
        let (work_tx, work_rx) = std::sync::mpsc::sync_channel(num_workers * 2);

        let total = match classified {
            ClassifiedInput::Arrays(arrays) => {
                let num_arrays = arrays.len();
                // Convert all arrays to work items and send them
                for (i, array) in arrays.iter().enumerate() {
                    let (dim1, dim2) = extract_array_dims(array, "float64")?;
                    let slice = array.as_slice().map_err(|err| {
                        PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
                    })?;
                    let frame = Frame {
                        name: format!("array[{}]", i),
                        data: slice.to_vec(),
                        dim1,
                        dim2,
                    };
                    let _ = work_tx.send(WorkItem::Frame {
                        index: i,
                        name: frame.name.clone(),
                        frame,
                    });
                }
                drop(work_tx);
                num_arrays
            }
            ClassifiedInput::Iterator(iter) => {
                let iter = iter.unbind();
                std::thread::spawn(move || {
                    let mut index = 0usize;
                    Python::attach(|py| {
                        let iter = iter.bind(py);
                        loop {
                            match iter.call_method0("__next__") {
                                Ok(item) => {
                                    let work_item = match extract_frame_from_pyany(&item, index) {
                                        Ok(FrameOrPath::Path(path)) => {
                                            WorkItem::Path { index, path }
                                        }
                                        Ok(FrameOrPath::Frame(frame)) => WorkItem::Frame {
                                            index,
                                            name: frame.name.clone(),
                                            frame,
                                        },
                                        Err(_) => break,
                                    };
                                    if work_tx.send(work_item).is_err() {
                                        break;
                                    }
                                    index += 1;
                                }
                                Err(err) => {
                                    if !err.is_instance_of::<pyo3::exceptions::PyStopIteration>(py)
                                    {
                                        let _ = work_tx.send(WorkItem::Frame {
                                            index,
                                            name: format!("iter[{}]", index),
                                            frame: Frame {
                                                name: format!("error: {}", err),
                                                data: vec![],
                                                dim1: 0,
                                                dim2: 0,
                                            },
                                        });
                                    }
                                    break;
                                }
                            }
                        }
                    });
                });
                0 // Unknown total for iterators
            }
            ClassifiedInput::Paths(_) => unreachable!(), // Handled above
        };

        let integrator = Arc::clone(&self.integrator);

        match (errors, azimuthal_range) {
            (false, None) => {
                let task = Integration2D {
                    n_radial,
                    n_azimuthal,
                };
                let batch =
                    process_queue_streaming(work_rx, integrator, task, num_workers, num_threads);
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                    buffer: BTreeMap::new(),
                    next_index: 0,
                })
            }
            (true, None) => {
                let task = Integration2DWithErrors {
                    n_radial,
                    n_azimuthal,
                };
                let batch =
                    process_queue_streaming(work_rx, integrator, task, num_workers, num_threads);
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                    buffer: BTreeMap::new(),
                    next_index: 0,
                })
            }
            (false, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let batch =
                    process_queue_streaming(work_rx, integrator, task, num_workers, num_threads);
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                    buffer: BTreeMap::new(),
                    next_index: 0,
                })
            }
            (true, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DWithAzimuthalRangeAndErrors {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                };
                let batch =
                    process_queue_streaming(work_rx, integrator, task, num_workers, num_threads);
                Ok(PyBatchIterator {
                    receiver: BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    completed: 0,
                    buffer: BTreeMap::new(),
                    next_index: 0,
                })
            }
        }
    }
}
