use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::frame::Frame;
use crate::integration::PixelSplitMode;
use crate::multi::{Multi2dOptions, MultiIntegrator, MultiNormalization};

use super::integrator::{PyIntegrator, resolve_azimuthal_bins};
use super::results::PyIntegrationResult;
use super::utils::{ImageInput, extract_array_dims};

#[pyclass(name = "MultiIntegrator", unsendable)]
pub struct PyMultiIntegrator {
    multi: MultiIntegrator,
}

#[pymethods]
impl PyMultiIntegrator {
    /// Combine one image per detector on shared radial/azimuthal edges.
    ///
    /// Radial edges use the integrators' common radial_range and unit. Azimuthal
    /// edges are radians, default (-pi, pi); sectors are clipped, not folded.
    /// Images are C-contiguous float64 arrays or TIFF paths (mixed lists allowed).
    /// errors=True propagates Poisson max(image,0) with squared split weights.
    /// normalization='absolute' matches MultiGeometry(correctSolidAngle=True)
    /// scaling; 'relative' retains ordinary Dioptrin solid-angle scaling.
    /// normalization_factors are one positive monitor denominator per detector.
    /// Result intensity/errors reshape to (n_azimuthal,n_radial); both coordinate
    /// arrays contain bin centers. Empty bins contain zero.
    /// Instead of n_azimuthal, azimuthal_bracket (radians, keyword only) gives
    /// contiguous rows of centre +- bracket; the first row is centred on the
    /// start of azimuthal_range, which then defaults to (0, 2*pi), and the span
    /// must be a whole number of rows.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (images, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, *, normalization="relative", normalization_factors=None, azimuthal_bracket=None))]
    fn integrate2d(
        &self,
        py: Python<'_>,
        images: Vec<ImageInput>,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        normalization: &str,
        normalization_factors: Option<Vec<f64>>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<PyIntegrationResult> {
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        if images.len() != self.multi.integrators().len() {
            return Err(PyValueError::new_err(format!(
                "Expected {} images (one per detector), got {}",
                self.multi.integrators().len(),
                images.len()
            )));
        }
        let mut options = Multi2dOptions {
            errors,
            normalization_factors,
            ..Default::default()
        };
        if let Some(range) = azimuthal_range {
            options.azimuthal_range = range;
        }
        options.normalization = match normalization {
            "relative" => MultiNormalization::Relative,
            "absolute" => MultiNormalization::Absolute,
            _ => {
                return Err(PyValueError::new_err(
                    "normalization must be 'relative' or 'absolute'",
                ));
            }
        };
        let mut buffers = Vec::with_capacity(images.len());
        for (i, image) in images.into_iter().enumerate() {
            buffers.push(match image {
                ImageInput::Path(path) => {
                    let frame = Frame::from_tiff(path)
                        .map_err(|e| PyValueError::new_err(format!("detector {i}: {e}")))?;
                    (frame.data, frame.dim1, frame.dim2)
                }
                ImageInput::Array(array) => {
                    let (d1, d2) = extract_array_dims(&array, "float64")?;
                    (array.as_slice()?.to_vec(), d1, d2)
                }
            });
        }
        let slices: Vec<_> = buffers
            .iter()
            .map(|(data, d1, d2)| (data.as_slice(), *d1, *d2))
            .collect();
        let result = py
            .detach(|| {
                self.multi
                    .integrate2d_from_slices(&slices, n_radial, n_azimuthal, &options)
            })
            .map_err(PyValueError::new_err)?;
        let mut output = if let Some(errors) = result.errors {
            PyIntegrationResult::from_2d_with_errors(
                result.intensity,
                errors,
                n_radial,
                n_azimuthal,
            )
        } else {
            PyIntegrationResult::from_2d(result.intensity, n_radial, n_azimuthal)
        };
        output.radial = Some(result.radial);
        output.azimuthal = Some(result.azimuthal);
        Ok(output)
    }

    /// Create a new MultiIntegrator from a list of Integrator instances.
    ///
    /// All integrators must have the same radial_unit and radial_range set.
    /// The Histogram method is not supported.
    ///
    /// # Arguments
    ///
    /// * `integrators` - List of Integrator instances, one per detector
    /// * `pixel_split_mode` - "fractional" (default) or "pyfai_histogram" for
    ///   closer full-histogram compatibility in 2D only. Beam-containing pixels
    ///   use detector-plane sampling in both modes; negative weights are avoided.
    ///
    /// # Example
    ///
    /// ```python
    /// import dioptrin
    /// int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
    /// int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
    /// multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
    /// ```
    #[new]
    #[pyo3(signature = (integrators, *, pixel_split_mode="fractional"))]
    fn new(integrators: Vec<PyRef<'_, PyIntegrator>>, pixel_split_mode: &str) -> PyResult<Self> {
        let mode = match pixel_split_mode {
            "fractional" => PixelSplitMode::Fractional,
            "pyfai_histogram" => PixelSplitMode::PyfaiHistogram,
            _ => {
                return Err(PyValueError::new_err(
                    "pixel_split_mode must be 'fractional' or 'pyfai_histogram'",
                ));
            }
        };
        if integrators.is_empty() {
            return Err(PyValueError::new_err(
                "MultiIntegrator requires at least one integrator",
            ));
        }

        let arcs: Vec<_> = integrators
            .iter()
            .map(|py_int| py_int.inner().clone())
            .collect();

        let multi = MultiIntegrator::from_arcs(arcs)
            .map_err(PyValueError::new_err)?
            .with_pixel_split_mode(mode);

        Ok(Self { multi })
    }

    /// Integrate one image per detector and merge into a single 1D pattern.
    ///
    /// # Arguments
    ///
    /// * `images` - List of images (file paths or NumPy arrays), one per detector
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    ///
    /// # Returns
    ///
    /// IntegrationResult with merged radial values, intensities, and optional errors
    #[pyo3(signature = (images, n_points, errors=false))]
    fn integrate1d(
        &self,
        images: Vec<ImageInput>,
        n_points: usize,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let n_det = self.multi.integrators().len();
        if images.len() != n_det {
            return Err(PyValueError::new_err(format!(
                "Expected {} images (one per detector), got {}",
                n_det,
                images.len()
            )));
        }

        // Determine if all inputs are file paths or all are arrays
        let mut frames: Vec<Frame> = Vec::new();
        let mut slices_data: Vec<(Vec<f64>, usize, usize)> = Vec::new();
        let mut use_frames = true;

        for (i, image) in images.into_iter().enumerate() {
            match image {
                ImageInput::Path(tiff_path) => {
                    let frame = Frame::from_tiff(tiff_path).map_err(|err| {
                        PyValueError::new_err(format!(
                            "Failed to load TIFF for detector {}: {}",
                            i, err
                        ))
                    })?;
                    frames.push(frame);
                }
                ImageInput::Array(array) => {
                    use_frames = false;
                    let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                    let data = array
                        .as_slice()
                        .map_err(|err| {
                            PyValueError::new_err(format!("Array must be contiguous: {}", err))
                        })?
                        .to_vec();
                    slices_data.push((data, dim1, dim2));
                }
            }
        }

        if use_frames && !frames.is_empty() {
            let frame_refs: Vec<&Frame> = frames.iter().collect();
            if errors {
                let (radial, intensity, errs) =
                    self.multi.integrate1d_with_errors(&frame_refs, n_points);
                Ok(PyIntegrationResult::from_1d_with_errors(
                    radial, intensity, errs,
                ))
            } else {
                let (radial, intensity) = self.multi.integrate1d(&frame_refs, n_points);
                Ok(PyIntegrationResult::from_1d(radial, intensity))
            }
        } else {
            let slice_refs: Vec<(&[f64], usize, usize)> = slices_data
                .iter()
                .map(|(data, dim1, dim2)| (data.as_slice(), *dim1, *dim2))
                .collect();
            if errors {
                let (radial, intensity, errs) = self
                    .multi
                    .integrate1d_with_errors_from_slices(&slice_refs, n_points);
                Ok(PyIntegrationResult::from_1d_with_errors(
                    radial, intensity, errs,
                ))
            } else {
                let (radial, intensity) = self.multi.integrate1d_from_slices(&slice_refs, n_points);
                Ok(PyIntegrationResult::from_1d(radial, intensity))
            }
        }
    }

    /// Batch integrate multiple sets of detector images.
    ///
    /// # Arguments
    ///
    /// * `images` - List of image-lists, e.g. `[[det1_frame1, det2_frame1], ...]`
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute error estimates (default: False)
    ///
    /// # Returns
    ///
    /// List of IntegrationResult, one per frame set
    #[pyo3(signature = (images, n_points, errors=false))]
    fn batch1d(
        &self,
        images: Vec<Vec<ImageInput>>,
        n_points: usize,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let mut results = Vec::with_capacity(images.len());
        for (frame_idx, image_set) in images.into_iter().enumerate() {
            let n_det = self.multi.integrators().len();
            if image_set.len() != n_det {
                return Err(PyValueError::new_err(format!(
                    "Frame set {} has {} images, expected {} (one per detector)",
                    frame_idx,
                    image_set.len(),
                    n_det
                )));
            }

            let mut frames: Vec<Frame> = Vec::new();
            let mut slices_data: Vec<(Vec<f64>, usize, usize)> = Vec::new();
            let mut use_frames = true;

            for (i, image) in image_set.into_iter().enumerate() {
                match image {
                    ImageInput::Path(tiff_path) => {
                        let frame = Frame::from_tiff(tiff_path).map_err(|err| {
                            PyValueError::new_err(format!(
                                "Failed to load TIFF for detector {} frame {}: {}",
                                i, frame_idx, err
                            ))
                        })?;
                        frames.push(frame);
                    }
                    ImageInput::Array(array) => {
                        use_frames = false;
                        let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                        let data = array
                            .as_slice()
                            .map_err(|err| {
                                PyValueError::new_err(format!("Array must be contiguous: {}", err))
                            })?
                            .to_vec();
                        slices_data.push((data, dim1, dim2));
                    }
                }
            }

            if use_frames && !frames.is_empty() {
                let frame_refs: Vec<&Frame> = frames.iter().collect();
                if errors {
                    let (radial, intensity, errs) =
                        self.multi.integrate1d_with_errors(&frame_refs, n_points);
                    results.push(PyIntegrationResult::from_1d_with_errors(
                        radial, intensity, errs,
                    ));
                } else {
                    let (radial, intensity) = self.multi.integrate1d(&frame_refs, n_points);
                    results.push(PyIntegrationResult::from_1d(radial, intensity));
                }
            } else {
                let slice_refs: Vec<(&[f64], usize, usize)> = slices_data
                    .iter()
                    .map(|(data, dim1, dim2)| (data.as_slice(), *dim1, *dim2))
                    .collect();
                if errors {
                    let (radial, intensity, errs) = self
                        .multi
                        .integrate1d_with_errors_from_slices(&slice_refs, n_points);
                    results.push(PyIntegrationResult::from_1d_with_errors(
                        radial, intensity, errs,
                    ));
                } else {
                    let (radial, intensity) =
                        self.multi.integrate1d_from_slices(&slice_refs, n_points);
                    results.push(PyIntegrationResult::from_1d(radial, intensity));
                }
            }
        }
        Ok(results)
    }
}
