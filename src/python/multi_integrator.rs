use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::frame::Frame;
use crate::integration::PixelSplitMode;
use crate::multi::{Multi2dOptions, MultiIntegrator, MultiNormalization};

use super::integrator::{PyIntegrator, resolve_azimuthal_bins};
use super::results::PyIntegrationResult;
use super::utils::{ImageInput, check_mask_shape, check_positive, extract_array_dims, mask_shape};

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
    /// start of azimuthal_range, which then defaults to (0, 2*pi) and has an
    /// exclusive end, and the span must be a whole number of rows. A first
    /// centre in [-pi, 2*pi] is kept; any other is corrected to [-pi, pi).
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
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
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
        let buffers = self.load_images(images, "")?;
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
        py: Python<'_>,
        images: Vec<ImageInput>,
        n_points: usize,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        check_positive("n_points", n_points)?;
        let n_det = self.multi.integrators().len();
        if images.len() != n_det {
            return Err(PyValueError::new_err(format!(
                "Expected {} images (one per detector), got {}",
                n_det,
                images.len()
            )));
        }
        let buffers = self.load_images(images, "")?;
        Ok(self.merge1d(py, &buffers, n_points, errors))
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
        py: Python<'_>,
        images: Vec<Vec<ImageInput>>,
        n_points: usize,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        check_positive("n_points", n_points)?;
        let n_det = self.multi.integrators().len();
        let mut results = Vec::with_capacity(images.len());
        for (frame_idx, image_set) in images.into_iter().enumerate() {
            if image_set.len() != n_det {
                return Err(PyValueError::new_err(format!(
                    "Frame set {} has {} images, expected {} (one per detector)",
                    frame_idx,
                    image_set.len(),
                    n_det
                )));
            }
            let buffers = self.load_images(image_set, &format!("frame set {frame_idx}, "))?;
            results.push(self.merge1d(py, &buffers, n_points, errors));
        }
        Ok(results)
    }
}

/// One detector image as owned pixels and (dim1, dim2).
type ImageBuffer = (Vec<f64>, usize, usize);

impl PyMultiIntegrator {
    /// Load one image per detector (TIFF paths and arrays may be mixed) and
    /// check each against its detector's mask. `context` prefixes messages.
    fn load_images(&self, images: Vec<ImageInput>, context: &str) -> PyResult<Vec<ImageBuffer>> {
        images
            .into_iter()
            .zip(self.multi.integrators())
            .enumerate()
            .map(|(i, (image, integrator))| {
                let (data, dim1, dim2) = match image {
                    ImageInput::Path(path) => {
                        let frame = Frame::from_tiff(path).map_err(|err| {
                            PyValueError::new_err(format!(
                                "{context}detector {i}: failed to load TIFF: {err}"
                            ))
                        })?;
                        (frame.data, frame.dim1, frame.dim2)
                    }
                    ImageInput::Array(array) => {
                        let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                        let data = array.as_slice().map_err(|err| {
                            PyValueError::new_err(format!(
                                "{context}detector {i}: array must be contiguous: {err}"
                            ))
                        })?;
                        (data.to_vec(), dim1, dim2)
                    }
                };
                check_mask_shape(
                    mask_shape(integrator),
                    (dim1, dim2),
                    &format!("{context}detector {i} image"),
                )?;
                Ok((data, dim1, dim2))
            })
            .collect()
    }

    /// Merge one owned buffer per detector, releasing the GIL meanwhile.
    fn merge1d(
        &self,
        py: Python<'_>,
        buffers: &[ImageBuffer],
        n_points: usize,
        errors: bool,
    ) -> PyIntegrationResult {
        let slices: Vec<(&[f64], usize, usize)> = buffers
            .iter()
            .map(|(data, dim1, dim2)| (data.as_slice(), *dim1, *dim2))
            .collect();
        py.detach(|| {
            if errors {
                let (radial, intensity, errs) = self
                    .multi
                    .integrate1d_with_errors_from_slices(&slices, n_points);
                PyIntegrationResult::from_1d_with_errors(radial, intensity, errs)
            } else {
                let (radial, intensity) = self.multi.integrate1d_from_slices(&slices, n_points);
                PyIntegrationResult::from_1d(radial, intensity)
            }
        })
    }
}
