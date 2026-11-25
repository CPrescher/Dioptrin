use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::sync::{Arc, Mutex};

use crate::batch::{
    FileWorkers, Integration1D, Integration1DWithAzimuthalStd, Integration1DWithErrors,
    Integration2D, Integration2DOnAzimuthalRange, Integration2DWithErrors, IntegrationTask,
    StreamControl, map_in_pools, process_queue_with_pool,
};
use crate::frame::{Frame, TiffFrame};
use crate::geometry::Geometry;
use crate::integration::{IntegrationGrid2d, Integrator, MaskMode, RadialUnit, SparseLayout};
use crate::mask::Mask;
use crate::poni::Poni;

use super::results::{BatchEntry, BatchReceiver, PyBatchIterator, PyIntegrationResult};
use super::streaming::QueueStream;
use super::utils::{
    BatchImageInput, ClassifiedInput, ImageInput, StackImageInput, batch_images, batch2d_images,
    check_mask_shape, check_polarization_factor, check_positive, check_unit_wavelength,
    classify_batch_input, extract_array_dims, extract_item_dims, mask_shape, parse_method,
    parse_unit, poni_from_dict, resolve_workers_threads, validate_integrator_options,
};

/// Raise ValueError for an invalid `azimuthal_range=(min, max)` before a batch
/// starts, instead of one error per frame.
fn validate_azimuthal_range(
    azimuthal_range: Option<(f64, f64)>,
    n_azimuthal: usize,
) -> PyResult<()> {
    if let Some(range) = azimuthal_range {
        IntegrationGrid2d::new(1, n_azimuthal, (0.0, 1.0), range).map_err(PyValueError::new_err)?;
    }
    Ok(())
}

/// Resolve the azimuthal rows of a 2D call: either `n_azimuthal` (with
/// `azimuthal_range` as outer edges, as before) or `azimuthal_bracket` (rows of
/// centre ± bracket, first centre at the start of `azimuthal_range`, default
/// 0 to 2*pi). Returns the bin count and the range to integrate on.
pub(super) fn resolve_azimuthal_bins(
    n_azimuthal: Option<usize>,
    azimuthal_range: Option<(f64, f64)>,
    azimuthal_bracket: Option<f64>,
) -> PyResult<(usize, Option<(f64, f64)>)> {
    match (n_azimuthal, azimuthal_bracket) {
        (Some(n), None) => Ok((n, azimuthal_range)),
        (None, Some(bracket)) => {
            let centres = azimuthal_range.unwrap_or((0.0, std::f64::consts::TAU));
            let (n, edges) = crate::integration::azimuthal_bracket_bins(centres, bracket)
                .map_err(PyValueError::new_err)?;
            Ok((n, Some(edges)))
        }
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "give either n_azimuthal or azimuthal_bracket, not both",
        )),
        (None, None) => Err(PyValueError::new_err(
            "give either n_azimuthal or azimuthal_bracket",
        )),
    }
}

/// Integrate one image to 1D as `Integrator.integrate1d` does, without
/// touching Python: callers run it detached from the interpreter.
///
/// `batch1d`'s array branch deliberately keeps its own closure instead of
/// calling this. Merging the two, or adding any new call site into the 1D
/// `integrate1d*_from_slice` entry points, changed fat-LTO inlining when tried
/// and slowed the 1D pixel_split/pyfai_full kernels by up to ~20% on every 1D
/// path, default ones included. Re-benchmark any such refactor against
/// v0.14.0 on the release wheel build.
fn integrate1d_slice(
    integrator: &Integrator,
    data: &[f64],
    dim1: usize,
    dim2: usize,
    n_points: usize,
    errors: bool,
    azimuthal_std: bool,
) -> PyIntegrationResult {
    if azimuthal_std {
        PyIntegrationResult::from_azimuthal(
            integrator
                .integrate1d_with_azimuthal_std_from_slice(data, dim1, dim2, n_points, errors),
        )
    } else if errors {
        let (radial, intensity, errs) =
            integrator.integrate1d_with_errors_from_slice(data, dim1, dim2, n_points);
        PyIntegrationResult::from_1d_with_errors(radial, intensity, errs)
    } else {
        let (radial, intensity) = integrator.integrate1d_from_slice(data, dim1, dim2, n_points);
        PyIntegrationResult::from_1d(radial, intensity)
    }
}

/// Binning of one Python 2D call, after its arguments were resolved.
#[derive(Clone, Copy)]
struct Bins2d {
    n_radial: usize,
    n_azimuthal: usize,
    errors: bool,
    azimuthal_range: Option<(f64, f64)>,
}

/// Integrate one image to 2D as `Integrator.integrate2d` does, without
/// touching Python: callers run it detached from the interpreter. `radial`
/// is the radial axis to report for an explicit radial range. Invalid input
/// on an azimuthal range is returned as `Err`.
fn integrate2d_slice(
    integrator: &Integrator,
    data: &[f64],
    dim1: usize,
    dim2: usize,
    bins: Bins2d,
    radial: Option<Vec<f64>>,
) -> Result<PyIntegrationResult, String> {
    let Bins2d {
        n_radial,
        n_azimuthal,
        errors,
        azimuthal_range,
    } = bins;
    Ok(match (errors, azimuthal_range) {
        (false, None) => PyIntegrationResult::from_2d(
            integrator.integrate2d_from_slice(data, dim1, dim2, n_radial, n_azimuthal),
            n_radial,
            n_azimuthal,
        )
        .with_radial(radial),
        (true, None) => {
            let (intensity, errs) = integrator.integrate2d_with_errors_from_slice(
                data,
                dim1,
                dim2,
                n_radial,
                n_azimuthal,
            );
            PyIntegrationResult::from_2d_with_errors(intensity, errs, n_radial, n_azimuthal)
                .with_radial(radial)
        }
        (errors, Some(range)) => PyIntegrationResult::from_azimuthal_range_2d(
            integrator.integrate2d_on_azimuthal_range_from_slice(
                data,
                dim1,
                dim2,
                n_radial,
                n_azimuthal,
                range,
                errors,
            )?,
            n_radial,
            n_azimuthal,
        ),
    })
}

fn parse_mask_mode(value: &str) -> PyResult<MaskMode> {
    match value {
        "fixed" => Ok(MaskMode::Fixed),
        "dynamic" => Ok(MaskMode::Dynamic),
        _ => Err(PyValueError::new_err(
            "mask_mode must be 'fixed' or 'dynamic'",
        )),
    }
}

fn parse_sparse_layout(value: &str) -> PyResult<SparseLayout> {
    match value {
        "csc" => Ok(SparseLayout::Csc),
        "csr" => Ok(SparseLayout::Csr),
        _ => Err(PyValueError::new_err(
            "sparse_layout must be 'csc' or 'csr'",
        )),
    }
}

#[derive(FromPyObject)]
pub enum MaskInput<'p> {
    Path(String),
    ArrayU8(PyReadonlyArray2<'p, u8>),
    ArrayBool(PyReadonlyArray2<'p, bool>),
}

#[pyclass(name = "Integrator", unsendable)]
pub struct PyIntegrator {
    pub(super) integrator: Arc<Integrator>,
    pub(super) file_workers: Mutex<Option<Arc<FileWorkers>>>,
    /// Workers of the last Python-iterator stream, with that stream's control;
    /// lent to the next such stream once that stream's workers are done.
    pub(super) iterator_workers: Mutex<Option<(Arc<FileWorkers>, Arc<StreamControl>)>>,
}

impl PyIntegrator {
    pub fn inner(&self) -> &Arc<Integrator> {
        &self.integrator
    }

    /// Shared constructor body: validates options core would accept unchecked.
    #[allow(clippy::too_many_arguments)]
    fn build(
        poni: Poni,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
        radial_range: Option<(f64, f64)>,
        fast_math: bool,
        mask_mode: &str,
        sparse_layout: &str,
    ) -> PyResult<Self> {
        let unit = parse_unit(unit)?;
        validate_integrator_options(&poni, unit, polarization_factor, radial_range)?;
        let method = parse_method(method, n_ss)?;
        let geometry = Geometry::new(0, 0, poni);
        let mut integrator = Integrator::new(geometry)
            .with_method(method)
            .with_radial_unit(unit)
            .with_fast_math(fast_math)
            .with_mask_mode(parse_mask_mode(mask_mode)?)
            .with_sparse_layout(parse_sparse_layout(sparse_layout)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }
        if let Some((min, max)) = radial_range {
            integrator = integrator.with_radial_range(min, max);
        }

        Ok(Self {
            integrator: Arc::new(integrator),
            file_workers: Mutex::new(None),
            iterator_workers: Mutex::new(None),
        })
    }

    /// Radial bin centres of a 2D call with an explicit radial range (None
    /// without one, where 2D results keep `radial = None`). Raises ValueError
    /// when the range cannot serve as 2D radial edges (e.g. a negative
    /// minimum), before any work starts.
    fn radial_axis_2d(&self, n_radial: usize) -> PyResult<Option<Vec<f64>>> {
        if self.integrator.radial_range().is_none() {
            return Ok(None);
        }
        self.integrator
            .radial_centers_2d(0, 0, n_radial)
            .map(Some)
            .map_err(PyValueError::new_err)
    }

    /// Raise ValueError when an image does not match the stored mask.
    pub(super) fn check_mask(&self, image: (usize, usize), label: &str) -> PyResult<()> {
        check_mask_shape(mask_shape(&self.integrator), image, label)
    }

    /// Single-frame native-pixel (`f32`/`u16`/`u32`/`i32`) 1D integration.
    fn integrate1d_array_pixels<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        array: &PyReadonlyArray2<'_, P>,
        type_name: &str,
        n_points: usize,
        errors: bool,
        azimuthal_std: bool,
    ) -> PyResult<PyIntegrationResult> {
        let (dim1, dim2) = extract_array_dims(array, type_name)?;
        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        self.check_mask((dim1, dim2), "image")?;
        let integrator = &*self.integrator;
        Ok(py.detach(|| {
            if azimuthal_std {
                let converted: Vec<f64> = slice.iter().map(|&x| x.to_f64()).collect();
                PyIntegrationResult::from_azimuthal(
                    integrator.integrate1d_with_azimuthal_std_from_slice(
                        &converted, dim1, dim2, n_points, errors,
                    ),
                )
            } else {
                let (radial, intensity, errs) =
                    integrator.integrate1d_pixels_from_slice(slice, dim1, dim2, n_points, errors);
                match errs {
                    Some(e) => PyIntegrationResult::from_1d_with_errors(radial, intensity, e),
                    None => PyIntegrationResult::from_1d(radial, intensity),
                }
            }
        }))
    }

    /// Single-frame native-pixel (`f32`/`u16`/`u32`/`i32`) 2D integration.
    fn integrate2d_array_pixels<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        array: &PyReadonlyArray2<'_, P>,
        type_name: &str,
        bins: Bins2d,
        radial: Option<Vec<f64>>,
    ) -> PyResult<PyIntegrationResult> {
        let (dim1, dim2) = extract_array_dims(array, type_name)?;
        let slice = array
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Array must be contiguous: {}", err)))?;
        self.check_mask((dim1, dim2), "image")?;
        let integrator = &*self.integrator;
        py.detach(|| -> Result<PyIntegrationResult, String> {
            let (intensity, errs, axes) = integrator.integrate2d_pixels_from_slice(
                slice,
                dim1,
                dim2,
                bins.n_radial,
                bins.n_azimuthal,
                bins.azimuthal_range,
                bins.errors,
            )?;
            let result = if let Some((rad, az)) = axes {
                PyIntegrationResult::from_azimuthal_range_2d(
                    crate::integration::AzimuthalRange2dResult {
                        intensity,
                        errors: errs,
                        radial: rad,
                        azimuthal: az,
                    },
                    bins.n_radial,
                    bins.n_azimuthal,
                )
            } else {
                match errs {
                    Some(e) => PyIntegrationResult::from_2d_with_errors(
                        intensity,
                        e,
                        bins.n_radial,
                        bins.n_azimuthal,
                    )
                    .with_radial(radial),
                    None => {
                        PyIntegrationResult::from_2d(intensity, bins.n_radial, bins.n_azimuthal)
                            .with_radial(radial)
                    }
                }
            };
            Ok(result)
        })
        .map_err(PyValueError::new_err)
    }

    /// Zero-intermediate-copy 2D batch integration directly into a contiguous
    /// `(n_frames, n_azimuthal, n_radial)` NumPy array (and optional errors array).
    fn batch2d_stack_slices<P: crate::frame::Pixel>(
        &self,
        py: Python<'_>,
        frames: &[&[P]],
        dim1: usize,
        dim2: usize,
        bins: Bins2d,
    ) -> PyResult<Py<PyAny>> {
        let n_frames = frames.len();
        let total_bins = bins
            .n_radial
            .checked_mul(bins.n_azimuthal)
            .ok_or_else(|| PyValueError::new_err("n_radial * n_azimuthal overflows usize"))?;
        let intensity =
            unsafe { PyArray3::<f64>::new(py, [n_frames, bins.n_azimuthal, bins.n_radial], false) };
        if n_frames == 0 {
            return if bins.errors {
                let err_arr = unsafe {
                    PyArray3::<f64>::new(py, [0, bins.n_azimuthal, bins.n_radial], false)
                };
                Ok((intensity.unbind(), err_arr.unbind())
                    .into_pyobject(py)?
                    .into_any()
                    .unbind())
            } else {
                Ok(intensity.unbind().into_any())
            };
        }

        use rayon::prelude::*;
        let integrator = &*self.integrator;

        let run_fill = |i_slice: &mut [f64],
                        mut e_slice: Option<&mut [f64]>|
         -> Result<(), String> {
            integrator.prepare2d(
                dim1,
                dim2,
                bins.n_radial,
                bins.n_azimuthal,
                bins.azimuthal_range,
            );
            let min_tiled = 4 * rayon::current_num_threads().max(1);
            let use_tiled = frames.len() >= min_tiled
                && integrator.can_integrate2d_batch4(
                    dim1,
                    dim2,
                    bins.n_radial,
                    bins.n_azimuthal,
                    bins.azimuthal_range,
                    bins.errors,
                );
            let n_tiled = if use_tiled { (frames.len() / 4) * 4 } else { 0 };

            if n_tiled > 0 {
                let (i_tiled, i_tail) = i_slice.split_at_mut(n_tiled * total_bins);
                if let Some(e_all) = e_slice.as_deref_mut() {
                    let (e_tiled, e_tail) = e_all.split_at_mut(n_tiled * total_bins);
                    frames[..n_tiled]
                        .par_chunks_exact(4)
                        .zip(i_tiled.par_chunks_exact_mut(4 * total_bins))
                        .zip(e_tiled.par_chunks_exact_mut(4 * total_bins))
                        .try_for_each(|((f_chunk, i_chunk), e_chunk)| -> Result<(), String> {
                            let (y0, rest) = i_chunk.split_at_mut(total_bins);
                            let (y1, rest) = rest.split_at_mut(total_bins);
                            let (y2, y3) = rest.split_at_mut(total_bins);
                            let (e0, rest) = e_chunk.split_at_mut(total_bins);
                            let (e1, rest) = rest.split_at_mut(total_bins);
                            let (e2, e3) = rest.split_at_mut(total_bins);
                            let ok = integrator.try_integrate2d_batch4_into(
                                [f_chunk[0], f_chunk[1], f_chunk[2], f_chunk[3]],
                                dim1,
                                dim2,
                                bins.n_radial,
                                bins.n_azimuthal,
                                bins.azimuthal_range,
                                [y0, y1, y2, y3],
                                Some([e0, e1, e2, e3]),
                            );
                            if !ok {
                                return Err(
                                    "2D batch4 tiled kernel unexpectedly declined chunk".into()
                                );
                            }
                            Ok(())
                        })?;
                    if n_tiled < frames.len() {
                        frames[n_tiled..]
                            .par_iter()
                            .zip(i_tail.par_chunks_exact_mut(total_bins))
                            .zip(e_tail.par_chunks_exact_mut(total_bins))
                            .try_for_each(|((&data, dst_i), dst_e)| -> Result<(), String> {
                                let (y, e, _) = integrator.integrate2d_pixels_from_slice(
                                    data,
                                    dim1,
                                    dim2,
                                    bins.n_radial,
                                    bins.n_azimuthal,
                                    bins.azimuthal_range,
                                    true,
                                )?;
                                dst_i.copy_from_slice(&y);
                                dst_e.copy_from_slice(e.as_ref().expect("errors requested"));
                                Ok(())
                            })?;
                    }
                } else {
                    frames[..n_tiled]
                        .par_chunks_exact(4)
                        .zip(i_tiled.par_chunks_exact_mut(4 * total_bins))
                        .try_for_each(|(f_chunk, i_chunk)| -> Result<(), String> {
                            let (y0, rest) = i_chunk.split_at_mut(total_bins);
                            let (y1, rest) = rest.split_at_mut(total_bins);
                            let (y2, y3) = rest.split_at_mut(total_bins);
                            let ok = integrator.try_integrate2d_batch4_into(
                                [f_chunk[0], f_chunk[1], f_chunk[2], f_chunk[3]],
                                dim1,
                                dim2,
                                bins.n_radial,
                                bins.n_azimuthal,
                                bins.azimuthal_range,
                                [y0, y1, y2, y3],
                                None,
                            );
                            if !ok {
                                return Err(
                                    "2D batch4 tiled kernel unexpectedly declined chunk".into()
                                );
                            }
                            Ok(())
                        })?;
                    if n_tiled < frames.len() {
                        frames[n_tiled..]
                            .par_iter()
                            .zip(i_tail.par_chunks_exact_mut(total_bins))
                            .try_for_each(|(&data, dst_i)| -> Result<(), String> {
                                let (y, _, _) = integrator.integrate2d_pixels_from_slice(
                                    data,
                                    dim1,
                                    dim2,
                                    bins.n_radial,
                                    bins.n_azimuthal,
                                    bins.azimuthal_range,
                                    false,
                                )?;
                                dst_i.copy_from_slice(&y);
                                Ok(())
                            })?;
                    }
                }
            } else if let Some(e_all) = e_slice.as_deref_mut() {
                frames
                    .par_iter()
                    .zip(i_slice.par_chunks_exact_mut(total_bins))
                    .zip(e_all.par_chunks_exact_mut(total_bins))
                    .try_for_each(|((&data, dst_i), dst_e)| -> Result<(), String> {
                        let (y, e, _) = integrator.integrate2d_pixels_from_slice(
                            data,
                            dim1,
                            dim2,
                            bins.n_radial,
                            bins.n_azimuthal,
                            bins.azimuthal_range,
                            true,
                        )?;
                        dst_i.copy_from_slice(&y);
                        dst_e.copy_from_slice(e.as_ref().expect("errors requested"));
                        Ok(())
                    })?;
            } else {
                frames
                    .par_iter()
                    .zip(i_slice.par_chunks_exact_mut(total_bins))
                    .try_for_each(|(&data, dst_i)| -> Result<(), String> {
                        let (y, _, _) = integrator.integrate2d_pixels_from_slice(
                            data,
                            dim1,
                            dim2,
                            bins.n_radial,
                            bins.n_azimuthal,
                            bins.azimuthal_range,
                            false,
                        )?;
                        dst_i.copy_from_slice(&y);
                        Ok(())
                    })?;
            }
            Ok(())
        };

        if bins.errors {
            let err_arr = unsafe {
                PyArray3::<f64>::new(py, [n_frames, bins.n_azimuthal, bins.n_radial], false)
            };
            {
                let mut i_rw = intensity.readwrite();
                let mut e_rw = err_arr.readwrite();
                let i_slice = i_rw.as_slice_mut().map_err(|e| {
                    PyRuntimeError::new_err(format!("intensity output not contiguous: {e}"))
                })?;
                let e_slice = e_rw.as_slice_mut().map_err(|e| {
                    PyRuntimeError::new_err(format!("errors output not contiguous: {e}"))
                })?;
                py.detach(|| run_fill(i_slice, Some(e_slice)))
                    .map_err(PyValueError::new_err)?;
            }
            Ok((intensity.unbind(), err_arr.unbind())
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            {
                let mut i_rw = intensity.readwrite();
                let i_slice = i_rw.as_slice_mut().map_err(|e| {
                    PyRuntimeError::new_err(format!("intensity output not contiguous: {e}"))
                })?;
                py.detach(|| run_fill(i_slice, None))
                    .map_err(PyValueError::new_err)?;
            }
            Ok(intensity.unbind().into_any())
        }
    }

    fn batch2d_stack_3d<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        stack: &PyReadonlyArray3<'_, P>,
        bins: Bins2d,
    ) -> PyResult<Py<PyAny>> {
        let view = stack.as_array();
        if !view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "3D image stack must be C-contiguous (row-major)",
            ));
        }
        let shape = view.shape();
        let (n_frames, dim1, dim2) = (shape[0], shape[1], shape[2]);
        if dim1 == 0 || dim2 == 0 {
            return Err(PyValueError::new_err(format!(
                "Image frames must be non-empty, got shape ({dim1}, {dim2})"
            )));
        }
        self.check_mask((dim1, dim2), "images")?;
        let raw = stack
            .as_slice()
            .map_err(|err| PyValueError::new_err(format!("Stack must be contiguous: {err}")))?;
        let frame_len = dim1 * dim2;
        let frames: Vec<&[P]> = (0..n_frames)
            .map(|i| &raw[i * frame_len..(i + 1) * frame_len])
            .collect();
        self.batch2d_stack_slices(py, &frames, dim1, dim2, bins)
    }

    fn batch2d_stack_list<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        arrays: &[PyReadonlyArray2<'_, P>],
        bins: Bins2d,
    ) -> PyResult<Py<PyAny>> {
        if arrays.is_empty() {
            return self.batch2d_stack_slices::<P>(py, &[], 0, 0, bins);
        }
        let mut frames = Vec::with_capacity(arrays.len());
        let mut first_shape: Option<(usize, usize)> = None;
        for (i, array) in arrays.iter().enumerate() {
            let (dim1, dim2) = extract_item_dims(array, i)?;
            let slice = array.as_slice().map_err(|err| {
                PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
            })?;
            self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
            if let Some((h, w)) = first_shape {
                if (dim1, dim2) != (h, w) {
                    return Err(PyValueError::new_err(format!(
                        "images[{i}] has shape ({dim1}, {dim2}), expected uniform shape ({h}, {w})"
                    )));
                }
            } else {
                first_shape = Some((dim1, dim2));
            }
            frames.push(slice);
        }
        let (dim1, dim2) = first_shape.unwrap();
        self.batch2d_stack_slices(py, &frames, dim1, dim2, bins)
    }

    /// Native-pixel (`f32`/`u16`/`u32`/`i32`) 1D array batch path.
    ///
    /// Kept separate from `batch1d`'s `f64` arm to preserve fat-LTO inlining on
    /// the hot `f64` 1D entry points.
    #[allow(clippy::too_many_arguments)]
    fn batch1d_arrays_pixels<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        arrays: &[PyReadonlyArray2<'_, P>],
        n_points: usize,
        errors: bool,
        azimuthal_std: bool,
        split: Option<(usize, usize)>,
    ) -> PyResult<Vec<BatchEntry>> {
        let frames = arrays
            .iter()
            .enumerate()
            .map(|(i, array)| {
                let (dim1, dim2) = extract_item_dims(array, i)?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
                })?;
                self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
                Ok((slice, dim1, dim2))
            })
            .collect::<PyResult<Vec<_>>>()?;

        use rayon::prelude::*;
        let integrator = &*self.integrator;
        let integrate = |(i, &(data, dim1, dim2)): (usize, &(&[P], usize, usize))| {
            let integration_result = if azimuthal_std {
                let converted: Vec<f64> = data.iter().map(|&x| x.to_f64()).collect();
                PyIntegrationResult::from_azimuthal(
                    integrator.integrate1d_with_azimuthal_std_from_slice(
                        &converted, dim1, dim2, n_points, errors,
                    ),
                )
            } else {
                let (radial, intensity, errs) =
                    integrator.integrate1d_pixels_from_slice(data, dim1, dim2, n_points, errors);
                match errs {
                    Some(e) => PyIntegrationResult::from_1d_with_errors(radial, intensity, e),
                    None => PyIntegrationResult::from_1d(radial, intensity),
                }
            };
            BatchEntry {
                filename: format!("array[{}]", i),
                result: Some(integration_result),
                error: None,
            }
        };

        if let Some((num_workers, num_threads)) = split {
            let prepare = || {
                if let Some(&(_, dim1, dim2)) = frames.first()
                    && frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
                {
                    integrator.prepare1d(dim1, dim2, n_points);
                }
            };
            let outcomes = py
                .detach(|| {
                    map_in_pools(frames.len(), num_workers, num_threads, prepare, |i| {
                        let mut one: Vec<BatchEntry> = frames[i..=i]
                            .par_iter()
                            .enumerate()
                            .map(integrate)
                            .collect();
                        let mut result = one.pop().expect("one frame, one result");
                        result.filename = format!("array[{}]", i);
                        result
                    })
                })
                .map_err(PyValueError::new_err)?;
            return Ok(outcomes
                .into_iter()
                .map(|outcome| outcome.unwrap_or_else(|payload| std::panic::resume_unwind(payload)))
                .collect());
        }

        let results = py.detach(|| {
            let uniform = frames
                .first()
                .filter(|&&(_, dim1, dim2)| frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2)))
                .map(|&(_, dim1, dim2)| (dim1, dim2));
            if let Some((dim1, dim2)) = uniform {
                integrator.prepare1d(dim1, dim2, n_points);
                let min_tiled = 4 * rayon::current_num_threads().max(1);
                if !azimuthal_std
                    && frames.len() >= min_tiled
                    && integrator.can_integrate1d_batch4(dim1, dim2, n_points, errors)
                {
                    let n_tiled = (frames.len() / 4) * 4;
                    let mut out: Vec<BatchEntry> = frames[..n_tiled]
                        .par_chunks_exact(4)
                        .enumerate()
                        .flat_map(|(chunk_idx, chunk)| {
                            let base = chunk_idx * 4;
                            let batch = integrator
                                .try_integrate1d_batch4(
                                    [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0],
                                    dim1,
                                    dim2,
                                    n_points,
                                    errors,
                                )
                                .expect("can_integrate1d_batch4 pre-checked eligibility");
                            batch
                                .into_iter()
                                .enumerate()
                                .map(|(j, (radial, intensity, errs))| {
                                    let res = match errs {
                                        Some(e) => PyIntegrationResult::from_1d_with_errors(
                                            radial, intensity, e,
                                        ),
                                        None => PyIntegrationResult::from_1d(radial, intensity),
                                    };
                                    BatchEntry {
                                        filename: format!("array[{}]", base + j),
                                        result: Some(res),
                                        error: None,
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    if n_tiled < frames.len() {
                        out.par_extend(
                            frames[n_tiled..]
                                .par_iter()
                                .enumerate()
                                .map(|(j, item)| integrate((n_tiled + j, item))),
                        );
                    }
                    return out;
                }
            }
            frames.par_iter().enumerate().map(integrate).collect()
        });

        Ok(results)
    }

    /// `batch2d` over a list of arrays: every array is checked first (dtype,
    /// shape, contiguity, mask), then the frames are integrated without
    /// copying and with the GIL released. The read-only borrow guards in
    /// `arrays` outlive the workers; only plain slices cross into them. An
    /// invalid azimuthal range never gets here: the caller
    /// (`validate_azimuthal_range`) raises ValueError up front, as for TIFF
    /// paths.
    ///
    /// `split` is `None` when the caller gave neither `num_workers` nor
    /// `num_threads`: the frames then run as in `batch1d`'s array branch, a
    /// `par_iter` on the global Rayon pool. Otherwise `Some((workers,
    /// threads))`: at most `workers` frames at a time, each in its own Rayon
    /// pool of `threads` threads ([`map_in_pools`]); a panic in one frame is
    /// raised after all frames finished. Only the per-frame closure is 2D
    /// specific, so 1D could use the same structure.
    fn batch2d_arrays<P: crate::frame::Pixel + numpy::Element>(
        &self,
        py: Python<'_>,
        arrays: &[PyReadonlyArray2<'_, P>],
        bins: Bins2d,
        radial: Option<Vec<f64>>,
        split: Option<(usize, usize)>,
    ) -> PyResult<Vec<BatchEntry>> {
        let frames = arrays
            .iter()
            .enumerate()
            .map(|(i, array)| {
                let (dim1, dim2) = extract_item_dims(array, i)?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
                })?;
                self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
                Ok((slice, dim1, dim2))
            })
            .collect::<PyResult<Vec<_>>>()?;

        use rayon::prelude::*;
        let integrator = &*self.integrator;
        let integrate = |i: usize| {
            let (data, dim1, dim2) = frames[i];
            let filename = format!("array[{}]", i);
            match integrator.integrate2d_pixels_from_slice(
                data,
                dim1,
                dim2,
                bins.n_radial,
                bins.n_azimuthal,
                bins.azimuthal_range,
                bins.errors,
            ) {
                Ok((intensity, errs, axes)) => {
                    let result = if let Some((rad, az)) = axes {
                        PyIntegrationResult::from_azimuthal_range_2d(
                            crate::integration::AzimuthalRange2dResult {
                                intensity,
                                errors: errs,
                                radial: rad,
                                azimuthal: az,
                            },
                            bins.n_radial,
                            bins.n_azimuthal,
                        )
                    } else {
                        match errs {
                            Some(e) => PyIntegrationResult::from_2d_with_errors(
                                intensity,
                                e,
                                bins.n_radial,
                                bins.n_azimuthal,
                            )
                            .with_radial(radial.clone()),
                            None => PyIntegrationResult::from_2d(
                                intensity,
                                bins.n_radial,
                                bins.n_azimuthal,
                            )
                            .with_radial(radial.clone()),
                        }
                    };
                    BatchEntry {
                        filename,
                        result: Some(result),
                        error: None,
                    }
                }
                Err(error) => BatchEntry {
                    filename,
                    result: None,
                    error: Some(error),
                },
            }
        };
        // Build the table once before the frames run, when they share a shape.
        let prepare = || {
            if let Some(&(_, dim1, dim2)) = frames.first()
                && frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
            {
                integrator.prepare2d(
                    dim1,
                    dim2,
                    bins.n_radial,
                    bins.n_azimuthal,
                    bins.azimuthal_range,
                );
            }
        };
        let Some((workers, threads)) = split else {
            return Ok(py.detach(|| {
                prepare();
                let uniform = frames
                    .first()
                    .filter(|&&(_, dim1, dim2)| {
                        frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
                    })
                    .map(|&(_, dim1, dim2)| (dim1, dim2));
                if let Some((dim1, dim2)) = uniform {
                    let min_tiled = 4 * rayon::current_num_threads().max(1);
                    if frames.len() >= min_tiled
                        && integrator.can_integrate2d_batch4(
                            dim1,
                            dim2,
                            bins.n_radial,
                            bins.n_azimuthal,
                            bins.azimuthal_range,
                            bins.errors,
                        )
                    {
                        let az_axes = bins.azimuthal_range.and_then(|range| {
                            integrator.azimuthal_range_axes_2d(
                                dim1,
                                dim2,
                                bins.n_radial,
                                bins.n_azimuthal,
                                range,
                            )
                        });
                        let n_tiled = (frames.len() / 4) * 4;
                        let mut out: Vec<BatchEntry> = frames[..n_tiled]
                            .par_chunks_exact(4)
                            .enumerate()
                            .flat_map(|(chunk_idx, chunk)| {
                                let base = chunk_idx * 4;
                                let batch = integrator
                                    .try_integrate2d_batch4(
                                        [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0],
                                        dim1,
                                        dim2,
                                        bins.n_radial,
                                        bins.n_azimuthal,
                                        bins.azimuthal_range,
                                        bins.errors,
                                    )
                                    .expect("can_integrate2d_batch4 pre-checked eligibility");
                                batch
                                    .into_iter()
                                    .enumerate()
                                    .map(|(j, (intensity, errs))| {
                                        let res = if let Some((ref rad, ref az)) = az_axes {
                                            PyIntegrationResult::from_azimuthal_range_2d(
                                                crate::integration::AzimuthalRange2dResult {
                                                    intensity,
                                                    errors: errs,
                                                    radial: rad.clone(),
                                                    azimuthal: az.clone(),
                                                },
                                                bins.n_radial,
                                                bins.n_azimuthal,
                                            )
                                        } else {
                                            match errs {
                                                Some(e) => {
                                                    PyIntegrationResult::from_2d_with_errors(
                                                        intensity,
                                                        e,
                                                        bins.n_radial,
                                                        bins.n_azimuthal,
                                                    )
                                                    .with_radial(radial.clone())
                                                }
                                                None => PyIntegrationResult::from_2d(
                                                    intensity,
                                                    bins.n_radial,
                                                    bins.n_azimuthal,
                                                )
                                                .with_radial(radial.clone()),
                                            }
                                        };
                                        BatchEntry {
                                            filename: format!("array[{}]", base + j),
                                            result: Some(res),
                                            error: None,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                        if n_tiled < frames.len() {
                            out.par_extend((n_tiled..frames.len()).into_par_iter().map(integrate));
                        }
                        return out;
                    }
                }
                (0..frames.len()).into_par_iter().map(integrate).collect()
            }));
        };
        let outcomes = py
            .detach(|| map_in_pools(frames.len(), workers, threads, prepare, integrate))
            .map_err(PyValueError::new_err)?;
        // Re-raise the first panic in input order, as the default path would.
        Ok(outcomes
            .into_iter()
            .map(|outcome| outcome.unwrap_or_else(|payload| std::panic::resume_unwind(payload)))
            .collect())
    }
}

#[pymethods]
impl PyIntegrator {
    /// Build one immutable geometry/weight/correction snapshot shared by frame workers.
    /// The stored mask is excluded: each worker receives its full mask per call.
    /// Changes to this Integrator do not change previously prepared plans.
    fn prepare_dynamic1d(
        &self,
        py: Python<'_>,
        shape: (usize, usize),
        n_points: usize,
    ) -> PyResult<super::dynamic::PyDynamic1dPlan> {
        let plan = py
            .detach(|| self.integrator.prepare_dynamic1d(shape, n_points))
            .map_err(PyValueError::new_err)?;
        Ok(super::dynamic::PyDynamic1dPlan {
            plan: Arc::new(plan),
        })
    }

    /// Experimental: upload this integrator's fixed-mask 1D table to a CUDA
    /// device. The plan is a snapshot of the current mask, corrections,
    /// method, unit and radial range; `fast_math` and `sparse_layout` are
    /// ignored. Raises `CudaUnavailableError` when this build has no CUDA
    /// support or the device cannot be used, `ValueError` for an unsupported
    /// configuration (dynamic mask or histogram method) or
    /// invalid input. The GIL is released while preparing.
    #[pyo3(signature = (shape, n_points, *, precision="f32_compensated", device=0, max_batch_bytes=None))]
    fn prepare_cuda1d(
        &self,
        py: Python<'_>,
        shape: (usize, usize),
        n_points: usize,
        precision: &str,
        device: usize,
        max_batch_bytes: Option<usize>,
    ) -> PyResult<super::cuda::PyCuda1dPlan> {
        super::cuda::prepare(
            py,
            &self.integrator,
            shape,
            n_points,
            precision,
            device,
            max_batch_bytes,
        )
    }

    /// Experimental: upload this integrator's fixed-mask 2D table to a CUDA
    /// device. Mirrors `prepare_cuda1d` for 2D azimuthal integration.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (shape, n_radial, n_azimuthal=None, *, azimuthal_range=None, azimuthal_bracket=None, precision="f32_compensated", device=0, max_batch_bytes=None))]
    fn prepare_cuda2d(
        &self,
        py: Python<'_>,
        shape: (usize, usize),
        n_radial: usize,
        n_azimuthal: Option<usize>,
        azimuthal_range: Option<(f64, f64)>,
        azimuthal_bracket: Option<f64>,
        precision: &str,
        device: usize,
        max_batch_bytes: Option<usize>,
    ) -> PyResult<super::cuda::PyCuda2dPlan> {
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        super::cuda::prepare_2d(
            py,
            &self.integrator,
            shape,
            n_radial,
            n_azimuthal,
            azimuthal_range,
            precision,
            device,
            max_batch_bytes,
        )
    }

    /// Experimental: upload this integrator's fixed-mask 1D table to a Metal
    /// GPU device on macOS.
    #[pyo3(signature = (shape, n_points, *, precision="f32_compensated", device=0, max_batch_bytes=None))]
    fn prepare_metal1d(
        &self,
        py: Python<'_>,
        shape: (usize, usize),
        n_points: usize,
        precision: &str,
        device: usize,
        max_batch_bytes: Option<usize>,
    ) -> PyResult<super::metal::PyMetal1dPlan> {
        super::metal::prepare(
            py,
            &self.integrator,
            shape,
            n_points,
            precision,
            device,
            max_batch_bytes,
        )
    }

    /// Experimental: upload this integrator's fixed-mask 2D table to a Metal
    /// GPU device on macOS. Mirrors `prepare_metal1d` for 2D azimuthal integration.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (shape, n_radial, n_azimuthal=None, *, azimuthal_range=None, azimuthal_bracket=None, precision="f32_compensated", device=0, max_batch_bytes=None))]
    fn prepare_metal2d(
        &self,
        py: Python<'_>,
        shape: (usize, usize),
        n_radial: usize,
        n_azimuthal: Option<usize>,
        azimuthal_range: Option<(f64, f64)>,
        azimuthal_bracket: Option<f64>,
        precision: &str,
        device: usize,
        max_batch_bytes: Option<usize>,
    ) -> PyResult<super::metal::PyMetal2dPlan> {
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        super::metal::prepare_2d(
            py,
            &self.integrator,
            shape,
            n_radial,
            n_azimuthal,
            azimuthal_range,
            precision,
            device,
            max_batch_bytes,
        )
    }

    /// Release cached file workers and decoding buffers: the pool of the
    /// synchronous and list batches, and the workers kept from the last
    /// finished generator-fed stream for the next one. Admitted streaming work
    /// finishes normally; the next batch or stream creates a new pool.
    fn clear_file_workers(&self) {
        self.file_workers
            .lock()
            .expect("file worker cache lock poisoned")
            .take();
        self.iterator_workers
            .lock()
            .expect("iterator worker cache lock poisoned")
            .take();
    }

    /// Create a new Integrator instance.
    ///
    /// The integrator is configured with detector geometry from a PONI file.
    /// The dimensions will be determined automatically when the first image
    /// is integrated.
    ///
    /// # Arguments
    ///
    /// * `poni_path` - Path to the PONI calibration file (v1 or v2 format)
    /// * `method` - Integration method: "pixel_split" (default), "pyfai_full", "counting", or "supersampled"
    /// * `polarization_factor` - Optional polarization factor P in [-1, 1]: +1 fully
    ///   polarised along the horizontal axis (chi = 0, detector dim2; the usual
    ///   synchrotron case, typically 0.95-0.99), -1 vertical, 0 unpolarised
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    /// * `radial_range` - Optional explicit (min, max) radial bin range in `unit`.
    ///   Not supported with "histogram", which always uses the automatic range.
    ///
    /// # Raises
    ///
    /// ValueError for invalid options, including `radial_range` combined with
    /// the "histogram" method.
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (poni_path, method=None, polarization_factor=None, n_ss=None, unit=None, radial_range=None, *, fast_math=true, mask_mode="fixed", sparse_layout="csc"))]
    fn new(
        poni_path: &str,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
        radial_range: Option<(f64, f64)>,
        fast_math: bool,
        mask_mode: &str,
        sparse_layout: &str,
    ) -> PyResult<Self> {
        let poni = Poni::load(poni_path)
            .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
        Self::build(
            poni,
            method,
            polarization_factor,
            n_ss,
            unit,
            radial_range,
            fast_math,
            mask_mode,
            sparse_layout,
        )
    }

    /// Create an Integrator from a PONI dictionary.
    ///
    /// The dictionary should contain PONI calibration parameters as keys:
    /// "pixel1", "pixel2", "distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength".
    ///
    /// # Arguments
    ///
    /// * `poni_dict` - Dictionary containing PONI calibration parameters
    /// * `method` - Integration method: "pixel_split" (default), "pyfai_full", "counting", or "supersampled"
    /// * `polarization_factor` - Optional polarization factor P in [-1, 1]: +1 fully
    ///   polarised along the horizontal axis (chi = 0, detector dim2; the usual
    ///   synchrotron case, typically 0.95-0.99), -1 vertical, 0 unpolarised
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (default: 5)
    /// * `unit` - Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
    /// * `radial_range` - Optional explicit (min, max) radial bin range in `unit`.
    ///   Not supported with "histogram", which always uses the automatic range.
    ///
    /// # Raises
    ///
    /// ValueError for invalid options, including `radial_range` combined with
    /// the "histogram" method.
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
    #[allow(clippy::too_many_arguments)] // Python keyword options plus the class receiver.
    #[pyo3(signature = (poni_dict, method=None, polarization_factor=None, n_ss=None, unit=None, radial_range=None, *, fast_math=true, mask_mode="fixed", sparse_layout="csc"))]
    fn from_poni_dict(
        _cls: &Bound<'_, PyType>,
        poni_dict: &Bound<'_, PyDict>,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
        unit: Option<&str>,
        radial_range: Option<(f64, f64)>,
        fast_math: bool,
        mask_mode: &str,
        sparse_layout: &str,
    ) -> PyResult<Self> {
        Self::build(
            poni_from_dict(poni_dict)?,
            method,
            polarization_factor,
            n_ss,
            unit,
            radial_range,
            fast_math,
            mask_mode,
            sparse_layout,
        )
    }

    /// Perform 1D radial integration.
    ///
    /// # Arguments
    ///
    /// * `image` - Path to the TIFF image file or 2D NumPy array
    /// * `n_points` - Number of bins for the 1D integration output
    /// * `errors` - Whether to compute Poisson errors (default: False)
    /// * `azimuthal_std` - Also return weighted population spread (default: False)
    ///
    /// # Returns
    ///
    /// IntegrationResult with radial values, intensities, and optional errors
    ///
    /// The GIL is released while a TIFF loads and while the image is integrated,
    /// so other Python threads run meanwhile (one Integrator per thread: an
    /// Integrator cannot be shared between threads). A NumPy input is borrowed
    /// read-only, without copying, for the duration of the call and must not be
    /// mutated concurrently.
    ///
    /// # Example
    ///
    /// ```python
    /// integrator = dioptrin.Integrator("calibration.poni")
    /// result = integrator.integrate1d("pattern.tif", n_points=1000)
    /// result = integrator.integrate1d("pattern.tif", n_points=1000, errors=True)
    /// ```
    #[pyo3(signature = (image, n_points, errors=false, *, azimuthal_std=false))]
    fn integrate1d(
        &self,
        py: Python<'_>,
        image: ImageInput,
        n_points: usize,
        errors: bool,
        azimuthal_std: bool,
    ) -> PyResult<PyIntegrationResult> {
        check_positive("n_points", n_points)?;
        let integrator = &*self.integrator;
        // The GIL is released while the TIFF loads and while the image is
        // integrated; only plain Rust data crosses into the detached regions.
        match image {
            ImageInput::Path(tiff_path) => {
                let frame = py
                    .detach(|| {
                        let mut frame = TiffFrame::default();
                        frame
                            .reload(&tiff_path)
                            .map(|_| frame)
                            .map_err(|err| err.to_string())
                    })
                    .map_err(|err| {
                        PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                    })?;
                self.check_mask((frame.dim1, frame.dim2), "image")?;
                Ok(py.detach(|| {
                    if azimuthal_std {
                        return PyIntegrationResult::from_azimuthal(
                            Integration1DWithAzimuthalStd {
                                n_bins: n_points,
                                errors,
                            }
                            .execute_tiff(integrator, &frame),
                        );
                    }
                    if errors {
                        let (radial, intensity, errs) =
                            integrator.integrate_tiff1d::<true>(&frame, n_points);
                        PyIntegrationResult::from_1d_with_errors(radial, intensity, errs)
                    } else {
                        let (radial, intensity, _) =
                            integrator.integrate_tiff1d::<false>(&frame, n_points);
                        PyIntegrationResult::from_1d(radial, intensity)
                    }
                }))
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                self.check_mask((dim1, dim2), "image")?;
                // `array` keeps its read-only borrow guard until this call
                // returns; the array must not be mutated concurrently.
                Ok(py.detach(|| {
                    integrate1d_slice(
                        integrator,
                        slice,
                        dim1,
                        dim2,
                        n_points,
                        errors,
                        azimuthal_std,
                    )
                }))
            }
            ImageInput::ArrayF32(array) => self.integrate1d_array_pixels(
                py,
                &array,
                "float32",
                n_points,
                errors,
                azimuthal_std,
            ),
            ImageInput::ArrayU16(array) => {
                self.integrate1d_array_pixels(py, &array, "uint16", n_points, errors, azimuthal_std)
            }
            ImageInput::ArrayU32(array) => {
                self.integrate1d_array_pixels(py, &array, "uint32", n_points, errors, azimuthal_std)
            }
            ImageInput::ArrayI32(array) => {
                self.integrate1d_array_pixels(py, &array, "int32", n_points, errors, azimuthal_std)
            }
        }
    }

    /// Perform 2D integration.
    ///
    /// Set `errors=True` to include per-bin error estimates. Use `azimuthal_range`
    /// to integrate a sector or to choose where the azimuthal axis starts.
    ///
    /// # Arguments
    ///
    /// * `image` - Path to the TIFF image file or 2D NumPy array
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins. Give either this or `azimuthal_bracket`.
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_bracket` - Keyword only, instead of `n_azimuthal`: half-width in radians of
    ///   contiguous rows given by their centres. The first row is centred on the start of
    ///   `azimuthal_range` (default (0, 2*pi)), every row covers its centre +- the bracket, and
    ///   the number of rows is span / (2 * bracket), which must be a whole number. For example
    ///   `azimuthal_bracket=np.radians(2.5)` gives 72 rows centred on 0, 5, ..., 355 degrees.
    ///   `result.azimuthal` holds the centres and `result.n_azimuthal` the row count.
    ///   In this mode `azimuthal_range` is (first centre, end) rather than edges, and the end is
    ///   exclusive: (0, pi/2) with a 5 degree bracket gives centres 0, 10, ..., 80 degrees,
    ///   covering -5 to 85 degrees. A first centre in [-pi, 2*pi] is kept as given; any other
    ///   is corrected to the equivalent angle in [-pi, pi). With `method="pyfai_full"` the seam lies one bracket
    ///   below the first centre, so rows differ slightly from a default pyFAI cake.
    /// * `azimuthal_range` - Optional finite (min, max) azimuthal edges in radians, with any start
    ///   and 0 < max - min <= 2*pi. Starts outside ±2*pi are corrected to
    ///   [-pi, pi), including the returned axis. The `n_azimuthal` bins split it evenly
    ///   (bin i covers [min + i*step, min + (i+1)*step)) and pixels outside are left out,
    ///   so sectors work, also across the +-pi cut. `result.azimuthal` holds the bin
    ///   centres. Radial bins are the same as without an azimuthal range.
    ///   With `method="pyfai_full"`, individual bins depend on how the range is
    ///   written (a range shifted by 2*pi, or a sector vs the same rows of a full
    ///   circle, can differ, as in pyFAI itself); prefer `"pixel_split"` for
    ///   sectors.
    ///
    /// # Returns
    ///
    /// IntegrationResult with flat intensity (and optional errors) sized
    /// `n_radial * n_azimuthal`; with `azimuthal_range`, also the radial and azimuthal
    /// bin centres. With the Integrator's `radial_range`, every method bins onto
    /// `n_radial` equal bins between its edges (pixels outside are left out) and
    /// `result.radial` holds their centres; otherwise `radial` is None without
    /// `azimuthal_range`.
    ///
    /// The GIL is released while a TIFF loads and while the image is integrated,
    /// so other Python threads run meanwhile (one Integrator per thread: an
    /// Integrator cannot be shared between threads). A NumPy input is borrowed
    /// read-only, without copying, for the duration of the call and must not be
    /// mutated concurrently.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the integrator uses the Histogram method, which has
    /// no azimuthal binning, or if its `radial_range` has a negative minimum
    /// (2D radial edges must be nonnegative).
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
    #[allow(clippy::too_many_arguments)] // Python keyword options plus the GIL token.
    #[pyo3(signature = (image, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, *, azimuthal_bracket=None))]
    fn integrate2d(
        &self,
        py: Python<'_>,
        image: ImageInput,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<PyIntegrationResult> {
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        let radial = self.radial_axis_2d(n_radial)?;
        let integrator = &*self.integrator;
        let bins = Bins2d {
            n_radial,
            n_azimuthal,
            errors,
            azimuthal_range,
        };
        // The GIL is released while the TIFF loads and while the image is
        // integrated; only plain Rust data crosses into the detached regions.
        match image {
            ImageInput::Path(tiff_path) => {
                let frame = py
                    .detach(|| Frame::from_tiff(tiff_path).map_err(|err| err.to_string()))
                    .map_err(|err| {
                        PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                    })?;
                self.check_mask((frame.dim1, frame.dim2), "image")?;
                py.detach(|| {
                    integrate2d_slice(
                        integrator,
                        &frame.data,
                        frame.dim1,
                        frame.dim2,
                        bins,
                        radial,
                    )
                })
                .map_err(PyValueError::new_err)
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                self.check_mask((dim1, dim2), "image")?;
                // `array` keeps its read-only borrow guard until this call
                // returns; the array must not be mutated concurrently.
                py.detach(|| integrate2d_slice(integrator, slice, dim1, dim2, bins, radial))
                    .map_err(PyValueError::new_err)
            }
            ImageInput::ArrayF32(array) => {
                self.integrate2d_array_pixels(py, &array, "float32", bins, radial)
            }
            ImageInput::ArrayU16(array) => {
                self.integrate2d_array_pixels(py, &array, "uint16", bins, radial)
            }
            ImageInput::ArrayU32(array) => {
                self.integrate2d_array_pixels(py, &array, "uint32", bins, radial)
            }
            ImageInput::ArrayI32(array) => {
                self.integrate2d_array_pixels(py, &array, "int32", bins, radial)
            }
        }
    }

    /// Set the integration method.
    ///
    /// Whether reassociated CSR accumulation is enabled (default: true).
    #[getter]
    fn fast_math(&self) -> bool {
        self.integrator.fast_math()
    }

    /// Opt into reassociated CSR sums and hardware FMA where available.
    /// LUTs/geometry are unchanged. Histogram and azimuthal_std keep their
    /// existing algorithms. Cannot be changed while an async batch is in use.
    fn set_fast_math(&mut self, enabled: bool) -> PyResult<()> {
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_fast_math(enabled);
        Ok(())
    }

    /// Requested fast 1D sparse layout (default CSC).
    #[getter]
    fn sparse_layout(&self) -> &'static str {
        match self.integrator.sparse_layout() {
            SparseLayout::Csc => "csc",
            SparseLayout::Csr => "csr",
        }
    }
    fn set_sparse_layout(&mut self, layout: &str) -> PyResult<()> {
        let layout = parse_sparse_layout(layout)?;
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_sparse_layout(layout);
        Ok(())
    }

    /// Mask execution policy: "fixed" (default) or "dynamic".
    #[getter]
    fn mask_mode(&self) -> &'static str {
        match self.integrator.mask_mode() {
            MaskMode::Fixed => "fixed",
            MaskMode::Dynamic => "dynamic",
        }
    }

    /// Change mask policy while retaining geometry and pixel weights.
    /// Fixed mode supports edits; dynamic mode avoids rebuilding per-frame tables.
    fn set_mask_mode(&mut self, mode: &str) -> PyResult<()> {
        let mode = parse_mask_mode(mode)?;
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_mask_mode(mode);
        Ok(())
    }

    /// Changes the integration method used for subsequent integrations.
    /// Available methods: "pixel_split", "pyfai_full", "counting", or "supersampled".
    ///
    /// # Arguments
    ///
    /// * `method` - Integration method name
    /// * `n_ss` - Number of subpixels per pixel for supersampled method (required if method is "supersampled")
    ///
    /// # Raises
    ///
    /// ValueError if the method name is invalid, or if the method is "histogram"
    /// while a `radial_range` is set (Histogram always uses the automatic range)
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_method("supersampled", n_ss=7)
    /// ```
    #[pyo3(signature = (method, n_ss=None))]
    fn set_method(&mut self, method: &str, n_ss: Option<usize>) -> PyResult<()> {
        let method = parse_method(Some(method), n_ss)?;
        let integrator = Arc::get_mut(&mut self.integrator)
            .ok_or_else(|| PyValueError::new_err("Integrator is in use by a batch operation"))?;
        integrator.set_method(method);
        Ok(())
    }

    /// Set the polarization correction factor.
    ///
    /// Applies the correction 0.5*(1 + cos^2(2th) - P*cos(2chi)*sin^2(2th)). P must be
    /// in [-1, 1]: +1 fully polarised along the horizontal axis (chi = 0, detector
    /// dim2; the usual synchrotron case, typically 0.95-0.99), -1 vertical,
    /// 0 unpolarised. Set to None to disable polarization correction.
    ///
    /// # Arguments
    ///
    /// * `polarization_factor` - Polarization factor P in [-1, 1], or None to disable
    ///
    /// # Example
    ///
    /// ```python
    /// integrator.set_polarization_factor(0.99)  # Enable correction
    /// integrator.set_polarization_factor(None)  # Disable correction
    /// ```
    fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) -> PyResult<()> {
        check_polarization_factor(polarization_factor)?;
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
        let unit = parse_unit(Some(unit))?;
        check_unit_wavelength(unit, integrator.geometry().poni.wavelength)?;
        integrator.set_radial_unit(unit);
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
    /// * `errors` - Whether to compute Poisson errors (default: False)
    /// * `azimuthal_std` - Also return weighted population spread (default: False)
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename/index and IntegrationResult.
    /// NumPy inputs are borrowed for this synchronous call without copying image
    /// pixels. Their read-only borrow guards remain held until all native workers
    /// finish; input arrays must not be mutated concurrently. The GIL is released
    /// while the workers run. For arrays, `num_workers` is the number of frames
    /// integrated at a time and `num_threads` the Rayon threads each of them
    /// uses, in a pool of its own (built for the call), as for TIFF paths; a
    /// missing one of the two is resolved as for TIFF paths (`num_threads`
    /// defaults to 2, `num_workers` to cores / num_threads - 1, at least 1).
    /// With neither given, arrays run on the global Rayon pool (all frames in
    /// parallel), configured with `set_rayon_threads` or `RAYON_NUM_THREADS`.
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
    #[allow(clippy::too_many_arguments)] // Python keyword options plus the GIL token.
    #[pyo3(signature = (images, n_points, errors=false, num_workers=None, num_threads=None, *, azimuthal_std=false))]
    fn batch1d<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = batch_images)] images: BatchImageInput<'py>,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_std: bool,
    ) -> PyResult<Vec<BatchEntry>> {
        check_positive("n_points", n_points)?;
        // Arrays keep the global pool unless a split was asked for.
        let split_requested = num_workers.is_some() || num_threads.is_some();
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match images {
            BatchImageInput::Paths(tiff_paths) => {
                if azimuthal_std {
                    let task = Integration1DWithAzimuthalStd {
                        n_bins: n_points,
                        errors,
                    };
                    return Ok(self
                        .process_files(
                            py,
                            &tiff_paths,
                            Arc::clone(&self.integrator),
                            task,
                            num_workers,
                            num_threads,
                        )
                        .into_iter()
                        .map(|(filename, result)| match result {
                            Ok(result) => BatchEntry {
                                filename,
                                result: Some(PyIntegrationResult::from_azimuthal(result)),
                                error: None,
                            },
                            Err(error) => BatchEntry {
                                filename,
                                result: None,
                                error: Some(error),
                            },
                        })
                        .collect());
                }
                // Original file-based processing
                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let results = self.process_files(
                        py,
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(results
                        .into_iter()
                        .map(|(filename, result)| match result {
                            Ok((radial, intensity, errs)) => BatchEntry {
                                filename,
                                result: Some(PyIntegrationResult::from_1d_with_errors(
                                    radial, intensity, errs,
                                )),
                                error: None,
                            },
                            Err(err) => BatchEntry {
                                filename,
                                result: None,
                                error: Some(err),
                            },
                        })
                        .collect())
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let results = self.process_files(
                        py,
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    );
                    Ok(results
                        .into_iter()
                        .map(|(filename, result)| match result {
                            Ok((radial, intensity)) => BatchEntry {
                                filename,
                                result: Some(PyIntegrationResult::from_1d(radial, intensity)),
                                error: None,
                            },
                            Err(err) => BatchEntry {
                                filename,
                                result: None,
                                error: Some(err),
                            },
                        })
                        .collect())
                }
            }
            BatchImageInput::Arrays(arrays) => {
                // Keep every PyReadonlyArray guard alive until Rayon finishes.
                // Only plain Rust slices/dimensions cross into native workers;
                // no Python objects or borrowed data escape this synchronous call.
                let frames: Result<Vec<_>, PyErr> = arrays
                    .iter()
                    .enumerate()
                    .map(|(i, array)| {
                        let (dim1, dim2) = extract_item_dims(array, i)?;
                        let slice = array.as_slice().map_err(|err| {
                            PyValueError::new_err(format!(
                                "Array {} must be contiguous: {}",
                                i, err
                            ))
                        })?;
                        self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
                        Ok((slice, dim1, dim2))
                    })
                    .collect();

                let frames = frames?;

                // Process frames in parallel using rayon
                use rayon::prelude::*;
                // Deliberately its own closure, shared with the worker split
                // below only through the same parallel iterator: replacing it
                // by `integrate1d_slice`, or adding new call sites into the 1D
                // `*_from_slice` entry points, changed fat-LTO inlining and
                // slowed the 1D pixel_split/pyfai_full kernels by up to ~20%
                // when tried. Re-benchmark any such refactor against v0.14.0
                // on the release wheel build.
                let integrate = |(i, &(data, dim1, dim2))| {
                    let integration_result = if azimuthal_std {
                        PyIntegrationResult::from_azimuthal(
                            self.integrator.integrate1d_with_azimuthal_std_from_slice(
                                data, dim1, dim2, n_points, errors,
                            ),
                        )
                    } else if errors {
                        let (radial, intensity, errs) = self
                            .integrator
                            .integrate1d_with_errors_from_slice(data, dim1, dim2, n_points);
                        PyIntegrationResult::from_1d_with_errors(radial, intensity, errs)
                    } else {
                        let (radial, intensity) = self
                            .integrator
                            .integrate1d_from_slice(data, dim1, dim2, n_points);
                        PyIntegrationResult::from_1d(radial, intensity)
                    };

                    BatchEntry {
                        filename: format!("array[{}]", i),
                        result: Some(integration_result),
                        error: None,
                    }
                };
                if split_requested {
                    // At most `num_workers` frames at a time, each on its own
                    // pool of `num_threads` threads, as for TIFF paths; the
                    // table is built first in the first worker's pool.
                    //
                    // Each frame runs through the very same parallel iterator
                    // as the global-pool path below (one frame long), not a
                    // new call into the integrator: with fat LTO, an extra
                    // call site measurably changed how the shared 1D kernels
                    // were optimised (pixel_split/pyfai_full up to 20% slower
                    // on every 1D path, default ones included).
                    let prepare = || {
                        if let Some(&(_, dim1, dim2)) = frames.first()
                            && frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
                        {
                            self.integrator.prepare1d(dim1, dim2, n_points);
                        }
                    };
                    let outcomes = py
                        .detach(|| {
                            map_in_pools(frames.len(), num_workers, num_threads, prepare, |i| {
                                let mut one: Vec<BatchEntry> = frames[i..=i]
                                    .par_iter()
                                    .enumerate()
                                    .map(integrate)
                                    .collect();
                                let mut result = one.pop().expect("one frame, one result");
                                result.filename = format!("array[{}]", i);
                                result
                            })
                        })
                        .map_err(PyValueError::new_err)?;
                    // Re-raise the first panic in input order, as the global
                    // pool path would.
                    return Ok(outcomes
                        .into_iter()
                        .map(|outcome| {
                            outcome.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
                        })
                        .collect());
                }
                // Release the GIL while the workers run (see `process_files`).
                // `frames` holds only plain slices whose borrow guards live in
                // `arrays`, which outlives this region.
                let results = py.detach(|| {
                    let uniform = frames
                        .first()
                        .filter(|&&(_, dim1, dim2)| {
                            frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
                        })
                        .map(|&(_, dim1, dim2)| (dim1, dim2));
                    if let Some((dim1, dim2)) = uniform {
                        self.integrator.prepare1d(dim1, dim2, n_points);
                        let min_tiled = 4 * rayon::current_num_threads().max(1);
                        if !azimuthal_std
                            && frames.len() >= min_tiled
                            && self
                                .integrator
                                .can_integrate1d_batch4(dim1, dim2, n_points, errors)
                        {
                            let n_tiled = (frames.len() / 4) * 4;
                            let mut out: Vec<BatchEntry> = frames[..n_tiled]
                                .par_chunks_exact(4)
                                .enumerate()
                                .flat_map(|(chunk_idx, chunk)| {
                                    let base = chunk_idx * 4;
                                    let batch = self
                                        .integrator
                                        .try_integrate1d_batch4(
                                            [chunk[0].0, chunk[1].0, chunk[2].0, chunk[3].0],
                                            dim1,
                                            dim2,
                                            n_points,
                                            errors,
                                        )
                                        .expect("can_integrate1d_batch4 pre-checked eligibility");
                                    batch
                                        .into_iter()
                                        .enumerate()
                                        .map(|(j, (radial, intensity, errs))| {
                                            let res = match errs {
                                                Some(e) => {
                                                    PyIntegrationResult::from_1d_with_errors(
                                                        radial, intensity, e,
                                                    )
                                                }
                                                None => {
                                                    PyIntegrationResult::from_1d(radial, intensity)
                                                }
                                            };
                                            BatchEntry {
                                                filename: format!("array[{}]", base + j),
                                                result: Some(res),
                                                error: None,
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect();
                            if n_tiled < frames.len() {
                                out.par_extend(
                                    frames[n_tiled..]
                                        .par_iter()
                                        .enumerate()
                                        .map(|(j, item)| integrate((n_tiled + j, item))),
                                );
                            }
                            return out;
                        }
                    }
                    frames.par_iter().enumerate().map(integrate).collect()
                });

                Ok(results)
            }
            BatchImageInput::ArraysF32(arrays) => self.batch1d_arrays_pixels(
                py,
                &arrays,
                n_points,
                errors,
                azimuthal_std,
                split_requested.then_some((num_workers, num_threads)),
            ),
            BatchImageInput::ArraysU16(arrays) => self.batch1d_arrays_pixels(
                py,
                &arrays,
                n_points,
                errors,
                azimuthal_std,
                split_requested.then_some((num_workers, num_threads)),
            ),
            BatchImageInput::ArraysU32(arrays) => self.batch1d_arrays_pixels(
                py,
                &arrays,
                n_points,
                errors,
                azimuthal_std,
                split_requested.then_some((num_workers, num_threads)),
            ),
            BatchImageInput::ArraysI32(arrays) => self.batch1d_arrays_pixels(
                py,
                &arrays,
                n_points,
                errors,
                azimuthal_std,
                split_requested.then_some((num_workers, num_threads)),
            ),
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
    /// * `errors` - Whether to compute Poisson errors (default: False)
    /// * `azimuthal_std` - Also return weighted population spread (default: False)
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
    #[pyo3(signature = (images, n_points, errors=false, num_workers=None, num_threads=None, *, azimuthal_std=false))]
    fn batch1d_iter(
        &self,
        images: &Bound<'_, PyAny>,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_std: bool,
    ) -> PyResult<PyBatchIterator> {
        check_positive("n_points", n_points)?;
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;

        match classify_batch_input(images)? {
            ClassifiedInput::Paths(tiff_paths) => {
                // File-based streaming
                let total = tiff_paths.len();

                if azimuthal_std {
                    let task = Integration1DWithAzimuthalStd {
                        n_bins: n_points,
                        errors,
                    };
                    let batch = self.process_files_streaming(
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    return Ok(PyBatchIterator::new(
                        BatchReceiver::Integration1DWithAzimuthalStd(Mutex::new(
                            batch.into_receiver(),
                        )),
                        total,
                    ));
                }
                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let batch = self.process_files_streaming(
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    Ok(PyBatchIterator::new(
                        BatchReceiver::Integration1DWithErrors(Mutex::new(batch.into_receiver())),
                        total,
                    ))
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let batch = self.process_files_streaming(
                        &tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    Ok(PyBatchIterator::new(
                        BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                        total,
                    ))
                }
            }
            classified => {
                // Arrays or an iterator: a producer thread pulls items lazily.
                let (input, total) = self.stream_input(classified)?;
                let QueueStream {
                    work_rx,
                    pool,
                    stream_error,
                    control,
                } = self.open_queue_stream(input, num_workers, num_threads)?;

                let integrator = Arc::clone(&self.integrator);
                if azimuthal_std {
                    let task = Integration1DWithAzimuthalStd {
                        n_bins: n_points,
                        errors,
                    };
                    let batch = process_queue_with_pool(
                        work_rx,
                        integrator,
                        task,
                        pool,
                        Arc::clone(&control),
                    );
                    return Ok(PyBatchIterator::with_control(
                        BatchReceiver::Integration1DWithAzimuthalStd(Mutex::new(
                            batch.into_receiver(),
                        )),
                        total,
                        stream_error.clone(),
                        Arc::clone(&control),
                    ));
                }
                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let batch = process_queue_with_pool(
                        work_rx,
                        integrator,
                        task,
                        pool,
                        Arc::clone(&control),
                    );
                    Ok(PyBatchIterator::with_control(
                        BatchReceiver::Integration1DWithErrors(Mutex::new(batch.into_receiver())),
                        total,
                        stream_error.clone(),
                        Arc::clone(&control),
                    ))
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let batch = process_queue_with_pool(
                        work_rx,
                        integrator,
                        task,
                        pool,
                        Arc::clone(&control),
                    );
                    Ok(PyBatchIterator::with_control(
                        BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                        total,
                        stream_error.clone(),
                        Arc::clone(&control),
                    ))
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
    /// * `tiff_paths` - List of TIFF file paths or list of NumPy arrays (2D float64). The
    ///   parameter keeps its historical name; it accepts arrays like `batch1d`'s `images`.
    /// * `n_radial` - Number of radial bins
    /// * `n_azimuthal` - Number of azimuthal bins. Give either this or `azimuthal_bracket`.
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_bracket` - Keyword only, instead of `n_azimuthal`: half-width in radians of
    ///   contiguous rows given by their centres. The first row is centred on the start of
    ///   `azimuthal_range` (default (0, 2*pi)), every row covers its centre +- the bracket, and
    ///   the number of rows is span / (2 * bracket), which must be a whole number. For example
    ///   `azimuthal_bracket=np.radians(2.5)` gives 72 rows centred on 0, 5, ..., 355 degrees.
    ///   `result.azimuthal` holds the centres and `result.n_azimuthal` the row count.
    ///   In this mode `azimuthal_range` is (first centre, end) rather than edges, and the end is
    ///   exclusive: (0, pi/2) with a 5 degree bracket gives centres 0, 10, ..., 80 degrees,
    ///   covering -5 to 85 degrees. A first centre in [-pi, 2*pi] is kept as given; any other
    ///   is corrected to the equivalent angle in [-pi, pi). With `method="pyfai_full"` the seam lies one bracket
    ///   below the first centre, so rows differ slightly from a default pyFAI cake.
    /// * `azimuthal_range` - Optional finite (min, max) azimuthal edges in radians, with any start
    ///   and 0 < max - min <= 2*pi. Starts outside ±2*pi are corrected to
    ///   [-pi, pi), including the returned axis. The `n_azimuthal` bins split it evenly
    ///   (bin i covers [min + i*step, min + (i+1)*step)) and pixels outside are left out,
    ///   so sectors work, also across the +-pi cut. `result.azimuthal` holds the bin
    ///   centres. Radial bins are the same as without an azimuthal range.
    ///   With `method="pyfai_full"`, individual bins depend on how the range is
    ///   written (a range shifted by 2*pi, or a sector vs the same rows of a full
    ///   circle, can differ, as in pyFAI itself); prefer `"pixel_split"` for
    ///   sectors.
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename (`array[i]` for arrays)
    /// and IntegrationResult, in input order. With the Integrator's `radial_range`,
    /// results carry its radial bin centres in `radial`.
    /// NumPy inputs are borrowed for this synchronous call without copying image
    /// pixels. Their read-only borrow guards remain held until all native workers
    /// finish; input arrays must not be mutated concurrently. The GIL is released
    /// while the workers run. For arrays, `num_workers` is the number of frames
    /// integrated at a time and `num_threads` the Rayon threads each of them
    /// uses, in a pool of its own (built for the call), as for TIFF paths: with
    /// `num_workers=8, num_threads=1` eight frames run side by side on one
    /// thread each; with `num_workers=1, num_threads=8` one frame after another
    /// on eight threads. A missing one of the two is resolved as for TIFF paths
    /// (`num_threads` defaults to 2, `num_workers` to cores / num_threads - 1,
    /// at least 1). With neither given, arrays run on the global Rayon pool (all
    /// frames in parallel, as in `batch1d`), configured with `set_rayon_threads`
    /// or `RAYON_NUM_THREADS`.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the integrator uses the Histogram method, which has
    /// no azimuthal binning, or for an array that is not C-contiguous or does not
    /// match the mask; `TypeError` for an array that is not 2D float64, a single
    /// path, or a list mixing paths and arrays (use `batch2d_iter` for those).
    ///
    /// # Example
    ///
    /// ```python
    /// # Full 2D integration
    /// results = integrator.batch2d(files, n_radial=100, n_azimuthal=360)
    ///
    /// # From HDF5 via NumPy arrays
    /// with h5py.File('data.h5', 'r') as f:
    ///     frames = [f['/entry/data/data'][i].astype(np.float64) for i in range(100)]
    /// results = integrator.batch2d(frames, n_radial=100, n_azimuthal=360)
    ///
    /// # With azimuthal range and errors
    /// results = integrator.batch2d(
    ///     files, n_radial=100, n_azimuthal=90,
    ///     azimuthal_range=(0, 1.57),  # 0 to pi/2
    ///     errors=True
    /// )
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, num_workers=None, num_threads=None, *, azimuthal_bracket=None))]
    fn batch2d<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = batch2d_images)] tiff_paths: BatchImageInput<'py>,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<Vec<BatchEntry>> {
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        let radial = self.radial_axis_2d(n_radial)?;
        // Arrays keep the global pool unless a split was asked for.
        let split_requested = num_workers.is_some() || num_threads.is_some();
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        validate_azimuthal_range(azimuthal_range, n_azimuthal)?;
        let tiff_paths = match tiff_paths {
            BatchImageInput::Paths(paths) => paths,
            BatchImageInput::Arrays(arrays) => {
                let bins = Bins2d {
                    n_radial,
                    n_azimuthal,
                    errors,
                    azimuthal_range,
                };
                let split = split_requested.then_some((num_workers, num_threads));
                return self.batch2d_arrays(py, &arrays, bins, radial, split);
            }
            BatchImageInput::ArraysF32(arrays) => {
                let bins = Bins2d {
                    n_radial,
                    n_azimuthal,
                    errors,
                    azimuthal_range,
                };
                let split = split_requested.then_some((num_workers, num_threads));
                return self.batch2d_arrays(py, &arrays, bins, radial, split);
            }
            BatchImageInput::ArraysU16(arrays) => {
                let bins = Bins2d {
                    n_radial,
                    n_azimuthal,
                    errors,
                    azimuthal_range,
                };
                let split = split_requested.then_some((num_workers, num_threads));
                return self.batch2d_arrays(py, &arrays, bins, radial, split);
            }
            BatchImageInput::ArraysU32(arrays) => {
                let bins = Bins2d {
                    n_radial,
                    n_azimuthal,
                    errors,
                    azimuthal_range,
                };
                let split = split_requested.then_some((num_workers, num_threads));
                return self.batch2d_arrays(py, &arrays, bins, radial, split);
            }
            BatchImageInput::ArraysI32(arrays) => {
                let bins = Bins2d {
                    n_radial,
                    n_azimuthal,
                    errors,
                    azimuthal_range,
                };
                let split = split_requested.then_some((num_workers, num_threads));
                return self.batch2d_arrays(py, &arrays, bins, radial, split);
            }
        };

        match (errors, azimuthal_range) {
            (false, None) => {
                let task = Integration2D {
                    n_radial,
                    n_azimuthal,
                };
                let results = self.process_files(
                    py,
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok(intensity) => BatchEntry {
                            filename,
                            result: Some(
                                PyIntegrationResult::from_2d(intensity, n_radial, n_azimuthal)
                                    .with_radial(radial.clone()),
                            ),
                            error: None,
                        },
                        Err(err) => BatchEntry {
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
                let results = self.process_files(
                    py,
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| match result {
                        Ok((intensity, errs)) => BatchEntry {
                            filename,
                            result: Some(
                                PyIntegrationResult::from_2d_with_errors(
                                    intensity,
                                    errs,
                                    n_radial,
                                    n_azimuthal,
                                )
                                .with_radial(radial.clone()),
                            ),
                            error: None,
                        },
                        Err(err) => BatchEntry {
                            filename,
                            result: None,
                            error: Some(err),
                        },
                    })
                    .collect())
            }
            (errors, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DOnAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                    errors,
                };
                let results = self.process_files(
                    py,
                    &tiff_paths,
                    Arc::clone(&self.integrator),
                    task,
                    num_workers,
                    num_threads,
                );
                Ok(results
                    .into_iter()
                    .map(|(filename, result)| {
                        PyIntegrationResult::batch_azimuthal_range_2d(
                            filename,
                            result,
                            n_radial,
                            n_azimuthal,
                        )
                    })
                    .collect())
            }
        }
    }

    /// Perform batch 2D integration into a contiguous 3D NumPy array
    /// `(n_frames, n_azimuthal, n_radial)` (or `(intensity, errors)` when
    /// `errors=True`).
    ///
    /// Accepts a 3D `(n_frames, dim1, dim2)` C-contiguous NumPy array or a list
    /// of uniform-shape 2D C-contiguous NumPy arrays in `float64`, `float32`,
    /// `uint16`, `uint32`, or `int32`. Writes directly into the output array
    /// without per-frame allocations.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (images, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, *, azimuthal_bracket=None))]
    fn batch2d_stack(
        &self,
        py: Python<'_>,
        images: StackImageInput<'_>,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        let _radial = self.radial_axis_2d(n_radial)?;
        validate_azimuthal_range(azimuthal_range, n_azimuthal)?;
        let bins = Bins2d {
            n_radial,
            n_azimuthal,
            errors,
            azimuthal_range,
        };
        match images {
            StackImageInput::StackF64(s) => self.batch2d_stack_3d(py, &s, bins),
            StackImageInput::StackF32(s) => self.batch2d_stack_3d(py, &s, bins),
            StackImageInput::StackU16(s) => self.batch2d_stack_3d(py, &s, bins),
            StackImageInput::StackU32(s) => self.batch2d_stack_3d(py, &s, bins),
            StackImageInput::StackI32(s) => self.batch2d_stack_3d(py, &s, bins),
            StackImageInput::ListF64(a) => self.batch2d_stack_list(py, &a, bins),
            StackImageInput::ListF32(a) => self.batch2d_stack_list(py, &a, bins),
            StackImageInput::ListU16(a) => self.batch2d_stack_list(py, &a, bins),
            StackImageInput::ListU32(a) => self.batch2d_stack_list(py, &a, bins),
            StackImageInput::ListI32(a) => self.batch2d_stack_list(py, &a, bins),
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
    /// * `n_azimuthal` - Number of azimuthal bins. Give either this or `azimuthal_bracket`.
    /// * `errors` - Whether to compute error estimates (default: False)
    /// * `azimuthal_bracket` - Keyword only, instead of `n_azimuthal`: half-width in radians of
    ///   contiguous rows given by their centres. The first row is centred on the start of
    ///   `azimuthal_range` (default (0, 2*pi)), every row covers its centre +- the bracket, and
    ///   the number of rows is span / (2 * bracket), which must be a whole number. For example
    ///   `azimuthal_bracket=np.radians(2.5)` gives 72 rows centred on 0, 5, ..., 355 degrees.
    ///   `result.azimuthal` holds the centres and `result.n_azimuthal` the row count.
    ///   In this mode `azimuthal_range` is (first centre, end) rather than edges, and the end is
    ///   exclusive: (0, pi/2) with a 5 degree bracket gives centres 0, 10, ..., 80 degrees,
    ///   covering -5 to 85 degrees. A first centre in [-pi, 2*pi] is kept as given; any other
    ///   is corrected to the equivalent angle in [-pi, pi). With `method="pyfai_full"` the seam lies one bracket
    ///   below the first centre, so rows differ slightly from a default pyFAI cake.
    /// * `azimuthal_range` - Optional finite (min, max) azimuthal edges in radians, with any start
    ///   and 0 < max - min <= 2*pi. Starts outside ±2*pi are corrected to
    ///   [-pi, pi), including the returned axis. The `n_azimuthal` bins split it evenly
    ///   (bin i covers [min + i*step, min + (i+1)*step)) and pixels outside are left out,
    ///   so sectors work, also across the +-pi cut. `result.azimuthal` holds the bin
    ///   centres. Radial bins are the same as without an azimuthal range.
    ///   With `method="pyfai_full"`, individual bins depend on how the range is
    ///   written (a range shifted by 2*pi, or a sector vs the same rows of a full
    ///   circle, can differ, as in pyFAI itself); prefer `"pixel_split"` for
    ///   sectors.
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// BatchIterator that yields BatchResult objects as they complete.
    /// The iterator has `total`, `completed`, and `progress` properties.
    /// When using a generator, `total` is 0 and `progress` returns None.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the integrator uses the Histogram method, which has
    /// no azimuthal binning.
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
    #[pyo3(signature = (images, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, num_workers=None, num_threads=None, *, azimuthal_bracket=None))]
    fn batch2d_iter(
        &self,
        images: &Bound<'_, PyAny>,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<PyBatchIterator> {
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        let radial = self.radial_axis_2d(n_radial)?;
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        validate_azimuthal_range(azimuthal_range, n_azimuthal)?;

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
                    let batch = self.process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    Ok(PyBatchIterator::new(
                        BatchReceiver::Integration2D {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                            radial: radial.clone(),
                        },
                        total,
                    ))
                }
                (true, None) => {
                    let task = Integration2DWithErrors {
                        n_radial,
                        n_azimuthal,
                    };
                    let batch = self.process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    Ok(PyBatchIterator::new(
                        BatchReceiver::Integration2DWithErrors {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                            radial: radial.clone(),
                        },
                        total,
                    ))
                }
                (errors, Some((azimuthal_min, azimuthal_max))) => {
                    let task = Integration2DOnAzimuthalRange {
                        n_radial,
                        n_azimuthal,
                        azimuthal_min,
                        azimuthal_max,
                        errors,
                    };
                    let batch = self.process_files_streaming(
                        tiff_paths,
                        Arc::clone(&self.integrator),
                        task,
                        num_workers,
                        num_threads,
                    )?;
                    Ok(PyBatchIterator::new(
                        BatchReceiver::Integration2DOnAzimuthalRange {
                            receiver: Mutex::new(batch.into_receiver()),
                            n_radial,
                            n_azimuthal,
                        },
                        total,
                    ))
                }
            };
        }

        // Arrays or an iterator: a producer thread pulls items lazily.
        let (input, total) = self.stream_input(classified)?;
        let QueueStream {
            work_rx,
            pool,
            stream_error,
            control,
        } = self.open_queue_stream(input, num_workers, num_threads)?;

        let integrator = Arc::clone(&self.integrator);

        match (errors, azimuthal_range) {
            (false, None) => {
                let task = Integration2D {
                    n_radial,
                    n_azimuthal,
                };
                let batch =
                    process_queue_with_pool(work_rx, integrator, task, pool, Arc::clone(&control));
                Ok(PyBatchIterator::with_control(
                    BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                        radial: radial.clone(),
                    },
                    total,
                    stream_error.clone(),
                    Arc::clone(&control),
                ))
            }
            (true, None) => {
                let task = Integration2DWithErrors {
                    n_radial,
                    n_azimuthal,
                };
                let batch =
                    process_queue_with_pool(work_rx, integrator, task, pool, Arc::clone(&control));
                Ok(PyBatchIterator::with_control(
                    BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                        radial: radial.clone(),
                    },
                    total,
                    stream_error.clone(),
                    Arc::clone(&control),
                ))
            }
            (errors, Some((azimuthal_min, azimuthal_max))) => {
                let task = Integration2DOnAzimuthalRange {
                    n_radial,
                    n_azimuthal,
                    azimuthal_min,
                    azimuthal_max,
                    errors,
                };
                let batch =
                    process_queue_with_pool(work_rx, integrator, task, pool, Arc::clone(&control));
                Ok(PyBatchIterator::with_control(
                    BatchReceiver::Integration2DOnAzimuthalRange {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    stream_error.clone(),
                    Arc::clone(&control),
                ))
            }
        }
    }
}
