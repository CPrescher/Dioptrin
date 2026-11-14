use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use crate::batch::{
    FileWorkers, Integration1D, Integration1DWithAzimuthalStd, Integration1DWithErrors,
    Integration2D, Integration2DOnAzimuthalRange, Integration2DWithErrors, IntegrationTask,
    StreamingBatch, WorkItem, process_files_streaming_with_pool, process_files_with_pool,
    process_queue_with_pool,
};
use crate::frame::{Frame, TiffFrame};
use crate::geometry::Geometry;
use crate::integration::{
    IntegrationGrid2d, IntegrationMethod, Integrator, MaskMode, RadialUnit, SparseLayout,
};
use crate::mask::Mask;
use crate::poni::Poni;

use super::results::{
    BatchReceiver, PyBatchIterator, PyBatchResult, PyIntegrationResult, StreamError, panic_message,
};
use super::utils::{
    BatchImageInput, ClassifiedInput, FrameOrPath, ImageInput, check_mask_shape,
    check_method_radial_range, check_polarization_factor, check_positive, check_unit_wavelength,
    classify_batch_input, extract_array_dims, extract_frame_from_pyany, extract_item_dims,
    mask_shape, parse_method, parse_unit, poni_from_dict, resolve_workers_threads,
    validate_integrator_options,
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

/// Pull items from a Python iterator on a producer thread and queue them for
/// the workers. An invalid item, a mask-shape mismatch, or an exception raised
/// by the iterator itself is recorded in `stream_error` at its position (so the
/// consumer raises it after all earlier results) and ends production.
fn spawn_iterator_producer(
    iter: Py<PyAny>,
    work_tx: SyncSender<WorkItem>,
    mask: Option<(usize, usize)>,
    stream_error: StreamError,
) {
    std::thread::spawn(move || {
        Python::attach(move |py| {
            let iter = iter.bind(py);
            for index in 0.. {
                let item = match iter.call_method0("__next__") {
                    Ok(item) => item,
                    Err(err) if err.is_instance_of::<PyStopIteration>(py) => break,
                    Err(err) => {
                        stream_error.set(index, err);
                        break;
                    }
                };
                let work_item = match extract_frame_from_pyany(&item, index) {
                    Ok(FrameOrPath::Path(path)) => WorkItem::Path { index, path },
                    Ok(FrameOrPath::Frame(frame)) => {
                        let label = format!("images[{index}]");
                        if let Err(err) = check_mask_shape(mask, (frame.dim1, frame.dim2), &label) {
                            stream_error.set(index, err);
                            break;
                        }
                        WorkItem::Frame {
                            index,
                            name: frame.name.clone(),
                            frame,
                        }
                    }
                    Err(err) => {
                        stream_error.set(index, err);
                        break;
                    }
                };
                // Wait for queue space without holding the GIL.
                if py.detach(|| work_tx.send(work_item)).is_err() {
                    break; // Workers gone
                }
            }
        });
        // work_tx dropped here → channel closes → workers drain and stop
    });
}

/// Integrate owned frames on a detached thread, streaming results as they
/// complete. A panic while integrating a frame is recorded in `stream_error`
/// and raised by the iterator at that position instead of ending it silently.
fn spawn_array_stream<T: Send + 'static>(
    frames: Vec<Frame>,
    stream_error: StreamError,
    integrate: impl Fn(&Frame) -> T + Send + Sync + 'static,
) -> Receiver<(usize, String, Result<T, String>)> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use rayon::prelude::*;
        frames
            .par_iter()
            .enumerate()
            .for_each_with(tx, |tx, (i, frame)| {
                match catch_unwind(AssertUnwindSafe(|| integrate(frame))) {
                    Ok(result) => {
                        let _ = tx.send((i, format!("array[{i}]"), Ok(result)));
                    }
                    Err(payload) => stream_error.set(
                        i,
                        PyRuntimeError::new_err(format!(
                            "integration of images[{i}] failed: {}",
                            panic_message(&*payload)
                        )),
                    ),
                }
            });
    });
    rx
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
    integrator: Arc<Integrator>,
    file_workers: Mutex<Option<Arc<FileWorkers>>>,
}

impl PyIntegrator {
    pub fn inner(&self) -> &Arc<Integrator> {
        &self.integrator
    }

    /// The Histogram method has no azimuthal binning, so 2D integration rejects
    /// it before any work starts instead of panicking inside Rust.
    fn reject_histogram_2d(&self, function: &str) -> PyResult<()> {
        if matches!(self.integrator.method(), IntegrationMethod::Histogram) {
            return Err(PyValueError::new_err(format!(
                "{function} is not supported for the 'histogram' method; \
                 use 'counting', 'pixel_split', 'pyfai_full' or 'supersampled'"
            )));
        }
        Ok(())
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
        check_method_radial_range(&method, radial_range)?;
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
        })
    }

    /// Raise ValueError when an image does not match the stored mask.
    fn check_mask(&self, image: (usize, usize), label: &str) -> PyResult<()> {
        check_mask_shape(mask_shape(&self.integrator), image, label)
    }

    fn file_pool(&self, workers: usize, threads: usize) -> PyResult<Arc<FileWorkers>> {
        let mut cache = self
            .file_workers
            .lock()
            .expect("file worker cache lock poisoned");
        if let Some(pool) = cache.as_ref()
            && pool.workers() == workers
            && pool.num_threads == threads
        {
            return Ok(Arc::clone(pool));
        }
        let pool = FileWorkers::new(workers, threads).map_err(PyValueError::new_err)?;
        *cache = Some(Arc::clone(&pool));
        Ok(pool)
    }

    /// Workers for a stream fed by a Python iterator. Such a stream holds every
    /// worker of its pool until the iterator is exhausted (or the stream is
    /// dropped), so it gets a pool of its own: on the shared cached pool a
    /// later synchronous batch, or a second stream, would queue behind it for
    /// as long as the caller's generator keeps producing.
    fn iterator_pool(workers: usize, threads: usize) -> PyResult<Arc<FileWorkers>> {
        FileWorkers::new(workers, threads).map_err(PyValueError::new_err)
    }

    /// Integrate `paths` on the cached file workers and wait for all results.
    ///
    /// The wait releases the GIL: a stream fed by a Python iterator needs it to
    /// pull its next item, and its jobs may be ahead of this batch in a worker
    /// queue. Nothing inside the detached region touches Python objects.
    fn process_files<T: IntegrationTask + Clone + 'static>(
        &self,
        py: Python<'_>,
        paths: &[String],
        integrator: Arc<Integrator>,
        task: T,
        workers: usize,
        threads: usize,
    ) -> Vec<(String, Result<T::Output, String>)>
    where
        T::Output: 'static,
    {
        // Worker construction errors are rare resource errors; preserve one
        // ordered error result per input file, including for empty batches.
        if paths.is_empty() {
            return Vec::new();
        }
        match self.file_pool(workers, threads) {
            Ok(pool) => py.detach(|| process_files_with_pool(paths, integrator, task, pool)),
            Err(e) => paths
                .iter()
                .map(|p| (p.clone(), Err(e.to_string())))
                .collect(),
        }
    }

    fn process_files_streaming<T: IntegrationTask + Clone + 'static>(
        &self,
        paths: &[String],
        integrator: Arc<Integrator>,
        task: T,
        workers: usize,
        threads: usize,
    ) -> PyResult<StreamingBatch<T::Output>>
    where
        T::Output: 'static,
    {
        Ok(process_files_streaming_with_pool(
            paths,
            integrator,
            task,
            self.file_pool(workers, threads)?,
        ))
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

    /// Release cached file workers and decoding buffers. Admitted streaming work
    /// finishes normally; the next file batch creates a new pool.
    fn clear_file_workers(&self) {
        self.file_workers
            .lock()
            .expect("file worker cache lock poisoned")
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
    /// * `method` - Integration method: "pixel_split" (default), "pyfai_full", "counting", "histogram", or "supersampled"
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
    /// * `method` - Integration method: "pixel_split" (default), "pyfai_full", "counting", "histogram", or "supersampled"
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
        image: ImageInput,
        n_points: usize,
        errors: bool,
        azimuthal_std: bool,
    ) -> PyResult<PyIntegrationResult> {
        check_positive("n_points", n_points)?;
        match image {
            ImageInput::Path(tiff_path) => {
                let mut frame = TiffFrame::default();
                frame.reload(&tiff_path).map_err(|err| {
                    PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                })?;
                self.check_mask((frame.dim1, frame.dim2), "image")?;
                if azimuthal_std {
                    return Ok(PyIntegrationResult::from_azimuthal(
                        Integration1DWithAzimuthalStd {
                            n_bins: n_points,
                            errors,
                        }
                        .execute_tiff(&self.integrator, &frame),
                    ));
                }
                if errors {
                    let (radial, intensity, errs) =
                        self.integrator.integrate_tiff1d::<true>(&frame, n_points);
                    Ok(PyIntegrationResult::from_1d_with_errors(
                        radial, intensity, errs,
                    ))
                } else {
                    let (radial, intensity, _) =
                        self.integrator.integrate_tiff1d::<false>(&frame, n_points);
                    Ok(PyIntegrationResult::from_1d(radial, intensity))
                }
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                self.check_mask((dim1, dim2), "image")?;
                if azimuthal_std {
                    return Ok(PyIntegrationResult::from_azimuthal(
                        self.integrator.integrate1d_with_azimuthal_std_from_slice(
                            slice, dim1, dim2, n_points, errors,
                        ),
                    ));
                }
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
    ///   centres. Radial bins are the same as without a range.
    ///   With `method="pyfai_full"`, individual bins depend on how the range is
    ///   written (a range shifted by 2*pi, or a sector vs the same rows of a full
    ///   circle, can differ, as in pyFAI itself); prefer `"pixel_split"` for
    ///   sectors.
    ///
    /// # Returns
    ///
    /// IntegrationResult with flat intensity (and optional errors) sized
    /// `n_radial * n_azimuthal`; with `azimuthal_range`, also the azimuthal bin centres
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the integrator uses the Histogram method, which has
    /// no azimuthal binning.
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
    #[pyo3(signature = (image, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, *, azimuthal_bracket=None))]
    fn integrate2d(
        &self,
        image: ImageInput,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<PyIntegrationResult> {
        self.reject_histogram_2d("integrate2d")?;
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        match image {
            ImageInput::Path(tiff_path) => {
                let frame = Frame::from_tiff(tiff_path).map_err(|err| {
                    PyValueError::new_err(format!("Failed to load TIFF: {}", err))
                })?;
                self.check_mask((frame.dim1, frame.dim2), "image")?;

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
                    (errors, Some(range)) => PyIntegrationResult::from_azimuthal_range_2d(
                        self.integrator
                            .integrate2d_on_azimuthal_range(
                                &frame,
                                n_radial,
                                n_azimuthal,
                                range,
                                errors,
                            )
                            .map_err(PyValueError::new_err)?,
                        n_radial,
                        n_azimuthal,
                    ),
                };
                Ok(result)
            }
            ImageInput::Array(array) => {
                let (dim1, dim2) = extract_array_dims(&array, "float64")?;
                let slice = array.as_slice().map_err(|err| {
                    PyValueError::new_err(format!("Array must be contiguous: {}", err))
                })?;
                self.check_mask((dim1, dim2), "image")?;
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
                    (errors, Some(range)) => PyIntegrationResult::from_azimuthal_range_2d(
                        self.integrator
                            .integrate2d_on_azimuthal_range_from_slice(
                                slice,
                                dim1,
                                dim2,
                                n_radial,
                                n_azimuthal,
                                range,
                                errors,
                            )
                            .map_err(PyValueError::new_err)?,
                        n_radial,
                        n_azimuthal,
                    ),
                };
                Ok(result)
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
    /// Available methods: "pixel_split", "pyfai_full", "counting", "histogram", or "supersampled".
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
        check_method_radial_range(&method, self.integrator.radial_range())?;
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
    /// while the workers run.
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
    fn batch1d(
        &self,
        py: Python<'_>,
        images: BatchImageInput,
        n_points: usize,
        errors: bool,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_std: bool,
    ) -> PyResult<Vec<PyBatchResult>> {
        check_positive("n_points", n_points)?;
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
                            Ok(result) => PyBatchResult {
                                filename,
                                result: Some(PyIntegrationResult::from_azimuthal(result)),
                                error: None,
                            },
                            Err(error) => PyBatchResult {
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

                    PyBatchResult {
                        filename: format!("array[{}]", i),
                        result: Some(integration_result),
                        error: None,
                    }
                };
                // Release the GIL while the workers run (see `process_files`).
                // `frames` holds only plain slices whose borrow guards live in
                // `arrays`, which outlives this region.
                let results = py.detach(|| {
                    if let Some(&(_, dim1, dim2)) = frames.first()
                        && frames.iter().all(|&(_, h, w)| (h, w) == (dim1, dim2))
                    {
                        self.integrator.prepare1d(dim1, dim2, n_points);
                    }
                    frames.par_iter().enumerate().map(integrate).collect()
                });

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
            ClassifiedInput::Arrays(arrays) => {
                let num_arrays = arrays.len();

                // Convert all arrays to owned Frame objects
                let frames: Result<Vec<Frame>, PyErr> = arrays
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
                        Ok(Frame {
                            name: format!("array[{}]", i),
                            data: slice.to_vec(),
                            dim1,
                            dim2,
                        })
                    })
                    .collect();

                let frames = frames?;

                let integrator = Arc::clone(&self.integrator);
                let stream_error = StreamError::default();
                let receiver =
                    if azimuthal_std {
                        BatchReceiver::Integration1DWithAzimuthalStd(Mutex::new(
                            spawn_array_stream(frames, stream_error.clone(), move |frame| {
                                integrator.integrate1d_with_azimuthal_std(frame, n_points, errors)
                            }),
                        ))
                    } else if errors {
                        BatchReceiver::Integration1DWithErrors(Mutex::new(spawn_array_stream(
                            frames,
                            stream_error.clone(),
                            move |frame| integrator.integrate1d_with_errors(frame, n_points),
                        )))
                    } else {
                        BatchReceiver::Integration1D(Mutex::new(spawn_array_stream(
                            frames,
                            stream_error.clone(),
                            move |frame| integrator.integrate1d(frame, n_points),
                        )))
                    };
                Ok(PyBatchIterator::with_stream_error(
                    receiver,
                    num_arrays,
                    stream_error,
                ))
            }
            ClassifiedInput::Iterator(iter) => {
                // Generator/iterator path: a producer thread pulls items lazily.
                let pool = Self::iterator_pool(num_workers, num_threads)?;
                let (work_tx, work_rx) = std::sync::mpsc::sync_channel(num_workers * 2);
                let stream_error = StreamError::default();
                spawn_iterator_producer(
                    iter.unbind(),
                    work_tx,
                    mask_shape(&self.integrator),
                    stream_error.clone(),
                );

                let integrator = Arc::clone(&self.integrator);
                if azimuthal_std {
                    let task = Integration1DWithAzimuthalStd {
                        n_bins: n_points,
                        errors,
                    };
                    let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                    return Ok(PyBatchIterator::with_stream_error(
                        BatchReceiver::Integration1DWithAzimuthalStd(Mutex::new(
                            batch.into_receiver(),
                        )),
                        0,
                        stream_error.clone(),
                    ));
                }
                if errors {
                    let task = Integration1DWithErrors { n_bins: n_points };
                    let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                    Ok(PyBatchIterator::with_stream_error(
                        BatchReceiver::Integration1DWithErrors(Mutex::new(batch.into_receiver())),
                        0,
                        stream_error.clone(),
                    ))
                } else {
                    let task = Integration1D { n_bins: n_points };
                    let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                    Ok(PyBatchIterator::with_stream_error(
                        BatchReceiver::Integration1D(Mutex::new(batch.into_receiver())),
                        0,
                        stream_error.clone(),
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
    /// * `tiff_paths` - List of TIFF file paths to process
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
    ///   centres. Radial bins are the same as without a range.
    ///   With `method="pyfai_full"`, individual bins depend on how the range is
    ///   written (a range shifted by 2*pi, or a sector vs the same rows of a full
    ///   circle, can differ, as in pyFAI itself); prefer `"pixel_split"` for
    ///   sectors.
    /// * `num_workers` - Number of parallel workers (optional)
    /// * `num_threads` - Threads per worker (optional)
    ///
    /// # Returns
    ///
    /// List of BatchResult objects, each containing filename and IntegrationResult.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the integrator uses the Histogram method, which has
    /// no azimuthal binning.
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
    #[pyo3(signature = (tiff_paths, n_radial, n_azimuthal=None, errors=false, azimuthal_range=None, num_workers=None, num_threads=None, *, azimuthal_bracket=None))]
    fn batch2d(
        &self,
        py: Python<'_>,
        tiff_paths: Vec<String>,
        n_radial: usize,
        n_azimuthal: Option<usize>,
        errors: bool,
        azimuthal_range: Option<(f64, f64)>,
        num_workers: Option<usize>,
        num_threads: Option<usize>,
        azimuthal_bracket: Option<f64>,
    ) -> PyResult<Vec<PyBatchResult>> {
        self.reject_histogram_2d("batch2d")?;
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
        let (num_workers, num_threads) = resolve_workers_threads(num_workers, num_threads)?;
        validate_azimuthal_range(azimuthal_range, n_azimuthal)?;

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
    ///   centres. Radial bins are the same as without a range.
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
        self.reject_histogram_2d("batch2d_iter")?;
        check_positive("n_radial", n_radial)?;
        let (n_azimuthal, azimuthal_range) =
            resolve_azimuthal_bins(n_azimuthal, azimuthal_range, azimuthal_bracket)?;
        check_positive("n_azimuthal", n_azimuthal)?;
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

        // For arrays or iterators, use the queue-based streaming approach.
        // Array streams are finite and need no Python, so they share the
        // cached workers; a Python iterator gets its own (see `iterator_pool`).
        let pool = if matches!(classified, ClassifiedInput::Iterator(_)) {
            Self::iterator_pool(num_workers, num_threads)?
        } else {
            self.file_pool(num_workers, num_threads)?
        };
        let (work_tx, work_rx) = std::sync::mpsc::sync_channel(num_workers * 2);
        let stream_error = StreamError::default();

        let total = match classified {
            ClassifiedInput::Arrays(arrays) => {
                let num_arrays = arrays.len();
                // Convert all arrays to work items and send them
                let mut items = Vec::with_capacity(num_arrays);
                for (i, array) in arrays.iter().enumerate() {
                    let (dim1, dim2) = extract_item_dims(array, i)?;
                    let slice = array.as_slice().map_err(|err| {
                        PyValueError::new_err(format!("Array {} must be contiguous: {}", i, err))
                    })?;
                    self.check_mask((dim1, dim2), &format!("images[{i}]"))?;
                    let frame = Frame {
                        name: format!("array[{}]", i),
                        data: slice.to_vec(),
                        dim1,
                        dim2,
                    };
                    items.push(WorkItem::Frame {
                        index: i,
                        name: frame.name.clone(),
                        frame,
                    });
                }
                // Start sending after returning to the consumer setup below;
                // filling a bounded queue here deadlocks before workers exist.
                std::thread::spawn(move || {
                    for item in items {
                        if work_tx.send(item).is_err() {
                            break;
                        }
                    }
                });
                num_arrays
            }
            ClassifiedInput::Iterator(iter) => {
                spawn_iterator_producer(
                    iter.unbind(),
                    work_tx,
                    mask_shape(&self.integrator),
                    stream_error.clone(),
                );
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
                let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                Ok(PyBatchIterator::with_stream_error(
                    BatchReceiver::Integration2D {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    stream_error.clone(),
                ))
            }
            (true, None) => {
                let task = Integration2DWithErrors {
                    n_radial,
                    n_azimuthal,
                };
                let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                Ok(PyBatchIterator::with_stream_error(
                    BatchReceiver::Integration2DWithErrors {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    stream_error.clone(),
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
                let batch = process_queue_with_pool(work_rx, integrator, task, pool);
                Ok(PyBatchIterator::with_stream_error(
                    BatchReceiver::Integration2DOnAzimuthalRange {
                        receiver: Mutex::new(batch.into_receiver()),
                        n_radial,
                        n_azimuthal,
                    },
                    total,
                    stream_error.clone(),
                ))
            }
        }
    }
}
