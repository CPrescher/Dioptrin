//! Experimental CUDA 1D plans: `Integrator.prepare_cuda1d`, `Cuda1dPlan`,
//! `cuda_available`, `cuda_info` and `CudaUnavailableError`.
//!
//! These names exist in every build. Without the `cuda` cargo feature the Rust
//! core reports `compiled: false` and `prepare_cuda1d` raises
//! `CudaUnavailableError` after the configuration checks.
use numpy::{
    Element, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::results::PyIntegrationResult;
use crate::integration::{
    Cuda1dOptions, Cuda1dPlan, CudaError, CudaPixel, CudaPrecision, Fixed1dResult, Integrator,
    cuda_info,
};

create_exception!(
    dioptrin,
    CudaUnavailableError,
    PyRuntimeError,
    "Experimental: CUDA cannot be used. Raised when dioptrin was built without \
     the `cuda` feature, or when the driver, NVRTC or the device cannot be \
     initialised. A subclass of RuntimeError."
);

const NOT_COMPILED: &str = "dioptrin was built without CUDA support; build with: \
     maturin build --release --features python,cuda";

/// Python exception for a CUDA error (contract table: not compiled and
/// unavailable raise `CudaUnavailableError`, configuration and input errors
/// `ValueError`, driver errors at run time `RuntimeError`).
fn cuda_error(error: CudaError) -> PyErr {
    match error {
        CudaError::NotCompiled => CudaUnavailableError::new_err(NOT_COMPILED),
        CudaError::Unavailable(_) => CudaUnavailableError::new_err(error.to_string()),
        CudaError::Unsupported(_) | CudaError::InvalidInput(_) => {
            PyValueError::new_err(error.to_string())
        }
        CudaError::Device(_) => PyRuntimeError::new_err(error.to_string()),
    }
}

fn to_python(result: Fixed1dResult) -> PyIntegrationResult {
    PyIntegrationResult {
        radial: Some(result.radial),
        azimuthal: None,
        intensity: result.intensity,
        errors: result.errors,
        azimuthal_std: None,
        n_radial: None,
        n_azimuthal: None,
    }
}

/// Uploads the integrator's fixed-mask 1D table; called by
/// `Integrator.prepare_cuda1d`. The GIL is released while the table is built
/// and uploaded.
pub(crate) fn prepare(
    py: Python<'_>,
    integrator: &Integrator,
    shape: (usize, usize),
    n_points: usize,
    precision: &str,
    device: usize,
    max_batch_bytes: Option<usize>,
) -> PyResult<PyCuda1dPlan> {
    let precision: CudaPrecision = precision.parse().map_err(cuda_error)?;
    let mut options = Cuda1dOptions::new()
        .with_precision(precision)
        .with_device(device);
    if let Some(bytes) = max_batch_bytes {
        options = options.with_max_batch_bytes(bytes);
    }
    let plan = py
        .detach(|| integrator.prepare_cuda1d(shape, n_points, &options))
        .map_err(cuda_error)?;
    Ok(PyCuda1dPlan { plan })
}

/// Experimental: whether CUDA device `device` can be used by this build.
#[pyfunction]
#[pyo3(signature = (device=0))]
pub fn cuda_available(py: Python<'_>, device: usize) -> bool {
    py.detach(|| cuda_info(device)).available
}

/// Experimental: describe the CUDA build and device `device` as a dict.
#[pyfunction(name = "cuda_info")]
#[pyo3(signature = (device=0))]
pub fn cuda_info_py(py: Python<'_>, device: usize) -> PyResult<Bound<'_, PyDict>> {
    let info = py.detach(|| cuda_info(device));
    let dict = PyDict::new(py);
    dict.set_item("compiled", info.compiled)?;
    dict.set_item("available", info.available)?;
    dict.set_item("reason", info.reason)?;
    dict.set_item("device", info.device)?;
    dict.set_item("device_name", info.device_name)?;
    dict.set_item("compute_capability", info.compute_capability)?;
    dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
    dict.set_item("driver_version", info.driver_version)?;
    dict.set_item("nvrtc_version", info.nvrtc_version)?;
    dict.set_item("bindings_version", info.bindings_version)?;
    Ok(dict)
}

/// Pixel type of one input array.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pixel {
    F32,
    F64,
}

impl Pixel {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

/// What `obj` is, for a `TypeError`: its qualified type name, and for a NumPy
/// scalar such as `np.float64(1.0)` that it is a scalar, not an array.
fn received(obj: &Bound<'_, PyAny>) -> String {
    let ty = obj.get_type();
    let name = ty
        .fully_qualified_name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "object".to_owned());
    let scalar = obj
        .py()
        .import("numpy")
        .and_then(|numpy| numpy.getattr("generic"))
        .and_then(|generic| obj.is_instance(&generic))
        .unwrap_or(false);
    if scalar {
        format!("a numpy scalar of type {name}, not an array")
    } else {
        name
    }
}

/// The array and pixel type of `obj`: `TypeError` unless it is a NumPy array
/// of native float64 or float32.
fn pixel_type<'a, 'py>(
    obj: &'a Bound<'py, PyAny>,
    what: &str,
) -> PyResult<(&'a Bound<'py, PyUntypedArray>, Pixel)> {
    let py = obj.py();
    let array = obj.cast::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a numpy array with dtype float64 or float32, got {}",
            received(obj)
        ))
    })?;
    let dtype = array.dtype();
    let pixel = if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        Pixel::F64
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        Pixel::F32
    } else {
        return Err(PyTypeError::new_err(format!(
            "{what} must have dtype float64 or float32, got dtype {dtype}"
        )));
    };
    Ok((array, pixel))
}

/// Shape and layout checks shared by single images and stacks: `ValueError`
/// for a wrong number of dimensions, a wrong image shape, a layout that is
/// not C contiguous, or data that is not aligned for the dtype.
fn check_layout(
    array: &Bound<'_, PyUntypedArray>,
    what: &str,
    ndim: usize,
    shape: (usize, usize),
) -> PyResult<()> {
    let actual = array.shape();
    if actual.len() != ndim {
        return Err(PyValueError::new_err(format!(
            "{what} must be a {ndim}D array, got shape {actual:?}"
        )));
    }
    if actual[ndim - 2..] != [shape.0, shape.1] {
        return Err(PyValueError::new_err(format!(
            "{what} shape {actual:?} does not match prepared plan shape ({}, {})",
            shape.0, shape.1
        )));
    }
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{what} must be C contiguous"
        )));
    }
    // An empty array needs no alignment: it is borrowed as an empty slice.
    if array.len() > 0 && !array.is_aligned() {
        return Err(PyValueError::new_err(format!(
            "{what} is not aligned for its dtype (flags.aligned is False); pass an \
             aligned copy, for example {what}.copy()"
        )));
    }
    Ok(())
}

/// `ValueError` for a plan shape with a zero side. Plans never have one
/// (`prepare_cuda1d` rejects it); this guards the private test hook.
fn check_shape(shape: (usize, usize)) -> PyResult<()> {
    if shape.0 == 0 || shape.1 == 0 {
        return Err(PyValueError::new_err(format!(
            "plan shape ({}, {}) must have non-zero sides",
            shape.0, shape.1
        )));
    }
    Ok(())
}

/// Borrow a checked 2D array without copying. Read-only arrays and
/// `np.memmap` views are accepted: the borrow is shared.
fn borrow2<'py, T: Element>(
    obj: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray2<'py, T>> {
    obj.cast::<PyArray2<T>>()
        .map_err(|err| PyTypeError::new_err(format!("{what}: {err}")))?
        .try_readonly()
        .map_err(|err| PyRuntimeError::new_err(format!("{what}: {err}")))
}

fn borrow3<'py, T: Element>(
    obj: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray3<'py, T>> {
    obj.cast::<PyArray3<T>>()
        .map_err(|err| PyTypeError::new_err(format!("{what}: {err}")))?
        .try_readonly()
        .map_err(|err| PyRuntimeError::new_err(format!("{what}: {err}")))
}

fn contiguous<'a, T: Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C contiguous and aligned")))
}

/// Checked input arrays of one call, borrowed (never copied) until the call
/// returns.
enum Arrays<'py> {
    F64(Vec<PyReadonlyArray2<'py, f64>>),
    F32(Vec<PyReadonlyArray2<'py, f32>>),
    Stack64(PyReadonlyArray3<'py, f64>),
    Stack32(PyReadonlyArray3<'py, f32>),
}

/// The arrays of one call and how its error messages name them: `image` for
/// `integrate1d`, `images[i]` or `images` for `batch1d`.
struct Borrowed<'py> {
    arrays: Arrays<'py>,
    single: bool,
}

fn borrow_list<'py, T: Element>(
    items: &[Bound<'py, PyAny>],
) -> PyResult<Vec<PyReadonlyArray2<'py, T>>> {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| borrow2::<T>(item, &format!("images[{index}]")))
        .collect()
}

impl<'py> Borrowed<'py> {
    /// One 2D image (`Cuda1dPlan.integrate1d`).
    fn image(image: &Bound<'py, PyAny>, shape: (usize, usize)) -> PyResult<Self> {
        check_shape(shape)?;
        let (array, pixel) = pixel_type(image, "image")?;
        check_layout(array, "image", 2, shape)?;
        let arrays = match pixel {
            Pixel::F64 => Arrays::F64(vec![borrow2(image, "image")?]),
            Pixel::F32 => Arrays::F32(vec![borrow2(image, "image")?]),
        };
        Ok(Self {
            arrays,
            single: true,
        })
    }

    /// A list or tuple of 2D images of one dtype, or one 3D stack
    /// (`Cuda1dPlan.batch1d`).
    fn batch(images: &Bound<'py, PyAny>, shape: (usize, usize)) -> PyResult<Self> {
        check_shape(shape)?;
        let arrays = Self::batch_arrays(images, shape)?;
        Ok(Self {
            arrays,
            single: false,
        })
    }

    fn batch_arrays(images: &Bound<'py, PyAny>, shape: (usize, usize)) -> PyResult<Arrays<'py>> {
        if images.cast::<PyUntypedArray>().is_ok() {
            let (array, pixel) = pixel_type(images, "images")?;
            if array.ndim() != 3 {
                return Err(PyValueError::new_err(format!(
                    "images must be a list or tuple of 2D arrays or one 3D array, got shape {:?}",
                    array.shape()
                )));
            }
            check_layout(array, "images", 3, shape)?;
            return Ok(match pixel {
                Pixel::F64 => Arrays::Stack64(borrow3(images, "images")?),
                Pixel::F32 => Arrays::Stack32(borrow3(images, "images")?),
            });
        }
        if !(images.is_instance_of::<PyList>() || images.is_instance_of::<PyTuple>()) {
            return Err(PyTypeError::new_err(format!(
                "images must be a list or tuple of 2D numpy arrays or one 3D numpy array, got {}",
                received(images)
            )));
        }
        let items = images
            .try_iter()?
            .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?;
        let Some(first) = items.first() else {
            return Ok(Arrays::F64(Vec::new()));
        };
        // Dtypes first, so a mixed list is a TypeError whatever the shapes.
        let (_, pixel) = pixel_type(first, "images[0]")?;
        for (index, item) in items.iter().enumerate().skip(1) {
            let what = format!("images[{index}]");
            let (_, other) = pixel_type(item, &what)?;
            if other != pixel {
                return Err(PyTypeError::new_err(format!(
                    "{what} has dtype {}, but images[0] has dtype {}; all images in one \
                     call must have the same dtype",
                    other.name(),
                    pixel.name()
                )));
            }
        }
        for (index, item) in items.iter().enumerate() {
            let what = format!("images[{index}]");
            let (array, _) = pixel_type(item, &what)?;
            check_layout(array, &what, 2, shape)?;
        }
        Ok(match pixel {
            Pixel::F64 => Arrays::F64(borrow_list(&items)?),
            Pixel::F32 => Arrays::F32(borrow_list(&items)?),
        })
    }

    fn pixel(&self) -> Pixel {
        match self.arrays {
            Arrays::F64(_) | Arrays::Stack64(_) => Pixel::F64,
            Arrays::F32(_) | Arrays::Stack32(_) => Pixel::F32,
        }
    }

    /// The name of list item `index` in error messages.
    fn label(&self, index: usize) -> String {
        if self.single {
            "image".to_owned()
        } else {
            format!("images[{index}]")
        }
    }

    fn slices<'a, T: Element>(
        &self,
        arrays: &'a [PyReadonlyArray2<'_, T>],
    ) -> PyResult<Vec<&'a [T]>> {
        arrays
            .iter()
            .enumerate()
            .map(|(index, array)| contiguous(array, &self.label(index)))
            .collect()
    }

    /// Number of frames, after taking the same slices `run` passes on.
    fn frames(&self, shape: (usize, usize)) -> PyResult<usize> {
        let pixels = shape.0 * shape.1;
        let stack = |len: usize| {
            len.checked_div(pixels).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "plan shape ({}, {}) must have non-zero sides",
                    shape.0, shape.1
                ))
            })
        };
        match &self.arrays {
            Arrays::F64(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::F32(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::Stack64(array) => stack(contiguous(array, "images")?.len()),
            Arrays::Stack32(array) => stack(contiguous(array, "images")?.len()),
        }
    }

    /// Integrate on the device with the GIL released. The guards in `self`
    /// stay alive until the device call has synchronised.
    fn run(
        &self,
        py: Python<'_>,
        plan: &Cuda1dPlan,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let results = match &self.arrays {
            Arrays::F64(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::F32(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::Stack64(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
            Arrays::Stack32(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
        }
        .map_err(cuda_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }
}

fn frames_call<T: CudaPixel>(
    py: Python<'_>,
    plan: &Cuda1dPlan,
    frames: &[&[T]],
    errors: bool,
) -> Result<Vec<Fixed1dResult>, CudaError> {
    py.detach(|| plan.integrate_frames(frames, errors))
}

fn stack_call<T: CudaPixel>(
    py: Python<'_>,
    plan: &Cuda1dPlan,
    stack: &[T],
    errors: bool,
) -> Result<Vec<Fixed1dResult>, CudaError> {
    py.detach(|| plan.integrate_stack(stack, errors))
}

/// Private test hook: run the input checks and borrows of
/// `Cuda1dPlan.integrate1d` (`batch=False`) or `Cuda1dPlan.batch1d`
/// (`batch=True`) for a plan of `shape`, without a device. Returns the number
/// of frames and their dtype name. Not part of the API.
#[pyfunction(name = "_cuda_check_input")]
#[pyo3(signature = (images, shape, batch))]
pub fn cuda_check_input(
    images: &Bound<'_, PyAny>,
    shape: (usize, usize),
    batch: bool,
) -> PyResult<(usize, &'static str)> {
    let borrowed = if batch {
        Borrowed::batch(images, shape)?
    } else {
        Borrowed::image(images, shape)?
    };
    Ok((borrowed.frames(shape)?, borrowed.pixel().name()))
}

/// Experimental: a fixed-mask 1D lookup table uploaded to one CUDA device.
///
/// Created by `Integrator.prepare_cuda1d`. A snapshot of the integrator's
/// mask, corrections, method, unit and radial range; it may be shared between
/// threads, and calls on one plan run one at a time.
#[pyclass(name = "Cuda1dPlan", frozen)]
pub struct PyCuda1dPlan {
    plan: Cuda1dPlan,
}

#[pymethods]
impl PyCuda1dPlan {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.plan.shape()
    }
    #[getter]
    fn n_points(&self) -> usize {
        self.plan.n_points()
    }
    #[getter]
    fn precision(&self) -> &'static str {
        self.plan.precision().as_str()
    }
    #[getter]
    fn device(&self) -> usize {
        self.plan.device()
    }
    #[getter]
    fn device_name(&self) -> String {
        self.plan.device_name().to_owned()
    }
    #[getter]
    fn radial(&self) -> Vec<f64> {
        self.plan.radial().to_vec()
    }
    #[getter]
    fn max_batch_frames(&self) -> usize {
        self.plan.max_batch_frames()
    }
    #[getter]
    fn table_entries(&self) -> usize {
        self.plan.table_entries()
    }

    fn __repr__(&self) -> String {
        let (dim1, dim2) = self.plan.shape();
        format!(
            "Cuda1dPlan(shape=({dim1}, {dim2}), n_points={}, precision={:?}, device={}, device_name={:?})",
            self.plan.n_points(),
            self.plan.precision().as_str(),
            self.plan.device(),
            self.plan.device_name()
        )
    }

    /// Integrate one C-contiguous float64 or float32 image of the plan's
    /// shape. The array is borrowed without copying and the GIL is released
    /// while the frame is copied to the device, integrated and copied back.
    #[pyo3(signature = (image, *, errors=false))]
    fn integrate1d(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let borrowed = Borrowed::image(image, self.plan.shape())?;
        borrowed
            .run(py, &self.plan, errors)?
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("CUDA integration returned no result"))
    }

    /// Integrate a list or tuple of 2D images of one dtype, or one C-contiguous
    /// 3D array `(n, dim1, dim2)`. Arrays are borrowed without copying for
    /// this synchronous call and must not be mutated concurrently; the GIL is
    /// released during the device work. Returns one IntegrationResult per
    /// image; an empty list returns [].
    #[pyo3(signature = (images, *, errors=false))]
    fn batch1d(
        &self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        Borrowed::batch(images, self.plan.shape())?.run(py, &self.plan, errors)
    }
}
