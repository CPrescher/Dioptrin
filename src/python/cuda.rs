//! Experimental CUDA 1D plans: `Integrator.prepare_cuda1d`, `Cuda1dPlan`,
//! `cuda_available`, `cuda_info` and `CudaUnavailableError`.
//!
//! These names exist in every build. Without the `cuda` cargo feature the Rust
//! core reports `compiled: false` and `prepare_cuda1d` raises
//! `CudaUnavailableError` after the configuration checks.
use std::ffi::c_int;
use std::sync::Mutex;

use numpy::{
    Element, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::results::PyIntegrationResult;
use crate::integration::{
    Bslz4Dtype, Cuda1dOptions, Cuda1dPlan, Cuda2dPlan, CudaError, CudaPinnedBytes, CudaPixel,
    CudaPrecision, Fixed1dResult, Fixed2dResult, Integrator, cuda_info,
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

fn to_python_2d(result: Fixed2dResult) -> PyIntegrationResult {
    PyIntegrationResult {
        radial: result.radial,
        azimuthal: result.azimuthal,
        intensity: result.intensity,
        errors: result.errors,
        azimuthal_std: None,
        n_radial: Some(result.n_radial),
        n_azimuthal: Some(result.n_azimuthal),
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

/// Uploads the integrator's fixed-mask 2D table; called by
/// `Integrator.prepare_cuda2d`. The GIL is released while the table is built
/// and uploaded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_2d(
    py: Python<'_>,
    integrator: &Integrator,
    shape: (usize, usize),
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    precision: &str,
    device: usize,
    max_batch_bytes: Option<usize>,
) -> PyResult<PyCuda2dPlan> {
    let precision: CudaPrecision = precision.parse().map_err(cuda_error)?;
    let mut options = Cuda1dOptions::new()
        .with_precision(precision)
        .with_device(device);
    if let Some(bytes) = max_batch_bytes {
        options = options.with_max_batch_bytes(bytes);
    }
    let plan = py
        .detach(|| {
            integrator.prepare_cuda2d(shape, n_radial, n_azimuthal, azimuthal_range, &options)
        })
        .map_err(cuda_error)?;
    Ok(PyCuda2dPlan { plan })
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
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl Pixel {
    fn name(self) -> &'static str {
        match self {
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::I32 => "int32",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    fn itemsize(self) -> usize {
        match self {
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::I32 => size_of::<i32>(),
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
        }
    }
}

/// Owns a CUDA page-locked host allocation exposed to NumPy via the CPython
/// buffer protocol so `np.frombuffer(owner, dtype=...).reshape(...)` yields a
/// standard writable NumPy array whose `.base` keeps the allocation alive.
#[pyclass(name = "_CudaPinnedBuffer", module = "dioptrin")]
struct PinnedBufferOwner {
    _buf: Mutex<CudaPinnedBytes>,
    ptr: usize,
    len: isize,
}

#[pymethods]
impl PinnedBufferOwner {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let owner = slf.borrow();
        let ret = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                owner.ptr as *mut std::ffi::c_void,
                owner.len,
                0, // writable
                flags,
            )
        };
        if ret != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
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
/// of native float64, float32, uint16, uint32, or int32.
fn pixel_type<'a, 'py>(
    obj: &'a Bound<'py, PyAny>,
    what: &str,
) -> PyResult<(&'a Bound<'py, PyUntypedArray>, Pixel)> {
    let py = obj.py();
    let array = obj.cast::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a numpy array with dtype float64, float32, uint16, uint32 or int32, got {}",
            received(obj)
        ))
    })?;
    let dtype = array.dtype();
    let pixel = if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        Pixel::F64
    } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        Pixel::F32
    } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
        Pixel::U16
    } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
        Pixel::U32
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        Pixel::I32
    } else {
        return Err(PyTypeError::new_err(format!(
            "{what} must have dtype float64, float32, uint16, uint32 or int32, got dtype {dtype}"
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
    U16(Vec<PyReadonlyArray2<'py, u16>>),
    U32(Vec<PyReadonlyArray2<'py, u32>>),
    I32(Vec<PyReadonlyArray2<'py, i32>>),
    F64(Vec<PyReadonlyArray2<'py, f64>>),
    F32(Vec<PyReadonlyArray2<'py, f32>>),
    StackU16(PyReadonlyArray3<'py, u16>),
    StackU32(PyReadonlyArray3<'py, u32>),
    StackI32(PyReadonlyArray3<'py, i32>),
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
            Pixel::U16 => Arrays::U16(vec![borrow2(image, "image")?]),
            Pixel::U32 => Arrays::U32(vec![borrow2(image, "image")?]),
            Pixel::I32 => Arrays::I32(vec![borrow2(image, "image")?]),
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
                Pixel::U16 => Arrays::StackU16(borrow3(images, "images")?),
                Pixel::U32 => Arrays::StackU32(borrow3(images, "images")?),
                Pixel::I32 => Arrays::StackI32(borrow3(images, "images")?),
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
            Pixel::U16 => Arrays::U16(borrow_list(&items)?),
            Pixel::U32 => Arrays::U32(borrow_list(&items)?),
            Pixel::I32 => Arrays::I32(borrow_list(&items)?),
            Pixel::F64 => Arrays::F64(borrow_list(&items)?),
            Pixel::F32 => Arrays::F32(borrow_list(&items)?),
        })
    }

    fn pixel(&self) -> Pixel {
        match self.arrays {
            Arrays::U16(_) | Arrays::StackU16(_) => Pixel::U16,
            Arrays::U32(_) | Arrays::StackU32(_) => Pixel::U32,
            Arrays::I32(_) | Arrays::StackI32(_) => Pixel::I32,
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
            Arrays::U16(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::U32(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::I32(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::F64(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::F32(arrays) => Ok(self.slices(arrays)?.len()),
            Arrays::StackU16(array) => stack(contiguous(array, "images")?.len()),
            Arrays::StackU32(array) => stack(contiguous(array, "images")?.len()),
            Arrays::StackI32(array) => stack(contiguous(array, "images")?.len()),
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
            Arrays::U16(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::U32(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::I32(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::F64(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::F32(arrays) => frames_call(py, plan, &self.slices(arrays)?, errors),
            Arrays::StackU16(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
            Arrays::StackU32(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
            Arrays::StackI32(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
            Arrays::Stack64(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
            Arrays::Stack32(array) => stack_call(py, plan, contiguous(array, "images")?, errors),
        }
        .map_err(cuda_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }

    fn run_2d(
        &self,
        py: Python<'_>,
        plan: &Cuda2dPlan,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let results = match &self.arrays {
            Arrays::U16(arrays) => frames_call_2d(py, plan, &self.slices(arrays)?, errors),
            Arrays::U32(arrays) => frames_call_2d(py, plan, &self.slices(arrays)?, errors),
            Arrays::I32(arrays) => frames_call_2d(py, plan, &self.slices(arrays)?, errors),
            Arrays::F64(arrays) => frames_call_2d(py, plan, &self.slices(arrays)?, errors),
            Arrays::F32(arrays) => frames_call_2d(py, plan, &self.slices(arrays)?, errors),
            Arrays::StackU16(array) => {
                stack_call_2d(py, plan, contiguous(array, "images")?, errors)
            }
            Arrays::StackU32(array) => {
                stack_call_2d(py, plan, contiguous(array, "images")?, errors)
            }
            Arrays::StackI32(array) => {
                stack_call_2d(py, plan, contiguous(array, "images")?, errors)
            }
            Arrays::Stack64(array) => stack_call_2d(py, plan, contiguous(array, "images")?, errors),
            Arrays::Stack32(array) => stack_call_2d(py, plan, contiguous(array, "images")?, errors),
        }
        .map_err(cuda_error)?;
        Ok(results.into_iter().map(to_python_2d).collect())
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

fn frames_call_2d<T: CudaPixel>(
    py: Python<'_>,
    plan: &Cuda2dPlan,
    frames: &[&[T]],
    errors: bool,
) -> Result<Vec<Fixed2dResult>, CudaError> {
    py.detach(|| plan.integrate_frames(frames, errors))
}

fn stack_call_2d<T: CudaPixel>(
    py: Python<'_>,
    plan: &Cuda2dPlan,
    stack: &[T],
    errors: bool,
) -> Result<Vec<Fixed2dResult>, CudaError> {
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

    /// Allocate a writable 3D NumPy array `(n_frames, dim1, dim2)` backed by
    /// CUDA page-locked (pinned) host memory on this plan's device.
    ///
    /// Readers such as `dioptrin-io` or `h5py` (`dataset.read_direct(buf)`) can
    /// decode directly into the returned array; passing it to `batch1d` or
    /// `integrate1d` then skips the CPU staging copy and DMAs directly to the
    /// GPU. Supported dtypes: `uint16`, `uint32`, `int32`, `float32`, `float64`.
    #[pyo3(signature = (n_frames, dtype=None))]
    fn alloc_pinned_batch<'py>(
        &self,
        py: Python<'py>,
        n_frames: usize,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        let np_dtype = match dtype {
            Some(d) => numpy.call_method1("dtype", (d,))?,
            None => numpy.call_method1("dtype", ("float32",))?,
        };
        let descr = np_dtype
            .cast::<numpy::PyArrayDescr>()
            .map_err(|_| PyTypeError::new_err("dtype must be a valid numpy dtype"))?;
        let pixel = if descr.is_equiv_to(&numpy::dtype::<u16>(py)) {
            Pixel::U16
        } else if descr.is_equiv_to(&numpy::dtype::<u32>(py)) {
            Pixel::U32
        } else if descr.is_equiv_to(&numpy::dtype::<i32>(py)) {
            Pixel::I32
        } else if descr.is_equiv_to(&numpy::dtype::<f32>(py)) {
            Pixel::F32
        } else if descr.is_equiv_to(&numpy::dtype::<f64>(py)) {
            Pixel::F64
        } else {
            return Err(PyTypeError::new_err(format!(
                "dtype must be native uint16, uint32, int32, float32 or float64, got {descr}"
            )));
        };
        let (dim1, dim2) = self.plan.shape();
        let elems = n_frames
            .checked_mul(dim1)
            .and_then(|n| n.checked_mul(dim2))
            .ok_or_else(|| PyValueError::new_err("pinned batch shape overflows usize"))?;
        let bytes = elems
            .checked_mul(pixel.itemsize())
            .ok_or_else(|| PyValueError::new_err("pinned batch byte size overflows usize"))?;
        let len = isize::try_from(bytes)
            .map_err(|_| PyValueError::new_err("pinned batch byte size overflows isize"))?;
        let mut pinned = py
            .detach(|| self.plan.alloc_pinned_bytes(bytes))
            .map_err(cuda_error)?;
        let ptr = pinned.as_mut_ptr().map_err(cuda_error)? as usize;
        let owner = Py::new(
            py,
            PinnedBufferOwner {
                _buf: Mutex::new(pinned),
                ptr,
                len,
            },
        )?;
        let flat = numpy.call_method1("frombuffer", (owner.bind(py), np_dtype))?;
        flat.call_method1("reshape", ((n_frames, dim1, dim2),))
    }

    /// Decompress raw HDF5 Bitshuffle+LZ4 chunks directly on the GPU and
    /// integrate them in VRAM without transferring uncompressed frames over PCIe.
    ///
    /// `chunks` is a list or tuple of raw compressed chunk buffers (`bytes`,
    /// `bytearray`, `memoryview`, or C-contiguous `uint8` arrays — e.g. from
    /// `h5py.Dataset.id.read_direct_chunk` or `dioptrin-io`).
    /// Supported dtypes: `uint16`, `uint32`, `int32`.
    #[pyo3(signature = (chunks, dtype="uint32", *, errors=false))]
    fn batch1d_bslz4(
        &self,
        py: Python<'_>,
        chunks: &Bound<'_, PyAny>,
        dtype: &str,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let bslz4_dtype = match dtype {
            "uint16" | "u16" => Bslz4Dtype::U16,
            "uint32" | "u32" => Bslz4Dtype::U32,
            "int32" | "i32" => Bslz4Dtype::I32,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "dtype must be 'uint16', 'uint32' or 'int32', got {dtype:?}"
                )));
            }
        };
        if !(chunks.is_instance_of::<PyList>() || chunks.is_instance_of::<PyTuple>()) {
            return Err(PyTypeError::new_err(format!(
                "chunks must be a list or tuple of byte buffers, got {}",
                received(chunks)
            )));
        }
        let mut buffers: Vec<PyBuffer<u8>> = Vec::new();
        for item in chunks.try_iter()? {
            let item = item?;
            let buf = PyBuffer::<u8>::get(&item).map_err(|e| {
                PyTypeError::new_err(format!("chunk must support the byte buffer protocol: {e}"))
            })?;
            if !buf.is_c_contiguous() {
                return Err(PyValueError::new_err("chunk buffer must be C contiguous"));
            }
            buffers.push(buf);
        }
        if buffers.is_empty() {
            return Ok(Vec::new());
        }
        let slices: Vec<&[u8]> = buffers
            .iter()
            .map(|buf| {
                // SAFETY: `buf` holds a live Py_buffer lock for the duration of
                // this synchronous call and is verified C-contiguous.
                unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.len_bytes()) }
            })
            .collect();
        let results = py
            .detach(|| self.plan.integrate_bslz4(&slices, bslz4_dtype, errors))
            .map_err(cuda_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }
}

/// Experimental: a fixed-mask 2D lookup table uploaded to one CUDA device.
///
/// Created by `Integrator.prepare_cuda2d`. Mirrors `Cuda1dPlan` for 2D cakes.
#[pyclass(name = "Cuda2dPlan", frozen)]
pub struct PyCuda2dPlan {
    plan: Cuda2dPlan,
}

#[pymethods]
impl PyCuda2dPlan {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.plan.shape()
    }
    #[getter]
    fn n_radial(&self) -> usize {
        self.plan.n_radial()
    }
    #[getter]
    fn n_azimuthal(&self) -> usize {
        self.plan.n_azimuthal()
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
    fn radial(&self) -> Option<Vec<f64>> {
        self.plan.radial().map(|r| r.to_vec())
    }
    #[getter]
    fn azimuthal(&self) -> Option<Vec<f64>> {
        self.plan.azimuthal().map(|a| a.to_vec())
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
            "Cuda2dPlan(shape=({dim1}, {dim2}), n_radial={}, n_azimuthal={}, precision={:?}, device={}, device_name={:?})",
            self.plan.n_radial(),
            self.plan.n_azimuthal(),
            self.plan.precision().as_str(),
            self.plan.device(),
            self.plan.device_name()
        )
    }

    #[pyo3(signature = (image, *, errors=false))]
    fn integrate2d(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let borrowed = Borrowed::image(image, self.plan.shape())?;
        borrowed
            .run_2d(py, &self.plan, errors)?
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("CUDA integration returned no result"))
    }

    #[pyo3(signature = (images, *, errors=false))]
    fn batch2d(
        &self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        Borrowed::batch(images, self.plan.shape())?.run_2d(py, &self.plan, errors)
    }

    #[pyo3(signature = (n_frames, dtype=None))]
    fn alloc_pinned_batch<'py>(
        &self,
        py: Python<'py>,
        n_frames: usize,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        let np_dtype = match dtype {
            Some(d) => numpy.call_method1("dtype", (d,))?,
            None => numpy.call_method1("dtype", ("float32",))?,
        };
        let descr = np_dtype
            .cast::<numpy::PyArrayDescr>()
            .map_err(|_| PyTypeError::new_err("dtype must be a valid numpy dtype"))?;
        let pixel = if descr.is_equiv_to(&numpy::dtype::<u16>(py)) {
            Pixel::U16
        } else if descr.is_equiv_to(&numpy::dtype::<u32>(py)) {
            Pixel::U32
        } else if descr.is_equiv_to(&numpy::dtype::<i32>(py)) {
            Pixel::I32
        } else if descr.is_equiv_to(&numpy::dtype::<f32>(py)) {
            Pixel::F32
        } else if descr.is_equiv_to(&numpy::dtype::<f64>(py)) {
            Pixel::F64
        } else {
            return Err(PyTypeError::new_err(format!(
                "dtype must be native uint16, uint32, int32, float32 or float64, got {descr}"
            )));
        };
        let (dim1, dim2) = self.plan.shape();
        let elems = n_frames
            .checked_mul(dim1)
            .and_then(|n| n.checked_mul(dim2))
            .ok_or_else(|| PyValueError::new_err("pinned batch shape overflows usize"))?;
        let bytes = elems
            .checked_mul(pixel.itemsize())
            .ok_or_else(|| PyValueError::new_err("pinned batch byte size overflows usize"))?;
        let len = isize::try_from(bytes)
            .map_err(|_| PyValueError::new_err("pinned batch byte size overflows isize"))?;
        let mut pinned = py
            .detach(|| self.plan.alloc_pinned_bytes(bytes))
            .map_err(cuda_error)?;
        let ptr = pinned.as_mut_ptr().map_err(cuda_error)? as usize;
        let owner = Py::new(
            py,
            PinnedBufferOwner {
                _buf: Mutex::new(pinned),
                ptr,
                len,
            },
        )?;
        let flat = numpy.call_method1("frombuffer", (owner.bind(py), np_dtype))?;
        flat.call_method1("reshape", ((n_frames, dim1, dim2),))
    }

    #[pyo3(signature = (chunks, dtype="uint32", *, errors=false))]
    fn batch2d_bslz4(
        &self,
        py: Python<'_>,
        chunks: &Bound<'_, PyAny>,
        dtype: &str,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let bslz4_dtype = match dtype {
            "uint16" | "u16" => Bslz4Dtype::U16,
            "uint32" | "u32" => Bslz4Dtype::U32,
            "int32" | "i32" => Bslz4Dtype::I32,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "dtype must be 'uint16', 'uint32' or 'int32', got {dtype:?}"
                )));
            }
        };
        if !(chunks.is_instance_of::<PyList>() || chunks.is_instance_of::<PyTuple>()) {
            return Err(PyTypeError::new_err(format!(
                "chunks must be a list or tuple of byte buffers, got {}",
                received(chunks)
            )));
        }
        let mut buffers: Vec<PyBuffer<u8>> = Vec::new();
        for item in chunks.try_iter()? {
            let item = item?;
            let buf = PyBuffer::<u8>::get(&item).map_err(|e| {
                PyTypeError::new_err(format!("chunk must support the byte buffer protocol: {e}"))
            })?;
            if !buf.is_c_contiguous() {
                return Err(PyValueError::new_err("chunk buffer must be C contiguous"));
            }
            buffers.push(buf);
        }
        if buffers.is_empty() {
            return Ok(Vec::new());
        }
        let slices: Vec<&[u8]> = buffers
            .iter()
            .map(|buf| {
                // SAFETY: `buf` holds a live Py_buffer lock for the duration of
                // this synchronous call and is verified C-contiguous.
                unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.len_bytes()) }
            })
            .collect();
        let results = py
            .detach(|| self.plan.integrate_bslz4(&slices, bslz4_dtype, errors))
            .map_err(cuda_error)?;
        Ok(results.into_iter().map(to_python_2d).collect())
    }
}
