//! Experimental Metal 1D and 2D plans: `Integrator.prepare_metal1d`,
//! `Integrator.prepare_metal2d`, `Metal1dPlan`, `Metal2dPlan`,
//! `metal_available`, `metal_info`, and `MetalUnavailableError`.
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
    Bslz4Dtype, Fixed1dResult, Fixed2dResult, Integrator, Metal1dOptions, Metal1dPlan, Metal2dPlan,
    MetalError, MetalPixel, MetalPrecision, MetalSharedBytes, metal_info,
};

create_exception!(
    dioptrin,
    MetalUnavailableError,
    PyRuntimeError,
    "Experimental: Metal cannot be used. Raised when dioptrin was built without \
     the `metal` feature, on non-macOS platforms, or when the Metal device or \
     kernels cannot be initialised. A subclass of RuntimeError."
);

const NOT_COMPILED: &str = "dioptrin was built without Metal support; build with: \
     maturin build --release --features python,metal";

fn metal_error(error: MetalError) -> PyErr {
    match error {
        MetalError::NotCompiled => MetalUnavailableError::new_err(NOT_COMPILED),
        MetalError::Unavailable(_) => MetalUnavailableError::new_err(error.to_string()),
        MetalError::Unsupported(_) | MetalError::InvalidInput(_) => {
            PyValueError::new_err(error.to_string())
        }
        MetalError::Device(_) => PyRuntimeError::new_err(error.to_string()),
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

pub(crate) fn prepare(
    py: Python<'_>,
    integrator: &Integrator,
    shape: (usize, usize),
    n_points: usize,
    precision: &str,
    device: usize,
    max_batch_bytes: Option<usize>,
) -> PyResult<PyMetal1dPlan> {
    let precision: MetalPrecision = precision.parse().map_err(metal_error)?;
    let mut options = Metal1dOptions::new()
        .with_precision(precision)
        .with_device(device);
    if let Some(bytes) = max_batch_bytes {
        options = options.with_max_batch_bytes(bytes);
    }
    let plan = py
        .detach(|| integrator.prepare_metal1d(shape, n_points, &options))
        .map_err(metal_error)?;
    Ok(PyMetal1dPlan { plan })
}

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
) -> PyResult<PyMetal2dPlan> {
    let precision: MetalPrecision = precision.parse().map_err(metal_error)?;
    let mut options = Metal1dOptions::new()
        .with_precision(precision)
        .with_device(device);
    if let Some(bytes) = max_batch_bytes {
        options = options.with_max_batch_bytes(bytes);
    }
    let plan = py
        .detach(|| {
            integrator.prepare_metal2d(shape, n_radial, n_azimuthal, azimuthal_range, &options)
        })
        .map_err(metal_error)?;
    Ok(PyMetal2dPlan { plan })
}

#[pyfunction]
#[pyo3(signature = (device=0))]
pub fn metal_available(py: Python<'_>, device: usize) -> bool {
    py.detach(|| metal_info(device).available)
}

#[pyfunction(name = "metal_info")]
#[pyo3(signature = (device=0))]
pub fn metal_info_py<'py>(py: Python<'py>, device: usize) -> PyResult<Bound<'py, PyDict>> {
    let info = py.detach(|| metal_info(device));
    let dict = PyDict::new(py);
    dict.set_item("compiled", info.compiled)?;
    dict.set_item("available", info.available)?;
    dict.set_item("reason", info.reason)?;
    dict.set_item("device", info.device)?;
    dict.set_item("device_name", info.device_name)?;
    dict.set_item("has_unified_memory", info.has_unified_memory)?;
    dict.set_item(
        "recommended_max_working_set_size",
        info.recommended_max_working_set_size,
    )?;
    dict.set_item(
        "max_threadgroup_memory_length",
        info.max_threadgroup_memory_length,
    )?;
    Ok(dict)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArrayDtype {
    F64,
    F32,
    U16,
    U32,
    I32,
}

fn array_dtype<'py>(array: &Bound<'py, PyUntypedArray>) -> PyResult<ArrayDtype> {
    let py = array.py();
    let descr = array.dtype();
    if descr.is_equiv_to(&numpy::dtype::<f64>(py)) {
        Ok(ArrayDtype::F64)
    } else if descr.is_equiv_to(&numpy::dtype::<f32>(py)) {
        Ok(ArrayDtype::F32)
    } else if descr.is_equiv_to(&numpy::dtype::<u16>(py)) {
        Ok(ArrayDtype::U16)
    } else if descr.is_equiv_to(&numpy::dtype::<u32>(py)) {
        Ok(ArrayDtype::U32)
    } else if descr.is_equiv_to(&numpy::dtype::<i32>(py)) {
        Ok(ArrayDtype::I32)
    } else {
        Err(PyTypeError::new_err(
            "expected float64, float32, uint16, uint32 or int32 numpy array",
        ))
    }
}

fn check_2d_array<'py, T: Element>(
    obj: &Bound<'py, PyAny>,
    expected: (usize, usize),
) -> PyResult<PyReadonlyArray2<'py, T>> {
    let arr: PyReadonlyArray2<'py, T> = obj.extract()?;
    if arr.shape() != [expected.0, expected.1] {
        return Err(PyValueError::new_err(format!(
            "image shape {:?} does not match plan shape {:?}",
            arr.shape(),
            expected
        )));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err("image must be C-contiguous"));
    }
    Ok(arr)
}

fn check_3d_array<'py, T: Element>(
    obj: &Bound<'py, PyAny>,
    expected: (usize, usize),
) -> PyResult<PyReadonlyArray3<'py, T>> {
    let arr: PyReadonlyArray3<'py, T> = obj.extract()?;
    let s = arr.shape();
    if s[1] != expected.0 || s[2] != expected.1 {
        return Err(PyValueError::new_err(format!(
            "stack frame shape ({}, {}) does not match plan shape {:?}",
            s[1], s[2], expected
        )));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err("image stack must be C-contiguous"));
    }
    Ok(arr)
}

fn parse_bslz4_dtype(dtype: &str) -> PyResult<Bslz4Dtype> {
    match dtype {
        "uint16" | "u16" => Ok(Bslz4Dtype::U16),
        "uint32" | "u32" => Ok(Bslz4Dtype::U32),
        "int32" | "i32" => Ok(Bslz4Dtype::I32),
        _ => Err(PyValueError::new_err(format!(
            "unsupported bslz4 dtype {dtype:?}; expected \"uint16\", \"uint32\" or \"int32\""
        ))),
    }
}

fn collect_bslz4_buffers(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyBuffer<u8>>> {
    let mut bufs = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let buf = PyBuffer::<u8>::get(&item)?;
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "bslz4 chunk must be contiguous bytes",
            ));
        }
        bufs.push(buf);
    }
    Ok(bufs)
}

#[pyclass(name = "_MetalSharedBuffer")]
struct MetalSharedOwner {
    buf: Mutex<MetalSharedBytes>,
    len: usize,
}

#[pymethods]
impl MetalSharedOwner {
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let mut guard = slf
            .buf
            .lock()
            .map_err(|_| PyRuntimeError::new_err("shared buffer lock poisoned"))?;
        let ptr = guard.as_mut_ptr();
        unsafe {
            ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr.cast(), slf.len as isize, 0, flags);
        }
        Ok(())
    }
}

fn alloc_shared_numpy<'py>(
    py: Python<'py>,
    alloc: impl FnOnce(usize) -> Result<MetalSharedBytes, MetalError> + Send,
    shape: (usize, usize),
    n_frames: usize,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (elem_bytes, np_dtype) = match dtype {
        "float32" | "f32" => (size_of::<f32>(), "float32"),
        "uint16" | "u16" => (size_of::<u16>(), "uint16"),
        "uint32" | "u32" => (size_of::<u32>(), "uint32"),
        "int32" | "i32" => (size_of::<i32>(), "int32"),
        "float64" | "f64" => (size_of::<f64>(), "float64"),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported shared batch dtype {dtype:?}; expected float32, uint16, uint32, int32 or float64"
            )));
        }
    };
    let bytes = n_frames
        .checked_mul(shape.0)
        .and_then(|v| v.checked_mul(shape.1))
        .and_then(|v| v.checked_mul(elem_bytes))
        .ok_or_else(|| PyValueError::new_err("shared batch byte size overflow"))?;
    let buf = py.detach(|| alloc(bytes)).map_err(metal_error)?;
    let owner = Py::new(
        py,
        MetalSharedOwner {
            buf: Mutex::new(buf),
            len: bytes,
        },
    )?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("frombuffer", (owner, np_dtype))?;
    arr.call_method1("reshape", ((n_frames, shape.0, shape.1),))
}

#[pyclass(name = "Metal1dPlan")]
pub struct PyMetal1dPlan {
    plan: Metal1dPlan,
}

impl PyMetal1dPlan {
    fn run_frame_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let arr = check_2d_array::<P>(obj, self.plan.shape())?;
        let slice = arr.as_slice()?;
        let res = py
            .detach(|| self.plan.integrate_frame(slice, errors))
            .map_err(metal_error)?;
        Ok(to_python(res))
    }

    fn run_stack_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let arr = check_3d_array::<P>(obj, self.plan.shape())?;
        let slice = arr.as_slice()?;
        let results = py
            .detach(|| self.plan.integrate_stack(slice, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }

    fn run_list_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        items: &[Bound<'_, PyAny>],
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let arrays: Vec<PyReadonlyArray2<'_, P>> = items
            .iter()
            .map(|item| check_2d_array::<P>(item, self.plan.shape()))
            .collect::<PyResult<_>>()?;
        let slices: Vec<&[P]> = arrays
            .iter()
            .map(|a| a.as_slice().map_err(PyErr::from))
            .collect::<PyResult<_>>()?;
        let results = py
            .detach(|| self.plan.integrate_frames(&slices, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }
}

#[pymethods]
impl PyMetal1dPlan {
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
    fn device_name(&self) -> &str {
        self.plan.device_name()
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

    #[pyo3(signature = (image, *, errors=false))]
    fn integrate1d(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let untyped = image
            .cast::<PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("image must be a 2D numpy array"))?;
        match array_dtype(untyped)? {
            ArrayDtype::F64 => self.run_frame_typed::<f64>(py, image, errors),
            ArrayDtype::F32 => self.run_frame_typed::<f32>(py, image, errors),
            ArrayDtype::U16 => self.run_frame_typed::<u16>(py, image, errors),
            ArrayDtype::U32 => self.run_frame_typed::<u32>(py, image, errors),
            ArrayDtype::I32 => self.run_frame_typed::<i32>(py, image, errors),
        }
    }

    #[pyo3(signature = (images, *, errors=false))]
    fn batch1d(
        &self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        if let Ok(untyped) = images.cast::<PyUntypedArray>() {
            return match array_dtype(untyped)? {
                ArrayDtype::F64 => self.run_stack_typed::<f64>(py, images, errors),
                ArrayDtype::F32 => self.run_stack_typed::<f32>(py, images, errors),
                ArrayDtype::U16 => self.run_stack_typed::<u16>(py, images, errors),
                ArrayDtype::U32 => self.run_stack_typed::<u32>(py, images, errors),
                ArrayDtype::I32 => self.run_stack_typed::<i32>(py, images, errors),
            };
        }
        let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = images.cast::<PyList>() {
            list.iter().collect()
        } else if let Ok(tuple) = images.cast::<PyTuple>() {
            tuple.iter().collect()
        } else {
            return Err(PyTypeError::new_err(
                "images must be a 3D numpy array or a list/tuple of 2D numpy arrays",
            ));
        };
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let first = items[0]
            .cast::<PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("each frame must be a 2D numpy array"))?;
        let dt = array_dtype(first)?;
        for item in &items[1..] {
            let u = item
                .cast::<PyUntypedArray>()
                .map_err(|_| PyTypeError::new_err("each frame must be a 2D numpy array"))?;
            if array_dtype(u)? != dt {
                return Err(PyTypeError::new_err("mixed dtypes in one batch1d call"));
            }
        }
        match dt {
            ArrayDtype::F64 => self.run_list_typed::<f64>(py, &items, errors),
            ArrayDtype::F32 => self.run_list_typed::<f32>(py, &items, errors),
            ArrayDtype::U16 => self.run_list_typed::<u16>(py, &items, errors),
            ArrayDtype::U32 => self.run_list_typed::<u32>(py, &items, errors),
            ArrayDtype::I32 => self.run_list_typed::<i32>(py, &items, errors),
        }
    }

    #[pyo3(signature = (chunks, dtype="uint32", *, errors=false))]
    fn batch1d_bslz4(
        &self,
        py: Python<'_>,
        chunks: &Bound<'_, PyAny>,
        dtype: &str,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let dt = parse_bslz4_dtype(dtype)?;
        let bufs = collect_bslz4_buffers(chunks)?;
        let slices: Vec<&[u8]> = bufs
            .iter()
            .map(|b| unsafe { std::slice::from_raw_parts(b.buf_ptr().cast::<u8>(), b.len_bytes()) })
            .collect();
        let results = py
            .detach(|| self.plan.integrate_bslz4(&slices, dt, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python).collect())
    }

    #[pyo3(signature = (n_frames, dtype="float32"))]
    fn alloc_shared_batch<'py>(
        &self,
        py: Python<'py>,
        n_frames: usize,
        dtype: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        alloc_shared_numpy(
            py,
            |bytes| self.plan.alloc_shared_bytes(bytes),
            self.plan.shape(),
            n_frames,
            dtype,
        )
    }
}

#[pyclass(name = "Metal2dPlan")]
pub struct PyMetal2dPlan {
    plan: Metal2dPlan,
}

impl PyMetal2dPlan {
    fn run_frame_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let arr = check_2d_array::<P>(obj, self.plan.shape())?;
        let slice = arr.as_slice()?;
        let res = py
            .detach(|| self.plan.integrate_frame(slice, errors))
            .map_err(metal_error)?;
        Ok(to_python_2d(res))
    }

    fn run_stack_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let arr = check_3d_array::<P>(obj, self.plan.shape())?;
        let slice = arr.as_slice()?;
        let results = py
            .detach(|| self.plan.integrate_stack(slice, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python_2d).collect())
    }

    fn run_list_typed<P: MetalPixel + Element>(
        &self,
        py: Python<'_>,
        items: &[Bound<'_, PyAny>],
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let arrays: Vec<PyReadonlyArray2<'_, P>> = items
            .iter()
            .map(|item| check_2d_array::<P>(item, self.plan.shape()))
            .collect::<PyResult<_>>()?;
        let slices: Vec<&[P]> = arrays
            .iter()
            .map(|a| a.as_slice().map_err(PyErr::from))
            .collect::<PyResult<_>>()?;
        let results = py
            .detach(|| self.plan.integrate_frames(&slices, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python_2d).collect())
    }

    #[allow(clippy::type_complexity)]
    fn run_stack_into_numpy<'py, P: MetalPixel + Element>(
        &self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        errors: bool,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Option<Bound<'py, PyArray3<f64>>>)> {
        let arr = check_3d_array::<P>(obj, self.plan.shape())?;
        let n_frames = arr.shape()[0];
        let slice = arr.as_slice()?;
        let n_rad = self.plan.n_radial();
        let n_az = self.plan.n_azimuthal();
        let out_int = PyArray3::<f64>::zeros(py, [n_frames, n_az, n_rad], false);
        let out_err = errors.then(|| PyArray3::<f64>::zeros(py, [n_frames, n_az, n_rad], false));
        // SAFETY: newly allocated exclusive C-contiguous NumPy arrays.
        let int_slice = unsafe { out_int.as_slice_mut()? };
        let err_slice = match &out_err {
            Some(e) => Some(unsafe { e.as_slice_mut()? }),
            None => None,
        };
        py.detach(|| self.plan.integrate_stack_into(slice, int_slice, err_slice))
            .map_err(metal_error)?;
        Ok((out_int, out_err))
    }
}

#[pymethods]
impl PyMetal2dPlan {
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
    fn radial(&self) -> Vec<f64> {
        self.plan.radial().to_vec()
    }
    #[getter]
    fn azimuthal(&self) -> Vec<f64> {
        self.plan.azimuthal().to_vec()
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
    fn device_name(&self) -> &str {
        self.plan.device_name()
    }
    #[getter]
    fn max_batch_frames(&self) -> usize {
        self.plan.max_batch_frames()
    }
    #[getter]
    fn table_entries(&self) -> usize {
        self.plan.table_entries()
    }

    #[pyo3(signature = (image, *, errors=false))]
    fn integrate2d(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<PyIntegrationResult> {
        let untyped = image
            .cast::<PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("image must be a 2D numpy array"))?;
        match array_dtype(untyped)? {
            ArrayDtype::F64 => self.run_frame_typed::<f64>(py, image, errors),
            ArrayDtype::F32 => self.run_frame_typed::<f32>(py, image, errors),
            ArrayDtype::U16 => self.run_frame_typed::<u16>(py, image, errors),
            ArrayDtype::U32 => self.run_frame_typed::<u32>(py, image, errors),
            ArrayDtype::I32 => self.run_frame_typed::<i32>(py, image, errors),
        }
    }

    #[pyo3(signature = (images, *, errors=false))]
    fn batch2d(
        &self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        if let Ok(untyped) = images.cast::<PyUntypedArray>() {
            return match array_dtype(untyped)? {
                ArrayDtype::F64 => self.run_stack_typed::<f64>(py, images, errors),
                ArrayDtype::F32 => self.run_stack_typed::<f32>(py, images, errors),
                ArrayDtype::U16 => self.run_stack_typed::<u16>(py, images, errors),
                ArrayDtype::U32 => self.run_stack_typed::<u32>(py, images, errors),
                ArrayDtype::I32 => self.run_stack_typed::<i32>(py, images, errors),
            };
        }
        let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = images.cast::<PyList>() {
            list.iter().collect()
        } else if let Ok(tuple) = images.cast::<PyTuple>() {
            tuple.iter().collect()
        } else {
            return Err(PyTypeError::new_err(
                "images must be a 3D numpy array or a list/tuple of 2D numpy arrays",
            ));
        };
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let first = items[0]
            .cast::<PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("each frame must be a 2D numpy array"))?;
        let dt = array_dtype(first)?;
        for item in &items[1..] {
            let u = item
                .cast::<PyUntypedArray>()
                .map_err(|_| PyTypeError::new_err("each frame must be a 2D numpy array"))?;
            if array_dtype(u)? != dt {
                return Err(PyTypeError::new_err("mixed dtypes in one batch2d call"));
            }
        }
        match dt {
            ArrayDtype::F64 => self.run_list_typed::<f64>(py, &items, errors),
            ArrayDtype::F32 => self.run_list_typed::<f32>(py, &items, errors),
            ArrayDtype::U16 => self.run_list_typed::<u16>(py, &items, errors),
            ArrayDtype::U32 => self.run_list_typed::<u32>(py, &items, errors),
            ArrayDtype::I32 => self.run_list_typed::<i32>(py, &items, errors),
        }
    }

    #[pyo3(signature = (images, *, errors=false))]
    fn batch2d_stack<'py>(
        &self,
        py: Python<'py>,
        images: &Bound<'py, PyAny>,
        errors: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let untyped = images
            .cast::<PyUntypedArray>()
            .map_err(|_| PyTypeError::new_err("images must be a 3D numpy array"))?;
        let (int_arr, err_arr) = match array_dtype(untyped)? {
            ArrayDtype::F64 => self.run_stack_into_numpy::<f64>(py, images, errors)?,
            ArrayDtype::F32 => self.run_stack_into_numpy::<f32>(py, images, errors)?,
            ArrayDtype::U16 => self.run_stack_into_numpy::<u16>(py, images, errors)?,
            ArrayDtype::U32 => self.run_stack_into_numpy::<u32>(py, images, errors)?,
            ArrayDtype::I32 => self.run_stack_into_numpy::<i32>(py, images, errors)?,
        };
        match err_arr {
            Some(e) => Ok((int_arr, e).into_pyobject(py)?.into_any()),
            None => Ok(int_arr.into_any()),
        }
    }

    #[pyo3(signature = (chunks, dtype="uint32", *, errors=false))]
    fn batch2d_bslz4(
        &self,
        py: Python<'_>,
        chunks: &Bound<'_, PyAny>,
        dtype: &str,
        errors: bool,
    ) -> PyResult<Vec<PyIntegrationResult>> {
        let dt = parse_bslz4_dtype(dtype)?;
        let bufs = collect_bslz4_buffers(chunks)?;
        let slices: Vec<&[u8]> = bufs
            .iter()
            .map(|b| unsafe { std::slice::from_raw_parts(b.buf_ptr().cast::<u8>(), b.len_bytes()) })
            .collect();
        let results = py
            .detach(|| self.plan.integrate_bslz4(&slices, dt, errors))
            .map_err(metal_error)?;
        Ok(results.into_iter().map(to_python_2d).collect())
    }

    #[pyo3(signature = (n_frames, dtype="float32"))]
    fn alloc_shared_batch<'py>(
        &self,
        py: Python<'py>,
        n_frames: usize,
        dtype: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        alloc_shared_numpy(
            py,
            |bytes| self.plan.alloc_shared_bytes(bytes),
            self.plan.shape(),
            n_frames,
            dtype,
        )
    }
}

#[allow(dead_code)]
fn _keep_pyarray2_import(_: &PyArray2<f64>) {}
